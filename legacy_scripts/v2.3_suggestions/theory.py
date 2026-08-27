"""
Theory-validation machinery for QS-Net / CQ-ZDR, implementing the constructs from
`Theoretical_Results.pdf` (QuantumSentinel / QS-Net) that are not already covered by
the rest of the `scripts/` package:

  - Proposition 1 (noise contraction): depolarizing channel + exact-identity check,
    evaluated on REAL data-derived density matrices only (see check_proposition1_on_data).
  - Section 4 (Lipschitz constant for angle encoding): analytic bound L_phi = R/2,
    with the sampled-ratio estimator demoted to a tightness diagnostic only.
  - Proposition 2 (adversarial-novelty separation): worst-case F_in / F_out, the gap
    Delta, epsilon*, and the quantile-relaxed Corollary 1 (F_out^(beta), epsilon^(beta)).
  - Section 3.1 note: L2 <-> L_inf perturbation-budget conversion for d-dim inputs.
  - Attacks that target the CQ-ZDR novelty score directly (Section 3, "empirical
    breakpoint"), as opposed to attacks.py's classifier-head CE-loss attacks.
  - Proposition 3 support: a two-sample exchangeability (AUROC) diagnostic and an exact
    (Clopper-Pearson) confidence interval for reporting detection rates honestly.

DATA POLICY: every density matrix that feeds a reported *scientific* quantity in this
module (Proposition 1's rho/sigma pairs, F_in/F_out, Lipschitz ratios, attack targets)
is the trained encoder+ansatz's own output on real dataset samples -- rho(x) for real x,
looked up from a scripts.cache.ForwardCache wherever possible instead of being
recomputed. The ONE exception is verify_fidelity_convention(), which is a pure SOFTWARE
unit test of the fidelity() *implementation* (checking it satisfies a textbook identity
and the Fuchs-van de Graaf inequalities) -- it uses synthetic Ginibre-random matrices on
purpose, the same way you'd unit-test `add(x, y)` with arbitrary numbers; it is not a
measurement about the dataset or the model and makes no claim about either.

PERFORMANCE NOTES (fgsm_attack_nonconformity / pgd_attack_nonconformity):
  - Both attacks process X internally in chunks of `batch_size` samples, so peak GPU
    memory for a call is bounded by O(batch_size) regardless of how large X is.
  - The prototype stack is built once (`stack_prototypes`) and reused across every
    chunk / every PGD step / every call in a sweep if a precomputed `proto_stack` is
    passed in.
  - fgsm_gradient_sign() / apply_fgsm_perturbation() split the one-step FGSM attack into
    its two logically separate parts: FGSM's attack DIRECTION sign(grad_x s(x)) does not
    depend on the perturbation budget eps at all, only the step size does. Sweeping many
    eps values against the same starting batch therefore needs exactly ONE forward+
    backward pass (fgsm_gradient_sign), followed by a cheap, gradient-free clamp+scale
    per eps (apply_fgsm_perturbation) -- instead of one full forward+backward pass PER
    epsilon, which is what naively calling fgsm_attack_nonconformity(eps=...) in a loop
    does. fgsm_attack_nonconformity() itself is kept as a single-call convenience
    wrapper around the same two functions for callers that only need one eps value.
"""

import math

import numpy as np
import torch

from scripts.conformal import (  # noqa: F401  (re-exported)
    min_calibration_size,
    nonconformity_score,
)
from scripts.constants import (
    DEFAULT_BETA,
    DEFAULT_NOISE_RATE,
    INPUT_DIM_D,
    PROP1_RESIDUAL_TOL,
)
from scripts.quantum_metrics import (
    fidelity,
    fidelity_pairwise,
    max_fidelity_to_prototypes,
    stack_prototypes,
    trace_distance,
)
from scripts.utils import to_torch_batch_x

# --------------------------------------------------------------------------------------
# Fidelity-convention self-check (SOFTWARE unit test -- see module docstring)
# --------------------------------------------------------------------------------------


def verify_fidelity_convention(dim=4, n_trials=20, seed=0, tol=1e-6):
    """
    Confirms scripts.quantum_metrics is using the NON-SQUARED fidelity convention
    F(rho,sigma) = tr(sqrt(sqrt(rho) sigma sqrt(rho))) required by the theory doc
    (Section 1), and that fidelity() / fidelity_pairwise() agree with each other and
    with the Fuchs-van de Graaf inequalities (Eq. 1).

    This is a code-correctness unit test, not a scientific measurement -- the synthetic
    Ginibre matrices below are arbitrary test fixtures (like calling add(2, 3) to check
    an addition function), never used for any reported result.
    """
    rng = np.random.default_rng(seed)
    max_err_pair, max_fvg_violation = 0.0, 0.0

    for _ in range(n_trials):
        rho = _random_test_fixture_matrix(dim, rng)
        sigma = _random_test_fixture_matrix(dim, rng)

        f_a = float(fidelity(rho, sigma))
        f_b = float(fidelity_pairwise(rho.unsqueeze(0), sigma).squeeze())
        max_err_pair = max(max_err_pair, abs(f_a - f_b))

        d_tr = float(trace_distance(rho, sigma))
        lower = 1.0 - f_a
        upper = math.sqrt(max(1.0 - f_a**2, 0.0))
        violation = max(lower - d_tr, d_tr - upper, 0.0)
        max_fvg_violation = max(max_fvg_violation, violation)

    assert max_err_pair < tol, (
        f"fidelity() and fidelity_pairwise() disagree by {max_err_pair:.3e}. "
        "They must use the same (non-squared) convention -- check quantum_metrics.py."
    )
    assert max_fvg_violation < 1e-3, (
        f"Fuchs-van de Graaf bounds (Eq. 1) violated by {max_fvg_violation:.3e}. "
        "This means fidelity() is still returning the SQUARED convention "
        "(qml.math.fidelity returns F^2) -- patch quantum_metrics.py to take sqrt()."
    )
    return {
        "max_pairwise_disagreement": max_err_pair,
        "max_fvg_violation": max_fvg_violation,
    }


def _random_test_fixture_matrix(dim, rng):
    """Arbitrary Ginibre-random matrix used ONLY as a unit-test fixture (see docstring above)."""
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    A = torch.tensor(A, dtype=torch.cdouble)
    rho = A @ A.conj().T
    return rho / torch.trace(rho).real


# --------------------------------------------------------------------------------------
# Proposition 1: noise contraction  (Section 2, Eq. 2) -- REAL DATA ONLY
# --------------------------------------------------------------------------------------


def depolarizing_channel_global(rho, p):
    """Lambda_p(rho) = (1-p) rho + p * I / d   -- the channel Proposition 1 is stated for."""
    d = rho.shape[-1]
    I = torch.eye(d, dtype=rho.dtype, device=rho.device)
    return (1.0 - p) * rho + p * I / d


def check_proposition1_on_data(
    rho_real, p_values=(0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0), n_pairs=25, seed=0
):
    """
    Numerically confirms D_tr(Lambda_p(rho), Lambda_p(sigma)) = (1-p) D_tr(rho,sigma)
    (Eq. 2) using pairs of the encoder's OWN density matrices rho(x) on REAL dataset
    samples -- e.g. rho_real = cache.get("test")["rho"] from a scripts.cache.ForwardCache
    that was already computed for classification/fidelity, so this check adds no new
    circuit evaluations. No synthetic or random density matrices are used: only the
    choice of WHICH real (x_i, x_j) pair to test is randomized, not the states themselves.

    Per the doc's "Note for implementation", the residual should be <= ~1e-10 (machine
    precision) for every p and every pair -- NOT ~1e-2.
    """
    n = len(rho_real)
    if n < 2:
        raise ValueError("need at least 2 cached density matrices to form pairs")
    rng = np.random.default_rng(seed)
    idx1 = rng.integers(0, n, size=n_pairs)
    idx2 = rng.integers(0, n, size=n_pairs)

    records = []
    for i1, i2 in zip(idx1, idx2):
        rho = rho_real[i1].to(torch.complex128)
        sigma = rho_real[i2].to(torch.complex128)
        d0 = float(trace_distance(rho, sigma))
        for p in p_values:
            d_measured = float(
                trace_distance(
                    depolarizing_channel_global(rho, p),
                    depolarizing_channel_global(sigma, p),
                )
            )
            d_theory = (1.0 - p) * d0
            records.append(
                {
                    "p": float(p),
                    "sample_i": int(i1),
                    "sample_j": int(i2),
                    "D_tr_measured": d_measured,
                    "D_tr_theory": d_theory,
                    "abs_error": abs(d_measured - d_theory),
                }
            )
    return records


def assert_proposition1(records, tol=PROP1_RESIDUAL_TOL):
    max_err = max(r["abs_error"] for r in records)
    assert max_err <= tol, (
        f"Proposition 1 residual {max_err:.3e} exceeds tolerance {tol:.1e}. "
        "Eq. 2 is an EXACT identity -- a residual this large means the channel "
        "(or the density-matrix sampling) is implemented incorrectly, not numerical noise."
    )
    return max_err


# --------------------------------------------------------------------------------------
# Section 4: analytic Lipschitz bound for angle encoding (Lemma 1)
# --------------------------------------------------------------------------------------


def analytic_lipschitz_bound(n_layers, reupload=True):
    """
    Lemma 1 (+ its data-reuploading extension): for single-qubit-per-feature angle
    encoding, D_tr(|psi(x)>,|psi(x')>) <= (R/2) ||x-x'||_2, where R is the number of
    times the data is re-embedded. R = n_layers if reupload else 1.
    This is the certified L_phi -- it should be used in the certified-radius / epsilon*
    formulas, NOT a sampled percentile (see scripts.cache.cached_lipschitz_percentile).
    """
    R = n_layers if reupload else 1
    return R / 2.0


def check_lipschitz_tightness(diag, l_phi_bound):
    """Flags whether any sampled ratio exceeds the analytic bound (encoder-bug signal)."""
    max_ratio = diag["max"]
    return {
        "max_sampled_ratio": max_ratio,
        "analytic_bound": l_phi_bound,
        "within_bound": bool(max_ratio <= l_phi_bound + 1e-6),
        "tightness_gap": l_phi_bound - max_ratio,
    }


# --------------------------------------------------------------------------------------
# Section 3.1: L2 <-> L_inf perturbation-budget conversion
# --------------------------------------------------------------------------------------


def linf_to_l2_budget(eps_inf, d=INPUT_DIM_D):
    """||delta||_2 <= sqrt(d) ||delta||_inf."""
    return math.sqrt(d) * eps_inf


def l2_to_linf_budget(eps_2, d=INPUT_DIM_D):
    return eps_2 / math.sqrt(d)


# --------------------------------------------------------------------------------------
# F_max / F_in / F_out  (Section 1 "nearest-prototype quantities"; Assumptions A2/A3/A3')
# --------------------------------------------------------------------------------------


def f_max_batch(X, theta, prototypes, forward_circuit, device=None, batch_size=64):
    """
    F_max(x) = max_{c in K} F(rho(x), rho_c) for every sample in X (known prototypes only).
    Prefer scripts.cache.cached_f_max if you already have a ForwardCache entry for X --
    this variant re-runs the circuit and exists for callers without a cache.
    """
    scores = nonconformity_score(
        X, theta, prototypes, forward_circuit, device=device, batch_size=batch_size
    )
    return 1.0 - scores


def worst_case_f_in(f_max_known, percentile=0.0):
    """
    F_in per Assumption (A2): every known sample satisfies F_max(x) >= F_in.
    percentile=0 -> the literal worst case (minimum); pass e.g. 1 to use the 1st
    percentile instead if a single outlier makes the strict minimum degenerate
    (report both -- the doc only formalizes the strict worst case for F_in).
    """
    return float(np.percentile(f_max_known, percentile))


def quantile_f_out(f_max_zeroday, beta=DEFAULT_BETA):
    """
    F_out^(beta) per (A3'): the (1-beta)-quantile of F_max(z) over zero-day samples,
    so at least a (1-beta) fraction satisfy F_max(z) <= F_out^(beta).
    beta=0 recovers the strict worst case (A3): F_out = max(F_max(z)).
    """
    return float(np.percentile(f_max_zeroday, 100.0 * (1.0 - beta)))


# --------------------------------------------------------------------------------------
# Proposition 2: gap Delta, epsilon*, quantile-relaxed epsilon^(beta)   (Eq. 3, Eq. 4)
# --------------------------------------------------------------------------------------


def proposition2_gap(F_in, F_out):
    """Delta = (1 - F_out) - sqrt(1 - F_in^2)   (Eq. 3). Delta > 0 is Assumption (A5)."""
    return (1.0 - F_out) - math.sqrt(max(1.0 - F_in**2, 0.0))


def proposition2_epsilon_star(F_in, F_out, L_phi, p=DEFAULT_NOISE_RATE):
    """
    epsilon* = Delta / (2 (1-p) L_phi)   (Eq. 4) -- the certified L2 perturbation
    budget below which adversarially-perturbed known samples stay accepted and
    genuine zero-day samples stay rejected. NOTE: no C_f term here (that only
    appears in the operational certified radius R of Algorithm 3, Section 3.4).

    epsilon* can be negative -- that is a valid, reportable outcome (Section 3.1, A5):
    it means the worst-case separability precondition currently fails, not an error.
    Use plot_safe_epsilon() when visualizing this value on a perturbation-budget axis.
    """
    delta = proposition2_gap(F_in, F_out)
    eps_star = delta / (2.0 * (1.0 - p) * L_phi)
    return delta, eps_star


def proposition2_epsilon_beta(
    F_in, f_max_zeroday, L_phi, p=DEFAULT_NOISE_RATE, beta=DEFAULT_BETA
):
    """Corollary 1 (quantile-relaxed): returns (F_out_beta, Delta_beta, epsilon_beta)."""
    F_out_beta = quantile_f_out(f_max_zeroday, beta=beta)
    delta_beta, eps_beta = proposition2_epsilon_star(F_in, F_out_beta, L_phi, p=p)
    return F_out_beta, delta_beta, eps_beta


def plot_safe_epsilon(eps_star):
    """
    epsilon* (or epsilon^(beta)) can be <= 0 when the worst-case separability
    precondition (A5) fails -- that is a valid result, but 0 perturbation budgets are
    the natural floor of a "perturbation size" axis, so a negative value has no sensible
    x-position on such a plot (matplotlib will just silently pad the axis into a
    confusing, mostly-empty range). Returns (plot_value_or_None, is_vacuous):
    draw an axvline at plot_value only when it's not None; otherwise annotate the
    non-separability in text instead of drawing a line off the axis.
    """
    if eps_star is None or not np.isfinite(eps_star) or eps_star <= 0:
        return None, True
    return float(eps_star), False


# --------------------------------------------------------------------------------------
# Attacks that directly target the CQ-ZDR novelty score (for the Day-19/20/25 experiments)
# --------------------------------------------------------------------------------------


def _as_bound_tensor(bound, device):
    """Accepts None / a python scalar / a tensor and returns a device-matched tensor (or None)."""
    if bound is None:
        return None
    if torch.is_tensor(bound):
        return bound.to(device=device) if device is not None else bound
    return torch.tensor(float(bound), device=device)


def _clear_cuda_cache_if_needed(device):
    """Cheap relative to a circuit forward/backward pass; keeps memory flat across a long sweep."""
    if device is not None and torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.empty_cache()


def fgsm_gradient_sign(
    X, theta, prototypes, forward_circuit, device=None, batch_size=32, proto_stack=None
):
    """
    Computes sign(grad_x [1 - F_max(x)]) ONCE -- the FGSM attack DIRECTION, which does
    NOT depend on the perturbation budget eps. Reuse the returned tensor across an
    entire epsilon sweep via apply_fgsm_perturbation() instead of recomputing a full
    forward+backward pass at every eps (a one-step attack's gradient is eps-independent
    by construction -- eps only scales the already-computed step).
    """
    if proto_stack is None:
        _, proto_stack = stack_prototypes(prototypes)
    proto_stack = proto_stack.to(device=device) if device is not None else proto_stack

    X_np = np.asarray(X, dtype=np.float32)
    n = len(X_np)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 0
    if n == 0:
        return torch.empty((0, n_features))

    chunks = []
    for i in range(0, n, batch_size):
        X_t = (
            to_torch_batch_x(X_np[i : i + batch_size], device=device)
            .detach()
            .requires_grad_(True)
        )
        _, rho = forward_circuit(X_t, theta)
        max_f = max_fidelity_to_prototypes(rho, proto_stack.to(device=rho.device))
        score = (1.0 - max_f).mean()
        grad = torch.autograd.grad(score, X_t)[0]
        chunks.append(grad.sign().detach())
        _clear_cuda_cache_if_needed(device)

    return torch.cat(chunks, dim=0)


def apply_fgsm_perturbation(X, grad_sign, eps, device=None, x_min=None, x_max=None):
    """
    Cheap, gradient-free step: X_adv = clip(X + eps * grad_sign, x_min, x_max). No
    circuit call -- pairs with fgsm_gradient_sign() to sweep many eps values for the
    price of one forward+backward pass instead of one per eps.
    """
    X_t = to_torch_batch_x(np.asarray(X, dtype=np.float32), device=device)
    grad_sign = grad_sign.to(device=X_t.device)
    x_min_t = _as_bound_tensor(x_min, X_t.device)
    x_max_t = _as_bound_tensor(x_max, X_t.device)

    X_adv = X_t + eps * grad_sign
    if x_min_t is not None or x_max_t is not None:
        X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
    return X_adv.detach()


def fgsm_attack_nonconformity(
    X,
    theta,
    prototypes,
    forward_circuit,
    eps,
    device=None,
    x_min=None,
    x_max=None,
    batch_size=32,
    proto_stack=None,
):
    """
    One-step FGSM that maximizes the nonconformity score s(x) = 1 - F_max(x) directly,
    i.e. attacks the CQ-ZDR accept/reject boundary rather than the classifier head's
    cross-entropy loss (which attacks.py's fgsm_attack targets).

    Convenience wrapper around fgsm_gradient_sign() + apply_fgsm_perturbation() for a
    SINGLE eps value. If you need multiple eps values against the same X (a sweep),
    call those two functions directly instead -- see the module docstring.
    """
    grad_sign = fgsm_gradient_sign(
        X,
        theta,
        prototypes,
        forward_circuit,
        device=device,
        batch_size=batch_size,
        proto_stack=proto_stack,
    )
    return apply_fgsm_perturbation(
        X, grad_sign, eps, device=device, x_min=x_min, x_max=x_max
    )


def pgd_attack_nonconformity(
    X,
    theta,
    prototypes,
    forward_circuit,
    eps,
    alpha,
    steps,
    device=None,
    x_min=None,
    x_max=None,
    random_start=True,
    batch_size=32,
    proto_stack=None,
):
    """
    Multi-step PGD version of fgsm_attack_nonconformity, L_inf-projected onto the
    eps-ball. Unlike one-step FGSM, each step's gradient genuinely depends on eps (via
    the projection radius), so there is no eps-independent quantity to precompute and
    reuse across a sweep here -- chunking by `batch_size` is still exact (each sample's
    trajectory is independent of every other sample) and bounds peak memory.
    """
    if proto_stack is None:
        _, proto_stack = stack_prototypes(prototypes)
    proto_stack = proto_stack.to(device=device) if device is not None else proto_stack

    X_np = np.asarray(X, dtype=np.float32)
    n = len(X_np)
    n_features = X_np.shape[1] if X_np.ndim > 1 else 0
    if n == 0:
        return torch.empty((0, n_features))

    x_min_t = _as_bound_tensor(x_min, device)
    x_max_t = _as_bound_tensor(x_max, device)

    chunks = []
    for i in range(0, n, batch_size):
        X0 = to_torch_batch_x(X_np[i : i + batch_size], device=device).detach()
        X_adv = (
            X0 + torch.empty_like(X0).uniform_(-eps, eps)
            if random_start
            else X0.clone()
        )
        if x_min_t is not None or x_max_t is not None:
            X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
        X_adv = X_adv.detach()

        for _ in range(steps):
            X_adv.requires_grad_(True)
            _, rho = forward_circuit(X_adv, theta)
            max_f = max_fidelity_to_prototypes(rho, proto_stack.to(device=rho.device))
            score = (1.0 - max_f).mean()
            grad = torch.autograd.grad(score, X_adv)[0]

            with torch.no_grad():
                X_adv = X_adv + alpha * grad.sign()
                X_adv = torch.clamp(
                    X_adv, min=X0 - eps, max=X0 + eps
                )  # fused L_inf-ball projection
                if x_min_t is not None or x_max_t is not None:
                    X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
            X_adv = X_adv.detach()

        chunks.append(X_adv)
        _clear_cuda_cache_if_needed(device)

    return torch.cat(chunks, dim=0)


# --------------------------------------------------------------------------------------
# Proposition 3 support: exchangeability diagnostic + honest interval reporting
# --------------------------------------------------------------------------------------


def two_sample_discriminability_auroc(X_cal, X_test, seed=0):
    """
    Trains a simple classifier to distinguish calibration vs. test features and reports
    its AUROC. AUROC close to 0.50 is the supporting evidence for the exchangeability
    assumption Proposition 3 relies on; well above 0.50 suggests cal/test come from
    different distributions and the coverage guarantee should not be invoked as-is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([np.asarray(X_cal), np.asarray(X_test)], axis=0)
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=1000)
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(np.mean(aucs))


def clopper_pearson_ci(successes, n, alpha=0.05):
    """
    Exact (Clopper-Pearson) 100(1-alpha)% confidence interval for a binomial proportion.
    Use this to report detection rates like "TPR = 1.000" honestly -- e.g. n=1500,
    successes=1500 gives a 95% CI, not a claim of exactly zero miss rate; report the
    interval alongside the point estimate rather than the bare fraction.
    """
    from scipy.stats import beta as beta_dist

    if n == 0:
        return (float("nan"), float("nan"))
    lo = (
        0.0
        if successes == 0
        else float(beta_dist.ppf(alpha / 2, successes, n - successes + 1))
    )
    hi = (
        1.0
        if successes == n
        else float(beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes))
    )
    return lo, hi


def robust_f_in(f_max_known, lower_percentile=5.0):
    """
    Direct-percentile alias for worst_case_f_in(): F_in = the `lower_percentile`-th
    percentile of F_max over KNOWN samples (e.g. lower_percentile=5 -> the 5th
    percentile), trimming the low-F_max tail instead of taking the strict minimum.
    Same rationale as robust_f_out() below: ignore statistical outliers while
    preserving the majority structure of the known-class distribution. Not formalized
    as a named assumption in the source document (only F_out has a stated quantile-
    relaxed corollary, A3'/Corollary 1) -- report it as an explicit extension, not as
    Corollary 1 itself.
    """
    return worst_case_f_in(f_max_known, percentile=lower_percentile)


def robust_f_out(f_max_zeroday, upper_percentile=97.0):
    """
    Direct-percentile alias for quantile_f_out(): F_out = the `upper_percentile`-th
    percentile of F_max over ZERO-DAY samples (e.g. upper_percentile=97 -> the 97th
    percentile). Equivalent to quantile_f_out(..., beta=1 - upper_percentile/100); this
    wrapper exists purely for readability when you want to think in "Nth percentile"
    terms directly rather than the Corollary-1 "beta fraction excluded" framing.
    """
    beta = 1.0 - upper_percentile / 100.0
    return quantile_f_out(f_max_zeroday, beta=beta)


def proposition2_epsilon_robust(
    F_max_known,
    F_max_zeroday,
    L_phi,
    p=DEFAULT_NOISE_RATE,
    lower_percentile=5.0,
    upper_percentile=97.0,
):
    """
    A SYMMETRIC, doubly-relaxed variant of Eq. 3/4 that trims both tails instead of only
    the zero-day side. The source document's Corollary 1 only formalizes relaxing F_out
    (via a (1-beta)-quantile) and keeps F_in as the strict worst-case minimum (Assumption
    A2 is stated with "every known sample", no quantile-relaxed version given). This
    function additionally replaces F_in with a LOWER percentile of F_max over known
    samples, on the same "ignore statistical outliers, preserve the majority structure"
    rationale, to investigate whether a small number of extreme points on EITHER side
    (not just the zero-day side) are responsible for a failing worst-case certificate.

    This is a diagnostic extension beyond what the source document proves -- report it
    explicitly as such (e.g. "doubly quantile-relaxed, non-formalized variant"), not as
    Corollary 1 itself, alongside the strict and F_out-only-relaxed numbers.

    Returns (F_in_robust, F_out_robust, delta_robust, epsilon_robust).
    """
    F_in_robust = robust_f_in(F_max_known, lower_percentile=lower_percentile)
    F_out_robust = robust_f_out(F_max_zeroday, upper_percentile=upper_percentile)
    delta_robust, eps_robust = proposition2_epsilon_star(
        F_in_robust, F_out_robust, L_phi, p=p
    )
    return F_in_robust, F_out_robust, delta_robust, eps_robust
