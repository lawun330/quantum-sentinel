"""


  - Proposition 1 (noise contraction): depolarizing channel + exact-identity check.
  - Section 4 (Lipschitz constant for angle encoding): analytic bound L_phi = R/2,
    with the sampled-ratio estimator demoted to a tightness diagnostic only.
  - Proposition 2 (adversarial-novelty separation): worst-case F_in / F_out, the gap
    Delta, epsilon*, and the quantile-relaxed Corollary 1 (F_out^(beta), epsilon^(beta)).
  - Section 3.1 note: L2 <-> L_inf perturbation-budget conversion for d-dim inputs.
  - Attacks that target the CQ-ZDR novelty score directly (Section 3, "empirical
    breakpoint"), as opposed to attacks.py's classifier-head CE-loss attacks.
  - Proposition 3 support: a two-sample exchangeability (AUROC) diagnostic
    (min_calibration_size itself lives in conformal.py, re-exported here for
    convenience).



PERFORMANCE NOTES (fgsm_attack_nonconformity / pgd_attack_nonconformity):

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
# Fidelity-convention self-check
# --------------------------------------------------------------------------------------


def verify_fidelity_convention(dim=4, n_trials=20, seed=0, tol=1e-6):
    """
    Confirms scripts.quantum_metrics is using the NON-SQUARED fidelity convention
    F(rho,sigma) = tr(sqrt(sqrt(rho) sigma sqrt(rho))) required by the theory doc
    (Section 1), and that fidelity() / fidelity_pairwise() agree with each other and
    with the Fuchs-van de Graaf inequalities.


    """
    rng = np.random.default_rng(seed)
    max_err_pair, max_fvg_violation = 0.0, 0.0

    for _ in range(n_trials):
        rho = _random_density_matrix(dim, rng)
        sigma = _random_density_matrix(dim, rng)

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


def _random_density_matrix(dim, rng):
    A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    A = torch.tensor(A, dtype=torch.cdouble)
    rho = A @ A.conj().T
    return rho / torch.trace(rho).real


# --------------------------------------------------------------------------------------
# Proposition 1: noise contraction
# --------------------------------------------------------------------------------------


def depolarizing_channel_global(rho, p):
    """Lambda_p(rho) = (1-p) rho + p * I / d   -- the channel Proposition 1 is stated for."""
    d = rho.shape[-1]
    I = torch.eye(d, dtype=rho.dtype, device=rho.device)
    return (1.0 - p) * rho + p * I / d


def check_proposition1(
    dim, p_values=(0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0), n_pairs=25, seed=0
):
    """
    Numerically confirms D_tr(Lambda_p(rho), Lambda_p(sigma)) = (1-p) D_tr(rho,sigma)
    (Eq. 2). Per the doc's "Note for implementation", the residual should be
    <= ~1e-10 (machine precision) for every p and every pair -- NOT ~1e-2.
    """
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n_pairs):
        rho = _random_density_matrix(dim, rng)
        sigma = _random_density_matrix(dim, rng)
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
    formulas, NOT a sampled percentile (see estimate_lipschitz_percentile below).
    """
    R = n_layers if reupload else 1
    return R / 2.0


def estimate_lipschitz_percentile(
    X,
    theta,
    forward_circuit,
    n_pairs=300,
    min_dist=1e-4,
    device=None,
    batch_size=64,
    seed=0,
    percentile=95,
):
    """
    TIGHTNESS DIAGNOSTIC ONLY (not a certified bound -- a percentile estimate is not an
    upper bound, since by construction some pairs exceed it.
    Every sampled ratio D_tr/||dx|| should sit at or below analytic_lipschitz_bound();
    a violation indicates a bug in the encoder, not a looser true constant.
    """
    X_np = np.asarray(X, dtype=np.float32)
    n = len(X_np)
    rng = np.random.default_rng(seed)

    idx1 = rng.integers(0, n, size=n_pairs * 3)
    idx2 = rng.integers(0, n, size=n_pairs * 3)
    keep = idx1 != idx2
    idx1, idx2 = idx1[keep], idx2[keep]
    dists = np.linalg.norm(X_np[idx1] - X_np[idx2], axis=1)
    keep2 = dists > min_dist
    idx1, idx2, dists = (
        idx1[keep2][:n_pairs],
        idx2[keep2][:n_pairs],
        dists[keep2][:n_pairs],
    )

    ratios = []
    with torch.no_grad():
        for i in range(0, len(idx1), batch_size):
            x1 = to_torch_batch_x(X_np[idx1[i : i + batch_size]], device=device)
            x2 = to_torch_batch_x(X_np[idx2[i : i + batch_size]], device=device)
            _, rho1 = forward_circuit(x1, theta)
            _, rho2 = forward_circuit(x2, theta)
            for k in range(rho1.shape[0]):
                d_out = float(trace_distance(rho1[k], rho2[k]))
                ratios.append(d_out / dists[i + k])
    ratios = np.array(ratios)
    return {
        "ratios": ratios,
        f"p{percentile}": float(np.percentile(ratios, percentile))
        if len(ratios)
        else float("nan"),
        "max": float(ratios.max()) if len(ratios) else float("nan"),
    }


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
    """F_max(x) = max_{c in K} F(rho(x), rho_c) for every sample in X (known prototypes only)."""
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
    cross-entropy loss (which attacks.py's fgsm_attack targets). This is the more
    faithful attack for testing Proposition 2's "known sample stays accepted" claim.

    Processes X internally in chunks of `batch_size` -- peak memory is bounded by the
    chunk size regardless of len(X), and the same call is safe to retry at a smaller
    `batch_size` after a CUDA OOM (e.g. via a generic run_batched_safely-style wrapper).
    Pass a precomputed `proto_stack`  to
    skip rebuilding it on every call of a sweep.
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
        X_t = (
            to_torch_batch_x(X_np[i : i + batch_size], device=device)
            .detach()
            .requires_grad_(True)
        )
        _, rho = forward_circuit(X_t, theta)
        max_f = max_fidelity_to_prototypes(rho, proto_stack.to(device=rho.device))
        score = (1.0 - max_f).mean()  # nonconformity score, to be maximized
        grad = torch.autograd.grad(score, X_t)[0]

        with torch.no_grad():
            X_adv = X_t + eps * grad.sign()
            if x_min_t is not None or x_max_t is not None:
                X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
        chunks.append(X_adv.detach())
        _clear_cuda_cache_if_needed(device)

    return torch.cat(chunks, dim=0)


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
    eps-ball. Each per-sample trajectory is independent of every other sample, so
    chunking by `batch_size` .
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
# Proposition 3 support: two-sample exchangeability diagnostic
# --------------------------------------------------------------------------------------


def two_sample_discriminability_auroc(X_cal, X_test, seed=0):
    """
    Trains a simple classifier to distinguish calibration vs. test features and reports
    its AUROC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([np.asarray(X_cal), np.asarray(X_test)], axis=0)
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=1000)
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(np.mean(aucs))
