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

"""

import math

import numpy as np
import torch

from scripts.constants import (
    DEFAULT_BETA,
    DEFAULT_MIX,
    DEFAULT_N_PAIRS,
    DEFAULT_NOISE_RATE,
    DEFAULT_L_PHI,
    DEFAULT_P_VALS,
    DEFAULT_SEED,
    INPUT_DIM_D,
    PROP1_RESIDUAL_TOL,
    DEFAULT_BATCH_SIZE,
)
from scripts.quantum_metrics import (
    _as_complex_hermitian,
    fidelity,
    fidelity_pairwise,
    trace_distance,
    max_fidelity_to_prototypes,
    stack_prototypes,
)
from scripts.conformal import nonconformity_score, min_calibration_size  # noqa: F401  (re-exported)
from scripts.utils import get_torch_device, to_torch_batch_x, to_np_y, _as_bound_tensor, _clear_cuda_cache_if_needed

# --------------------------------------------------------------------------------------
# Fidelity-convention self-check
# --------------------------------------------------------------------------------------

def verify_fidelity_convention(dim=4, n_trials=20, seed=DEFAULT_SEED, tol1=1e-6, tol2=1e-3):
    """
    (Section 1, Eq. 1)
    Check non-squared fidelity convention and Fuchs-van de Graaf bounds.
    """
    torch.manual_seed(seed)
    device = get_torch_device()
    max_err, max_viol = 0.0, 0.0    # error and Fuchs-van de Graaf violation

    for _ in range(n_trials):
        rho = _random_density_matrix(dim, device)
        sigma = _random_density_matrix(dim, device)

        # check non-squared fidelity convention
        f = float(fidelity(rho, sigma))
        f_pw = float(fidelity_pairwise(rho.unsqueeze(0), sigma).squeeze())
        max_err = max(max_err, abs(f - f_pw))

        # check Fuchs-van de Graaf bounds
        d = float(trace_distance(rho, sigma))
        lower_bound = 1.0 - f
        upper_bound = math.sqrt(max(1.0 - f * f, 0.0))
        viol = max(lower_bound - d, d - upper_bound, 0.0)
        max_viol = max(max_viol, viol)

    assert max_err < tol1, f"fidelity mismatch: {max_err:.3e} (please ensure non-squared fidelity is used!)"
    assert max_viol < tol2, f"FvG violation: {max_viol:.3e} (please ensure non-squared fidelity is used!)"
    return {"max_pairwise_disagreement": max_err, "max_fvg_violation": max_viol}


def _random_density_matrix(dim, device, dtype=torch.complex128, mix_val=DEFAULT_MIX):
    """
    Sample a random density matrix: Haar-random pure state via QR, then mix with I/d.
    
    Defaults to complex128 so Proposition-1 residual stays near machine precision.
    """
    z = torch.randn(dim, dim, device=device, dtype=dtype)   # random matrix
    q, _ = torch.linalg.qr(z)   # orthonomal columns
    psi = q[:, 0]   # first column as unit vector
    rho = torch.outer(psi, psi.conj())
    mix = float(torch.rand(1, dtype=torch.float64).item()) * mix_val
    I = torch.eye(dim, device=device, dtype=dtype) / dim
    rho = (1.0 - mix) * rho + mix * I
    return 0.5 * (rho + rho.mH) # symmetrize to make it a density matrix

# --------------------------------------------------------------------------------------
# Proposition 1: noise contraction  
# --------------------------------------------------------------------------------------

def depolarizing_channel_global(rho, p):
    """
    (Section 2, Proposition 1, Eq. 2)
    Global n-qubit depolarizing channel.

    Contracts every pair of states by exactly (1-p) under the trace distance.
    """
    if not (0.0 <= float(p) <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    rho = _as_complex_hermitian(rho)
    d = rho.shape[-1]
    I = torch.eye(d, dtype=rho.dtype, device=rho.device)
    while I.ndim < rho.ndim:
        I = I.unsqueeze(0)
    return (1.0 - p) * rho + p * I / d


def check_proposition1(dim, p_values=DEFAULT_P_VALS, n_pairs=DEFAULT_N_PAIRS, seed=DEFAULT_SEED):
    """
    (Section 2, Proposition 1, Eq. 2)
    Compute the residual of Proposition 1 for each p value and return a list of records.
    """
    device = get_torch_device()
    records = []
    for _ in range(n_pairs):
        rho = _random_density_matrix(dim, device)
        sigma = _random_density_matrix(dim, device)
        d_clean = float(trace_distance(rho, sigma))
        for p in p_values:
            d_noisy = float(trace_distance(depolarizing_channel_global(rho, p), depolarizing_channel_global(sigma, p)))
            d_expected = (1.0 - p) * d_clean
            records.append({"p": float(p), "D_tr_clean": d_clean, "D_tr_noisy": d_noisy,
                            "D_tr_expected": d_expected, "residual": abs(d_noisy - d_expected)})
    return records


def assert_proposition1(records, tol=PROP1_RESIDUAL_TOL):
    """
    (Section 2, Proposition 1, Eq. 2)
    Assert Proposition 1: the depolarizing channel contracts the trace distance by exactly (1-p) with a tolerance of 1e-10.
    """
    max_err = max(r["residual"] for r in records)
    assert max_err <= tol, (
        f"Proposition 1 residual {max_err:.3e} exceeds tolerance {tol:.1e} while it should be almost exactly zero."
    )
    return max_err


def _mean_pairwise_td(states):
    """
    Mean trace distance over all unique pairs of states.
    """
    vals = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            vals.append(float(trace_distance(states[i], states[j])))
    return float(np.mean(vals)) if vals else 0.0


def regulariser_contraction_curve(rho_list, p_grid=None):
    """
    (Team-B Day-15): To validate p as a regularizer.

    Sweep p and check mean pairwise trace distance shrinks as (1-p) * D_tr(p=0).
    """
    if p_grid is None:
        p_grid = np.linspace(0.0, 1.0, 11)
    p_grid = np.asarray(p_grid, dtype=float)
    d0 = _mean_pairwise_td(rho_list)    # D_tr(p=0)
    d_p = np.asarray([
        _mean_pairwise_td([depolarizing_channel_global(r, float(p)) for r in rho_list])
        for p in p_grid
    ], dtype=float) # D_tr(p)
    d_theory = (1.0 - p_grid) * d0  # (1-p) * D_tr(p=0)
    return p_grid, d0, d_p, d_theory

# --------------------------------------------------------------------------------------
# Proposition 2: adversarial-novelty separation (F_max / F_in / F_out)
# --------------------------------------------------------------------------------------

def collect_known_fidelities(X, y, theta, prototypes, forward_circuit, device, batch_size):
    """
    Return non-squared F_own and F_max on known-labeled samples.
    """
    y_np = to_np_y(y).astype(int)
    class_ids, proto_stack = stack_prototypes(prototypes, device=device)
    id_to_pos = {c: i for i, c in enumerate(class_ids)}

    f_own_list, f_max_list = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            y_chunk = y_np[i:i + batch_size]

            f_max_batch = max_fidelity_to_prototypes(rho_chunk, proto_stack)
            for j in range(rho_chunk.shape[0]):
                c = int(y_chunk[j])
                if c not in id_to_pos:
                    continue
                f_own_val = fidelity(rho_chunk[j], prototypes[c])
                f_own_list.append(float(f_own_val.item() if torch.is_tensor(f_own_val) else f_own_val))
                f_max_list.append(float(f_max_batch[j].item()))
    return np.asarray(f_own_list, dtype=np.float64), np.asarray(f_max_list, dtype=np.float64)


def collect_zeroday_fmax(X, theta, prototypes, forward_circuit, device, batch_size):
    """
    Return non-squared F_max of zero-day samples vs known prototypes only.
    """
    f_max_list = nonconformity_score(X, theta, prototypes, forward_circuit, device=device, batch_size=batch_size)
    return 1.0 - f_max_list


def _percentile_f(f_vals, percentile):
    """
    Returns the percentile-th percentile of f_vals.
    """
    return float(np.percentile(f_vals, percentile))


def worst_case_f_in(f_max_known, percentile=0.0):
    """
    (Section 3.1, Proposition 2, A2)
    Returns the percentile-th percentile of F_max(x) over known samples.
    """
    return _percentile_f(f_max_known, percentile)


def quantile_f_out(f_max_zeroday, beta=DEFAULT_BETA):
    """
    (Section 3.5, Proposition 2, A3', Corollary 1)
    Returns the (1-beta)-quantile of F_max(z) over zero-day samples.
    """
    return _percentile_f(f_max_zeroday, 100.0 * (1.0 - beta))

# --------------------------------------------------------------------------------------
# Proposition 2: gap Delta, epsilon*, quantile-relaxed epsilon_beta
# --------------------------------------------------------------------------------------

def proposition2_gap(F_in, F_out):
    """
    (Section 3.1, Proposition 2, Eq. 3)
    Delta and eps* ingredients from non-squared fidelities.
    """
    d_in = float(np.sqrt(max(0.0, 1.0 - F_in ** 2)))
    d_out = float(1.0 - F_out)
    delta = d_out - d_in
    return d_in, d_out, delta


def proposition2_epsilon_star(delta, p=DEFAULT_NOISE_RATE, L_phi=DEFAULT_L_PHI):
    """
    (Section 3.2, Proposition 2, Eq. 4)
    Proposition-2 predicted separable budget (epsilon*) from distance gap, noise rate, and L_phi.
    """
    denom = 2.0 * (1.0 - p) * L_phi
    if denom <= 0:
        return float("nan")
    return float(delta / denom)


def proposition2_epsilon_beta(F_in, f_max_zeroday, L_phi=DEFAULT_L_PHI, p=DEFAULT_NOISE_RATE, beta=DEFAULT_BETA):
    """
    (Section 3.5, Proposition 2, A3', Corollary 1)
    Quantile-relaxed Proposition-2 separation budget (epsilon_beta).
    """
    F_out_beta = quantile_f_out(f_max_zeroday, beta=beta)
    _, _, delta_beta = proposition2_gap(F_in, F_out_beta)
    eps_beta = proposition2_epsilon_star(delta_beta, p=p, L_phi=L_phi)
    return F_out_beta, delta_beta, eps_beta

# --------------------------------------------------------------------------------------
# FGSM + PGD attacks that directly target the CQ-ZDR novelty score
# --------------------------------------------------------------------------------------

def fgsm_attack_nonconformity(X, theta, prototypes, forward_circuit, eps, device=None,
                               x_min=None, x_max=None, batch_size=DEFAULT_BATCH_SIZE, proto_stack=None):
    """
    (Section 3.1, Proposition 2, A1)
    Batched one-step FGSM that maximizes the CQ-ZDR novelty score s(x) = 1 - F_max(x) directly,
    i.e. attacks the CQ-ZDR accept/reject boundary.

    - Chunks X by `batch_size` (OOM-safe).
    - Pass `proto_stack` to reuse precomputed prototypes across a sweep.
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
        X_t = to_torch_batch_x(X_np[i:i + batch_size], device=device).detach().requires_grad_(True)
        _, rho = forward_circuit(X_t, theta)
        max_f = max_fidelity_to_prototypes(rho, proto_stack.to(device=rho.device))
        score = (1.0 - max_f).mean()    # CQ-ZDR novelty score, to be maximized
        grad = torch.autograd.grad(score, X_t)[0]

        with torch.no_grad():
            X_adv = X_t + eps * grad.sign()
            if x_min_t is not None or x_max_t is not None:
                X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
        chunks.append(X_adv.detach())
        _clear_cuda_cache_if_needed(device)

    return torch.cat(chunks, dim=0)


def pgd_attack_nonconformity(X, theta, prototypes, forward_circuit, eps, alpha, steps,
                              device=None, x_min=None, x_max=None, random_start=True,
                              batch_size=DEFAULT_BATCH_SIZE, proto_stack=None):
    """
    (Section 3.1, Proposition 2, A1)
    Batched multi-step PGD with L_inf projection onto the eps-ball around X.

    - Chunks X by `batch_size` (OOM-safe).
    - Pass `proto_stack` to reuse precomputed prototypes across a sweep.
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
        X0 = to_torch_batch_x(X_np[i:i + batch_size], device=device).detach()
        X_adv = X0 + torch.empty_like(X0).uniform_(-eps, eps) if random_start else X0.clone()
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
                X_adv = torch.clamp(X_adv, min=X0 - eps, max=X0 + eps)  # fused L_inf-ball projection
                if x_min_t is not None or x_max_t is not None:
                    X_adv = torch.clamp(X_adv, min=x_min_t, max=x_max_t)
            X_adv = X_adv.detach()

        chunks.append(X_adv)
        _clear_cuda_cache_if_needed(device)

    return torch.cat(chunks, dim=0)

# --------------------------------------------------------------------------------------
# Proposition 3: two-sample exchangeability diagnostic
# --------------------------------------------------------------------------------------

def two_sample_discriminability_auroc(X_cal, X_test, seed=DEFAULT_SEED):
    """
    (Section 5, Proposition 3)
    Trains a simple classifier to distinguish calibration vs. test features and reports its AUROC.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X = np.concatenate([np.asarray(X_cal), np.asarray(X_test)], axis=0)
    y = np.concatenate([np.zeros(len(X_cal)), np.ones(len(X_test))])
    clf = LogisticRegression(max_iter=1000)
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(np.mean(aucs))

# --------------------------------------------------------------------------------------
# Section 4: analytic Lipschitz bound for angle encoding (Lemma 1)
# --------------------------------------------------------------------------------------

def analytic_lipschitz_bound(n_layers, reupload=True):
    """
    (Section 3.1, A1) - (Section 4, Lemma 1)
    Lemma 1: For single-qubit-per-feature angle encoding, D_tr(|psi(x)>,|psi(x')>) <= (R/2) ||x-x'||_2,
    where R = n_layers if reupload else 1.

    Returns the Lipschitz constant R/2.
    """
    R = n_layers if reupload else 1
    return R / 2.0


def estimate_lipschitz_percentile(X, theta, forward_circuit, n_pairs=300, min_dist=1e-4,
                                   device=None, batch_size=64, seed=0, percentile=95):
    """
    (Section 3.1, A1) - (Section 4, Lemma 1)
    Estimates the Lipschitz constant by sampling pairs (x, x') and computing D_tr/||x-x'||_2.    

    Every sampled pair should satisfy Lemma 1,
    and any violation indicates a bug in the encoder, not a looser true constant.
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
    idx1, idx2, dists = idx1[keep2][:n_pairs], idx2[keep2][:n_pairs], dists[keep2][:n_pairs]

    ratios = []
    with torch.no_grad():
        for i in range(0, len(idx1), batch_size):
            x1 = to_torch_batch_x(X_np[idx1[i:i + batch_size]], device=device)
            x2 = to_torch_batch_x(X_np[idx2[i:i + batch_size]], device=device)
            _, rho1 = forward_circuit(x1, theta)
            _, rho2 = forward_circuit(x2, theta)
            for k in range(rho1.shape[0]):
                d_out = float(trace_distance(rho1[k], rho2[k]))
                ratios.append(d_out / dists[i + k])
    ratios = np.array(ratios)
    return {
        "ratios": ratios,
        f"p{percentile}": float(np.percentile(ratios, percentile)) if len(ratios) else float("nan"),
        "max": float(ratios.max()) if len(ratios) else float("nan"),
    }


def check_lipschitz_tightness(diag, l_phi_bound):
    """
    (Section 4, Lemma 1)
    Flags whether any sampled ratio exceeds the analytic bound (encoder-bug signal).
    """
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