"""Algorithm 2: Conformal Zero-Day Threshold Calibration (CQ-ZDR)."""

import numpy as np
import torch

from scripts.constants import DEFAULT_ALPHA, DEFAULT_BATCH_SIZE
from scripts.quantum_metrics import max_fidelity_to_prototypes, stack_prototypes
from scripts.utils import to_torch_batch_x


def nonconformity_score(
    X, theta, prototypes, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE
):
    """
    Compute one conformal nonconformity score per sample in X: s(x) = 1 - F_max(x)
    (Proposition 3), using the non-squared fidelity convention (Section 1).
    """
    scores = []
    _, proto_stack = stack_prototypes(prototypes)

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i : i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            max_f = max_fidelity_to_prototypes(rho_chunk, proto_stack)
            scores.extend((1.0 - max_f).detach().cpu().tolist())

    return np.asarray(scores, dtype=np.float64)


def min_calibration_size(alpha):
    """Proposition 3, Section 5.1: n >= 1/alpha - 1 is required for a valid threshold to exist."""
    return 1.0 / alpha - 1.0


def calibrate_threshold(
    theta,
    X_cal,
    prototypes,
    forward_circuit,
    alpha=DEFAULT_ALPHA,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Calibrate the CQ-ZDR threshold q from the calibration split (Proposition 3).

    Per Section 5.1: if n is too small for the chosen alpha, no valid threshold
    exists and the method must ABSTAIN (raise) rather than silently fall back to
    the largest calibration score.
    """
    if X_cal is None or len(X_cal) == 0:
        raise ValueError(
            "X_cal must contain at least one sample for conformal calibration (n == 0)"
        )
    if not prototypes:
        raise ValueError("prototypes must be non-empty")

    scores = nonconformity_score(
        X_cal,
        theta,
        prototypes,
        forward_circuit,
        device=device,
        batch_size=batch_size,
    )
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1

    if k > n - 1:
        raise ValueError(
            f"Calibration set too small for alpha={alpha}: n={n}, but Proposition 3 "
            f"requires n >= 1/alpha - 1 = {min_calibration_size(alpha):.1f}. "
            "Abstaining rather than silently using the largest calibration score -- "
            "collect more calibration data or increase alpha."
        )
    k = max(k, 0)
    q = float(scores_sorted[k])
    return q, scores_sorted


def conformal_alpha_sweep(
    theta,
    X_cal,
    X_test_known,
    prototypes,
    forward_circuit,
    alphas=(0.01, 0.05, 0.1, 0.2),
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Team-A Day-16 diagnostic: for each alpha, calibrate q and report the achieved
    (empirical) false-zero-day rate on a held-out known-class set, to check it
    tracks the target alpha (Proposition 3, Eq. 5).
    """
    test_scores = nonconformity_score(
        X_test_known,
        theta,
        prototypes,
        forward_circuit,
        device=device,
        batch_size=batch_size,
    )
    rows = []
    for alpha in alphas:
        q, _ = calibrate_threshold(
            theta,
            X_cal,
            prototypes,
            forward_circuit,
            alpha=alpha,
            device=device,
            batch_size=batch_size,
        )
        empirical_far = float(np.mean(test_scores > q))
        rows.append(
            {
                "alpha": alpha,
                "q": q,
                "empirical_false_alarm_rate": empirical_far,
                "min_calibration_size": min_calibration_size(alpha),
                "n_cal": len(X_cal),
            }
        )
    return rows
