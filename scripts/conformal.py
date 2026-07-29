"""Algorithm 2: Conformal Zero-Day Threshold Calibration (CQ-ZDR)."""

import numpy as np
import torch

from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_ALPHA
from scripts.quantum_metrics import max_fidelity_to_prototypes, stack_prototypes
from scripts.utils import to_torch_batch_x


def nonconformity_score(X, theta, prototypes, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    (Section 5, Proposition 3)
    Compute one conformal nonconformity score per sample in X: s(x) = 1 - F_max(x).

    Runs forward_circuit in internal mini-batches and compares each rho(x)
    to the full prototype bank (max over classes).
    """
    scores = []
    _, proto_stack = stack_prototypes(prototypes)

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            max_f = max_fidelity_to_prototypes(rho_chunk, proto_stack)
            scores.extend((1.0 - max_f).detach().cpu().tolist())

    return np.asarray(scores, dtype=np.float64)


def min_calibration_size(alpha):
    """
    (Section 5.1, Proposition 3)
    n >= (1/alpha) - 1 is required for a valid threshold to exist.
    """
    return ((1.0 / alpha) - 1.0)


def threshold_from_scores(scores, alpha=DEFAULT_ALPHA):
    """
    (Section 5.1, Proposition 3)
    Sort already-computed array of calibration nonconformity scores, and return the threshold q.

    Abstain if n < min_calibration_size(alpha), don't fall back to the max score.
    """
    scores_sorted = np.sort(np.asarray(scores, dtype=np.float64))
    n = len(scores_sorted)
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1

    if k > n - 1:
        raise ValueError(
            f"Calibration set too small for alpha={alpha}: n={n}, but Proposition 3 "
            f"requires n >= (1/alpha) - 1 = {min_calibration_size(alpha):.2f}. "
            "Abstaining rather than using the largest calibration score -- "
            "collect more calibration data or increase alpha."
        )
    k = max(k, 0)
    q = float(scores_sorted[k])
    return q, scores_sorted


def calibrate_threshold(theta, X_cal, prototypes, forward_circuit, alpha=DEFAULT_ALPHA, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    (Section 5.1, Proposition 3)
    Calibrate the CQ-ZDR threshold, q, from the calibration split.

    Abstain if n < min_calibration_size(alpha), don't fall back to the max score.
    """
    if X_cal is None or len(X_cal) == 0:
        raise ValueError("X_cal must contain at least one sample for conformal calibration (n == 0)")
    if not prototypes:
        raise ValueError("prototypes must be non-empty")

    scores = nonconformity_score(
        X_cal, theta, prototypes, forward_circuit, device=device, batch_size=batch_size,
    )
    q, scores_sorted = threshold_from_scores(scores, alpha=alpha)

    return q, scores_sorted


def conformal_alpha_sweep(theta, X_cal, X_test_known, prototypes, forward_circuit,
                           alphas=(0.01, 0.05, 0.1, 0.2), device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    (Section 5, Proposition 3, Eq. 5) - (Team-A Day-16 diagnostic)

    For each alpha, calibrate q and report the achieved (empirical) false-zero-day rate
    on a held-out known-class set, to check it tracks the target alpha.
    """
    test_scores = nonconformity_score(X_test_known, theta, prototypes, forward_circuit,
                                       device=device, batch_size=batch_size)
    cal_scores = nonconformity_score(X_cal, theta, prototypes, forward_circuit,
                                      device=device, batch_size=batch_size)
    rows = []
    for alpha in alphas:
        q, _ = threshold_from_scores(cal_scores, alpha=alpha)
        empirical_far = float(np.mean(test_scores > q))
        rows.append({"alpha": alpha, "q": q, "empirical_false_alarm_rate": empirical_far,
                     "min_calibration_size": min_calibration_size(alpha), "n_cal": len(X_cal)})
    return rows

# --------------------------------------------------------------------------------------
# Class-conditional (Mondrian) calibration
# --------------------------------------------------------------------------------------
#
# (Section 5.1): Marginal vs. Class-Conditional (Mondrian) Coverage
# -----------------------------------------------------------------
# - Marginal (`conformal.calibrate_threshold()`)
#   - Guarantees `P(s(X) > q) <= alpha` overall.
#   - A pooled global q can over-reject diffuse classes and under-reject concentrated ones.
#
# - Class-Conditional (`conformal.class_conditional_calibrate()`)
#   - Computes a threshold q_c per class so `P(s(X) > q_c | Y=c) <= alpha`, ensuring per-class false-alarm guarantees.
#   - Each class gets its own false-alarm guarantee.

def class_conditional_calibrate(theta, X_cal, y_cal, prototypes, forward_circuit, alpha=DEFAULT_ALPHA,
                                 device=None, batch_size=DEFAULT_BATCH_SIZE, fallback="global"):
    """
    (Section 5.1, Proposition 3)
    Mondrian (label-conditional) calibration: one threshold q_c per known class, using only that class's calibration samples.

    Per-class guarantee also needs `conformal.min_calibration_size(alpha)` WITHIN EVERY CLASS
    (not just in total). If a class is too small for alpha:

    - fallback="global" (default): use the marginal/global q for that class only;
        record reason in `meta` (shows which classes get a per-class guarantee and which get the marginal one).

    - fallback="abstain": raise for that class (same all-or-nothing as
        `conformal.calibrate_threshold()`); use when mixing guarantee strengths is not OK.

    Returns (q_by_class: dict[class_id -> float], meta: dict[class_id -> dict]);
    - meta also has "_global" with the marginal threshold for reference.
    """
    if X_cal is None or len(X_cal) == 0:
        raise ValueError("X_cal must contain at least one sample for conformal calibration (n == 0)")
    if not prototypes:
        raise ValueError("prototypes must be non-empty")
    if fallback not in ("global", "abstain"):
        raise ValueError(f"fallback must be 'global' or 'abstain', got {fallback!r}")

    y_cal = np.asarray(y_cal)
    class_ids = sorted(prototypes.keys())

    global_scores = nonconformity_score(X_cal, theta, prototypes, forward_circuit,
                                         device=device, batch_size=batch_size)
    global_q, _ = threshold_from_scores(global_scores, alpha=alpha)

    q_by_class, meta = {}, {}
    for c in class_ids:
        mask = y_cal == c
        n_c = int(mask.sum())
        class_scores = global_scores[mask]
        try:
            q_c, _ = threshold_from_scores(class_scores, alpha=alpha)
            meta[c] = {"n": n_c, "status": "ok", "q": q_c}
            q_by_class[c] = q_c
        except ValueError as e:
            if fallback == "abstain":
                raise ValueError(f"Class {c}: {e}") from e
            meta[c] = {"n": n_c, "status": f"insufficient for alpha={alpha} -- used GLOBAL threshold ({e})",
                       "q": global_q}
            q_by_class[c] = global_q

    meta["_global"] = {"q": global_q, "n": len(global_scores), "status": "marginal (Proposition 3 baseline)"}
    return q_by_class, meta


def per_class_empirical_far(theta, X_known, y_known, prototypes, forward_circuit,
                             q_by_class, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Measure empirical false-alarm rate per true class on a held-out known-class set (e.g. test),
    under a threshold map `q_by_class`.

    To compare marginal vs Mondrian per-class fairness,
    - call once with `q_by_class = {c: global_q for c in classes}` for marginal calibration
    - call once with `conformal.class_conditional_calibrate()` output for Mondrian calibration
    - compare the spread across classes.
    """
    y_known = np.asarray(y_known)
    scores = nonconformity_score(X_known, theta, prototypes, forward_circuit,
                                  device=device, batch_size=batch_size)
    rows = []
    for c in sorted(prototypes.keys()):
        mask = y_known == c
        n_c = int(mask.sum())
        if n_c == 0:
            rows.append({
                "class": c,
                "n": 0,
                "q": q_by_class.get(c, float("nan")),
                "empirical_far": float("nan"),
            })
            continue
        far_c = float(np.mean(scores[mask] > q_by_class[c]))
        rows.append({
            "class": c,
            "n": n_c,
            "q": q_by_class[c],
            "empirical_far": far_c,
        })
    return rows