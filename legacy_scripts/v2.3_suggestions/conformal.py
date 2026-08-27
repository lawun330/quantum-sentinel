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


def threshold_from_scores(scores, alpha=DEFAULT_ALPHA):
    """
    Shared conformal-quantile logic (Proposition 3): given an already-computed array of
    calibration nonconformity scores, returns (q, scores_sorted) -- abstaining (raising)
    if n is too small for alpha (Section 5.1), rather than silently using the largest
    score.

    This is factored out of calibrate_threshold() so that any caller who already has
    precomputed scores (e.g. scripts.cache's cached_calibrate_threshold, or the
    per-class partitions in class_conditional_calibrate below) shares EXACTLY the same
    abstention/quantile logic and can never silently diverge from it.
    """
    scores_sorted = np.sort(np.asarray(scores, dtype=np.float64))
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
    return float(scores_sorted[k]), scores_sorted


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
    Calibrate the CQ-ZDR threshold q from the calibration split (Proposition 3, marginal
    coverage -- one threshold shared by every known class).
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
    return threshold_from_scores(scores, alpha=alpha)


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

    Both the calibration and test nonconformity scores are computed ONCE up front and
    reused across every alpha in the sweep.
    """
    test_scores = nonconformity_score(
        X_test_known,
        theta,
        prototypes,
        forward_circuit,
        device=device,
        batch_size=batch_size,
    )
    cal_scores = nonconformity_score(
        X_cal, theta, prototypes, forward_circuit, device=device, batch_size=batch_size
    )
    rows = []
    for alpha in alphas:
        q, _ = threshold_from_scores(cal_scores, alpha=alpha)
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


# --------------------------------------------------------------------------------------
# Class-conditional (Mondrian) calibration
# --------------------------------------------------------------------------------------
#
# Section 5.1 of the theory doc distinguishes MARGINAL coverage (calibrate_threshold
# above: P(s(X) > q) <= alpha, averaged over the whole known-class distribution) from
# CLASS-CONDITIONAL / Mondrian coverage (P(s(X) > q_c | Y=c) <= alpha, separately for
# every class c). Marginal coverage does not guarantee any individual class is well
# covered -- if one class's fidelity distribution is naturally more diffuse than
# another's (e.g. Class A concentrated at F_max~0.96-0.99 vs. Class B at ~0.79-0.82),
# a single global threshold calibrated on the pooled distribution can systematically
# over-reject the diffuse class and under-reject the concentrated one relative to the
# target alpha. class_conditional_calibrate() computes one threshold q_c per class
# instead, so each class gets its own false-alarm guarantee.


def class_conditional_calibrate(
    theta,
    X_cal,
    y_cal,
    prototypes,
    forward_circuit,
    alpha=DEFAULT_ALPHA,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
    fallback="global",
):
    """
    Mondrian (label-conditional) conformal calibration: one threshold q_c per known
    class, using only that class's calibration samples.

    Per Section 5.1, a per-class guarantee additionally requires the minimum-
    calibration-size condition to hold WITHIN EVERY CLASS (not just in total). If a
    class doesn't have enough calibration samples for the requested alpha:
      - fallback="global" (default): use the GLOBAL (marginal) threshold for that
        class only, with the reason recorded in `meta` -- keeps the pipeline usable for
        rare classes while being explicit about which classes get the stronger
        per-class guarantee and which fall back to the weaker marginal one.
      - fallback="abstain": raise for that class, matching calibrate_threshold's
        all-or-nothing behavior -- use this if mixing guarantee strengths across
        classes is not acceptable for your reporting.

    Returns (q_by_class: dict[class_id -> float], meta: dict[class_id -> dict]), where
    meta also contains a "_global" entry with the marginal threshold for reference.
    """
    if X_cal is None or len(X_cal) == 0:
        raise ValueError(
            "X_cal must contain at least one sample for conformal calibration (n == 0)"
        )
    if not prototypes:
        raise ValueError("prototypes must be non-empty")
    if fallback not in ("global", "abstain"):
        raise ValueError(f"fallback must be 'global' or 'abstain', got {fallback!r}")

    y_cal = np.asarray(y_cal)
    class_ids = sorted(prototypes.keys())

    global_scores = nonconformity_score(
        X_cal, theta, prototypes, forward_circuit, device=device, batch_size=batch_size
    )
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
            meta[c] = {
                "n": n_c,
                "status": f"insufficient for alpha={alpha} -- used GLOBAL threshold ({e})",
                "q": global_q,
            }
            q_by_class[c] = global_q

    meta["_global"] = {
        "q": global_q,
        "n": len(global_scores),
        "status": "marginal (Proposition 3 baseline)",
    }
    return q_by_class, meta


def per_class_empirical_far(
    theta,
    X_known,
    y_known,
    prototypes,
    forward_circuit,
    q_by_class,
    device=None,
    batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Measures the empirical false-alarm rate SEPARATELY for each true class on a
    held-out known-class set (e.g. the test split), under an arbitrary per-class (or
    constant, if you pass the same q for every class) threshold map. Use this to
    compare the marginal vs. Mondrian calibrations' per-class fairness directly:
    call once with q_by_class = {c: global_q for c in classes} and once with the
    output of class_conditional_calibrate(), and compare the spread across classes.
    """
    y_known = np.asarray(y_known)
    scores = nonconformity_score(
        X_known,
        theta,
        prototypes,
        forward_circuit,
        device=device,
        batch_size=batch_size,
    )
    rows = []
    for c in sorted(prototypes.keys()):
        mask = y_known == c
        n_c = int(mask.sum())
        if n_c == 0:
            rows.append(
                {
                    "class": c,
                    "n": 0,
                    "q": q_by_class.get(c, float("nan")),
                    "empirical_far": float("nan"),
                }
            )
            continue
        far_c = float(np.mean(scores[mask] > q_by_class[c]))
        rows.append({"class": c, "n": n_c, "q": q_by_class[c], "empirical_far": far_c})
    return rows
