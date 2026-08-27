"""
=== FORWARD-PASS CACHE FOR QS-NET / CQ-ZDR ===

Caches circuit outputs (z, rho) once per dataset key so diagnostics can reuse them
without re-running the expensive forward pass.

Original -> uses; discards
- predict_labels                  -> z; discards rho
- nonconformity_score             -> rho; discards z
- calibrate_threshold             -> rho (via scores); discards z
- conformal_alpha_sweep           -> rho (via scores); discards z
- f_max_batch                     -> rho; discards z
- qsnet_infer_batch               -> rho; discards z
- qsnet_infer_batch_per_class     -> rho; discards z
- estimate_lipschitz_percentile   -> rho + X (pair distances); discards z

> ForwardCache runs the circuit ONCE per (key, theta, p) and stores both z and rho
(CPU by default to keep GPU free for training/attacks).
> Stores real encoder outputs on real samples. No synthetic states.
"""

import numpy as np
import torch

from scripts.conformal import (
    min_calibration_size,
    threshold_from_scores,
)
from scripts.constants import DEFAULT_ALPHA, DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.inference import algo3_postprocess_f_mat, f_maps_from_f_mat
from scripts.quantum_metrics import (
    all_fidelities_to_prototypes,
    fidelity_pairwise,
    max_fidelity_to_prototypes,
    stack_prototypes,
    trace_distance,
)
from scripts.utils import expectations_to_tensor, to_np_y, to_torch_batch_x

# --------------------------------------------------------------------------------------
# ForwardCache
# --------------------------------------------------------------------------------------


class ForwardCache:
    """
    Store {"z": (N, n_qubits), "rho": (N, d, d)} for named datasets (CPU by default).
    Downstream helpers move only the active chunk to `device` for cheap math.
    """

    def __init__(self, store_device="cpu"):
        self.store_device = store_device
        self._entries = {}  # key -> {"z", "rho", "n", "p", "X"}

    def compute(
        self,
        key,
        X,
        theta,
        forward_circuit,
        device=None,
        batch_size=64,
        p=None,
        force=False,
    ):
        """Run circuit once over X; cache (z, rho) under `key` (idempotent unless force=True)."""
        if key in self._entries and not force:
            return self._entries[key]

        X_np = np.asarray(X, dtype=np.float32)
        z_chunks, rho_chunks = [], []
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                x_chunk = to_torch_batch_x(X_np[i : i + batch_size], device=device)
                z_chunk, rho_chunk = forward_circuit(x_chunk, theta)
                z_chunks.append(expectations_to_tensor(z_chunk).to(self.store_device))
                rho_chunks.append(rho_chunk.detach().to(self.store_device))

        entry = {
            "z": torch.cat(z_chunks, dim=0) if z_chunks else torch.empty(0),
            "rho": torch.cat(rho_chunks, dim=0) if rho_chunks else torch.empty(0),
            "n": len(X_np),
            "p": p,
            "X": X_np,
        }
        self._entries[key] = entry
        return entry

    def get(self, key):
        return self._entries[key]

    def __contains__(self, key):
        return key in self._entries

    def slice(self, key, idx):
        """Return a new (uncached) entry dict restricted to the given sample indices."""
        entry = self._entries[key]
        idx = np.asarray(idx)
        return {
            "z": entry["z"][idx],
            "rho": entry["rho"][idx],
            "n": len(idx),
            "p": entry["p"],
            "X": entry["X"][idx],
        }

    def clear(self, key=None):
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def memory_report(self):
        rows = []
        for key, entry in self._entries.items():
            nbytes = (
                entry["rho"].element_size() * entry["rho"].nelement()
                + entry["z"].element_size() * entry["z"].nelement()
            )
            rows.append(
                {
                    "key": key,
                    "n_samples": entry["n"],
                    "p": entry["p"],
                    "MB": nbytes / 1e6,
                }
            )
        return rows


# --------------------------------------------------------------------------------------
# Cached conformal / F_max helpers (rho; no circuit)
# --------------------------------------------------------------------------------------


def cached_f_max(entry, prototypes, device=None, batch_size=256):
    """F_max(x) for every sample in entry['rho']."""
    _, proto_stack = stack_prototypes(prototypes)
    rho_all = entry["rho"]
    n = len(rho_all)
    out = np.empty(n, dtype=np.float64)
    for i in range(0, n, batch_size):
        rho_chunk = rho_all[i : i + batch_size]
        rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
        max_f = max_fidelity_to_prototypes(
            rho_chunk, proto_stack.to(device=rho_chunk.device)
        )
        out[i : i + batch_size] = max_f.detach().cpu().numpy()
    return out


def cached_nonconformity_scores(entry, prototypes, device=None, batch_size=256):
    """s(x) = 1 - F_max(x)."""
    return 1.0 - cached_f_max(entry, prototypes, device=device, batch_size=batch_size)


def cached_calibrate_threshold(
    entry, prototypes, alpha=DEFAULT_ALPHA, device=None, batch_size=256
):
    """Calibrate conformal q from cached scores."""
    scores = cached_nonconformity_scores(
        entry, prototypes, device=device, batch_size=batch_size
    )
    return threshold_from_scores(scores, alpha=alpha)


def conformal_alpha_sweep_from_scores(
    cal_scores, test_scores, alphas=(0.01, 0.05, 0.1, 0.2)
):
    """
    Alpha sweep of q and empirical FAR from already-computed score arrays (no fidelity pass).
    """
    cal_scores = np.asarray(cal_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    rows = []
    for alpha in alphas:
        q, _ = threshold_from_scores(cal_scores, alpha=alpha)
        rows.append(
            {
                "alpha": float(alpha),
                "q": q,
                "empirical_false_alarm_rate": float(np.mean(test_scores > q)),
                "min_calibration_size": min_calibration_size(alpha),
                "n_cal": len(cal_scores),
            }
        )
    return rows


def cached_conformal_alpha_sweep(
    entry_cal,
    entry_test_known,
    prototypes,
    alphas=(0.01, 0.05, 0.1, 0.2),
    device=None,
    batch_size=256,
    cal_scores=None,
    test_scores=None,
):
    """
    Alpha sweep of q and empirical FAR using cached cal/test scores.

    Pass precomputed `cal_scores` / `test_scores` (e.g. from calibrate + Global infer).
    """
    if cal_scores is None:
        cal_scores = cached_nonconformity_scores(
            entry_cal, prototypes, device=device, batch_size=batch_size
        )
    if test_scores is None:
        test_scores = cached_nonconformity_scores(
            entry_test_known, prototypes, device=device, batch_size=batch_size
        )
    return conformal_alpha_sweep_from_scores(cal_scores, test_scores, alphas=alphas)


# --------------------------------------------------------------------------------------
# Cached inference helpers (z / rho; no circuit)
# --------------------------------------------------------------------------------------


def cached_predict_labels(entry, y_true, head, device=None, batch_size=1024):
    """
    Cached variant of `inference.predict_labels()`.

    Batched prediction of class labels using cached circuit expectations + classical head.
    """
    head_device = next(head.parameters()).device if device is None else device
    z_all = entry["z"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(z_all), batch_size):
            z_chunk = z_all[i : i + batch_size].to(device=head_device)
            preds.append(head(z_chunk).argmax(dim=1).cpu().numpy())
    return to_np_y(y_true).astype(int), (
        np.concatenate(preds) if preds else np.array([], dtype=int)
    )


def cached_qsnet_infer(
    entry,
    prototypes,
    q,
    p=DEFAULT_NOISE_RATE,
    L_phi=None,
    Cf=DEFAULT_CF,
    zero_day=ZERO_DAY,
    device=None,
    batch_size=256,
    return_f_maps=True,
):
    """
    Cached variant of `inference.qsnet_infer_batch()`.

    Algo3 pipeline: p(x) -> per-class fidelity -> nearest class c_star -> conformal test with global q -> label + certified radius.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (see theory.analytic_lipschitz_bound)")
    class_ids, proto_stack = stack_prototypes(prototypes, device=device)
    rho_all = entry["rho"]
    n = len(rho_all)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            rho_chunk = rho_all[i : i + batch_size]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            f_mat = all_fidelities_to_prototypes(rho_chunk, proto_stack)
            labels_b, radii_b, scores_b = algo3_postprocess_f_mat(
                f_mat,
                class_ids,
                q=q,
                p=p,
                L_phi=L_phi,
                Cf=Cf,
                zero_day=zero_day,
            )
            labels.append(labels_b.cpu().numpy())
            radii.append(radii_b.cpu().numpy())
            scores.append(scores_b.cpu().numpy())
            if return_f_maps:
                f_maps.extend(f_maps_from_f_mat(f_mat, class_ids))

    labels_out = np.concatenate(labels) if labels else np.array([], dtype=int)
    radii_out = np.concatenate(radii) if radii else np.array([], dtype=np.float64)
    scores_out = np.concatenate(scores) if scores else np.array([], dtype=np.float64)
    return labels_out, radii_out, scores_out, (f_maps if return_f_maps else [])


def cached_qsnet_infer_per_class(
    entry,
    prototypes,
    q_by_class,
    p=DEFAULT_NOISE_RATE,
    L_phi=None,
    Cf=DEFAULT_CF,
    zero_day=ZERO_DAY,
    device=None,
    batch_size=256,
    return_f_maps=True,
):
    """
    Cached variant of `inference.qsnet_infer_batch_per_class()`.

    Algo3 pipeline: p(x) -> per-class fidelity -> nearest class c_star -> conformal test with per-class q_c -> label + certified radius.

    See `conformal.class_conditional_calibrate()` for how to build q_by_class.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (see theory.analytic_lipschitz_bound)")
    class_ids = sorted(prototypes.keys())
    missing = [c for c in class_ids if c not in q_by_class]
    if missing:
        raise ValueError(f"q_by_class is missing thresholds for classes: {missing}")
    _, proto_stack = stack_prototypes(prototypes, device=device)
    rho_all = entry["rho"]
    n = len(rho_all)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            rho_chunk = rho_all[i : i + batch_size]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            f_mat = all_fidelities_to_prototypes(rho_chunk, proto_stack)
            labels_b, radii_b, scores_b = algo3_postprocess_f_mat(
                f_mat,
                class_ids,
                q=None,
                p=p,
                L_phi=L_phi,
                Cf=Cf,
                zero_day=zero_day,
                q_by_class=q_by_class,
            )
            labels.append(labels_b.cpu().numpy())
            radii.append(radii_b.cpu().numpy())
            scores.append(scores_b.cpu().numpy())
            if return_f_maps:
                f_maps.extend(f_maps_from_f_mat(f_mat, class_ids))

    labels_out = np.concatenate(labels) if labels else np.array([], dtype=int)
    radii_out = np.concatenate(radii) if radii else np.array([], dtype=np.float64)
    scores_out = np.concatenate(scores) if scores else np.array([], dtype=np.float64)
    return labels_out, radii_out, scores_out, (f_maps if return_f_maps else [])


# --------------------------------------------------------------------------------------
# Cached Lipschitz diagnostic (rho pairs; no circuit)
# --------------------------------------------------------------------------------------


def cached_lipschitz_percentile(
    entry, n_pairs=300, min_dist=1e-4, seed=0, percentile=95
):
    """
    (Section 3.1, A1) - (Section 4, Lemma 1)
    Cached variant of `theory.estimate_lipschitz_percentile()`.

    Estimates the Lipschitz constant by sampling pairs (x, x') and computing D_tr/||x-x'||_2.
    """
    X_np, rho_all = entry["X"], entry["rho"]
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

    ratios = (
        np.array(
            [
                float(trace_distance(rho_all[i1], rho_all[i2])) / d
                for i1, i2, d in zip(idx1, idx2, dists)
            ]
        )
        if len(idx1)
        else np.array([])
    )

    return {
        "ratios": ratios,
        f"p{percentile}": float(np.percentile(ratios, percentile))
        if len(ratios)
        else float("nan"),
        "max": float(ratios.max()) if len(ratios) else float("nan"),
    }


def cached_class_conditional_calibrate(
    entry,
    y_cal,
    prototypes,
    alpha=DEFAULT_ALPHA,
    device=None,
    batch_size=256,
    fallback="global",
    scores=None,
):
    """
    Cached variant of `conformal.class_conditional_calibrate()`.

    Mondrian (label-conditional) calibration: one threshold q_c per known class,
    using only that class's calibration samples -- computed from a cached `rho`
    entry, so no circuit forward pass is re-run.

    Pass precomputed `scores` (same order as `entry` / `y_cal`) to skip an extra
    nonconformity pass when cal scores were already computed for global q.

    Per-class guarantee also needs `conformal.min_calibration_size(alpha)` WITHIN
    EVERY CLASS (not just in total). If a class is too small for alpha:

    - fallback="global" (default): use the marginal/global q for that class only;
        record reason in `meta`.
    - fallback="abstain": raise for that class (same all-or-nothing as
        `conformal.calibrate_threshold()`).

    Returns (q_by_class: dict[class_id -> float], meta: dict[class_id -> dict]);
    meta also has "_global" with the marginal threshold for reference.
    """
    if fallback not in ("global", "abstain"):
        raise ValueError(f"fallback must be 'global' or 'abstain', got {fallback!r}")

    y_cal = np.asarray(y_cal)
    class_ids = sorted(prototypes.keys())

    if scores is None:
        global_scores = cached_nonconformity_scores(
            entry, prototypes, device=device, batch_size=batch_size
        )
    else:
        global_scores = np.asarray(scores, dtype=np.float64)
        if len(global_scores) != len(y_cal):
            raise ValueError(
                f"scores length {len(global_scores)} != y_cal length {len(y_cal)}"
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


def cached_per_class_far(
    entry, y_known, prototypes, q_by_class, device=None, batch_size=256, scores=None
):
    """
    Cached variant of `conformal.per_class_empirical_far()`.

    Measure empirical false-alarm rate per true class on a held-out known-class
    cached entry (e.g. test), under a threshold map `q_by_class` -- no circuit
    forward pass is re-run.

    Pass precomputed `scores` (e.g. `scores_test_g` from Global infer) to skip
    another nonconformity pass; scores are strategy-agnostic (s = 1 - F_max).

    To compare marginal vs Mondrian per-class fairness,
    - call once with `q_by_class = {c: global_q for c in classes}` for marginal calibration
    - call once with `cached_class_conditional_calibrate()` output for Mondrian calibration
    - compare the spread across classes.
    """
    y_known = np.asarray(y_known)
    if scores is None:
        scores = cached_nonconformity_scores(
            entry, prototypes, device=device, batch_size=batch_size
        )
    else:
        scores = np.asarray(scores, dtype=np.float64)
        if len(scores) != len(y_known):
            raise ValueError(
                f"scores length {len(scores)} != y_known length {len(y_known)}"
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
        rows.append(
            {
                "class": c,
                "n": n_c,
                "q": q_by_class[c],
                "empirical_far": far_c,
            }
        )
    return rows


def cached_class_conditional_f_in_f_out(
    entry_known, y_known, entry_zeroday, prototypes, device=None, batch_size=256
):
    """
    Cached, per-class extension of Proposition 2's F_in / F_out (Section 3.1/3.5).

    For each known class c:
    - F_in_c  = worst-case (min) fidelity of TRUE-class-c samples to THEIR OWN prototype rho_c.
    - F_out_c = worst-case (max) fidelity that ANY zero-day sample achieves against
                THAT class's prototype rho_c (not the pooled F_max over all classes).

    Computed entirely from cached `rho` (no circuit calls).

    Returns a list of dict rows: {"class", "n_known", "F_in_c", "F_out_c"}.
    """
    y_known = np.asarray(y_known)
    class_ids = sorted(prototypes.keys())
    rho_known_all = entry_known["rho"]
    rho_zday_all = entry_zeroday["rho"]

    rows = []
    for c in class_ids:
        proto_c = (
            prototypes[c].to(device=device) if device is not None else prototypes[c]
        )
        mask_c = y_known == c
        n_c = int(mask_c.sum())

        if n_c == 0:
            rows.append(
                {
                    "class": c,
                    "n_known": 0,
                    "F_in_c": float("nan"),
                    "F_out_c": float("nan"),
                }
            )
            continue

        # F_in_c: min fidelity of class-c's own samples to rho_c
        f_own_vals = []
        idx_c = np.where(mask_c)[0]
        for i in range(0, len(idx_c), batch_size):
            chunk = idx_c[i : i + batch_size]
            rho_chunk = rho_known_all[chunk]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            proto_b = proto_c.unsqueeze(0).expand(rho_chunk.shape[0], -1, -1)
            f_own_vals.append(
                fidelity_pairwise(rho_chunk, proto_b).detach().cpu().numpy()
            )
        f_own_c = np.concatenate(f_own_vals)
        F_in_c = float(f_own_c.min())

        # F_out_c: max fidelity of ALL zero-day samples against THIS class's prototype
        f_zday_vals = []
        n_z = len(rho_zday_all)
        for i in range(0, n_z, batch_size):
            rho_chunk = rho_zday_all[i : i + batch_size]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            proto_b = proto_c.unsqueeze(0).expand(rho_chunk.shape[0], -1, -1)
            f_zday_vals.append(
                fidelity_pairwise(rho_chunk, proto_b).detach().cpu().numpy()
            )
        f_zday_c = np.concatenate(f_zday_vals)
        F_out_c = float(f_zday_c.max())

        rows.append({"class": c, "n_known": n_c, "F_in_c": F_in_c, "F_out_c": F_out_c})

    return rows
