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

from scripts.constants import DEFAULT_ALPHA, DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.conformal import threshold_from_scores, min_calibration_size  # noqa: F401  (re-exported)
from scripts.quantum_metrics import fidelity, max_fidelity_to_prototypes, stack_prototypes, trace_distance
from scripts.utils import to_torch_batch_x, to_np_y, expectations_to_tensor

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

    def compute(self, key, X, theta, forward_circuit, device=None, batch_size=64,
                p=None, force=False):
        """Run circuit once over X; cache (z, rho) under `key` (idempotent unless force=True)."""
        if key in self._entries and not force:
            return self._entries[key]

        X_np = np.asarray(X, dtype=np.float32)
        z_chunks, rho_chunks = [], []
        with torch.no_grad():
            for i in range(0, len(X_np), batch_size):
                x_chunk = to_torch_batch_x(X_np[i:i + batch_size], device=device)
                z_chunk, rho_chunk = forward_circuit(x_chunk, theta)
                z_chunks.append(expectations_to_tensor(z_chunk).to(self.store_device))
                rho_chunks.append(rho_chunk.detach().to(self.store_device))

        entry = {
            "z": torch.cat(z_chunks, dim=0) if z_chunks else torch.empty(0),
            "rho": torch.cat(rho_chunks, dim=0) if rho_chunks else torch.empty(0),
            "n": len(X_np), "p": p, "X": X_np,
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
            rows.append({"key": key, "n_samples": entry["n"], "p": entry["p"], "MB": nbytes / 1e6})
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
        rho_chunk = rho_all[i:i + batch_size]
        rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
        max_f = max_fidelity_to_prototypes(rho_chunk, proto_stack.to(device=rho_chunk.device))
        out[i:i + batch_size] = max_f.detach().cpu().numpy()
    return out


def cached_nonconformity_scores(entry, prototypes, device=None, batch_size=256):
    """s(x) = 1 - F_max(x)."""
    return 1.0 - cached_f_max(entry, prototypes, device=device, batch_size=batch_size)


def cached_calibrate_threshold(entry, prototypes, alpha=DEFAULT_ALPHA, device=None, batch_size=256):
    """Calibrate conformal q from cached scores."""
    scores = cached_nonconformity_scores(entry, prototypes, device=device, batch_size=batch_size)
    return threshold_from_scores(scores, alpha=alpha)


def cached_conformal_alpha_sweep(entry_cal, entry_test_known, prototypes,
                                  alphas=(0.01, 0.05, 0.1, 0.2), device=None, batch_size=256):
    """Alpha sweep of q and empirical FAR using cached cal/test scores."""
    cal_scores = cached_nonconformity_scores(entry_cal, prototypes, device=device, batch_size=batch_size)
    test_scores = cached_nonconformity_scores(entry_test_known, prototypes, device=device, batch_size=batch_size)
    rows = []
    for alpha in alphas:
        q, _ = threshold_from_scores(cal_scores, alpha=alpha)
        rows.append({
            "alpha": alpha,
            "q": q,
            "empirical_false_alarm_rate": float(np.mean(test_scores > q)),
            "min_calibration_size": min_calibration_size(alpha),
            "n_cal": len(cal_scores),
        })
    return rows

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
            z_chunk = z_all[i:i + batch_size].to(device=head_device)
            preds.append(head(z_chunk).argmax(dim=1).cpu().numpy())
    return to_np_y(y_true).astype(int), (np.concatenate(preds) if preds else np.array([], dtype=int))


def cached_qsnet_infer(entry, prototypes, q, p=DEFAULT_NOISE_RATE, L_phi=None, Cf=DEFAULT_CF,
                        zero_day=ZERO_DAY, device=None, batch_size=256):
    """
    Cached variant of `inference.qsnet_infer_batch()`.

    Algo3 pipeline: p(x) -> per-class fidelity -> nearest class c_star -> conformal test with global q -> label + certified radius.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (see theory.analytic_lipschitz_bound)")
    class_ids = sorted(prototypes.keys())
    proto_on_device = {
        c: prototypes[c].to(device=device) if device is not None else prototypes[c]
        for c in class_ids
    }
    rho_all = entry["rho"]
    n = len(rho_all)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            rho_chunk = rho_all[i:i + batch_size]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            for j in range(rho_chunk.shape[0]):
                rho_x = rho_chunk[j]
                f_map = {c: float(fidelity(rho_x, proto_on_device[c])) for c in class_ids}
                f_vals = [f_map[c] for c in class_ids]
                c_star = class_ids[int(np.argmax(f_vals))]
                s = 1.0 - f_map[c_star]

                if s > q:
                    labels.append(zero_day)
                    radii.append(0.0)
                else:
                    sorted_f = sorted(f_vals, reverse=True)
                    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
                    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)
                    labels.append(c_star)
                    radii.append(float(radius))
                scores.append(s)
                f_maps.append(f_map)

    return np.array(labels), np.array(radii), np.array(scores), f_maps


def cached_qsnet_infer_per_class(entry, prototypes, q_by_class, p=DEFAULT_NOISE_RATE, L_phi=None, Cf=DEFAULT_CF,
                                  zero_day=ZERO_DAY, device=None, batch_size=256):
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
    proto_on_device = {
        c: prototypes[c].to(device=device) if device is not None else prototypes[c]
        for c in class_ids
    }
    rho_all = entry["rho"]
    n = len(rho_all)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, n, batch_size):
            rho_chunk = rho_all[i:i + batch_size]
            rho_chunk = rho_chunk.to(device=device) if device is not None else rho_chunk
            for j in range(rho_chunk.shape[0]):
                rho_x = rho_chunk[j]
                f_map = {c: float(fidelity(rho_x, proto_on_device[c])) for c in class_ids}
                f_vals = [f_map[c] for c in class_ids]
                c_star = class_ids[int(np.argmax(f_vals))]
                s = 1.0 - f_map[c_star]
                q_c = q_by_class[c_star]

                if s > q_c:
                    labels.append(zero_day)
                    radii.append(0.0)
                else:
                    sorted_f = sorted(f_vals, reverse=True)
                    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
                    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)
                    labels.append(c_star)
                    radii.append(float(radius))
                scores.append(s)
                f_maps.append(f_map)

    return np.array(labels), np.array(radii), np.array(scores), f_maps

# --------------------------------------------------------------------------------------
# Cached Lipschitz diagnostic (rho pairs; no circuit)
# --------------------------------------------------------------------------------------

def cached_lipschitz_percentile(entry, n_pairs=300, min_dist=1e-4, seed=0, percentile=95):
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
    idx1, idx2, dists = idx1[keep2][:n_pairs], idx2[keep2][:n_pairs], dists[keep2][:n_pairs]

    ratios = np.array([
        float(trace_distance(rho_all[i1], rho_all[i2])) / d
        for i1, i2, d in zip(idx1, idx2, dists)
    ]) if len(idx1) else np.array([])

    return {
        "ratios": ratios,
        f"p{percentile}": float(np.percentile(ratios, percentile)) if len(ratios) else float("nan"),
        "max": float(ratios.max()) if len(ratios) else float("nan"),
    }
