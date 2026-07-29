"""
Forward-pass cache for QS-Net / CQ-ZDR.

The quantum circuit forward pass (rho(x) = Lambda_p(U(theta)|psi(x)><psi(x)|U(theta)^dag))
is by far the most expensive operation in this pipeline -- it scales as O(4^n_qubits) per
sample for the default.mixed density-matrix simulator. Across a typical run of the
theory-validation notebook, the SAME (theta_final, dataset, noise level p) combination is
forward-passed independently by several different diagnostics that each only need a
different piece of the same output:

  - predict_labels            wants z (classifier expectations), discards rho
  - nonconformity_score       wants rho (for fidelity), discards z
  - qsnet_infer_batch         wants rho, discards z
  - f_max_batch               wants rho (via nonconformity_score), discards z
  - estimate_lipschitz_percentile  wants rho for two disjoint SAMPLES already elsewhere
                               in the dataset, discards z

ForwardCache runs the circuit ONCE per (key, theta, p) and stores both z and rho (on CPU,
to keep GPU memory free for training/attacks); every function in this module derives its
result purely from the cached tensors -- no circuit call, no synthetic/random data. The
cache holds exactly the encoder's own outputs on the real dataset, nothing else.
"""

import numpy as np
import torch

from scripts.constants import DEFAULT_ALPHA, DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.conformal import threshold_from_scores, min_calibration_size  # noqa: F401  (re-exported)
from scripts.quantum_metrics import fidelity, max_fidelity_to_prototypes, stack_prototypes, trace_distance
from scripts.utils import to_torch_batch_x, to_np_y, expectations_to_tensor


class ForwardCache:
    """
    Computes and stores {"z": (N, n_qubits), "rho": (N, d, d)} for named datasets.
    Everything is cached on CPU by default; downstream cached_* functions move only the
    chunk currently being processed back to `device` for the (much cheaper) fidelity /
    classification math.
    """

    def __init__(self, store_device="cpu"):
        self.store_device = store_device
        self._entries = {}   # key -> {"z":.., "rho":.., "n":.., "p":.., "theta_id":..}

    def compute(self, key, X, theta, forward_circuit, device=None, batch_size=64,
                p=None, force=False):
        """Runs the circuit once over X and caches (z, rho) under `key`. Idempotent unless force=True."""
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
        """Returns a new (uncached) entry dict restricted to the given sample indices -- no circuit call."""
        entry = self._entries[key]
        idx = np.asarray(idx)
        return {"z": entry["z"][idx], "rho": entry["rho"][idx], "n": len(idx),
                "p": entry["p"], "X": entry["X"][idx]}

    def clear(self, key=None):
        if key is None:
            self._entries.clear()
        else:
            self._entries.pop(key, None)

    def memory_report(self):
        rows = []
        for key, entry in self._entries.items():
            nbytes = entry["rho"].element_size() * entry["rho"].nelement() + \
                     entry["z"].element_size() * entry["z"].nelement()
            rows.append({"key": key, "n_samples": entry["n"], "p": entry["p"], "MB": nbytes / 1e6})
        return rows


# --------------------------------------------------------------------------------------
# Cache-aware replacements for scripts.inference / scripts.conformal, operating purely
# on already-computed (z, rho) -- zero additional circuit calls.
# --------------------------------------------------------------------------------------

def cached_f_max(entry, prototypes, device=None, batch_size=256):
    """F_max(x) for every cached sample -- reuses entry['rho'], no circuit call."""
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
    return 1.0 - cached_f_max(entry, prototypes, device=device, batch_size=batch_size)


def cached_calibrate_threshold(entry, prototypes, alpha=DEFAULT_ALPHA, device=None, batch_size=256):
    scores = cached_nonconformity_scores(entry, prototypes, device=device, batch_size=batch_size)
    return threshold_from_scores(scores, alpha=alpha)


def cached_conformal_alpha_sweep(entry_cal, entry_test_known, prototypes,
                                  alphas=(0.01, 0.05, 0.1, 0.2), device=None, batch_size=256):
    cal_scores = cached_nonconformity_scores(entry_cal, prototypes, device=device, batch_size=batch_size)
    test_scores = cached_nonconformity_scores(entry_test_known, prototypes, device=device, batch_size=batch_size)
    rows = []
    for alpha in alphas:
        q, _ = threshold_from_scores(cal_scores, alpha=alpha)
        rows.append({"alpha": alpha, "q": q, "empirical_false_alarm_rate": float(np.mean(test_scores > q)),
                     "min_calibration_size": min_calibration_size(alpha), "n_cal": len(cal_scores)})
    return rows


def cached_predict_labels(entry, y_true, head, device=None, batch_size=1024):
    """Classifier predictions from cached z -- no circuit call."""
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
    Algorithm 3 unified inference (p(x) -> fidelity -> score -> conformal test ->
    label + certified radius), reusing cached rho -- no circuit call. Mirrors
    scripts.inference.qsnet_infer_batch exactly, just sourced from the cache.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (see theory.analytic_lipschitz_bound)")
    class_ids = sorted(prototypes.keys())
    proto_on_device = {c: prototypes[c].to(device=device) if device is not None else prototypes[c]
                        for c in class_ids}
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
                    labels.append(zero_day); radii.append(0.0)
                else:
                    sorted_f = sorted(f_vals, reverse=True)
                    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
                    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)
                    labels.append(c_star); radii.append(float(radius))
                scores.append(s)
                f_maps.append(f_map)

    return np.array(labels), np.array(radii), np.array(scores), f_maps







def cached_qsnet_infer_per_class(entry, prototypes, q_by_class, p=DEFAULT_NOISE_RATE, L_phi=None, Cf=DEFAULT_CF,
                                  zero_day=ZERO_DAY, device=None, batch_size=256):
    """
    Same as cached_qsnet_infer, but compares s(x) against a PER-CLASS threshold
    q_by_class[c_star] (the threshold calibrated for whichever known class x is nearest
    to) instead of one global q. See scripts.conformal.class_conditional_calibrate for
    how to build q_by_class.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (see theory.analytic_lipschitz_bound)")
    class_ids = sorted(prototypes.keys())
    missing = [c for c in class_ids if c not in q_by_class]
    if missing:
        raise ValueError(f"q_by_class is missing thresholds for classes: {missing}")
    proto_on_device = {c: prototypes[c].to(device=device) if device is not None else prototypes[c]
                        for c in class_ids}
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
                    labels.append(zero_day); radii.append(0.0)
                else:
                    sorted_f = sorted(f_vals, reverse=True)
                    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
                    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)
                    labels.append(c_star); radii.append(float(radius))
                scores.append(s)
                f_maps.append(f_map)

    return np.array(labels), np.array(radii), np.array(scores), f_maps








def cached_lipschitz_percentile(entry, n_pairs=300, min_dist=1e-4, seed=0, percentile=95):
    """
    Day-16 Lipschitz tightness diagnostic, reusing cached rho for BOTH ends of every
    sampled pair -- zero circuit calls (previously this issued 2 fresh forward passes
    per pair, i.e. up to 2*n_pairs redundant circuit evaluations per dataset, on
    samples whose rho had already been computed for classification/fidelity).
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