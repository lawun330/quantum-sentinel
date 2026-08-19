"""Algorithm 3: Unified inference with disentangled rejection."""

import numpy as np
import torch

from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_CF, DEFAULT_NOISE_RATE, DEFAULT_N_PROBE, DEFAULT_DELTA, ZERO_DAY
from scripts.quantum_metrics import all_fidelities_to_prototypes, fidelity, stack_prototypes, trace_distance
from scripts.utils import expectations_to_tensor, to_np_batch_x, to_torch_batch_x, to_torch_x, to_np_y


def algo3_postprocess_f_mat(f_mat, class_ids, q, p, L_phi, Cf, zero_day=ZERO_DAY, q_by_class=None):
    """
    Algo-3 labels, certified radii, and nonconformity scores from a (B, C) fidelity matrix.

    Keeps computation on `f_mat.device` until the caller moves results to CPU.
    """
    b, c = f_mat.shape
    if c != len(class_ids):
        raise ValueError(f"f_mat has {c} classes but class_ids has {len(class_ids)}")

    device = f_mat.device
    class_id_t = torch.tensor(class_ids, device=device, dtype=torch.long)
    row_idx = torch.arange(b, device=device)

    c_idx = f_mat.argmax(dim=1)
    s = 1.0 - f_mat[row_idx, c_idx]

    if q_by_class is not None:
        q_vec = torch.tensor([q_by_class[cid] for cid in class_ids], device=device, dtype=f_mat.dtype)
        reject = s > q_vec[c_idx]
    else:
        reject = s > q

    f_sorted, _ = torch.sort(f_mat, dim=1, descending=True)
    if c > 1:
        margin = f_sorted[:, 0] - f_sorted[:, 1]
    else:
        margin = f_sorted[:, 0]
    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)

    labels = torch.empty(b, dtype=torch.long, device=device)
    radii = torch.zeros(b, dtype=f_mat.dtype, device=device)
    labels[reject] = zero_day
    labels[~reject] = class_id_t[c_idx[~reject]]
    radii[~reject] = radius[~reject]
    return labels, radii, s


def f_maps_from_f_mat(f_mat, class_ids):
    """Build per-sample fidelity dicts from a (B, C) matrix (one CPU sync per batch)."""
    f_np = f_mat.detach().cpu().numpy()
    return [{class_ids[c]: float(f_np[j, c]) for c in range(len(class_ids))} for j in range(f_np.shape[0])]


def predict_labels(X, y, theta, classifier_head, forward_circuit, device, batch_size=DEFAULT_BATCH_SIZE):
    """
    Batched prediction of class labels using circuit expectations + classical head.
    """
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            z, _ = forward_circuit(x_chunk, theta)
            logits = classifier_head(expectations_to_tensor(z))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_true = to_np_y(y).astype(int)
    return y_true, np.concatenate(preds)


def estimate_lipschitz(X, theta, forward_circuit, n_probe=DEFAULT_N_PROBE, delta=DEFAULT_DELTA,
                       device=None, batch_size=DEFAULT_BATCH_SIZE, percentile=None, return_ratios=False):
    """
    DEPRECATED for certificates / Day-16 deliverable.
    USE `theory.estimate_lipschitz_percentile()` instead.

    Batched estimation of encoder Lipschitz constant from data.

    For each probe x, forms a δ-perturbed twin and records
    D_tr(ρ(x), ρ(x+δξ)) / δ.

    - By default returns the max ratio (legacy).
    - Pass percentile (e.g. 95) for the Day-16 robust estimator; 
    - Set return_ratios=True to also get the full ratio distribution.
    """
    X_np = to_np_batch_x(X)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)

    n = X_np.shape[0]
    indices = np.random.choice(n, min(n_probe, n), replace=False)

    x_probe = to_torch_batch_x(X_np[indices], device=device)
    x_probe_pert = x_probe + delta * torch.randn_like(x_probe)

    ratios = []
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            _, r1 = forward_circuit(x_probe[i:i + batch_size], theta)
            _, r2 = forward_circuit(x_probe_pert[i:i + batch_size], theta)
            for j in range(r1.shape[0]):
                ratios.append(trace_distance(r1[j], r2[j]) / delta)

    ratios_t = torch.stack(ratios)
    ratios_np = np.asarray([float(r.item()) for r in ratios_t], dtype=np.float64)

    if percentile is None:
        L_phi = float(torch.max(ratios_t).item())
    else:
        L_phi = float(np.percentile(ratios_np, percentile))

    if return_ratios:
        return L_phi, ratios_np
    return L_phi


def qsnet_infer_single(x, theta, prototypes, q, forward_circuit, p=DEFAULT_NOISE_RATE,
                L_phi=None, Cf=DEFAULT_CF, zero_day=ZERO_DAY, device=None):
    """
    Single-sample unified inference with disentangled rejection.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (estimate with estimate_lipschitz)")

    with torch.no_grad():
        _, rho_batch = forward_circuit(to_torch_x(x, device=device), theta)
        rho_x = rho_batch[0]
        class_ids = sorted(prototypes.keys())
        f_map = {c: float(fidelity(rho_x, prototypes[c]).item()) for c in class_ids}
        f_vals = [f_map[c] for c in class_ids]

    c_star = class_ids[int(np.argmax(f_vals))]
    s = 1.0 - f_map[c_star]

    if s > q:
        return zero_day, 0.0, s, f_map

    sorted_f = sorted(f_vals, reverse=True)
    margin = sorted_f[0] - sorted_f[1] if len(sorted_f) > 1 else sorted_f[0]
    radius = margin / (2.0 * (1.0 - p) * L_phi * Cf)

    return c_star, float(radius), s, f_map


def qsnet_infer_batch(X, theta, prototypes, q, forward_circuit, p=DEFAULT_NOISE_RATE,
                       L_phi=None, Cf=DEFAULT_CF, zero_day=ZERO_DAY, device=None,
                       batch_size=DEFAULT_BATCH_SIZE, return_f_maps=True):
    """
    Batched unified inference with disentangled rejection.

    Algo3 pipeline: p(x) -> per-class fidelity -> nearest class c_star -> conformal test with global q -> label + certified radius.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided.")

    class_ids, proto_stack = stack_prototypes(prototypes, device=device)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            f_mat = all_fidelities_to_prototypes(rho_chunk, proto_stack)
            labels_b, radii_b, scores_b = algo3_postprocess_f_mat(
                f_mat, class_ids, q=q, p=p, L_phi=L_phi, Cf=Cf, zero_day=zero_day,
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


def qsnet_infer_batch_per_class(X, theta, prototypes, q_by_class, forward_circuit, p=DEFAULT_NOISE_RATE,
                                 L_phi=None, Cf=DEFAULT_CF, zero_day=ZERO_DAY, device=None,
                                 batch_size=DEFAULT_BATCH_SIZE, return_f_maps=True):
    """
    Class-conditional (Mondrian) variant of `inference.qsnet_infer_batch()`.
    Non-cached variant of `cache.cached_qsnet_infer_per_class()`.

    Algo3 pipeline: p(x) -> per-class fidelity -> nearest class c_star -> conformal test with per-class q_c -> label + certified radius.

    See `conformal.class_conditional_calibrate()` for how to build q_by_class.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided.")

    class_ids = sorted(prototypes.keys())
    missing = [c for c in class_ids if c not in q_by_class]
    if missing:
        raise ValueError(f"q_by_class is missing thresholds for classes: {missing}")

    _, proto_stack = stack_prototypes(prototypes, device=device)
    labels, radii, scores, f_maps = [], [], [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_chunk = to_torch_batch_x(X[i:i + batch_size], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            f_mat = all_fidelities_to_prototypes(rho_chunk, proto_stack)
            labels_b, radii_b, scores_b = algo3_postprocess_f_mat(
                f_mat, class_ids, q=None, p=p, L_phi=L_phi, Cf=Cf, zero_day=zero_day,
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