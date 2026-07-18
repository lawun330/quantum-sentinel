"""Algorithm 3: Unified inference with disentangled rejection."""

import numpy as np
import torch

from scripts.constants import DEFAULT_CF, DEFAULT_NOISE_RATE, ZERO_DAY
from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_np_x, to_torch_x


def predict_labels(X, y, theta, classifier_head, forward_circuit, device):
    """
    Predict class labels via circuit expectations + classical head.
    """
    preds = []
    with torch.no_grad():
        for x in X:
            z, _ = forward_circuit(to_torch_x(x, device=device), theta)
            z = expectations_to_tensor(z)
            preds.append(int(classifier_head(z).argmax().item()))
    y_true = to_np_x(y).astype(int)
    return y_true, np.asarray(preds)


def estimate_lipschitz(X, theta, forward_circuit, n_probe=30, delta=0.05, device=None):
    """
    Estimate Lipschitz constant from data.
    """
    if torch.is_tensor(X):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = np.asarray(X, dtype=np.float32)
    if X_np.ndim == 1:
        X_np = X_np.reshape(1, -1)

    n = X_np.shape[0]
    indices = np.random.choice(n, min(n_probe, n), replace=False)
    ratios = []

    with torch.no_grad():
        for i in indices:
            x = to_torch_x(X_np[i], device=device)
            x_pert = x + delta * torch.randn_like(x)
            _, r1 = forward_circuit(x, theta)
            _, r2 = forward_circuit(x_pert, theta)
            ratios.append(trace_distance(r1, r2) / delta)

    return float(torch.max(torch.stack(ratios)).item())


def qsnet_infer(
    x,
    theta,
    prototypes,
    q,
    forward_circuit,
    p=DEFAULT_NOISE_RATE,
    L_phi=None,
    Cf=DEFAULT_CF,
    zero_day=ZERO_DAY,
    device=None,
):
    """
    Unified inference with disentangled rejection.
    """
    if L_phi is None:
        raise ValueError("L_phi must be provided (estimate with estimate_lipschitz)")

    with torch.no_grad():
        _, rho_x = forward_circuit(to_torch_x(x, device=device), theta)
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


def predict_batch(X, theta, prototypes, q, forward_circuit, **infer_kwargs):
    """
    Make batch predictions.
    """
    labels, radii, scores = [], [], []
    for x in X:
        label, radius, score, _ = qsnet_infer(
            x, theta, prototypes, q, forward_circuit, **infer_kwargs
        )
        labels.append(label)
        radii.append(radius)
        scores.append(score)
    return np.array(labels), np.array(radii), np.array(scores)
