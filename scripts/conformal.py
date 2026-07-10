"""Algorithm 2: Conformal Zero-Day Threshold Calibration (CQ-ZDR)."""

import numpy as np
import torch

from scripts.quantum_metrics import fidelity
from scripts.utils import to_torch_x


def nonconformity_score(x, theta, prototypes, forward_circuit, device=None):
    """
    Nonconformity score (PyTorch version).
    """
    with torch.no_grad():
        _, rho_x = forward_circuit(to_torch_x(x, device=device), theta)
        fids = [fidelity(rho_x, prototypes[c]) for c in sorted(prototypes)]
        return float((1.0 - torch.max(torch.stack(fids))).item())


def calibrate_threshold(theta, X_cal, prototypes, forward_circuit, alpha=0.05, device=None):
    """
    Calibrate the threshold.
    """
    scores = np.array([
        nonconformity_score(x, theta, prototypes, forward_circuit, device=device)
        for x in X_cal
    ])
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    k = min(max(k, 0), n - 1)
    q = float(scores_sorted[k])
    return q, scores_sorted
