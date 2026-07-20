"""Algorithm 2: Conformal Zero-Day Threshold Calibration (CQ-ZDR)."""

import numpy as np
import torch

from scripts.constants import DEFAULT_BATCH_SIZE, DEFAULT_ALPHA
from scripts.quantum_metrics import max_fidelity_to_prototypes, stack_prototypes
from scripts.utils import to_torch_batch_x


def nonconformity_score(X, theta, prototypes, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Compute one conformal nonconformity score per sample in X.

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


def calibrate_threshold(theta, X_cal, prototypes, forward_circuit, alpha=DEFAULT_ALPHA, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Calibrate the CQ-ZDR threshold q from the calibration split.
    """
    if X_cal is None or len(X_cal) == 0:
        raise ValueError("X_cal must contain at least one sample for conformal calibration (n == 0)")
    if not prototypes:
        raise ValueError("prototypes must be non-empty")

    scores = nonconformity_score(
        X_cal, theta, prototypes, forward_circuit, device=device, batch_size=batch_size,
    )
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    k = min(max(k, 0), n - 1)  # clamp to [0, n-1]
    q = float(scores_sorted[k])
    return q, scores_sorted