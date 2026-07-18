"""Tensor and array conversion helpers."""

import numpy as np
import torch


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_x(x, device=None):
    """
    Single sample -> torch tensor of shape (1, n_features).
    Kept as a batch-of-one so it can be fed directly into the now-batched
    forward_circuit without special-casing single-sample callers.
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(x):
        t = x.detach().clone().to(device=device, dtype=torch.float32).reshape(-1)
    else:
        t = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1)
    return t.unsqueeze(0)


def to_torch_batch(X, device=None):
    """
    NEW: Batch of samples -> torch tensor of shape (batch, n_features).
    This is what makes the circuit batchable (see circuit.py) instead of
    looping one QNode call per sample.
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(X):
        return X.detach().clone().to(device=device, dtype=torch.float32)
    return torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device)


def to_np_x(x):
    """
    Transform a 1D label vector to numpy. Kept for label arrays (y),
    where flattening to 1D is correct.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def to_np_2d(X):
    """
    NEW: Transform a 2D feature batch to numpy WITHOUT flattening.
    (to_np_x would incorrectly collapse a (batch, n_features) array to 1D.)
    """
    if torch.is_tensor(X):
        return X.detach().cpu().numpy()
    return np.asarray(X, dtype=np.float32)


def expectations_to_tensor(z):
    """
    Convert PennyLane batched expectation values to a (batch, n_qubits) tensor.
    z is a list of n_qubits tensors, each shaped (batch,).
    """
    if torch.is_tensor(z):
        return z.float()
    return torch.stack(z, dim=1).float()