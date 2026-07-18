"""Tensor and array conversion helpers."""

import numpy as np
import torch


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_x(x, device=None):
    """
    Transform data to 1D torch vector.
    """
    if device is None:
        device = get_torch_device()
    if torch.is_tensor(x):
        return x.detach().clone().to(device=device, dtype=torch.float32).reshape(-1)
    return torch.tensor(x, dtype=torch.float32, device=device)


def to_np_x(x):
    """
    Transform data to 1D numpy vector.
    """
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().reshape(-1)
    return np.asarray(x, dtype=np.float32).reshape(-1)


def expectations_to_tensor(z):
    """
    Convert PennyLane expectation values to a 1D float tensor.
    """
    if torch.is_tensor(z):
        return z.float().reshape(-1)
    return torch.stack(z).float()
