"""Quantum state distance and similarity metrics."""

import pennylane as qp
import torch


def fidelity(rho_a, rho_b, eps=1e-7):
    """
    Fidelity between two density matrices, clamped to [0, 1-eps].
    FIX: raw qp.math.fidelity can drift slightly above 1.0 due to floating-point
    error in mixed-state simulation, which silently poisons 1-F "infidelity"
    losses downstream (loss.py, conformal.py) with negative values. Clamping
    is your friend's `safe_fidelity` pattern, merged here as the canonical impl
    so every caller gets it for free.
    """
    f = qp.math.fidelity(rho_a, rho_b)
    if torch.is_tensor(f):
        f = f.real if torch.is_complex(f) else f
        return torch.clamp(f, 0.0, 1.0 - eps)
    return min(max(float(f), 0.0), 1.0 - eps)


def trace_distance(rho_a, rho_b):
    """
    Trace distance between two density matrices, clamped to [0, 1].
    """
    td = qp.math.trace_distance(rho_a, rho_b)
    if torch.is_tensor(td):
        td = td.real if torch.is_complex(td) else td
        return torch.clamp(td, 0.0, 1.0)
    return min(max(float(td), 0.0), 1.0)