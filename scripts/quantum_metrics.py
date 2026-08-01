"""Quantum state distance and similarity metrics."""

import pennylane as qp
import torch

from scripts.constants import DEFAULT_EPS


def fidelity(rho_a, rho_b, eps=DEFAULT_EPS):
    """
    Non-squared Uhlmann fidelity between two density matrices, clamped to [0, 1-eps] to avoid numerical instability.

    PennyLane returns the squared Uhlmann fidelity -> take the square root.

    F(rho, sigma) = tr( sqrt( sqrt(rho) * sigma * sqrt(rho) ) )
    """
    squared_f = qp.math.fidelity(rho_a, rho_b)
    if torch.is_tensor(squared_f):
        squared_f = squared_f.real if torch.is_complex(squared_f) else squared_f
        squared_f = torch.clamp(squared_f, min=0.0) # ensure non-negative before taking the square root
        f = torch.sqrt(squared_f) # take the square root
        return torch.clamp(f, 0.0, 1.0 - eps)
    squared_f = max(float(squared_f), 0.0) # ensure non-negative before taking the square root
    f = squared_f ** 0.5 # take the square root
    return min(f, 1.0 - eps)


def _as_complex_hermitian(mat):
    """
    Cast density matrices to a complex dtype and symmetrize numerically.
    """
    mat = mat.to(torch.complex128)
    return 0.5 * (mat + mat.mH)


def _stabilize_psd(mat, eps=DEFAULT_EPS):
    """
    Mix a tiny multiple of identity so eigh stays well-conditioned under attack.
    """
    d = mat.shape[-1]
    eye = torch.eye(d, device=mat.device, dtype=mat.dtype)
    # broadcast eye to mat's batch shape
    while eye.ndim < mat.ndim:
        eye = eye.unsqueeze(0)
    return (1.0 - eps) * mat + eps * eye / d


def _psd_sqrt(mat, eps=DEFAULT_EPS):
    """
    Matrix square root for batched Hermitian PSD mats shaped (..., d, d).
    """
    mat = _stabilize_psd(_as_complex_hermitian(mat), eps=eps)
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals.real, min=0.0).to(evecs.dtype)
    return (evecs * torch.sqrt(evals).unsqueeze(-2)) @ evecs.mH


def fidelity_pairwise_psd(rho, sigma, eps=DEFAULT_EPS):
    """
    Batched non-squared Uhlmann fidelity with torch broadcasting.

    rho, sigma: (..., d, d) broadcastable against each other.
    Returns real fidelities clamped to [0, 1-eps] with leading broadcast shape.
    """
    rho = _stabilize_psd(_as_complex_hermitian(rho), eps=eps)
    sigma = _stabilize_psd(_as_complex_hermitian(sigma), eps=eps)
    sqrt_rho = _psd_sqrt(rho, eps=eps)
    mid = _stabilize_psd(sqrt_rho @ sigma @ sqrt_rho, eps=eps)
    evals = torch.linalg.eigvalsh(mid)
    evals = torch.clamp(evals.real, min=0.0)
    fid = torch.sqrt(evals).sum(dim=-1) # no squaring
    return torch.clamp(fid.real, 0.0, 1.0 - eps)


def fidelity_pairwise(rho, sigma, eps=DEFAULT_EPS):
    """
    Variant of `fidelity_pairwise_psd` that uses the PennyLane function.
    """
    squared_f = qp.math.fidelity(rho, sigma)
    if torch.is_tensor(squared_f):
        squared_f = squared_f.real if torch.is_complex(squared_f) else squared_f
        squared_f = torch.clamp(squared_f, min=0.0)
        f = torch.sqrt(squared_f)
        return torch.clamp(f, 0.0, 1.0 - eps)
    squared_f = max(float(squared_f), 0.0)
    return min(squared_f ** 0.5, 1.0 - eps)


def stack_prototypes(prototypes, device=None, dtype=None):
    """
    Stack a prototype dict into a (C, d, d) tensor in sorted class-id order.
    """
    if not prototypes:
        raise ValueError("prototypes must be non-empty")
    class_ids = sorted(prototypes)
    stacked = torch.stack([prototypes[c] for c in class_ids], dim=0)
    if device is not None or dtype is not None:
        stacked = stacked.to(device=device, dtype=dtype)
    return class_ids, stacked


def max_fidelity_to_prototypes(rho_batch, proto_stack, eps=DEFAULT_EPS):
    """
    Max fidelity of each state in a batch against a prototype stack.

    Loops over classes (usually few) and batches over samples to avoid a
    (B, C, d, d) memory blow-up on larger qubit counts.

    rho_batch: (B, d, d) or (d, d)
    proto_stack: (C, d, d)
    returns: (B,) or scalar tensor
    """
    squeezed = False
    if rho_batch.ndim == 2:
        rho_batch = rho_batch.unsqueeze(0)
        squeezed = True

    proto_stack = proto_stack.to(device=rho_batch.device)
    max_f = torch.zeros(rho_batch.shape[0], device=rho_batch.device, dtype=torch.float32)
    for c in range(proto_stack.shape[0]):
        f_c = fidelity(rho_batch, proto_stack[c], eps=eps)  # use fidelity function instead of fidelity_pairwise to avoid broadcasting
        if f_c.ndim == 0:  # if f_c is a scalar, expand it to the batch size
            f_c = f_c.expand(rho_batch.shape[0])
        max_f = torch.maximum(max_f, f_c.float())

    max_f = torch.clamp(max_f, 0.0, 1.0 - eps)
    return max_f[0] if squeezed else max_f


def trace_distance(rho_a, rho_b):
    """
    Trace distance between two density matrices, clamped to [0, 1].
    """
    td = qp.math.trace_distance(rho_a, rho_b)
    if torch.is_tensor(td):
        td = td.real if torch.is_complex(td) else td
        return torch.clamp(td, 0.0, 1.0)
    return min(max(float(td), 0.0), 1.0)