"""Quantum state distance and similarity metrics.

Fidelity convention (see Theoretical_Results.pdf, Section 1): all fidelities
returned by this module are the NON-SQUARED Uhlmann fidelity
    F(rho, sigma) = tr( sqrt( sqrt(rho) sigma sqrt(rho) ) )  in [0, 1].
pennylane's qml.math.fidelity() returns the SQUARED quantity F^2, so fidelity()
takes an explicit sqrt() to convert. fidelity_pairwise() is a from-scratch batched
implementation and must NOT square its result. Every fidelity-based quantity in the
codebase (nonconformity score, F_in/F_out, intra/inter-class diagnostics, Proposition
2's epsilon*) depends on this convention being consistent everywhere.
"""

import pennylane as qp
import torch

from scripts.constants import DEFAULT_EPS


def fidelity(rho_a, rho_b, eps=DEFAULT_EPS):
    """
    Non-squared Uhlmann fidelity F(rho,sigma) = tr(sqrt(sqrt(rho) sigma sqrt(rho))),
    clamped to [0, 1-eps]. qml.math.fidelity returns F^2, hence the sqrt() below.
    """
    f_sq = qp.math.fidelity(rho_a, rho_b)
    if torch.is_tensor(f_sq):
        f_sq = f_sq.real if torch.is_complex(f_sq) else f_sq
        f_sq = torch.clamp(f_sq, min=0.0)
        f = torch.sqrt(f_sq)
        return torch.clamp(f, 0.0, 1.0 - eps)
    f_sq = max(float(f_sq), 0.0)
    f = f_sq**0.5
    return min(f, 1.0 - eps)


def _as_complex_hermitian(mat):
    """
    Cast density matrices to a complex dtype and symmetrize numerically.
    """
    if not torch.is_complex(mat):
        mat = mat.to(torch.complex64)
    return 0.5 * (mat + mat.mH)


def _psd_sqrt(mat):
    """
    Matrix square root for batched Hermitian PSD mats shaped (..., d, d).
    """
    evals, evecs = torch.linalg.eigh(_as_complex_hermitian(mat))
    evals = torch.clamp(evals.real, min=0.0).to(evecs.dtype)
    return (evecs * torch.sqrt(evals).unsqueeze(-2)) @ evecs.mH


def fidelity_pairwise(rho, sigma, eps=DEFAULT_EPS):
    """
    Batched, NON-SQUARED Uhlmann fidelity with torch broadcasting:
    F(rho, sigma) = tr( sqrt( sqrt(rho) sigma sqrt(rho) ) ).

    rho, sigma: (..., d, d) broadcastable against each other.
    Returns real fidelities clamped to [0, 1-eps] with leading broadcast shape.
    """
    rho = _as_complex_hermitian(rho)
    sigma = _as_complex_hermitian(sigma)
    sqrt_rho = _psd_sqrt(rho)
    mid = sqrt_rho @ sigma @ sqrt_rho
    evals = torch.linalg.eigvalsh(_as_complex_hermitian(mid))
    evals = torch.clamp(evals.real, min=0.0)
    fid = torch.sqrt(evals).sum(dim=-1)  # NOTE: no ** 2 here -- non-squared F
    return torch.clamp(fid.real, 0.0, 1.0 - eps)


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
    Max fidelity of each state in a batch against a prototype stack: F_max(x) (Section 1).

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
    max_f = torch.zeros(
        rho_batch.shape[0], device=rho_batch.device, dtype=torch.float32
    )
    for c in range(proto_stack.shape[0]):
        f_c = fidelity_pairwise(rho_batch, proto_stack[c], eps=eps)
        max_f = torch.maximum(max_f, f_c.float())

    max_f = torch.clamp(max_f, 0.0, 1.0 - eps)
    return max_f[0] if squeezed else max_f


def trace_distance(rho_a, rho_b):
    """
    Trace distance D_tr(rho,sigma) = (1/2)||rho-sigma||_1, clamped to [0, 1].
    Matches Section 1's definition directly (qml.math.trace_distance uses the same
    convention), so no correction is needed here.
    """
    td = qp.math.trace_distance(rho_a, rho_b)
    if torch.is_tensor(td):
        td = td.real if torch.is_complex(td) else td
        return torch.clamp(td, 0.0, 1.0)
    return min(max(float(td), 0.0), 1.0)
