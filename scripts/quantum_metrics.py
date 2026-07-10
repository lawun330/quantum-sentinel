"""Quantum state distance and similarity metrics."""

import pennylane as qp


def fidelity(rho_a, rho_b):
    """
    Compute the fidelity between two density matrices.
    """
    return qp.math.fidelity(rho_a, rho_b)


def trace_distance(rho_a, rho_b):
    """
    Compute the trace distance between two density matrices.
    """
    return qp.math.trace_distance(rho_a, rho_b)
