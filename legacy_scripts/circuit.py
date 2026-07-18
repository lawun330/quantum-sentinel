"""Quantum device setup and variational forward circuit."""

import pennylane as qp
import torch

from scripts.constants import DEFAULT_NOISE_RATE


def create_quantum_device(num_qubits, backend="default.mixed"):
    return qp.device(backend, wires=num_qubits)


def initialize_weights(n_layers, n_wires, device):
    """
    Initialize weights for the variational circuit using identity rotations to be barren-plateau-safe.
    """
    shape = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    return torch.nn.Parameter(torch.zeros(shape, device=device))


def build_forward_circuit(dev, num_qubits, num_layers, noise_rate=DEFAULT_NOISE_RATE):
    """
    Build the QS-Net forward QNode: encode, variational layers, depolarizing noise, readout.

    Returns a callable ``forward_circuit(x, theta) -> (z, rho)``.
    """

    @qp.qnode(dev, interface="torch", diff_method="backprop")
    def forward_circuit(x, theta):
        """
        Circuit for encoding, variational, adding noise, and readout.
        """
        for layer in range(num_layers):
            qp.AngleEmbedding(x, wires=range(num_qubits), rotation="Y")
            qp.StronglyEntanglingLayers(theta[layer : layer + 1], wires=range(num_qubits))
            for w in range(num_qubits):
                qp.DepolarizingChannel(noise_rate, wires=w)
        z = [qp.expval(qp.PauliZ(w)) for w in range(num_qubits)]
        rho = qp.density_matrix(wires=range(num_qubits))
        return z, rho

    return forward_circuit
