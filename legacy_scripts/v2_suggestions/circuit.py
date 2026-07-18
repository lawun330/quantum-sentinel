"""Quantum device setup and BATCHED variational forward circuit."""

import numpy as np
import pennylane as qp
import torch

from scripts.constants import DEFAULT_NOISE_RATE
from scripts.utils import get_torch_device


def create_quantum_device(num_qubits, backend="default.mixed"):
    return qp.device(backend, wires=num_qubits)


def initialize_weights(n_layers, n_wires, device):
    """Identity-block init (barren-plateau-safe): near-zero, not exactly zero."""
    shape = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    return torch.nn.Parameter(torch.zeros(shape, device=device))


def initialize_weights_random_identity(n_layers, n_wires, device, eps=1e-2, seed=None):
    """
     small-random identity-block init instead of
    exact zeros. Exact-zero init means every random probe in a barren-plateau
    check starts from the identical point, so it can't tell you whether
    GRADIENT VARIANCE across different inits is healthy -- only whether one
    single init is. Small random perturbation around identity lets
    gradient_variance_probe() below actually measure init-to-init variance,
    which is the thing barren-plateau theory cares about.
    """
    shape = qp.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    theta = eps * torch.randn(shape, generator=gen, device=device)
    return torch.nn.Parameter(theta)


def build_forward_circuit(dev, num_qubits, num_layers, noise_rate=DEFAULT_NOISE_RATE, reupload=True):
    """
    Build the QS-Net forward QNode: encode, variational layers, depolarizing noise, readout.

     accepts a BATCH of inputs,
    x_batch of shape (batch, n_features), and returns (z, rho) for the whole
    batch in a single QNode call, instead of one call per sample. Every
    caller elsewhere in this codebase (loss.py, prototypes.py, inference.py,
    conformal.py, hilbert.py) was looping `for x in X: forward_circuit(x, theta)`
    -- that's O(N) circuit traces per epoch. Batching this one function fixes
    all of them at the source.

    reupload flag : re-embeds x_batch at every
    variational layer instead of only once. This is the "data re-uploading"
    trick -- it increases effective circuit expressivity at fixed qubit count.
    Defaults to True; set False to reproduce the original single-embedding
    behavior for an A/B comparison.

    Returns a callable forward_circuit(x_batch, theta) -> (z, rho) where:
      z   : list of num_qubits tensors, each shaped (batch,)
      rho : tensor shaped (batch, dim, dim)
    """

    @qp.qnode(dev, interface="torch", diff_method="backprop")
    def forward_circuit(x_batch, theta):
        for layer in range(num_layers):
            if reupload or layer == 0:
                qp.AngleEmbedding(x_batch, wires=range(num_qubits), rotation="Y")
            qp.StronglyEntanglingLayers(theta[layer:layer + 1], wires=range(num_qubits))
            for w in range(num_qubits):
                qp.DepolarizingChannel(noise_rate, wires=w)
        z = [qp.expval(qp.PauliZ(w)) for w in range(num_qubits)]
        rho = qp.density_matrix(wires=range(num_qubits))
        return z, rho

    return forward_circuit


def gradient_variance_probe(forward_circuit, x_probe, n_qubits, n_layers, n_trials=30, eps=1e-2, device=None):
    """
     pre-training barren-plateau check.
    Runs n_trials fresh small-random inits through the circuit on a fixed
    probe batch, backprops a purity-based toy loss, and reports the variance
    of a single gradient component across inits. Run this BEFORE committing
    to a full training run on a new circuit config (qubit count, layer count,
    reupload on/off) -- catches a barren-plateau'd architecture cheaply,
    instead of finding out after a full training run stalls.
    """
    device = device or get_torch_device()
    grad_samples = []
    for _ in range(n_trials):
        theta = initialize_weights_random_identity(n_layers, n_qubits, device, eps=eps)
        _, rho = forward_circuit(x_probe, theta)
        loss = torch.stack([torch.real(torch.trace(r @ r)) for r in rho]).mean()
        loss.backward()
        grad_samples.append(theta.grad.reshape(-1)[0].item())
    grad_samples = np.array(grad_samples)
    var = float(grad_samples.var())
    return var, grad_samples