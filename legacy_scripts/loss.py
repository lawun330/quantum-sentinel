"""Algorithm 1: Margin-Aware Quantum Training (MAQT) loss."""

import torch

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_torch_x


def _forward_batch(theta, X_batch, y_batch, forward_circuit, device):
    """
    Run the circuit on a batch; return lists of (y_int, z, rho_x).
    """
    samples = []
    for x, y_i in zip(X_batch, y_batch):
        x = to_torch_x(x, device=device)
        y_int = int(y_i.item()) if torch.is_tensor(y_i) else int(y_i)
        z, rho_x = forward_circuit(x, theta)
        samples.append((y_int, z, rho_x))
    return samples


def ce_loss_term(samples, classifier_head, ce_loss_fn, device):
    """
    Cross-entropy term L_CE from circuit expectations + linear head.
    """
    if not samples:
        return torch.tensor(0.0, device=device)

    ce = torch.tensor(0.0, device=device)
    for y_int, z, _ in samples:
        logits = classifier_head(expectations_to_tensor(z))
        ce = ce + ce_loss_fn(
            logits.unsqueeze(0),
            torch.tensor([y_int], device=device, dtype=torch.long),
        )
    return ce / len(samples)


def intra_loss_term(samples, prototypes, device):
    """
    Intra-class term L_intra: mean infidelity (1 - F) to each class prototype.
    """
    intra_terms = {c: [] for c in prototypes}
    for y_int, _, rho_x in samples:
        if y_int not in prototypes:
            continue
        intra_terms[y_int].append(1.0 - fidelity(rho_x, prototypes[y_int]))

    l_intra = torch.tensor(0.0, device=device)
    active_classes = [c for c, terms in intra_terms.items() if terms]
    for c in active_classes:
        l_intra = l_intra + torch.stack(intra_terms[c]).mean()
    if active_classes:
        l_intra = l_intra / len(active_classes)
    return l_intra


def inter_loss_term(prototypes, device):
    """
    Inter-class term L_inter: negative mean trace distance between prototypes.
    """
    class_list = sorted(prototypes.keys())
    inter_pairs = []
    for i, c in enumerate(class_list):
        for c_prime in class_list[i + 1 :]:
            inter_pairs.append(trace_distance(prototypes[c], prototypes[c_prime]))
    if not inter_pairs:
        return torch.tensor(0.0, device=device)
    return -torch.stack(inter_pairs).mean()


def compute_l_ce(
    theta,
    classifier_head,
    ce_loss_fn,
    X_batch,
    y_batch,
    forward_circuit,
    device=None,
):
    """unit-testable L_CE over a batch."""
    device = device or theta.device
    samples = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return ce_loss_term(samples, classifier_head, ce_loss_fn, device)


def compute_l_intra(theta, X_batch, y_batch, prototypes, forward_circuit, device=None):
    """unit-testable L_intra over a batch."""
    device = device or theta.device
    samples = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return intra_loss_term(samples, prototypes, device)


def compute_l_inter(prototypes, device=None):
    """unit-testable L_inter from class prototypes only."""
    if device is None:
        any_rho = next(iter(prototypes.values()))
        device = any_rho.device
    return inter_loss_term(prototypes, device)


def maqt_loss(
    theta,
    classifier_head,
    ce_loss_fn,
    X_batch,
    y_batch,
    prototypes,
    forward_circuit,
    lambda1=0.5,
    lambda2=0.3,
    device=None,
):
    """
    MAQT loss function: L = L_CE + λ1·L_intra + λ2·L_inter.
    """
    device = device or theta.device
    samples = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)

    l_ce = ce_loss_term(samples, classifier_head, ce_loss_fn, device)
    l_intra = intra_loss_term(samples, prototypes, device)
    l_inter = inter_loss_term(prototypes, device)

    loss = l_ce + lambda1 * l_intra + lambda2 * l_inter
    return loss, l_ce, l_intra, l_inter


def gradient_variance(
    theta,
    classifier_head,
    ce_loss_fn,
    X_batch,
    y_batch,
    prototypes,
    forward_circuit,
    lambda1=0.5,
    lambda2=0.3,
    device=None,
):
    """
    Fresh grad of MAQT loss w.r.t. circuit theta, then variance.
    """
    device = device or theta.device
    loss, _, _, _ = maqt_loss(
        theta,
        classifier_head,
        ce_loss_fn,
        X_batch,
        y_batch,
        prototypes,
        forward_circuit,
        lambda1=lambda1,
        lambda2=lambda2,
        device=device,
    )
    grads = torch.autograd.grad(loss, theta, retain_graph=False, create_graph=False)[0]
    flat = grads.reshape(-1)
    return float(flat.var().item()), grads
