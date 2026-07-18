"""Algorithm 1: Margin-Aware Quantum Training (MAQT) loss -- batched, with curriculum + focal option."""

import torch
import torch.nn.functional as F

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_torch_batch


def curriculum_weight(epoch, total_epochs, base, warmup_frac=0.2):
    """
    NEW : linearly ramp a lambda from 0 to `base`
    over the first warmup_frac of training. Early in training the circuit's
    prototypes are meaningless (near-identity init), so forcing L_intra/
    L_inter to matter immediately can push the circuit toward a degenerate
    prototype geometry before it has learned anything about the features.
    """
    warmup = max(int(warmup_frac * total_epochs), 1)
    return base * min(1.0, epoch / warmup)


def focal_ce(logits, targets, gamma=2.0):
    """NEW : focal loss, optional alternative to plain CE."""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _forward_batch(theta, X_batch, y_batch, forward_circuit, device):
    """
    one batched circuit call instead of a Python loop calling
    forward_circuit once per sample.
    """
    x_t = to_torch_batch(X_batch, device=device)
    if torch.is_tensor(y_batch):
        y_t = y_batch.to(device=device, dtype=torch.long)
    else:
        y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
    z, rho = forward_circuit(x_t, theta)
    return y_t, z, rho


def ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal=False, focal_gamma=2.0):
    """
    Cross-entropy (or focal) term L_CE from batched circuit expectations + linear head.
    """
    if y_t.numel() == 0:
        return torch.tensor(0.0, device=device)
    logits = classifier_head(expectations_to_tensor(z))  # (batch, n_classes)
    if use_focal:
        return focal_ce(logits, y_t, gamma=focal_gamma)
    return ce_loss_fn(logits, y_t)


def intra_loss_term(y_t, rho, prototypes, device):
    """
    Intra-class term L_intra: mean infidelity (1 - F) to each class prototype.
    rho is now a batched tensor (batch, dim, dim); fidelity is still computed
    per-sample against its class prototype -- that part is cheap classical
    linear algebra on already-materialized density matrices, NOT a new
    circuit execution, so it doesn't need batching for performance.
    """
    y_np = y_t.detach().cpu().numpy()
    intra_terms = {c: [] for c in prototypes}
    for i, c_val in enumerate(y_np):
        c = int(c_val)
        if c not in prototypes:
            continue
        intra_terms[c].append(1.0 - fidelity(rho[i], prototypes[c]))

    l_intra = torch.tensor(0.0, device=device)
    active_classes = [c for c, terms in intra_terms.items() if terms]
    for c in active_classes:
        l_intra = l_intra + torch.stack(intra_terms[c]).mean()
    if active_classes:
        l_intra = l_intra / len(active_classes)
    return l_intra


def inter_loss_term(prototypes, device):
    """Inter-class term L_inter: negative mean trace distance between prototypes."""
    class_list = sorted(prototypes.keys())
    inter_pairs = []
    for i, c in enumerate(class_list):
        for c_prime in class_list[i + 1:]:
            inter_pairs.append(trace_distance(prototypes[c], prototypes[c_prime]))
    if not inter_pairs:
        return torch.tensor(0.0, device=device)
    return -torch.stack(inter_pairs).mean()


def compute_l_ce(theta, classifier_head, ce_loss_fn, X_batch, y_batch, forward_circuit,
                  device=None, use_focal=False, focal_gamma=2.0):
    """Unit-testable L_CE over a batch."""
    device = device or theta.device
    y_t, z, _ = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)


def compute_l_intra(theta, X_batch, y_batch, prototypes, forward_circuit, device=None):
    """Unit-testable L_intra over a batch."""
    device = device or theta.device
    y_t, _, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return intra_loss_term(y_t, rho, prototypes, device)


def compute_l_inter(prototypes, device=None):
    """Unit-testable L_inter from class prototypes only."""
    if device is None:
        any_rho = next(iter(prototypes.values()))
        device = any_rho.device
    return inter_loss_term(prototypes, device)


def maqt_loss(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
              forward_circuit, lambda1=0.5, lambda2=0.3, device=None,
              use_focal=False, focal_gamma=2.0):
    """
    MAQT loss: L = L_CE + lambda1 * L_intra + lambda2 * L_inter.

    prototypes may be either PrototypeBank.means() (a fixed snapshot) or
    EMAPrototypeBank.snapshot() (the pre-update EMA prototypes) -- caller's
    choice, see train.py for the intended EMA usage pattern.

    Returns rho too now, so the caller can feed it into
    EMAPrototypeBank.batch_class_means()/.update() AFTER this loss's
    backward() -- keeping the EMA update strictly post-loss (see prototypes.py).
    """
    device = device or theta.device
    y_t, z, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)

    l_ce = ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)
    l_intra = intra_loss_term(y_t, rho, prototypes, device)
    l_inter = inter_loss_term(prototypes, device)

    loss = l_ce + lambda1 * l_intra + lambda2 * l_inter
    return loss, l_ce, l_intra, l_inter, y_t, rho


def gradient_variance(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
                       forward_circuit, lambda1=0.5, lambda2=0.3, device=None):
    """Fresh grad of MAQT loss w.r.t. circuit theta, then variance."""
    device = device or theta.device
    loss, *_ = maqt_loss(
        theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
        forward_circuit, lambda1=lambda1, lambda2=lambda2, device=device,
    )
    grads = torch.autograd.grad(loss, theta, retain_graph=False, create_graph=False)[0]
    flat = grads.reshape(-1)
    return float(flat.var().item()), grads