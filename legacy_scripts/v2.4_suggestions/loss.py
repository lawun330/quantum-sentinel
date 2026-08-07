"""Algorithm 1: Margin-Aware Quantum Training (MAQT) loss."""

import torch
import torch.nn.functional as F

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_torch_batch_x, to_torch_y, to_np_y
from scripts.constants import (
    DEFAULT_FOCAL, DEFAULT_FOCAL_GAMMA, DEFAULT_LAMBDA1, DEFAULT_LAMBDA2,
    DEFAULT_INTER_MARGIN, DEFAULT_WARMUP_FRAC,
)


def _forward_batch(theta, X_batch, y_batch, forward_circuit, device):
    """
    Run the circuit on a batch; return (y_t, z, rho).
    """
    x_t = to_torch_batch_x(X_batch, device=device)
    y_t = to_torch_y(y_batch, device=device)
    z, rho = forward_circuit(x_t, theta)
    return y_t, z, rho


def focal_ce(logits, targets, gamma=DEFAULT_FOCAL_GAMMA):
    """
    Focal loss, an optional alternative to plain cross-entropy.
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    Cross-entropy or focal loss term (L_CE) from batched circuit expectations + linear head.
    """
    if y_t.numel() == 0:
        return torch.tensor(0.0, device=device)
    logits = classifier_head(expectations_to_tensor(z))  # (batch, n_classes)
    if use_focal:
        return focal_ce(logits, y_t, gamma=focal_gamma)
    return ce_loss_fn(logits, y_t)


def intra_loss_term(y_t, rho, prototypes, device):
    """
    Intra-class term (L_intra): mean infidelity (1 - F) to each class prototype.

    Gradient flows through `rho` (this step's live forward pass) toward `theta`.
    `prototypes` are frozen EMA targets (detached) -- this is correct and unchanged.
    """
    y_np = to_np_y(y_t)
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


def inter_loss_term(y_t, rho, prototypes, device, margin=DEFAULT_INTER_MARGIN, hardest_only=True):
    """
    Inter-class term (L_inter): hinge repulsion of each live sample from its
    NEAREST wrong-class prototype (hardest negative).


    hardest_only=True uses only the single nearest wrong-class prototype per
    sample (hinge only fires below `margin`), concentrating gradient on the
    classes that are actually confusable (e.g. DDoS <-> DoS) instead of
    diluting it across easy pairs that are already well separated.
    """
    y_np = to_np_y(y_t)
    class_ids = sorted(prototypes.keys())
    terms = []

    for i, c_val in enumerate(y_np):
        c = int(c_val)
        if c not in prototypes:
            continue
        neg_classes = [cc for cc in class_ids if cc != c]
        if not neg_classes:
            continue

        neg_d = torch.stack([trace_distance(rho[i], prototypes[cc]) for cc in neg_classes])
        d_ref = neg_d.min() if hardest_only else neg_d.mean()
        terms.append(torch.relu(margin - d_ref))  

    if not terms:
        return torch.tensor(0.0, device=device)
    return torch.stack(terms).mean()


def compute_l_ce(theta, classifier_head, ce_loss_fn, X_batch, y_batch, forward_circuit, device=None,
                  use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    Unit-testable L_CE over a batch.
    """
    device = device or theta.device
    y_t, z, _ = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)


def compute_l_intra(theta, X_batch, y_batch, prototypes, forward_circuit, device=None):
    """
    Unit-testable L_intra over a batch.
    """
    device = device or theta.device
    y_t, _, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return intra_loss_term(y_t, rho, prototypes, device)


def compute_l_inter(theta, X_batch, y_batch, prototypes, forward_circuit, device=None,
                     margin=DEFAULT_INTER_MARGIN, hardest_only=True):
    """
    Unit-testable L_inter over a batch.

    """
    device = device or theta.device
    y_t, _, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)
    return inter_loss_term(y_t, rho, prototypes, device, margin=margin, hardest_only=hardest_only)


def maqt_loss(theta, classifier_head, ce_loss_fn, X_batch, y_batch, prototypes,
            forward_circuit, lambda1=DEFAULT_LAMBDA1, lambda2=DEFAULT_LAMBDA2,
            margin=DEFAULT_INTER_MARGIN, hardest_only=True, device=None,
            use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA):
    """
    MAQT loss: L = L_CE + lambda1 * L_intra + lambda2 * L_inter.

    `margin` and `hardest_only` are new, optional, and default-backward-compatible
    -- no caller (train.py, gradient.py) needs to change unless they want to tune them.
    """
    device = device or theta.device
    y_t, z, rho = _forward_batch(theta, X_batch, y_batch, forward_circuit, device)

    l_ce = ce_loss_term(y_t, z, classifier_head, ce_loss_fn, device, use_focal, focal_gamma)
    l_intra = intra_loss_term(y_t, rho, prototypes, device)
    l_inter = inter_loss_term(y_t, rho, prototypes, device, margin=margin, hardest_only=hardest_only)

    loss = l_ce + lambda1 * l_intra + lambda2 * l_inter
    return loss, l_ce, l_intra, l_inter, y_t, rho


def curriculum_weight(epoch, total_epochs, base, warmup_frac=DEFAULT_WARMUP_FRAC):
    """
    Linearly ramp a lambda from 0 to `base` over the first `warmup_frac` of training.

    This prevents the circuit from being prematurely constrained by uninformative prototypes during early training.
    """
    warmup = max(int(warmup_frac * total_epochs), 1)
    return base * min(1.0, epoch / warmup)