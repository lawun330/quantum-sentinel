"""Algorithm 1: Margin-Aware Quantum Training (MAQT) loss."""

import torch

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import expectations_to_tensor, to_torch_x


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
    MAQT loss function (PyTorch version).
    """
    device = device or theta.device
    ce = torch.tensor(0.0, device=device)
    intra_terms = {c: [] for c in prototypes}

    for x, y_i in zip(X_batch, y_batch):
        x = to_torch_x(x, device=device)
        y_int = int(y_i.item()) if torch.is_tensor(y_i) else int(y_i)

        z, rho_x = forward_circuit(x, theta)
        logits = classifier_head(expectations_to_tensor(z))
        ce = ce + ce_loss_fn(
            logits.unsqueeze(0),
            torch.tensor([y_int], device=device, dtype=torch.long),
        )
        proto = prototypes[y_int]
        intra_terms[y_int].append(1.0 - fidelity(rho_x, proto))

    l_ce = ce / len(X_batch)

    l_intra = torch.tensor(0.0, device=device)
    active_classes = [c for c, terms in intra_terms.items() if terms]
    for c in active_classes:
        terms_t = torch.stack(intra_terms[c])
        l_intra = l_intra + terms_t.mean()
    if active_classes:
        l_intra = l_intra / len(active_classes)

    class_list = sorted(prototypes.keys())
    inter_pairs = []
    for i, c in enumerate(class_list):
        for c_prime in class_list[i + 1 :]:
            inter_pairs.append(trace_distance(prototypes[c], prototypes[c_prime]))
    l_inter = -torch.stack(inter_pairs).mean() if inter_pairs else torch.tensor(0.0, device=device)

    loss = l_ce + lambda1 * l_intra + lambda2 * l_inter
    return loss, l_ce, l_intra, l_inter
