"""FGSM / PGD adversarial attacks on the classical feature input."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from scripts.utils import expectations_to_tensor, to_np_x, to_torch_x


def ce_loss_on_x(x, y, theta, classifier_head, ce_loss_fn, forward_circuit, device):
    """
    CE loss on a single sample with grad enabled on the input features.
    """
    x = to_torch_x(x, device=device).detach().requires_grad_(True)
    y_t = torch.tensor([int(y)], device=device, dtype=torch.long)
    z, _ = forward_circuit(x, theta)
    logits = classifier_head(expectations_to_tensor(z)).unsqueeze(0)
    # attacks use unweighted CE even if training CE is class-weighted
    if getattr(ce_loss_fn, "weight", None) is None:
        loss = ce_loss_fn(logits, y_t)
    else:
        loss = nn.CrossEntropyLoss()(logits, y_t)
    return loss, x


def fgsm_attack(
    x,
    y,
    theta,
    classifier_head,
    forward_circuit,
    eps,
    device,
    x_min=None,
    x_max=None,
    ce_loss_fn=None,
):
    """
    One-step FGSM: x_adv = x + eps * sign(grad_x L_CE).
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    loss, x_var = ce_loss_on_x(
        x, y, theta, classifier_head, ce_loss_fn, forward_circuit, device
    )
    loss.backward()
    x_adv = x_var + eps * x_var.grad.sign()
    if x_min is not None:
        x_adv = torch.maximum(x_adv, x_min)
    if x_max is not None:
        x_adv = torch.minimum(x_adv, x_max)
    return x_adv.detach()


def pgd_attack(
    x,
    y,
    theta,
    classifier_head,
    forward_circuit,
    eps,
    alpha,
    steps,
    device,
    x_min=None,
    x_max=None,
    ce_loss_fn=None,
):
    """
    Multi-step PGD with L_inf projection onto the eps-ball around x.
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    x0 = to_torch_x(x, device=device).detach()
    x_adv = x0.clone()
    for _ in range(steps):
        loss, x_var = ce_loss_on_x(
            x_adv, y, theta, classifier_head, ce_loss_fn, forward_circuit, device
        )
        loss.backward()
        x_adv = x_var + alpha * x_var.grad.sign()

        # project onto L_inf ball around x0
        x_adv = torch.maximum(torch.minimum(x_adv, x0 + eps), x0 - eps)
        if x_min is not None:
            x_adv = torch.maximum(x_adv, x_min)
        if x_max is not None:
            x_adv = torch.minimum(x_adv, x_max)
        x_adv = x_adv.detach()
    return x_adv


def eval_attacked(
    attack_fn,
    X,
    y,
    theta,
    classifier_head,
    forward_circuit,
    device,
    labels=None,
    **attack_kwargs,
):
    """
    Run an attack over a dataset and report accuracy + macro-F1.
    """
    y_true = to_np_x(y).astype(int)
    preds = []
    with torch.enable_grad():
        for i, x in enumerate(X):
            y_i = int(y_true[i])
            x_adv = attack_fn(
                x,
                y_i,
                theta,
                classifier_head,
                forward_circuit,
                device=device,
                **attack_kwargs,
            )
            with torch.no_grad():
                z, _ = forward_circuit(x_adv, theta)
                pred = int(classifier_head(expectations_to_tensor(z)).argmax().item())
            preds.append(pred)

    preds = np.asarray(preds)
    return {
        "acc": float(np.mean(y_true == preds)),
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                average="macro",
                labels=labels,
                zero_division=0,
            )
        ),
        "y_true": y_true,
        "y_pred": preds,
    }
