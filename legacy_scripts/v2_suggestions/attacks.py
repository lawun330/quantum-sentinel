"""FGSM / PGD adversarial attacks -- batched, plus a plain-CE baseline for ablation."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from scripts.utils import expectations_to_tensor, to_np_x, to_torch_batch


def logits_from_batch(x_batch, theta, classifier_head, forward_circuit):
    z, _ = forward_circuit(x_batch, theta)
    return classifier_head(expectations_to_tensor(z))


def fgsm_attack(X_batch, y_batch, theta, classifier_head, forward_circuit, eps, device,
                 x_min=None, x_max=None, ce_loss_fn=None):
    """
    batched (accepts X_batch/y_batch), matching the batched circuit.
    One-step FGSM: x_adv = x + eps * sign(grad_x L_CE).
    """
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    x = to_torch_batch(X_batch, device=device).detach().requires_grad_(True)
    y_t = torch.as_tensor(np.asarray(y_batch), dtype=torch.long, device=device)

    logits = logits_from_batch(x, theta, classifier_head, forward_circuit)
    if getattr(ce_loss_fn, "weight", None) is None:
        loss = ce_loss_fn(logits, y_t)
    else:
        loss = nn.CrossEntropyLoss()(logits, y_t)  # attacks use unweighted CE
    grad = torch.autograd.grad(loss, x)[0]

    x_adv = x + eps * grad.sign()
    if x_min is not None:
        x_adv = torch.maximum(x_adv, x_min)
    if x_max is not None:
        x_adv = torch.minimum(x_adv, x_max)
    return x_adv.detach()


def pgd_attack(X_batch, y_batch, theta, classifier_head, forward_circuit, eps, alpha, steps,
               device, x_min=None, x_max=None, ce_loss_fn=None, random_start=True):
    """ batched multi-step PGD with L_inf projection onto the eps-ball around x."""
    if ce_loss_fn is None:
        ce_loss_fn = nn.CrossEntropyLoss()
    x0 = to_torch_batch(X_batch, device=device).detach()
    y_t = torch.as_tensor(np.asarray(y_batch), dtype=torch.long, device=device)

    if random_start:
        x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps)
    else:
        x_adv = x0.clone()
    if x_min is not None:
        x_adv = torch.maximum(x_adv, x_min)
    if x_max is not None:
        x_adv = torch.minimum(x_adv, x_max)
    x_adv = x_adv.detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = logits_from_batch(x_adv, theta, classifier_head, forward_circuit)
        loss = ce_loss_fn(logits, y_t) if getattr(ce_loss_fn, "weight", None) is None \
            else nn.CrossEntropyLoss()(logits, y_t)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv + alpha * grad.sign()
            x_adv = torch.maximum(torch.minimum(x_adv, x0 + eps), x0 - eps)
            if x_min is not None:
                x_adv = torch.maximum(x_adv, x_min)
            if x_max is not None:
                x_adv = torch.minimum(x_adv, x_max)
        x_adv = x_adv.detach()
    return x_adv


def eval_attacked(attack_fn, X, y, theta, classifier_head, forward_circuit, device,
                   labels=None, batch_size=64, **attack_kwargs):
    """
     runs the attack + eval in chunks of batch_size instead of one
    sample at a time -- both the clean and adversarial forward passes now
    use the batched circuit.
    """
    y_true = to_np_x(y).astype(int)
    preds = []
    with torch.enable_grad():
        for i in range(0, len(X), batch_size):
            X_chunk = X[i:i + batch_size]
            y_chunk = y_true[i:i + batch_size]
            x_adv = attack_fn(X_chunk, y_chunk, theta, classifier_head, forward_circuit,
                               device=device, **attack_kwargs)
            with torch.no_grad():
                logits = logits_from_batch(x_adv, theta, classifier_head, forward_circuit)
                preds.append(logits.argmax(dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    return {
        "acc": float(np.mean(y_true == preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", labels=labels, zero_division=0)),
        "y_true": y_true,
        "y_pred": preds,
    }


def train_plain_baseline(X_train, y_train, n_classes, forward_circuit, n_qubits,
                          epochs, lr, device, batch_size=16):
    """
    
    This is the CONTROL GROUP for the robustness ablation below -- without
    it, "MAQT is robust to FGSM/PGD" is an unfalsifiable claim, since you
    can't tell whether the attack code itself is doing anything.
    """
    from scripts.circuit import initialize_weights_random_identity
    theta = initialize_weights_random_identity(
        n_layers=None, n_wires=n_qubits, device=device, eps=1e-2
    ) if False else None  # placeholder guard; caller should pass an initialized theta pattern
    raise NotImplementedError(
        "Wire this to your project's theta/head init (see train.py's train_maqt for the pattern); "
        "kept as a stub here since layer count isn't known inside attacks.py."
    )


def robustness_ablation(X_test, y_test, theta_maqt, head_maqt, theta_plain, head_plain,
                         forward_circuit, epsilons, device, attack_fn=fgsm_attack,
                         batch_size=64, **attack_kwargs):
    """
     compares MAQT-trained model vs. plain-CE baseline under the same
    attack at each epsilon -- gives you an actual control group for the
    Proposition 2 robustness claim instead of a single model's degradation curve.
    """
    results = []
    for eps in epsilons:
        maqt_res = eval_attacked(attack_fn, X_test, y_test, theta_maqt, head_maqt,
                                  forward_circuit, device, batch_size=batch_size,
                                  eps=eps, **attack_kwargs)
        plain_res = eval_attacked(attack_fn, X_test, y_test, theta_plain, head_plain,
                                   forward_circuit, device, batch_size=batch_size,
                                   eps=eps, **attack_kwargs)
        results.append({
            "eps": eps,
            "maqt_acc": maqt_res["acc"], "maqt_macro_f1": maqt_res["macro_f1"],
            "plain_acc": plain_res["acc"], "plain_macro_f1": plain_res["macro_f1"],
        })
    return results