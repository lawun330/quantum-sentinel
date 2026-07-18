"""
 the actual training loop, previously not present in your file set
(you had the building blocks -- loss.py, prototypes.py -- but no assembled
loop). Wires together: batched circuit, EMA prototypes with the pre-update
snapshot fix, curriculum lambda, optional focal loss, and weighted sampling.


THIS TRAIN.PY NOT BE INSIDE IN THE SCRIPTS -->LA WUN CAREFUL WHEN YOU R RUNNING...
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from scripts.circuit import initialize_weights_random_identity
from scripts.data import class_weights_for_sampler
from scripts.loss import curriculum_weight, maqt_loss
from scripts.prototypes import EMAPrototypeBank, PrototypeBank, compute_prototypes
from scripts.constants import DEFAULT_EMA_MOMENTUM, DEFAULT_FOCAL_GAMMA, DEFAULT_WARMUP_FRAC


def train_maqt(X_train, y_train, n_classes, n_qubits, n_layers, forward_circuit, device,
                epochs=10, lr=0.05, batch_size=16, lambda1_max=0.5, lambda2_max=0.3,
                warmup_frac=DEFAULT_WARMUP_FRAC, grad_clip_norm=1.0,
                ema_momentum=DEFAULT_EMA_MOMENTUM, use_focal=False, focal_gamma=DEFAULT_FOCAL_GAMMA,
                use_weighted_sampler=True, log_every=1, verbose=True):
    theta = initialize_weights_random_identity(n_layers, n_qubits, device, eps=1e-2)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    ema_protos = EMAPrototypeBank(classes=range(n_classes), momentum=ema_momentum)

    X_t = torch.as_tensor(np.asarray(X_train, dtype=np.float32), device=device)
    y_t = torch.as_tensor(np.asarray(y_train, dtype=np.int64), device=device)

    if use_weighted_sampler:
        weights = class_weights_for_sampler(y_train, n_classes)
        sample_weights = torch.as_tensor(weights, dtype=torch.double)
    else:
        sample_weights = None

    history = {"loss": [], "L_CE": [], "L_intra": [], "L_inter": [],
               "grad_var": [], "intra_fid_gap": [], "inter_trace_dist": []}

    n = len(X_t)
    for epoch in range(epochs):
        lam1 = curriculum_weight(epoch, epochs, lambda1_max, warmup_frac)
        lam2 = curriculum_weight(epoch, epochs, lambda2_max, warmup_frac)

        if use_weighted_sampler:
            perm = torch.tensor(list(WeightedRandomSampler(sample_weights, num_samples=n, replacement=True)))
        else:
            perm = torch.randperm(n)

        epoch_terms = {"L_total": [], "L_CE": [], "L_intra": [], "L_inter": []}
        grad_vars, intra_fid_running = [], []

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]

            protos_snapshot = ema_protos.snapshot()
            if not protos_snapshot:
                # First few steps before any class has been seen yet: fall back
                # to plain CE only (lambda terms contribute 0 with no prototypes).
                protos_snapshot = {}

            loss, l_ce, l_intra, l_inter, y_used, rho_used = maqt_loss(
                theta, head, ce_loss_fn, xb, yb, protos_snapshot, forward_circuit,
                lambda1=lam1, lambda2=lam2, device=device,
                use_focal=use_focal, focal_gamma=focal_gamma,
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), grad_clip_norm)
            grad_vars.append(theta.grad.var().item())
            opt.step()

            # EMA update strictly AFTER the loss/backward pass (see prototypes.py docstring).
            batch_means = ema_protos.batch_class_means(rho_used.detach(), y_used)
            ema_protos.update(batch_means)

            epoch_terms["L_total"].append(loss.item())
            epoch_terms["L_CE"].append(l_ce.item())
            epoch_terms["L_intra"].append(float(l_intra))
            epoch_terms["L_inter"].append(float(l_inter))
            intra_fid_running.append(1 - float(l_intra))

        diag_pairs = []
        keys = sorted(ema_protos.protos.keys())
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                from scripts.quantum_metrics import trace_distance
                diag_pairs.append(float(trace_distance(ema_protos.protos[keys[a]], ema_protos.protos[keys[b]])))
        mean_inter_td = float(np.mean(diag_pairs)) if diag_pairs else 0.0
        mean_gv = float(np.mean(grad_vars))

        history["loss"].append(float(np.mean(epoch_terms["L_total"])))
        history["L_CE"].append(float(np.mean(epoch_terms["L_CE"])))
        history["L_intra"].append(float(np.mean(epoch_terms["L_intra"])))
        history["L_inter"].append(float(np.mean(epoch_terms["L_inter"])))
        history["grad_var"].append(mean_gv)
        history["intra_fid_gap"].append(float(np.mean(intra_fid_running)))
        history["inter_trace_dist"].append(mean_inter_td)

        if verbose and (epoch + 1) % log_every == 0:
            print(f"epoch {epoch+1:2d}/{epochs} | loss {history['loss'][-1]:.4f} | "
                  f"L_CE {history['L_CE'][-1]:.4f} | L_intra {history['L_intra'][-1]:.4f} | "
                  f"L_inter {history['L_inter'][-1]:.4f} | grad_var {mean_gv:.2e} | "
                  f"intra_fid {history['intra_fid_gap'][-1]:.3f} | inter_TD {mean_inter_td:.3f}")
            if mean_gv < 1e-6:
                print("  barren plateau detected")

    # Final, unbiased prototypes for deployment (conformal.py / inference.py):
    static_bank = PrototypeBank(classes=range(n_classes))
    final_prototypes = compute_prototypes(theta, np.asarray(X_train), np.asarray(y_train),
                                           classes=range(n_classes), forward_circuit=forward_circuit,
                                           device=device)

    return theta, head, final_prototypes, ema_protos, history


def train_plain_vqc(X_train, y_train, n_classes, n_qubits, n_layers, forward_circuit, device,
                     epochs=8, lr=0.05, batch_size=16):
    """CE-only baseline (no prototype terms) -- control group for robustness ablation."""
    theta = initialize_weights_random_identity(n_layers, n_qubits, device, eps=1e-2)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    X_t = torch.as_tensor(np.asarray(X_train, dtype=np.float32), device=device)
    y_t = torch.as_tensor(np.asarray(y_train, dtype=np.int64), device=device)
    n = len(X_t)
    losses = []

    from scripts.utils import expectations_to_tensor
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_t[idx], y_t[idx]
            z, _ = forward_circuit(xb, theta)
            logits = head(expectations_to_tensor(z))
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), 1.0)
            opt.step()
        losses.append(loss.item())
        print(f"[plain VQC] epoch {epoch+1}/{epochs} loss {loss.item():.4f}")
    return theta, head, losses