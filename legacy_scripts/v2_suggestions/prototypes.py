"""Algorithm 1: Class prototype bank(s) for MAQT -- static (final) and EMA (live-training)."""

import torch

from scripts.utils import to_np_x, to_torch_batch


class PrototypeBank:
    """
    Running (exact, order-independent) average of rho(x) per class.
    Unchanged  original -- use this for the FINAL, unbiased
    prototype set (deployed for conformal.py calibration / inference.py),
    computed via compute_prototypes() below in one full pass with theta frozen.
    """

    def __init__(self, classes):
        self.classes = sorted(classes)
        self.reset()

    def reset(self):
        self._sums = {c: None for c in self.classes}
        self._counts = {c: 0 for c in self.classes}

    def update(self, label, rho):
        if self._sums[label] is None:
            self._sums[label] = rho.detach().clone()
        else:
            self._sums[label] = self._sums[label] + rho.detach()
        self._counts[label] += 1

    def means(self):
        return {
            c: self._sums[c] / self._counts[c]
            for c in self.classes
            if self._counts[c] > 0
        }


class EMAPrototypeBank:
    """
    NEW : exponential-moving-average
    prototype per class, updated every minibatch. Cheap (O(batch), no full
    pass needed), and adapts to representation drift during training -- use
    this as the LIVE supervision signal inside the training loop's loss,
    NOT as the final deployed prototype (use PrototypeBank.compute_prototypes
    for that).

    it computed
    the EMA update and the loss consumed the JUST-UPDATED prototype in the
    same step -- i.e. the target for L_intra was contaminated by the very
    samples being pulled toward it in that step (a same-step self-reference;
    the update() call does .detach() so no literal gradient leak, but the
    *target itself* still moves before the loss that uses it as a fixed
    point is computed, which is the collapse-risk pattern EMA-target
    architectures like MoCo/DINO specifically avoid with a delayed target).
    Here, snapshot() returns the pre-update prototype to use as the loss
    target; call update() AFTER computing the loss for that batch.
    """

    def __init__(self, classes, momentum=0.9):
        self.classes = sorted(classes)
        self.momentum = momentum
        self.protos = {}

    def batch_class_means(self, rho_batch, y_batch):
        means = {}
        for c in torch.unique(y_batch):
            c_int = int(c)
            mask = y_batch == c
            means[c_int] = rho_batch[mask].mean(dim=0)
        return means

    def snapshot(self):
        """Prototypes as of BEFORE this batch's update -- use as loss target."""
        return {c: p.clone() for c, p in self.protos.items()}

    def update(self, batch_means):
        """Call AFTER the loss for this batch has been computed."""
        for c_int, batch_mean in batch_means.items():
            if c_int not in self.protos:
                self.protos[c_int] = batch_mean.detach().clone()
            else:
                self.protos[c_int] = (
                    self.momentum * self.protos[c_int]
                    + (1 - self.momentum) * batch_mean.detach()
                )
        return self.protos


def compute_prototypes(theta, X, y, classes, forward_circuit, device=None, batch_size=64):
    """
    Build final class prototypes by averaging states over training samples,
    with theta frozen (torch.no_grad()).

    FIX: now processes samples in chunks of `batch_size` through the BATCHED
    forward_circuit instead of one circuit call per sample -- same math
    (exact mean), far fewer circuit traces.
    """
    prototypes = {}
    y_labels = to_np_x(y).astype(int)

    with torch.no_grad():
        for c in classes:
            idx = (y_labels == c).nonzero()[0] if hasattr((y_labels == c), "nonzero") else None
            idx = (y_labels == c).nonzero()[0]
            if len(idx) == 0:
                continue

            proto_sum = None
            count = 0
            for i in range(0, len(idx), batch_size):
                chunk_idx = idx[i:i + batch_size]
                x_chunk = to_torch_batch(X[chunk_idx], device=device)
                _, rho_chunk = forward_circuit(x_chunk, theta)
                chunk_sum = rho_chunk.sum(dim=0)

                if proto_sum is None:
                    proto_sum = chunk_sum.detach().clone()
                else:
                    proto_sum = proto_sum + chunk_sum.detach()
                count += len(chunk_idx)
                del rho_chunk, chunk_sum

            prototypes[c] = proto_sum / count
            del proto_sum

    return prototypes


def prototype_summary(prototypes, class_names=None):
    """Summarize the prototype density matrices."""
    rows = []
    for c in sorted(prototypes):
        rho_c = prototypes[c]
        name = class_names[c] if class_names is not None else str(c)
        rows.append({
            "class": name,
            "dim": int(rho_c.shape[0]),
            "trace": float(torch.trace(rho_c).real.item()),
        })
    return rows