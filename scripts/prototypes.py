"""Algorithm 1: Class prototype bank and prototype computation for MAQT."""

import torch

from scripts.utils import to_np_x, to_torch_x


class PrototypeBank:
    """Running average of rho(x) per class during MAQT minibatches (PyTorch version)"""

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


def compute_prototypes(theta, X, y, classes, forward_circuit, device=None):
    """
    Build final class prototypes by averaging noisy states over training samples one circuit at a time, with no stack. (PyTorch version).
    """
    prototypes = {}

    y_labels = to_np_x(y).astype(int)

    with torch.no_grad():
        for c in classes:
            mask = y_labels == c
            if not mask.any():
                continue

            proto_sum = None
            count = 0

            for x in X[mask]:
                _, rho_x = forward_circuit(to_torch_x(x, device=device), theta)

                if proto_sum is None:
                    proto_sum = rho_x.detach().clone()
                else:
                    proto_sum = proto_sum + rho_x.detach()
                count += 1

                del rho_x

            prototypes[c] = proto_sum / count
            del proto_sum

    return prototypes


def prototype_summary(prototypes, class_names=None):
    """
    Summarize the prototype density matrices.
    """
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
