"""H1 Hilbert-geometry diagnostics (intra/inter fidelity gaps)."""

import numpy as np
import torch

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import to_np_x, to_torch_x


def fidelity_gap_proxy(prototypes, l_intra):
    """
    Cheap H1 proxy for logging during training.
    mean_intra ≈ 1 - L_intra; mean_inter = mean pairwise F(ρ_c, ρ_c').
    """
    classes = sorted(prototypes)
    inter_fids = []
    for i, c in enumerate(classes):
        for c2 in classes[i + 1 :]:
            f = fidelity(prototypes[c], prototypes[c2])
            inter_fids.append(float(f.real.item() if torch.is_tensor(f) else f))
    mean_intra_fid_proxy = 1.0 - float(l_intra)
    mean_inter_fid = float(np.mean(inter_fids)) if inter_fids else float("nan")
    return {
        "mean_intra_fid_proxy": mean_intra_fid_proxy,
        "mean_inter_fid": mean_inter_fid,
        "fidelity_gap_proxy": mean_intra_fid_proxy - mean_inter_fid,
    }


@torch.no_grad()
def hilbert_geometry_diagnostics(
    theta,
    X,
    y,
    prototypes,
    forward_circuit,
    class_names=None,
    device=None,
    max_per_class=None,
    seed=42,
):
    """
    Compute H1 (intra/inter fidelity gaps) in Hilbert space.
    """
    y_np = to_np_x(y).astype(int)
    rng = np.random.default_rng(seed)

    per_class = {}
    mean_intra_fid_per_class = []

    for c in sorted(prototypes):
        idx = np.where(y_np == c)[0]
        if len(idx) == 0:
            continue
        if max_per_class is not None and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)

        fids = []
        for i in idx:
            _, rho_x = forward_circuit(to_torch_x(X[i], device=device), theta)
            f = fidelity(rho_x, prototypes[c])
            fids.append(float(f.real.item() if torch.is_tensor(f) else f))

        mean_intra_fid_c = float(np.mean(fids))
        name = class_names[c] if class_names is not None else str(c)
        per_class[name] = {
            "n": int(len(fids)),
            "mean_intra_fid_c": mean_intra_fid_c,
            "mean_intra_infidelity_c": 1.0 - mean_intra_fid_c,  # same spirit as L_intra
        }
        mean_intra_fid_per_class.append(mean_intra_fid_c)

    # inter-class fidelity + trace distance between prototypes
    classes = sorted(prototypes)
    pair_inter_fids, pair_inter_tds = [], []
    pair_rows = []
    for i, c in enumerate(classes):
        for c2 in classes[i + 1 :]:
            f = fidelity(prototypes[c], prototypes[c2])
            td = trace_distance(prototypes[c], prototypes[c2])
            f = float(f.real.item() if torch.is_tensor(f) else f)
            td = float(td.real.item() if torch.is_tensor(td) else td)
            pair_inter_fids.append(f)
            pair_inter_tds.append(td)
            n1 = class_names[c] if class_names is not None else str(c)
            n2 = class_names[c2] if class_names is not None else str(c2)
            pair_rows.append(
                {"pair": f"{n1}\u2194{n2}", "pair_inter_fid": f, "pair_trace_distance": td}
            )

    mean_intra_fid = (
        float(np.mean(mean_intra_fid_per_class)) if mean_intra_fid_per_class else float("nan")
    )
    mean_inter_fid = float(np.mean(pair_inter_fids)) if pair_inter_fids else float("nan")
    fidelity_gap = mean_intra_fid - mean_inter_fid

    return {
        "mean_intra_fid": mean_intra_fid,
        "mean_inter_fid": mean_inter_fid,
        "fidelity_gap": fidelity_gap,  # H1 headline metric
        "mean_inter_trace_distance": (
            float(np.mean(pair_inter_tds)) if pair_inter_tds else float("nan")
        ),
        "per_class": per_class,
        "pairs": pair_rows,
    }


def print_h1_report(report):
    """
    Print a compact H1 fidelity-gap summary.
    """
    print("=== H1 Hilbert geometry (fidelity gaps) ===")
    print(f"mean intra-class fidelity : {report['mean_intra_fid']:.4f}")
    print(f"mean inter-class fidelity : {report['mean_inter_fid']:.4f}")
    print(f"fidelity gap (intra-inter): {report['fidelity_gap']:.4f}  \u2190 want \u2191")
    print(f"mean inter trace distance : {report['mean_inter_trace_distance']:.4f}  \u2190 want \u2191")
    print("\nper-class intra fidelity:")
    for name, row in report["per_class"].items():
        print(f"  {name:28s} n={row['n']:4d}  F={row['mean_intra_fid_c']:.4f}")
