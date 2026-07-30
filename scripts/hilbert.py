"""H1 Hilbert-geometry diagnostics + trust-region 2D projections."""

import numpy as np
import torch

from scripts.quantum_metrics import fidelity, trace_distance
from scripts.utils import to_np_y, to_torch_batch_x
from scripts.constants import DEFAULT_SEED, DEFAULT_BATCH_SIZE


@torch.no_grad()
def encode_density_matrices(X, theta, forward_circuit, device=None, batch_size=DEFAULT_BATCH_SIZE):
    """
    Run the forward circuit and collect post-channel density matrices rho(x).

    Return a CPU tensor of shape (n, d, d).
    """
    rhos = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = to_torch_batch_x(X[i : i + batch_size], device=device)
        _, rho = forward_circuit(xb, theta)
        rhos.append(rho.detach().cpu())
    return torch.cat(rhos, dim=0)


@torch.no_grad()
def pairwise_trace_distance_matrix(rhos):
    """
    Pairwise trace distance on a stack of density matrices.

    Return a float64 numpy array of shape (N, N), zeros on the diagonal.
    """
    if torch.is_tensor(rhos):
        rhos_t = rhos.detach().cpu().to(torch.complex128)
    else:
        rhos_t = torch.as_tensor(rhos, dtype=torch.complex128)
    n = rhos_t.shape[0]
    dist = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        rest = rhos_t[i + 1 :]
        if rest.shape[0] == 0:
            continue
        diff = rhos_t[i].unsqueeze(0) - rest
        diff = 0.5 * (diff + diff.mH)
        evals = torch.linalg.eigvalsh(diff)
        td = 0.5 * evals.abs().sum(dim=-1).real.clamp(0.0, 1.0)
        dist[i, i + 1 :] = td
        dist[i + 1 :, i] = td
    return dist.numpy()


@torch.no_grad()
def fidelity_to_prototypes_matrix(rhos, prototypes):
    """
    Each state -> vector of F(rho, rho_c) over sorted prototype ids.
    """
    class_ids = sorted(prototypes)
    proto_stack = torch.stack(
        [prototypes[c].detach().cpu() for c in class_ids], dim=0
    ).to(torch.complex128)
    rhos_t = rhos.detach().cpu() if torch.is_tensor(rhos) else torch.as_tensor(rhos)
    feats = np.zeros((rhos_t.shape[0], len(class_ids)), dtype=np.float64)
    for j, c in enumerate(class_ids):
        for i in range(rhos_t.shape[0]):
            f = fidelity(rhos_t[i], proto_stack[j])
            feats[i, j] = float(f.real.item() if torch.is_tensor(f) else f)
    return class_ids, feats


def mds_2d_from_distances(dist, random_state=DEFAULT_SEED):
    """
    Classical MDS to 2D from a precomputed distance matrix.
    """
    from sklearn.manifold import MDS

    kwargs = dict(
        n_components=2,
        random_state=random_state,
        normalized_stress="auto",
        init="random",
    )
    try:
        embedding = MDS(metric="precomputed", **kwargs)
    except TypeError:
        embedding = MDS(dissimilarity="precomputed", **kwargs)
    return embedding.fit_transform(dist)


def pca_2d(features):
    """
    PCA to 2D (used on fidelity-to-prototype features).
    """
    from sklearn.decomposition import PCA

    return PCA(n_components=2).fit_transform(features)


def fidelity_gap_proxy(prototypes, l_intra):
    """
    Cheap H1 proxy for logging during training.
    mean_intra ≈ 1 - L_intra; mean_inter = mean pairwise F(ρ_c, ρ_c').
    """
    class_ids = sorted(prototypes)
    inter_fids = []
    for i, c in enumerate(class_ids):
        for c2 in class_ids[i + 1 :]:
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
def hilbert_geometry_diagnostics(theta, X, y, prototypes, forward_circuit, class_names=None,
                                device=None, max_per_class=None, seed=DEFAULT_SEED, batch_size=DEFAULT_BATCH_SIZE,
):
    """
    Compute H1 (intra/inter fidelity gaps) in Hilbert space.
    """
    y_np = to_np_y(y).astype(int)
    rng = np.random.default_rng(seed)

    class_ids = sorted(prototypes)
    per_class = {}
    mean_intra_fid_per_class = []

    for c in class_ids:
        idx = np.where(y_np == c)[0]
        if len(idx) == 0:
            continue
        if max_per_class is not None and len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)

        fids = []
        for i in range(0, len(idx), batch_size): # batch the forward passes
            chunk_idx = idx[i:i + batch_size]
            x_chunk = to_torch_batch_x(X[chunk_idx], device=device)
            _, rho_chunk = forward_circuit(x_chunk, theta)
            for j in range(rho_chunk.shape[0]):
                f = fidelity(rho_chunk[j], prototypes[c])
                fids.append(float(f.real.item() if torch.is_tensor(f) else f))

        # intra-class fidelity
        mean_intra_fid_c = float(np.mean(fids))
        name = class_names[c] if class_names is not None else str(c)
        per_class[name] = {
            "n": int(len(fids)),
            "mean_intra_fid_c": mean_intra_fid_c,
            "mean_intra_infidelity_c": 1.0 - mean_intra_fid_c,  # same spirit as L_intra
        }
        mean_intra_fid_per_class.append(mean_intra_fid_c)

    # inter-class fidelity + trace distance between prototypes
    pair_inter_fids, pair_inter_tds = [], []
    pair_rows = []
    for i, c in enumerate(class_ids):
        for c2 in class_ids[i + 1 :]: # pair-wise inter-class fidelity
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

    # fidelity gap
    mean_intra_fid = float(np.mean(mean_intra_fid_per_class)) if mean_intra_fid_per_class else float("nan")
    mean_inter_fid = float(np.mean(pair_inter_fids)) if pair_inter_fids else float("nan")
    fidelity_gap = mean_intra_fid - mean_inter_fid

    return {
        "mean_intra_fid": mean_intra_fid,
        "mean_inter_fid": mean_inter_fid,
        "fidelity_gap": fidelity_gap,
        "mean_inter_trace_distance": float(np.mean(pair_inter_tds)) if pair_inter_tds else float("nan"),
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