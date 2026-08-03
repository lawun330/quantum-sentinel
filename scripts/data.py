"""Dataset loading, subsampling, and class-balance helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

from scripts.constants import DEFAULT_SEED, DEFAULT_MIN_PCT
from scripts.utils import to_np_y

# ============================================================================
# Dataset loading and preprocessing helpers
# ============================================================================

def load_split(data_path, name, column_name, categories=None, csv=False, selected_cols=None, return_df=False):
    """
    Load a split of the dataset from a csv or parquet file.

    - data_cols: list of ALL columns in the dataset
    - selected_cols: list of columns to select from the dataset (e.g. FROZEN features)
    - feature_cols: list of columns to use as features that may or may not be explicitly selected
    - categories: optional fixed label order. If None, classes are taken from the
      label column in this file (sorted). Pass an explicit list when loading
      several splits so integer codes stay aligned across train/test/etc.
    - selected_cols: columns to use as features (e.g. FROZEN features); if None,
      all non-label columns are used.
    - return_df: return the dataframe with the selected columns and label column or not
    """
    if csv:
        df = pd.read_csv(f"{data_path}/{name}.csv")
    else:
        df = pd.read_parquet(f"{data_path}/{name}.parquet")
    df = df[df[column_name].notna()].copy()
    data_cols = list(df.columns)

    if selected_cols is None:
        # no feature selection, use all columns except the label column
        feature_cols = [c for c in data_cols if not c.startswith("label")]
    else:
        # feature selection, check if any label column is selected
        if any(c.startswith("label") for c in selected_cols):
            raise ValueError("Label column cannot be selected for feature selection")
        # check if all selected columns are in the dataset
        missing_cols = set(selected_cols) - set(data_cols)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in dataset")
        # use only the selected columns
        feature_cols = list(selected_cols)

    if categories is None:
        categories = sorted(df[column_name].unique().tolist())
    else:
        categories = list(categories)

    X = df[feature_cols].values
    y = pd.Categorical(df[column_name], categories=categories).codes

    # drop negative labels (i.e. unknown labels)
    mask = y >= 0
    X, y, df = X[mask], y[mask], df.loc[mask].reset_index(drop=True)

    if return_df:
        return X, y, categories, df[feature_cols + [column_name]].copy()
    return X, y, categories


def stratified_head(X, y, n, seed=DEFAULT_SEED, return_index=False):
    """
    Take a stratified subset of size n (or all if n >= len(X)).
    """
    n_rows = len(X)
    if n >= n_rows:
        idx = np.arange(n_rows)
        X_sub, y_sub = X.copy(), np.asarray(y).copy()
    else:
        _, idx = train_test_split(
            np.arange(n_rows),
            test_size=n, stratify=y, random_state=seed,
        )
        X_sub, y_sub = X[idx], np.asarray(y)[idx]
    if return_index:
        return X_sub, y_sub, idx
    return X_sub, y_sub


def balanced_sample(X, y, n_per_class=None, seed=DEFAULT_SEED):
    """
    Undersample each class to have the same count (min class size by default).
    
    Used for quick, strictly-balanced baseline runs.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    classes = np.unique(y)

    if n_per_class is None:
        n_per_class = min(int((y == c).sum()) for c in classes)

    idxs = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) < n_per_class:
            raise ValueError(f"class {c} has only {len(c_idx)} samples, need {n_per_class}")
        idxs.append(rng.choice(c_idx, size=n_per_class, replace=False))

    idxs = rng.permutation(np.concatenate(idxs))
    return X[idxs], y[idxs]


def capped_sample(X, y, per_class_cap, seed=DEFAULT_SEED):
    """
    Cap each class at `per_class_cap` samples (no replacement) WITHOUT forcing all classes down to the smallest one.
    
    Used with `class_weights_for_sampler()` for real training.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    idxs = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        take = min(per_class_cap, len(c_idx))
        idxs.append(rng.choice(c_idx, size=take, replace=False))
    idxs = rng.permutation(np.concatenate(idxs))
    return X[idxs], y[idxs]


def to_angles(X_raw, scaler, x_min, x_max, pca=None, angle_max=np.pi):
    """
    Convert raw features to angles.
    """
    Xs = scaler.transform(X_raw)
    Xp = pca.transform(Xs) if pca is not None else Xs
    return (Xp - x_min) / (x_max - x_min + 1e-12) * angle_max


def class_weights_for_sampler(y, n_classes):
    """
    Per-sample inverse-frequency weights for torch's WeightedRandomSampler(weights, num_samples=len(y), replacement=True).

    Used with `capped_sample()` for real training.
    """
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=n_classes)
    class_w = 1.0 / np.maximum(counts, 1)
    return class_w[y]

# ============================================================================
# Geometric Coreset Sampling (k-Center and Greedy DPP)
# ============================================================================

def greedy_kcenter_sample(X, y, per_class_cap, seed=DEFAULT_SEED):
    """
    Selects `per_class_cap` samples per class using Greedy k-Center.

    Guarantees that sparse edges and multi-modal boundaries are preserved,
    preventing mode-collapse in the training subset.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    selected_indices = []

    # Scale internally purely for distance calculation so features with
    # larger scales don't dominate the geometric selection.
    X_scaled = StandardScaler().fit_transform(X)

    for c in classes:
        mask = (y == c)
        X_c = X_scaled[mask]
        idx_c = np.where(mask)[0]

        n_samples = len(idx_c)
        cap = min(per_class_cap, n_samples)

        if cap == n_samples:
            selected_indices.extend(idx_c)
            continue

        sel = [rng.integers(0, n_samples)]
        min_dists = euclidean_distances(X_c, X_c[sel[0]:sel[0]+1]).flatten()

        for _ in range(1, cap):
            min_dists_masked = min_dists.copy()
            min_dists_masked[sel] = -1
            next_i = np.argmax(min_dists_masked)
            sel.append(next_i)
            dists_new = euclidean_distances(X_c, X_c[next_i:next_i+1]).flatten()
            min_dists = np.minimum(min_dists, dists_new)

        selected_indices.extend(idx_c[sel])

    selected_indices = rng.permutation(np.array(selected_indices))
    return X[selected_indices], y[selected_indices]


def greedy_dpp_sample(X, y, per_class_cap, seed=DEFAULT_SEED):
    """
    Selects `per_class_cap` samples per class using Greedy Determinantal Point Processes.

    Strictly superior to k-Center: it covers sparse edges (like k-center) but also
    maintains proper density representation in the dense center of the data manifold.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    selected_indices = []

    X_scaled = StandardScaler().fit_transform(X)

    for c in classes:
        mask = (y == c)
        X_c = X_scaled[mask]
        idx_c = np.where(mask)[0]

        n_samples = len(idx_c)
        cap = min(per_class_cap, n_samples)

        if cap == n_samples:
            selected_indices.extend(idx_c)
            continue

        gamma = 1.0 / X_c.shape[1]
        sel = [rng.integers(0, n_samples)]

        dists_sq = euclidean_distances(X_c, X_c[sel[0]:sel[0]+1]).flatten() ** 2
        L_diag = np.exp(-gamma * dists_sq)

        for _ in range(1, cap):
            L_masked = L_diag.copy()
            L_masked[sel] = -1
            next_i = np.argmax(L_masked)
            sel.append(next_i)

            dists_sq = euclidean_distances(X_c, X_c[next_i:next_i+1]).flatten() ** 2
            K_new = np.exp(-gamma * dists_sq)

            # Efficient incremental update avoiding O(k^3) matrix inversion
            L_diag = L_diag - (K_new ** 2) / (1 + L_diag[sel[-1]] + 1e-10)
            L_diag = np.clip(L_diag, 0, 1)

        selected_indices.extend(idx_c[sel])

    selected_indices = rng.permutation(np.array(selected_indices))
    return X[selected_indices], y[selected_indices]


def _fast_kcenter_idx_2d(X_2d, y, cap, seed):
    """
    Internal helper: runs k-center directly on 2D PCA for fast visualization.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    indices = []
    for c in np.unique(y):
        mask = (y == c)
        X_c, idx_c = X_2d[mask], np.where(mask)[0]
        cap_c = min(cap, len(idx_c))
        if cap_c == len(idx_c): indices.extend(idx_c); continue
        sel = [rng.integers(0, len(idx_c))]
        min_dists = euclidean_distances(X_c, X_c[sel[0]:sel[0]+1]).flatten()
        for _ in range(1, cap_c):
            min_dists_masked = min_dists.copy(); min_dists_masked[sel] = -1
            next_i = np.argmax(min_dists_masked); sel.append(next_i)
            dists_new = euclidean_distances(X_c, X_c[next_i:next_i+1]).flatten()
            min_dists = np.minimum(min_dists, dists_new)
        indices.extend(idx_c[sel])
    return np.array(indices)


def _fast_dpp_idx_2d(X_2d, y, cap, seed):
    """
    Internal helper: runs DPP directly on 2D PCA for fast visualization.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    indices = []
    for c in np.unique(y):
        mask = (y == c)
        X_c, idx_c = X_2d[mask], np.where(mask)[0]
        cap_c = min(cap, len(idx_c))
        if cap_c == len(idx_c): indices.extend(idx_c); continue
        gamma = 1.0 / X_c.shape[1]
        sel = [rng.integers(0, len(idx_c))]
        dists_sq = euclidean_distances(X_c, X_c[sel[0]:sel[0]+1]).flatten()**2
        L_diag = np.exp(-gamma * dists_sq)
        for _ in range(1, cap_c):
            L_masked = L_diag.copy(); L_masked[sel] = -1
            next_i = np.argmax(L_masked); sel.append(next_i)
            dists_sq = euclidean_distances(X_c, X_c[next_i:next_i+1]).flatten()**2
            K_new = np.exp(-gamma * dists_sq)
            L_diag = L_diag - (K_new**2) / (1 + L_diag[sel[-1]] + 1e-10)
            L_diag = np.clip(L_diag, 0, 1)
        indices.extend(idx_c[sel])
    return np.array(indices)

# ============================================================================
# Visualization helpers
# ============================================================================

def class_balance_table(y, class_names):
    """
    Return per-class counts and percentages for label vector y (zeros included).
    """
    y = to_np_y(y).astype(int)
    counts = pd.Series(y).value_counts()
    df = pd.DataFrame({
        "class": class_names,
        "count": [int(counts.get(i, 0)) for i in range(len(class_names))],
    })
    df["pct"] = 100 * df["count"] / df["count"].sum()
    return df


def plot_class_balance_pie(y, class_names, title="Class balance", ax=None, min_pct=DEFAULT_MIN_PCT):
    """
    Plot a pie chart of class frequencies for label vector y (zeros dropped).
    """
    df = class_balance_table(y, class_names)
    df = df[df["count"] > 0]    # pie can't show 0-count slices cleanly

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    def autopct(p):
        return f"{p:.1f}%" if p >= min_pct else ""
    wedges, _, _ = ax.pie(
        df["count"],
        labels=None,
        autopct=autopct,
        startangle=90,
        pctdistance=0.7,
    )
    ax.legend(
        wedges,
        [f"{c}: {n} ({p:.1f}%)" for c, n, p in zip(df["class"], df["count"], df["pct"])],
        title="class",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
    )
    ax.set_title(title)
    return ax


def plot_class_balance_bars(y, class_names, title="Class balance", ax=None):
    """
    Plot a bar chart of class frequencies for label vector y (zeros dropped).
    """
    df = class_balance_table(y, class_names).sort_values("count")

    if ax is None:
        height = max(4, 0.4 * len(df))
        _, ax = plt.subplots(figsize=(8, height))

    ax.barh(df["class"], df["count"])
    ax.set_xlabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=8)

    for i, (n, p) in enumerate(zip(df["count"], df["pct"])):
        ax.text(n, i, f" {n} ({p:.1f}%)", va="center", fontsize=8)

    plt.tight_layout()
    return ax


def plot_dataset_and_sampling_analysis(X_raw, y_raw, per_class_cap, class_names, seed=DEFAULT_SEED):
    """
    Plots a 2x3 grid showing:
    Row 1: True Raw Data Distribution (Geometric 2D PCA & Class Frequencies)
    Row 2: Sampling Strategy Comparison (Random vs k-Center vs DPP)
    """
    y_raw = np.asarray(y_raw).astype(int)
    n_classes = len(class_names)

    # 1. Project to 2D for fast geometric visualization
    pca_2d = PCA(n_components=2, random_state=seed)
    X_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X_raw))

    # 2. Calculate indices for the 3 sampling methods on the 2D projection
    rng = np.random.default_rng(seed)
    random_idx = np.array([], dtype=int)
    for c in np.unique(y_raw):
        cls_idx = np.where(y_raw == c)[0]
        random_idx = np.concatenate([random_idx, rng.choice(cls_idx, size=min(per_class_cap, len(cls_idx)), replace=False)])

    kcenter_idx = _fast_kcenter_idx_2d(X_2d, y_raw, per_class_cap, seed)
    dpp_idx = _fast_dpp_idx_2d(X_2d, y_raw, per_class_cap, seed)

    # 3. Setup Plot
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.25)

    try:
        cmap = plt.cm.get_cmap('tab10', n_classes)
    except:
        cmap = plt.colormaps['tab10'].resampled(n_classes)  # for version mismatch

    # --- TOP LEFT: True Raw Geometric ---
    ax_raw_geo = fig.add_subplot(gs[0, 0])
    for c_id, class_name in enumerate(class_names):
        mask = (y_raw == c_id)
        ax_raw_geo.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap(c_id)],
                           label=f"{class_name} (n={mask.sum():,})", s=2, alpha=0.3, rasterized=True)
    ax_raw_geo.set_title("True Raw Geometric Distribution\n(2D PCA Projection)", fontweight='bold')
    ax_raw_geo.legend(markerscale=4, loc='best', fontsize=7)
    ax_raw_geo.grid(True, alpha=0.2)

    # --- TOP MIDDLE & RIGHT: True Raw Frequency ---
    ax_raw_freq = fig.add_subplot(gs[0, 1:])
    unique_classes, counts = np.unique(y_raw, return_counts=True)
    bar_colors = [cmap(i) for i in unique_classes]
    bars = ax_raw_freq.bar([class_names[i] for i in unique_classes], counts, color=bar_colors)
    ax_raw_freq.set_title("True Raw Frequency Distribution (Class Imbalance)\nBefore Capping", fontweight='bold')
    ax_raw_freq.set_ylabel("Number of Samples")
    ax_raw_freq.set_xlabel("Class")
    for bar, count in zip(bars, counts):
        ax_raw_freq.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(counts)*0.01,
                         f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    ax_raw_freq.tick_params(axis='x', rotation=0)

    # --- BOTTOM ROW: Sampling Comparisons ---
    configs = [
        (random_idx, "Random Sampling", "royalblue", "Clumps in dense center,\nmisses sparse edges"),
        (kcenter_idx, "Greedy k-Center", "darkorange", "Hits sparse edges,\nbut leaves holes in center"),
        (dpp_idx, "Greedy DPP (Best)", "crimson", "Covers edges AND dense\ncenter uniformly")
    ]

    for i, (idx, title, color, subtitle) in enumerate(configs):
        ax = fig.add_subplot(gs[1, i])
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=3, alpha=0.3, label='Discarded')
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color, s=4, alpha=0.7, label=f'Selected ({len(idx):,})')
        ax.set_title(f"{title}\n({subtitle})", fontsize=10, fontweight='bold')
        ax.legend(markerscale=3, loc='upper right', fontsize=7)
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"Dataset Analysis & Geometric Impact of Capping Strategy (Cap={per_class_cap:,}/class)", fontsize=14, y=1.01)
    plt.show()