"""Dataset loading, subsampling, and class-balance helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.utils import to_np_x


def load_split(data_path, name, column_name, categories):
    df = pd.read_parquet(f"{data_path}/{name}.parquet")
    df = df[df["label_multiclass"].notna()].copy()

    feature_cols = [c for c in df.columns if not c.startswith("label")]
    X = df[feature_cols].values
    y = pd.Categorical(df[column_name], categories=categories).codes

    return X, y


def stratified_head(X, y, n, seed=42):
    """
    Take a stratified subset of size n (or all if n >= len(X))
    """
    if n >= len(X):
        return X.copy(), y.copy()
    _, X_sub, _, y_sub = train_test_split(
        X, y, test_size=n, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def balanced_sample(X, y, n_per_class=None, seed=42):
    """
    Undersample so each class has the same count (min class size by default).
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


def class_balance_table(y, class_names):
    """
    Return per-class counts and percentages for label vector y (zeros included).
    """
    y = to_np_x(y).astype(int)
    counts = pd.Series(y).value_counts()
    df = pd.DataFrame({
        "class": class_names,
        "count": [int(counts.get(i, 0)) for i in range(len(class_names))],
    })
    df["pct"] = 100 * df["count"] / df["count"].sum()
    return df


def plot_class_balance_pie(y, class_names, title="Class balance", ax=None, min_pct=5.0):
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
        labels=None,          # no long names on wedges
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