"""Dataset loading, subsampling, and class-balance helpers."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.utils import to_np_x


def load_split(name, categories):
    df = pd.read_parquet(f"data/TON_IoT/quantum/q8_{name}.parquet")
    df = df[df["label_multiclass"].notna()].copy()

    feature_cols = [c for c in df.columns if not c.startswith("label")]
    X = df[feature_cols].values
    y = pd.Categorical(df["label_multiclass"], categories=categories).codes

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


def plot_class_balance_pie(y, class_names, title="Class balance", ax=None):
    """
    Plot a pie chart of class frequencies for label vector y (zeros dropped).
    """
    df = class_balance_table(y, class_names)
    df = df[df["count"] > 0]    # pie can't show 0-count slices cleanly

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.pie(
        df["count"],
        labels=df["class"],
        autopct=lambda p: f"{p:.1f}%\n({int(p * df['count'].sum() / 100)})",
        startangle=90,
    )
    ax.set_title(title)
    return ax
