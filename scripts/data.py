"""Dataset loading and subsampling."""

import pandas as pd
from sklearn.model_selection import train_test_split


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
