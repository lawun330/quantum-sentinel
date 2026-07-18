"""NEW MODULE: generalized hyperparameter sweep (from friend's ad hoc run_lambda_sweep)."""

import numpy as np
import pandas as pd


def run_sweep(train_fn, param_grid, score_fn):
    """
    train_fn(**params) -> (theta, head, prototypes, ema_protos, history)
    score_fn(history, prototypes) -> float (higher is better)
    param_grid: list of dicts, each a kwargs set for train_fn.
    """
    records = []
    for params in param_grid:
        theta, head, prototypes, ema_protos, history = train_fn(**params)
        score = score_fn(history, prototypes)
        records.append({**params, "score": score, "final_loss": history["loss"][-1],
                         "final_intra_fid": history["intra_fid_gap"][-1],
                         "min_grad_var": min(history["grad_var"])})
        print(f"{params} -> score={score:.4f}")
    df = pd.DataFrame(records).sort_values("score", ascending=False)
    return df


def default_score_fn(history, prototypes):
    """Matches friend's scoring heuristic: separation + tightness - instability penalty."""
    stable = min(history["grad_var"]) > 1e-6
    return history["inter_trace_dist"][-1] + history["intra_fid_gap"][-1] - (0 if stable else 1)