"""Training history logging helpers."""

import json
import time
import traceback
from pathlib import Path

from scripts.constants import DEFAULT_LOG_DIR, DEFAULT_SWEEP_LOG_DIR


def to_jsonable(obj):
    """
    Convert nested `history` metrics into JSON-serializable values.
    """
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def write_history_log(history, name, extra=None, log_dir=DEFAULT_LOG_DIR):
    """
    Write training `history` to `<notebook_name>.log` as JSON.
    """
    log_path = Path(log_dir) / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"history": to_jsonable(history)}
    if extra:
        payload["extra"] = to_jsonable(extra)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def write_sweep_log(sweep_df, name, log_dir=DEFAULT_SWEEP_LOG_DIR):
    """
    Write a hyperparameter sweep DataFrame (from `sweep.py`) to `<sweep_name>.json` as JSON.
    """
    log_path = Path(log_dir) / f"{name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = to_jsonable(sweep_df.to_dict(orient="records"))

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path


def append_jsonl(record, name, log_dir=DEFAULT_LOG_DIR):
    """
    Append one JSON line to `<log_dir>/<name>.jsonl`.
    """
    log_path = Path(log_dir) / f"{name}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")
    return log_path


def read_jsonl(name, log_dir=DEFAULT_LOG_DIR):
    """
    Read records written by `append_jsonl`, in order.
    """
    log_path = Path(log_dir) / f"{name}.jsonl"
    if not log_path.exists():
        return []
    with open(log_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_crash_log(
    exc, history=None, extra=None, name="maqt", log_dir=DEFAULT_LOG_DIR
):
    """
    Dump exception + optional training state to `<name>-crash-<timestamp>.json`.
    """
    log_path = Path(log_dir) / f"{name}-crash-{int(time.time())}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "history": to_jsonable(history) if history is not None else None,
        "extra": to_jsonable(extra) if extra else None,
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return log_path
