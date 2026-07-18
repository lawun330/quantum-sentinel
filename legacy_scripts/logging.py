"""Training history logging helpers."""

import json
from pathlib import Path


def to_jsonable(obj):
    """
    Convert nested history / metrics into JSON-serializable values.
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


def write_history_log(history, notebook, extra=None, log_dir="."):
    """
    Write training `history` to `<notebook_stem>.log` as JSON.
    """
    stem = Path(notebook).stem
    log_path = Path(log_dir) / f"{stem}.log"

    payload = {
        "notebook": Path(notebook).name,
        "history": to_jsonable(history),
    }
    if extra:
        payload["extra"] = to_jsonable(extra)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return log_path
