"""Helpers that free GPU memory and recover from out-of-memory errors during training."""

import gc
import torch


def is_oom_error(e):
    """
    Return True when the error looks like the GPU ran out of memory.
    """
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(e, RuntimeError):
        return False
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error" in msg or "cublas" in msg


def is_fatal_cuda_error(e):
    """
    Return True when the CUDA context is likely corrupted and retrying is unsafe.
    """
    msg = str(e).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "misaligned address" in msg
    )


def free_memory(device=None):
    """
    Free unused GPU memory (CUDA cache) when the device is CUDA (or when any CUDA GPU is available).
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    if device is not None and "cuda" not in str(device):
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if is_fatal_cuda_error(e):
            print("\t[FATAL] CUDA context corrupted -- restart the kernel before further GPU cells.")
        else:
            raise


def safe_empty_cache():
    """
    Free GPU memory (CUDA cache) on all devices in a careful way.
    """
    free_memory(device="cuda" if torch.cuda.is_available() else None)
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError as e:
            if is_fatal_cuda_error(e):
                print("\t[FATAL] CUDA context corrupted -- restart the kernel before further GPU cells.")
            else:
                raise


def gpu_memory_snapshot(device=None):
    """
    Report how much GPU memory is in use and the peak so far.

    Returns an empty dict on CPU-only runs.
    """
    if not torch.cuda.is_available():
        return {}
    devices = [device] if device is not None else list(range(torch.cuda.device_count()))
    out = {}
    for d in devices:
        idx = d if isinstance(d, int) else (d.index if getattr(d, "index", None) is not None else 0)
        out[f"cuda:{idx}"] = {
            "allocated_mb": torch.cuda.memory_allocated(idx) / 1e6,
            "reserved_mb": torch.cuda.memory_reserved(idx) / 1e6,
            "max_allocated_mb": torch.cuda.max_memory_allocated(idx) / 1e6,
        }
    return out


def peak_gpu_mb(device=None):
    """
    Return the highest GPU memory usage across devices in megabytes, or None on CPU-only runs.
    """
    snap = gpu_memory_snapshot(device)
    if not snap:
        return None
    return max(v["max_allocated_mb"] for v in snap.values())


def print_gpu_memory(device=None, tag=""):
    """
    Print a short summary of current and peak GPU memory use.
    """
    snap = gpu_memory_snapshot(device)
    prefix = f"[mem{f' {tag}' if tag else ''}]"
    if not snap:
        print(f"{prefix} CPU-only")
        return
    for dev, stats in snap.items():
        print(
            f"{prefix} {dev}: alloc={stats['allocated_mb']:.0f}MB "
            f"reserved={stats['reserved_mb']:.0f}MB peak={stats['max_allocated_mb']:.0f}MB"
        )


def run_with_oom_retry(xb, yb, step_fn, device, max_retries):
    """
    Run one training step on a batch of size `len(xb)`.
    If the GPU runs out of memory, split the batch in half and retry.
    Numeric fields are size-weighted averages; other fields take the later half.
    """
    try:
        return step_fn(xb, yb)
    except RuntimeError as e:
        if is_oom_error(e) and max_retries > 0 and len(xb) > 1:
            free_memory(device)
            mid = len(xb) // 2
            r1 = run_with_oom_retry(xb[:mid], yb[:mid], step_fn, device, max_retries - 1)
            r2 = run_with_oom_retry(xb[mid:], yb[mid:], step_fn, device, max_retries - 1)
            w1, w2 = r1["n"], r2["n"]
            merged = {"n": w1 + w2}
            for k in r1:
                if k == "n":
                    continue
                v1, v2 = r1.get(k), r2.get(k)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    merged[k] = (v1 * w1 + v2 * w2) / (w1 + w2)
                else:
                    merged[k] = v2
            return merged
        raise


def run_batched_safely(fn, *args, batch_size=8, min_batch=1, on_fail=None, label="", **kwargs):
    """
    Call a function with a batch size, and keep halving that size if the GPU runs out of memory.
    """
    bs = batch_size
    while True:
        try:
            return fn(*args, batch_size=bs, **kwargs)
        except RuntimeError as e:
            if is_fatal_cuda_error(e):
                print(f"\t[FATAL{' ' + label if label else ''}] non-recoverable CUDA error, not retrying: {e}")
                safe_empty_cache()
                return on_fail() if on_fail is not None else None
            if is_oom_error(e) and bs > min_batch:
                print(
                    f"\t[OOM{' ' + label if label else ''}] batch_size={bs} failed "
                    f"-> retrying at {max(bs // 2, min_batch)}"
                )
                safe_empty_cache()
                bs = max(bs // 2, min_batch)
                continue
            if is_oom_error(e):
                print(f"\t[OOM{' ' + label if label else ''}] failed even at batch_size={min_batch}. Skipping.")
                safe_empty_cache()
                return on_fail() if on_fail is not None else None
            raise