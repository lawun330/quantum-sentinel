"""
MAQT training loop: batched circuit, EMA prototypes, curriculum lambda, optional focal
loss, weighted sampling -- with crash-safe checkpointing built for constrained Kaggle
GPU sessions (dual T4) where a single epoch on an 8-qubit `default.mixed` density-matrix
simulation can run long and get killed by OOM, a Kaggle time limit, or Ctrl+C.

Whatever happens, `<checkpoint_dir>/{maqt,plain}-latest.pt` holds the most recent state
(theta, head, optimizer, EMA prototypes, epoch + in-epoch step position) written
atomically every `checkpoint_every_steps` batches -- not just at epoch boundaries. Re-run
the same cell with `resume_from=<that path>` and it continues from the exact batch it
was on, not from epoch 0.

Recovery behaviors:
- CUDA OOM on a batch  -> free cache, halve the micro-batch, retry (recursively).
- SIGINT/SIGTERM/Ctrl+C -> finish the current batch, checkpoint, return cleanly
                           (history["stop_reason"] tells you why it stopped).
- `time_budget_sec` exceeded -> same graceful stop, before Kaggle kills the kernel.
- any other exception  -> emergency checkpoint + crash log written, then re-raised so
                           the traceback still shows in the notebook.
"""

import gc
import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

from scripts.circuit import initialize_random_weights
from scripts.constants import (
    BARREN_PLATEAU_VAR_THRESHOLD,
    DEFAULT_CONTROL_BATCH_SIZE,
    DEFAULT_EMA_MOMENTUM,
    DEFAULT_EPOCHS,
    DEFAULT_FOCAL,
    DEFAULT_FOCAL_GAMMA,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_LAMBDA1,
    DEFAULT_LAMBDA2,
    DEFAULT_LR,
    DEFAULT_WARMUP_FRAC,
    DEFAULT_WEIGHT_INIT_EPS,
)
try:
    from scripts.constants import DEFAULT_PATIENCE
except ImportError:  # constants.py doesn't define it yet -- see one-line addition above
    DEFAULT_PATIENCE = 3

from scripts.data import class_weights_for_sampler
from scripts.loss import curriculum_weight, maqt_loss
from scripts.prototypes import EMAPrototypeBank, PrototypeBank
from scripts.quantum_metrics import trace_distance
from scripts.utils import expectations_to_tensor, to_np_batch_x, to_np_y, to_torch_batch_x, to_torch_y
from scripts.logging import append_jsonl, write_crash_log, gpu_memory_snapshot


# ============================================================================
# Graceful-shutdown machinery
# ============================================================================

_STOP_REQUESTED = {"flag": False}


class _GracefulStop(Exception):
    """Internal control-flow signal only: state is already checkpointed by the time
    this is raised, so the outer handler just needs to stop quietly, not crash."""


@contextmanager
def _graceful_signals():
    """
    Install SIGINT/SIGTERM handlers that set a flag instead of killing the process
    immediately, so the training loop can finish its current batch and checkpoint
    before stopping. Always restores the previous handlers on exit -- so a Ctrl+C in
    some *other* cell later in the notebook still behaves normally.
    """
    _STOP_REQUESTED["flag"] = False

    def _handler(signum, frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        print(f"\n[signal] received {name} -- finishing current step, then checkpointing and stopping")
        _STOP_REQUESTED["flag"] = True

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # not the main thread / not supported here -- skip silently

    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def _is_oom_error(err):
    msg = str(err).lower()
    return isinstance(err, RuntimeError) and (
        "out of memory" in msg or "cuda error" in msg or "cublas" in msg
    )


def _free_memory(device):
    gc.collect()
    if device is not None and torch.cuda.is_available() and "cuda" in str(device):
        torch.cuda.empty_cache()


def _peak_gpu_mb():
    snap = gpu_memory_snapshot()
    return max((v["max_allocated_mb"] for v in snap.values()), default=None)


def _run_with_oom_retry(xb, yb, step_fn, device, max_retries):
    """
    Call `step_fn(xb, yb) -> dict` (does zero_grad/backward/opt.step internally, must
    return at least {"n": len(xb), ...numeric scalars..., ...tensors...}).

    On CUDA OOM: free the cache, halve the micro-batch, retry each half as its own
    optimizer step (recursively, up to `max_retries` splits). Numeric fields in the
    returned dict are merged as a size-weighted average; non-numeric fields (e.g. the
    batch's y/rho used for the EMA update) fall back to the later half's value.
    """
    try:
        return step_fn(xb, yb)
    except RuntimeError as e:
        if _is_oom_error(e) and max_retries > 0 and len(xb) > 1:
            _free_memory(device)
            mid = len(xb) // 2
            r1 = _run_with_oom_retry(xb[:mid], yb[:mid], step_fn, device, max_retries - 1)
            r2 = _run_with_oom_retry(xb[mid:], yb[mid:], step_fn, device, max_retries - 1)
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


# ============================================================================
# Checkpoint I/O (atomic writes so a kill mid-save can't corrupt the file)
# ============================================================================

def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)  # atomic on POSIX
    return path


def _history_to_cpu(history):
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in history.items()}


def _build_payload(*, epoch, epochs_total, step_in_epoch, perm, theta, head, opt, history,
                    seed, early_stopping, early_stop_state, ema_protos=None, best_bundle=None,
                    epoch_terms=None, lam1=None, lam2=None, extra=None):
    payload = {
        "epoch": int(epoch),                  # 0-based epoch index to resume at
        "epochs_total": int(epochs_total),
        "step_in_epoch": int(step_in_epoch),   # batches already done in that epoch (0 = fresh epoch)
        "perm": (perm.detach().cpu() if torch.is_tensor(perm) else perm),
        "theta": theta.detach().cpu(),
        "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "optimizer_state_dict": opt.state_dict(),
        "history": _history_to_cpu(history),
        "epoch_terms": epoch_terms,            # partial-epoch running stats, if mid-epoch
        "lam1": lam1, "lam2": lam2,
        "seed": seed,
        "early_stopping": bool(early_stopping),
        "early_stop_state": early_stop_state,
    }
    if ema_protos is not None:
        payload["ema_protos"] = {int(k): v.detach().cpu() for k, v in ema_protos.protos.items()}
        payload["ema_momentum"] = float(ema_protos.momentum)
    if best_bundle is not None:
        payload["best_bundle"] = best_bundle
    if extra:
        payload["extra"] = extra
    return payload


def _prune_old_checkpoints(ckpt_dir, prefix, keep_last_n):
    if not keep_last_n or keep_last_n <= 0:
        return
    files = sorted(Path(ckpt_dir).glob(f"{prefix}-epoch*.pt"))
    for f in files[:-keep_last_n]:
        try:
            f.unlink()
        except OSError:
            pass


def _save_all(ckpt_dir, prefix, *, write_epoch_file, keep_last_n, **kwargs):
    """
    Always overwrites `<prefix>-latest.pt` (used for every resume). Additionally writes
    a numbered `<prefix>-epochNNN.pt` snapshot when `write_epoch_file=True` (only at
    completed-epoch boundaries, so mid-epoch checkpoints don't spam the disk), pruning
    older numbered snapshots beyond `keep_last_n`. Never raises -- a failed checkpoint
    write is logged as a warning but does not abort training.
    """
    payload = _build_payload(**kwargs)
    ckpt_dir = Path(ckpt_dir)
    saved = None
    try:
        saved = _atomic_torch_save(payload, ckpt_dir / f"{prefix}-latest.pt")
    except Exception as e:
        print(f"warning: failed to write checkpoint (latest): {e!r}")
    if write_epoch_file:
        try:
            saved = _atomic_torch_save(payload, ckpt_dir / f"{prefix}-epoch{kwargs['epoch']:03d}.pt")
            _prune_old_checkpoints(ckpt_dir, prefix, keep_last_n)
        except Exception as e:
            print(f"warning: failed to write checkpoint (epoch file): {e!r}")
    return saved


def _load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def _restore_early_stop(early_stop, state):
    if not state:
        return
    early_stop.best_score = state.get("best_score")
    early_stop.best_epoch = int(state.get("best_epoch", 0))
    early_stop.bad_epochs = int(state.get("bad_epochs", 0))
    early_stop.should_stop = bool(state.get("should_stop", False))


def _early_stop_state_dict(early_stop):
    return {
        "best_score": early_stop.best_score,
        "best_epoch": early_stop.best_epoch,
        "bad_epochs": early_stop.bad_epochs,
        "should_stop": early_stop.should_stop,
        "patience": early_stop.patience,
        "min_delta": early_stop.min_delta,
        "mode": early_stop.mode,
    }


# ============================================================================
# Early stopping
# ============================================================================

class EarlyStopping:
    """
    Track best score (loss or accuracy/F1) and signal stop after `patience` non-improving
    epochs. mode="min" for loss, mode="max" for accuracy/F1.
    """

    def __init__(self, patience=3, min_delta=0.0, mode="min"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best_score = None
        self.bad_epochs = 0
        self.should_stop = False
        self.best_epoch = 0

    def step(self, score, epoch):
        score = float(score)
        if self.best_score is None:
            improved = True
        elif self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True
        return improved, self.should_stop


# ============================================================================
# MAQT training
# ============================================================================

def train_maqt(
    X_train, y_train, n_classes, n_qubits, n_layers, forward_circuit, device,
    epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    lambda1_max=DEFAULT_LAMBDA1, lambda2_max=DEFAULT_LAMBDA2, warmup_frac=DEFAULT_WARMUP_FRAC,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM, ema_momentum=DEFAULT_EMA_MOMENTUM,
    use_focal=DEFAULT_FOCAL, focal_gamma=DEFAULT_FOCAL_GAMMA, weight_init_eps=DEFAULT_WEIGHT_INIT_EPS,
    use_weighted_sampler=True, log_every=1, verbose=True, seed=None,
    early_stopping=False, patience=DEFAULT_PATIENCE, min_delta=0.0,
    checkpoint_dir=None, save_every_epoch=True, resume_from=None, checkpoint_extra=None,
    checkpoint_every_steps=50, keep_last_n_checkpoints=3, time_budget_sec=None,
    oom_max_retries=3, heartbeat_every_steps=25, log_dir=None, notebook_name="maqt",
):
    """
    Train the MAQT model. See module docstring for the crash/OOM/interrupt recovery
    behavior. Existing call sites (positional/keyword args from earlier versions) keep
    working -- all new knobs have safe defaults.
    """
    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps, seed=seed)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    ema_protos = EMAPrototypeBank(classes=range(n_classes), momentum=ema_momentum)

    # Dataset stays on CPU as numpy; only the active mini-batch is moved to `device`
    # each step (via maqt_loss -> to_torch_batch_x/to_torch_y). Frees GPU headroom for
    # the actual bottleneck: forward/backward through the 8-qubit density matrices.
    X_np = to_np_batch_x(X_train)
    y_np = to_np_y(y_train).astype(int)
    n = len(X_np)

    sample_weights = (
        torch.as_tensor(class_weights_for_sampler(y_np, n_classes), dtype=torch.double)
        if use_weighted_sampler else None
    )

    history = {
        "loss": [], "L_CE": [], "L_intra": [], "L_inter": [],
        "grad_var": [], "intra_fid_gap": [], "inter_trace_dist": [],
        "epoch_sec": [], "gpu_mem_mb": [],
    }

    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
    best_bundle = None
    start_epoch, start_step = 0, 0
    resumed_perm, resumed_epoch_terms = None, None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_dir = Path(log_dir) if log_dir is not None else (ckpt_dir if ckpt_dir is not None else Path("logs"))

    if resume_from is not None and Path(resume_from).exists():
        try:
            ckpt = _load_checkpoint(resume_from, device)
            with torch.no_grad():
                theta.copy_(ckpt["theta"].to(device))
            head.load_state_dict(ckpt["head_state_dict"]); head.to(device)
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            history = _history_to_cpu(ckpt["history"])
            start_epoch = int(ckpt.get("epoch", 0))
            start_step = int(ckpt.get("step_in_epoch", 0))
            if ckpt.get("perm") is not None and start_step > 0:
                resumed_perm = ckpt["perm"]
                resumed_epoch_terms = ckpt.get("epoch_terms")
            if "ema_protos" in ckpt:
                ema_protos.protos = {int(k): v.to(device) for k, v in ckpt["ema_protos"].items()}
            _restore_early_stop(early_stop, ckpt.get("early_stop_state"))
            if ckpt.get("best_bundle") is not None:
                best_bundle = ckpt["best_bundle"]
            if verbose:
                where = f"epoch {start_epoch}" + (f", step {start_step} (mid-epoch)" if start_step else "")
                print(f"resumed MAQT from {resume_from} at {where} / {epochs}")
        except Exception as e:
            print(f"warning: failed to load checkpoint {resume_from} ({e!r}); starting fresh")
            start_epoch, start_step = 0, 0

    train_t0 = time.perf_counter()
    stopped_time_budget = False
    epoch, step_count, perm, epoch_terms, lam1, lam2 = start_epoch, 0, None, None, None, None

    with _graceful_signals():
        try:
            for epoch in range(start_epoch, epochs):
                epoch_t0 = time.perf_counter()
                lam1 = curriculum_weight(epoch, epochs, lambda1_max, warmup_frac)
                lam2 = curriculum_weight(epoch, epochs, lambda2_max, warmup_frac)

                if seed is not None:
                    torch.manual_seed(int(seed) + epoch)

                if resumed_perm is not None and len(resumed_perm) == n:
                    perm = resumed_perm if torch.is_tensor(resumed_perm) else torch.as_tensor(resumed_perm)
                elif use_weighted_sampler:
                    perm = torch.tensor(list(WeightedRandomSampler(sample_weights, num_samples=n, replacement=True)))
                else:
                    perm = torch.randperm(n)
                resumed_perm = None

                epoch_terms = resumed_epoch_terms or {
                    "L_total": [], "L_CE": [], "L_intra": [], "L_inter": [],
                    "grad_vars": [], "intra_fid_running": [],
                }
                resumed_epoch_terms = None

                step_start_this_epoch = start_step if epoch == start_epoch else 0
                start_step = 0  # only honor the resumed offset once
                step_count = step_start_this_epoch

                for i in range(step_start_this_epoch * batch_size, n, batch_size):
                    idx = perm[i : i + batch_size].numpy()
                    xb_np, yb_np = X_np[idx], y_np[idx]
                    protos_snapshot = ema_protos.snapshot()

                    def _step_fn(xb_s, yb_s, _protos=protos_snapshot, _lam1=lam1, _lam2=lam2):
                        opt.zero_grad()
                        loss, l_ce, l_intra, l_inter, y_used, rho_used = maqt_loss(
                            theta, head, ce_loss_fn, xb_s, yb_s, _protos, forward_circuit,
                            lambda1=_lam1, lambda2=_lam2, device=device,
                            use_focal=use_focal, focal_gamma=focal_gamma,
                        )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), grad_clip_norm)
                        grad_var = theta.grad.var().item()
                        opt.step()
                        return {"n": len(xb_s), "loss": loss.item(), "l_ce": l_ce.item(),
                                "l_intra": l_intra.item(), "l_inter": l_inter.item(),
                                "grad_var": grad_var, "y_used": y_used, "rho_used": rho_used}

                    result = _run_with_oom_retry(xb_np, yb_np, _step_fn, device, oom_max_retries)

                    ema_protos.update(ema_protos.batch_class_means(result["rho_used"].detach(), result["y_used"]))

                    epoch_terms["L_total"].append(result["loss"])
                    epoch_terms["L_CE"].append(result["l_ce"])
                    epoch_terms["L_intra"].append(result["l_intra"])
                    epoch_terms["L_inter"].append(result["l_inter"])
                    epoch_terms["grad_vars"].append(result["grad_var"])
                    epoch_terms["intra_fid_running"].append(1 - result["l_intra"])
                    step_count += 1

                    if verbose and heartbeat_every_steps and step_count % heartbeat_every_steps == 0:
                        elapsed = time.perf_counter() - epoch_t0
                        frac = min(1.0, (i + batch_size) / n)
                        eta = elapsed / max(frac, 1e-6) - elapsed
                        print(f"  epoch {epoch+1} step {step_count} ({frac*100:.0f}%) "
                              f"loss={epoch_terms['L_total'][-1]:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s")

                    time_budget_hit = (
                        time_budget_sec is not None and (time.perf_counter() - train_t0) > time_budget_sec
                    )
                    if ckpt_dir is not None and (
                        (checkpoint_every_steps and step_count % checkpoint_every_steps == 0)
                        or _STOP_REQUESTED["flag"] or time_budget_hit
                    ):
                        _save_all(
                            ckpt_dir, "maqt", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                            epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                            theta=theta, head=head, opt=opt, history=history, seed=seed,
                            early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                            ema_protos=ema_protos, best_bundle=best_bundle, epoch_terms=epoch_terms,
                            lam1=lam1, lam2=lam2, extra=checkpoint_extra,
                        )
                        if verbose and (_STOP_REQUESTED["flag"] or time_budget_hit):
                            print(f"  saved mid-epoch checkpoint (epoch {epoch+1}, step {step_count})")

                    if step_count % 200 == 0:
                        _free_memory(device)

                    if _STOP_REQUESTED["flag"] or time_budget_hit:
                        stopped_time_budget = time_budget_hit
                        raise _GracefulStop()

                # ---- epoch finished: aggregate + log + checkpoint ----
                diag_pairs = []
                keys = sorted(ema_protos.protos.keys())
                for a in range(len(keys)):
                    for b in range(a + 1, len(keys)):
                        diag_pairs.append(float(trace_distance(ema_protos.protos[keys[a]], ema_protos.protos[keys[b]])))
                mean_inter_td = float(np.mean(diag_pairs)) if diag_pairs else 0.0
                mean_gv = float(np.mean(epoch_terms["grad_vars"])) if epoch_terms["grad_vars"] else float("nan")

                history["loss"].append(float(np.mean(epoch_terms["L_total"])))
                history["L_CE"].append(float(np.mean(epoch_terms["L_CE"])))
                history["L_intra"].append(float(np.mean(epoch_terms["L_intra"])))
                history["L_inter"].append(float(np.mean(epoch_terms["L_inter"])))
                history["grad_var"].append(mean_gv)
                history["intra_fid_gap"].append(float(np.mean(epoch_terms["intra_fid_running"])))
                history["inter_trace_dist"].append(mean_inter_td)
                history["epoch_sec"].append(time.perf_counter() - epoch_t0)
                history["gpu_mem_mb"].append(_peak_gpu_mb())

                epoch_1based = epoch + 1
                improved, stop = False, False
                if early_stopping:
                    improved, stop = early_stop.step(history["loss"][-1], epoch=epoch_1based)
                    if improved:
                        best_bundle = {
                            "theta": theta.detach().cpu().clone(),
                            "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            "ema_protos": {int(k): v.detach().cpu().clone() for k, v in ema_protos.protos.items()},
                        }

                if verbose and (epoch_1based % log_every == 0):
                    es_msg = (
                        f" | best={early_stop.best_score:.4f}@ep{early_stop.best_epoch} | bad={early_stop.bad_epochs}/{patience}"
                        if early_stopping else ""
                    )
                    mem_msg = f" | peak_mem={history['gpu_mem_mb'][-1]:.0f}MB" if history["gpu_mem_mb"][-1] else ""
                    print(
                        f"epoch {epoch_1based:2d}/{epochs} | loss {history['loss'][-1]:.4f} | "
                        f"L_CE {history['L_CE'][-1]:.4f} | L_intra {history['L_intra'][-1]:.4f} | "
                        f"L_inter {history['L_inter'][-1]:.4f} | grad_var {mean_gv:.2e} | "
                        f"intra_fid {history['intra_fid_gap'][-1]:.3f} | inter_TD {mean_inter_td:.3f} | "
                        f"time {history['epoch_sec'][-1]:.1f}s{mem_msg}{es_msg}"
                    )
                    if mean_gv < BARREN_PLATEAU_VAR_THRESHOLD:
                        print("  barren plateau detected")

                if ckpt_dir is not None and save_every_epoch:
                    saved = _save_all(
                        ckpt_dir, "maqt", write_epoch_file=True, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch_1based, epochs_total=epochs, step_in_epoch=0, perm=None,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=ema_protos, best_bundle=best_bundle, epoch_terms=None,
                        lam1=lam1, lam2=lam2, extra=checkpoint_extra,
                    )
                    if verbose and saved:
                        print(f"  saved checkpoint: {saved}")

                if log_dir is not None:
                    try:
                        append_jsonl(
                            {"epoch": epoch_1based, "loss": history["loss"][-1], "L_CE": history["L_CE"][-1],
                             "L_intra": history["L_intra"][-1], "L_inter": history["L_inter"][-1],
                             "grad_var": mean_gv, "epoch_sec": history["epoch_sec"][-1],
                             "gpu_mem_mb": history["gpu_mem_mb"][-1]},
                            notebook_name, log_dir=log_dir,
                        )
                    except Exception:
                        pass

                if early_stopping and stop:
                    if verbose:
                        print(f"early stopping at epoch {epoch_1based}: no train-loss improvement for "
                              f"{patience} epochs (best={early_stop.best_score:.4f} @ epoch {early_stop.best_epoch})")
                    break

        except _GracefulStop:
            pass
        except BaseException as e:
            crash_ckpt = None
            if ckpt_dir is not None:
                try:
                    crash_ckpt = _save_all(
                        ckpt_dir, "maqt", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=ema_protos, best_bundle=best_bundle, epoch_terms=epoch_terms,
                        lam1=lam1, lam2=lam2, extra=checkpoint_extra,
                    )
                except Exception:
                    pass
            crash_log = None
            if log_dir is not None:
                try:
                    crash_log = write_crash_log(e, history=history, extra=checkpoint_extra,
                                                 name=notebook_name, log_dir=log_dir)
                except Exception:
                    pass
            if verbose:
                kind = "KeyboardInterrupt" if isinstance(e, KeyboardInterrupt) else type(e).__name__
                print(f"\n[train_maqt] stopped by {kind}: {e}")
                if crash_ckpt: print(f"  emergency checkpoint saved: {crash_ckpt}")
                if crash_log: print(f"  crash log written: {crash_log}")
                if ckpt_dir: print(f"  re-run with resume_from={ckpt_dir / 'maqt-latest.pt'} to continue")
            raise

        interrupted = stopped_time_budget or _STOP_REQUESTED["flag"]

        if early_stopping and best_bundle is not None and not interrupted:
            with torch.no_grad():
                theta.copy_(best_bundle["theta"].to(device))
            head.load_state_dict(best_bundle["head_state_dict"]); head.to(device)
            ema_protos.protos = {int(k): v.to(device) for k, v in best_bundle["ema_protos"].items()}
            if verbose:
                print(f"restored best MAQT weights from epoch {early_stop.best_epoch} "
                      f"(train loss={early_stop.best_score:.4f})")

        history["best_epoch"] = int(early_stop.best_epoch) if early_stopping else int(len(history["loss"]))
        history["best_score"] = (
            float(early_stop.best_score) if (early_stopping and early_stop.best_score is not None)
            else (float(history["loss"][-1]) if history["loss"] else float("nan"))
        )
        history["stopped_early"] = bool(early_stopping and early_stop.should_stop)
        history["epochs_ran"] = int(len(history["loss"]))
        history["interrupted"] = bool(interrupted)
        history["stop_reason"] = (
            "time_budget" if stopped_time_budget else
            "signal" if _STOP_REQUESTED["flag"] else
            "early_stopping" if history["stopped_early"] else
            "completed"
        )

        if interrupted:
            if verbose:
                print(f"training paused ({history['stop_reason']}) after {history['epochs_ran']} epoch(s) "
                      f"-- re-run with resume_from=<latest checkpoint> to continue")
            # skip the full-dataset exact-prototype recompute (could itself be slow/OOM
            # right when we're trying to exit quickly) -- return the live EMA prototypes.
            final_prototypes = {k: v.detach().clone() for k, v in ema_protos.protos.items()}
            return theta, head, final_prototypes, ema_protos, history

        try:
            final_bank = PrototypeBank(classes=range(n_classes))
            final_prototypes = final_bank.compute(theta, X_np, y_np, forward_circuit=forward_circuit, device=device)
        except Exception as e:
            if verbose:
                print(f"warning: failed to compute final exact prototypes ({e!r}); falling back to EMA prototypes")
            final_prototypes = {k: v.detach().clone() for k, v in ema_protos.protos.items()}

        return theta, head, final_prototypes, ema_protos, history


# ============================================================================
# Plain-CE baseline (robustness ablation) -- same crash-safety machinery
# ============================================================================

def train_plain_vqc(
    X_train, y_train, n_classes, n_qubits, n_layers, forward_circuit, device,
    epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    weight_init_eps=DEFAULT_WEIGHT_INIT_EPS, grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    seed=None, early_stopping=False, patience=DEFAULT_PATIENCE, min_delta=0.0,
    checkpoint_dir=None, save_every_epoch=True, resume_from=None, checkpoint_extra=None,
    checkpoint_every_steps=50, keep_last_n_checkpoints=3, time_budget_sec=None,
    oom_max_retries=3, heartbeat_every_steps=25, log_dir=None, notebook_name="plain_vqc",
    verbose=True,
):
    """CE-only baseline (no prototype terms) for robustness ablation."""
    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps, seed=seed)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce = nn.CrossEntropyLoss()

    X_np = to_np_batch_x(X_train)
    y_np = to_np_y(y_train).astype(int)
    n = len(X_np)

    history = {"loss": [], "grad_var": [], "epoch_sec": [], "gpu_mem_mb": []}
    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
    best_bundle = None
    start_epoch, start_step = 0, 0
    resumed_perm, resumed_epoch_terms = None, None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_dir = Path(log_dir) if log_dir is not None else (ckpt_dir if ckpt_dir is not None else Path("logs"))

    if resume_from is not None and Path(resume_from).exists():
        try:
            ckpt = _load_checkpoint(resume_from, device)
            with torch.no_grad():
                theta.copy_(ckpt["theta"].to(device))
            head.load_state_dict(ckpt["head_state_dict"]); head.to(device)
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            history = _history_to_cpu(ckpt["history"])
            start_epoch = int(ckpt.get("epoch", 0))
            start_step = int(ckpt.get("step_in_epoch", 0))
            if ckpt.get("perm") is not None and start_step > 0:
                resumed_perm = ckpt["perm"]
                resumed_epoch_terms = ckpt.get("epoch_terms")
            _restore_early_stop(early_stop, ckpt.get("early_stop_state"))
            if ckpt.get("best_bundle") is not None:
                best_bundle = ckpt["best_bundle"]
            if verbose:
                where = f"epoch {start_epoch}" + (f", step {start_step} (mid-epoch)" if start_step else "")
                print(f"resumed plain VQC from {resume_from} at {where} / {epochs}")
        except Exception as e:
            print(f"warning: failed to load checkpoint {resume_from} ({e!r}); starting fresh")
            start_epoch, start_step = 0, 0

    train_t0 = time.perf_counter()
    stopped_time_budget = False
    epoch, step_count, perm, epoch_terms = start_epoch, 0, None, None

    with _graceful_signals():
        try:
            for epoch in range(start_epoch, epochs):
                epoch_t0 = time.perf_counter()
                if seed is not None:
                    torch.manual_seed(int(seed) + epoch)

                if resumed_perm is not None and len(resumed_perm) == n:
                    perm = resumed_perm if torch.is_tensor(resumed_perm) else torch.as_tensor(resumed_perm)
                else:
                    perm = torch.randperm(n)
                resumed_perm = None

                epoch_terms = resumed_epoch_terms or {"loss": [], "grad_vars": []}
                resumed_epoch_terms = None

                step_start_this_epoch = start_step if epoch == start_epoch else 0
                start_step = 0
                step_count = step_start_this_epoch

                for i in range(step_start_this_epoch * batch_size, n, batch_size):
                    idx = perm[i : i + batch_size].numpy()
                    xb_np, yb_np = X_np[idx], y_np[idx]

                    def _step_fn(xb_s, yb_s):
                        x_t = to_torch_batch_x(xb_s, device=device)
                        y_t = to_torch_y(yb_s, device=device)
                        opt.zero_grad()
                        z, _ = forward_circuit(x_t, theta)
                        logits = head(expectations_to_tensor(z))
                        loss = ce(logits, y_t)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), grad_clip_norm)
                        grad_var = theta.grad.var().item()
                        opt.step()
                        return {"n": len(xb_s), "loss": loss.item(), "grad_var": grad_var}

                    result = _run_with_oom_retry(xb_np, yb_np, _step_fn, device, oom_max_retries)
                    epoch_terms["loss"].append(result["loss"])
                    epoch_terms["grad_vars"].append(result["grad_var"])
                    step_count += 1

                    if verbose and heartbeat_every_steps and step_count % heartbeat_every_steps == 0:
                        elapsed = time.perf_counter() - epoch_t0
                        frac = min(1.0, (i + batch_size) / n)
                        eta = elapsed / max(frac, 1e-6) - elapsed
                        print(f"  [plain] epoch {epoch+1} step {step_count} ({frac*100:.0f}%) "
                              f"loss={epoch_terms['loss'][-1]:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s")

                    time_budget_hit = (
                        time_budget_sec is not None and (time.perf_counter() - train_t0) > time_budget_sec
                    )
                    if ckpt_dir is not None and (
                        (checkpoint_every_steps and step_count % checkpoint_every_steps == 0)
                        or _STOP_REQUESTED["flag"] or time_budget_hit
                    ):
                        _save_all(
                            ckpt_dir, "plain", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                            epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                            theta=theta, head=head, opt=opt, history=history, seed=seed,
                            early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                            ema_protos=None, best_bundle=best_bundle, epoch_terms=epoch_terms,
                            lam1=None, lam2=None, extra=checkpoint_extra,
                        )
                        if verbose and (_STOP_REQUESTED["flag"] or time_budget_hit):
                            print(f"  saved mid-epoch checkpoint (epoch {epoch+1}, step {step_count})")

                    if step_count % 200 == 0:
                        _free_memory(device)

                    if _STOP_REQUESTED["flag"] or time_budget_hit:
                        stopped_time_budget = time_budget_hit
                        raise _GracefulStop()

                mean_gv = float(np.mean(epoch_terms["grad_vars"])) if epoch_terms["grad_vars"] else float("nan")
                history["loss"].append(float(np.mean(epoch_terms["loss"])))
                history["grad_var"].append(mean_gv)
                history["epoch_sec"].append(time.perf_counter() - epoch_t0)
                history["gpu_mem_mb"].append(_peak_gpu_mb())

                epoch_1based = epoch + 1
                improved, stop = False, False
                if early_stopping:
                    improved, stop = early_stop.step(history["loss"][-1], epoch=epoch_1based)
                    if improved:
                        best_bundle = {
                            "theta": theta.detach().cpu().clone(),
                            "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                        }

                if verbose:
                    es_msg = (
                        f" | best={early_stop.best_score:.4f}@ep{early_stop.best_epoch} | bad={early_stop.bad_epochs}/{patience}"
                        if early_stopping else ""
                    )
                    print(f"[plain VQC] epoch {epoch_1based}/{epochs} loss {history['loss'][-1]:.4f} "
                          f"| grad_var {history['grad_var'][-1]:.2e}{es_msg}")

                if ckpt_dir is not None and save_every_epoch:
                    saved = _save_all(
                        ckpt_dir, "plain", write_epoch_file=True, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch_1based, epochs_total=epochs, step_in_epoch=0, perm=None,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=None, best_bundle=best_bundle, epoch_terms=None,
                        lam1=None, lam2=None, extra=checkpoint_extra,
                    )
                    if verbose and saved:
                        print(f"  saved checkpoint: {saved}")

                if log_dir is not None:
                    try:
                        append_jsonl(
                            {"epoch": epoch_1based, "loss": history["loss"][-1], "grad_var": mean_gv,
                             "epoch_sec": history["epoch_sec"][-1], "gpu_mem_mb": history["gpu_mem_mb"][-1]},
                            notebook_name, log_dir=log_dir,
                        )
                    except Exception:
                        pass

                if early_stopping and stop:
                    if verbose:
                        print(f"early stopping at epoch {epoch_1based}: no improvement for "
                              f"{patience} epochs (best={early_stop.best_score:.4f} @ epoch {early_stop.best_epoch})")
                    break

        except _GracefulStop:
            pass
        except BaseException as e:
            crash_ckpt = None
            if ckpt_dir is not None:
                try:
                    crash_ckpt = _save_all(
                        ckpt_dir, "plain", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=None, best_bundle=best_bundle, epoch_terms=epoch_terms,
                        lam1=None, lam2=None, extra=checkpoint_extra,
                    )
                except Exception:
                    pass
            crash_log = None
            if log_dir is not None:
                try:
                    crash_log = write_crash_log(e, history=history, extra=checkpoint_extra,
                                                 name=notebook_name, log_dir=log_dir)
                except Exception:
                    pass
            if verbose:
                kind = "KeyboardInterrupt" if isinstance(e, KeyboardInterrupt) else type(e).__name__
                print(f"\n[train_plain_vqc] stopped by {kind}: {e}")
                if crash_ckpt: print(f"  emergency checkpoint saved: {crash_ckpt}")
                if crash_log: print(f"  crash log written: {crash_log}")
            raise

        interrupted = stopped_time_budget or _STOP_REQUESTED["flag"]
        if early_stopping and best_bundle is not None and not interrupted:
            with torch.no_grad():
                theta.copy_(best_bundle["theta"].to(device))
            head.load_state_dict(best_bundle["head_state_dict"]); head.to(device)
            if verbose:
                print(f"restored best plain-VQC weights from epoch {early_stop.best_epoch} "
                      f"(train loss={early_stop.best_score:.4f})")

        history["best_epoch"] = int(early_stop.best_epoch) if early_stopping else int(len(history["loss"]))
        history["best_score"] = (
            float(early_stop.best_score) if (early_stopping and early_stop.best_score is not None)
            else (float(history["loss"][-1]) if history["loss"] else float("nan"))
        )
        history["stopped_early"] = bool(early_stopping and early_stop.should_stop)
        history["epochs_ran"] = int(len(history["loss"]))
        history["interrupted"] = bool(interrupted)
        history["stop_reason"] = (
            "time_budget" if stopped_time_budget else
            "signal" if _STOP_REQUESTED["flag"] else
            "early_stopping" if history["stopped_early"] else
            "completed"
        )
        if interrupted and verbose:
            print(f"training paused ({history['stop_reason']}) after {history['epochs_ran']} epoch(s) "
                  f"-- re-run with resume_from=<latest checkpoint> to continue")

        return theta, head, history