"""Train MAQT and plain-VQC quantum models with checkpoints, safe stops, and memory-friendly batching."""

import signal
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    DEFAULT_PATIENCE,
    DEFAULT_WARMUP_FRAC,
    DEFAULT_WEIGHT_INIT_EPS,
)
from scripts.data import class_weights_for_sampler
from scripts.logging import append_jsonl, write_crash_log
from scripts.loss import curriculum_weight, maqt_loss
from scripts.memory import free_memory, peak_gpu_mb, run_with_oom_retry
from scripts.prototypes import EMAPrototypeBank, PrototypeBank
from scripts.quantum_metrics import trace_distance
from scripts.utils import expectations_to_tensor, to_np_batch_x, to_np_y, to_torch_batch_x, to_torch_y

# ============================================================================
# Graceful-shutdown machinery
# ============================================================================

_STOP_REQUESTED = {"flag": False}


class _GracefulStop(Exception):
    """
    Stop training after the current batch is saved.
    """


@contextmanager
def _graceful_signals():
    """
    Catch Ctrl+C and similar signals so training can save and stop cleanly.
    Example:
    - SIGINT (Ctrl+C)
    - SIGTERM (kill)
    """
    _STOP_REQUESTED["flag"] = False

    def _handler(signum, frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)
        print(f"\n[signal] {name} -- finishing current step, then checkpointing")
        _STOP_REQUESTED["flag"] = True

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
    try:
        yield
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

# ============================================================================
# Early stopping
# ============================================================================

class EarlyStopping:
    """
    Stop training when the score stops getting better for several `patience` epochs.

    Use mode="min" when lower is better (loss), or mode="max" when higher is better.
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
        """
        Update the best score and decide whether training should stop.
        """
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
# Checkpoint I/O (atomic writes so a kill mid-save can't corrupt the file)
# ============================================================================

def _history_to_cpu(history):
    """
    Copy history lists so they are safe to save to disk.
    """
    return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in history.items()}


def _atomic_torch_save(payload, path):
    """
    Save a file in two steps so a crash mid-write does not leave a broken checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def _build_payload(
    *,
    epoch,
    epochs_total,
    step_in_epoch,
    perm,
    theta,
    head,
    opt,
    history,
    seed,
    early_stopping,
    early_stop_state,
    ema_protos=None,
    best_bundle=None,
    epoch_terms=None,
    lam1=None,
    lam2=None,
    extra=None,
    scheduler=None,
):
    """
    Build the dictionary that gets written into a checkpoint file.
    """
    payload = {
        "epoch": int(epoch),  # 1-based completed epoch
        "epochs_total": int(epochs_total),
        "step_in_epoch": int(step_in_epoch),
        "perm": (perm.detach().cpu() if torch.is_tensor(perm) else perm),
        "theta": theta.detach().cpu(),
        "head_state_dict": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "optimizer_state_dict": opt.state_dict(),
        "history": _history_to_cpu(history),
        "epoch_terms": epoch_terms,
        "lam1": lam1,
        "lam2": lam2,
        "seed": seed,
        "early_stopping": bool(early_stopping),
        "early_stop_state": early_stop_state,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if ema_protos is not None:
        payload["ema_protos"] = {int(k): v.detach().cpu() for k, v in ema_protos.protos.items()}
        payload["ema_momentum"] = float(ema_protos.momentum)
    if best_bundle is not None:
        payload["best_bundle"] = best_bundle
    if extra:
        payload["extra"] = extra
    return payload


def _make_cosine_scheduler(opt, epochs, lr, lr_min=None, last_epoch=-1):
    """
    Cosine decay from `lr` down to `lr_min` (default 1% of `lr`) over `epochs`.
    """
    eta_min = float(lr * 0.01 if lr_min is None else lr_min)
    return CosineAnnealingLR(opt, T_max=max(int(epochs), 1), eta_min=eta_min, last_epoch=last_epoch)


def _current_lr(opt):
    return float(opt.param_groups[0]["lr"])


def _prune_old_checkpoints(ckpt_dir, prefix, keep_last_n):
    """
    Delete older numbered epoch files so only the newest few remain.
    """
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
    Always save the latest checkpoint file: `<prefix>-latest.pt`

    Optionally also save a numbered epoch file: `<prefix>-epoch<epoch>.pt`
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
    """
    Load a checkpoint onto the given device.
    """
    return torch.load(path, map_location=device, weights_only=False)


def _restore_early_stop(early_stop, state):
    """
    Restore early-stopping counters from a saved checkpoint.
    """
    if not state:
        return
    early_stop.best_score = state.get("best_score")
    early_stop.best_epoch = int(state.get("best_epoch", 0))
    early_stop.bad_epochs = int(state.get("bad_epochs", 0))
    early_stop.should_stop = bool(state.get("should_stop", False))


def _reset_early_stop_patience(early_stop):
    """
    Clear the patience counter so resumed training gets a fresh run of bad epochs.
    Keeps the best score and best epoch so they can still be restored at the end.
    """
    early_stop.bad_epochs = 0
    early_stop.should_stop = False


def _early_stop_state_dict(early_stop):
    """
    Pack early-stopping fields so they can be saved and restored later.
    """
    return {
        "best_score": early_stop.best_score,
        "best_epoch": early_stop.best_epoch,
        "bad_epochs": early_stop.bad_epochs,
        "should_stop": early_stop.should_stop,
        "patience": early_stop.patience,
        "min_delta": early_stop.min_delta,
        "mode": early_stop.mode,
    }


def _finalize_history(history, early_stop, early_stopping, interrupted, stopped_time_budget):
    """
    Add final summary fields to the training history before returning.
    """
    history["best_epoch"] = int(early_stop.best_epoch) if early_stopping else int(len(history["loss"]))
    history["best_score"] = (
        float(early_stop.best_score)
        if (early_stopping and early_stop.best_score is not None)
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
    return history


def _ensure_history_key(history, key):
    """
    Ensure a list key exists after resume from older checkpoints.
    """
    if key not in history or history[key] is None:
        history[key] = []
    return history


@torch.no_grad()
def _mean_maqt_val_loss(
    theta,
    head,
    ce_loss_fn,
    X_val_np,
    y_val_np,
    prototypes,
    forward_circuit,
    lam1,
    lam2,
    device,
    batch_size,
    use_focal,
    focal_gamma,
    oom_max_retries,
):
    """
    Mean MAQT total loss on a held-out val set (no grad, no EMA update).
    """
    n_val = len(X_val_np)
    if n_val == 0:
        return float("nan")
    totals = []
    for i in range(0, n_val, batch_size):
        xb_np, yb_np = X_val_np[i : i + batch_size], y_val_np[i : i + batch_size]

        def _eval_fn(xb_s, yb_s, _protos=prototypes, _lam1=lam1, _lam2=lam2):
            loss, _, _, _, _, _ = maqt_loss(
                theta, head, ce_loss_fn, xb_s, yb_s, _protos, forward_circuit,
                lambda1=_lam1, lambda2=_lam2, device=device,
                use_focal=use_focal, focal_gamma=focal_gamma,
            )
            return {"loss": float(loss.item())}

        result = run_with_oom_retry(xb_np, yb_np, _eval_fn, device, oom_max_retries)
        totals.append(result["loss"])
    return float(np.mean(totals))


@torch.no_grad()
def _mean_ce_val_loss(theta, head, ce_loss_fn, X_val_np, y_val_np, forward_circuit, device, batch_size, oom_max_retries):
    """
    Mean CE loss on a held-out val set (no grad).
    """
    n_val = len(X_val_np)
    if n_val == 0:
        return float("nan")
    totals = []
    for i in range(0, n_val, batch_size):
        xb_np, yb_np = X_val_np[i : i + batch_size], y_val_np[i : i + batch_size]

        def _eval_fn(xb_s, yb_s):
            x_t = to_torch_batch_x(xb_s, device=device)
            y_t = to_torch_y(yb_s, device=device)
            z, _ = forward_circuit(x_t, theta)
            loss = ce_loss_fn(head(expectations_to_tensor(z)), y_t)
            return {"loss": float(loss.item())}

        result = run_with_oom_retry(xb_np, yb_np, _eval_fn, device, oom_max_retries)
        totals.append(result["loss"])
    return float(np.mean(totals))

# ============================================================================
# MAQT training
# ============================================================================

def train_maqt(
    X_train,
    y_train,
    n_classes,
    n_qubits,
    n_layers,
    forward_circuit,
    device,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    lambda1_max=DEFAULT_LAMBDA1,
    lambda2_max=DEFAULT_LAMBDA2,
    warmup_frac=DEFAULT_WARMUP_FRAC,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    ema_momentum=DEFAULT_EMA_MOMENTUM,
    use_focal=DEFAULT_FOCAL,
    focal_gamma=DEFAULT_FOCAL_GAMMA,
    weight_init_eps=DEFAULT_WEIGHT_INIT_EPS,
    use_weighted_sampler=True,
    log_every=1,
    verbose=True,
    seed=None,
    early_stopping=False,
    patience=DEFAULT_PATIENCE,
    min_delta=0.0,
    X_val=None,
    y_val=None,
    use_lr_schedule=True,
    lr_min=None,
    checkpoint_dir=None,
    save_every_epoch=True,
    resume_from=None,
    reset_early_stopping_on_resume=False,
    checkpoint_extra=None,
    checkpoint_every_steps=50,
    keep_last_n_checkpoints=3,
    time_budget_sec=None,
    oom_max_retries=3,
    heartbeat_every_steps=25,
    log_dir=None,
    notebook_name="maqt",
):
    """
    Train the MAQT model.

    Keep the full training set on CPU and only move each small batch to the GPU when needed.
    """
    if (X_val is None) ^ (y_val is None):
        raise ValueError("pass both X_val and y_val, or neither")

    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps, seed=seed)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()
    ema_protos = EMAPrototypeBank(classes=range(n_classes), momentum=ema_momentum)

    X_np = to_np_batch_x(X_train)
    y_np = to_np_y(y_train).astype(int)
    n = len(X_np)

    X_val_np = to_np_batch_x(X_val) if X_val is not None else None
    y_val_np = to_np_y(y_val).astype(int) if y_val is not None else None
    use_val = X_val_np is not None

    sample_weights = (
        torch.as_tensor(class_weights_for_sampler(y_np, n_classes), dtype=torch.double)
        if use_weighted_sampler else None
    )

    history = {
        "loss": [], "L_CE": [], "L_intra": [], "L_inter": [],
        "val_loss": [], "lr": [],
        "grad_var": [], "intra_fid_gap": [], "inter_trace_dist": [],
        "epoch_sec": [], "gpu_mem_mb": [],
    }

    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
    best_bundle = None
    start_epoch, start_step = 0, 0
    resumed_perm, resumed_epoch_terms = None, None
    resumed_scheduler_state = None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_dir = Path(log_dir) if log_dir is not None else (ckpt_dir if ckpt_dir is not None else Path("logs"))

    if resume_from is not None and Path(resume_from).exists():
        try:
            ckpt = _load_checkpoint(resume_from, device)
            with torch.no_grad():
                theta.copy_(ckpt["theta"].to(device))
            head.load_state_dict(ckpt["head_state_dict"])
            head.to(device)
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            history = _history_to_cpu(ckpt["history"])
            _ensure_history_key(history, "val_loss")
            _ensure_history_key(history, "lr")
            start_epoch = int(ckpt.get("epoch", 0))
            start_step = int(ckpt.get("step_in_epoch", 0))
            if ckpt.get("perm") is not None and start_step > 0:
                resumed_perm = ckpt["perm"]
                resumed_epoch_terms = ckpt.get("epoch_terms")
            if "ema_protos" in ckpt:
                ema_protos.protos = {int(k): v.to(device) for k, v in ckpt["ema_protos"].items()}
            _restore_early_stop(early_stop, ckpt.get("early_stop_state"))
            if reset_early_stopping_on_resume:
                _reset_early_stop_patience(early_stop)
            if ckpt.get("best_bundle") is not None:
                best_bundle = ckpt["best_bundle"]
            resumed_scheduler_state = ckpt.get("scheduler_state_dict")
            if verbose:
                where = f"epoch {start_epoch}" + (f", step {start_step} (mid-epoch)" if start_step else "")
                es_note = "\n(patience reset)" if reset_early_stopping_on_resume else ""
                print(f"resumed MAQT from {resume_from} at {where} / {epochs}{es_note}")
        except Exception as e:
            print(f"warning: failed to load checkpoint {resume_from} ({e!r}); starting fresh")
            start_epoch, start_step = 0, 0
            resumed_scheduler_state = None

    scheduler = None
    if use_lr_schedule:
        if resumed_scheduler_state is not None:
            scheduler = _make_cosine_scheduler(opt, epochs, lr, lr_min=lr_min, last_epoch=-1)
            scheduler.load_state_dict(resumed_scheduler_state)
        else:
            # old ckpts / fresh: last_epoch = completed epochs - 1
            scheduler = _make_cosine_scheduler(
                opt, epochs, lr, lr_min=lr_min, last_epoch=start_epoch - 1
            )

    train_t0 = time.perf_counter()
    stopped_time_budget = False
    epoch, step_count, perm, epoch_terms, lam1, lam2 = start_epoch, 0, None, None, None, None
    es_metric = "val_loss" if use_val else "train_loss"

    with _graceful_signals():
        try:
            for epoch in range(start_epoch, epochs):
                epoch_t0 = time.perf_counter()
                lam1 = curriculum_weight(epoch, epochs, lambda1_max, warmup_frac)
                lam2 = curriculum_weight(epoch, epochs, lambda2_max, warmup_frac)

                if seed is not None:
                    torch.manual_seed(int(seed) + epoch)

                if resumed_perm is not None and len(resumed_perm) == n:
                    # keep on CPU: map_location=device would put ckpt perm on CUDA, and .numpy() needs host memory
                    perm = (
                        resumed_perm.detach().cpu()
                        if torch.is_tensor(resumed_perm)
                        else torch.as_tensor(resumed_perm)
                    )
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
                start_step = 0
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
                        return {
                            "n": len(xb_s),
                            "loss": loss.item(),
                            "l_ce": l_ce.item(),
                            "l_intra": l_intra.item(),
                            "l_inter": l_inter.item(),
                            "grad_var": grad_var,
                            "y_used": y_used,
                            "rho_used": rho_used,
                        }

                    result = run_with_oom_retry(xb_np, yb_np, _step_fn, device, oom_max_retries)
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
                        print(
                            f"\tepoch {epoch+1} step {step_count} ({frac*100:.0f}%) "
                            f"loss={epoch_terms['L_total'][-1]:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s"
                        )

                    time_budget_hit = (
                        time_budget_sec is not None
                        and (time.perf_counter() - train_t0) > time_budget_sec
                    )
                    if ckpt_dir is not None and (
                        (checkpoint_every_steps and step_count % checkpoint_every_steps == 0)
                        or _STOP_REQUESTED["flag"]
                        or time_budget_hit
                    ):
                        _save_all(
                            ckpt_dir, "maqt", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                            epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                            theta=theta, head=head, opt=opt, history=history, seed=seed,
                            early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                            ema_protos=ema_protos, best_bundle=best_bundle, epoch_terms=epoch_terms,
                            lam1=lam1, lam2=lam2, extra=checkpoint_extra, scheduler=scheduler,
                        )
                        if verbose and (_STOP_REQUESTED["flag"] or time_budget_hit):
                            print(f"  saved mid-epoch checkpoint (epoch {epoch+1}, step {step_count})")

                    if step_count % 200 == 0:
                        free_memory(device)

                    if _STOP_REQUESTED["flag"] or time_budget_hit:
                        stopped_time_budget = time_budget_hit
                        raise _GracefulStop()

                # epoch finished
                diag_pairs = []
                keys = sorted(ema_protos.protos.keys())
                for a in range(len(keys)):
                    for b in range(a + 1, len(keys)):
                        diag_pairs.append(
                            float(trace_distance(ema_protos.protos[keys[a]], ema_protos.protos[keys[b]]))
                        )
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
                history["gpu_mem_mb"].append(peak_gpu_mb())

                if use_val:
                    val_loss = _mean_maqt_val_loss(
                        theta, head, ce_loss_fn, X_val_np, y_val_np,
                        ema_protos.snapshot(), forward_circuit, lam1, lam2, device,
                        batch_size, use_focal, focal_gamma, oom_max_retries,
                    )
                else:
                    val_loss = float("nan")
                history["val_loss"].append(val_loss)

                epoch_lr = _current_lr(opt)
                history["lr"].append(epoch_lr)

                epoch_1based = epoch + 1
                improved, stop = False, False
                if early_stopping:
                    score = val_loss if use_val else history["loss"][-1]
                    improved, stop = early_stop.step(score, epoch=epoch_1based)
                    if improved:
                        best_bundle = {
                            "theta": theta.detach().cpu().clone(),
                            "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            "ema_protos": {int(k): v.detach().cpu().clone() for k, v in ema_protos.protos.items()},
                        }

                if verbose and (epoch_1based % log_every == 0):
                    es_msg = (
                        f" | best={early_stop.best_score:.4f}@ep{early_stop.best_epoch} "
                        f"| bad={early_stop.bad_epochs}/{patience}"
                        if early_stopping else ""
                    )
                    mem = history["gpu_mem_mb"][-1]
                    mem_msg = f" | peak_mem={mem:.0f}MB" if mem else ""
                    val_msg = f" | val_loss {val_loss:.4f}" if use_val else ""
                    lr_msg = f" | lr {epoch_lr:.2e}" if use_lr_schedule else ""
                    print(
                        f"epoch {epoch_1based:2d}/{epochs} | time {history['epoch_sec'][-1]:.1f}s | "
                        f"loss {history['loss'][-1]:.4f} | "
                        f"L_CE {history['L_CE'][-1]:.4f} | L_intra {history['L_intra'][-1]:.4f} | "
                        f"L_inter {history['L_inter'][-1]:.4f} | grad_var {mean_gv:.2e} | "
                        f"intra_fid {history['intra_fid_gap'][-1]:.3f} | inter_TD {mean_inter_td:.3f}"
                        f"{val_msg}{lr_msg}{mem_msg}{es_msg}"
                    )
                    if mean_gv < BARREN_PLATEAU_VAR_THRESHOLD:
                        print("  barren plateau detected")

                if scheduler is not None:
                    scheduler.step()

                if ckpt_dir is not None and save_every_epoch:
                    saved = _save_all(
                        ckpt_dir, "maqt", write_epoch_file=True, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch_1based, epochs_total=epochs, step_in_epoch=0, perm=None,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=ema_protos, best_bundle=best_bundle, epoch_terms=None,
                        lam1=lam1, lam2=lam2, extra=checkpoint_extra, scheduler=scheduler,
                    )
                    if verbose and saved:
                        print(f"  saved checkpoint: {saved}")

                if log_dir is not None:
                    try:
                        row = {
                            "epoch": epoch_1based,
                            "loss": history["loss"][-1],
                            "L_CE": history["L_CE"][-1],
                            "L_intra": history["L_intra"][-1],
                            "L_inter": history["L_inter"][-1],
                            "grad_var": mean_gv,
                            "lr": epoch_lr,
                            "epoch_sec": history["epoch_sec"][-1],
                            "gpu_mem_mb": history["gpu_mem_mb"][-1],
                        }
                        if use_val:
                            row["val_loss"] = val_loss
                        append_jsonl(row, notebook_name, log_dir=log_dir)
                    except Exception:
                        pass

                if early_stopping and stop:
                    if verbose:
                        print(
                            f"early stopping at epoch {epoch_1based}: "
                            f"no {es_metric} improvement for {patience} epochs "
                            f"(best={early_stop.best_score:.4f} @ epoch {early_stop.best_epoch})"
                        )
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
                        lam1=lam1, lam2=lam2, extra=checkpoint_extra, scheduler=scheduler,
                    )
                except Exception:
                    pass
            crash_log = None
            if log_dir is not None:
                try:
                    crash_log = write_crash_log(
                        e, history=history, extra=checkpoint_extra, name=notebook_name, log_dir=log_dir
                    )
                except Exception:
                    pass
            if verbose:
                kind = "KeyboardInterrupt" if isinstance(e, KeyboardInterrupt) else type(e).__name__
                print(f"\n[train_maqt] stopped by {kind}: {e}")
                if crash_ckpt:
                    print(f"  emergency checkpoint saved: {crash_ckpt}")
                if crash_log:
                    print(f"  crash log written: {crash_log}")
                if ckpt_dir:
                    print(f"  re-run with resume_from={ckpt_dir / 'maqt-latest.pt'} to continue")
            raise

        interrupted = stopped_time_budget or _STOP_REQUESTED["flag"]

        if early_stopping and best_bundle is not None and not interrupted:
            with torch.no_grad():
                theta.copy_(best_bundle["theta"].to(device))
            head.load_state_dict(best_bundle["head_state_dict"])
            head.to(device)
            ema_protos.protos = {int(k): v.to(device) for k, v in best_bundle["ema_protos"].items()}
            if verbose:
                print(
                    f"restored best MAQT weights from epoch {early_stop.best_epoch} "
                    f"({es_metric}={early_stop.best_score:.4f})"
                )

        _finalize_history(history, early_stop, early_stopping, interrupted, stopped_time_budget)

        if interrupted:
            if verbose:
                print(
                    f"training paused ({history['stop_reason']}) after {history['epochs_ran']} epoch(s) "
                    f"-- re-run with resume_from=<latest checkpoint> to continue"
                )
            final_prototypes = {k: v.detach().clone() for k, v in ema_protos.protos.items()}
            return theta, head, final_prototypes, ema_protos, history

        try:
            final_bank = PrototypeBank(classes=range(n_classes))
            final_prototypes = final_bank.compute(
                theta, X_np, y_np, forward_circuit=forward_circuit, device=device
            )
        except Exception as e:
            if verbose:
                print(f"warning: failed to compute final exact prototypes ({e!r}); using EMA")
            final_prototypes = {k: v.detach().clone() for k, v in ema_protos.protos.items()}

        return theta, head, final_prototypes, ema_protos, history

# ============================================================================
# Plain-CE baseline (same loop as MAQT; CE loss only)
# ============================================================================

def train_plain_vqc(
    X_train,
    y_train,
    n_classes,
    n_qubits,
    n_layers,
    forward_circuit,
    device,
    epochs=DEFAULT_EPOCHS,
    lr=DEFAULT_LR,
    batch_size=DEFAULT_CONTROL_BATCH_SIZE,
    grad_clip_norm=DEFAULT_GRAD_CLIP_NORM,
    weight_init_eps=DEFAULT_WEIGHT_INIT_EPS,
    use_weighted_sampler=True,
    log_every=1,
    verbose=True,
    seed=None,
    early_stopping=False,
    patience=DEFAULT_PATIENCE,
    min_delta=0.0,
    X_val=None,
    y_val=None,
    use_lr_schedule=True,
    lr_min=None,
    checkpoint_dir=None,
    save_every_epoch=True,
    resume_from=None,
    reset_early_stopping_on_resume=False,
    checkpoint_extra=None,
    checkpoint_every_steps=50,
    keep_last_n_checkpoints=3,
    time_budget_sec=None,
    oom_max_retries=3,
    heartbeat_every_steps=25,
    log_dir=None,
    notebook_name="plain_vqc",
):
    """
    Train the CE-only baseline (zero lambdas) with only class labels for robustness ablation.
    """
    if (X_val is None) ^ (y_val is None):
        raise ValueError("pass both X_val and y_val, or neither")

    theta = initialize_random_weights(n_layers, n_qubits, device, eps=weight_init_eps, seed=seed)
    head = nn.Linear(n_qubits, n_classes).to(device)
    opt = torch.optim.Adam(list([theta]) + list(head.parameters()), lr=lr)
    ce_loss_fn = nn.CrossEntropyLoss()

    X_np = to_np_batch_x(X_train)
    y_np = to_np_y(y_train).astype(int)
    n = len(X_np)

    X_val_np = to_np_batch_x(X_val) if X_val is not None else None
    y_val_np = to_np_y(y_val).astype(int) if y_val is not None else None
    use_val = X_val_np is not None

    sample_weights = (
        torch.as_tensor(class_weights_for_sampler(y_np, n_classes), dtype=torch.double)
        if use_weighted_sampler else None
    )

    history = {
        "loss": [], "L_CE": [],
        "val_loss": [], "lr": [],
        "grad_var": [],
        "epoch_sec": [], "gpu_mem_mb": [],
    }

    early_stop = EarlyStopping(patience=patience, min_delta=min_delta, mode="min")
    best_bundle = None
    start_epoch, start_step = 0, 0
    resumed_perm, resumed_epoch_terms = None, None
    resumed_scheduler_state = None
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    log_dir = Path(log_dir) if log_dir is not None else (ckpt_dir if ckpt_dir is not None else Path("logs"))

    if resume_from is not None and Path(resume_from).exists():
        try:
            ckpt = _load_checkpoint(resume_from, device)
            with torch.no_grad():
                theta.copy_(ckpt["theta"].to(device))
            head.load_state_dict(ckpt["head_state_dict"])
            head.to(device)
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            history = _history_to_cpu(ckpt["history"])
            _ensure_history_key(history, "val_loss")
            _ensure_history_key(history, "lr")
            start_epoch = int(ckpt.get("epoch", 0))
            start_step = int(ckpt.get("step_in_epoch", 0))
            if ckpt.get("perm") is not None and start_step > 0:
                resumed_perm = ckpt["perm"]
                resumed_epoch_terms = ckpt.get("epoch_terms")
            _restore_early_stop(early_stop, ckpt.get("early_stop_state"))
            if reset_early_stopping_on_resume:
                _reset_early_stop_patience(early_stop)
            if ckpt.get("best_bundle") is not None:
                best_bundle = ckpt["best_bundle"]
            resumed_scheduler_state = ckpt.get("scheduler_state_dict")
            if verbose:
                where = f"epoch {start_epoch}" + (f", step {start_step} (mid-epoch)" if start_step else "")
                es_note = "\n(patience reset)" if reset_early_stopping_on_resume else ""
                print(f"resumed plain VQC from {resume_from} at {where} / {epochs}{es_note}")
        except Exception as e:
            print(f"warning: failed to load checkpoint {resume_from} ({e!r}); starting fresh")
            start_epoch, start_step = 0, 0
            resumed_scheduler_state = None

    scheduler = None
    if use_lr_schedule:
        if resumed_scheduler_state is not None:
            scheduler = _make_cosine_scheduler(opt, epochs, lr, lr_min=lr_min, last_epoch=-1)
            scheduler.load_state_dict(resumed_scheduler_state)
        else:
            scheduler = _make_cosine_scheduler(
                opt, epochs, lr, lr_min=lr_min, last_epoch=start_epoch - 1
            )

    train_t0 = time.perf_counter()
    stopped_time_budget = False
    epoch, step_count, perm, epoch_terms = start_epoch, 0, None, None
    es_metric = "val_loss" if use_val else "train_loss"

    with _graceful_signals():
        try:
            for epoch in range(start_epoch, epochs):
                epoch_t0 = time.perf_counter()

                if seed is not None:
                    torch.manual_seed(int(seed) + epoch)

                if resumed_perm is not None and len(resumed_perm) == n:
                    # keep on CPU: map_location=device would put ckpt perm on CUDA, and .numpy() needs host memory
                    perm = (
                        resumed_perm.detach().cpu()
                        if torch.is_tensor(resumed_perm)
                        else torch.as_tensor(resumed_perm)
                    )
                elif use_weighted_sampler:
                    perm = torch.tensor(list(WeightedRandomSampler(sample_weights, num_samples=n, replacement=True)))
                else:
                    perm = torch.randperm(n)
                resumed_perm = None

                epoch_terms = resumed_epoch_terms or {
                    "L_total": [], "L_CE": [], "grad_vars": [],
                }
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
                        loss = ce_loss_fn(logits, y_t)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(list([theta]) + list(head.parameters()), grad_clip_norm)
                        grad_var = theta.grad.var().item()
                        opt.step()
                        return {
                            "n": len(xb_s),
                            "loss": loss.item(),
                            "l_ce": loss.item(),
                            "grad_var": grad_var,
                        }

                    result = run_with_oom_retry(xb_np, yb_np, _step_fn, device, oom_max_retries)

                    epoch_terms["L_total"].append(result["loss"])
                    epoch_terms["L_CE"].append(result["l_ce"])
                    epoch_terms["grad_vars"].append(result["grad_var"])
                    step_count += 1

                    if verbose and heartbeat_every_steps and step_count % heartbeat_every_steps == 0:
                        elapsed = time.perf_counter() - epoch_t0
                        frac = min(1.0, (i + batch_size) / n)
                        eta = elapsed / max(frac, 1e-6) - elapsed
                        print(
                            f"\tepoch {epoch+1} step {step_count} ({frac*100:.0f}%) "
                            f"loss={epoch_terms['L_total'][-1]:.4f} elapsed={elapsed:.0f}s eta={eta:.0f}s"
                        )

                    time_budget_hit = (
                        time_budget_sec is not None
                        and (time.perf_counter() - train_t0) > time_budget_sec
                    )
                    if ckpt_dir is not None and (
                        (checkpoint_every_steps and step_count % checkpoint_every_steps == 0)
                        or _STOP_REQUESTED["flag"]
                        or time_budget_hit
                    ):
                        _save_all(
                            ckpt_dir, "plain", write_epoch_file=False, keep_last_n=keep_last_n_checkpoints,
                            epoch=epoch, epochs_total=epochs, step_in_epoch=step_count, perm=perm,
                            theta=theta, head=head, opt=opt, history=history, seed=seed,
                            early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                            ema_protos=None, best_bundle=best_bundle, epoch_terms=epoch_terms,
                            lam1=None, lam2=None, extra=checkpoint_extra, scheduler=scheduler,
                        )
                        if verbose and (_STOP_REQUESTED["flag"] or time_budget_hit):
                            print(f"  saved mid-epoch checkpoint (epoch {epoch+1}, step {step_count})")

                    if step_count % 200 == 0:
                        free_memory(device)

                    if _STOP_REQUESTED["flag"] or time_budget_hit:
                        stopped_time_budget = time_budget_hit
                        raise _GracefulStop()

                mean_gv = float(np.mean(epoch_terms["grad_vars"])) if epoch_terms["grad_vars"] else float("nan")
                history["loss"].append(float(np.mean(epoch_terms["L_total"])))
                history["L_CE"].append(float(np.mean(epoch_terms["L_CE"])))
                history["grad_var"].append(mean_gv)
                history["epoch_sec"].append(time.perf_counter() - epoch_t0)
                history["gpu_mem_mb"].append(peak_gpu_mb())

                if use_val:
                    val_loss = _mean_ce_val_loss(
                        theta, head, ce_loss_fn, X_val_np, y_val_np, forward_circuit,
                        device, batch_size, oom_max_retries,
                    )
                else:
                    val_loss = float("nan")
                history["val_loss"].append(val_loss)

                epoch_lr = _current_lr(opt)
                history["lr"].append(epoch_lr)

                epoch_1based = epoch + 1
                improved, stop = False, False
                if early_stopping:
                    score = val_loss if use_val else history["loss"][-1]
                    improved, stop = early_stop.step(score, epoch=epoch_1based)
                    if improved:
                        best_bundle = {
                            "theta": theta.detach().cpu().clone(),
                            "head_state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                        }

                if verbose and (epoch_1based % log_every == 0):
                    es_msg = (
                        f" | best={early_stop.best_score:.4f}@ep{early_stop.best_epoch} "
                        f"| bad={early_stop.bad_epochs}/{patience}"
                        if early_stopping else ""
                    )
                    mem = history["gpu_mem_mb"][-1]
                    mem_msg = f" | peak_mem={mem:.0f}MB" if mem else ""
                    val_msg = f" | val_loss {val_loss:.4f}" if use_val else ""
                    lr_msg = f" | lr {epoch_lr:.2e}" if use_lr_schedule else ""
                    print(
                        f"epoch {epoch_1based:2d}/{epochs} | time {history['epoch_sec'][-1]:.1f}s | "
                        f"loss {history['loss'][-1]:.4f} | "
                        f"L_CE {history['L_CE'][-1]:.4f} | grad_var {mean_gv:.2e}"
                        f"{val_msg}{lr_msg}{mem_msg}{es_msg}"
                    )
                    if mean_gv < BARREN_PLATEAU_VAR_THRESHOLD:
                        print("  barren plateau detected")

                if scheduler is not None:
                    scheduler.step()

                if ckpt_dir is not None and save_every_epoch:
                    saved = _save_all(
                        ckpt_dir, "plain", write_epoch_file=True, keep_last_n=keep_last_n_checkpoints,
                        epoch=epoch_1based, epochs_total=epochs, step_in_epoch=0, perm=None,
                        theta=theta, head=head, opt=opt, history=history, seed=seed,
                        early_stopping=early_stopping, early_stop_state=_early_stop_state_dict(early_stop),
                        ema_protos=None, best_bundle=best_bundle, epoch_terms=None,
                        lam1=None, lam2=None, extra=checkpoint_extra, scheduler=scheduler,
                    )
                    if verbose and saved:
                        print(f"  saved checkpoint: {saved}")

                if log_dir is not None:
                    try:
                        row = {
                            "epoch": epoch_1based,
                            "loss": history["loss"][-1],
                            "L_CE": history["L_CE"][-1],
                            "grad_var": mean_gv,
                            "lr": epoch_lr,
                            "epoch_sec": history["epoch_sec"][-1],
                            "gpu_mem_mb": history["gpu_mem_mb"][-1],
                        }
                        if use_val:
                            row["val_loss"] = val_loss
                        append_jsonl(row, notebook_name, log_dir=log_dir)
                    except Exception:
                        pass

                if early_stopping and stop:
                    if verbose:
                        print(
                            f"early stopping at epoch {epoch_1based}: "
                            f"no {es_metric} improvement for {patience} epochs "
                            f"(best={early_stop.best_score:.4f} @ epoch {early_stop.best_epoch})"
                        )
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
                        lam1=None, lam2=None, extra=checkpoint_extra, scheduler=scheduler,
                    )
                except Exception:
                    pass
            crash_log = None
            if log_dir is not None:
                try:
                    crash_log = write_crash_log(
                        e, history=history, extra=checkpoint_extra, name=notebook_name, log_dir=log_dir
                    )
                except Exception:
                    pass
            if verbose:
                kind = "KeyboardInterrupt" if isinstance(e, KeyboardInterrupt) else type(e).__name__
                print(f"\n[train_plain_vqc] stopped by {kind}: {e}")
                if crash_ckpt:
                    print(f"  emergency checkpoint saved: {crash_ckpt}")
                if crash_log:
                    print(f"  crash log written: {crash_log}")
                if ckpt_dir:
                    print(f"  re-run with resume_from={ckpt_dir / 'plain-latest.pt'} to continue")
            raise

        interrupted = stopped_time_budget or _STOP_REQUESTED["flag"]

        if early_stopping and best_bundle is not None and not interrupted:
            with torch.no_grad():
                theta.copy_(best_bundle["theta"].to(device))
            head.load_state_dict(best_bundle["head_state_dict"])
            head.to(device)
            if verbose:
                print(
                    f"restored best plain-VQC weights from epoch {early_stop.best_epoch} "
                    f"({es_metric}={early_stop.best_score:.4f})"
                )

        _finalize_history(history, early_stop, early_stopping, interrupted, stopped_time_budget)

        if interrupted and verbose:
            print(
                f"training paused ({history['stop_reason']}) after {history['epochs_ran']} epoch(s) "
                f"-- re-run with resume_from=<latest checkpoint> to continue"
            )

        return theta, head, history
