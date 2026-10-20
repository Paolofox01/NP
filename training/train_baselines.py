"""Loss, evaluation and training loop of the baselines (models/baselines.py).

Same procedure as train_np.train for the Neural Processes: fixed validation set every
validate_every epochs, plateau scheduler, early stopping, best weights restored at the end.
"""

import copy
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["baseline_loss", "evaluate_baseline", "train_baseline"]

Prediction = Tuple[Tensor, Optional[Tensor]]


# ----------------------------------------------------------------------------- loss and evaluation

def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Mean over the real targets: per task, then over the batch (as np_loss does)."""
    if mask is None:
        return values.mean()
    keep = (~mask).float()
    return ((values * keep).sum(-1) / keep.sum(-1).clamp(min=1)).mean()


def baseline_loss(prediction: Prediction, y: Tensor, target_mask: Optional[Tensor] = None,
                  mse_weight: float = 0.0) -> Dict[str, Tensor]:
    """Training loss per value, ignoring padded targets. y, target_mask: (B, T).
    Deterministic: MSE. Probabilistic: (1 - w) NLL + w MSE."""
    mean, var = prediction
    mse = _masked_mean((mean - y) ** 2, target_mask)
    if var is None:
        return {"loss": mse, "mse": mse}
    nll = _masked_mean(0.5 * (math.log(2 * math.pi) + var.log() + (y - mean) ** 2 / var), target_mask)
    return {"loss": (1 - mse_weight) * nll + mse_weight * mse, "mse": mse, "nll": nll}


@torch.no_grad()
def evaluate_baseline(model: nn.Module, batches, device: torch.device) -> Dict[str, float]:
    """MSE (and NLL, for a probabilistic model) per value, averaged over the batches."""
    model.eval()
    totals: Dict[str, float] = {}
    for batch in batches:
        batch = {k: v.to(device) for k, v in batch.items()}
        terms = baseline_loss(model(batch), batch["y_target"].squeeze(-1), batch.get("target_mask"))
        for name in ("mse", "nll"):
            if name in terms:
                totals[name] = totals.get(name, 0.0) + terms[name].item()
    return {name: total / len(batches) for name, total in totals.items()}


# ----------------------------------------------------------------------------- training

def _save(path: Path, model: nn.Module, epoch: int, history) -> None:
    torch.save({"class": type(model).__name__, "kwargs": model.kwargs, "epoch": epoch,
                "model_state_dict": model.state_dict(), "history": history}, path)


def train_baseline(
    model: nn.Module,
    train_batches,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    val_batches=None,
    validate_every: int = 1,
    scheduler=None,
    mse_warmup_epochs: int = 0,
    grad_clip: Optional[float] = 1.0,
    early_stopping_patience: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    verbose: bool = True,
    print_every: int = 1,
) -> Dict[str, List[float]]:
    """Same procedure as training.train for the Neural Processes.

    Validation metric (best model, early stopping, ReduceLROnPlateau): the NLL for a probabilistic
    model, the MSE otherwise. mse_warmup_epochs (probabilistic only): the MSE weight in the loss
    falls linearly from 1 to 0 over these epochs, and the best model / early stopping / plateau
    scheduler start after them. The model ends with the weights of the best validation.
    """
    probabilistic = getattr(model, "probabilistic", False)
    warmup = mse_warmup_epochs if probabilistic else 0
    monitor = "nll" if probabilistic else "mse"
    history: Dict[str, List[float]] = {k: [] for k in ("train_loss", "train_mse", "lr", "val_epoch", "val_mse")}
    if probabilistic:
        history.update(train_nll=[], val_nll=[], mse_weight=[])
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    best_value, best_state, best_epoch = float("inf"), None, None
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        mse_weight = max(0.0, 1.0 - epoch / warmup) if warmup else 0.0
        model.train()
        sums: Dict[str, float] = {}
        for batch in train_batches:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            terms = baseline_loss(model(batch), batch["y_target"].squeeze(-1), batch.get("target_mask"),
                                  mse_weight)
            terms["loss"].backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value.item()
        for name, total in sums.items():
            history[f"train_{name}"].append(total / len(train_batches))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        if probabilistic:
            history["mse_weight"].append(mse_weight)

        stop = False
        validated = val_batches is not None and ((epoch + 1) % validate_every == 0 or epoch == epochs - 1)
        if validated:
            metrics = evaluate_baseline(model, val_batches, device)
            history["val_epoch"].append(epoch + 1)
            for name, value in metrics.items():
                history[f"val_{name}"].append(value)
            if epoch >= warmup:
                if metrics[monitor] < best_value:
                    best_value, best_epoch = metrics[monitor], epoch
                    best_state = copy.deepcopy(model.state_dict())
                    if checkpoint_dir:
                        _save(checkpoint_dir / "best_model.pt", model, epoch, history)
                since = epoch - (best_epoch if best_epoch is not None else warmup - 1)
                stop = early_stopping_patience is not None and since >= early_stopping_patience

        if scheduler is not None:
            if not plateau:
                scheduler.step()
            elif validated and epoch >= warmup:
                scheduler.step(history[f"val_{monitor}"][-1])

        if verbose and (epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1 or stop):
            message = f"Epoch {epoch + 1}/{epochs}  loss {history['train_loss'][-1]:.4f}"
            if probabilistic:
                message += f" (nll {history['train_nll'][-1]:.4f}, mse {history['train_mse'][-1]:.4f}, w={mse_weight:.2f})"
            if validated:
                message += f"  |  val mse {history['val_mse'][-1]:.4f}"
                if probabilistic:
                    message += f", nll {history['val_nll'][-1]:.4f}"
            message += f"  |  lr {history['lr'][-1]:.2e}"
            print(message)
        if stop:
            if verbose:
                print(f"Early stopping: no validation improvement for {early_stopping_patience} epochs")
            break

    if checkpoint_dir:
        _save(checkpoint_dir / "final_model.pt", model, epoch, history)
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored the best model (epoch {best_epoch + 1}, val {monitor} {best_value:.4f})")
    return history
