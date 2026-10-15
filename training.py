"""Training loop, evaluation and checkpoints for the Neural Process models.

Batches are dictionaries of tensors:
    x_context (B, N, x_dim), y_context (B, N, y_dim),
    x_target  (B, T, x_dim), y_target  (B, T, y_dim),
and optionally context_mask (B, N), target_mask (B, T) (True = padding).
"""

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from base import NeuralProcess, NPConfig, NPOutput
from loss import gaussian_nll, masked_mse, np_loss

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None

__all__ = ["train", "evaluate", "linear_kl_schedule", "save_checkpoint", "load_model"]

Batch = Dict[str, Tensor]


# ----------------------------------------------------------------------------- helpers

def linear_kl_schedule(warmup_epochs: int) -> List[float]:
    """KL weight rising linearly from 1/warmup_epochs to 1 (KL annealing)."""
    return [(epoch + 1) / warmup_epochs for epoch in range(warmup_epochs)]


def _to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _forward(model: NeuralProcess, batch: Batch, num_samples: int, with_target: bool) -> NPOutput:
    return model(
        batch["x_context"], batch["y_context"], batch["x_target"],
        batch["y_target"] if with_target else None,
        num_samples=num_samples,
        context_mask=batch.get("context_mask"),
        target_mask=batch.get("target_mask"),
    )


def _autocast(device: torch.device, use_amp: bool):
    """bfloat16 mixed precision on CUDA: no gradient scaler needed, unlike float16."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=use_amp and device.type == "cuda")


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model: NeuralProcess, loader: DataLoader, device: torch.device,
             num_samples: int = 1, use_amp: bool = False) -> Dict[str, float]:
    """Predictive NLL and MSE per value on the targets, predicting from the context only."""
    model.eval()
    totals = {"nll": 0.0, "mse": 0.0}
    for batch in loader:
        batch = _to_device(batch, device)
        with _autocast(device, use_amp):
            out = _forward(model, batch, num_samples, with_target=False)
        mask = batch.get("target_mask")
        totals["nll"] += gaussian_nll(out.y, batch["y_target"], mask).item()
        totals["mse"] += masked_mse(out.y, batch["y_target"], mask).item()
    return {name: total / len(loader) for name, total in totals.items()}


# ----------------------------------------------------------------------------- checkpoints

def save_checkpoint(path: Path, model: NeuralProcess, optimizer: torch.optim.Optimizer,
                    epoch: int, history: Dict[str, List[float]]) -> None:
    """Saves the weights together with the config, so the model can be rebuilt from the file."""
    torch.save({
        "epoch": epoch,
        "config": asdict(model.cfg),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, path)


def load_model(path: Path, model_class: Type[NeuralProcess],
               device: torch.device = torch.device("cpu")) -> NeuralProcess:
    """Rebuilds a model from a checkpoint written by save_checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # the config holds classes
    model = model_class(NPConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


# ----------------------------------------------------------------------------- training

def train(
    model: NeuralProcess,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    validate_every: int = 1,
    scheduler=None,
    warmup_epochs: int = 0,
    kl_schedule: Optional[Sequence[float]] = None,
    num_samples: int = 1,
    grad_clip: Optional[float] = 1.0,
    early_stopping_patience: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
    use_amp: bool = False,
    verbose: bool = True,
    print_every: int = 1,
) -> Dict[str, List[float]]:
    """Train with the negative ELBO; validate on the predictive NLL.

    Args:
        validate_every: Validate every this many epochs (and always after the last one).
            A ReduceLROnPlateau scheduler is stepped at each validation, so its patience
            counts validations; early_stopping_patience still counts epochs.
        warmup_epochs: Epochs during which a ReduceLROnPlateau scheduler is not stepped and
            early stopping / best-model tracking is off (early losses are not representative).
            Usually the length of the KL schedule.
        kl_schedule: KL weight per epoch; after its end the last value is kept. None = 1.
        num_samples: Latent samples per task, for training and validation.
        grad_clip: Max gradient norm; None disables clipping.
        early_stopping_patience: Stop after this many epochs without validation improvement.
        checkpoint_dir: If given, saves best_model.pt and final_model.pt there.
        use_amp: bfloat16 mixed precision (CUDA only).

    The model ends with the weights of the best validation epoch, if there is a validation set.
    Returns the history: one value per epoch for the training quantities; val_nll and val_mse
    have one value per validation, at the epochs (1-based) listed in val_epoch.
    """
    if validate_every < 1:
        raise ValueError("validate_every must be >= 1")
    history: Dict[str, List[float]] = {key: [] for key in (
        "train_loss", "train_nll", "train_kl", "kl_weight", "lr", "val_epoch", "val_nll", "val_mse")}
    plateau = isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    best_nll, best_state, best_epoch = float("inf"), None, None
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        kl_weight = kl_schedule[min(epoch, len(kl_schedule) - 1)] if kl_schedule else 1.0

        # --- training -----------------------------------------------------------
        model.train()
        sums = {"loss": 0.0, "nll": 0.0, "kl": 0.0}
        batches = train_loader
        if verbose and tqdm is not None:
            batches = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for batch in batches:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp):
                out = _forward(model, batch, num_samples, with_target=True)
            terms = np_loss(out, batch["y_target"], batch.get("target_mask"), kl_weight)
            terms.total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            for name, value in (("loss", terms.total), ("nll", terms.nll), ("kl", terms.kl)):
                sums[name] += value.item()
            if verbose and tqdm is not None:
                batches.set_postfix(loss=f"{terms.total.item():.4f}")

        for name, total in sums.items():
            history[f"train_{name}"].append(total / len(train_loader))
        history["kl_weight"].append(kl_weight)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        # --- validation, best model, early stopping -----------------------------
        stop = False
        validated = val_loader is not None and ((epoch + 1) % validate_every == 0 or epoch == epochs - 1)
        if validated:
            metrics = evaluate(model, val_loader, device, num_samples, use_amp)
            history["val_epoch"].append(epoch + 1)
            history["val_nll"].append(metrics["nll"])
            history["val_mse"].append(metrics["mse"])

            if epoch >= warmup_epochs:
                if metrics["nll"] < best_nll:
                    best_nll, best_epoch = metrics["nll"], epoch
                    best_state = copy.deepcopy(model.state_dict())
                    if checkpoint_dir:
                        save_checkpoint(checkpoint_dir / "best_model.pt", model, optimizer, epoch, history)
                # counted in epochs, from the best epoch or the end of the warmup
                since = epoch - (best_epoch if best_epoch is not None else warmup_epochs - 1)
                stop = early_stopping_patience is not None and since >= early_stopping_patience

        # --- learning rate ------------------------------------------------------
        if scheduler is not None:
            if not plateau:
                scheduler.step()
            elif epoch >= warmup_epochs:
                if validated:
                    scheduler.step(history["val_nll"][-1])
                elif val_loader is None:
                    scheduler.step(history["train_loss"][-1])

        # --- report -------------------------------------------------------------
        if verbose and (epoch == 0 or (epoch + 1) % print_every == 0 or epoch == epochs - 1 or stop):
            message = (f"Epoch {epoch + 1}/{epochs}  loss {history['train_loss'][-1]:.4f} "
                       f"(nll {history['train_nll'][-1]:.4f}, kl {history['train_kl'][-1]:.4f}, "
                       f"β={kl_weight:.3f})")
            if validated:
                message += f"  |  val nll {history['val_nll'][-1]:.4f}, mse {history['val_mse'][-1]:.4f}"
            message += f"  |  lr {history['lr'][-1]:.2e}"
            print(message)

        if stop:
            if verbose:
                print(f"Early stopping: no validation improvement for {early_stopping_patience} epochs")
            break

    if checkpoint_dir:
        save_checkpoint(checkpoint_dir / "final_model.pt", model, optimizer, epoch, history)
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Restored the best model (epoch {best_epoch + 1}, val nll {best_nll:.4f})")
    return history