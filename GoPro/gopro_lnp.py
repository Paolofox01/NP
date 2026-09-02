from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GoPro.gopro_common import DEFAULT_HISTORY_LENGTH, make_np_loaders, np_eval_inputs, prepare_data, set_epoch_targets
from LNP.LATNPsimple import LatNP_simple
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np


def save_sensor_location_plot(test_dataset, sensors, height, width, output_path: Path) -> None:
    _, frame_history = test_dataset[0]
    frame = frame_history[-1].reshape(height, width)
    sensor_rows = [sensor // width for sensor in sensors]
    sensor_columns = [sensor % width for sensor in sensors]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.imshow(frame, cmap="gray")
    axis.scatter(sensor_columns, sensor_rows, marker="x", s=100, c="cyan", linewidths=2)
    axis.set_title("GoPro sensor locations")
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved sensor location plot to {output_path}")


def print_max_history_test_result(model, test_dataset, coords, sensors, height, width, device, output_path: Path) -> None:
    x_context, y_context, x_target, y_target = np_eval_inputs(
        [test_dataset[0]], coords, sensors, history=DEFAULT_HISTORY_LENGTH
    )
    x_context, y_context, x_target, y_target = (
        tensor.to(device) for tensor in (x_context, y_context, x_target, y_target)
    )
    model.eval()
    with torch.no_grad():
        y_pred, y_var, *_ = model(x_context, y_context, x_target, y_target)
    error = y_pred - y_target
    print(
        f"Max-history test case ({DEFAULT_HISTORY_LENGTH} frames) - "
        f"MSE: {error.square().mean().item():.6f}, MAE: {error.abs().mean().item():.6f}"
    )
    target_image = y_target[0, :, 0].cpu().reshape(height, width)
    prediction_image = y_pred[0, :, 0].cpu().reshape(height, width)
    variance_image = y_var[0, :, 0].cpu().reshape(height, width)
    sensor_rows = [sensor // width for sensor in sensors]
    sensor_columns = [sensor % width for sensor in sensors]
    value_min = min(target_image.min().item(), prediction_image.min().item())
    value_max = max(target_image.max().item(), prediction_image.max().item())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, image, title, cmap, vmin, vmax in (
        (axes[0], target_image, "Test target", "gray", value_min, value_max),
        (axes[1], prediction_image, "Prediction", "gray", value_min, value_max),
        (axes[2], variance_image, "Predictive variance", "magma", 0.0, variance_image.max().item()),
    ):
        plot = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.scatter(sensor_columns, sensor_rows, marker="x", s=80, c="cyan", linewidths=2)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Maximum-history test prediction ({DEFAULT_HISTORY_LENGTH} frames)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved test prediction diagnostic to {output_path}")


def main() -> None:
    print("[GoPro LNP] Starting training script...")
    data_dir = Path(__file__).resolve().parent
    checkpoints = data_dir / "checkpoints_gopro_lnp"
    print(f"[GoPro LNP] Data directory: {data_dir}")
    print(f"[GoPro LNP] Checkpoint directory: {checkpoints}")
    print("[GoPro LNP] Preparing GoPro data...")
    datasets, coords, sensors, _, height, width, _, _ = prepare_data(data_dir)
    save_sensor_location_plot(
        datasets["test"], sensors, height, width, checkpoints / "sensor_locations.png"
    )
    print("[GoPro LNP] Building train/validation loaders...")
    train_loader, val_loader = make_np_loaders(datasets, coords, sensors, batch_size=32)
    print(f"[GoPro LNP] Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GoPro LNP] Using device: {device}")
    print("[GoPro LNP] Initializing model...")
    model = LatNP_simple(
        x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2,
        activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3,
        num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=False,
    ).to(device)
    
    # =====================================================================
    # PHASE 1: DETERMINISTIC WARMUP (Beta = 0)
    # =====================================================================
    print("\n" + "="*60)
    print("PHASE 1: DETERMINISTIC WARMUP (200 Epochs)")
    print("="*60)
    
    epochs_p1 = 200
    optimizer_p1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.0)
    beta_schedule_p1 = [0.0] * epochs_p1
    p1_checkpoints_dir = checkpoints / "phase1"
    p1_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    train_np(
        train_loader, model, optimizer_p1, ELBOLossNP(beta=1.0), device,
        epochs=epochs_p1, val_loader=val_loader, scheduler=None, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(p1_checkpoints_dir), beta_schedule=beta_schedule_p1,
        early_stopping_start_epoch=epochs_p1 + 1,
        on_epoch_start=lambda _: set_epoch_targets(
            coords.shape[0], sensors, num_target=8192, history_options=(10, 20, 30, 40), drop_sensor_options=(0, 1),
        ),
    )
    
    phase1_complete_path = checkpoints / "phase1_complete.pt"
    torch.save(model.state_dict(), phase1_complete_path)
    print(f"\nPhase 1 Complete! Model saved to {phase1_complete_path}")
    
    # =====================================================================
    # PHASE 2: STOCHASTIC FINE-TUNING (Beta Ramp + LR Decay)
    # =====================================================================
    print("\n" + "="*60)
    print("PHASE 2: STOCHASTIC FINE-TUNING (600 Epochs)")
    print("="*60)
    
    epochs_p2 = 800
    ramp_epochs = 600
    beta_target = 1.0
    
    model.load_state_dict(torch.load(phase1_complete_path, map_location=device))
    for param in model.latent.parameters():
        param.requires_grad = True
    
    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode="min", factor=0.5, patience=64, min_lr=1e-6)
    beta_schedule_p2 = [beta_target * (e / ramp_epochs) for e in range(ramp_epochs)] + [beta_target] * (epochs_p2 - ramp_epochs)
    p2_checkpoints_dir = checkpoints / "phase2"
    p2_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    history = train_np(
        train_loader, model, optimizer_p2, ELBOLossNP(beta=1.0), device,
        epochs=epochs_p2, val_loader=val_loader, scheduler=scheduler_p2, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(p2_checkpoints_dir), beta_schedule=beta_schedule_p2,
        early_stopping_start_epoch=ramp_epochs,
        on_epoch_start=lambda _: set_epoch_targets(
            coords.shape[0], sensors, num_target=2048, history_options=(10, 20, 30, 40), drop_sensor_options=(0, 1),
        ),
    )
    
    best_model_path = p2_checkpoints_dir / "best_model.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best Phase 2 checkpoint was not saved: {best_model_path}")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\nTraining completely finished!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print_max_history_test_result(
        model, datasets["test"], coords, sensors, height, width, device,
        checkpoints / "max_history_test_prediction.png",
    )


if __name__ == "__main__":
    main()
