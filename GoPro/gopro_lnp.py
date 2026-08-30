from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GoPro.gopro_common import make_np_loaders, prepare_data, set_epoch_targets
from LNP.LATNPsimple import LatNP_simple
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np


def main() -> None:
    print("[GoPro LNP] Starting training script...")
    data_dir = Path(__file__).resolve().parent
    checkpoints = data_dir / "checkpoints_gopro_lnp"
    print(f"[GoPro LNP] Data directory: {data_dir}")
    print(f"[GoPro LNP] Checkpoint directory: {checkpoints}")
    print("[GoPro LNP] Preparing GoPro data...")
    datasets, coords, sensors, _, _, _, _, _ = prepare_data(data_dir)
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
    print("PHASE 1: DETERMINISTIC WARMUP (500 Epochs)")
    print("="*60)
    
    epochs_p1 = 500
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
        on_epoch_start=lambda _: set_epoch_targets(coords.shape[0], sensors, num_target=8192, history_options=(10, 20, 30, 40)),
    )
    
    phase1_complete_path = checkpoints / "phase1_complete.pt"
    torch.save(model.state_dict(), phase1_complete_path)
    print(f"\nPhase 1 Complete! Model saved to {phase1_complete_path}")
    
    # =====================================================================
    # PHASE 2: STOCHASTIC FINE-TUNING (Beta Ramp + LR Decay)
    # =====================================================================
    print("\n" + "="*60)
    print("PHASE 2: STOCHASTIC FINE-TUNING (2500 Epochs)")
    print("="*60)
    
    epochs_p2 = 2500
    ramp_epochs = 1000
    beta_target = 1.0
    
    model.load_state_dict(torch.load(phase1_complete_path, map_location=device))
    for param in model.latent.parameters():
        param.requires_grad = True
    
    optimizer_p2 = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
    scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode="min", factor=0.5, patience=256, min_lr=1e-6)
    beta_schedule_p2 = [beta_target * (e / ramp_epochs) for e in range(ramp_epochs)] + [beta_target] * (epochs_p2 - ramp_epochs)
    p2_checkpoints_dir = checkpoints / "phase2"
    p2_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    history = train_np(
        train_loader, model, optimizer_p2, ELBOLossNP(beta=1.0), device,
        epochs=epochs_p2, val_loader=val_loader, scheduler=scheduler_p2, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(p2_checkpoints_dir), beta_schedule=beta_schedule_p2,
        early_stopping_start_epoch=ramp_epochs,
        on_epoch_start=lambda _: set_epoch_targets(coords.shape[0], sensors, num_target=8192, history_options=(10, 20, 30, 40)),
    )
    
    print("\nTraining completely finished!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
