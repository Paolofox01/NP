from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GoPro.gopro_common import make_np_loaders, prepare_data
from LNP.LATNPsimple import LatNP_simple
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np


def main() -> None:
    print("[GoPro LNP] Starting training script...")
    data_dir = Path(__file__).resolve().parent
    checkpoints = data_dir / "checkpoints_gopro_lnp" / "phase2"
    print(f"[GoPro LNP] Data directory: {data_dir}")
    print(f"[GoPro LNP] Checkpoint directory: {checkpoints}")
    print("[GoPro LNP] Preparing GoPro data...")
    datasets, coords, sensors, _, _, _, _, _ = prepare_data(data_dir)
    print("[GoPro LNP] Building train/validation loaders...")
    train_loader, val_loader = make_np_loaders(datasets, coords, sensors)
    print(f"[GoPro LNP] Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GoPro LNP] Using device: {device}")
    print("[GoPro LNP] Initializing model...")
    model = LatNP_simple(
        x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2,
        activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3,
        num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=True,
    ).to(device)
    epochs, ramp = 3000, 1000
    print(f"[GoPro LNP] Training config: epochs={epochs}, ramp={ramp}, lr=5e-4")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=256, min_lr=1e-6
    )
    print("[GoPro LNP] Starting training loop...")
    train_np(
        train_loader, model, optimizer, ELBOLossNP(beta=1.0), device,
        epochs=epochs, val_loader=val_loader, scheduler=scheduler, gradient_clip=1.0,
        early_stopping_patience=256, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(checkpoints),
        beta_schedule=[min(1.0, epoch / ramp) for epoch in range(epochs)],
        early_stopping_start_epoch=ramp,
    )
    print("[GoPro LNP] Training finished.")


if __name__ == "__main__":
    print("[GoPro LNP] Script entry point reached.")
    main()
