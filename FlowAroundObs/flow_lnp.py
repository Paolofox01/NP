from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FlowAroundObs.flow_common import make_loaders, prepare_data
from LNP.LATNPsimple import LatNP_simple
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    datasets, coordinates, sensors = prepare_data(data_dir)
    train_loader, val_loader = make_loaders(datasets, coordinates, sensors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FlowAroundObs LNP] Device: {device}; train batches: {len(train_loader)}")

    model = LatNP_simple(
        x_dim=7, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2,
        activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3,
        num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=False,
    ).to(device)
    checkpoints = data_dir / "checkpoints_flow_lnp"

    phase1_epochs = 500
    phase1_optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    train_np(
        train_loader, model, phase1_optimizer, ELBOLossNP(beta=1.0), device,
        epochs=phase1_epochs, val_loader=val_loader, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(checkpoints / "phase1"), beta_schedule=[0.0] * phase1_epochs,
        early_stopping_start_epoch=phase1_epochs + 1,
    )
    phase1_path = checkpoints / "phase1_complete.pt"
    phase1_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), phase1_path)

    phase2_epochs, ramp_epochs = 2500, 1000
    model.load_state_dict(torch.load(phase1_path, map_location=device, weights_only=True))
    phase2_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(phase2_optimizer, mode="min", factor=0.5, patience=256, min_lr=1e-6)
    beta_schedule = [epoch / ramp_epochs for epoch in range(ramp_epochs)] + [1.0] * (phase2_epochs - ramp_epochs)
    train_np(
        train_loader, model, phase2_optimizer, ELBOLossNP(beta=1.0), device,
        epochs=phase2_epochs, val_loader=val_loader, scheduler=scheduler, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(checkpoints / "phase2"), beta_schedule=beta_schedule,
        early_stopping_start_epoch=ramp_epochs,
    )


if __name__ == "__main__":
    main()