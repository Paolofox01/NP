from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GoPro.gopro_common import DeepONetDeterministic, make_don_loaders, prepare_data


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for history, coords, targets in loader:
            losses.append(F.mse_loss(model(history.to(device), coords.to(device)), targets.to(device)).item())
    return sum(losses) / len(losses)


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    checkpoint_dir = data_dir / "checkpoints_gopro_don_det"
    checkpoint_dir.mkdir(exist_ok=True)
    datasets, coords, sensors, _, _, _, _, _ = prepare_data(data_dir)
    train_loader, val_loader = make_don_loaders(datasets, coords, sensors)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepONetDeterministic(len(sensors)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=128, min_lr=1e-6)
    best = float("inf")
    for epoch in range(1, 3001):
        model.train()
        losses = []
        for history, coords_batch, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(history.to(device), coords_batch.to(device)), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | train MSE {sum(losses) / len(losses):.6f} | val MSE {val_loss:.6f}")


if __name__ == "__main__":
    main()
