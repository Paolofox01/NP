from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
import torch.nn as nn

from TRL.trl_common import (
    DEFAULT_DATA_FILENAME,
    DEFAULT_SENSORS,
    DEFAULT_TARGET_POINTS,
    choose_sensors,
    eval_inputs,
    make_loaders,
    prepare_data,
)
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np


def save_sensor_locations(
    coordinates: torch.Tensor,
    sensors: torch.Tensor,
    spatial_shape: tuple[int, int],
    output_dir: Path,
) -> None:
    sensor_coordinates = coordinates[sensors].reshape(-1, 2)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.scatter(coordinates[:, 1], coordinates[:, 0], s=1, alpha=0.12, color="gray")
    axis.scatter(
        sensor_coordinates[:, 1], sensor_coordinates[:, 0],
        s=45, color="magenta", edgecolors="black", linewidths=0.7,
    )
    for index, (x_coord, y_coord) in zip(sensors.tolist(), sensor_coordinates.tolist()):
        axis.annotate(str(index), (y_coord, x_coord), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set_xlabel("y coordinate")
    axis.set_ylabel("x coordinate")
    axis.set_title(f"TRL sensor locations ({len(sensors)} sensors, grid {spatial_shape[0]}x{spatial_shape[1]})")
    axis.set_aspect("equal")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sensor_locations.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved TRL sensor locations to {output_path}")


def save_test_trajectory_gif(
    test_trajectory: torch.Tensor,
    output_dir: Path,
    spatial_shape: tuple[int, int] | None = None,
    sensors: torch.Tensor | None = None,
    show_sensors: bool = True,
) -> None:
    frames = test_trajectory.detach().cpu().numpy()
    value_min, value_max = frames.min(), frames.max()
    figure, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(frames[0], cmap="viridis", vmin=value_min, vmax=value_max, animated=True)
    sensor_scatter = None
    if show_sensors and spatial_shape is not None and sensors is not None:
        # imshow uses raw pixel-index axes, so sensors must be plotted in pixel space too.
        _, cols = spatial_shape
        sensor_indices = sensors.cpu().numpy()
        sensor_rows, sensor_cols = sensor_indices // cols, sensor_indices % cols
        sensor_scatter = axis.scatter(
            sensor_cols, sensor_rows,
            s=45, color="magenta", edgecolors="black", linewidths=0.7,
        )
    axis.set_title("TRL test density trajectory, frame 0")
    axis.axis("off")

    def update(frame_index: int):
        image.set_array(frames[frame_index])
        axis.set_title(f"TRL test density trajectory, frame {frame_index}")
        return (image, sensor_scatter) if sensor_scatter is not None else (image,)

    animation = FuncAnimation(figure, update, frames=len(frames), interval=100, blit=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_density_trajectory.gif"
    animation.save(output_path, writer=PillowWriter(fps=10))
    plt.close(figure)
    print(f"Saved TRL test trajectory GIF to {output_path}")


def train_trl_model(
    model_class: type[nn.Module],
    model_name: str,
    batch_size: int = 16,
    num_target_points: int = DEFAULT_TARGET_POINTS,
    num_sensors: int = DEFAULT_SENSORS,
    phase1_epochs: int = 500,
    phase2_epochs: int = 2500,
    ramp_epochs: int = 1000,
) -> None:
    data_dir = Path(__file__).resolve().parent
    checkpoints = data_dir / f"checkpoints_trl_{model_name.lower()}"
    print(f"[TRL {model_name}] Loading full 2D density data...")
    datasets, coordinates, spatial_shape = prepare_data(data_dir, DEFAULT_DATA_FILENAME)
    sensors = choose_sensors(
        len(coordinates), num_sensors, spatial_shape=spatial_shape, col_fraction_range=(1 / 3, 1 / 2)
    )
    save_sensor_locations(coordinates, sensors, spatial_shape, checkpoints)
    save_test_trajectory_gif(datasets["test"][0], checkpoints, spatial_shape, sensors)
    train_loader, val_loader = make_loaders(
        datasets, coordinates, sensors, batch_size=batch_size, num_target_points=num_target_points,
        boundary_col_fraction_range=(1 / 3, 1 / 2), boundary_target_fraction=0.6,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRL {model_name}] Device: {device}; shape: {spatial_shape}; sensors: {len(sensors)}")
    print(f"[TRL {model_name}] Target points per batch: {num_target_points}")

    model = model_class(
        x_dim=4, y_dim=1, r_dim=128, z_dim=256, hidden_dim=256, n_hidden=2,
        activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3,
        num_frequencies=64, learnable_fourier=True, use_deeponet_decoder=False,
    ).to(device)
    print(f"[TRL {model_name}] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    phase1_optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.0)
    phase1_dir = checkpoints / "phase1"
    train_np(
        train_loader, model, phase1_optimizer, ELBOLossNP(beta=1.0), device,
        epochs=phase1_epochs, val_loader=val_loader, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(phase1_dir), beta_schedule=[0.0] * phase1_epochs,
        early_stopping_start_epoch=phase1_epochs + 1,
    )
    phase1_path = checkpoints / "phase1_complete.pt"
    phase1_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), phase1_path)

    model.load_state_dict(torch.load(phase1_path, map_location=device, weights_only=True))
    phase2_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        phase2_optimizer, mode="min", factor=0.5, patience=256, min_lr=1e-6
    )
    beta_schedule = [epoch / ramp_epochs for epoch in range(ramp_epochs)] + [1.0] * (phase2_epochs - ramp_epochs)
    phase2_dir = checkpoints / "phase2"
    train_np(
        train_loader, model, phase2_optimizer, ELBOLossNP(beta=1.0), device,
        epochs=phase2_epochs, val_loader=val_loader, scheduler=scheduler, gradient_clip=1.0,
        early_stopping_patience=1000, is_meta_learning=True, verbose=True, print_every=10,
        checkpoint_dir=str(phase2_dir), beta_schedule=beta_schedule,
        early_stopping_start_epoch=ramp_epochs,
    )

    best_checkpoint = torch.load(phase2_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()
    print_max_history_test_result(model, datasets, coordinates, sensors, spatial_shape, device, checkpoints)



def print_max_history_test_result(
    model: nn.Module,
    datasets,
    coordinates: torch.Tensor,
    sensors: torch.Tensor,
    spatial_shape: tuple[int, int],
    device: torch.device,
    output_dir: Path,
) -> None:
    test_trajectory = datasets["test"][0]
    time_index = test_trajectory.shape[0] - 1
    lag = min(19, time_index - 1)
    cooling_timescale = datasets["test"].cooling_timescales[0]
    x_context, y_context, x_target, y_target = eval_inputs(
        test_trajectory, cooling_timescale, coordinates, sensors, time_index=time_index, lag=lag
    )
    x_context, y_context, x_target, y_target = (
        tensor.to(device) for tensor in (x_context, y_context, x_target, y_target)
    )
    with torch.no_grad():
        y_pred, y_var, *_ = model(x_context, y_context, x_target, y_target)

    error = y_pred - y_target
    print(
        f"Max-history TRL test case (lag={lag}) - "
        f"MSE: {error.square().mean().item():.6f}, MAE: {error.abs().mean().item():.6f}"
    )

    target_field = y_target[0, :, 0].cpu().reshape(spatial_shape)
    prediction_field = y_pred[0, :, 0].cpu().reshape(spatial_shape)
    variance_field = y_var[0, :, 0].cpu().reshape(spatial_shape)
    value_min = min(target_field.min().item(), prediction_field.min().item())
    value_max = max(target_field.max().item(), prediction_field.max().item())

    # imshow uses raw pixel-index axes, so sensors must be plotted in pixel space too.
    _, cols = spatial_shape
    sensor_indices = sensors.cpu().numpy()
    sensor_rows, sensor_cols = sensor_indices // cols, sensor_indices % cols

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, image, title, cmap, vmin, vmax in (
        (axes[0], target_field, "Test target", "viridis", value_min, value_max),
        (axes[1], prediction_field, "Prediction", "viridis", value_min, value_max),
        (axes[2], variance_field, "Predictive variance", "magma", 0.0, variance_field.max().item()),
    ):
        plot = axis.imshow(image.numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.scatter(sensor_cols, sensor_rows, marker="x", s=60, c="cyan", linewidths=1.5)
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("TRL test prediction, final time")
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "max_history_test_prediction.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved TRL test prediction diagnostic to {output_path}")
