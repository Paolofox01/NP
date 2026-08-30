from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_FILENAME = "FlowAroundObstacle_data.npz"
MESH_FILENAME = "FlowAroundObstacle_mesh_reference.xml"
SENSOR_FILENAME = "FlowAroundObstacle_idx_sensors_velocity.pt"


class FlowDataset(Dataset):
    def __init__(self, fields: torch.Tensor, parameters: torch.Tensor):
        self.fields = fields.float()
        self.parameters = parameters.float()

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fields[index], self.parameters[index]


def _load_data(data_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    path = data_dir / DATA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing steady FlowAroundObstacle snapshots: {path}")
    with np.load(path) as data:
        if "v" not in data or "mu" not in data:
            raise KeyError(f"{path} must contain 'v' and 'mu' arrays.")
        velocity = torch.from_numpy(data["v"])
        parameters = torch.from_numpy(data["mu"])

    if velocity.ndim != 3 or parameters.ndim != 3:
        raise ValueError("Expected 'v' and 'mu' with shape (trajectories, time, features).")
    return velocity.float(), parameters.float()


def _velocity_x_coordinates(data_dir: Path, nstate: int) -> torch.Tensor:
    mesh_path = data_dir / MESH_FILENAME
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing reference mesh: {mesh_path}")

    try:
        from dolfin import Mesh, VectorFunctionSpace
    except ImportError as error:
        raise ImportError("FlowAroundObs training requires FEniCS/dolfin to load mesh coordinates.") from error

    mesh = Mesh(str(mesh_path))
    velocity_space = VectorFunctionSpace(mesh, "CG", 2)
    coordinates = torch.as_tensor(
        velocity_space.sub(0).collapse().tabulate_dof_coordinates(), dtype=torch.float32
    )
    if len(coordinates) != nstate:
        raise ValueError(
            f"Velocity-x data has {nstate} states but the reference mesh has {len(coordinates)} x-component DOFs."
        )
    mins = coordinates.min(dim=0).values
    ranges = (coordinates.max(dim=0).values - mins).clamp_min(1e-8)
    return 2.0 * (coordinates - mins) / ranges - 1.0


def prepare_data(data_dir: Path):
    velocity, parameters = _load_data(data_dir)
    velocity_x = velocity[:, :, 0::2]
    coordinates = _velocity_x_coordinates(data_dir, velocity_x.shape[-1])

    sensor_path = data_dir / SENSOR_FILENAME
    if not sensor_path.exists():
        raise FileNotFoundError(f"Missing velocity-x sensor indices: {sensor_path}")
    sensors = torch.as_tensor(torch.load(sensor_path, weights_only=True), dtype=torch.long).unique()
    if sensors.numel() == 0 or sensors.min() < 0 or sensors.max() >= velocity_x.shape[-1]:
        raise ValueError("FlowAroundObstacle sensor indices are empty or outside the velocity-x state range.")

    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(velocity_x), generator=generator)
    train_end = round(0.8 * len(indices))
    remaining = indices[train_end:]
    splits = {
        "train": indices[:train_end],
        "val": remaining[::2],
        "test": remaining[1::2],
    }
    datasets = {
        name: FlowDataset(velocity_x[index], parameters[index]) for name, index in splits.items()
    }
    return datasets, coordinates, sensors


def spatiotemporal_collate_fn(
    batch,
    mesh_coords: torch.Tensor,
    fixed_sensor_locations: torch.Tensor,
    num_target_min: int = 2048,
    num_target_max: int = 4096,
):
    fields = torch.stack([item[0] for item in batch])
    parameters = torch.stack([item[1] for item in batch])
    batch_size, ntimes, nstate = fields.shape
    sensors = fixed_sensor_locations

    if ntimes <= 20:
        raise ValueError("FlowAroundObstacle trajectories need more than 20 time steps.")
    time_index = torch.randint(20, ntimes, ()).item()
    lag = (0, 4, 9, 19)[torch.randint(4, ()).item()]
    context_times = torch.arange(time_index - lag, time_index + 1)
    context_state_indices = sensors.repeat(lag + 1)
    context_time_indices = context_times.repeat_interleave(len(sensors))

    available = torch.ones(nstate, dtype=torch.bool)
    available[sensors] = False
    extras = torch.arange(nstate)[available]
    num_extra = min(torch.randint(num_target_min, num_target_max + 1, ()).item(), len(extras))
    target_indices = torch.cat([sensors, extras[torch.randperm(len(extras))[:num_extra]]])
    target_times = torch.full((len(target_indices),), time_index, dtype=torch.long)

    context_coords = torch.cat([
        ((context_time_indices - time_index).float() / ntimes).unsqueeze(-1),
        mesh_coords[context_state_indices],
    ], dim=-1)
    target_coords = torch.cat([
        torch.zeros((len(target_indices), 1)), mesh_coords[target_indices],
    ], dim=-1)
    mu_context = parameters[:, context_time_indices, :]
    mu_target = parameters[:, target_times, :]
    x_context = torch.cat([mu_context, context_coords.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
    x_target = torch.cat([mu_target, target_coords.unsqueeze(0).expand(batch_size, -1, -1)], dim=-1)
    y_context = fields[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = fields[:, target_times, target_indices].unsqueeze(-1)
    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()


def make_loaders(datasets, coordinates: torch.Tensor, sensors: torch.Tensor, batch_size: int = 16):
    collate = partial(
        spatiotemporal_collate_fn,
        mesh_coords=coordinates,
        fixed_sensor_locations=sensors,
    )
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader