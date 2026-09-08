from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_FILENAME = "trl_density.npz"
DEFAULT_SENSORS = 16
DEFAULT_TARGET_POINTS = 4096


class TRLDataset(Dataset):
    """One full turbulent radiative layer density trajectory per sample."""

    def __init__(self, fields: torch.Tensor):
        if fields.ndim != 4:
            raise ValueError(
                "Expected TRL fields with shape (trajectories, time, x, y), "
                f"got {tuple(fields.shape)}."
            )
        self.fields = fields.float()

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.fields[index]


def build_grid(shape: tuple[int, int]) -> torch.Tensor:
    """Build normalized Cartesian coordinates for a 2D grid."""
    axes = [
        torch.linspace(-1.0, 1.0, length) if length > 1 else torch.zeros(length)
        for length in shape
    ]
    x, y = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([x, y], dim=-1).reshape(-1, 2).float()


def prepare_data(data_dir: Path, data_filename: str = DEFAULT_DATA_FILENAME):
    data_path = data_dir / data_filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing TRL density data: {data_path}. Run TRL/export_trl.py first."
        )
    with np.load(data_path) as data:
        datasets = {
            name: TRLDataset(torch.from_numpy(data[name])) for name in ("train", "valid", "test")
        }
    # Normalize using train-split statistics so the raw physical density scale
    # doesn't mismatch the model's near-unit-scale output/variance initialization.
    data_min = datasets["train"].fields.min()
    data_max = datasets["train"].fields.max()
    for dataset in datasets.values():
        dataset.fields = (dataset.fields - data_min) / (data_max - data_min + 1e-8)
    first_field = datasets["train"].fields[0]
    coordinates = build_grid(first_field.shape[1:])
    return datasets, coordinates, first_field.shape[1:]



def choose_sensors(
    nstate: int,
    num_sensors: int = DEFAULT_SENSORS,
    seed: int = 0,
    spatial_shape: tuple[int, int] | None = None,
    col_fraction_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Pick sensor node indices, stratified across the grid when a spatial shape is given.

    A plain ``randperm`` over a large, elongated domain can by chance draw all of its
    (few) samples from one corner; splitting the grid into cells and drawing one sensor
    per cell guarantees coverage across both spatial dimensions. ``col_fraction_range``
    restricts sensors to a horizontal band (e.g. ``(1/3, 1/2)``) while still spreading
    them across the full vertical extent, useful for targeting a known interface region.
    """
    if not 1 <= num_sensors <= nstate:
        raise ValueError(f"num_sensors must be in [1, {nstate}], got {num_sensors}.")
    generator = torch.Generator().manual_seed(seed)
    if spatial_shape is None:
        if col_fraction_range is not None:
            raise ValueError("col_fraction_range requires spatial_shape.")
        return torch.randperm(nstate, generator=generator)[:num_sensors].sort().values

    rows, cols = spatial_shape
    if rows * cols != nstate:
        raise ValueError(f"spatial_shape {spatial_shape} does not match nstate {nstate}.")

    if col_fraction_range is None:
        col_lo, col_hi = 0, cols
    else:
        low_frac, high_frac = col_fraction_range
        if not 0.0 <= low_frac < high_frac <= 1.0:
            raise ValueError(f"col_fraction_range must satisfy 0 <= low < high <= 1, got {col_fraction_range}.")
        col_lo = int(round(low_frac * cols))
        col_hi = max(int(round(high_frac * cols)), col_lo + 1)
    band_cols = col_hi - col_lo

    grid_cols = max(1, round((num_sensors * band_cols / rows) ** 0.5))
    grid_rows = max(1, -(-num_sensors // grid_cols))  # ceil division
    row_bounds = torch.linspace(0, rows, grid_rows + 1).round().long()
    col_bounds = torch.linspace(col_lo, col_hi, grid_cols + 1).round().long()

    cell_choices = []
    for row_start, row_end in zip(row_bounds[:-1].tolist(), row_bounds[1:].tolist()):
        if row_end <= row_start:
            continue
        for col_start, col_end in zip(col_bounds[:-1].tolist(), col_bounds[1:].tolist()):
            if col_end <= col_start:
                continue
            row_pick = row_start + torch.randint(row_end - row_start, (1,), generator=generator).item()
            col_pick = col_start + torch.randint(col_end - col_start, (1,), generator=generator).item()
            cell_choices.append(row_pick * cols + col_pick)

    cell_order = torch.randperm(len(cell_choices), generator=generator)[:num_sensors]
    sensors = torch.as_tensor(cell_choices, dtype=torch.long)[cell_order]
    return sensors.sort().values


def sample_target_indices(
    nstate: int,
    target_count: int,
    spatial_shape: tuple[int, int] | None = None,
    boundary_col_fraction_range: tuple[float, float] | None = None,
    boundary_target_fraction: float = 0.5,
) -> torch.Tensor:
    """Sample target node indices, optionally oversampling a horizontal boundary band.

    Uniform target sampling barely supervises a thin, high-interest band (e.g. a fluid
    interface): most points land in the much larger surrounding region. Reserving a
    fraction of targets for the band gives the loss real gradient signal there.
    """
    if spatial_shape is None or boundary_col_fraction_range is None:
        return torch.randperm(nstate)[:target_count]

    rows, cols = spatial_shape
    low_frac, high_frac = boundary_col_fraction_range
    col_lo = int(round(low_frac * cols))
    col_hi = max(int(round(high_frac * cols)), col_lo + 1)
    col_indices = torch.arange(nstate) % cols
    boundary_mask = (col_indices >= col_lo) & (col_indices < col_hi)
    boundary_pool = boundary_mask.nonzero(as_tuple=True)[0]
    other_pool = (~boundary_mask).nonzero(as_tuple=True)[0]

    boundary_count = min(int(round(target_count * boundary_target_fraction)), len(boundary_pool))
    rest_count = min(target_count - boundary_count, len(other_pool))
    boundary_sample = boundary_pool[torch.randperm(len(boundary_pool))[:boundary_count]]
    rest_sample = other_pool[torch.randperm(len(other_pool))[:rest_count]]
    return torch.cat([boundary_sample, rest_sample])


def trl_collate_fn(
    batch,
    mesh_coords: torch.Tensor,
    sensor_locations: torch.Tensor,
    num_target_points: int = DEFAULT_TARGET_POINTS,
    history_options: tuple = (0, 4, 9, 19),
    boundary_col_fraction_range: tuple[float, float] | None = None,
    boundary_target_fraction: float = 0.5,
):
    """Create sparse 2D context-target batches with [t, x, y] coordinates."""
    trajectories = torch.stack(batch).float()
    batch_size, ntimes, *spatial_shape = trajectories.shape
    fields = trajectories.reshape(batch_size, ntimes, -1)
    nstate = fields.shape[-1]
    sensors = sensor_locations.long()

    if ntimes <= max(history_options) + 1:
        raise ValueError("TRL trajectories are too short for the requested history options.")

    time_index = torch.randint(max(history_options) + 1, ntimes, ()).item()
    lag = history_options[torch.randint(len(history_options), ()).item()]
    context_times = torch.arange(time_index - lag, time_index + 1)
    context_state_indices = sensors.repeat(lag + 1)
    context_time_indices = context_times.repeat_interleave(len(sensors))

    target_count = min(num_target_points, nstate)
    target_indices = sample_target_indices(
        nstate, target_count, tuple(spatial_shape), boundary_col_fraction_range, boundary_target_fraction
    )
    target_times = torch.full((len(target_indices),), time_index, dtype=torch.long)

    x_context = torch.cat(
        [
            ((context_time_indices - time_index).float() / ntimes).unsqueeze(-1),
            mesh_coords[context_state_indices],
        ],
        dim=-1,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    x_target = torch.cat(
        [torch.zeros((target_count, 1)), mesh_coords[target_indices]],
        dim=-1,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    y_context = fields[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = fields[:, target_times, target_indices].unsqueeze(-1)
    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()


def make_loaders(
    datasets,
    mesh_coords: torch.Tensor,
    sensors: torch.Tensor,
    batch_size: int = 2,
    num_target_points: int = DEFAULT_TARGET_POINTS,
    boundary_col_fraction_range: tuple[float, float] | None = None,
    boundary_target_fraction: float = 0.5,
):
    collate = partial(
        trl_collate_fn,
        mesh_coords=mesh_coords,
        sensor_locations=sensors,
        num_target_points=num_target_points,
        boundary_col_fraction_range=boundary_col_fraction_range,
        boundary_target_fraction=boundary_target_fraction,
    )
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def eval_inputs(
    trajectory: torch.Tensor,
    mesh_coords: torch.Tensor,
    sensor_locations: torch.Tensor,
    time_index: int,
    lag: int = 19,
):
    """Build one full-field prediction problem for testing."""
    fields = trajectory.float().unsqueeze(0).reshape(1, trajectory.shape[0], -1)
    ntimes, nstate = fields.shape[1:]
    sensors = sensor_locations.long()
    if not 0 <= lag < time_index < ntimes:
        raise ValueError("Evaluation time index must leave room for the requested lag.")

    context_times = torch.arange(time_index - lag, time_index + 1)
    context_state_indices = sensors.repeat(lag + 1)
    context_time_indices = context_times.repeat_interleave(len(sensors))
    target_indices = torch.arange(nstate)
    target_times = torch.full((nstate,), time_index, dtype=torch.long)

    x_context = torch.cat(
        [
            ((context_time_indices - time_index).float() / ntimes).unsqueeze(-1),
            mesh_coords[context_state_indices],
        ],
        dim=-1,
    ).unsqueeze(0)
    x_target = torch.cat(
        [torch.zeros((nstate, 1)), mesh_coords[target_indices]],
        dim=-1,
    ).unsqueeze(0)
    y_context = fields[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = fields[:, target_times, target_indices].unsqueeze(-1)
    return x_context, y_context, x_target, y_target
