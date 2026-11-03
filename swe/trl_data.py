"""Turbulent Radiative Layer 2D (The Well) as Neural Process tasks, with the task builder of swe_data.py.

The data: 90 trajectories (72 train, 9 valid, 9 test; one file per cooling time tcool and split) of 101 time
steps on a 128 x 384 grid, x in [-0.5, 0.5] (periodic) and y in [-1, 2]. Cold dense gas at the bottom and hot
dilute gas at the top: they mix through a layer that starts at y = 0 and rises to y = 0.5 or so.
The whole dataset is 6.9 GB, so it is loaded in memory, from The Well's files on disk or streamed from the
Hugging Face Hub.

A task is one trajectory at one time t0:
- context: the readings of the sensors at the lag + 1 times t0 - lag, ..., t0 (a few sensors randomly dropped);
- targets: grid points at t0 that are not sensors.
The sensors lie on the fluid separation: a SENSOR_GRID (5 x 2) lattice with x spread over the periodic width and
y within SENSOR_Y_FRACTION of the height of the domain, (1/3, 1/2), that is y in [0, 0.5].

x = ([log10 tcool,] relative time, x, y); the cooling time is only an input with use_tcool=True.
y = (density, pressure, velocity_x, velocity_y), standardized per channel with training-set statistics.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from swe_data import HF_BASE, SWEData, _download, _list_files, array_info, read_array

__all__ = ["DATASET", "VARIABLES", "HISTORY_OPTIONS", "DROP_SENSOR_OPTIONS", "SENSOR_GRID", "SENSOR_Y_FRACTION",
           "prepare_trl"]

DATASET = "turbulent_radiative_layer_2D"
VARIABLES = ["density", "pressure", "velocity_x", "velocity_y"]
HISTORY_OPTIONS = (1,)             # past readings (lag) a task can use; the readings are consecutive time steps
DROP_SENSOR_OPTIONS = (0, 1, 2, 5)  # sensors a task can drop, out of SENSOR_GRID[0] * SENSOR_GRID[1] = 10
SENSOR_GRID = (5, 2)               # sensors along x, and rows along y
SENSOR_Y_FRACTION = (1 / 3, 1 / 2)  # where the sensor rows lie, as fractions of the domain height y in [-1, 2]
X_RANGE, Y_RANGE = (-0.5, 0.5), (-1.0, 2.0)


def _root(base_path) -> str:
    """The folder holding <dataset>/data/<split>: base_path itself, or its `datasets` folder (where
    the-well-download puts the files)."""
    base = str(base_path)
    if "://" not in base and (Path(base) / "datasets").is_dir():
        return str(Path(base) / "datasets")
    return base


def _sensor_points(x: Tensor, y: Tensor, grid: Tuple[int, int], y_fraction: Tuple[float, float]) -> Tensor:
    """Flat grid indices (x-major) of the grid points closest to a lattice of sensors: `grid[0]` positions along
    the periodic x axis, `grid[1]` rows between the two fractions of the height of the domain."""
    n_x, n_rows = grid
    sensor_x = X_RANGE[0] + (torch.arange(n_x) + 0.5) / n_x * (X_RANGE[1] - X_RANGE[0])
    low, high = (Y_RANGE[0] + f * (Y_RANGE[1] - Y_RANGE[0]) for f in y_fraction)
    sensor_y = low + (torch.arange(n_rows) + 0.5) / n_rows * (high - low)
    ix = (x[None, :] - sensor_x[:, None]).abs().argmin(dim=1)
    iy = (y[None, :] - sensor_y[:, None]).abs().argmin(dim=1)
    points = (ix[:, None] * len(y) + iy[None, :]).reshape(-1)
    if points.unique().numel() != len(points):
        raise ValueError("could not place the sensors on distinct grid points")
    return points.sort().values


def _load_split(fs, root: str, split: str, workers: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """All trajectories of a split: (trajectories, times, points, 4) raw values, log10 tcool (trajectories,),
    and the x and y grids."""
    files = _list_files(fs, root, DATASET, split)
    fields = ["t0_fields/density", "t0_fields/pressure", "t1_fields/velocity", "scalars/tcool"]
    infos = [array_info(fs, f, fields + (["dimensions/x", "dimensions/y"] if k == 0 else []))
             for k, f in enumerate(files)]
    counts = [info["t0_fields/density"][0][0] for info in infos]
    n_times, n_x, n_y = infos[0]["t0_fields/density"][0][1:]
    y_all = torch.empty(sum(counts), n_times, n_x * n_y, 4)
    log_tcool = torch.empty(sum(counts))
    starts = np.concatenate([[0], np.cumsum(counts)])

    def work(k: int) -> None:
        path, info, a, b = files[k], infos[k], starts[k], starts[k + 1]
        density, pressure = (read_array(fs, path, name, info[name]) for name in fields[:2])
        velocity = read_array(fs, path, fields[2], info[fields[2]])
        tcool = float(read_array(fs, path, fields[3], info[fields[3]]))
        flat = lambda values: torch.from_numpy(values).reshape(b - a, n_times, n_x * n_y, -1)
        y_all[a:b] = torch.cat([flat(density), flat(pressure), flat(velocity)], dim=-1)
        log_tcool[a:b] = math.log10(tcool)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for future in [executor.submit(work, k) for k in range(len(files))]:
            future.result()
    x = torch.as_tensor(read_array(fs, files[0], "dimensions/x", infos[0]["dimensions/x"]))
    y = torch.as_tensor(read_array(fs, files[0], "dimensions/y", infos[0]["dimensions/y"]))
    return y_all, log_tcool, x, y


def prepare_trl(base_path=HF_BASE, splits: Sequence[str] = ("train", "valid", "test"), use_tcool: bool = False,
                sensor_grid: Tuple[int, int] = SENSOR_GRID, sensor_y_fraction: Tuple[float, float] = SENSOR_Y_FRACTION,
                history_options: Sequence[int] = HISTORY_OPTIONS, drop_options: Sequence[int] = DROP_SENSOR_OPTIONS,
                workers: int = 4, download: bool = False) -> SWEData:
    """Loads the dataset into memory, standardized per channel with the training mean and std.

    base_path: a local folder with The Well's files (<base_path>/datasets/<dataset>/data/<split>/*.hdf5, or
    <base_path>/<dataset>/data/...), or the Hugging Face Hub (the default, streamed: about 7 GB to transfer).
    download: first download the files into base_path with the-well-download (local folders only).
    use_tcool: also give the models log10 of the cooling time of the trajectory as an input.
    """
    import fsspec

    if download:
        _download(Path(base_path), DATASET)
    fs, root = fsspec.url_to_fs(_root(base_path))
    loaded = {split: _load_split(fs, root, split, workers) for split in splits}
    if "train" not in loaded:
        raise ValueError("the training split is needed for the standardization statistics")
    x, y = loaded["train"][2], loaded["train"][3]
    coordinates = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)     # x-major, as the fields
    sensors = _sensor_points(x, y, sensor_grid, sensor_y_fraction)

    total, total_sq, count = (torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64), 0)
    for trajectory in loaded["train"][0]:
        values = trajectory.reshape(-1, 4).double()
        total += values.sum(0)
        total_sq += (values ** 2).sum(0)
        count += len(values)
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp(min=1e-16).sqrt().clamp(min=1e-8)
    mean, std = mean.float(), std.float()
    trajectories = {}
    for split, (values, _, _, _) in loaded.items():
        trajectories[split] = values.sub_(mean).div_(std)

    params, x_low, x_high = None, [-1.0, X_RANGE[0], Y_RANGE[0]], [0.0, X_RANGE[1], Y_RANGE[1]]
    if use_tcool:
        params = {split: log_tcool.unsqueeze(-1) for split, (_, log_tcool, _, _) in loaded.items()}
        train_tcool = loaded["train"][1]
        x_low, x_high = [train_tcool.min().item()] + x_low, [train_tcool.max().item()] + x_high
    weights = torch.ones(len(coordinates))
    weights[sensors] = 0.0                                                                      # targets are not sensors
    return SWEData(trajectories, coordinates, mean, std, sensors, 1, weights, geometry="plane",
                   history_options=tuple(history_options), drop_options=tuple(drop_options), params=params,
                   x_low=x_low, x_high=x_high)
