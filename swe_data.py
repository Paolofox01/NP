"""PlanetSWE (The Well) as Neural Process tasks.

A task is one trajectory at one time t0:
- context: the sensors at the lag + 1 sampled times up to t0 (a few sensors randomly dropped);
- targets: TARGET_POINTS grid points at t0, sampled in proportion to their cell area.

x = (relative time, x, y, z): relative time in [-1, 0] (0 = t0), position on the unit sphere.
y = (h, u_theta, u_phi), standardized per channel with training-set statistics.

Batches are the dictionaries training.py expects.
"""

import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler

__all__ = ["VARIABLES", "X_LOW", "X_HIGH", "prepare_data", "load_sensors",
           "make_train_loader", "make_validation_batches"]

VARIABLES = ["h", "u_theta", "u_phi"]
X_LOW = [-1.0, -1.0, -1.0, -1.0]   # domain of x = (relative time, x, y, z)
X_HIGH = [0.0, 1.0, 1.0, 1.0]

SENSOR_COUNT = 20
SENSOR_FILE = Path(__file__).with_name("SWE_sensors_coordinates.pt")
TARGET_POINTS = 4096
HISTORY_OPTIONS = (10, 20, 50)     # possible numbers of past sensor readings (lag)
STRIDE = 5                         # time steps between consecutive readings
DROP_SENSOR_OPTIONS = (0, 2, 5, 10)


# ----------------------------------------------------------------------------- loading

def _download(base_path: Path, dataset_name: str) -> None:
    downloader = shutil.which("the-well-download")
    command = ([downloader] if downloader else
               [sys.executable, "-c", "from the_well.utils.download import main; main()"])
    base_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        print(f"[The Well] Downloading {dataset_name}/{split}", flush=True)
        subprocess.run(command + ["--base-path", str(base_path), "--dataset", dataset_name,
                                  "--split", split], check=True)


def _load_split(base_path: Path, dataset_name: str, split: str) -> Tuple[Tensor, Tensor]:
    """All trajectories of a split, (samples, time, points, 3), and the (theta, phi) grid."""
    try:
        from the_well.data import WellDataset
    except ImportError as error:
        raise ImportError("Install The Well with `pip install the_well` to load SWE data.") from error

    root = base_path / "datasets" if (base_path / "datasets").is_dir() else base_path
    dataset = WellDataset(well_base_path=str(root), well_dataset_name=dataset_name,
                          well_split_name=split, n_steps_input=1008, n_steps_output=0,
                          use_normalization=False, return_grid=True, flatten_tensors=True)
    grid = torch.as_tensor(dataset[0]["space_grid"], dtype=torch.float32).reshape(-1, 2)
    fields = torch.as_tensor(np.asarray([sample["input_fields"] for sample in dataset]),
                             dtype=torch.float32)
    if fields.ndim == 5:  # (samples, time, Lx, Ly, channels) -> (samples, time, points, channels)
        fields = fields.flatten(2, 3)
    if fields.ndim != 4 or fields.shape[-1] != 3:
        raise ValueError(f"expected trajectories of shape (samples, time, points, 3), got {tuple(fields.shape)}")
    return fields, grid


def prepare_data(base_path, dataset_name: str = "planetswe", splits: Sequence[str] = ("train", "valid"),
                 download: bool = False) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor]:
    """Loads the requested splits into memory, standardized per channel.

    The training-set mean and std are computed once and cached next to the data
    (<dataset_name>_channel_stats.pt). Returns (trajectories per split, coordinates, mean, std);
    mean and std (3,) convert predictions back to physical units.
    """
    base_path = Path(base_path)
    if download:
        _download(base_path, dataset_name)

    stats_path = base_path / f"{dataset_name}_channel_stats.pt"
    mean = std = None
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
        mean, std = stats["mean"].flatten(), stats["std"].flatten()

    to_load = list(splits)
    if mean is None and "train" not in to_load:
        print("[SWE] No cached statistics: loading 'train' once to compute them.")
        to_load.insert(0, "train")

    trajectories, coordinates = {}, None
    for split in to_load:
        fields, grid = _load_split(base_path, dataset_name, split)
        if mean is None:
            mean = fields.mean(dim=(0, 1, 2))
            std = fields.std(dim=(0, 1, 2)).clamp(min=1e-8)
            torch.save({"mean": mean, "std": std}, stats_path)
        fields.sub_(mean).div_(std)  # in place: no second full-size copy
        if split in splits:
            trajectories[split] = fields
        coordinates = grid if coordinates is None else coordinates
    return trajectories, coordinates, mean, std


# ----------------------------------------------------------------------------- sensors

def spherical_to_cartesian(coordinates: Tensor) -> Tensor:
    """(theta = colatitude, phi = longitude) in radians -> (x, y, z) on the unit sphere.

    Removes the longitude wrap-around and the pole degeneracy of raw angles, and makes
    Euclidean distance follow great-circle distance.
    """
    theta, phi = coordinates[..., 0], coordinates[..., 1]
    return torch.stack([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()], dim=-1)


def _spread_sensors(coordinates: Tensor, count: int) -> Tensor:
    """Grid points closest to `count` points spread evenly over the sphere (Fibonacci lattice)."""
    i = torch.arange(count, dtype=coordinates.dtype)
    z = 1.0 - 2.0 * (i + 0.5) / count
    phi = i * torch.pi * (3.0 - 5.0 ** 0.5)
    radius = (1.0 - z ** 2).clamp(min=0).sqrt()
    points = torch.stack([radius * phi.cos(), radius * phi.sin(), z], dim=-1)
    sensors = (points @ spherical_to_cartesian(coordinates).T).argmax(dim=1)
    if sensors.unique().numel() != count:
        raise ValueError("could not place the sensors on distinct grid points")
    return sensors.sort().values


def load_sensors(coordinates: Tensor, path: Path = SENSOR_FILE, count: int = SENSOR_COUNT) -> Tensor:
    """Sensor grid indices from `path` (flat indices or (row, column) pairs), or evenly
    spread sensors if the file is missing or holds a different number of sensors."""
    path = Path(path)
    if not path.exists():
        print(f"[SWE] No sensor file at {path}: spreading {count} sensors evenly.")
        return _spread_sensors(coordinates, count)

    saved = torch.as_tensor(torch.load(path, map_location="cpu", weights_only=False)).long()
    if saved.ndim == 2 and saved.shape[1] == 2:  # (row, column) -> flat index
        n_columns = torch.unique(coordinates[:, 1]).numel()
        if not ((0 <= saved[:, 1]) & (saved[:, 1] < n_columns)).all():
            raise ValueError(f"sensor columns in {path} must be in [0, {n_columns})")
        saved = saved[:, 0] * n_columns + saved[:, 1]
    if saved.ndim != 1 or saved.unique().numel() != saved.numel() or not (0 <= saved).all() \
            or not (saved < len(coordinates)).all():
        raise ValueError(f"{path} must hold unique sensor indices inside the grid")
    if saved.numel() != count:
        print(f"[SWE] {path} has {saved.numel()} sensors, {count} requested: spreading them evenly.")
        return _spread_sensors(coordinates, count)
    print(f"[SWE] Loaded {count} sensors from {path}.")
    return saved


# ----------------------------------------------------------------------------- tasks

class _Windows(Dataset):
    """One item per (trajectory, time t0): the max(HISTORY_OPTIONS) + 1 readings ending at t0."""

    def __init__(self, trajectories: Tensor):
        self.trajectories = trajectories
        self.span = max(HISTORY_OPTIONS) * STRIDE
        self.n_times = trajectories.shape[1] - self.span - 1
        if self.n_times <= 0:
            raise ValueError("trajectories are too short for the largest history option")

    def __len__(self) -> int:
        return self.trajectories.shape[0] * self.n_times

    def __getitem__(self, index: int) -> Tensor:
        trajectory, t = divmod(index, self.n_times)
        t0 = t + self.span + 1
        return self.trajectories[trajectory, t0 - self.span: t0 + 1: STRIDE]


def _collate(windows: List[Tensor], coordinates: Tensor, sensors: Tensor) -> Dict[str, Tensor]:
    """Builds a batch of tasks; the lag, dropped sensors and targets are shared within a batch."""
    windows = torch.stack(windows).float()          # (B, readings, points, 3)
    B, n_readings, n_points, _ = windows.shape
    last = n_readings - 1

    lag = HISTORY_OPTIONS[torch.randint(len(HISTORY_OPTIONS), ()).item()]
    n_drop = DROP_SENSOR_OPTIONS[torch.randint(len(DROP_SENSOR_OPTIONS), ()).item()]
    active = sensors[torch.randperm(len(sensors))[n_drop:]] if n_drop else sensors

    # Targets at t0; cells shrink towards the poles as sin(theta), so weight by area.
    if TARGET_POINTS >= n_points:
        targets = torch.arange(n_points)
    else:
        area = torch.sin(coordinates[:, 0]).clamp(min=1e-6)
        targets = torch.multinomial(area, TARGET_POINTS, replacement=False)

    offsets = torch.arange(-lag, 1)                                   # readings before t0, and t0
    readings = (last + offsets).repeat_interleave(len(active))       # (N,)
    points = active.repeat(lag + 1)                                   # (N,)
    time_c = (offsets.float() / last).repeat_interleave(len(active)).unsqueeze(-1)

    x_context = torch.cat([time_c, spherical_to_cartesian(coordinates[points])], dim=-1)
    x_target = torch.cat([torch.zeros(len(targets), 1), spherical_to_cartesian(coordinates[targets])], dim=-1)
    return {
        "x_context": x_context.expand(B, -1, -1).contiguous(),
        "y_context": windows[:, readings, points],
        "x_target": x_target.expand(B, -1, -1).contiguous(),
        "y_target": windows[:, last, targets],
    }


def make_train_loader(trajectories: Tensor, coordinates: Tensor, sensors: Tensor, steps_per_epoch: int,
                      batch_size: int = 16, num_workers: int = 4) -> DataLoader:
    """An "epoch" is steps_per_epoch batches of tasks, sampled with replacement.

    num_workers: on Linux, worker processes share the trajectories in memory for free. On Windows
    and macOS, workers start as new processes and the whole training set has to be handed over to
    them, which for PlanetSWE can be slow and memory-hungry: if the first epoch is slow to start
    or memory runs out, use num_workers=0.
    """
    windows = _Windows(trajectories)
    return DataLoader(
        windows, batch_size=batch_size,
        sampler=RandomSampler(windows, replacement=True, num_samples=steps_per_epoch * batch_size),
        collate_fn=partial(_collate, coordinates=coordinates, sensors=sensors),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_validation_batches(trajectories: Tensor, coordinates: Tensor, sensors: Tensor, n_batches: int,
                            batch_size: int = 16, seed: int = 0) -> List[Dict[str, Tensor]]:
    """A fixed list of validation batches, drawn once with `seed`.

    Every epoch is then validated on exactly the same tasks, so changes in the
    validation metric come from the model, not from which tasks happened to be drawn.
    """
    windows = _Windows(trajectories)
    generator = torch.Generator().manual_seed(seed)
    collate = partial(_collate, coordinates=coordinates, sensors=sensors)
    state = torch.get_rng_state()
    torch.manual_seed(seed)  # _collate draws from the global generator
    try:
        indices = torch.randint(len(windows), (n_batches, batch_size), generator=generator)
        return [collate([windows[i] for i in batch.tolist()]) for batch in indices]
    finally:
        torch.set_rng_state(state)