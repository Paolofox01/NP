"""PlanetSWE (The Well) as Neural Process tasks.

A task is one trajectory at one time t0:
- context: the sensors' readings at the lag + 1 sampled times up to t0 (a few sensors randomly dropped);
- targets: grid points at t0, sampled in proportion to their cell area.

Random tasks (training, validation, the random-task test set) are drawn independently for every task in
a batch: t0, lag (HISTORY_OPTIONS), which sensors are dropped, how many targets (TARGET_RANGE) and which.
Tasks of different sizes are padded to the largest one in the batch, with context_mask / target_mask
(True = padding) marking the padded points. This is the same task distribution as pinball/pinball_data.py.

x = (relative time, x, y, z): relative time in [-1, 0] (0 = t0), position on the unit sphere.
y = (h, u_theta, u_phi), standardized per channel with training-set statistics.

Batches are the dictionaries training/train_np.py expects, plus the sensor-history inputs of the
baselines (models/baselines.py), which the Neural Processes ignore.

Two ways to get the data (both give a SWEData):
- prepare_data: the full PlanetSWE grid, from The Well's files on disk, all in memory (about 190 GB).
- build_cache + load_cache: the data is streamed once from the Hugging Face Hub, reduced to the
  sensors plus a fixed pool of grid points on the times t0 can take, and saved (a few GB). The tasks
  are then drawn from that pool. This is what fits on Colab or Kaggle.
"""

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler

__all__ = ["VARIABLES", "X_LOW", "X_HIGH", "HISTORY_OPTIONS", "TARGET_RANGE", "SWEData", "prepare_data",
           "load_sensors", "spherical_to_cartesian", "make_train_loader", "make_random_batches",
           "fixed_targets", "make_test_tasks", "build_cache", "load_cache"]

VARIABLES = ["h", "u_theta", "u_phi"]
X_LOW = [-1.0, -1.0, -1.0, -1.0]   # domain of x = (relative time, x, y, z)
X_HIGH = [0.0, 1.0, 1.0, 1.0]

SENSOR_COUNT = 20
SENSOR_FILE = Path(__file__).with_name("SWE_sensors_coordinates.pt")
TARGET_RANGE = (2048, 4096)        # random target points per task
HISTORY_OPTIONS = (10, 20, 50)     # possible numbers of past sensor readings (lag)
STRIDE = 5                         # time steps between consecutive readings
DROP_SENSOR_OPTIONS = (0, 2, 5, 10)


@dataclass
class SWEData:
    """Trajectories and everything the tasks need to be drawn from them."""

    trajectories: Dict[str, Tensor]   # split -> (trajectories, times, points, 3), standardized
    coordinates: Tensor               # (points, 2): (theta, phi) of each point
    y_mean: Tensor                    # (3,), to convert predictions back to physical units
    y_std: Tensor                     # (3,)
    sensors: Tensor                   # (S,) indices of the sensor points
    stride: int                       # steps of the stored time axis between consecutive readings of a window
    target_weights: Tensor            # (points,) relative probability of each point as a target


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
                 download: bool = False) -> SWEData:
    """Loads the requested splits into memory, standardized per channel.

    The training-set mean and std are computed once and cached next to the data
    (<dataset_name>_channel_stats.pt). Returns a SWEData; its y_mean and y_std (3,) convert predictions
    back to physical units.
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
    return SWEData(trajectories, coordinates, mean, std, load_sensors(coordinates), STRIDE,
                   _cell_area(coordinates))


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
    """One item per (trajectory, time t0): the max(HISTORY_OPTIONS) + 1 readings ending at t0, `stride`
    steps of the stored time axis apart."""

    def __init__(self, trajectories: Tensor, stride: int = STRIDE):
        self.trajectories, self.stride = trajectories, stride
        self.span = max(HISTORY_OPTIONS) * stride
        self.n_times = trajectories.shape[1] - self.span - 1
        if self.n_times <= 0:
            raise ValueError("trajectories are too short for the largest history option")

    def __len__(self) -> int:
        return self.trajectories.shape[0] * self.n_times

    def __getitem__(self, index: int) -> Tensor:
        trajectory, t = divmod(index, self.n_times)
        t0 = t + self.span + 1
        return self.trajectories[trajectory, t0 - self.span: t0 + 1: self.stride]   # a view, not a copy


def _cell_area(coordinates: Tensor) -> Tensor:
    """Cells shrink towards the poles as sin(theta): the weight of a point as a target."""
    return torch.sin(coordinates[:, 0]).clamp(min=1e-6)


def _build_batch(windows: List[Tensor], coordinates: Tensor, sensors: Tensor, lags: List[int],
                 active: Tensor, targets: List[Tensor]) -> Dict[str, Tensor]:
    """One task per window, each with its own lag, active sensors and targets.

    windows: B tensors (readings, points, 3), the last reading being t0; lags: B ints (past readings used);
    active: (B, S) bool; targets: B index tensors (of different lengths).

    The context is laid out on a (reading, sensor) grid of L = max(lag) + 1 rows ending at t0, so the
    readings of a task are its last lag + 1 rows; the rest, and dropped sensors, are padding.
    Besides the Neural Process inputs (x/y_context, x/y_target and the two masks), the task has what
    the baselines need:
      sensor_history (B, L, S, 3): the readings on that grid, 0 at padding;
      sensor_on (B, L, S): 1 for a real reading, 0 for padding (dropped sensor or before the lag);
      sensor_time (B, L, 1): relative time of each row;
      sensor_position (B, S, 3): position of each sensor slot (also for dropped sensors);
      target_index (B, T): grid point of each target (padding: the first target).
    """
    B, S = len(windows), len(sensors)
    last = windows[0].shape[0] - 1
    L = max(lags) + 1
    offset = torch.arange(-(L - 1), 1)                                                 # (L,)
    on = (offset[None, :] >= -torch.as_tensor(lags)[:, None])[:, :, None] & active[:, None, :]   # (B, L, S)
    rel_time = offset.float() / last                                                   # (L,)
    history = torch.stack([w[last - L + 1:][:, sensors] for w in windows])             # (B, L, S, 3)

    sensor_xyz = spherical_to_cartesian(coordinates[sensors])                          # (S, 3)
    x_context = torch.cat([rel_time.view(1, L, 1, 1).expand(B, L, S, 1),
                           sensor_xyz.view(1, 1, S, 3).expand(B, L, S, 3)], dim=-1)

    n_targets = torch.as_tensor([len(t) for t in targets])
    T = int(n_targets.max())
    index = torch.stack([F.pad(t, (0, T - len(t)), value=int(t[0])) for t in targets])  # (B, T)
    target_mask = torch.arange(T)[None, :] >= n_targets[:, None]                       # (B, T)
    x_target = torch.cat([torch.zeros(B, T, 1), spherical_to_cartesian(coordinates[index])], dim=-1)
    y_target = torch.stack([w[last][i] for w, i in zip(windows, index)])               # (B, T, 3)

    return {
        "x_context": x_context.reshape(B, L * S, -1),
        "y_context": history.reshape(B, L * S, -1),
        "context_mask": ~on.reshape(B, L * S),
        "x_target": x_target,
        "y_target": y_target,
        "target_mask": target_mask,
        "sensor_history": history * on.unsqueeze(-1),
        "sensor_on": on.float(),
        "sensor_time": rel_time.view(1, L, 1).expand(B, L, 1).contiguous(),
        "sensor_position": sensor_xyz.unsqueeze(0).expand(B, -1, -1).contiguous(),
        "target_index": index,
    }


def _random_tasks(windows: List[Tensor], coordinates: Tensor, sensors: Tensor, weights: Tensor,
                  target_range: Tuple[int, int] = TARGET_RANGE) -> Dict[str, Tensor]:
    """Random tasks, drawn independently for every window: lag, dropped sensors, number of targets
    (uniform in target_range) and which points (without replacement, with probability `weights`)."""
    B, S = len(windows), len(sensors)
    n_available = int((weights > 0).sum())
    lags = [HISTORY_OPTIONS[i] for i in torch.randint(len(HISTORY_OPTIONS), (B,)).tolist()]

    drop_options = torch.as_tensor(DROP_SENSOR_OPTIONS)
    n_drop = drop_options[torch.randint(len(drop_options), (B,))]
    rank = torch.rand(B, S).argsort(dim=1).argsort(dim=1)                              # random order
    active = rank >= n_drop[:, None]                                                   # (B, S)

    n_targets = torch.randint(target_range[0], target_range[1] + 1, (B,)).clamp(max=n_available)
    targets = [torch.multinomial(weights, int(n), replacement=False) for n in n_targets]
    return _build_batch(windows, coordinates, sensors, lags, active, targets)


def make_train_loader(data: SWEData, steps_per_epoch: int, batch_size: int = 16, num_workers: int = 4,
                      target_range: Tuple[int, int] = TARGET_RANGE) -> DataLoader:
    """An "epoch" is steps_per_epoch batches of tasks on the training trajectories, sampled with replacement.

    num_workers: on Linux, worker processes share the trajectories in memory for free. On Windows
    and macOS, workers start as new processes and the whole training set has to be handed over to
    them, which for PlanetSWE can be slow and memory-hungry: if the first epoch is slow to start
    or memory runs out, use num_workers=0.
    """
    windows = _Windows(data.trajectories["train"], data.stride)
    return DataLoader(
        windows, batch_size=batch_size,
        sampler=RandomSampler(windows, replacement=True, num_samples=steps_per_epoch * batch_size),
        collate_fn=partial(_random_tasks, coordinates=data.coordinates, sensors=data.sensors,
                           weights=data.target_weights, target_range=target_range),
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def make_random_batches(data: SWEData, split: str, n_batches: int, batch_size: int = 16, seed: int = 0,
                        target_range: Tuple[int, int] = TARGET_RANGE) -> List[Dict[str, Tensor]]:
    """A fixed list of batches of random tasks on one split, drawn once with `seed` (same distribution as
    training).

    Used for validation and for the random-task test set: every evaluation then runs on exactly the
    same tasks, so changes in a metric come from the model, not from which tasks happened to be drawn.
    """
    windows = _Windows(data.trajectories[split], data.stride)
    generator = torch.Generator().manual_seed(seed)
    state = torch.get_rng_state()
    torch.manual_seed(seed)  # the tasks draw from the global generator
    try:
        indices = torch.randint(len(windows), (n_batches, batch_size), generator=generator)
        return [_random_tasks([windows[i] for i in batch.tolist()], data.coordinates, data.sensors,
                              data.target_weights, target_range) for batch in indices]
    finally:
        torch.set_rng_state(state)


def fixed_targets(data: SWEData, n_points: int, seed: int = 0) -> Tensor:
    """A fixed set of points (without replacement, with the target probabilities) for the fixed-time
    test: the whole grid is too large for a Neural Process to predict with many latent samples."""
    generator = torch.Generator().manual_seed(seed)
    n_points = min(n_points, int((data.target_weights > 0).sum()))
    return torch.multinomial(data.target_weights, n_points, replacement=False, generator=generator)


def make_test_tasks(data: SWEData, lag: int, targets: Tensor) -> List[Dict[str, Tensor]]:
    """Given lag and targets: one task per test trajectory, at the last time of the trajectory, with all
    the sensors active (each task is a batch of one: no padding)."""
    windows = _Windows(data.trajectories["test"], data.stride)
    all_active = torch.ones(1, len(data.sensors), dtype=torch.bool)
    return [_build_batch([windows[k * windows.n_times + windows.n_times - 1]], data.coordinates, data.sensors,
                         [lag], all_active, [targets]) for k in range(data.trajectories["test"].shape[0])]


# ----------------------------------------------------------------------------- streamed cache

HF_BASE = "hf://datasets/polymathic-ai/"
CACHE_META = "meta.pt"


class _RangeReader:
    """Reads byte ranges of a (possibly remote) file, reopening it and retrying if a request fails."""

    def __init__(self, fs, path: str, retries: int = 5):
        self.fs, self.path, self.retries, self.file = fs, path, retries, None

    def _open(self):
        try:
            return self.fs.open(self.path, "rb", cache_type="none")   # exact ranges, no read-ahead
        except TypeError:
            return self.fs.open(self.path, "rb")

    def read(self, start: int, size: int) -> bytes:
        for attempt in range(self.retries):
            try:
                if self.file is None:
                    self.file = self._open()
                self.file.seek(start)
                data = self.file.read(size)
                if len(data) != size:
                    raise IOError(f"short read: {len(data)} of {size} bytes")
                return data
            except Exception:
                if self.file is not None:
                    self.file.close()
                    self.file = None
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)

    def close(self):
        if self.file is not None:
            self.file.close()


def _read_trajectory(fs, path: str, frames: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The values (h, u_theta, u_phi) of the trajectory file `path` at the time steps `frames` and the flat
    grid indices `points`: (len(frames), len(points), 3) float32.

    The files store every field as one contiguous float32 block (no chunks, no compression), so each frame
    is one byte range of the file: only the frames asked for are transferred, not the whole file. If a file
    is laid out differently, it is read through h5py instead (much slower over the network).
    """
    import h5py

    with fs.open(path, "rb", block_size=2 ** 20) as f, h5py.File(f, "r") as h:
        height, velocity = h["t0_fields/height"], h["t1_fields/velocity"]
        n_times, n_theta, n_phi = height.shape[1:]
        if frames.max() >= n_times:
            raise ValueError(f"{path} has {n_times} time steps, expected more than {frames.max()}")
        out = np.empty((len(frames), len(points), 3), dtype=np.float32)
        contiguous = all(d.chunks is None and d.compression is None and d.id.get_offset() is not None
                         and d.dtype == np.dtype("<f4") for d in (height, velocity))
        if not contiguous:
            for k, t in enumerate(frames):
                out[k, :, 0] = height[0, t].reshape(-1)[points]
                out[k, :, 1:] = velocity[0, t].reshape(-1, 2)[points]
            return out
        offset_h, offset_v = height.id.get_offset(), velocity.id.get_offset()

    size_h, size_v = n_theta * n_phi * 4, n_theta * n_phi * 2 * 4
    reader = _RangeReader(fs, path)
    try:
        for k, t in enumerate(frames):
            frame_h = np.frombuffer(reader.read(offset_h + int(t) * size_h, size_h), dtype="<f4")
            frame_v = np.frombuffer(reader.read(offset_v + int(t) * size_v, size_v), dtype="<f4")
            out[k, :, 0] = frame_h[points]
            out[k, :, 1:] = frame_v.reshape(-1, 2)[points]
    finally:
        reader.close()
    return out


def _list_files(fs, root: str, dataset_name: str, split: str) -> List[str]:
    files = sorted(f for f in fs.glob(f"{root.rstrip('/')}/{dataset_name}/data/{split}/*")
                   if f.endswith((".hdf5", ".h5")))
    if not files:
        raise FileNotFoundError(f"no HDF5 files in {root}/{dataset_name}/data/{split}")
    return files


def build_cache(cache_dir, base_path: str = HF_BASE, dataset_name: str = "planetswe",
                splits: Sequence[str] = ("train", "valid", "test"), files_per_split: Optional[Dict[str, int]] = None,
                pool_size: int = 8192, seed: int = 0, workers: int = 8, sensor_file: Path = SENSOR_FILE) -> Path:
    """Streams PlanetSWE once (from the Hugging Face Hub by default, or from a local copy of The Well's
    files) into a compact cache in `cache_dir`, which load_cache reads.

    Per trajectory, only the sensors and a fixed pool of `pool_size` grid points (drawn in proportion to
    their cell area, with `seed`) are kept, and only at the times t0 can take: every STRIDE-th time step,
    ending at the last one, so that all the readings of a window are among them. The cache is about
    pool_size * 12 bytes * 200 times per trajectory (20 MB for 8192 points), against 1.6 GB for a file, and
    only about a fifth of each file is transferred.

    files_per_split: use only this many trajectories of a split, evenly spaced (default: all).
    The job is resumable: finished trajectories are skipped, and settings that differ from the ones the
    cache was started with are rejected.
    """
    import fsspec
    import h5py

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.url_to_fs(str(base_path))
    listing = {}
    for split in splits:
        files = _list_files(fs, root, dataset_name, split)
        n = (files_per_split or {}).get(split)
        if n is not None and n < len(files):
            files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, n).round().astype(int))]
        listing[split] = files

    with fs.open(listing[splits[0]][0], "rb", block_size=2 ** 20) as f, h5py.File(f, "r") as h:
        theta = torch.as_tensor(h["dimensions/theta"][:], dtype=torch.float32)
        phi = torch.as_tensor(h["dimensions/phi"][:], dtype=torch.float32)
        n_times = h["t0_fields/height"].shape[1]
    coordinates = torch.stack(torch.meshgrid(theta, phi, indexing="ij"), dim=-1).reshape(-1, 2)
    sensors = load_sensors(coordinates, sensor_file)
    generator = torch.Generator().manual_seed(seed)
    pool = torch.multinomial(_cell_area(coordinates), pool_size, replacement=False, generator=generator)
    points = torch.cat([sensors, pool])
    frames = np.arange((n_times - 1) % STRIDE, n_times, STRIDE)
    meta = {"coordinates": coordinates[points], "points": points, "n_sensors": len(sensors), "pool_size": pool_size,
            "seed": seed, "frames": torch.as_tensor(frames), "n_times": n_times, "dataset": dataset_name}

    meta_path = cache_dir / CACHE_META
    if meta_path.exists():
        saved = torch.load(meta_path, map_location="cpu")
        if not (torch.equal(saved["points"], points) and torch.equal(saved["frames"], meta["frames"])):
            raise ValueError(f"{cache_dir} was started with other settings (sensors, pool size, seed or "
                             "times): use another folder, or delete this one")
    else:
        torch.save(meta, meta_path)

    jobs = []
    for split, files in listing.items():
        (cache_dir / split).mkdir(exist_ok=True)
        jobs += [(split, path, cache_dir / split / f"{Path(path).stem}.pt") for path in files]
    todo = [job for job in jobs if not job[2].exists()]
    print(f"[cache] {len(jobs)} trajectories ({len(jobs) - len(todo)} already done), {len(frames)} times each, "
          f"{len(points)} points; {workers} parallel downloads", flush=True)

    def work(split, path, out_path):
        y = _read_trajectory(fs, path, frames, points.numpy())
        tmp = out_path.with_suffix(".tmp")
        torch.save({"y": torch.from_numpy(y)}, tmp)
        tmp.replace(out_path)                                   # a file is only there once it is complete
        return split, Path(path).name

    start, failed = time.time(), []
    with ThreadPoolExecutor(max_workers=workers) as pool_executor:
        futures = {pool_executor.submit(work, *job): job for job in todo}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                split, name = future.result()
                print(f"[cache] {done}/{len(todo)} {split}/{name}  ({(time.time() - start) / 60:.1f} min)", flush=True)
            except Exception as error:
                failed.append(futures[future][1])
                print(f"[cache] FAILED {futures[future][1]}: {error!r}", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} trajectories failed: run build_cache again to retry them")
    return cache_dir


def load_cache(cache_dir, splits: Sequence[str] = ("train", "valid", "test")) -> SWEData:
    """Loads a cache written by build_cache, standardized per channel with the statistics of the training
    trajectories (over their pool points). The windows then read the stored times one after the other
    (stride 1: they are already STRIDE time steps apart), and the targets are drawn from the pool, never
    from the sensors."""
    cache_dir = Path(cache_dir)
    meta = torch.load(cache_dir / CACHE_META, map_location="cpu")
    n_sensors = meta["n_sensors"]
    trajectories = {}
    for split in splits:
        files = sorted((cache_dir / split).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"no trajectories in {cache_dir / split}: run build_cache first")
        trajectories[split] = torch.stack([torch.load(f, map_location="cpu")["y"] for f in files])
    if "train" not in trajectories:
        raise ValueError("the training split is needed for the standardization statistics")

    total, total_sq, count = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0
    for trajectory in trajectories["train"]:
        values = trajectory[:, n_sensors:].reshape(-1, 3).double()
        total += values.sum(0)
        total_sq += (values ** 2).sum(0)
        count += len(values)
    mean = total / count
    std = (total_sq / count - mean ** 2).clamp(min=1e-16).sqrt().clamp(min=1e-8)
    mean, std = mean.float(), std.float()
    for values in trajectories.values():
        values.sub_(mean).div_(std)

    n_points = meta["coordinates"].shape[0]
    weights = torch.cat([torch.zeros(n_sensors), torch.ones(n_points - n_sensors)])
    return SWEData(trajectories, meta["coordinates"], mean, std, torch.arange(n_sensors), 1, weights)
