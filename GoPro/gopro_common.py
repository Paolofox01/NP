from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from architectures.Fourier import LearnableFourierFeatures


DEFAULT_HISTORY_LENGTH = 40
DEFAULT_STRIDE = 3
DEFAULT_SENSOR_COORDS = ((287, 464), (110, 256), (297, 256))


class TimestepHistoryDataset(Dataset):
    """Valid target times from one video, retaining a fixed history window."""

    def __init__(self, data: torch.Tensor, indexes: np.ndarray, history_length: int, stride: int):
        self.data = data.float()
        self.indexes = np.sort(indexes)
        self.history_length = history_length
        self.stride = stride

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target_time = int(self.indexes[index])
        times = torch.arange(
            target_time - self.history_length * self.stride,
            target_time + 1,
            self.stride,
        )
        return times, self.data[times]


def load_videos(data_dir: Path) -> tuple[torch.Tensor, int, int, int, int, torch.Tensor, torch.Tensor]:
    """Load, clip, normalize, and flatten the two grayscale GIF videos."""
    frames = []
    for filename in ("GoPro_video1.gif", "GoPro_video2.gif"):
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing GoPro input video: {path}")
        with Image.open(path) as gif:
            video = []
            try:
                while True:
                    video.append(torch.from_numpy(np.asarray(gif.convert("L"), dtype=np.float32) / 255.0))
                    gif.seek(gif.tell() + 1)
            except EOFError:
                pass
        frames.append(torch.stack(video))

    if frames[0].shape != frames[1].shape:
        raise ValueError(f"GoPro videos must have equal dimensions, got {frames[0].shape} and {frames[1].shape}.")

    videos = torch.stack(frames)
    nframes, height, width = videos.shape[1:]
    source_height, source_width = height, width
    
    # Optional: Downsample video resolution to speed up training
    # Set scale_factor to 0.5 for 50% resolution, 0.33 for 33%, etc.
    scale_factor = 0.5  # Change this to downsample (e.g., 0.5, 0.33, 0.25)
    if scale_factor < 1.0:
        videos_flat = videos.view(2 * nframes, 1, height, width)
        videos_flat = F.interpolate(videos_flat, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        videos = videos_flat.view(2, nframes, int(height * scale_factor), int(width * scale_factor))
        height, width = int(height * scale_factor), int(width * scale_factor)
        print(f"[GoPro] Downsampled video to {height}x{width}")
    
    videos = videos.clamp(max=0.5)
    data_min, data_max = videos.min(), videos.max()
    normalized = (videos - data_min) / (data_max - data_min + 1e-8)
    return normalized.view(2, nframes, height * width), height, width, source_height, source_width, data_min, data_max


def build_splits(videos: torch.Tensor, history_length: int = DEFAULT_HISTORY_LENGTH, stride: int = DEFAULT_STRIDE):
    """Use the same per-video temporal split convention as the original GoPro scripts."""
    first_valid = history_length * stride
    candidates = np.arange(first_valid, videos.shape[1])
    if len(candidates) < 4:
        raise ValueError("Videos are too short for the requested GoPro history window.")

    rng = np.random.default_rng(0)
    datasets = {"train": [], "val": [], "test": []}
    for video in videos:
        train_indices = rng.choice(candidates, size=round(0.8 * len(candidates)), replace=False)
        remaining = np.setdiff1d(candidates, train_indices)
        datasets["train"].append(TimestepHistoryDataset(video, train_indices, history_length, stride))
        datasets["val"].append(TimestepHistoryDataset(video, remaining[::2], history_length, stride))
        datasets["test"].append(TimestepHistoryDataset(video, remaining[1::2], history_length, stride))
    return {name: ConcatDataset(items) for name, items in datasets.items()}


def build_coordinates(
    height: int,
    width: int,
    source_height: int | None = None,
    source_width: int | None = None,
) -> tuple[torch.Tensor, list[int]]:
    row, col = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    coords = torch.stack([row, col], dim=-1).reshape(-1, 2).float()
    coords[:, 0] = 2 * coords[:, 0] / max(height - 1, 1) - 1
    coords[:, 1] = 2 * coords[:, 1] / max(width - 1, 1) - 1
    source_height = source_height or height
    source_width = source_width or width
    scaled_sensor_coords = [
        (
            round(r * max(height - 1, 1) / max(source_height - 1, 1)),
            round(c * max(width - 1, 1) / max(source_width - 1, 1)),
        )
        for r, c in DEFAULT_SENSOR_COORDS
    ]
    if any(r < 0 or r >= height or c < 0 or c >= width for r, c in scaled_sensor_coords):
        raise ValueError("Configured GoPro sensor coordinates exceed the video dimensions.")
    sensors = [r * width + c for r, c in scaled_sensor_coords]
    return coords, sensors


def prepare_data(data_dir: Path):
    videos, height, width, source_height, source_width, data_min, data_max = load_videos(data_dir)
    coordinates, sensors = build_coordinates(height, width, source_height, source_width)
    return build_splits(videos), coordinates, sensors, videos.shape[1], height, width, data_min, data_max


def _stack_batch(batch):
    return torch.stack([item[1] for item in batch])


def build_epoch_target_indices(nstate: int, fixed_sensor_locations: list[int], num_target: int = 128):
    """Pre-generate fixed target indices for each epoch to balance speed vs. diversity.
    
    Call this once per epoch to get epoch-specific targets.
    Targets change per epoch (spatial diversity) but stay fixed within epoch (fast collate).
    """
    sensor_indices = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    all_indices = torch.arange(nstate)
    extra_mask = torch.ones(nstate, dtype=torch.bool)
    extra_mask[sensor_indices] = False
    
    # Shuffle extra indices for this epoch
    extra_indices_all = all_indices[extra_mask]
    perm = torch.randperm(len(extra_indices_all))
    extra_indices = extra_indices_all[perm][:num_target]
    
    target_indices = torch.cat([sensor_indices, extra_indices])
    return target_indices


# Global state for per-epoch randomization
_GLOBAL_TARGET_INDICES = None
_GLOBAL_HISTORY_LENGTH = 20


def set_epoch_targets(nstate: int, fixed_sensor_locations: list[int], num_target: int = 128, history_options: tuple = (10, 20, 30, 40)):
    """Call this at the start of each epoch to update target indices and history length.
    
    Both are randomized once per epoch, then fixed for all batches in that epoch.
    """
    global _GLOBAL_TARGET_INDICES, _GLOBAL_HISTORY_LENGTH
    _GLOBAL_TARGET_INDICES = build_epoch_target_indices(nstate, fixed_sensor_locations, num_target)
    _GLOBAL_HISTORY_LENGTH = int(history_options[torch.randint(len(history_options), ()).item()])


def np_collate_fn(
    batch,
    mesh_coords: torch.Tensor,
    fixed_sensor_locations: list[int],
    num_target: int = 128,
    history: int = 20,
):
    """Minimal collate: Uses epoch-specific fixed targets and history (set via set_epoch_targets).
    
    Pre-computing per-epoch reduces CPU overhead significantly:
    - No per-batch random sampling (still per-epoch randomization)
    - No history_options selection per batch
    - Direct indexing only
    """
    global _GLOBAL_TARGET_INDICES, _GLOBAL_HISTORY_LENGTH
    
    windows = _stack_batch(batch)
    batch_size, max_history_plus_one, nstate = windows.shape
    
    # Use global history length (set once per epoch)
    if _GLOBAL_HISTORY_LENGTH is None:
        _GLOBAL_HISTORY_LENGTH = 20
    history = _GLOBAL_HISTORY_LENGTH
    
    if history >= max_history_plus_one:
        raise ValueError("Requested GoPro history exceeds the dataset window.")
    
    sensor_indices = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    
    # Use global target indices (set once per epoch)
    if _GLOBAL_TARGET_INDICES is None:
        _GLOBAL_TARGET_INDICES = build_epoch_target_indices(nstate, fixed_sensor_locations, num_target)
    target_indices = _GLOBAL_TARGET_INDICES

    context_values = windows[:, -(history + 1):, sensor_indices]
    offsets = torch.linspace(-1.0, 0.0, history + 1).repeat_interleave(len(sensor_indices)).unsqueeze(-1)
    context_coords = mesh_coords[sensor_indices].repeat(history + 1, 1)
    x_context = torch.cat([offsets, context_coords], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
    y_context = context_values.reshape(batch_size, -1, 1)
    x_target = torch.cat([torch.zeros((len(target_indices), 1)), mesh_coords[target_indices]], dim=-1)
    x_target = x_target.unsqueeze(0).expand(batch_size, -1, -1)
    y_target = windows[:, -1, target_indices].unsqueeze(-1)
    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()


def np_eval_inputs(batch, mesh_coords: torch.Tensor, fixed_sensor_locations: list[int], history: int = DEFAULT_HISTORY_LENGTH):
    windows = _stack_batch(batch)
    batch_size, max_history_plus_one, nstate = windows.shape
    if history >= max_history_plus_one:
        raise ValueError("Requested GoPro history exceeds the dataset window.")
    sensors = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    values = windows[:, -(history + 1):, sensors]
    offsets = torch.linspace(-1.0, 0.0, history + 1).repeat_interleave(len(sensors)).unsqueeze(-1)
    context_coords = mesh_coords[sensors].repeat(history + 1, 1)
    x_context = torch.cat([offsets, context_coords], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
    y_context = values.reshape(batch_size, -1, 1)
    all_indices = torch.arange(nstate)
    x_target = torch.cat([torch.zeros((nstate, 1)), mesh_coords], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
    y_target = windows[:, -1, all_indices].unsqueeze(-1)
    return x_context, y_context, x_target, y_target


def don_collate_fn(batch, mesh_coords: torch.Tensor, fixed_sensor_locations: list[int], points_per_batch: int = 2048):
    """Create fixed-width `(sensor_1, sensor_2, sensor_3, time)` LSTM histories."""
    windows = _stack_batch(batch)
    batch_size, history_plus_one, nstate = windows.shape
    sensors = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    sensor_values = windows[:, :, sensors]
    relative_time = torch.linspace(-1.0, 0.0, history_plus_one, dtype=windows.dtype).view(1, -1, 1)
    sensor_history = torch.cat([sensor_values, relative_time.expand(batch_size, -1, -1)], dim=-1)
    points = torch.randint(nstate, (batch_size, points_per_batch))
    coords = torch.cat([torch.zeros((batch_size, points_per_batch, 1)), mesh_coords[points]], dim=-1)
    targets = torch.gather(windows[:, -1], 1, points)
    return sensor_history, coords, targets


def don_eval_inputs(batch, mesh_coords: torch.Tensor, fixed_sensor_locations: list[int]):
    windows = _stack_batch(batch)
    batch_size, history_plus_one, nstate = windows.shape
    sensors = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    sensor_values = windows[:, :, sensors]
    relative_time = torch.linspace(-1.0, 0.0, history_plus_one, dtype=windows.dtype).view(1, -1, 1)
    sensor_history = torch.cat([sensor_values, relative_time.expand(batch_size, -1, -1)], dim=-1)
    coords = torch.cat([torch.zeros((nstate, 1)), mesh_coords], dim=-1).unsqueeze(0).expand(batch_size, -1, -1)
    return sensor_history, coords, windows[:, -1]


class DeepONetDeterministic(nn.Module):
    def __init__(self, num_sensors: int, coord_dim: int = 3, p: int = 128, num_frequencies: int = 32):
        super().__init__()
        self.p = p
        self.branch_lstm = nn.LSTM(num_sensors + 1, 256, num_layers=2, batch_first=True, dropout=0.1)
        self.branch_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p))
        self.fourier = LearnableFourierFeatures(coord_dim, num_frequencies, init_scale=1.0)
        self.trunk = nn.Sequential(nn.Linear(coord_dim + 2 * num_frequencies, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, p))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensor_history: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.branch_lstm(sensor_history)
        branch = self.branch_head(hidden[-1]).unsqueeze(1)
        trunk = self.trunk(torch.cat([coords, self.fourier(coords)], dim=-1))
        return (branch * trunk).sum(dim=-1) / math.sqrt(self.p) + self.bias


class DeepONetMeanVar(nn.Module):
    def __init__(self, num_sensors: int, coord_dim: int = 3, p: int = 128, num_frequencies: int = 32):
        super().__init__()
        self.p = p
        self.branch_lstm = nn.LSTM(num_sensors + 1, 256, num_layers=2, batch_first=True, dropout=0.1)
        self.branch_mean = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p))
        self.branch_var = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p))
        self.fourier = LearnableFourierFeatures(coord_dim, num_frequencies, init_scale=1.0)
        self.trunk_base = nn.Sequential(nn.Linear(coord_dim + 2 * num_frequencies, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.trunk_mean = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p))
        self.trunk_var = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p))
        self.mean_bias = nn.Parameter(torch.zeros(1))
        self.var_bias = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, sensor_history: torch.Tensor, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.branch_lstm(sensor_history)
        features = self.trunk_base(torch.cat([coords, self.fourier(coords)], dim=-1))
        mean = (self.branch_mean(hidden[-1]).unsqueeze(1) * self.trunk_mean(features)).sum(dim=-1) / math.sqrt(self.p) + self.mean_bias
        raw_var = (self.branch_var(hidden[-1]).unsqueeze(1) * self.trunk_var(features)).sum(dim=-1) / math.sqrt(self.p) + self.var_bias
        return mean, F.softplus(raw_var) + 1e-6


def gaussian_nll(mean: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return 0.5 * (math.log(2.0 * math.pi) + variance.log() + (target - mean).square() / variance).mean()


def make_np_loaders(datasets, coords, sensors, batch_size: int = 32, num_workers: int = 0):
    collate = partial(np_collate_fn, mesh_coords=coords, fixed_sensor_locations=sensors)
    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate, 
                  num_workers=num_workers, pin_memory=True),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate,
                  num_workers=num_workers, pin_memory=True),
    )


def make_don_loaders(datasets, coords, sensors, batch_size: int = 32, num_workers: int = 0):
    collate = partial(don_collate_fn, mesh_coords=coords, fixed_sensor_locations=sensors)
    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate,
                  num_workers=num_workers, pin_memory=True),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, collate_fn=collate,
                  num_workers=num_workers, pin_memory=True),
    )