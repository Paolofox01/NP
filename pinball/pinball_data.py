"""Fluidic pinball (FEniCS) data as Neural Process tasks.

A task is one trajectory at one time t0:
- context: the readings of the fixed sensors at the lag + 1 times t0 - lag, ..., t0;
- targets: a set of mesh nodes at t0 that are not active sensors (the context is not repeated among
  them; the model adds it itself for the posterior, with target_includes_context=False).

Random tasks (training, validation, the random-task test set) are drawn independently for every
task in a batch: t0 in MIN_TIME ... last time, lag in 0 ... MAX_LAG, which sensors are dropped,
how many targets and which. Tasks of different sizes are padded to the largest one in the batch,
with context_mask / target_mask (True = padding) marking the padded points.

x = ([mu,] relative time, x, y): relative time (t - t0) / ntimes, so 0 at t0 and negative before;
mu (the 3 flow parameters, taken at t0) only when use_mu=True.
y = the state value, standardized with the training-set mean and std.

Batches are the dictionaries training/train_np.py expects, plus the sensor-history inputs of the
baselines (models/baselines.py), which the Neural Processes ignore. The whole dataset is small
(about 350 MB of training data), so it is kept on the GPU and batches are built there directly:
no loader workers and no copies between CPU and GPU (call data.to(device) once).
"""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["PinballData", "load_pinball", "x_bounds", "TrainBatches", "make_random_batches",
           "make_validation_batches", "make_test_batch", "MAX_LAG"]

N_TRAJECTORIES = 500
N_TIMES = 31                       # T = 3.0, dt = 0.1
MAX_LAG = 19                       # up to MAX_LAG past readings (MAX_LAG + 1 readings in all)
MIN_TIME = 20                      # t0 = MIN_TIME ... N_TIMES - 1, so the longest lag fits
TARGET_RANGE = (2048, 4096)        # random target nodes per task, active sensors excluded
DROP_SENSOR_OPTIONS = (0, 1)       # how many sensors a random task drops


@dataclass
class PinballData:
    function_space: object         # FEniCS function space, for plotting fields
    coordinates: Tensor            # (nstate, 2) node coordinates
    y: Dict[str, Tensor]           # split -> (trajectories, ntimes, nstate), standardized
    mu: Dict[str, Tensor]          # split -> (trajectories, ntimes, nparams)
    y_mean: float
    y_std: float
    n_times: int = N_TIMES

    def to(self, device) -> "PinballData":
        """The same data, with every tensor moved to `device`."""
        return replace(self, coordinates=self.coordinates.to(device),
                       y={k: v.to(device) for k, v in self.y.items()},
                       mu={k: v.to(device) for k, v in self.mu.items()})


# ----------------------------------------------------------------------------- loading

def load_pinball(data_dir, mesh_file: str = "Pinball_mesh.xml",
                 data_file: str = "Pinball_data.npz", seed: int = 0) -> PinballData:
    """Loads mesh and snapshots, and splits them as the original scripts did (same seed):
    80% train, the remaining 20% alternately to validation and test."""
    from dolfin import FunctionSpace, Mesh  # FEniCS is only needed here and for plotting

    data_dir = Path(data_dir)
    mesh = Mesh(str(data_dir / mesh_file))
    space = FunctionSpace(mesh, "CG", 1)
    n_state = space.dim()
    coordinates = torch.as_tensor(space.tabulate_dof_coordinates(), dtype=torch.float32)

    data = np.load(data_dir / data_file)
    y_all = torch.as_tensor(data["y"], dtype=torch.float32)
    if y_all.numel() != N_TRAJECTORIES * N_TIMES * n_state:
        raise ValueError(f"{data_file}: y has shape {tuple(y_all.shape)}, expected {N_TRAJECTORIES} "
                         f"trajectories x {N_TIMES} times x {n_state} mesh nodes")
    y_all = y_all.reshape(N_TRAJECTORIES, N_TIMES, n_state)
    mu_all = _mu_per_time(torch.as_tensor(data["mu"], dtype=torch.float32), data_file)

    rng = np.random.RandomState(seed)
    n_train = round(0.8 * N_TRAJECTORIES)
    idx_train = rng.choice(N_TRAJECTORIES, size=n_train, replace=False)
    rest = np.setdiff1d(np.arange(N_TRAJECTORIES), idx_train)
    splits = {"train": idx_train, "valid": rest[::2], "test": rest[1::2]}

    y_mean, y_std = y_all[idx_train].mean().item(), y_all[idx_train].std().item()
    y = {name: (y_all[idx] - y_mean) / y_std for name, idx in splits.items()}
    mu = {name: mu_all[idx] for name, idx in splits.items()}
    return PinballData(space, coordinates, y, mu, y_mean, y_std)


def _mu_per_time(mu: Tensor, data_file: str) -> Tensor:
    """mu as (trajectories, times, nparams). The original scripts store it per time step (and read it
    at the target time); one value per trajectory is accepted too, and repeated over the times."""
    per_time = (mu.shape[0] == N_TRAJECTORIES * N_TIMES                         # (500 * 31, nparams)
                or (mu.ndim == 3 and mu.shape[:2] == (N_TRAJECTORIES, N_TIMES))  # (500, 31, nparams)
                or (mu.ndim == 2 and mu.shape[0] == N_TRAJECTORIES               # (500, 31 * nparams)
                    and mu.shape[1] % N_TIMES == 0))
    if per_time:
        return mu.reshape(N_TRAJECTORIES, N_TIMES, -1)
    if mu.ndim == 2 and mu.shape[0] == N_TRAJECTORIES:                           # (500, nparams)
        return mu.unsqueeze(1).expand(-1, N_TIMES, -1).contiguous()
    raise ValueError(f"{data_file}: mu has shape {tuple(mu.shape)}, expected ({N_TRAJECTORIES}, {N_TIMES}, "
                     f"nparams), ({N_TRAJECTORIES * N_TIMES}, nparams) or ({N_TRAJECTORIES}, nparams)")


def x_bounds(data: PinballData, use_mu: bool) -> Tuple[List[float], List[float]]:
    """Domain of x, for the model's input scaling: [mu,] relative time, x, y."""
    time_low = -MAX_LAG / data.n_times
    low = [time_low] + data.coordinates.min(0).values.tolist()
    high = [0.0] + data.coordinates.max(0).values.tolist()
    if use_mu:
        mu = data.mu["train"].flatten(0, 1)                     # every trajectory and time
        low = mu.min(0).values.tolist() + low
        high = mu.max(0).values.tolist() + high
    return low, high


# ----------------------------------------------------------------------------- tasks

def _build_tasks(y: Tensor, mu: Tensor, idx: Tensor, coordinates: Tensor, sensors: Tensor,
                 use_mu: bool, t0: Tensor, lag: Tensor, active: Tensor, targets: Tensor,
                 n_targets: Tensor) -> Dict[str, Tensor]:
    """One task per trajectory in idx, each with its own t0, lag, active sensors and targets.

    y, mu: one split; idx (B,): trajectories; sensors (S,): sensor nodes; t0, lag (B,): ints;
    active (B, S): bool; targets (B, T): node indices, of which the first n_targets (B,) are real.
    Everything stays on y's device.

    The context is laid out on a (time, sensor) grid of L = max(lag) + 1 rows ending at t0, so the
    readings of a task are its last lag + 1 rows; the rest, and dropped sensors, are padding.
    Besides the Neural Process inputs (x/y_context, x/y_target and the two masks), the task has
    what the baselines need:
      sensor_history (B, L, S): the readings on that grid, 0 at padding;
      sensor_on (B, L, S): 1 for a real reading, 0 for padding (dropped sensor or before the lag);
      sensor_time (B, L, 1): relative time of each row;
      sensor_position (B, S, 2): (x, y) of each sensor slot (also for dropped sensors);
      target_index (B, T): mesh node of each target (padding: the node of the first target);
      mu (B, nparams): mu at t0, only with use_mu.
    """
    device, n_times = y.device, y.shape[1]
    B, S, T = len(idx), len(sensors), targets.shape[1]
    L = int(lag.max()) + 1

    # context grid: row l is offset l - (L - 1) from t0
    offset = torch.arange(-(L - 1), 1, device=device)                                  # (L,)
    times = (t0[:, None] + offset).clamp(min=0)                                        # (B, L)
    on = (offset[None, :] >= -lag[:, None])[:, :, None] & active[:, None, :]          # (B, L, S)
    rel_time = (offset.float() / n_times)                                              # (L,)
    history = y[idx[:, None, None], times[:, :, None], sensors[None, None, :]]         # (B, L, S)

    x_context = torch.cat([rel_time.view(1, L, 1, 1).expand(B, L, S, 1),
                           coordinates[sensors].view(1, 1, S, 2).expand(B, L, S, 2)], dim=-1)
    x_target = torch.cat([torch.zeros(B, T, 1, device=device), coordinates[targets]], dim=-1)
    if use_mu:
        mu_t0 = mu[idx, t0]                                                            # (B, nparams)
        x_context = torch.cat([mu_t0.view(B, 1, 1, -1).expand(B, L, S, -1), x_context], dim=-1)
        x_target = torch.cat([mu_t0.unsqueeze(1).expand(B, T, -1), x_target], dim=-1)

    target_mask = torch.arange(T, device=device)[None, :] >= n_targets[:, None]        # (B, T)
    task = {
        "x_context": x_context.reshape(B, L * S, -1),
        "y_context": history.reshape(B, L * S, 1),
        "context_mask": ~on.reshape(B, L * S),
        "x_target": x_target,
        "y_target": y[idx[:, None], t0[:, None], targets].unsqueeze(-1),
        "target_mask": target_mask,
        "sensor_history": history * on,
        "sensor_on": on.float(),
        "sensor_time": rel_time.view(1, L, 1).expand(B, L, 1).contiguous(),
        "sensor_position": coordinates[sensors].unsqueeze(0).expand(B, -1, -1).contiguous(),
        "target_index": targets,
    }
    if use_mu:
        task["mu"] = mu_t0
    return task


def _random_tasks(y: Tensor, mu: Tensor, idx: Tensor, coordinates: Tensor, sensors: Tensor,
                  use_mu: bool, target_range: Tuple[int, int] = TARGET_RANGE) -> Dict[str, Tensor]:
    """Random tasks, drawn independently for every trajectory in idx: t0, lag, dropped sensors,
    number of targets (uniform in target_range) and which nodes (any node that is not an active
    sensor of that task)."""
    device, n_state, S, B = coordinates.device, coordinates.shape[0], len(sensors), len(idx)
    t0 = torch.randint(MIN_TIME, y.shape[1], (B,), device=device)
    lag = torch.randint(0, MAX_LAG + 1, (B,), device=device)

    drop_options = torch.as_tensor(DROP_SENSOR_OPTIONS, device=device)
    n_drop = drop_options[torch.randint(len(drop_options), (B,), device=device)]
    rank = torch.rand(B, S, device=device).argsort(dim=1).argsort(dim=1)               # random order
    active = rank >= n_drop[:, None]                                                   # (B, S)

    n_targets = torch.randint(target_range[0], target_range[1] + 1, (B,), device=device)
    n_targets = n_targets.clamp(max=n_state - S)
    scores = torch.rand(B, n_state, device=device)
    scores[:, sensors] = torch.where(active, 2.0, scores[:, sensors])                 # active sensors last
    T = int(n_targets.max())
    targets = scores.topk(T, dim=1, largest=False).indices                             # (B, T), random nodes
    return _build_tasks(y, mu, idx, coordinates, sensors, use_mu, t0, lag, active, targets, n_targets)


class TrainBatches:
    """Training batches built on the data's device. One epoch = one pass over the training
    trajectories in random order, each batch a random task."""

    def __init__(self, data: PinballData, sensors: Sequence[int], use_mu: bool, batch_size: int = 16,
                 target_range: Tuple[int, int] = TARGET_RANGE):
        self.y, self.mu, self.coordinates = data.y["train"], data.mu["train"], data.coordinates
        self.sensors = torch.as_tensor(sensors, device=self.coordinates.device)
        self.use_mu, self.batch_size, self.target_range = use_mu, batch_size, target_range

    def __len__(self) -> int:
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        order = torch.randperm(len(self.y), device=self.y.device)
        for start in range(0, len(order), self.batch_size):
            yield _random_tasks(self.y, self.mu, order[start:start + self.batch_size],
                                self.coordinates, self.sensors, self.use_mu, self.target_range)


def make_random_batches(data: PinballData, split: str, sensors: Sequence[int], use_mu: bool,
                        n_rounds: int = 4, batch_size: int = 16, seed: int = 0,
                        target_range: Tuple[int, int] = TARGET_RANGE) -> List[Dict[str, Tensor]]:
    """A fixed set of random tasks on one split: n_rounds passes over its trajectories, drawn once
    with `seed`, so every evaluation uses exactly the same tasks. Same distribution as training."""
    y, mu, coordinates = data.y[split], data.mu[split], data.coordinates
    sensors = torch.as_tensor(sensors, device=coordinates.device)
    devices = [y.device] if y.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):  # leaves the training random state untouched
        torch.manual_seed(seed)
        batches = []
        for _ in range(n_rounds):
            order = torch.randperm(len(y), device=y.device)
            for start in range(0, len(order), batch_size):
                batches.append(_random_tasks(y, mu, order[start:start + batch_size], coordinates,
                                             sensors, use_mu, target_range))
        return batches


def make_validation_batches(data: PinballData, sensors: Sequence[int], use_mu: bool,
                            n_rounds: int = 4, batch_size: int = 16, seed: int = 0,
                            target_range: Tuple[int, int] = TARGET_RANGE) -> List[Dict[str, Tensor]]:
    """The fixed validation set: make_random_batches on the validation split."""
    return make_random_batches(data, "valid", sensors, use_mu, n_rounds, batch_size, seed, target_range)


def make_test_batch(data: PinballData, indices: Sequence[int], sensors: Sequence[int], use_mu: bool,
                    t0: int, lag: int, split: str = "test") -> Dict[str, Tensor]:
    """Given trajectories, t0 and lag, all sensors active and every mesh node a target (for the
    test metrics and plots of the full field; no padding, the masks are all False)."""
    device = data.coordinates.device
    idx = torch.as_tensor(indices, device=device)
    sensors = torch.as_tensor(sensors, device=device)
    B, n_state = len(idx), data.coordinates.shape[0]
    full = lambda value: torch.full((B,), value, dtype=torch.long, device=device)
    return _build_tasks(data.y[split], data.mu[split], idx, data.coordinates, sensors, use_mu,
                        full(t0), full(lag), torch.ones(B, len(sensors), dtype=torch.bool, device=device),
                        torch.arange(n_state, device=device).expand(B, -1), full(n_state))