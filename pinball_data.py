"""Fluidic pinball (FEniCS) data as Neural Process tasks.

A task is one trajectory at one time t0:
- context: the fixed sensors at the lag + 1 times t0 - lag, ..., t0 (sometimes one sensor dropped);
- targets: a random set of other mesh nodes at t0 (the context is not repeated among them;
  the model adds it itself for the posterior, with target_includes_context=False).

x = ([mu,] relative time, x, y): relative time (t - t0) / ntimes, so 0 at t0 and negative before;
mu (the 3 flow parameters) only when use_mu=True.
y = the state value, standardized with the training-set mean and std.

Batches are the dictionaries training.py expects. The whole dataset is small (about 350 MB of
training data), so it is kept on the GPU and batches are built there directly: no loader workers
and no copies between CPU and GPU (call data.to(device) once).
"""

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["PinballData", "load_pinball", "x_bounds", "TrainBatches",
           "make_validation_batches", "make_test_batch", "LAG_OPTIONS"]

N_TRAJECTORIES = 500
N_TIMES = 31                       # T = 3.0, dt = 0.1
LAG_OPTIONS = (0, 4, 9, 19)        # number of past readings in the context
MIN_TIME = 20                      # t0 >= MIN_TIME, so the longest lag fits
TARGET_RANGE = (2048, 4096)        # random target nodes per task, sensors excluded
DROP_SENSOR_OPTIONS = (0, 1)


@dataclass
class PinballData:
    function_space: object         # FEniCS function space, for plotting fields
    coordinates: Tensor            # (nstate, 2) node coordinates
    y: Dict[str, Tensor]           # split -> (trajectories, ntimes, nstate), standardized
    mu: Dict[str, Tensor]          # split -> (trajectories, nparams)
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
    y_all = torch.as_tensor(data["y"], dtype=torch.float32).reshape(N_TRAJECTORIES, N_TIMES, n_state)
    mu_all = torch.as_tensor(data["mu"], dtype=torch.float32).reshape(N_TRAJECTORIES, -1)

    rng = np.random.RandomState(seed)
    n_train = round(0.8 * N_TRAJECTORIES)
    idx_train = rng.choice(N_TRAJECTORIES, size=n_train, replace=False)
    rest = np.setdiff1d(np.arange(N_TRAJECTORIES), idx_train)
    splits = {"train": idx_train, "valid": rest[::2], "test": rest[1::2]}

    y_mean, y_std = y_all[idx_train].mean().item(), y_all[idx_train].std().item()
    y = {name: (y_all[idx] - y_mean) / y_std for name, idx in splits.items()}
    mu = {name: mu_all[idx] for name, idx in splits.items()}
    return PinballData(space, coordinates, y, mu, y_mean, y_std)


def x_bounds(data: PinballData, use_mu: bool) -> Tuple[List[float], List[float]]:
    """Domain of x, for the model's input scaling: [mu,] relative time, x, y."""
    time_low = -max(LAG_OPTIONS) / data.n_times
    low = [time_low] + data.coordinates.min(0).values.tolist()
    high = [0.0] + data.coordinates.max(0).values.tolist()
    if use_mu:
        low = data.mu["train"].min(0).values.tolist() + low
        high = data.mu["train"].max(0).values.tolist() + high
    return low, high


# ----------------------------------------------------------------------------- tasks

def _build_task(y: Tensor, mu: Tensor, idx: Tensor, coordinates: Tensor, sensors: Tensor,
                use_mu: bool, t0: int, lag: int, targets: Tensor) -> Dict[str, Tensor]:
    """Context = sensors over t0 - lag ... t0; targets = given nodes at t0.

    y, mu: one split; idx: the trajectories of the batch. Everything stays on y's device.
    """
    device, n_times = y.device, y.shape[1]
    B = len(idx)
    times = torch.arange(t0 - lag, t0 + 1, device=device).repeat_interleave(len(sensors))  # (N,)
    points = sensors.repeat(lag + 1)                                                        # (N,)
    time_c = ((times - t0).float() / n_times).unsqueeze(-1)

    x_context = torch.cat([time_c, coordinates[points]], dim=-1).expand(B, -1, -1)
    x_target = torch.cat([torch.zeros(len(targets), 1, device=device), coordinates[targets]],
                         dim=-1).expand(B, -1, -1)
    if use_mu:
        mu_b = mu[idx].unsqueeze(1)
        x_context = torch.cat([mu_b.expand(-1, x_context.shape[1], -1), x_context], dim=-1)
        x_target = torch.cat([mu_b.expand(-1, x_target.shape[1], -1), x_target], dim=-1)
    rows = idx.unsqueeze(1)
    return {
        "x_context": x_context.contiguous(),
        "y_context": y[rows, times.unsqueeze(0), points.unsqueeze(0)].unsqueeze(-1),
        "x_target": x_target.contiguous(),
        "y_target": y[rows, t0, targets.unsqueeze(0)].unsqueeze(-1),
    }


def _random_task(y: Tensor, mu: Tensor, idx: Tensor, coordinates: Tensor, sensors: Tensor,
                 use_mu: bool) -> Dict[str, Tensor]:
    """Random task, shared by the whole batch: dropped sensors, t0, lag and target nodes.
    Targets are drawn from the nodes that are not active sensors."""
    device, n_state = coordinates.device, coordinates.shape[0]
    n_drop = DROP_SENSOR_OPTIONS[torch.randint(len(DROP_SENSOR_OPTIONS), ()).item()]
    active = sensors[torch.randperm(len(sensors), device=device)[n_drop:]] if n_drop else sensors

    others = torch.ones(n_state, dtype=torch.bool, device=device)
    others[active] = False
    others = others.nonzero().squeeze(1)
    n_targets = min(int(torch.randint(TARGET_RANGE[0], TARGET_RANGE[1] + 1, ())), len(others))
    targets = others[torch.randperm(len(others), device=device)[:n_targets]]

    t0 = int(torch.randint(MIN_TIME, y.shape[1] - 1, ()))
    lag = LAG_OPTIONS[torch.randint(len(LAG_OPTIONS), ()).item()]
    return _build_task(y, mu, idx, coordinates, active, use_mu, t0, lag, targets)


class TrainBatches:
    """Training batches built on the data's device. One epoch = one pass over the training
    trajectories in random order, each batch a random task."""

    def __init__(self, data: PinballData, sensors: Sequence[int], use_mu: bool, batch_size: int = 16):
        self.y, self.mu, self.coordinates = data.y["train"], data.mu["train"], data.coordinates
        self.sensors = torch.as_tensor(sensors, device=self.coordinates.device)
        self.use_mu, self.batch_size = use_mu, batch_size

    def __len__(self) -> int:
        return math.ceil(len(self.y) / self.batch_size)

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        order = torch.randperm(len(self.y), device=self.y.device)
        for start in range(0, len(order), self.batch_size):
            yield _random_task(self.y, self.mu, order[start:start + self.batch_size],
                               self.coordinates, self.sensors, self.use_mu)


def make_validation_batches(data: PinballData, sensors: Sequence[int], use_mu: bool,
                            n_rounds: int = 4, batch_size: int = 16, seed: int = 0) -> List[Dict[str, Tensor]]:
    """A fixed validation set: n_rounds passes over the validation trajectories, with random
    tasks drawn once with `seed`, so every epoch is validated on exactly the same tasks."""
    y, mu, coordinates = data.y["valid"], data.mu["valid"], data.coordinates
    sensors = torch.as_tensor(sensors, device=coordinates.device)
    devices = [y.device] if y.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):  # leaves the training random state untouched
        torch.manual_seed(seed)
        batches = []
        for _ in range(n_rounds):
            order = torch.randperm(len(y), device=y.device)
            for start in range(0, len(order), batch_size):
                batches.append(_random_task(y, mu, order[start:start + batch_size], coordinates,
                                            sensors, use_mu))
        return batches


def make_test_batch(data: PinballData, indices: Sequence[int], sensors: Sequence[int], use_mu: bool,
                    t0: int, lag: int, split: str = "test") -> Dict[str, Tensor]:
    """Given trajectories, sensors, t0 and lag; every mesh node is a target (for evaluation and
    plots of the full field: the posterior is not used at test time)."""
    device = data.coordinates.device
    return _build_task(data.y[split], data.mu[split], torch.as_tensor(indices, device=device),
                       data.coordinates, torch.as_tensor(sensors, device=device), use_mu, t0, lag,
                       torch.arange(data.coordinates.shape[0], device=device))