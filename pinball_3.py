#IMPORT LIBRARIES


from __future__ import print_function

from dolfin import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import sys
from IPython.display import clear_output as clc
from pathlib import Path
from processdata import trajectory, trajectories, multiplot
from torch.utils.data import Dataset, DataLoader
from functools import partial
from LNP.LATNPsimple import LatNP_simple
from LNP.LATNPsimple_bttime import LatNP_simple_2 as LatNP_simple_bttime
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np, DelayedReduceLROnPlateau
import math
from scipy.stats import norm as sp_norm   # add after line 23 (import math)

from pinball_paths import resolve_pinball_asset

plt.style.use('default')
set_log_level(LogLevel.ERROR)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def select_sensor_locations(fixed_sensor_locations, nstate, use_all_sensors=False, drop_random_sensors=0):
    if drop_random_sensors < 0:
        raise ValueError("drop_random_sensors must be >= 0")

    if fixed_sensor_locations is not None:
        sensor_locations = torch.as_tensor(fixed_sensor_locations, dtype=torch.long)
    elif use_all_sensors:
        sensor_locations = torch.arange(nstate, dtype=torch.long)
    else:
        return None

    if sensor_locations.numel() == 0:
        raise ValueError("fixed_sensor_locations must contain at least one index")
    if sensor_locations.min().item() < 0 or sensor_locations.max().item() >= nstate:
        raise ValueError("fixed_sensor_locations contain out-of-range indices")

    sensor_locations = torch.unique(sensor_locations)
    if drop_random_sensors > 0:
        if drop_random_sensors >= sensor_locations.numel():
            raise ValueError("drop_random_sensors must be less than the number of available sensors")
        perm = torch.randperm(sensor_locations.numel())
        keep = perm[drop_random_sensors:]
        sensor_locations = sensor_locations[keep]

    return sensor_locations


def choose_drop_random_sensors(drop_random_sensors, drop_random_sensors_options):
    if drop_random_sensors_options is None:
        return drop_random_sensors

    options = list(drop_random_sensors_options)
    if len(options) == 0:
        raise ValueError("drop_random_sensors_options must not be empty")
    if any(opt < 0 for opt in options):
        raise ValueError("drop_random_sensors_options values must be >= 0")

    choice_idx = torch.randint(0, len(options), (1,)).item()
    return int(options[choice_idx])

def spatiotemporal_collate_fn(batch, num_context_sensors_min=2, num_context_sensors_max=10,
                              num_target_min=10, num_target_max=30, mesh_coords=None,
                              fixed_sensor_locations=None, use_all_sensors=False,
                              drop_random_sensors=0, drop_random_sensors_options=None, return_mu_as_label=False):
    if mesh_coords is None:
        raise ValueError("mesh_coords must be provided!")
        
    if not isinstance(mesh_coords, torch.Tensor):
        mesh_coords = torch.as_tensor(mesh_coords, dtype=torch.float32)

    batch_size = len(batch)
    
    # --- Safely handle batches with or without 'mu' ---
    if isinstance(batch[0], (tuple, list)):
        ntimes, nstate = batch[0][0].shape
        batch_trajectories = torch.stack([item[0] for item in batch])
        batch_mu = torch.stack([item[1] for item in batch])
    else:
        ntimes, nstate = batch[0].shape
        batch_trajectories = torch.stack(batch)
        batch_mu = None

    # 1. Sample spatial locations (Once per batch)
    chosen_drop = choose_drop_random_sensors(drop_random_sensors, drop_random_sensors_options)
    sensor_locations_fixed = select_sensor_locations(
        fixed_sensor_locations, nstate, use_all_sensors, chosen_drop
    )
    
    if sensor_locations_fixed is not None:
        num_sensors = sensor_locations_fixed.numel()
        sensor_locations = sensor_locations_fixed
        
        remaining_mask = torch.ones(nstate, dtype=torch.bool)
        remaining_mask[sensor_locations] = False
        remaining_indices = torch.nonzero(remaining_mask, as_tuple=False).squeeze(1)
        
        num_target = min(torch.randint(num_target_min, num_target_max + 1, (1,)).item(), nstate - num_sensors)
        if num_target > 0:
            if remaining_indices.numel() > num_target:
                rand_idx = torch.randperm(remaining_indices.numel())[:num_target]
                target_extra = remaining_indices[rand_idx]
            else:
                target_extra = remaining_indices
            target_state_indices = torch.cat([sensor_locations, target_extra], dim=0)
        else:
            target_state_indices = sensor_locations
    else:
        num_sensors = torch.randint(num_context_sensors_min, num_context_sensors_max + 1, (1,)).item()
        num_target = min(torch.randint(num_target_min, num_target_max + 1, (1,)).item(), nstate - num_sensors)
        all_sampled_locations = torch.randperm(nstate)[:num_sensors + num_target]
        sensor_locations = all_sampled_locations[:num_sensors]
        target_state_indices = all_sampled_locations

    total_locations = num_sensors + num_target

    # 2. Sample temporal locations using PyTorch RNG!
    time_idx = torch.randint(20, ntimes - 1, (1,)).item()
    lag_options = torch.tensor([0, 4, 9, 19])
    lag = lag_options[torch.randint(0, len(lag_options), (1,))].item()
    
    time_window = torch.arange(time_idx - lag, time_idx + 1)
    context_time_indices = time_window.repeat_interleave(num_sensors)
    target_time_indices = torch.full((total_locations,), time_idx, dtype=torch.long)
    context_state_indices = sensor_locations.repeat(1 + lag)

    # 3. Build Base Coordinates and Times (RELATIVE TIME SHIFT)
    norm_context_time = ((context_time_indices - time_idx).float() / ntimes).unsqueeze(1)
    norm_target_time = torch.zeros((total_locations, 1), dtype=torch.float32)
    
    context_coords = mesh_coords[context_state_indices]
    target_coords = mesh_coords[target_state_indices]

    x_ctx_base = torch.cat([norm_context_time, context_coords], dim=-1)
    x_tgt_base = torch.cat([norm_target_time, target_coords], dim=-1)

    x_context = x_ctx_base.unsqueeze(0).expand(batch_size, -1, -1)
    x_target = x_tgt_base.unsqueeze(0).expand(batch_size, -1, -1)

    # 4. Handle Mu parameters
    if batch_mu is not None and not return_mu_as_label:
        # existing path: prepend mu to x (USE_MU=True feature mode)
        if batch_mu.dim() == 4 and batch_mu.size(1) == 1:
            batch_mu = batch_mu.squeeze(1)
        if batch_mu.dim() == 3:
            mu_context = batch_mu[:, context_time_indices, :]
            mu_target  = batch_mu[:, target_time_indices, :]
        else:
            mu_context = batch_mu.unsqueeze(1).expand(-1, len(context_time_indices), -1)
            mu_target  = batch_mu.unsqueeze(1).expand(-1, len(target_time_indices), -1)
        x_context = torch.cat([mu_context, x_context], dim=-1)
        x_target  = torch.cat([mu_target,  x_target],  dim=-1)

    # 5. Extract Y values (unchanged)
    y_context = batch_trajectories[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target  = batch_trajectories[:, target_time_indices,  target_state_indices].unsqueeze(-1)

    # 6. Return mu as label if requested
    if batch_mu is not None and return_mu_as_label:
        # batch_mu is (B, nparams) — constant per trajectory, no time indexing needed
        theta = batch_mu if batch_mu.dim() == 2 else batch_mu[:, 0, :]
        return (x_context.contiguous(), y_context.contiguous(),
                x_target.contiguous(),  y_target.contiguous(),
                theta.contiguous())

    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()

def spatiotemporal_test_collate_fn(batch, num_context_sensors_min=2, num_context_sensors_max=10, 
                                   mesh_coords=None, fixed_sensor_locations=None,
                                   use_all_sensors=False, drop_random_sensors=0,
                                   drop_random_sensors_options=None,
                                   time_idx=None, lag=None, return_mu_as_label=False):
    if mesh_coords is None:
        raise ValueError("mesh_coords must be provided!")

    if not isinstance(mesh_coords, torch.Tensor):
        mesh_coords = torch.as_tensor(mesh_coords, dtype=torch.float32)

    batch_size = len(batch)

    if isinstance(batch[0], (tuple, list)):
        ntimes, nstate = batch[0][0].shape
        batch_trajectories = torch.stack([item[0] for item in batch])
        batch_mu = torch.stack([item[1] for item in batch])
    else:
        ntimes, nstate = batch[0].shape
        batch_trajectories = torch.stack(batch)
        batch_mu = None

    chosen_drop = choose_drop_random_sensors(drop_random_sensors, drop_random_sensors_options)
    sensor_locations_fixed = select_sensor_locations(
        fixed_sensor_locations, nstate, use_all_sensors, chosen_drop
    )
    
    if sensor_locations_fixed is not None:
        sensor_locations = sensor_locations_fixed
        num_sensors = sensor_locations.numel()
    else:
        num_sensors = torch.randint(num_context_sensors_min, num_context_sensors_max + 1, (1,)).item()
        sensor_locations = torch.randperm(nstate)[:num_sensors]

    # Override random choices with PyTorch RNG
    if time_idx is None:
        time_idx = torch.randint(20, ntimes - 1, (1,)).item()
    if lag is None:
        lag_options = torch.tensor([0, 4, 9, 19])
        lag = lag_options[torch.randint(0, len(lag_options), (1,))].item()

    time_window = torch.arange(time_idx - lag, time_idx + 1)
    context_time_indices = time_window.repeat_interleave(num_sensors)
    context_state_indices = sensor_locations.repeat(1 + lag)
    
    target_time_indices = torch.full((nstate,), time_idx, dtype=torch.long)
    target_state_indices = torch.arange(nstate)

    norm_context_time = ((context_time_indices - time_idx).float() / ntimes).unsqueeze(1)
    norm_target_time = torch.zeros((nstate, 1), dtype=torch.float32)
    
    context_coords = mesh_coords[context_state_indices]
    target_coords = mesh_coords[target_state_indices]

    x_ctx_base = torch.cat([norm_context_time, context_coords], dim=-1)
    x_tgt_base = torch.cat([norm_target_time, target_coords], dim=-1)

    x_context = x_ctx_base.unsqueeze(0).expand(batch_size, -1, -1)
    x_target = x_tgt_base.unsqueeze(0).expand(batch_size, -1, -1)

    if batch_mu is not None:
        if batch_mu.dim() == 4 and batch_mu.size(1) == 1:
            batch_mu = batch_mu.squeeze(1)
            
        if batch_mu.dim() == 3:
            mu_context = batch_mu[:, context_time_indices, :]
            mu_target = batch_mu[:, target_time_indices, :]
        else:
            mu_context = batch_mu.unsqueeze(1).expand(-1, len(context_time_indices), -1)
            mu_target = batch_mu.unsqueeze(1).expand(-1, len(target_time_indices), -1)

        x_context = torch.cat([mu_context, x_context], dim=-1)
        x_target = torch.cat([mu_target, x_target], dim=-1)

    y_context = batch_trajectories[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = batch_trajectories[:, target_time_indices, target_state_indices].unsqueeze(-1)

    if batch_mu is not None and return_mu_as_label:
        theta = batch_mu if batch_mu.dim() == 2 else batch_mu[:, 0, :]
        return (x_context.contiguous(), y_context.contiguous(),
                    x_target.contiguous(), y_target.contiguous(),
                    theta.contiguous())
    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()

class BTTimeInputAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, time_feature_idx: int):
        super().__init__()
        self.base_model = base_model
        self.time_feature_idx = time_feature_idx
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


    def forward(self, x_context: torch.Tensor, y_context: torch.Tensor,
                x_target: torch.Tensor, y_target: torch.Tensor = None,
                num_samples: int = 1):
        t_context_flat = x_context[..., self.time_feature_idx]
        t_target = x_target[..., self.time_feature_idx]

        x_context_no_time = torch.cat(
            [x_context[..., :self.time_feature_idx], x_context[..., self.time_feature_idx + 1:]],
            dim=-1
        )
        x_target_no_time = torch.cat(
            [x_target[..., :self.time_feature_idx], x_target[..., self.time_feature_idx + 1:]],
            dim=-1
        )

        batch_size, num_context_total, feat_dim = x_context_no_time.shape
        t_dim = torch.unique_consecutive(t_context_flat[0]).numel()

        if num_context_total % t_dim != 0:
            raise ValueError("Unable to reshape context into (batch, t_dim, num_context, features).")

        num_context = num_context_total // t_dim

        x_context_bt = x_context_no_time.reshape(batch_size, t_dim, num_context, feat_dim)
        y_context_bt = y_context.reshape(batch_size, t_dim, num_context, -1)
        t_context_bt = t_context_flat.reshape(batch_size, t_dim, num_context)[:, :, 0]
        context_mask = torch.ones((batch_size, t_dim, num_context), dtype=torch.bool, device=x_context.device)

        return self.base_model(
            x_context_bt, y_context_bt, t_context_bt, context_mask,
            x_target_no_time, y_target=y_target, t_target=t_target,
            num_samples=num_samples
        )
        
class SpatiotemporalDataset(Dataset):
    """Dataset for Neural Process training on spatiotemporal fields"""
    def __init__(self, data, mu_params=None):
        """
        Args:
            data: tensor of shape (n_trajectories, ntimes, nstate)
            mu_params: tensor of shape (n_trajectories, nparams) - optional parameters for each trajectory
        """
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        if mu_params is not None:
            if isinstance(mu_params, np.ndarray):
                self.mu_params = torch.from_numpy(mu_params).float()
            else:
                self.mu_params = mu_params.float()
        else:
            self.mu_params = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns trajectory and optionally MU parameters
        if self.mu_params is not None:
            return self.data[idx], self.mu_params[idx]  # (ntimes, nstate), (nparams,)
        
        return self.data[idx]

# FUNCTION TO CONVERT VECTORS INTO FUNCTIONS

def vec2fun(yvec, Yh):
    '''
    Convert a vector into a fenics function
    Input: vector of degrees of freedom and functional space
    Output: function
    '''
    y = Function(Yh)
    y.vector()[:] = yvec
    return y

def plot_with_colorbar(y, Yh, ax=None, cmap="jet", vmin=None, vmax=None, label=None, cbar_kwargs=None):
    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)
    mappable = plot(vec2fun(y, Yh), cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar_kwargs is None:
        cbar_kwargs = {"shrink": 0.75, "pad": 0.02}
    cbar = plt.colorbar(mappable, ax=ax, **cbar_kwargs)

    # Change the numbers
    cbar.ax.tick_params(labelsize=14)

    # Change the main label text
    cbar.set_label(label, size=16)
    return mappable

def main():
    
    USE_MU = True
    USE_BT_TIME = False   # True -> use LatNP_simple_bttime, False -> use LatNP_simple   
    ESTIMATE_PARAMS = False
    
    print(f"USE_MU: {USE_MU}, USE_BT_TIME: {USE_BT_TIME}")
    
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / f"logs_pinball_{'bt' if USE_BT_TIME else 'no_bt'}_{'mu' if USE_MU else 'no_mu'}_3"
    checkpoints_dir = script_dir / f"checkpoints_pinball_{'bt' if USE_BT_TIME else 'no_bt'}_{'mu' if USE_MU else 'no_mu'}_3"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    mesh = Mesh(str(resolve_pinball_asset(script_dir, "Pinball_mesh.xml")))

    plt.figure(figsize=(8, 6))
    plot(mesh, color="grey", linewidth=0.75)
    plt.title(f"Mesh ($N_h$={mesh.num_vertices()})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(logs_dir / "mesh.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Mesh plot saved to {logs_dir / 'mesh.png'}")    
    
    Yh = FunctionSpace(mesh, "CG", 1)
    nstate = Yh.dim()
    
    # VISUALIZE MESH NODE ORDERING
    # Color each node by its index in the mesh vector to show ordering

    # Create a function where each node gets its index value as the "state"
    node_indices = Function(Yh)
    node_indices.vector()[:] = np.arange(nstate)

    # Plot with colormap
    plt.figure(figsize=(10, 8))
    p = plot(node_indices, cmap="viridis", title=f"Mesh Node Ordering (nstate={nstate})")
    plt.colorbar(p, label="Node Index")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Mesh Node Ordering Visualization\n(Each point colored by its index in the mesh vector)")
    plt.tight_layout()
    plt.savefig(logs_dir / "mesh_node_ordering.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Mesh node ordering plot saved to {logs_dir / 'mesh_node_ordering.png'}")
    
    print(f"Total number of mesh nodes: {nstate}")
    print(f"Node indices range: [0, {nstate-1}]")
    print(f"\nThis shows how FEniCS orders the nodes in the mesh vector.")
    print(f"The gradient shows the spatial arrangement of node indices.")
    
    # LOAD SNAPSHOTS MATRICES

    dt = Constant(0.1)
    T = 3.0
    ntimesteps = round(T / dt)
    ntimes = ntimesteps + 1 # Time series length

    ntrajectories = 500
    nparams = 3

    Data = np.load(str(resolve_pinball_asset(script_dir, "Pinball_data.npz")))
    Y = torch.tensor(Data["y"])
    MU = torch.tensor(Data["mu"])
    VNS = torch.tensor(Data["v"])

    del Data
    
    plot_state = lambda y: plot_with_colorbar(y, Yh, cmap="jet", vmin=vmin, vmax=vmax, label="State value")
    
    which = (0, 1, 2)
    plotlist = [Y[which[0]], Y[which[1]], Y[which[2]]]
    vmin = min(plotlist[i].min() for i in range(len(plotlist)))
    vmax = max(plotlist[i].max() for i in range(len(plotlist)))

    trajectories(plotlist, plot_state, titles = ("Trajectory 1", "Trajectory 2", "Trajectory 3"), figsize = (15, 15), save = True, name = logs_dir / "trajectories")

    # TRAIN-VALIDATION-TEST SPLITTING

    np.random.seed(0)

    ntrain = round(0.8 * ntrajectories)

    idx_train = np.random.choice(ntrajectories, size = ntrain, replace = False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask!=0)[0]]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]

    nvalid = idx_valid.shape[0]
    ntest = idx_test.shape[0]

    Ytrain = Y[idx_train]
    Yvalid = Y[idx_valid]
    Ytest = Y[idx_test]
    MUtrain = MU[idx_train]
    MUvalid = MU[idx_valid]
    MUtest = MU[idx_test]

    del Y, MU
    
    # Reshape data for spatiotemporal Neural Process training
    # From (n_trajectories, ntimes * nstate) to (n_trajectories, ntimes, nstate)

    Ytrain = Ytrain.reshape(ntrain, ntimes, nstate)
    Yvalid = Yvalid.reshape(nvalid, ntimes, nstate)
    Ytest = Ytest.reshape(ntest, ntimes, nstate)

    # # --- ADD THIS: Normalize Y (State Values) to [0, 1] ---
    # y_min = Ytrain.min()
    # y_max = Ytrain.max()

    # print(f"\nNormalizing Y values to [0, 1]:")
    # print(f"  Training Min: {y_min:.4f}, Max: {y_max:.4f}")

    # # Add 1e-8 to the denominator to mathematically prevent division by zero
    # scale_range = y_max - y_min + 1e-8

    # Ytrain = (Ytrain - y_min) / scale_range
    # Yvalid = (Yvalid - y_min) / scale_range
    # Ytest  = (Ytest - y_min) / scale_range

    # print(f"Ytrain shape: {Ytrain.shape}")
    # print(f"Yvalid shape: {Yvalid.shape}")
    # print(f"Ytest shape: {Ytest.shape}")
    
    # # --- Normalize MU to the range [-1, 1] ---
    # # .min(dim=0)[0] extracts just the values (ignoring the indices)
    # mu_min = MUtrain.min(dim=0)[0]
    # mu_max = MUtrain.max(dim=0)[0]
    
    # # Add a tiny epsilon to prevent division by zero if a parameter is constant
    # mu_range = (mu_max - mu_min).clamp(min=1e-8)
    
    # MUtrain = 2.0 * (MUtrain - mu_min) / mu_range - 1.0
    # MUvalid = 2.0 * (MUvalid - mu_min) / mu_range - 1.0
    # MUtest  = 2.0 * (MUtest  - mu_min) / mu_range - 1.0

    # Create dataset objects with MU parametersprint(f"Validation dataset: {len(val_dataset)} trajectories")
    if USE_MU:
        train_dataset = SpatiotemporalDataset(Ytrain, MUtrain)
        val_dataset = SpatiotemporalDataset(Yvalid, MUvalid)
        test_dataset = SpatiotemporalDataset(Ytest, MUtest)
    else:
        train_dataset = SpatiotemporalDataset(Ytrain)#, MUtrain)
        val_dataset = SpatiotemporalDataset(Yvalid)#, MUvalid)
        test_dataset = SpatiotemporalDataset(Ytest)#, MUtest)
    
    print(f"Train dataset: {len(train_dataset)} trajectories")
    
    # Get mesh coordinates for spatial information
    mesh_coordinates = Yh.tabulate_dof_coordinates()  # (nstate, 2) - actual (x,y) positions
    mesh_coordinates = torch.as_tensor(mesh_coordinates, dtype=torch.float32)

    # # --- ADD THIS: Normalize X/Y Spatial Coordinates ---
    # x1_min, x1_max = mesh_coordinates[:, 0].min(), mesh_coordinates[:, 0].max()
    # x2_min, x2_max = mesh_coordinates[:, 1].min(), mesh_coordinates[:, 1].max()

    # print(f"\nNormalizing Spatial Coordinates:")
    # print(f"  X range: [{x1_min:.4f}, {x1_max:.4f}] -> [-1, 1]")
    # print(f"  Y range: [{x2_min:.4f}, {x2_max:.4f}] -> [-1, 1]")

    # mesh_coordinates_norm = mesh_coordinates.clone()
    # # Scale to [-1, 1]
    # mesh_coordinates_norm[:, 0] = 2.0 * (mesh_coordinates[:, 0] - x1_min) / (x1_max - x1_min) - 1.0
    # mesh_coordinates_norm[:, 1] = 2.0 * (mesh_coordinates[:, 1] - x2_min) / (x2_max - x2_min) - 1.0
    
    mesh_coordinates_norm = mesh_coordinates  # Use actual coordinates without normalization
    # ---------------------------------------------------

    # fixed_sens = [3501 ,1100 , 4590 ,6888, 4300, 2540]  # Example fixed sensor locations (5 sensors)
    # fixed_sens = [3496, 3595, 4803, 5583, 710, 1907, 1521, 4830, 4732, 6888]
    fixed_sens = [1573, 6925, 1986]
    # fixed_sens = [3496, 3595, 5240, 6180]

    # Create data loaders
    # Context: 2-10 fixed sensors × 31 time steps = 62-310 context points
    # Target: 10-30 fixed spatial locations × 31 time steps = 310-930 target points (memory-safe!)
    # Input: [time, x, y] - 3D coordinates with actual spatial positions
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,           # <--- ADD THIS (try 4 to 8 depending on your CPU cores)
        pin_memory=True,         # <--- ADD THIS (speeds up CPU to GPU transfer)
        prefetch_factor=2,       # <--- ADD THIS (keeps 2 batches ready in the queue)
        persistent_workers=True, # <--- ADD THIS (prevents worker respawn overhead)
        collate_fn=partial(spatiotemporal_collate_fn,
                        num_context_sensors_min=4,  num_context_sensors_max=32,
                        num_target_min=2048, num_target_max=4096,
                        mesh_coords=mesh_coordinates_norm,
                        fixed_sensor_locations=fixed_sens,
                        use_all_sensors=True,
                        drop_random_sensors_options=[0],
                        return_mu_as_label=ESTIMATE_PARAMS)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=partial(spatiotemporal_collate_fn,
                        num_context_sensors_min=4,  num_context_sensors_max=32,
                        num_target_min=2048, num_target_max=4096,
                        mesh_coords=mesh_coordinates_norm,
                        fixed_sensor_locations=fixed_sens,
                        use_all_sensors=True,
                        drop_random_sensors_options=[0],
                        return_mu_as_label=ESTIMATE_PARAMS)
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Test the collate function
    batch_sample = next(iter(train_loader))
    if ESTIMATE_PARAMS:
        x_c, y_c, x_t, y_t, mu_t = batch_sample
    else:
        x_c, y_c, x_t, y_t = batch_sample

    print(f"\nBatch shapes:")
    print(f"Context y range: [{y_c.min():.3f}, {y_c.max():.3f}]")

    print(f"  Context: x_c {x_c.shape}, y_c {y_c.shape}")
    print(f"\nContext x range: [{x_c.min():.3f}, {x_c.max():.3f}]")

    print(f"  Target:  x_t {x_t.shape}, y_t {y_t.shape}")
    
    y_dim = 1  # Output dimension: state value
    r_dim = 128  # Representation dimension
    z_dim = 128 # Latent dimension
    hidden_dim = 128  # Hidden layer dimension
    n_hidden = 2  # Number of hidden layers

    if USE_BT_TIME:
        x_dim = 5 if USE_MU else 2   # [mu(3) + x,y] or [x,y]
    else:
        x_dim = 6 if USE_MU else 3   # [mu(3) + t,x,y] or [t,x,y]
    y_dim = 1  # Output dimension: state value
    r_dim = 128  # Representation dimension
    z_dim = 128 # Latent dimension
    hidden_dim = 128  # Hidden layer dimension
    n_hidden = 2  # Number of hidden layers

    # Create model
    if USE_BT_TIME:
        base_model = LatNP_simple_bttime(
            x_dim=x_dim,
            y_dim=y_dim,
            r_dim=r_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            activation=nn.ReLU,
            dropout=0.0,
            is_normalized=True,
            norm_type='layer',
            fourier_vars=2,  # Fourier on spatial coords only
            num_frequencies=32,
            fourier_scale=1.0,
            learnable_fourier=True,
            use_deeponet_decoder=True,
        ).to(device)

        time_feature_idx = 3 if USE_MU else 0
        model = BTTimeInputAdapter(base_model, time_feature_idx=time_feature_idx).to(device)
    else:
        model = LatNP_simple(
            x_dim=x_dim,
            y_dim=y_dim,
            r_dim=r_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            activation=nn.ReLU,
            dropout=0.0,
            is_normalized=True,
            norm_type='layer',
            fourier_vars=3,
            num_frequencies=32,
            fourier_scale=1.0,
            learnable_fourier=True,
            use_deeponet_decoder=True,
            parameter_estimation=3 if ESTIMATE_PARAMS else 0
        ).to(device)
        
        
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining on full-dimensional data:")
    print(f"  State dimension: {nstate}")
    print(f"  Time steps: {ntimes}")
    print(f"  Total spatiotemporal points per trajectory: {nstate * ntimes}")
    print(f"\nModel architecture:\n{model}")
    
    # DATA STATISTICS AND SANITY CHECKS
    print("=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"\n1. Spatiotemporal Grid:")
    print(f"   - State dimension (nstate): {nstate}")
    print(f"   - Time steps (ntimes): {ntimes}")
    print(f"   - Total points per trajectory: {nstate * ntimes:,}")
    print(f"\n2. Dataset Sizes:")
    print(f"   - Training: {ntrain} trajectories")
    print(f"   - Validation: {nvalid} trajectories")
    print(f"   - Test: {ntest} trajectories")
    print(f"\n3. Data Ranges:")
    print(f"   - Ytrain: [{Ytrain.min():.4f}, {Ytrain.max():.4f}]")
    print(f"   - Mean: {Ytrain.mean():.4f}, Std: {Ytrain.std():.4f}")
    print(f"\n4. Training Configuration:")
    batch_sample = next(iter(train_loader))
    print(f"   - Batch size: {batch_sample[0].shape[0]} trajectories")
    print(f"   - Context points per trajectory: {batch_sample[0].shape[1]}")
    print(f"   - Target points per trajectory: {batch_sample[2].shape[1]}")
    print(f"   - Memory per batch (approx): ~{(batch_sample[0].numel() + batch_sample[2].numel()) * 4 / 1e6:.1f} MB")
    print(f"\n5. Context Coverage:")
    max_context = 10 * ntimes
    print(f"   - Max context points: {max_context} (10 sensors × {ntimes} times)")
    print(f"   - Coverage: {max_context / (nstate * ntimes) * 100:.3f}% of trajectory")
    print(f"   - This is SPARSE - testing generalization!")
    print(f"   - Context structure: FIXED sensors across time (more realistic)")

    criterion = ELBOLossNP(beta=1.0) # Base criterion (beta overriden by schedules below)

    # =====================================================================
    # PHASE 1: DETERMINISTIC WARMUP (Flat LR, Latent Frozen, Beta = 0)
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: DETERMINISTIC WARMUP (500 Epochs)")
    print("=" * 60)
    
    epochs_p1 = 500
    
    # 1. Freeze the latent space
    # for param in model.latent.parameters():
    #     param.requires_grad = False
        
    # 2. Setup Phase 1 Optimizer (Flat LR: 2e-4, passing only unfrozen parameters)
    optimizer_p1 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=0.0)
    
    # 3. Setup Phase 1 Beta Schedule (Strictly 0.0)
    beta_schedule_p1 = [0.0] * epochs_p1
    
    # Create a specific directory for Phase 1 checkpoints
    # p1_checkpoints_dir = checkpoints_dir / "phase1"
    # p1_checkpoints_dir.mkdir(exist_ok=True)
    
    # train_np(
    #     train_loader=train_loader,
    #     model=model,
    #     optimizer=optimizer_p1,
    #     loss_fn=criterion,
    #     device=device,
    #     epochs=epochs_p1,
    #     val_loader=val_loader,
    #     scheduler=None,           # Keep learning rate perfectly flat
    #     gradient_clip=1.0,
    #     early_stopping_patience=1000, # Disable early stopping during warmup
    #     is_meta_learning=True,
    #     verbose=True,
    #     print_every=10,
    #     checkpoint_dir=str(p1_checkpoints_dir),
    #     beta_schedule=beta_schedule_p1,
    #     early_stopping_start_epoch=epochs_p1 + 1 
    # )
    
    # # Save a guaranteed complete Phase 1 checkpoint
    # phase1_complete_path = checkpoints_dir / "phase1_complete.pt"
    # torch.save(model.state_dict(), phase1_complete_path)
    # print(f"\nPhase 1 Complete! Model saved to {phase1_complete_path}")


    # # =====================================================================
    # # PHASE 2: STOCHASTIC FINE-TUNING (LR Spike+Decay, Latent Active, Beta Ramp)
    # # =====================================================================
    # print("\n" + "=" * 60)
    # print("PHASE 2: STOCHASTIC FINE-TUNING (4500 Epochs)")
    # print("=" * 60)
    
    # epochs_p2 = 4500
    # ramp_epochs = 1000
    
    # # Note: I strongly recommend 0.1 instead of 1.0 to prevent the collapse you saw earlier.
    # # If you want to try 1.0 again, change this variable.
    # beta_target = 1.0 
    
    # # 1. Reload the best warmup weights (safety net)
    # model.load_state_dict(torch.load(phase1_complete_path, map_location=device))
    
    # # 2. Unfreeze the latent space!
    # for param in model.latent.parameters():
    #     param.requires_grad = True
        
    # # 3. Setup Phase 2 Optimizer (THE SPIKE: Jump up to 5e-4)
    # optimizer_p2 = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.0)
    
    # # 4. Setup Phase 2 Scheduler (THE DECAY: Cosine curve from 5e-4 down to 1e-6)
    # scheduler_p2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_p2, mode='min', factor=0.5, patience=128, min_lr=1e-6)
    
    # # 5. Setup Phase 2 Beta Schedule (Ramp up to target, then hold)
    # beta_schedule_p2 = [beta_target * (e / ramp_epochs) for e in range(ramp_epochs)] + \
    #                    [beta_target] * (epochs_p2 - ramp_epochs)
                       
    # # Create specific directory for Phase 2 checkpoints
    p2_checkpoints_dir = checkpoints_dir / "phase2"
    p2_checkpoints_dir.mkdir(exist_ok=True)
    
    # history = train_np(
    #     train_loader=train_loader,
    #     model=model,
    #     optimizer=optimizer_p2,
    #     loss_fn=criterion,
    #     device=device,
    #     epochs=epochs_p2,
    #     val_loader=val_loader,
    #     scheduler=scheduler_p2,
    #     gradient_clip=1.0,
    #     early_stopping_patience=1000, 
    #     is_meta_learning=True,
    #     verbose=True,
    #     print_every=10,
    #     checkpoint_dir=str(p2_checkpoints_dir),
    #     beta_schedule=beta_schedule_p2,
    #     early_stopping_start_epoch=ramp_epochs # Don't allow early stopping until the beta ramp is finished
    # )

    # print("\nTraining completely finished!")
    # print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    # print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    # print(f"Final KL divergence: {history['train_kl'][-1]:.4f}")
    # print(f"Final reconstruction: {history['train_recon'][-1]:.4f}")
    
    # Load the absolute best model from Phase 2 for testing
    best_model_path = p2_checkpoints_dir / "best_model.pt"
    checkpoint = torch.load(best_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    # ============================================================
    # TEST SET PREDICTIONS WITH BATCH FROM 3 TEST SAMPLES
    # ============================================================

    # ============================================================
    # TEST SET PREDICTIONS WITH BATCH FROM 3 TEST SAMPLES
    # ============================================================

    model.eval()

    test_indices = np.array([0, 1, 2])
    test_batch = [test_dataset[idx] for idx in test_indices]
    
    # Define a fixed time index to observe. Ensure this is >= 20 so a lag of 20 is valid
    eval_time_idx = 30 
    
    for current_lag in [0, 9, 19]:
        print(f"\n{'='*60}")
        print(f"BATCH TESTING WITH 3 TEST SAMPLES | TIME: {eval_time_idx} | LAG: {current_lag}")
        print(f"{'='*60}")

        collate_out = spatiotemporal_test_collate_fn(
            test_batch,
            mesh_coords=mesh_coordinates_norm,
            fixed_sensor_locations=fixed_sens,
            use_all_sensors=True,
            time_idx=eval_time_idx,
            lag=current_lag,
            return_mu_as_label=ESTIMATE_PARAMS
        )
        x_context, y_context, x_target, y_target = collate_out[:4]
        theta_true_batch = collate_out[4].cpu() if len(collate_out) >= 5 else None  # (B, param_dim)

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target  = x_target.to(device)
        y_target  = y_target.to(device)

        # Inference
        num_mc_samples = 100
        with torch.no_grad():
            result = model(x_context, y_context, x_target, num_samples=num_mc_samples)
            y_pred_mean, y_pred_var = result[0], result[1]
            # param outputs present when parameter_estimator is not None
            # inference path returns: y_mu, y_var, z_c_mu, z_c_var, [param_mu, param_var]
            param_mu_mc  = result[4].cpu() if len(result) >= 6 else None  # (S, B, param_dim) or (B, param_dim)
            param_var_mc = result[5].cpu() if len(result) >= 6 else None
        
        # After plt.close(ll_fig), still inside for batch_idx loop:

        
        # Squeeze feature dim and move to CPU
        # Shapes: (M, batch_size, num_target)
        y_pred_mc = y_pred_mean.squeeze(-1).cpu()
        y_pred_var_mc = y_pred_var.squeeze(-1).cpu()
        y_target_cpu = y_target.squeeze(-1).cpu() 
        
        # Calculate raw physical predictions
        y_pred = y_pred_mc.mean(dim=0)
        y_pred_run_std = y_pred_mc.std(dim=0)
        y_pred_var_epistemic = y_pred_mc.var(dim=0, unbiased=False)
        y_pred_var_aleatoric = y_pred_var_mc.mean(dim=0)
        y_pred_var_total = y_pred_var_epistemic + y_pred_var_aleatoric

        # --- FIX: Calculate true Log-Likelihood of the MC Mixture ---
        var_clamp = y_pred_var_mc.clamp_min(1e-8)
        ll_const = math.log(2.0 * math.pi)

        # Broadcast y_target_cpu (batch, nodes) to match y_pred_mc (M, batch, nodes)
        sample_lls = -0.5 * (ll_const + torch.log(var_clamp) + ((y_target_cpu.unsqueeze(0) - y_pred_mc)**2) / var_clamp)

        # Safely LogSumExp across the M samples
        M = y_pred_mc.shape[0]
        log_lik_all = torch.logsumexp(sample_lls, dim=0) - math.log(M)

        ll_vmin_global = log_lik_all.min().item()
        ll_vmax_global = log_lik_all.max().item()
        if ll_vmax_global == ll_vmin_global:
            ll_vmax_global = ll_vmin_global + 1e-8

        plot_state_fixed = lambda y: plot_with_colorbar(
            y, Yh, cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value"
        )

        for batch_idx in range(3):
            ground_truth, _ = test_dataset[test_indices[batch_idx]] if isinstance(test_dataset[0], (tuple, list)) else (test_dataset[test_indices[batch_idx]], None)
            ground_truth = ground_truth.cpu()

            if USE_MU:
                context = x_context[batch_idx].cpu()[:, 4:6]  
            else:
                context = x_context[batch_idx].cpu()[:, 1:3]  
            
            sample_pred_runs = y_pred_mc[:, batch_idx, :]
            sample_pred = y_pred[batch_idx]
            sample_target = y_target_cpu[batch_idx]
            
            sample_total_var = y_pred_var_total[batch_idx]
            sample_total_std = torch.sqrt(sample_total_var.clamp_min(1e-8))
            sample_sq_error = (sample_pred - sample_target) ** 2
            sample_log_lik = log_lik_all[batch_idx]

            state_vmin = torch.min(sample_pred_runs.min(), sample_target.min()).item()
            state_vmax = torch.max(sample_pred_runs.max(), sample_target.max()).item()
            if state_vmax == state_vmin:
                state_vmax = state_vmin + 1e-8
            
            std_vmin, std_vmax = sample_total_std.min().item(), sample_total_std.max().item()
            if std_vmax == std_vmin: std_vmax = std_vmin + 1e-8
            
            sq_err_vmin, sq_err_vmax = sample_sq_error.min().item(), sample_sq_error.max().item()
            if sq_err_vmax == sq_err_vmin: sq_err_vmax = sq_err_vmin + 1e-8
            sq_err_vmin = 0.0  

            sample_sse = sample_sq_error / sample_total_var.clamp_min(1e-8)
            sse_vmin, sse_vmax = sample_sse.min().item(), sample_sse.max().item()
            if sse_vmax == sse_vmin: sse_vmax = sse_vmin + 1e-8
            sse_vmin = 0.0

            # ----------------------------------------------------
            # 1. 2x3 AGGREGATE METRICS GRID
            # ----------------------------------------------------
            fig, axes = plt.subplots(2, 3, figsize=(30, 16))
            
            # --- ROW 1 ---
            # Col 0: Ground Truth
            plot_with_colorbar(sample_target, Yh, ax=axes[0, 0], cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            axes[0, 0].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[0, 0].set_title("Truth", fontsize=25)
            axes[0, 0].axis('off')

            # Col 1: Mean Prediction
            plot_with_colorbar(sample_pred, Yh, ax=axes[0, 1], cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            axes[0, 1].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[0, 1].set_title(f"Mean Prediction ({num_mc_samples} MC Runs)", fontsize=25)
            axes[0, 1].axis('off')

            # Col 2: Squared Error (MSE)
            plot_with_colorbar(sample_sq_error, Yh, ax=axes[0, 2], cmap="magma", vmin=sq_err_vmin, vmax=sq_err_vmax, label="Squared Error (MSE)")
            axes[0, 2].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[0, 2].set_title("Squared Error", fontsize=25)
            axes[0, 2].axis('off')

            # --- ROW 2 ---
            # Col 0: Log-Likelihood Map
            plot_with_colorbar(sample_log_lik, Yh, ax=axes[1, 0], cmap="magma", vmin=ll_vmin_global, vmax=ll_vmax_global, label="Log-likelihood")
            axes[1, 0].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[1, 0].set_title("Log-likelihood map", fontsize=25)
            axes[1, 0].axis('off')

            # Col 1: Standardized SE
            plot_with_colorbar(sample_sse, Yh, ax=axes[1, 1], cmap="magma", vmin=sse_vmin, vmax=sse_vmax, label="Standardized SE")
            axes[1, 1].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[1, 1].set_title("Standardized SE", fontsize=25)
            axes[1, 1].axis('off')
            
            # Col 2: Standardized Deviation (Std)
            plot_with_colorbar(sample_total_std, Yh, ax=axes[1, 2], cmap="magma", vmin=std_vmin, vmax=std_vmax, label="Std")
            axes[1, 2].scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[1, 2].set_title("Standardized Deviation (Std)", fontsize=25)
            axes[1, 2].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            grid_path = logs_dir / f"multiplot_grid_2x3_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
            plt.savefig(grid_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # ----------------------------------------------------
            # 2. 10 INDIVIDUAL MONTE CARLO SAMPLES PLOT (2x5 Grid)
            # ----------------------------------------------------
            fig_mc, axes_mc = plt.subplots(2, 5, figsize=(25, 10))
            fig_mc.suptitle(f"10 Monte Carlo Samples | Sample {test_indices[batch_idx]} | Time: {eval_time_idx} | Lag: {current_lag}", fontsize=25)

            for i in range(10):
                row = i // 5
                col = i % 5
                plt.sca(axes_mc[row, col])
                plot_state_fixed(sample_pred_runs[i])
                plt.scatter(context[:, 0], context[:, 1], color='red', s=10)
                axes_mc[row, col].set_title(f"MC Run {i+1}", fontsize=25)
                axes_mc[row, col].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            mc_path = logs_dir / f"mc_10_samples_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
            plt.savefig(mc_path, dpi=300, bbox_inches="tight")
            plt.close(fig_mc)

            # ----------------------------------------------------
            # 3. LOG-LIKELIHOOD HISTOGRAM
            # ----------------------------------------------------
            ll_fig, ll_ax = plt.subplots(figsize=(6, 4))
            ll_ax.hist(sample_log_lik.cpu().numpy().flatten(), bins=50, edgecolor='black', alpha=0.7)
            ll_ax.set_title(f"Log-likelihood (truth | pred) - Lag {current_lag}", fontsize=25)
            ll_ax.set_xlabel("Log-likelihood", fontsize=25)
            ll_ax.set_ylabel("Frequency", fontsize=25)
            ll_ax.set_xlim(ll_vmin_global, ll_vmax_global)
            ll_ax.grid(True, alpha=0.3)
            ll_fig.tight_layout()
            
            ll_path = logs_dir / f"loglik_hist_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
            ll_fig.savefig(ll_path, dpi=300, bbox_inches="tight")
            plt.close(ll_fig)
            
            # ----------------------------------------------------
            # 4. PARAMETER DISTRIBUTION (MC Samples)
            # ----------------------------------------------------
            if param_mu_mc is not None:
                # param_mu_mc: (S, B, param_dim) or (B, param_dim) when num_samples==1
                if param_mu_mc.dim() == 3:
                    # (S, B, param_dim) -> for this batch item: (S, param_dim)
                    sample_params = param_mu_mc[:, batch_idx, :]   # (100, param_dim)
                    sample_param_var = param_var_mc[:, batch_idx, :]
                else:
                    # squeezed case — shouldn't happen with num_mc_samples=100 but handle it
                    sample_params = param_mu_mc[batch_idx].unsqueeze(0)
                    sample_param_var = param_var_mc[batch_idx].unsqueeze(0)

                param_dim = sample_params.shape[-1]
                param_names = [f"θ_{i+1}" for i in range(param_dim)]

                # --- True parameter values for this sample (if available) ---
                true_params = None
                if theta_true_batch is not None:
                    true_params = theta_true_batch[batch_idx].squeeze()

                fig_p, axes_p = plt.subplots(1, param_dim, figsize=(5 * param_dim, 4))
                if param_dim == 1:
                    axes_p = [axes_p]

                fig_p.suptitle(
                    f"Parameter Posterior | Sample {test_indices[batch_idx]} | "
                    f"Time {eval_time_idx} | Lag {current_lag}",
                    fontsize=16
                )

                for pi in range(param_dim):
                    ax = axes_p[pi]
                    vals = sample_params[:, pi].numpy()          # (100,) MC draws of mu
                    stds = torch.sqrt(sample_param_var[:, pi].clamp_min(1e-8)).numpy()

                    # Histogram of the 100 predicted means
                    ax.hist(vals, bins=20, density=True, alpha=0.6, color='steelblue',
                            edgecolor='white', label='MC sample means')

                    # Overlay a Gaussian fitted to the MC spread
                    from scipy.stats import norm as sp_norm
                    x_range = np.linspace(vals.min() - 3*vals.std(), vals.max() + 3*vals.std(), 200)
                    ax.plot(x_range, sp_norm.pdf(x_range, vals.mean(), vals.std()),
                            'steelblue', lw=2, linestyle='--', label='Fitted Gaussian')

                    # Shade mean ± 1 aleatoric std (average predicted uncertainty)
                    mean_aleatoric_std = stds.mean()
                    ax.axvspan(vals.mean() - mean_aleatoric_std,
                            vals.mean() + mean_aleatoric_std,
                            alpha=0.15, color='orange', label=f'±1 aleatoric std')

                    # Predicted mean
                    ax.axvline(vals.mean(), color='steelblue', lw=2, label=f'Pred mean: {vals.mean():.3f}')

                    # True value
                    if true_params is not None:
                        ax.axvline(true_params[pi].item(), color='red', lw=2,
                                linestyle='--', label=f'True: {true_params[pi].item():.3f}')

                    ax.set_title(param_names[pi], fontsize=13)
                    ax.set_xlabel("Parameter value", fontsize=11)
                    ax.set_ylabel("Density", fontsize=11)
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)

                fig_p.tight_layout(rect=[0, 0, 1, 0.93])
                param_path = logs_dir / f"param_dist_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
                fig_p.savefig(param_path, dpi=300, bbox_inches="tight")
                plt.close(fig_p)
        # Batch Aggregation 
        abs_error_batch = torch.abs(y_pred - y_target_cpu)
        rel_error_batch = abs_error_batch / (torch.abs(y_target_cpu) + 1e-8)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(abs_error_batch.numpy().flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_title(f'Absolute Error Distribution\n(Time {eval_time_idx}, Lag {current_lag})', fontsize=25)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(rel_error_batch.numpy().flatten(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_title(f'Relative Error Distribution\n(Time {eval_time_idx}, Lag {current_lag})', fontsize=25)
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        
        err_dist_path = logs_dir / f"test_batch_error_distribution_time{eval_time_idx}_lag{current_lag}.png"
        plt.savefig(err_dist_path, dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_target_cpu.numpy().flatten(), y_pred.numpy().flatten(), alpha=0.3, s=1)
        all_vals = torch.cat([y_target_cpu.flatten(), y_pred.flatten()])
        min_val, max_val = all_vals.min().item(), all_vals.max().item()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        ax.set_title(f'Prediction vs Ground Truth\n(Time {eval_time_idx}, Lag {current_lag})', fontsize=25)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        pred_vs_truth_path = logs_dir / f"test_batch_pred_vs_truth_time{eval_time_idx}_lag{current_lag}.png"
        plt.savefig(pred_vs_truth_path, dpi=300, bbox_inches="tight")
        plt.close()
        
    print("\n" + "="*60)
    print("STARTING FULL TEST SET EVALUATION...")
    print("="*60)

    import matplotlib.patches as mpatches

    # --- Plotting Functions ---
    METHOD_STYLES = {}

    def _plot_row(ax_hist, ax_box, data_dict, title, xlabel, bins, clip_pct, log_scale=False):
        if not data_dict: return

        all_vals = np.concatenate([v for v in data_dict.values()])
        x_lo, x_hi = np.percentile(all_vals, clip_pct), np.percentile(all_vals, 100 - clip_pct)
        legend_handles = []

        for name, vals in data_dict.items():
            style = METHOD_STYLES.get(name, {})
            clipped = vals[(vals >= x_lo) & (vals <= x_hi)]
            eff_bins = min(bins, max(10, len(clipped) // 2))

            counts, edges = np.histogram(clipped, bins=eff_bins, density=True)
            ax_hist.hist(clipped, bins=eff_bins, density=True, color=style["color"], alpha=style["alpha"]*0.55)
            ax_hist.plot(0.5*(edges[:-1]+edges[1:]), counts, color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"])
            ax_hist.axvline(float(np.median(vals)), color=style["color"], linewidth=1.2, linestyle="--")
            legend_handles.append(mpatches.Patch(color=style["color"], label=name))

        ax_hist.set_xlim(x_lo, x_hi)
        ax_hist.set_xlabel(xlabel, fontsize=12)
        ax_hist.set_ylabel("Density", fontsize=12)
        ax_hist.set_title(title, fontsize=15)
        ax_hist.tick_params(axis='both', labelsize=12) 
        ax_hist.legend(handles=legend_handles, framealpha=0.85, fontsize=12) 

        bp_data, bp_names = list(data_dict.values()), list(data_dict.keys())
        flier_style = dict(marker='o', markerfacecolor='black', markersize=2, alpha=0.1, linestyle='none', markeredgecolor='none')

        bplot = ax_box.boxplot(bp_data, vert=True, patch_artist=True, notch=True, showfliers=False, flierprops=flier_style)

        for patch, colour in zip(bplot["boxes"], [METHOD_STYLES[n]["color"] for n in bp_names]):
            patch.set_facecolor(colour); patch.set_alpha(0.70)

        ax_box.set_xticks(range(1, len(bp_names) + 1))
        ax_box.set_xticklabels(bp_names, rotation=0, fontsize=10)
        ax_box.set_ylabel(xlabel, fontsize=12)
        ax_box.set_title(f"Box Plot: {title}", fontsize=15)
        ax_box.tick_params(axis='both', labelsize=12) 

        if log_scale == True or log_scale == 'log':
            ax_box.set_yscale('log')
        elif log_scale == 'symlog':
            ax_box.set_yscale('symlog')

    def plot_all_distributions(ll_dict, se_dict, sse_dict, mse_dict, out_path, bins=80, clip_pct=5):
        fig, axes = plt.subplots(4, 2, figsize=(18, 15), gridspec_kw={"width_ratios": [2, 1.25]})

        _plot_row(axes[0, 0], axes[0, 1], ll_dict, "Log-Likelihood Distribution", "Per-node log-likelihood", bins, clip_pct, log_scale=False)
        _plot_row(axes[1, 0], axes[1, 1], sse_dict, "Standardized Squared Errors (SSE)", "Per-sample SSE", bins, clip_pct, log_scale=False)
        _plot_row(axes[2, 0], axes[2, 1], se_dict, "Squared Error (SE) Distribution", "Per-node Squared Error", bins, clip_pct * 2, log_scale=False)
        _plot_row(axes[3, 0], axes[3, 1], mse_dict, "Mean Squared Errors (MSE)", "Per-sample MSE", bins, clip_pct, log_scale=False)

        fig.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.4, wspace=0.25)
        plt.show()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    # --- Reusable Evaluation Engine (Without Un-normalization) ---
    def evaluate_scenario(dataset, time_idx, lag, sensors_to_use, drop_options, mc_samples=100):
        all_ll, all_se, all_sse, all_mse = [], [], [], []

        for idx in range(len(dataset)):
            test_batch = [dataset[idx]]

            test_out = spatiotemporal_test_collate_fn(
                test_batch,
                mesh_coords=mesh_coordinates_norm,
                fixed_sensor_locations=sensors_to_use,
                use_all_sensors=True,
                drop_random_sensors_options=drop_options,
                time_idx=time_idx,
                lag=lag,
                return_mu_as_label=ESTIMATE_PARAMS
            )
            
            x_c, y_c, x_t, y_t = test_out[:4]
            theta_true_batch = test_out[4].cpu() if len(test_out) >= 5 else None  # (B, param_dim)

            x_c = x_c.to(device)
            y_c = y_c.to(device)
            x_t  = x_t.to(device)

            x_c, y_c, x_t = x_c.to(device), y_c.to(device), x_t.to(device)

            with torch.no_grad():
                mu_mc, var_mc, *_ = model(x_c, y_c, x_t, num_samples=mc_samples)
                
                # Strip feature and batch dimensions (Batch is 1 here)
                mc_means = mu_mc.squeeze(-1).squeeze(1).cpu()   # Shape: (M, num_target)
                mc_vars = var_mc.squeeze(-1).squeeze(1).cpu()   # Shape: (M, num_target)
                y_true = y_t.squeeze(-1).squeeze(0).cpu()       # Shape: (num_target)

                # Aggregate predictions
                pred_mean = mc_means.mean(dim=0)
                pred_var = mc_vars.mean(dim=0) + mc_means.var(dim=0, unbiased=False)

                # --- CALCULATE TRUE MIXTURE LOG-LIKELIHOOD ---
                var_clamp = mc_vars.clamp_min(1e-8)
                ll_const = math.log(2.0 * math.pi)
                
                # Sample-wise LL: Shape (M, num_target)
                sample_lls = -0.5 * (ll_const + torch.log(var_clamp) + ((y_true.unsqueeze(0) - mc_means)**2) / var_clamp)
                
                # Safely LogSumExp across M samples
                M = mc_means.shape[0]
                ll = torch.logsumexp(sample_lls, dim=0) - math.log(M)

                # --- CALCULATE ERRORS ---
                se = (y_true - pred_mean) ** 2
                sse = se / pred_var.clamp_min(1e-8)

            all_ll.append(ll.flatten())
            all_se.append(se.flatten())
            all_sse.append(sse.flatten())
            all_mse.append(se.mean().view(1))

            del x_c, y_c, x_t, y_t

        return (
            torch.cat(all_ll).cpu().numpy(),
            torch.cat(all_se).cpu().numpy(),
            torch.cat(all_sse).cpu().numpy(),
            torch.cat(all_mse).cpu().numpy()
        )

    # --- SCENARIO A: Same Sensors, Lags = 0, 9, 19 ---
    print("\nRunning Scenario A: Varying Lags (0, 9, 19)...")

    ll_dict_A, se_dict_A, sse_dict_A, mse_dict_A = {}, {}, {}, {}
    lags_to_test = [0, 9, 19]
    colors_A = ["#E63946", "#457B9D", "#2A9D8F"]

    for i, lag in enumerate(lags_to_test):
        label = f"NP (Lag {lag + 1})"
        print(f"  Evaluating {label}...")

        METHOD_STYLES[label] = {"color": colors_A[i], "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            dataset=test_dataset,
            time_idx=30,
            lag=lag,
            sensors_to_use=fixed_sens,
            drop_options=[0]
        )

        ll_dict_A[label], se_dict_A[label], sse_dict_A[label], mse_dict_A[label] = ll, se, sse, mse

    print("Generating plots for Scenario A...")
    plot_all_distributions(ll_dict_A, se_dict_A, sse_dict_A, mse_dict_A, out_path=logs_dir / "diagnostics_lags_0_9_19.png")

    # --- SCENARIO B: Fixed Lag 9, Drop 1 Sensor ---
    print("\nRunning Scenario B: Lag 9, Dropping Sensors...")

    ll_dict_B, se_dict_B, sse_dict_B, mse_dict_B = {}, {}, {}, {}

    # Updated configs_B to match your `fixed_sens` array: [1573, 6925, 1986]
    configs_B = [
        ("All Sensors",         [1573, 6925, 1986], "#000000"),
        ("Missing Sensor 1573", [6925, 1986], "#E63946"),
        ("Missing Sensor 6925", [1573, 1986], "#F4A261"),
        ("Missing Sensor 1986", [1573, 6925], "#2A9D8F")
    ]

    for label, sens_list, color in configs_B:
        print(f"  Evaluating {label}...")

        METHOD_STYLES[label] = {"color": color, "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            dataset=test_dataset,
            time_idx=30,
            lag=9,
            sensors_to_use=sens_list,
            drop_options=[0]
        )

        ll_dict_B[label], se_dict_B[label], sse_dict_B[label], mse_dict_B[label] = ll, se, sse, mse

    print("Generating plots for Scenario B...")
    plot_all_distributions(ll_dict_B, se_dict_B, sse_dict_B, mse_dict_B, out_path=logs_dir / "diagnostics_lag9_drop_sensors.png")

    # --- Plot Ground Truth with Sensor Annotations ---
    print("\nGenerating final ground truth plot with sensor locations...")
    time_idx = 30
    sample_idx = 0

    if torch.is_tensor(Ytest):
        truth_field = Ytest[sample_idx, time_idx].cpu().numpy()
    else:
        truth_field = Ytest[sample_idx, time_idx]

    sensor_coords = mesh_coordinates[fixed_sens].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_with_colorbar(truth_field, Yh, ax=ax, cmap="jet", label="True State")

    for i, sensor_idx in enumerate(fixed_sens):
        x, y = sensor_coords[i, 0], sensor_coords[i, 1]

        ax.scatter(x, y, color='red', s=80, marker='X', edgecolor='black', linewidth=1.5, zorder=5)

        ax.annotate(str(sensor_idx),
                    (x, y),
                    xytext=(8, 8), textcoords='offset points',
                    color='black', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.9),
                    zorder=6)

    ax.set_title(f"Ground Truth (Test Trajectory {sample_idx}, Time = {time_idx}) with Sensor Locations")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    plt.tight_layout()
    plt.show()
    fig.savefig(logs_dir / "ground_truth_sensors.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()