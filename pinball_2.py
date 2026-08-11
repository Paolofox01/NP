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
from LNP.LatentNP import LatNP
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np
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
                              drop_random_sensors=0, drop_random_sensors_options=None,
                              return_mu_as_label=False):
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

    # 2. Sample temporal locations (Once per batch)
    time_idx = np.random.randint(20, ntimes - 1)  
    lag = np.random.choice([0, 4, 9, 19])  
    
    time_window = torch.arange(time_idx - lag, time_idx + 1)
    context_time_indices = time_window.repeat_interleave(num_sensors)
    target_time_indices = torch.full((total_locations,), time_idx, dtype=torch.long)
    context_state_indices = sensor_locations.repeat(1 + lag)

    # ====================================================================
    # 3. Build Base Coordinates and Times (RELATIVE TIME SHIFT)
    # Target time is ALWAYS exactly 0.0
    # Context times are shifted backwards (negative) relative to the target
    # ====================================================================
    norm_context_time = ((context_time_indices - time_idx).float() / ntimes).unsqueeze(1)
    norm_target_time = torch.zeros((total_locations, 1), dtype=torch.float32)
    
    context_coords = mesh_coords[context_state_indices]
    target_coords = mesh_coords[target_state_indices]

    # Shape: (num_points, 3)
    x_ctx_base = torch.cat([norm_context_time, context_coords], dim=-1)
    x_tgt_base = torch.cat([norm_target_time, target_coords], dim=-1)

    # Expand to match batch size: (B, num_points, 3)
    x_context = x_ctx_base.unsqueeze(0).expand(batch_size, -1, -1)
    x_target = x_tgt_base.unsqueeze(0).expand(batch_size, -1, -1)

    # 4. Handle Mu parameters efficiently
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

    # 5. Extract Y values natively
    y_context = batch_trajectories[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = batch_trajectories[:, target_time_indices, target_state_indices].unsqueeze(-1)

    if batch_mu is not None and return_mu_as_label:
        theta = batch_mu if batch_mu.dim() == 2 else batch_mu[:, 0, :]
        return (
            x_context.contiguous(),
            y_context.contiguous(),
            x_target.contiguous(),
            y_target.contiguous(),
            theta.contiguous(),
        )

    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()

def spatiotemporal_test_collate_fn(batch, num_context_sensors_min=2, num_context_sensors_max=10, 
                                   mesh_coords=None, fixed_sensor_locations=None,
                                   use_all_sensors=False, drop_random_sensors=0,
                                   drop_random_sensors_options=None,
                                   time_idx=None, lag=None, return_mu_as_label=False):
    """
    Collate function for spatiotemporal Neural Process TESTING.
    Targets ALL states. Context sensors can be randomly sampled or explicitly provided.
    Fully vectorized to support multiprocessing and optional 'mu' parameters.
    """
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

    # --- 1. DETERMINE SENSOR CONFIGURATION ---
    chosen_drop = choose_drop_random_sensors(drop_random_sensors, drop_random_sensors_options)
    sensor_locations_fixed = select_sensor_locations(
        fixed_sensor_locations,
        nstate,
        use_all_sensors=use_all_sensors,
        drop_random_sensors=chosen_drop,
    )
    
    if sensor_locations_fixed is not None:
        sensor_locations = sensor_locations_fixed
        num_sensors = sensor_locations.numel()
    else:
        num_sensors = np.random.randint(num_context_sensors_min, num_context_sensors_max + 1)
        sensor_locations = torch.randperm(nstate)[:num_sensors]

    # Override random choices if explicit values are provided
    if time_idx is None:
        time_idx = np.random.randint(20, ntimes - 1) 
    if lag is None:
        lag = np.random.choice([0, 4, 9, 19]) 

    # --- 2. PREPARE INDICES (Vectorized) ---
    time_window = torch.arange(time_idx - lag, time_idx + 1)
    context_time_indices = time_window.repeat_interleave(num_sensors)
    context_state_indices = sensor_locations.repeat(1 + lag)
    
    target_time_indices = torch.full((nstate,), time_idx, dtype=torch.long)
    target_state_indices = torch.arange(nstate)

    # ====================================================================
    # --- 3. BUILD BASE COORDINATES (RELATIVE TIME SHIFT) ---
    # Target time is ALWAYS exactly 0.0
    # Context times are shifted backwards (negative) relative to the target
    # ====================================================================
    norm_context_time = ((context_time_indices - time_idx).float() / ntimes).unsqueeze(1)
    norm_target_time = torch.zeros((nstate, 1), dtype=torch.float32)
    
    context_coords = mesh_coords[context_state_indices]
    target_coords = mesh_coords[target_state_indices]

    x_ctx_base = torch.cat([norm_context_time, context_coords], dim=-1)
    x_tgt_base = torch.cat([norm_target_time, target_coords], dim=-1)

    # Expand to match batch size
    x_context = x_ctx_base.unsqueeze(0).expand(batch_size, -1, -1)
    x_target = x_tgt_base.unsqueeze(0).expand(batch_size, -1, -1)

    # --- 4. HANDLE MU PARAMETERS ---
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

    # --- 5. EXTRACT Y TARGETS NATIVELY ---
    y_context = batch_trajectories[:, context_time_indices, context_state_indices].unsqueeze(-1)
    y_target = batch_trajectories[:, target_time_indices, target_state_indices].unsqueeze(-1)

    if batch_mu is not None and return_mu_as_label:
        theta = batch_mu if batch_mu.dim() == 2 else batch_mu[:, 0, :]
        return (
            x_context.contiguous(),
            y_context.contiguous(),
            x_target.contiguous(),
            y_target.contiguous(),
            theta.contiguous(),
        )

    return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()

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

def run_experiment(USE_MU, USE_DEEPONET_DECODER=False):
    
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / f"logs_pinball_{'dndec' if USE_DEEPONET_DECODER else 'no_dndec'}_{'mu' if USE_MU else 'no_mu'}_new_5sens"
    checkpoints_dir = script_dir / f"checkpoints_pinball_{'dndec' if USE_DEEPONET_DECODER else 'no_dndec'}_{'mu' if USE_MU else 'no_mu'}_new_5sens"
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
    # fixed_sens = [3496, 3595, 5240, 6180, 1521]

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
                        drop_random_sensors_options=[0, 1])
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
                        drop_random_sensors_options=[0, 1])
    )

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Test the collate function
    x_c, y_c, x_t, y_t = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"Context y range: [{y_c.min():.3f}, {y_c.max():.3f}]")

    print(f"  Context: x_c {x_c.shape}, y_c {y_c.shape}")
    print(f"\nContext x range: [{x_c.min():.3f}, {x_c.max():.3f}]")

    print(f"  Target:  x_t {x_t.shape}, y_t {y_t.shape}")
    
    # Model hyperparameters - For full-dimensional spatiotemporal data
    if USE_MU:
        x_dim = 6
    else:
        x_dim = 3  # Input dimension: [time, x, y] coordinates (actual mesh coordinates!)
    y_dim = 1  # Output dimension: state value
    r_dim = 128  # Representation dimension
    z_dim = 128 # Latent dimension
    hidden_dim = 128  # Hidden layer dimension
    n_hidden = 2  # Number of hidden layers

    # Create model
    model = LatNP(
        x_dim=x_dim,
        y_dim=y_dim,
        r_dim=r_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation=nn.ReLU,
        dropout=0.0,
        is_normalized=True,  # Layer normalization helps training stability
        norm_type='layer',
        fourier_vars = 3,
        num_frequencies = 32, #number of fourier features for the spatial coordinates (x,y)
        num_heads = 4, #number of attention heads in the cross-attention module
        fourier_scale = 1.0, #scale of the fourier features for the spatial coordinates (x,y)
        learnable_fourier = True, #whether to learn the fourier features for the spatial coordinates (x,y)
        use_skip = True, #whether to use skip connections in the decoder
        use_deeponet_decoder = USE_DEEPONET_DECODER, #whether to use a DeepONet-style decoder
    ).to(device)
    
        
    num_epochs   = 3000
    beta_target  = 1.0   # final KL weight — low enough to avoid collapse, high enough to regularise
    warmup_steps = 1500   # linearly ramp beta from 0 → beta_target over the first 1000 epochs
    
    # Loss function
    criterion = ELBOLossNP(beta=1.0)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=0.0)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=128, min_lr=1e-5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Beta (KL weight): {criterion.beta}")
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
    print(f"   - Batch size: 32 trajectories")
    print(f"   - Context points per trajectory: {batch_sample[0].shape[1]}")
    print(f"   - Target points per trajectory: {batch_sample[2].shape[1]}")
    print(f"   - Memory per batch (approx): ~{(batch_sample[0].numel() + batch_sample[2].numel()) * 4 / 1e6:.1f} MB")

    print(f"\n5. Context Coverage:")
    max_context = 10 * ntimes  # 10 sensors × ntimes
    print(f"   - Max context points: {max_context} (10 sensors × {ntimes} times)")
    print(f"   - Coverage: {max_context / (nstate * ntimes) * 100:.3f}% of trajectory")
    print(f"   - This is SPARSE - testing generalization!")
    print(f"   - Context structure: FIXED sensors across time (more realistic)")

    # Fixed reference scale for test-time noise injection (Scenario C)
    training_noise_std = float(Ytrain.std().item())
    print(f"   - Fixed noise std from training data: {training_noise_std:.6f}")

    print("\n" + "=" * 60)

    beta0 = 500
            
    if num_epochs > warmup_steps and warmup_steps > beta0:
        # Build per-epoch beta schedule: linear warmup, then constant
        beta_schedule = (
            [0.0] * beta0 +
            [beta_target * (e / (warmup_steps - beta0)) for e in range(0, warmup_steps - beta0)]   # ramp-up
            + [beta_target] * (num_epochs - warmup_steps)                      # constant
        )
    elif warmup_steps < beta0:
        beta_schedule = (
            [beta_target * (e / (warmup_steps)) for e in range(0, warmup_steps)]   # ramp-up
            + [beta_target] * (num_epochs - warmup_steps)                      # constant
        )
    else:
        beta_schedule = [beta_target] * num_epochs  # constant
        
        
    # best_model_path = checkpoints_dir / "final_model.pt"
    # checkpoint = torch.load(best_model_path, map_location=device)
    # if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    #     model.load_state_dict(checkpoint["model_state_dict"])
    # else:
    #     model.load_state_dict(checkpoint)
    # model.eval()
    
    # Train the model using the unified training function
    history = train_np(
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=criterion,
        device=device,
        epochs=num_epochs,  # Reduced: 1000 was excessive, monitor for convergence
        val_loader=val_loader,
        scheduler=scheduler,
        gradient_clip=1.0,  # Gradient clipping for stability
        early_stopping_patience=1000,
        is_meta_learning=True,  # Enable Neural Process mode
        verbose=False,
        print_every=10,  # Print every 10 epochs
        checkpoint_dir=str(checkpoints_dir),  # Directory to save checkpoints
        beta_schedule=beta_schedule,
        early_stopping_start_epoch=200,  # Don't consider early stopping until after 200 epochs
    )

    print("\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final KL divergence: {history['train_kl'][-1]:.4f}")
    print(f"Final reconstruction: {history['train_recon'][-1]:.4f}")
    
    best_model_path = checkpoints_dir / "best_model.pt"
    checkpoint = torch.load(best_model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # ============================================================
    # TEST SET PREDICTIONS WITH BATCH FROM 3 TEST SAMPLES
    # ============================================================

    model.eval()

    test_indices = np.array([0, 1, 2])
    test_batch = [test_dataset[idx] for idx in test_indices]
    
    # Define a fixed time index to observe. Ensure this is >= 20 so a lag of 20 is valid
    # Assuming ntimes > 25 for this dataset.
    eval_time_idx = 30 
    
    for current_lag in [0, 9, 19]:
        print(f"\n{'='*60}")
        print(f"BATCH TESTING WITH 3 TEST SAMPLES | TIME: {eval_time_idx} | LAG: {current_lag}")
        print(f"{'='*60}")

        x_context, y_context, x_target, y_target = spatiotemporal_test_collate_fn(
            test_batch,
            mesh_coords=mesh_coordinates_norm,
            fixed_sensor_locations=fixed_sens,
            use_all_sensors=True,
            time_idx=eval_time_idx,   # Pass explicit time index
            lag=current_lag           # Pass explicit lag
        )

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        # Inference
        num_mc_samples = 100
        with torch.no_grad():
            # Vectorized Monte Carlo sampling (runs all 100 samples in one pass)
            y_pred_mean, y_pred_var, _, _ = model(
                x_context, y_context, x_target, num_samples=num_mc_samples
            )
            
        # Squeeze the feature dimension (y_dim) and move to CPU
        # Shape: (num_mc_samples, batch_size, num_target)
        y_pred_mc = y_pred_mean.squeeze(-1).cpu()
        y_pred_var_mc = y_pred_var.squeeze(-1).cpu()
        
        # Un-normalize
        y_pred_mc = y_pred_mc #* scale_range.cpu()) + y_min.cpu()
        y_pred_var_mc = y_pred_var_mc # * (scale_range.cpu() ** 2)
        
        y_target_cpu = y_target.squeeze(-1).cpu() # * scale_range.cpu()) + y_min.cpu()

        y_pred = y_pred_mc.mean(dim=0)
        y_pred_run_std = y_pred_mc.std(dim=0)
        y_pred_var_epistemic = y_pred_mc.var(dim=0, unbiased=False)
        y_pred_var_aleatoric = y_pred_var_mc.mean(dim=0)
        y_pred_var_total = y_pred_var_epistemic + y_pred_var_aleatoric


        ll_var_mc = y_pred_var_mc.clamp_min(1e-8)
        ll_const_mc = torch.log(torch.tensor(2.0 * np.pi, dtype=ll_var_mc.dtype))
        ll_diff_mc = y_target_cpu.unsqueeze(0) - y_pred_mc
        per_mc_log_lik = -0.5 * (ll_const_mc + torch.log(ll_var_mc) + (ll_diff_mc ** 2) / ll_var_mc)
        log_lik_all = torch.logsumexp(per_mc_log_lik, dim=0) - np.log(per_mc_log_lik.shape[0])
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
                context = x_context[batch_idx].cpu()[:, 4:6]  # Extract spatial coordinates (x,y) for plotting
            else :
                context = x_context[batch_idx].cpu()[:, 1:3]  # Extract spatial coordinates (x,y) for plotting
            
            sample_pred_runs = y_pred_mc[:, batch_idx, :]
            sample_pred = y_pred[batch_idx]
            sample_target = y_target_cpu[batch_idx]
            
            # Extract Total Variance and Calculate Std / Squared Error
            sample_total_var = y_pred_var_total[batch_idx]
            sample_total_std = torch.sqrt(sample_total_var.clamp_min(1e-8))
            sample_sq_error = (sample_pred - sample_target) ** 2
            sample_log_lik = log_lik_all[batch_idx]

            state_vmin = torch.min(sample_pred_runs.min(), sample_target.min()).item()
            state_vmax = torch.max(sample_pred_runs.max(), sample_target.max()).item()
            if state_vmax == state_vmin:
                state_vmax = state_vmin + 1e-8
            
            # Calculate limits for colorbars
            std_vmin, std_vmax = sample_total_std.min().item(), sample_total_std.max().item()
            if std_vmax == std_vmin: std_vmax = std_vmin + 1e-8
            
            sq_err_vmin, sq_err_vmax = sample_sq_error.min().item(), sample_sq_error.max().item()
            if sq_err_vmax == sq_err_vmin: sq_err_vmax = sq_err_vmin + 1e-8
            sq_err_vmin = 0.0  # Best practice: anchor squared error at 0

            # ----------------------------------------------------
            # 1. 1x5 AGGREGATE METRICS GRID
            # ----------------------------------------------------
            fig, axes = plt.subplots(1, 5, figsize=(40, 6))
            
            # Col 0: Ground Truth
            plt.sca(axes[0])
            plot_with_colorbar(sample_target, Yh, cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[0].set_title("Truth", fontsize=25)
            axes[0].axis('off')

            # Col 1: Mean Prediction
            plt.sca(axes[1])
            plot_with_colorbar(sample_pred, Yh, cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[1].set_title(f"Mean Prediction ({num_mc_samples} MC Runs)", fontsize=25)
            axes[1].axis('off')

            # Col 2: Standard Deviation
            plt.sca(axes[2])
            plot_with_colorbar(sample_total_std, Yh, cmap="magma", vmin=std_vmin, vmax=std_vmax, label="Std")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[2].set_title("Standard Deviation", fontsize=25)
            axes[2].axis('off')

            # Col 3: Squared Error
            plt.sca(axes[3])
            plot_with_colorbar(sample_sq_error, Yh, cmap="magma", vmin=sq_err_vmin, vmax=sq_err_vmax, label="Squared Error")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[3].set_title("Squared Error", fontsize=25)
            axes[3].axis('off')

            # Col 4: Log-Likelihood
            plt.sca(axes[4])
            plot_with_colorbar(sample_log_lik, Yh, cmap="magma", vmin=ll_vmin_global, vmax=ll_vmax_global, label="Log-likelihood")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[4].set_title("Log-likelihood map", fontsize=25)
            axes[4].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            grid_path = logs_dir / f"multiplot_grid_1x5_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
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

    import math
    import matplotlib.patches as mpatches

    # --- 1. Math Helpers ---
    def gaussian_mixture_log_lik(mc_means, mc_vars, y_true):
        mc_vars = mc_vars.clamp_min(1e-8)
        ll_const = math.log(2.0 * math.pi)
        diff = y_true.unsqueeze(0) - mc_means
        per_mc_ll = -0.5 * (ll_const + torch.log(mc_vars) + (diff ** 2) / mc_vars)
        return torch.logsumexp(per_mc_ll, dim=0) - math.log(mc_means.shape[0])

    def compute_standardized_se(y_pred_mean, y_pred_var, y_true):
        var_clamp = y_pred_var.clamp_min(1e-8)
        return ((y_true - y_pred_mean)**2) / var_clamp

    # --- 2. Plotting Functions ---
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

        # --- AGGIORNAMENTO FONT: Istogramma ---
        ax_hist.set_xlim(x_lo, x_hi)
        ax_hist.set_xlabel(xlabel, fontsize=12)
        ax_hist.set_ylabel("Density", fontsize=12)
        ax_hist.set_title(title, fontsize=15)
        ax_hist.tick_params(axis='both', labelsize=12) # Dimensione numeri assi
        ax_hist.legend(handles=legend_handles, framealpha=0.85, fontsize=12) # Legenda leggermente più piccola per leggibilità

        # ---------------------------------------------------------
        # BOXPLOT FIX: Show fliers (outliers) so we can see DoN's failures
        # ---------------------------------------------------------
        bp_data, bp_names = list(data_dict.values()), list(data_dict.keys())
        flier_style = dict(marker='o', markerfacecolor='black', markersize=2, alpha=0.1, linestyle='none', markeredgecolor='none')

        bplot = ax_box.boxplot(bp_data, vert=True, patch_artist=True, notch=True, showfliers=False, flierprops=flier_style)

        for patch, colour in zip(bplot["boxes"], [METHOD_STYLES[n]["color"] for n in bp_names]):
            patch.set_facecolor(colour); patch.set_alpha(0.70)

        # --- AGGIORNAMENTO FONT: Box Plot ---
        ax_box.set_xticks(range(1, len(bp_names) + 1))
        ax_box.set_xticklabels(bp_names, rotation=0, fontsize=10)
        ax_box.set_ylabel(xlabel, fontsize=12)
        ax_box.set_title(f"Box Plot: {title}", fontsize=15)
        ax_box.tick_params(axis='both', labelsize=12) # Dimensione numeri assi

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

    # --- 3. Reusable Evaluation Engine ---
    def evaluate_scenario(dataset, time_idx, lag, sensors_to_use, drop_options, mc_samples=100,
                          context_noise_amplitude=0.0):
        all_ll, all_se, all_sse, all_mse = [], [], [], []

        for idx in range(len(dataset)):
            test_batch = [dataset[idx]]

            x_c, y_c, x_t, y_t = spatiotemporal_test_collate_fn(
                test_batch,
                mesh_coords=mesh_coordinates_norm,
                fixed_sensor_locations=sensors_to_use,
                use_all_sensors=True,
                drop_random_sensors_options=drop_options,
                time_idx=time_idx,
                lag=lag
            )

            x_c, y_c, x_t = x_c.to(device), y_c.to(device), x_t.to(device)

            if context_noise_amplitude > 0.0:
                noisy_context = context_noise_amplitude * training_noise_std * torch.randn_like(y_c)
                y_c = y_c + noisy_context

            with torch.no_grad():
                try:
                    mu_mc, var_mc, *_ = model(x_c, y_c, x_t, num_samples=mc_samples)
                    mc_means = mu_mc.squeeze(-1).cpu()
                    mc_vars = var_mc.squeeze(-1).cpu()
                except TypeError:
                    means, vars_ = zip(*[(m.squeeze(-1).cpu(), v.squeeze(-1).cpu()) for m, v, *_ in [model(x_c, y_c, x_t) for _ in range(mc_samples)]])
                    mc_means = torch.stack(means, 0)
                    mc_vars = torch.stack(vars_, 0)

                pred_mean = mc_means.mean(0).squeeze(0)
                pred_var = mc_vars.mean(0).squeeze(0) + mc_means.var(0, unbiased=False).squeeze(0)

                y_true = y_t.squeeze(-1).squeeze(0).cpu()

                ll = gaussian_mixture_log_lik(mc_means.squeeze(1), mc_vars.squeeze(1), y_true)
                se = (y_true - pred_mean) ** 2
                sse = compute_standardized_se(pred_mean, pred_var, y_true)

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

    # --- 4. SCENARIO A: Same Sensors, Lags = 0, 9, 19 ---
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

    # --- 5. SCENARIO B: Fixed Lag 9, Drop 1 Sensor ---
    print("\nRunning Scenario B: Lag 9, Dropping Sensors...")

    ll_dict_B, se_dict_B, sse_dict_B, mse_dict_B = {}, {}, {}, {}

    configs_B = [
        ("All Sensors",         [3496, 3595, 5240, 6180, 1521], "#000000"),
        ("Missing Sensor 3496", [3595, 5240, 6180, 1521], "#E63946"),
        ("Missing Sensor 3595", [3496, 5240, 6180, 1521], "#E9C46A"),
        ("Missing Sensor 6180",  [3496, 3595, 5240, 1521], "#2A9D8F"),
        ("Missing Sensor 5240", [3496, 3595, 6180, 1521], "#0F962697"),
        ("Missing Sensor 1521", [3496, 3595, 5240, 6180], "#FF3CA4")
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

    # --- 6. SCENARIO C: Fixed Lag 9, Vary Context Noise Amplitude ---
    print("\nRunning Scenario C: Lag 9, Varying Context Noise Amplitude...")

    ll_dict_C, se_dict_C, sse_dict_C, mse_dict_C = {}, {}, {}, {}
    noise_amplitudes = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]
    colors_C = ["#1D3557", "#2A9D8F", "#E9C46A", "#F4A261", "#E63946", "#FF3CA4"]

    scenario_c_summary = {
        "noise_amplitude": [],
        "mean_ll": [],
        "median_ll": [],
        "mean_mse": [],
        "median_mse": [],
    }

    def generate_noisy_multiplots_for_amplitude(noise_amp, sample_indices=(0, 1, 2),
                                                time_idx=30, lag=9, num_mc_samples=100):
        sample_batch = [test_dataset[idx] for idx in sample_indices]

        x_context, y_context, x_target, y_target = spatiotemporal_test_collate_fn(
            sample_batch,
            mesh_coords=mesh_coordinates_norm,
            fixed_sensor_locations=fixed_sens,
            use_all_sensors=True,
            time_idx=time_idx,
            lag=lag,
        )

        x_context = x_context.to(device)
        y_context = y_context.to(device)
        x_target = x_target.to(device)
        y_target = y_target.to(device)

        if noise_amp > 0.0:
            y_context = y_context + noise_amp * training_noise_std * torch.randn_like(y_context)

        with torch.no_grad():
            y_pred_mean, y_pred_var, _, _ = model(x_context, y_context, x_target, num_samples=num_mc_samples)

        y_pred_mc = y_pred_mean.squeeze(-1).cpu()
        y_pred_var_mc = y_pred_var.squeeze(-1).cpu()
        y_target_cpu = y_target.squeeze(-1).cpu()

        y_pred = y_pred_mc.mean(dim=0)
        y_pred_var_epistemic = y_pred_mc.var(dim=0, unbiased=False)
        y_pred_var_aleatoric = y_pred_var_mc.mean(dim=0)
        y_pred_var_total = y_pred_var_epistemic + y_pred_var_aleatoric

        ll_var_mc = y_pred_var_mc.clamp_min(1e-8)
        ll_const_mc = torch.log(torch.tensor(2.0 * np.pi, dtype=ll_var_mc.dtype))
        ll_diff_mc = y_target_cpu.unsqueeze(0) - y_pred_mc
        per_mc_log_lik = -0.5 * (ll_const_mc + torch.log(ll_var_mc) + (ll_diff_mc ** 2) / ll_var_mc)
        log_lik_all = torch.logsumexp(per_mc_log_lik, dim=0) - np.log(per_mc_log_lik.shape[0])

        ll_vmin_global = log_lik_all.min().item()
        ll_vmax_global = log_lik_all.max().item()
        if ll_vmax_global == ll_vmin_global:
            ll_vmax_global = ll_vmin_global + 1e-8

        for batch_idx, sample_idx in enumerate(sample_indices):
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
            if std_vmax == std_vmin:
                std_vmax = std_vmin + 1e-8

            sq_err_vmin, sq_err_vmax = sample_sq_error.min().item(), sample_sq_error.max().item()
            if sq_err_vmax == sq_err_vmin:
                sq_err_vmax = sq_err_vmin + 1e-8
            sq_err_vmin = 0.0

            fig, axes = plt.subplots(1, 5, figsize=(40, 6))

            plt.sca(axes[0])
            plot_with_colorbar(sample_target, Yh, cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[0].set_title("Truth", fontsize=25)
            axes[0].axis('off')

            plt.sca(axes[1])
            plot_with_colorbar(sample_pred, Yh, cmap="jet", vmin=state_vmin, vmax=state_vmax, label="State value")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[1].set_title(f"Mean Prediction ({num_mc_samples} MC Runs)", fontsize=25)
            axes[1].axis('off')

            plt.sca(axes[2])
            plot_with_colorbar(sample_total_std, Yh, cmap="magma", vmin=std_vmin, vmax=std_vmax, label="Std")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[2].set_title("Standard Deviation", fontsize=25)
            axes[2].axis('off')

            plt.sca(axes[3])
            plot_with_colorbar(sample_sq_error, Yh, cmap="magma", vmin=sq_err_vmin, vmax=sq_err_vmax, label="Squared Error")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[3].set_title("Squared Error", fontsize=25)
            axes[3].axis('off')

            plt.sca(axes[4])
            plot_with_colorbar(sample_log_lik, Yh, cmap="magma", vmin=ll_vmin_global, vmax=ll_vmax_global, label="Log-likelihood")
            plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
            axes[4].set_title("Log-likelihood map", fontsize=25)
            axes[4].axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            amp_tag = str(f"{noise_amp:.2f}").replace('.', 'p')
            noisy_grid_path = logs_dir / (
                f"noisy_multiplot_grid_1x5_sample{sample_idx}_time{time_idx}_lag{lag}_amp{amp_tag}.png"
            )
            plt.savefig(noisy_grid_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    for i, amp in enumerate(noise_amplitudes):
        label = f"Noise amp = {amp:.2f}"
        print(f"  Evaluating {label}...")

        METHOD_STYLES[label] = {"color": colors_C[i], "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            dataset=test_dataset,
            time_idx=30,
            lag=9,
            sensors_to_use=fixed_sens,
            drop_options=[0],
            context_noise_amplitude=amp,
        )

        ll_dict_C[label], se_dict_C[label], sse_dict_C[label], mse_dict_C[label] = ll, se, sse, mse

        scenario_c_summary["noise_amplitude"].append(amp)
        scenario_c_summary["mean_ll"].append(float(np.mean(ll)))
        scenario_c_summary["median_ll"].append(float(np.median(ll)))
        scenario_c_summary["mean_mse"].append(float(np.mean(mse)))
        scenario_c_summary["median_mse"].append(float(np.median(mse)))

        print(f"  Generating 3 noisy 1x5 multiplots for amplitude {amp:.2f}...")
        generate_noisy_multiplots_for_amplitude(
            noise_amp=amp,
            sample_indices=(0, 1, 2),
            time_idx=30,
            lag=9,
            num_mc_samples=100,
        )

    print("Generating plots for Scenario C...")
    plot_all_distributions(
        ll_dict_C,
        se_dict_C,
        sse_dict_C,
        mse_dict_C,
        out_path=logs_dir / "diagnostics_lag9_noise_amplitudes.png"
    )

    # Compact comparison plot across noise amplitudes
    fig_noise, axes_noise = plt.subplots(1, 2, figsize=(14, 5))

    x_amp = np.array(scenario_c_summary["noise_amplitude"])
    mean_ll = np.array(scenario_c_summary["mean_ll"])
    med_ll = np.array(scenario_c_summary["median_ll"])
    mean_mse = np.array(scenario_c_summary["mean_mse"])
    med_mse = np.array(scenario_c_summary["median_mse"])

    axes_noise[0].plot(x_amp, mean_ll, marker='o', linewidth=2.2, color="#1D3557", label="Mean LL")
    axes_noise[0].plot(x_amp, med_ll, marker='s', linewidth=2.2, color="#457B9D", label="Median LL")
    axes_noise[0].set_title("Scenario C: Log-Likelihood vs Noise")
    axes_noise[0].set_xlabel("Noise amplitude")
    axes_noise[0].set_ylabel("Log-likelihood")
    axes_noise[0].grid(True, alpha=0.3)
    axes_noise[0].legend()

    axes_noise[1].plot(x_amp, mean_mse, marker='o', linewidth=2.2, color="#E76F51", label="Mean MSE")
    axes_noise[1].plot(x_amp, med_mse, marker='s', linewidth=2.2, color="#F4A261", label="Median MSE")
    axes_noise[1].set_title("Scenario C: MSE vs Noise")
    axes_noise[1].set_xlabel("Noise amplitude")
    axes_noise[1].set_ylabel("MSE")
    axes_noise[1].grid(True, alpha=0.3)
    axes_noise[1].legend()

    plt.tight_layout()
    fig_noise.savefig(logs_dir / "diagnostics_lag9_noise_comparison_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig_noise)

    print("Scenario C complete. Saved:")
    print(f"  - {logs_dir / 'diagnostics_lag9_noise_amplitudes.png'}")
    print(f"  - {logs_dir / 'diagnostics_lag9_noise_comparison_summary.png'}")

    # --- 7. Plot Ground Truth with Sensor Annotations ---
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

def main():
    for USE_MU in [True, False]:
        for USE_DEEPONET_DECODER in [True, False]:
            print(f"\n{'=' * 80}\nRUNNING EXPERIMENT WITH USE_MU={USE_MU}, USE_DEEPONET_DECODER={USE_DEEPONET_DECODER}\n{'=' * 80}")
            run_experiment(USE_MU=USE_MU, USE_DEEPONET_DECODER=USE_DEEPONET_DECODER)

if __name__ == "__main__":
    main()