from __future__ import print_function

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dolfin import *

# Assuming these are available in your local directory structure
from architectures.Fourier import FourierFeatures, LearnableFourierFeatures
from pinball_paths import resolve_pinball_asset
from processdata import multiplot, trajectories
from torch.utils.data import DataLoader, Dataset

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
plt.style.use("default")
set_log_level(LogLevel.ERROR)


# ==============================================================================
# DATASETS & DATA LOADERS
# ==============================================================================

class TrajectoryDataset(Dataset):
    """Dataset returning full trajectories and their parameters."""
    def __init__(self, data, mu=None):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.mu = torch.as_tensor(mu, dtype=torch.float32) if mu is not None else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if self.mu is not None:
            return self.data[idx], self.mu[idx]
        return self.data[idx]

def deeponet_collate_fn(
    batch, 
    fixed_sensor_locations, 
    mesh_coordinates, 
    ntimes, 
    history_length=20,     
    points_per_batch=2048,
    use_mu=False
):
    """Builds a batch for the DeepONet Baseline using Relative Time."""
    # Handle whether the batch items are tuples (traj, mu) or just trajs
    if isinstance(batch[0], (tuple, list)):
        batch_trajs = torch.stack([item[0] for item in batch])
        if use_mu:
            batch_mus = torch.stack([item[1] for item in batch])
    else:
        batch_trajs = torch.stack([item for item in batch])
        if use_mu:
            raise ValueError("use_mu is True, but the dataset did not return MU.")
    
    batch_size, ntimes_total, nstate = batch_trajs.shape
    device = batch_trajs.device

    # 1. Sample absolute target time indices (to slice the data correctly)
    time_indices = torch.randint(history_length, ntimes_total, (batch_size,), device=device)

    # 2. Extract the sliding window of sensor history for each trajectory
    offsets = torch.arange(-history_length + 1, 1, device=device)
    time_windows = time_indices.unsqueeze(1) + offsets
    batch_indices_2d = torch.arange(batch_size, device=device).unsqueeze(1)

    history_states = batch_trajs[batch_indices_2d, time_windows]
    sensor_history_3d = history_states[:, :, fixed_sensor_locations]

    # ====================================================================
    # RELATIVE TIME SHIFT
    # ====================================================================
    
    # Branch (History) Time: Negative offsets [-19, ..., 0] normalized
    t_relative = (offsets.float() / float(ntimes_total)).unsqueeze(0).unsqueeze(-1)
    t_repeated = t_relative.expand(batch_size, history_length, -1)
    
    # Trunk (Target) Time: Always 0.0
    t_target = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
    t_expanded = t_target.unsqueeze(1).expand(-1, points_per_batch, -1)

    # --- CHANGED: sensor_history strictly contains (sensors + time) ---
    sensor_history = torch.cat([sensor_history_3d, t_repeated], dim=-1)

    # --- CHANGED: Extract static params separately ---
    if use_mu:
        batch_mus = batch_mus.view(batch_size, ntimes_total, -1).to(device)
        # Extract the parameter at the target time index (Shape: [Batch, num_params])
        batch_indices_1d = torch.arange(batch_size, device=device)
        static_params = batch_mus[batch_indices_1d, time_indices]
    else:
        static_params = None

    # 3. Extract the full target mesh state at the target time
    batch_indices_1d = torch.arange(batch_size, device=device)
    states_at_t = batch_trajs[batch_indices_1d, time_indices]
    
    # 4. Sample random spatial points for the Trunk network
    point_indices = torch.randint(0, nstate, (batch_size, points_per_batch), device=device)
    
    if mesh_coordinates.device != device:
        mesh_coordinates = mesh_coordinates.to(device)
        
    spatial_coords = mesh_coordinates[point_indices] 
    coords = torch.cat([t_expanded, spatial_coords], dim=-1)
    
    # 5. Gather the ground truth values at those exact (t, x, y) points
    y_target = torch.gather(states_at_t, 1, point_indices)
    
    # --- CHANGED: Return 4 items instead of 3 ---
    return sensor_history, static_params, coords, y_target
# ==============================================================================
# MODEL DEFINITION
# ==============================================================================

class DeepONetMeanVar(nn.Module):
    def __init__(self, num_sensors=10, num_params=0, history_length=20, coord_dim=3, p=128, num_frequencies=32):
        super().__init__()
        self.p = p
        self.num_params = num_params
        
        # --- SHARED BRANCH BASE ---
        # 1. Removed num_params from LSTM input. It now only sees dynamic data!
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1, 
            hidden_size=256, num_layers=2, batch_first=True, dropout=0.1
        )
        
        # Separate Branch Heads
        self.branch_mean_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p)
        )
        self.branch_var_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p)
        )

        # --- SHARED TRUNK BASE ---
        self.fourier_mapping = LearnableFourierFeatures(
            input_dim=coord_dim, num_frequencies=num_frequencies, init_scale=1.0
        )
        
        # 2. Added num_params to the trunk's input dimension
        trunk_input_dim = coord_dim + (2 * num_frequencies) + num_params
        
        self.trunk_base = nn.Sequential(
            nn.Linear(trunk_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        
        # Separate Trunk Heads
        self.trunk_mean_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p)
        )
        self.trunk_var_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, p)
        )

        # Biases
        self.mean_bias = nn.Parameter(torch.zeros(1))
        # Start variance small to force learning the mean
        self.var_bias = nn.Parameter(torch.tensor([-3.0])) 

    def forward(self, sensor_history, static_params, coords):
        # Expected Shapes:
        # sensor_history: (B, T, num_sensors + 1)
        # static_params:  (B, num_params)
        # coords:         (B, N, coord_dim)

        # 1. Branch processing (Time-series ONLY)
        _, (h_n, _) = self.branch_lstm(sensor_history)
        branch_features = h_n[-1]
        
        branch_mean = self.branch_mean_head(branch_features).unsqueeze(1) # (B, 1, p)
        branch_var = self.branch_var_head(branch_features).unsqueeze(1)   # (B, 1, p)
        
        # 2. Trunk processing (Coords + Fourier + Params)
        coords_fourier = self.fourier_mapping(coords)
        
        # 3. Expand static parameters to match spatial dimension N
        # Transforms (B, num_params) -> (B, 1, num_params) -> (B, N, num_params)
        if static_params is not None:
            static_params_expanded = static_params.unsqueeze(1).expand(-1, coords.size(1), -1)
            trunk_input = torch.cat([coords, coords_fourier, static_params_expanded], dim=-1)
        else:
            trunk_input = torch.cat([coords, coords_fourier], dim=-1)
        
        
        trunk_features = self.trunk_base(trunk_input)
        
        trunk_mean = self.trunk_mean_head(trunk_features) # (B, N, p)
        trunk_var = self.trunk_var_head(trunk_features)   # (B, N, p)
        
        # 3. Independent Dot Products
        mean = torch.sum(branch_mean * trunk_mean, dim=-1) / math.sqrt(self.p) + self.mean_bias
        
        var_raw = torch.sum(branch_var * trunk_var, dim=-1) / math.sqrt(self.p) + self.var_bias
        var = F.softplus(var_raw) + 1e-6
        
        return mean, var


# ==============================================================================
# LOSS & TRAINING UTILS
# ==============================================================================

def gaussian_nll(mean, var, target):
    return 0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(var)
        + (target - mean) ** 2 / var
    ).mean()

def train_one_DoN_epoch(model, loader, optimizer, device, mse_weight=0.0):
    model.train()
    running_nll = 0.0
    running_mse = 0.0
    
    for sensors, params, coords, y_target in loader:
        sensors = sensors.to(device)
        coords = coords.to(device)
        y_target = y_target.to(device)
        
        # Safely handle the static parameters
        if params is not None:
            params = params.to(device)

        optimizer.zero_grad()
        mean, var = model(sensors, params, coords)
        
        nll_loss = gaussian_nll(mean, var, y_target)
        
        if mse_weight > 0.0:
            mse_loss = F.mse_loss(mean, y_target)
            loss = nll_loss * (1 - mse_weight) + (mse_weight * mse_loss)
            running_mse += mse_loss.item() * sensors.size(0)
        else:
            loss = nll_loss
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_nll += nll_loss.item() * sensors.size(0)
        
    avg_nll = running_nll / len(loader.dataset)
    avg_mse = running_mse / len(loader.dataset) if mse_weight > 0.0 else 0.0
    
    return avg_nll, avg_mse

def evaluate_DoN_loss(model, loader, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for sensors, params, coords, y_target in loader:
            sensors = sensors.to(device)
            coords = coords.to(device)
            y_target = y_target.to(device)
            
            # Safely handle the static parameters
            if params is not None:
                params = params.to(device)

            mean, var = model(sensors, params, coords)
            loss = gaussian_nll(mean, var, y_target)
            running += loss.item() * sensors.size(0)
    return running / len(loader.dataset)

def evaluate_DoN_metrics(model, loader, device, max_batches=3):
    model.eval()
    abs_errors = []
    rel_errors = []
    with torch.no_grad():
        for batch_idx, (sensors, params, coords, y_target) in enumerate(loader):
            sensors = sensors.to(device)
            coords = coords.to(device)
            y_target = y_target.to(device)

            # Safely handle the static parameters
            if params is not None:
                params = params.to(device)

            mean, _ = model(sensors, params, coords)

            pred = mean
            target = y_target

            abs_error = torch.abs(pred - target)
            rel_error = abs_error / (torch.abs(target) + 1e-8)

            abs_errors.append(abs_error.cpu().flatten())
            rel_errors.append(rel_error.cpu().flatten())

            if batch_idx + 1 >= max_batches:
                break

    if not abs_errors:
        return None

    abs_errors = torch.cat(abs_errors)
    rel_errors = torch.cat(rel_errors)

    return abs_errors.mean().item(), rel_errors.mean().item()


# ==============================================================================
# PLOTTING UTILS
# ==============================================================================

def vec2fun(yvec, Yh):
    y = Function(Yh)
    if torch.is_tensor(yvec):
        yvec = yvec.detach().cpu().numpy()
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
    plt.colorbar(mappable, ax=ax, label=label, **cbar_kwargs)
    return mappable


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_experiment(USE_MU):
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / f"logs_pinball_fc_baseline_{'with_mu' if USE_MU else 'without_mu'}"
    checkpoints_dir = script_dir / f"checkpoints_pinball_fc_baseline_{'with_mu' if USE_MU else 'without_mu'}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. Setup Mesh
    # --------------------------------------------------------------------------
    mesh = Mesh(str(resolve_pinball_asset(script_dir, "Pinball_mesh.xml")))
    Yh = FunctionSpace(mesh, "CG", 1)
    nstate = Yh.dim()
    
    # --------------------------------------------------------------------------
    # 2. Load and Prepare Data
    # --------------------------------------------------------------------------
    dt = Constant(0.1)
    T = 3.0
    ntimesteps = round(T / dt)
    ntimes = ntimesteps + 1

    ntrajectories = 500

    Data = np.load(str(resolve_pinball_asset(script_dir, "Pinball_data.npz")))
    Y = torch.tensor(Data["y"], dtype=torch.float32)
    MU = torch.tensor(Data["mu"], dtype=torch.float32)
    nparams = MU.shape[-1]
    del Data


    # Splitting
    
    np.random.seed(0)
            
    ntrain = round(0.8 * ntrajectories)
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories, dtype=bool)
    mask[idx_train] = False
    idx_valid_test = np.arange(ntrajectories)[mask]
    idx_valid = idx_valid_test[::2]
    idx_test = idx_valid_test[1::2]

    Ytrain = Y[idx_train]
    Yvalid = Y[idx_valid]
    Ytest = Y[idx_test]
    
    MUtrain = MU[idx_train]
    MUvalid = MU[idx_valid]
    MUtest = MU[idx_test]

    del Y, MU
    
    # Reshape Data
    Ytrain = Ytrain.reshape(ntrain, ntimes, nstate)
    Yvalid = Yvalid.reshape(len(idx_valid), ntimes, nstate)
    Ytest = Ytest.reshape(len(idx_test), ntimes, nstate)


    # Sensor Definitions
    fixed_sens = [1573, 6925, 1986]
    mesh_coordinates = Yh.tabulate_dof_coordinates()
    sensor_coords = mesh_coordinates[fixed_sens]

    # Normalize Coordinates
    mesh_coords_tensor = torch.tensor(mesh_coordinates, dtype=torch.float32)
    coord_min = mesh_coords_tensor.min(dim=0, keepdim=True)[0]
    coord_max = mesh_coords_tensor.max(dim=0, keepdim=True)[0]
    coord_range = coord_max - coord_min
    coord_range[coord_range == 0] = 1.0 
    mesh_coords_tensor = (mesh_coords_tensor - coord_min) / coord_range

    # --------------------------------------------------------------------------
    # 3. Setup Training Components
    # --------------------------------------------------------------------------
    train_dataset = TrajectoryDataset(Ytrain, mu=MUtrain if USE_MU else None)
    val_dataset = TrajectoryDataset(Yvalid, mu=MUvalid if USE_MU else None)
    test_dataset = TrajectoryDataset(Ytest, mu=MUtest if USE_MU else None)

    batch_size = 16
    collate = partial(
        deeponet_collate_fn, 
        fixed_sensor_locations=fixed_sens,
        mesh_coordinates=mesh_coords_tensor,
        ntimes=ntimes,
        points_per_batch=2048,
        use_mu=USE_MU
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Conditionally pass nparams to the model based on the toggle
    model = DeepONetMeanVar(
        num_sensors=len(fixed_sens), 
        num_params=nparams if USE_MU else 0,
        coord_dim=3, 
        p=128
    ).to(device)

    num_epochs = 3000
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    best_val = float("inf")
    patience = 1000
    no_improve = 0
    initial_mse_weight = 1.0   
    mse_decay_epochs = 1500     

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print(f"\n--- Starting Training (USE_MU = {USE_MU}) ---")
    
    
    for epoch in range(1, num_epochs + 1):
        mse_weight = initial_mse_weight * (mse_decay_epochs - epoch) / mse_decay_epochs if epoch <= mse_decay_epochs else 0.0
        train_nll, train_mse = train_one_DoN_epoch(model, train_loader, optimizer, device, mse_weight=mse_weight)
        
        val_loss = evaluate_DoN_loss(model, val_loader, device)
        scheduler.step()

        if val_loss < best_val and epoch > mse_decay_epochs:
            best_val = val_loss
            no_improve = 0
            torch.save(model.state_dict(), checkpoints_dir / "best_model.pt")
        elif val_loss >= best_val and epoch > mse_decay_epochs:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            if mse_weight > 0.0:
                print(f"Epoch {epoch:4d} | Train NLL: {train_nll:.6f} | Train MSE: {train_mse:.6f} | Val NLL: {val_loss:.6f} | MSE Wt: {mse_weight:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch:4d} | Train NLL: {train_nll:.6f} | Val NLL: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print("Early stopping triggered")
            break
    
    
    # --------------------------------------------------------------------------
    # 5. Evaluation & Visualization
    # --------------------------------------------------------------------------
    best_path = checkpoints_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss = evaluate_DoN_loss(model, test_loader, device)
    metrics = evaluate_DoN_metrics(model, test_loader, device, max_batches=3)

    print(f"Test NLL (mean over batches): {test_loss:.6f}")
    if metrics is not None:
        mae, mre = metrics
        print(f"Test MAE (3 batches): {mae:.6f}")
        print(f"Test MRE (3 batches): {mre:.4%}")

    # Visualizing specific batch
    test_indices = [0, 1, 2] 
    test_batch = [test_dataset[idx] for idx in test_indices]

    history_length = 20
    time_idx = 30
    batch_size = len(test_batch)

    if isinstance(test_batch[0], (tuple, list)):
        trajectories_test = torch.stack([item[0] for item in test_batch])
        if USE_MU:
            mu_test_batch = torch.stack([item[1] for item in test_batch]).to(device)
            mu_test_batch = mu_test_batch.view(batch_size, ntimes, -1)
            static_params = mu_test_batch[:, time_idx, :]
        else:
            static_params = None
    else:
        trajectories_test = torch.stack(test_batch)
        static_params = None
    
    batch_size = trajectories_test.shape[0]
    
    # ---------------------------------------------------------
    # CORRECTED: Branch Input Extraction (Relative Time)
    # ---------------------------------------------------------
    history_window = trajectories_test[:, time_idx - history_length + 1 : time_idx + 1, fixed_sens].to(device)
    
    offsets = torch.arange(-history_length + 1, 1, dtype=torch.float32).to(device)
    t_col = (offsets / float(ntimes)).unsqueeze(0).unsqueeze(-1).expand(batch_size, history_length, 1)
    
    # --- CHANGED: sensor_history strictly contains (sensors + time) ---
    sensor_history = torch.cat([history_window, t_col], dim=-1)
    
    # ---------------------------------------------------------
    # CORRECTED: Trunk Input Extraction (Target Time = 0.0)
    # ---------------------------------------------------------
    t_norm = torch.zeros((batch_size, nstate, 1), dtype=torch.float32).to(device)
    spatial_coords = mesh_coords_tensor.unsqueeze(0).expand(batch_size, -1, -1).to(device)
    full_coords = torch.cat([t_norm, spatial_coords], dim=-1)
    
    # Target Ground Truth
    y_test_norm = trajectories_test[:, time_idx, :].to(device)
    
    with torch.no_grad():
        # --- CHANGED: Forward pass now expects 3 inputs ---
        pred_mean_norm, pred_var_norm = model(sensor_history, static_params, full_coords)
        
    # Correct Un-normalization Math
    pred_mean = pred_mean_norm 
    y_target = y_test_norm 
    pred_var = pred_var_norm 
    pred_std = torch.sqrt(pred_var)
    
    # Calculations strictly in Un-normalized space
    pred_mse = (pred_mean - y_target) ** 2
    pred_loglik = -0.5 * torch.log(2.0 * math.pi * pred_var) - 0.5 * (pred_mse / pred_var)

    # Plot scale calculations
    state_vmin = torch.min(pred_mean.min(), y_target.min()).item()
    state_vmax = torch.max(pred_mean.max(), y_target.max()).item()
    if state_vmax == state_vmin: state_vmax = state_vmin + 1e-8

    uncert_vmin = pred_std.min().item()
    uncert_vmax = pred_std.max().item()
    if uncert_vmax == uncert_vmin: uncert_vmax = uncert_vmin + 1e-8

    err_vmin = pred_mse.min().item()
    err_vmax = pred_mse.max().item()
    if err_vmax == err_vmin: err_vmax = err_vmin + 1e-8
    
    ll_vmin_global = pred_loglik.min().item()
    ll_vmax_global = pred_loglik.max().item()
    if ll_vmax_global == ll_vmin_global: ll_vmax_global = ll_vmin_global + 1e-8
    
    # 1x5 CUSTOM GRID PLOTTING
    def plot_custom_grid(y, Yh, ax, cmap, vmin, vmax, title, tick_size=30):
        plt.sca(ax)
        mappable = plot(vec2fun(y, Yh), cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=tick_size)
        plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color='red', s=200, zorder=5)
        ax.set_title(title, fontsize=40)
        ax.axis('off')
        return mappable

    for i in range(len(test_indices)):
        # Expanded to 1x5 to fit the Ground Truth
        fig, axes = plt.subplots(1, 5, figsize=(60, 10))
        
        # Plot 1: Ground Truth (New)
        plot_custom_grid(y_target[i].cpu(), Yh, axes[0], "jet", state_vmin, state_vmax, "Ground Truth")
        # Plot 2: Mean Prediction
        plot_custom_grid(pred_mean[i].cpu(), Yh, axes[1], "jet", state_vmin, state_vmax, "Mean prediction")
        # Plot 3: Standard Deviation
        plot_custom_grid(pred_std[i].cpu(), Yh, axes[2], "magma", uncert_vmin, uncert_vmax, "Standard Deviation")
        # Plot 4: MSE
        plot_custom_grid(pred_mse[i].cpu(), Yh, axes[3], "magma", err_vmin, err_vmax, "Squared Error")
        # Plot 5: Log-Likelihood
        plot_custom_grid(pred_loglik[i].cpu(), Yh, axes[4], "magma", ll_vmin_global, ll_vmax_global, "Log-likelihood map")

        plt.tight_layout()
        grid_path = logs_dir / f"multiplot_fc_baseline_1x5_index_{test_indices[i]}_at_time_{time_idx}_mu_{USE_MU}.png"
        plt.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"1x5 Grid plot saved to {grid_path}")


def main():
    for USE_MU in [False, True]:
        print(f"\n{'=' * 80}\nRUNNING EXPERIMENT WITH USE_MU={USE_MU}\n{'=' * 80}")
        run_experiment(USE_MU)

if __name__ == "__main__":
    main()