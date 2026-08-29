from __future__ import print_function

import math
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dolfin import *

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Assuming these are available in your local directory structure
from architectures.Fourier import LearnableFourierFeatures
from pinball.pinball_paths import resolve_pinball_asset
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
    """Builds a batch for DeepONet separating dynamic sensors from static parameters."""
    if isinstance(batch[0], (tuple, list)):
        batch_trajs = torch.stack([item[0] for item in batch])
        if use_mu:
            batch_mus = torch.stack([item[1] for item in batch])
    else:
        batch_trajs = torch.stack([item for item in batch])
        if use_mu:
            raise ValueError("use_mu is True, but the dataset did not return MU.")
    
    batch_size, ntimes_total, nstate = batch_trajs.shape
    dev = batch_trajs.device

    # 1. Sample target time indices
    time_indices = torch.randint(history_length - 1, ntimes_total, (batch_size,), device=dev)

    # 2. Extract sliding window of sensor history
    offsets = torch.arange(-history_length + 1, 1, device=dev)
    time_windows = time_indices.unsqueeze(1) + offsets
    batch_indices_2d = torch.arange(batch_size, device=dev).unsqueeze(1)

    history_states = batch_trajs[batch_indices_2d, time_windows]
    sensor_history_3d = history_states[:, :, fixed_sensor_locations]

    # --- RELATIVE TIME SHIFT ---
    t_relative = (offsets.float() / float(ntimes_total)).unsqueeze(0).unsqueeze(-1)
    t_repeated = t_relative.expand(batch_size, history_length, -1)
    
    # Branch only takes sensors + relative time
    sensor_history = torch.cat([sensor_history_3d, t_repeated], dim=-1)

    # Static parameters extracted at target time index
    if use_mu:
        batch_mus = batch_mus.view(batch_size, ntimes_total, -1).to(dev)
        static_params = batch_mus[torch.arange(batch_size, device=dev), time_indices]
    else:
        static_params = None

    # Trunk Target Time: Always 0.0
    t_target = torch.zeros((batch_size, 1), dtype=torch.float32, device=dev)
    t_expanded = t_target.unsqueeze(1).expand(-1, points_per_batch, -1)

    # 3. Extract target states and sample random spatial points
    states_at_t = batch_trajs[torch.arange(batch_size, device=dev), time_indices]
    point_indices = torch.randint(0, nstate, (batch_size, points_per_batch), device=dev)
    
    if mesh_coordinates.device != dev:
        mesh_coordinates = mesh_coordinates.to(dev)
        
    spatial_coords = mesh_coordinates[point_indices] 
    coords = torch.cat([t_expanded, spatial_coords], dim=-1)
    
    # 4. Gather ground truth
    y_target = torch.gather(states_at_t, 1, point_indices)
    
    return sensor_history, static_params, coords, y_target


def build_don_eval_inputs(
    batch,
    fixed_sensor_locations,
    mesh_coordinates,
    ntimes,
    history_length=20,
    time_idx=30,
    use_mu=False,
):
    if isinstance(batch[0], (tuple, list)):
        batch_trajs = torch.stack([item[0] for item in batch])
        batch_mus = torch.stack([item[1] for item in batch]) if use_mu else None
    else:
        batch_trajs = torch.stack([item for item in batch])
        batch_mus = None

    batch_size, _, nstate = batch_trajs.shape
    dev = batch_trajs.device

    time_window = torch.arange(time_idx - history_length + 1, time_idx + 1, device=dev)
    history_states = batch_trajs[:, time_window]
    sensor_history_3d = history_states[:, :, fixed_sensor_locations]

    offsets = torch.arange(-history_length + 1, 1, device=dev)
    t_relative = (offsets.float() / float(ntimes)).unsqueeze(0).unsqueeze(-1)
    t_repeated = t_relative.expand(batch_size, history_length, -1)

    sensor_history = torch.cat([sensor_history_3d, t_repeated], dim=-1)

    if use_mu and batch_mus is not None:
        batch_mus = batch_mus.view(batch_size, ntimes, -1).to(dev)
        static_params = batch_mus[:, time_idx]
    else:
        static_params = None

    t_target = torch.zeros((batch_size, 1), dtype=torch.float32, device=dev)
    t_expanded = t_target.unsqueeze(1).expand(-1, nstate, -1)

    if mesh_coordinates.device != dev:
        mesh_coordinates = mesh_coordinates.to(dev)

    spatial_coords = mesh_coordinates.unsqueeze(0).expand(batch_size, -1, -1)
    coords = torch.cat([t_expanded, spatial_coords], dim=-1)
    y_target = batch_trajs[:, time_idx, :]

    return sensor_history, static_params, coords, y_target


# ==============================================================================
# MODEL DEFINITION (DETERMINISTIC)
# ==============================================================================

class DeepONetDeterministic(nn.Module):
    def __init__(
        self,
        num_sensors=10,
        num_params=0,
        history_length=20,
        coord_dim=3,
        p=128,
        num_frequencies=32,
    ):
        super().__init__()
        self.p = p
        self.num_params = num_params

        # 1. Branch Net — LSTM only sees dynamic signals (sensors + relative time)
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Branch Head
        self.branch_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, p),
        )

        # 2. Learnable Fourier Features
        self.fourier_mapping = LearnableFourierFeatures(
            input_dim=coord_dim,
            num_frequencies=num_frequencies,
            init_scale=1.0,
        )

        # 3. Trunk Net (Coordinates + Fourier + static parameters mu)
        trunk_input_dim = coord_dim + (2 * num_frequencies) + num_params
        self.trunk_base = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Trunk Head
        self.trunk_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, p),
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensor_history, static_params, coords):
        # 1. Branch Processing
        _, (h_n, _) = self.branch_lstm(sensor_history)
        branch_features = h_n[-1]
        branch_out = self.branch_head(branch_features).unsqueeze(1)  # (B, 1, p)

        # 2. Trunk Processing
        coords_fourier = self.fourier_mapping(coords)

        if static_params is not None:
            static_params_expanded = static_params.unsqueeze(1).expand(-1, coords.size(1), -1)
            trunk_input = torch.cat([coords, coords_fourier, static_params_expanded], dim=-1)
        else:
            trunk_input = torch.cat([coords, coords_fourier], dim=-1)

        trunk_features = self.trunk_base(trunk_input)
        trunk_out = self.trunk_head(trunk_features)  # (B, N, p)

        # 3. Dot Product
        pred = torch.sum(branch_out * trunk_out, dim=-1) / math.sqrt(self.p) + self.bias
        return pred


# ==============================================================================
# LOSS & TRAINING UTILS
# ==============================================================================

def train_one_DoN_epoch(model, loader, optimizer, device):
    model.train()
    running_mse = 0.0
    
    for sensors, static_params, coords, y_target in loader:
        sensors = sensors.to(device)
        coords = coords.to(device)
        y_target = y_target.to(device)
        if static_params is not None:
            static_params = static_params.to(device)

        optimizer.zero_grad()
        pred = model(sensors, static_params, coords)
        
        loss = F.mse_loss(pred, y_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_mse += loss.item() * sensors.size(0)
        
    avg_mse = running_mse / len(loader.dataset)
    return avg_mse

def evaluate_DoN_loss(model, loader, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for sensors, static_params, coords, y_target in loader:
            sensors = sensors.to(device)
            coords = coords.to(device)
            y_target = y_target.to(device)
            if static_params is not None:
                static_params = static_params.to(device)
            
            pred = model(sensors, static_params, coords)
            loss = F.mse_loss(pred, y_target)
            running += loss.item() * sensors.size(0)
    return running / len(loader.dataset)

def evaluate_DoN_metrics(model, loader, device, max_batches=3):
    model.eval()
    abs_errors = []
    rel_errors = []
    with torch.no_grad():
        for batch_idx, (sensors, static_params, coords, y_target) in enumerate(loader):
            sensors = sensors.to(device)
            coords = coords.to(device)
            y_target = y_target.to(device)
            if static_params is not None:
                static_params = static_params.to(device)
            
            pred = model(sensors, static_params, coords)
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
# EXPERIMENT WRAPPER
# ==============================================================================

def run_experiment(
    use_mu, 
    train_dataset, val_dataset, test_dataset, 
    fixed_sens, mesh_coords_tensor, ntimes, nstate, nparams, Yh, sensor_coords, script_dir
):
    print(f"\n{'='*60}")
    print(f"--- Starting Training (USE_MU = {use_mu}) ---")
    print(f"{'='*60}")

    logs_dir = script_dir / f"logs_pinball_don_det_{'with_mu' if use_mu else 'without_mu'}"
    checkpoints_dir = script_dir / f"checkpoints_pinball_don_det_{'with_mu' if use_mu else 'without_mu'}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    collate = partial(
        deeponet_collate_fn, 
        fixed_sensor_locations=fixed_sens, 
        mesh_coordinates=mesh_coords_tensor, 
        ntimes=ntimes, 
        points_per_batch=2048, 
        use_mu=use_mu
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = DeepONetDeterministic(
        num_sensors=len(fixed_sens), 
        num_params=nparams if use_mu else 0, 
        coord_dim=3, 
        p=128
    ).to(device)

    num_epochs = 3000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    best_val = float("inf")
    patience = 1000
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_mse = train_one_DoN_epoch(model, train_loader, optimizer, device)
        val_mse = evaluate_DoN_loss(model, val_loader, device)
        scheduler.step()

        if val_mse < best_val:
            best_val = val_mse
            no_improve = 0
            torch.save(model.state_dict(), checkpoints_dir / "best_model.pt")
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if no_improve >= patience:
            print("Early stopping triggered")
            break
    
    # --------------------------------------------------------------------------
    # Evaluation & Visualization
    # --------------------------------------------------------------------------
    best_path = checkpoints_dir / "best_model.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss = evaluate_DoN_loss(model, test_loader, device)
    metrics = evaluate_DoN_metrics(model, test_loader, device, max_batches=3)

    print(f"\nFinal Test MSE (mean over batches): {test_loss:.6f}")
    if metrics is not None:
        mae, mre = metrics
        print(f"Test MAE (3 batches): {mae:.6f}")
        print(f"Test MRE (3 batches): {mre:.4%}")

    test_indices = [0, 1, 2] 
    test_batch = [test_dataset[idx] for idx in test_indices]
    
    history_length = 20
    time_idx = 30  
    
    sensor_history, static_params, full_coords, y_test_norm = build_don_eval_inputs(
        test_batch,
        fixed_sensor_locations=fixed_sens,
        mesh_coordinates=mesh_coords_tensor,
        ntimes=ntimes,
        history_length=history_length,
        time_idx=time_idx,
        use_mu=use_mu,
    )
    sensor_history = sensor_history.to(device)
    full_coords = full_coords.to(device)
    y_target = y_test_norm.to(device)
    if static_params is not None:
        static_params = static_params.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(sensor_history, static_params, full_coords)
        
    pred_mse = (pred - y_target) ** 2

    state_vmin = torch.min(pred.min(), y_target.min()).item()
    state_vmax = torch.max(pred.max(), y_target.max()).item()
    if state_vmax == state_vmin: state_vmax = state_vmin + 1e-8

    err_vmin = pred_mse.min().item()
    err_vmax = pred_mse.max().item()
    if err_vmax == err_vmin: err_vmax = err_vmin + 1e-8
    
    def plot_custom_grid(y, Yh, ax, cmap, vmin, vmax, title, tick_size=30):
        plt.sca(ax)
        mappable = plot(vec2fun(y, Yh), cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=tick_size)
        plt.scatter(sensor_coords[:, 0], sensor_coords[:, 1], color='red', s=200, zorder=5)
        ax.set_title(title, fontsize=40)
        ax.axis('off')
        return mappable

    # Generate 1x3 Grids
    for i in range(len(test_indices)):
        fig, axes = plt.subplots(1, 3, figsize=(40, 10))
        
        plot_custom_grid(y_target[i].cpu(), Yh, axes[0], "jet", state_vmin, state_vmax, "Ground Truth")
        plot_custom_grid(pred[i].cpu(), Yh, axes[1], "jet", state_vmin, state_vmax, "Prediction")
        plot_custom_grid(pred_mse[i].cpu(), Yh, axes[2], "magma", err_vmin, err_vmax, "Squared Error")

        plt.tight_layout()
        grid_path = logs_dir / f"multiplot_don_det_1x3_idx_{test_indices[i]}_at_time_{time_idx}_mu_{use_mu}.png"
        plt.savefig(grid_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved inference plot to {grid_path}")


# ==============================================================================
# MAIN 
# ==============================================================================

def main():
    script_dir = Path(__file__).resolve().parent
    
    # --------------------------------------------------------------------------
    # Setup Mesh & Data
    # --------------------------------------------------------------------------
    mesh = Mesh(str(resolve_pinball_asset(script_dir, "Pinball_mesh.xml")))
    Yh = FunctionSpace(mesh, "CG", 1)
    nstate = Yh.dim()
    
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
    
    Ytrain = Ytrain.reshape(ntrain, ntimes, nstate)
    Yvalid = Yvalid.reshape(len(idx_valid), ntimes, nstate)
    Ytest = Ytest.reshape(len(idx_test), ntimes, nstate)

    fixed_sens = [1573, 6925, 1986]
    mesh_coordinates = Yh.tabulate_dof_coordinates()
    sensor_coords = mesh_coordinates[fixed_sens]

    # Raw physical coordinates (no scaling)
    mesh_coords_tensor = torch.tensor(mesh_coordinates, dtype=torch.float32)

    for use_mu in [True, False]:
        train_dataset = TrajectoryDataset(Ytrain, mu=MUtrain if use_mu else None)
        val_dataset = TrajectoryDataset(Yvalid, mu=MUvalid if use_mu else None)
        test_dataset = TrajectoryDataset(Ytest, mu=MUtest if use_mu else None)

        run_experiment(
            use_mu=use_mu,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            fixed_sens=fixed_sens,
            mesh_coords_tensor=mesh_coords_tensor,
            ntimes=ntimes,
            nstate=nstate,
            nparams=nparams,
            Yh=Yh,
            sensor_coords=sensor_coords,
            script_dir=script_dir,
        )


if __name__ == "__main__":
    main()