
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
from LNP.LATNPsimple import LatNP_simple
from architectures.Fourier import FourierFeatures, LearnableFourierFeatures
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np
import math
import matplotlib.patches as mpatches
import torch.nn.functional as F

from pinball_paths import resolve_pinball_asset


# ============================================================
# 1. CONFIGURATION & DATA LOADING
# ============================================================
def load_data(script_dir):
    """Loads the main datasets, trajectory parameters, and sensors."""
    print("Loading data...")
    filename = resolve_pinball_asset(script_dir, "Pinball_data.npz")

    Data = np.load(str(filename))
    Y = torch.tensor(Data["y"])
    MU = torch.tensor(Data["mu"])
    VNS = torch.tensor(Data["v"])
    
    # Load fixed sensors
    idx_sensors_path = resolve_pinball_asset(script_dir, "Pinball_idx_fixedsensors.pt")
    if idx_sensors_path.exists():
        fixed_sens = torch.load(str(idx_sensors_path), weights_only=False)
    else:
        fixed_sens = None
        
    return Y, MU, VNS, fixed_sens

# ============================================================
# 1. HELPERS & COLLATERS (From your pinball_2.py)
# ============================================================

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
    choice_idx = torch.randint(0, len(options), (1,)).item()
    return int(options[choice_idx])

def unified_test_collate_fn(
    batch, 
    mesh_coords, 
    fixed_sensor_locations=None,
    num_context_sensors_min=2, 
    num_context_sensors_max=10, 
    use_all_sensors=False, 
    drop_random_sensors=0,
    drop_random_sensors_options=None, 
    time_idx=None, 
    lag=None,
    use_mu=False,
    model_format="np"  # <--- Use "np" or "don"
):
    """Unified test collater for Neural Processes and DeepONets."""
    if mesh_coords is None: raise ValueError("mesh_coords must be provided!")
    if not isinstance(mesh_coords, torch.Tensor): mesh_coords = torch.as_tensor(mesh_coords, dtype=torch.float32)

    batch_size = len(batch)
    device = batch[0][0].device if isinstance(batch[0], (tuple, list)) else batch[0].device
    
    # 1. Extract Batch Elements
    if isinstance(batch[0], (tuple, list)):
        ntimes, nstate = batch[0][0].shape
        batch_trajs = torch.stack([item[0] for item in batch])
        batch_mus = torch.stack([item[1] for item in batch]) if use_mu else None
    else:
        ntimes, nstate = batch[0].shape
        batch_trajs = torch.stack(batch)
        batch_mus = None
        if use_mu: raise ValueError("use_mu is True, but dataset did not return MU.")

    # 2. Select Sensors
    chosen_drop = choose_drop_random_sensors(drop_random_sensors, drop_random_sensors_options)
    sensor_locations_fixed = select_sensor_locations(
        fixed_sensor_locations, nstate, use_all_sensors=use_all_sensors, drop_random_sensors=chosen_drop
    )
    
    if sensor_locations_fixed is not None:
        sensor_locations = sensor_locations_fixed
        num_sensors = sensor_locations.numel()
    else:
        num_sensors = np.random.randint(num_context_sensors_min, num_context_sensors_max + 1)
        sensor_locations = torch.randperm(nstate)[:num_sensors]

    # 3. Select Time Window (Lag serves as history_length for DON)
    if time_idx is None: time_idx = np.random.randint(20, ntimes - 1) 
    if lag is None: lag = np.random.choice([0, 4, 9, 19]) 

    time_window = torch.arange(time_idx - lag, time_idx + 1)
    history_len = lag + 1

    # =========================================================
    # A. NEURAL PROCESS FORMATTING
    # =========================================================
    if model_format == "np":
        context_time_indices = time_window.repeat_interleave(num_sensors)
        context_state_indices = sensor_locations.repeat(history_len)
        
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

        if batch_mus is not None:
            if batch_mus.dim() == 4 and batch_mus.size(1) == 1: batch_mus = batch_mus.squeeze(1)
            if batch_mus.dim() == 3:
                mu_context = batch_mus[:, context_time_indices, :]
                mu_target = batch_mus[:, target_time_indices, :]
            else:
                mu_context = batch_mus.unsqueeze(1).expand(-1, len(context_time_indices), -1)
                mu_target = batch_mus.unsqueeze(1).expand(-1, len(target_time_indices), -1)
            x_context = torch.cat([mu_context, x_context], dim=-1)
            x_target = torch.cat([mu_target, x_target], dim=-1)

        y_context = batch_trajs[:, context_time_indices, context_state_indices].unsqueeze(-1)
        y_target = batch_trajs[:, target_time_indices, target_state_indices].unsqueeze(-1)

        return x_context.contiguous(), y_context.contiguous(), x_target.contiguous(), y_target.contiguous()

    # =========================================================
    # B. DEEPONET FORMATTING
    # =========================================================
    elif model_format == "don":
        batch_indices_2d = torch.arange(batch_size).unsqueeze(1)
        history_states = batch_trajs[batch_indices_2d, time_window]
        
        # Branch Input: (Batch, History_Len, Num_Sensors)
        sensor_history_3d = history_states[:, :, sensor_locations]

        # Normalized Relative Time Shift
        offsets = torch.arange(-lag, 1)
        t_relative = (offsets.float() / float(ntimes)).unsqueeze(0).unsqueeze(-1)
        t_repeated = t_relative.expand(batch_size, history_len, -1)
        
        if batch_mus is not None:
            batch_mus = batch_mus.view(batch_size, ntimes, -1)
            mu_history = batch_mus[batch_indices_2d, time_window]
            sensor_history = torch.cat([sensor_history_3d, t_repeated, mu_history], dim=-1)
        else:
            sensor_history = torch.cat([sensor_history_3d, t_repeated], dim=-1)

        # Trunk Input: Target time (0.0) + spatial coords
        t_target = torch.zeros((batch_size, nstate, 1), dtype=torch.float32)
        spatial_coords = mesh_coords.unsqueeze(0).expand(batch_size, -1, -1) 
        coords = torch.cat([t_target, spatial_coords], dim=-1)

        # DeepONet Target is typically shaped (B, Nstate, 1) to match NP layout safely
        y_target = batch_trajs[:, time_idx, :].unsqueeze(-1) 
        
        # We return `None` for y_context since DON doesn't use it, 
        # keeping the 4-item return signature consistent!
        return sensor_history.contiguous(), None, coords.contiguous(), y_target.contiguous()

    else:
        raise ValueError("model_format must be 'np' or 'don'")
    
    
class SpatiotemporalDataset(Dataset):
    def __init__(self, data, mu_params=None):
        self.data = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        if mu_params is not None:
            self.mu_params = torch.from_numpy(mu_params).float() if isinstance(mu_params, np.ndarray) else mu_params.float()
        else:
            self.mu_params = None

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        if self.mu_params is not None: return self.data[idx], self.mu_params[idx]
        return self.data[idx]

def vec2fun(yvec, Yh):
    y = Function(Yh)
    y.vector()[:] = yvec
    return y

def plot_with_colorbar(y, Yh, ax=None, cmap="jet", vmin=None, vmax=None, label=None, cbar_kwargs=None):
    if ax is None: ax = plt.gca()
    else: plt.sca(ax)
    mappable = plot(vec2fun(y, Yh), cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar_kwargs is None: cbar_kwargs = {"shrink": 0.75, "pad": 0.02}
    cbar = plt.colorbar(mappable, ax=ax, **cbar_kwargs)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(label, size=16)
    return mappable


# ==============================================================================
# MODEL DEFINITION
# ==============================================================================
class DeepONetDeterministic(nn.Module):
    def __init__(self, num_sensors=10, num_params=0, history_length=20, coord_dim=3, p=128, num_frequencies=32):
        super().__init__()
        self.p = p
        self.num_params = num_params

        # 1. Branch Net — LSTM
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1 + num_params,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.branch_proj = nn.Linear(256, p) # Only predicts p features
        
        # 2. Fourier Features (Now matches stochastic: input_dim = coord_dim)
        self.fourier_mapping = FourierFeatures(
            input_dim=coord_dim, 
            num_frequencies=num_frequencies,
            scale=6.0,
            learnable=False
        )

        # 3. Trunk Net (Dropout removed to match stochastic)
        trunk_input_dim = coord_dim + (2 * num_frequencies)
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, p) 
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, sensor_history, coords):
        _, (h_n, _) = self.branch_lstm(sensor_history)
        branch_out = self.branch_proj(h_n[-1]).unsqueeze(1)  # (B, 1, p)
        
        # Fourier applied to ALL coordinates (time + space)
        coords_fourier = self.fourier_mapping(coords[: , :, :])
        trunk_input = torch.cat([coords, coords_fourier], dim=-1)
        trunk_out = self.trunk(trunk_input) # (B, N_points, p)
        
        # Simple dot product for deterministic prediction
        pred = torch.sum(branch_out * trunk_out, dim=-1) / math.sqrt(self.p) + self.bias
        
        return pred



class DeepONetMeanVar(nn.Module):
    def __init__(self, num_sensors=10, num_params=0, history_length=20, coord_dim=3, p=128, num_frequencies=32):
        super().__init__()
        self.p = p
        self.num_params = num_params

        # 1. Branch Net — LSTM
        # Input: Sensors + 1 (Time) + Params
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1 + num_params,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.branch_proj = nn.Linear(256, p * 2)
        
        # 2. Fourier Features
        self.fourier_mapping = LearnableFourierFeatures(
            input_dim=coord_dim, 
            num_frequencies=num_frequencies,
            init_scale=1.0
        )

        # 3. Trunk Net
        trunk_input_dim = coord_dim + (2 * num_frequencies)
        
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, p * 2) 
        )

        self.mean_bias = nn.Parameter(torch.zeros(1))
        self.var_bias = nn.Parameter(torch.tensor([-3.0]))

    def forward(self, sensor_history, coords):
        _, (h_n, _) = self.branch_lstm(sensor_history)
        branch_out = self.branch_proj(h_n[-1]).unsqueeze(1)  # (B, 1, p*2)
        
        coords_fourier = self.fourier_mapping(coords[: , :, :])
        trunk_input = torch.cat([coords, coords_fourier], dim=-1)
        trunk_out = self.trunk(trunk_input)
        
        branch_mean, branch_var = torch.split(branch_out, self.p, dim=-1)
        trunk_mean, trunk_var = torch.split(trunk_out, self.p, dim=-1)
        
        mean = torch.sum(branch_mean * trunk_mean, dim=-1) / math.sqrt(self.p) + self.mean_bias
        
        var_raw = torch.sum(branch_var * trunk_var, dim=-1) / math.sqrt(self.p) + self.var_bias
        var = F.softplus(var_raw) + 1e-6
        
        return mean, var
    
import gpytorch

# ==============================================================================
# GP MODEL DEFINITION
# ==============================================================================
class ContextConditionedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # 6 input dimensions: [mu_0, mu_1, mu_2, time, x, y]
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=6)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ============================================================
# 2. MATH & METRICS HELPERS
# ============================================================
def gaussian_log_lik(y_pred_mean, y_pred_var, y_true):
    var_clamp = y_pred_var.clamp_min(1e-8)
    ll_const = math.log(2.0 * math.pi)
    return -0.5 * (ll_const + torch.log(var_clamp) + ((y_true - y_pred_mean)**2) / var_clamp)

def compute_standardized_se(y_pred_mean, y_pred_var, y_true):
    var_clamp = y_pred_var.clamp_min(1e-8)
    return ((y_true - y_pred_mean)**2) / var_clamp

# ============================================================
# 3. PLOTTING STYLES & HELPERS
# ============================================================
plt.style.use('default')

METHOD_STYLES = {
    "ANP": {"color": "#E63946", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85},
    "NP": {"color": "#457B9D", "linestyle": "--", "linewidth": 2.2, "alpha": 0.85},
    "Prob-DeepONet": {"color": "#2A9D8F", "linestyle": "-.", "linewidth": 2.2, "alpha": 0.85},
    "SHRED": {"color": "#8338EC", "linestyle": ":", "linewidth": 2.2, "alpha": 0.85},
    "Context-GP": {"color": "#F4A261", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85},
    "DeepONet": {"color": "#457B9D", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}
}

def _plot_row(ax_hist, ax_box, data_dict, title, xlabel, bins, clip_pct, log_scale=False):
    if not data_dict: return

    all_vals = np.concatenate([v for v in data_dict.values()])
    x_lo, x_hi = np.percentile(all_vals, clip_pct), np.percentile(all_vals, 100 - clip_pct)
    legend_handles = []

    for name, vals in data_dict.items():
        style = METHOD_STYLES.get(name, {"color": "#000000", "linestyle": "-", "linewidth": 2, "alpha": 0.8})
        clipped = vals[(vals >= x_lo) & (vals <= x_hi)]
        eff_bins = min(bins, max(10, len(clipped) // 2))

        counts, edges = np.histogram(clipped, bins=eff_bins, density=True)
        ax_hist.hist(clipped, bins=eff_bins, density=True, color=style["color"], alpha=style["alpha"]*0.55)
        ax_hist.plot(0.5*(edges[:-1]+edges[1:]), counts, color=style["color"], linestyle=style["linestyle"], linewidth=style["linewidth"])
        ax_hist.axvline(float(np.median(vals)), color=style["color"], linewidth=1.2, linestyle="--")
        legend_handles.append(mpatches.Patch(color=style["color"], label=name))

    # Format Histogram
    ax_hist.set_xlim(x_lo, x_hi)
    ax_hist.set_xlabel(xlabel, fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title(title, fontsize=15)
    ax_hist.tick_params(axis='both', labelsize=12)
    ax_hist.legend(handles=legend_handles, framealpha=0.85, fontsize=12)

    # Format Boxplot
    bp_data, bp_names = list(data_dict.values()), list(data_dict.keys())
    flier_style = dict(marker='o', markerfacecolor='black', markersize=2, alpha=0.1, linestyle='none', markeredgecolor='none')

    bplot = ax_box.boxplot(bp_data, vert=True, patch_artist=True, notch=True, showfliers=False, flierprops=flier_style)
    for patch, colour in zip(bplot["boxes"], [METHOD_STYLES.get(n, {"color": "#000"})["color"] for n in bp_names]):
        patch.set_facecolor(colour); patch.set_alpha(0.70)

    ax_box.set_xticks(range(1, len(bp_names) + 1))
    ax_box.set_xticklabels(bp_names, rotation=0, fontsize=10)
    ax_box.set_ylabel(xlabel, fontsize=12)
    ax_box.set_title(f"Box Plot: {title}", fontsize=15)
    ax_box.tick_params(axis='both', labelsize=12)

    if log_scale in [True, 'log']:
        ax_box.set_yscale('log')
    elif log_scale == 'symlog':
        ax_box.set_yscale('symlog')

def plot_all_distributions(ll_dict, se_dict, sse_dict, mse_dict, out_path, bins=80, clip_pct=0.5):
    fig, axes = plt.subplots(4, 2, figsize=(18, 15), gridspec_kw={"width_ratios": [2, 1.25]})

    _plot_row(axes[0, 0], axes[0, 1], ll_dict, "Log-Likelihood Distribution", "Per-node log-likelihood", bins, clip_pct, log_scale=False)
    _plot_row(axes[1, 0], axes[1, 1], sse_dict, "Standardized Squared Errors (SSE)", "Per-node SSE", bins, clip_pct, log_scale=False)
    _plot_row(axes[2, 0], axes[2, 1], se_dict, "Squared Error (SE) Distribution", "Per-node Squared Error", bins, clip_pct * 2, log_scale=False)
    _plot_row(axes[3, 0], axes[3, 1], mse_dict, "Mean Squared Errors (MSE)", "Per-sample MSE", bins, clip_pct, log_scale=False)

    fig.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 4. SINGLE BATCH EVALUATION & 2x3 GRID PLOTTING
# ============================================================
def plot_batch_diagnostics(model, test_dataset, spatiotemporal_test_collate_fn, mesh_coordinates_norm, fixed_sens, Yh, USE_MU, device, logs_dir, model_format="np"):
    """Evaluates a batch and outputs the 2x3 diagnostic grid and 2x5 MC plots."""
    print("\n" + "="*60)
    print("BATCH TESTING WITH 3 TEST SAMPLES")
    print("="*60)

    model.eval()
    test_indices = np.array([0, 1, 2])
    test_batch = [test_dataset[idx] for idx in test_indices]
    eval_time_idx = 30 

    for current_lag in [0, 9, 19]:
        x_context, y_context, x_target, y_target = spatiotemporal_test_collate_fn(
            test_batch,
            mesh_coords=mesh_coordinates_norm,
            fixed_sensor_locations=fixed_sens,
            use_all_sensors=True,
            time_idx=eval_time_idx,
            lag=current_lag,
            use_mu=USE_MU,
            model_format=model_format
        )

        x_context, y_context, x_target = x_context.to(device), y_context.to(device), x_target.to(device)

        # Inference
        num_mc_samples = 100
        with torch.no_grad():
            y_pred_mean, y_pred_var, _, _ = model(
                x_context, y_context, x_target, num_samples=num_mc_samples
            )
            
        y_pred_mc = y_pred_mean.squeeze(-1).cpu()
        y_pred_var_mc = y_pred_var.squeeze(-1).cpu()
        y_target_cpu = y_target.squeeze(-1).cpu()

        # Target point aggregations
        y_pred = y_pred_mc.mean(dim=0)
        y_pred_var_epistemic = y_pred_mc.var(dim=0, unbiased=False)
        y_pred_var_aleatoric = y_pred_var_mc.mean(dim=0)
        y_pred_var_total = y_pred_var_epistemic + y_pred_var_aleatoric

        # Mixture Log-Likelihood Calculation (using LogSumExp)
        var_clamp = y_pred_var_mc.clamp_min(1e-8)
        ll_const = math.log(2.0 * math.pi)
        sample_lls = -0.5 * (ll_const + torch.log(var_clamp) + ((y_target_cpu - y_pred_mc)**2) / var_clamp)
        
        M = y_pred_mc.shape[0]
        log_lik_all = torch.logsumexp(sample_lls, dim=0) - math.log(M)
        
        ll_vmin_global = log_lik_all.min().item()
        ll_vmax_global = log_lik_all.max().item()
        if ll_vmax_global == ll_vmin_global:
            ll_vmax_global = ll_vmin_global + 1e-8

        # --- Plotting per sample in batch ---
        for batch_idx in range(3):
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
            
            sample_sse = sample_sq_error / sample_total_var.clamp_min(1e-8)

            # Colormap Limits
            state_vmin = torch.min(sample_pred_runs.min(), sample_target.min()).item()
            state_vmax = torch.max(sample_pred_runs.max(), sample_target.max()).item()
            if state_vmax == state_vmin: state_vmax = state_vmin + 1e-8
            
            std_vmin, std_vmax = sample_total_std.min().item(), sample_total_std.max().item()
            if std_vmax == std_vmin: std_vmax = std_vmin + 1e-8
            
            sq_err_vmin, sq_err_vmax = 0.0, sample_sq_error.max().item()
            if sq_err_vmax == sq_err_vmin: sq_err_vmax = sq_err_vmin + 1e-8
                
            sse_vmin, sse_vmax = 0.0, sample_sse.max().item()
            if sse_vmax == sse_vmin: sse_vmax = sse_vmin + 1e-8

            # NOTE: Assuming `plot_with_colorbar` is provided or imported from your custom libs
            # We mock the call structure here as per your original script
            def _local_plot(val_array, cmap, vmin, vmax, title, ax):
                plt.sca(ax)
                plot_with_colorbar(val_array, Yh, cmap=cmap, vmin=vmin, vmax=vmax, label=title)
                plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
                ax.set_title(title, fontsize=25)
                ax.axis('off')

            # ----------------------------------------------------
            # A. 2x3 AGGREGATE METRICS GRID
            # ----------------------------------------------------
            fig, axes = plt.subplots(2, 3, figsize=(30, 16))
            
            _local_plot(sample_target, "jet", state_vmin, state_vmax, "Truth", axes[0, 0])
            _local_plot(sample_pred, "jet", state_vmin, state_vmax, f"Mean Prediction", axes[0, 1])
            _local_plot(sample_sq_error, "magma", sq_err_vmin, sq_err_vmax, "Squared Error (MSE)", axes[0, 2])
            _local_plot(sample_total_std, "magma", std_vmin, std_vmax, "Standard Deviation", axes[1, 0])
            _local_plot(sample_log_lik, "magma", ll_vmin_global, ll_vmax_global, "Log-likelihood", axes[1, 1])
            _local_plot(sample_sse, "magma", sse_vmin, sse_vmax, "Standardized SE", axes[1, 2])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            grid_path = logs_dir / f"multiplot_grid_2x3_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
            plt.savefig(grid_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            # ----------------------------------------------------
            # B. 2x5 MONTE CARLO SAMPLES
            # ----------------------------------------------------
            fig_mc, axes_mc = plt.subplots(2, 5, figsize=(25, 10))
            fig_mc.suptitle(f"10 Monte Carlo Samples | Sample {test_indices[batch_idx]} | Time: {eval_time_idx} | Lag: {current_lag}", fontsize=25)

            for i in range(10):
                row, col = i // 5, i % 5
                _local_plot(sample_pred_runs[i], "jet", state_vmin, state_vmax, f"MC Run {i+1}", axes_mc[row, col])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            mc_path = logs_dir / f"mc_10_samples_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png"
            plt.savefig(mc_path, dpi=300, bbox_inches="tight")
            plt.close(fig_mc)

def plot_non_mc_batch_diagnostics(model, test_dataset, spatiotemporal_test_collate_fn, mesh_coordinates_norm, fixed_sens, Yh, USE_MU, device, logs_dir, is_probabilistic=False, model_format="np", likelihood=None, y_mean=None, y_std=None):
    """Evaluates a batch for Non-Monte Carlo methods, including GPs."""
    print("\n" + "="*60)
    mode_str = "PROBABILISTIC (GP)" if model_format == "gp" else ("PROBABILISTIC (Non-MC)" if is_probabilistic else "DETERMINISTIC")
    print(f"BATCH TESTING SAMPLES | {mode_str}")
    print("="*60)

    model.eval()
    if likelihood is not None: likelihood.eval()
    
    test_indices = np.array([0, 1, 2])
    test_batch = [test_dataset[idx] for idx in test_indices]
    eval_time_idx = 30 

    for current_lag in [0, 9, 19]:
        x_context, y_context, x_target, y_target = spatiotemporal_test_collate_fn(
            test_batch, mesh_coords=mesh_coordinates_norm, fixed_sensor_locations=fixed_sens,
            use_all_sensors=True, time_idx=eval_time_idx, lag=current_lag,
            use_mu=USE_MU, model_format="np" if model_format == "gp" else model_format
        )

        x_context, y_context, x_target = x_context.to(device), y_context.to(device) if y_context is not None else None, x_target.to(device)

        with torch.no_grad():
            if model_format == "gp":
                y_pred_list, y_pred_var_list = [], []
                # GP must be conditioned one sample at a time
                for b_idx in range(len(test_batch)):
                    x_ctx = x_context[b_idx]
                    y_ctx = y_context[b_idx].squeeze(-1)
                    x_tgt = x_target[b_idx]
                    
                    if y_mean is not None:
                        y_ctx = (y_ctx - y_mean) / y_std
                        
                    model.set_train_data(inputs=x_ctx, targets=y_ctx, strict=False)
                    with gpytorch.settings.fast_pred_var():
                        preds = likelihood(model(x_tgt))
                        pm, pv = preds.mean.cpu(), preds.variance.cpu()
                        
                    if y_mean is not None:
                        pm = pm * y_std + y_mean
                        pv = pv * (y_std ** 2)
                    y_pred_list.append(pm)
                    y_pred_var_list.append(pv)
                    
                y_pred = torch.stack(y_pred_list)
                y_pred_var = torch.stack(y_pred_var_list)
            else:
                outputs = model(x_context, x_target) if y_context is None else model(x_context, y_context, x_target)
                if isinstance(outputs, tuple):
                    y_pred, y_pred_var = outputs[0], outputs[1] if (is_probabilistic and len(outputs) > 1) else None
                else:
                    y_pred, y_pred_var = outputs, None
                y_pred = y_pred.squeeze(-1).cpu()
                if y_pred_var is not None: y_pred_var = y_pred_var.squeeze(-1).cpu()

        y_target_cpu = y_target.squeeze(-1).cpu()

        if is_probabilistic and y_pred_var is not None:
            var_clamp = y_pred_var.clamp_min(1e-8)
            ll_const = math.log(2.0 * math.pi)
            log_lik_all = -0.5 * (ll_const + torch.log(var_clamp) + ((y_target_cpu - y_pred)**2) / var_clamp)
            ll_vmin_global, ll_vmax_global = log_lik_all.min().item(), log_lik_all.max().item()
            if ll_vmax_global == ll_vmin_global: ll_vmax_global = ll_vmin_global + 1e-8
        else:
            log_lik_all = None

        for batch_idx in range(3):
            context = x_context[batch_idx].cpu()[:, 4:6] if USE_MU else x_context[batch_idx].cpu()[:, 1:3]
            sample_pred = y_pred[batch_idx]
            sample_target = y_target_cpu[batch_idx]
            sample_sq_error = (sample_pred - sample_target) ** 2
            
            state_vmin, state_vmax = torch.min(sample_pred.min(), sample_target.min()).item(), torch.max(sample_pred.max(), sample_target.max()).item()
            if state_vmax == state_vmin: state_vmax = state_vmin + 1e-8
            
            sq_err_vmin, sq_err_vmax = 0.0, sample_sq_error.max().item()
            if sq_err_vmax == sq_err_vmin: sq_err_vmax = sq_err_vmin + 1e-8

            def _local_plot(val_array, cmap, vmin, vmax, title, ax):
                plt.sca(ax)
                plot_with_colorbar(val_array, Yh, cmap=cmap, vmin=vmin, vmax=vmax, label=title)
                plt.scatter(context[:, 0], context[:, 1], color='red', s=20)
                ax.set_title(title, fontsize=25)
                ax.axis('off')

            if is_probabilistic and y_pred_var is not None:
                sample_var = y_pred_var[batch_idx]
                sample_std = torch.sqrt(sample_var.clamp_min(1e-8))
                sample_log_lik = log_lik_all[batch_idx]
                sample_sse = sample_sq_error / sample_var.clamp_min(1e-8)

                std_vmin, std_vmax = sample_std.min().item(), sample_std.max().item()
                if std_vmax == std_vmin: std_vmax = std_vmin + 1e-8
                sse_vmin, sse_vmax = 0.0, sample_sse.max().item()
                if sse_vmax == sse_vmin: sse_vmax = sse_vmin + 1e-8

                fig, axes = plt.subplots(2, 3, figsize=(30, 16))
                _local_plot(sample_target, "jet", state_vmin, state_vmax, "Truth", axes[0, 0])
                _local_plot(sample_pred, "jet", state_vmin, state_vmax, "Mean Prediction", axes[0, 1])
                _local_plot(sample_sq_error, "magma", sq_err_vmin, sq_err_vmax, "Squared Error (MSE)", axes[0, 2])
                _local_plot(sample_std, "magma", std_vmin, std_vmax, "Standard Deviation", axes[1, 0])
                _local_plot(sample_log_lik, "magma", ll_vmin_global, ll_vmax_global, "Log-likelihood", axes[1, 1])
                _local_plot(sample_sse, "magma", sse_vmin, sse_vmax, "Standardized SE", axes[1, 2])
                grid_suffix = "prob_2x3"
            else:
                fig, axes = plt.subplots(1, 3, figsize=(30, 8))
                _local_plot(sample_target, "jet", state_vmin, state_vmax, "Truth", axes[0])
                _local_plot(sample_pred, "jet", state_vmin, state_vmax, "Prediction", axes[1])
                _local_plot(sample_sq_error, "magma", sq_err_vmin, sq_err_vmax, "Squared Error", axes[2])
                grid_suffix = "det_1x3"

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(logs_dir / f"multiplot_grid_{grid_suffix}_sample{test_indices[batch_idx]}_time{eval_time_idx}_lag{current_lag}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

def evaluate_scenario(model, dataset, spatiotemporal_test_collate_fn, mesh_coordinates_norm, device, time_idx, lag, sensors_to_use, drop_options, mc_samples=100, is_mc=True, model_format="np", likelihood=None, y_mean=None, y_std=None):
    all_ll, all_se, all_sse, all_mse = [], [], [], []

    for idx in range(len(dataset)):
        test_batch = [dataset[idx]]
        x_c, y_c, x_t, y_t = spatiotemporal_test_collate_fn(
            test_batch, mesh_coords=mesh_coordinates_norm, fixed_sensor_locations=sensors_to_use,
            use_all_sensors=True, drop_random_sensors_options=drop_options, time_idx=time_idx, lag=lag,
            model_format="np" if model_format == "gp" else model_format 
        )

        x_c, x_t, y_t = x_c.to(device), x_t.to(device), y_t.to(device)
        if y_c is not None: y_c = y_c.to(device)

        with torch.no_grad():
            if model_format == "gp":
                # ==========================================
                # EXACT GAUSSIAN PROCESS INFERENCE
                # ==========================================
                x_ctx, y_ctx, x_tgt = x_c[0], y_c[0].squeeze(-1), x_t[0]
                y_true = y_t[0].squeeze(-1).cpu()

                if y_mean is not None: y_ctx = (y_ctx - y_mean) / y_std

                model.set_train_data(inputs=x_ctx, targets=y_ctx, strict=False)
                with gpytorch.settings.fast_pred_var():
                    preds = likelihood(model(x_tgt))
                    pred_mean, pred_var = preds.mean.cpu(), preds.variance.cpu()

                if y_mean is not None:
                    pred_mean = pred_mean * y_std + y_mean
                    pred_var = pred_var * (y_std ** 2)

                final_ll = gaussian_log_lik(pred_mean, pred_var, y_true)
            elif is_mc:
                # ==========================================
                # MONTE CARLO INFERENCE (NP)
                # ==========================================
                try:
                    mu_mc, var_mc, *_ = model(x_c, y_c, x_t, num_samples=mc_samples)
                    mc_means, mc_vars = mu_mc.squeeze(-1).cpu(), var_mc.squeeze(-1).cpu()
                except TypeError:
                    means, vars_ = zip(*[(m.squeeze(-1).cpu(), v.squeeze(-1).cpu()) for m, v, *_ in [model(x_c, y_c, x_t) for _ in range(mc_samples)]])
                    mc_means, mc_vars = torch.stack(means, 0), torch.stack(vars_, 0)

                pred_mean = mc_means.mean(0).squeeze(0)
                pred_var = mc_vars.mean(0).squeeze(0) + mc_means.var(0, unbiased=False).squeeze(0)
                y_true = y_t.squeeze(-1).squeeze(0).cpu()
                final_ll = torch.logsumexp(gaussian_log_lik(mc_means.squeeze(1), mc_vars.squeeze(1), y_true.unsqueeze(0)), dim=0) - math.log(mc_samples)
            else:
                # ==========================================
                # DETERMINISTIC (DeepONet)
                # ==========================================
                outputs = model(x_c, x_t) if y_c is None else model(x_c, y_c, x_t)
                if isinstance(outputs, tuple):
                    pred_mean, pred_var = outputs[0], outputs[1]
                else:
                    pred_mean, pred_var = outputs, None

                pred_mean = pred_mean.squeeze(-1).squeeze(0).cpu()
                y_true = y_t.squeeze(-1).squeeze(0).cpu()

                if pred_var is not None:
                    pred_var = pred_var.squeeze(-1).squeeze(0).cpu()
                    final_ll = gaussian_log_lik(pred_mean, pred_var, y_true)
                else:
                    pred_var = torch.ones_like(pred_mean)
                    final_ll = torch.zeros_like(pred_mean)

        se = (y_true - pred_mean) ** 2
        sse = compute_standardized_se(pred_mean, pred_var, y_true)

        all_ll.append(final_ll.flatten())
        all_se.append(se.flatten())
        all_sse.append(sse.flatten())
        all_mse.append(se.mean().view(1))

    return torch.cat(all_ll).numpy(), torch.cat(all_se).numpy(), torch.cat(all_sse).numpy(), torch.cat(all_mse).numpy()

def main():
    USE_MU = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # 1. SETUP PATHS & DIRECTORIES
    # ==========================================
    # Matches the directory structure from pinball_2.py
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir / f"logs_compare"
    checkpoints_dir_anp = script_dir / f"checkpoints_pinball_{'mu' if USE_MU else 'no_mu'}_new_5sens"
    checkpoints_dir_lnp = script_dir / f"checkpoints_pinball_{'mu' if USE_MU else 'no_mu'}_3"
    checkpoints_dir_deeponet = script_dir / f"checkpoints_pinball_fc_deterministic_{'with_mu' if USE_MU else 'without_mu'}_lag_20"
    checkpoints_dir_shred = script_dir / f"checkpoints_pinball_shred_{'with_mu' if USE_MU else 'without_mu'}_lag_20"
    checkpoints_dir_gp = script_dir / f"checkpoints_pinball_gp_lag_20"
    checkpoints_dir_probdeeponet = script_dir / f"checkpoints_pinball_fc_{'with_mu' if USE_MU else 'without_mu'}_lag20"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "logs_anp").mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    (logs_dir / "logs_lnp").mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    (logs_dir / "logs_deeponet").mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    (logs_dir / "logs_shred").mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    (logs_dir / "logs_gp").mkdir(parents=True, exist_ok=True) # <--- ADD THIS
    (logs_dir / "logs_probdeeponet").mkdir(parents=True, exist_ok=True) # <--- ADD THIS

    # ==========================================
    # 2. LOAD MESH & FENICS SPACE
    # ==========================================
    # Requires FEniCS import here or at the top of the file
    from dolfin import Mesh, FunctionSpace
    
    print("Loading FEniCS Mesh...")
    mesh = Mesh(str(resolve_pinball_asset(script_dir, "Pinball_mesh.xml")))
    Yh = FunctionSpace(mesh, "CG", 1)
    nstate = Yh.dim()
    
    mesh_coordinates = torch.as_tensor(Yh.tabulate_dof_coordinates(), dtype=torch.float32)
    mesh_coordinates_norm = mesh_coordinates  # Using actual coordinates without normalization
    
    fixed_sens = [1573, 6925, 1986]

    # ==========================================
    # 3. LOAD & SPLIT DATA
    # ==========================================
    print("Loading NPZ data and splitting...")
    Data = np.load(str(resolve_pinball_asset(script_dir, "Pinball_data.npz")))
    Y = torch.tensor(Data["y"])
    MU = torch.tensor(Data["mu"])
    
    ntrajectories = 500
    dt = 0.1
    ntimes = round(3.0 / dt) + 1

    # Exact Train/Valid/Test split from pinball_2.py
    np.random.seed(0)
    ntrain = round(0.8 * ntrajectories)
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask!=0)[0]]
    idx_test = idx_valid_test[1::2]

    # After generating idx_train...
    Ytrain = Y[idx_train]
    y_mean = Ytrain.mean().item()
    y_std = Ytrain.std().item()

    # Reshape specifically for the spatiotemporal setup
    Ytest = Y[idx_test].reshape(idx_test.shape[0], ntimes, nstate)
    MUtest = MU[idx_test]

    if USE_MU:
        test_dataset = SpatiotemporalDataset(Ytest, MUtest)
    else:
        test_dataset = SpatiotemporalDataset(Ytest)

    # ==========================================
    # 4. INITIALIZE & LOAD MODEL
    # ==========================================
    print("Initializing LatNP model...")
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
    model_anp = LatNP(
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
        use_deeponet_decoder = True, #whether to use a DeepONet-style decoder
    ).to(device)

    best_model_path = checkpoints_dir_anp / "best_model.pt"
    if best_model_path.exists():
        print(f"Loading weights from {best_model_path}...")
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_anp.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_anp.load_state_dict(checkpoint)
    else:
        print(f"\n[!] WARNING: Model checkpoint not found at {best_model_path}! Running with untrained weights.\n")

    model_anp.eval()

    # ==========================================
    # 5. RUN EVALUATION DIAGNOSTICS
    # ==========================================

    # ==========================================
    # 5D. Plot Ground Truth with Sensor Annotations
    # ==========================================
    
    print("\nGenerating final ground truth plot with sensor locations...")
    time_idx = 30
    sample_idx = 0

    # Extract the truth field for the specific sample and time
    if torch.is_tensor(Ytest):
        truth_field = Ytest[sample_idx, time_idx].cpu().numpy()
    else:
        truth_field = Ytest[sample_idx, time_idx]

    # Get the physical (x, y) coordinates of the fixed sensors
    sensor_coords = mesh_coordinates[fixed_sens].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the base state field
    plot_with_colorbar(truth_field, Yh, ax=ax, cmap="jet", label="True State")

    # Overlay the sensors
    for i, sensor_idx in enumerate(fixed_sens):
        x, y = sensor_coords[i, 0], sensor_coords[i, 1]

        # Draw the red 'X' marker
        ax.scatter(x, y, color='red', s=80, marker='X', edgecolor='black', linewidth=1.5, zorder=5)

        # Annotate with the sensor index number
        ax.annotate(str(sensor_idx),
                    (x, y),
                    xytext=(8, 8), textcoords='offset points',
                    color='black', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.9),
                    zorder=6)

    ax.set_title(f"Ground Truth (Test Trajectory {sample_idx}, Time = {time_idx}) with Sensor Locations", fontsize=15)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    plt.tight_layout()
    
    # Save the figure
    sensor_plot_path = logs_dir / "ground_truth_sensors.png"
    fig.savefig(sensor_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensor position map to {sensor_plot_path}")    
    
    
    # 5A. Single Batch Grid (2x3 Plot + 2x5 MC plot)
    plot_batch_diagnostics(
        model=model_anp, 
        test_dataset=test_dataset, 
        spatiotemporal_test_collate_fn=unified_test_collate_fn, 
        mesh_coordinates_norm=mesh_coordinates_norm, 
        fixed_sens=fixed_sens, 
        Yh=Yh, 
        USE_MU=USE_MU,
        device=device, 
        logs_dir=logs_dir / "logs_anp",
        model_format="np"
    )

    # 5B. Full Dataset Global Distribution
    print("\nRunning Multi-Lag Global Distribution Evaluation...")
    ll_dict_A_anp, se_dict_A_anp, sse_dict_A_anp, mse_dict_A_anp = {}, {}, {}, {}
    lags_to_test = [0, 9, 19]
    colors_A = ["#E63946", "#457B9D", "#2A9D8F"]

    for i, lag in enumerate(lags_to_test):
        label = f"NP (Lag {lag + 1})"
        print(f"  Evaluating {label}...")
        METHOD_STYLES[label] = {"color": colors_A[i], "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            model=model_anp,
            dataset=test_dataset,
            spatiotemporal_test_collate_fn=unified_test_collate_fn,
            mesh_coordinates_norm=mesh_coordinates_norm,
            device=device,
            time_idx=30,
            lag=lag,
            sensors_to_use=fixed_sens,
            drop_options=[0]
        )
        ll_dict_A_anp[label], se_dict_A_anp[label], sse_dict_A_anp[label], mse_dict_A_anp[label] = ll, se, sse, mse

    print("Saving global distribution plots...")
    out_path = logs_dir / "diagnostics_anp_lags_0_9_19.png"
    plot_all_distributions(ll_dict_A_anp, se_dict_A_anp, sse_dict_A_anp, mse_dict_A_anp, out_path=out_path)
    
    # ==========================================
    # 5C. Scenario B: Sensor Ablation Diagnostics
    # ==========================================
    print("\nRunning Scenario B: Lag 9, Dropping Sensors (Ablation Diagnostics)...")
    
    ll_dict_B_anp, se_dict_B_anp, sse_dict_B_anp, mse_dict_B_anp = {}, {}, {}, {}

    # Updated configs to match your 3 fixed sensors: [1573, 6925, 1986]
    configs_B = [
        ("All Sensors",         [1573, 6925, 1986], "#000000"),
        ("Missing Sensor 1573", [6925, 1986],       "#E63946"),
        ("Missing Sensor 6925", [1573, 1986],       "#F4A261"),
        ("Missing Sensor 1986", [1573, 6925],       "#2A9D8F")
    ]

    for label, sens_list, color in configs_B:
        print(f"  Evaluating {label}...")

        # Set styling for the boxplots/histograms
        METHOD_STYLES[label] = {"color": color, "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            model=model_anp,
            dataset=test_dataset,
            spatiotemporal_test_collate_fn=unified_test_collate_fn,
            mesh_coordinates_norm=mesh_coordinates_norm,
            device=device,
            time_idx=30,   # Keep time index consistent
            lag=9,         # Fixed lag for ablation test
            sensors_to_use=sens_list,
            drop_options=[0],
            mc_samples=100
        )

        ll_dict_B_anp[label] = ll
        se_dict_B_anp[label] = se
        sse_dict_B_anp[label] = sse
        mse_dict_B_anp[label] = mse

    print("Saving ablation diagnostics plot...")
    ablation_out_path = logs_dir / "diagnostics_anp_lag9_drop_sensors.png"
    plot_all_distributions(
        ll_dict_B_anp, 
        se_dict_B_anp, 
        sse_dict_B_anp, 
        mse_dict_B_anp, 
        out_path=ablation_out_path
    )
    
    #==========================================
    # 6. Load Latent Neural Process (LatNP) Model
    #==========================================
    
    # Create model
    # Create model
    model_lnp = LatNP_simple(
        x_dim=x_dim,
        y_dim=y_dim,
        r_dim=r_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation=nn.SiLU,
        dropout=0.0,
        is_normalized=True,  # Layer normalization helps training stability
        norm_type='layer',
        fourier_vars=3,
        num_frequencies=32, # number of fourier features for the spatial coordinates (x,y)
        fourier_scale=1.0,  # scale of the fourier features for the spatial coordinates (x,y)
        learnable_fourier=True, # whether to learn the fourier features for the spatial coordinates (x,y)
        use_deeponet_decoder=True, # whether to use a DeepONet decoder for the latent variable z
        p=128,
    ).to(device)
    
    best_model_path_lnp = checkpoints_dir_lnp / "best_model.pt"
    if best_model_path_lnp.exists():
        print(f"Loading weights from {best_model_path_lnp}...")
        checkpoint = torch.load(best_model_path_lnp, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model_lnp.load_state_dict(checkpoint["model_state_dict"])
        else:
            model_lnp.load_state_dict(checkpoint)
    else:
        print(f"\n[!] WARNING: Model checkpoint not found at {best_model_path_lnp}! Running with untrained weights.\n")

    model_lnp.eval()
    
    #==========================================
    # 6a. plots for Latent Neural Process (LatNP) Model
    #==========================================
    
    print("\nGenerating final ground truth plot with sensor locations...")
    time_idx = 30
    sample_idx = 0

    # Extract the truth field for the specific sample and time
    if torch.is_tensor(Ytest):
        truth_field = Ytest[sample_idx, time_idx].cpu().numpy()
    else:
        truth_field = Ytest[sample_idx, time_idx]

    # Get the physical (x, y) coordinates of the fixed sensors
    sensor_coords = mesh_coordinates[fixed_sens].cpu().numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the base state field
    plot_with_colorbar(truth_field, Yh, ax=ax, cmap="jet", label="True State")

    # Overlay the sensors
    for i, sensor_idx in enumerate(fixed_sens):
        x, y = sensor_coords[i, 0], sensor_coords[i, 1]

        # Draw the red 'X' marker
        ax.scatter(x, y, color='red', s=80, marker='X', edgecolor='black', linewidth=1.5, zorder=5)

        # Annotate with the sensor index number
        ax.annotate(str(sensor_idx),
                    (x, y),
                    xytext=(8, 8), textcoords='offset points',
                    color='black', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8, alpha=0.9),
                    zorder=6)

    ax.set_title(f"Ground Truth (Test Trajectory {sample_idx}, Time = {time_idx}) with Sensor Locations", fontsize=15)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)

    plt.tight_layout()
    
    # Save the figure
    sensor_plot_path = logs_dir / "ground_truth_sensors.png"
    fig.savefig(sensor_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensor position map to {sensor_plot_path}")    
    
    
    # 5A. Single Batch Grid (2x3 Plot + 2x5 MC plot)
    plot_batch_diagnostics(
        model=model_lnp, 
        test_dataset=test_dataset, 
        spatiotemporal_test_collate_fn=unified_test_collate_fn, 
        mesh_coordinates_norm=mesh_coordinates_norm, 
        fixed_sens=fixed_sens, 
        Yh=Yh, 
        USE_MU=USE_MU,
        device=device, 
        logs_dir=logs_dir / "logs_lnp",
        model_format="np"
    )

    # 5B. Full Dataset Global Distribution
    print("\nRunning Multi-Lag Global Distribution Evaluation...")
    ll_dict_A_lnp, se_dict_A_lnp, sse_dict_A_lnp, mse_dict_A_lnp = {}, {}, {}, {}
    lags_to_test = [0, 9, 19]
    colors_A = ["#E63946", "#457B9D", "#2A9D8F"]

    for i, lag in enumerate(lags_to_test):
        label = f"NP (Lag {lag + 1})"
        print(f"  Evaluating {label}...")
        METHOD_STYLES[label] = {"color": colors_A[i], "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            model=model_lnp,
            dataset=SpatiotemporalDataset(Ytest, MUtest),
            spatiotemporal_test_collate_fn=unified_test_collate_fn,
            mesh_coordinates_norm=mesh_coordinates_norm,
            device=device,
            time_idx=30,
            lag=lag,
            sensors_to_use=fixed_sens,
            drop_options=[0]
        )
        ll_dict_A_lnp[label], se_dict_A_lnp[label], sse_dict_A_lnp[label], mse_dict_A_lnp[label] = ll, se, sse, mse

    print("Saving global distribution plots...")
    out_path = logs_dir / "diagnostics_lnp_lags_0_9_19.png"
    plot_all_distributions(ll_dict_A_lnp, se_dict_A_lnp, sse_dict_A_lnp, mse_dict_A_lnp, out_path=out_path)

    # ==========================================
    # 5C. Scenario B: Sensor Ablation Diagnostics
    # ==========================================
    print("\nRunning Scenario B: Lag 9, Dropping Sensors (Ablation Diagnostics)...")
    
    ll_dict_B_lnp, se_dict_B_lnp, sse_dict_B_lnp, mse_dict_B_lnp = {}, {}, {}, {}

    # Updated configs to match your 3 fixed sensors: [1573, 6925, 1986]
    configs_B = [
        ("All Sensors",         [1573, 6925, 1986], "#000000"),
        ("Missing Sensor 1573", [6925, 1986],       "#E63946"),
        ("Missing Sensor 6925", [1573, 1986],       "#F4A261"),
        ("Missing Sensor 1986", [1573, 6925],       "#2A9D8F")
    ]

    for label, sens_list, color in configs_B:
        print(f"  Evaluating {label}...")

        # Set styling for the boxplots/histograms
        METHOD_STYLES[label] = {"color": color, "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}

        ll, se, sse, mse = evaluate_scenario(
            model=model_lnp,
            dataset=test_dataset,
            spatiotemporal_test_collate_fn=unified_test_collate_fn,
            mesh_coordinates_norm=mesh_coordinates_norm,
            device=device,
            time_idx=30,   # Keep time index consistent
            lag=9,         # Fixed lag for ablation test
            sensors_to_use=sens_list,
            drop_options=[0],
            mc_samples=100
        )

        ll_dict_B_lnp[label] = ll
        se_dict_B_lnp[label] = se
        sse_dict_B_lnp[label] = sse
        mse_dict_B_lnp[label] = mse

    print("Saving ablation diagnostics plot...")
    ablation_out_path = logs_dir / "diagnostics_lnp_lag9_drop_sensors.png"
    plot_all_distributions(
        ll_dict_B_lnp, 
        se_dict_B_lnp, 
        sse_dict_B_lnp, 
        mse_dict_B_lnp, 
        out_path=ablation_out_path
    )
    
    #==========================================
    # 7. Additional models (DeepONet, Shred, GP, ProbDeepONet) can be loaded and evaluated similarly
    #==========================================
    
    # Conditionally pass nparams to the model based on the toggle
    model_probdeeponet = DeepONetMeanVar(
        num_sensors=len(fixed_sens), 
        num_params=3 if USE_MU else 0,
        coord_dim=3, 
        p=128
    ).to(device)
    
    best_path = checkpoints_dir_probdeeponet / "best_model.pt"
    if best_path.exists():
        model_probdeeponet.load_state_dict(torch.load(best_path, map_location=device))

    
    # 7A. Single Batch Grid (2x3 Plot + 2x5 MC plot)
    plot_non_mc_batch_diagnostics(
        model=model_probdeeponet, 
        test_dataset=test_dataset,       # <--- Use the unified dataset
        spatiotemporal_test_collate_fn=unified_test_collate_fn, 
        mesh_coordinates_norm=mesh_coordinates_norm, 
        fixed_sens=fixed_sens, 
        Yh=Yh, 
        USE_MU=USE_MU,
        device=device, 
        logs_dir=logs_dir / "logs_probdeeponet",  # <--- Fix: Don't overwrite the lnp logs!
        is_probabilistic=True,
        model_format="don"
    )
    
    # ==============================================================================
    # 8. DeepONet (Deterministic)
    # ==============================================================================
    model_don = DeepONetDeterministic(
        num_sensors=len(fixed_sens), 
        num_params=3 if USE_MU else 0,
        coord_dim=3, 
        p=128
    ).to(device)
    
    print("\nRunning DeepONet Evaluation...")
    best_path_don = checkpoints_dir_deeponet / "best_model.pt"
    if best_path_don.exists():
        model_don.load_state_dict(torch.load(best_path_don, map_location=device))
    else:
        print(f"[!] WARNING: Model checkpoint not found at {best_path_don}!")
        
    model_don.eval()

    # 8A. Single Batch Grid (1x3 Plot for deterministic)
    plot_non_mc_batch_diagnostics(
        model=model_don, 
        test_dataset=test_dataset,       # <--- Use unified dataset
        spatiotemporal_test_collate_fn=unified_test_collate_fn, 
        mesh_coordinates_norm=mesh_coordinates_norm, 
        fixed_sens=fixed_sens, 
        Yh=Yh, 
        USE_MU=USE_MU,
        device=device, 
        logs_dir=logs_dir / "logs_deeponet",
        is_probabilistic=False,          # <--- Forces the 1x3 deterministic grid
        model_format="don"
    )

    # ==============================================================================
    # 9. Context-Conditioned GP
    # ==============================================================================
    print("\nRunning Context-Conditioned GP Evaluation...")
    
    likelihood_gp = gpytorch.likelihoods.GaussianLikelihood().to(device)
    dummy_x = torch.zeros(2, 6).to(device)
    dummy_y = torch.zeros(2).to(device)
    model_gp = ContextConditionedGP(dummy_x, dummy_y, likelihood_gp).to(device)

    # Use whatever checkpoint path you saved the GP in
    gp_ckpt_path = checkpoints_dir_gp / "sensor_history_gp.pth" 
    if gp_ckpt_path.exists():
        state_dict = torch.load(gp_ckpt_path, map_location=device)
        model_gp.load_state_dict(state_dict['model_state_dict'])
        likelihood_gp.load_state_dict(state_dict['likelihood_state_dict'])
        print("Loaded GP Weights successfully!")
    else:
        print(f"[!] Warning: GP checkpoint not found at {gp_ckpt_path}! Using untrained weights.")

    model_gp.eval()
    likelihood_gp.eval()

    # 9A. Single Batch Grid (2x3 Plot)
    plot_non_mc_batch_diagnostics(
        model=model_gp, 
        test_dataset=test_dataset, 
        spatiotemporal_test_collate_fn=unified_test_collate_fn, 
        mesh_coordinates_norm=mesh_coordinates_norm, 
        fixed_sens=fixed_sens, 
        Yh=Yh, 
        USE_MU=USE_MU,
        device=device, 
        logs_dir=logs_dir / "logs_gp",
        is_probabilistic=True,
        model_format="gp",              # <--- Important!
        likelihood=likelihood_gp,       # <--- Pass the likelihood
        y_mean=y_mean, y_std=y_std      # <--- Pass the normalizers
    )

    # ==============================================================================
    # 10. SHRED (Deterministic)
    # ==============================================================================
    print("\nRunning SHRED Evaluation...")
    
    # SHRED requires the POD basis matrix V to decode its output back to the spatial field.
    kstate = 100  # Number of POD modes
    
    try:
        from utils.models import SHRED
        
        print(f"Computing manual POD basis matrix (V) for top {kstate} modes...")
        # Flatten time and batch dimensions into a single Snapshot Matrix
        Ytrain_flat = Ytrain.reshape(-1, nstate).to(device)
        
        # Manually compute POD via low-rank SVD (Y = U * S * V^T)
        # V returned by PyTorch has shape (nstate, kstate)
        U, S, V = torch.svd_lowrank(Ytrain_flat, q=kstate)
        
        # Transpose to (kstate, nstate) for easy decoding: (Batch, kstate) @ (kstate, nstate)
        V_matrix = V.T 

        # Initialize base SHRED model
        shred_base = SHRED(
            len(fixed_sens) + (3 if USE_MU else 0), # in_dim
            kstate,                                 # out_dim
            hidden_size=64, 
            hidden_layers=2, 
            decoder_sizes=[350, 400], 
            dropout=0.1
        ).to(device)
        
        best_path_shred = checkpoints_dir_shred / "best_model.pt"
        if best_path_shred.exists():
            shred_base.load_state_dict(torch.load(best_path_shred, map_location=device))
            print("Loaded SHRED weights successfully!")
        else:
            print(f"[!] WARNING: Model checkpoint not found at {best_path_shred}!")
            
        shred_base.eval()

        # Wrap SHRED so it cleanly ignores the DON relative time shift and projects via V
        class SHREDWrapper(nn.Module):
            def __init__(self, shred_model, V_matrix, use_mu):
                super().__init__()
                self.shred = shred_model
                self.V = V_matrix
                self.use_mu = use_mu
                
            def forward(self, sensor_history, coords):
                # sensor_history shape: (Batch, History_Len, Sensors + Time + Params)
                nsens = len(fixed_sens)
                sensors = sensor_history[:, :, :nsens]
                if self.use_mu:
                    mu = sensor_history[:, :, -3:]
                    shred_in = torch.cat([sensors, mu], dim=-1)
                else:
                    shred_in = sensors
                    
                # Predict POD coefficients (Batch, kstate)
                pod_coeffs = self.shred(shred_in)
                
                # Reconstruct full spatial state (Batch, nstate)
                pred_state = torch.matmul(pod_coeffs, self.V)
                
                return pred_state.unsqueeze(-1)

        model_shred = SHREDWrapper(shred_base, V_matrix, USE_MU).to(device)
        model_shred.eval()

        # 10A. Single Batch Grid (1x3 Plot for deterministic)
        plot_non_mc_batch_diagnostics(
            model=model_shred, 
            test_dataset=test_dataset, 
            spatiotemporal_test_collate_fn=unified_test_collate_fn, 
            mesh_coordinates_norm=mesh_coordinates_norm, 
            fixed_sens=fixed_sens, 
            Yh=Yh, 
            USE_MU=USE_MU,
            device=device, 
            logs_dir=logs_dir / "logs_shred",
            is_probabilistic=False,
            model_format="don"  # Passes sensor_history correctly
        )
    except ImportError:
        print("[!] Could not import SHRED from utils.models. Skipping SHRED.")
        model_shred = None

    # ==============================================================================
    # 11. DIRECT MODEL COMPARISON: ANP vs LNP vs Prob-DeepONet vs DeepONet
    # ==============================================================================
    print("\n" + "="*60)
    print("RUNNING DIRECT COMPARISON (All Sensors, Lag 19)")
    print("="*60)

    ll_dict_cmp, se_dict_cmp, sse_dict_cmp, mse_dict_cmp = {}, {}, {}, {}
    max_lag = 19
    
    # 1. Evaluate ANP
    print("  Evaluating ANP...")
    METHOD_STYLES["ANP"] = {"color": "#E63946", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}
    ll, se, sse, mse = evaluate_scenario(
        model=model_anp, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=True, model_format="np"
    )
    ll_dict_cmp["ANP"], se_dict_cmp["ANP"], sse_dict_cmp["ANP"], mse_dict_cmp["ANP"] = ll, se, sse, mse

    # 2. Evaluate LNP
    print("  Evaluating LNP...")
    METHOD_STYLES["LNP"] = {"color": "#457B9D", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}
    ll, se, sse, mse = evaluate_scenario(
        model=model_lnp, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=True, model_format="np"
    )
    ll_dict_cmp["LNP"], se_dict_cmp["LNP"], sse_dict_cmp["LNP"], mse_dict_cmp["LNP"] = ll, se, sse, mse

    # 3. Evaluate Probabilistic DeepONet
    print("  Evaluating Prob-DeepONet...")
    METHOD_STYLES["Prob-DeepONet"] = {"color": "#2A9D8F", "linestyle": "-.", "linewidth": 2.2, "alpha": 0.85}
    ll, se, sse, mse = evaluate_scenario(
        model=model_probdeeponet, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="don"
    )
    ll_dict_cmp["Prob-DeepONet"], se_dict_cmp["Prob-DeepONet"], sse_dict_cmp["Prob-DeepONet"], mse_dict_cmp["Prob-DeepONet"] = ll, se, sse, mse

    # 4. Evaluate Standard (Deterministic) DeepONet
    print("  Evaluating DeepONet (Deterministic)...")
    METHOD_STYLES["DeepONet"] = {"color": "#E9C46A", "linestyle": ":", "linewidth": 2.2, "alpha": 0.85} # Unique style
    ll, se, sse, mse = evaluate_scenario(
        model=model_don, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="don"
    )
    
    # 5. Evaluate Context-Conditioned GP
    print("  Evaluating Context-GP...")
    METHOD_STYLES["Context-GP"] = {"color": "#F4A261", "linestyle": "-", "linewidth": 2.2, "alpha": 0.85}
    ll, se, sse, mse = evaluate_scenario(
        model=model_gp, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="gp",
        likelihood=likelihood_gp, y_mean=y_mean, y_std=y_std
    )
    ll_dict_cmp["Context-GP"] = ll
    se_dict_cmp["Context-GP"] = se
    sse_dict_cmp["Context-GP"] = sse
    mse_dict_cmp["Context-GP"] = mse
    
    # 6. Evaluate SHRED
    if 'model_shred' in locals() and model_shred is not None:
        print("  Evaluating SHRED (Deterministic)...")
        METHOD_STYLES["SHRED"] = {"color": "#8338EC", "linestyle": ":", "linewidth": 2.2, "alpha": 0.85}
        ll, se, sse, mse = evaluate_scenario(
            model=model_shred, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
            mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
            sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="don"
        )
        # Note: Skip LL and SSE for SHRED because it is deterministic!
        se_dict_cmp["SHRED"], mse_dict_cmp["SHRED"] = se, mse
    
    # NOTE: We intentionally skip adding DeepONet to ll_dict_cmp and sse_dict_cmp!
    # Because it is deterministic, it has no variance, making LL and SSE invalid.
    se_dict_cmp["DeepONet"], mse_dict_cmp["DeepONet"] = se, mse

    # 5. Generate the Unified Plot
    print("Saving comparison diagnostics plot...")
    cmp_out_path = logs_dir / "diagnostics_comparison_all_sensors_lag19.png"
    plot_all_distributions(ll_dict_cmp, se_dict_cmp, sse_dict_cmp, mse_dict_cmp, out_path=cmp_out_path)
    

if __name__ == "__main__":
    main()