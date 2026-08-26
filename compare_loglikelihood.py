from __future__ import print_function

from dolfin import *
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import sys
import os
from pathlib import Path
from functools import partial
import math
import matplotlib.patches as mpatches
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Domain & Model modules
from processdata import trajectory, trajectories, multiplot
from LNP.LatentNP import LatNP
from LNP.LATNPsimple import LatNP_simple
from architectures.Fourier import FourierFeatures, LearnableFourierFeatures
from LNP.loss_np import ELBOLossNP
from LNP.training import train_np
import gpytorch

# ============================================================
# 0. KAGGLE PATH CONFIGURATION
# ============================================================
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    BASE_DATA_DIR = Path("/kaggle/input/datasets/filippovolpicelli/pinball-data")
    KAGGLE_NB1 = Path("/kaggle/input/notebooks/filippovolpicelli/pinball-nb/NP")
    KAGGLE_NB2 = Path("/kaggle/input/notebooks/filippovolpicelli/notebookae593e1af4/NP")
    
    OUTPUT_LOGS_DIR = Path("/kaggle/working/logs_compare")
    
    # Define checkpoint paths mapping
    CHECKPOINT_PATHS = {
        # Data & Assets
        "mesh": BASE_DATA_DIR / "Pinball_mesh.xml",
        "data": BASE_DATA_DIR / "Pinball_data.npz",
        "shred": BASE_DATA_DIR / "Pinball_idx_fixedsensors.pt",
        "gp": BASE_DATA_DIR / "sensor_history_gp.pth",

        # Model Checkpoints (Adjust keys to match your models)
        "anp_mu": KAGGLE_NB2 / "checkpoints_pinball_no_dndec_mu_new_5sens/best_model.pt",
        "anp_no_mu": KAGGLE_NB2 / "checkpoints_pinball_no_dndec_no_mu_new_5sens/best_model.pt",
        "lnp_mu": KAGGLE_NB1 / "checkpoints_pinball_no_dndec_mu_3/phase2/best_model.pt",
        "lnp_no_mu": KAGGLE_NB1 / "checkpoints_pinball_no_dndec_no_mu_3/phase2/best_model.pt",
        
        # DeepONet / GP / SHRED checkpoints (update with your Kaggle paths if available)
        "probdeeponet": KAGGLE_NB1 / "checkpoints_pinball_fc_without_mu_lag20/best_model.pt",
        "deeponet": KAGGLE_NB1 / "checkpoints_pinball_fc_deterministic_without_mu_lag_20/best_model.pt",
        
        }
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    from pinball_paths import resolve_pinball_asset
    
    OUTPUT_LOGS_DIR = SCRIPT_DIR / "logs_compare"
    USE_MU = False
    
    CHECKPOINT_PATHS = {
        "mesh": resolve_pinball_asset(SCRIPT_DIR, "Pinball_mesh.xml"),
        "data": resolve_pinball_asset(SCRIPT_DIR, "Pinball_data.npz"),
        "fixed_sensors": resolve_pinball_asset(SCRIPT_DIR, "Pinball_idx_fixedsensors.pt"),
        
        "anp": SCRIPT_DIR / f"checkpoints_pinball_{'mu' if USE_MU else 'no_mu'}_new_5sens/best_model.pt",
        "lnp": SCRIPT_DIR / f"checkpoints_pinball_{'mu' if USE_MU else 'no_mu'}_3/best_model.pt",
        "probdeeponet": SCRIPT_DIR / f"checkpoints_pinball_fc_{'with_mu' if USE_MU else 'without_mu'}_lag20/best_model.pt",
        "deeponet": SCRIPT_DIR / f"checkpoints_pinball_fc_deterministic_{'with_mu' if USE_MU else 'without_mu'}_lag_20/best_model.pt",
        "gp": SCRIPT_DIR / "checkpoints_pinball_gp_lag_20/sensor_history_gp.pth",
        "shred": SCRIPT_DIR / f"checkpoints_pinball_shred_{'with_mu' if USE_MU else 'without_mu'}_lag_20/best_model.pt",
    }


def load_model_checkpoint(model, path, device):
    """Safely loads model weights handling checkpoint dict wrappers."""
    path = Path(path)
    if not path.exists():
        print(f"[!] WARNING: Checkpoint not found at: {path}. Using random weights.")
        return model
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(str(path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model


# ============================================================
# 1. HELPERS & COLLATERS
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
    model_format="np"
):
    if mesh_coords is None: raise ValueError("mesh_coords must be provided!")
    if not isinstance(mesh_coords, torch.Tensor): mesh_coords = torch.as_tensor(mesh_coords, dtype=torch.float32)

    batch_size = len(batch)
    
    if isinstance(batch[0], (tuple, list)):
        ntimes, nstate = batch[0][0].shape
        batch_trajs = torch.stack([item[0] for item in batch])
        batch_mus = torch.stack([item[1] for item in batch]) if use_mu else None
    else:
        ntimes, nstate = batch[0].shape
        batch_trajs = torch.stack(batch)
        batch_mus = None
        if use_mu: raise ValueError("use_mu is True, but dataset did not return MU.")

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

    if time_idx is None: time_idx = np.random.randint(20, ntimes - 1) 
    if lag is None: lag = np.random.choice([0, 4, 9, 19]) 

    time_window = torch.arange(time_idx - lag, time_idx + 1)
    history_len = lag + 1

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

    elif model_format == "don":
        batch_indices_2d = torch.arange(batch_size).unsqueeze(1)
        history_states = batch_trajs[batch_indices_2d, time_window]
        sensor_history_3d = history_states[:, :, sensor_locations]

        offsets = torch.arange(-lag, 1)
        t_relative = (offsets.float() / float(ntimes)).unsqueeze(0).unsqueeze(-1)
        t_repeated = t_relative.expand(batch_size, history_len, -1)
        
        if batch_mus is not None:
            batch_mus = batch_mus.view(batch_size, ntimes, -1)
            mu_history = batch_mus[batch_indices_2d, time_window]
            sensor_history = torch.cat([sensor_history_3d, t_repeated, mu_history], dim=-1)
        else:
            sensor_history = torch.cat([sensor_history_3d, t_repeated], dim=-1)

        t_target = torch.zeros((batch_size, nstate, 1), dtype=torch.float32)
        spatial_coords = mesh_coords.unsqueeze(0).expand(batch_size, -1, -1) 
        coords = torch.cat([t_target, spatial_coords], dim=-1)
        y_target = batch_trajs[:, time_idx, :].unsqueeze(-1) 
        
        return sensor_history.contiguous(), None, coords.contiguous(), y_target.contiguous()
    else:
        raise ValueError("model_format must be 'np' or 'don'")

class SpatiotemporalDataset(Dataset):
    def __init__(self, data, mu_params=None):
        self.data = torch.from_numpy(data).float() if isinstance(data, np.ndarray) else data.float()
        self.mu_params = torch.from_numpy(mu_params).float() if isinstance(mu_params, np.ndarray) else (mu_params.float() if mu_params is not None else None)

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
# 2. MODEL DEFINITIONS
# ==============================================================================
class DeepONetDeterministic(nn.Module):
    def __init__(self, num_sensors=10, num_params=0, history_length=20, coord_dim=3, p=128, num_frequencies=32):
        super().__init__()
        self.p = p
        self.num_params = num_params
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1 + num_params,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.branch_proj = nn.Linear(256, p)
        self.fourier_mapping = FourierFeatures(
            input_dim=coord_dim, 
            num_frequencies=num_frequencies,
            scale=6.0,
            learnable=False
        )
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
        branch_out = self.branch_proj(h_n[-1]).unsqueeze(1)
        coords_fourier = self.fourier_mapping(coords[:, :, :])
        trunk_input = torch.cat([coords, coords_fourier], dim=-1)
        trunk_out = self.trunk(trunk_input)
        pred = torch.sum(branch_out * trunk_out, dim=-1) / math.sqrt(self.p) + self.bias
        return pred

class DeepONetMeanVar(nn.Module):
    def __init__(self, num_sensors=10, num_params=0, history_length=20, coord_dim=3, p=128, num_frequencies=32):
        super().__init__()
        self.p = p
        self.num_params = num_params
        self.branch_lstm = nn.LSTM(
            input_size=num_sensors + 1 + num_params,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.branch_proj = nn.Linear(256, p * 2)
        self.fourier_mapping = LearnableFourierFeatures(
            input_dim=coord_dim, 
            num_frequencies=num_frequencies,
            init_scale=1.0
        )
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
        branch_out = self.branch_proj(h_n[-1]).unsqueeze(1)
        coords_fourier = self.fourier_mapping(coords[:, :, :])
        trunk_input = torch.cat([coords, coords_fourier], dim=-1)
        trunk_out = self.trunk(trunk_input)
        branch_mean, branch_var = torch.split(branch_out, self.p, dim=-1)
        trunk_mean, trunk_var = torch.split(trunk_out, self.p, dim=-1)
        mean = torch.sum(branch_mean * trunk_mean, dim=-1) / math.sqrt(self.p) + self.mean_bias
        var_raw = torch.sum(branch_var * trunk_var, dim=-1) / math.sqrt(self.p) + self.var_bias
        var = F.softplus(var_raw) + 1e-6
        return mean, var

class ContextConditionedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=6)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ============================================================
# 3. MATH, METRICS & PLOTTING HELPERS
# ============================================================
def gaussian_log_lik(y_pred_mean, y_pred_var, y_true):
    var_clamp = y_pred_var.clamp_min(1e-8)
    ll_const = math.log(2.0 * math.pi)
    return -0.5 * (ll_const + torch.log(var_clamp) + ((y_true - y_pred_mean)**2) / var_clamp)

def compute_standardized_se(y_pred_mean, y_pred_var, y_true):
    var_clamp = y_pred_var.clamp_min(1e-8)
    return ((y_true - y_pred_mean)**2) / var_clamp

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

    ax_hist.set_xlim(x_lo, x_hi)
    ax_hist.set_xlabel(xlabel, fontsize=12)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.set_title(title, fontsize=15)
    ax_hist.tick_params(axis='both', labelsize=12)
    ax_hist.legend(handles=legend_handles, framealpha=0.85, fontsize=12)

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

    if log_scale in [True, 'log']: ax_box.set_yscale('log')
    elif log_scale == 'symlog': ax_box.set_yscale('symlog')

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

# ============================================================
# 4. MAIN EXECUTION
# ============================================================
def main():
    USE_MU = False  # Toggle whether to use mu conditioning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device} | Kaggle Environment: {IS_KAGGLE}")

    # Create all log subdirectories in output location
    logs_dir = OUTPUT_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["logs_anp", "logs_lnp", "logs_deeponet", "logs_shred", "logs_gp", "logs_probdeeponet"]:
        (logs_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1. Load Mesh
    print("Loading FEniCS Mesh...")
    mesh = Mesh(str(CHECKPOINT_PATHS["mesh"]))
    Yh = FunctionSpace(mesh, "CG", 1)
    nstate = Yh.dim()
    mesh_coordinates = torch.as_tensor(Yh.tabulate_dof_coordinates(), dtype=torch.float32)
    mesh_coordinates_norm = mesh_coordinates

    # 2. Load Sensors & Data
    if CHECKPOINT_PATHS["fixed_sensors"].exists():
        fixed_sens = torch.load(str(CHECKPOINT_PATHS["fixed_sensors"]), weights_only=False)
        if isinstance(fixed_sens, torch.Tensor):
            fixed_sens = fixed_sens.tolist()
    else:
        fixed_sens = [1573, 6925, 1986]

    print("Loading NPZ data and splitting...")
    Data = np.load(str(CHECKPOINT_PATHS["data"]))
    Y = torch.tensor(Data["y"])
    MU = torch.tensor(Data["mu"])
    
    ntrajectories = 500
    dt = 0.1
    ntimes = round(3.0 / dt) + 1

    np.random.seed(0)
    ntrain = round(0.8 * ntrajectories)
    idx_train = np.random.choice(ntrajectories, size=ntrain, replace=False)
    mask = np.ones(ntrajectories)
    mask[idx_train] = 0
    idx_valid_test = np.arange(0, ntrajectories)[np.where(mask!=0)[0]]
    idx_test = idx_valid_test[1::2]

    Ytrain = Y[idx_train]
    y_mean = Ytrain.mean().item()
    y_std = Ytrain.std().item()

    Ytest = Y[idx_test].reshape(idx_test.shape[0], ntimes, nstate)
    MUtest = MU[idx_test]
    test_dataset = SpatiotemporalDataset(Ytest, MUtest if USE_MU else None)

    # 3. Model Hyperparameters
    x_dim = 6 if USE_MU else 3
    y_dim = 1
    r_dim = 128
    z_dim = 128
    hidden_dim = 128
    n_hidden = 2

    # Choose correct checkpoint key depending on USE_MU flag
    anp_key = "anp_mu" if (IS_KAGGLE and USE_MU) else ("anp_no_mu" if IS_KAGGLE else "anp")
    lnp_key = "lnp_mu" if (IS_KAGGLE and USE_MU) else ("lnp_no_mu" if IS_KAGGLE else "lnp")

    # 4. Initialize ANP (Matching checkpoints with `no_dndec`)
    print("Initializing ANP...")
    model_anp = LatNP(
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
        num_heads=4,
        fourier_scale=1.0,
        learnable_fourier=True,
        use_skip=True,
        use_deeponet_decoder=False, # Set to False for 'no_dndec' checkpoints
    ).to(device)
    model_anp = load_model_checkpoint(model_anp, CHECKPOINT_PATHS[anp_key], device)
    model_anp.eval()

    # 5. Initialize LatNP
    print("Initializing LNP...")
    model_lnp = LatNP_simple(
        x_dim=x_dim,
        y_dim=y_dim,
        r_dim=r_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        n_hidden=n_hidden,
        activation=nn.SiLU,
        dropout=0.0,
        is_normalized=True,
        norm_type='layer',
        fourier_vars=3,
        num_frequencies=32,
        fourier_scale=1.0,
        learnable_fourier=True,
        use_deeponet_decoder=False, # Set to False for 'no_dndec' checkpoints
        p=128,
    ).to(device)
    model_lnp = load_model_checkpoint(model_lnp, CHECKPOINT_PATHS[lnp_key], device)
    model_lnp.eval()

    # 6. Initialize Probabilistic DeepONet
    model_probdeeponet = DeepONetMeanVar(
        num_sensors=len(fixed_sens), 
        num_params=3 if USE_MU else 0,
        coord_dim=3, 
        p=128
    ).to(device)
    model_probdeeponet = load_model_checkpoint(model_probdeeponet, CHECKPOINT_PATHS["probdeeponet"], device)
    model_probdeeponet.eval()

    # 7. Initialize Deterministic DeepONet
    model_don = DeepONetDeterministic(
        num_sensors=len(fixed_sens), 
        num_params=3 if USE_MU else 0,
        coord_dim=3, 
        p=128
    ).to(device)
    model_don = load_model_checkpoint(model_don, CHECKPOINT_PATHS["deeponet"], device)
    model_don.eval()

    # 8. Initialize Context GP
    likelihood_gp = gpytorch.likelihoods.GaussianLikelihood().to(device)
    dummy_x = torch.zeros(2, 6).to(device)
    dummy_y = torch.zeros(2).to(device)
    model_gp = ContextConditionedGP(dummy_x, dummy_y, likelihood_gp).to(device)
    if Path(CHECKPOINT_PATHS["gp"]).exists():
        state_dict = torch.load(str(CHECKPOINT_PATHS["gp"]), map_location=device)
        model_gp.load_state_dict(state_dict['model_state_dict'])
        likelihood_gp.load_state_dict(state_dict['likelihood_state_dict'])
        print("Loaded GP weights.")
    model_gp.eval()
    likelihood_gp.eval()

    # ==============================================================================
    # 9. RUN COMPARISON BENCHMARK (All Models, Lag 19)
    # ==============================================================================
    print("\n" + "="*60)
    print("RUNNING DIRECT COMPARISON ON KAGGLE (Lag 19)")
    print("="*60)

    ll_dict_cmp, se_dict_cmp, sse_dict_cmp, mse_dict_cmp = {}, {}, {}, {}
    max_lag = 19

    # 1. ANP
    ll, se, sse, mse = evaluate_scenario(
        model=model_anp, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=True, model_format="np"
    )
    ll_dict_cmp["ANP"], se_dict_cmp["ANP"], sse_dict_cmp["ANP"], mse_dict_cmp["ANP"] = ll, se, sse, mse

    # 2. LNP
    ll, se, sse, mse = evaluate_scenario(
        model=model_lnp, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=True, model_format="np"
    )
    ll_dict_cmp["LNP"], se_dict_cmp["LNP"], sse_dict_cmp["LNP"], mse_dict_cmp["LNP"] = ll, se, sse, mse

    # 3. Prob-DeepONet
    ll, se, sse, mse = evaluate_scenario(
        model=model_probdeeponet, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="don"
    )
    ll_dict_cmp["Prob-DeepONet"], se_dict_cmp["Prob-DeepONet"], sse_dict_cmp["Prob-DeepONet"], mse_dict_cmp["Prob-DeepONet"] = ll, se, sse, mse

    # 4. DeepONet (Deterministic)
    ll, se, sse, mse = evaluate_scenario(
        model=model_don, dataset=test_dataset, spatiotemporal_test_collate_fn=unified_test_collate_fn,
        mesh_coordinates_norm=mesh_coordinates_norm, device=device, time_idx=30, lag=max_lag,
        sensors_to_use=fixed_sens, drop_options=[0], is_mc=False, model_format="don"
    )
    se_dict_cmp["DeepONet"], mse_dict_cmp["DeepONet"] = se, mse

    # 5. Context-Conditioned GP
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

    # Generate Final Plot
    cmp_out_path = logs_dir / "diagnostics_comparison_all_sensors_lag19.png"
    plot_all_distributions(ll_dict_cmp, se_dict_cmp, sse_dict_cmp, mse_dict_cmp, out_path=cmp_out_path)
    print(f"\nSaved final comparison plot to: {cmp_out_path}")

if __name__ == "__main__":
    main()