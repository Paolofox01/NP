from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GoPro.gopro_common import (
    DeepONetDeterministic,
    DeepONetMeanVar,
    don_eval_inputs,
    np_eval_inputs,
    prepare_data,
)
from LNP.LATNPsimple import LatNP_simple
from LNP.LatentNP import LatNP

METHOD_STYLES = {
    "ANP": {"color": "#E63946"},
    "LNP": {"color": "#457B9D"},
    "Prob-DeepONet": {"color": "#2A9D8F"},
    "DeepONet": {"color": "#E9C46A"},
}


def load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Run its trainer first.")
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.eval()


def collect_probabilistic_np(model, loader, coordinates, sensors, device, samples=10):
    log_likelihood, squared_error, standardized_error, sample_mse = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x_context, y_context, x_target, y_target = np_eval_inputs(batch, coordinates, sensors)
            output = model(x_context.to(device), y_context.to(device), x_target.to(device), num_samples=samples)
            means, variances = output[:2]
            if means.dim() == 3:
                means, variances = means.unsqueeze(0), variances.unsqueeze(0)
            prediction = means.mean(dim=0)
            variance = variances.mean(dim=0) + means.var(dim=0, unbiased=False)
            target = y_target.to(device)
            variance = variance.clamp_min(1e-8)
            error = (target - prediction).square()
            log_likelihood.append((-0.5 * (math.log(2.0 * math.pi) + variance.log() + error / variance)).cpu().numpy().ravel())
            standardized_error.append((error / variance).cpu().numpy().ravel())
            squared_error.append(error.cpu().numpy().ravel())
            sample_mse.extend(error.mean(dim=(1, 2)).cpu().tolist())
    return log_likelihood, squared_error, standardized_error, sample_mse


def collect_probabilistic_don(model, loader, coordinates, sensors, device):
    log_likelihood, squared_error, standardized_error, sample_mse = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            history, coords, target = don_eval_inputs(batch, coordinates, sensors)
            prediction, variance = model(history.to(device), coords.to(device))
            variance = variance.clamp_min(1e-8)
            error = (target.to(device) - prediction).square()
            log_likelihood.append((-0.5 * (math.log(2.0 * math.pi) + variance.log() + error / variance)).cpu().numpy().ravel())
            standardized_error.append((error / variance).cpu().numpy().ravel())
            squared_error.append(error.cpu().numpy().ravel())
            sample_mse.extend(error.mean(dim=1).cpu().tolist())
    return log_likelihood, squared_error, standardized_error, sample_mse


def collect_deterministic_don(model, loader, coordinates, sensors, device):
    squared_error, sample_mse = [], []
    with torch.no_grad():
        for batch in loader:
            history, coords, target = don_eval_inputs(batch, coordinates, sensors)
            error = (target.to(device) - model(history.to(device), coords.to(device))).square()
            squared_error.append(error.cpu().numpy().ravel())
            sample_mse.extend(error.mean(dim=1).cpu().tolist())
    return squared_error, sample_mse


def print_diagnostic_summary(metrics) -> None:
    """Print descriptive statistics for every model in each diagnostic."""
    diagnostic_names = {
        "ll": "Log likelihood",
        "sse": "Standardized squared error",
        "se": "Squared error",
        "mse": "Per-frame MSE",
    }
    print("\nDIAGNOSTIC SUMMARY")
    for key, diagnostic_name in diagnostic_names.items():
        print(f"\n{diagnostic_name}")
        for model_name, values in metrics[key].items():
            values = np.asarray(values, dtype=float).reshape(-1)
            if values.size == 0:
                print(f"  {model_name}: no values")
                continue

            mean = values.mean()
            median = np.median(values)
            std = values.std()
            relative_std = std / abs(mean) if not np.isclose(mean, 0.0) else np.nan
            relative_std_text = f"{relative_std:.2%}" if np.isfinite(relative_std) else "n/a (mean is zero)"
            print(
                f"  {model_name}: mean={mean:.6g}, median={median:.6g}, "
                f"std={std:.6g}, relative std={relative_std_text}"
            )


def plot_distributions(metrics, output_path: Path) -> None:
    print_diagnostic_summary(metrics)
    rows = [("Log likelihood", "ll"), ("Standardized squared error", "sse"), ("Squared error", "se"), ("Per-frame MSE", "mse")]
    figure, axes = plt.subplots(len(rows), 1, figsize=(10, 14))
    for axis, (title, key) in zip(axes, rows):
        nonempty_values = [np.asarray(values).reshape(-1) for values in metrics[key].values() if np.asarray(values).size]
        if nonempty_values:
            all_values = np.concatenate(nonempty_values)
            lower, upper = np.percentile(all_values, [0.5, 99.5])
            if np.isclose(lower, upper):
                upper = lower + 1e-8
            bin_edges = np.linspace(lower, upper, 81)
            for name, values in metrics[key].items():
                values = np.asarray(values).reshape(-1)
                if values.size:
                    visible_values = values[(values >= lower) & (values <= upper)]
                    style = METHOD_STYLES.get(name, {"color": "#000000"})
                    axis.hist(visible_values, bins=bin_edges, density=True, alpha=0.35, color=style["color"], label=name)
            axis.set_xlim(lower, upper)
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    output_dir = data_dir / "logs_compare"
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, coordinates, sensors, _, _, _, _, _ = prepare_data(data_dir)
    loader = DataLoader(datasets["test"], batch_size=2, shuffle=False)

    anp = load_checkpoint(LatNP(x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2, activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3, num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=True).to(device), data_dir / "checkpoints_gopro_anp/phase2/best_model.pt", device)
    lnp = load_checkpoint(LatNP_simple(x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2, activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3, num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=True).to(device), data_dir / "checkpoints_gopro_lnp/phase2/best_model.pt", device)
    prob_don = load_checkpoint(DeepONetMeanVar(len(sensors)).to(device), data_dir / "checkpoints_gopro_prob_don/best_model.pt", device)
    don = load_checkpoint(DeepONetDeterministic(len(sensors)).to(device), data_dir / "checkpoints_gopro_don_det/best_model.pt", device)

    metrics = {"ll": {}, "sse": {}, "se": {}, "mse": {}}
    for name, model in (("ANP", anp), ("LNP", lnp)):
        ll, se, sse, mse = collect_probabilistic_np(model, loader, coordinates, sensors, device)
        metrics["ll"][name], metrics["se"][name], metrics["sse"][name], metrics["mse"][name] = np.concatenate(ll), np.concatenate(se), np.concatenate(sse), mse
    ll, se, sse, mse = collect_probabilistic_don(prob_don, loader, coordinates, sensors, device)
    metrics["ll"]["Prob-DeepONet"], metrics["se"]["Prob-DeepONet"], metrics["sse"]["Prob-DeepONet"], metrics["mse"]["Prob-DeepONet"] = np.concatenate(ll), np.concatenate(se), np.concatenate(sse), mse
    se, mse = collect_deterministic_don(don, loader, coordinates, sensors, device)
    metrics["se"]["DeepONet"], metrics["mse"]["DeepONet"] = np.concatenate(se), mse
    plot_distributions(metrics, output_dir / "gopro_model_comparison.png")
    for name, values in metrics["mse"].items():
        print(f"{name:16s} MSE: {np.mean(values):.6f}")


if __name__ == "__main__":
    main()
