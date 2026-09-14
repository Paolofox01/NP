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


def keep_dataset_samples(batch):
    """Keep GoPro dataset tuples intact for the existing NP input builder."""
    return batch


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
    figure, axes = plt.subplots(
        len(rows), 2, figsize=(16, 14), gridspec_kw={"width_ratios": [2.0, 1.25]}
    )
    for row_index, (title, key) in enumerate(rows):
        axis, box_axis = axes[row_index]
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
        box_values = [np.asarray(values).reshape(-1) for values in metrics[key].values()]
        box_names = list(metrics[key].keys())
        box_plot = box_axis.boxplot(
            box_values, patch_artist=True, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
        )
        for patch, name in zip(box_plot["boxes"], box_names):
            patch.set_facecolor(METHOD_STYLES.get(name, {"color": "#000000"})["color"])
            patch.set_alpha(0.7)
        box_axis.set_xticks(range(1, len(box_names) + 1))
        box_axis.set_xticklabels(box_names, rotation=20, ha="right")
        box_axis.set_title(f"{title} boxplot")
        box_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _image_sensor_coordinates(sensors, height: int, width: int):
    sensor_indices = np.asarray(sensors, dtype=int)
    return sensor_indices % width, sensor_indices // width


def plot_probabilistic_image_diagnostics(
    model,
    model_name: str,
    test_dataset,
    coordinates,
    sensors,
    device,
    height: int,
    width: int,
    output_dir: Path,
    samples: int = 10,
) -> None:
    """Save Pinball-style field diagnostics for one GoPro probabilistic model."""
    batch = [test_dataset[index] for index in range(min(2, len(test_dataset)))]
    x_context, y_context, x_target, y_target = np_eval_inputs(batch, coordinates, sensors)
    with torch.no_grad():
        output = model(
            x_context.to(device), y_context.to(device), x_target.to(device), num_samples=samples
        )
    means, variances = output[:2]
    if means.dim() == 3:
        means, variances = means.unsqueeze(0), variances.unsqueeze(0)

    target = y_target.to(device).squeeze(-1).cpu()
    mean_prediction = means.mean(dim=0).squeeze(-1).cpu()
    variance = (variances.mean(dim=0) + means.var(dim=0, unbiased=False)).clamp_min(1e-8)
    total_std = variance.sqrt().squeeze(-1).cpu()
    sample_predictions = means.squeeze(-1).cpu()
    sample_variances = variances.squeeze(-1).clamp_min(1e-8).cpu()
    sample_log_likelihoods = -0.5 * (
        math.log(2.0 * math.pi)
        + sample_variances.log()
        + (target.unsqueeze(0) - sample_predictions).square() / sample_variances
    )
    log_likelihood = torch.logsumexp(sample_log_likelihoods, dim=0) - math.log(samples)
    squared_error = (mean_prediction - target).square()
    standardized_error = squared_error / variance.squeeze(-1).cpu()

    state_min = min(target.min().item(), mean_prediction.min().item())
    state_max = max(target.max().item(), mean_prediction.max().item())
    error_max = max(squared_error.max().item(), 1e-8)
    std_max = max(total_std.max().item(), 1e-8)
    ll_min, ll_max = log_likelihood.min().item(), log_likelihood.max().item()
    sse_max = max(standardized_error.max().item(), 1e-8)
    sensor_columns, sensor_rows = _image_sensor_coordinates(sensors, height, width)

    print(f"\n{model_name} image diagnostics (first test batch):")
    for batch_index in range(len(batch)):
        print(
            f"  sample {batch_index}: MSE={squared_error[batch_index].mean():.6f}, "
            f"MAE={(mean_prediction[batch_index] - target[batch_index]).abs().mean():.6f}, "
            f"mean log-likelihood={log_likelihood[batch_index].mean():.6f}"
        )

        def draw(axis, image, title, cmap, vmin, vmax):
            plot = axis.imshow(image.reshape(height, width).numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            axis.scatter(sensor_columns, sensor_rows, color="cyan", marker="x", s=45, linewidths=1.5)
            axis.set_title(title)
            axis.axis("off")
            return plot

        figure, axes = plt.subplots(2, 3, figsize=(15, 8))
        panels = (
            (target[batch_index], "Truth", "gray", state_min, state_max),
            (mean_prediction[batch_index], "Mean prediction", "gray", state_min, state_max),
            (squared_error[batch_index], "Squared error", "magma", 0.0, error_max),
            (total_std[batch_index], "Total standard deviation", "magma", 0.0, std_max),
            (log_likelihood[batch_index], "Log-likelihood", "magma", ll_min, ll_max),
            (standardized_error[batch_index], "Standardized squared error", "magma", 0.0, sse_max),
        )
        for axis, panel in zip(axes.flat, panels):
            draw(axis, *panel)
        figure.suptitle(f"{model_name} diagnostics, test sample {batch_index}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{model_name.lower()}_diagnostics_sample{batch_index}.png", dpi=200)
        plt.close(figure)

        figure, axes = plt.subplots(2, 5, figsize=(20, 8))
        for sample_index, axis in enumerate(axes.flat):
            draw(axis, sample_predictions[sample_index, batch_index], f"MC sample {sample_index + 1}", "gray", state_min, state_max)
        figure.suptitle(f"{model_name} Monte Carlo samples, test sample {batch_index}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{model_name.lower()}_mc_samples_sample{batch_index}.png", dpi=200)
        plt.close(figure)


def plot_deterministic_image_diagnostics(
    model,
    model_name: str,
    test_dataset,
    coordinates,
    sensors,
    device,
    height: int,
    width: int,
    output_dir: Path,
) -> None:
    """Save target, prediction, and squared-error images for a deterministic model."""
    batch = [test_dataset[index] for index in range(min(2, len(test_dataset)))]
    history, coords, target = don_eval_inputs(batch, coordinates, sensors)
    with torch.no_grad():
        output = model(history.to(device), coords.to(device))
        if isinstance(output, tuple):
            prediction, variance = output
            variance = variance.clamp_min(1e-8).sqrt().cpu()
        else:
            prediction, variance = output, None
        prediction = prediction.cpu()
    target = target.cpu()
    squared_error = (prediction - target).square()
    sensor_columns, sensor_rows = _image_sensor_coordinates(sensors, height, width)
    state_min = min(target.min().item(), prediction.min().item())
    state_max = max(target.max().item(), prediction.max().item())
    error_max = max(squared_error.max().item(), 1e-8)

    for batch_index in range(len(batch)):
        panels = [
            (target[batch_index], "Truth", "gray", state_min, state_max),
            (prediction[batch_index], "Prediction", "gray", state_min, state_max),
            (squared_error[batch_index], "Squared error", "magma", 0.0, error_max),
        ]
        if variance is not None:
            panels.append((variance[batch_index], "Standard deviation", "magma", 0.0, max(variance.max().item(), 1e-8)))
        figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
        axes = np.atleast_1d(axes)
        for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
            plot = axis.imshow(image.reshape(height, width).numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            axis.scatter(sensor_columns, sensor_rows, color="cyan", marker="x", s=45, linewidths=1.5)
            axis.set_title(title)
            axis.axis("off")
            figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(f"{model_name} diagnostics, test sample {batch_index}")
        figure.tight_layout()
        figure.savefig(output_dir / f"{model_name.lower()}_diagnostics_sample{batch_index}.png", dpi=200)
        plt.close(figure)


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    output_dir = data_dir / "logs_compare"
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, coordinates, sensors, _, height, width, _, _ = prepare_data(data_dir)
    loader = DataLoader(
        datasets["test"], batch_size=2, shuffle=False, collate_fn=keep_dataset_samples
    )

    anp = load_checkpoint(LatNP(x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2, activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3, num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=False).to(device), data_dir / "checkpoints_gopro_anp/phase2/best_model.pt", device)
    lnp = load_checkpoint(LatNP_simple(x_dim=3, y_dim=1, r_dim=128, z_dim=128, hidden_dim=128, n_hidden=2, activation=nn.ReLU, is_normalized=True, norm_type="layer", fourier_vars=3, num_frequencies=32, learnable_fourier=True, use_deeponet_decoder=False).to(device), data_dir / "checkpoints_gopro_lnp/phase2/best_model.pt", device)
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
    plot_probabilistic_image_diagnostics(
        anp, "ANP", datasets["test"], coordinates, sensors, device,
        height, width, output_dir,
    )
    plot_probabilistic_image_diagnostics(
        lnp, "LNP", datasets["test"], coordinates, sensors, device,
        height, width, output_dir,
    )
    plot_deterministic_image_diagnostics(
        prob_don, "Prob-DeepONet", datasets["test"], coordinates, sensors, device,
        height, width, output_dir,
    )
    plot_deterministic_image_diagnostics(
        don, "DeepONet", datasets["test"], coordinates, sensors, device,
        height, width, output_dir,
    )
    for name, values in metrics["mse"].items():
        print(f"{name:16s} MSE: {np.mean(values):.6f}")


if __name__ == "__main__":
    main()
