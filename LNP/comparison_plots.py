"""
comparison_plots.py
-------------------
Side-by-side comparison figures for several models on the same test sample, built AT THE FINAL PRINTED SIZE
(so that the text is not shrunk when the figure is placed on an A4 page).  Independent of the benchmark and of
PyTorch: the benchmark-specific part (how a field is drawn on a sphere, on an image, or on a finite element mesh)
is passed in as a ``draw_fn``.

Two figures:

  1. plot_model_comparison(...)
       Truth | Prediction | Squared error   (Prediction = predictive mean for probabilistic models,
       point prediction for deterministic ones).
  2. plot_probabilistic_diagnostics(...)
       Truth | Mean | Squared error | Std. deviation | Log-likelihood | Standardized SE, probabilistic models only.

Orientation
  "models_as_rows"    : one row per model, one column per quantity (models one above the other).
  "models_as_columns" : one column per model, one row per quantity (truth above prediction above error, ...).
Every quantity shares one colour scale and one colour bar, so the models can be compared directly.

Sizes (all in inches / points, at print size)
  print_width : width of the figure = the width it will have on the page (A4 portrait text width ~ 6.3 in;
                A4 landscape ~ 9.4 in; one column of a two-column layout ~ 3.3 in).
  max_height  : the figure is made narrower if it would be taller than this (A4 text height ~ 9.2 in).
  font_size   : base font size in points (titles 8, labels 7, ticks 6 by default): keep it >= 7 for print.
  panel_aspect: width / height of one panel (sphere 2.0, image W/H, ...).

Data format
  entries : dict  model name -> {"pred": 1-D array (N,), "var": 1-D array (N,) or None}
            ("var" is the total predictive variance; None for deterministic models)
  truth   : 1-D array (N,)
  draw_fn : callable(ax, values, cmap, vmin, vmax) -> matplotlib mappable   (used for the colour bar)
  color_limits : dict with the keys "state", "squared_error", "standard_deviation", "log_likelihood",
                 "standardized_error", each a (vmin, vmax) tuple (as in compute_dataset_diagnostic_limits).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_2PI = math.log(2.0 * math.pi)
DEFAULT_MODEL_ORDER = ("ANP", "LNP", "Context-GP", "Prob-DeepONet", "SHRED", "DeepONet")

A4_PORTRAIT_TEXT_WIDTH = 6.3     # inches (~16 cm)
A4_TEXT_HEIGHT = 9.2             # inches (~23.4 cm)


def to_numpy(x) -> np.ndarray:
    """Convert a torch tensor (or anything array-like) to a float64 numpy array without importing torch."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _vec(x) -> np.ndarray:
    return to_numpy(x).reshape(-1)


def compute_fields(truth, pred, var=None) -> dict:
    """Point-wise diagnostics, with the same definitions as the per-model diagnostic figures of the scripts."""
    truth, pred = _vec(truth), _vec(pred)
    fields = {"truth": truth, "pred": pred, "squared_error": (pred - truth) ** 2}
    if var is not None:
        var = np.clip(_vec(var), 1e-8, None)
        sse = fields["squared_error"] / var
        fields.update(
            standard_deviation=np.sqrt(var),
            standardized_error=sse,
            log_likelihood=-0.5 * (LOG_2PI + np.log(var) + sse),
        )
    return fields


def default_color_limits(truth, max_standardized_error: float = 10.0) -> dict:
    """Fallback colour limits (same definition as compute_dataset_diagnostic_limits in the scripts)."""
    t = _vec(truth)
    lo, hi = float(t.min()), float(t.max())
    if np.isclose(lo, hi):
        lo, hi = lo - 1e-4, hi + 1e-4
    std = max(float(t.std()), 1e-8)
    ll_max = -0.5 * math.log(2.0 * math.pi * std ** 2)
    return {
        "state": (lo, hi),
        "squared_error": (0.0, (2.0 * std) ** 2),
        "standard_deviation": (0.0, 1.5 * std),
        "log_likelihood": (ll_max - 0.5 * max_standardized_error, ll_max),
        "standardized_error": (0.0, max_standardized_error),
    }


def _ordered(entries: dict, model_order=None) -> list:
    order = list(model_order) if model_order is not None else list(DEFAULT_MODEL_ORDER)
    known = [n for n in order if n in entries]
    rest = [n for n in entries if n not in known]
    return known + rest


def _wrap_name(name: str) -> str:
    """Insert a line break after a hyphen in long model names (e.g. Prob-DeepONet -> Prob-/DeepONet)."""
    if len(name) > 9 and "-" in name:
        head, tail = name.split("-", 1)
        return head + "-\n" + tail
    return name


def _figure_geometry(n_models, n_quant, orientation, print_width, max_height, panel_aspect):
    """Figure size (in) at print size, and the panel size, so that the fonts are not rescaled afterwards."""
    if orientation == "models_as_rows":
        n_cols, n_rows = n_quant, n_models
        left, right = 0.42, 0.10          # row labels
        top, bottom = 0.55, 0.50          # column titles, horizontal colour bars
        panel_w = (print_width - left - right) / n_cols
        panel_h = panel_w / panel_aspect
        height = n_rows * panel_h + top + bottom
        if height > max_height:           # narrower figure instead of smaller text
            panel_h = (max_height - top - bottom) / n_rows
            panel_w = panel_h * panel_aspect
        width = n_cols * panel_w + left + right
        height = n_rows * panel_h + top + bottom
    else:
        n_cols, n_rows = n_models, n_quant
        left, right = 0.42, 0.62          # row labels, vertical colour bars with tick labels
        top, bottom = 0.45, 0.10          # model names
        panel_w = (print_width - left - right) / n_cols
        panel_h = panel_w / panel_aspect
        height = n_rows * panel_h + top + bottom
        if height > max_height:
            panel_h = (max_height - top - bottom) / n_rows
            panel_w = panel_h * panel_aspect
        width = n_cols * panel_w + left + right
        height = n_rows * panel_h + top + bottom
    return width, height, n_rows, n_cols


def _render(model_rows, columns, truth, draw_fn, limits, out_path, title, orientation, print_width, max_height,
            panel_aspect, font_size, state_cmap, error_cmap, dpi):
    """model_rows: list of (name, fields); columns: list of (field_key, label, kind, limit_key)."""
    n_models, n_quant = len(model_rows), len(columns)
    width, height, n_rows, n_cols = _figure_geometry(n_models, n_quant, orientation, print_width, max_height,
                                                      panel_aspect)
    fs_title, fs_label, fs_tick = font_size, font_size - 1.0, font_size - 2.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    horizontal = orientation == "models_as_rows"
    row_labels = []
    for q, (key, label, kind, limit_key) in enumerate(columns):
        cmap = state_cmap if kind == "state" else error_cmap
        vmin, vmax = limits[limit_key]
        group, mappable = [], None
        for m, (name, fields) in enumerate(model_rows):
            r, c = (m, q) if horizontal else (q, m)
            ax = axes[r, c]
            values = truth if key == "truth" else fields[key]
            mp = draw_fn(ax, values, cmap, vmin, vmax)
            mappable = mp if mappable is None else mappable
            group.append(ax)
            if horizontal:
                if m == 0:
                    ax.set_title(label, fontsize=fs_title, pad=2)
                if q == 0:   # rotated model name; long names are wrapped so that they never exceed the row height
                    row_labels.append((r, _wrap_name(name), fs_label, "bold"))
            else:
                if q == 0:
                    ax.set_title(name, fontsize=fs_title, fontweight="bold", pad=2)
                if m == 0:
                    row_labels.append((r, label, fs_tick, "normal"))
        cbar = fig.colorbar(mappable, ax=group, orientation="horizontal" if horizontal else "vertical",
                            shrink=0.9 if horizontal else 0.85, pad=0.02, aspect=28 if horizontal else 14)
        cbar.ax.tick_params(labelsize=fs_tick, length=2, pad=1)
    if title:
        fig.suptitle(title, fontsize=fs_title + 1)
    fig.canvas.draw()                                  # finalize the constrained layout before measuring
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    for r, text, size, weight in row_labels:
        box = axes[r, 0].get_tightbbox(renderer)       # includes the tick labels of the first column
        (x0, y0), (_, y1) = to_fig.transform((box.x0, box.y0)), to_fig.transform((box.x1, box.y1))
        fig.text(x0 - 0.006, 0.5 * (y0 + y1), text, rotation=90, ha="right", va="center", fontsize=size,
                 fontweight=weight, multialignment="center")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved comparison figure to {out_path}  (print size {width:.1f} x {height:.1f} in, fonts {fs_tick:.0f}-{fs_title + 1:.0f} pt)")
    return out_path


def plot_model_comparison(entries, truth, draw_fn, out_path, color_limits=None, title=None, model_order=None,
                          orientation="models_as_rows", print_width=A4_PORTRAIT_TEXT_WIDTH, max_height=A4_TEXT_HEIGHT,
                          panel_aspect=1.6, font_size=8.0, state_cmap="jet", error_cmap="magma", dpi=300):
    """Truth | Prediction (mean or point prediction) | Squared error, for all the models."""
    if not entries:
        print("  [comparison] no model data collected, skipping the comparison figure.")
        return None
    limits = color_limits if color_limits is not None else default_color_limits(truth)
    names = _ordered(entries, model_order)
    rows = [(n, compute_fields(truth, entries[n]["pred"], entries[n].get("var"))) for n in names]
    columns = [
        ("truth", "Truth", "state", "state"),
        ("pred", "Prediction", "state", "state"),
        ("squared_error", "Squared error", "error", "squared_error"),
    ]
    if orientation == "models_as_columns":
        columns = [("truth", "Truth", "state", "state"), ("pred", "Prediction\n(mean)", "state", "state"),
                   ("squared_error", "Squared\nerror", "error", "squared_error")]
    return _render(rows, columns, _vec(truth), draw_fn, limits, out_path, title, orientation, print_width, max_height,
                   panel_aspect, font_size, state_cmap, error_cmap, dpi)


def plot_probabilistic_diagnostics(entries, truth, draw_fn, out_path, color_limits=None, title=None, model_order=None,
                                   orientation="models_as_columns", print_width=A4_PORTRAIT_TEXT_WIDTH,
                                   max_height=A4_TEXT_HEIGHT, panel_aspect=1.6, font_size=8.0, state_cmap="jet",
                                   error_cmap="magma", dpi=300):
    """Truth | Mean | Squared error | Std. deviation | Log-likelihood | Standardized SE, probabilistic models only.
    The default orientation (models as columns, quantities stacked) fits an A4 portrait page at print size."""
    prob = {n: e for n, e in entries.items() if e.get("var") is not None}
    if not prob:
        print("  [comparison] no probabilistic model collected, skipping the diagnostics figure.")
        return None
    limits = color_limits if color_limits is not None else default_color_limits(truth)
    names = _ordered(prob, model_order)
    rows = [(n, compute_fields(truth, prob[n]["pred"], prob[n]["var"])) for n in names]
    columns = [
        ("truth", "Truth", "state", "state"),
        ("pred", "Mean", "state", "state"),
        ("squared_error", "Squared\nerror", "error", "squared_error"),
        ("standard_deviation", "Std.\ndeviation", "error", "standard_deviation"),
        ("log_likelihood", "Log-\nlikelihood", "error", "log_likelihood"),
        ("standardized_error", "Standardized\nSE", "error", "standardized_error"),
    ]
    return _render(rows, columns, _vec(truth), draw_fn, limits, out_path, title, orientation, print_width, max_height,
                   panel_aspect, font_size, state_cmap, error_cmap, dpi)
