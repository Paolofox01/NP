from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


WELL_DATASET_NAME = "turbulent_radiative_layer_2D"
FIELD_NAME = "density"
COOLING_TIMESCALE_NAMES = (
    "cooling_timescale",
    "cooling_time",
    "cooling_timescale_coefficient",
    "cooling_coefficient",
)
DEFAULT_OUTPUT = "trl_density.npz"


def load_well_dataset(
    well_base_path: str,
    split: str,
    max_trajectories: int | None,
):
    try:
        from the_well.data import WellDataset
    except ImportError as error:
        raise ImportError(
            "The TRL exporter requires The Well package. Install it with `pip install the_well`."
        ) from error

    return WellDataset(
        well_base_path=well_base_path,
        well_dataset_name=WELL_DATASET_NAME,
        well_split_name=split,
        n_steps_input=1,
        n_steps_output=1,
        restrict_num_trajectories=max_trajectories,
        use_normalization=False,
        return_grid=True,
        flatten_tensors=True,
    )


def _extract_field_array(dataset, max_trajectories: int | None) -> np.ndarray:
    arrays = dataset.to_xarray(backend="numpy")
    if FIELD_NAME not in arrays:
        available = ", ".join(arrays.data_vars)
        raise KeyError(
            f"The Well {WELL_DATASET_NAME} dataset does not contain '{FIELD_NAME}'. "
            f"Available fields: {available}"
        )

    field = arrays[FIELD_NAME]
    spatial_dims = [dim for dim in field.dims if dim not in {"sample", "time"}]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2D {FIELD_NAME}, got dimensions {field.dims}.")
    if max_trajectories is not None:
        field = field.isel(sample=slice(0, max_trajectories))
    return np.asarray(field.transpose("sample", "time", *spatial_dims).values, dtype=np.float32)


def _extract_cooling_timescale(dataset, max_trajectories: int | None) -> np.ndarray:
    arrays = dataset.to_xarray(backend="numpy")
    cooling_name = next((name for name in COOLING_TIMESCALE_NAMES if name in arrays), None)
    if cooling_name is None:
        available = ", ".join(arrays.data_vars)
        raise KeyError(
            "Could not find the TRL cooling-timescale coefficient. "
            f"Tried {COOLING_TIMESCALE_NAMES}; available variables: {available}"
        )

    coefficient = arrays[cooling_name]
    if "sample" not in coefficient.dims:
        raise ValueError(
            f"Expected {cooling_name!r} to have a sample dimension, got {coefficient.dims}."
        )
    if max_trajectories is not None:
        coefficient = coefficient.isel(sample=slice(0, max_trajectories))
    coefficient = np.asarray(coefficient.values, dtype=np.float32)
    if coefficient.ndim == 1:
        return coefficient
    if coefficient.ndim == 2 and coefficient.shape[1] == 1:
        return coefficient[:, 0]
    raise ValueError(
        f"Expected {cooling_name!r} to be one scalar per trajectory, got shape {coefficient.shape}."
    )


def export_density_fields(
    well_base_path: str,
    output_path: Path,
    max_train: int | None = None,
    max_valid: int | None = None,
    max_test: int | None = None,
) -> None:
    """Export 2D density fields from The Well's turbulent radiative layer dataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_limits = {"train": max_train, "valid": max_valid, "test": max_test}
    split_arrays = {}
    cooling_timescales = {}
    for split, max_trajectories in split_limits.items():
        print(f"[TRL export] Loading {split} split from {well_base_path}...")
        dataset = load_well_dataset(well_base_path, split, max_trajectories)
        split_arrays[split] = _extract_field_array(dataset, max_trajectories)
        cooling_timescales[split] = _extract_cooling_timescale(dataset, max_trajectories)
        print(f"[TRL export] {split}: {split_arrays[split].shape}")

    shapes = {name: array.shape[-2:] for name, array in split_arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Split spatial shapes differ: {shapes}")

    np.savez_compressed(
        output_path,
        train=split_arrays["train"],
        valid=split_arrays["valid"],
        test=split_arrays["test"],
        train_cooling_timescale=cooling_timescales["train"],
        valid_cooling_timescale=cooling_timescales["valid"],
        test_cooling_timescale=cooling_timescales["test"],
        field=np.asarray(FIELD_NAME),
        source_dataset=np.asarray(WELL_DATASET_NAME),
    )
    print(f"[TRL export] Saved 2D density fields to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 2D density fields from The Well's turbulent radiative layer data."
    )
    parser.add_argument("--well-base-path", required=True, help="Directory containing The Well datasets.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / DEFAULT_OUTPUT)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-valid", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    args = parser.parse_args()

    export_density_fields(
        args.well_base_path,
        args.output,
        max_train=args.max_train,
        max_valid=args.max_valid,
        max_test=args.max_test,
    )


if __name__ == "__main__":
    main()
