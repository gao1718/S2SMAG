"""Build the C2–C8 training, C9 validation, and C1 test datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent
FEATURES = ("settlement", "settlement_rate", "vacuum")
POINTS = tuple(f"C{i}" for i in range(1, 10))
SOURCE_COLUMNS = {
    "settlement": POINTS,
    "settlement_rate": POINTS,
    "vacuum": tuple(f"B416-MM{i}" for i in (1, 2, 3, 6, 8, 9, 10, 11, 12)),
}
PARTITIONS = {"train": tuple(range(1, 8)), "val": (8,), "test": (0,)}
HISTORY = HORIZON = 10
STRIDE = 1
MONITORING_DAYS = 143
SPLIT_SCHEMA = "C1_test_C9_val_spatial_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_monitoring_records(raw_dir: Path) -> tuple[np.ndarray, pd.DatetimeIndex, dict]:
    """Read the mapped columns without changing the source observations."""
    arrays, source_hashes, column_indices = [], {}, {}
    dates = None
    for feature, columns in SOURCE_COLUMNS.items():
        path = raw_dir / f"{feature}.xlsx"
        frame = pd.read_excel(path, engine="openpyxl")
        if len(frame) != MONITORING_DAYS:
            raise ValueError(f"{path.name}: expected {MONITORING_DAYS} daily records, got {len(frame)}.")
        dates_in_file = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], errors="raise"))
        if dates_in_file.hasnans:
            raise ValueError(f"{path.name}: monitoring dates cannot be missing.")
        expected = pd.date_range(dates_in_file[0], periods=len(frame), freq="D")
        if not dates_in_file.equals(expected):
            raise ValueError(f"{path.name}: expected unique and consecutive daily records.")
        if dates is not None and not dates.equals(dates_in_file):
            raise ValueError(f"{path.name}: dates differ from the settlement workbook.")
        dates = dates_in_file
        missing = [column for column in columns if column not in frame.columns[1:]]
        if missing:
            raise ValueError(f"{path.name}: required columns missing: {missing}")
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64).T
        if not np.isfinite(values).all():
            raise ValueError(f"{path.name}: monitoring values must be finite.")
        arrays.append(values)
        column_indices[feature] = [int(frame.columns.get_loc(column) - 1) for column in columns]
        source_hashes[feature] = file_sha256(path)
    records = np.stack(arrays, axis=-1)
    daily_change = np.diff(records[:, :, 0], axis=1, prepend=records[:, :1, 0])
    if not np.allclose(records[:, :, 1], daily_change, rtol=0, atol=1e-10):
        raise ValueError("The settlement-rate workbook does not match the daily settlement increments.")
    provenance = {
        "column_indices_after_time": column_indices,
        "source_columns": {name: list(cols) for name, cols in SOURCE_COLUMNS.items()},
        "source_files": {feature: f"{feature}.xlsx" for feature in FEATURES},
        "source_hashes": source_hashes,
    }
    return records, dates, provenance


def fit_training_scaler(records: np.ndarray) -> dict[str, np.ndarray]:
    training = records[list(PARTITIONS["train"])]
    minimum = training.min(axis=(0, 1))
    maximum = training.max(axis=(0, 1))
    scale = np.where(maximum == minimum, 1.0, maximum - minimum)
    return {
        "feature_min": minimum,
        "feature_max": maximum,
        "feature_scale": scale,
        "target_min": np.asarray(minimum[0]),
        "target_max": np.asarray(maximum[0]),
        "target_scale": np.asarray(scale[0]),
    }


def make_windows(scaled: np.ndarray, points: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Make consecutive history and forecast windows within each point."""
    starts = np.arange(0, scaled.shape[1] - HISTORY - HORIZON + 1, STRIDE, dtype=np.int64)
    point_ids = np.repeat(np.asarray(points, dtype=np.int64), len(starts))
    starts = np.tile(starts, len(points))
    historical_days = starts[:, None] + np.arange(HISTORY)
    future_days = starts[:, None] + HISTORY + np.arange(HORIZON)
    return {
        "encoder_inputs": scaled[point_ids[:, None], historical_days, :].astype(np.float32),
        "future_inputs": scaled[point_ids[:, None], future_days, 2:3].astype(np.float32),
        "targets": scaled[point_ids[:, None], future_days, 0:1].astype(np.float32),
        "point_indices": point_ids,
        "window_starts": starts,
    }


def split_identifier(metadata: dict, scaler: dict) -> str:
    # Use the field order and serialization of the archived C1/C9 protocol.
    keys = (
        "source_hashes", "column_indices_after_time", "source_columns", "feature_names",
        "point_names", "dates", "encoder_length", "horizon", "stride", "train_points",
        "test_point", "val_point",
    )
    payload = {key: metadata[key] for key in keys}
    payload.update(norm=scaler, split_id_schema=SPLIT_SCHEMA)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare(raw_dir: Path, output_dir: Path) -> dict:
    """Generate a self-contained dataset directory from the original workbooks."""
    raw_dir, output_dir = raw_dir.resolve(), output_dir.resolve()
    if output_dir.is_relative_to(raw_dir):
        raise ValueError("Choose an output directory outside the raw-data directory.")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    records, dates, provenance = read_monitoring_records(raw_dir)
    norm = fit_training_scaler(records)
    scaled = (records - norm["feature_min"]) / norm["feature_scale"]
    partitions = {label: make_windows(scaled, points) for label, points in PARTITIONS.items()}
    scaler = {name: values.tolist() for name, values in norm.items()}
    reference_name = "C1_observed_source.csv"
    metadata = {
        **provenance,
        **scaler,
        "schema_version": 2,
        "partition_schema_version": 1,
        "feature_names": list(FEATURES),
        "future_feature_names": ["vacuum"],
        "target_feature_names": ["settlement"],
        "point_names": list(POINTS),
        "dates": [day.isoformat() for day in dates],
        "encoder_length": HISTORY,
        "horizon": HORIZON,
        "stride": STRIDE,
        "train_points": list(PARTITIONS["train"]),
        "val_point": PARTITIONS["val"][0],
        "validation_points": list(PARTITIONS["val"]),
        "test_point": PARTITIONS["test"][0],
        "scaler_fit": "train_points",
        "scaler_fit_points": list(PARTITIONS["train"]),
        "sample_counts": {name: len(arrays["targets"]) for name, arrays in partitions.items()},
        "array_shapes": {
            name: {key: list(array.shape) for key, array in arrays.items()}
            for name, arrays in partitions.items()
        },
        "preparation_status": "complete",
        "observed_reference_file": reference_name,
        "future_covariate_note": "Measured vacuum is treated as a given forecast condition.",
    }
    metadata["split_id"] = split_identifier(metadata, scaler)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=output_dir.parent) as staging:
        prepared = Path(staging) / "dataset"
        prepared.mkdir()
        archives = {name: f"{name}_dataset_norm.npz" for name in partitions}
        for name, arrays in partitions.items():
            np.savez_compressed(prepared / archives[name], **arrays)
        np.savez_compressed(prepared / "norm_params.npz", **norm)
        archives["norm"] = "norm_params.npz"
        metadata["file_hashes"] = {name: file_sha256(prepared / filename) for name, filename in archives.items()}

        reference = pd.DataFrame(records[metadata["test_point"]], columns=list(FEATURES))
        reference.insert(0, "time_index", np.arange(len(dates)))
        reference.insert(0, "date", dates.strftime("%Y-%m-%d"))
        reference.to_csv(
            prepared / reference_name, index=False, encoding="utf-8-sig", float_format="%.17g",
        )
        metadata["observed_reference_sha256"] = file_sha256(prepared / reference_name)
        (prepared / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        prepared.rename(output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "raw")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for prepared datasets.")
    args = parser.parse_args()
    metadata = prepare(args.raw_dir, args.output_dir)
    print(json.dumps({"sample_counts": metadata["sample_counts"], "split_id": metadata["split_id"]}, indent=2))


if __name__ == "__main__":
    main()
