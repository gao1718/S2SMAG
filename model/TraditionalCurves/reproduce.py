"""Reproduce the C1 classical comparisons and empirical forecast errors."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.S2SMAG.reproduce import (
    autoregressive_predictions,
    load_checkpoint,
    load_normalization,
    load_partition,
)
from model.TraditionalCurves.block_recursive_forecast import block_recursive_predict
from model.TraditionalCurves.classical_point_models import fit_segment


SHORT_CASES = (("Loading", 15), ("Maintained vacuum", 36), ("Vacuum loss response", 124))
LONG_CASES = (("30% history", 42), ("40% history", 57), ("50% history", 71))
CLASSICAL_METHODS = ("Hyperbolic", "ThreePoint", "Asaoka")
PHASES = ("Loading", "Maintained vacuum", "Vacuum loss response")
HORIZON = 10


def load_observations(path):
    observations = pd.read_csv(path, float_precision="round_trip", parse_dates=["date"])
    observations = observations[["date", "time_index", "settlement", "settlement_rate", "vacuum"]]
    time_index = observations.time_index.to_numpy()
    date_diff = observations.date.diff().iloc[1:]
    features = observations.iloc[:, 2:].to_numpy(dtype=np.float64)
    if (len(observations) != 143 or observations.date.isna().any()
            or not date_diff.eq(pd.Timedelta(days=1)).all()
            or not np.array_equal(time_index, np.arange(len(observations)))
            or not np.isfinite(features).all()):
        raise ValueError("Expected the 143 consecutive daily C1 observations, indexed from zero.")
    return observations, features


def forecast_windows(model, state, data_dir, device, observations, features, norm):
    arrays = load_partition(data_dir, "test")
    window_starts = np.asarray(arrays["window_starts"]).reshape(-1)
    point_indices = np.asarray(arrays["point_indices"]).reshape(-1)
    starts = np.arange(len(observations) - 2 * HORIZON + 1)
    if not np.array_equal(window_starts, starts) or not (point_indices == 0).all():
        raise ValueError("The test data must contain all stride-one C1 windows in date order.")
    leads = np.arange(1, HORIZON + 1)
    target_indices = starts[:, None] + HORIZON - 1 + leads
    history_indices = starts[:, None] + np.arange(HORIZON)
    normalized = (features - norm["feature_min"]) / norm["feature_scale"]
    expected = {
        "encoder_inputs": normalized[history_indices],
        "future_inputs": normalized[target_indices, 2, None],
        "targets": ((features[target_indices, 0] - norm["target_min"]) / norm["target_scale"])[..., None],
    }
    for name, values in expected.items():
        np.testing.assert_array_equal(arrays[name], values.astype(np.float32),
                                      err_msg=f"{name} does not match the original C1 data.")
    y_scaled = autoregressive_predictions(
        model, arrays, device, batch_size=int(state["config"]["BATCH_SIZE"])
    )
    predictions = y_scaled.astype(np.float64).reshape(-1) * norm["target_scale"] + norm["target_min"]
    if not np.isfinite(predictions).all():
        raise ValueError("The checkpoint produced nonfinite C1 predictions.")
    indices = target_indices.ravel()
    return pd.DataFrame({
        "start": np.repeat(starts, HORIZON),
        "step": np.tile(leads, len(starts)),
        "time_index": indices,
        "date": observations.date.to_numpy()[indices],
        "actual": features[indices, 0],
        "predicted": predictions,
        "future_vacuum": features[indices, 2],
    })


def use_archived_predictions(windows, path):
    archive = pd.read_csv(path, float_precision="round_trip")
    keys = ["start", "step", "time_index"]
    paired = windows.merge(archive[[*keys, "actual", "predicted"]], on=keys,
                           suffixes=("", "_archive"), validate="one_to_one")
    if len(paired) != len(windows) or len(archive) != len(windows):
        raise ValueError("Archived predictions must include exactly the original C1 window–lead pairs.")
    # Allow the rounding of archived labels that were inverse-transformed from float32.
    np.testing.assert_allclose(paired.actual_archive, paired.actual, rtol=0, atol=2e-5)
    differences = np.abs(paired.predicted_archive - paired.predicted)
    largest_difference = float(differences.max())
    if not np.isfinite(differences).all() or largest_difference > 0.1:
        raise ValueError(f"Archived predictions differ from the checkpoint by up to {largest_difference:g} mm.")
    windows["predicted"] = paired.predicted_archive.to_numpy()
    return largest_difference


def assign_error_phases(windows, observations):
    vacuum = observations.vacuum.to_numpy()
    threshold_crossings = np.flatnonzero(vacuum >= 80.0)
    if not len(threshold_crossings):
        raise ValueError("The C1 phase analysis requires a measured vacuum of at least 80 kPa.")
    first80 = int(threshold_crossings[0])
    loss_crossings = np.flatnonzero((np.arange(len(vacuum)) > first80) & (vacuum < 70.0))
    if not len(loss_crossings):
        raise ValueError("The C1 phase analysis requires a subsequent vacuum below 70 kPa.")
    loss = int(loss_crossings[0])
    windows["phase"] = np.asarray(PHASES)[np.searchsorted(
        [first80, loss], windows.time_index, side="right"
    )]
    windows["error"] = windows.predicted - windows.actual
    return first80, loss


def comparison_cases(model, device, observations, features, norm, windows, first80):
    rows, fits, trace = [], [], []
    time = observations.time_index.to_numpy()
    for task, cases in (("short", SHORT_CASES), ("long", LONG_CASES)):
        for label, cutoff in cases:
            indices = time[cutoff:cutoff + HORIZON] if task == "short" else time[cutoff:]
            given = features[indices, 2]
            common = {"task": task, "case": label, "history_days": cutoff}

            def record(protocol, method, predicted, status, reason, eligible):
                for step, (index, value) in enumerate(zip(indices, predicted, strict=True), 1):
                    actual = features[index, 0]
                    rows.append({"protocol": protocol, **common, "method": method,
                                 "step": step, "time_index": index, "date": observations.date.iloc[index],
                                 "actual": actual, "predicted": value, "error": value - actual,
                                 "future_vacuum": features[index, 2], "status": status,
                                 "reason": reason, "ranking_eligible": eligible})

            if task == "short":
                prediction = windows.loc[windows.start == cutoff - HORIZON, "predicted"].to_numpy()
                record("first80", "S2SMAG", prediction, "ok", "", True)
            else:
                recursion = block_recursive_predict(
                    model, features[cutoff - HORIZON:cutoff], given, norm, device=device
                )
                trace.extend({**common, **entry} for entry in recursion["history_trace"])
                record("first80", "S2SMAG", recursion["prediction"], recursion["status"],
                       recursion["reason"], recursion["status"] == "ok")

            for protocol, start in (("first80", min(first80, cutoff)), ("full_prefix_diagnostic", 0)):
                for method in CLASSICAL_METHODS:
                    result = fit_segment(method, time[:cutoff], features[:cutoff, 0], indices, start)
                    diag = result["diagnostics"]
                    rank_ok = method == "ThreePoint" or diag.get("rank") == 2
                    admissible = (result["status"] == "ok" and rank_ok
                                  and diag.get("parameter_condition_met", False))
                    eligible = protocol == "first80" and admissible
                    fits.append({**common, "method": method, "protocol": protocol,
                                 "status": result["status"], "reason": result["reason"],
                                 "result": diag, "parameters": result["parameters"],
                                 "used_indices": result["used_indices"].tolist(),
                                 "ranking_eligible": eligible})
                    if protocol == "first80" and not admissible:
                        if np.isfinite(result["prediction"]).any():
                            record("first80_diagnostic", method, result["prediction"],
                                   result["status"], result["reason"], False)
                        record(protocol, method, np.full(len(indices), np.nan), "inapplicable",
                               result["reason"] or "The fit does not meet the classical parameter conditions.", False)
                    else:
                        record(protocol, method, result["prediction"], result["status"], result["reason"], eligible)
    return pd.DataFrame(rows), fits, pd.DataFrame(trace)


def summarize(frame, keys):
    rows = []
    for group, part in frame.groupby(keys, sort=False):
        row = dict(zip(keys, group if isinstance(group, tuple) else (group,)))
        row.update(n_total=len(part), n_evaluated=int(np.isfinite(part.error).sum()))
        if "ranking_eligible" in part:
            row.update(ranking_eligible=bool(part.ranking_eligible.all()), status=part.status.iloc[0])
        finite = part.loc[np.isfinite(part.error)]
        if len(finite):
            error, actual = finite.error.to_numpy(), finite.actual.to_numpy()
            squared = np.square(error)
            total_variation = np.square(actual - actual.mean()).sum()
            q05, q25, median, q75, q95 = np.quantile(error, [.05, .25, .5, .75, .95])
            row.update(MAE=np.abs(error).mean(), RMSE=np.sqrt(squared.mean()), bias=error.mean(),
                       R2=1 - squared.sum() / total_variation if total_variation > 0 else np.nan,
                       q05=q05, q25=q25, median=median, q75=q75, q95=q95)
        rows.append(row)
    return pd.DataFrame(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, frame):
    frame.to_csv(path, index=False, float_format="%.17g", date_format="%Y-%m-%d")


def write_json(path, data):
    path.write_text(json.dumps(json_ready(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared C1/C9 split directory")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a locally trained checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--archived-predictions", type=Path,
                        help="Optional original C1 prediction CSV, checked against the checkpoint")
    parser.add_argument("--plots", action="store_true", help="Export SVG and PNG figures")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to keep previous runs intact.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu or --device auto.")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    observed_path = args.data_dir / "C1_observed_source.csv"
    observations, features = load_observations(observed_path)
    model, state = load_checkpoint(args.checkpoint, device)
    if state.get("file_hashes"):
        for split, filename in (("norm", "norm_params.npz"), ("test", "test_dataset_norm.npz")):
            if state["file_hashes"][split] != sha256(args.data_dir / filename):
                raise ValueError(f"{filename} does not match the checkpoint's data manifest.")
    norm = load_normalization(args.data_dir)
    windows = forecast_windows(model, state, args.data_dir, device, observations, features, norm)
    archive_difference = None
    if args.archived_predictions:
        archive_difference = use_archived_predictions(windows, args.archived_predictions)
    first80, loss = assign_error_phases(windows, observations)
    comparisons, fits, trace = comparison_cases(model, device, observations, features, norm, windows, first80)

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "window_errors.csv", windows)
    write_csv(args.output / "lead_metrics.csv", summarize(windows, ["step"]))
    write_csv(args.output / "phase_metrics.csv", summarize(windows, ["phase"]))
    write_csv(args.output / "phase_lead_metrics.csv", summarize(windows, ["phase", "step"]))
    write_csv(args.output / "classical_comparison_predictions.csv", comparisons)
    write_csv(args.output / "classical_comparison_metrics.csv",
              summarize(comparisons, ["protocol", "task", "case", "history_days", "method"]))
    write_csv(args.output / "recursive_trace.csv", trace)
    write_json(args.output / "classical_fit_parameters.json", fits)
    sources = [observed_path, args.data_dir / "norm_params.npz", args.data_dir / "test_dataset_norm.npz",
               args.checkpoint]
    if args.archived_predictions:
        sources.append(args.archived_predictions)
    write_json(args.output / "analysis_manifest.json", {
        "checkpoint_epoch": state["epoch"],
        "device": str(device),
        "source_files": [{"name": path.name, "sha256": sha256(path)} for path in sources],
        "short_and_window_prediction_source": "archive" if args.archived_predictions else "checkpoint_forward",
        "archive_vs_checkpoint_max_difference_mm": archive_difference,
        "future_vacuum": "Observed C1 vacuum for retrospective conditional forecasts.",
        "classical_fit": "Fixed first day with vacuum at least 80 kPa within the observed prefix, "
                         "with a minimum of seven calibration days. Full-prefix fits are diagnostics.",
        "recursion": "Native ten-day blocks initialized from ten observations. Later blocks receive "
                     "predicted settlement, its daily difference, and the given vacuum inputs.",
        "error_definition": "Predicted minus observed cumulative settlement, in mm.",
        "error_assignment": "Target date. Loading ends before the first recorded vacuum of at least "
                            "80 kPa. Vacuum loss starts at the first subsequent reading below 70 kPa. "
                            "These are descriptive phase proxies.",
        "first_80_kPa_date": observations.date.iloc[first80].date().isoformat(),
        "first_subsequent_below_70_kPa_date": observations.date.iloc[loss].date().isoformat(),
        "window_lead_pairs": len(windows),
        "phase_counts": windows.phase.value_counts(sort=False).to_dict(),
        "error_quantiles": "Empirical distribution of errors over overlapping windows.",
    })
    if args.plots:
        from model.TraditionalCurves.reproduction_plots import save_plots

        save_plots(args.output, observations, comparisons, windows)
    print(f"C1 analysis complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
