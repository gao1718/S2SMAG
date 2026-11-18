"""Reproduce the 1,000-epoch C2–C8 training, C9 validation, and C1 test protocol."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from model.comparison_runtime import make_loader, read_json, select_device, sha256, write_json
from model.evaluation import regression_metrics
from model.optimizers import Lookahead

from .loss_functions import supervised_loss
from .model_Config import Config
from .s2smag_model import build_model


class ReproductionConfig(Config):
    EXPERIMENT_NAME = "C1_test_C9_val_1000"
    EPOCHS = 1000
    BEST_MODEL_FILENAME = "checkpoint_best.pt"
    FINAL_MODEL_FILENAME = "checkpoint_final.pt"
    TRAIN_MONITOR_WINDOW_STARTS = (0, 30, 60, 90, 120)
    SELECTION_METRIC = "C9_autoregressive_validation_mse"

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in dir(self)
            if name.isupper() and name != "SCRIPT_DIR"
            and not isinstance(getattr(type(self), name), property)
        }


PARTITION_POINTS = {"train": set(range(1, 8)), "val": {8}, "test": {0}}
FILE_NAMES = {
    "train": "train_dataset_norm.npz",
    "val": "val_dataset_norm.npz",
    "test": "test_dataset_norm.npz",
    "norm": "norm_params.npz",
}


def load_partition(data_dir, partition):
    """Load the named author split and check its stations and daily window indices."""
    expected_points = PARTITION_POINTS[partition]
    path = Path(data_dir) / FILE_NAMES[partition]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in (
            "encoder_inputs", "future_inputs", "targets", "point_indices", "window_starts",
        )}
    points = arrays["point_indices"]
    starts = arrays["window_starts"]
    count = 124 * len(expected_points)
    for name, shape in (
        ("encoder_inputs", (count, 10, 3)),
        ("future_inputs", (count, 10, 1)),
        ("targets", (count, 10, 1)),
    ):
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path.name}: {name} must be a finite array with shape {shape}.")
    if (points.shape != (count,) or starts.shape != (count,)
            or not np.issubdtype(points.dtype, np.integer)
            or not np.issubdtype(starts.dtype, np.integer)
            or set(points.tolist()) != expected_points):
        raise ValueError(f"{path.name}: unexpected point indices or window starts.")
    for point in expected_points:
        if not np.array_equal(np.sort(starts[points == point]), np.arange(124)):
            raise ValueError(f"{path.name}: C{point + 1} must have one window per start from 0 to 123.")
    return arrays


def load_normalization(data_dir):
    """Read the training-only feature and target scaling for the published split."""
    with np.load(Path(data_dir) / FILE_NAMES["norm"], allow_pickle=False) as archive:
        scales = {name: archive[name] for name in (
            "feature_min", "feature_max", "feature_scale", "target_min", "target_max", "target_scale",
        )}
    for name, values in scales.items():
        shape = (3,) if name.startswith("feature_") else ()
        if values.shape != shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid normalization array {name}; expected finite shape {shape}.")
    if np.any(scales["feature_scale"] <= 0) or scales["target_scale"] <= 0:
        raise ValueError("Normalization scales must be positive.")
    return scales


def _verified_metadata(data_dir, partitions):
    metadata = read_json(Path(data_dir) / "metadata.json")
    if (metadata["train_points"] != list(range(1, 8)) or metadata["val_point"] != 8
            or metadata["test_point"] != 0 or len(metadata["dates"]) != 143):
        raise ValueError("Expected C2–C8 training, C9 validation, C1 test, and 143 daily observations.")
    for partition in partitions:
        path = Path(data_dir) / FILE_NAMES[partition]
        if sha256(path) != metadata["file_hashes"][partition]:
            raise ValueError(f"{path.name} does not match metadata.json. Check the prepared data directory.")
    return metadata


def _prepare_output(output_dir):
    try:
        output_dir.mkdir(parents=True)
    except FileExistsError:
        raise FileExistsError(f"Choose a new output directory: {output_dir}") from None


@torch.inference_mode()
def _validation_mse(model, loader, device):
    model.eval()
    squared_error = 0.0
    element_count = 0
    for history, future_vacuum, target in loader:
        prediction = model(history.to(device), future_vacuum.to(device))
        residual = prediction.double() - target.to(device).double()
        squared_error += residual.square().sum().item()
        element_count += target.numel()
    mse = squared_error / element_count
    if not math.isfinite(mse):
        raise FloatingPointError("Autoregressive evaluation produced a non-finite MSE.")
    return mse


@torch.inference_mode()
def autoregressive_predictions(model, arrays, device, *, batch_size=16):
    """Predict normalized settlement; feedback uses previous predictions in eval mode."""
    cfg = SimpleNamespace(BATCH_SIZE=batch_size, RANDOM_SEED=42)
    loader = make_loader(arrays, cfg, training=False)
    model.eval()
    return np.concatenate([
        model(history.to(device), future_vacuum.to(device)).cpu().numpy()
        for history, future_vacuum, _ in loader
    ])


def _save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(checkpoint, device):
    """Load a locally generated or original author checkpoint using its model configuration."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(Config.from_dict(state["config"]))
    model.load_state_dict(state["model_state_dict"])
    return model.to(device).eval(), state


def _train_epoch(model, loader, optimizer, forcing_ratio, device, gradient_log, epoch):
    model.train()
    squared_error = 0.0
    element_count = 0
    norms = []
    for batch, (history, future_vacuum, target) in enumerate(loader, start=1):
        history, future_vacuum, target = history.to(device), future_vacuum.to(device), target.to(device)
        optimizer.zero_grad()
        prediction = model(history, future_vacuum, target, teacher_forcing_ratio=forcing_ratio)
        loss = supervised_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}, batch {batch}.")
        loss.backward()
        with torch.no_grad():
            norms_by_parameter = [torch.linalg.vector_norm(parameter.grad)
                                  for parameter in model.parameters() if parameter.grad is not None]
            gradient_norm = torch.linalg.vector_norm(torch.stack(norms_by_parameter)).item()
        gradient_log.writerow((epoch, batch, gradient_norm))
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, batch {batch}.")
        norms.append(gradient_norm)
        optimizer.step()
        squared_error += loss.item() * target.numel()
        element_count += target.numel()
    return squared_error / element_count, np.asarray(norms)


def train(data_dir, output_dir, epochs, device):
    """Fit on C2–C8 and select the earliest minimum C9 MSE without reading C1 windows."""
    cfg = ReproductionConfig()
    cfg.EPOCHS = epochs
    cfg.DEVICE = str(device)
    metadata = _verified_metadata(data_dir, ("train", "val", "norm"))
    scale_squared = float(load_normalization(data_dir)["target_scale"]) ** 2
    training = load_partition(data_dir, "train")
    validation = load_partition(data_dir, "val")
    monitor_mask = np.isin(training["window_starts"], cfg.TRAIN_MONITOR_WINDOW_STARTS)
    monitor = {key: values[monitor_mask] for key, values in training.items()}

    _prepare_output(output_dir)
    config = cfg.as_dict()
    write_json(output_dir / "config.json", config)
    torch.set_num_threads(cfg.CPU_THREADS)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(cfg).to(device)
    train_loader = make_loader(training, cfg, training=True)
    monitor_loader = make_loader(monitor, cfg, training=False)
    validation_loader = make_loader(validation, cfg, training=False)
    adamw = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS,
                             eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
    optimizer = Lookahead(adamw, cfg.LOOKAHEAD_K, cfg.LOOKAHEAD_ALPHA)
    checkpoint = {
        "format_version": 2, "model_name": cfg.MODEL_NAME, "config": config,
        "metadata": metadata, "file_hashes": metadata["file_hashes"],
        "selection_metric": cfg.SELECTION_METRIC,
    }
    best_mse = math.inf
    best_epoch = 0
    with ((output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as history_file,
          (output_dir / "gradient_batches.csv").open("w", newline="", encoding="utf-8") as gradient_file):
        history = csv.writer(history_file)
        gradients = csv.writer(gradient_file)
        history.writerow(("epoch", "train_mse", "train_mse_mm2", "train_monitor_mse",
                          "train_monitor_mse_mm2", "val_mse", "val_mse_mm2",
                          "gradient_min", "gradient_median", "gradient_p95", "gradient_max"))
        gradients.writerow(("epoch", "batch", "gradient_norm"))
        for epoch in range(1, epochs + 1):
            train_mse, norms = _train_epoch(model, train_loader, optimizer,
                                            cfg.TEACHER_FORCING_RATIO, device, gradients, epoch)
            monitor_mse = _validation_mse(model, monitor_loader, device)
            val_mse = _validation_mse(model, validation_loader, device)
            history.writerow((epoch, train_mse, train_mse * scale_squared, monitor_mse,
                              monitor_mse * scale_squared, val_mse, val_mse * scale_squared,
                              norms.min(), np.median(norms), np.quantile(norms, 0.95), norms.max()))
            history_file.flush()
            gradient_file.flush()
            checkpoint.update(epoch=epoch, training_mse=train_mse, validation_mse=val_mse)
            if val_mse < best_mse or epoch == epochs:
                checkpoint.update(model_state_dict=model.state_dict(),
                                  optimizer_state_dict=optimizer.state_dict())
            if val_mse < best_mse:
                best_mse, best_epoch = val_mse, epoch
                _save_checkpoint(output_dir / cfg.BEST_MODEL_FILENAME, checkpoint)
            if epoch == epochs:
                _save_checkpoint(output_dir / cfg.FINAL_MODEL_FILENAME, checkpoint)
            if epoch == 1 or epoch % cfg.LOG_INTERVAL == 0 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs} | train MSE {train_mse:.6g} | "
                      f"C9 MSE {val_mse:.6g} | best epoch {best_epoch}", flush=True)

    write_json(output_dir / "training_summary.json", {
        "epochs": epochs, "best_epoch": best_epoch, "best_validation_mse": best_mse,
        "training_samples": len(training["targets"]), "validation_samples": len(validation["targets"]),
        "train_monitor_samples": len(monitor["targets"]), "device": str(device),
        "torch_version": str(torch.__version__), "split_id": metadata["split_id"],
        "selection_metric": cfg.SELECTION_METRIC,
    })


def test(data_dir, checkpoint, output_dir, device):
    """Evaluate a completed run on each native ten-day window of the held-out C1 point."""
    metadata = _verified_metadata(data_dir, ("test", "norm"))
    scales = load_normalization(data_dir)
    arrays = load_partition(data_dir, "test")
    torch.set_num_threads(ReproductionConfig.CPU_THREADS)
    model, state = load_checkpoint(checkpoint, device)
    if (state["metadata"]["split_id"] != metadata["split_id"]
            or any(state["file_hashes"][name] != metadata["file_hashes"][name]
                   for name in ("test", "norm"))):
        raise ValueError("Checkpoint and test data refer to different splits or normalization parameters.")

    predicted_normalized = autoregressive_predictions(
        model, arrays, device, batch_size=state["config"]["BATCH_SIZE"],
    ).astype(np.float64)
    actual_normalized = arrays["targets"].astype(np.float64)
    target_min, target_scale = float(scales["target_min"]), float(scales["target_scale"])
    predicted = predicted_normalized * target_scale + target_min
    actual = actual_normalized * target_scale + target_min
    errors = predicted - actual
    overall = {**regression_metrics(actual, predicted), "Bias": float(errors.mean())}
    horizon = actual.shape[1]
    by_lead = [
        {"step": step + 1, **regression_metrics(actual[:, step], predicted[:, step]),
         "Bias": float(errors[:, step].mean())}
        for step in range(horizon)
    ]

    _prepare_output(output_dir)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "point", "start", "step", "time_index", "date", "actual", "predicted", "error"))
        for sample, start in enumerate(arrays["window_starts"].tolist()):
            for step in range(horizon):
                index = start + arrays["encoder_inputs"].shape[1] + step
                writer.writerow((sample, 0, start, step + 1, index, metadata["dates"][index],
                                 actual[sample, step, 0], predicted[sample, step, 0], errors[sample, step, 0]))
    write_json(output_dir / "metrics.json", {
        "test_point": "C1", "windows": len(actual), "horizon": horizon,
        "checkpoint_epoch": state["epoch"], "checkpoint_sha256": sha256(checkpoint),
        "split_id": metadata["split_id"], "units": "mm",
        "observations": "Inverse-transformed normalized window targets",
        "future_input": "Recorded vacuum pressure supplied for each target day",
        "normalized_mse": float(np.square(predicted_normalized - actual_normalized).mean()),
        "overall": overall, "by_lead": by_lead,
    })
    print(f"C1, epoch {state['epoch']} | MAE {overall['MAE']:.3f} mm | RMSE {overall['RMSE']:.3f} mm", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True, dest="command")
    for command in ("train", "test"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--data-dir", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    commands.choices["train"].add_argument("--epochs", type=int, default=ReproductionConfig.EPOCHS)
    commands.choices["test"].add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    device = select_device(args.device)
    if args.command == "train":
        if args.epochs < 1:
            parser.error("--epochs must be at least 1")
        train(args.data_dir, args.output_dir, args.epochs, device)
    else:
        test(args.data_dir, args.checkpoint, args.output_dir, device)


if __name__ == "__main__":
    main()
