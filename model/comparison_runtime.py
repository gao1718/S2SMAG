"""由模型本地Config驱动的统一训练与测试流程。

参考gnn_simulation_code/model/comparison_runtime.py的依赖注入方式：
每个模型显式传入自己的cfg、build_model和损失函数，本模块不通过全局
注册表猜测架构。训练只加载训练文件，测试只加载测试文件；模型目录
拥有自己的实验配置、save_model、results和plotting_scripts。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import re
import sys
import time

from project_paths import configure_environment

configure_environment()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.evaluation import regression_metrics
from model.optimizers import Lookahead


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_run(cfg, argv, *, training):
    """命令行仅覆盖本次配置，不更改模型目录中的model_Config.py。"""
    parser = argparse.ArgumentParser(description=f"{cfg.MODEL_NAME} {'训练' if training else '独立测试'}")
    parser.add_argument("--experiment", help="实验名；默认使用本模型Config")
    parser.add_argument("--fold", default=str(cfg.FOLD), help="留出点号0..8，或all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="仅验证流程：1轮、2批次、小隐藏层，与正式实验隔离")
    if training:
        parser.add_argument("--epochs", type=int)
        parser.add_argument("--max-batches", type=int, help="调试用批次数上限；正式实验省略")
    else:
        parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    args = parser.parse_args(argv)
    result = cfg.clone()
    if args.device:
        result.DEVICE = args.device
    if args.smoke:
        result.EXPERIMENT_NAME = "smoke_check"
    if result.SMOKE or args.smoke:
        result.SMOKE = True
        result.HIDDEN_DIM = 16
        result.DECODER_HIDDEN_DIM = 16
        result.EPOCHS = 1
        result.BATCH_SIZE = 32
    if args.experiment:
        result.EXPERIMENT_NAME = args.experiment
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", result.EXPERIMENT_NAME):
        raise ValueError("实验名称须以字母/数字开头，仅包含字母、数字、点、下划线、连字符")
    folds = list(range(9)) if args.fold == "all" else [int(args.fold)]
    if any(not 0 <= fold <= 8 for fold in folds):
        raise ValueError("fold必须为0..8或all，对应C1..C9")
    if training:
        if args.epochs is not None and not result.SMOKE:
            result.EPOCHS = args.epochs
        result.MAX_BATCHES = 2 if result.SMOKE else args.max_batches
        if result.EPOCHS < 1 or result.BATCH_SIZE < 1 or (result.MAX_BATCHES is not None and result.MAX_BATCHES < 1):
            raise ValueError("轮数、批量大小和批次数上限须为正整数")
    return result, folds, args


def select_device(name):
    if str(name) == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前Python没有可用CUDA，请用--device cpu")
    return device


def load_data(cfg, partition):
    """校验来源后只打开所需分区；归一化参数从独立文件只读加载。"""
    paths = {"train": cfg.DATA_PATH, "test": cfg.TEST_DATA_PATH, "norm": cfg.NORM_DATA_PATH}
    if not all(Path(path).exists() for path in [*paths.values(), cfg.METADATA_PATH]):
        raise FileNotFoundError("缺少数据文件，请先运行code/original_data/prepare_data.py")
    metadata = read_json(cfg.METADATA_PATH)
    # 保存三份文件的指纹，以防训练后换表或替换划分；不把测试标签加载进训练。
    fingerprints = {key: sha256(path) for key, path in paths.items()}
    if fingerprints != metadata["file_hashes"]:
        raise ValueError("数据文件指纹与metadata不一致，请重新生成划分")
    if metadata["test_point"] != cfg.FOLD or cfg.FOLD in metadata["train_points"]:
        raise ValueError("元信息中的留出点号错误或训练/测试点重叠")
    with np.load(paths[partition], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    with np.load(cfg.NORM_DATA_PATH, allow_pickle=False) as archive:
        norm = {key: archive[key] for key in archive.files}
    encoder, future, target = (arrays[key] for key in ("encoder_inputs", "future_inputs", "targets"))
    expected_shapes = [(cfg.ENCODER_LENGTH, cfg.INPUT_SIZE), (cfg.HORIZON, cfg.FUTURE_INPUT_SIZE),
                       (cfg.HORIZON, cfg.OUTPUT_SIZE)]
    for array, shape in zip((encoder, future, target), expected_shapes):
        if array.ndim != 3 or array.shape[1:] != shape or len(array) != len(encoder) or not len(array):
            raise ValueError(f"数据形状{array.shape}与配置要求[样本,{shape}]不一致")
        if not np.isfinite(array).all():
            raise ValueError("输入或标签含NaN/Inf")
    for key in ("point_indices", "window_starts"):
        if arrays[key].shape != (len(encoder),) or not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError(f"{key}必须为每个窗口对应的整数索引")
    points = set(arrays["point_indices"].tolist())
    expected_points = set(metadata["train_points"]) if partition == "train" else {cfg.FOLD}
    if points != expected_points:
        raise ValueError("实际窗口点位与独立训练/测试划分不一致")
    if np.any(arrays["window_starts"] < 0):
        raise ValueError("窗口起点不能为负数")
    for key in ("feature_min", "feature_max", "feature_scale", "target_min", "target_max", "target_scale"):
        if key not in norm or not np.isfinite(norm[key]).all():
            raise ValueError("归一化参数缺失或包含非有限值")
    if np.any(norm["feature_scale"] <= 0) or np.any(norm["target_scale"] <= 0):
        raise ValueError("归一化尺度必须为正，常数列使用1")
    return arrays, norm, metadata, fingerprints


def make_loader(arrays, cfg, *, training):
    dataset = TensorDataset(*(torch.from_numpy(np.asarray(arrays[key], dtype=np.float32))
                              for key in ("encoder_inputs", "future_inputs", "targets")))
    generator = torch.Generator().manual_seed(cfg.RANDOM_SEED)
    return DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                      shuffle=training and cfg.SHUFFLE_TRAINING, generator=generator, num_workers=0)


def train_one_fold(cfg, build_model, loss_function):
    arrays, norm, metadata, fingerprints = load_data(cfg, "train")
    device = select_device(cfg.DEVICE)
    if cfg.RUN_DIR.exists() and any(cfg.RUN_DIR.iterdir()):
        raise FileExistsError(f"实验目录已有内容，拒绝覆盖，请更改--experiment：{cfg.RUN_DIR}")
    cfg.SAVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(cfg.RUN_DIR / "config.json", cfg.as_dict())
    write_json(cfg.RUN_DIR / "data_metadata.json", metadata)
    torch.set_num_threads(cfg.CPU_THREADS)
    random.seed(cfg.RANDOM_SEED)
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = build_model(cfg, arrays).to(device)
    loader = make_loader(arrays, cfg, training=True)
    adamw = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, betas=tuple(cfg.BETAS),
                             eps=cfg.EPS, weight_decay=cfg.WEIGHT_DECAY)
    optimizer = Lookahead(adamw, cfg.LOOKAHEAD_K, cfg.LOOKAHEAD_ALPHA)
    scheduler = None
    if getattr(cfg, "SCHEDULER_ENABLED", True):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            adamw, T_0=cfg.SCHEDULER_T_0, T_mult=cfg.SCHEDULER_T_MULT, eta_min=cfg.MIN_LEARNING_RATE)
    history, best_loss, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    summary = {"model": cfg.MODEL_NAME, "experiment": cfg.EXPERIMENT_NAME, "fold": cfg.FOLD,
               "smoke": cfg.SMOKE, "selection_metric": "training_mse", "validation_set": None,
               "split_id": metadata["split_id"], "file_hashes": fingerprints,
               "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
               "started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "status": "running"}
    summary.update(python_executable=sys.executable, planned_epochs=cfg.EPOCHS,
                   completed_epochs=0, train_samples=len(arrays["targets"]))
    if device.type == "cuda":
        summary.update(gpu_name=torch.cuda.get_device_name(device),
                       gpu_capability=list(torch.cuda.get_device_capability(device)),
                       torch_cuda_version=torch.version.cuda)
    summary_path = cfg.RUN_DIR / "training_summary.json"
    write_json(summary_path, summary)
    print(f"{cfg.MODEL_NAME} | fold {cfg.FOLD:02d} | {len(arrays['targets'])}训练窗口 | {device}")
    print(f"配置：{cfg.RUN_DIR / 'config.json'}")
    if cfg.SMOKE:
        print("流程验证模式：不作为正式训练或论文性能结果。")
    history_path = cfg.RESULT_DIR / "training_history.csv"
    history_columns = ["epoch", "train_mse", "learning_rate", "elapsed_seconds", "train_samples"]
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=history_columns).writeheader()
    try:
        for epoch in range(1, cfg.EPOCHS + 1):
            model.train()
            squared_loss, elements, samples = 0.0, 0, 0
            learning_rate = adamw.param_groups[0]["lr"]
            for batch_index, (encoder, future, target) in enumerate(loader, start=1):
                encoder, future, target = (value.to(device) for value in (encoder, future, target))
                optimizer.zero_grad()
                predicted = model(encoder, future, target, teacher_forcing_ratio=cfg.TEACHER_FORCING_RATIO)
                if predicted.shape != target.shape:
                    raise ValueError("模型预测与标签形状不同，禁止广播计算损失")
                loss = loss_function(predicted, target)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("损失必须为有限标量")
                loss.backward()
                if cfg.GRADIENT_CLIP_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM, error_if_nonfinite=True)
                optimizer.step()
                squared_loss += float(loss.detach()) * target.numel()
                elements += target.numel()
                samples += len(target)
                if cfg.MAX_BATCHES is not None and batch_index >= cfg.MAX_BATCHES:
                    break
            if scheduler is not None:
                scheduler.step(epoch)
            mean_loss = squared_loss / elements
            history.append({"epoch": epoch, "train_mse": mean_loss, "learning_rate": learning_rate,
                            "elapsed_seconds": time.perf_counter() - started, "train_samples": samples})
            # CSV每轮追加，供实时曲线只读读取；Excel仍按周期保存完整逐轮记录。
            with history_path.open("a", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=history_columns).writerow(history[-1])
            state = {"format_version": 2, "model_name": cfg.MODEL_NAME, "config": cfg.as_dict(),
                     "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                     "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},
                     "epoch": epoch, "training_mse": mean_loss,
                     "metadata": metadata, "file_hashes": fingerprints}
            if mean_loss < best_loss:
                best_loss, best_epoch = mean_loss, epoch
                torch.save(state, cfg.MODEL_SAVE_PATH)
            if epoch == 1 or epoch == cfg.EPOCHS or epoch % cfg.CHECKPOINT_INTERVAL == 0:
                torch.save(state, cfg.FINAL_MODEL_PATH)
                pd.DataFrame(history).to_excel(cfg.RESULT_DIR / "training_losses.xlsx", index=False)
                summary.update(completed_epochs=epoch, best_epoch=best_epoch,
                               best_training_mse=best_loss, latest_training_mse=mean_loss,
                               elapsed_seconds=time.perf_counter() - started)
                write_json(summary_path, summary)
            if epoch == 1 or epoch == cfg.EPOCHS or epoch % cfg.LOG_INTERVAL == 0:
                print(f"epoch={epoch}/{cfg.EPOCHS} | training MSE={mean_loss:.8f}")
    except BaseException as error:
        if history:
            pd.DataFrame(history).to_excel(cfg.RESULT_DIR / "training_losses.xlsx", index=False)
        summary.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                       completed_epochs=len(history), error=str(error))
        write_json(summary_path, summary)
        raise
    summary.update(status="complete", completed_epochs=len(history), best_epoch=best_epoch,
                   best_training_mse=best_loss, elapsed_seconds=time.perf_counter() - started)
    write_json(summary_path, summary)
    print(f"最佳权重：{cfg.MODEL_SAVE_PATH}\n逐轮损失：{cfg.RESULT_DIR / 'training_losses.xlsx'}")
    return summary


def train_model(cfg, build_model, loss_function=None, argv=None):
    selected, folds, _ = configure_run(cfg, argv, training=True)
    if loss_function is None:
        raise ValueError("请显式传入本模型loss_functions.py中的损失函数")
    summaries = []
    for fold in folds:
        current = selected.clone()
        current.FOLD = fold
        summaries.append(train_one_fold(current, build_model, loss_function))
    return summaries[0] if len(summaries) == 1 else summaries


def load_checkpoint(cfg, build_model, checkpoint="best"):
    path = cfg.MODEL_SAVE_PATH if checkpoint == "best" else cfg.FINAL_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"尚未训练该配置的模型：{path}")
    device = select_device(cfg.DEVICE)
    state = torch.load(path, map_location=device, weights_only=True)
    if state["format_version"] != 2 or state["model_name"] != cfg.MODEL_NAME:
        raise ValueError("权重版本或模型名称与当前模型不匹配")
    stored = type(cfg).from_dict(state["config"])
    for key in ("FOLD", "EXPERIMENT_NAME", "SMOKE"):
        if getattr(stored, key) != getattr(cfg, key):
            raise ValueError(f"权重中的{key}与当前实验不一致")
    # 架构按权重内的快照构建，之后修改model_Config的隐藏层不会破坏旧实验加载。
    model = build_model(stored).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model, stored, state, device


def test_one_fold(cfg, build_model, checkpoint):
    model, stored, state, device = load_checkpoint(cfg, build_model, checkpoint)
    arrays, norm, metadata, fingerprints = load_data(stored, "test")
    if fingerprints != state["file_hashes"] or metadata != state["metadata"]:
        raise ValueError("当前数据/缩放参数与训练时不同，请重新训练，不能覆盖旧实验测试结果")
    torch.set_num_threads(stored.CPU_THREADS)
    loader = make_loader(arrays, stored, training=False)
    predictions = []
    with torch.inference_mode():
        for encoder, future, _ in loader:
            # 不向模型传入测试target，保证自回归解码不会读取未来真实沉降。
            predictions.append(model(encoder.to(device), future.to(device)).cpu().numpy())
    prediction = np.concatenate(predictions).astype(np.float64)
    actual = arrays["targets"].astype(np.float64)
    if prediction.shape != actual.shape or not np.isfinite(prediction).all():
        raise ValueError("预测形状错误或出现非有限值")
    scale, minimum = float(norm["target_scale"]), float(norm["target_min"])
    actual, prediction = actual * scale + minimum, prediction * scale + minimum
    overall = regression_metrics(actual, prediction)
    horizon = stored.HORIZON
    by_horizon = [{"step": step + 1, **regression_metrics(actual[:, step], prediction[:, step])}
                  for step in range(horizon)]
    count = len(actual)
    steps = np.tile(np.arange(1, horizon + 1), count)
    starts = np.repeat(arrays["window_starts"], horizon)
    time_indices = starts + stored.ENCODER_LENGTH + steps - 1
    frame = pd.DataFrame({"sample": np.repeat(np.arange(count), horizon),
                          "point": np.repeat(arrays["point_indices"], horizon), "start": starts,
                          "step": steps, "time_index": time_indices, "actual": actual.reshape(-1),
                          "predicted": prediction.reshape(-1), "error": (prediction - actual).reshape(-1)})
    if "dates" in metadata:
        frame["date"] = np.asarray(metadata["dates"])[time_indices]
    # final权重的评估单独保存，避免覆盖best结果后留下仍对应best的旧图片。
    output_directory = cfg.RESULT_DIR if checkpoint == "best" else cfg.RESULT_DIR / "final_checkpoint"
    output_directory.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_directory / "predictions.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="predictions", index=False)
    with pd.ExcelWriter(output_directory / "metrics.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(list(overall.items()), columns=["metric", "value"]).to_excel(writer, sheet_name="overall", index=False)
        pd.DataFrame(by_horizon).to_excel(writer, sheet_name="by_horizon", index=False)
    evaluation = {"model": cfg.MODEL_NAME, "experiment": cfg.EXPERIMENT_NAME, "fold": cfg.FOLD,
                  "smoke": stored.SMOKE, "split_id": metadata["split_id"], "source_hashes": metadata["source_hashes"],
                  "file_hashes": fingerprints, "checkpoint": checkpoint, "checkpoint_epoch": state["epoch"],
                  "checkpoint_sha256": sha256(cfg.MODEL_SAVE_PATH if checkpoint == "best" else cfg.FINAL_MODEL_PATH),
                  "test_samples": count, "horizon": horizon, "overall": overall,
                  "units": "same_as_source_settlement_table", "test_labels_used_as_model_input": False,
                  "overlapping_windows": True}
    write_json(output_directory / "evaluation_metadata.json", evaluation)
    print(f"{cfg.MODEL_NAME} | fold={cfg.FOLD:02d} | MAE={overall['MAE']:.6g} | RMSE={overall['RMSE']:.6g}")
    print(f"测试结果：{output_directory}")
    return evaluation


def test_model(cfg, build_model, argv=None):
    selected, folds, args = configure_run(cfg, argv, training=False)
    summaries = []
    for fold in folds:
        current = selected.clone()
        current.FOLD = fold
        summaries.append(test_one_fold(current, build_model, args.checkpoint))
    return summaries[0] if len(summaries) == 1 else summaries
