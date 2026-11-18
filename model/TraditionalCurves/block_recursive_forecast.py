"""完整十步分块递推，区别于旧实验每次仅取第一步的递推协议。

未来接口只接受真空路径，没有未来沉降或速率参数。每块十个预测都被消费，
相邻预测沉降差重新生成速率，连同对应真空形成下一块的十行历史。
末块不足十天时仅截断输出；补齐的真空和未使用预测不会成为评分观测。
"""

import numpy as np
import torch


def block_recursive_predict(model, observed_history, future_vacuum, norm, *, horizon=10, device="cpu"):
    """返回 prediction、history_trace、blocks、final_history_raw、status、reason。

    observed_history 严格为 [10,3] 原始量纲，列顺序为沉降、逐日沉降增量、
    真空幅值。C1数据为等间隔逐日观测，因此逐日增量数值对应 mm/day。
    """
    observed = np.asarray(observed_history, dtype=np.float64).copy()
    vacuum = np.asarray(future_vacuum, dtype=np.float64).reshape(-1)
    if horizon != 10 or observed.shape != (10, 3):
        raise ValueError("本实验固定10历史、10输出；observed_history应为[10,3]。")
    if model.training:
        raise ValueError("分块递推必须先调用 model.eval()。")
    if not len(vacuum) or not np.isfinite(observed).all() or not np.isfinite(vacuum).all():
        raise ValueError("初始历史与给定未来真空必须为有限非空数据。")
    minimum = np.asarray(norm["feature_min"], dtype=np.float64)
    scale = np.asarray(norm["feature_scale"], dtype=np.float64)
    target_min, target_scale = float(norm["target_min"]), float(norm["target_scale"])
    if minimum.shape != (3,) or scale.shape != (3,) or np.any(scale <= 0) or target_scale <= 0:
        raise ValueError("训练归一化参数形状或尺度不正确。")

    history = observed.copy()
    predictions = np.full(len(vacuum), np.nan, dtype=np.float64)
    traces, blocks = [], []
    status, reason = "ok", ""
    with torch.inference_mode():
        for offset in range(0, len(vacuum), horizon):
            used = min(horizon, len(vacuum) - offset)
            given = vacuum[offset:offset + used]
            padded = np.pad(given, (0, horizon - used), mode="edge")
            history_tensor = torch.as_tensor((history - minimum) / scale, dtype=torch.float32,
                                              device=device).unsqueeze(0)
            future_tensor = torch.as_tensor((padded - minimum[2]) / scale[2], dtype=torch.float32,
                                             device=device).reshape(1, horizon, 1)
            # 每次都调用相同10→10模型；下一次重新编码整块预测历史，不跨块复用隐状态。
            normalized = model(history_tensor, future_tensor)
            if tuple(normalized.shape) != (1, horizon, 1):
                raise ValueError(f"模型输出应为[1,10,1]，实际为{tuple(normalized.shape)}。")
            complete_output = normalized[0, :, 0].cpu().numpy().astype(np.float64) * target_scale + target_min
            block_output = complete_output[:used]
            previous = np.concatenate(([history[-1, 0]], block_output[:-1]))
            increments = block_output - previous
            next_rows = np.column_stack((block_output, increments, given))
            block_number = len(blocks) + 1
            for local in range(used):
                traces.append({"step": offset + local + 1, "block_index": block_number,
                               "block_start_offset": offset, "block_step": local + 1,
                               "previous_settlement": float(previous[local]),
                               "predicted_settlement": float(block_output[local]),
                               "predicted_rate": float(increments[local]), "future_vacuum": float(given[local]),
                               "history_source": "observed_initial_10" if offset == 0 else "previous_full_predicted_block"})
            block = {"block_index": block_number, "block_start_offset": offset, "used_length": used,
                     "history_input_raw": history.tolist(), "future_vacuum_full": padded.tolist(),
                     "future_padding_count": horizon - used, "complete_output_raw": complete_output.tolist(),
                     "used_output_raw": block_output.tolist()}
            predictions[offset:offset + used] = block_output
            if not np.isfinite(next_rows).all():
                block["next_history_raw"] = None
                blocks.append(block)
                status, reason = "failed", f"第{block_number}块出现非有限预测，剩余预测保留NaN，未裁剪或用真值填充。"
                break
            # 完整块替代全部十行；最后不足十行时只追加实际使用行，且不再调用模型。
            history = np.concatenate((history, next_rows), axis=0)[-10:]
            block["next_history_raw"] = history.tolist()
            blocks.append(block)
    return {"prediction": predictions, "history_trace": traces, "blocks": blocks,
            "final_history_raw": history, "status": status, "reason": reason}
