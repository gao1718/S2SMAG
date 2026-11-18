"""四模型共同使用的原量纲指标，定义在比较前保持一致。"""
import numpy as np


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("真值和预测必须等长且非空")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("指标输入含NaN/Inf")
    error = predicted - actual
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    nonzero = np.abs(actual) > 1e-12
    return {"MAE": float(np.mean(np.abs(error))), "MSE": mse, "RMSE": mse ** 0.5,
            "R2": float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else None,
            "MAPE_percent": float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
            "MAPE_valid_count": int(nonzero.sum())}
