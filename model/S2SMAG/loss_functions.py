"""S2SMAG 的实际训练目标：严格同形的多步沉降 MSE。

该函数由本目录 train.py 注入共享运行时，不是未调用的示意损失文件。
严格检查形状，避免 PyTorch 广播把一个时间步错误地当成整个预测序列。
"""

import torch
import torch.nn.functional as F


def supervised_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """输入均为 [batch, horizon, 1]，返回参与反向传播的标量 MSE。"""
    if prediction.ndim != 3 or target.ndim != 3 or prediction.shape != target.shape:
        raise ValueError(f"预测和标签必须为相同的三维张量，当前为 {prediction.shape} 和 {target.shape}。")
    if prediction.shape[-1] != 1 or prediction.numel() == 0:
        raise ValueError("沉降回归的输出特征维度必须为1，且张量不能为空。")
    return F.mse_loss(prediction, target)
