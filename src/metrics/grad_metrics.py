"""B13 空间梯度/边缘清晰度指标（开尔文域）。

用于评估预报是否「糊」：梯度 MAE 越低，说明云顶边缘与纹理越接近真值。
"""
from __future__ import annotations

import numpy as np
import torch


def gradient_mae_b13_norm(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    channel: int = 3,
) -> torch.Tensor:
    """B13 通道空间梯度 MAE（norm 域，可反传）。

    pred, true: ``[B, C, T, H, W]``。
    """
    p = pred[:, channel]
    t = true[:, channel]
    p_dx = p[..., :, 1:] - p[..., :, :-1]
    t_dx = t[..., :, 1:] - t[..., :, :-1]
    p_dy = p[..., 1:, :] - p[..., :-1, :]
    t_dy = t[..., 1:, :] - t[..., :-1, :]
    return 0.5 * ((p_dx - t_dx).abs().mean() + (p_dy - t_dy).abs().mean())


def gradient_mae_b13_kelvin(pred_k: np.ndarray, true_k: np.ndarray) -> float:
    """B13 空间梯度 MAE（开尔文域，评估用）。

    pred_k, true_k: 任意前缀 + ``[..., H, W]``，例如 ``[B,T,H,W]``。
    返回单位约为 K/像素（相邻像素亮温差之差的平均绝对值）。
    """
    s, n = gradient_mae_b13_kelvin_sum(pred_k, true_k)
    return s / max(n, 1)


def gradient_mae_b13_kelvin_sum(pred_k: np.ndarray, true_k: np.ndarray) -> tuple[float, int]:
    """返回 (梯度绝对误差之和, 梯度元素个数)，便于跨 batch micro 聚合。"""
    p_dx = pred_k[..., :, 1:] - pred_k[..., :, :-1]
    t_dx = true_k[..., :, 1:] - true_k[..., :, :-1]
    p_dy = pred_k[..., 1:, :] - pred_k[..., :-1, :]
    t_dy = true_k[..., 1:, :] - true_k[..., :-1, :]
    s = float(np.abs(p_dx - t_dx).sum() + np.abs(p_dy - t_dy).sum())
    n = int(p_dx.size + p_dy.size)
    return s, n
