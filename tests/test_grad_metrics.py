"""梯度清晰度指标单元测试。"""
from __future__ import annotations

import numpy as np
import torch

from src.metrics.grad_metrics import (
    gradient_mae_b13_kelvin,
    gradient_mae_b13_kelvin_sum,
    gradient_mae_b13_norm,
)


def test_gradient_mae_norm_zero_on_identical():
    x = torch.randn(2, 4, 3, 8, 8)
    g = gradient_mae_b13_norm(x, x, channel=3)
    assert g.ndim == 0
    assert float(g) == 0.0


def test_gradient_mae_kelvin_matches_sum():
    pred = np.random.randn(2, 4, 8).astype(np.float32) * 10 + 250
    true = pred + np.random.randn(2, 4, 8).astype(np.float32)
    s, n = gradient_mae_b13_kelvin_sum(pred, true)
    assert n > 0
    assert abs(gradient_mae_b13_kelvin(pred, true) - s / n) < 1e-6


def test_gradient_mae_kelvin_increases_with_blur():
    """简单平滑会使梯度 MAE 变大（相对锐利真值）。"""
    rng = np.random.default_rng(0)
    true = rng.normal(250, 15, size=(1, 16, 16)).astype(np.float32)
    pred_sharp = true + rng.normal(0, 1, size=true.shape).astype(np.float32)
    pred_blur = true.copy()
    pred_blur[:, 1:-1, 1:-1] = (
        true[:, :-2, 1:-1] + true[:, 2:, 1:-1] + true[:, 1:-1, :-2] + true[:, 1:-1, 2:]
    ) / 4.0
    assert gradient_mae_b13_kelvin(pred_blur, true) > gradient_mae_b13_kelvin(pred_sharp, true)
