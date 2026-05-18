"""nowcast 评估指标单元测试。"""
from __future__ import annotations

import torch

from src.metrics.nowcast import (
    NowcastMetricState,
    finalize_nowcast_metrics,
    persistence_forecast,
    update_nowcast_metrics,
)


def test_persistence_forecast_shape():
    past = torch.randn(2, 4, 6, 8, 8)
    out = persistence_forecast(past, 12)
    assert out.shape == (2, 4, 12, 8, 8)
    assert torch.equal(out[:, :, 0], past[:, :, -1])
    assert torch.equal(out[:, :, -1], past[:, :, -1])


def test_finalize_mae_and_csi():
    state = NowcastMetricState()
    pred = torch.full((1, 4, 2, 4, 4), -0.5)
    true = torch.full((1, 4, 2, 4, 4), -0.4)
    update_nowcast_metrics(state, pred, true, [240.0])
    m = finalize_nowcast_metrics(state)
    assert "MAE_B13_K" in m
    assert m["MAE_B13_K"] >= 0
    assert "CSI_B13_240K_global" in m
    assert "CSI_B13_240K_mean_per_sample" in m
