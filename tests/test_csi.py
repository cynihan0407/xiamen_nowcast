"""CSI 指标单元测试。"""
from __future__ import annotations

import numpy as np

from src.metrics.csi import binary_csi, csi_at_threshold_k


def test_binary_csi_perfect():
    x = np.ones((10, 10), dtype=bool)
    m = binary_csi(x, x)
    assert abs(m["CSI"] - 1.0) < 1e-6


def test_csi_at_threshold():
    pred = np.full((5, 5), 230.0)
    true = np.full((5, 5), 235.0)
    m = csi_at_threshold_k(pred, true, 240.0)
    assert "CSI" in m
