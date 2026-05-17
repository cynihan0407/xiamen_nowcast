"""基于 B13 亮温（开尔文）阈值的二分类 CSI 等。"""
from __future__ import annotations

import numpy as np


def binary_csi(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-8) -> dict[str, float]:
    """逐像素二分类的 CSI / POD / FAR / BIAS。

    pred_mask, true_mask: 同形状 bool 数组。
    """
    pred = pred_mask.astype(bool).reshape(-1)
    true = true_mask.astype(bool).reshape(-1)
    hits = int(np.logical_and(pred, true).sum())
    misses = int(np.logical_and(~pred, true).sum())
    false_alarms = int(np.logical_and(pred, ~true).sum())
    csi = hits / max(hits + misses + false_alarms, eps)
    pod = hits / max(hits + misses, eps)
    far = false_alarms / max(hits + false_alarms, eps)
    bias = (hits + false_alarms) / max(hits + misses, eps)
    return {"CSI": float(csi), "POD": float(pod), "FAR": float(far), "BIAS": float(bias)}


def csi_at_threshold_k(
    pred_b13_K: np.ndarray,
    true_b13_K: np.ndarray,
    threshold_K: float,
    *,
    event_is_cold: bool = True,
) -> dict[str, float]:
    """B13 开尔文场：默认「事件 = 云顶足够冷（≤ 阈值）」。"""
    if event_is_cold:
        pred_m = pred_b13_K <= threshold_K
        true_m = true_b13_K <= threshold_K
    else:
        pred_m = pred_b13_K >= threshold_K
        true_m = true_b13_K >= threshold_K
    return binary_csi(pred_m, true_m)
