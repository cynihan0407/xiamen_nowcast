"""临近预报评估指标聚合（B13 开尔文域）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.normalizers import norm_to_kelvin_np
from src.metrics.csi import binary_csi, csi_at_threshold_k


@dataclass
class CSIAccumulator:
    """全局（micro）二分类计数，用于跨 batch 汇总 CSI。"""

    hits: int = 0
    misses: int = 0
    false_alarms: int = 0

    def update_masks(self, pred_mask: np.ndarray, true_mask: np.ndarray) -> None:
        pred = pred_mask.astype(bool).reshape(-1)
        true = true_mask.astype(bool).reshape(-1)
        self.hits += int(np.logical_and(pred, true).sum())
        self.misses += int(np.logical_and(~pred, true).sum())
        self.false_alarms += int(np.logical_and(pred, ~true).sum())

    def update_kelvin(self, pred_k: np.ndarray, true_k: np.ndarray, threshold_k: float) -> None:
        self.update_masks(pred_k <= threshold_k, true_k <= threshold_k)

    def to_dict(self) -> dict[str, float]:
        denom = self.hits + self.misses + self.false_alarms
        if denom == 0:
            return {"CSI": 0.0, "POD": 0.0, "FAR": 0.0, "BIAS": 0.0}
        csi = self.hits / denom
        pod = self.hits / max(self.hits + self.misses, 1)
        far = self.false_alarms / max(self.hits + self.false_alarms, 1)
        bias = (self.hits + self.false_alarms) / max(self.hits + self.misses, 1)
        return {"CSI": float(csi), "POD": float(pod), "FAR": float(far), "BIAS": float(bias)}


@dataclass
class NowcastMetricState:
    """跨 batch 累积状态。"""

    n_samples: int = 0
    sum_abs_err_k: float = 0.0
    sum_sq_err_k: float = 0.0
    n_pixels: int = 0
    csi_global: dict[float, CSIAccumulator] = field(default_factory=dict)
    csi_per_sample: dict[float, list[float]] = field(default_factory=dict)

    def ensure_threshold(self, thr: float) -> None:
        if thr not in self.csi_global:
            self.csi_global[thr] = CSIAccumulator()
        if thr not in self.csi_per_sample:
            self.csi_per_sample[thr] = []


def tensor_future_b13_kelvin(x_bcthw: torch.Tensor) -> np.ndarray:
    """``[B, C, T, H, W]`` → ``[B, T, H, W]`` 开尔文。"""
    return norm_to_kelvin_np(x_bcthw[:, 3], "B13")


def update_nowcast_metrics(
    state: NowcastMetricState,
    pred_future: torch.Tensor,
    true_future: torch.Tensor,
    thresholds_k: list[float],
) -> None:
    """用一批 future 预报更新指标状态。

    pred_future / true_future: ``[B, C, T, H, W]``，norm 域。
    """
    pred_k = tensor_future_b13_kelvin(pred_future)
    true_k = tensor_future_b13_kelvin(true_future)
    B, T, H, W = pred_k.shape
    state.n_samples += B
    diff = pred_k - true_k
    state.sum_abs_err_k += float(np.abs(diff).sum())
    state.sum_sq_err_k += float((diff**2).sum())
    state.n_pixels += int(B * T * H * W)

    for thr in thresholds_k:
        state.ensure_threshold(thr)
        # 全局 micro CSI
        state.csi_global[thr].update_kelvin(pred_k.reshape(-1), true_k.reshape(-1), thr)
        # 逐样本：先对 T 帧平均 CSI，再记入列表（与 02_baselines 一致）
        for b in range(B):
            frame_csis = [
                csi_at_threshold_k(pred_k[b, t], true_k[b, t], thr)["CSI"] for t in range(T)
            ]
            state.csi_per_sample[thr].append(float(np.mean(frame_csis)))


def finalize_nowcast_metrics(state: NowcastMetricState) -> dict[str, float]:
    """输出扁平 dict，便于写 JSON / 打印。"""
    out: dict[str, float] = {}
    if state.n_pixels > 0:
        out["MAE_B13_K"] = state.sum_abs_err_k / state.n_pixels
        out["RMSE_B13_K"] = (state.sum_sq_err_k / state.n_pixels) ** 0.5
    for thr, acc in state.csi_global.items():
        m = acc.to_dict()
        tag = int(thr) if thr == int(thr) else thr
        out[f"CSI_B13_{tag}K_global"] = m["CSI"]
        out[f"POD_B13_{tag}K_global"] = m["POD"]
        out[f"FAR_B13_{tag}K_global"] = m["FAR"]
        per = state.csi_per_sample.get(thr, [])
        if per:
            out[f"CSI_B13_{tag}K_mean_per_sample"] = float(np.mean(per))
    out["n_samples"] = float(state.n_samples)
    out["n_pixels"] = float(state.n_pixels)
    return out


def persistence_forecast(past: torch.Tensor, n_future: int) -> torch.Tensor:
    """末帧复制：``past [B,C,T_p,H,W]`` → ``[B,C,n_future,H,W]``。"""
    last = past[:, :, -1:, :, :]
    return last.expand(-1, -1, n_future, -1, -1).contiguous()
