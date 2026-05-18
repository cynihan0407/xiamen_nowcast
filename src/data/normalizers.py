"""亮温归一化与反归一化工具。

与 ``prod_v7_ultimate.py`` 中的物理极值线性映射完全对齐：

    norm = clip((K - mi) / (ma - mi), 0, 1) * 2 - 1

约定的通道顺序固定为 ``B08 -> B09 -> B10 -> B13`` ，与 ``.npz`` 中
``data['x']`` 的 channel 维一致。任何下游模块都应使用本模块提供的常量与函数，
避免将物理极值散落在多处。

本模块同时提供：

* ``kelvin_to_norm`` / ``norm_to_kelvin``  ：通道感知的标量/张量转换；
* ``BrightnessTemperatureNormalizer``      ：批量张量级的封装，配合 PyTorch 使用；
* ``b13_norm_threshold_for_kelvin``        ：把 B13 的开尔文阈值翻译到 norm 域，
  用于评估指标（CSI/POD/...）以及加权采样器。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


BAND_ORDER: tuple[str, ...] = ("B08", "B09", "B10", "B13")
"""频段在通道维上的固定顺序，下游所有模块都依赖此顺序。"""


NORM_LIMITS: dict[str, tuple[float, float]] = {
    "B08": (190.0, 260.0),
    "B09": (190.0, 270.0),
    "B10": (190.0, 280.0),
    "B13": (180.0, 310.0),
}
"""每个频段的物理极值（开尔文），与 prod_v7_ultimate.py 完全一致。"""


B13_INDEX: int = BAND_ORDER.index("B13")
"""B13 通道在 channel 维上的索引（=3），评估指标与采样器频繁用到。"""


# ---------------------------------------------------------------------------
# 标量 / 数组级 API
# ---------------------------------------------------------------------------
def kelvin_to_norm(value: ArrayLike, band: str) -> ArrayLike:
    """开尔文 → ``[-1, 1]`` 归一化值。

    Args:
        value: 任意形状的开尔文亮温（``np.ndarray`` 或 ``torch.Tensor``）。
        band:  ``B08`` / ``B09`` / ``B10`` / ``B13``。

    Returns:
        与输入同类型同形状的归一化结果，已截断到 ``[-1, 1]``。
    """
    mi, ma = NORM_LIMITS[band]
    if isinstance(value, torch.Tensor):
        out = (value - mi) / (ma - mi)
        out = out.clamp_(0.0, 1.0)
        return out * 2.0 - 1.0
    arr = np.asarray(value)
    out = (arr - mi) / (ma - mi)
    np.clip(out, 0.0, 1.0, out=out)
    return out * 2.0 - 1.0


def norm_to_kelvin(value: ArrayLike, band: str) -> ArrayLike:
    """``[-1, 1]`` 归一化值 → 开尔文（不做截断，调用方按需 clip）。"""
    mi, ma = NORM_LIMITS[band]
    half_range = (ma - mi) * 0.5
    mid = (ma + mi) * 0.5
    if isinstance(value, torch.Tensor):
        return value * half_range + mid
    return np.asarray(value) * half_range + mid


def norm_to_kelvin_np(value: torch.Tensor, band: str) -> np.ndarray:
    """Tensor → 开尔文 ``float32`` NumPy 数组（bf16/fp16 混合精度训练安全）。"""
    return norm_to_kelvin(value.detach().float(), band).cpu().numpy()


def b13_norm_threshold_for_kelvin(temperature_K: float) -> float:
    """把 B13 的开尔文阈值翻译到归一化域。

    例如 240 K (深对流上界) -> 约 0.857 的 norm 值（>0 即比中位低）。
    """
    return float(kelvin_to_norm(np.asarray([temperature_K], dtype=np.float64), "B13").item())


# ---------------------------------------------------------------------------
# 张量级封装
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrightnessTemperatureNormalizer:
    """批量张量级的归一化 / 反归一化。

    输入张量必须满足通道维顺序 = :data:`BAND_ORDER`。形状无要求，
    只要存在一个 channel 维（默认是第 0 维之外的某一维），由 ``channel_dim`` 指定。
    典型用法：

    >>> norm = BrightnessTemperatureNormalizer()
    >>> kelvin = torch.full((2, 4, 18, 64, 64), 250.0)   # [B, C, T, H, W]
    >>> normed = norm.encode(kelvin, channel_dim=1)        # -> [-1, 1]
    >>> back = norm.decode(normed, channel_dim=1)          # -> Kelvin
    """

    band_order: Sequence[str] = BAND_ORDER
    limits: Mapping[str, tuple[float, float]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # dataclass(frozen=True) 下不能直接赋值，使用 object.__setattr__ 兜底。
        if self.limits is None:
            object.__setattr__(self, "limits", NORM_LIMITS)

        # 校验
        for band in self.band_order:
            if band not in self.limits:
                raise KeyError(f"BAND {band} 缺失物理极值定义")

    # ------------------------------------------------------------------ utils
    def _per_channel_params(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """返回每个通道的 (mid, half_range)。"""
        mids = []
        halves = []
        for band in self.band_order:
            mi, ma = self.limits[band]
            mids.append((mi + ma) * 0.5)
            halves.append((ma - mi) * 0.5)
        mid_t = torch.tensor(mids, dtype=torch.float32, device=device)
        half_t = torch.tensor(halves, dtype=torch.float32, device=device)
        return mid_t, half_t

    @staticmethod
    def _broadcast_shape(channel_dim: int, n_channels: int, ndim: int) -> tuple[int, ...]:
        shape = [1] * ndim
        shape[channel_dim] = n_channels
        return tuple(shape)

    # ------------------------------------------------------------------ API
    def encode(self, kelvin: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        """开尔文 → ``[-1, 1]``。"""
        if kelvin.size(channel_dim) != len(self.band_order):
            raise ValueError(
                f"channel_dim={channel_dim} 上的尺寸 {kelvin.size(channel_dim)} != {len(self.band_order)}"
            )
        mid, half = self._per_channel_params(kelvin.device)
        view_shape = self._broadcast_shape(channel_dim, len(self.band_order), kelvin.ndim)
        mid = mid.view(view_shape)
        half = half.view(view_shape)
        out = (kelvin - mid) / half
        return out.clamp_(-1.0, 1.0)

    def decode(self, normed: torch.Tensor, channel_dim: int = 1, *, clip_to_limits: bool = False) -> torch.Tensor:
        """``[-1, 1]`` → 开尔文。``clip_to_limits=True`` 时裁剪到物理极值域内。"""
        if normed.size(channel_dim) != len(self.band_order):
            raise ValueError(
                f"channel_dim={channel_dim} 上的尺寸 {normed.size(channel_dim)} != {len(self.band_order)}"
            )
        mid, half = self._per_channel_params(normed.device)
        view_shape = self._broadcast_shape(channel_dim, len(self.band_order), normed.ndim)
        mid = mid.view(view_shape)
        half = half.view(view_shape)
        kelvin = normed * half + mid
        if clip_to_limits:
            mins = torch.tensor(
                [self.limits[b][0] for b in self.band_order], dtype=kelvin.dtype, device=kelvin.device
            ).view(view_shape)
            maxs = torch.tensor(
                [self.limits[b][1] for b in self.band_order], dtype=kelvin.dtype, device=kelvin.device
            ).view(view_shape)
            kelvin = torch.maximum(torch.minimum(kelvin, maxs), mins)
        return kelvin

    def b13_threshold_norm(self, temperature_K: float) -> float:
        """把 B13 的开尔文阈值翻译到归一化域，与全局函数等价。"""
        return b13_norm_threshold_for_kelvin(temperature_K)
