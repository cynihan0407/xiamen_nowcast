"""物理感知数据增强。

设计原则：
1. **仅几何变换**：临近预报中亮温的统计分布、谱特性、通道间物理关系都不能动；
   因此禁止任何亮度/对比度/色彩抖动、CutOut、MixUp 等会破坏物理分布的增强。
2. **同一序列共享同一组随机参数**：保证时间一致性（不然光流被破坏）。
3. **过去/未来共享同一变换**：在 V11 设定下，past/future 是同一个序列的切片，
   必须用同一组几何变换以保留时空因果性。
4. **可关闭**：评估阶段必须 ``enable=False``。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class GeometricAugConfig:
    """几何增强超参。"""

    enable: bool = True
    flip_horizontal: bool = True
    flip_vertical: bool = True
    rot90_p: float = 0.5
    rot90_choices: tuple[int, ...] = (0, 1, 2, 3)


def _maybe_flip(seq: torch.Tensor, dim: int, do_flip: bool) -> torch.Tensor:
    return torch.flip(seq, dims=(dim,)) if do_flip else seq


def _rot90(seq: torch.Tensor, k: int, dims: tuple[int, int]) -> torch.Tensor:
    if k % 4 == 0:
        return seq
    return torch.rot90(seq, k=k, dims=dims)


class SequenceGeometricAug:
    """对 ``[T, C, H, W]`` 或 ``[C, T, H, W]`` 序列应用几何增强。

    每次 ``__call__`` 重新采样一组随机参数，序列内部所有帧共享。

    Args:
        cfg:        增强配置；``cfg.enable=False`` 时直接返回原张量。
        layout:     ``"TCHW"`` (默认) 或 ``"CTHW"`` ，决定 H/W 所在轴。
        rng:        可选的 ``random.Random`` 实例，用于可复现实验。
    """

    def __init__(
        self,
        cfg: GeometricAugConfig,
        *,
        layout: str = "TCHW",
        rng: Optional[random.Random] = None,
    ) -> None:
        if layout not in ("TCHW", "CTHW"):
            raise ValueError(f"未知 layout: {layout}")
        self.cfg = cfg
        self.layout = layout
        self.rng = rng or random.Random()

    @property
    def _hw_dims(self) -> tuple[int, int]:
        # H/W 永远在最后两维
        return (-2, -1)

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable:
            return seq

        h_dim, w_dim = self._hw_dims

        # === 1. 翻转 ===========================================================
        if self.cfg.flip_horizontal and self.rng.random() < 0.5:
            seq = _maybe_flip(seq, w_dim, do_flip=True)
        if self.cfg.flip_vertical and self.rng.random() < 0.5:
            seq = _maybe_flip(seq, h_dim, do_flip=True)

        # === 2. 90° 旋转 =======================================================
        if self.cfg.rot90_p > 0.0 and self.rng.random() < self.cfg.rot90_p:
            k = self.rng.choice(self.cfg.rot90_choices)
            seq = _rot90(seq, k=k, dims=(h_dim, w_dim))

        return seq.contiguous()


class CropTransform:
    """空间裁剪：训练随机、评估中心。

    输入张量布局 ``[T, C, H, W]`` 或 ``[C, T, H, W]``，输出形状一致但 H/W 改为 crop_size。
    """

    def __init__(self, crop_size: int, mode: str = "random", *, rng: Optional[random.Random] = None) -> None:
        if mode not in ("random", "center"):
            raise ValueError(f"mode 必须是 random|center，收到: {mode}")
        self.crop_size = int(crop_size)
        self.mode = mode
        self.rng = rng or random.Random()

    def __call__(self, seq: torch.Tensor) -> torch.Tensor:
        h, w = seq.shape[-2], seq.shape[-1]
        if self.crop_size > h or self.crop_size > w:
            raise ValueError(f"crop_size={self.crop_size} 超过原始尺寸 H×W={h}×{w}")
        if self.mode == "random":
            y0 = self.rng.randint(0, h - self.crop_size)
            x0 = self.rng.randint(0, w - self.crop_size)
        else:
            y0 = (h - self.crop_size) // 2
            x0 = (w - self.crop_size) // 2

        return seq[..., y0 : y0 + self.crop_size, x0 : x0 + self.crop_size].contiguous()


def numpy_to_tensor(seq: np.ndarray, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``np.ndarray`` → ``torch.Tensor``，自动 contiguous 且不共享内存。

    避免 ``np.load`` 关闭后 mmap 被释放导致的悬空指针。
    """
    arr = np.ascontiguousarray(seq).copy()
    return torch.from_numpy(arr).to(dtype=dtype)
