"""3D U-Net 主干（适配 latent 扩散）。

输入 / 输出：``[B, C, T, H, W]``，``T`` 在网络内部 **保持不变**（只在 H,W 下采样）。

关键设计：
* **(2+1)D 卷积**：空间 ``Conv3d(kernel=(1,3,3))`` + 时间 ``Conv3d(kernel=(3,1,1))``；
* **AdaGN 注入 sigma**：根据 EDM 的 ``c_noise`` 时间嵌入产生 ``(scale, shift)``，
  作用于每个 ResBlock 的 GroupNorm 输出；
* **Factorized attention**（可选）：
  * spatial attention：把 ``T`` 维并入 batch，对 ``(H*W)`` 做 self-attention；
  * temporal attention：把 ``(H*W)`` 维并入 batch，对 ``T`` 做 self-attention；

为单卡 A100 友好，``base_channels=64~96`` 即可获得足够容量。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 基础组件
# ---------------------------------------------------------------------------
def _zero_init(m: nn.Module) -> nn.Module:
    """Conv 输出层零初始化，便于扩散训练早期稳定。"""
    if hasattr(m, "weight") and m.weight is not None:
        nn.init.zeros_(m.weight)
    if hasattr(m, "bias") and m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class SinusoidalTimeEmbedding(nn.Module):
    """log-sigma → 正弦时间嵌入；适配 EDM 的 ``c_noise``（标量）。"""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"time embed dim 必须为偶数，得到 {dim}")
        self.dim = dim
        self.max_period = max_period

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        device = c_noise.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, dtype=torch.float32, device=device) / half
        )
        args = c_noise.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb.to(c_noise.dtype)


def _pick_groups(channels: int, max_groups: int = 32) -> int:
    """选择能整除 ``channels`` 的最大 ``num_groups``（≤ max_groups）。"""
    for g in (max_groups, 16, 8, 4, 2, 1):
        if g <= channels and channels % g == 0:
            return g
    return 1


class AdaGN(nn.Module):
    """根据时间嵌入产生 (scale, shift) 注入 GroupNorm 输出。"""

    def __init__(self, channels: int, time_dim: int, num_groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_pick_groups(channels, num_groups), channels, affine=False)
        self.proj = nn.Linear(time_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.proj(temb).chunk(2, dim=-1)
        # 广播到 [B, C, 1, 1, 1]
        while scale.ndim < x.ndim:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return h * (1.0 + scale) + shift


class TwoPlusOneDConv(nn.Module):
    """(2+1)D 卷积：先空间 (1,3,3) 后时间 (3,1,1)。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.spatial = nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.temporal = nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1), padding=(1, 0, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(x))


class ResBlock3D(nn.Module):
    """(2+1)D ResBlock + AdaGN(sigma)。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.adagn1 = AdaGN(in_ch, time_dim)
        self.conv1 = TwoPlusOneDConv(in_ch, out_ch)
        self.adagn2 = AdaGN(out_ch, time_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = TwoPlusOneDConv(out_ch, out_ch)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.adagn1(x, temb)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.adagn2(h, temb)))
        return h + self.skip(x)


class FactorizedAttention(nn.Module):
    """空间 / 时间 self-attention（可选启用，按 head 切分）。"""

    def __init__(self, channels: int, num_heads: int = 4, spatial: bool = True, temporal: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.spatial_enable = spatial
        self.temporal_enable = temporal
        if spatial:
            self.norm_s = nn.GroupNorm(_pick_groups(channels), channels)
            self.attn_s = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.proj_s = _zero_init(nn.Linear(channels, channels))
        if temporal:
            self.norm_t = nn.GroupNorm(_pick_groups(channels), channels)
            self.attn_t = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.proj_t = _zero_init(nn.Linear(channels, channels))

    def _spatial(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm_s(x)
        h = h.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        h, _ = self.attn_s(h, h, h, need_weights=False)
        h = self.proj_s(h)
        h = h.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x + h

    def _temporal(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        h = self.norm_t(x)
        h = h.permute(0, 3, 4, 2, 1).reshape(B * H * W, T, C)
        h, _ = self.attn_t(h, h, h, need_weights=False)
        h = self.proj_t(h)
        h = h.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()
        return x + h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_enable:
            x = self._spatial(x)
        if self.temporal_enable:
            x = self._temporal(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# 主网络
# ---------------------------------------------------------------------------
@dataclass
class UNet3DConfig:
    in_channels: int = 16          # latent C_z + cond C_z
    out_channels: int = 8          # latent C_z
    base_channels: int = 64
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16, 8)
    time_embed_dim: int = 256
    dropout: float = 0.0
    num_heads: int = 4
    attention_spatial: bool = True
    attention_temporal: bool = True

    def __post_init__(self) -> None:
        self.channel_mult = tuple(self.channel_mult)
        self.attn_resolutions = tuple(self.attn_resolutions)


class UNet3D(nn.Module):
    """latent 扩散用 3D U-Net。

    forward 签名（与 ``EDMDiffusion.denoiser`` 一致）::

        unet(x_in, c_noise, cond)
            x_in:    [B, out_channels, T, H, W]
            c_noise: [B]
            cond:    [B, cond_channels, T, H, W]，cond_channels == in_channels - out_channels

    返回与 ``x_in`` 同形状。
    """

    def __init__(self, cfg: Optional[UNet3DConfig] = None, **kwargs) -> None:
        super().__init__()
        kwargs.pop("_target_", None)
        if cfg is None:
            fields = UNet3DConfig.__dataclass_fields__
            cfg = UNet3DConfig(**{k: v for k, v in kwargs.items() if k in fields})
        self.cfg = cfg

        ch = cfg.base_channels
        time_dim = cfg.time_embed_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(ch),
            nn.Linear(ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv3d(cfg.in_channels, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # 通道列表：[base*mult[0], base*mult[1], ...]
        chs = [ch * m for m in cfg.channel_mult]
        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.down_skip_channels: list[int] = [ch]
        cur_ch = ch
        cur_res = -1  # 仅作分辨率比较用；在 forward 中通过 H 实际值与 attn_resolutions 对比
        for level, out_ch in enumerate(chs):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlock3D(cur_ch, out_ch, time_dim, dropout=cfg.dropout))
                attns.append(
                    FactorizedAttention(
                        out_ch,
                        num_heads=cfg.num_heads,
                        spatial=cfg.attention_spatial,
                        temporal=cfg.attention_temporal,
                    )
                )
                cur_ch = out_ch
                self.down_skip_channels.append(cur_ch)
            self.down_blocks.append(blocks)
            self.down_attn.append(attns)
            if level < len(chs) - 1:
                self.down_skip_channels.append(cur_ch)  # 下采样前再 push 一次跳跃

        self.downsamples = nn.ModuleList(
            [Downsample3D(chs[i]) for i in range(len(chs) - 1)]
        )

        # 中间块
        mid_ch = chs[-1]
        self.mid_block1 = ResBlock3D(mid_ch, mid_ch, time_dim, dropout=cfg.dropout)
        self.mid_attn = FactorizedAttention(
            mid_ch, num_heads=cfg.num_heads, spatial=cfg.attention_spatial, temporal=cfg.attention_temporal
        )
        self.mid_block2 = ResBlock3D(mid_ch, mid_ch, time_dim, dropout=cfg.dropout)

        # 上采样路径（与下采样对称）
        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        skip_channels = list(self.down_skip_channels)
        for level in reversed(range(len(chs))):
            out_ch = chs[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(cfg.num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                blocks.append(ResBlock3D(cur_ch + skip_ch, out_ch, time_dim, dropout=cfg.dropout))
                attns.append(
                    FactorizedAttention(
                        out_ch,
                        num_heads=cfg.num_heads,
                        spatial=cfg.attention_spatial,
                        temporal=cfg.attention_temporal,
                    )
                )
                cur_ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attn.append(attns)
        self.upsamples = nn.ModuleList(
            [Upsample3D(chs[i]) for i in reversed(range(1, len(chs)))]
        )

        self.out_norm = nn.GroupNorm(_pick_groups(cur_ch), cur_ch)
        self.out_conv = _zero_init(nn.Conv3d(cur_ch, cfg.out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1)))

        self._attn_resolutions = set(int(r) for r in cfg.attn_resolutions)

    # ------------------------------------------------------------------ helpers
    def _attn_enabled(self, h: torch.Tensor) -> bool:
        return int(h.shape[-1]) in self._attn_resolutions

    def forward(
        self,
        x_in: torch.Tensor,
        c_noise: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if cond is not None:
            if cond.shape[2:] != x_in.shape[2:]:
                raise ValueError(
                    f"cond 与 x_in 的 (T,H,W) 必须一致，得到 {cond.shape} vs {x_in.shape}"
                )
            x = torch.cat([x_in, cond], dim=1)
        else:
            x = x_in
        temb = self.time_embed(c_noise)

        h = self.input_conv(x)
        skips: list[torch.Tensor] = [h]

        n_levels = len(self.cfg.channel_mult)
        for level in range(n_levels):
            blocks = self.down_blocks[level]
            attns = self.down_attn[level]
            for blk, attn in zip(blocks, attns):
                h = blk(h, temb)
                if self._attn_enabled(h):
                    h = attn(h)
                skips.append(h)
            if level < n_levels - 1:
                h = self.downsamples[level](h)
                skips.append(h)

        h = self.mid_block1(h, temb)
        if self._attn_enabled(h):
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for up_idx, level in enumerate(reversed(range(n_levels))):
            blocks = self.up_blocks[up_idx]
            attns = self.up_attn[up_idx]
            for blk, attn in zip(blocks, attns):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = blk(h, temb)
                if self._attn_enabled(h):
                    h = attn(h)
            if level > 0:
                h = self.upsamples[up_idx](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
