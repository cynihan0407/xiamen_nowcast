"""轻量临近预报基线。

* ``ConcatConvNowcast``：把过去 ``T_p`` 帧沿通道拼接，用 2D CNN 直接回归未来 ``T_f`` 帧（强基线、训练快）。
* ``SimpleConvLSTMNowcast``：单层 ConvLSTM 在特征图上递推时间，再解码为多帧输出。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcatConvNowcast(nn.Module):
    """``[B, C*T_p, H, W] -> [B, C*T_f, H, W]`` 的 U-Net 风格浅网络。"""

    def __init__(self, in_channels: int = 4, past_len: int = 6, future_len: int = 12, hidden: int = 64):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.in_channels = in_channels
        ic = in_channels * past_len
        oc = in_channels * future_len
        self.net = nn.Sequential(
            nn.Conv2d(ic, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, oc, 3, padding=1),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        # past: [B, C, T_p, H, W]
        b, c, t, h, w = past.shape
        x = past.reshape(b, c * t, h, w)
        y = self.net(x)
        return y.reshape(b, c, self.future_len, h, w)


class ConvLSTMCell(nn.Module):
    def __init__(self, in_c: int, hidden: int, k: int = 3):
        super().__init__()
        pad = k // 2
        self.hidden = hidden
        self.gates = nn.Conv2d(in_c + hidden, hidden * 4, k, padding=pad)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        g = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g_ = torch.chunk(g, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g_ = torch.tanh(g_)
        c = f * c + i * g_
        h = o * torch.tanh(c)
        return h, c


class SimpleConvLSTMNowcast(nn.Module):
    """先用 1x1 把 ``C`` 投到 ``hidden``，再在 ``T_p`` 上跑 ConvLSTM，最后 1x1 解到 ``C * T_f``。"""

    def __init__(self, in_channels: int = 4, past_len: int = 6, future_len: int = 12, hidden: int = 48):
        super().__init__()
        self.past_len = past_len
        self.future_len = future_len
        self.in_channels = in_channels
        self.embed = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.cell = ConvLSTMCell(hidden, hidden)
        self.head = nn.Sequential(
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, in_channels * future_len, kernel_size=1),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        b, c, t, hi, wi = past.shape
        hid = self.cell.hidden
        h = torch.zeros(b, hid, hi, wi, device=past.device, dtype=past.dtype)
        cc = torch.zeros_like(h)
        for ti in range(t):
            x = self.embed(past[:, :, ti])
            h, cc = self.cell(x, h, cc)
        out = self.head(h)
        return out.reshape(b, c, self.future_len, hi, wi)
