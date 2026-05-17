"""时空 VAE（Stage-A）：将亮温序列压缩到 latent，并重建全序列。

输入 / 输出张量布局：``[B, C, T, H, W]``，其中 ``C=4``（B08–B13），``T=18``。
空间下采样仅在 ``H,W`` 维进行（``T`` 保持不变），符合 v1.1 对时间分辨率的要求。

实现要点：
* 编码器末端输出 ``mu`` 与 ``logvar``（各 ``latent_channels`` 维），重参数化得到 ``z``；
* 解码器从 ``z`` 上采样回 ``C`` 通道；
* 默认假设 ``H=W=256`` 且 ``H`` 可被 ``2**num_down`` 整除（训练时用 ``crop_size=256``）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def _same_hw(h: int, w: int) -> bool:
    if h != w:
        raise ValueError(f"当前实现要求 H==W，收到 H={h}, W={w}")
    return h


@dataclass
class STVAEConfig:
    in_channels: int = 4
    latent_channels: int = 8
    base_channels: int = 32
    num_down: int = 4          # 256 -> 16 需要 4 次 /2
    seq_len: int = 18


class STVAE(nn.Module):
    """3D 卷积时空 VAE。"""

    def __init__(self, cfg: STVAEConfig | None = None, **kwargs: object) -> None:
        super().__init__()
        kwargs = {k: v for k, v in kwargs.items() if k != "_target_"}
        if cfg is None:
            fields = STVAEConfig.__dataclass_fields__
            valid = {k: v for k, v in kwargs.items() if k in fields}
            cfg = STVAEConfig(**valid)
        self.cfg = cfg
        c0 = cfg.in_channels
        bc = cfg.base_channels
        lat = cfg.latent_channels
        nd = cfg.num_down

        chs = [bc * (2**i) for i in range(nd)]  # 32,64,128,256
        enc_layers: list[nn.Module] = []
        cin = c0
        for cout in chs:
            enc_layers.append(
                nn.Sequential(
                    nn.Conv3d(cin, cout, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)),
                    nn.GroupNorm(min(32, cout), cout),
                    nn.SiLU(),
                )
            )
            cin = cout
        self.encoder = nn.ModuleList(enc_layers)
        self.to_mu = nn.Conv3d(chs[-1], lat, kernel_size=1)
        self.to_logvar = nn.Conv3d(chs[-1], lat, kernel_size=1)

        dec_layers: list[nn.Module] = []
        cin = lat
        for i in range(nd - 1, -1, -1):
            cout = bc * (2**i)
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        cin, cout, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1), output_padding=(0, 0, 0)
                    ),
                    nn.GroupNorm(min(32, cout), cout),
                    nn.SiLU(),
                )
            )
            cin = cout
        self.decoder = nn.ModuleList(dec_layers)
        self.out_conv = nn.Conv3d(bc, c0, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B,C,T,H,W] -> mu, logvar: [B,lat,T,h',w']"""
        h = x
        for layer in self.encoder:
            h = layer(h)
        return self.to_mu(h), self.to_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for layer in self.decoder:
            h = layer(h)
        return self.out_conv(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 ``(recon, mu, logvar)``，``recon`` 与 ``x`` 同形状。"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def expected_latent_hw(image_hw: int, num_down: int) -> int:
        """给定正方形边长，返回 latent 空间 ``H'=W'``。"""
        _same_hw(image_hw, image_hw)
        return image_hw // (2**num_down)

    @torch.no_grad()
    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)
