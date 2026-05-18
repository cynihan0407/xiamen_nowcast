"""EDM 扩散过程：预条件、损失、Heun ODE 采样测试。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.diffusion.edm import EDMConfig, EDMDiffusion


class _ToyDenoiser(nn.Module):
    """一个保形 denoiser，用于走通流程；输出与输入同形状。"""

    def __init__(self, channels: int = 4, cond_channels: int = 0) -> None:
        super().__init__()
        self.proj = nn.Conv3d(channels + cond_channels, channels, kernel_size=1)
        self.cond_channels = cond_channels

    def forward(self, x_in: torch.Tensor, c_noise: torch.Tensor, cond=None) -> torch.Tensor:
        del c_noise  # 测试用，不使用
        if cond is not None:
            x_in = torch.cat([x_in, cond], dim=1)
        return self.proj(x_in)


def test_edm_loss_finite_and_backward():
    net = _ToyDenoiser(channels=4, cond_channels=2)
    diff = EDMDiffusion(net, EDMConfig(num_steps=4))
    x = torch.randn(2, 4, 6, 8, 8)
    cond = torch.randn(2, 2, 6, 8, 8)
    loss, logs = diff.compute_loss(x, cond)
    assert torch.isfinite(loss)
    loss.backward()
    assert "train/edm_mse" in logs
    assert "train/sigma_mean" in logs


def test_edm_sigma_schedule_monotonic():
    diff = EDMDiffusion(_ToyDenoiser(), EDMConfig(num_steps=10))
    s = diff.build_sigma_schedule(num_steps=10)
    assert s.numel() == 11
    diffs = s[1:] - s[:-1]
    assert (diffs <= 0).all(), "sigma schedule 应严格递减"
    assert float(s[-1]) == 0.0


def test_edm_heun_sample_shape():
    net = _ToyDenoiser(channels=4, cond_channels=2)
    diff = EDMDiffusion(net, EDMConfig(num_steps=4))
    cond = torch.randn(2, 2, 6, 8, 8)
    out = diff.heun_sample((2, 4, 6, 8, 8), cond=cond, num_steps=4)
    assert out.shape == (2, 4, 6, 8, 8)
    assert torch.isfinite(out).all()


def test_edm_denoise_at_zero_sigma_close_to_identity():
    # sigma -> 0 时 c_skip -> 1, c_out -> 0；denoise(x; ~0) ≈ x
    net = _ToyDenoiser(channels=4)
    diff = EDMDiffusion(net, EDMConfig())
    x = torch.randn(2, 4, 4, 8, 8)
    sigma = torch.full((2,), 1e-4)
    out = diff.denoise(x, sigma, cond=None)
    assert torch.allclose(out, x, atol=1e-2)
