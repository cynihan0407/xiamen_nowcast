"""UNet3D 形状与基本前向测试。"""
from __future__ import annotations

import torch

from src.models.unet3d.unet3d import UNet3D, UNet3DConfig


def test_unet3d_forward_with_cond():
    cfg = UNet3DConfig(
        in_channels=16,
        out_channels=8,
        base_channels=16,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=1,
        attn_resolutions=(4, 2),
        time_embed_dim=64,
        num_heads=2,
    )
    m = UNet3D(cfg)
    B, T, H, W = 2, 12, 16, 16
    x = torch.randn(B, 8, T, H, W)
    cond = torch.randn(B, 8, T, H, W)
    c_noise = torch.randn(B)
    y = m(x, c_noise, cond)
    assert y.shape == (B, 8, T, H, W)


def test_unet3d_forward_no_cond():
    m = UNet3D(UNet3DConfig(
        in_channels=4, out_channels=4, base_channels=16, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(), time_embed_dim=32, num_heads=2,
    ))
    B, T, H, W = 1, 6, 8, 8
    x = torch.randn(B, 4, T, H, W)
    c_noise = torch.randn(B)
    y = m(x, c_noise, None)
    assert y.shape == x.shape


def test_unet3d_grad_flow():
    """out_conv 零初始化时 y==0，但只要把 out_conv 的权重略扰动，整网应有梯度。"""
    m = UNet3D(UNet3DConfig(
        in_channels=8, out_channels=4, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(), time_embed_dim=16, num_heads=1,
    ))
    with torch.no_grad():
        m.out_conv.weight.add_(torch.randn_like(m.out_conv.weight) * 0.05)
    x = torch.randn(1, 4, 4, 8, 8)
    cond = torch.randn(1, 4, 4, 8, 8)
    c_noise = torch.zeros(1)
    y = m(x, c_noise, cond)
    target = torch.ones_like(y)
    loss = (y - target).pow(2).mean()
    loss.backward()
    grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())
    assert grads_ok
