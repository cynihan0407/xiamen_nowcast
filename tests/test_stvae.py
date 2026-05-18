"""STVAE 与基线前向形状测试。"""
from __future__ import annotations

import torch

from src.models.baselines.convlstm_nowcast import ConcatConvNowcast, SimpleConvLSTMNowcast
from src.models.vae.losses import vae_total_loss
from src.models.vae.stvae import STVAE, STVAEConfig


def test_stvae_forward_256():
    m = STVAE(STVAEConfig())
    x = torch.randn(2, 4, 18, 256, 256)
    r, mu, lv = m(x)
    assert r.shape == x.shape
    assert mu.shape[1] == 8


def test_stvae_large_base_channels_48():
    """P1-B 配置：base_channels=48 时 GroupNorm 须可整除。"""
    m = STVAE(STVAEConfig(base_channels=48, latent_channels=12))
    x = torch.randn(1, 4, 18, 256, 256)
    r, mu, lv = m(x)
    assert r.shape == x.shape
    assert mu.shape[1] == 12


def test_vae_loss_finite():
    m = STVAE(STVAEConfig())
    x = torch.randn(2, 4, 18, 256, 256)
    r, mu, lv = m(x)
    loss, logs = vae_total_loss(r, x, mu, lv)
    assert loss.item() == loss.item()
    assert "train/kl" in logs


def test_concat_conv_baseline():
    m = ConcatConvNowcast()
    p = torch.randn(2, 4, 6, 64, 64)
    y = m(p)
    assert y.shape == (2, 4, 12, 64, 64)


def test_simple_convlstm_baseline():
    m = SimpleConvLSTMNowcast(hidden=32)
    p = torch.randn(2, 4, 6, 32, 32)
    y = m(p)
    assert y.shape == (2, 4, 12, 32, 32)
