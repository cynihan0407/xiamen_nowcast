"""VAE losses: soft-Dice 分支测试。"""
from __future__ import annotations

import torch

from src.models.vae.losses import b13_soft_dice_loss, vae_total_loss


def test_soft_dice_lower_when_prediction_matches_truth() -> None:
    # 构造明显冷云事件：norm 越小越冷
    true_b13 = torch.full((2, 4, 8, 8), -0.3)
    pred_good = true_b13.clone()
    pred_bad = torch.full_like(true_b13, 0.3)
    lg = b13_soft_dice_loss(pred_good, true_b13, thresholds_K=(240.0,), tau=0.02)
    lb = b13_soft_dice_loss(pred_bad, true_b13, thresholds_K=(240.0,), tau=0.02)
    assert lg < lb


def test_vae_total_loss_with_dice_backward() -> None:
    x = torch.randn(2, 4, 18, 32, 32)
    recon = x.clone().requires_grad_(True)
    mu = torch.zeros(2, 8, 18, 2, 2)
    logvar = torch.zeros_like(mu)
    loss, logs = vae_total_loss(
        recon,
        x,
        mu,
        logvar,
        kl_weight=1e-6,
        b13_weight=2.0,
        dice_weight=1.0,
        dice_tau=0.02,
        dice_thresholds_K=(240.0, 230.0),
    )
    assert "train/dice" in logs
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None
