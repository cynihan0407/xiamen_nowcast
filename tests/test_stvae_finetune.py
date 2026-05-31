"""Path 2：decoder sharpening 微调相关测试（梯度损失 + 仅解冻 decoder 末层）。"""
from __future__ import annotations

import torch

from src.engine.stvae_module import STVAELightningModule
from src.models.vae.losses import gradient_sharpen_loss, vae_total_loss
from src.models.vae.stvae import STVAE, STVAEConfig


def test_gradient_sharpen_loss_zero_when_identical():
    x = torch.randn(2, 4, 4, 16, 16)
    assert float(gradient_sharpen_loss(x, x)) == 0.0


def test_gradient_sharpen_loss_penalizes_blur():
    true = torch.randn(2, 4, 4, 16, 16)
    # 模糊预测：对 B13 做均值平滑，高频被抹掉 → 梯度损失应 > 0
    blurred = true.clone()
    blurred[:, 3] = true[:, 3].mean(dim=(-1, -2), keepdim=True)
    sharp = true.clone()
    assert gradient_sharpen_loss(blurred, true) > gradient_sharpen_loss(sharp, true)


def test_vae_total_loss_with_sharpen_backward():
    x = torch.randn(2, 4, 8, 32, 32)
    recon = x.clone().requires_grad_(True)
    mu = torch.zeros(2, 8, 8, 2, 2)
    logvar = torch.zeros_like(mu)
    loss, logs = vae_total_loss(recon, x, mu, logvar, sharpen_weight=0.3)
    assert "train/sharpen" in logs
    assert torch.isfinite(loss)
    loss.backward()
    assert recon.grad is not None


def test_finetune_decoder_only_freezes_encoder():
    model = STVAE(STVAEConfig(in_channels=4, latent_channels=4, base_channels=8, num_down=2, seq_len=4))
    lit = STVAELightningModule(model, finetune_decoder_only=True, sharpen_weight=0.3)

    # encoder / to_mu / to_logvar 应全部冻结
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(not p.requires_grad for p in model.to_mu.parameters())
    # decoder 末层 + out_conv 应可训练
    assert all(p.requires_grad for p in model.decoder[-1].parameters())
    assert all(p.requires_grad for p in model.out_conv.parameters())
    # decoder 非末层应冻结
    assert all(not p.requires_grad for p in model.decoder[0].parameters())

    # optimizer 只应包含可训练参数
    opt = lit.configure_optimizers()
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_opt == n_train
    assert 0 < n_train < sum(p.numel() for p in model.parameters())
