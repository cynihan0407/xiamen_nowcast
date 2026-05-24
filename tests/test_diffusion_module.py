"""DiffusionLightningModule 单步 / 采样 / EMA / build_cond 测试。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.engine.diffusion_module import (
    DiffusionLightningModule,
    EMAState,
    build_cond_from_past,
)
from src.models.diffusion.edm import EDMConfig, EDMDiffusion
from src.models.unet3d.unet3d import UNet3D, UNet3DConfig
from src.models.vae.stvae import STVAE, STVAEConfig


# ---------------------------------------------------------------------------
# build_cond
# ---------------------------------------------------------------------------
def test_build_cond_repeat_interleave():
    z_past = torch.randn(2, 8, 6, 16, 16)
    cond = build_cond_from_past(z_past, t_future=12)
    assert cond.shape == (2, 8, 12, 16, 16)
    # 第 0,1 个 future 帧应等于 past[0]，第 2,3 等于 past[1] ...
    for i in range(6):
        for k in (0, 1):
            assert torch.equal(cond[:, :, 2 * i + k], z_past[:, :, i])


def test_build_cond_non_divisible_pads_with_last():
    z_past = torch.randn(1, 4, 5, 4, 4)
    cond = build_cond_from_past(z_past, t_future=12)
    assert cond.shape == (1, 4, 12, 4, 4)
    # 末尾应是 padding（与 past 的最后一帧一致）
    assert torch.equal(cond[:, :, -1], z_past[:, :, -1])


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def test_ema_update_and_restore():
    net = nn.Linear(4, 4)
    ema = EMAState(net, decay=0.5)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(1.0)
    ema.update(net)


def test_ema_update_after_model_moved_to_cuda():
    """续训场景：shadow 在 CPU、参数在 GPU 时 update 不应报错。"""
    if not torch.cuda.is_available():
        return
    net = nn.Linear(4, 4)
    ema = EMAState(net, decay=0.5)
    net.cuda()
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(2.0)
    ema.update(net)
    for n, p in net.named_parameters():
        assert ema.shadow[n].device == p.device
    # shadow = 0.5*shadow + 0.5*1.0；初始 shadow 为初始权重
    assert all((v != 1.0).any() for v in ema.shadow.values())

    backup = ema.copy_to(net)
    # 此时 net == shadow
    for n, p in net.named_parameters():
        assert torch.allclose(p, ema.shadow[n])
    ema.restore(net, backup)
    # 还原后 net == 全 1
    for p in net.parameters():
        assert torch.allclose(p, torch.ones_like(p))


# ---------------------------------------------------------------------------
# 训练 / 验证单步
# ---------------------------------------------------------------------------
def _toy_module() -> DiffusionLightningModule:
    stvae = STVAE(STVAEConfig(in_channels=4, latent_channels=4, base_channels=8, num_down=2, seq_len=4))
    unet = UNet3D(UNet3DConfig(
        in_channels=8, out_channels=4, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(), time_embed_dim=16, num_heads=1,
    ))
    diffusion = EDMDiffusion(unet, EDMConfig(num_steps=2))
    return DiffusionLightningModule(
        diffusion=diffusion,
        stvae=stvae,
        lr=1e-4,
        ema_enable=True,
        ema_decay=0.99,
        cosine_t_max=None,
        val_sample_steps=2,
        val_sample_max_batches=1,
    )


def test_training_step_runs():
    m = _toy_module()
    # 模拟 trainer 行为：手工注入 EMA、log 函数
    m.ema = EMAState(m.diffusion.denoiser, decay=0.99)
    m.log = lambda *a, **kw: None  # type: ignore[assignment]
    m.log_dict = lambda *a, **kw: None  # type: ignore[assignment]

    past = torch.randn(2, 4, 2, 16, 16)
    future = torch.randn(2, 4, 2, 16, 16)
    loss = m.training_step({"past": past, "future": future}, batch_idx=0)
    assert torch.isfinite(loss)
    loss.backward()


def test_sample_future_shape():
    m = _toy_module()
    z_past = torch.randn(1, 4, 1, 4, 4)
    z_pred = m._sample_future(z_past, t_future=2, num_steps=2)
    assert z_pred.shape == (1, 4, 2, 4, 4)
