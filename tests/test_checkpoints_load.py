"""回归测试：load_diffusion_lit 不得用扩散 ckpt 内嵌的 STVAE 覆盖外部 STVAE。"""
from __future__ import annotations

import torch

from src.engine.diffusion_module import DiffusionLightningModule
from src.models.diffusion.edm import EDMConfig, EDMDiffusion
from src.models.unet3d.unet3d import UNet3D, UNet3DConfig
from src.models.vae.stvae import STVAE, STVAEConfig
from src.utils.checkpoints import load_diffusion_lit


def _make_lit() -> DiffusionLightningModule:
    stvae = STVAE(STVAEConfig(in_channels=4, latent_channels=4, base_channels=8, num_down=2, seq_len=4))
    unet = UNet3D(UNet3DConfig(
        in_channels=8, out_channels=4, base_channels=8, channel_mult=(1, 2),
        num_res_blocks=1, attn_resolutions=(), time_embed_dim=16, num_heads=1,
    ))
    diffusion = EDMDiffusion(unet, EDMConfig(num_steps=2))
    return DiffusionLightningModule(diffusion=diffusion, stvae=stvae, ema_enable=False)


def test_load_diffusion_lit_preserves_external_stvae(tmp_path):
    # 1) 训练态 lit：把 stvae.out_conv 权重设为全 1，存成扩散 ckpt
    trained = _make_lit()
    with torch.no_grad():
        for p in trained.stvae.out_conv.parameters():
            p.fill_(1.0)
    ckpt = {"state_dict": trained.state_dict()}
    ckpt_path = tmp_path / "diff.ckpt"
    torch.save(ckpt, ckpt_path)

    # 2) 评估态 lit：外部装入"新 STVAE"，out_conv 设为全 2（模拟替换后的 decoder）
    fresh = _make_lit()
    with torch.no_grad():
        for p in fresh.stvae.out_conv.parameters():
            p.fill_(2.0)

    load_diffusion_lit(fresh, ckpt_path, use_ema=False, device="cpu")

    # 3) 加载扩散 ckpt 后，stvae.out_conv 应仍为 2（未被 ckpt 里的 1 覆盖）
    for p in fresh.stvae.out_conv.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0)), "外部 STVAE 被扩散 ckpt 覆盖了！"
