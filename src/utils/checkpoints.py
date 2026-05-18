"""Checkpoint 加载工具（Stage-A STVAE / Stage-B 扩散）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from src.engine.diffusion_module import DiffusionLightningModule, EMAState
from src.models.diffusion.edm import EDMDiffusion
from src.models.vae.stvae import STVAE


def load_stvae_weights(stvae: STVAE, ckpt_path: str | Path) -> tuple[list[str], list[str]]:
    """从 Lightning STVAE checkpoint 加载权重到 ``stvae``。"""
    p = Path(ckpt_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"stvae_ckpt_path 不存在: {p}")
    state = torch.load(p, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    model_state: dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
        elif k.startswith("stvae."):
            model_state[k[len("stvae."):]] = v
        elif not k.startswith(("diffusion.", "ema")):
            model_state[k] = v
    missing, unexpected = stvae.load_state_dict(model_state, strict=False)
    stvae.eval()
    for param in stvae.parameters():
        param.requires_grad_(False)
    return list(missing), list(unexpected)


def load_diffusion_lit(
    lit: DiffusionLightningModule,
    ckpt_path: str | Path,
    *,
    use_ema: bool = True,
    device: torch.device | str = "cpu",
) -> None:
    """加载 Stage-B Lightning checkpoint 到 ``lit``，可选应用 EMA 到 denoiser。"""
    p = Path(ckpt_path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"diffusion_ckpt_path 不存在: {p}")
    state = torch.load(p, map_location=device, weights_only=False)
    sd = state.get("state_dict", state)
    lit.load_state_dict(sd, strict=False)

    if use_ema and "ema_shadow" in state:
        shadow = state["ema_shadow"]
        if lit.ema is None:
            lit.ema = EMAState(lit.diffusion.denoiser, decay=lit._ema_decay)
        for k, v in shadow.items():
            if k in lit.ema.shadow:
                lit.ema.shadow[k] = v.to(device)
        lit.ema.copy_to(lit.diffusion.denoiser)

    lit.eval()
    lit.stvae.eval()


def build_lit_from_components(
    diffusion: EDMDiffusion,
    stvae: STVAE,
    **lit_kwargs: Any,
) -> DiffusionLightningModule:
    return DiffusionLightningModule(diffusion=diffusion, stvae=stvae, **lit_kwargs)
