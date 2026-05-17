"""时空 VAE 训练用损失：重建 L1（B13 冷云加权）+ KL。"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_total_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float = 1e-6,
    b13_weight: float = 2.0,
    b13_cold_norm_thr: float = -0.0769,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 ``(loss, log_dict)``。

    * ``b13_cold_norm_thr``：约 240 K 对应的 norm 阈值（与 ``b13_norm_threshold_for_kelvin(240)`` 一致）。
    * 对 B13 通道中「冷于阈值」的像素施加 ``b13_weight`` 倍 L1 权重。
    """
    diff = (recon - x).abs()
    b13 = x[:, 3:4]
    cold = (b13 < b13_cold_norm_thr).to(diff.dtype)
    w = 1.0 + (b13_weight - 1.0) * cold
    l1_w = (diff * w).mean()
    l1 = diff.mean()
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = l1_w + kl_weight * kl
    logs = {
        "train/l1": l1.detach(),
        "train/l1_weighted": l1_w.detach(),
        "train/kl": kl.detach(),
    }
    return total, logs
