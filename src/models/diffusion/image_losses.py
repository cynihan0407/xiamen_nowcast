"""Stage-B 图像空间辅助损失：在 STVAE 解码后的 B13 上约束清晰度与冷云结构。"""
from __future__ import annotations

import torch

from src.metrics.grad_metrics import gradient_mae_b13_norm
from src.models.vae.losses import b13_soft_dice_loss


def diffusion_image_aux_loss(
    recon: torch.Tensor,
    future: torch.Tensor,
    *,
    l1_weight: float = 0.0,
    grad_weight: float = 0.0,
    dice_weight: float = 0.0,
    dice_thresholds_K: tuple[float, ...] = (240.0, 220.0),
    dice_tau: float = 0.02,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """解码图像上的 B13 辅助损失（STVAE 冻结，梯度回传到扩散 denoiser）。

    recon, future: ``[B, 4, T, H, W]``，norm 域。
    """
    zero = recon.new_zeros(())
    logs: dict[str, torch.Tensor] = {
        "train/img_l1": zero,
        "train/img_grad": zero,
        "train/img_dice": zero,
    }
    if l1_weight <= 0 and grad_weight <= 0 and dice_weight <= 0:
        return zero, logs

    total = zero
    if l1_weight > 0:
        l1 = (recon[:, 3:4] - future[:, 3:4]).abs().mean()
        logs["train/img_l1"] = l1.detach()
        total = total + l1_weight * l1
    if grad_weight > 0:
        grad = gradient_mae_b13_norm(recon, future, channel=3)
        logs["train/img_grad"] = grad.detach()
        total = total + grad_weight * grad
    if dice_weight > 0:
        dice = b13_soft_dice_loss(
            pred_b13_norm=recon[:, 3],
            true_b13_norm=future[:, 3],
            thresholds_K=dice_thresholds_K,
            tau=dice_tau,
        )
        logs["train/img_dice"] = dice.detach()
        total = total + dice_weight * dice
    return total, logs
