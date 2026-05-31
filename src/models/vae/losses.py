"""时空 VAE 训练用损失：重建 L1（B13 冷云加权）+ KL + 可选 soft-Dice。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.data.normalizers import b13_norm_threshold_for_kelvin


def b13_soft_dice_loss(
    pred_b13_norm: torch.Tensor,
    true_b13_norm: torch.Tensor,
    *,
    thresholds_K: tuple[float, ...] = (240.0,),
    tau: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """B13 阈值事件（冷云）上的 soft-Dice 损失。

    使用可微近似掩膜:
        soft_mask = sigmoid((thr_norm - x_norm) / tau)

    其中 ``x_norm`` 越小表示越冷，越可能是事件像素。
    """
    if tau <= 0:
        raise ValueError(f"tau 必须为正，得到 tau={tau}")
    if len(thresholds_K) == 0:
        raise ValueError("thresholds_K 不能为空")

    losses: list[torch.Tensor] = []
    for thr_k in thresholds_K:
        thr_norm = b13_norm_threshold_for_kelvin(float(thr_k))
        thr_t = pred_b13_norm.new_tensor(thr_norm)
        p = torch.sigmoid((thr_t - pred_b13_norm) / tau)
        t = torch.sigmoid((thr_t - true_b13_norm) / tau)

        # 每个样本单独算 Dice，再做 batch 平均
        p_flat = p.reshape(p.size(0), -1)
        t_flat = t.reshape(t.size(0), -1)
        inter = (p_flat * t_flat).sum(dim=1)
        denom = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        losses.append(1.0 - dice.mean())
    return torch.stack(losses).mean()


def gradient_sharpen_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    *,
    channel: int = 3,
) -> torch.Tensor:
    """空间高频（梯度）一致性损失，用于抑制 decoder 输出的糊化。

    对指定通道（默认 B13=3）在 ``H, W`` 维做一阶有限差分，惩罚预测与真值的
    梯度差异，从而鼓励重建保留云顶边缘等高频结构。

    pred, true: ``[B, C, T, H, W]``（norm 域）。
    """
    p = pred[:, channel]
    t = true[:, channel]
    p_dx = p[..., :, 1:] - p[..., :, :-1]
    t_dx = t[..., :, 1:] - t[..., :, :-1]
    p_dy = p[..., 1:, :] - p[..., :-1, :]
    t_dy = t[..., 1:, :] - t[..., :-1, :]
    return (p_dx - t_dx).abs().mean() + (p_dy - t_dy).abs().mean()


def vae_total_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    kl_weight: float = 1e-6,
    b13_weight: float = 2.0,
    b13_cold_norm_thr: float = -0.0769,
    dice_weight: float = 0.0,
    dice_tau: float = 0.02,
    dice_thresholds_K: tuple[float, ...] = (240.0,),
    sharpen_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """返回 ``(loss, log_dict)``。

    * ``b13_cold_norm_thr``：约 240 K 对应的 norm 阈值（与 ``b13_norm_threshold_for_kelvin(240)`` 一致）。
    * 对 B13 通道中「冷于阈值」的像素施加 ``b13_weight`` 倍 L1 权重。
    * ``dice_weight > 0`` 时，额外加入 B13 阈值事件上的 soft-Dice 损失。
    * ``sharpen_weight > 0`` 时，额外加入 B13 空间梯度一致性损失（抑制糊化，Path 2 decoder 微调用）。
    """
    diff = (recon - x).abs()
    b13 = x[:, 3:4]
    cold = (b13 < b13_cold_norm_thr).to(diff.dtype)
    w = 1.0 + (b13_weight - 1.0) * cold
    l1_w = (diff * w).mean()
    l1 = diff.mean()
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    dice = l1.new_zeros(())
    if dice_weight > 0:
        dice = b13_soft_dice_loss(
            pred_b13_norm=recon[:, 3],
            true_b13_norm=x[:, 3],
            thresholds_K=dice_thresholds_K,
            tau=dice_tau,
        )
    sharpen = l1.new_zeros(())
    if sharpen_weight > 0:
        sharpen = gradient_sharpen_loss(recon, x)

    total = l1_w + kl_weight * kl + dice_weight * dice + sharpen_weight * sharpen
    logs = {
        "train/l1": l1.detach(),
        "train/l1_weighted": l1_w.detach(),
        "train/kl": kl.detach(),
        "train/dice": dice.detach(),
        "train/sharpen": sharpen.detach(),
    }
    return total, logs
