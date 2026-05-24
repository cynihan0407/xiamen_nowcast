"""Stage-B 扩散 Lightning 模块：冻结 STVAE + EDM 训练 / 采样。

数据流（训练）::

    batch["past"]:   [B, 4, 6, 256, 256]   ── STVAE.encode → z_past   [B, 8, 6, 16, 16]
    batch["future"]: [B, 4, 12, 256, 256]  ── STVAE.encode → z_future [B, 8, 12, 16, 16]
    cond = build_cond(z_past)              形状 [B, 8, 12, 16, 16]
    loss = EDM.compute_loss(z_future, cond)

数据流（验证）::

    z_future_pred = EDM.heun_sample(shape, cond)
    future_pred   = STVAE.decode(z_future_pred)
    指标 = CSI/MAE on B13 (开尔文域)
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.normalizers import norm_to_kelvin_np
from src.metrics.csi import csi_at_threshold_k
from src.models.diffusion.edm import EDMDiffusion
from src.models.vae.stvae import STVAE


# ---------------------------------------------------------------------------
# 条件构造
# ---------------------------------------------------------------------------
def build_cond_from_past(z_past: torch.Tensor, t_future: int) -> torch.Tensor:
    """把 ``z_past [B, C, T_past, H, W]`` 扩展为 ``cond [B, C, T_future, H, W]``。

    策略：``repeat_interleave``（每个 past 帧覆盖 ``t_future / t_past`` 个 future 帧）。
    若不能整除，则后部用最后一帧 padding。
    """
    B, C, T_p, H, W = z_past.shape
    if t_future == T_p:
        return z_past
    rep = max(1, t_future // T_p)
    cond = z_past.repeat_interleave(rep, dim=2)
    if cond.size(2) < t_future:
        pad = z_past[:, :, -1:, :, :].expand(-1, -1, t_future - cond.size(2), -1, -1)
        cond = torch.cat([cond, pad], dim=2)
    else:
        cond = cond[:, :, :t_future]
    return cond


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMAState:
    """对外提供 ``update`` / ``copy_to`` 的简单 EMA 容器。"""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow[n]
            if s.device != p.device:
                s = s.to(device=p.device, dtype=p.dtype)
                self.shadow[n] = s
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.device, p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, torch.Tensor]) -> None:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n].to(device=p.device, dtype=p.dtype))


# ---------------------------------------------------------------------------
# Lightning 模块
# ---------------------------------------------------------------------------
class DiffusionLightningModule(LightningModule):
    def __init__(
        self,
        diffusion: EDMDiffusion,
        stvae: STVAE,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.999),
        ema_decay: float = 0.9999,
        ema_enable: bool = True,
        cosine_t_max: Optional[int] = None,
        cosine_eta_min: float = 1e-6,
        csi_threshold_K: float = 240.0,
        val_sample_steps: int = 18,
        val_sample_max_batches: int = 4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["diffusion", "stvae"])

        self.diffusion = diffusion
        self.stvae = stvae
        for p in self.stvae.parameters():
            p.requires_grad_(False)
        self.stvae.eval()

        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = tuple(betas)
        self.cosine_t_max = cosine_t_max
        self.cosine_eta_min = cosine_eta_min
        self.csi_threshold_K = csi_threshold_K
        self.val_sample_steps = int(val_sample_steps)
        self.val_sample_max_batches = int(val_sample_max_batches)

        self.ema: Optional[EMAState] = None
        self._ema_decay = ema_decay
        self._ema_enable = ema_enable

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode_seq(self, x: torch.Tensor) -> torch.Tensor:
        """STVAE 编码 ``[B, C=4, T, H, W] -> [B, C_z, T, H', W']``，取 ``mu`` 作确定性编码。"""
        was_training = self.stvae.training
        self.stvae.eval()
        mu, _ = self.stvae.encode(x)
        if was_training:
            self.stvae.train()
        return mu

    @torch.no_grad()
    def decode_seq(self, z: torch.Tensor) -> torch.Tensor:
        was_training = self.stvae.training
        self.stvae.eval()
        out = self.stvae.decode(z)
        if was_training:
            self.stvae.train()
        return out

    # ------------------------------------------------------------------ optimizer / EMA
    def _sync_ema_device(self) -> None:
        """续训时 ``ema_shadow`` 常在 CPU，模型已在 GPU；统一设备避免 update 报错。"""
        if self.ema is None:
            return
        device = next(self.diffusion.denoiser.parameters()).device
        for k, v in list(self.ema.shadow.items()):
            if v.device != device:
                self.ema.shadow[k] = v.to(device=device)

    def on_fit_start(self) -> None:
        if self._ema_enable and self.ema is None:
            self.ema = EMAState(self.diffusion.denoiser, decay=self._ema_decay)
        self._sync_ema_device()

    def configure_optimizers(self):
        opt = AdamW(
            self.diffusion.denoiser.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        if self.cosine_t_max is None:
            return opt
        sch = CosineAnnealingLR(opt, T_max=int(self.cosine_t_max), eta_min=float(self.cosine_eta_min))
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    # ------------------------------------------------------------------ training
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        past = batch["past"]
        future = batch["future"]
        B = past.size(0)

        z_past = self.encode_seq(past)
        z_future = self.encode_seq(future)
        cond = build_cond_from_past(z_past, t_future=z_future.size(2))

        loss, logs = self.diffusion.compute_loss(z_future, cond)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=B)
        self.log_dict(logs, prog_bar=False, on_step=True, on_epoch=True, batch_size=B)
        return loss

    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.ema is not None:
            self._sync_ema_device()
            self.ema.update(self.diffusion.denoiser)

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def _sample_future(self, z_past: torch.Tensor, t_future: int, num_steps: int) -> torch.Tensor:
        B, C_z, T_p, H, W = z_past.shape
        shape = (B, C_z, t_future, H, W)
        cond = build_cond_from_past(z_past, t_future=t_future)
        return self.diffusion.heun_sample(shape, cond=cond, num_steps=num_steps)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        past = batch["past"]
        future = batch["future"]
        B = past.size(0)

        z_past = self.encode_seq(past)
        z_future = self.encode_seq(future)
        cond = build_cond_from_past(z_past, t_future=z_future.size(2))
        loss, _ = self.diffusion.compute_loss(z_future, cond)
        self.log("val/edm_loss", loss, prog_bar=True, on_epoch=True, batch_size=B)

        # 仅前 N 个 batch 跑采样评估，避免每 epoch 太慢
        if batch_idx >= self.val_sample_max_batches:
            return

        # 用 EMA 权重采样
        if self.ema is not None:
            self._sync_ema_device()
        backup = self.ema.copy_to(self.diffusion.denoiser) if self.ema is not None else None
        try:
            z_pred = self._sample_future(z_past, t_future=z_future.size(2), num_steps=self.val_sample_steps)
            future_pred = self.decode_seq(z_pred)
        finally:
            if backup is not None and self.ema is not None:
                self.ema.restore(self.diffusion.denoiser, backup)

        # B13 @240K CSI（开尔文域）
        pred_k = norm_to_kelvin_np(future_pred[:, 3], "B13")
        true_k = norm_to_kelvin_np(future[:, 3], "B13")
        m = csi_at_threshold_k(pred_k, true_k, self.csi_threshold_K)
        self.log("val/csi_b13_240K", float(m["CSI"]), prog_bar=True, on_epoch=True, batch_size=B)
        self.log("val/pod_b13_240K", float(m["POD"]), on_epoch=True, batch_size=B)
        self.log("val/far_b13_240K", float(m["FAR"]), on_epoch=True, batch_size=B)
        mae_K = float(((pred_k - true_k) ** 2).mean() ** 0.5)
        self.log("val/rmse_b13_K", mae_K, on_epoch=True, batch_size=B)

    # ------------------------------------------------------------------ checkpoint EMA 持久化
    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema is not None:
            checkpoint["ema_shadow"] = {k: v.detach().cpu() for k, v in self.ema.shadow.items()}

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        shadow = checkpoint.get("ema_shadow")
        if shadow and self._ema_enable:
            if self.ema is None:
                self.ema = EMAState(self.diffusion.denoiser, decay=self._ema_decay)
            for k, v in shadow.items():
                if k in self.ema.shadow:
                    # 先落到 CPU；``on_fit_start`` 里会 ``_sync_ema_device`` 到 GPU
                    self.ema.shadow[k] = v.detach().cpu()
