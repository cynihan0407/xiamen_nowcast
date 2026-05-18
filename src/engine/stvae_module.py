"""Lightning 模块：Stage-A 时空 VAE 训练。"""
from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW

from src.data.normalizers import norm_to_kelvin_np
from src.metrics.csi import csi_at_threshold_k
from src.models.vae.losses import vae_total_loss


class STVAELightningModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lr: float = 1e-4,
        weight_decay: float = 1e-6,
        kl_weight: float = 1e-6,
        b13_weight: float = 2.0,
        csi_threshold_K: float = 240.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.b13_weight = b13_weight
        self.csi_threshold_K = csi_threshold_K

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        recon, mu, logvar = self.model(x)
        loss, logs = vae_total_loss(
            recon, x, mu, logvar, kl_weight=self.kl_weight, b13_weight=self.b13_weight
        )
        self.log_dict(logs, prog_bar=False, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    @torch.no_grad()
    def _val_csi_b13(self, recon: torch.Tensor, x: torch.Tensor) -> float:
        """在 B13 @ 240K 上算逐像素 CSI（norm 域转 K 后阈值化）。"""
        pred_k = norm_to_kelvin_np(recon[:, 3], "B13")
        true_k = norm_to_kelvin_np(x[:, 3], "B13")
        m = csi_at_threshold_k(pred_k, true_k, self.csi_threshold_K)
        return float(m["CSI"])

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        x = batch["x"]
        recon, mu, logvar = self.model(x)
        loss, logs = vae_total_loss(
            recon, x, mu, logvar, kl_weight=self.kl_weight, b13_weight=self.b13_weight
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        self.log("val/l1", logs["train/l1"], on_epoch=True, batch_size=x.size(0))
        self.log("val/kl", logs["train/kl"], on_epoch=True, batch_size=x.size(0))
        csi = self._val_csi_b13(recon, x)
        self.log("val/csi_b13_240K", csi, prog_bar=True, on_epoch=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
