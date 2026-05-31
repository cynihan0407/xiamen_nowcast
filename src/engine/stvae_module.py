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
        dice_weight: float = 0.0,
        dice_tau: float = 0.02,
        dice_thresholds_K: tuple[float, ...] = (240.0,),
        csi_threshold_K: float = 240.0,
        sharpen_weight: float = 0.0,
        finetune_decoder_only: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.kl_weight = kl_weight
        self.b13_weight = b13_weight
        self.dice_weight = dice_weight
        self.dice_tau = dice_tau
        self.dice_thresholds_K = tuple(dice_thresholds_K)
        self.csi_threshold_K = csi_threshold_K
        self.sharpen_weight = sharpen_weight
        self.finetune_decoder_only = bool(finetune_decoder_only)

        if self.finetune_decoder_only:
            self._freeze_except_decoder_tail()

    def _freeze_except_decoder_tail(self) -> None:
        """冻结全部参数，仅解冻 decoder 最后一个上采样块与 ``out_conv``。

        用于 Path 2「decoder sharpening」微调：encoder 与 latent 分布完全不变，
        Stage-B 扩散模型无需重训，评估时只换 STVAE decoder 权重即可。
        """
        for p in self.model.parameters():
            p.requires_grad_(False)
        trainable: list[str] = []
        # decoder 末层（nn.ModuleList 的最后一个 Sequential）
        if hasattr(self.model, "decoder") and len(self.model.decoder) > 0:
            for n, p in self.model.decoder[-1].named_parameters():
                p.requires_grad_(True)
                trainable.append(f"decoder[-1].{n}")
        # 输出层 out_conv
        if hasattr(self.model, "out_conv"):
            for n, p in self.model.out_conv.named_parameters():
                p.requires_grad_(True)
                trainable.append(f"out_conv.{n}")
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[stvae-finetune] 仅解冻 decoder 末层 + out_conv："
              f"{n_train}/{n_total} 参数可训练（{len(trainable)} 个张量）")

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["x"]
        recon, mu, logvar = self.model(x)
        loss, logs = vae_total_loss(
            recon,
            x,
            mu,
            logvar,
            kl_weight=self.kl_weight,
            b13_weight=self.b13_weight,
            dice_weight=self.dice_weight,
            dice_tau=self.dice_tau,
            dice_thresholds_K=self.dice_thresholds_K,
            sharpen_weight=self.sharpen_weight,
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
            recon,
            x,
            mu,
            logvar,
            kl_weight=self.kl_weight,
            b13_weight=self.b13_weight,
            dice_weight=self.dice_weight,
            dice_tau=self.dice_tau,
            dice_thresholds_K=self.dice_thresholds_K,
            sharpen_weight=self.sharpen_weight,
        )
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, batch_size=x.size(0))
        self.log("val/l1", logs["train/l1"], on_epoch=True, batch_size=x.size(0))
        self.log("val/kl", logs["train/kl"], on_epoch=True, batch_size=x.size(0))
        self.log("val/dice", logs["train/dice"], on_epoch=True, batch_size=x.size(0))
        self.log("val/sharpen", logs["train/sharpen"], on_epoch=True, batch_size=x.size(0))
        csi = self._val_csi_b13(recon, x)
        self.log("val/csi_b13_240K", csi, prog_bar=True, on_epoch=True, batch_size=x.size(0))

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            params = list(self.model.parameters())
        return AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
