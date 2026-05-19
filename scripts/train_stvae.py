#!/usr/bin/env python3
"""Stage-A：训练时空 VAE。

用法（在 ``xiamen_nowcast`` 仓库根目录、已 ``conda activate xn``）::

    python scripts/train_stvae.py

默认读取 ``configs/train_stvae.yaml``。覆盖示例::

    python scripts/train_stvae.py train.epochs=10 data.loader.batch_size=1

从 checkpoint 微调（权重 + 优化器状态恢复，epoch 计数继续）::

    python scripts/train_stvae.py ckpt_path=outputs/stvae/.../checkpoints/last.ckpt \\
        train.epochs=110 train.lr=3e-5 train.b13_weight=5.0 \\
        hydra.run.dir=outputs/stvae/finetune_b13w5

环境变量 ``XN_TRAIN_DIR`` / ``XN_VAL_DIR`` 等与 ``configs/data/h8_v7.yaml`` 一致。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.stvae_module import STVAELightningModule


def _resolve_ckpt_path(cfg: DictConfig) -> str | None:
    """支持 ``ckpt_path=...``（推荐）或已废弃的 ``train.trainer.resume_ckpt``。"""
    raw = OmegaConf.select(cfg, "ckpt_path", default=None)
    if raw is None:
        raw = OmegaConf.select(cfg, "train.trainer.resume_ckpt", default=None)
    if raw is None:
        return None
    path = Path(str(raw)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ckpt_path 不存在: {path}")
    return str(path.resolve())


def _log_data_roots(cfg: DictConfig) -> None:
    roots = OmegaConf.to_container(cfg.data.roots, resolve=True)
    assert isinstance(roots, dict)
    for split, path in roots.items():
        n = len(list(Path(str(path)).glob("*.npz"))) if Path(str(path)).is_dir() else -1
        print(f"[data] {split}: {path}  (顶层 .npz 约 {n} 个)")


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="train_stvae")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    _log_data_roots(cfg)
    dm = instantiate(cfg.data)
    dm.setup("fit")

    model = instantiate(cfg.model)
    lit = STVAELightningModule(
        model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        kl_weight=float(cfg.train.kl_weight),
        b13_weight=float(cfg.train.b13_weight),
        dice_weight=float(OmegaConf.select(cfg, "train.dice_weight", default=0.0)),
        dice_tau=float(OmegaConf.select(cfg, "train.dice_tau", default=0.02)),
        dice_thresholds_K=tuple(
            float(x) for x in OmegaConf.select(cfg, "train.dice_thresholds_K", default=[240.0])
        ),
        csi_threshold_K=float(cfg.train.csi_threshold_K),
    )

    ckpt_cfg = cfg.train.checkpoint
    out_dir = HydraConfig.get().runtime.output_dir
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(out_dir, "checkpoints"),
            monitor=str(ckpt_cfg.monitor),
            mode=str(ckpt_cfg.mode),
            save_top_k=int(ckpt_cfg.save_top_k),
            save_last=bool(ckpt_cfg.save_last),
            filename=str(ckpt_cfg.filename),
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    log_dir = os.path.join(out_dir, "csv_logs")
    logger = CSVLogger(save_dir=log_dir, name="stvae")

    trainer_kw = OmegaConf.to_container(cfg.train.trainer, resolve=True)
    assert isinstance(trainer_kw, dict)
    trainer_kw.pop("resume_ckpt", None)  # 勿传入 Lightning Trainer
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_kw)

    ckpt_path = _resolve_ckpt_path(cfg)
    weights_only = bool(OmegaConf.select(cfg, "ckpt_weights_only", default=False))
    fit_ckpt: str | None = ckpt_path
    if ckpt_path and weights_only:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        lit.load_state_dict(state["state_dict"], strict=True)
        fit_ckpt = None
        print(f"[train] 仅加载模型权重（新优化器 / 新 b13_weight 等）: {ckpt_path}")
    elif ckpt_path:
        print(f"[train] 完整恢复 checkpoint（含优化器与 epoch）: {ckpt_path}")

    trainer.fit(
        lit,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
        ckpt_path=fit_ckpt,
    )


if __name__ == "__main__":
    main()
