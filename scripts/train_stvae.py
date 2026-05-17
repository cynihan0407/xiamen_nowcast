#!/usr/bin/env python3
"""Stage-A：训练时空 VAE。

用法（在 ``xiamen_nowcast`` 仓库根目录、已 ``conda activate xn``）::

    python scripts/train_stvae.py

默认读取 ``configs/train_stvae.yaml``。覆盖示例::

    python scripts/train_stvae.py train.epochs=10 data.loader.batch_size=1

环境变量 ``XN_TRAIN_DIR`` / ``XN_VAL_DIR`` 等与 ``configs/data/h8_v7.yaml`` 一致。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.stvae_module import STVAELightningModule


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="train_stvae")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)

    dm = instantiate(cfg.data)
    dm.setup("fit")

    model = instantiate(cfg.model)
    lit = STVAELightningModule(
        model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        kl_weight=float(cfg.train.kl_weight),
        b13_weight=float(cfg.train.b13_weight),
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
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_kw)

    trainer.fit(
        lit,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )


if __name__ == "__main__":
    main()
