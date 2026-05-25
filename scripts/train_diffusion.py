#!/usr/bin/env python3
"""Stage-B：训练 latent 扩散（3D U-Net + EDM）。

用法（仓库根目录，已 ``conda activate xn``）::

    python scripts/train_diffusion.py \\
        stvae_ckpt_path=/share/.../outputs/stvae/<run>/checkpoints/last.ckpt

覆盖示例（small 冒烟）::

    python scripts/train_diffusion.py \\
        stvae_ckpt_path=... \\
        train.epochs=2 data.loader.batch_size=2 data.loader.num_workers=2 \\
        +train.trainer.limit_train_batches=50 +train.trainer.limit_val_batches=10

环境变量与 Stage-A 一致：``XN_TRAIN_DIR / XN_VAL_DIR / XN_BLACKLIST``。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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

from src.engine.diffusion_module import DiffusionLightningModule  # noqa: E402
from src.models.diffusion.edm import EDMDiffusion  # noqa: E402
from src.models.vae.stvae import STVAE  # noqa: E402


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _log_data_roots(cfg: DictConfig) -> None:
    roots = OmegaConf.to_container(cfg.data.roots, resolve=True)
    assert isinstance(roots, dict)
    for split, path in roots.items():
        n = len(list(Path(str(path)).glob("*.npz"))) if Path(str(path)).is_dir() else -1
        print(f"[data] {split}: {path}  (顶层 .npz 约 {n} 个)")


def _load_stvae(cfg: DictConfig) -> STVAE:
    """实例化 STVAE 并从 Stage-A checkpoint 加载权重，整体冻结。"""
    stvae: STVAE = instantiate(cfg.stvae)
    ckpt_path = OmegaConf.select(cfg, "stvae_ckpt_path", default=None)
    if not ckpt_path:
        raise ValueError(
            "必须通过 stvae_ckpt_path=... 提供 Stage-A 训练好的 STVAE checkpoint"
        )
    p = Path(str(ckpt_path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"stvae_ckpt_path 不存在: {p}")

    print(f"[stvae] 加载 Stage-A 权重: {p}")
    state = torch.load(p, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    # Lightning 保存的 STVAELightningModule 权重前缀是 "model."
    model_state: dict[str, Any] = {}
    for k, v in sd.items():
        if k.startswith("model."):
            model_state[k[len("model."):]] = v
        elif k.startswith("stvae."):
            model_state[k[len("stvae."):]] = v
        else:
            model_state[k] = v
    missing, unexpected = stvae.load_state_dict(model_state, strict=False)
    if missing:
        print(f"[stvae] 缺失参数 {len(missing)} 个（前 5）:", missing[:5])
    if unexpected:
        print(f"[stvae] 多余参数 {len(unexpected)} 个（前 5）:", unexpected[:5])
    stvae.eval()
    for param in stvae.parameters():
        param.requires_grad_(False)
    return stvae


def _resolve_resume_ckpt(cfg: DictConfig) -> str | None:
    raw = OmegaConf.select(cfg, "ckpt_path", default=None)
    if raw is None:
        return None
    p = Path(str(raw)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"ckpt_path 不存在: {p}")
    return str(p.resolve())


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="train_diffusion")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(int(cfg.seed), workers=True)
    print(OmegaConf.to_yaml(cfg))

    _log_data_roots(cfg)

    # 1) 数据
    dm = instantiate(cfg.data)
    dm.setup("fit")

    # 2) 模型 + STVAE
    stvae = _load_stvae(cfg)
    denoiser = instantiate(cfg.model)
    diffusion = EDMDiffusion(denoiser=denoiser, **{k: v for k, v in OmegaConf.to_container(cfg.diffusion, resolve=True).items() if k != "_target_"})

    cosine_t_max = int(cfg.train.epochs)
    lit = DiffusionLightningModule(
        diffusion=diffusion,
        stvae=stvae,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=tuple(cfg.train.betas),
        ema_enable=bool(cfg.train.ema_enable),
        ema_decay=float(cfg.train.ema_decay),
        cosine_t_max=cosine_t_max,
        cosine_eta_min=1e-6,
        csi_threshold_K=float(cfg.train.csi_threshold_K),
        val_sample_steps=int(cfg.train.val_sample_steps),
        val_sample_max_batches=int(cfg.train.val_sample_max_batches),
        cold_weight_enable=bool(OmegaConf.select(cfg, "train.cold_weight_enable", default=False)),
        cold_weight_threshold_K=float(OmegaConf.select(cfg, "train.cold_weight_threshold_K", default=220.0)),
        cold_weight_factor=float(OmegaConf.select(cfg, "train.cold_weight_factor", default=3.0)),
    )

    # 3) Trainer
    out_dir = HydraConfig.get().runtime.output_dir
    ckpt_cfg = cfg.train.checkpoint
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
    logger = CSVLogger(save_dir=log_dir, name="diffusion")

    trainer_kw = OmegaConf.to_container(cfg.train.trainer, resolve=True)
    assert isinstance(trainer_kw, dict)
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_kw)

    resume_ckpt = _resolve_resume_ckpt(cfg)
    if resume_ckpt:
        print(f"[train] 从 checkpoint 续训: {resume_ckpt}")

    trainer.fit(
        lit,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
        ckpt_path=resume_ckpt,
    )


if __name__ == "__main__":
    main()
