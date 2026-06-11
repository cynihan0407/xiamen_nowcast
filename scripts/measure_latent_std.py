#!/usr/bin/env python3
"""测量 Stage-A latent 的均值/标准差，用于确定 Stage-B 的 EDM ``sigma_data``。

EDM 的理想设置是 ``sigma_data ≈ latent std``。换了 STVAE 结构（如 num_down=3）
后 latent 尺度会变，必须重测再训 Stage-B。

复用 ``evaluate_nowcast`` 配置（自带 data + stvae 结构），只需提供 STVAE 权重::

    python scripts/measure_latent_std.py \\
        stvae_ckpt_path=outputs/stvae/<run>/checkpoints/last.ckpt \\
        split=test max_batches=30

输出：past / future / all 的 mean、std、|z|max，以及对 sigma_data 的建议值。
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.checkpoints import load_stvae_weights  # noqa: E402


class _RunningStat:
    """流式统计 mean / std / |max|（避免一次性堆显存）。"""

    def __init__(self) -> None:
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.absmax = 0.0

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        self.n += x.numel()
        self.sum += x.sum().item()
        self.sumsq += (x * x).sum().item()
        self.absmax = max(self.absmax, x.abs().max().item())

    @property
    def mean(self) -> float:
        return self.sum / max(self.n, 1)

    @property
    def std(self) -> float:
        var = self.sumsq / max(self.n, 1) - self.mean ** 2
        return float(var ** 0.5) if var > 0 else 0.0


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[latent] CUDA 不可用，回退到 CPU")
        return torch.device("cpu")
    return torch.device(name)


def _get_dataloader(dm, split: str):
    if split == "test":
        dm.setup("test")
        return dm.test_dataloader()
    if split == "val":
        dm.setup("fit")
        return dm.val_dataloader()
    raise ValueError(f"split 必须是 test 或 val，得到 {split!r}")


@torch.no_grad()
@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="evaluate_nowcast")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    stvae_ckpt = OmegaConf.select(cfg, "stvae_ckpt_path", default=None)
    if not stvae_ckpt:
        raise ValueError("必须提供 stvae_ckpt_path=...")

    device = _resolve_device(str(cfg.device))
    max_batches = cfg.max_batches
    max_batches = int(max_batches) if max_batches is not None else 30

    # residual=true 时测残差 (z_future - Persistence) 的尺度，用于残差预报的 sigma_data
    residual = bool(OmegaConf.select(cfg, "residual", default=False))

    dm = instantiate(cfg.data)
    loader = _get_dataloader(dm, str(cfg.split))

    stvae = instantiate(cfg.stvae)
    missing, unexpected = load_stvae_weights(stvae, stvae_ckpt)
    print(f"[latent] STVAE ← {stvae_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[latent] residual={residual}")
    stvae = stvae.to(device).eval()

    st_past, st_future, st_all = _RunningStat(), _RunningStat(), _RunningStat()
    st_resid = _RunningStat()

    n = 0
    for batch in tqdm(loader, desc=f"latent/{cfg.split}"):
        if n >= max_batches:
            break
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        mu_p, _ = stvae.encode(past)
        mu_f, _ = stvae.encode(future)
        for st, mu in ((st_past, mu_p), (st_future, mu_f)):
            st.update(mu)
            st_all.update(mu)
        if residual:
            persist = mu_p[:, :, -1:, :, :].expand(-1, -1, mu_f.size(2), -1, -1)
            st_resid.update(mu_f - persist)
        n += 1

    print(f"\nlatent 统计 ({cfg.split}, {n} batch, {st_all.n} 个元素)")
    print(f"  past   : mean={st_past.mean:.4f}  std={st_past.std:.4f}  |z|max={st_past.absmax:.4f}")
    print(f"  future : mean={st_future.mean:.4f}  std={st_future.std:.4f}  |z|max={st_future.absmax:.4f}")
    print(f"  all    : mean={st_all.mean:.4f}  std={st_all.std:.4f}  |z|max={st_all.absmax:.4f}")
    if residual:
        print(f"  residual(z_future-Persistence): mean={st_resid.mean:.4f}  std={st_resid.std:.4f}  |z|max={st_resid.absmax:.4f}")
        print("\n>>> 残差预报 Stage-B 建议: train.predict_residual=true  diffusion.sigma_data={:.2f}".format(round(st_resid.std, 2)))
    else:
        print("\n>>> Stage-B 建议: 训练时设 diffusion.sigma_data={:.2f}".format(round(st_all.std, 2)))


if __name__ == "__main__":
    main()
