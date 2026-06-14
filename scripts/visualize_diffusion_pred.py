#!/usr/bin/env python3
"""目视检查 Stage-B 扩散预报效果：把 12 帧未来 B13 画成 真值 / 预报 / Persistence 网格图。

复用 ``configs/evaluate_nowcast.yaml``，新增 ``viz.*`` 字段（不在 yaml 里默认即可，通过
``+viz.xxx=...`` 覆盖）。

典型用法（仓库根目录，``conda activate xn``）::

    CUDA_VISIBLE_DEVICES=0 python scripts/visualize_diffusion_pred.py \
        stvae_ckpt_path=$STVAE_CKPT \
        diffusion_ckpt_path=/path/to/best.ckpt \
        stvae.base_channels=48 stvae.latent_channels=12 \
        model.in_channels=24 model.out_channels=12 \
        split=val \
        +viz.num_samples=6 \
        +viz.num_steps=18 \
        +viz.fig_dir=reports/diffusion/viz_$(date +%Y%m%d_%H%M%S)

每个样本输出 1 个 PNG：``sample_{i}.png``，3×T 网格（真值 / 预报 / Persistence），
颜色为 B13 开尔文，越冷越白（``Greys_r``，vmin=190 K, vmax=300 K）。同时打印
每个样本的 ``MAE_K`` / ``CSI@240K``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.normalizers import norm_to_kelvin_np  # noqa: E402
from src.engine.diffusion_module import DiffusionLightningModule  # noqa: E402
from src.metrics.csi import csi_at_threshold_k  # noqa: E402
from src.metrics.nowcast import persistence_forecast  # noqa: E402
from src.models.diffusion.edm import EDMDiffusion  # noqa: E402
from src.utils.checkpoints import load_diffusion_lit, load_stvae_weights  # noqa: E402


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[viz] CUDA 不可用，回退到 CPU（采样会很慢）")
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


def _save_grid(
    out_path: Path,
    truth_k: np.ndarray,        # [T, H, W]
    pred_k: np.ndarray,         # [T, H, W]
    persist_k: np.ndarray,      # [T, H, W]
    *,
    vmin: float,
    vmax: float,
    title: str,
    threshold_K: float = 240.0,
) -> None:
    """3×T 网格图：行 = 真值 / 预报 / Persistence；列 = 帧。"""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    T = truth_k.shape[0]
    fig, axes = plt.subplots(
        nrows=3,
        ncols=T,
        figsize=(1.6 * T, 5.2),
        constrained_layout=True,
    )
    if T == 1:
        axes = axes.reshape(3, 1)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "Greys_r"

    rows = [
        ("Truth", truth_k),
        ("Predict", pred_k),
        ("Persistence", persist_k),
    ]
    for r, (label, arr) in enumerate(rows):
        for t in range(T):
            ax = axes[r, t]
            ax.imshow(arr[t], cmap=cmap, norm=norm, interpolation="nearest")
            ax.contour(arr[t] <= threshold_K, levels=[0.5], colors="red", linewidths=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t+{t + 1}", fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=10)

    # 共享 colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.01)
    cbar.set_label("B13 [K]")

    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _per_sample_metrics(pred_k: np.ndarray, true_k: np.ndarray, threshold_K: float) -> dict[str, float]:
    """单样本 T 帧 平均 CSI / 整体 MAE。"""
    T = pred_k.shape[0]
    csis = [
        csi_at_threshold_k(pred_k[t], true_k[t], threshold_K)["CSI"]
        for t in range(T)
    ]
    return {
        "MAE_K": float(np.abs(pred_k - true_k).mean()),
        "RMSE_K": float(((pred_k - true_k) ** 2).mean() ** 0.5),
        f"CSI_{int(threshold_K)}K_mean": float(np.mean(csis)),
        f"CSI_{int(threshold_K)}K_min": float(np.min(csis)),
        f"CSI_{int(threshold_K)}K_max": float(np.max(csis)),
    }


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="evaluate_nowcast")
def main(cfg: DictConfig) -> None:
    seed = int(OmegaConf.select(cfg, "seed", default=2025))
    torch.manual_seed(seed)

    # viz 参数（默认在 cfg 中不存在，通过 +viz.xxx=... 覆盖）
    num_samples = int(OmegaConf.select(cfg, "viz.num_samples", default=4))
    num_steps = int(OmegaConf.select(cfg, "viz.num_steps",
                                     default=OmegaConf.select(cfg, "eval.inference.num_steps", default=18)))
    use_ema = bool(OmegaConf.select(cfg, "viz.use_ema",
                                    default=OmegaConf.select(cfg, "eval.inference.use_ema", default=True)))
    threshold_K = float(OmegaConf.select(cfg, "viz.threshold_K",
                                         default=240.0))
    vmin = float(OmegaConf.select(cfg, "viz.vmin", default=190.0))
    vmax = float(OmegaConf.select(cfg, "viz.vmax", default=300.0))
    sample_stride = int(OmegaConf.select(cfg, "viz.sample_stride", default=1))

    fig_dir_raw = OmegaConf.select(cfg, "viz.fig_dir", default=None)
    fig_dir = Path(str(fig_dir_raw)) if fig_dir_raw else Path(str(cfg.output_dir)) / "viz"
    fig_dir.mkdir(parents=True, exist_ok=True)

    stvae_ckpt = OmegaConf.select(cfg, "stvae_ckpt_path", default=None)
    diff_ckpt = OmegaConf.select(cfg, "diffusion_ckpt_path", default=None)
    if not stvae_ckpt or not diff_ckpt:
        raise ValueError("必须提供 stvae_ckpt_path=... 与 diffusion_ckpt_path=...")

    device = _resolve_device(str(cfg.device))

    # 数据
    dm = instantiate(cfg.data)
    loader = _get_dataloader(dm, str(cfg.split))

    # 模型
    stvae = instantiate(cfg.stvae)
    load_stvae_weights(stvae, stvae_ckpt)
    denoiser = instantiate(cfg.model)
    diff_kw = {k: v for k, v in OmegaConf.to_container(cfg.diffusion, resolve=True).items() if k != "_target_"}
    diffusion = EDMDiffusion(denoiser=denoiser, **diff_kw)

    # 从 ckpt 超参自动读取 predict_residual，确保与训练/评估一致
    _blob = torch.load(diff_ckpt, map_location="cpu", weights_only=False)
    _hp = _blob.get("hyper_parameters", {}) if isinstance(_blob, dict) else {}
    predict_residual = bool(
        _hp.get("predict_residual", OmegaConf.select(cfg, "predict_residual", default=False))
    )
    advect_residual = bool(
        _hp.get("advect_residual", OmegaConf.select(cfg, "advect_residual", default=False))
    )
    del _blob
    print(f"[viz] predict_residual={predict_residual}  advect_residual={advect_residual}（来自 ckpt 超参）")

    lit = DiffusionLightningModule(
        diffusion=diffusion,
        stvae=stvae,
        ema_enable=use_ema,
        val_sample_steps=num_steps,
        predict_residual=predict_residual,
        advect_residual=advect_residual,
        flow_max_disp=int(_hp.get("flow_max_disp", 6)),
        flow_win=int(_hp.get("flow_win", 9)),
        flow_scale=int(_hp.get("flow_scale", 4)),
    )
    load_diffusion_lit(lit, diff_ckpt, use_ema=use_ema, device=device)
    lit = lit.to(device)
    lit.eval()

    print(f"[viz] STVAE     ← {stvae_ckpt}")
    print(f"[viz] Diffusion ← {diff_ckpt}  EMA={use_ema}  Heun steps={num_steps}")
    print(f"[viz] split={cfg.split}  num_samples={num_samples}  sample_stride={sample_stride}")
    print(f"[viz] 输出目录: {fig_dir.resolve()}")

    rows_meta: list[dict[str, object]] = []
    saved = 0
    sample_global_idx = 0

    for batch_idx, batch in enumerate(loader):
        if saved >= num_samples:
            break
        past = batch["past"].to(device)
        future = batch["future"].to(device)

        with torch.no_grad():
            pred = lit.forecast(past, t_future=future.size(2), num_steps=num_steps)
            persist = persistence_forecast(past, future.size(2))

        # [B, C, T, H, W]，取 B13 通道
        true_k_b = norm_to_kelvin_np(future[:, 3], "B13")
        pred_k_b = norm_to_kelvin_np(pred[:, 3], "B13")
        persist_k_b = norm_to_kelvin_np(persist[:, 3], "B13")

        B = true_k_b.shape[0]
        for b in range(B):
            if saved >= num_samples:
                break
            if sample_stride > 1 and (sample_global_idx % sample_stride) != 0:
                sample_global_idx += 1
                continue
            sample_global_idx += 1

            t_k = true_k_b[b]
            p_k = pred_k_b[b]
            pe_k = persist_k_b[b]

            m_pred = _per_sample_metrics(p_k, t_k, threshold_K)
            m_pers = _per_sample_metrics(pe_k, t_k, threshold_K)

            out_path = fig_dir / f"sample_{saved:02d}.png"
            title = (
                f"sample {saved}  (batch {batch_idx}, in-batch {b})\n"
                f"Pred  MAE={m_pred['MAE_K']:.2f}K  CSI@{int(threshold_K)}K={m_pred[f'CSI_{int(threshold_K)}K_mean']:.3f}   |   "
                f"Persist MAE={m_pers['MAE_K']:.2f}K  CSI@{int(threshold_K)}K={m_pers[f'CSI_{int(threshold_K)}K_mean']:.3f}"
            )
            _save_grid(
                out_path,
                t_k, p_k, pe_k,
                vmin=vmin, vmax=vmax,
                title=title,
                threshold_K=threshold_K,
            )

            rows_meta.append(
                {
                    "index": saved,
                    "batch_idx": batch_idx,
                    "in_batch_idx": b,
                    "png": str(out_path),
                    **{f"pred/{k}": v for k, v in m_pred.items()},
                    **{f"persist/{k}": v for k, v in m_pers.items()},
                }
            )
            print(
                f"  [{saved:02d}] PNG={out_path.name}  "
                f"pred_CSI={m_pred[f'CSI_{int(threshold_K)}K_mean']:.3f}  "
                f"persist_CSI={m_pers[f'CSI_{int(threshold_K)}K_mean']:.3f}  "
                f"pred_MAE_K={m_pred['MAE_K']:.2f}  persist_MAE_K={m_pers['MAE_K']:.2f}"
            )
            saved += 1

    # 摘要
    if rows_meta:
        import pandas as pd

        df = pd.DataFrame(rows_meta)
        csv_path = fig_dir / "metrics_per_sample.csv"
        df.to_csv(csv_path, index=False)
        cmean = df[f"pred/CSI_{int(threshold_K)}K_mean"].mean()
        pmean = df[f"persist/CSI_{int(threshold_K)}K_mean"].mean()
        print(f"\n[viz] 保存 {saved} 个 PNG → {fig_dir}")
        print(f"[viz] 平均 pred CSI@{int(threshold_K)}K = {cmean:.4f}   (persist = {pmean:.4f})")
        print(f"[viz] 详细指标: {csv_path}")
    else:
        print("[viz] 未生成任何样本（dataloader 为空？）")


if __name__ == "__main__":
    main()
