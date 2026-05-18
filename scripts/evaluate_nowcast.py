#!/usr/bin/env python3
"""评估 Stage-B 临近预报：扩散采样 + STVAE 解码 + B13 指标。

用法（仓库根目录）::

    python scripts/evaluate_nowcast.py \\
        stvae_ckpt_path=outputs/stvae/nohup_manual/checkpoints/last.ckpt \\
        diffusion_ckpt_path=outputs/diffusion/<run>/checkpoints/last.ckpt

冒烟（只跑 10 个 batch）::

    python scripts/evaluate_nowcast.py \\
        stvae_ckpt_path=... diffusion_ckpt_path=... \\
        max_batches=10 split=val

结果写入 ``reports/eval/<时间>/metrics.json`` 与终端摘要。
"""
from __future__ import annotations

import json
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

from src.engine.diffusion_module import DiffusionLightningModule  # noqa: E402
from src.metrics.nowcast import (  # noqa: E402
    NowcastMetricState,
    finalize_nowcast_metrics,
    persistence_forecast,
    update_nowcast_metrics,
)
from src.models.diffusion.edm import EDMDiffusion  # noqa: E402
from src.utils.checkpoints import load_diffusion_lit, load_stvae_weights  # noqa: E402
from src.utils.training_report import to_json_safe  # noqa: E402


@torch.no_grad()
def predict_future(
    lit: DiffusionLightningModule,
    past: torch.Tensor,
    t_future: int,
    num_steps: int,
) -> torch.Tensor:
    z_past = lit.encode_seq(past)
    z_pred = lit._sample_future(z_past, t_future=t_future, num_steps=num_steps)
    return lit.decode_seq(z_pred)


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[eval] CUDA 不可用，回退到 CPU")
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


@hydra.main(version_base=None, config_path=str(ROOT / "configs"), config_name="evaluate_nowcast")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))

    stvae_ckpt = OmegaConf.select(cfg, "stvae_ckpt_path", default=None)
    diff_ckpt = OmegaConf.select(cfg, "diffusion_ckpt_path", default=None)
    if not stvae_ckpt or not diff_ckpt:
        raise ValueError("必须提供 stvae_ckpt_path=... 与 diffusion_ckpt_path=...")

    device = _resolve_device(str(cfg.device))
    num_steps = int(cfg.eval.inference.num_steps)
    use_ema = bool(cfg.eval.inference.use_ema)
    thresholds = [float(t) for t in cfg.eval.metrics.meteorological.thresholds_K]
    compare_persist = bool(cfg.compare_persistence)

    out_dir = Path(str(cfg.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    dm = instantiate(cfg.data)
    loader = _get_dataloader(dm, str(cfg.split))
    max_batches = cfg.max_batches
    if max_batches is not None:
        max_batches = int(max_batches)

    # 模型
    stvae = instantiate(cfg.stvae)
    missing, unexpected = load_stvae_weights(stvae, stvae_ckpt)
    print(f"[eval] STVAE ← {stvae_ckpt}  missing={len(missing)} unexpected={len(unexpected)}")

    denoiser = instantiate(cfg.model)
    diff_kw = {
        k: v
        for k, v in OmegaConf.to_container(cfg.diffusion, resolve=True).items()
        if k != "_target_"
    }
    diffusion = EDMDiffusion(denoiser=denoiser, **diff_kw)
    lit = DiffusionLightningModule(
        diffusion=diffusion,
        stvae=stvae,
        ema_enable=use_ema,
        val_sample_steps=num_steps,
    )
    load_diffusion_lit(lit, diff_ckpt, use_ema=use_ema, device=device)
    lit = lit.to(device)
    print(f"[eval] Diffusion ← {diff_ckpt}  EMA={use_ema}  Heun steps={num_steps}")

    model_state = NowcastMetricState()
    persist_state = NowcastMetricState() if compare_persist else None

    n_batches = 0
    pbar = tqdm(loader, desc=f"eval/{cfg.split}")
    for batch in pbar:
        if max_batches is not None and n_batches >= max_batches:
            break
        past = batch["past"].to(device)
        future = batch["future"].to(device)
        pred = predict_future(lit, past, t_future=future.size(2), num_steps=num_steps)
        update_nowcast_metrics(model_state, pred, future, thresholds)
        if persist_state is not None:
            pred_p = persistence_forecast(past, future.size(2))
            update_nowcast_metrics(persist_state, pred_p, future, thresholds)
        n_batches += 1

    results: dict[str, object] = {
        "split": str(cfg.split),
        "n_batches": n_batches,
        "stvae_ckpt_path": str(Path(stvae_ckpt).resolve()),
        "diffusion_ckpt_path": str(Path(diff_ckpt).resolve()),
        "num_steps": num_steps,
        "use_ema": use_ema,
        "model": finalize_nowcast_metrics(model_state),
    }
    if persist_state is not None:
        results["persistence"] = finalize_nowcast_metrics(persist_state)

    # 打印摘要
    m = results["model"]
    assert isinstance(m, dict)
    print("\n=== 扩散模型 ===")
    for k in sorted(m.keys()):
        if "CSI" in k or "MAE" in k or "RMSE" in k:
            print(f"  {k}: {m[k]:.4f}")
    if persist_state is not None:
        p = results["persistence"]
        assert isinstance(p, dict)
        print("\n=== Persistence 对照 ===")
        for k in sorted(p.keys()):
            if "CSI" in k or "MAE" in k:
                print(f"  {k}: {p[k]:.4f}")

    if bool(cfg.save_json):
        json_path = out_dir / "metrics.json"
        json_path.write_text(
            json.dumps(to_json_safe(results), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n[eval] 已保存: {json_path}")


if __name__ == "__main__":
    main()
