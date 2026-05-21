#!/usr/bin/env python3
"""分析 Stage-B 扩散训练的 loss / CSI 曲线，找出最佳 epoch + 排名 ckpt。

用法（仓库根目录，``conda activate xn``）::

    # 自动选 outputs/diffusion 下最新一次 run
    python scripts/analyze_diffusion_run.py

    # 指定 run 目录
    python scripts/analyze_diffusion_run.py \
        --run-dir outputs/diffusion/run_20260520_211037

    # 仅列出可用 run
    python scripts/analyze_diffusion_run.py --list

产物默认写入 ``reports/diffusion/<run_name>/``：
  - epoch_metrics.csv         每个 epoch 的所有指标
  - epoch_metrics_brief.csv   仅核心列，方便快速看
  - summary.json              最佳 epoch + 健康检查
  - checkpoints_ranked.csv    按 val_edm_loss 升序排列的 ckpt
  - loss_curves.png, val_csi_b13_240K.png, ... 等图

注意：训练日志中 ``val/csi_b13_240K`` 只在前 N=4 个 val batch 上采样 + 随机 noise，
方差较大；该脚本只用它做趋势参考，**真实预报精度请用 ``evaluate_nowcast.py``** 在
更大样本上评估。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.training_report import (  # noqa: E402
    DIFFUSION_EPOCH_METRICS,
    discover_diffusion_runs,
    epoch_summary,
    evaluate_diffusion_summary,
    find_metrics_csv,
    load_metrics_csv,
    rank_diffusion_checkpoints,
    save_diffusion_plots,
    to_json_safe,
)

DEFAULT_DIFFUSION_ROOT = ROOT / "outputs" / "diffusion"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "diffusion"

BRIEF_COLS = (
    "epoch",
    "train/loss_epoch",
    "val/edm_loss",
    "val/csi_b13_240K",
    "val/pod_b13_240K",
    "val/far_b13_240K",
    "val/rmse_b13_K",
)


def _print_epoch_table(epoch_df: pd.DataFrame, *, every: int) -> None:
    show_cols = [c for c in BRIEF_COLS if c in epoch_df.columns]
    sub = epoch_df[show_cols].copy()
    if every > 1 and len(sub) > 0:
        # 始终保留首、尾，以及 best_loss / best_csi 行（main 里追加打印）
        sub = sub[(sub["epoch"] % every == 0) | (sub["epoch"] == sub["epoch"].iloc[-1])]
    print(sub.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_summary(s) -> None:
    print("\n=== Stage-B 训练摘要 ===")
    print(f"  total epochs                : {s.n_epochs}")
    print(f"  best val/edm_loss           : {s.best_loss_val:.4f}  @ epoch {s.best_loss_epoch}")
    print(f"  best val/csi_b13_240K (训练) : {s.best_csi_val:.4f}  @ epoch {s.best_csi_epoch}")
    print(f"  final epoch ({s.final_epoch:>3d})           : "
          f"train_loss={s.final_train_loss:.4f}  val_loss={s.final_val_loss:.4f}  "
          f"val_csi={s.final_val_csi:.4f}")
    print(f"  val_loss 相对首轮明显下降    : {'是' if s.val_loss_decreased else '否'}")
    print(f"  val_loss 是否进入平台期      : {'是' if s.val_loss_plateau else '否'}")
    if s.notes:
        print("\n  说明 / 提示:")
        for n in s.notes:
            print(f"    - {n}")


def _print_ckpts(rows: list[dict], top: int = 10) -> None:
    if not rows:
        print("\n[ckpt] 未在 checkpoints/ 下找到 edm-*.ckpt（请检查 train.checkpoint.filename）")
        return
    print(f"\n=== Top-{min(top, len(rows))} checkpoints（按 val_edm_loss 升序） ===")
    print(f"  {'rank':>4}  {'epoch':>5}  {'val_edm_loss':>12}  {'size_MB':>8}  path")
    for i, r in enumerate(rows[:top]):
        print(f"  {i + 1:>4}  {r['epoch']:>5}  {r['val_edm_loss']:>12.4f}  {r['size_mb']:>8.2f}  {r['ckpt']}")


def _resolve_ckpt_dir(run_dir: Path) -> Path:
    """优先 ``<run>/checkpoints``，找不到时回退到第一个匹配的子目录。"""
    primary = run_dir / "checkpoints"
    if primary.is_dir():
        return primary
    candidates = list(run_dir.rglob("checkpoints"))
    return candidates[0] if candidates else primary


def main() -> None:
    p = argparse.ArgumentParser(description="分析 Stage-B 扩散训练 metrics.csv 与 checkpoints")
    p.add_argument("--run-dir", type=Path, default=None, help="Hydra 运行目录")
    p.add_argument(
        "--diffusion-root",
        type=Path,
        default=DEFAULT_DIFFUSION_ROOT,
        help="扫描所有 run 的根目录（默认 outputs/diffusion）",
    )
    p.add_argument("--list", action="store_true", help="列出可用 run 后退出")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="终端只打印每隔 N 个 epoch 一行，长 run 推荐 5 / 10",
    )
    p.add_argument("--top-ckpts", type=int, default=10, help="打印多少个最佳 ckpt")
    p.add_argument("--csi-threshold-K", type=float, default=240.0)
    args = p.parse_args()

    runs = discover_diffusion_runs(args.diffusion_root)
    if args.list:
        if not runs:
            print(f"在 {args.diffusion_root} 下未找到含 metrics.csv 的 run")
            return
        for r in runs:
            print(r)
        return

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--run-dir 不存在: {run_dir}")
    elif runs:
        run_dir = runs[-1]
        print(f"[analyze] 使用最新 run: {run_dir}")
    else:
        raise SystemExit(
            f"未找到训练记录。请指定 --run-dir 或确认 {args.diffusion_root} 下有 csv_logs/.../metrics.csv"
        )

    metrics_path = find_metrics_csv(run_dir)
    raw = load_metrics_csv(metrics_path)
    epoch_df = epoch_summary(raw)

    report_sub = args.report_dir / run_dir.name
    report_sub.mkdir(parents=True, exist_ok=True)

    epoch_df.to_csv(report_sub / "epoch_metrics.csv", index=False)
    brief_cols = [c for c in BRIEF_COLS if c in epoch_df.columns]
    epoch_df[brief_cols].to_csv(report_sub / "epoch_metrics_brief.csv", index=False)

    plots = save_diffusion_plots(epoch_df, report_sub, title=run_dir.name)
    summary = evaluate_diffusion_summary(epoch_df, csi_threshold_K=args.csi_threshold_K)

    ckpt_dir = _resolve_ckpt_dir(run_dir)
    ranked = rank_diffusion_checkpoints(ckpt_dir)
    if ranked:
        pd.DataFrame(ranked).to_csv(report_sub / "checkpoints_ranked.csv", index=False)

    last_ckpt = ckpt_dir / "last.ckpt"
    summary_dict = {
        "run_dir": str(run_dir.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
        "ckpt_dir": str(ckpt_dir.resolve()),
        "last_ckpt": str(last_ckpt.resolve()) if last_ckpt.is_file() else None,
        "n_epochs": int(len(epoch_df)),
        "summary": summary.to_dict(),
        "best_ckpt_by_val_edm_loss": ranked[0] if ranked else None,
        "plots": [str(x) for x in plots],
        "tracked_columns": [c for c in DIFFUSION_EPOCH_METRICS if c in epoch_df.columns],
    }
    (report_sub / "summary.json").write_text(
        json.dumps(to_json_safe(summary_dict), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[analyze] metrics : {metrics_path}")
    print(f"[analyze] 报告目录 : {report_sub}")
    print(f"[analyze] ckpt 目录: {ckpt_dir}")

    print("\n=== 逐 epoch（核心列） ===")
    _print_epoch_table(epoch_df, every=int(args.print_every))

    _print_summary(summary)
    _print_ckpts(ranked, top=int(args.top_ckpts))

    print("\n[analyze] 建议下一步：")
    if ranked:
        best = ranked[0]
        print(f"  1) 用最佳 ckpt 跑大样本评估（推荐 val 全量或 ≥50 batch）：")
        print(f"     CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_nowcast.py \\")
        print(f"       stvae_ckpt_path=$STVAE_CKPT \\")
        print(f"       diffusion_ckpt_path={best['ckpt']} \\")
        print(f"       stvae.base_channels=48 stvae.latent_channels=12 \\")
        print(f"       model.in_channels=24 model.out_channels=12 \\")
        print(f"       split=val max_batches=100")
    if last_ckpt.is_file():
        print(f"  2) 也可以用 last.ckpt 看末轮效果：diffusion_ckpt_path={last_ckpt}")
    print(f"  3) 目视检查 12 帧预报形态：python scripts/visualize_diffusion_pred.py ...")


if __name__ == "__main__":
    main()
