#!/usr/bin/env python3
"""分析 Stage-A STVAE 训练的 loss / CSI 曲线，并输出 Stage-B 门槛判断。

用法（仓库根目录）::

    # 自动选 outputs/stvae 下最新一次 run
    python scripts/analyze_stvae_run.py

    # 指定 run 目录（含 csv_logs/）
    python scripts/analyze_stvae_run.py --run-dir outputs/stvae/nohup_20260517_120000

    # 列出所有 run
    python scripts/analyze_stvae_run.py --list

产物默认写入 ``reports/training/<run_name>/``：
  - epoch_metrics.csv
  - summary.json
  - loss_curves.png, csi_b13_240K.png 等
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.training_report import (  # noqa: E402
    discover_run_dirs,
    epoch_summary,
    evaluate_stage_a_gate,
    find_metrics_csv,
    load_metrics_csv,
    save_training_plots,
    to_json_safe,
)

DEFAULT_STVAE_ROOT = ROOT / "outputs" / "stvae"
DEFAULT_REPORT_ROOT = ROOT / "reports" / "training"


def _print_epoch_table(epoch_df) -> None:
    cols = ["epoch", "train/loss_epoch", "val/loss", "val/l1", "val/csi_b13_240K"]
    show = [c for c in cols if c in epoch_df.columns]
    print(epoch_df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_gate(gate) -> None:
    print("\n=== Stage-A → Stage-B 验收 ===")
    print(f"  硬门槛 CSI ≥ {gate.csi_threshold:.2f}  : {'通过' if gate.checks.get('csi_reaches_threshold') else '未通过'}")
    print(f"  最佳 epoch {gate.best_epoch}  val/csi={gate.best_val_csi:.4f}  val/loss={gate.best_val_loss:.4f}")
    print(f"  末 epoch           val/csi={gate.final_val_csi:.4f}  val/loss={gate.final_val_loss:.4f}")
    print(f"  综合判定: {'可进入 Stage-B' if gate.passed else '继续训练 / 调参'}")
    print("\n  分项检查:")
    for k, v in gate.checks.items():
        print(f"    [{ '✓' if v else '×' }] {k}")
    if gate.notes:
        print("\n  说明:")
        for n in gate.notes:
            print(f"    - {n}")


def main() -> None:
    p = argparse.ArgumentParser(description="分析 STVAE 训练 metrics.csv")
    p.add_argument("--run-dir", type=Path, default=None, help="Hydra 运行目录")
    p.add_argument(
        "--stvae-root",
        type=Path,
        default=DEFAULT_STVAE_ROOT,
        help="扫描所有 run 的根目录（默认 outputs/stvae）",
    )
    p.add_argument("--list", action="store_true", help="列出可用 run 后退出")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT)
    p.add_argument("--csi-threshold", type=float, default=0.95)
    p.add_argument("--min-epochs", type=int, default=10)
    args = p.parse_args()

    runs = discover_run_dirs(args.stvae_root)
    if args.list:
        if not runs:
            print(f"在 {args.stvae_root} 下未找到含 metrics.csv 的 run")
            return
        for r in runs:
            print(r)
        return

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif runs:
        run_dir = runs[-1]
        print(f"[analyze] 使用最新 run: {run_dir}")
    else:
        raise SystemExit(
            f"未找到训练记录。请指定 --run-dir 或确认 {args.stvae_root} 下有 csv_logs/.../metrics.csv"
        )

    metrics_path = find_metrics_csv(run_dir)
    raw = load_metrics_csv(metrics_path)
    epoch_df = epoch_summary(raw)

    report_sub = args.report_dir / run_dir.name
    report_sub.mkdir(parents=True, exist_ok=True)
    epoch_df.to_csv(report_sub / "epoch_metrics.csv", index=False)

    plots = save_training_plots(epoch_df, report_sub, title=run_dir.name)

    gate = evaluate_stage_a_gate(
        epoch_df,
        csi_threshold=args.csi_threshold,
        min_epochs=args.min_epochs,
    )
    summary = {
        "run_dir": str(run_dir.resolve()),
        "metrics_csv": str(metrics_path.resolve()),
        "n_epochs": int(len(epoch_df)),
        "gate": gate.to_dict(),
        "plots": [str(x) for x in plots],
    }
    (report_sub / "summary.json").write_text(
        json.dumps(to_json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[analyze] metrics: {metrics_path}")
    print(f"[analyze] 报告目录: {report_sub}\n")
    _print_epoch_table(epoch_df)
    _print_gate(gate)


if __name__ == "__main__":
    main()
