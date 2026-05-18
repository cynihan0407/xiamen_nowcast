"""Stage-A STVAE 训练日志解析与验收判断。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Lightning CSVLogger 常见列名
EPOCH_METRICS = (
    "train/loss_epoch",
    "train/l1_epoch",
    "train/l1_weighted_epoch",
    "train/kl_epoch",
    "val/loss",
    "val/l1",
    "val/csi_b13_240K",
)


@dataclass
class StageAGateResult:
    """Stage-A → Stage-B 门槛检查结果。"""

    passed: bool
    best_epoch: int
    best_val_csi: float
    best_val_loss: float
    final_val_csi: float
    final_val_loss: float
    csi_threshold: float
    checks: dict[str, bool]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return to_json_safe(asdict(self))


def to_json_safe(obj: Any) -> Any:
    """将 numpy / pandas 标量转为 JSON 可序列化的 Python 原生类型。"""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, (Path,)):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def find_metrics_csv(run_dir: Path) -> Path:
    """在 Hydra 运行目录下定位 ``metrics.csv``。"""
    run_dir = Path(run_dir)
    candidates = sorted(run_dir.glob("csv_logs/**/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"未找到 metrics.csv: {run_dir}/csv_logs/...")
    return candidates[-1]


def load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"metrics.csv 缺少 epoch 列: {path}")
    return df


def epoch_summary(df: pd.DataFrame) -> pd.DataFrame:
    """按 epoch 聚合：每个指标取该 epoch 内最后一次非空记录。"""
    epochs = sorted(df["epoch"].dropna().unique())
    rows: list[dict[str, Any]] = []
    for ep in epochs:
        sub = df[df["epoch"] == ep]
        row: dict[str, Any] = {"epoch": int(ep)}
        for col in df.columns:
            if col in ("epoch", "step", "lr-AdamW", "lr-AdamW/pg1"):
                continue
            vals = sub[col].dropna()
            if len(vals):
                row[col] = float(vals.iloc[-1])
        rows.append(row)
    out = pd.DataFrame(rows)
    if "val/csi_b13_240K" in out.columns:
        out = out.sort_values("epoch").reset_index(drop=True)
    return out


def _series_at(epoch_df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in epoch_df.columns:
        return np.array([])
    return epoch_df[col].dropna().to_numpy(dtype=float)


def _is_plateau(values: np.ndarray, *, patience: int = 5, rel_eps: float = 0.01) -> bool:
    """最近 ``patience`` 个 epoch 相对改进 < ``rel_eps``。"""
    if len(values) < patience + 1:
        return False
    tail = values[-patience:]
    ref = values[-patience - 1]
    if ref <= 0:
        return bool(np.std(tail) < 1e-6)
    improvements = (ref - tail) / max(abs(ref), 1e-8)
    return bool(np.all(improvements < rel_eps))


def evaluate_stage_a_gate(
    epoch_df: pd.DataFrame,
    *,
    csi_threshold: float = 0.95,
    max_val_loss: Optional[float] = None,
    max_train_val_gap: float = 0.20,
    plateau_patience: int = 5,
    plateau_rel_eps: float = 0.01,
    min_epochs: int = 10,
) -> StageAGateResult:
    """Stage-A 验收：以 ``val/csi_b13_240K`` 为主门槛，辅以 loss 健康检查。"""
    notes: list[str] = []
    checks: dict[str, bool] = {}

    val_csi = _series_at(epoch_df, "val/csi_b13_240K")
    val_loss = _series_at(epoch_df, "val/loss")
    train_loss = _series_at(epoch_df, "train/loss_epoch")

    n_ep = len(val_csi)
    if n_ep == 0:
        return StageAGateResult(
            passed=False,
            best_epoch=-1,
            best_val_csi=float("nan"),
            best_val_loss=float("nan"),
            final_val_csi=float("nan"),
            final_val_loss=float("nan"),
            csi_threshold=csi_threshold,
            checks={"has_val_csi": False},
            notes=["无 val/csi_b13_240K 记录，请确认训练已跑完至少 1 个完整 epoch"],
        )

    best_i = int(np.nanargmax(val_csi))
    best_epoch = int(epoch_df["epoch"].iloc[best_i])
    best_val_csi = float(val_csi[best_i])
    best_val_loss = float(val_loss[best_i]) if len(val_loss) else float("nan")
    final_val_csi = float(val_csi[-1])
    final_val_loss = float(val_loss[-1]) if len(val_loss) else float("nan")

    checks["min_epochs"] = n_ep >= min_epochs
    if not checks["min_epochs"]:
        notes.append(f"训练 epoch 数 {n_ep} < {min_epochs}，建议继续训练")

    checks["csi_reaches_threshold"] = best_val_csi >= csi_threshold
    if not checks["csi_reaches_threshold"]:
        notes.append(
            f"最佳 val/csi_b13_240K={best_val_csi:.4f} < {csi_threshold}（epoch {best_epoch}）"
        )

    if max_val_loss is not None and len(val_loss):
        checks["val_loss_below_cap"] = best_val_loss <= max_val_loss
        if not checks["val_loss_below_cap"]:
            notes.append(f"最佳 val/loss={best_val_loss:.4f} > 上限 {max_val_loss}")
    else:
        checks["val_loss_below_cap"] = True

    if len(val_loss) >= 3:
        checks["val_loss_decreased"] = val_loss[-1] < val_loss[0] * 0.98
        if not checks["val_loss_decreased"]:
            notes.append("val/loss 相对首轮几乎未下降，检查学习率或数据")
    else:
        checks["val_loss_decreased"] = False

    checks["val_loss_plateau"] = _is_plateau(val_loss, patience=plateau_patience, rel_eps=plateau_rel_eps)
    if checks["val_loss_plateau"]:
        notes.append(f"val/loss 近 {plateau_patience} 个 epoch 已平台期（可停止或降 lr 微调）")

    if len(train_loss) and len(val_loss):
        gap = float(train_loss[-1] - val_loss[-1])
        checks["no_severe_overfit"] = gap < max_train_val_gap
        if not checks["no_severe_overfit"]:
            notes.append(f"train/loss - val/loss = {gap:.4f}，过拟合风险偏高")
    else:
        checks["no_severe_overfit"] = True

    # 硬门槛：CSI；软门槛：最少 epoch + loss 有下降（平台期可选）
    passed = bool(
        checks["csi_reaches_threshold"]
        and checks["min_epochs"]
        and checks["val_loss_decreased"]
        and checks["no_severe_overfit"]
        and checks["val_loss_below_cap"]
    )
    checks = {k: bool(v) for k, v in checks.items()}

    return StageAGateResult(
        passed=passed,
        best_epoch=best_epoch,
        best_val_csi=best_val_csi,
        best_val_loss=best_val_loss,
        final_val_csi=final_val_csi,
        final_val_loss=final_val_loss,
        csi_threshold=csi_threshold,
        checks=checks,
        notes=notes,
    )


def discover_run_dirs(root: Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    runs: list[Path] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if list(p.glob("csv_logs/**/metrics.csv")):
            runs.append(p)
    return sorted(runs, key=lambda x: x.stat().st_mtime)


def save_training_plots(epoch_df: pd.DataFrame, out_dir: Path, title: str = "") -> list[Path]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _plot(cols: list[str], ylabel: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ep = epoch_df["epoch"]
        for c in cols:
            if c in epoch_df.columns:
                ax.plot(ep, epoch_df[c], marker="o", label=c, linewidth=1.5)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    _plot(["val/loss", "train/loss_epoch"], "loss", "loss_curves.png")
    if "val/csi_b13_240K" in epoch_df.columns:
        _plot(["val/csi_b13_240K"], "CSI", "csi_b13_240K.png")
    _plot(["val/l1", "train/l1_epoch"], "L1", "l1_curves.png")
    if "train/kl_epoch" in epoch_df.columns:
        _plot(["train/kl_epoch"], "KL", "kl_curve.png")

    return saved
