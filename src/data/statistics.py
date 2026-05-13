"""数据审计与坏帧检测。

设计目标：
1. 给出**全数据集**的健康画像（按月样本数、各通道分位数、有效像素率），
   作为论文 Section 3 / 立项答辩的"数据集图表"。
2. 自动识别坏帧 / 撕裂帧，输出可追加到 ``problematic_checkpoints.csv`` 的黑名单。
3. **仅依赖单帧统计**，不读完整 18 帧之外的内容，可在审计 notebook 中按需触发。

判定规则（保守，可在 ``QualityRule`` 里调）：
* ``valid_ratio < 0.95``：B13 通道有效像素（非 NaN 且未触发归一化截断）少于 95%。
* ``b13_median_K > 295`` **且** ``b13_q05_K < 200``：典型"撕裂帧"——大部分极暖
  但少量极冷异常并存，常见于云检测错误或文件损坏。
* ``b13_std < 0.5 K`` **且** ``b13_mean_K > 290``：均匀晴空帧但被 prod_v7 误判保留。
  （这一项默认仅记录，不强制拉黑。）
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .h8_dataset import SeqMeta
from .normalizers import BAND_ORDER, NORM_LIMITS, norm_to_kelvin


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class QualityRule:
    valid_ratio_min: float = 0.95
    tearing_b13_median_K_min: float = 295.0
    tearing_b13_q05_K_max: float = 200.0
    flat_clear_b13_std_K_max: float = 0.5
    flat_clear_b13_mean_K_min: float = 290.0


# ---------------------------------------------------------------------------
# 单文件统计
# ---------------------------------------------------------------------------
def _frame_stats(arr_norm: np.ndarray, band_idx: int, band: str) -> dict[str, float]:
    """对单通道 [T, H, W] 计算关键统计量（开尔文）。"""
    band_norm = arr_norm[:, band_idx, :, :].astype(np.float32)
    # 反归一化为开尔文（不做截断）
    K = norm_to_kelvin(band_norm, band)
    flat = K.reshape(-1)
    # 有效像素：未触达边界（即归一化值不在 ±1 端点附近）
    norm_flat = band_norm.reshape(-1)
    valid_mask = (norm_flat > -0.999) & (norm_flat < 0.999)
    valid_ratio = float(valid_mask.mean())
    return {
        f"{band}_min_K": float(np.min(K)),
        f"{band}_q05_K": float(np.quantile(flat, 0.05)),
        f"{band}_median_K": float(np.median(K)),
        f"{band}_q95_K": float(np.quantile(flat, 0.95)),
        f"{band}_max_K": float(np.max(K)),
        f"{band}_mean_K": float(np.mean(K)),
        f"{band}_std_K": float(np.std(K)),
        f"{band}_valid_ratio": valid_ratio,
    }


def stat_one_file(path: str) -> Optional[dict[str, float]]:
    """计算单个 .npz 序列的全通道统计量。读取失败返回 None。"""
    try:
        with np.load(path) as data:
            arr_norm = data["x"]                                 # [T, C, H, W]
    except Exception:  # noqa: BLE001
        return None
    if arr_norm.ndim != 4 or arr_norm.shape[1] != len(BAND_ORDER):
        return None
    out: dict[str, float] = {}
    for i, band in enumerate(BAND_ORDER):
        out.update(_frame_stats(arr_norm, i, band))
    return out


# ---------------------------------------------------------------------------
# 批量审计
# ---------------------------------------------------------------------------
def audit_dataset(
    metas: Sequence[SeqMeta],
    *,
    show_progress: bool = True,
    n_jobs: int = 0,
) -> pd.DataFrame:
    """对全数据集做单文件级统计，返回 DataFrame。"""
    rows: list[dict[str, float | str | int]] = []

    if n_jobs and n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        paths = [m.path for m in metas]
        results: list[Optional[dict[str, float]]] = [None] * len(paths)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            fut_to_idx = {ex.submit(stat_one_file, p): i for i, p in enumerate(paths)}
            it: Iterable = as_completed(fut_to_idx)
            if show_progress:
                it = tqdm(it, total=len(paths), desc="auditing")
            for fut in it:
                idx = fut_to_idx[fut]
                results[idx] = fut.result()
        stats_iter = zip(metas, results)
    else:
        it: Iterable = metas
        if show_progress:
            it = tqdm(metas, desc="auditing")
        stats_iter = ((m, stat_one_file(m.path)) for m in it)

    for m, stat in stats_iter:
        row: dict[str, float | str | int] = {
            "path": m.path,
            "start_timestamp": m.start_timestamp,
            "year": m.year,
            "month": m.month,
            "hour": m.hour,
            "load_ok": stat is not None,
        }
        if stat is not None:
            row.update(stat)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 黑名单扩充
# ---------------------------------------------------------------------------
def detect_problematic(audit_df: pd.DataFrame, rule: Optional[QualityRule] = None) -> pd.DataFrame:
    """根据规则标记问题样本，返回包含 ``reason`` 列的子表（仅含问题项）。"""
    if rule is None:
        rule = QualityRule()
    df = audit_df.copy()
    reasons: list[str] = []
    flags: list[bool] = []

    for _, r in df.iterrows():
        if not r.get("load_ok", True):
            flags.append(True)
            reasons.append("load_failed")
            continue

        bad: list[str] = []

        # 规则 1：B13 有效像素率过低
        if r.get("B13_valid_ratio", 1.0) < rule.valid_ratio_min:
            bad.append(f"low_valid_ratio={r['B13_valid_ratio']:.3f}")

        # 规则 2：撕裂帧（极暖中混极冷）
        if (
            r.get("B13_median_K", 0.0) > rule.tearing_b13_median_K_min
            and r.get("B13_q05_K", 1e9) < rule.tearing_b13_q05_K_max
        ):
            bad.append(
                f"tearing(median={r['B13_median_K']:.1f}K, q05={r['B13_q05_K']:.1f}K)"
            )

        # 规则 3：异常平坦帧（默认仅记录，不一定要拉黑）
        if (
            r.get("B13_std_K", 1e9) < rule.flat_clear_b13_std_K_max
            and r.get("B13_mean_K", 0.0) > rule.flat_clear_b13_mean_K_min
        ):
            bad.append(
                f"flat_clear(std={r['B13_std_K']:.2f}K, mean={r['B13_mean_K']:.1f}K)"
            )

        flags.append(len(bad) > 0)
        reasons.append("; ".join(bad))

    df["is_problematic"] = flags
    df["reason"] = reasons
    return df.loc[df["is_problematic"]].copy()


def merge_blacklist(
    existing_path: Optional[Path | str],
    new_problems: pd.DataFrame,
    output_path: Path | str,
) -> pd.DataFrame:
    """合并已有黑名单与新检出，写出 CSV 并返回最终表。"""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["timestamp", "reason"]
    new_rows = new_problems[["start_timestamp", "reason"]].rename(
        columns={"start_timestamp": "timestamp"}
    )
    new_rows["reason"] = new_rows["reason"].fillna("")

    if existing_path is not None and Path(existing_path).exists():
        existing = pd.read_csv(existing_path)
        if "timestamp" not in existing.columns:
            raise KeyError(f"已有黑名单缺少 'timestamp' 列: {existing_path}")
        if "reason" not in existing.columns:
            existing["reason"] = ""
        merged = pd.concat([existing[columns], new_rows[columns]], ignore_index=True)
    else:
        merged = new_rows[columns]

    merged = merged.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    return merged


# ---------------------------------------------------------------------------
# 高层报告
# ---------------------------------------------------------------------------
def monthly_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    """每月样本量与关键统计量摘要。"""
    g = audit_df.groupby(["year", "month"], dropna=False)
    summary = g.agg(
        n_samples=("path", "count"),
        n_load_failed=("load_ok", lambda s: int((~s.astype(bool)).sum())),
        b13_min_K_p10=("B13_min_K", lambda x: float(np.nanquantile(x, 0.10))),
        b13_min_K_p50=("B13_min_K", lambda x: float(np.nanquantile(x, 0.50))),
        b13_min_K_p90=("B13_min_K", lambda x: float(np.nanquantile(x, 0.90))),
        b13_valid_ratio_mean=("B13_valid_ratio", "mean"),
    ).reset_index()
    return summary


def channel_distribution_summary(audit_df: pd.DataFrame) -> pd.DataFrame:
    """各通道全数据集分布摘要（开尔文域）。"""
    rows = []
    for band in BAND_ORDER:
        col_min = f"{band}_min_K"
        col_med = f"{band}_median_K"
        col_max = f"{band}_max_K"
        rows.append(
            {
                "band": band,
                "phys_min_K": NORM_LIMITS[band][0],
                "phys_max_K": NORM_LIMITS[band][1],
                "obs_min_K_p10": float(np.nanquantile(audit_df[col_min], 0.10)),
                "obs_min_K_p50": float(np.nanquantile(audit_df[col_min], 0.50)),
                "obs_max_K_p90": float(np.nanquantile(audit_df[col_max], 0.90)),
                "obs_median_K_mean": float(np.nanmean(audit_df[col_med])),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "QualityRule",
    "audit_dataset",
    "channel_distribution_summary",
    "detect_problematic",
    "merge_blacklist",
    "monthly_summary",
    "stat_one_file",
]
