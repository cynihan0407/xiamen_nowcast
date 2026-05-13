"""分层加权采样器。

设计目标：
* 在 ``prod_v7_ultimate.py`` 的"晴空 10% 抽样"基础上，进一步**抑制弱对流主导损失曲面**；
* 按 ``(年月份 × 对流强度)`` 分桶，强对流桶相对采样概率显著提高；
* 一次扫描一遍数据集（每个 ``.npz`` 只读 ``B13`` 通道、计算 1 个标量），
  结果可缓存到 parquet，下次直接复用；
* 采样器采用 PyTorch 标准 ``WeightedRandomSampler`` 接口，无侵入。

对流强度分桶（B13 最低值 norm 域 ``[-1, 1]``）：
    deep    : norm_min < -0.6923  (~< 200 K)
    strong  : norm_min < -0.3846  (~< 220 K)
    active  : norm_min < -0.0769  (~< 240 K)
    weak    : 其余                 (~>= 240 K，含晴空)

强度判定使用 ``min`` 而非 ``mean``：与原始数据生产逻辑保持一致，
也是文献中（如 PreDiff、CasCast）对"是否含强对流核心"最直接的判别量。
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from .h8_dataset import H8Dataset, SeqMeta
from .normalizers import B13_INDEX

INTENSITY_LEVELS: tuple[str, ...] = ("deep", "strong", "active", "weak")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class IntensityBin:
    """对流强度分桶定义：B13 (norm 域) 的最低值是否小于阈值。"""

    name: str
    b13_norm_min_lt: float  # min(B13_norm) < 此阈值则属于此桶（自上而下生效）


@dataclass
class SamplerConfig:
    enable: bool = True
    intensity_bins: Sequence[IntensityBin] = (
        IntensityBin("deep", -0.6923),    # < 200 K
        IntensityBin("strong", -0.3846),  # < 220 K
        IntensityBin("active", -0.0769),  # < 240 K
        IntensityBin("weak", 1.0),        # 兜底，永远 True
    )
    bin_weights: dict[str, float] = None  # type: ignore[assignment]
    cache_scan: bool = True
    cache_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.bin_weights is None:
            self.bin_weights = {"deep": 5.0, "strong": 3.0, "active": 2.0, "weak": 1.0}
        for b in self.intensity_bins:
            if b.name not in self.bin_weights:
                raise KeyError(f"bin_weights 缺少桶 '{b.name}' 的权重")


# ---------------------------------------------------------------------------
# 扫描与分桶
# ---------------------------------------------------------------------------
def _b13_norm_min_for_path(path: str) -> Optional[float]:
    """读取一个 .npz，计算 B13 通道的全局最低值（norm 域）。"""
    try:
        with np.load(path) as data:
            arr = data["x"]                                     # [T, C, H, W]
            b13 = arr[:, B13_INDEX, :, :]                       # [T, H, W]
            return float(np.min(b13))
    except Exception:  # noqa: BLE001
        return None


def assign_intensity_bin(b13_norm_min: float, bins: Sequence[IntensityBin]) -> str:
    """按 ``b13_norm_min`` 自上而下找到第一个匹配的桶名。"""
    for b in bins:
        if b13_norm_min < b.b13_norm_min_lt:
            return b.name
    return bins[-1].name  # 理论上不会触达（兜底桶 = 1.0）


def scan_dataset(
    metas: Sequence[SeqMeta],
    bins: Sequence[IntensityBin],
    *,
    show_progress: bool = True,
    n_jobs: int = 0,
) -> pd.DataFrame:
    """扫描数据集，为每个样本计算 B13_min 并打分桶标签。

    返回 DataFrame 列：
        path, start_timestamp, year, month, b13_norm_min, intensity_bin
    """
    if n_jobs and n_jobs > 1:
        # 并行扫描（多进程）
        try:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            paths = [m.path for m in metas]
            results: list[Optional[float]] = [None] * len(paths)
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                fut_to_idx = {ex.submit(_b13_norm_min_for_path, p): i for i, p in enumerate(paths)}
                it: Iterable = as_completed(fut_to_idx)
                if show_progress:
                    it = tqdm(it, total=len(paths), desc="scanning B13_min")
                for fut in it:
                    idx = fut_to_idx[fut]
                    results[idx] = fut.result()
            mins = results
        except Exception:  # noqa: BLE001
            mins = None
    else:
        mins = None

    if mins is None:
        mins = []
        it: Iterable = metas
        if show_progress:
            it = tqdm(metas, desc="scanning B13_min")
        for m in it:
            mins.append(_b13_norm_min_for_path(m.path))

    rows = []
    for m, v in zip(metas, mins):
        if v is None:
            # 读取失败的样本：归到 weak 桶并标记，下游可选丢弃
            label = "weak"
            v = 1.0
            valid = False
        else:
            label = assign_intensity_bin(v, bins)
            valid = True
        rows.append(
            {
                "path": m.path,
                "start_timestamp": m.start_timestamp,
                "year": m.year,
                "month": m.month,
                "b13_norm_min": float(v),
                "intensity_bin": label,
                "valid": valid,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 加权采样器
# ---------------------------------------------------------------------------
class StratifiedConvectiveSampler(WeightedRandomSampler):
    """按 (月份 × 对流强度) 联合分桶的加权采样器。

    权重计算：
        w_i = bin_weight[intensity] / count_in_(month, intensity_bin)
    这样保证：
    * 强对流桶相对常被抽中；
    * 同一强度下，样本量稀少的月份权重提升（避免季节性失衡）；
    * 总体期望抽样次数与 ``num_samples`` 严格相符。
    """

    def __init__(
        self,
        scan_df: pd.DataFrame,
        cfg: SamplerConfig,
        *,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if not {"month", "intensity_bin", "valid"}.issubset(scan_df.columns):
            raise KeyError("scan_df 必须包含 month / intensity_bin / valid 列")

        df = scan_df.copy()
        df["weight_raw"] = df["intensity_bin"].map(cfg.bin_weights).astype(float)
        df.loc[~df["valid"], "weight_raw"] = 0.0  # 无效样本永不抽中

        # 月份 × 桶 内归一化（按桶内样本数倒数加权）
        joint = df.groupby(["month", "intensity_bin"])["weight_raw"].transform("count").clip(lower=1)
        df["weight"] = df["weight_raw"] / joint
        weights = torch.as_tensor(np.ascontiguousarray(df["weight"].to_numpy(copy=True)), dtype=torch.double)

        if num_samples is None:
            num_samples = int((weights > 0).sum().item())

        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )
        self._scan_df = df.reset_index(drop=True)

    @property
    def scan_df(self) -> pd.DataFrame:
        return self._scan_df


# ---------------------------------------------------------------------------
# 工厂：与 H8DataModule / Hydra 配置对接
# ---------------------------------------------------------------------------
def build_sampler(
    dataset: H8Dataset,
    cfg_dict: dict,
    *,
    num_samples: Optional[int] = None,
    n_jobs: int = 0,
    show_progress: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Optional[StratifiedConvectiveSampler]:
    """从 Hydra 配置 dict 构造采样器；``enable=False`` 时返回 ``None``。

    ``cfg_dict`` 形如 ``configs/data/h8_v7.yaml`` 的 ``sampler`` 段。
    """
    if not cfg_dict.get("enable", False):
        return None

    intensity_bins_raw = cfg_dict.get("intensity_bins", {})
    bins: list[IntensityBin] = []
    if isinstance(intensity_bins_raw, dict):
        # 保持顺序：按用户在 YAML 中给出的顺序
        for name, spec in intensity_bins_raw.items():
            bins.append(IntensityBin(name=str(name), b13_norm_min_lt=float(spec["b13_norm_min_lt"])))
    elif isinstance(intensity_bins_raw, list):
        for item in intensity_bins_raw:
            bins.append(IntensityBin(name=str(item["name"]), b13_norm_min_lt=float(item["b13_norm_min_lt"])))
    else:
        raise TypeError("intensity_bins 必须是 dict 或 list")

    sampler_cfg = SamplerConfig(
        enable=True,
        intensity_bins=tuple(bins),
        bin_weights={str(k): float(v) for k, v in cfg_dict.get("bin_weights", {}).items()},
        cache_scan=bool(cfg_dict.get("cache_scan", True)),
        cache_path=cfg_dict.get("cache_path"),
    )

    cache = Path(sampler_cfg.cache_path) if (sampler_cfg.cache_scan and sampler_cfg.cache_path) else None
    if cache is not None and cache.exists():
        scan_df = pd.read_parquet(cache)
        # 简单一致性校验：路径集合是否对齐
        ds_paths = {m.path for m in dataset.metas}
        if set(scan_df["path"]) != ds_paths:
            scan_df = scan_dataset(dataset.metas, sampler_cfg.intensity_bins, show_progress=show_progress, n_jobs=n_jobs)
            cache.parent.mkdir(parents=True, exist_ok=True)
            scan_df.to_parquet(cache, index=False)
    else:
        scan_df = scan_dataset(dataset.metas, sampler_cfg.intensity_bins, show_progress=show_progress, n_jobs=n_jobs)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            scan_df.to_parquet(cache, index=False)

    # 数据集 metas 顺序 = scan_df 顺序：在 scan_dataset 中已保证
    return StratifiedConvectiveSampler(
        scan_df=scan_df,
        cfg=sampler_cfg,
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )


__all__ = [
    "INTENSITY_LEVELS",
    "IntensityBin",
    "SamplerConfig",
    "StratifiedConvectiveSampler",
    "assign_intensity_bin",
    "build_sampler",
    "scan_dataset",
]
