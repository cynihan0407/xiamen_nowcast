"""加权采样器单元测试。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.h8_dataset import H8Dataset
from src.data.samplers import (
    IntensityBin,
    SamplerConfig,
    StratifiedConvectiveSampler,
    assign_intensity_bin,
    build_sampler,
    scan_dataset,
)


# ---------------------------------------------------------------------------
# 合成数据：构造 4 个强度桶 × 2 个月份共 8 类 × 50 样本
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_dataset(tmp_path: Path) -> H8Dataset:
    rng = np.random.default_rng(2025)
    d = tmp_path / "train"
    d.mkdir(parents=True)
    # 强度 -> B13 极冷值（norm 域）
    intensity_to_b13 = {
        "deep": -0.85,
        "strong": -0.55,
        "active": -0.20,
        "weak": 0.50,
    }
    months = [6, 7]
    counter = 0
    for month in months:
        for level, b13 in intensity_to_b13.items():
            for _ in range(50):
                counter += 1
                # 每个文件唯一的时间戳
                day = (counter % 28) + 1
                hh = (counter // 28) % 24
                mm = (counter // (28 * 24)) % 6 * 10
                ts = f"2018{month:02d}{day:02d}_{hh:02d}{mm:02d}"
                arr = rng.uniform(-0.05, 0.05, size=(18, 4, 16, 16)).astype(np.float16)
                # 人为注入 B13 极冷区
                arr[5:7, 3, 4:8, 4:8] = b13
                np.savez_compressed(d / f"seq_18F_{ts}.npz", x=arr)
    return H8Dataset(d, mode="raw")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def test_assign_intensity_bin_orders_correctly():
    bins = (
        IntensityBin("deep", -0.6923),
        IntensityBin("strong", -0.3846),
        IntensityBin("active", -0.0769),
        IntensityBin("weak", 1.0),
    )
    assert assign_intensity_bin(-0.85, bins) == "deep"
    assert assign_intensity_bin(-0.50, bins) == "strong"
    assert assign_intensity_bin(-0.20, bins) == "active"
    assert assign_intensity_bin(0.40, bins) == "weak"


# ---------------------------------------------------------------------------
# 扫描 + 加权采样
# ---------------------------------------------------------------------------
def test_scan_dataset_assigns_correct_bins(fake_dataset: H8Dataset):
    bins = (
        IntensityBin("deep", -0.6923),
        IntensityBin("strong", -0.3846),
        IntensityBin("active", -0.0769),
        IntensityBin("weak", 1.0),
    )
    df = scan_dataset(fake_dataset.metas, bins, show_progress=False)
    assert df["valid"].all()
    counts = df["intensity_bin"].value_counts().to_dict()
    # 4 桶 × 2 月份 × 50 样本 = 100 / 桶
    for level in ("deep", "strong", "active", "weak"):
        assert counts.get(level, 0) == 100, f"桶 {level} 计数不符: {counts}"


def test_stratified_sampler_amplifies_strong_bins(fake_dataset: H8Dataset):
    bins = (
        IntensityBin("deep", -0.6923),
        IntensityBin("strong", -0.3846),
        IntensityBin("active", -0.0769),
        IntensityBin("weak", 1.0),
    )
    scan_df = scan_dataset(fake_dataset.metas, bins, show_progress=False)

    cfg = SamplerConfig(
        enable=True,
        intensity_bins=bins,
        bin_weights={"deep": 5.0, "strong": 3.0, "active": 2.0, "weak": 1.0},
    )
    g = torch.Generator().manual_seed(0)
    sampler = StratifiedConvectiveSampler(scan_df, cfg, num_samples=20_000, generator=g)
    indices = list(iter(sampler))
    drawn = scan_df.iloc[indices]
    counts = drawn["intensity_bin"].value_counts().to_dict()

    # 期望相对比例 ≈ 权重比 (因每个桶样本数相等)
    deep, strong, active, weak = counts["deep"], counts["strong"], counts["active"], counts["weak"]
    total = sum((deep, strong, active, weak))
    p_deep, p_strong, p_active, p_weak = deep / total, strong / total, active / total, weak / total
    # 权重和 = 5+3+2+1 = 11 → 期望比例 ≈ 5/11, 3/11, 2/11, 1/11
    assert abs(p_deep - 5 / 11) < 0.03, p_deep
    assert abs(p_strong - 3 / 11) < 0.03, p_strong
    assert abs(p_active - 2 / 11) < 0.03, p_active
    assert abs(p_weak - 1 / 11) < 0.03, p_weak


def test_stratified_sampler_excludes_invalid_rows():
    df = pd.DataFrame(
        {
            "month": [6, 6, 7, 7],
            "intensity_bin": ["deep", "deep", "weak", "weak"],
            "valid": [True, False, True, True],
        }
    )
    cfg = SamplerConfig(
        intensity_bins=(IntensityBin("deep", -0.6923), IntensityBin("weak", 1.0)),
        bin_weights={"deep": 5.0, "weak": 1.0},
    )
    sampler = StratifiedConvectiveSampler(df, cfg, num_samples=10_000)
    indices = list(iter(sampler))
    # 索引 1 是 invalid，理论上权重为 0，绝不应被抽中
    assert 1 not in set(indices)


# ---------------------------------------------------------------------------
# build_sampler 工厂（含缓存）
# ---------------------------------------------------------------------------
def test_build_sampler_with_cache(fake_dataset: H8Dataset, tmp_path: Path):
    cache = tmp_path / "scan.parquet"
    cfg_dict = {
        "enable": True,
        "intensity_bins": {
            "deep": {"b13_norm_min_lt": -0.6923},
            "strong": {"b13_norm_min_lt": -0.3846},
            "active": {"b13_norm_min_lt": -0.0769},
            "weak": {"b13_norm_min_lt": 1.0},
        },
        "bin_weights": {"deep": 5.0, "strong": 3.0, "active": 2.0, "weak": 1.0},
        "cache_scan": True,
        "cache_path": str(cache),
    }
    s1 = build_sampler(fake_dataset, cfg_dict, show_progress=False)
    assert cache.exists()
    s2 = build_sampler(fake_dataset, cfg_dict, show_progress=False)
    assert isinstance(s1, StratifiedConvectiveSampler)
    assert isinstance(s2, StratifiedConvectiveSampler)


def test_build_sampler_disabled(fake_dataset: H8Dataset):
    out = build_sampler(fake_dataset, {"enable": False}, show_progress=False)
    assert out is None
