"""normalizers.py 单元测试。

核心要求：
* 与 ``prod_v7_ultimate.py`` 的归一化等价；
* encode 后 decode 与原值一致（开尔文域 round-trip）。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.normalizers import (
    BAND_ORDER,
    NORM_LIMITS,
    BrightnessTemperatureNormalizer,
    b13_norm_threshold_for_kelvin,
    kelvin_to_norm,
    norm_to_kelvin,
)


# ---------------------------------------------------------------------------
# 与 prod_v7 的金标准对齐
# ---------------------------------------------------------------------------
def _prod_v7_scale(arr: np.ndarray, band: str) -> np.ndarray:
    """复刻 prod_v7_ultimate.py.scale_to_neg1_pos1。"""
    mi, ma = NORM_LIMITS[band]
    norm = (arr - mi) / (ma - mi)
    norm = np.clip(norm, 0.0, 1.0)
    return norm * 2.0 - 1.0


@pytest.mark.parametrize("band", BAND_ORDER)
def test_scale_to_neg1_pos1_matches_prod_v7(band: str):
    rng = np.random.default_rng(42)
    arr = rng.uniform(150.0, 320.0, size=(8, 16, 16)).astype(np.float32)
    expected = _prod_v7_scale(arr, band)
    got = kelvin_to_norm(arr, band)
    np.testing.assert_allclose(got, expected, atol=1e-7)


@pytest.mark.parametrize("band", BAND_ORDER)
def test_norm_to_kelvin_round_trip(band: str):
    mi, ma = NORM_LIMITS[band]
    K = np.linspace(mi + 1, ma - 1, 50, dtype=np.float64)  # 全在物理范围内
    normed = kelvin_to_norm(K, band)
    back = norm_to_kelvin(normed, band)
    np.testing.assert_allclose(back, K, atol=1e-6)


def test_scale_clips_extremes():
    rng = np.random.default_rng(0)
    arr = rng.uniform(50.0, 400.0, size=(4, 4)).astype(np.float32)  # 故意越界
    out = kelvin_to_norm(arr, "B13")
    assert (out >= -1.0 - 1e-6).all() and (out <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# 张量级封装
# ---------------------------------------------------------------------------
def test_normalizer_encode_decode_tensor():
    norm = BrightnessTemperatureNormalizer()
    K = torch.full((2, 4, 6, 8, 8), 230.0)  # [B, C, T, H, W]
    K[:, 0] = 200.0  # B08
    K[:, 1] = 220.0  # B09
    K[:, 2] = 230.0  # B10
    K[:, 3] = 250.0  # B13
    normed = norm.encode(K, channel_dim=1)
    assert normed.shape == K.shape
    assert (normed >= -1.0).all() and (normed <= 1.0).all()
    back = norm.decode(normed, channel_dim=1)
    torch.testing.assert_close(back, K, atol=1e-4, rtol=1e-4)


def test_normalizer_channel_check():
    norm = BrightnessTemperatureNormalizer()
    bad = torch.zeros(2, 3, 6, 8, 8)  # C != 4
    with pytest.raises(ValueError):
        norm.encode(bad, channel_dim=1)


def test_b13_threshold_norm():
    # B13 物理范围 [180, 310]，中点 245 K
    assert abs(b13_norm_threshold_for_kelvin(245.0)) < 1e-6
    # 240 K 应该略小于 0（偏冷端）—— 实际计算 (240-180)/130*2-1 ≈ -0.077
    expected = (240.0 - 180.0) / 130.0 * 2.0 - 1.0
    assert abs(b13_norm_threshold_for_kelvin(240.0) - expected) < 1e-6
