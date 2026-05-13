"""H8Dataset / DataModule 单元测试。

使用临时合成数据，**不**依赖真实 H8 文件。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.h8_dataset import H8DataModule, H8Dataset, SeqMeta, load_blacklist
from src.data.transforms import CropTransform, GeometricAugConfig, SequenceGeometricAug


# ---------------------------------------------------------------------------
# 数据 fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_h8_dirs(tmp_path: Path) -> dict[str, Path]:
    """在临时目录构造 train/val/test 各几个合成 .npz。"""
    rng = np.random.default_rng(0)
    dirs = {"train": tmp_path / "train", "val": tmp_path / "val", "test": tmp_path / "test"}
    timestamps = {
        "train": ["20180601_0000", "20180601_0030", "20180712_1200"],
        "val": ["20210601_0000", "20210715_0900"],
        "test": ["20220601_0000", "20220715_0900"],
    }
    for split, ts_list in timestamps.items():
        d = dirs[split]
        d.mkdir(parents=True, exist_ok=True)
        for ts in ts_list:
            arr = rng.uniform(-1.0, 1.0, size=(18, 4, 64, 64)).astype(np.float16)
            # 让 train 第一个样本含有强对流核心（B13 极冷区）
            if ts == "20180601_0000":
                arr[:, 3, 30:34, 30:34] = -0.95  # ~183K
            np.savez_compressed(d / f"seq_18F_{ts}.npz", x=arr)
    return dirs


# ---------------------------------------------------------------------------
# SeqMeta 解析
# ---------------------------------------------------------------------------
def test_seqmeta_parses_filename(tmp_path: Path):
    p = tmp_path / "seq_18F_20210815_2350.npz"
    p.write_bytes(b"")
    meta = SeqMeta.from_path(str(p))
    assert meta.start_timestamp == "20210815_2350"
    assert (meta.year, meta.month, meta.day, meta.hour, meta.minute) == (2021, 8, 15, 23, 50)


def test_seqmeta_rejects_invalid_name(tmp_path: Path):
    p = tmp_path / "no_timestamp_here.npz"
    p.write_bytes(b"")
    with pytest.raises(ValueError):
        SeqMeta.from_path(str(p))


# ---------------------------------------------------------------------------
# Dataset 加载、形状、模式
# ---------------------------------------------------------------------------
def test_dataset_split_mode_shapes(fake_h8_dirs):
    ds = H8Dataset(fake_h8_dirs["train"], past_len=6, future_len=12, mode="split")
    sample = ds[0]
    assert sample["past"].shape == (4, 6, 64, 64)
    assert sample["future"].shape == (4, 12, 64, 64)
    assert isinstance(sample["timestamp"], str)
    assert sample["past"].dtype == torch.float32


def test_dataset_raw_mode_shapes(fake_h8_dirs):
    ds = H8Dataset(fake_h8_dirs["train"], mode="raw")
    sample = ds[0]
    assert sample["x"].shape == (4, 18, 64, 64)


def test_dataset_with_crop_and_aug(fake_h8_dirs):
    crop = CropTransform(48, mode="random")
    aug = SequenceGeometricAug(GeometricAugConfig(), layout="CTHW")
    ds = H8Dataset(fake_h8_dirs["train"], crop=crop, aug=aug)
    s = ds[0]
    assert s["past"].shape == (4, 6, 48, 48)
    assert s["future"].shape == (4, 12, 48, 48)


def test_dataset_blacklist_filters(fake_h8_dirs):
    blacklist = {"20180601_0000"}
    ds = H8Dataset(fake_h8_dirs["train"], blacklist=blacklist)
    assert ds.skipped_blacklist == 1
    assert all(m.start_timestamp != "20180601_0000" for m in ds.metas)


def test_dataset_metas_dataframe(fake_h8_dirs):
    ds = H8Dataset(fake_h8_dirs["train"])
    df = ds.metas_dataframe()
    assert {"path", "start_timestamp", "year", "month"}.issubset(df.columns)
    assert len(df) == len(ds)


# ---------------------------------------------------------------------------
# DataModule 工厂
# ---------------------------------------------------------------------------
def test_datamodule_setup_and_loaders(fake_h8_dirs):
    dm = H8DataModule(
        roots={k: str(v) for k, v in fake_h8_dirs.items()},
        spatial={"crop_size": 48, "eval_crop_size": None, "random_crop_train": True},
        augmentation={"enable": True, "rot90": {"p": 0.0, "choices": (0,)}},
        loader={"batch_size": 1, "eval_batch_size": 1, "num_workers": 0, "drop_last": False},
        sampler={"enable": False},
    )
    dm.setup("fit")
    dm.setup("test")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    batch = next(iter(train_loader))
    assert batch["past"].shape == (1, 4, 6, 48, 48)
    assert batch["future"].shape == (1, 4, 12, 48, 48)

    val_batch = next(iter(val_loader))
    # 评估保留完整 64x64
    assert val_batch["past"].shape == (1, 4, 6, 64, 64)
    test_batch = next(iter(test_loader))
    assert test_batch["future"].shape == (1, 4, 12, 64, 64)


# ---------------------------------------------------------------------------
# 几何增强一致性
# ---------------------------------------------------------------------------
def test_geometric_aug_keeps_shape_and_is_deterministic_with_same_rng():
    import random as _r

    cfg = GeometricAugConfig(enable=True)
    seq = torch.randn(4, 18, 16, 16)

    aug1 = SequenceGeometricAug(cfg, layout="CTHW", rng=_r.Random(123))
    aug2 = SequenceGeometricAug(cfg, layout="CTHW", rng=_r.Random(123))
    out1 = aug1(seq)
    out2 = aug2(seq)
    assert out1.shape == seq.shape
    torch.testing.assert_close(out1, out2)


def test_geometric_aug_disabled_is_identity():
    cfg = GeometricAugConfig(enable=False)
    seq = torch.randn(4, 18, 16, 16)
    out = SequenceGeometricAug(cfg, layout="CTHW")(seq)
    torch.testing.assert_close(out, seq)


def test_load_blacklist_missing_returns_empty(tmp_path: Path):
    assert load_blacklist(tmp_path / "nope.csv") == set()
    assert load_blacklist(None) == set()
