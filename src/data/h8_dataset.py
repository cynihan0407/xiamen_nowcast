"""Himawari-8 序列数据集与 LightningDataModule。

数据约定（与 ``prod_v7_ultimate.py`` 完全对齐）：

* 文件名：``seq_18F_<YYYYMMDD>_<HHMM>.npz``
* 字段：``x``，形状 ``[T=18, C=4, H=501, W=501]``，``float16``，已归一化到 ``[-1, 1]``
* 通道顺序：``B08, B09, B10, B13``
* 时间步：每帧 10 min；前 6 帧为过去（输入条件），后 12 帧为未来（预测目标）

本数据集相对 ``dataset_v11.py`` 的主要升级：

1. **元信息提取**：从文件名解析时间戳（年/月/日/时分），驱动分层加权采样。
2. **黑名单合并**：支持加载 ``problematic_checkpoints.csv``，自动剔除问题样本。
3. **两种返回模式**：``raw`` 返回完整 18 帧（VAE 训练 / 数据审计），
   ``split`` 返回 ``(past, future)`` 两段（扩散主干训练）。
4. **可选裁剪**：训练随机、评估中心；亦可关闭裁剪保留完整 501x501（VAE 阶段推荐）。
5. **几何增强一致性**：past/future 共享同一组随机变换。
6. **DataModule**：与 PyTorch Lightning + Hydra 对接的标准入口。

返回的张量布局采用 ``[C, T, H, W]`` ，与 3D U-Net 主干（``Conv3d`` 默认布局
``[B, C, D, H, W]``）天然兼容。
"""
from __future__ import annotations

import glob
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .transforms import CropTransform, GeometricAugConfig, SequenceGeometricAug, numpy_to_tensor

# ---------------------------------------------------------------------------
# 元信息工具
# ---------------------------------------------------------------------------
_TIMESTAMP_PATTERN = re.compile(r"(\d{8})_(\d{4})")


@dataclass(frozen=True)
class SeqMeta:
    """序列样本的元信息（**仅依赖文件名**，O(1) 提取）。"""

    path: str
    start_timestamp: str       # YYYYMMDD_HHMM
    year: int
    month: int
    day: int
    hour: int
    minute: int

    @classmethod
    def from_path(cls, path: str) -> "SeqMeta":
        name = Path(path).stem
        m = _TIMESTAMP_PATTERN.search(name)
        if not m:
            raise ValueError(f"无法从文件名解析时间戳: {path}")
        date_str, hm_str = m.group(1), m.group(2)
        return cls(
            path=path,
            start_timestamp=f"{date_str}_{hm_str}",
            year=int(date_str[:4]),
            month=int(date_str[4:6]),
            day=int(date_str[6:8]),
            hour=int(hm_str[:2]),
            minute=int(hm_str[2:]),
        )


def discover_npz_files(data_dir: str | os.PathLike, *, recursive: bool = False) -> list[str]:
    """枚举数据目录下的 ``seq_18F_*.npz``（默认仅顶层，可选递归子目录）。"""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    if recursive:
        return sorted(str(p) for p in root.rglob("*.npz"))
    return sorted(glob.glob(str(root / "*.npz")))


def _raise_no_npz_error(data_dir: str) -> None:
    root = Path(data_dir)
    exists = root.is_dir()
    flat = len(glob.glob(str(root / "*.npz"))) if exists else 0
    nested = len(list(root.rglob("*.npz"))) if exists else 0
    raise FileNotFoundError(
        f"在 {data_dir} 中找不到任何 .npz 文件。\n"
        f"  目录存在: {exists}\n"
        f"  顶层 *.npz 数量: {flat}\n"
        f"  递归 **/*.npz 数量: {nested}\n"
        "请检查：\n"
        "  1) 是否 export XN_TRAIN_DIR / XN_VAL_DIR（nohup 不会读 ~/.bashrc 里未 export 的变量）\n"
        "  2) 或命令行覆盖: data.roots.train=/你的/train路径 data.roots.val=/你的/val路径\n"
        "  3) 若 .npz 在子目录中，加 data.glob_recursive=true"
    )


def load_blacklist(path: Optional[str | os.PathLike]) -> set[str]:
    """加载 ``problematic_checkpoints.csv``，返回起始时间戳集合。

    若文件不存在则返回空集合（与 prod_v7 行为一致）。
    """
    if path is None:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    if "timestamp" not in df.columns:
        raise KeyError(f"黑名单 CSV 缺少 'timestamp' 列: {p}")
    return set(df["timestamp"].astype(str).tolist())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
ReturnMode = Literal["split", "raw"]


class H8Dataset(Dataset):
    """单一 split (train/val/test) 的 H8 数据集。"""

    EXPECTED_NDIM: int = 4
    EXPECTED_CHANNELS: int = 4

    def __init__(
        self,
        data_dir: str | os.PathLike,
        *,
        past_len: int = 6,
        future_len: int = 12,
        mode: ReturnMode = "split",
        crop: Optional[CropTransform] = None,
        aug: Optional[SequenceGeometricAug] = None,
        blacklist: Optional[set[str]] = None,
        files: Optional[Sequence[str]] = None,
        glob_recursive: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.data_dir = str(data_dir)
        self.past_len = int(past_len)
        self.future_len = int(future_len)
        self.seq_len = self.past_len + self.future_len
        if self.past_len <= 0 or self.future_len <= 0:
            raise ValueError("past_len 与 future_len 必须为正整数")
        self.mode: ReturnMode = mode
        self.crop = crop
        self.aug = aug
        self.dtype = dtype

        self._blacklist = blacklist or set()

        # 文件列表：要么外部注入（用于已经预扫描过的场景），要么 glob
        if files is not None:
            file_list = list(files)
        else:
            file_list = discover_npz_files(self.data_dir, recursive=glob_recursive)
            if len(file_list) == 0 and not glob_recursive:
                file_list = discover_npz_files(self.data_dir, recursive=True)
        if len(file_list) == 0:
            _raise_no_npz_error(self.data_dir)

        # 先解析元信息再过滤黑名单（保留可观测的过滤量）
        metas: list[SeqMeta] = []
        skipped_blacklist = 0
        for p in file_list:
            try:
                m = SeqMeta.from_path(p)
            except ValueError:
                continue
            if m.start_timestamp in self._blacklist:
                skipped_blacklist += 1
                continue
            metas.append(m)
        self._metas: list[SeqMeta] = metas
        self._skipped_blacklist = skipped_blacklist

    # ------------------------------------------------------------------ public
    def __len__(self) -> int:
        return len(self._metas)

    @property
    def metas(self) -> list[SeqMeta]:
        return self._metas

    @property
    def skipped_blacklist(self) -> int:
        return self._skipped_blacklist

    def metas_dataframe(self) -> pd.DataFrame:
        """便于上层（采样器、审计 notebook）以表格形式访问。"""
        return pd.DataFrame(
            {
                "path": [m.path for m in self._metas],
                "start_timestamp": [m.start_timestamp for m in self._metas],
                "year": [m.year for m in self._metas],
                "month": [m.month for m in self._metas],
                "day": [m.day for m in self._metas],
                "hour": [m.hour for m in self._metas],
                "minute": [m.minute for m in self._metas],
            }
        )

    # ------------------------------------------------------------------ I/O
    def _load_seq(self, path: str) -> np.ndarray:
        """读取并校验 ``[T, C, H, W]`` 序列。"""
        try:
            with np.load(path) as data:
                if "x" not in data:
                    raise KeyError(f"npz 缺少 'x' 字段: {path}")
                seq = data["x"]
        except Exception as e:
            raise RuntimeError(f"读取 .npz 失败 {path}: {e}") from e

        if seq.ndim != self.EXPECTED_NDIM:
            raise ValueError(f"期望 4D，得到 shape={seq.shape} ({path})")
        T, C, H, W = seq.shape
        if T != self.seq_len:
            raise ValueError(f"期望 T={self.seq_len}，得到 T={T} ({path})")
        if C != self.EXPECTED_CHANNELS:
            raise ValueError(f"期望 C={self.EXPECTED_CHANNELS}，得到 C={C} ({path})")
        return seq

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, idx: int) -> dict[str, Any]:
        meta = self._metas[idx]
        seq_np = self._load_seq(meta.path)                     # [T, C, H, W]
        seq = numpy_to_tensor(seq_np, dtype=self.dtype)        # [T, C, H, W]

        # 维度重排到 [C, T, H, W]，与 Conv3d 兼容
        seq = seq.permute(1, 0, 2, 3).contiguous()

        # 1. 裁剪（同一序列共享同一窗口）
        if self.crop is not None:
            seq = self.crop(seq)

        # 2. 几何增强（同一序列共享同一组变换）
        if self.aug is not None:
            seq = self.aug(seq)

        sample: dict[str, Any] = {
            "timestamp": meta.start_timestamp,
            "year": meta.year,
            "month": meta.month,
        }

        if self.mode == "raw":
            sample["x"] = seq                                   # [C, T_total, H, W]
        else:  # split
            sample["past"] = seq[:, : self.past_len, :, :].contiguous()         # [C, T_past, H, W]
            sample["future"] = seq[:, self.past_len :, :, :].contiguous()       # [C, T_future, H, W]
        return sample


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class H8DataModule:
    """Lightning DataModule 风格的工厂。

    本类有意 **不继承** ``pytorch_lightning.LightningDataModule`` ，
    避免在数据审计 / 单元测试场景下强行依赖 Lightning。
    上层训练入口可在外面包一层 ``LightningDataModule`` 适配类即可。
    """

    def __init__(
        self,
        roots: dict[str, str],
        *,
        past_len: int = 6,
        future_len: int = 12,
        n_channels: int = 4,
        # 以下字段仅写在 Hydra YAML 中作说明，不参与 DataModule 逻辑
        seq_len: Optional[int] = None,
        norm_limits: Optional[dict[str, tuple[float, float]]] = None,
        band_order: Optional[Sequence[str]] = None,
        spatial: Optional[dict[str, Any]] = None,
        augmentation: Optional[dict[str, Any]] = None,
        sampler: Optional[dict[str, Any]] = None,
        loader: Optional[dict[str, Any]] = None,
        mode: ReturnMode = "split",
        blacklist_path: Optional[str | os.PathLike] = None,
        train_sampler: Optional[Sampler[int]] = None,
        glob_recursive: bool = False,
        **kwargs: Any,
    ) -> None:
        if seq_len is not None and int(seq_len) != past_len + future_len:
            raise ValueError(
                f"seq_len={seq_len} 与 past_len+future_len={past_len + future_len} 不一致"
            )
        if kwargs:
            # Hydra 可能传入仅用于文档的键，忽略即可
            pass
        self.roots = roots
        self.past_len = past_len
        self.future_len = future_len
        self.n_channels = n_channels
        _ = norm_limits, band_order  # 归一化常量见 src.data.normalizers
        self.spatial = spatial or {}
        self.augmentation = augmentation or {}
        self.sampler_cfg = sampler or {}
        self.loader_cfg = loader or {}
        self.mode: ReturnMode = mode
        self.blacklist_path = blacklist_path
        self.train_sampler = train_sampler
        self.glob_recursive = bool(glob_recursive)

        self._train: Optional[H8Dataset] = None
        self._val: Optional[H8Dataset] = None
        self._test: Optional[H8Dataset] = None
        self._blacklist: Optional[set[str]] = None

    # ------------------------------------------------------------------ setup
    def setup(self, stage: Optional[str] = None) -> None:  # noqa: D401
        if self._blacklist is None:
            self._blacklist = load_blacklist(self.blacklist_path)

        train_aug = self._build_aug(train=True)
        eval_aug = self._build_aug(train=False)
        train_crop = self._build_crop(train=True)
        eval_crop = self._build_crop(train=False)

        if stage in (None, "fit"):
            gr = self.glob_recursive
            self._train = H8Dataset(
                self.roots["train"],
                past_len=self.past_len,
                future_len=self.future_len,
                mode=self.mode,
                crop=train_crop,
                aug=train_aug,
                blacklist=self._blacklist,
                glob_recursive=gr,
            )
            self._val = H8Dataset(
                self.roots["val"],
                past_len=self.past_len,
                future_len=self.future_len,
                mode=self.mode,
                crop=eval_crop,
                aug=eval_aug,
                blacklist=self._blacklist,
                glob_recursive=gr,
            )
        if stage in (None, "test", "predict"):
            self._test = H8Dataset(
                self.roots["test"],
                past_len=self.past_len,
                future_len=self.future_len,
                mode=self.mode,
                crop=eval_crop,
                aug=eval_aug,
                blacklist=self._blacklist,
                glob_recursive=self.glob_recursive,
            )

    # ------------------------------------------------------------------ helpers
    def _build_crop(self, *, train: bool) -> Optional[CropTransform]:
        crop_size = self.spatial.get("crop_size") if train else self.spatial.get("eval_crop_size")
        if not crop_size:
            return None
        if train and self.spatial.get("random_crop_train", True):
            return CropTransform(crop_size, mode="random")
        return CropTransform(crop_size, mode="center")

    def _build_aug(self, *, train: bool) -> Optional[SequenceGeometricAug]:
        if not train:
            return None
        if not self.augmentation.get("enable", False):
            return None
        cfg = GeometricAugConfig(
            enable=True,
            flip_horizontal=bool(self.augmentation.get("flip_horizontal", True)),
            flip_vertical=bool(self.augmentation.get("flip_vertical", True)),
            rot90_p=float(self.augmentation.get("rot90", {}).get("p", 0.5)),
            rot90_choices=tuple(self.augmentation.get("rot90", {}).get("choices", (0, 1, 2, 3))),
        )
        return SequenceGeometricAug(cfg, layout="CTHW")

    # ------------------------------------------------------------------ accessors
    @property
    def train_dataset(self) -> H8Dataset:
        if self._train is None:
            raise RuntimeError("DataModule 未 setup")
        return self._train

    @property
    def val_dataset(self) -> H8Dataset:
        if self._val is None:
            raise RuntimeError("DataModule 未 setup")
        return self._val

    @property
    def test_dataset(self) -> H8Dataset:
        if self._test is None:
            raise RuntimeError("DataModule 未 setup")
        return self._test

    # ------------------------------------------------------------------ loaders
    def _common_loader_kwargs(self, *, eval_loader: bool) -> dict[str, Any]:
        kw = dict(
            num_workers=int(self.loader_cfg.get("num_workers", 4)),
            pin_memory=bool(self.loader_cfg.get("pin_memory", True)),
            persistent_workers=bool(self.loader_cfg.get("persistent_workers", True))
            and int(self.loader_cfg.get("num_workers", 4)) > 0,
            prefetch_factor=int(self.loader_cfg.get("prefetch_factor", 2))
            if int(self.loader_cfg.get("num_workers", 4)) > 0
            else None,
        )
        if kw["prefetch_factor"] is None:
            kw.pop("prefetch_factor")
        kw["batch_size"] = int(
            self.loader_cfg.get("eval_batch_size", self.loader_cfg.get("batch_size", 4))
            if eval_loader
            else self.loader_cfg.get("batch_size", 4)
        )
        return kw

    def train_dataloader(self) -> DataLoader:
        kw = self._common_loader_kwargs(eval_loader=False)
        if self.train_sampler is not None:
            kw["sampler"] = self.train_sampler
            kw["shuffle"] = False
        else:
            kw["shuffle"] = True
        kw["drop_last"] = bool(self.loader_cfg.get("drop_last", True))
        return DataLoader(self.train_dataset, **kw)

    def val_dataloader(self) -> DataLoader:
        kw = self._common_loader_kwargs(eval_loader=True)
        kw["shuffle"] = False
        kw["drop_last"] = False
        return DataLoader(self.val_dataset, **kw)

    def test_dataloader(self) -> DataLoader:
        kw = self._common_loader_kwargs(eval_loader=True)
        kw["shuffle"] = False
        kw["drop_last"] = False
        return DataLoader(self.test_dataset, **kw)
