#!/usr/bin/env python3
"""一次性生成 notebooks/02_baselines.ipynb（开发用）。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s: str) -> dict:
    lines = [ln + "\n" for ln in s.strip().split("\n")]
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(s: str) -> dict:
    lines = [ln + "\n" for ln in s.strip().split("\n")]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


cells: list[dict] = []
cells.append(
    md(
        """
# 02 · 临近预报基线（Persistence / PySTEPS* / Conv baseline）

**目的**：在与扩散模型相同的数据与划分上，建立可复现的弱基线与中等基线，输出 MAE 与 B13@240K CSI，供论文对比。

**说明**：
* 主评分使用 **B13**；PySTEPS 外推仅在 **B13 场** 上运行（亮温非降水率，光流外推在形态上仍具参照意义）。
* 若未安装 ``pysteps``，第 4 节会跳过并提示安装。
"""
    )
)

cells.append(
    md(
        """
## 步骤 0：环境检查

1. ``conda activate xn``，且 ``import torch`` 成功。
2. 已设置 ``XN_TRAIN_DIR / XN_VAL_DIR / XN_TEST_DIR``（与 01 审计相同）。
3. 在仓库根目录启动 Jupyter；下方 cell 会把仓库根加入 ``sys.path``。
"""
    )
)

cells.append(
    code(
        r"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = Path.cwd().resolve()
if not (REPO_ROOT / "src").is_dir():
    REPO_ROOT = Path.cwd().resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.data.h8_dataset import H8Dataset, load_blacklist
from src.data.normalizers import norm_to_kelvin, B13_INDEX
from src.metrics.csi import csi_at_threshold_k
from src.models.baselines.convlstm_nowcast import ConcatConvNowcast

print("REPO_ROOT =", REPO_ROOT)
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
"""
    )
)

cells.append(
    md(
        """
## 步骤 1：路径与数据集
"""
    )
)

cells.append(
    code(
        r"""
TRAIN_DIR = os.environ.get("XN_TRAIN_DIR", "/share/home/sera_hujun/train_data_v7_unbiased_501")
VAL_DIR = os.environ.get("XN_VAL_DIR", "/share/home/sera_hujun/val_data_v7_unbiased_501")
TEST_DIR = os.environ.get("XN_TEST_DIR", "/share/home/sera_hujun/test_data_v7_unbiased_501")
BLACKLIST = os.environ.get("XN_BLACKLIST", str(REPO_ROOT / "problematic_checkpoints.csv"))

for name, p in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
    print(name, p, "exists=", Path(p).exists())

bl = load_blacklist(BLACKLIST)
ds_test = H8Dataset(TEST_DIR, mode="split", blacklist=bl)
print("test samples:", len(ds_test))
"""
    )
)

cells.append(
    md(
        """
## 步骤 2：评估函数（B13 MAE + 逐帧 CSI@240K 再平均）
"""
    )
)

cells.append(
    code(
        r"""
THRESH_K = 240.0


def b13_kelvin(x_cthw: torch.Tensor) -> torch.Tensor:
    b13n = x_cthw[B13_INDEX]
    return norm_to_kelvin(b13n, "B13")


def mae_b13_k(pred: torch.Tensor, true: torch.Tensor) -> float:
    pk = b13_kelvin(pred).numpy()
    tk = b13_kelvin(true).numpy()
    return float(np.mean(np.abs(pk - tk)))


def mean_csi_b13_240(pred: torch.Tensor, true: torch.Tensor) -> float:
    pk = b13_kelvin(pred).numpy()
    tk = b13_kelvin(true).numpy()
    T = pk.shape[0]
    csis = [csi_at_threshold_k(pk[t], tk[t], THRESH_K)["CSI"] for t in range(T)]
    return float(np.mean(csis))


def evaluate_forecast(pred_future: torch.Tensor, true_future: torch.Tensor) -> dict:
    return {
        "MAE_B13_K": mae_b13_k(pred_future, true_future),
        "mean_CSI_B13_240K": mean_csi_b13_240(pred_future, true_future),
    }
"""
    )
)

cells.append(
    md(
        """
## 步骤 3：基线 A — Persistence（末帧复制 12 步）
"""
    )
)

cells.append(
    code(
        r"""
@torch.no_grad()
def baseline_persistence(past: torch.Tensor) -> torch.Tensor:
    c, tp, h, w = past.shape
    return past[:, -1:, :, :].expand(-1, 12, -1, -1).clone()


N_EVAL = min(200, len(ds_test))
rows = []
for i in range(N_EVAL):
    batch = ds_test[i]
    past, fut = batch["past"], batch["future"]
    pred = baseline_persistence(past)
    m = evaluate_forecast(pred, fut)
    m["i"] = i
    rows.append(m)

df_p = pd.DataFrame(rows)
print(df_p.describe().T)
"""
    )
)

cells.append(
    md(
        """
## 步骤 4：基线 B — PySTEPS（可选）

若导入失败: ``pip install pysteps`` 后重启 kernel。

**注意**：不同 ``pysteps`` 版本的 ``dense_lucaskanade`` 输入维度可能不同；若本节报错，请根据你环境内的 API 调整 ``R`` 的形状，或暂时跳过本节。
"""
    )
)

cells.append(
    code(
        r"""
PYSTEPS_OK = False
try:
    from pysteps.motion.lucaskanade import dense_lucaskanade
    from pysteps.extrapolation.semilagrangian import extrapolate

    PYSTEPS_OK = True
    print("pysteps: OK")
except Exception as e:
    print("pysteps: SKIP ->", e)


@torch.no_grad()
def baseline_pysteps_b13(past: torch.Tensor, n_lead: int = 12) -> torch.Tensor:
    c, tp, h, w = past.shape
    out = past[:, -1:, :, :].expand(-1, n_lead, -1, -1).clone()
    if not PYSTEPS_OK:
        return out
    b13n = past[B13_INDEX].numpy().astype(np.float32)
    R = (-b13n).copy()
    # 常见 API: 输入 (T,ny,nx) 或 (1,T,ny,nx)；若报错请查阅本地 pysteps 文档改此处
    V = dense_lucaskanade(R[-2:])
    last = R[-1]
    seq = extrapolate(last, V, n_lead)
    out[B13_INDEX] = torch.from_numpy((-seq).astype(np.float32))
    return out


if PYSTEPS_OK:
    rows2 = []
    for i in range(N_EVAL):
        batch = ds_test[i]
        past, fut = batch["past"], batch["future"]
        pred = baseline_pysteps_b13(past)
        m = evaluate_forecast(pred, fut)
        m["i"] = i
        rows2.append(m)
    df_s = pd.DataFrame(rows2)
    print(df_s.describe().T)
else:
    print("跳过 PySTEPS 评估")
"""
    )
)

cells.append(
    md(
        """
## 步骤 5：基线 C — ConcatConv（子集快速训练）

在训练集随机子集上训练 ``MAX_STEPS`` 步，再在测试子集上评估。
"""
    )
)

cells.append(
    code(
        r"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBSET = 512
MAX_STEPS = 300
LR = 3e-4

ds_tr = H8Dataset(TRAIN_DIR, mode="split", blacklist=bl)
g = torch.Generator().manual_seed(2025)
perm = torch.randperm(len(ds_tr), generator=g)[:SUBSET]
subset_paths = [ds_tr.metas[int(i)].path for i in perm]
ds_small = H8Dataset(TRAIN_DIR, mode="split", blacklist=bl, files=subset_paths)

model_c = ConcatConvNowcast().to(DEVICE)
opt = torch.optim.AdamW(model_c.parameters(), lr=LR)
loss_fn = nn.L1Loss()
model_c.train()
step = 0
while step < MAX_STEPS:
    for i in range(len(ds_small)):
        b = ds_small[i]
        p = b["past"].unsqueeze(0).to(DEVICE)
        f = b["future"].unsqueeze(0).to(DEVICE)
        pred = model_c(p)
        loss = loss_fn(pred, f)
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step >= MAX_STEPS:
            break

model_c.eval()
rows3 = []
with torch.no_grad():
    for i in range(N_EVAL):
        b = ds_test[i]
        p = b["past"].unsqueeze(0).to(DEVICE)
        f = b["future"]
        pr = model_c(p).cpu()[0]
        rows3.append(evaluate_forecast(pr, f))
print("ConcatConv mean:", pd.DataFrame(rows3).mean())
"""
    )
)

cells.append(
    md(
        """
## 步骤 6：导出 CSV
"""
    )
)

cells.append(
    code(
        r"""
out_dir = REPO_ROOT / "reports" / "baselines"
out_dir.mkdir(parents=True, exist_ok=True)
df_p.to_csv(out_dir / "persistence_test_subset.csv", index=False)
if PYSTEPS_OK:
    df_s.to_csv(out_dir / "pysteps_b13_test_subset.csv", index=False)
pd.DataFrame(rows3).to_csv(out_dir / "concatconv_test_subset.csv", index=False)
print("saved to", out_dir)
"""
    )
)

cells.append(
    md(
        """
## 下一步

1. Stage-A VAE: ``python scripts/train_stvae.py``（见 README）。
2. VAE 重建 B13@240K CSI 建议 >= 0.95 再进入扩散主干。
"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "cells": cells,
}
out = ROOT / "notebooks" / "02_baselines.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Wrote", out, "cells=", len(cells))
