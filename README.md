# xiamen-nowcast / PG-ST-LFM v1.1

厦门市突发性强对流生消过程的智能临近预报。基于 **3D U-Net + EDM 扩散** 的生成式临近预报模型，仅使用葵花-8 (Himawari-8) 4 通道亮温（B08 / B09 / B10 / B13）。

> **第一阶段目标**：仅凭亮温序列，让模型学到对流系统从初生到消散的完整时空动力学，输出未来 0–2 小时、10 min 步长、2 km 网格的 4 通道预报。

## 1. 数据与切片

- 数据源：Himawari-8 全圆盘亮温的厦门子区域（501×501，~2 km/格）
- 序列：`SEQ_LEN = 18 = 6 (过去 1h) + 12 (未来 2h)`，10 min 步长
- 通道：`B08, B09, B10, B13`（按此顺序）
- 划分：Train 2016–2020，Val 2021，Test 2022（按年份完全隔离）
- 归一化：物理极值线性映射到 `[-1, 1]`，与 `prod_v7_ultimate.py` 完全对齐
  - `B08: [190, 260] K`
  - `B09: [190, 270] K`
  - `B10: [190, 280] K`
  - `B13: [180, 310] K`

## 2. 技术路线（v1.1）

| 模块 | 选型 |
|---|---|
| 工程 | PyTorch Lightning + Hydra + WandB（DeepSpeed ZeRO-2 / FSDP） |
| 表示 | Stage-A 时空 VAE，将 `[18,4,501,501]` 压缩到 `[18,8,64,64]` 量级的 latent |
| 主干 | **3D U-Net (Spatiotemporal U-Net)**：(2+1)D ResBlock + factorized spatial/temporal attention |
| 扩散 | **EDM (Karras 2022)** 主线；DDPM / Flow Matching 作为消融 |
| 条件 | 过去 6 帧 latent 沿通道维 concat + AdaGN 注入 σ / 时刻嵌入 |
| 推理 | Heun-ODE 18/35 步；50 成员集合预报 |
| 物理约束 | **第一阶段不引入**，预留接口在 `src/losses/(physics/)` |

## 3. 目录结构

```
xiamen_nowcast/
├── configs/        # Hydra 多文件配置
├── src/
│   ├── data/       # H8 数据集、加权采样、归一化、统计审计
│   ├── models/     # 时空 VAE、3D U-Net、扩散调度器
│   ├── losses/     # EDM 损失、像素损失（物理损失二期再开）
│   ├── metrics/    # 像素 / 气象阈值 / 概率 / 生命周期评分
│   ├── engine/     # Lightning trainer / 推理 / rollout
│   └── utils/      # 日志、可视化、分布式
├── scripts/        # train_stvae / train_diffusion / evaluate
├── notebooks/      # 数据审计、基线对照、个例分析
└── tests/          # 单元测试
```

## 4. 路线图

| 阶段 | 周期 | 关键产出 |
|---|---|---|
| P0 数据审计 + Baseline | 第 1–3 周 | 数据健康报告；PySTEPS / ConvLSTM 基线表 |
| P1 时空 VAE | 第 4–7 周 | `B13 240K CSI ≥ 0.95` 的重建门槛 |
| P2 3D U-Net + EDM | 第 8–14 周 | 6→12 帧 latent 扩散收敛 |
| P3 物理精调（二期） | 第 15–18 周 | 平流 / 谱 / 多通道一致性 |
| P4 集合化 + 个例分析 | 第 19–22 周 | 50 成员集合，2022 年厦门强对流个例闭环 |
| P5 论文 / 工程化 | 第 23–26 周 | 论文初稿、模型权重发布、推理 API |

## 5. 快速开始

### 5.1 服务器端首次安装

```bash
conda create -n xn python=3.10 -y && conda activate xn

# 与服务器 CUDA 版本对齐（示例：cu121）
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -e ".[dev,baseline]"

# 数据路径环境变量（建议写入 ~/.bashrc）
export XN_TRAIN_DIR=/share/home/sera_hujun/train_data_v7_unbiased_501
export XN_VAL_DIR=/share/home/sera_hujun/val_data_v7_unbiased_501
export XN_TEST_DIR=/share/home/sera_hujun/test_data_v7_unbiased_501
export XN_BLACKLIST=$PWD/problematic_checkpoints.csv

pytest tests/ -q        # 期望 29 passed
```

### 5.2 日常迭代

```bash
# 数据审计（先跑这个，输出新版黑名单）
jupyter lab notebooks/01_data_audit.ipynb

# 训练时空 VAE（Stage-A）
python scripts/train_stvae.py +train=stage_a

# 训练 3D U-Net 扩散（Stage-B）
python scripts/train_diffusion.py +train=stage_b

# 评估
python scripts/evaluate.py +eval=nowcast ckpt_path=...
```

### 5.3 本地↔服务器同步（git 工作流）

首次：在本地仓库根目录执行

```bash
git init -b main && git add . && git commit -m "init"
git remote add origin <your-git-url>
git push -u origin main
```

服务器端：

```bash
git clone <your-git-url> /share/home/sera_hujun/xiamen-nowcast
```

日常一键同步（本地侧）：

```bash
# 在 ~/.zshrc 或 ~/.bashrc 中配置一次
export XN_SERVER_HOST=user@host                                  # 或 ~/.ssh/config 中的别名
export XN_SERVER_PATH=/share/home/sera_hujun/xiamen-nowcast

# 之后每次：
bash scripts/push_to_server.sh "your commit message"             # 本地 commit+push, 服务器 pull
bash scripts/push_to_server.sh --no-pull "msg"                   # 仅 push
bash scripts/push_to_server.sh --dry-run "msg"                   # 只打印不执行
```

### 5.4 模块路径说明

数据相关代码在 **`src/data/`** 包内（例如 `src/data/normalizers.py`、`src/data/h8_dataset.py`），**不在** `src/` 根目录。导入形式为：

```python
from src.data.normalizers import kelvin_to_norm
from src.data.h8_dataset import H8Dataset
```

### 5.5 pytest 收集阶段报错（服务器缺 `src/data`）

若出现 `ModuleNotFoundError: No module named 'src.data'` 或收集阶段 3 个 ERROR，常见原因是 **`.gitignore` 里曾使用 `data/`**，Git 会误忽略 **`src/data/`** 整个源码目录，导致 push 后服务器仓库里没有这些文件。

**处理：**

1. 拉取已修复的 `.gitignore`（规则改为仅忽略仓库根目录的 `/data/`）。
2. 在仓库根目录执行：

```bash
git check-ignore -v src/data/normalizers.py   # 应无输出（表示不再被忽略）
git add -f src/data/
git status                                      # 应能看到若干 .py 被纳入暂存
git commit -m "fix: track src/data Python package (was ignored by data/)"
git push
```

3. 服务器 `git pull` 后确认：`ls src/data` 应列出 `normalizers.py`、`h8_dataset.py` 等。

## 7. 引用

如使用本仓库代码或方法，请引用（占位，待论文发表后补充）：

```bibtex
@misc{xiamen_nowcast_2026,
  title  = {Physics-Guided Spatiotemporal Latent Flow-Matching for Convective Nowcasting over Xiamen},
  author = {...},
  year   = {2026},
}
```
