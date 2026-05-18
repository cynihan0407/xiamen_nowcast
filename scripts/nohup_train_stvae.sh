#!/usr/bin/env bash
# 在已分配到 GPU 的节点上后台启动 Stage-A VAE（nohup，不依赖 tmux）
#
# 用法（必须先 ssh/srun 到带 GPU 的计算节点，再执行）：
#   cd /share/home/sera_hujun/xiamen_nowcast
#   bash scripts/nohup_train_stvae.sh
#
# 查看日志：
#   tail -f outputs/stvae/nohup_<时间戳>/train.log
# 停止：
#   kill $(cat outputs/stvae/nohup_<时间戳>/train.pid)

set -euo pipefail

XN_ROOT="${XN_ROOT:-/share/home/sera_hujun/xiamen_nowcast}"
cd "${XN_ROOT}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[nohup] 错误：当前节点无 nvidia-smi，请先 srun/ssh 到 GPU 节点再运行。" >&2
  exit 1
fi

# Conda
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate xn
fi

export XN_ROOT
export XN_TRAIN_DIR="${XN_TRAIN_DIR:-/share/home/sera_hujun/train_data_v7_unbiased_501}"
export XN_VAL_DIR="${XN_VAL_DIR:-/share/home/sera_hujun/val_data_v7_unbiased_501}"
export XN_TEST_DIR="${XN_TEST_DIR:-/share/home/sera_hujun/test_data_v7_unbiased_501}"
export XN_BLACKLIST="${XN_BLACKLIST:-${XN_ROOT}/problematic_checkpoints.csv}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="${XN_ROOT}/outputs/stvae/nohup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

echo "[nohup] host=$(hostname) run_dir=${RUN_DIR}"
nvidia-smi -L || true

nohup python scripts/train_stvae.py \
  hydra.run.dir="${RUN_DIR}" \
  train.epochs=80 \
  data.loader.batch_size=4 \
  data.loader.eval_batch_size=4 \
  data.loader.num_workers=8 \
  data.loader.persistent_workers=true \
  data.loader.prefetch_factor=4 \
  > "${RUN_DIR}/train.log" 2>&1 &

echo $! > "${RUN_DIR}/train.pid"
echo "[nohup] PID=$(cat "${RUN_DIR}/train.pid")"
echo "[nohup] tail -f ${RUN_DIR}/train.log"
echo "[nohup] metrics: ${RUN_DIR}/csv_logs/stvae/version_0/metrics.csv"
