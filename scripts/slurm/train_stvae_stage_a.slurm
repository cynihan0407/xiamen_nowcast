#!/bin/bash
#SBATCH --job-name=stvae_a1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=/share/home/sera_hujun/xiamen_nowcast/outputs/stvae/slurm_%j.out
#SBATCH --error=/share/home/sera_hujun/xiamen_nowcast/outputs/stvae/slurm_%j.err

# Stage-A 时空 VAE 正式训练（SLURM 批作业）
#
# 提交（在 login 节点、项目根目录）：
#   mkdir -p outputs/stvae
#   sbatch scripts/slurm/train_stvae_stage_a.slurm
#
# 查看：
#   squeue -u $USER
#   tail -f outputs/stvae/slurm_<JOBID>.out
#
# 按集群实际情况修改：--partition、--nodelist、路径、conda 环境名。

set -euo pipefail

# ---------- 项目路径（改成你的仓库目录）------------------------------------------
export XN_ROOT=/share/home/sera_hujun/xiamen_nowcast
cd "${XN_ROOT}"

# ---------- Conda（按服务器安装位置调整）--------------------------------------
# 若已在 ~/.bashrc 里 conda init，可只写：source ~/.bashrc && conda activate xn
source "${HOME}/.bashrc" 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate xn
fi

# ---------- 数据路径 ----------------------------------------------------------------
export XN_TRAIN_DIR=/share/home/sera_hujun/train_data_v7_unbiased_501
export XN_VAL_DIR=/share/home/sera_hujun/val_data_v7_unbiased_501
export XN_TEST_DIR=/share/home/sera_hujun/test_data_v7_unbiased_501
export XN_BLACKLIST="${XN_ROOT}/problematic_checkpoints.csv"

# ---------- 运行目录（Hydra 输出 + checkpoint + csv_logs）---------------------------
RUN_DIR="${XN_ROOT}/outputs/stvae/slurm_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"
echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID} run_dir=${RUN_DIR}"
echo "[slurm] $(date -Iseconds) starting train_stvae.py"
nvidia-smi -L || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_stvae.py \
  hydra.run.dir="${RUN_DIR}" \
  train.epochs=80 \
  data.loader.batch_size=4 \
  data.loader.eval_batch_size=4 \
  data.loader.num_workers=8 \
  data.loader.persistent_workers=true \
  data.loader.prefetch_factor=4

echo "[slurm] $(date -Iseconds) finished. checkpoints: ${RUN_DIR}/checkpoints"
