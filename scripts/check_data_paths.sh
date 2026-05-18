#!/usr/bin/env bash
# 启动训练前检查 XN_TRAIN_DIR / XN_VAL_DIR 是否可读且含 .npz
set -euo pipefail

TRAIN="${XN_TRAIN_DIR:-/share/home/sera_hujun/train_data_v7_unbiased_501}"
VAL="${XN_VAL_DIR:-/share/home/sera_hujun/val_data_v7_unbiased_501}"

check_one() {
  local name="$1" dir="$2"
  echo "=== $name: $dir ==="
  if [[ ! -d "$dir" ]]; then
    echo "  [FAIL] 目录不存在"
    return 1
  fi
  local flat nested
  flat=$(find "$dir" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
  nested=$(find "$dir" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
  echo "  顶层 .npz: $flat"
  echo "  递归 .npz: $nested"
  if [[ "$nested" -eq 0 ]]; then
    echo "  [FAIL] 无数据文件"
    return 1
  fi
  echo "  [OK] 示例: $(find "$dir" -name 'seq_18F_*.npz' 2>/dev/null | head -1)"
}

check_one train "$TRAIN"
check_one val "$VAL"
echo "检查通过。可启动 train_stvae.py"
