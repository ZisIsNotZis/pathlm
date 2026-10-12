#!/usr/bin/env bash
# 2624-composable-followups-13m · 新臂训练驱动（三点里需要训练的部分）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
# 共享卡上另有非本仓负载（两个 llama-server ≈19.9GB），锁只管本仓 GPU 线。
#
# 三臂（步时先测见 measure_step_time.sh / README §6；配方其余字段逐字段同 2623）：
#   mha_mlp5        : MHA + --mlp-mult 5（① 的容量括号下端，6k）
#   gqa_10k         : +GQA(2 KV 头)，10k（③ 的无 carry 点）
#   gqa_carry_10k   : +GQA +carry(delta)，10k（③ 的 carry 点）
set -u
cd "$(dirname "$0")/../../../../.."    # repo root
HERE=axes/M-carrier/M1-carrier-type/experiments/2624-composable-followups-13m
source "$HERE/gpu_lock.sh"

COMMON_BASE="--d 256 --layers 12 --heads 8 --seq-len 256 --dense --topology sequential \
--batch-size 32 --lr 1e-3 --seed 0 --synth-task assoc-mqar --synth-pairs 8 \
--synth-queries 8 --synth-gap 14 --out-root $HERE/train"

echo "== 跑前 nvidia-smi（free,used,util）: $(gpu_free_mib)"
run_arm() {
  local name=$1 steps=$2; shift 2
  echo "=== [$name] $(date -Is) 取锁 ==="
  acquire_lock 3600 || { echo "[$name] 等锁超时（>60min），中止"; exit 9; }
  echo "=== [$name] 训练开始 $(date -Is)（steps=$steps）==="
  # shellcheck disable=SC2086
  python3 train_depth_ar.py "$name" $COMMON_BASE --steps "$steps" "$@" \
      > "$HERE/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ==="
  release_lock
  if [ $rc -ne 0 ]; then
    echo "[$name] 训练失败，见 $HERE/train_${name}.log"; tail -20 "$HERE/train_${name}.log"; exit $rc
  fi
  echo "=== [$name] 臂间窗口 65s $(date -Is) ==="
  sleep 65
}

run_arm mha_mlp5      6000  --mlp-mult 5
run_arm gqa_10k       10000 --mlp-mult 6 --n-kv-heads 2
run_arm gqa_carry_10k 10000 --mlp-mult 6 --n-kv-heads 2 --carry-write delta

echo "== 跑后 nvidia-smi（free,used,util）: $(gpu_free_mib)"
echo "ALL FOLLOWUP ARMS DONE $(date -Is)"
