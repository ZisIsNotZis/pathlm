#!/usr/bin/env bash
# 2631-carry-reseed · 六臂训练驱动（三臂 × seed {0,1}）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
# 共享卡上另有非本仓负载（llama-server）与本波其它 GPU 线；锁只管本仓 GPU 线。
#
# 六臂（唯一变量 = carry_rule 与 seed，其余逐字段一致；12k 步 × bs32 × T256 = 98.3M token）：
#   none_s0  / plain_s0  / delta_s0   （seed=0）
#   none_s1  / plain_s1  / delta_s1   （seed=1）
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1))
    if [ $((n % 8)) -eq 0 ]; then echo "  [wait-lock] 已等 $((n*15))s，holder=$hp $(date -Is)"; fi
    [ $n -gt 240 ] && { echo "等锁超时（>60min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null || true
}

# 12k 步：与 2618（ticket 35 的单次观测）同 token 预算，便于跨种子直接对照。
BASE="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
--steps 12000 --batch-size 32 --lr 1e-3 --exit-mode uniform --train-window 16 --train-anchors 0 \
--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14"

run_arm() {
  local name=$1 seed=$2; shift 2
  echo "=== [$name] $(date -Is) 取锁 ==="
  acquire
  echo "=== [$name] 训练开始 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
  python3 train_depth_ar.py "$name" $BASE --seed "$seed" --out-root "$EXP/train" "$@" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ==="
  release
  if [ $rc -ne 0 ]; then echo "[$name] 训练失败，见 $EXP/train_${name}.log"; tail -20 "$EXP/train_${name}.log"; fi
  echo "=== [$name] 臂间窗口 65s ==="
  sleep 65
}

run_arm none_s0  0
run_arm plain_s0 0 --carry-write plain
run_arm delta_s0 0 --carry-write delta
run_arm none_s1  1
run_arm plain_s1 1 --carry-write plain
run_arm delta_s1 1 --carry-write delta

echo "ALL ARMS DONE $(date -Is)"
