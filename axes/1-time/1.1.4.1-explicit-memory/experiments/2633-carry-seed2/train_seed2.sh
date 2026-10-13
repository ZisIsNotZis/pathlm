#!/usr/bin/env bash
# 2633-carry-seed2 · 三臂（seed=2）训练驱动。判据线不动；配方逐字段 = 2631 的 BASE，
# 唯一差别 = --seed 2（对照复用证据见 equality_vs_2631.py）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2633-carry-seed2
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

# 12k 步 / bs32 / T256 = 98.3M token（与 2631 的六臂同 token 预算 ⇒ §9 复用可比）。
BASE="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
--steps 12000 --batch-size 32 --lr 1e-3 --exit-mode uniform --train-window 16 --train-anchors 0 \
--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14"

run_arm() {
  local name=$1; shift
  echo "=== [$name] $(date -Is) 取锁 ==="
  acquire
  echo "=== [$name] 训练开始 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
  python3 train_depth_ar.py "$name" $BASE --seed 2 --out-root "$EXP/train" "$@" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ==="
  release
  if [ $rc -ne 0 ]; then echo "[$name] 训练失败，见 $EXP/train_${name}.log"; tail -20 "$EXP/train_${name}.log"; fi
  echo "=== [$name] 臂间窗口 65s ==="
  sleep 65
}

run_arm none_s2
run_arm plain_s2 --carry-write plain
run_arm delta_s2 --carry-write delta

echo "ALL ARMS DONE $(date -Is)"
