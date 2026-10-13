#!/usr/bin/env bash
# 2631-spec-three-arms · 训练驱动（按臂取/放 GPU 锁，SCHEMA §20 / §20.1）。
#
# 用法：bash train_arms.sh [arm ...]     默认 = 13m_s0 13m_s1（先做 13M，跑完再谈大档）
#   arm 命名：<scale>_s<seed>，scale ∈ {13m, 53m}
#
# 纪律：**按臂**释放锁、臂间 ≥60s 窗口；不整程独占。共享卡上另有非本仓负载
# （llama-server）与本波其它 GPU 线（L4 的 carry-reseed）。锁只管本仓 GPU 线。
#
# 四臂（唯一变量 = seed；同一 scale 内配方逐字段一致；6000 步 × bs32 × T512 = 98.3M token）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/H-horizon/H3-composition/experiments/2631-spec-three-arms
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
    if [ $((n % 8)) -eq 0 ]; then echo "  [wait-lock] 已等 $((n*15))s holder=$hp $(date -Is)"; fi
    [ $n -gt 80 ] && { echo "等锁超时（>20min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; }

run_arm() {
  local arm=$1 scale=${1%%_*} seed=${1##*_s}
  local cfg="config_${scale}.json"
  echo "=== [$arm] $(date -Is) 取锁 ==="
  acquire
  echo "=== [$arm] 训练开始 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
  python3 train_m1.py "$arm" --config "$EXP/$cfg" --seed "$seed" \
      --out-root "$EXP/train" > "$EXP/train_${arm}.log" 2>&1
  local rc=$?
  echo "=== [$arm] 训练结束 rc=$rc $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
  release
  if [ $rc -ne 0 ]; then echo "[$arm] 训练失败，见 $EXP/train_${arm}.log"; tail -25 "$EXP/train_${arm}.log"; fi
  echo "=== [$arm] 臂间窗口 65s ==="
  sleep 65
}

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then ARMS=(13m_s0 13m_s1); fi
for a in "${ARMS[@]}"; do run_arm "$a"; done
echo "ALL ARMS DONE $(date -Is)"
