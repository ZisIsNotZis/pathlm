#!/usr/bin/env bash
# 轴 4 · 坍缩可观测性探针（4.2 前提）：13M、top-1、无 aux loss、单臂，记录逐 100 步
# 专家使用率分布（熵 / 最大占比）。§20 串行取锁；§20.1 短任务整程持锁（本程 >10min 但单臂、
# 无对照，按 §20.1 的「长训练 nohup 后台 + 轮询 touch heartbeat」执行）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$HERE/../../../axes/4-moe/4.1-sparse-experts/experiments/2610-moe-arm-ledger-13m"
EXP="$(cd "$EXP" && pwd)"
LOCK=/tmp/pathlm_gpu.lock
STEPS="${STEPS:-1200}"
OUT="${OUT:-$EXP/probe_collapse_${STEPS}.json}"

# --- §20 取锁（有界：最多 40×15s = 10min）---
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
  stale=0
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
  if [ $stale -eq 1 ]; then
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "LOCK TIMEOUT"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT

echo "lock acquired at $(date -Iseconds); running $STEPS steps"
cd "$EXP"
python3 run.py train --arm moe-top1-E8 --steps "$STEPS" --log-every 100 \
        --aux-weight 0.0 --seed 0 --out "$OUT"
echo "done at $(date -Iseconds)"
