#!/usr/bin/env bash
# 2681-int8-equalbytes-frontier-13m · 训练驱动（纯窗臂 win60 / win64，anchors=0）。
# 配方与 2638-carry-frontier-13m/train_arms.sh 逐字段相同，唯一变量 = --train-window。
# §20 GPU 互斥锁：按臂取/放锁，臂间 ≥60s。
# 用法：bash train_arms.sh          # 训 win60 win64
#       ARMS="60" bash train_arms.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m"
ARMS="${ARMS:-60 64}"
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "LOCK TIMEOUT"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released"
}

for K in $ARMS; do
  acquire
  python3 train_depth_ar.py "win${K}" --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
    --lr 1e-3 --seed 0 --exit-mode uniform --train-window "$K" --train-anchors 0 \
    --eval-batches 4 --carry-write none --out-root "$EXP/train" \
    > "$EXP/train_win${K}.log" 2>&1
  rc=$?
  release
  echo "[$(date -Iseconds)] arm win$K done rc=$rc"
  [ $rc -ne 0 ] && { echo "arm win$K FAILED"; exit $rc; }
  sleep 61
done
echo "[$(date -Iseconds)] all arms done"
