#!/usr/bin/env bash
# 2641 · stock 路径 300 步冒烟（只为量步时/显存；§20 整程持锁，因整程 <10 min，§20.1.1）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m"
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
acquire
echo "[$(date -Iseconds)] nvidia-smi free before:"; nvidia-smi --query-gpu=memory.free --format=csv,noheader
t0=$(date +%s)
python3 train_depth_ar.py stock-smoke-300 --d 256 --layers 8 --heads 8 --seq-len 256 \
  --mlp-mult 4 --steps 300 --batch-size 32 --seed 0 --exit-mode uniform --dense \
  --eval-batches 2 --out-root "$EXP/smoke" > "$EXP/smoke_stock_300.log" 2>&1
rc=$?
t1=$(date +%s)
echo "[$(date -Iseconds)] smoke rc=$rc wall=$((t1-t0))s"
echo "[$(date -Iseconds)] nvidia-smi free after:"; nvidia-smi --query-gpu=memory.free --format=csv,noheader
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
echo "[$(date -Iseconds)] lock released"
