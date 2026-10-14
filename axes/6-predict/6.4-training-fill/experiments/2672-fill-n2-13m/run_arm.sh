#!/usr/bin/env bash
# 2672 WP1b — 单臂取锁/训练/放锁（SCHEMA §20 与 §20.1：逐 run 取放锁，等锁 ≤6 min）。
# 用法：STEPS=2000 bash run_arm.sh <fill|nofill|projfill> <seed>
# 落盘目录 = snap<STEPS/1000>k-<arm>-s<seed>（刻度写进目录名，避免与 12k 档混淆）。
set -u
ARM="$1"; SEED="$2"; STEPS="${STEPS:-2000}"
TAG="$((STEPS / 1000))k"
RUN="snap${TAG}-${ARM}-s${SEED}"
D="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/z/vibe/pathlm
LOCK=/tmp/pathlm_gpu.lock
LOG="$D/${RUN}.train.log"

case "$ARM" in
  fill)     FILLARGS="" ;;
  nofill)   FILLARGS="--no-fill-kv" ;;
  projfill) FILLARGS="--proj-fill" ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] $RUN: waiting for GPU lock" | tee -a "$LOG"
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  stale=0; reason=""
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp dead"; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
    stale=1; reason="no holder_pid & lock age >15min"; fi
  if [ $stale -eq 1 ]; then
    echo "[$(date -Is)] reclaim: $reason" | tee -a "$LOG"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 0.3; n=$((n+1))
  if [ $n -gt 1200 ]; then echo "[$(date -Is)] LOCK TIMEOUT (6min)" | tee -a "$LOG"; exit 9; fi
done
echo $$ > "$LOCK/holder_pid"; echo "wp2672-$RUN" > "$LOCK/name"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
echo "[$(date -Is)] $RUN: lock acquired (pid $$)" | tee -a "$LOG"

python3 -u train_depth_ar.py "$RUN" \
  --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
  --steps "$STEPS" --batch-size 32 --lr 1e-3 --seed "$SEED" \
  --out-root "$D" $FILLARGS >> "$LOG" 2>&1
rc=$?
echo "[$(date -Is)] $RUN: exit $rc" | tee -a "$LOG"
exit $rc
