#!/usr/bin/env bash
# 2641 · 训练驱动（臂 × seed）。§20 GPU 互斥锁：**按臂**取/放，臂间 ≥60 s（§20.1）。
# 用法：ARMS="U4:0 U2:0 PYR:0 APYR:0" bash train_arms.sh
#       STEPS=12000 ARMS="PYR:1" bash train_arms.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m"
ARMS="${ARMS:-U4:0 U2:0 PYR:0 APYR:0 U2:1 PYR:1 APYR:1}"
STEPS="${STEPS:-12000}"
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
      echo "[$(date -Iseconds)] reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
  return 0
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released"
}

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  acquire || exit 9
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | sed "s/^/[$(date -Iseconds)] free before: /"
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" \
    > "$EXP/train_${ARM}_s${SEED}.log" 2>&1
  rc=$?
  release
  echo "[$(date -Iseconds)] arm $ARM seed $SEED done rc=$rc"
  [ $rc -ne 0 ] && { echo "[$(date -Iseconds)] arm $ARM-$SEED FAILED (见 train_${ARM}_s${SEED}.log)"; exit $rc; }
  sleep 61
done
echo "[$(date -Iseconds)] all arms done"
