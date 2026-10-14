#!/usr/bin/env bash
# 2642 · 训练驱动（臂 × seed）。§20 GPU 互斥锁：**按臂**取/放，臂间 ≥61 s。
# 用法：ARMS="D192:0 D128:0 WD256:0" bash train_arms.sh
#       STEPS=12000 ARMS="D192:1" bash train_arms.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m"
ARMS="${ARMS:-D192:0 D128:0 WD256:0}"
STEPS="${STEPS:-12000}"
LOG_PREFIX="${LOG_PREFIX:-train}"
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
    sleep 4; n=$((n+1)); [ $n -gt 375 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT (>25min)"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$ ($1)" >> "$EXP/run_log.txt"
  return 0
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released" >> "$EXP/run_log.txt"
}

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  acquire "$ARM-s$SEED" || exit 9
  echo "[$(date -Iseconds)] free before $ARM-s$SEED: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)" >> "$EXP/run_log.txt"
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" \
    >> "$EXP/${LOG_PREFIX}_${ARM}_s${SEED}.log" 2>&1
  rc=$?
  release
  echo "[$(date -Iseconds)] arm $ARM seed $SEED done rc=$rc" >> "$EXP/run_log.txt"
  if [ $rc -ne 0 ]; then
    echo "[$(date -Iseconds)] arm $ARM-$SEED FAILED (见 ${LOG_PREFIX}_${ARM}_s${SEED}.log)" >> "$EXP/run_log.txt"
    [ "${FAIL_FAST:-1}" = "1" ] && exit $rc
  fi
  sleep 61
done
echo "[$(date -Iseconds)] all arms done" >> "$EXP/run_log.txt"
