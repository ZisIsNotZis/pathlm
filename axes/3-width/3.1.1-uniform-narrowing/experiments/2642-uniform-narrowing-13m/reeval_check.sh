#!/usr/bin/env bash
# 2642 · 评测确定性核对（训练后跑）：`--no-train` 重评并与首评**逐位比对**（P4-①）。
# §20：整程 <10 min ⇒ 允许整程持锁；臂间 ≥61 s。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m"
LOCK=/tmp/pathlm_gpu.lock
ARMS="${ARMS:-D192:0 D128:0 WD256:0}"
STEPS="${STEPS:-12000}"

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ $stale -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; continue
    fi
    sleep 4; n=$((n+1)); [ $n -gt 375 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT" >> "$EXP/run_log.txt"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] reeval lock acquired pid=$$" >> "$EXP/run_log.txt"
}

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  SNAP="$EXP/results_${ARM}_s${SEED}.json"
  [ -f "$SNAP" ] || { echo "[$(date -Iseconds)] skip $ARM-$SEED（无结果文件）" >> "$EXP/run_log.txt"; continue; }
  acquire || exit 9
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" --no-train \
    >> "$EXP/reeval_driver.log" 2>&1
  rc=$?
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] reeval $ARM-$SEED rc=$rc" >> "$EXP/run_log.txt"
  sleep 61
done
echo "[$(date -Iseconds)] reeval done" >> "$EXP/run_log.txt"
