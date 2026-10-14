#!/usr/bin/env bash
# 2662-hard-window-anchor-13m · 全流程驱动：冒烟 → 两臂训练 → 回填 config → 评测。
# 每步按臂取/放 §20 GPU 锁（/tmp/pathlm_gpu.lock）；stale 回收照抄 2639/train_arms.sh。
# 墙钟只作过程指标，不进判据。用法：nohup bash run_all.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOG="$EXP/driver.log"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 20 * 60 ))

acquire() {
  local n=0 waited=0
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
    sleep 15; n=$((n+1)); waited=$((waited+15))
    if [ $((n % 8)) -eq 0 ]; then echo "[$(date -Iseconds)] wait-lock ${waited}s holder=$hp"; fi
    if [ $waited -gt $MAX_WAIT_S ]; then
      echo "[$(date -Iseconds)] LOCK WAIT TIMEOUT (>20min) — 退让"
      return 9
    fi
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released"
}

echo "[$(date -Iseconds)] DRIVER START pid=$$" > "$LOG"

# 1) 冒烟（300 步，测步时；train_arms.sh 自带取/放锁）
MODE=smoke ARMS="smk16a" bash "$EXP/train_arms.sh" >> "$LOG" 2>&1
echo "[$(date -Iseconds)] smoke done rc=$?" >> "$LOG"

# 2) 两臂训练（win16a_s0 → win16a_s1，臂间 ≥61s；train_arms.sh 按臂取/放锁）
ARMS="win16a_s0 win16a_s1" bash "$EXP/train_arms.sh" >> "$LOG" 2>&1
echo "[$(date -Iseconds)] train done rc=$?" >> "$LOG"

# 3) 回填 config（逐臂 path + sha256）
python3 "$EXP/make_config.py" --require-new >> "$LOG" 2>&1
echo "[$(date -Iseconds)] make_config done rc=$?" >> "$LOG"

# 4) 评测（取锁；run.py 落盘 results.json / run_log.txt）
if acquire; then
  echo "[$(date -Iseconds)] EVAL START" >> "$LOG"
  python3 "$EXP/run.py" > "$EXP/run_log.txt" 2>&1
  echo "[$(date -Iseconds)] EVAL done rc=$?" >> "$LOG"
  release
fi
echo "[$(date -Iseconds)] DRIVER END" >> "$LOG"
