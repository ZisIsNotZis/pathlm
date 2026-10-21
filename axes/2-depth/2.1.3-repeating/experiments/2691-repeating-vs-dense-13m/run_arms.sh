#!/usr/bin/env bash
# 2691 有界后台 launcher：P0 仪器门 + 四臂训练/评测，每步按 SCHEMA §20.1 取/放 GPU 锁。
# 单臂 timeout 1200 s（契约 §6：单臂 ≤20 min）；最多 ROUNDS 轮等锁（每轮 acquire ≤10 min）；
# 不杀他人进程、不动他人锁。完成后自动 aggregate 并打印 DONE 横幅（判活看横幅，不看进程名）。
set -u
cd "$(dirname "$0")"
LOCK=/tmp/pathlm_gpu.lock
ROUNDS=${ROUNDS:-6}
SEP_SEC=${SEP_SEC:-60}
LOG=run_log.txt
ARMS=(dense6 dense12 repeat2x repeat2x-tail)

acquire() {   # 最多等 10 min（40×15s）
  local n=0 hp stale reason
  while ! mkdir "$LOCK" 2>/dev/null; do
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    stale=0; reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp dead"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="no holder_pid, age>15m"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date +%FT%T)] reclaim: $reason" | tee -a "$LOG"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && return 9
  done
  echo $$ > "$LOCK/holder_pid"; echo wp2691 > "$LOCK/name"; touch "$LOCK/heartbeat"
}

release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }

run_under_lock() {   # $1=label, rest=run.py args
  local label="$1"; shift
  local r
  for r in $(seq 1 "$ROUNDS"); do
    if acquire; then
      echo "[$(date +%FT%T)] === $label (round $r) ===" | tee -a "$LOG"
      timeout 1200 python3 run.py "$@" 2>&1 | tee -a "$LOG"
      local rc=${PIPESTATUS[0]}
      release
      if [ "$rc" -eq 124 ]; then
        echo "[$(date +%FT%T)] TIMEOUT(1200s) $label ⇒ 该臂按失败登记" | tee -a "$LOG"
      fi
      sleep "$SEP_SEC"
      return $rc
    fi
    echo "[$(date +%FT%T)] lock timeout for $label (round $r)" | tee -a "$LOG"
    sleep 30
  done
  echo "[$(date +%FT%T)] GIVE UP on $label after $ROUNDS rounds" | tee -a "$LOG"
  return 9
}

echo "[$(date +%FT%T)] launcher start (pid $$)" | tee -a "$LOG"

# P0：先用本实验的评测代码重评历史锚（证明仪器可比），再花钱训练。
run_under_lock p0 --p0-only
rc_p0=$?
echo "[$(date +%FT%T)] p0 rc=$rc_p0" | tee -a "$LOG"

for arm in "${ARMS[@]}"; do
  run_under_lock "$arm" --arms "$arm" --budget-min 20
done

run_under_lock aggregate --aggregate
echo "[$(date +%FT%T)] LAUNCHER DONE" | tee -a "$LOG"
