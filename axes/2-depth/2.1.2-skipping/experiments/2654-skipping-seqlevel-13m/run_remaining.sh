#!/usr/bin/env bash
# 2654 有界后台 launcher：20 步冒烟 + 逐臂训练/评测，每臂按 SCHEMA §20.1 取/放 GPU 锁。
# 最多 ROUNDS 轮等锁（每轮 acquire 最多 10 min）；不杀他人进程、不动他人锁。
# 完成后自动 aggregate 并打印 DONE 横幅（判活看横幅，不看进程名）。
set -u
cd "$(dirname "$0")"
LOCK=/tmp/pathlm_gpu.lock
ROUNDS=${ROUNDS:-6}
SEP_SEC=${SEP_SEC:-60}
ARMS=(stock skip-mid layers6 layerdrop-pt)

acquire() {   # 最多等 10 min（40×15s）
  local n=0 hp stale reason
  while ! mkdir "$LOCK" 2>/dev/null; do
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    stale=0; reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp dead"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="no holder_pid, age>15m"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date +%FT%T)] reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && return 9
  done
  echo $$ > "$LOCK/holder_pid"; echo wp2654 > "$LOCK/name"; touch "$LOCK/heartbeat"
}

release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }

run_under_lock() {   # $1=label, rest=run.py args
  local label="$1"; shift
  local r
  for r in $(seq 1 "$ROUNDS"); do
    if acquire; then
      echo "[$(date +%FT%T)] === $label (round $r) ==="
      python3 run.py "$@"
      local rc=$?
      release
      sleep "$SEP_SEC"
      return $rc
    fi
    echo "[$(date +%FT%T)] lock timeout for $label (round $r)"
    sleep 30
  done
  echo "[$(date +%FT%T)] GIVE UP on $label after $ROUNDS rounds"
  return 9
}

echo "[$(date +%FT%T)] launcher start (pid $$)"
run_under_lock smoke --smoke --steps 20 || { echo "[$(date +%FT%T)] SMOKE FAILED"; exit 1; }

for arm in "${ARMS[@]}"; do
  run_under_lock "$arm" --arms "$arm" --budget-min 30
done

python3 run.py --aggregate
echo "[$(date +%FT%T)] LAUNCHER DONE"
