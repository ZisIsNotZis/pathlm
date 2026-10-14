#!/usr/bin/env bash
# 在 GPU 锁内执行任意命令（SCHEMA §20）：取锁 → 执行 → 放锁（等锁 ≤6 min，不杀他人进程）。
# 用法：bash lock_exec.sh <cmd> [args...]
set -u
LOCK=/tmp/pathlm_gpu.lock
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 5; n=$((n+1))
  if [ $n -gt 72 ]; then echo "[lock_exec] LOCK TIMEOUT (6min)" >&2; exit 9; fi
done
echo $$ > "$LOCK/holder_pid"; echo "wp2672-eval" > "$LOCK/name"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
echo "[$(date -Is)] lock_exec: lock acquired (pid $$): $*"
"$@"
rc=$?
echo "[$(date -Is)] lock_exec: exit $rc"
exit $rc
