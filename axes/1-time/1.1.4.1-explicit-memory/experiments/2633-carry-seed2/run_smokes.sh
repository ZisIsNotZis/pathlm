#!/usr/bin/env bash
# 三步线的步时冒烟总驱动（SCHEMA §20.1：整程 <10 min ⇒ 允许整程持锁一次）。
# 全部三个实验的步时数字都出自本次实测（§17.8：不借用其它实验的步时）。
set -u
cd "$(dirname "$0")/../../../../../.." 2>/dev/null || cd /home/z/vibe/pathlm
ROOT=/home/z/vibe/pathlm
cd "$ROOT"
LOCK=/tmp/pathlm_gpu.lock
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null); stale=0
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
  if [ "$stale" -eq 1 ]; then rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; continue; fi
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "等锁超时"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT

B=axes/1-time/1.1.4.1-explicit-memory/experiments
for s in "$B/2633-carry-seed2/smoke_steptime.sh" \
         "$B/2634-carry-realtext-13m/smoke_steptime.sh" \
         "$B/2635-recursive-carrier-13m/smoke_steptime.sh"; do
  echo "########## $s $(date -Is)"
  timeout 900 bash "$s" || echo "!! $s rc=$?"
  sleep 62
done
echo "ALL SMOKE DONE $(date -Is)"
