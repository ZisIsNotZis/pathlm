#!/usr/bin/env bash
# 2620：评测/分析驱动（持 /tmp/pathlm_gpu.lock；SCHEMA §20 修正版协议）。
# 释放与回收一律「先 rm -f 内容、再 rmdir」；回收只用 holder_pid 存活性判据。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  stale=0; reason=""
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
    stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
  if [ $stale -eq 1 ]; then
    echo "[$(date -Is)] reclaim: $reason"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 15; n=$((n+1))
  [ $n -gt 40 ] && { echo "GPU LOCK TIMEOUT after $((n*15))s (holder=${hp:-<missing>})"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
echo "[$(date -Is)] GPU LOCK acquired (waited $((n*15))s, holder $$)"

cd "$ROOT"
python3 "$HERE/run.py" "$@" 2>&1 | tee "$HERE/run_log.txt" | tail -80
rc=${PIPESTATUS[0]}
echo "run.py rc=$rc $(date -Is)"
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "[$(date -Is)] GPU LOCK released"
exit $rc
