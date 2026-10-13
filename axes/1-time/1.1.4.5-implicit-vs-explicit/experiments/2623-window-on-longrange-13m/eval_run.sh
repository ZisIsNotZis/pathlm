#!/usr/bin/env bash
# 2623 评测驱动：先取 /tmp/pathlm_gpu.lock（SCHEMA §20），再跑纯推理电池。
# 用法： ./eval_run.sh            # 全电池
#        ./eval_run.sh --smoke    # 冒烟（缩批）
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
    echo "reclaim: $reason"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "GPU LOCK TIMEOUT (holder=${hp:-<missing>})"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
echo "LOCK acquired $(date -Is) waited $((n*15))s"
cd "$ROOT"
python3 "$HERE/run.py" "$@"
rc=$?
echo "run.py rc=$rc"
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "LOCK released $(date -Is)"
exit $rc
