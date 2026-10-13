#!/usr/bin/env bash
# 契约 U1：评测驱动（持 /tmp/pathlm_gpu.lock；SCHEMA §20 + 编排方自愈协议）。
# 注意：释放用 `rm -rf` 而不是 `rmdir`——锁目录里写了 holder_pid 后 rmdir 会失败
#（本轮 U1 第一次释放就踩了这个坑，见 analysis.md §GPU 锁台账）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

has_repo_gpu_proc() { pgrep -a python3 2>/dev/null | grep -qE "train_depth_ar\.py|evalkit|/run\.py"; }

n=0
until mkdir "$LOCK" 2>/dev/null; do
  hp="$(cat "$LOCK/holder_pid" 2>/dev/null || echo '')"
  if { [ -z "$hp" ] || ! kill -0 "$hp" 2>/dev/null; } && ! has_repo_gpu_proc; then
    echo "reclaim: holder_pid='${hp:-<missing>}' 不存在/已死 $(date -Is)"; rm -rf "$LOCK"; continue
  fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
  if [ "$age" -gt 600 ] && ! has_repo_gpu_proc; then
    echo "reclaim: 陈旧锁 ${age}s $(date -Is)"; rm -rf "$LOCK"; continue
  fi
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "GPU LOCK TIMEOUT after $((n*15))s"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"
trap 'rm -rf "$LOCK"' EXIT
echo "GPU LOCK acquired $(date -Is) (waited $((n*15))s, holder $$)"

cd "$ROOT"
python3 "$HERE/run.py" "$@" 2>&1 | tail -60
rc=${PIPESTATUS[0]}
echo "run.py rc=$rc $(date -Is)"
rm -rf "$LOCK"; trap - EXIT
echo "GPU LOCK released $(date -Is)"
exit $rc
