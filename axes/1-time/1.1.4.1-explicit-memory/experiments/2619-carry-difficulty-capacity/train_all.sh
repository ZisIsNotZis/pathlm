#!/usr/bin/env bash
# 契约 U1（ticket 36 波二）：四臂容量曲线训练驱动。
# 持 /tmp/pathlm_gpu.lock（SCHEMA §20 + 编排方 2026-10-02 自愈协议）：
#   启动时做一次**一次性回收**（仅当 空锁 + 无 holder_pid + 无本仓 GPU python 进程），
#   然后取锁、写 holder_pid + 心跳；后台 nohup 训练 + ≤110s 轮询（轮询期间保持持锁）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

has_repo_gpu_proc() { pgrep -a python3 2>/dev/null | grep -qE "train_depth_ar\.py|evalkit|run\.py"; }

# 一次性回收（编排方 2026-10-02 授权：认定 18:39:07 的空锁为孤儿锁）。
# 回收前三件事同时成立：目录存在且为空 + 无本仓 python GPU 进程 + holder_pid 不存在/已死。
if [ -d "$LOCK" ]; then
  pid="$(cat "$LOCK/holder_pid" 2>/dev/null || echo '')"
  if [ -z "$(ls -A "$LOCK" 2>/dev/null)" ] \
     && [ -z "$pid" ] && ! has_repo_gpu_proc; then
    echo "reclaim: 空锁 + holder_pid 不存在 + 无本仓 GPU 进程 $(date -Is)"
    rmdir "$LOCK" 2>/dev/null
  else
    echo "no-reclaim: lock 非空或持有者仍在（pid='${pid:-<missing>}'）"
  fi
fi

n=0
until mkdir "$LOCK" 2>/dev/null; do
  sleep 15; n=$((n+1))
  [ $n -gt 40 ] && { echo "GPU LOCK TIMEOUT after $((n*15))s"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"
touch "$LOCK/heartbeat"
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
echo "GPU LOCK acquired $(date -Is) (waited $((n*15))s, holder $$)"

cd "$ROOT"
declare -A DH=( [D16]=16 [D32]=32 [D64]=64 [D128]=128 )
for arm in D16 D32 D64 D128; do
  dh="${DH[$arm]}"
  nohup python3 train_depth_ar.py "$arm" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps 10000 --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 \
    --carry-write delta --carry-state-dim "$dh" \
    --out-root "$HERE/train" > "$HERE/train_${arm}.log" 2>&1 &
  pid=$!
  echo "[$(date -Is)] $arm (dh=$dh) started pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 100; touch "$LOCK/heartbeat"; done
  wait "$pid"; rc=$?
  echo "[$(date -Is)] $arm done rc=$rc"
  [ "$rc" -ne 0 ] && { echo "ARM $arm FAILED rc=$rc"; tail -20 "$HERE/train_${arm}.log"; }
done

echo "[$(date -Is)] ALL ARMS DONE"
rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "GPU LOCK released $(date -Is)"
