#!/usr/bin/env bash
# ticket 38 W2 · GPU 串行驱动（SCHEMA §20 锁协议 + §20.1 锁粒度）。
#
# 用途：在共享 4090 上取锁 → 跑纯推理全量 → 释放。整程预计 <10 min（先冒烟测步时），
# 因此按 §20.1 允许整程持锁；若超时则按臂边界人工重取。
#
# 用法：bash gpu_run.sh <ckpt> <out_dir> [prompts] [repeats]
set -u
cd "$(dirname "$0")/../../../../.."   # 仓库根
CKPT="${1:-.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt}"
OUT="${2:-axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e}"
PROMPTS="${3:-4}"
REPEATS="${4:-3}"
EXP=axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e
LOCK=/tmp/pathlm_gpu.lock

# ---------- 取锁（自愈：holder_pid 死了 / 无 holder_pid 且锁龄 >15min）----------
hp=""
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; echo "[lock] released at $(date +%H:%M:%S)"; }
trap release EXIT
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
  sleep 15; n=$((n + 1))
  if [ $n -gt 40 ]; then echo "等锁超时（>10min）：holder=$hp"; exit 9; fi
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
echo "[lock] acquired by $$ at $(date +%H:%M:%S)  prev_holder=${hp:-none}(waited $((n*15))s)"

# ---------- 后台跑 + ≤110s 轮询心跳 ----------
OMP_NUM_THREADS=4 python3 "$EXP/run.py" --ckpt "$CKPT" --out "$OUT" \
    --device cuda --repeats "$REPEATS" --prompts "$PROMPTS" \
    >> "$OUT/run_log.txt" 2>&1 &
PID=$!
t0=$(date +%s)
while kill -0 "$PID" 2>/dev/null; do
  sleep 30; touch "$LOCK/heartbeat"
  el=$(( $(date +%s) - t0 ))
  echo "[poll] ${el}s alive (lock heartbeat touched)"
  if [ $el -gt 1500 ]; then echo "[poll] 超过 25min 预算，终止"; kill "$PID"; break; fi
done
wait "$PID"; rc=$?
echo "[done] rc=$rc elapsed=$(( $(date +%s) - t0 ))s"
exit $rc
