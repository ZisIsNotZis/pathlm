#!/usr/bin/env bash
# 契约 V2 / 2620：三臂（S16 / S128 / S16_rep）24k 步训练驱动。
#
# GPU 互斥锁协议（SCHEMA §20，2026-10-02 修正版）：
#   * 取锁 mkdir，最多等 10 分钟（40×15s），等不到就退出码 9 上报，**不杀别人的进程**；
#   * 持锁时写 holder_pid + heartbeat；
#   * 回收只用 **holder_pid 存活性**判据（无 holder_pid 的老锁按锁龄 >15min），**不用 pgrep**；
#   * 回收/释放一律 **先 `rm -f` 内容再 `rmdir`**（非空目录 rmdir 会失败 ⇒ 漏锁）；
#   * 循环路径上始终有 `sleep`；
#   * 长训练 nohup 后台 + ≤110 s 轮询，轮询期间保持持锁并 touch heartbeat。
#
# 用法： ./train_all.sh            # 3 臂 × 24000 步（默认）
#        ./train_all.sh --smoke    # 20 步冒烟（3 臂，~1 min）
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

STEPS=24000
SMOKE=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1;;
    *) echo "unknown arg $a"; exit 2;;
  esac
done
[ "$SMOKE" -eq 1 ] && STEPS=20
OUT="$HERE/train"
[ "$SMOKE" -eq 1 ] && OUT="$HERE/smoke/train20"

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
declare -A DH=( [S16]=16 [S128]=128 [S16_rep]=16 )
for arm in S16 S128 S16_rep; do
  dh="${DH[$arm]}"
  nohup python3 train_depth_ar.py "$arm" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps "$STEPS" --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --carry-write delta --carry-state-dim "$dh" \
    --out-root "$OUT" > "$HERE/train_${arm}${SMOKE:+_smoke}.log" 2>&1 &
  pid=$!
  echo "[$(date -Is)] $arm (dh=$dh, steps=$STEPS) started pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 100; touch "$LOCK/heartbeat"; done
  wait "$pid"; rc=$?
  echo "[$(date -Is)] $arm done rc=$rc  (elapsed since lock: $(($(date +%s) - $(stat -c %Y "$LOCK"))))s"
  [ "$rc" -ne 0 ] && { echo "ARM $arm FAILED rc=$rc"; tail -20 "$HERE/train_${arm}${SMOKE:+_smoke}.log"; }
  touch "$LOCK/heartbeat"
done

echo "[$(date -Is)] ALL ARMS DONE"
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "[$(date -Is)] GPU LOCK released"
