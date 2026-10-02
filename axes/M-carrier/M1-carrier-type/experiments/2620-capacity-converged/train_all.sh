#!/usr/bin/env bash
# 契约 V2 / 2620：三臂（S16 / S128 / S16_rep）24k 步训练驱动。
#
# GPU 互斥锁协议（SCHEMA §20 修正版 + 2026-10-02 编排方「锁粒度」修正）：
#   * **按臂释放/重取锁**：每臂结束后释放，并留 **≥60 秒窗口**再取下一臂
#     （整程 <10 min 的短活才允许整程持锁）；
#   * 取锁 mkdir，最多等 10 分钟（40×15s），等不到就退出码 9 上报，**不杀别人的进程**；
#   * 持锁时写 holder_pid + heartbeat；
#   * 回收只用 **holder_pid 存活性**判据（无 holder_pid 的老锁按锁龄 >15min），**不用 pgrep**；
#   * 回收/释放一律 **先 `rm -f` 内容再 `rmdir`**（非空目录 rmdir 会失败 ⇒ 漏锁）；
#   * 循环路径上始终有 `sleep`；长训练 nohup 后台 + ≤110 s 轮询（轮询期间保持持锁）。
#
# **口径披露（重要）**：2026-10-02 21:07–22:47 的**正式运行**用的是**本脚本的上一版**
# （整程持锁 100 分钟，无臂间窗口）——那是「锁粒度」修正下发之前的行为。本版按修正后的
# 协议实现「按臂释放/重取 + ≥60s 窗口」，供**复现**使用；两版对训练结果无影响
# （锁只影响并发，不影响数值），差异记在 `analysis.md` §GPU 锁台账与 `stability.md`。
#
# 用法： ./train_all.sh            # 3 臂 × 24000 步（默认）
#        ./train_all.sh --smoke    # 20 步冒烟（3 臂，~1 min；<10 min 故整程持锁）
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

STEPS=24000
SUFFIX=""
OUT="$HERE/train"
for a in "$@"; do
  case "$a" in
    --smoke) STEPS=20; SUFFIX="_smoke"; OUT="$HERE/smoke/train20";;
    *) echo "unknown arg $a"; exit 2;;
  esac
done

HOLDING=0
release_lock() {
  if [ "$HOLDING" -eq 1 ]; then
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null
    HOLDING=0
    echo "[$(date -Is)] GPU LOCK released"
  fi
}
trap 'release_lock' EXIT

take_lock() {
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
  HOLDING=1
  echo "[$(date -Is)] GPU LOCK acquired (waited $((n*15))s, holder $$)"
}

cd "$ROOT"
declare -A DH=( [S16]=16 [S128]=128 [S16_rep]=16 )
first=1
for arm in S16 S128 S16_rep; do
  dh="${DH[$arm]}"
  [ "$first" -eq 0 ] && { echo "[$(date -Is)] 臂间窗口：sleep 60（协议修正：按臂释放/重取锁）"; sleep 60; }
  first=0
  take_lock
  nohup python3 train_depth_ar.py "$arm" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps "$STEPS" --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --carry-write delta --carry-state-dim "$dh" \
    --out-root "$OUT" > "$HERE/train_${arm}${SUFFIX}.log" 2>&1 &
  pid=$!
  echo "[$(date -Is)] $arm (dh=$dh, steps=$STEPS) started pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 100; touch "$LOCK/heartbeat"; done
  wait "$pid"; rc=$?
  echo "[$(date -Is)] $arm done rc=$rc"
  [ "$rc" -ne 0 ] && { echo "ARM $arm FAILED rc=$rc"; tail -20 "$HERE/train_${arm}${SUFFIX}.log"; }
  release_lock
done

echo "[$(date -Is)] ALL ARMS DONE"
