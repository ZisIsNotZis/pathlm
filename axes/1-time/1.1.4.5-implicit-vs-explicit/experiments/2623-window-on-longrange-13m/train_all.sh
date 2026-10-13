#!/usr/bin/env bash
# 契约 Y3 / 2623：A9 合成长程判决——两臂训练驱动（full / win16 / 可选 win16_rep）。
#
# 唯一变量 = 训练/评测可见掩码：full（无掩码）vs win16（window 16, anchors 0）。
# 其余逐字段一致（同架构/同数据/同优化器/同步数/同 seed）。
#
# GPU 互斥锁协议（SCHEMA §20 修正版 + §20.1 锁粒度修正）：
#   * **按臂释放/重取锁**，臂间留 **≥60 秒窗口**（整程 <10 min 的短活才允许整程持锁）；
#   * 取锁 mkdir，最多等 10 分钟（40×15s），等不到退出码 9 上报，**不杀别人的进程**；
#   * 持锁写 holder_pid + heartbeat；回收只用 holder_pid 存活性（无 holder_pid 的老锁按
#     锁龄 >15min），**不用 pgrep**；回收/释放一律**先 rm -f 内容再 rmdir**；
#   * 长训练 nohup 后台 + ≤110 s 轮询（轮询期间保持持锁）。
#
# 用法： ./train_all.sh            # 3 臂（full, win16, win16_rep）× 12000 步
#        ./train_all.sh --smoke    # 20 步冒烟（3 臂，<10 min 故整程持锁）
#        ./train_all.sh --no-rep   # 只跑 full+win16（预算不够时砍复现臂，契约 §成本）
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock

STEPS=12000
SUFFIX=""
OUT="$HERE/train"
ARMS="full win16 win16_rep"
WANT_ARMS=""
while [ $# -gt 0 ]; do
  a="$1"
  case "$a" in
    --smoke) STEPS=20; SUFFIX="_smoke"; OUT="$HERE/smoke/train20";;
    --no-rep) ARMS="full win16";;
    --arms) shift; WANT_ARMS="$1";;
    *) echo "unknown arg $a"; exit 2;;
  esac
  shift
done
# --arms 只跑指定臂（用于锁竞争下分次补跑，不重训已完成的臂）
if [ -n "$WANT_ARMS" ]; then ARMS="$WANT_ARMS"; SUFFIX="_single"; fi

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
first=1
for arm in $ARMS; do
  [ "$first" -eq 0 ] && { echo "[$(date -Is)] 臂间窗口：sleep 60（§20.1 按臂释放/重取锁）"; sleep 60; }
  first=0
  take_lock
  # 唯一变量 = 掩码：win16* 臂加 --train-window 16 --train-anchors 0；full 臂不加（缺省 0）
  WINFLAGS=""
  case "$arm" in
    win16|win16_rep) WINFLAGS="--train-window 16 --train-anchors 0";;
  esac
  nohup python3 train_depth_ar.py "$arm" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --seq-len 256 \
    --steps "$STEPS" --batch-size 32 --seed 0 --lr 0.001 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    $WINFLAGS \
    --out-root "$OUT" > "$HERE/train_${arm}${SUFFIX}.log" 2>&1 &
  pid=$!
  echo "[$(date -Is)] $arm (steps=$STEPS$WINFLAGS) started pid=$pid"
  while kill -0 "$pid" 2>/dev/null; do sleep 100; touch "$LOCK/heartbeat"; done
  wait "$pid"; rc=$?
  echo "[$(date -Is)] $arm done rc=$rc"
  [ "$rc" -ne 0 ] && { echo "ARM $arm FAILED rc=$rc"; tail -20 "$HERE/train_${arm}${SUFFIX}.log"; }
  release_lock
done

echo "[$(date -Is)] ALL ARMS DONE"
