#!/usr/bin/env bash
# 2681-anchor-recall-seeds23-13m · 四臂训练驱动（窗口 16 + 锚点 8 / 窗口 16 + 锚点 0，seed 2/3）。
#
# §20 GPU 互斥锁：**逐臂取/放锁**（/tmp/pathlm_gpu.lock），臂间 ≥61s（§20.1）；stale 回收
# 照抄 2662/train_arms.sh。**绝不杀别人的进程**；共享机器上 local_lm_research 优先 ⇒
# 等锁期间只等待，不抢。
# 配方与 2662 的 win16a_s{0,1} / 2634 的 rt_none_s{0,1} **逐字段一致**，唯一差别 = seed：
#   --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential
#   --steps 12000 --batch-size 32 --lr 1e-3 --seed {2,3} --exit-mode uniform
#   --train-window 16 --train-anchors {8,0} --eval-batches 4 --out-root .../train
#
# 用法： ARMS="win16a_s2 win16a_s3 win16_s2 win16_s3" bash train_arms.sh
#        MODE=smoke ARMS="win16a_s2 win16_s3" bash train_arms.sh   # 300 步冒烟（测步时）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.1.1-hard-window/experiments/2681-anchor-recall-seeds23-13m"
ARMS="${ARMS:-win16a_s2 win16a_s3 win16_s2 win16_s3}"
MODE="${MODE:-full}"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 40 * 60 ))          # 等锁 >40 min 退让并如实登记（本 WP 契约）

python3 tools/prereg_bind.py "$EXP" --require || exit 3

acquire() {
  local n=0 waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date -Iseconds)] reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); waited=$((waited+15))
    if [ $((n % 8)) -eq 0 ]; then
      echo "[$(date -Iseconds)] wait-lock ${waited}s holder=$hp"
    fi
    if [ $waited -gt $MAX_WAIT_S ]; then
      echo "[$(date -Iseconds)] LOCK WAIT TIMEOUT (>40min) — 退让并返回（如实登记）"
      return 9
    fi
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
}

release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released"
}

run_arm() {   # $1 = 臂名, $2 = seed, $3 = anchors, $4 = steps
  local name="$1" seed="$2" anchors="$3" steps="$4"
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=$seed anchors=$anchors steps=$steps gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  python3 train_depth_ar.py "$name" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps "$steps" --batch-size 32 \
    --lr 1e-3 --seed "$seed" --exit-mode uniform --train-window 16 --train-anchors "$anchors" \
    --eval-batches 4 --out-root "$EXP/train" \
    > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

if [ "$MODE" = "smoke" ]; then
  run_arm win16a_s2 2 8 300 || exit $?
  echo "[$(date -Iseconds)] smoke done"
  exit 0
fi

for A in $ARMS; do
  case "$A" in
    win16a_s2) run_arm win16a_s2 2 8 12000 || exit $? ;;
    win16a_s3) run_arm win16a_s3 3 8 12000 || exit $? ;;
    win16_s2)  run_arm win16_s2  2 0 12000 || exit $? ;;
    win16_s3)  run_arm win16_s3  3 0 12000 || exit $? ;;
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
