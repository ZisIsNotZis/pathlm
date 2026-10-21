#!/usr/bin/env bash
# 2686-window-kcurve-seed2-13m · 三个 seed-2 臂的训练驱动（win128_s2、win64_s2、win64a_s2）。
#
# §20 GPU 互斥锁：**逐臂取/放锁**（/tmp/pathlm_gpu.lock），臂间 ≥61s（§20.1）；stale 回收
# 照抄 2682/train_arms.sh。**绝不杀别人的进程**；共享机器上 local_lm_research 优先 ⇒
# 等锁期间只等待，不抢。单臂 `timeout 1200`（≤20 min，本 WP 契约）。
# 配方与 2682 的 win64*/win128* / win64a* **逐字段一致**，唯一差别 = --seed：
#   --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential
#   --steps 12000 --batch-size 32 --lr 1e-3 --seed 2 --exit-mode uniform
#   --train-window {64,128} --train-anchors {0,8} --eval-batches 4 --out-root .../train
#
# 用法： ARMS="win128_s2 win64_s2 win64a_s2" bash train_arms.sh
#        MODE=smoke ARMS="win128_s2" bash train_arms.sh   # 300 步冒烟（测步时）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.1.1-hard-window/experiments/2686-window-kcurve-seed2-13m"
ARMS="${ARMS:-win128_s2 win64_s2 win64a_s2}"
MODE="${MODE:-full}"
ARM_TIMEOUT_S=$(( 20 * 60 ))
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

run_arm() {   # $1 = 臂名, $2 = 锚点数, $3 = 窗口, $4 = 步数（seed 恒为 2）
  local name="$1" anchors="$2" window="$3" steps="$4"
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=2 window=$window anchors=$anchors steps=$steps gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT_S" python3 train_depth_ar.py "$name" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps "$steps" --batch-size 32 \
    --lr 1e-3 --seed 2 --exit-mode uniform --train-window "$window" \
    --train-anchors "$anchors" \
    --eval-batches 4 --out-root "$EXP/train" \
    > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  if [ $rc -eq 124 ]; then echo "[$(date -Iseconds)] TRAIN TIMEOUT(>${ARM_TIMEOUT_S}s) $name"; fi
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

if [ "$MODE" = "smoke" ]; then
  run_arm win128_s2 0 128 300 || exit $?
  echo "[$(date -Iseconds)] smoke done"
  exit 0
fi

for A in $ARMS; do
  case "$A" in
    win128_s2) run_arm win128_s2 0 128 12000 || exit $? ;;
    win64_s2)  run_arm win64_s2  0 64  12000 || exit $? ;;
    win64a_s2) run_arm win64a_s2 8 64  12000 || exit $? ;;
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
