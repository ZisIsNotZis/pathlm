#!/usr/bin/env bash
# 2677-local-global-vs-band-13m · 两臂训练驱动（均匀条带 win24a0 的 seed 0/1）。
#
# §20 GPU 互斥锁：**逐臂取/放锁**（/tmp/pathlm_gpu.lock），臂间 ≥61s（§20.1）；stale 回收照抄
# 既有多臂驱动。**绝不杀别人的进程**。
# 配方与 2662 的 win16a8 臂**逐字段一致**，唯一差别 = 掩码的 (window, anchors)：
#   --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential
#   --steps 12000 --batch-size 32 --lr 1e-3 --seed {0,1} --exit-mode uniform
#   --train-window 24 --train-anchors 0 --eval-batches 4 --out-root .../train
#
# 用法： bash train_arms.sh                 # 两臂全跑（12000 步）
#        MODE=smoke bash train_arms.sh     # 单臂 300 步冒烟（量步时，写 smoke/）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.2.2-block-sparse/experiments/2677-local-global-vs-band-13m"
ARMS="${ARMS:-win24a0_s0 win24a0_s1}"
MODE="${MODE:-full}"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 20 * 60 ))
ARM_TIMEOUT=20m

# 预登记绑定必须先于训练（tools/check_tree.py 机检定序）；缺绑定即拒绝启动。
if ! python3 tools/prereg_bind.py "$EXP" --require; then
  echo "先绑定契约：python3 tools/prereg_bind.py $EXP"; exit 3
fi

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
    [ $((n % 8)) -eq 0 ] && echo "[$(date -Iseconds)] wait-lock ${waited}s holder=$hp"
    if [ $waited -gt $MAX_WAIT_S ]; then
      echo "[$(date -Iseconds)] LOCK WAIT TIMEOUT (>20min) — 退让并如实登记"
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

run_arm() {   # $1 = 臂名, $2 = seed, $3 = steps, $4 = out-root
  local name="$1" seed="$2" steps="$3" root="$4"
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=$seed steps=$steps gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT" python3 train_depth_ar.py "$name" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps "$steps" --batch-size 32 \
    --lr 1e-3 --seed "$seed" --exit-mode uniform --train-window 24 --train-anchors 0 \
    --eval-batches 4 --out-root "$root" \
    > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

if [ "$MODE" = "smoke" ]; then
  run_arm "smk24" 0 300 "$EXP/smoke"
  echo "[$(date -Iseconds)] smoke done"
  exit $?
fi

for A in $ARMS; do
  case "$A" in
    win24a0_s0) run_arm win24a0_s0 0 12000 "$EXP/train" || exit $? ;;
    win24a0_s1) run_arm win24a0_s1 1 12000 "$EXP/train" || exit $? ;;
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
