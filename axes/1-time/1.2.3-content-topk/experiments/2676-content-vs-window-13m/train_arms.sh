#!/usr/bin/env bash
# 2676-content-vs-window-13m · 两臂训练驱动（按位置取最近 k 的 win32 / win64）。
#
# §20 GPU 互斥锁：**逐臂取/放锁**（/tmp/pathlm_gpu.lock），臂间 ≥61s（§20.1）；stale 回收照抄
# 既有多臂驱动。**绝不杀别人的进程**。
# 配方与 2614 的 full/topk 臂**逐字段一致**，唯一差别 = 可见键的选择规则（位置 vs 内容）：
#   --d 256 --layers 8 --heads 8 --mlp-mult 4 --seq-len 256 --dense --topology sequential
#   --steps 12000 --batch-size 32 --lr 1e-3 --exit-mode uniform --eval-batches 20 --out-root .../runs
# 每臂唯一的干预：`--train-window {32,64} --train-anchors 0`（位置臂）或 `--attn-sparse-k 32`（内容臂）；
# 升格对（topk32_s1 / win32_s1）与同名 seed 0 臂逐字段相同，只换 `--seed 1`。
# （不开 --attn-sparse-k 与 --train-window 的组合：两臂是同一研究项的并列臂。）
#
# 用法： bash train_arms.sh                 # 两臂全跑（12000 步，seed 0）
#        ARMS="topk32_s1 win32_s1" bash train_arms.sh   # 主判 k=32 的升格对（seed 1）
#        MODE=smoke bash train_arms.sh     # 单臂 40 步冒烟（量步时，不写正式 run 目录）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m"
ARMS="${ARMS:-win32 win64}"
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

run_arm() {   # $1 = 臂名, $2 = seed, $3 = steps, $4 = out-root, $5.. = 本臂额外开关
  local name="$1" seed="$2" steps="$3" root="$4"; shift 4
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=$seed steps=$steps gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT" python3 train_depth_ar.py "$name" \
    --d 256 --layers 8 --heads 8 --mlp-mult 4 \
    --seq-len 256 --dense --topology sequential --steps "$steps" --batch-size 32 \
    --lr 1e-3 --seed "$seed" --exit-mode uniform "$@" \
    --eval-batches 20 --out-root "$root" \
    > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

if [ "$MODE" = "smoke" ]; then
  run_arm "smk32" 0 40 "$EXP/smoke" --train-window 32 --train-anchors 0
  echo "[$(date -Iseconds)] smoke done"
  exit $?
fi

for A in $ARMS; do
  case "$A" in
    win32)     run_arm win32     0 12000 "$EXP/runs" --train-window 32 --train-anchors 0 || exit $? ;;
    win64)     run_arm win64     0 12000 "$EXP/runs" --train-window 64 --train-anchors 0 || exit $? ;;
    topk32_s1) run_arm topk32_s1 1 12000 "$EXP/runs" --attn-sparse-k 32 || exit $? ;;
    win32_s1)  run_arm win32_s1  1 12000 "$EXP/runs" --train-window 32 --train-anchors 0 || exit $? ;;
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
