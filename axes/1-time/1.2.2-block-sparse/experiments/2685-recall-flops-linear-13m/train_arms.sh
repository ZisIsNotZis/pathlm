#!/usr/bin/env bash
# 2685-recall-flops-linear-13m · 四条新臂的训练驱动（full × 2 seed、linear × 2 seed）。
#
# §20 GPU 互斥锁：**逐臂取/放锁**（/tmp/pathlm_gpu.lock），臂间 ≥61 s（§20.1）；stale 回收照抄既有多臂驱动。
# **绝不杀别人的进程**。配方与 2677 / 2683 的臂逐字段一致（唯一变量 = 注意力算子）：
#   --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential
#   --steps 12000 --batch-size 32 --lr 1e-3 --seed {0,1} --exit-mode uniform --eval-batches 4
# `linear` 臂多一个 `--attn-linear`，并走 `train_linear.py`（它把 φ 换成无消去求值、把分母 guard 的
# eps 降到 1e-30，见 stable_linear.py）。
#
# 用法： bash train_arms.sh                     # 四臂全跑（12000 步）
#        MODE=smoke bash train_arms.sh          # 两档 300 步冒烟（引擎默认 φ vs 无消去 φ，不进判据）
#        ARMS="linear_s0" bash train_arms.sh    # 只跑指定臂
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m"
ARMS="${ARMS:-linear_s0 full_s0 linear_s1 full_s1}"
MODE="${MODE:-full}"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 20 * 60 ))
# 25 min（本仓惯例 20 min；线性臂 12k 步实测 ≈19 min、余量只剩 ~1 min ⇒ 统一抬到 25 min，见 README §5）
# 25 min（全因果臂）/ 45 min（线性臂：共享卡被别的 agent 的 GPU 测试占用时实测 150–220 ms/step）
ARM_TIMEOUT=25m
ARM_TIMEOUT_LINEAR=45m

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

common_args() {   # $1 = seed
  echo --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
    --lr 1e-3 --seed "$1" --exit-mode uniform --eval-batches 4
}

run_dense() {     # $1 = 臂名, $2 = seed
  local name="$1" seed="$2"
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=$seed gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT" python3 train_depth_ar.py "$name" $(common_args "$seed") \
    --out-root "$EXP/train" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

run_linear() {    # $1 = 臂名, $2 = seed
  local name="$1" seed="$2"
  acquire || return 9
  echo "[$(date -Iseconds)] TRAIN START $name seed=$seed (attn-linear, stable phi) gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT_LINEAR" python3 "$EXP/train_linear.py" "$name" $(common_args "$seed") \
    --attn-linear --linear-chunk 32 \
    --out-root "$EXP/train" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; tail -20 "$EXP/train_${name}.log"; return $rc; }
  sleep 61
  return 0
}

run_smoke() {     # $1 = 档名（engine|stable）, $2 = out-dir 名
  local kind="$1" name="$2"
  acquire || return 9
  echo "[$(date -Iseconds)] SMOKE START $name ($kind phi) gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  if [ "$kind" = "engine" ]; then
    timeout 10m python3 train_depth_ar.py "$name" \
      --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
      --steps 300 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform --attn-linear \
      --eval-batches 2 --out-root "$EXP/smoke" > "$EXP/smoke_${name}.log" 2>&1
  else
    timeout 10m python3 "$EXP/train_linear.py" "$name" \
      --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
      --steps 300 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform --attn-linear \
      --linear-chunk 32 --eval-batches 2 --out-root "$EXP/smoke" > "$EXP/smoke_${name}.log" 2>&1
  fi
  local rc=$?
  release
  echo "[$(date -Iseconds)] SMOKE END $name rc=$rc"
  [ $rc -ne 0 ] && { echo "smoke $name FAILED"; tail -20 "$EXP/smoke_${name}.log"; return $rc; }
  sleep 61
  return 0
}

if [ "$MODE" = "smoke" ]; then
  run_smoke engine smk_engine_phi || exit $?
  run_smoke stable smk_stable_phi || exit $?
  echo "[$(date -Iseconds)] smoke done"
  exit 0
fi

for A in $ARMS; do
  case "$A" in
    full_s0)   run_dense full_s0 0 || exit $? ;;
    full_s1)   run_dense full_s1 1 || exit $? ;;
    linear_s0) run_linear linear_s0 0 || exit $? ;;
    linear_s1) run_linear linear_s1 1 || exit $? ;;
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
