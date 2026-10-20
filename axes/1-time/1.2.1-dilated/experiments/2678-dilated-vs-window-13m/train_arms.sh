#!/usr/bin/env bash
# 2678-dilated-vs-window-13m · 空洞臂训练驱动（逐臂取/放 §20 GPU 锁，臂间 ≥61 s）。
#
# 配方 = 参考臂 `2676` 的 win32 逐字段一致（dense / sequential / 12k 步 / bs32 / lr1e-3），
# **唯一差别 = 注意力可见集**（连续窗口 32 → 近端窗口 16 + 16 个空洞点铺满前缀 = 同 32 键）。
# 掩码在 `run.py` 的 `DilatedBlock` 里，`pathlm/**` 与 `train_depth_ar.py` 不改。
#
# 用法：
#   bash train_arms.sh                 # dil32 seed 0 + seed 1（12k）
#   STEPS=40 bash train_arms.sh        # 40 步接线冒烟（写 runs-smoke/，不进判据）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m"
STEPS="${STEPS:-12000}"
SEEDS="${SEEDS:-0 1}"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 20 * 60 ))
ARM_TIMEOUT=20m

# 预登记绑定必须先于训练（tools/check_tree.py 机检定序）；缺绑定即拒绝启动。
if ! python3 tools/prereg_bind.py "$EXP" --require; then
  echo "先绑定契约：python3 tools/prereg_bind.py $EXP --contract README.md config.json run.py"
  exit 3
fi

acquire() {
  local n=0 waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] \
       && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
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

for S in $SEEDS; do
  acquire || exit 9
  echo "[$(date -Iseconds)] TRAIN START dil32 seed=$S steps=$STEPS gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  timeout "$ARM_TIMEOUT" python3 -u "$EXP/run.py" --stage train --arm dil32 \
    --seed "$S" --steps "$STEPS" > "$EXP/train_dil32_s${S}.log" 2>&1
  rc=$?
  release
  echo "[$(date -Iseconds)] TRAIN END dil32 seed=$S rc=$rc"
  if [ $rc -ne 0 ]; then echo "arm dil32 seed $S FAILED"; tail -20 "$EXP/train_dil32_s${S}.log"; exit $rc; fi
  sleep 61
done
echo "[$(date -Iseconds)] all arms done"
