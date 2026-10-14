#!/usr/bin/env bash
# 2644 · 训练驱动（臂 × seed）。§20 GPU 互斥锁：**按臂**取/放，臂间 ≥61 s。
# 优先级（契约 §9）：K50:0 → F50:0 → K25:0；锁被占最多等 25 min（单次）。
# 用法：ARMS="K50:0 F50:0 K25:0" bash train_arms.sh
#       STEPS=60 HOLD_ALL=1 ARMS="K50:0 F50:0 K25:0" bash train_arms.sh  # 冒烟：整程持锁（<10min）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m"
ARMS="${ARMS:-K50:0 F50:0 K25:0}"
STEPS="${STEPS:-12000}"
LOG_PREFIX="${LOG_PREFIX:-train}"
HOLD_ALL="${HOLD_ALL:-0}"
NOTRAIN="${NOTRAIN:-0}"          # 1 = 只评测（复用已有 model.pt）
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    # 不按「无 holder_pid + 锁龄」回收：他线的锁实现可能不写 holder_pid（会误回收活锁），
    # 只有当锁目录里既无 holder_pid 也无 name（无人认领）时才按锁龄回收。
    if [ -z "$hp" ] && [ ! -e "$LOCK/name" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid/name 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date -Iseconds)] reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 100 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT (>25min)"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$ ($1)" >> "$EXP/run_log.txt"
  return 0
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released" >> "$EXP/run_log.txt"
}

if [ "$HOLD_ALL" = "1" ]; then
  acquire "smoke-hold-all" || exit 9
fi

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  if [ "$HOLD_ALL" != "1" ]; then
    acquire "$ARM-s$SEED" || exit 9
    echo "[$(date -Iseconds)] free before $ARM-s$SEED: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)" >> "$EXP/run_log.txt"
  fi
  extra=""; [ "$NOTRAIN" = "1" ] && extra="--no-train"
  python3 -u "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" $extra \
    >> "$EXP/${LOG_PREFIX}_${ARM}_s${SEED}.log" 2>&1
  rc=$?
  if [ "$HOLD_ALL" != "1" ]; then
    release
    echo "[$(date -Iseconds)] arm $ARM seed $SEED done rc=$rc" >> "$EXP/run_log.txt"
  fi
  if [ $rc -ne 0 ]; then
    echo "[$(date -Iseconds)] arm $ARM-$SEED FAILED rc=$rc (见 ${LOG_PREFIX}_${ARM}_s${SEED}.log)" >> "$EXP/run_log.txt"
    [ "${FAIL_FAST:-1}" = "1" ] && { [ "$HOLD_ALL" = "1" ] && release; exit $rc; }
  fi
  [ "$HOLD_ALL" != "1" ] && sleep 61
done

[ "$HOLD_ALL" = "1" ] && release
echo "[$(date -Iseconds)] all arms done (steps=$STEPS)" >> "$EXP/run_log.txt"
