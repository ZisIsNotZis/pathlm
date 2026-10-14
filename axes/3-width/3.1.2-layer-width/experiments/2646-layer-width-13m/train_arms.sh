#!/usr/bin/env bash
# 2646 · 训练驱动（臂 × seed）。§20 GPU 互斥锁：**按臂**取/放，臂间 ≥61 s。
# 协议照抄 2642 train_arms.sh（同仓既有取放协议）；**不杀他人进程**。
#
# 用法：ARMS="EQ:0 PYR:0 HG:0 APYR:0" bash train_arms.sh
#       STEPS=60 HOLD_ALL=1 ARMS="EQ:99 PYR:99 HG:99 APYR:99" bash train_arms.sh   # 冒烟
# HOLD_ALL=1 ⇒ 全程只取一次锁（冒烟用：60 步的臂之间不值得放锁再等 61 s）。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.2-layer-width/experiments/2646-layer-width-13m"
ARMS="${ARMS:-EQ:0 PYR:0 HG:0 APYR:0}"
STEPS="${STEPS:-12000}"
HOLD_ALL="${HOLD_ALL:-0}"
LOG_PREFIX="${LOG_PREFIX:-train}"
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
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
    sleep 4; n=$((n+1)); [ $n -gt 375 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT (>25min)"; return 9; }
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

HELD=0
if [ "$HOLD_ALL" = "1" ]; then
  acquire "HOLD_ALL" || exit 9
  HELD=1
fi

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  if [ "$HELD" != "1" ]; then
    acquire "$ARM-s$SEED" || exit 9
    echo "[$(date -Iseconds)] free before $ARM-s$SEED: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)" >> "$EXP/run_log.txt"
  fi
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" \
    >> "$EXP/${LOG_PREFIX}_${ARM}_s${SEED}.log" 2>&1
  rc=$?
  if [ "$HELD" != "1" ]; then
    release
    echo "[$(date -Iseconds)] arm $ARM seed $SEED done rc=$rc" >> "$EXP/run_log.txt"
  fi
  if [ $rc -ne 0 ]; then
    echo "[$(date -Iseconds)] arm $ARM-$SEED FAILED rc=$rc (见 ${LOG_PREFIX}_${ARM}_s${SEED}.log)" >> "$EXP/run_log.txt"
    [ "${FAIL_FAST:-1}" = "1" ] && { [ "$HELD" = "1" ] && release; exit $rc; }
  fi
  [ "$HELD" != "1" ] && sleep 61
done

[ "$HELD" = "1" ] && release
echo "[$(date -Iseconds)] all arms done (HOLD_ALL=$HOLD_ALL)" >> "$EXP/run_log.txt"
