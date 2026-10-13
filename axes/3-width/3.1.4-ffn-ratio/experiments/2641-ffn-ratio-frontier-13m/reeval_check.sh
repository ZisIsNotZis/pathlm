#!/usr/bin/env bash
# 2641 · 评测确定性核对（训练后跑）：对已训好的臂**只评测**（--no-train），
# 与 results_<arm>_s<seed>.json 记录的 clean bpc 逐位比对（评测口径确定性证据）。
# §20：整程 <10 min（单臂评测 ~40 s）⇒ 允许整程持锁；臂间 ≥61 s。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m"
LOCK=/tmp/pathlm_gpu.lock
ARMS="${ARMS:-U4:0 U2:0 PYR:0 APYR:0 U2:1 PYR:1 APYR:1 U4:1}"
STEPS="${STEPS:-12000}"

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ $stale -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "[$(date -Iseconds)] LOCK TIMEOUT"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
}

for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  SNAP="$EXP/results_${ARM}_s${SEED}.json"
  [ -f "$SNAP" ] || { echo "[$(date -Iseconds)] skip $ARM-$SEED（无结果文件）"; continue; }
  acquire || exit 9
  cp "$SNAP" "/tmp/2641_prev_${ARM}_${SEED}.json"
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" --no-train \
    >> "$EXP/run_log.txt" 2>&1
  rc=$?
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
  python3 - "$ARM" "$SEED" "$SNAP" <<'PY'
import json, sys
arm, seed, snap = sys.argv[1], sys.argv[2], sys.argv[3]
prev = json.load(open(f"/tmp/2641_prev_{arm}_{seed}.json"))
now = json.load(open(snap))
same = prev["eval"]["clean_bpc"] == now["eval"]["clean_bpc"]
print(f"[reeval] {arm}-s{seed} prev={prev['eval']['clean_bpc']} "
      f"now={now['eval']['clean_bpc']} bitwise_equal={same}")
PY
  echo "[$(date -Iseconds)] reeval $ARM-$SEED rc=$rc"
  sleep 61
done
echo "[$(date -Iseconds)] reeval done"
