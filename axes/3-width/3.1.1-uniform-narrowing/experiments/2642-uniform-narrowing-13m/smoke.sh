#!/usr/bin/env bash
# 2642 · 冒烟（120 步）：确认干预真的到了模型（宽度采样/活跃维/参数审计）、量步时与显存。
# 跑在 seed 99 ⇒ run 目录与结果文件都与正式的 seed 0 分开，不入判据。
# 用法：STEPS=120 ARMS="D192:99 D128:99 WD256:99" bash smoke.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m"
ARMS="${ARMS:-D192:99 D128:99 WD256:99}"
STEPS="${STEPS:-120}"
for spec in $ARMS; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  echo "[$(date -Iseconds)] smoke $ARM-s$SEED steps=$STEPS" >> "$EXP/run_log.txt"
  python3 "$EXP/run.py" --arm "$ARM" --seed "$SEED" --steps "$STEPS" \
    >> "$EXP/smoke_${ARM}_s${SEED}.log" 2>&1
  rc=$?
  echo "[$(date -Iseconds)] smoke $ARM-s$SEED rc=$rc" >> "$EXP/run_log.txt"
done
