#!/usr/bin/env bash
# 2644 · 驱动：冒烟（rc=0 前置）→ 12k 臂（按时间预算降级臂数）→ 汇总。
# 降级规则（预登记在 analysis.md）：<22:00 三臂；22:00–22:30 两臂；≥22:30 只 K50。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m"
LOG() { echo "[$(date -Iseconds)] $*" >> "$EXP/drive.log"; }

LOG "=== drive start ==="
SMOKE=1 STEPS=60 HOLD_ALL=1 LOG_PREFIX=smoke ARMS="K50:0 F50:0 K25:0" \
  bash "$EXP/train_arms.sh"
rc=$?
LOG "smoke rc=$rc"
if [ "$rc" -ne 0 ]; then LOG "smoke FAILED → 不跑 12k"; exit "$rc"; fi

HM=$(date +%H%M)
if [ "$HM" -lt 2200 ]; then ARMS12="K50:0 F50:0 K25:0";
elif [ "$HM" -lt 2230 ]; then ARMS12="K50:0 F50:0";
else ARMS12="K50:0"; fi
LOG "12k arms = $ARMS12 (now $(date -Iseconds))"
STEPS=12000 ARMS="$ARMS12" bash "$EXP/train_arms.sh"
LOG "12k rc=$?"

python3 "$EXP/run.py" --aggregate >> "$EXP/drive.log" 2>&1
LOG "aggregate rc=$?"
LOG "=== drive done ==="
