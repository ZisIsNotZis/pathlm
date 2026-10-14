#!/usr/bin/env bash
# 2643 · 驱动：冒烟（rc=0 前置）→ 12k 臂（22:20 后只跑 U:0）→ 汇总。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.2.3-tied-untied/experiments/2643-tied-untied-13m"
LOG() { echo "[$(date -Iseconds)] $*" >> "$EXP/drive.log"; }

LOG "=== drive start ==="
# 1) 冒烟（整程持锁；60 步 × 2 臂 <10 min）
SMOKE=1 STEPS=60 HOLD_ALL=1 LOG_PREFIX=smoke ARMS="U:0 PR:0" \
  bash "$EXP/train_arms.sh"
rc=$?
LOG "smoke rc=$rc"
if [ "$rc" -ne 0 ]; then
  if [ "$rc" = "9" ]; then LOG "锁等待超时 → 本轮无法起跑（不伪造读数）"; else LOG "smoke FAILED → 不跑 12k"; fi
  exit "$rc"
fi

# 2) 12k 臂（优先级 U:0 → PR:0；22:20 后放弃 PR:0）
HM=$(date +%H%M)
if [ "$HM" -lt 2220 ]; then ARMS12="U:0 PR:0"; else ARMS12="U:0"; fi
LOG "12k arms = $ARMS12 (now $(date -Iseconds))"
STEPS=12000 ARMS="$ARMS12" bash "$EXP/train_arms.sh"
LOG "12k rc=$?"

# 3) T 复评（复用 2641 锚；纯评测，按臂取锁）——已有结果则跳过（不重复占锁）
if [ -s "$EXP/results_T_s0.json" ]; then
  LOG "T 复评已有结果，跳过"
else
  NOTRAIN=1 STEPS=12000 LOG_PREFIX=reeval ARMS="T:0" bash "$EXP/train_arms.sh"
  LOG "reeval T rc=$?"
fi

# 4) 汇总
python3 "$EXP/run.py" --aggregate >> "$EXP/drive.log" 2>&1
LOG "aggregate rc=$?"
LOG "=== drive done ==="
