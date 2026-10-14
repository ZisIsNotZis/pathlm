#!/usr/bin/env bash
# 2645 · 驱动：冒烟（rc=0 前置）→ 锚重测 → （可选）12k 臂 → 汇总。
#
# 本 worker 会话的硬约束：单条 bash 命令 ≤ 180 s，且引擎无 `--resume` ⇒ 12k 臂
# （2610 同规模实测 ≈ 12.7 min）**不能**在前台单次命令内跑完。故缺省只跑
# 「冒烟 + 锚重测」；12k 臂须显式 RUN_12K=1（由能承受 >12 min 前台命令的会话执行）。
#
#   bash drive.sh                     # 冒烟 + 锚重测 + 汇总
#   RUN_12K=1 bash drive.sh           # 追加 PE:0（首选臂）
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m"
RUN_12K="${RUN_12K:-0}"
LOG() { echo "[$(date -Iseconds)] $*" >> "$EXP/drive.log"; }

LOG "=== drive start (RUN_12K=$RUN_12K) ==="
STEPS=60 HOLD_ALL=1 ARMS="T:0" LOG_PREFIX=smoke WAIT_ITERS=5 \
  bash "$EXP/train_arms.sh"
rc=$?
LOG "smoke rc=$rc"
if [ "$rc" -ne 0 ]; then LOG "smoke FAILED → 不跑 12k"; exit "$rc"; fi

WAIT_ITERS=5 python3 -u "$EXP/run.py" --stage anchor >> "$EXP/drive.log" 2>&1
LOG "anchor rc=$?"

if [ "$RUN_12K" = "1" ]; then
  ARMS="PE:0" STEPS=12000 bash "$EXP/train_arms.sh"
  LOG "12k rc=$?"
fi

python3 -u "$EXP/run.py" --aggregate --steps 12000 >> "$EXP/drive.log" 2>&1
LOG "aggregate rc=$?"
LOG "=== drive done ==="
