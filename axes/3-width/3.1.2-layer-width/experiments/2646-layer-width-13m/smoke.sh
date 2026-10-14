#!/usr/bin/env bash
# 2646 · 冒烟（60 步）：确认干预真的到了模型（逐层活跃维 / qkv 输出宽 / 参数审计）、
# 量步时与显存。跑在 **seed 99** ⇒ run 目录（runs/<arm>-s99-0k）与结果文件
# （results_<arm>_s99.json）都与正式的 seed 0 分开，**不入判据**、也不会被 supervisor
# 误认成「该臂已完成」。
#
# 用法：STEPS=60 ARMS="EQ:99 PYR:99 HG:99 APYR:99" HOLD_ALL=1 bash smoke.sh
set -uo pipefail
cd "$(dirname "$0")"
export STEPS="${STEPS:-60}"
export ARMS="${ARMS:-EQ:99 PYR:99 HG:99 APYR:99}"
export HOLD_ALL="${HOLD_ALL:-1}"
export LOG_PREFIX="smoke"
bash ./train_arms.sh
