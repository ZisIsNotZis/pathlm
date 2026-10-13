#!/usr/bin/env bash
# 2633-carry-seed2 · 评测驱动（纯推理）——**§20.1.2 短任务插队例外**。
#
# 为什么不用 eval.sh 的「整程持锁」：本波评测期遇到外部线（R7/MoD 的 run_arms.sh）
# 长时间占锁、多次等锁超时（原始日志 `eval_*`）。本实验的评测是纯推理、<2 min、
# 峰值 <1 GB，满足 §20.1.2 的插队条件 ⇒ **不取锁、不动锁目录**，并在报告里登记例外。
# 前后各记一次空闲显存，并把结果落成**结构化产物** `lock_accounting.json`（供
# `results.json::cost.lock_accounting` 引用）。
#
# 用法：bash eval_concurrent.sh [--smoke]
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2633-carry-seed2

SMOKE=""
[ "${1:-}" = "--smoke" ] && SMOKE="--smoke"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
FREE_BEFORE=$(free_mib)
T0=$(date +%s)
# **§9 复用前置门**（与 eval.sh 一致）：verdict 非 reuse_ok 系 ⇒ 不评测，回退自包含双跑。
# 注意：`run.py` 本身**不**读 verdict（它是评测入口，不是门）；门在这里与 equality_vs_2631.py
# 的返回码处。
python3 "$EXP/equality_vs_2631.py" || { echo "§9 复用证据不通过 ⇒ 回退自包含双跑"; exit 1; }
python3 "$EXP/fill_config_hashes.py" || { echo "权重未齐（先训练）"; exit 1; }
python3 "$EXP/run.py" $SMOKE 2>&1 | tee "$EXP/run_log.txt"
RC=$?
T1=$(date +%s)
FREE_AFTER=$(free_mib)
# 记账：**同一处**写 lock_accounting.json + 刷新 results.json 的 accounting **与 note**，
# 并在覆写后自检口径一致（复核 P1：驱动事后覆写不得只改一半）
python3 "$EXP/lock_accounting.py" record --mode section_20_1_2_short_job_exception \
  --free-before "$FREE_BEFORE" --free-after "$FREE_AFTER" --wall-s "$((T1 - T0))" --rc "$RC"
echo "=== DONE $(date -Is) rc=$RC ==="
