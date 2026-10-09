#!/usr/bin/env bash
# 2634-carry-realtext-13m · 评测驱动（纯推理）——**§20.1.2 短任务插队例外**。
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
EXP=axes/M-carrier/M1-carrier-type/experiments/2634-carry-realtext-13m

SMOKE=""
[ "${1:-}" = "--smoke" ] && SMOKE="--smoke"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits; }
FREE_BEFORE=$(free_mib)
T0=$(date +%s)
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
