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
EXP=axes/M-carrier/M1-carrier-type/experiments/2633-carry-seed2

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
python3 - "$EXP" "$FREE_BEFORE" "$FREE_AFTER" "$((T1 - T0))" "$RC" <<'PY'
import json, os, sys
exp, fb, fa, wall, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
res_path = os.path.join(exp, "results.json")
peak = None
if os.path.exists(res_path):
    try:
        peak = json.load(open(res_path))["cost"]["eval_peak_gpu_mem_mb"]
    except Exception:
        peak = None
out = {
    "mode": "section_20_1_2_short_job_exception",
    "lock_taken": False,
    "lock_dir_touched": False,
    "why": ("训练按臂取/放锁（§20.1）；评测为纯推理、峰值 <1 GB、墙钟 <2.5 min，"
            "在外部线长时间占锁（等锁多次超时）时按 §20.1.2 插队执行"),
    "free_gpu_mem_mib_before": fb,
    "free_gpu_mem_mib_after": fa,
    "free_headroom_gate_mib": 3072,
    "headroom_ok": bool(min(fb, fa) >= 3072),
    "wall_s": wall,
    "eval_peak_gpu_mem_mb": peak,
    "returncode": rc,
    "note": ("若 wall_s > 120 或 eval_peak_gpu_mem_mb > 1024 ⇒ 已越过 §20.1.2 的字面门槛，"
             "必须在 analysis §20 显式登记为偏离（本文件即证据）"),
}
with open(os.path.join(exp, "lock_accounting.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(json.dumps(out, ensure_ascii=False))
PY
echo "=== DONE $(date -Is) rc=$RC ==="
