#!/usr/bin/env bash
# 2633-carry-seed2 · 评测驱动（纯推理）：先回填三臂新 checkpoint 的 sha256（并核 §9 复用证据），
# 再冒烟，再全电池（九臂一次评完 ⇒ 主腿/毁源腿逐臂配对）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2633-carry-seed2
LOCK=/tmp/pathlm_gpu.lock

python3 "$EXP/equality_vs_2631.py" || { echo "§9 复用证据不通过 ⇒ 回退自包含双跑"; exit 1; }
python3 "$EXP/fill_config_hashes.py" || { echo "权重未齐（先训练）"; exit 1; }

n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null); stale=0
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
  if [ "$stale" -eq 1 ]; then rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; continue; fi
  sleep 15; n=$((n+1));
  if [ $((n % 8)) -eq 0 ]; then echo "  [wait-lock] 已等 $((n*15))s，holder=$hp $(date -Is)"; fi
  [ $n -gt 160 ] && { echo "等锁超时（>40min）"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT

FREE_BEFORE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
T0=$(date +%s)
echo "=== 冒烟 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
python3 "$EXP/run.py" --smoke --out-dir "$EXP/smoke/eval"
echo "=== 全电池 $(date -Is) ==="
python3 "$EXP/run.py" 2>&1 | tee "$EXP/run_log.txt"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
FREE_AFTER=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
# §20 记账：本驱动走「整程持锁」口径 ⇒ mode=section_20_lock；并回填 results.json（复核 P2-2）
python3 - "$EXP" "$FREE_BEFORE" "$FREE_AFTER" "$((T1 - T0))" "$RC" <<'PYEOF'
import json, os, sys
exp, fb, fa, wall, rc = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
res_path = os.path.join(exp, "results.json")
peak = None
if os.path.exists(res_path):
    try:
        peak = json.load(open(res_path))["cost"]["eval_peak_gpu_mem_mb"]
    except Exception:
        peak = None
out = {"mode": "section_20_lock", "lock_taken": True, "lock_dir_touched": True,
       "why": "评测驱动 eval.sh：整程持锁（本驱动 <10 min）；免费额度前后各记一次",
       "free_gpu_mem_mib_before": fb, "free_gpu_mem_mib_after": fa,
       "free_headroom_gate_mib": 3072, "headroom_ok": bool(min(fb, fa) >= 3072),
       "wall_s": wall, "eval_peak_gpu_mem_mb": peak, "returncode": rc,
       "note": "wall_s 含 §9 前置门 + 冒烟 + 全电池；若 >120 s 属超 §20.1.2 字面门槛的整程持锁，已登记"}
with open(os.path.join(exp, "lock_accounting.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False); f.write("\n")
if os.path.exists(res_path):
    res = json.load(open(res_path))
    res.setdefault("cost", {})["lock_accounting"] = out
    with open(res_path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("results.json::cost.lock_accounting 已同步")
print(json.dumps(out, ensure_ascii=False))
PYEOF
echo "=== DONE $(date -Is) rc=$RC ==="
