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
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo "等锁超时"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT

echo "=== 冒烟 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
python3 "$EXP/run.py" --smoke --out-dir "$EXP/smoke/eval"
echo "=== 全电池 $(date -Is) ==="
python3 "$EXP/run.py" 2>&1 | tee "$EXP/run_log.txt"
echo "=== DONE $(date -Is) ==="
