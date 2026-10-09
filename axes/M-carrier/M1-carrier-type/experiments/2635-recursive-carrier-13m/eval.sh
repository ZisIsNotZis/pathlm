#!/usr/bin/env bash
# 2635-recursive-carrier-13m · 评测驱动（纯推理）：先回填 sha256，再冒烟，再全电池。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2635-recursive-carrier-13m
LOCK=/tmp/pathlm_gpu.lock

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

echo "=== 冒烟 $(date -Is) ==="
python3 "$EXP/run.py" --smoke --out-dir "$EXP/smoke/eval"
echo "=== 全电池 $(date -Is) ==="
python3 "$EXP/run.py" 2>&1 | tee "$EXP/run_log.txt"
echo "=== DONE $(date -Is) ==="
