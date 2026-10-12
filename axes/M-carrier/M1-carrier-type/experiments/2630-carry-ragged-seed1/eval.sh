#!/usr/bin/env bash
# 2630-carry-ragged-seed1 · 评测驱动（纯推理）：先 `run.py --smoke`（冒烟，验接线）再全电池。
#
# 纪律（SCHEMA §20 / §20.1）：整程 <10 min ⇒ 允许整程持锁；结束后释放。
# 峰值 ≪5GB net（<1GB）；若共享卡被外部占用降速，如实记账（§20.3）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2630-carry-ragged-seed1
LOCK=/tmp/pathlm_gpu.lock

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

echo "=== 冒烟 $(date -Is) ==="
python3 "$EXP/run.py" --smoke
echo "=== 全电池 $(date -Is) ==="
python3 "$EXP/run.py"
echo "=== DONE $(date -Is) ==="
