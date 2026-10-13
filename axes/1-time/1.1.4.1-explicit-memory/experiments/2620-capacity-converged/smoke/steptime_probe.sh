#!/usr/bin/env bash
set -u
ROOT=/home/z/vibe/pathlm
HERE=$ROOT/axes/1-time/1.1.4.1-explicit-memory/experiments/2620-capacity-converged
LOCK=/tmp/pathlm_gpu.lock
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  stale=0; reason=""
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
    stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
  if [ $stale -eq 1 ]; then
    echo "reclaim: $reason"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 15; n=$((n+1)); [ $n -gt 40 ] && exit 9
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
echo "LOCK acquired $(date -Is) waited $((n*15))s"
cd "$ROOT"
for dh in 16 128; do
  /usr/bin/time -f "ARM dh=$dh wall_s=%e maxrss_kb=%M" -o "$HERE/smoke/time_dh${dh}.txt" \
    python3 train_depth_ar.py "smoke_dh$dh" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps 600 --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 \
    --carry-write delta --carry-state-dim "$dh" \
    --out-root "$HERE/smoke" > "$HERE/smoke/steptime_dh${dh}.log" 2>&1
  echo "--- dh=$dh: $(cat "$HERE/smoke/time_dh${dh}.txt")"
  python3 -c "
import json;d=json.load(open('$HERE/smoke/smoke_dh$dh/results.json'))
print('  wall_minutes',d.get('wall_minutes'),'peak_mb',d.get('peak_gpu_mem_mb'),'params',d.get('params'))"
done
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "LOCK released $(date -Is)"
