#!/usr/bin/env bash
# 2623 步时探针：**先测步时再定预算**（SCHEMA §17.8；不外推训练时间字段）。
# 对两臂各跑 PROBE 步（与正式训练同配置），量 ms/step 与峰值显存。
# 用法： ./smoke/steptime_probe.sh            # 缺省 PROBE=400
#        PROBE=600 ./smoke/steptime_probe.sh
set -u
ROOT=/home/z/vibe/pathlm
HERE="$ROOT/axes/A-mixing/A9-old-info-retention/experiments/2623-window-on-longrange-13m"
LOCK=/tmp/pathlm_gpu.lock
PROBE="${PROBE:-400}"
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
for arm in full win16; do
  WINFLAGS=""
  [ "$arm" = "win16" ] && WINFLAGS="--train-window 16 --train-anchors 0"
  python3 train_depth_ar.py "probe_${arm}" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --seq-len 256 \
    --steps "$PROBE" --batch-size 32 --seed 0 --lr 0.001 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    $WINFLAGS \
    --out-root "$HERE/smoke" > "$HERE/smoke/probe_${arm}.log" 2>&1
  python3 -c "
import json
d=json.load(open('$HERE/smoke/probe_${arm}/results.json'))
rows=[json.loads(l) for l in open('$HERE/smoke/probe_${arm}/train_log.jsonl') if l.strip()]
last=rows[-1]; ms=1000*60*last['min']/last['step']
print('ARM $arm: steps=%d ms/step=%.1f wall_min=%.2f peak_mb=%.0f params=%d'
      % (last['step'], ms, d['wall_minutes'], d['peak_gpu_mem_mb'], d['params']))
print('  last_window_mask=%s' % (last.get('window_mask')))
"
done
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
trap - EXIT
echo "LOCK released $(date -Is)"
