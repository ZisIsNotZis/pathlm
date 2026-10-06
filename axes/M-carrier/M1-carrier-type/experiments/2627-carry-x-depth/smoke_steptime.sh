#!/usr/bin/env bash
# 2627-carry-x-depth · 先测步时（SCHEMA §20.3 / §17.8）：四臂各 400 步，实测 ms/step。
# 取一次 GPU 锁（整程 <10 min，§20.1 允许整程持锁），结束后释放。
# 输出：$EXP/smoke/steps400/<arm>/train_log.jsonl（步时由 step0→step399 的 "min" 差算）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2627-carry-x-depth
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

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 400 --batch-size 32 --lr 1e-3 --seed 0 --train-window 16 --train-anchors 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --eval-batches 1 --out-root $EXP/smoke/steps400"

for spec in "base:" "carry:--carry-write delta" "depth:--dense-aux" "both:--carry-write delta --dense-aux"; do
  name=${spec%%:*}; flags=${spec#*:}
  /usr/bin/time -f "$name WALL %e s MAXRSS %M KB" python3 train_depth_ar.py "$name" $COMMON $flags 2>&1 | tail -1
  python3 - "$name" <<'PY'
import json,sys
a=sys.argv[1]
p=f"axes/M-carrier/M1-carrier-type/experiments/2627-carry-x-depth/smoke/steps400/{a}/train_log.jsonl"
rows=[json.loads(l) for l in open(p) if l.strip()]
lo,hi=rows[0],rows[-1]
ms=(hi['min']-lo['min'])*60000/max(hi['step']-lo['step'],1)
print(f"  {a:6s} {ms:7.1f} ms/step (step {lo['step']}->{hi['step']})")
PY
done

echo "DONE $(date -Is)"
