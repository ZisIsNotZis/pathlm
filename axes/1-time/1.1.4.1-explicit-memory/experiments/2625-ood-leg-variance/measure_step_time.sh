#!/usr/bin/env bash
# 2625-ood-leg-variance · 步时先测（SCHEMA §17.8/§20.3）：200 步 × bs32 × T256 的实测步时，
# 用来定预算（**先测步时再定预算**）。只测、不训练到预算；产物落 /tmp（临时，不入库）。
set -u
cd "$(dirname "$0")/../../../../.."    # repo root
HERE=axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; continue; fi
    sleep 15; n=$((n+1)); [ $n -gt 40 ] && { echo 等锁超时; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; }

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 200 --batch-size 32 --lr 1e-3 --seed 1 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 --train-anchors 0 --n-kv-heads 2 --out-root /tmp/steptime2625"

echo "== 跑前 nvidia-smi: $(nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader) $(date -Is)"
acquire
echo "== 取锁 OK $(date -Is)"
python3 train_depth_ar.py st_gqa_carry_win $COMMON 2>&1 | tail -3
release
echo "== 跑后 nvidia-smi: $(nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader) $(date -Is)"
python3 - <<'EOF'
import json, glob, os
for p in sorted(glob.glob('/tmp/steptime2625/*/train_log.jsonl')):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    if not rows: continue
    r=rows[-1]
    n_steps=r['step']+1
    ms=1000.0*r['min']*60/n_steps
    print(f"{os.path.basename(os.path.dirname(p)):24s} step{n_steps-1} min={r['min']:.3f} => {ms:.1f} ms/step")
for p in sorted(glob.glob('/tmp/steptime2625/*/results.json')):
    d=json.load(open(p))
    print(os.path.basename(os.path.dirname(p)), "wall_min=", d.get("wall_minutes"), "peak_mb=", d.get("peak_gpu_mem_mb"))
EOF
