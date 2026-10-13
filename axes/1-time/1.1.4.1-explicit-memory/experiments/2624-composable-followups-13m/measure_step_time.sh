#!/usr/bin/env bash
# 步时先测（SCHEMA §17.8/§20.3）：200 步 × bs32 × T256 的实测步时，用来定预算。
# 只测、不训练到预算；产物落 /tmp/steptime2624（临时，不入库）。
set -u
cd "$(dirname "$0")/../../../../.."    # repo root
HERE=axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m
source "$HERE/gpu_lock.sh"

COMMON="--d 256 --layers 12 --heads 8 --seq-len 256 --dense \
--topology sequential --steps 200 --batch-size 32 --lr 1e-3 --seed 0 \
--synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
--out-root /tmp/steptime2624"

echo "== 跑前 nvidia-smi（free,used,util）: $(gpu_free_mib)"
acquire_lock 900 || exit 9
for spec in "st_gqa --mlp-mult 6 --n-kv-heads 2" \
            "st_gqa_carry --mlp-mult 6 --n-kv-heads 2 --carry-write delta" \
            "st_gqa_carry_window --mlp-mult 6 --n-kv-heads 2 --carry-write delta --train-window 16 --train-anchors 0" \
            "st_mha_mlp5 --mlp-mult 5"; do
  name=${spec%% *}; flags=${spec#* }
  echo "--- [$name] $(date -Is)"
  # shellcheck disable=SC2086
  python3 train_depth_ar.py "$name" $COMMON $flags 2>&1 | tail -2
done
release_lock
echo "== 跑后 nvidia-smi（free,used,util）: $(gpu_free_mib)"
python3 - <<'EOF'
import json, glob, os
for p in sorted(glob.glob('/tmp/steptime2624/*/train_log.jsonl')):
    rows=[json.loads(l) for l in open(p) if l.strip()]
    a=[r for r in rows if r['step']==200]
    if not a: continue
    r=a[0]
    # train_log 的 "min" = 累计分钟
    mins=r['min']; ms=1000.0*mins*60/200 if 200 else None
    print(f"{os.path.basename(os.path.dirname(p)):24s} step200 min={mins:.3f} "
          f"=> {ms:.1f} ms/step (mean over 200 steps)")
EOF
