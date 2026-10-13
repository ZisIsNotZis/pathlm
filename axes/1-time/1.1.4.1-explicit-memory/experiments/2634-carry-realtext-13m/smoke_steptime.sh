#!/usr/bin/env bash
# 2634-carry-realtext-13m · 先测步时（SCHEMA §17.8 / §20.3）：本实验的两个真实文本臂
# 各跑 300 步，实测稳态 ms/step 与峰值显存，**再由本数字定全跑预算**（不借用 2618/2631）。
# 输出：smoke/steps300/<arm>/{train_log.jsonl,gate_log.jsonl}；ms/step 在 stdout。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m

BASE="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
--steps 300 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform --eval-batches 1 \
--train-window 16 --train-anchors 0 --out-root $EXP/smoke/steps300"

echo "GPU free before: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
for spec in "rt_none:" "rt_delta:--carry-write delta"; do
  name=${spec%%:*}; flags=${spec#*:}
  /usr/bin/time -f "$name WALL %e s MAXRSS %M KB" \
    python3 train_depth_ar.py "$name" $BASE $flags 2>&1 | tail -1
  python3 - "$EXP/smoke/steps300/$name" "$name" <<'PY'
import json,sys,os
log=os.path.join(sys.argv[1],"train_log.jsonl")
rows=[json.loads(l) for l in open(log) if l.strip()]
lo,hi=rows[0],rows[-1]
ms=(hi['min']-lo['min'])*60000/max(hi['step']-lo['step'],1)
res=json.load(open(os.path.join(sys.argv[1],"results.json")))
print(f"  {sys.argv[2]:10s} {ms:7.1f} ms/step (step {lo['step']}->{hi['step']}) "
      f"peak {res['peak_gpu_mem_mb']} MB wall {res['wall_minutes']:.2f} min")
PY
done
echo "GPU free after: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo "DONE $(date -Is)"
