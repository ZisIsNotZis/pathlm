#!/bin/bash
# 2655 有界后台 launcher：按臂跑（run.py 内部按 §20 取锁、按 §20.1 按臂释放、臂间 ≥60 s）。
# 最多 8 轮，每轮最多 3 臂 × (10 min 等锁 + ~8 min 训练/评测)；已完成的臂自动跳过（可续跑）。
# 用法：setsid nohup bash run_remaining.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$EXP/run_stdout.txt"

for i in $(seq 1 8); do
  echo "[sh] attempt $i start $(date -Iseconds)" >> "$LOG"
  python3 "$EXP/run.py" --arms uniform,front-big,growing --seed 0 --steps 12000 \
      --lock-wait-s 600 --arm-gap-s 65 >> "$LOG" 2>&1
  rc=$?
  echo "[sh] attempt $i rc=$rc end $(date -Iseconds)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "[sh] DONE rc=0 $(date -Iseconds)" >> "$LOG"
    break
  fi
  sleep 30
done
echo "[sh] launcher exit $(date -Iseconds)" >> "$LOG"
