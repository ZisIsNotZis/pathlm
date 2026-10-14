#!/bin/bash
# 2653 · 无界 supervisor-retry 驱动（setsid nohup 后台）。
# 形态：不放弃重试；每轮调 run.py 按臂续跑 + 逐臂增量落盘；已完成的臂自动跳过。
# run.py 内部按 SCHEMA §20 取锁、§20.1 按臂释放（臂间 ≥60s 窗口），本驱动只负责在最外层重试。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 用法：setsid nohup bash supervise.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$EXP/run_stdout-supervise.txt"
attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[sup] attempt $attempt start $(date -Iseconds)" >> "$LOG"
  python3 "$EXP/run.py" --arms attn-scalar,attn-qkv --seed 0 --steps 12000 \
      --lock-wait-s 600 --arm-gap-s 65 >> "$LOG" 2>&1
  rc=$?
  echo "[sup] attempt $attempt rc=$rc end $(date -Iseconds)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "[sup] DONE rc=0 $(date -Iseconds)" >> "$LOG"
    break
  fi
  sleep 20
done
