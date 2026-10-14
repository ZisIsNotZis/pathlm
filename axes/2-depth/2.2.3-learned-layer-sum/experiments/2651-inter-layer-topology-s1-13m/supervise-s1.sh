#!/bin/bash
# 2651 · seed=1 无界 supervisor-retry 驱动（setsid nohup 后台）。
# 形态：不放弃重试；每轮调 run.py 按臂续跑 + 逐臂增量落盘；已训完的臂（sequential-12k-s1）
# 由 run.py::arm_trained 自动跳过重训（只重算电池，~3s）。
# 三条待训臂 = parallel / parallel-dual-ln / dense（seed=1，其余与 2613 s0 逐字段相同）。
# 注：run.py 的 --arms 校验要求必须含 sequential（判读需要它做基线）；sequential 已训完，
# 入选只是满足校验、不会重训。不用 --skip-s0-anchor 之外的额外开关。
# run.py 内部按 SCHEMA §20 取锁、§20.1 按臂释放（臂间 ≥60s 窗口）。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 用法：setsid nohup bash supervise-s1.sh >/dev/null 2>&1 &
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
  python3 "$EXP/run.py" --arms sequential,parallel,parallel-dual-ln,dense \
      --skip-s0-anchor --caps-min 14 --budget-min 42 >> "$LOG" 2>&1
  rc=$?
  echo "[sup] attempt $attempt rc=$rc end $(date -Iseconds)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "[sup] DONE rc=0 $(date -Iseconds)" >> "$LOG"
    break
  fi
  sleep 20
done
