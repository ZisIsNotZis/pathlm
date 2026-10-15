#!/bin/bash
# 2651 · seed=1 无界 supervisor-retry 驱动（setsid nohup 后台）。
# 排队形态：**快轮询抢窗口**——单次取锁 `--lock-wait-s 8 --lock-poll-s 5`（拿不到即返回，
# 不在前台 inline 死等）；外层每 5s 重排队，`while :` 不限轮数（不放弃重试）。
# 每臂结束由 run.py `gpu_lock_release(60)` 释放锁并留 ≥60s 窗口再取下一臂
# （`SCHEMA.md` §20.1）；**不**释放后 1s 抢回。
# 形态：不放弃重试；按臂续跑 + 逐臂增量落盘（run.py 每臂写 results.json + append
# run_log.txt）；已训完的臂（sequential-12k-s1）由 run.py::arm_trained 自动跳过重训
# （只重算电池，~3s）。
# 三条待训臂 = parallel / parallel-dual-ln / dense（seed=1，其余与 2613 s0 逐字段相同）。
# 注：run.py 的 --arms 校验要求必须含 sequential（判读需要它做基线）；sequential 已训完，
# 入选只是满足校验、不会重训。不用 --skip-s0-anchor 之外的额外开关。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 判活看末行横幅 `[sup] DONE rc=0`，不看进程名。
# 用法：setsid nohup bash supervise-s1.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$EXP/supervise.out"
echo $$ > "$EXP/supervise.pid"
attempt=0
while :; do
  attempt=$((attempt + 1))
  echo "[sup] attempt $attempt start $(date -Iseconds) holder=$(cat /tmp/pathlm_gpu.lock/holder_pid 2>/dev/null || echo none)" >> "$LOG"
  python3 "$EXP/run.py" --arms sequential,parallel,parallel-dual-ln,dense \
      --skip-s0-anchor --caps-min 14 --budget-min 42 \
      --lock-wait-s 8 --lock-poll-s 5 >> "$LOG" 2>&1
  rc=$?
  echo "[sup] attempt $attempt rc=$rc end $(date -Iseconds)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "[sup] DONE rc=0 $(date -Iseconds)" >> "$LOG"
    break
  fi
  sleep 5
done
