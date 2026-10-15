#!/bin/bash
# 2653 · 无界 supervisor-retry 驱动（setsid nohup 后台）。
# 排队形态：**快轮询抢窗口**——单次取锁 max_wait ≤ 8s（`gpu_lock` 内部每 2s 试一次），
# 拿不到就立刻返回；外层每 5s 重排队，`while :` 不限轮数（不放弃重试）。
# 每臂成功结束由 `gpu_lock.release()` 释放锁，run.py 再 `--arm-gap-s 65` 留 ≥60s 窗口
# 才取下一臂（`SCHEMA.md` §20.1）；**不**释放后 1s 抢回。
# 形态：不放弃重试；按臂续跑 + 逐臂增量落盘（results_<arm>_s<seed>.json 存在则跳过）。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 判活看末行横幅 `[sup] DONE rc=0`，不看进程名。
# 用法：setsid nohup bash supervise.sh >/dev/null 2>&1 &
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
  python3 "$EXP/run.py" --arms attn-scalar,attn-qkv --seed 0 --steps 12000 \
      --lock-wait-s 8 --arm-gap-s 65 >> "$LOG" 2>&1
  rc=$?
  echo "[sup] attempt $attempt rc=$rc end $(date -Iseconds)" >> "$LOG"
  if [ $rc -eq 0 ]; then
    echo "[sup] DONE rc=0 $(date -Iseconds)" >> "$LOG"
    break
  fi
  sleep 5
done
