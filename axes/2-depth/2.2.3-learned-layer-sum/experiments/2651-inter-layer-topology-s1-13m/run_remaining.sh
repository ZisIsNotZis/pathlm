#!/bin/bash
# 2651 seed=1 · 剩余三臂（parallel / parallel-dual-ln / dense）+ run.py train_arm 修复冒烟
# 增量落盘（run.py 每臂写 results.json + append run_log.txt）；GPU 锁按 §20 取、按臂释放。
# run.py 用仓库根相对路径读数据/数据切分 ⇒ 必须在仓库根运行。
# 用法：setsid nohup bash run_remaining.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: > "$EXP/run_stdout-smoke.txt"
: > "$EXP/run_stdout-full.txt"
ARMS="sequential,parallel,parallel-dual-ln,dense"

for i in $(seq 1 12); do
  if [ ! -f "$EXP/.smoke_ok" ]; then
    echo "[sh] smoke attempt $i $(date -Iseconds)" >> "$EXP/run_stdout-smoke.txt"
    if python3 "$EXP/run.py" --steps 20 --arms sequential,parallel \
         --skip-s0-anchor --caps-min 6 --poll-s 15 >> "$EXP/run_stdout-smoke.txt" 2>&1; then
      touch "$EXP/.smoke_ok"
    fi
    echo "[sh] smoke attempt $i smoke_ok=$([ -f "$EXP/.smoke_ok" ] && echo yes || echo no)" \
         "$(date -Iseconds)" >> "$EXP/run_stdout-smoke.txt"
    sleep 15
    continue
  fi
  echo "[sh] full attempt $i $(date -Iseconds)" >> "$EXP/run_stdout-full.txt"
  python3 "$EXP/run.py" --arms "$ARMS" --caps-min 14 --budget-min 40 \
      >> "$EXP/run_stdout-full.txt" 2>&1
  rc=$?
  echo "[sh] full attempt $i rc=$rc $(date -Iseconds)" >> "$EXP/run_stdout-full.txt"
  if [ $rc -eq 0 ]; then
    echo "[sh] DONE rc=0 $(date -Iseconds)" >> "$EXP/run_stdout-full.txt"
    break
  fi
  sleep 20
done
echo "[sh] launcher exit $(date -Iseconds)" >> "$EXP/run_stdout-full.txt"
