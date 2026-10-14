#!/usr/bin/env bash
# 2672 WP1b（接任段）— 6 个 run 的链式驱动：3 臂 × seed{0,1}，统一缩步数 STEPS（快照口径）。
# 逐 run 取放锁（SCHEMA §20）；到 DEADLINE（epoch 秒）后不再启动新 run（不杀进行中的 run，不杀他人进程）。
# 用法：STEPS=2000 DEADLINE=$(date -d '22:40' +%s) nohup bash run6.sh &
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/z/vibe/pathlm
STEPS="${STEPS:-2000}"
DEADLINE="${DEADLINE:-0}"
LOGD="$D/run_log.txt"
cd "$ROOT"

echo "[$(date -Is)] RUN6 start (pid $$) STEPS=$STEPS DEADLINE=$DEADLINE" >> "$LOGD"

# 顺序：先凑齐 P1 主判的 4 个 run（fill/nofill × seed{0,1}），再补 projfill 臂。
for spec in fill:0 nofill:0 fill:1 nofill:1 projfill:0 projfill:1; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  if [ "$DEADLINE" != "0" ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[$(date -Is)] RUN6 deadline reached before $ARM s$SEED — stop launching" >> "$LOGD"
    break
  fi
  if [ -f "$D/snap$((STEPS / 1000))k-${ARM}-s${SEED}/results.json" ]; then
    echo "[$(date -Is)] RUN6 skip $ARM s$SEED (results.json exists)" >> "$LOGD"
    continue
  fi
  STEPS="$STEPS" bash "$D/run_arm.sh" "$ARM" "$SEED" >> "$LOGD" 2>&1 \
    || echo "[$(date -Is)] RUN6 $ARM s$SEED nonzero exit (see snap*.train.log)" >> "$LOGD"
done

echo "[$(date -Is)] RUN6 training loop done" >> "$LOGD"

# 训练后收尾：等参数 rescore（需 GPU 锁）→ config 对平 → 汇聚。脚本缺失则跳过，由接管者补。
if [ -f "$D/finish.sh" ]; then
  bash "$D/finish.sh" >> "$LOGD" 2>&1
  echo "[$(date -Is)] RUN6 finish.sh done" >> "$LOGD"
fi
echo "[$(date -Is)] RUN6 exit" >> "$LOGD"
