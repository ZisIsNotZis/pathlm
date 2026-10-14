#!/usr/bin/env bash
# 轴 3 · GPU 作业 supervisor-retry 循环（Main 裁决 D1：不设配额，但 GPU 作业一律
# detached supervisor-retry —— 锁超时后继续重试、不放弃；§20.1 逐臂取/放锁由各实验的
# train_arms.sh 负责；本脚本只负责「哪一臂还没出结果就再排一次」）。
#
# 用法（detached）：
#   EXP=<实验目录> ARMS="U:0 PR:0" [STEPS=12000] [RFMT='results_%s_s%s.json'] \
#     [DEADLINE_EPOCH=<epoch>] [LOG=<日志路径>] \
#     setsid nohup bash .scratch/64-axis3-width/supervise_arms.sh >/dev/null 2>&1 &
#
# 退出条件：① 所有请求臂都出了结果文件；② 到达 DEADLINE_EPOCH（缺省 = 现在 +6h）。
# 记录：每轮 pending 与被跳过/失败的臂都写进 LOG（只增不改）。
set -uo pipefail
EXP="${EXP:?EXP 必给}"
ARMS="${ARMS:?ARMS 必给}"
STEPS="${STEPS:-12000}"
RFMT="${RFMT:-results_%s_s%s.json}"
LOG="${LOG:-$EXP/supervise.log}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-$(( $(date +%s) + 21600 ))}"
SLEEP_BETWEEN_ROUNDS="${SLEEP_BETWEEN_ROUNDS:-20}"

pending="$ARMS"
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Iseconds)] supervise start exp=$EXP arms='$ARMS' steps=$STEPS rfmt=$RFMT deadline=$(date -d @"$DEADLINE_EPOCH" -Iseconds)" >> "$LOG"

round=0
while :; do
  todo=""
  for s in $pending; do
    a="${s%%:*}"; sd="${s##*:}"
    f="$EXP/$(printf "$RFMT" "$a" "$sd")"
    [ -f "$f" ] || todo="$todo $s"
  done
  pending="${todo# }"
  [ -z "$pending" ] && { echo "[$(date -Iseconds)] supervise DONE (all arms have results)" >> "$LOG"; break; }
  if [ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]; then
    echo "[$(date -Iseconds)] supervise DEADLINE reached; pending='$pending'" >> "$LOG"; break
  fi
  round=$((round + 1))
  echo "[$(date -Iseconds)] round $round pending='$pending'" >> "$LOG"
  ARMS="$pending" STEPS="$STEPS" FAIL_FAST=0 bash "$EXP/train_arms.sh" >> "$LOG" 2>&1
  echo "[$(date -Iseconds)] round $round driver rc=$?" >> "$LOG"
  sleep "$SLEEP_BETWEEN_ROUNDS"
done
echo "[$(date -Iseconds)] supervise exit pending='$pending'" >> "$LOG"
