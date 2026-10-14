#!/usr/bin/env bash
# 2642 · 本 WP 的 GPU 批次驱动（一个后台 job 串起：WD256 微冒烟 → WP1 两臂 → WP2 一臂）。
# 失败不阻断后续（FAIL_FAST=0）：锁很贵，一次拿到就把它用满。
set -uo pipefail
E="$(cd "$(dirname "$0")" && pwd)"
export FAIL_FAST=0
STEPS=60   ARMS="WD256:99"               LOG_PREFIX=smoke bash "$E/train_arms.sh"
STEPS=12000 ARMS="D192:0 D128:0"          LOG_PREFIX=train bash "$E/train_arms.sh"
STEPS=12000 ARMS="WD256:0"                LOG_PREFIX=train bash "$E/train_arms.sh"
echo "[$(date -Iseconds)] run_batch done" >> "$E/run_log.txt"
