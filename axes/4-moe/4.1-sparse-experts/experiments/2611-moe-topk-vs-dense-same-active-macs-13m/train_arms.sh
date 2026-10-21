#!/usr/bin/env bash
# 2611 的逐臂训练驱动：**每臂**取/放 GPU 互斥锁，臂间留 61 s（harness/SCHEMA.md §20 / §20.1）。
# 用法：bash train_arms.sh <臂名> [<臂名> …]        （臂名见 config.json::arms）
#
# 单臂 timeout 20 min（README §6）：超时 ⇒ 该臂登记「未跑」，不推断（掉臂规则）。
# 不做重试循环：失败即停，由人/主 Agent 决定是否重跑该臂（methodology §18.1 的熔断纪律）。
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock
LOG="$EXP/train_arms.out"

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    stale=0
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ $stale -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15
    n=$((n + 1))
    if [ $n -gt 40 ]; then echo "[drv] 等锁超过 10 min，放弃（不杀别人的进程）" | tee -a "$LOG"; exit 9; fi
  done
  echo $$ > "$LOCK/holder_pid"
  touch "$LOCK/heartbeat"
}

release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null || true
}

trap release EXIT
cd "$ROOT" || exit 1
{
  echo "[drv] start $(date -Iseconds)  arms: $*"
} >> "$LOG"

for ARM in "$@"; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "[drv] $ARM: 取锁前 GPU 空闲 ${free_mib} MiB  $(date -Iseconds)" | tee -a "$LOG"
  # 外部占用（更高优先级项目）留下的余量不足 8 GB ⇒ 排队等锁后复查，不抢不杀（README §6）
  while [ "${free_mib:-0}" -lt 8000 ]; do
    echo "[drv] $ARM: 空闲 ${free_mib} MiB < 8 GB ⇒ 等 60 s 再查（让路给更高优先级项目）" | tee -a "$LOG"
    sleep 60
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  done
  acquire
  echo "[drv] $ARM: 已取锁 holder=$$  $(date -Iseconds)" | tee -a "$LOG"
  timeout 1200 python3 "$EXP/run.py" --stage train --arm "$ARM" >> "$LOG" 2>&1
  rc=$?
  echo "[drv] $ARM: exit $rc  $(date -Iseconds)  peak/free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ') MiB" | tee -a "$LOG"
  release
  if [ $rc -ne 0 ]; then
    echo "[drv] $ARM 失败（rc=$rc）⇒ 停在这里，余下臂未跑" | tee -a "$LOG"
    exit $rc
  fi
  sleep 61
done
echo "[drv] done $(date -Iseconds)" | tee -a "$LOG"
