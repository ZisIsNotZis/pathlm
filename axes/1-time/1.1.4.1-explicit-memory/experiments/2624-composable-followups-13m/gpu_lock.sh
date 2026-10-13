#!/usr/bin/env bash
# 2624-composable-followups-13m · GPU 互斥锁助手（SCHEMA §20 / §20.1）。
#
# 为什么单独抽出来：SCHEMA §20 的示例用 `while ! mkdir ...; do ...; done && ...`
# 这种写法在 bash 里**有坑**——循环体最后一条命令是 `[ $n -gt 8 ] && {...}`，
# 条件为假时循环的退出码为 1 ⇒ `&&` 后面的命令整条被跳过（本线在 2026-10-03
# 实测踩到一次：mkdir 成功但 holder_pid 没写、后续训练整段没跑）。
# 本助手把「取锁 → 写 holder_pid → 返回 0/9」写成显式 `if` 分支，退出码确定。
#
# 用法：source gpu_lock.sh; acquire_lock [wait_secs]; ... ; release_lock
LOCK=${LOCK:-/tmp/pathlm_gpu.lock}

acquire_lock() {
  local wait=${1:-1800} deadline=$(( $(date +%s) + ${1:-1800} ))
  while :; do
    if mkdir "$LOCK" 2>/dev/null; then
      echo $$ > "$LOCK/holder_pid"
      touch "$LOCK/heartbeat"
      echo "[lock] acquired by $$ $(date -Is)"
      return 0
    fi
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] \
       && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then
      echo "[lock] reclaim stale lock (holder=${hp:-none}) $(date -Is)"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[lock] TIMEOUT after ${wait}s (holder=${hp:-none}) $(date -Is)"
      return 9
    fi
    echo "[lock] waiting (holder=${hp:-none}) $(date -Is)"
    sleep 15
  done
}

release_lock() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null || true
  echo "[lock] released by $$ $(date -Is)"
}

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free,memory.used,utilization.gpu --format=csv,noheader
}
