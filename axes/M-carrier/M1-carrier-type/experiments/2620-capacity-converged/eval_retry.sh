#!/usr/bin/env bash
# 2620：评测/分析的**持久**驱动（持 /tmp/pathlm_gpu.lock）。
#
# 为什么单独有这个脚本：本实验的评测只需要 ~2–3 分钟 GPU，但共享卡上还有 V3 等长线；
# 按 SCHEMA §20 的「最多等 10 分钟」规则，单次等待很容易在别人的整段训练里超时。
# 本脚本把「取锁失败」变成**有界重试**（最多 N 次 × 每次 10 分钟），
# 每次内部用 **5 秒**轮询以及早抓住别人「按臂释放」的窗口。
#
# 锁协议（SCHEMA §20 修正版）：
#   * 回收只用 holder_pid 存活性（无 holder_pid 的老锁按锁龄 >15min），**不用 pgrep**；
#   * 回收/释放一律先 `rm -f` 内容再 `rmdir`；循环路径上始终有 sleep；
#   * **取锁成功的判定 = `mkdir` 成功那一刻**（不是事后看目录里有没有 holder_pid）——
#     2026-10-02 23:26 本脚本的上一版把判定写在写 holder_pid 之前，导致「自己创建了锁
#     目录、又判定自己没拿到」⇒ 空锁泄漏 5 分钟，阻塞其他线；已按下述结构修掉，
#     并把这次事故登记进 `analysis.md` §GPU 锁台账。
#
# 用法： ./eval_retry.sh [attempts]      # 默认 8 次尝试
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOCK=/tmp/pathlm_gpu.lock
ATTEMPTS="${1:-8}"
[ "$#" -gt 0 ] && shift

HOLDING=0
release_lock() {
  if [ "$HOLDING" -eq 1 ]; then
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null
    HOLDING=0
    echo "[$(date -Is)] GPU LOCK released"
  fi
}
trap 'release_lock' EXIT

cd "$ROOT"
for attempt in $(seq 1 "$ATTEMPTS"); do
  got=0; n=0
  while [ "$got" -eq 0 ]; do
    if mkdir "$LOCK" 2>/dev/null; then got=1; break; fi
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    stale=0; reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date -Is)] reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 5; n=$((n+1))
    if [ $n -gt 120 ]; then
      echo "[$(date -Is)] attempt $attempt/$ATTEMPTS: TIMEOUT after $((n*5))s (holder=${hp:-<missing>})"
      break
    fi
  done
  if [ "$got" -eq 0 ]; then sleep 20; continue; fi
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"; HOLDING=1
  echo "[$(date -Is)] GPU LOCK acquired (attempt $attempt, waited $((n*5))s, holder $$)"
  python3 "$HERE/run.py" "$@" 2>&1 | tee "$HERE/run_log.txt" | tail -80
  rc=${PIPESTATUS[0]}
  release_lock
  echo "[$(date -Is)] run.py rc=$rc"
  exit "$rc"
done
echo "[$(date -Is)] 全部 $ATTEMPTS 次取锁尝试均超时"
exit 9
