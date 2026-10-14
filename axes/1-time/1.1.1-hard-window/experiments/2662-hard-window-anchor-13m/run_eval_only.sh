#!/usr/bin/env bash
# **已作废，勿启动**（lead steer 后由 `run_all_missing.sh` 取代）：本实验的判据是跨臂统计
# （`run.py:394`/`409-415` 硬点名锚点臂两 seed 都在场）⇒ 缺 `win16a_s1` 时只跑评测必 `KeyError`、
# 0 读数（实测证据：`armgap_probe/probe_med_stdout.txt`）。保留仅为留痕。
#
# 2662-hard-window-anchor-13m · eval-only 驱动：取 §20 GPU 锁 → `run.py` → 放锁。
# 臂集不缩：`config.json` 由 `make_config.py` 逐臂生成（本格盘上 = 4 臂 = `win16a_s0` + 3 复用臂，
# 缺的新臂记在 `missing_new_arms`）。取/放锁写法照抄同目录 `run_all.sh`，一处不同：
# **不启用**「无 holder_pid 且锁龄 >15min ⇒ 回收」这条 stale 规则——本仓有他线作业只写 `name`
# 不写 `holder_pid`，按锁龄回收会打断正在跑的作业（`SCHEMA.md` §20 互斥）。只回收
# 「holder_pid 存在且该进程已死」的锁。墙钟只作过程指标，不进判据。
# 用法：setsid nohup bash run_eval_only.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../../../../.."
EXP="axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOG="$EXP/driver_eval.log"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 90 * 60 ))

acquire() {
  local n=0 waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ $stale -eq 1 ]; then
      echo "[$(date -Iseconds)] reclaim: $reason" >> "$LOG"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); waited=$((waited+15))
    if [ $((n % 8)) -eq 0 ]; then echo "[$(date -Iseconds)] wait-lock ${waited}s holder=$hp" >> "$LOG"; fi
    if [ $waited -gt $MAX_WAIT_S ]; then
      echo "[$(date -Iseconds)] LOCK WAIT TIMEOUT (>90min) — 退让" >> "$LOG"
      return 9
    fi
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$" >> "$LOG"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released" >> "$LOG"
}

echo "[$(date -Iseconds)] DRIVER START pid=$$ arm_set=$(python3 -c "import json;c=json.load(open('$EXP/config.json'));print([a['name'] for a in c['arms']],c.get('missing_new_arms'))")" > "$LOG"
if acquire; then
  echo "[$(date -Iseconds)] EVAL START" >> "$LOG"
  python3 "$EXP/run.py" > "$EXP/run_log.txt" 2>&1
  rc=$?
  echo "[$(date -Iseconds)] EVAL done rc=$rc" >> "$LOG"
  release
else
  echo "[$(date -Iseconds)] EVAL SKIPPED（等锁超时，未跑 run.py）" >> "$LOG"
fi
echo "[$(date -Iseconds)] DRIVER END" >> "$LOG"
