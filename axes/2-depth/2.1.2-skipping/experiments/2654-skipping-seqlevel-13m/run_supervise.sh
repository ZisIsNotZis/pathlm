#!/usr/bin/env bash
# 2654 supervisor-retry driver（detached 后台；Main 裁决的标准形态）。
#
#  1) 先跑 20 步冒烟（全四臂）作**闸门**：红则立即退出，不花 GPU 跑 12k。
#  2) 四臂 12k 训练+评测：**逐臂**取/放 GPU 锁（SCHEMA §20 / §20.1）——每臂结束必释放锁，
#     并留 SEP_SEC（缺省 60s）礼貌窗口再重取；长等待交给本循环（不在前台 inline 死等）。
#  3) 有界重试：单次 acquire 最多 WAIT_MAX（缺省 600s = 10 min），拿不到就记一条并重排队
#     （30s 后）；总预算 DEADLINE_MIN（缺省 240 min）。四臂齐则 aggregate 收口。
#  4) 判活看末行横幅 `SUPERVISE DONE` / `SUPERVISE GIVEUP`，**不看进程名**。
#
# 环境变量：DEADLINE_MIN / WAIT_MAX / SEP_SEC / ARMS_OVERRIDE
set -u
cd "$(dirname "$0")"

LOCK=/tmp/pathlm_gpu.lock
LOG=run_log.txt
SELECTOR=wp2654
DEADLINE_MIN=${DEADLINE_MIN:-240}
WAIT_MAX=${WAIT_MAX:-600}
SEP_SEC=${SEP_SEC:-60}
ARMS=(stock skip-mid layers6 layerdrop-pt)
HELD=0
START=$SECONDS

log() { echo "[$(date +%FT%T)] $*" | tee -a "$LOG"; }

holder_desc() {
  local hp name; hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
  name=$(cat "$LOCK/name" 2>/dev/null || true)
  if [ -n "$hp" ]; then echo "holder_pid=$hp name=${name:-?}"; else echo "无 holder_pid"; fi
}

acquire() {                      # 0 = 拿到锁；9 = 单次等锁超时
  local t0=$SECONDS hp age
  while ! mkdir "$LOCK" 2>/dev/null; do
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then
      log "reclaim: holder_pid $hp 已死"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null; continue
    fi
    if [ -z "$hp" ] && [ -d "$LOCK" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || echo $(date +%s)) ))
      if [ "$age" -gt 900 ]; then
        log "reclaim: 无 holder_pid 且锁龄 ${age}s > 900s"
        rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
        rmdir "$LOCK" 2>/dev/null; continue
      fi
    fi
    if [ $(( SECONDS - t0 )) -ge "$WAIT_MAX" ]; then
      log "lock timeout: 等 $(holder_desc) > ${WAIT_MAX}s，30s 后重排队"
      return 9
    fi
    sleep 15
  done
  echo $$ > "$LOCK/holder_pid"; echo "$SELECTOR" > "$LOCK/name"; touch "$LOCK/heartbeat"
  HELD=1
  log "lock acquired（$(holder_desc)）"
  return 0
}

release() {                      # 只在真持锁时释放（否则会误删他人 holder_pid）
  [ "$HELD" -eq 1 ] || return 0
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" "$LOCK/name" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  HELD=0
}
trap release EXIT

arm_done() {                     # 1 = results.json 已含该臂
  python3 - "$1" <<'PY'
import json, os, sys
p = "results.json"
if not os.path.exists(p):
    sys.exit(1)
try:
    r = json.load(open(p))
except Exception:
    sys.exit(1)
sys.exit(0 if sys.argv[1] in r.get("arms", {}) else 1)
PY
}

deadline_hit() { [ $(( (SECONDS - START) / 60 )) -ge "$DEADLINE_MIN" ]; }

log "=== supervise start (pid $$, deadline ${DEADLINE_MIN}min, wait ${WAIT_MAX}s/次, window ${SEP_SEC}s) ==="
log "lock 现状：$(holder_desc)"

# ---- 1) 冒烟闸门（全四臂）----
SMOKE_OK=0
while [ "$SMOKE_OK" -eq 0 ]; do
  if acquire; then
    log "--- smoke 闸门：python3 run.py --smoke --steps 20 ---"
    if python3 run.py --smoke --steps 20 >>"$LOG" 2>&1; then
      SMOKE_OK=1; log "SMOKE PASSED（四臂活性门全绿）"
    else
      log "SMOKE FAILED（驱动退出，不花 GPU 跑 12k）"
      release; exit 1
    fi
    release; sleep "$SEP_SEC"
  else
    sleep 30
  fi
  if deadline_hit; then log "DEADLINE ${DEADLINE_MIN}min 到于冒烟前；SUPERVISE GIVEUP"; exit 9; fi
done

# ---- 2) 逐臂训练+评测，直到四臂齐或超期 ----
ROUND=0
while :; do
  ROUND=$((ROUND + 1))
  REMAIN=""
  for arm in "${ARMS[@]}"; do arm_done "$arm" || REMAIN="$REMAIN $arm"; done
  if [ -z "$REMAIN" ]; then
    log "四臂齐（round $ROUND），aggregate"
    if acquire; then python3 run.py --aggregate >>"$LOG" 2>&1; release; fi
    log "SUPERVISE DONE"; break
  fi
  log "round $ROUND：未完成 =$REMAIN"
  STOPPED=0
  for arm in "${ARMS[@]}"; do
    arm_done "$arm" && continue
    if deadline_hit; then STOPPED=1; break; fi
    if acquire; then
      log "--- train arm: $arm (round $ROUND) ---"
      python3 run.py --arms "$arm" --budget-min 30 >>"$LOG" 2>&1
      RC=$?
      log "arm $arm 结束 rc=$RC；释放锁后留 ${SEP_SEC}s 窗口"
      release; sleep "$SEP_SEC"
    fi
  done
  if [ "$STOPPED" -eq 1 ]; then
    log "DEADLINE ${DEADLINE_MIN}min 到：未完成 =$REMAIN；SUPERVISE GIVEUP"; break
  fi
  sleep 20
done
