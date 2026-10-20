#!/usr/bin/env bash
# 2654 supervisor-retry driver（detached 后台；Main 裁决的标准形态）。
#
#  1) 先跑 20 步冒烟（全四臂）作**闸门**：红则立即退出，不花 GPU 跑 12k。
#  2) 四臂 12k 训练+评测：**逐臂**取/放 GPU 锁（SCHEMA §20 / §20.1）——每臂结束必释放锁，
#     并留 SEP_SEC（缺省 60s）礼貌窗口再重取；长等待交给命令自己（不在前台 inline 死等）。
#  3) 有界重试：**轮**（枚举未完成臂 → 逐臂训练）由 `tools/supervise.sh --command-owns-lock`
#     驱动（`docs/methodology.md` §18.1）。锁在本脚本里按臂 mkdir/rmdir（`run.py` 内部不取锁），
#     所以 supervisor 不探测、不触碰锁，只用 `--attempts-max` / `--fail-max` 给轮数封顶。
#     轮数上限由本脚本自己的 deadline 推出：一轮至少要跑完一臂的 ARM_BUDGET_MIN，
#     所以最多 DEADLINE_MIN/ARM_BUDGET_MIN + 1 轮（缺省 240/30+1 = 9）；连续失败上限取同一个数，
#     因为本驱动「一轮没齐」是正常推进而不是崩溃，真正的界是总轮数。单次 acquire 仍最多
#     WAIT_MAX（缺省 600s = 10 min），拿不到就记一条并退到下一轮；deadline 到后日志留
#     `SUPERVISE GIVEUP`，剩下的空转轮很快跑完（不再取锁、不再训练），直到 supervisor 写 `FUSED`。
#  4) 判活看末行横幅 `SUPERVISE DONE` / `SUPERVISE GIVEUP`，以及 supervisor 的
#     `[sup] DONE rc=0` / `[sup] FUSED reason=...`；**不看进程名**。
#
# 环境变量：DEADLINE_MIN / WAIT_MAX / SEP_SEC / ARM_BUDGET_MIN / ATTEMPTS_MAX / FAIL_MAX / GAP_S
# 内部用法：`--round` 由 supervisor 每轮调用一次（跑一轮，四臂齐则退 0，否则退非 0 等下一轮）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$HERE"

LOCK=/tmp/pathlm_gpu.lock
LOG=run_log.txt
SELECTOR=wp2654
DEADLINE_MIN=${DEADLINE_MIN:-240}
WAIT_MAX=${WAIT_MAX:-600}
SEP_SEC=${SEP_SEC:-60}
ARM_BUDGET_MIN=${ARM_BUDGET_MIN:-30}
GAP_S=${GAP_S:-20}
ATTEMPTS_MAX=${ATTEMPTS_MAX:-$(( DEADLINE_MIN / ARM_BUDGET_MIN + 1 ))}
FAIL_MAX=${FAIL_MAX:-$ATTEMPTS_MAX}
ARMS=(stock skip-mid layers6 layerdrop-pt)
HELD=0
# deadline 从整条驱动起算，跨轮不变（`--round` 子进程由 supervisor 继承这个环境变量）
START_EPOCH=${START_EPOCH:-$(date +%s)}
export START_EPOCH

log() { echo "[$(date +%FT%T)] $*" >>"$LOG"; }   # 只写日志文件：stdout 会被 supervisor 再追加进同一文件，tee 会变成双份

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
      log "lock timeout: 等 $(holder_desc) > ${WAIT_MAX}s，交给下一轮重排"
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

deadline_hit() { [ $(( $(date +%s) - START_EPOCH )) -ge $(( DEADLINE_MIN * 60 )) ]; }

round_no() {                     # 轮次号从本脚本自己写的日志行推出，不新增状态文件
  local n
  n=$(grep -c '\] round [0-9]*：未完成 =' "$LOG" 2>/dev/null || true)
  echo $(( ${n:-0} + 1 ))
}

run_round() {                    # 0 = 四臂齐（已 aggregate）；非 0 = 这一轮没齐，等 supervisor 下一轮
  local ROUND REMAIN="" arm RC STOPPED=0
  ROUND=$(round_no)
  for arm in "${ARMS[@]}"; do arm_done "$arm" || REMAIN="$REMAIN $arm"; done
  if [ -z "$REMAIN" ]; then
    log "四臂齐（round $ROUND），aggregate"
    if acquire; then python3 run.py --aggregate >>"$LOG" 2>&1; release; fi
    log "SUPERVISE DONE"
    return 0
  fi
  log "round $ROUND：未完成 =$REMAIN"
  for arm in "${ARMS[@]}"; do
    arm_done "$arm" && continue
    if deadline_hit; then STOPPED=1; break; fi
    if acquire; then
      log "--- train arm: $arm (round $ROUND) ---"
      python3 run.py --arms "$arm" --budget-min "$ARM_BUDGET_MIN" >>"$LOG" 2>&1
      RC=$?
      log "arm $arm 结束 rc=$RC；释放锁后留 ${SEP_SEC}s 窗口"
      release; sleep "$SEP_SEC"
    fi
  done
  if [ "$STOPPED" -eq 1 ]; then
    log "DEADLINE ${DEADLINE_MIN}min 到：未完成 =$REMAIN；SUPERVISE GIVEUP"
  fi
  return 1
}

# ---- supervisor 每轮调用一次：只跑一轮 ----
if [ "${1:-}" = "--round" ]; then
  run_round
  exit $?
fi

log "=== supervise start (pid $$, deadline ${DEADLINE_MIN}min, wait ${WAIT_MAX}s/次, window ${SEP_SEC}s, 轮数上限 ${ATTEMPTS_MAX}) ==="
log "lock 现状：$(holder_desc)"
echo "2654 driver: 进度与横幅都写 $LOG"

# ---- 1) 冒烟闸门（全四臂；deadline 内重排，超期即放弃）----
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

# ---- 2) 逐臂训练+评测，直到四臂齐或超期：轮由 supervisor 封顶 ----
exec bash "$ROOT/tools/supervise.sh" --command-owns-lock \
  --attempts-max "$ATTEMPTS_MAX" --fail-max "$FAIL_MAX" \
  --lock-wait-s 0 --gap-s "$GAP_S" --log "$LOG" \
  -- bash "$HERE/run_supervise.sh" --round
