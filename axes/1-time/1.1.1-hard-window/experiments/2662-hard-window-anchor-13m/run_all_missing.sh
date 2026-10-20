#!/usr/bin/env bash
# 2662-hard-window-anchor-13m · 后台全流程 driver：**补训缺的第二臂 → 重生成 config → 评测**。
# 事由：本实验预登记的判据是跨臂统计（`run.py:394` / `409-415` 的校验硬点名锚点臂两 seed 都在场）
# ⇒ 缺 `win16a_s1` 时**只跑评测**必然 `KeyError`、0 读数；故先补训缺臂再评测。
#
# 顺序与纪律：
#   ① 取锁 → 训练 `win16a_s1`（CLI 与 `train_arms.sh` 的 `run_arm` 体**逐字段相同**，只换臂名与
#      `--seed 1`；`--out-root` 同 = `$EXP/train`）→ 放锁 → 臂间 ≥61 s（§20.1）；
#      本 driver 自持锁、等锁上限 90 min —— 不用 `train_arms.sh` 自带的 20 min 上限（该 20 min 上限
#      正是 `21:51:57` 丢掉 `win16a_s1` 的原因）；
#   ② `make_config.py --require-new`（逐臂 path + sha256），并校验 `missing_new_arms` 已空；
#      仍缺臂 ⇒ **不跑评测**（如实登记，不让 run.py 白占 GPU 后崩）；
#   ③ 取锁 → `python3 run.py` → 放锁。
# 取/放锁写法照抄 `run_all.sh`（每段取、跑完即放）；**不启用**「无 holder_pid 且锁龄 >15min ⇒ 回收」
# 那条 stale 规则——本仓有他线作业只写 `name` 不写 `holder_pid`，按锁龄回收会打断正在跑的作业
# （`harness/SCHEMA.md` §20 互斥）。只回收「holder_pid 存在且该进程已死」的锁。**不杀任何进程**。
# 墙钟只作过程指标，不进判据。
# 用法：setsid nohup bash run_all_missing.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../../../../.."
EXP="axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOG="$EXP/driver_all.log"
LOCK=/tmp/pathlm_gpu.lock
MAX_WAIT_S=$(( 90 * 60 ))

log() { echo "[$(date -Iseconds)] $*" >> "$LOG"; }

acquire() {
  local n=0 waited=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ $stale -eq 1 ]; then
      log "reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); waited=$((waited+15))
    if [ $((n % 8)) -eq 0 ]; then log "wait-lock ${waited}s holder=$hp"; fi
    if [ $waited -gt $MAX_WAIT_S ]; then
      log "LOCK WAIT TIMEOUT (>90min) — 退让"
      return 9
    fi
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  log "lock acquired pid=$$"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  log "lock released"
}

echo "[$(date -Iseconds)] DRIVER START pid=$$" > "$LOG"

# ---------------- ① 补训缺的第二臂 `win16a_s1`（seed 1）----------------
if acquire; then
  log "TRAIN START win16a_s1 seed=1 steps=12000 gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  python3 train_depth_ar.py win16a_s1 \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
    --lr 1e-3 --seed 1 --exit-mode uniform --train-window 16 --train-anchors 8 \
    --eval-batches 4 --out-root "$EXP/train" \
    > "$EXP/train_win16a_s1.log" 2>&1
  t_rc=$?
  release
  log "TRAIN done win16a_s1 rc=$t_rc"
  if [ $t_rc -ne 0 ]; then
    log "ABORT：补训失败（不生成 config、不排评测）"
    tail -20 "$EXP/train_win16a_s1.log" >> "$LOG"
    log "DRIVER END"
    exit $t_rc
  fi
  log "ARM GAP sleep 61s"
  sleep 61
else
  log "ABORT：训练段等锁超时（win16a_s1 未训）"
  log "DRIVER END"
  exit 9
fi

# ---------------- ② 重生成 config（逐臂 sha256）+ 校验缺臂已清零 ----------------
python3 "$EXP/make_config.py" --require-new >> "$LOG" 2>&1
log "make_config done rc=$?"
python3 - "$EXP/config.json" >> "$LOG" 2>&1 <<'PY'
import hashlib, json, os, sys
c = json.load(open(sys.argv[1]))
for a in c["arms"]:
    p = os.path.join("/home/z/vibe/pathlm", a["path"])
    h = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    print(f"[verify] {a['name']:10s} path={a['path']} sha256_ok={h == a['sha256']}")
print(f"[verify] arms={[a['name'] for a in c['arms']]}")
print(f"[verify] missing_new_arms={c['missing_new_arms']}")
sys.exit(0 if not c["missing_new_arms"] else 3)
PY
v_rc=$?
log "config verify rc=$v_rc"

# ---------------- ③ 评测（缺臂未清零则不排）----------------
if [ $v_rc -ne 0 ]; then
  log "EVAL SKIPPED：config 仍缺臂（missing_new_arms 非空），run.py 会 KeyError"
else
  if acquire; then
    log "EVAL START"
    python3 "$EXP/run.py" > "$EXP/run_log.txt" 2>&1
    e_rc=$?
    log "EVAL done rc=$e_rc"
    release
  else
    log "EVAL SKIPPED（等锁超时）"
  fi
fi
log "DRIVER END"
