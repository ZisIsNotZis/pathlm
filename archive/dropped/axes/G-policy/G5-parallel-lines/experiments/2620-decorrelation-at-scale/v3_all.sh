#!/usr/bin/env bash
# 2620 · 去相关的规模验证（契约 V3）——冒烟 → 预算判定 → 三臂全跑 → 评测 → 对账
#
# GPU 锁协议：SCHEMA §20（已修版：释放/回收一律先 `rm -f` 内容再 `rmdir`；回收只认
# `holder_pid` 存活性，不用 pgrep）**+ §20.1 锁粒度**（2026-10-02 事故后新增）：
#   · 多臂驱动**按臂释放/重取锁**，每臂之间留 **≥60 s 窗口**（不得整程独占）；
#   · 只有整程 <10 min 的阶段（本脚本 = 冒烟 2 min、评测 5 min）才允许整程持锁。
# ⚠️ 本脚本已按 §20.1 改写；**2026-10-02 的首跑是整程持锁的**（§20.1 在首跑进行中才入宪），
#    该事实与「是否影响结论」记在 `analysis.md` §11。
#
# 用法：
#   nohup bash v3_all.sh > run_all.log 2>&1 &
# 环境变量：STEPS_FULL（默认 12000）、SKIP_EVAL=1、SKIP_BATTERY=1
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale
LOCK=/tmp/pathlm_gpu.lock
ARMS="stock l1 divl1"
SMOKE_STEPS=400
STEPS_FULL="${STEPS_FULL:-12000}"
BUDGET_MIN=115
EVAL_EST_MIN=8
HB=""

cd "$ROOT" || exit 1

acquire_lock() {
  local n=0 hp stale reason
  while ! mkdir "$LOCK" 2>/dev/null; do
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    stale=0; reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "reclaim: $reason"; rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true; continue
    fi
    sleep 15; n=$((n+1))
    [ $((n % 8)) -eq 0 ] && echo "waiting for GPU lock: $((n*15))s (holder=$hp)"
    [ $n -gt 720 ] && { echo "LOCK TIMEOUT after $((n*15))s (holder=$hp)"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
  ( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) & HB=$!
  echo "LOCK acquired $(date -Is) waited=$((n*15))s"
}

release_lock() {
  [ -n "$HB" ] && kill "$HB" 2>/dev/null; HB=""
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
  trap - EXIT
  echo "LOCK released $(date -Is)"
}

mkdir -p "$EXP/smoke" "$EXP/train" "$ROOT/.tmp/v3-ckpts"

# ---------------- ① 冒烟（整段 ~2 min < 10 min ⇒ 允许整程持锁） ----------------
acquire_lock || exit 9
echo "=== SMOKE ($SMOKE_STEPS steps/arm, bs16) ==="
for arm in $ARMS; do
  python3 train_m1.py "smoke_$arm" --config "$EXP/configs/$arm.json" \
      --steps "$SMOKE_STEPS" --out-root "$EXP/smoke" > "$EXP/smoke/smoke_$arm.log" 2>&1
  echo "smoke $arm exit=$?"
  [ -f "$EXP/smoke/smoke_$arm/results.json" ] || { echo "SMOKE FAILED for $arm"; tail -15 "$EXP/smoke/smoke_$arm.log"; release_lock; exit 8; }
done
python3 "$EXP/smoke_steptime.py" "$SMOKE_STEPS" > "$EXP/smoke/steptime_console.txt" 2>&1 || true

# 预算判定：超预算 ⇒ 三臂同步降步（README §6 的砍法；判据线不动）
STEPS=$(python3 - "$EXP" "$STEPS_FULL" "$BUDGET_MIN" "$EVAL_EST_MIN" <<'PY'
import json, os, sys
exp, steps_full, budget, eval_est = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
arms = ("stock", "l1", "divl1")
rows, proj = {}, {}
for a in arms:
    lines = [json.loads(x) for x in open(os.path.join(exp, "smoke", f"smoke_{a}", "train_log.jsonl")) if x.strip()]
    inner_min = max(lines[-1]["min"] - lines[0]["min"], 1e-9)
    win = max(lines[-1]["step"] - lines[0]["step"], 1)
    ms = inner_min * 60000 / win
    rows[a] = {"inner_ms_per_step": round(ms, 2), "smoke_peak_mb": json.load(open(
        os.path.join(exp, "smoke", f"smoke_{a}", "results.json"))).get("peak_gpu_mem_mb")}
    proj[a] = ms * steps_full / 60000
total = sum(proj.values())
plan = {"steps_full": steps_full, "budget_min": budget, "eval_est_min": eval_est,
        "smoke": rows, "projected_train_min": round(total, 1),
        "projected_grand_total_min": round(total + eval_est + 3, 1)}
if total + eval_est + 3 > budget:
    avail = budget - eval_est - 5
    scale = avail / total if total else 1.0
    chosen = max(int(steps_full * scale) // 200 * 200, 200)
    plan["decision"] = f"超预算 ⇒ 三臂同步降步 {steps_full} → {chosen}"
    plan["unequal_field"] = ("三臂同步数（阶梯内仍同步），但与 13M 锚点 token 数不再相等 ⇒ "
                             "写入 analysis.md 的不等字段")
else:
    chosen = steps_full
    plan["decision"] = f"预算内 ⇒ 保持 {steps_full} 步（与 13M 锚点同 token）"
    plan["unequal_field"] = None
plan["steps_chosen"] = chosen
json.dump(plan, open(os.path.join(exp, "smoke", "budget_plan.json"), "w"), indent=2, ensure_ascii=False)
print(f"# projected_train_min={total:.1f} eval_est={eval_est} budget={budget}", file=sys.stderr)
print(json.dumps(plan, indent=2, ensure_ascii=False), file=sys.stderr)
print(chosen)
PY
)
release_lock                     # §20.1：冒烟结束就还锁，留窗口给别的线
echo "steps_chosen=$STEPS  (see smoke/budget_plan.json)"
[ -z "$STEPS" ] && { echo "budget plan failed"; exit 7; }
sleep 60

# ---------------- ② 三臂全跑（**按臂取锁/释放**，§20.1） ----------------
START=$SECONDS
for arm in $ARMS; do
  acquire_lock || exit 9
  echo "=== FULL TRAIN $arm: $STEPS steps ==="
  python3 train_m1.py "$arm" --config "$EXP/configs/$arm.json" --steps "$STEPS" \
      --ckpt-every 6000 --ckpt-dir "$ROOT/.tmp/v3-ckpts/$arm" \
      --out-root "$EXP/train" > "$EXP/train_$arm.log" 2>&1
  echo "train $arm exit=$? elapsed=$(( (SECONDS-START)/60 ))min  $(date -Is)"
  release_lock
  [ "$arm" != "divl1" ] && sleep 60          # §20.1：臂间留 ≥60s 窗口
done
echo "total train wall: $(( (SECONDS-START)/60 )) min"

# ---------------- ③ 评测（~5 min < 10 min ⇒ 允许整程持锁） ----------------
[ "${SKIP_EVAL:-0}" = "1" ] && { echo "SKIP_EVAL=1"; exit 0; }
acquire_lock || exit 9
echo "=== EVAL ==="
python3 "$EXP/fill_config_hashes.py" 2>&1 | tail -3
if [ "${SKIP_BATTERY:-0}" = "1" ]; then
  python3 "$EXP/run.py" --no-battery > "$EXP/eval_console.txt" 2>&1
else
  python3 "$EXP/run.py" > "$EXP/eval_console.txt" 2>&1
fi
echo "EVAL exit=$?"; tail -45 "$EXP/eval_console.txt"
release_lock

# ---------------- ④ 对账（纯 CPU，不取锁） ----------------
echo "=== RECONCILE ==="
python3 "$EXP/reconcile_numbers.py" 2>&1 | tail -30 || true
echo "DONE $(date -Is)"
