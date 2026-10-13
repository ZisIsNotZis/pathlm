#!/usr/bin/env bash
# 2625-ood-leg-variance · 训练驱动（两条新臂：seed=1 / seed=2）。
#
# 纪律（SCHEMA §20 / §20.1 / §20.3）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
# 唯一变量 = `--seed`；其余逐字段相同（GQA2 + carry(delta) + window16，6k×32×256）。
# 复用的 seed=0 臂 = Z3（2624）的 `carry_gqa`（同配置、同预算），本脚本不重训它。
#
# S 步 × bs32 × T256 = S·8192 token（两臂严格同预算，且 == 复用臂的 49.0M token）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance
LOCK=/tmp/pathlm_gpu.lock
S="${W2_STEPS:-6000}"

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1))
    if [ $((n % 8)) -eq 0 ]; then echo "  [wait-lock] 已等 $((n*15))s，holder=$hp $(date -Is)"; fi
    [ $n -gt 240 ] && { echo "等锁超时（>60min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null || true
}

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps $S --batch-size 32 --lr 1e-3 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 --train-anchors 0 --n-kv-heads 2 --out-root $EXP/train"

run_arm() {
  local name=$1 seed=$2
  echo "=== [$name seed=$seed] $(date -Is) 取锁 ===" | tee -a "$EXP/train_arms.runlog"
  acquire
  echo "=== [$name] 训练开始 $(date -Is) ===" | tee -a "$EXP/train_arms.runlog"
  python3 train_depth_ar.py "$name" $COMMON --seed "$seed" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ===" | tee -a "$EXP/train_arms.runlog"
  release
  if [ $rc -ne 0 ]; then echo "[$name] 训练失败，见 $EXP/train_${name}.log"; tail -20 "$EXP/train_${name}.log"; fi
  echo "=== [$name] 臂间窗口 65s ===" | tee -a "$EXP/train_arms.runlog"
  sleep 65
}

run_arm gqa_carry_win_s1 1
run_arm gqa_carry_win_s2 2

echo "ALL ARMS DONE $(date -Is)" | tee -a "$EXP/train_arms.runlog"
