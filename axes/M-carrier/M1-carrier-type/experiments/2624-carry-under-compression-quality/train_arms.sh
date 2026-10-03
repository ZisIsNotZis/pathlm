#!/usr/bin/env bash
# 2624-carry-under-compression-quality · 训练驱动（两条被训练的臂）。
#
# 纪律（SCHEMA §20 / §20.1 / §20.3）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——
# 不整程独占。共享卡上有非本仓负载（两个 llama-server，实测 ~19.5GB / util ~99%），
# 实测步时见 README §4c；本线在争用下按 §20.3 **缩步数保臂间同刻度**（两条臂步数恒等）。
#
# 两条被训练的臂（唯一变量 = n_kv_heads；carry + window16 是常量，seed=0）：
#   carry_mha : MHA(8 KV 头) + carry(delta) + window(16, anchors 0) → CarryChannelBlock
#   carry_gqa : GQA(2 KV 头) + carry(delta) + window(16, anchors 0) → ComposableBlock（ENG1）
#
# S 步 × bs32 × T256 = S·8192 token（两臂严格同预算）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2624-carry-under-compression-quality
LOCK=/tmp/pathlm_gpu.lock
S="${Z3_STEPS:-6000}"

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

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps $S --batch-size 32 --lr 1e-3 --seed 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 --train-anchors 0 --out-root $EXP/train"

run_arm() {
  local name=$1; shift
  echo "=== [$name] $(date -Is) 取锁 ===" | tee -a "$EXP/train_arms.runlog"
  acquire
  echo "=== [$name] 训练开始 $(date -Is) ===" | tee -a "$EXP/train_arms.runlog"
  python3 train_depth_ar.py "$name" $COMMON "$@" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ===" | tee -a "$EXP/train_arms.runlog"
  release
  if [ $rc -ne 0 ]; then echo "[$name] 训练失败，见 $EXP/train_${name}.log"; tail -20 "$EXP/train_${name}.log"; fi
  echo "=== [$name] 臂间窗口 65s ===" | tee -a "$EXP/train_arms.runlog"
  sleep 65
}

run_arm carry_mha
run_arm carry_gqa --n-kv-heads 2

echo "ALL ARMS DONE $(date -Is)" | tee -a "$EXP/train_arms.runlog"
