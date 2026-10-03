#!/usr/bin/env bash
# 2623-composable-arms-13m · 四臂训练驱动（ENG1 解锁的组合阶梯）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
# 共享卡上另有非本仓负载（llama-server），锁只管本仓 GPU 线。
#
# 四臂（唯一变量 = 特性叠加，逐字段同配方，seed=0）：
#   stock            : MHA / 无 carry / 无 window
#   gqa              : +GQA(2 KV 头)
#   gqa_carry        : +GQA +carry(delta)
#   gqa_carry_window : +GQA +carry(delta) +window(16, anchors 0)
#
# 6k 步 × bs32 × T256 = 49.0M token（四臂严格同预算；主 Agent 裁决见 README §4c）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2623-composable-arms-13m
LOCK=/tmp/pathlm_gpu.lock

# 等待上限 60 min（240×15s）：本波多条 GPU 线（Y1/Y3）在同一锁上串行，
# 单条线持锁可能整程 >10 min（§20 的 10 min 是单任务口径）；等待时长全部记录。
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

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 6000 --batch-size 32 --lr 1e-3 --seed 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --out-root $EXP/train"

run_arm() {
  local name=$1; shift
  echo "=== [$name] $(date -Is) 取锁 ==="
  acquire
  echo "=== [$name] 训练开始 $(date -Is) ==="
  python3 train_depth_ar.py "$name" $COMMON "$@" > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 训练结束 rc=$rc $(date -Is) ==="
  release
  if [ $rc -ne 0 ]; then echo "[$name] 训练失败，见 $EXP/train_${name}.log"; tail -20 "$EXP/train_${name}.log"; fi
  echo "=== [$name] 臂间窗口 65s ==="
  sleep 65
}

run_arm stock
run_arm gqa              --n-kv-heads 2
run_arm gqa_carry        --n-kv-heads 2 --carry-write delta
run_arm gqa_carry_window --n-kv-heads 2 --carry-write delta --train-window 16 --train-anchors 0

echo "ALL ARMS DONE $(date -Is)"
