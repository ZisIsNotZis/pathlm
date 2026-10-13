#!/usr/bin/env bash
# 2630-carry-ragged-seed1 · seed=1 两臂训练驱动（carry × 随机退出 × proj-fill 的复现）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口——不整程独占。
# 共享卡上另有非本仓负载（llama-server）与本波其它 GPU 线；锁只管本仓 GPU 线。
#
# 两臂（唯一变量 = carry 的开关，逐字段同配方，seed=1，非 dense RAP 形态；
# 与 2629 的 seed=0 逐字段相同，只换 seed）：
#   projfill_s1       : 随机退出 + proj-fill（100M 出口资产配方的形态件），无 carry
#   projfill_carry_s1 : 同上 + carry(delta 写)
#
# 6k 步 × bs32 × T256 = 49.0M token（两臂严格同预算；步时先测后定，见 README §6）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2630-carry-ragged-seed1
LOCK=/tmp/pathlm_gpu.lock

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

COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --topology sequential --steps 6000 --batch-size 32 --lr 1e-3 --seed 1 --exit-mode uniform --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --proj-fill --final-weight 2.0 --out-root $EXP/train"

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

run_arm projfill_s1
run_arm projfill_carry_s1 --carry-write delta

echo "ALL ARMS DONE $(date -Is)"
