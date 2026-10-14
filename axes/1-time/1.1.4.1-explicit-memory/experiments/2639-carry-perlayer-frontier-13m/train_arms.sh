#!/usr/bin/env bash
# 2639-carry-perlayer-frontier-13m · 训练驱动（完整机制臂：逐层独立状态 + 内容写门 ± γ）。
#
# §20 GPU 互斥锁：逐臂取/放锁（/tmp/pathlm_gpu.lock），臂间 ≥61s（§20.1）。
# 与 2638 的差异**只在机制完备度开关**（E1）：`CARRY_PER_LAYER=1`，`CARRY_DECAY` 可选；
# 其余 CLI（d/layers/heads/mlp/seq/steps/bs/lr/seed/exit-mode/train-window/train-anchors/
# carry-write）逐字段与 2638/2634 相同。
#
# 用法：bash train_arms.sh                            # pl16（γ=1，seed 0）
#       ARMS="pl16 c16 pl16d" bash train_arms.sh      # 主臂 → chunk 匹配退化对照 → γ 臂
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m"
ARMS="${ARMS:-pl16}"
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "reclaim: $reason"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1)); [ $n -gt 60 ] && { echo "LOCK TIMEOUT"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "[$(date -Iseconds)] lock acquired pid=$$"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null
  echo "[$(date -Iseconds)] lock released"
}

run_arm() {   # $1 = 臂名, $2 = carry_per_layer(0/1), $3 = carry_decay, $4 = seed
  local name="$1" pl="$2" decay="$3" seed="$4"
  acquire
  CARRY_PER_LAYER="$pl" CARRY_DECAY="$decay" python3 "$EXP/train_pl.py" "$name" \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
    --lr 1e-3 --seed "$seed" --exit-mode uniform --train-window 16 --train-anchors 0 \
    --eval-batches 4 --carry-write delta --carry-chunk 256 --out-root "$EXP/train" \
    > "$EXP/train_${name}.log" 2>&1
  local rc=$?
  release
  echo "[$(date -Iseconds)] arm $name done rc=$rc"
  [ $rc -ne 0 ] && { echo "arm $name FAILED"; return $rc; }
  sleep 61
  return 0
}

for A in $ARMS; do
  case "$A" in
    pl16)  run_arm pl16  1 1.0  0 || exit $? ;;   # 完整机制（逐层 + 内容写门，γ=1）
    pl16d) run_arm pl16d 1 0.99 0 || exit $? ;;   # 完整机制 + 衰减 γ
    c16)   run_arm c16   0 1.0  0 || exit $? ;;   # chunk 匹配的单共享槽退化对照
    *) echo "unknown arm $A"; exit 2 ;;
  esac
done
echo "[$(date -Iseconds)] all arms done"
