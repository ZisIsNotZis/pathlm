#!/usr/bin/env bash
# 2635-recursive-carrier-13m · 两臂训练驱动（stock 全注意力 / 线性递推状态载体）。
#
# 纪律（SCHEMA §20 / §20.1）：**按臂**取/放 GPU 锁，臂间留 ≥60s 窗口。
# 每臂都过**停摆门**（stall_gate.py）：≥180s 无新步 ⇒ 采集诊断（触发步/CPU/显存/日志尾/
# faulthandler 栈）、终止该臂、返回码 3 ⇒ 本驱动把结果写进 stall_report.json 并停在那里
# （实现层停摆 = 判据①不通过，不给质量结论；不调参救——那是另一个干预）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/M-carrier/M1-carrier-type/experiments/2635-recursive-carrier-13m
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
    [ $n -gt 240 ] && { echo "等锁超时（>60min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
  rmdir "$LOCK" 2>/dev/null || true
}

BASE="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
--steps 3000 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform --eval-batches 4 \
--out-root $EXP/train"

run_arm() {
  local name=$1; shift
  echo "=== [$name] $(date -Is) 取锁 ==="
  acquire
  echo "=== [$name] 训练开始 $(date -Is) | GPU free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ==="
  python3 "$EXP/stall_gate.py" \
    --train-log "$EXP/train/$name/train_log.jsonl" \
    --report "$EXP/stall_report_${name}.json" \
    --stall-secs 180 --poll-secs 20 --dump-stack-secs 150 \
    -- python3 train_depth_ar.py "$name" $BASE "$@" >> "$EXP/train_${name}.log" 2>&1
  local rc=$?
  echo "=== [$name] 结束 rc=$rc $(date -Is) ==="
  release
  if [ $rc -eq 3 ]; then
    echo "!! [$name] 停摆：见 $EXP/stall_report_${name}.json（判据①不通过）"
    return 3
  fi
  if [ $rc -ne 0 ]; then echo "!! [$name] 训练失败 rc=$rc"; tail -20 "$EXP/train_${name}.log"; fi
  sleep 65
}

run_arm stock
run_arm linear --attn-linear

echo "ALL ARMS DONE $(date -Is)"
