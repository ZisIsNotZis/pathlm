#!/usr/bin/env bash
# 2643 · 单臂收口驱动：只跑 U:0（放弃 PR:0）。
# 等锁到 22:40 仍不可得 ⇒ 放弃（不伪造读数）。拿到锁后**一次持锁跑完
# 冒烟(60 步) + U 12k**（实测整程 ≈ 8 min < 10 min，符合 §20.1.1「整程 <10 min
# 才允许整程持锁」），随后释放。
set -uo pipefail
cd "$(dirname "$0")/../../../../.."
EXP="axes/3-width/3.2.3-tied-untied/experiments/2643-tied-untied-13m"
LOCK=/tmp/pathlm_gpu.lock
LOG() { echo "[$(date -Iseconds)] $*" >> "$EXP/run_log.txt"; }

deadline=$(date -d "$(date +%F) 22:40:00" +%s)
while ! mkdir "$LOCK" 2>/dev/null; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    LOG "U-ONCE: 锁等待到 22:40 仍不可得 → 放弃（本轮不起跑）"
    exit 9
  fi
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then
    LOG "U-ONCE reclaim: holder_pid $hp 已死"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 2
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
LOG "U-ONCE: lock acquired pid=$$ free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
t0=$(date +%s)

python3 -u "$EXP/run.py" --arm U --seed 0 --steps 60 >> "$EXP/smoke_U_s0.log" 2>&1
rc=$?
LOG "U-ONCE: smoke(60 步) rc=$rc"

rc2=9
if [ "$rc" -eq 0 ]; then
  python3 -u "$EXP/run.py" --arm U --seed 0 --steps 12000 >> "$EXP/train_U_s0.log" 2>&1
  rc2=$?
  LOG "U-ONCE: U-s0 12k rc=$rc2"
else
  LOG "U-ONCE: 冒烟失败 → 不跑 12k（不伪造读数）"
fi

rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
rmdir "$LOCK" 2>/dev/null
LOG "U-ONCE: release free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader) hold=$(( $(date +%s) - t0 ))s"

if [ "$rc2" -eq 0 ]; then
  python3 "$EXP/run.py" --aggregate >> "$EXP/drive.log" 2>&1
  LOG "U-ONCE aggregate rc=$?"
fi
LOG "U-ONCE: done"
