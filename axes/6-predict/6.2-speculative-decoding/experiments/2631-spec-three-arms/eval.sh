#!/usr/bin/env bash
# 2631-spec-three-arms · 评测驱动（取 GPU 锁；评测的墙钟是**主判据**，必须独享整程）。
#
# 用法：bash eval.sh [run.py 的额外参数]     例：bash eval.sh --smoke
# 纪律：与 train_arms.sh 同一把锁（/tmp/pathlm_gpu.lock）；握手用 holder_pid/heartbeat，
# 陈旧锁（holder 死掉或 >15min 无心跳）自动回收；trap EXIT 保证异常也释放。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms
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
    [ $((n % 8)) -eq 0 ] && echo "  [wait-lock] 已等 $((n*15))s holder=$hp $(date -Is)"
    [ $n -gt 200 ] && { echo "等锁超时（>50min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; }
trap release EXIT

echo "GPU free before: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date -Is)"
acquire
echo "GPU free (lock held): $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date -Is)"
python3 "$EXP/run.py" --ckpts "$EXP/train/*" --out "$EXP" "$@"
rc=$?
echo "GPU free after: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date -Is)"
echo "eval rc=$rc DONE $(date -Is)"
exit $rc
