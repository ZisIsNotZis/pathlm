#!/usr/bin/env bash
# 2620 · 去相关的规模验证（契约 V3）——训练 + 评测总入口
#
# GPU 互斥锁协议：SCHEMA §20（已修版：释放/回收一律先 `rm -f` 内容再 `rmdir`；
# 回收只认 `holder_pid` 存活性，不用 pgrep）。持锁期间 nohup 后台 + 后台心跳每 60s。
#
# 用法：
#   nohup bash train_all.sh smoke > smoke_run.log 2>&1 &   # ① 60 步冒烟：实测步时/显存
#   nohup bash train_all.sh full  > run_all.log   2>&1 &   # ② 三臂全跑 + 评测 + 对账
# 环境变量：STEPS（默认 6000）、SKIP_BATTERY=1（砍 200-batch battery 腿，README §6 砍法①）
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale
LOCK=/tmp/pathlm_gpu.lock
ARMS="stock l1 divl1"
MODE="${1:-full}"
STEPS="${STEPS:-6000}"
SMOKE_STEPS="${SMOKE_STEPS:-60}"

cd "$ROOT" || exit 1

# ---------------- 取锁（有界等待 45 min，打印进度） ----------------
n=0
while ! mkdir "$LOCK" 2>/dev/null; do
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
  stale=0; reason=""
  if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
  if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
    stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
  if [ $stale -eq 1 ]; then
    echo "reclaim: $reason"
    rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
    rmdir "$LOCK" 2>/dev/null || true
    continue
  fi
  sleep 15; n=$((n+1))
  [ $((n % 8)) -eq 0 ] && echo "waiting for GPU lock: $((n*15))s (holder=$hp)"
  [ $n -gt 180 ] && { echo "LOCK TIMEOUT after $((n*15))s (holder=$hp)"; exit 9; }
done
echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
trap 'rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null' EXIT
# 心跳：持锁期间每 60s 刷新（长训练单臂可能连续 30+ min 不返回 shell）
( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) &
HB=$!
echo "LOCK acquired $(date -Is) waited=$((n*15))s mode=$MODE steps=$STEPS"

mkdir -p "$EXP/smoke" "$EXP/train"

if [ "$MODE" = "smoke" ]; then
  echo "=== SMOKE ($SMOKE_STEPS steps/arm) ==="
  for arm in $ARMS; do
    python3 train_m1.py "smoke_$arm" --config "$EXP/configs/$arm.json" \
        --steps "$SMOKE_STEPS" --out-root "$EXP/smoke" > "$EXP/smoke/smoke_$arm.log" 2>&1
    echo "smoke $arm exit=$?"
    tail -3 "$EXP/smoke/smoke_$arm.log"
  done
  python3 "$EXP/smoke_steptime.py" "$SMOKE_STEPS" 2>&1
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
  kill $HB 2>/dev/null
  trap - EXIT
  echo "SMOKE DONE $(date -Is)"
  exit 0
fi

# ---------------- 三臂全跑 ----------------
echo "=== FULL TRAIN: steps=$STEPS ==="
START=$SECONDS
for arm in $ARMS; do
  python3 train_m1.py "$arm" --config "$EXP/configs/$arm.json" --steps "$STEPS" \
      --ckpt-every 3000 --ckpt-dir "$ROOT/.tmp/v3-ckpts/$arm" \
      --out-root "$EXP/train" > "$EXP/train_$arm.log" 2>&1
  echo "train $arm exit=$? elapsed=$(( (SECONDS-START)/60 ))min"
done
echo "total train wall: $(( (SECONDS-START)/60 )) min"

# ---------------- 评测 ----------------
echo "=== EVAL ==="
python3 "$EXP/fill_config_hashes.py" 2>&1 | tail -3
if [ "${SKIP_BATTERY:-0}" = "1" ]; then
  python3 "$EXP/run.py" --no-battery 2>&1 | tail -80
else
  python3 "$EXP/run.py" 2>&1 | tail -80
fi
echo "EVAL exit=$?"

# ---------------- 对账 ----------------
python3 "$EXP/reconcile_numbers.py" 2>&1 | tail -20 || true
rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
kill $HB 2>/dev/null
trap - EXIT
echo "DONE $(date -Is)"
