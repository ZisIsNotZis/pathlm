#!/usr/bin/env bash
# 2627 · G5 机制定论抖动补点（ticket 44 X1）—— 训练/评测总入口
#
# GPU 互斥锁协议：SCHEMA §20（释放/回收一律先 `rm -f` 内容再 `rmdir`；回收只认
# `holder_pid` 存活性，不用 pgrep）+ §20.1（臂间 ≥60s；本实验单臂，整程 <10min 的模式
# 允许整程持锁，仅 smoke）。
#
# 干预：把 2626（V2）的 `repr01` 臂**同配置、同 24000 步、同 seed=0 再跑一遍**
# （配置与 2626/configs13m/repr01.json 逐字节相同）；判据 = 两次 repr01 的
# `l1@K8 − repr01@K8` 净效应的**极差**（README §4 预登记）。
#
# 用法：
#   nohup bash train_all.sh smoke 13m > smoke13m.log 2>&1 &   # 250 步实测步时（§20.3）
#   nohup bash train_all.sh train 13m > train13m_driver.log 2>&1 &   # 24k 单臂
#   nohup bash train_all.sh eval  13m > eval13m.log 2>&1 &    # 评测（纯推理 >2min ⇒ 取锁）
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2627-mechanism-jitter
LOCK=/tmp/pathlm_gpu.lock
MODE="${1:-help}"
SCALE="${2:-13m}"

cd "$ROOT" || exit 1

take_lock() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp; hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    local stale=0 reason=""
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; reason="holder_pid $hp 已死"; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then
      stale=1; reason="无 holder_pid 且锁龄 >15min"; fi
    if [ $stale -eq 1 ]; then
      echo "reclaim: $reason"; rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true; continue
    fi
    sleep 15; n=$((n+1))
    [ $((n % 8)) -eq 0 ] && echo "waiting for GPU lock: $((n*15))s (holder=$hp)"
    [ $n -gt 600 ] && { echo "LOCK TIMEOUT after $((n*15))s (holder=$hp)"; return 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
  echo "LOCK acquired $(date -Is) waited=$((n*15))s mode=$MODE scale=$SCALE"
}

release_lock() {
  rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null
  echo "LOCK released $(date -Is)"
}

# 心跳：持锁期间每 60s 刷新
( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) &
HB=$!
trap 'release_lock' EXIT

mkdir -p "$EXP/train13m"

if [ "$MODE" = "smoke" ]; then
  # 纯预算判定：250 步，不外推（SCHEMA §17.5）；单臂 <1min ⇒ 整程持锁允许
  take_lock || exit 9
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/pre-free(smoke) /'
  python3 train_m1.py repr01 --config "$EXP/configs13m/repr01.json" --steps 250 \
      --out-root "$ROOT/.tmp/2627-smoke" > "$ROOT/.tmp/2627-smoke.log" 2>&1
  echo "smoke exit=$? $(date -Is)"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/post-free(smoke) /'
  release_lock
  exit 0
fi

if [ "$MODE" = "train" ]; then
  take_lock || exit 9
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/pre-free(repr01) /'
  python3 train_m1.py repr01 --config "$EXP/configs13m/repr01.json" --steps 24000 \
      --ckpt-every 6000 --ckpt-dir "$ROOT/.tmp/2627-ckpts/$SCALE/repr01" \
      --out-root "$EXP/train$SCALE" > "$EXP/train$SCALE/repr01.log" 2>&1
  echo "train $SCALE repr01 exit=$? $(date -Is)"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/post-free(repr01) /'
  release_lock
  echo "TRAIN DONE $SCALE $(date -Is)"; exit 0
fi

if [ "$MODE" = "eval" ]; then
  take_lock || exit 9
  python3 "$EXP/fill_config_hashes.py" 2>&1 | tail -5
  python3 "$EXP/run.py" --scale "$SCALE" 2>&1 | tail -120
  echo "EVAL $SCALE exit=$? $(date -Is)"
  python3 "$EXP/reconcile_numbers.py" --scale "$SCALE" 2>&1 | tail -25 || true
  release_lock
  echo "EVAL DONE $SCALE $(date -Is)"; exit 0
fi

echo "usage: train_all.sh {smoke 13m|train 13m|eval 13m}"
