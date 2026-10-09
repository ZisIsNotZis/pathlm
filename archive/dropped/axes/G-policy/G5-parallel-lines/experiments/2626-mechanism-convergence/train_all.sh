#!/usr/bin/env bash
# 2626 · G5 机制收敛收尾（ticket 43 V2）—— 训练/评测总入口
#
# GPU 互斥锁协议：SCHEMA §20（释放/回收一律先 `rm -f` 内容再 `rmdir`；回收只认
# `holder_pid` 存活性，不用 pgrep）+ §20.1（多臂驱动按臂释放/重取，臂间 ≥60s）。
#
# 干预：与 X1（2622）**同配置**（configs13m/{stock,l1,repr01}.json，`train.steps` 由 6000 → 24000），
# 即 4× token 的 24000 步 cosine 训练；每臂 ckpt@6000/12000/18000/24000。
#
# 用法：
#   nohup bash train_all.sh train 13m > train13m.log 2>&1 &   # stock/l1/repr01 三臂
#   nohup bash train_all.sh eval  13m > eval13m.log  2>&1 &   # 评测（纯推理 >2min ⇒ 取锁）
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2626-mechanism-convergence
LOCK=/tmp/pathlm_gpu.lock
CKPT_ROOT=$ROOT/.tmp/2626-ckpts
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

# 心跳：持锁期间每 60s 刷新（单臂 24k 步长约 20–29 min，连续不返回 shell）
( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) &
HB=$!
trap 'release_lock' EXIT

mkdir -p "$EXP/train13m" "$CKPT_ROOT/13m"

if [ "$MODE" = "train" ]; then
  ARMS="stock l1 repr01"; CFGDIR=configs13m; STEPS=24000; CKPT_EVERY=6000
  for arm in $ARMS; do
    take_lock || exit 9
    nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' \
        | sed "s/^/pre-free($arm) /"
    python3 train_m1.py "$arm" --config "$EXP/$CFGDIR/$arm.json" --steps "$STEPS" \
        --ckpt-every "$CKPT_EVERY" --ckpt-dir "$CKPT_ROOT/$SCALE/$arm" \
        --out-root "$EXP/train$SCALE" > "$EXP/train$SCALE/${arm}.log" 2>&1
    echo "train $SCALE $arm exit=$? $(date -Is)"
    nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' \
        | sed "s/^/post-free($arm) /"
    release_lock
    sleep 65   # §20.1：臂间留 ≥60s 窗口
  done
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

echo "usage: train_all.sh {train 13m|eval 13m}"
