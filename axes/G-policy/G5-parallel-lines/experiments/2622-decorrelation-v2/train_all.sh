#!/usr/bin/env bash
# 2622 · 去相关 v2 的决定性实验 —— 训练/评测总入口（X1）
#
# GPU 互斥锁协议：SCHEMA §20（已修版：释放/回收一律先 `rm -f` 内容再 `rmdir`；
# 回收只认 `holder_pid` 存活性，不用 pgrep）+ §20.1（多臂驱动按臂释放/重取，臂间 ≥60s）。
#
# 用法：
#   nohup bash train_all.sh smoke        > smoke_run.log 2>&1 &   # 三臂 × 400 步冒烟（整程 <10min，允许整程持锁）
#   nohup bash train_all.sh train 13m    > train13m.log   2>&1 &   # 13M 三臂训练（按臂取/放锁）
#   nohup bash train_all.sh train 50m    > train50m.log   2>&1 &   # 50M 两臂训练（stock 复用 2620 检查点）
#   nohup bash train_all.sh eval  13m    > eval13m.log    2>&1 &   # 评测（纯推理 >2min ⇒ 取锁）
# 环境变量：SMOKE_STEPS（默认 400）
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2622-decorrelation-v2
LOCK=/tmp/pathlm_gpu.lock
MODE="${1:-help}"
SCALE="${2:-13m}"
SMOKE_STEPS="${SMOKE_STEPS:-400}"

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

# 心跳：持锁期间每 60s 刷新（长训练单臂可能连续 30+ min 不返回 shell）
( while kill -0 $$ 2>/dev/null; do touch "$LOCK/heartbeat" 2>/dev/null; sleep 60; done ) &
HB=$!
trap 'release_lock' EXIT

mkdir -p "$EXP/smoke" "$EXP/train13m" "$EXP/train50m" "$ROOT/.tmp/v2-ckpts"

if [ "$MODE" = "smoke" ]; then
  # 整程 <10 min ⇒ 允许整程持锁（§20.1）
  take_lock || exit 9
  for arm in stock l1 repr; do
    python3 train_m1.py "smoke13m_$arm" --config "$EXP/configs13m/$arm.json" \
        --steps "$SMOKE_STEPS" --out-root "$EXP/smoke" > "$EXP/smoke/smoke13m_$arm.log" 2>&1
    echo "smoke 13m $arm exit=$?"; tail -3 "$EXP/smoke/smoke13m_$arm.log"
  done
  for arm in l1 repr; do
    python3 train_m1.py "smoke50m_$arm" --config "$EXP/configs50m/$arm.json" \
        --steps "$SMOKE_STEPS" --out-root "$EXP/smoke" > "$EXP/smoke/smoke50m_$arm.log" 2>&1
    echo "smoke 50m $arm exit=$?"; tail -3 "$EXP/smoke/smoke50m_$arm.log"
  done
  python3 "$EXP/smoke_steptime.py" 2>&1
  echo "SMOKE DONE $(date -Is)"; exit 0
fi

if [ "$MODE" = "train" ]; then
  if [ "$SCALE" = "13m" ]; then
    ARMS="stock l1 repr"; CFGDIR=configs13m; STEPS=6000; CKPT_EVERY=3000
  else
    ARMS="l1 repr"; CFGDIR=configs50m; STEPS=12000; CKPT_EVERY=6000
  fi
  for arm in $ARMS; do
    take_lock || exit 9
    python3 train_m1.py "$arm" --config "$EXP/$CFGDIR/$arm.json" --steps "$STEPS" \
        --ckpt-every "$CKPT_EVERY" --ckpt-dir "$ROOT/.tmp/v2-ckpts/$SCALE/$arm" \
        --out-root "$EXP/train$SCALE" > "$EXP/train$SCALE/${arm}.log" 2>&1
    echo "train $SCALE $arm exit=$? $(date -Is)"
    release_lock
    sleep 65   # §20.1：臂间留 ≥60s 窗口
  done
  echo "TRAIN DONE $SCALE $(date -Is)"; exit 0
fi

if [ "$MODE" = "eval" ]; then
  take_lock || exit 9
  python3 "$EXP/fill_config_hashes.py" 2>&1 | tail -3
  python3 "$EXP/run.py" --scale "$SCALE" 2>&1 | tail -100
  echo "EVAL $SCALE exit=$? $(date -Is)"
  python3 "$EXP/reconcile_numbers.py" --scale "$SCALE" 2>&1 | tail -20 || true
  release_lock
  echo "EVAL DONE $SCALE $(date -Is)"; exit 0
fi

echo "usage: train_all.sh {smoke|train <13m|50m>|eval <13m|50m>}"
