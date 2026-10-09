#!/usr/bin/env bash
# 2628 · G5 机制定论抖动**补全**（l1 臂，ticket 44-X1 后续）—— 训练/评测总入口
#
# GPU 互斥锁协议：SCHEMA §20（释放/回收一律先 `rm -f` 内容再 `rmdir`；回收只认
# `holder_pid` 存活性，不用 pgrep）+ §20.1（臂间 ≥60s；本实验单臂，无法分片释放 ⇒
# 整臂持锁，持锁期间后台心跳每 60s `touch`；smoke <1min 整程持锁允许）。
#
# 干预：把 2626（V2）的 `l1` 臂**同配置、同 24000 步、同 seed=0 再跑一遍**
# （配置与 2626/configs13m/l1.json 逐字节相同）；判据 = `l1`/`repr01` 两臂各自的
# `@K8` run-to-run 极差，及**两臂合并极差（取大）**（README §4 预登记）。
#
# 用法：
#   nohup bash train_all.sh smoke 13m > smoke13m.log 2>&1 &   # 250 步实测步时（§20.3）
#   nohup bash train_all.sh train 13m > train13m_driver.log 2>&1 &   # 24k 单臂
#   nohup bash train_all.sh eval  13m > eval13m.log 2>&1 &    # 评测（纯推理 >2min ⇒ 取锁）
set -u
ROOT=/home/z/vibe/pathlm
EXP=$ROOT/axes/G-policy/G5-parallel-lines/experiments/2628-jitter-l1
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
  python3 train_m1.py l1 --config "$EXP/configs13m/l1.json" --steps 250 \
      --out-root "$ROOT/.tmp/2628-smoke" > "$ROOT/.tmp/2628-smoke.log" 2>&1
  echo "smoke exit=$? $(date -Is)"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/post-free(smoke) /'
  release_lock
  exit 0
fi

if [ "$MODE" = "train" ]; then
  take_lock || exit 9
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/pre-free(l1) /'
  python3 train_m1.py l1 --config "$EXP/configs13m/l1.json" --steps 24000 \
      --ckpt-every 6000 --ckpt-dir "$ROOT/.tmp/2628-ckpts/$SCALE/l1" \
      --out-root "$EXP/train$SCALE" > "$EXP/train$SCALE/l1.log" 2>&1
  echo "train $SCALE l1 exit=$? $(date -Is)"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' ' | sed 's/^/post-free(l1) /'
  release_lock
  echo "TRAIN DONE $SCALE $(date -Is)"; exit 0
fi

if [ "$MODE" = "eval" ]; then
  take_lock || exit 9
  python3 "$EXP/fill_config_hashes.py" 2>&1 | tail -8
  python3 "$EXP/run.py" --scale "$SCALE" 2>&1 | tail -160
  echo "EVAL $SCALE exit=$? $(date -Is)"
  python3 "$EXP/reconcile_numbers.py" --scale "$SCALE" 2>&1 | tail -30 || true
  release_lock
  echo "EVAL DONE $SCALE $(date -Is)"; exit 0
fi

echo "usage: train_all.sh {smoke 13m|train 13m|eval 13m}"
