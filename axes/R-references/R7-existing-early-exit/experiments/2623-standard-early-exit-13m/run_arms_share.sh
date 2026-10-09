#!/usr/bin/env bash
# R7（WI-16）臂驱动：**按臂取/放 GPU 锁**（SCHEMA §20.1），臂间 ≥60 s。
#
# 用法：
#   bash run_arms.sh smoke        # 只跑 20 步冒烟（步时/显存/活性，SCHEMA §17.8）
#   bash run_arms.sh all          # 全部训练臂（契约 §6 的预算）
#   bash run_arms.sh <arm> [...]  # 指定臂（见 ARMS 表）
#
# 每条臂的命令行**逐字落盘**到本脚本（可复现），训练结果进
# `.scratch/52-r7-early-exit/evidence/<run>/`。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
cd "$ROOT"
LOCK=/tmp/pathlm_gpu.lock
OUT="${R7_OUT:-.scratch/52-r7-early-exit/evidence}"
LOG="$HERE/run_log.txt"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1

release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; }

acquire() {
  local n=0
  # 轮询间隔 15 s（SCHEMA §20 的规范值）；上限 720×15 s = 3 h：共享卡上多线竞争时会长时间排队；**绝不跳过臂**
  # （skip 会让「臂间同刻度」失效）。等不到就在日志里留痕并继续排。
  while [ "$n" -lt 720 ]; do
    if mkdir "$LOCK" 2>/dev/null; then
      echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"; return 0
    fi
    local hp=""
    [ -f "$LOCK/holder_pid" ] && hp="$(cat "$LOCK/holder_pid" 2>/dev/null || true)"
    local stale=0
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then
      echo "[lock] reclaim (holder_pid=$hp)" | tee -a "$LOG"
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    if [ $((n % 40)) -eq 0 ] && [ "$n" -gt 0 ]; then
      echo "[lock] 仍在排队（已等 $((n / 4)) min，holder=$(cat "$LOCK/holder_pid" 2>/dev/null || echo '?'))" | tee -a "$LOG"
    fi
    sleep 15; n=$((n + 1))
  done
  echo "[lock] 等待超时（720×15 s），按 SCHEMA §20 上报，不杀他人进程" | tee -a "$LOG"
  return 9
}

SPEC="--d 256 --layers 8 --heads 8 --mlp-mult 4 --seq-len 256 --batch-size 32 --lr 1e-3"
# 事故记录（2026-10-09）：首版漏了 --d/--layers，缺省落到 d=128/L=4，整臂变成玩具；
# 现在 ① 规格显式写死；② 每臂跑完**回读 results.json 断言**规格与步数（不靠 exit code）。
check_spec() {   # $1 = run 目录, $2 = layers, $3 = steps
  python3 - "$1" "$2" "$3" <<'PY' || return 1
import json, sys
d = json.load(open(sys.argv[1] + "/results.json"))
c, t = d["config"], d["train"]
want = {"d_model": 256, "n_heads": 8, "seq_len": 256, "mlp_mult": 4,
        "n_layers": int(sys.argv[2])}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
bad.update({k: (t.get(k), v) for k, v in
            {"batch_size": 32, "steps": int(sys.argv[3])}.items()
            if t.get(k) != v})
sys.exit(f"规格不符：{bad}" if bad else 0)
PY
}

run_one() {   # $1 = 臂名，${@:2} = train_depth_ar.py 的参数
  local name="$1"; shift
  local steps="${R7_STEPS:-12000}"
  local tag=""
  [ "$steps" = "20" ] && tag="-smoke"
  local args=("$@")
  echo "=== $(date -Is) arm=$name steps=$steps ===" | tee -a "$LOG"
  acquire || return 9
  trap 'release' EXIT
  local t0=$(date +%s)
  python3 train_depth_ar.py "$name$tag" --out-root "$OUT" --steps "$steps" \
      $SPEC "${args[@]}" >>"$OUT/$name$tag.stdout.log" 2>&1
  local rc=$?
  local layers=8
  for a in "${args[@]}"; do [ "$a" = "--layers" ] && layers=16; done
  if [ $rc -eq 0 ]; then
    check_spec "$OUT/$name$tag" "$layers" "$steps" || rc=3
  fi
  local t1=$(date +%s)
  release; trap - EXIT
  echo "[arm] $name rc=$rc wall=$((t1 - t0))s" | tee -a "$LOG"
  sleep 60
  return $rc
}

# 复核 P1-4 要求：**同份额对照臂** —— 配方轴的 confound 是「主目标份额」与
# 「全局损失尺度」被一起改了（renorm 除分母、additive 不除）。本臂用 additive 把
# 浅层权重提到 8：loss = CE + 8·mean(浅层 CE) ⇒ 主目标份额 = 1/9 = 11.1%，
# 与 C1 现配方（等权平均、份额 1/9）**同份额**（全局尺度不同，如实记账）。
OUT="${R7_OUT:-.scratch/52-r7-early-exit/evidence}"
mkdir -p "$OUT"
run_one r7-c1-fair-share-s0 --exit-mode uniform --aux-mode additive --aux-weight 8 \
    --final-weight 1.0 --seed 0
echo "=== done $(date -Is) ===" | tee -a "$LOG"
