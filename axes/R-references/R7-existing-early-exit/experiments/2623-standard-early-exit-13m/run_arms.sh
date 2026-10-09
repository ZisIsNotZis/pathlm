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

if [ "${1:-}" = "smoke" ]; then
  # 冒烟与全跑**分开落盘**（不许用 20 步的产物覆盖正式 run 目录）
  R7_STEPS=20
  OUT="${R7_OUT:-.scratch/52-r7-early-exit/smoke}"
  mkdir -p "$OUT"
fi
CMD="${1:-all}"

case "$CMD" in
  probe)
    # 先测步时再定预算（SCHEMA §17.8）：一个进程内跑各臂 60 步
    acquire || exit 9
    trap 'release' EXIT
    t0=$(date +%s)
    python3 .scratch/52-r7-early-exit/step_time_probe.py --steps 60 \
        --out .scratch/52-r7-early-exit/step_time.json >>"$OUT/step_time.log" 2>&1
    echo "[probe] rc=$? wall=$(( $(date +%s) - t0 ))s" | tee -a "$LOG"
    release; trap - EXIT
    ;;
  all|smoke)
    # ---- 锚点：stock（普通 transformer，全深度，仅最终 CE）----
    run_one r7-stock-s0   --dense --exit-mode full --seed 0
    # ---- 本仓 C1 现配方（carry 门 + 随机退出 + fill + 加权平均降权配方）----
    run_one r7-c1-legacy-s0 --exit-mode uniform --aux-weight 1.0 --final-weight 1.0 --seed 0
    # ---- C1 形态 + 公平配方（主目标不降权 + 轻量早退项 λ=0.1）----
    run_one r7-c1-fair-s0 --exit-mode uniform --aux-mode additive --aux-weight 0.1 \
        --final-weight 1.0 --seed 0
    # ---- LayerSkip 式：无门普通 transformer + 随机退出（layer dropout）+ 轻量早退项 ----
    run_one r7-ls-fair-s0 --exit-mode uniform --no-carry-gate --aux-mode additive \
        --aux-weight 0.1 --final-weight 1.0 --seed 0
    # ---- MoD 式：L=16 + 每序列 top-128/256 放行（isoFLOP with dense L=8）----
    run_one r7-mod-s0 --dense --layers 16 --mod-ratio 0.5 --mod-router-weight 0.01 \
        --exit-mode full --seed 0
    # ---- seed 1 重复（本格自测噪声 σ_cell；预算不够时从后往前砍）----
    if [ "$CMD" = "all" ]; then
      run_one r7-stock-s1   --dense --exit-mode full --seed 1
      run_one r7-c1-fair-s1 --exit-mode uniform --aux-mode additive --aux-weight 0.1 \
          --final-weight 1.0 --seed 1
      run_one r7-ls-fair-s1 --exit-mode uniform --no-carry-gate --aux-mode additive \
          --aux-weight 0.1 --final-weight 1.0 --seed 1
    fi
    ;;
  *) echo "unknown mode $CMD"; exit 2;;
esac
echo "=== done $(date -Is) ===" | tee -a "$LOG"
