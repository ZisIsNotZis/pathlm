#!/usr/bin/env bash
# 2623 编排补跑：先出 full+win16 的主结果，再补 win16_rep，最后重跑评测纳入复现臂。
# 每一步自己按标准锁协议取/放锁（`eval_run.sh` / `train_all.sh` 内的 `mkdir /tmp/pathlm_gpu.lock`，
# 等不到锁以 rc=9 上报）⇒ 重试驱动用 `tools/supervise.sh --command-owns-lock`：supervisor
# **不探测、不触碰锁**，锁的生命周期整条归这几步自己（见 `docs/methodology.md` §18.1）。
# 有界：每步最多 40 次尝试、连续 3 次非零即熔断（共享卡竞争下的稳健补跑，但不再无限重来）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"
LOG="$HERE/supervise_eval_then_rep.log"   # *.log 不入库（.gitignore）

sup() {   # sup <label> <cmd...>：0 = 该步成功；非 0 = 熔断（不再重试，由本脚本继续下一步并留痕）
  local label="$1"; shift
  echo "[$label] start $(date -Is)"
  bash "$ROOT/tools/supervise.sh" --command-owns-lock \
    --attempts-max 40 --fail-max 3 --lock-wait-s 0 --gap-s 20 \
    --log "$LOG" -- "$@"
  local rc=$?
  echo "[$label] done rc=$rc $(date -Is)"
  return "$rc"
}

echo "=== STAGE 1: eval (full+win16) $(date -Is) ==="
sup eval1 bash "$HERE/eval_run.sh"

echo "=== STAGE 2: train win16_rep $(date -Is) ==="
sup rep bash "$HERE/train_all.sh" --arms "win16_rep"
python3 "$HERE/fill_config_hashes.py"

echo "=== STAGE 3: eval (含 win16_rep) $(date -Is) ==="
sup eval2 bash "$HERE/eval_run.sh"

echo "=== ALL DONE $(date -Is) ==="
