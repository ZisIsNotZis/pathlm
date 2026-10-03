#!/usr/bin/env bash
# 2623 编排补跑：先出 full+win16 的主结果，再补 win16_rep，最后重跑评测纳入复现臂。
# 每一步都用标准锁协议；锁超时（rc=9）就重试（共享卡竞争下的稳健补跑）。
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

retry() {   # retry <label> <cmd...>：rc=9（锁超时）就重试
  local label="$1"; shift
  while true; do
    "$@"; rc=$?
    [ "$rc" -ne 9 ] && { echo "[$label] done rc=$rc $(date -Is)"; return $rc; }
    echo "[$label] lock timeout, retry after 20s $(date -Is)"
    sleep 20
  done
}

echo "=== STAGE 1: eval (full+win16) $(date -Is) ==="
retry eval1 bash "$HERE/eval_run.sh"

echo "=== STAGE 2: train win16_rep $(date -Is) ==="
retry rep bash "$HERE/train_all.sh" --arms "win16_rep"
python3 "$HERE/fill_config_hashes.py"

echo "=== STAGE 3: eval (含 win16_rep) $(date -Is) ==="
retry eval2 bash "$HERE/eval_run.sh"

echo "=== ALL DONE $(date -Is) ==="
