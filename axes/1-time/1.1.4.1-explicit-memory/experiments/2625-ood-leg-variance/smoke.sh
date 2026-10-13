#!/usr/bin/env bash
# 2625 · 冒烟：20 步训练（两条新臂）+ 全电池缩档评测（含复用臂）。只为验证接线与仪器自检
# （§17.5：20 步不外推时间/质量）。复用臂指向真实 checkpoint（sha256 断言照跑）。
set -u
cd "$(dirname "$0")/../../../../.."
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; continue; fi
    sleep 15; n=$((n+1)); [ $n -gt 80 ] && { echo 等锁超时; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; }

mkdir -p "$EXP/smoke/train"
COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 20 --batch-size 32 --lr 1e-3 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 --train-anchors 0 --n-kv-heads 2 --out-root $EXP/smoke/train"

acquire
python3 train_depth_ar.py gqa_carry_win_s1 $COMMON --seed 1 > "$EXP/smoke/train_s1.log" 2>&1; echo "s1 rc=$?"
python3 train_depth_ar.py gqa_carry_win_s2 $COMMON --seed 2 > "$EXP/smoke/train_s2.log" 2>&1; echo "s2 rc=$?"
release

python3 - "$EXP" <<'PY'
import hashlib, json, os, sys
exp = sys.argv[1]
cfg = json.load(open(os.path.join(exp, "config.json")))
cfg["train"]["steps"] = 20
cfg["checkpoint"]["dir"] = "axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance/smoke/train"
for n in ("gqa_carry_win_s1", "gqa_carry_win_s2"):
    p = os.path.join(exp, "smoke", "train", n, "model.pt")
    cfg["checkpoint"]["sha256"][n] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(cfg, open(os.path.join(exp, "smoke", "config.json"), "w"), indent=2, ensure_ascii=False)
print("smoke config written")
PY

python3 "$EXP/run.py" --smoke --config "$EXP/smoke/config.json" --out-dir "$EXP/smoke" 2>&1 | tail -30
