#!/usr/bin/env bash
# 2624 · 冒烟：20 步训练（两臂）+ 全电池缩档评测。只为验证接线（§17.5：20 步不外推时间）。
set -u
cd "$(dirname "$0")/../../../../.."
EXP=axes/1-time/1.1.4.1-explicit-memory/experiments/2624-carry-under-compression-quality
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
COMMON="--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 20 --batch-size 32 --lr 1e-3 --seed 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 --train-anchors 0 --out-root $EXP/smoke/train"

acquire
python3 train_depth_ar.py carry_mha $COMMON > "$EXP/smoke/train_mha.log" 2>&1; echo "carry_mha rc=$?"
python3 train_depth_ar.py carry_gqa $COMMON --n-kv-heads 2 > "$EXP/smoke/train_gqa.log" 2>&1; echo "carry_gqa rc=$?"
release

python3 - "$EXP" <<'PY'
import hashlib, json, os, sys
exp = sys.argv[1]
cfg = json.load(open(os.path.join(exp, "config.json")))
cfg["train"]["steps"] = 20
cfg["checkpoint"]["dir"] = "axes/1-time/1.1.4.1-explicit-memory/experiments/2624-carry-under-compression-quality/smoke/train"
cfg["checkpoint"]["sha256"] = {}
for n in ("carry_mha", "carry_gqa"):
    p = os.path.join(exp, "smoke", "train", n, "model.pt")
    cfg["checkpoint"]["sha256"][n] = hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump(cfg, open(os.path.join(exp, "smoke", "config.json"), "w"), indent=2, ensure_ascii=False)
print("smoke config written")
PY

python3 "$EXP/run.py" --smoke --config "$EXP/smoke/config.json" --out-dir "$EXP/smoke" 2>&1 | tail -25
