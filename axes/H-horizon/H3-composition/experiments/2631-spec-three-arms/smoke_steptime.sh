#!/usr/bin/env bash
# 2631-spec-three-arms · 先测步时（SCHEMA §17.8 / §20.3 / §8），300 步稳态后**才**定预算。
# 不借用任何其它实验的步时；两档规模各一次。整程 <10 min ⇒ §20.1 允许整程持锁一次。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
EXP=axes/H-horizon/H3-composition/experiments/2631-spec-three-arms
LOCK=/tmp/pathlm_gpu.lock

acquire() {
  local n=0
  while ! mkdir "$LOCK" 2>/dev/null; do
    local hp stale=0
    hp=$(cat "$LOCK/holder_pid" 2>/dev/null)
    if [ -n "$hp" ] && ! kill -0 "$hp" 2>/dev/null; then stale=1; fi
    if [ -z "$hp" ] && [ -d "$LOCK" ] && [ $(( $(date +%s) - $(stat -c %Y "$LOCK") )) -gt 900 ]; then stale=1; fi
    if [ "$stale" -eq 1 ]; then
      rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null
      rmdir "$LOCK" 2>/dev/null || true
      continue
    fi
    sleep 15; n=$((n+1))
    [ $((n % 8)) -eq 0 ] && echo "  [wait-lock] 已等 $((n*15))s holder=$hp $(date -Is)"
    [ $n -gt 200 ] && { echo "等锁超时（>50min）"; exit 9; }
  done
  echo $$ > "$LOCK/holder_pid"; touch "$LOCK/heartbeat"
}
release() { rm -f "$LOCK/holder_pid" "$LOCK/heartbeat" 2>/dev/null; rmdir "$LOCK" 2>/dev/null || true; }
trap release EXIT

OUT="$EXP/smoke/steps300"
mkdir -p "$OUT"

echo "GPU free before: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date -Is)"
acquire
echo "GPU free (lock held): $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"

for spec in "13m_s0:config_13m.json" "53m_s0:config_53m.json"; do
  name=${spec%%:*}; cfg=${spec#*:}
  echo "=== [smoke $name] $(date -Is) ==="
  /usr/bin/time -f "SMOKE $name WALL %e s MAXRSS %M KB" \
    python3 train_m1.py "$name" --config "$EXP/$cfg" --steps 300 --seed 0 \
      --out-root "$OUT" > "$OUT/$name.stdout" 2>&1
  tail -1 "$OUT/$name.stdout"
  python3 - "$OUT/$name/train_log.jsonl" "$name" <<'PY'
import json, sys
p, name = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(p) if l.strip()]
lo, hi = rows[0], rows[-1]
steps = hi["step"] - lo["step"]
ms = (hi["min"] - lo["min"]) * 60000 / max(steps, 1)
print(f"  {name:8s} {ms:8.1f} ms/step (step {lo['step']} -> {hi['step']})")
PY
  python3 - "$OUT/$name/results.json" "$name" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  {sys.argv[2]}: params={r['params']} peak_gpu_mem_mb={r.get('peak_gpu_mem_mb')} "
      f"wall_min={r.get('wall_minutes')} bpc={r['bpc']['bpc']} acc={r['bpc']['next_token_acc']}")
PY
done

echo "GPU free after: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) $(date -Is)"
echo "DONE $(date -Is)"
