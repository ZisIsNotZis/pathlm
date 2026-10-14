#!/usr/bin/env bash
# 2672 链式驱动：逐臂取放锁训练（fill → nofill），随后 seed-1 评测 + config 对平 + 汇聚。
# 每臂走 run_arm.sh 的 §20 取放锁（等锁 ≤10 min）；缺失 run 目录的后继步骤自动跳过。
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/z/vibe/pathlm
cd "$ROOT"

echo "[$(date -Is)] CHAIN start (pid $$)" >> "$D/run_log.txt"

bash "$D/run_arm.sh" fill 1
bash "$D/run_arm.sh" nofill 1

sleep 5
CK=""
[ -f "$D/run1-fill-s1/results.json" ] && CK="$CK $D/run1-fill-s1"
[ -f "$D/run1-nofill-s1/results.json" ] && CK="$CK $D/run1-nofill-s1"
if [ -n "$CK" ]; then
  python3 "$D/rescore_eval.py" --checkpoints $CK --out "$D/rescore-seed1.json" >> "$D/run_log.txt" 2>&1
fi
if [ -f "$D/run1-fill-s1/results.json" ]; then
  python3 "$D/config_match.py" --seed0 .scratch/23-depth-ar/evidence/23c-H2-w0-12k \
    --seed1 "$D/run1-fill-s1" --out "$D/config-parity-fill.json" >> "$D/run_log.txt" 2>&1
fi
if [ -f "$D/run1-nofill-s1/results.json" ]; then
  python3 "$D/config_match.py" --seed0 .scratch/23-depth-ar/evidence/23e-nofill-12k \
    --seed1 "$D/run1-nofill-s1" --out "$D/config-parity-nofill.json" >> "$D/run_log.txt" 2>&1
fi
python3 "$D/aggregate.py" >> "$D/run_log.txt" 2>&1

echo "[$(date -Is)] CHAIN done" >> "$D/run_log.txt"
