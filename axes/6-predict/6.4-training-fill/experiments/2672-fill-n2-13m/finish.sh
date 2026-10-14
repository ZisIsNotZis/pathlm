#!/usr/bin/env bash
# 2672 WP1b（接任段）收尾：等参数 rescore（GPU 锁内）→ 逐对 config 对平 → 汇聚 results.json。
# 只处理已落盘的 snap<TAG>-<arm>-s<seed>；缺的 run 自动跳过并由 aggregate.py 如实标注缺失。
set -u
D="$(cd "$(dirname "$0")" && pwd)"
ROOT=/home/z/vibe/pathlm
TAG="${TAG:-2k}"
E12="$ROOT/.scratch/23-depth-ar/evidence"
cd "$ROOT"

CK=""
for spec in fill:0 nofill:0 fill:1 nofill:1 projfill:0 projfill:1; do
  ARM="${spec%%:*}"; SEED="${spec##*:}"
  d="$D/snap${TAG}-${ARM}-s${SEED}"
  [ -f "$d/results.json" ] && CK="$CK $d"
done

echo "[$(date -Is)] FINISH checkpoints:$CK"
if [ -n "$CK" ]; then
  bash "$D/lock_exec.sh" python3 "$D/rescore_eval.py" --checkpoints $CK \
    --out "$D/rescore-snap${TAG}.json"
fi

# 同刻度种子对（唯一预期差异 = train.seed ⇒ 不许有 allowed）
for arm in fill nofill projfill; do
  a="$D/snap${TAG}-${arm}-s0"; b="$D/snap${TAG}-${arm}-s1"
  if [ -f "$a/results.json" ] && [ -f "$b/results.json" ]; then
    python3 "$D/config_match.py" --seed0 "$a" --seed1 "$b" \
      --note "L0 主门：同刻度（${TAG} 步）种子对，唯一预期差异 = train.seed" \
      --out "$D/config-parity-snap${TAG}-${arm}-s0-vs-s1.json"
  fi
done

# 同臂同 seed 对 12k 档（预期差异 = train.steps，逐条说明）
declare -A E12DIR=( [fill]="$E12/23c-H2-w0-12k" [nofill]="$E12/23e-nofill-12k" [projfill]="$E12/23e-projfill-12k" )
for arm in fill nofill projfill; do
  for seed in 0 1; do
    rd="$D/snap${TAG}-${arm}-s${seed}"; e0="${E12DIR[$arm]}"
    if [ -f "$rd/results.json" ] && [ -f "$e0/results.json" ]; then
      python3 "$D/config_match.py" --seed0 "$e0" --seed1 "$rd" --allow train.steps \
        --note "跨刻度参照：预期差异仅 train.steps（12k 档 vs ${TAG} 快照），其余字段须逐位相同" \
        --out "$D/config-parity-snap${TAG}-${arm}-s${seed}-vs-12k.json"
    fi
  done
done

python3 "$D/aggregate.py"
echo "[$(date -Is)] FINISH done"
