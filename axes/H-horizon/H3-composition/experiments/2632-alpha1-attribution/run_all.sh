#!/usr/bin/env bash
# 2632-alpha1-attribution · 一键复现（纯 CPU；接受率是 argmax 确定量 ⇒ 不需要 GPU 锁）。
#
# 步骤：分 checkpoint 跑 diag.py → merge_outputs.py 合并 → 跑 p1_only.py（P1 门禁）→
#       分 checkpoint 跑 probes.py → merge_outputs.py 合并探针。
# 产物与本目录已提交的 results.json / diag_records.jsonl / probes_output.json **逐字节一致**：
#   - acceptance 为 argmax 确定量，墙钟字段本实验不涉及；
#   - `probes_output.json` 的 P1 数值来自 `p1_only.py`（12 offsets, n=768），`P1_note` 为
#     冻结常量（`merge_outputs.py::P1_HISTORICAL_PREFIX`，不嵌本次运行值）⇒ 对输入确定。
#
# 用法：bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

CKPTS=(../2631-spec-three-arms/train/13m_s0 ../2631-spec-three-arms/train/13m_s1 ../2631-spec-three-arms/train/53m_s0)
TMP="$(mktemp -d)"
echo "[run_all] tmp=$TMP"

DIAG_DIRS=()
for ck in "${CKPTS[@]}"; do
  name="$(basename "$ck")"
  mkdir -p "$TMP/diag/$name"
  DIAG_DIRS+=("$TMP/diag/$name")
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 diag.py --ckpts "$ck" --out "$TMP/diag/$name" &
done
wait
python3 merge_outputs.py --mode diag --dirs "${DIAG_DIRS[@]}" --out .

python3 p1_only.py   # 写 probes_p1.json（P1 门禁，修正 off-by-one 后单独落盘）

PROBE_DIRS=()
for ck in "${CKPTS[@]}"; do
  name="$(basename "$ck")"
  mkdir -p "$TMP/probe/$name"
  PROBE_DIRS+=("$TMP/probe/$name")
  PROBE_OUT="$TMP/probe/$name" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 probes.py "$ck" &
done
wait
python3 merge_outputs.py --mode probes --dirs "${PROBE_DIRS[@]}" --out .

echo "[run_all] done; artifacts in $(pwd)"
