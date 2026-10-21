#!/usr/bin/env bash
# 夹具驱动：真正跑的是仓库根的训练器（见 .scratch/run-hash-smoke/SMOKE.md）。
set -euo pipefail
cd "$(dirname "$0")/../../.."
python3 train_depth_ar.py hash-smoke --d 64 --layers 2 --heads 2 --seq-len 64 \
  --batch-size 2 --steps 2 --dense --out-root .scratch/run-hash-smoke/exp/train
