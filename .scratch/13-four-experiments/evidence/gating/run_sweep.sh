#!/bin/bash
set -e
cd /home/z/vibe/pathlm
echo "START $(date)"
python3 probe_gate.py --config configs/C1M.json --ckpt .tmp/ckpts4/C1M_s0 --out .tmp/gating/C1M_s0_gate.jsonl
python3 probe_gate.py --config configs/C1.json  --ckpt .tmp/ckpts4/C1_s1  --out .tmp/gating/C1_s1_gate.jsonl
python3 probe_gate.py --config configs/C2M.json --ckpt .tmp/ckpts5/C2M   --out .tmp/gating/C2M_gate.jsonl
python3 probe_gate.py --config configs/C2.json  --ckpt .tmp/ckpts5/C2    --out .tmp/gating/C2_gate.jsonl
echo "ALL_DONE $(date)"
