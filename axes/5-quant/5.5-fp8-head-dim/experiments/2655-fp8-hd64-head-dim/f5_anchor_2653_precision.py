#!/usr/bin/env python3
"""F5 ③ 等价证据 · `2653` 的 `fp16` 臂在 `24×16@seed2` 上的 9 位小数单点读数（纯 CPU）。

为什么需要它：`2653/run.py::eval_arm` 把每臂 bpc 以 **6 位小数**落盘，而 F5 ③ 的等价证据
要把本机的 `fp16` 读数与 GPU 历史锚 `1.483569` 放到 1e-7 级上比。本脚本用**同一条**
`b21.b19.bpc_of` 路径在历史锚口径（`window_seed=2`、`n_batches=24`、`batch_size=16`、CPU）
上重算一次并保留 12 位小数。

边界：只读复用 `2653/fp8_arms.py` / `2653/run.py` / `2621/run.py`，**不修改 `2653` 的任何文件**；
结果只写本文件同级的 `f5-anchor-2653-rerun/precision_point.json`（`2655` 目录内的 F5 证据子目录）。

    cd /home/z/vibe/pathlm && python3 <expdir>/f5_anchor_2653_precision.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
P2653 = os.path.join(ROOT, "axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m")
OUT = os.path.join(HERE, "f5-anchor-2653-rerun", "precision_point.json")
sys.path.insert(0, ROOT)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2653 的入口（内含 `fp8_arms` 与 `2621` 的评测管线）；只读。
r2653 = _load_module(os.path.join(P2653, "run.py"), "fp8_vs_int8_2653")
fa, b21 = r2653.fa, r2653.b21

ANCHOR = 1.483569          # GPU 历史锚（2614/2610 口径，device=cuda；跨设备不作判据）
CALIBER = {"n_batches": 24, "batch_size": 16, "window_seed": 2, "device": "cpu"}


def main() -> int:
    t0 = time.time()
    cfg = json.load(open(os.path.join(P2653, "config.json")))
    parent = json.load(open(os.path.join(ROOT, cfg["parent"]["config"])))
    for k in ("model", "data", "tokens", "checkpoints"):
        cfg[k] = parent[k]
    T = cfg["model"]["seq_len"]
    _, eval_arr, _ = r2653.load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    trunk, ck = b21.b19.load_trunk(cfg, CALIBER["device"])
    trunk.eval()
    fn = fa.make_logits_fn(trunk, {"mode": "fp16"}, T, CALIBER["device"])
    bpc = b21.b19.bpc_of(fn, eval_arr, T, CALIBER["n_batches"], CALIBER["batch_size"],
                         CALIBER["window_seed"], CALIBER["device"])
    rec = {"arm": "fp16", "caliber": f"{CALIBER['n_batches']}×{CALIBER['batch_size']}"
                                     f"@seed{CALIBER['window_seed']}",
           "device": CALIBER["device"], "bpc": round(bpc, 12),
           "bpc_path": "2653/run.py::eval_arm 的同一路径（b21.b19.bpc_of）",
           "checkpoint": {k: ck[k] for k in ("path", "sha256", "sha256_verified")
                          if k in ck},
           "anchor_gpu": ANCHOR, "abs_diff_vs_anchor": abs(bpc - ANCHOR),
           "wall_seconds": round(time.time() - t0, 3)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(json.dumps(rec, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
