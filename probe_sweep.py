"""Standalone K-sweep probe for one checkpoint dir (ticket 09 diversity probe).

Usage: python3 probe_sweep.py NAME CKPT_DIR
Prints bpc/acc for K in {1,2,4,8} on the val split, then the verdict line.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_tts as P  # noqa: E402


def main() -> None:
    name, ckpt = sys.argv[1], sys.argv[2]
    blob = np.load("data/enwik8_full.npz")
    m = P.load_ckpt(ckpt)
    out = {}
    for k in (1, 2, 4, 8):
        r = P.bpc_k(m, blob["val"], k)
        out[f"K{k}"] = r
        print(f"{name} K={k}: bpc={r['bpc']} acc={r['acc']}", flush=True)
    try:
        sf = open(os.path.join(ckpt, "k_sweep.json"), "w")
    except OSError as e:
        raise RuntimeError(f"cannot write sweep results to {ckpt}: {e}") from e
    with sf as f:
        json.dump(out, f, indent=2)
    print("saved", os.path.join(ckpt, "k_sweep.json"))


if __name__ == "__main__":
    main()
