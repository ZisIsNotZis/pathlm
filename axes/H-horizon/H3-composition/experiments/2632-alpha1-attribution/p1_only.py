"""2632 · P1 单独重跑（incremental == batched 门禁）。

`probes.py` 的 P1 早先有一处 off-by-one（增量侧漏最后一位却按 L 除），
本脚本用修正后的 `probes.inc_vs_batched` 在 12 个 segment 上重跑，写
`probes_p1.json`。用途：排除「增量解码与批量前向不一致」这一解释。
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import diag  # noqa: E402
import probes  # noqa: E402

ROOT = diag.ROOT
SPEC = diag.SPEC_DIR


def main() -> int:
    val = np.load(os.path.join(ROOT, "data/enwik8_full.npz"))["val"]
    offsets = [i * 5000 for i in range(12)]
    out = {}
    for ck in ("13m_s0", "13m_s1", "53m_s0"):
        m = diag.load_ckpt(os.path.join(SPEC, "train", ck))
        r = probes.inc_vs_batched(m, val, offsets, L=64)
        out[ck] = r
        print(ck, "batched=%.6f incremental=%.6f n=%d equal=%s"
              % (r["batched"], r["incremental"], r["n"], r["batched"] == r["incremental"]))
        del m
    with open(os.path.join(HERE, "probes_p1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
