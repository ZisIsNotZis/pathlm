"""2625 · CPU 探针 3：已落盘 checkpoint 的「疤」指纹（跨 run 只读）。

对 2622/2623/2624 的 8 个 checkpoint 读权重，量测：
- `pos_embed` / `embed` 权重范数；
- 每层 `qkv` / `proj` 权重范数（重点：`blocks.1.qkv`）。

用途：确认爆炸臂留下的**永久指纹**（README/报告里引用的数字必须能回到本文件）。

用法：python3 probe_checkpoint_scar.py
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel        # noqa: E402

D, L, H, MLP = 256, 8, 8, 4
EXP_ROOT = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                        "experiments")
CKPTS = [
    ("full-256", "2622-longT-512", 256, 0),
    ("win128-256", "2622-longT-512", 256, 128),
    ("full-512", "2622-longT-512", 512, 0),
    ("win256-512", "2622-longT-512", 512, 256),
    ("full-1024", "2623-longT-1024", 1024, 0),
    ("win512-1024", "2623-longT-1024", 1024, 512),
    ("full-1024-seed1", "2624-longT-1024-seed1", 1024, 0),
    ("win512-1024-seed1", "2624-longT-1024-seed1", 1024, 512),
]


def main():
    out = {}
    for arm, exp, T, w in CKPTS:
        path = os.path.join(EXP_ROOT, exp, "runs", arm, "model.pt")
        if not os.path.exists(path):
            print(f"[scar] SKIP {arm}（无 checkpoint）")
            continue
        sd = torch.load(path, map_location="cpu")
        rec = {
            "seq_len": T, "window": w,
            "pos_embed_w_norm": round(float(sd["pos_embed.weight"].norm()), 4),
            "embed_w_norm": round(float(sd["embed.weight"].norm()), 4),
            "qkv_w_norm_by_layer": [round(float(sd[f"blocks.{i}.qkv.weight"]
                                               .norm()), 4) for i in range(L)],
            "proj_w_norm_by_layer": [round(float(sd[f"blocks.{i}.proj.weight"]
                                                .norm()), 4) for i in range(L)],
        }
        out[arm] = rec
        print(f"[scar] {arm:18s} T={T:4d} w={w:3d} pos={rec['pos_embed_w_norm']:6.2f} "
              f"emb={rec['embed_w_norm']:6.2f} L1.qkv={rec['qkv_w_norm_by_layer'][1]:8.2f} "
              f"L0.qkv={rec['qkv_w_norm_by_layer'][0]:6.2f}", flush=True)
    with open(os.path.join(HERE, "smoke", "ckpt_scar_probe.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
