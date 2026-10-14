#!/usr/bin/env python3
"""2643 · 复用门 ②（`SCHEMA.md` §9「配置层 equality」）：把复用锚（2641 `U4-s0`）的
resolved config 与本实验的对照配置**逐字段**列表对照。任何应等字段不等 ⇒ exit 1。

可失败：把本实验 config.json 的 `base.mlp_mult` 改成 2 再跑，本脚本必须非零退出。

    python3 compare_anchor_config.py            # 打印逐字段表；不等则 exit 1
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
ANCHOR = os.path.join(ROOT, "axes/3-width/3.1.4-ffn-ratio/experiments",
                      "2641-ffn-ratio-frontier-13m")

# 应逐字段相等的项：(锚来源文件, 字段路径, 本实验字段路径)
MUST_MATCH = [
    ("config.json", ("base", "d_model"), ("base", "d_model")),
    ("config.json", ("base", "n_layers"), ("base", "n_layers")),
    ("config.json", ("base", "n_heads"), ("base", "n_heads")),
    ("config.json", ("base", "seq_len"), ("base", "seq_len")),
    ("config.json", ("base", "mlp_mult"), ("base", "mlp_mult")),
    ("config.json", ("base", "dense"), ("base", "dense")),
    ("config.json", ("base", "dropout"), ("base", "dropout")),
    ("config.json", ("base", "norm_cap"), ("base", "norm_cap")),
    ("config.json", ("base", "topology"), ("base", "topology")),
    ("config.json", ("base", "exit_mode"), ("base", "exit_mode")),
    ("config.json", ("base", "fill_kv"), ("base", "fill_kv")),
    ("config.json", ("training", "steps"), ("training", "steps")),
    ("config.json", ("training", "batch_size"), ("training", "batch_size")),
    ("config.json", ("training", "lr"), ("training", "lr")),
    ("config.json", ("training", "train_tokens"), ("training", "train_tokens")),
    ("config.json", ("eval", "metric"), ("eval", "metric")),
    ("results_U4_s0.json", ("seed",), ("training", "seed")),
]


def dig(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def main() -> int:
    # 变异入口（证明本门能变红）：PATHLM_2643_MY_CONFIG=/tmp/bad.json python3 …
    mine = json.load(open(os.environ.get("PATHLM_2643_MY_CONFIG",
                                         os.path.join(HERE, "config.json"))))
    cache = {}
    rows, bad = [], 0
    for src, apath, mpath in MUST_MATCH:
        if src not in cache:
            cache[src] = json.load(open(os.path.join(ANCHOR, src)))
        a, b = dig(cache[src], apath), dig(mine, mpath)
        ok = (a == b)
        bad += 0 if ok else 1
        rows.append((src + ":" + ".".join(apath), ".".join(mpath), a, b, ok))
    print("| 锚字段（2641） | 本实验字段 | 锚值 | 本值 | 相等 |")
    print("|---|---|---|---|---|")
    for a, b, va, vb, ok in rows:
        print(f"| `{a}` | `{b}` | `{va}` | `{vb}` | {'✓' if ok else '✗'} |")
    print()
    print("**不等字段（显式列出）**：" + ("无" if bad == 0 else f"{bad} 项不等"))
    print("**已知口径**：评测 = `depth_curve` 最终深度 clean bpc，`n_batches=20`、"
          "`batch_size=16`、eval generator `seed=2`、训练 `steps=12000`、`seed=0`。")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
