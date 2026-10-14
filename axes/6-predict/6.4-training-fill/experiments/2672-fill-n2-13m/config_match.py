"""L0 协议保真硬门：seed-1 run 与既有 seed-0 run 的 config 逐字段对平。

对比字段 = `config.*`（结构）+ `train.*`（**除 seed**）+ 顶层协议开关（填充语义、
退出模式、损失配方、拓扑/注意力/读出/合成等）。seed-0 记录早于部分字段面世 ⇒
「仅 seed-1 有」的字段必须恰好等于引擎缺省（协议中性），否则报差异。

用法：python3 config_match.py --seed0 DIR --seed1 DIR --out OUT.json
"""

import argparse
import json
import os
import sys

ROOT = "/home/z/vibe/pathlm"
sys.path.insert(0, ROOT)
try:  # 引擎缺省值的独立来源（config.* 字段由 asdict(DepthARConfig()) 产出）
    from dataclasses import asdict

    from pathlm.depth_ar import DepthARConfig
    DEFAULTS = {"config." + k: v for k, v in asdict(DepthARConfig()).items()}
except Exception:  # pragma: no cover - 引擎不可导入时不据缺省解释
    DEFAULTS = {}

TOPLEVEL = [
    "exit_mode", "fill_kv", "proj_fill", "aux_weight", "aux_anneal_to",
    "shallow_stopgrad", "dense_aux", "final_weight", "topology", "attn_sparse_k",
    "attn_linear", "linear_chunk", "train_window", "train_anchors", "n_kv_heads",
    "carry_write", "carry_chunk", "carry_state_dim", "recon_window", "recon_lambda",
    "synth", "readout_dims", "readout_gated", "readout_linear_resid",
    "readout_perdepth", "readout_wide_shared", "mod_ratio",
]

# 引擎缺省（= 协议中性值）：仅 seed-1 记录里有、且取此值的字段不构成差异。
NEUTRAL_DEFAULT = {
    "topology": "sequential", "attn_sparse_k": 0, "attn_linear": False,
    "linear_chunk": 64, "train_window": 0, "train_anchors": 0,
    "n_kv_heads": None, "carry_write": "none", "carry_chunk": 64,
    "carry_state_dim": None, "recon_window": 0, "recon_lambda": 0.0,
    "synth": None, "readout_dims": None, "readout_gated": False,
    "readout_linear_resid": False, "readout_perdepth": False,
    "readout_wide_shared": 0, "mod_ratio": 0.0, "final_weight": 1.0,
    # 引擎 CLI 缺省（train_depth_ar.py：--no-fill-kv = store_false(dest=fill_kv, 缺省 True)；
    # --proj-fill = store_true(缺省 False)）⇒ 旧记录里缺席时取此值不构成协议差异。
    "fill_kv": True, "proj_fill": False,
}


def norm(meta):
    d = {}
    for k, v in meta.get("config", {}).items():
        d[f"config.{k}"] = v
    for k, v in meta.get("train", {}).items():
        if k == "seed":
            continue
        d[f"train.{k}"] = v
    for k in TOPLEVEL:
        if k in meta:
            d[k] = meta[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", required=True)
    ap.add_argument("--seed1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow", action="append", default=[],
                    help="预期差异的字段名（如 train.steps）：记入 allowed_diffs，不计入 passed")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    m0 = json.load(open(os.path.join(args.seed0, "results.json")))
    m1 = json.load(open(os.path.join(args.seed1, "results.json")))
    n0, n1 = norm(m0), norm(m1)

    diffs, explained, only1 = [], [], []
    for k in sorted(set(n0) | set(n1)):
        if k not in n0:
            only1.append(k)
            if k in NEUTRAL_DEFAULT and n1[k] == NEUTRAL_DEFAULT[k]:
                explained.append({"field": k, "value": n1[k],
                                  "note": "seed-0 记录早于该字段面世；值 = 引擎缺省 ⇒ 协议中性"})
            elif k in DEFAULTS and n1[k] == DEFAULTS[k]:
                explained.append({"field": k, "value": n1[k],
                                  "note": f"seed-0 记录早于该字段面世；值 = DepthARConfig() 缺省 ⇒ 协议中性"})
            else:
                diffs.append({"field": k, "seed0": "<absent>", "seed1": n1[k]})
        elif k not in n1:
            diffs.append({"field": k, "seed0": n0[k], "seed1": "<absent>"})
        elif n0[k] != n1[k]:
            diffs.append({"field": k, "seed0": n0[k], "seed1": n1[k]})

    allowed = [d for d in diffs if d["field"] in args.allow]
    unexp = [d for d in diffs if d["field"] not in args.allow]
    out = {
        "seed0_dir": args.seed0, "seed1_dir": args.seed1,
        "note": args.note,
        "fields_compared": sorted(set(n0) | set(n1)),
        "n_fields": len(set(n0) | set(n1)),
        "diffs": diffs,
        "allowed_diffs": allowed,
        "allowed_fields": list(args.allow),
        "unexplained_diffs": unexp,
        "explained_absent_in_seed0": explained,
        "passed": len(unexp) == 0,
        "seed0_train_seed": m0["train"]["seed"],
        "seed1_train_seed": m1["train"]["seed"],
        "params_seed0": m0.get("params"), "params_seed1": m1.get("params"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[config_match] {args.seed0} vs {args.seed1}: passed={out['passed']} "
          f"diffs={len(diffs)} allowed={len(allowed)} unexplained={len(unexp)} "
          f"explained_absent={len(explained)} "
          f"seeds {out['seed0_train_seed']}->{out['seed1_train_seed']} "
          f"params {out['params_seed0']}->{out['params_seed1']}", flush=True)
    for d in diffs:
        print("  DIFF", d, flush=True)
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
