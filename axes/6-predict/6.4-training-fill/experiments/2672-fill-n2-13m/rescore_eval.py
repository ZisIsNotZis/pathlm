"""等参数软/硬提交剥离评测（2672 WP2，纯评测、无训练）。

在**同一个 checkpoint** 上用引擎既有 `pathlm.depth_ar.generation_quality` 重跑
soft / hard / latent 三种提交方式的 rescore_bpc（深度 0 / L/2 / L 三点）。
同 checkpoint ⇒ 权重逐位相同 ⇒ adapter/capacity 因素被剥离，唯一变量 = 提交规则。

用法：
    python3 rescore_eval.py --checkpoints DIR [DIR ...] --out OUT.json

对 seed-0 的既有 checkpoint，本脚本会把结果与 checkpoint 自带 `results.json`
的 `generation` 块逐字段比对（应逐位相同），作为「加载/复现正确」的可失败门。
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = "/home/z/vibe/pathlm"
sys.path.insert(0, ROOT)
from pathlm.data import load_enwik8_full          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,   # noqa: E402
                             generation_quality)


def load_checkpoint(run_dir: str, device: str):
    """从落盘 results.json 重建 config、从 model.pt 载权重（逐位回放训练点）。"""
    with open(os.path.join(run_dir, "results.json")) as f:
        meta = json.load(f)
    cfg = DepthARConfig(**meta["config"])
    model = DepthARModel(cfg, meta["vocab_size"]).to(device)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"{run_dir}: state_dict 不匹配 missing={missing[:5]} "
                         f"unexpected={unexpected[:5]}")
    model.eval()
    return model, cfg, meta


def _cmp_stored(rows, meta):
    """与 checkpoint 自带 generation 块逐字段比对；返回差异列表（空 = 逐位一致）。"""
    stored = meta.get("generation")
    if not stored:
        return None
    key = lambda r: (r["depth"], r["commit"])            # noqa: E731
    got = {key(r): r for r in rows}
    diffs = []
    for s in stored:
        g = got.get(key(s))
        if g is None:
            diffs.append(f"missing {key(s)}")
            continue
        for fld in ("rescore_bpc", "distinct2", "legal", "exit_top1_prob_tf"):
            if g[fld] != s[fld]:
                diffs.append(f"{key(s)}.{fld}: got {g[fld]} != stored {s[fld]}")
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commits", nargs="+", default=["soft", "hard", "latent"])
    args = ap.parse_args()

    _, eval_arr, vocab = load_enwik8_full(f"{ROOT}/.tmp/enwik8",
                                          f"{ROOT}/data/enwik8_full.npz")
    result = {}
    for rd in args.checkpoints:
        rd = rd.rstrip("/")
        name = os.path.basename(rd)
        t0 = time.time()
        model, cfg, meta = load_checkpoint(rd, args.device)
        depths = [0, cfg.n_layers // 2, cfg.n_layers]
        rows = generation_quality(model, eval_arr, prompt_len=64, n_new=128,
                                  depths=depths, commits=tuple(args.commits),
                                  seed=5, device=args.device)
        diffs = _cmp_stored(rows, meta)
        result[name] = {
            "run_dir": rd,
            "run": meta["run"],
            "seed": meta["train"]["seed"],
            "fill_kv": meta.get("fill_kv", True),
            "proj_fill": meta.get("proj_fill", False),
            "depths": depths,
            "commits": list(args.commits),
            "generation": rows,
            "stored_match": None if diffs is None else (len(diffs) == 0),
            "stored_diffs": diffs,
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[eval] {name}: depths={depths} stored_match="
              f"{result[name]['stored_match']} ({result[name]['elapsed_s']}s)",
              flush=True)
        if diffs:
            print(f"  DIFFS: {diffs}", flush=True)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[eval] wrote {args.out} ({len(result)} checkpoints)", flush=True)


if __name__ == "__main__":
    main()
