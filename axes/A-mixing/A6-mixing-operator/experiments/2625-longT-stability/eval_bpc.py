"""2625 · clean bpc 评测（native 口径：full 臂 = 全因果，与 2622/2623 同口径）。

与 2623 的电池同数据流（`masked_bpc`，window_seed=2，20 批 × bs16），用于：
① 确认臂的质量读数；② 短 T 不回归检查（与 2622 的 `full-256` / `full-512` 比 Δbpc）。

架构类候选（`temp` / `qk-norm`）必须在**评测时同样施加** —— 否则量的是另一个模型。

用法：python3 eval_bpc.py <arm> --candidate temp2.0 [--seq-len 1024] [--window 0]
                         [--out-root runs]      # 写 <run_dir>/eval.json
"""

import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402
from pathlm.depth_ar import window_anchor_keep_mask           # noqa: E402
from pathlm.data import load_enwik8_full                      # noqa: E402
from tools.evalkit.retention import masked_bpc                # noqa: E402

D, L, H, MLP = 256, 8, 8, 4
EVAL_BATCHES, EVAL_BS, WINDOW_SEED = 20, 16, 2


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "stability_driver_eval", os.path.join(HERE, "stability_driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--candidate", default="none")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    sd = load_driver()
    _train, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                               "data/enwik8_full.npz")
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=args.seq_len,
                        mlp_mult=MLP, dense=True, train_window=args.window)
    model = DepthARModel(cfg, vocab).cuda()
    kind, params = sd.parse_candidate(args.candidate)
    if kind == "temp":
        sd.apply_attn_scale(model, "temp", params["tau"])
    elif kind == "qk-norm":
        sd.apply_attn_scale(model, "qk-norm", 1.0, params.get("min_t", 0))
    run_dir = os.path.join(args.out_root, args.arm)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"),
                                     map_location="cuda"))
    model.eval()
    km = (None if args.window <= 0 else
          window_anchor_keep_mask(args.seq_len, args.window, 0, "cuda"))
    bpc = masked_bpc(model, eval_arr, km, n_batches=EVAL_BATCHES,
                     batch_size=EVAL_BS, window_seed=WINDOW_SEED, device="cuda")
    out = {"arm": args.arm, "candidate": args.candidate, "seq_len": args.seq_len,
           "window": args.window, "bpc_native": round(bpc, 6),
           "n_batches": EVAL_BATCHES, "batch_size": EVAL_BS,
           "window_seed": WINDOW_SEED,
           "native_mask": "release(causal)" if km is None else "window"}
    with open(os.path.join(run_dir, "eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[eval] {args.arm}: bpc_native={bpc:.6f} "
          f"(T={args.seq_len} w={args.window} cand={args.candidate})", flush=True)


if __name__ == "__main__":
    main()
