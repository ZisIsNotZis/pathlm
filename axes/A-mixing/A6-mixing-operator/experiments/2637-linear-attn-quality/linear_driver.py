"""2637 · 线性注意力质量腿 —— 训练驱动器（实验本地；**不改任何共享文件**）。

背景（`2616`）：线性注意力（φ = elu+1 的归一化线性核，分块 O(T) 实装）在本仓三次
训练期**停摆**（进程停在 `loss.backward()`、CPU 满载、GPU 无进展），停摆前活性量
`den_min` 触到 clamp 下限 1e-6、`phi_min` 触 0、gnorm 尖峰 5.1e3。停摆是**非确定性**的
（独立进程跑 1200–3000 步正常）。

本驱动器做两件事：
1. 跑**同一实装**（`attn_linear=True`）以复现/排除停摆，并可选打开 `faulthandler`
   抓栈（`--stall-dump`）；
2. 提供**稳定化分母**档（`--qk-norm global`）：给每个 block 的 `qkv` 挂 2625 的
   QK-Norm hook（逐头 L2 归一 q/k，**零新增参数**），把 φ 的输入钉在有界区间 ⇒
   `φ = elu+1` 不再下溢到 0、`den` 不再塌到 1e-6。这只改**数值条件**，不改核形式
   （仍是 elu+1 归一化线性核），属契约 §「可换实现：稳定化分母」。

契约：同目录 `README.md`（判据线预登记）。
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                        # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel             # noqa: E402
from pathlm.data import load_enwik8_full                            # noqa: E402

D, L, H, MLP = 256, 8, 8, 4
QK_NORM_MIN_T = 512

W1_DRIVER = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                         "experiments", "2625-longT-stability",
                         "stability_driver.py")


def _load_w1():
    spec = importlib.util.spec_from_file_location("w1_stability_driver_lin", W1_DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def apply_qk_norm(model: DepthARModel, mode: str) -> dict:
    """`none` ⇒ 不挂 hook；`global` ⇒ 全部层逐头归一；`from512` ⇒ 只在 T > 512 归一。"""
    if mode == "none":
        return {"attn_scale_mode": "none", "n_qkv_hooks": 0, "active": False}
    min_t = 0 if mode == "global" else QK_NORM_MIN_T
    return _load_w1().apply_attn_scale(model, "qk-norm", 1.0, min_t=min_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--linear", action="store_true", help="用线性注意力块")
    ap.add_argument("--linear-chunk", type=int, default=64)
    ap.add_argument("--qk-norm", choices=("none", "global", "from512"),
                    default="none")
    ap.add_argument("--stall-dump", type=int, default=0,
                    help=">0 时用 faulthandler 每 N 秒 dump 训练栈（诊断停摆）")
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    args = ap.parse_args()

    if args.stall_dump:
        import faulthandler
        faulthandler.dump_traceback_later(args.stall_dump, repeat=True)

    run_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)

    train_arr, _eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                   "data/enwik8_full.npz")
    torch.manual_seed(args.seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=args.seq_len,
                        mlp_mult=MLP, dense=True, attn_linear=bool(args.linear),
                        linear_chunk=args.linear_chunk)
    model = DepthARModel(cfg, vocab).cuda()
    n_before = sum(p.numel() for p in model.parameters())
    patch = apply_qk_norm(model, args.qk_norm)
    n_after = sum(p.numel() for p in model.parameters())
    patch["params_added"] = n_after - n_before
    patch["attn_linear"] = bool(args.linear)
    patch["linear_chunk"] = args.linear_chunk
    patch["block_class"] = type(model.blocks[0]).__name__

    print(f"[driver] {args.run_name}: {n_after/1e6:.3f}M params linear={args.linear} "
          f"qk_norm={args.qk_norm} patch={patch} T={args.seq_len} steps={args.steps} "
          f"bs={args.batch_size} lr={args.lr} seed={args.seed}", flush=True)

    t0 = time.time()
    tcfg = {"steps": args.steps, "batch_size": args.batch_size,
            "lr": args.lr, "seed": args.seed}
    wall, avg_ce, avg_gate, stats = TDA.train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode="uniform")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    result = {"run": args.run_name, "patch": patch, "seq_len": args.seq_len,
              "steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
              "seed": args.seed, "params": n_after,
              "tokens": args.steps * args.batch_size * args.seq_len,
              "wall_min": round((time.time() - t0) / 60, 2),
              "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
              "gnorm_monitor": stats,
              "avg_train_depth_ce": [round(c, 4) for c in avg_ce]}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[driver] {args.run_name} done wall={wall/60:.2f}min "
          f"gnorm_max={stats['gnorm_max']} mean={stats['gnorm_mean']} "
          f"skips={stats['grad_skips']} peak={result['peak_gpu_mem_mb']}MB",
          flush=True)


if __name__ == "__main__":
    main()
