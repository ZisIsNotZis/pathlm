"""R7（WI-16）步时探针：先测步时再定预算（SCHEMA §17.8）。

按契约 §6，成本必须 = 臂数 × **本实验实测步时**，不得借用其它实验的步时。
本脚本在**一个进程内**（CPU 启动开销只付一次）依次跑每个臂的 60 个真实训练步
（forward + backward + clip + AdamW step，bf16 autocast，bs32 × seq256），报
ms/step 与推算的 12k 步墙钟。真训练仍走 `train_depth_ar.py`（同一份 train()）。

用法：python3 .scratch/52-r7-early-exit/step_time_probe.py [--steps 60]
"""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pathlm.data import load_enwik8_full, batch
from pathlm.depth_ar import DepthARConfig, DepthARModel, sample_exit_depths
from pathlm.mod_router import MoDModel

ARMS = {
    # name: (cfg-kwargs, dense_aux?, exit_mode, shallow_weight, model_cls)
    "stock_dense_L8": (dict(dense=True), False, "full", 1.0, DepthARModel),
    "c1_legacy_L8": (dict(carry_gate=True, aux_mode="renorm"), False, "uniform",
                     1.0, DepthARModel),
    "c1_fair_L8": (dict(carry_gate=True, aux_mode="additive"), False, "uniform",
                   0.1, DepthARModel),
    "ls_fair_L8_nogate": (dict(carry_gate=False, aux_mode="additive"), False,
                          "uniform", 0.1, DepthARModel),
    "mod_L16_r0.5": (dict(dense=True, n_layers=16, mod_ratio=0.5), False, "full",
                     1.0, MoDModel),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", default=".scratch/52-r7-early-exit/step_time.json")
    args = ap.parse_args()

    train_arr, _, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    out = {"steps": args.steps, "warmup": args.warmup, "bs": args.bs,
           "seq": args.seq, "arms": {}}
    gpu_before = torch.cuda.memory_allocated() / 2 ** 20
    for name, (kw, dense_aux, exit_mode, w, cls) in ARMS.items():
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        cfg = DepthARConfig(d_model=256, n_layers=kw.pop("n_layers", 8),
                            n_heads=8, seq_len=args.seq, mlp_mult=4, **kw)
        model = cls(cfg, vocab).cuda() if cls is DepthARModel else \
            cls(cfg, vocab, router_weight=0.01).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 200, 0.1 + 0.9 * 0.5
                               * (1 + torch.cos(torch.tensor(3.14159 * s / 12000)).item())))
        rng = torch.Generator().manual_seed(0)
        L = cfg.n_layers
        ts = []
        for step in range(args.warmup + args.steps):
            x, _ = batch(train_arr, args.bs, args.seq, rng)
            x = x.cuda()
            if step == args.warmup:
                torch.cuda.synchronize()
            t0 = time.time() if step >= args.warmup else None
            depths = (torch.full((args.bs, args.seq), L, dtype=torch.long).cuda()
                      if exit_mode == "full" else
                      sample_exit_depths(args.bs, args.seq, L, rng).cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, aux = model(x, exit_depths=depths, shallow_weight=w,
                                  final_weight=1.0, shallow_stopgrad=False,
                                  dense_aux=dense_aux)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if t0 is not None:
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
        ms = sorted(ts)[len(ts) // 2] * 1000
        peak = torch.cuda.max_memory_allocated() / 2 ** 20
        out["arms"][name] = {
            "ms_per_step_median": round(ms, 2),
            "ms_per_step_mean": round(sum(ts) / len(ts) * 1000, 2),
            "params": sum(p.numel() for p in model.parameters()),
            "peak_gpu_mem_mb": round(peak, 1),
            "est_12k_min": round(ms * 12000 / 1000 / 60, 2),
            "loss": round(float(loss.detach()), 4),
            "gnorm": round(float(gn), 3),
        }
        print(f"[probe] {name}: {ms:.1f} ms/step  est 12k = "
              f"{ms * 12000 / 1000 / 60:.1f} min  peak {peak:.0f} MB  "
              f"params {out['arms'][name]['params']}", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    out["gpu_allocated_before_mb"] = round(gpu_before, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out["arms"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
