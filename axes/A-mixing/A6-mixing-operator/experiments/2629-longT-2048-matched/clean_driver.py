"""2629 · T=2048 同收敛复测（matched-step @6000）—— 训练驱动器（实验本地；**不改任何共享文件**）。

背景：2628（ticket 45-L1）在 QK-Norm 修复后拿到 T=2048 的 **3000 步快照**（full 末 loss
1.5423），而 2626 的 T=1024 点是 **6000 步**（full 末 loss 1.2146）⇒ 四点收敛度不等。
本实验把 T=2048 两臂都跑到 **6000 步**（与 T=1024 点同 step），其余逐字段一致。

为什么要一个本地驱动器：引擎 `train_depth_ar.py` 把「QK-Norm」硬编码在模型里——
没有这个开关；它只在 2625 的实验本地驱动器里以 **forward hook** 形式实现。本驱动器
**直接复用 2625 已通过单测 + 变异验证的 hook 实现**（`stability_driver.py`，只读引用，
单一实现来源），再**逐字调用** `train_depth_ar.train()`。

候选（唯一变量 = 训练步数）：
- 两个臂 = T=2048 × {full（`train_window=0`）, window（= T/2 = 1024，同相对刻度 50%）}，
  **全部**启用长度条件档 `qk-normfrom512`（`min_t=512`）。
- 因 `min_t=512` 且 `seq_len=2048 > 512`：两个臂**都真的挂 hook**（各 8 个、逐头归一 q/k）。
- 两臂步数 = **6000**（与 2626 的 T=1024 点一致；token 预算 = 98.304M/臂）。

训练中探针（`probe.jsonl`，每 `--probe-every` 步一行）：逐层 `logit_max` / `logit_rms`
/ `qkv_w_norm` / 残差流范数。探针只在**训练**前向计数（`model.training` 门），
故 `window_mask_probe`（eval 模式）不会让步号错位。

契约：同目录 `README.md`（判据线逐字抄录）。
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
#: 长度条件档的阈值（= 2625 推荐的 `qk-normfrom512`；工程选择，非机制推导）
QK_NORM_MIN_T = 512

#: 2625 实验本地驱动器（只读引用；qk-norm hook + 探针的**单一实现来源**）
W1_DRIVER = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                         "experiments", "2625-longT-stability",
                         "stability_driver.py")


def _load_w1():
    spec = importlib.util.spec_from_file_location("w1_stability_driver", W1_DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W1 = _load_w1()


def apply_qk_norm(model: DepthARModel) -> dict:
    """套用长度条件档 QK-Norm（`min_t=512`）：T ≤ 512 零 hook，T > 512 逐头归一。"""
    return W1.apply_attn_scale(model, "qk-norm", 1.0, min_t=QK_NORM_MIN_T)


# --------------------------------------------------------------- 训练中探针

class StepProbe:
    """每 `every` 个**训练**前向采一次逐层统计（对齐训练日志步号）。

    只对 `model.training is True` 的前向计数：`train()` 开跑前会用
    `window_mask_probe`（eval 模式）做两次前向，那些不计入 ⇒ 探针步号与
    `train_log.jsonl` 的 `step` 逐点对齐。
    """

    def __init__(self, model: DepthARModel, every: int = 200, nl: int = 64):
        self.model, self.every, self.nl = model, every, nl
        self.n = 0                       # 已见的训练前向数
        self.layers: list[dict] = []
        self.resid_in: float | None = None
        self.snapshots: list[dict] = []

    def attach(self) -> "StepProbe":
        self.model.register_forward_pre_hook(self._pre)
        self.model.register_forward_hook(self._post)
        for i, blk in enumerate(self.model.blocks):
            blk.register_forward_pre_hook(
                lambda m, a, _i=i: self._record_in(_i, a[0]))
            blk.register_forward_hook(
                lambda m, a, o, _i=i: self._record_out(_i, o))
            blk.qkv.register_forward_hook(
                lambda m, a, o, _i=i: self._record_qkv(_i, o))
        return self

    def _pre(self, module, args):
        if self.model.training:
            self.n += 1

    def _post(self, module, inp, out):
        if self._on():
            self.snapshots.append(self._snapshot())
        self.layers, self.resid_in = [], None

    def _on(self) -> bool:
        # 训练步号（0-based）= n − 1（本次前向已经计入）
        return self.model.training and self.n >= 1 and (self.n - 1) % self.every == 0

    @torch.no_grad()
    def _record_in(self, i, x):
        if self.model.training and self.resid_in is None:
            self.resid_in = float(x.detach().float().norm(dim=-1).mean())

    @torch.no_grad()
    def _record_out(self, i, out):
        if not self._on():
            return
        h = out[0] if isinstance(out, tuple) else out
        self._slot(i)["resid_out"] = float(h.detach().float().norm(dim=-1).mean())

    @torch.no_grad()
    def _record_qkv(self, i, out):
        if not self._on():
            return
        d = out.shape[-1] // 3
        nh = self.model.cfg.n_heads
        dh = d // nh
        B, T = out.shape[0], out.shape[1]
        q = out[..., :d].detach().float().reshape(B, T, nh, dh)[:, -self.nl:]
        k = out[..., d:2 * d].detach().float().reshape(B, T, nh, dh)
        scores = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
                  / (dh ** 0.5))                                  # [B,nh,nl,T]
        dev = scores.device
        irow = (T - self.nl + torch.arange(self.nl, device=dev)).view(1, 1, -1, 1)
        jcol = torch.arange(T, device=dev).view(1, 1, 1, -1)
        allow = jcol <= irow                                      # 全因果口径
        neg = torch.finfo(scores.dtype).min
        logit = scores.masked_fill(~allow, neg)
        sm = logit.softmax(-1)
        self._slot(i).update({
            "qkv_w_norm": float(self.model.blocks[i].qkv.weight.detach()
                                .float().norm()),
            "logit_rms": float(torch.sqrt((scores ** 2 * allow).sum(-1)
                                          / allow.sum(-1)).mean()),
            "logit_max": float(logit.amax(-1).mean()),
            "top1": float(sm.amax(-1).mean()),
            "entropy": float((-(sm.clamp_min(1e-12).log() * sm)
                              .sum(-1)).mean()),
        })

    def _slot(self, i: int) -> dict:
        while len(self.layers) <= i:
            self.layers.append({})
        return self.layers[i]

    def _snapshot(self) -> dict:
        m = self.model
        return {"step": self.n - 1,
                "resid_in": self.resid_in,
                "pos_w_norm": float(m.pos_embed.weight.detach().float().norm()),
                "layers": [dict(x) for x in self.layers]}


# --------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default="qk-normfrom512")
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--probe-every", type=int, default=200)
    args = ap.parse_args()

    run_dir = os.path.join(args.out_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl", "probe.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)

    train_arr, _eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                   "data/enwik8_full.npz")
    torch.manual_seed(args.seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=args.seq_len,
                        mlp_mult=MLP, dense=True, train_window=args.window)
    model = DepthARModel(cfg, vocab).cuda()
    n_params_before = sum(p.numel() for p in model.parameters())

    patch_info: dict = {"candidate": args.candidate}
    if args.candidate == "qk-normfrom512":
        patch_info.update(apply_qk_norm(model))
    elif args.candidate in ("none", "baseline"):
        patch_info.update({"active": False, "n_qkv_hooks": 0})
    else:
        raise SystemExit(f"未知候选 {args.candidate!r}")
    n_params_after = sum(p.numel() for p in model.parameters())
    patch_info["params_added"] = n_params_after - n_params_before

    probe = StepProbe(model, every=args.probe_every, nl=64).attach()

    print(f"[driver] {args.run_name}: {n_params_after/1e6:.3f}M params "
          f"cand={args.candidate} patch={patch_info} T={args.seq_len} "
          f"w={args.window} steps={args.steps} bs={args.batch_size} "
          f"lr={args.lr} seed={args.seed}", flush=True)

    t0 = time.time()
    tcfg = {"steps": args.steps, "batch_size": args.batch_size,
            "lr": args.lr, "seed": args.seed}
    wall, avg_ce, avg_gate, stats = TDA.train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode="uniform")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    with open(os.path.join(run_dir, "probe.jsonl"), "w") as f:
        for s in probe.snapshots:
            f.write(json.dumps(s) + "\n")

    w_final = {f"blocks.{i}.qkv": round(float(model.blocks[i].qkv.weight
                                              .detach().float().norm()), 3)
               for i in range(len(model.blocks))}
    result = {"run": args.run_name, "candidate": args.candidate,
              "patch": patch_info, "seq_len": args.seq_len,
              "window": args.window, "steps": args.steps,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "params": n_params_after,
              "tokens": args.steps * args.batch_size * args.seq_len,
              "wall_min": round((time.time() - t0) / 60, 2),
              "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
              "gnorm_monitor": stats,
              "avg_train_depth_ce": [round(c, 4) for c in avg_ce],
              "final_qkv_w_norm": w_final,
              "n_probe_rows": len(probe.snapshots)}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[driver] {args.run_name} done wall={wall/60:.2f}min "
          f"gnorm_max={stats['gnorm_max']} mean={stats['gnorm_mean']} "
          f"skips={stats['grad_skips']} peak={result['peak_gpu_mem_mb']}MB "
          f"L1.qkv={w_final.get('blocks.1.qkv')} hooks={patch_info.get('n_qkv_hooks')}",
          flush=True)


if __name__ == "__main__":
    main()
