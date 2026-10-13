"""2625 · 稳定性修复驱动器（实验本地；**不改任何共享文件**）。

为什么要一个本地驱动器：2623/2624 的入口 `train_depth_ar.py` 把「优化器 / LR 调度 /
梯度裁剪」硬编码在 `train()` 里，候选单因素（warmup、clip、weight decay、注意力 logit
尺度）无法用 CLI 表达。本驱动器：

1. `import train_depth_ar` 并**逐字调用它的 `train()`**（数据流、随机数消耗、日志字段、
   梯度监控全部一致）；候选补丁只打在**模型/优化器对象**上：
   - `initpos<σ>` / `initemb<σ>`：构造后重设 `pos_embed` / `embed` 初始化尺度；
   - `temp<τ>`：给每个 block 的 `qkv` 挂 post-hook，把 q 缩小 `1/τ`（= logits 除以 τ）；
   - `qk-norm`：同 hook，把 q/k 按 head 做 L2 归一（乘 √d_h，无新增参数）；
   - `wd<v>`：进程内替换 `torch.optim.AdamW` 的 `weight_decay` 实参；
   - `warmup<N>`：进程内替换 `LambdaLR` 的调度函数为「线性到 1 的 N 步预热 + 同一余弦」；
   - `clip<v>`：进程内替换 `clip_grad_norm_` 的阈值实参。
2. 训练中探针（`probe.jsonl`，每 `--probe-every` 步一行）：逐层残差流范数、`qkv`
   权重范数、注意力 logits 的 RMS/最大/top1/熵（只采最后 64 个查询行、全因果口径）。

契约：同目录 `README.md`（判据线逐字抄录）。
"""

import argparse
import json
import math
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


# --------------------------------------------------------------- 候选补丁

def apply_init_scale(model: DepthARModel, pos_std: float | None,
                     emb_std: float | None) -> dict:
    """构造后重设位置/输入嵌入的初始化尺度（`None` = 不动）。"""
    out = {}
    if pos_std is not None:
        torch.nn.init.normal_(model.pos_embed.weight, std=pos_std)
        out["pos_embed_std"] = pos_std
    if emb_std is not None:
        torch.nn.init.normal_(model.embed.weight, std=emb_std)
        out["embed_std"] = emb_std      # tied unembed 同表 ⇒ 读出表一并变（口径）
    return out


def apply_attn_scale(model: DepthARModel, mode: str, tau: float,
                     min_t: int = 0) -> dict:
    """给每个 block 的 `qkv` 挂 post-hook：`temp` 缩 q，`qk-norm` 归一 q/k。

    `min_t > 0` ⇒ 只在 `seq_len > min_t` 时施加（长度条件开关；短 T 路径**逐位
    不变**：`n_qkv_hooks == 0`，与基线同一计算图）。
    """
    n_heads = model.cfg.n_heads
    n_hooks = 0
    if min_t > 0 and model.cfg.seq_len <= min_t:
        return {"attn_scale_mode": mode, "tau": tau, "min_t": min_t,
                "n_qkv_hooks": 0, "active": False}

    def make_hook(_nh, _mode, _tau):
        def hook(module, inp, out):
            d = out.shape[-1] // 3
            q, k, v = out[..., :d], out[..., d:2 * d], out[..., 2 * d:]
            if _mode == "temp":
                q = q / _tau
            else:                                    # qk-norm
                dh = d // _nh
                shp = q.shape[:-1] + (_nh, dh)
                qf, kf = q.reshape(shp), k.reshape(shp)
                q = (qf / qf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                     * (dh ** 0.5)).reshape_as(q)
                k = (kf / kf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                     * (dh ** 0.5)).reshape_as(k)
            return torch.cat([q, k, v], dim=-1)
        return hook

    for blk in model.blocks:
        if not hasattr(blk, "qkv"):
            continue
        blk.qkv.register_forward_hook(make_hook(n_heads, mode, tau))
        n_hooks += 1
    return {"attn_scale_mode": mode, "tau": tau, "min_t": min_t,
            "n_qkv_hooks": n_hooks, "active": True}


def patch_weight_decay(value: float) -> dict:
    orig = torch.optim.AdamW

    def wrapped(*a, **kw):
        kw["weight_decay"] = value
        return orig(*a, **kw)
    torch.optim.AdamW = wrapped
    return {"weight_decay": value}


def patch_warmup(warmup: int, steps: int) -> dict:
    """`LambdaLR` 调度换成「线性 N 步预热 + 同一余弦」（唯一差别 = 预热长度）。"""
    orig = torch.optim.lr_scheduler.LambdaLR

    def cosine(s):
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))

    def wrapped(opt, lr_lambda, *a, **kw):
        return orig(opt, lambda s: min((s + 1) / warmup, cosine(s)), *a, **kw)
    torch.optim.lr_scheduler.LambdaLR = wrapped
    return {"warmup_steps": warmup}


def patch_clip_threshold(value: float) -> dict:
    orig = torch.nn.utils.clip_grad_norm_

    def wrapped(params, max_norm, *a, **kw):
        return orig(params, value, *a, **kw)
    torch.nn.utils.clip_grad_norm_ = wrapped
    return {"clip_max_norm": value}


# --------------------------------------------------------------- 训练中探针

class ProbeHook:
    """每 `every` 步采一次逐层统计；只采最后 `nl` 行（成本可控）。

    `step_ref` 是 `{"n": int}` 共享计数器（由 model 的 forward_pre_hook 递增）——
    `train()` 每步恰好调用 `model(...)` 一次 ⇒ step 号与训练日志逐点对齐。
    """

    def __init__(self, model: DepthARModel, step_ref: dict, every: int = 200,
                 nl: int = 64):
        self.model, self.step_ref, self.every, self.nl = model, step_ref, every, nl
        self.layers: list[dict] = []
        self.resid_in: float | None = None
        self.snapshots: list[dict] = []

    def attach(self) -> "ProbeHook":
        """挂全部 hook：model 级步计数 + 逐 block 激活/qkv 记录 + 每步末尾快照。"""
        self.model.register_forward_pre_hook(
            lambda m, a: self.step_ref.__setitem__(
                "n", self.step_ref["n"] + 1))
        self.model.register_forward_hook(self._forward_end)
        for i, blk in enumerate(self.model.blocks):
            blk.register_forward_pre_hook(
                lambda m, a, _i=i: self.record_in(_i, a[0]))
            blk.register_forward_hook(
                lambda m, a, o, _i=i: self.record_out(_i, o))
            blk.qkv.register_forward_hook(
                lambda m, a, o, _i=i: self.record_qkv(_i, a[0], o))
        return self

    def _forward_end(self, module, inp, out):
        if self._on():                       # 只落采样步（每 every 步一行）
            self.snapshots.append(self.snapshot())
        self.reset()
    def _on(self) -> bool:
        return self.step_ref["n"] % self.every == 0

    @torch.no_grad()
    def record_in(self, i, x):
        if not self._on():
            return
        if self.resid_in is None:
            self.resid_in = float(x.detach().float().norm(dim=-1).mean())

    @torch.no_grad()
    def record_out(self, i, out):
        if not self._on():
            return
        h = out[0] if isinstance(out, tuple) else out
        self._slot(i)["resid_out"] = float(h.detach().float()
                                           .norm(dim=-1).mean())

    @torch.no_grad()
    def record_qkv(self, i, x_in, out):
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
        sm = scores.masked_fill(~allow, neg).softmax(-1)
        logit = scores.masked_fill(~allow, neg)
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

    def reset(self):
        self.layers, self.resid_in = [], None

    def snapshot(self) -> dict:
        m = self.model
        return {"step": self.step_ref["n"],
                "resid_in": self.resid_in,
                "pos_w_norm": float(m.pos_embed.weight.detach().float().norm()),
                "embed_w_norm": float(m.embed.weight.detach().float().norm()),
                "layers": [dict(x) for x in self.layers]}


# --------------------------------------------------------------- 候选解析

def parse_candidate(spec: str) -> tuple[str, dict]:
    if spec in ("none", "baseline"):
        return "none", {}
    for tag, key in (("temp", "tau"), ("wd", "wd"), ("warmup", "warmup"),
                     ("clip", "clip")):
        if spec.startswith(tag):
            return tag, {key: float(spec[len(tag):])}
    if spec.startswith("initpos"):
        return "initpos", {"pos_std": float(spec[len("initpos"):])}
    if spec.startswith("initemb"):
        return "initemb", {"emb_std": float(spec[len("initemb"):])}
    if spec == "qk-norm":
        return "qk-norm", {"tau": 1.0, "min_t": 0}
    if spec.startswith("qk-normfrom"):
        return "qk-norm", {"tau": 1.0, "min_t": int(spec[len("qk-normfrom"):])}
    raise SystemExit(f"未知候选 {spec!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default="none")
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--probe-every", type=int, default=200)
    args = ap.parse_args()

    kind, params = parse_candidate(args.candidate)
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

    patch_info: dict = {"candidate": args.candidate, "kind": kind}
    if kind == "initpos":
        patch_info.update(apply_init_scale(model, params["pos_std"], None))
    elif kind == "initemb":
        patch_info.update(apply_init_scale(model, None, params["emb_std"]))
    elif kind == "temp":
        patch_info.update(apply_attn_scale(model, "temp", params["tau"]))
    elif kind == "qk-norm":
        patch_info.update(apply_attn_scale(model, "qk-norm", 1.0,
                                           params.get("min_t", 0)))
    elif kind == "wd":
        patch_info.update(patch_weight_decay(params["wd"]))
    elif kind == "warmup":
        patch_info.update(patch_warmup(int(params["warmup"]), args.steps))
    elif kind == "clip":
        patch_info.update(patch_clip_threshold(params["clip"]))
    elif kind != "none":
        raise SystemExit(f"kind {kind} 未接线")

    n_params = sum(p.numel() for p in model.parameters())
    step_ref = {"n": -1}
    probe = ProbeHook(model, step_ref, every=args.probe_every, nl=64).attach()

    print(f"[driver] {args.run_name}: {n_params/1e6:.3f}M params "
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

    # 末态权重范数（疤的指纹：layer-1 qkv 范数）
    w_final = {f"blocks.{i}.qkv": round(float(model.blocks[i].qkv.weight
                                              .detach().float().norm()), 3)
               for i in range(len(model.blocks))}
    result = {"run": args.run_name, "candidate": args.candidate,
              "patch": patch_info, "seq_len": args.seq_len,
              "window": args.window, "steps": args.steps,
              "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed,
              "params": n_params,
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
          f"L1.qkv={w_final.get('blocks.1.qkv')}", flush=True)


if __name__ == "__main__":
    main()
