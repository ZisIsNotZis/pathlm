#!/usr/bin/env python3
"""轴 4 · 4.1 稀疏激活专家：等 FLOPs / 等参数臂**记账** + MoE 接线 pre-flight。

    python run.py ledger    [--out results.json]     # 臂的参数量/每 token MACs/tokens-per-param
    python run.py preflight [--out preflight.json]   # 透明替换 + 活性 + 单步可训 断言（CPU）
    python run.py train     --arm <name> [--steps N] ...   # 训练+评测一臂（CPU/GPU，下一阶段入口）

口径（全部等效理论量，禁止墙钟进判据）：
- 每 token FFN MACs（乘加）: dense = 2·d·h；MoE = k·2·d·h_e + E·d（路由 logits）。
  报 MACs，并注明 FLOPs = 2×MACs。
- tokens-per-param = 训练 token 预算 / 总参数量。
- 注意力/LN/embedding/读出三臂**逐位相同**（只改 FFN），故不重复计。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# repo root = 上溯 5 级: 2610-.../ → experiments/ → 4.1-.../ → 4-moe/ → axes/ → <root>
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, *([".."] * 5))))

from pathlm.depth_ar import DepthARConfig, DepthARModel  # noqa: E402
from moe import MoEDepthARModel, TopKMoEFFN, copy_shared_from  # noqa: E402

# ---- 13M 基底 = 本仓 configs/B0.json（d=256, L=12, H=8, mlp_mult=6）----
BASE = dict(d_model=256, n_layers=12, n_heads=8, seq_len=512, mlp_mult=6)
VOCAB = 206                       # enwik8 全量 distinct 字节 205 + [mask] 1
TOKEN_BUDGET = 6000 * 32 * 512    # B0.json: steps×batch×seq = 98,304,000 tokens


def _dense_cfg(**over):
    kw = dict(BASE)
    kw.update(over)
    return DepthARConfig(dense=True, **kw)


def arm_specs() -> dict:
    """三臂 + 两个等参数 dense 对照的配置（等 FLOPs / 等参数双对照）。"""
    d = BASE["d_model"]
    h = BASE["mlp_mult"] * d                      # 1536
    specs = {
        "dense-B0":        {"kind": "dense", "mlp_mult": BASE["mlp_mult"]},
        "moe-top1-E8":     {"kind": "moe", "n_experts": 8, "top_k": 1, "expert_hidden": h},
        "moe-top2-E8":     {"kind": "moe", "n_experts": 8, "top_k": 2, "expert_hidden": h // 2},
        # 等参数 dense 对照（先把 mlp_mult 占位，下面按解析式回填）
        "dense-eqparam-top1": {"kind": "dense", "mlp_mult": None},
        "dense-eqparam-top2": {"kind": "dense", "mlp_mult": None},
    }
    return specs


def _ffn_params_per_layer_dense(d: int, h: int) -> int:
    return (2 * d * h) + h + d


def _ffn_params_per_layer_moe(d: int, e: int, k: int, h_e: int) -> int:  # k 不影响参数量
    return e * ((2 * d * h_e) + h_e + d) + e * d


def solve_eqparam_mlp_mult(d: int, e: int, h_e: int) -> tuple[int, float]:
    """解 dense 的 h' 使每层 FFN 参数 == MoE 每层 FFN 参数；返回 (mlp_mult, h' 实数)。

    解 `2·d·h' + h' + d = target` ⇒ `h' = (target − d) / (2d + 1)`。
    """
    target = _ffn_params_per_layer_moe(d, e, 1, h_e)
    h2 = (target - d) / (2 * d + 1)                # 解 2·d·h' + h' + d = target
    return int(round(h2 / d)), float(h2)


def _params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_ffn_params(model: DepthARModel) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)


def ledger() -> dict:
    d = BASE["d_model"]
    specs = arm_specs()
    # 等参数 dense 对照：按解析式回填 mlp_mult（不靠猜）
    for moe_key, de_key in (("moe-top1-E8", "dense-eqparam-top1"),
                            ("moe-top2-E8", "dense-eqparam-top2")):
        s = specs[moe_key]
        mm, _ = solve_eqparam_mlp_mult(d, s["n_experts"], s["expert_hidden"])
        specs[de_key]["mlp_mult"] = mm
    out = {}
    for name, s in specs.items():
        if s["kind"] == "dense":
            m = DepthARModel(_dense_cfg(mlp_mult=s["mlp_mult"]), VOCAB)
            expert_macs = 2 * d * (s["mlp_mult"] * d)
            router_macs = 0
            ffn_macs = expert_macs
            meta = {"kind": "dense", "mlp_mult": s["mlp_mult"],
                    "expert_hidden": s["mlp_mult"] * d}
            router_params = 0
        else:
            m = MoEDepthARModel(_dense_cfg(), VOCAB, s["n_experts"], s["top_k"],
                                s["expert_hidden"])
            expert_macs = s["top_k"] * 2 * d * s["expert_hidden"]
            router_macs = s["n_experts"] * d
            ffn_macs = expert_macs + router_macs
            meta = {"kind": "moe", "n_experts": s["n_experts"], "top_k": s["top_k"],
                    "expert_hidden": s["expert_hidden"]}
            router_params = s["n_experts"] * d * BASE["n_layers"]
        total_params = _params(m)
        ffn_params = _count_ffn_params(m)
        out[name] = {
            **meta,
            "total_params": total_params,
            "ffn_params": ffn_params,
            "router_params": router_params,
            "expert_macs_per_token_per_layer": expert_macs,
            "router_macs_per_token_per_layer": router_macs,
            "ffn_macs_per_token_per_layer": ffn_macs,
            "expert_macs_per_token": expert_macs * BASE["n_layers"],
            "ffn_macs_per_token": ffn_macs * BASE["n_layers"],
            "ffn_flops_per_token": 2 * ffn_macs * BASE["n_layers"],
            "token_budget": TOKEN_BUDGET,
            "tokens_per_param": round(TOKEN_BUDGET / total_params, 4),
            "ffn_params_x_dense": round(ffn_params / _count_ffn_params(
                DepthARModel(_dense_cfg(), VOCAB)), 4),
        }
    dense_ffn_macs = out["dense-B0"]["ffn_macs_per_token"]
    dense_expert_macs = out["dense-B0"]["expert_macs_per_token"]
    for name, v in out.items():
        v["ffn_macs_ratio_vs_dense"] = round(v["ffn_macs_per_token"] / dense_ffn_macs, 6)
        v["expert_macs_ratio_vs_dense"] = round(
            v["expert_macs_per_token"] / dense_expert_macs, 6)
    return {
        "base_config": {**BASE, "vocab_size": VOCAB, "token_budget": TOKEN_BUDGET,
                        "token_budget_source": "configs/B0.json steps=6000 × batch=32 × seq=512"},
        "arms": out,
        "notes": [
            "MACs = 乘加；FLOPs = 2×MACs。注意力/LN/embedding/读出三臂逐位相同，未重复计。",
            "等 FLOPs 对照：moe-* 相对 dense-B0 的 expert_macs_ratio_vs_dense 恰为 1.0；"
            "ffn_macs_ratio_vs_dense > 1 的部分 = 路由器 E·d logits 的『路由税』（本配置 +0.26%）。",
            "等参数对照：dense-eqparam-* 的 total_params 应与对应 moe-* 同量级（见 eqparam_check）。",
        ],
        "eqparam_check": _eqparam_check(out),
    }


def _eqparam_check(out: dict) -> list:
    res = []
    for moe_key, de_key in (("moe-top1-E8", "dense-eqparam-top1"),
                            ("moe-top2-E8", "dense-eqparam-top2")):
        a, b = out[moe_key]["total_params"], out[de_key]["total_params"]
        res.append({"moe": moe_key, "dense": de_key, "moe_params": a,
                    "dense_params": b, "rel_gap": round(abs(a - b) / a, 5)})
    return res


def preflight(out_path: str | None = None) -> dict:
    torch.manual_seed(0)
    d = BASE["d_model"]
    h = BASE["mlp_mult"] * d
    cfg = _dense_cfg()
    dense = DepthARModel(cfg, VOCAB).eval()
    # (a) 透明替换：E=1,top_k=1 + 灌入 stock mlp 权重 ⇒ 与 dense 逐位一致
    moe1 = MoEDepthARModel(cfg, VOCAB, n_experts=1, top_k=1, expert_hidden=h).eval()
    extra = copy_shared_from(dense, moe1)
    for blk_d, blk_m in zip(dense.blocks, moe1.blocks):
        blk_m.mlp.experts[0].load_state_dict(blk_d.mlp.state_dict())
    torch.manual_seed(123)
    x = torch.randint(0, VOCAB, (2, 64))
    with torch.no_grad():
        _, aux_d = dense(x)
        _, aux_m = moe1(x)
        ld = aux_d["depth_logits"][-1]
        lm = aux_m["depth_logits"][-1]
    transparency = {
        "non_mlp_extra_keys": extra,
        "max_abs_diff_logits": float((ld - lm).abs().max()),
        "bitwise_equal": bool(torch.equal(ld, lm)),
    }
    # (b) 记账一致性：实测参数量 == 解析式
    d_ffn = _count_ffn_params(dense)
    analytic_dense = BASE["n_layers"] * _ffn_params_per_layer_dense(d, h)
    moe8 = MoEDepthARModel(cfg, VOCAB, n_experts=8, top_k=2, expert_hidden=h // 2)
    m_ffn = _count_ffn_params(moe8)
    analytic_moe = BASE["n_layers"] * _ffn_params_per_layer_moe(d, 8, 2, h // 2)
    ledger_consistency = {
        "dense_ffn_params_measured": d_ffn, "dense_ffn_params_analytic": analytic_dense,
        "dense_match": d_ffn == analytic_dense,
        "moe_ffn_params_measured": m_ffn, "moe_ffn_params_analytic": analytic_moe,
        "moe_match": m_ffn == analytic_moe,
    }
    # (c) 活性量（§10）：E=8,top-2 前向 ⇒ 每层使用率分布可读
    torch.manual_seed(124)
    xb = torch.randint(0, VOCAB, (4, 128))
    _, aux8 = moe8(xb)
    live = moe8.moe_liveness()
    # (d) 单步可训：路由器梯度非零（干预不是死的）
    moe8.train()
    opt = torch.optim.AdamW(moe8.parameters(), lr=1e-3)
    loss, _ = moe8(xb)
    loss.backward()
    rgrad = sum(float(p.grad.abs().sum()) for n, p in moe8.named_parameters()
                if ".mlp.router" in n)
    opt.zero_grad()
    liveness = {
        "global": live["global"],
        "router_grad_abs_sum": round(rgrad, 6),
        "router_alive": rgrad > 0.0,
        "loss_step": round(float(loss.detach()), 6),
    }
    # (e) 路由判别力（N6）：不是恒真断言——用「零权重专家」区分路由真的选到不同专家
    moe2 = TopKMoEFFN(d, 2, 1, max(8, d // 4)).eval()
    with torch.no_grad():
        moe2.router.weight.zero_()
        moe2.router.weight[0, 0] = 5.0            # x[:,0] > 0 ⇒ 选专家 0
        moe2.router.weight[1, 0] = -5.0           # x[:,0] < 0 ⇒ 选专家 1
        for p_ in moe2.experts[1].parameters():   # 专家 1 全零 ⇒ 选中它则输出恒 0
            p_.zero_()
        xa = torch.zeros(1, 3, d); xa[0, :, 0] = 1.0   # 全选专家 0
        xb = torch.zeros(1, 3, d); xb[0, :, 0] = -1.0  # 全选专家 1
        ya, yb = moe2(xa), moe2(xb)
    routing_discrimination = {
        "expert0_out_absmax": round(float(ya.abs().max()), 8),
        "expert1_out_absmax": round(float(yb.abs().max()), 8),
        "routes_to_different_experts": bool(float(ya.abs().max()) > 0.0
                                            and float(yb.abs().max()) == 0.0),
    }
    # (f) aux loss 反传（4.2 用）：aux 必须在计算图上，否则是 detached 的死项
    moe_aux = MoEDepthARModel(cfg, VOCAB, n_experts=8, top_k=2,
                              expert_hidden=h // 2, aux_weight=0.01)
    la, _ = moe_aux(torch.randint(0, VOCAB, (2, 64)))
    al = moe_aux.moe_aux_loss()
    aux_check = {"aux_requires_grad": bool(al.requires_grad),
                 "aux_value": round(float(al.detach()), 8)}
    res = {
        "n_experts_preflight": 8, "top_k_preflight": 2,
        "transparency_E1_top1": transparency,
        "ledger_consistency": ledger_consistency,
        "liveness": liveness,
        "routing_discrimination": routing_discrimination,
        "aux_loss_check": aux_check,
        "pass": (not extra) and transparency["bitwise_equal"]
                and ledger_consistency["dense_match"] and ledger_consistency["moe_match"]
                and liveness["router_alive"] and aux_check["aux_requires_grad"]
                and routing_discrimination["routes_to_different_experts"],
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res


def train(args) -> dict:
    """训练 + 评测一臂（四个臂见 arm_specs）。CPU 冒烟与 GPU 训练同一代码路径。"""
    from pathlm.data import load_enwik8_full, batch
    root = os.path.abspath(os.path.join(_HERE, *([".."] * 5)))
    cache = args.cache or os.path.join(root, "data", "enwik8_full.npz")
    data_path = args.data or cache
    train_arr, val_arr, vocab = load_enwik8_full(data_path, cache=cache)
    vocab = max(vocab, VOCAB)
    d = args.d
    h = BASE["mlp_mult"] * d
    specs = arm_specs()
    for moe_key, de_key, hh in (("moe-top1-E8", "dense-eqparam-top1", h),
                                ("moe-top2-E8", "dense-eqparam-top2", h // 2)):
        specs[moe_key]["expert_hidden"] = hh
        specs[de_key]["mlp_mult"] = solve_eqparam_mlp_mult(d, 8, hh)[0]
    assert args.arm in specs, f"unknown arm {args.arm}; choose from {sorted(specs)}"
    s = specs[args.arm]
    torch.manual_seed(args.seed)
    cfg = _dense_cfg(seq_len=args.seq_len, n_layers=args.layers, d_model=args.d,
                     n_heads=BASE["n_heads"], mlp_mult=s.get("mlp_mult") or BASE["mlp_mult"])
    if s["kind"] == "dense":
        model = DepthARModel(cfg, vocab)
    else:
        # 单因素公平：同一 seed 下先建 dense-B0 参考（mlp_mult 与 dense-B0 同），再把非 FFN
        # 参数逐位拷进 MoE 臂 ⇒ 注意力/LN/embedding 与 dense 臂**逐位一致**（只因 FFN 专家化而不同）。
        ref = DepthARModel(_dense_cfg(seq_len=args.seq_len, n_layers=args.layers,
                                      d_model=args.d, n_heads=BASE["n_heads"],
                                      mlp_mult=BASE["mlp_mult"]), vocab)
        model = MoEDepthARModel(cfg, vocab, s["n_experts"], s["top_k"], s["expert_hidden"],
                                aux_weight=args.aux_weight)
        extra = copy_shared_from(ref, model)
        assert not extra, f"MoE 臂比 dense 多了非 mlp 参数：{extra}（单因素被破坏）"
        del ref
    device = args.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    gen = torch.Generator().manual_seed(args.seed + 7)
    log = []
    model.train()
    t0 = time.time()
    t20 = None
    for step in range(1, args.steps + 1):
        x, y = batch(train_arr, args.batch_size, cfg.seq_len, gen)
        x, y = x.to(device), y.to(device)
        loss, aux = model(x, targets=y)
        total = loss + (model.moe_aux_loss().to(device) if isinstance(model, MoEDepthARModel) else 0.0)
        opt.zero_grad(); total.backward(); opt.step()
        if step == 20:
            t20 = time.time()
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": round(float(loss.detach()), 6)}
            if isinstance(model, MoEDepthARModel):
                live = model.moe_liveness()["global"]
                rec.update({"usage_entropy_norm": live.get("usage_entropy_norm"),
                            "usage_max_frac": live.get("usage_max_frac"),
                            "usage_entropy_norm_min_layer": min(
                                (r.get("usage_entropy_norm") if
                                 r.get("usage_entropy_norm") is not None else 1.0)
                                for r in model.moe_liveness()["per_layer"]),
                            "usage": live.get("usage")})
            log.append(rec)
    ev = eval_bpc(model, val_arr, cfg, args)
    res = {"arm": args.arm, "config": {**BASE, "d_model": args.d, "n_layers": args.layers,
           "seq_len": args.seq_len, "steps": args.steps, "batch_size": args.batch_size,
           "lr": args.lr, "seed": args.seed, "aux_weight": args.aux_weight},
           "total_params": _params(model), "eval": ev, "log": log,
           # 墙钟只作**过程指标**（SCHEMA §3：不得进 cost_model/判据）
           "timing_process_metric": {
               "first_20_steps_s": round((t20 - t0), 3) if t20 else None,
               "total_s": round(time.time() - t0, 3)}}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    return res


@torch.no_grad()
def eval_bpc(model, val_arr, cfg, args) -> dict:
    from pathlm.data import batch
    model.eval()
    gen = torch.Generator().manual_seed(4)
    nats = 0.0; n = 0
    for _ in range(args.eval_batches):
        x, _ = batch(val_arr, args.eval_batch_size, cfg.seq_len, gen)
        x = x.to(args.device)
        _, aux = model(x)
        lg = aux["depth_logits"][-1][:, :-1]
        tgt = x[:, 1:]
        import torch.nn.functional as F
        nats += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(),
                                      tgt.reshape(-1), reduction="sum"))
        n += tgt.numel()
    out = {"bpc": round(nats / n / math.log(2), 4), "n_eval_positions": n}
    if isinstance(model, MoEDepthARModel):
        lv = model.moe_liveness()
        out["moe_liveness"] = lv["global"]
        # N5：逐层活性（堵住「池化把整层坍缩抹平」的漏洞）。
        # 注意：活性量取自**最后一个 eval batch**（bpc 覆盖全部 batch），口径在 analysis 里写明。
        out["moe_liveness_per_layer"] = [
            {"usage_entropy_norm": r.get("usage_entropy_norm"),
             "usage_max_frac": r.get("usage_max_frac")} for r in lv["per_layer"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("ledger"); p1.add_argument("--out", default=None)
    p2 = sub.add_parser("preflight"); p2.add_argument("--out", default=None)
    p3 = sub.add_parser("train")
    p3.add_argument("--arm", required=True)
    p3.add_argument("--steps", type=int, default=6000)
    p3.add_argument("--batch-size", type=int, default=32)
    p3.add_argument("--seq-len", type=int, default=512)
    p3.add_argument("--d", type=int, default=BASE["d_model"])
    p3.add_argument("--layers", type=int, default=BASE["n_layers"])
    p3.add_argument("--lr", type=float, default=1e-3)
    p3.add_argument("--seed", type=int, default=0)
    p3.add_argument("--aux-weight", type=float, default=0.0)
    p3.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p3.add_argument("--data", default=None); p3.add_argument("--cache", default=None)
    p3.add_argument("--eval-batches", type=int, default=20)
    p3.add_argument("--eval-batch-size", type=int, default=16)
    p3.add_argument("--log-every", type=int, default=200)
    p3.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "ledger":
        res = ledger()
    elif args.cmd == "preflight":
        res = preflight(args.out)
    else:
        res = train(args)
    if args.out and args.cmd != "preflight":
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
