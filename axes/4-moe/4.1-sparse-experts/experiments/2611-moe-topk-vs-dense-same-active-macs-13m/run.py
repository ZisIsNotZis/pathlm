#!/usr/bin/env python3
"""2611 · `4.1` 的判决实验：同每 token 激活 MACs 下，top-1 稀疏专家 vs stock dense（13M）。

契约见 `README.md` §4（判据线先绑后用），常数在 `config.json`；两文件与 `run.py` / `train_arms.sh`
一起被 `tools/prereg_bind.py --contract README.md config.json run.py train_arms.sh` 绑住。

用法：
    python3 run.py --stage ledger                 # 臂记账（参数量 / 每 token MACs / token-per-param）
    python3 run.py --stage preflight              # E=1 逐位恒等 + 记账自洽 + 路由活性（CPU）
    python3 run.py --stage train --arm <name>     # 训练 + 单臂产物（GPU）
    python3 run.py --stage eval                   # 全臂配对评测 + 判词（GPU，无训练）

单因素：只改 FFN 的专家化（`E` / `k` / `h_e`）。注意力 / LN / embedding / 读出 / 层数 / 残差接线 /
优化器 / 数据切分 / 步数 / seed 全不动；MoE 臂的非 FFN 参数从同 seed 的 stock dense 臂逐位拷贝。
`aux_weight = 0`（均衡属 4.2）、专家均匀同宽（4.3）、top-k 权重归一（见 `config.json::gating`）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, *([".."] * 5)))
sys.path.insert(0, ROOT)

import train_depth_ar as tdt                                       # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel             # noqa: E402
from pathlm.data import load_enwik8_full, batch                     # noqa: E402
from tools.prereg_bind import require as require_prereg             # noqa: E402
from tools.run_provenance import write_results                      # noqa: E402

CFG = json.loads(Path(HERE, "config.json").read_text(encoding="utf-8"))
RECIPE = CFG["recipe"]
BAND = float(CFG["band_bpc"])
EVAL = CFG["eval"]
CHECKS = CFG["checks"]
ARMS = {a["name"]: a for a in CFG["arms"]}
LN2 = math.log(2.0)
D = RECIPE["d_model"]
NLAYER = RECIPE["n_layers"]
BASE_MLP = RECIPE["base_mlp_mult"]
SEQ = RECIPE["seq_len"]
DATA = os.path.join(ROOT, RECIPE["data"])
CACHE = os.path.join(ROOT, "data", "enwik8_full.npz")
DEVICE = "cuda"


# ---------------------------------------------------------------- MoE 模块（本实验自持，不改 pathlm/**）
class TopKMoEFFN(nn.Module):
    """E 个同宽专家 + 一个 d×E 路由器；每 token 走 top-k（权重归一到和为 1）。

    参数量：E·(2·d·h_e + h_e + d)（专家）+ E·d（路由器，无 bias）。
    每 token FFN MACs：k·(2·d·h_e)（专家）+ E·d（路由 logits）。
    """

    def __init__(self, d: int, n_experts: int, top_k: int, expert_hidden: int,
                 dropout: float = 0.0, renorm: bool = True):
        super().__init__()
        if n_experts < 1 or not 1 <= top_k <= n_experts or expert_hidden < 1:
            raise ValueError(f"非法 MoE 形状：E={n_experts} k={top_k} h_e={expert_hidden}")
        self.d, self.n_experts, self.top_k = d, n_experts, top_k
        self.expert_hidden = expert_hidden
        self.renorm = bool(renorm)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, expert_hidden), nn.GELU(),
                          nn.Linear(expert_hidden, d), nn.Dropout(dropout))
            for _ in range(n_experts)])
        self.router = nn.Linear(d, n_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.last_usage: torch.Tensor | None = None

    def route(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """路由：[N, d] → (top-k 权重 [N, k]（归一到和为 1）、top-k 专家号 [N, k]）。"""
        probs = self.router(flat).softmax(-1)                 # [N, E]
        topv, topi = probs.topk(self.top_k, dim=-1)           # [N, k]
        if self.renorm:
            topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        return topv, topi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """按专家分组的前向（绑定后版本）：整层一次排序 + 一次 gather + 一次 index_add。

        每专家的 GEMM 仍是**同一批行、同一形状**（同 `forward_naive`），只是把「每专家一次布尔索引
        + 一次掩码拷贝」换成「整层一次 gather + 连续切片 + 一次 scatter」——后者把每层 3E 次
        host 同步/掩码核降到 1 次，逐位结果不变（`test_moe_forward_equivalence.py` 钉住
        max_abs_diff == 0）。k>1 时同一 token 的 k 份贡献由 `index_add` 累加，累加顺序不保证
        位级一致；本实验的臂全是 k=1（`config.json::arms`），那里 `index_add` 等于逐行写入。
        """
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        topv, topi = self.route(flat)
        tok, val = topi.reshape(-1), topv.reshape(-1)          # [N·k]
        order = torch.argsort(tok, stable=True)                # 按专家分组，组内保持原序
        # (token, slot) 行 → 它属于哪个 token（k=1 时两者相同）
        token_of = order.div(self.top_k, rounding_mode="floor")
        counts = torch.bincount(tok, minlength=self.n_experts).tolist()
        flat_s = flat.index_select(0, token_of)
        val_s = val.gather(0, order).unsqueeze(-1)
        ys, off = [], 0
        for e, expert in enumerate(self.experts):
            n = counts[e]
            if n:
                ys.append(expert(flat_s.narrow(0, off, n)))
                off += n
        y = torch.cat(ys, 0) if len(ys) > 1 else ys[0]
        out = torch.zeros_like(flat).index_add(0, token_of, val_s * y)
        self.last_usage = torch.bincount(tok, minlength=self.n_experts).detach()
        return out.reshape(B, T, d)

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """**参照实现（重绑前那一版）**：逐专家布尔索引 + 掩码拷贝，每层 3E 次 host 同步。

        只用于等价性测试（`test_moe_forward_equivalence.py`），不参与训练：160M 级臂上它每步
        ≈0.40 s（20 min 单臂 timeout 撑不住），换成 `forward` 后同臂 ≈0.165 s。
        """
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        topv, topi = self.route(flat)
        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            sel = topi == e
            row = sel.any(-1)
            if not bool(row.any()):
                continue
            w = (topv * sel).sum(-1)                          # [N]
            y = expert(flat[row])                             # [n_sel, d]
            out[row] = out[row] + w[row].unsqueeze(-1) * y
        self.last_usage = torch.bincount(topi.reshape(-1),
                                         minlength=self.n_experts).detach()
        return out.reshape(B, T, d)


class MoEDepthARModel(DepthARModel):
    """stock `DepthARModel(dense=True)`：只在构造后把每层 `Block.mlp` 换成 `TopKMoEFFN`。"""

    def __init__(self, cfg: DepthARConfig, vocab_size: int, n_experts: int, top_k: int,
                 expert_hidden: int, renorm: bool = True):
        if not cfg.dense:
            raise ValueError("MoE 替换只定义在 dense=True（stock 无门主干）上")
        super().__init__(cfg, vocab_size)
        self.n_experts, self.top_k = n_experts, top_k
        self.expert_hidden = expert_hidden
        for blk in self.blocks:
            if not hasattr(blk, "mlp"):
                raise TypeError(f"{type(blk).__name__} 没有 .mlp，无法作 MoE 替换")
            blk.mlp = TopKMoEFFN(cfg.d_model, n_experts, top_k, expert_hidden,
                                 dropout=cfg.dropout, renorm=renorm)

    def moe_modules(self) -> list[TopKMoEFFN]:
        return [blk.mlp for blk in self.blocks]


def copy_shared_from(dense: DepthARModel, moe: MoEDepthARModel) -> list[str]:
    """把 dense 臂里**非 FFN** 的参数/缓冲逐位拷进 MoE 臂（单因素：只差 FFN）。

    返回 MoE 侧多出来、dense 侧没有的**非 mlp** 键（应为空）。
    """
    dsd = {k: v for k, v in dense.state_dict().items() if ".mlp." not in k}
    msd = moe.state_dict()
    moe.load_state_dict({k: v for k, v in dsd.items() if k in msd}, strict=False)
    return sorted(k for k in msd if ".mlp." not in k and k not in dsd)


# ---------------------------------------------------------------- 臂的构造（训练 / 评测共用）
def model_cfg(mlp_mult: int) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=NLAYER, n_heads=RECIPE["n_heads"], seq_len=SEQ,
                         mlp_mult=mlp_mult, dropout=RECIPE["dropout"],
                         norm_cap=RECIPE["norm_cap"], dense=True, topology="sequential",
                         train_window=0, train_anchors=0)


def build_model(arm: dict, vocab: int) -> tuple[nn.Module, dict]:
    """按臂造模型。MoE 臂的 trunk 从同 seed 的 stock dense 臂逐位拷贝（唯一差异 = FFN）。"""
    torch.manual_seed(int(arm["seed"]))
    if arm["kind"] == "dense":
        model = DepthARModel(model_cfg(int(arm["mlp_mult"])), vocab)
        return model, {"kind": "dense", "mlp_mult": int(arm["mlp_mult"])}
    ref = DepthARModel(model_cfg(BASE_MLP), vocab)          # 同 seed 的 dense-B0 参考
    model = MoEDepthARModel(model_cfg(BASE_MLP), vocab, int(arm["n_experts"]),
                            int(arm["top_k"]), int(arm["expert_hidden"]),
                            renorm=bool(CFG["gating"]["renorm"]))
    extra = copy_shared_from(ref, model)
    if extra:
        raise SystemExit(f"{arm['name']}: MoE 臂比 dense 多了非 mlp 参数 {extra}（单因素被破坏）")
    return model, {"kind": "moe", "n_experts": int(arm["n_experts"]),
                   "top_k": int(arm["top_k"]), "expert_hidden": int(arm["expert_hidden"]),
                   "renorm": bool(CFG["gating"]["renorm"])}


def count_params(model: nn.Module) -> tuple[int, int, int]:
    """(总参数, FFN 参数, 路由器参数)。"""
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)
    router = sum(p.numel() for n, p in model.named_parameters() if ".mlp.router." in n)
    return total, ffn, router


def arm_macs(arm: dict) -> dict:
    """每 token、每层的解析 MACs（乘加；FLOPs = 2×MACs）。"""
    if arm["kind"] == "dense":
        h = int(arm["mlp_mult"]) * D
        return {"expert_macs_per_layer": 2 * D * h, "router_macs_per_layer": 0}
    k, e, h_e = int(arm["top_k"]), int(arm["n_experts"]), int(arm["expert_hidden"])
    return {"expert_macs_per_layer": k * 2 * D * h_e, "router_macs_per_layer": e * D}


# ---------------------------------------------------------------- ledger
def ledger() -> dict:
    specs, seen = {}, {}
    for name, arm in ARMS.items():
        key = json.dumps({k: v for k, v in arm.items()
                          if k not in ("name", "seed", "role", "ckpt")}, sort_keys=True)
        if key not in seen:
            model, _ = build_model(arm, 206)
            seen[key] = count_params(model)
            del model
        total, ffn, router = seen[key]
        m = arm_macs(arm)
        specs[name] = {
            "kind": arm["kind"],
            "n_experts": arm.get("n_experts", 1),
            "top_k": arm.get("top_k", 1),
            "expert_hidden": (arm.get("expert_hidden") if arm["kind"] == "moe"
                              else int(arm["mlp_mult"]) * D),
            "total_params": total, "ffn_params": ffn, "router_params": router,
            "expert_macs_per_token": m["expert_macs_per_layer"] * NLAYER,
            "router_macs_per_token": m["router_macs_per_layer"] * NLAYER,
            "ffn_macs_per_token": (m["expert_macs_per_layer"]
                                   + m["router_macs_per_layer"]) * NLAYER,
            "tokens_per_arm": RECIPE["tokens_per_arm"],
            "tokens_per_param": round(RECIPE["tokens_per_arm"] / total, 4),
        }
    base_expert = specs["dense-B0-s0"]["expert_macs_per_token"]
    base_ffn = specs["dense-B0-s0"]["ffn_macs_per_token"]
    for v in specs.values():
        v["expert_macs_ratio_vs_dense"] = round(v["expert_macs_per_token"] / base_expert, 6)
        v["ffn_macs_ratio_vs_dense"] = round(v["ffn_macs_per_token"] / base_ffn, 6)
        v["params_x_dense"] = round(v["total_params"] / specs["dense-B0-s0"]["total_params"], 3)
    moe_expert = specs["moe-top1-E4-s0"]["expert_macs_per_token"]
    eq_gap = abs(specs["dense-eqparam-E4-s0"]["total_params"]
                 - specs["moe-top1-E4-s0"]["total_params"]) / specs["moe-top1-E4-s0"]["total_params"]
    macs_check = {a: specs[a]["expert_macs_ratio_vs_dense"]
                  for a in ("moe-top1-E4-s0", "moe-top1-E16-s0")}
    return {
        "base_config": {**RECIPE, "vocab_size": 206},
        "arms": specs,
        "checks": {
            "expert_macs_ratio_vs_dense": macs_check,
            "expert_macs_equal_all_main_arms": all(
                v == float(CHECKS["expert_macs_ratio"]) for v in macs_check.values()),
            "eqparam_e4_rel_gap": round(eq_gap, 5),
            "eqparam_e4_within_0p2pct": eq_gap < float(CHECKS["eqparam_rel_gap_max"]),
            "router_tax_e4_pct": round(100 * (specs["moe-top1-E4-s0"]["ffn_macs_per_token"]
                                             / base_ffn - 1), 3),
            "router_tax_e16_pct": round(100 * (specs["moe-top1-E16-s0"]["ffn_macs_per_token"]
                                              / base_ffn - 1), 3),
        },
        "notes": [
            "MACs = 乘加，FLOPs = 2×MACs；注意力 / LN / embedding / 读出各臂逐位相同，未重复计。",
            "等激活算力：moe-top1-E{4,16} 的 expert_macs_ratio_vs_dense 必须恰为 1.0；"
            "ffn_macs_ratio_vs_dense 多出的部分 = 路由器 E·d 的『路由税』，单列申报。",
            "等参数对照 dense-eqparam-E4（mlp_mult=24）与 moe-top1-E4 参数相对差见 checks；"
            "它的每 token 激活算力是 4×，只用于归因（SCHEMA §15），不是等算力对照。",
            f"moe 的每 token 专家 MACs 绝对值 = {moe_expert}（与 dense-B0 相同）。",
        ],
    }


# ---------------------------------------------------------------- preflight（CPU，判据⑤/⑥的前提）
def preflight() -> dict:
    torch.manual_seed(0)
    cfg = model_cfg(BASE_MLP)
    dense = DepthARModel(cfg, 206).eval()
    h = BASE_MLP * D
    moe1 = MoEDepthARModel(cfg, 206, 1, 1, h, renorm=True).eval()
    extra = copy_shared_from(dense, moe1)
    for blk_d, blk_m in zip(dense.blocks, moe1.blocks):
        blk_m.mlp.experts[0].load_state_dict(blk_d.mlp.state_dict())
    torch.manual_seed(123)
    x = torch.randint(0, 206, (2, 64))
    with torch.no_grad():
        lg_d = dense(x)[1]["depth_logits"][-1]
        lg_m = moe1(x)[1]["depth_logits"][-1]
    identity = {
        "non_mlp_extra_keys": extra,
        "max_abs_diff_logits": float((lg_d - lg_m).abs().max()),
        "bitwise_equal": bool(torch.equal(lg_d, lg_m)),
    }

    torch.manual_seed(7)
    moe4 = MoEDepthARModel(cfg, 206, 4, 1, h, renorm=True)
    total_d, ffn_d, _ = count_params(dense)
    total_m, ffn_m, router_m = count_params(moe4)
    analytic_d = NLAYER * (2 * D * h + h + D)
    analytic_m = NLAYER * (4 * (2 * D * h + h + D) + 4 * D)
    ledger_check = {
        "dense_ffn_params_measured": ffn_d, "dense_ffn_params_analytic": analytic_d,
        "dense_match": ffn_d == analytic_d,
        "moe4_ffn_params_measured": ffn_m, "moe4_ffn_params_analytic": analytic_m,
        "moe4_match": ffn_m == analytic_m,
        "moe4_total_params": total_m, "dense_total_params": total_d,
        "moe4_router_params": router_m,
    }

    moe4.train()
    xb = torch.randint(0, 206, (4, 128))
    loss, _ = moe4(xb)
    loss.backward()
    rgrad = sum(float(p.grad.abs().sum()) for n, p in moe4.named_parameters()
                if ".mlp.router." in n)
    final_usage = [int(v) for v in moe4.moe_modules()[0].last_usage.tolist()]
    liveness = {
        "router_grad_abs_sum": round(rgrad, 6), "router_alive": rgrad > 0.0,
        "layer0_usage": final_usage,
        "layer0_usage_entropy_norm": round(
            float(-(torch.tensor(final_usage, dtype=torch.float)
                    / sum(final_usage)).mul(
                (torch.tensor(final_usage, dtype=torch.float) / sum(final_usage)
                 + 1e-12).log()).sum()) / math.log(4), 6),
    }

    # 路由判别力：两个零权重专家 ⇒ 选中它的 token 输出恒 0（证明路由真的选到不同专家）
    probe = TopKMoEFFN(D, 2, 1, max(8, D // 4)).eval()
    with torch.no_grad():
        probe.router.weight.zero_()
        probe.router.weight[0, 0] = 5.0
        probe.router.weight[1, 0] = -5.0
        for p_ in probe.experts[1].parameters():
            p_.zero_()
        xa = torch.zeros(1, 3, D); xa[0, :, 0] = 1.0
        xb2 = torch.zeros(1, 3, D); xb2[0, :, 0] = -1.0
        ya, yb = probe(xa), probe(xb2)
    discrimination = {
        "expert0_out_absmax": round(float(ya.abs().max()), 8),
        "expert1_out_absmax": round(float(yb.abs().max()), 8),
        "routes_to_different_experts": bool(float(ya.abs().max()) > 0.0
                                            and float(yb.abs().max()) == 0.0),
    }
    res = {
        "identity_E1_top1": identity,
        "ledger_consistency": ledger_check,
        "liveness": liveness,
        "routing_discrimination": discrimination,
        "pass": (not extra) and identity["bitwise_equal"]
                and ledger_check["dense_match"] and ledger_check["moe4_match"]
                and liveness["router_alive"] and discrimination["routes_to_different_experts"],
    }
    write_results(os.path.join(HERE, "preflight.json"), res, source=__file__)
    return res


# ---------------------------------------------------------------- 训练（一臂）
def data_arrays():
    return load_enwik8_full(os.path.join(ROOT, ".tmp", "enwik8"), CACHE)


@torch.no_grad()
def usage_stats(model: nn.Module) -> dict | None:
    """逐层 + 全局专家使用率（归一化熵 / 最大占比）。非 MoE 臂返回 None。"""
    if not isinstance(model, MoEDepthARModel):
        return None
    per_layer = []
    for m in model.moe_modules():
        if m.last_usage is None:
            continue
        u = m.last_usage.double()
        p = u / u.sum().clamp_min(1.0)
        ent = float(-(p * (p + 1e-12).log()).sum())
        per_layer.append({"usage": [int(v) for v in u.tolist()],
                          "entropy_norm": round(ent / math.log(m.n_experts), 6),
                          "max_frac": round(float(p.max()), 6)})
    return {"per_layer": per_layer,
            "entropy_norm_min_layer": min(r["entropy_norm"] for r in per_layer),
            "max_frac_worst_layer": max(r["max_frac"] for r in per_layer),
            "n_experts": model.n_experts, "top_k": model.top_k}


def accumulate_usage(acc: list[torch.Tensor], model: nn.Module) -> None:
    for i, m in enumerate(model.moe_modules()):
        if m.last_usage is not None:
            acc[i] += m.last_usage.double()


def train_arm(name: str, steps: int | None = None, out_root: str = "runs") -> dict:
    arm = ARMS[name]
    steps = int(steps) if steps else int(RECIPE["steps"])
    train_arr, _, vocab = data_arrays()
    model, info = build_model(arm, vocab)
    total, ffn, router = count_params(model)
    run_dir = os.path.join(HERE, out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    model = model.cuda()

    # 训练前的活性快照（同一批固定 probe，便于与终评比方向）
    torch.manual_seed(999)
    probe = torch.randint(0, vocab, (4, 128), device=DEVICE)
    model.eval()
    with torch.no_grad():
        model(probe)
    start_usage = usage_stats(model)

    torch.cuda.reset_peak_memory_stats()
    tcfg = {"steps": steps, "batch_size": RECIPE["batch_size"], "lr": RECIPE["lr"],
            "seed": int(arm["seed"])}
    wall, _, _, stats = tdt.train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"), exit_mode="uniform")
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    model.eval()
    with torch.no_grad():
        model(probe)
    end_usage = usage_stats(model)

    gnorm_max = stats.get("gnorm_max")
    valid = (gnorm_max is not None and gnorm_max < float(CHECKS["stability_gnorm_max"])
             and int(stats.get("grad_skips", 0)) == 0)
    out = {
        "run": name, "arm": name, "seed": int(arm["seed"]), "steps": steps,
        "tokens": steps * RECIPE["batch_size"] * SEQ,
        "arm_spec": {k: v for k, v in arm.items() if k != "role"},
        "recipe": {k: RECIPE[k] for k in ("d_model", "n_layers", "n_heads", "seq_len",
                                          "batch_size", "lr", "dropout", "norm_cap")},
        "tokens_per_param": round(steps * RECIPE["batch_size"] * SEQ / total, 4),
        "n_params": {"total": total, "ffn": ffn, "router": router},
        "arch": info,
        "wall_minutes": round(wall / 60, 2),
        "peak_gpu_mem_mb": peak,
        "train_stats": {k: stats.get(k) for k in ("gnorm_max", "gnorm_mean", "grad_skips")},
        "stability_gate": {"valid": bool(valid),
                           "rule": "gnorm_max < 100 且 grad_skips == 0"},
        "usage_start": start_usage, "usage_end": end_usage,
        "wall_note": "墙钟与峰值显存只作申报，不进判据（harness/methodology.md §3）",
    }
    write_results(os.path.join(run_dir, "results.json"), out, source=__file__)
    print(f"{name}: {total/1e6:.2f}M params  wall {out['wall_minutes']} min  "
          f"peak {peak} MiB  gnorm_max {gnorm_max}  valid {valid}", flush=True)
    return out


# ---------------------------------------------------------------- 评测（全臂，配对）
@torch.no_grad()
def eval_arm(arm: dict, eval_arr, vocab: int) -> dict:
    model, _ = build_model(arm, vocab)
    model.load_state_dict(torch.load(os.path.join(HERE, arm["ckpt"]), map_location=DEVICE))
    model.to(DEVICE).eval()
    gen = torch.Generator().manual_seed(int(EVAL["window_seed"]))
    per_batch, acc = [], None
    for _ in range(int(EVAL["n_batches"])):
        x, _ = batch(eval_arr, int(EVAL["batch_size"]), SEQ, gen)
        x = x.to(DEVICE)
        with torch.autocast("cuda", enabled=False):
            lg = model(x)[1]["depth_logits"][-1][:, :-1].float()
        tgt = x[:, 1:]
        nats = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                              reduction="sum").item()
        per_batch.append(round(nats / tgt.numel() / LN2, 5))
        if isinstance(model, MoEDepthARModel):
            if acc is None:
                acc = [torch.zeros(m.n_experts, dtype=torch.float64) for m in model.moe_modules()]
            accumulate_usage(acc, model)
    out = {"bpc_by_batch": per_batch,
           "bpc": round(statistics.mean(per_batch), 5),
           "n_eval_batches": len(per_batch),
           "n_eval_positions": len(per_batch) * int(EVAL["batch_size"]) * (SEQ - 1)}
    if acc is not None:
        per_layer = []
        for u in acc:
            p = u / u.sum().clamp_min(1.0)
            ent = float(-(p * (p + 1e-12).log()).sum())
            per_layer.append({"usage": [int(v) for v in u.tolist()],
                              "entropy_norm": round(ent / math.log(len(u)), 6),
                              "max_frac": round(float(p.max()), 6)})
        glob = sum(acc)
        pg = glob / glob.sum().clamp_min(1.0)
        ent = float(-(pg * (pg + 1e-12).log()).sum())
        out["routing"] = {
            "per_layer": per_layer,
            "global": {"usage": [int(v) for v in glob.tolist()],
                       "entropy_norm": round(ent / math.log(len(glob)), 6),
                       "max_frac": round(float(pg.max()), 6),
                       "n_experts": len(glob)},
            "entropy_norm_min_layer": min(r["entropy_norm"] for r in per_layer),
            "max_frac_worst_layer": max(r["max_frac"] for r in per_layer),
            "collapse_flagged": bool(
                min(r["entropy_norm"] for r in per_layer) < float(CHECKS["collapse_entropy_norm"])
                or max(r["max_frac"] for r in per_layer) > float(CHECKS["collapse_max_frac"])),
            "windows_note": "使用率累计全部 20 个评测批（不是只看最后一批）",
        }
    return out


def paired_stats(comparator: list[float], arm: list[float]) -> dict:
    d = [c - a for c, a in zip(comparator, arm)]
    n = len(d)
    mean, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / n ** 0.5
    return {"n_batches": n, "delta_bpc": round(mean, 5), "sd_of_paired_delta": round(sd, 5),
            "se_of_mean": round(se, 5), "two_se": round(1.96 * se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "n_req_for_band": int(math.ceil((1.96 * sd / BAND) ** 2)) if sd > 0 else None,
            "delta_consistency_abs": None}      # 由调用方按未舍入均值填


def band_class(delta: float) -> str:
    if delta >= BAND:
        return "beyond_positive"
    if delta <= -BAND:
        return "beyond_negative"
    return "in_band"


def group_verdict(classes: list[str], kind: str) -> str:
    if len(classes) != 2:
        return f"只取得 {len(classes)} 个 seed ⇒ 不落判词"
    if all(c == classes[0] for c in classes):
        if classes[0] == "beyond_positive":
            return {"main": "同激活算力下 MoE 更好（越带、两 seed 同向，n=2）",
                    "expert_count": "专家更多更好（越带、两 seed 同向，n=2）",
                    "attr": "MoE 优于等参数 dense ⇒ 稀疏/路由本身有贡献（越带，n=2）"}[kind]
        if classes[0] == "beyond_negative":
            return {"main": "同激活算力下 MoE 更差（越带、两 seed 同向，n=2）",
                    "expert_count": "专家更多更差（越带、两 seed 同向，n=2）",
                    "attr": "MoE 劣于等参数 dense ⇒ 稀疏/路由本身更差（越带，n=2）"}[kind]
        return {"main": "两 seed 都落带内 ⇒ 同激活算力下不可排序（买不到质量）",
                "expert_count": "两 seed 都落带内 ⇒ 专家数买不到质量",
                "attr": "两 seed 都落带内 ⇒ 质量差异可由参数容量解释（路由/稀疏无独立贡献，SCHEMA §15）"}[kind]
    return "两 seed 不同向或未同时越带 ⇒ 本 n 下不可排序（不解释成噪声、不取平均掩盖）"


def evaluate() -> dict:
    _, eval_arr, vocab = data_arrays()
    per_arm, not_run = {}, []
    for name, arm in ARMS.items():
        ckpt = os.path.join(HERE, arm["ckpt"])
        if not os.path.exists(ckpt):
            not_run.append(name)
            continue
        per_arm[name] = eval_arm(arm, eval_arr, vocab)

    def meta(name):
        p = os.path.join(HERE, "runs", name, "results.json")
        return json.loads(Path(p).read_text(encoding="utf-8")) if os.path.exists(p) else {}

    invalid = [n for n in per_arm
               if not (meta(n).get("stability_gate", {}).get("valid", False))]
    pairs = {}
    for pname, spec in CFG["pairs"].items():
        cmp_n, arm_n = spec["comparator"], spec["arm"]
        if cmp_n not in per_arm or arm_n not in per_arm:
            pairs[pname] = {"skipped": f"臂缺读数：{cmp_n if cmp_n not in per_arm else arm_n}"}
            continue
        if cmp_n in invalid or arm_n in invalid:
            pairs[pname] = {"skipped": f"臂破稳定性门：{cmp_n if cmp_n in invalid else arm_n}"}
            continue
        cb, ab = per_arm[cmp_n]["bpc_by_batch"], per_arm[arm_n]["bpc_by_batch"]
        st = paired_stats(cb, ab)
        raw = statistics.mean(cb) - statistics.mean(ab)
        st["delta_consistency_abs"] = round(abs(st["delta_bpc"] - raw), 8)
        st["comparator"] = cmp_n
        st["arm"] = arm_n
        st["bpc_comparator"] = round(statistics.mean(cb), 5)
        st["bpc_arm"] = round(statistics.mean(ab), 5)
        st["band_class"] = band_class(st["delta_bpc"])
        pairs[pname] = st

    def group(prefix, kind):
        classes = [pairs[f"{prefix}-s{s}"]["band_class"]
                   for s in (0, 1) if f"{prefix}-s{s}" in pairs
                   and "band_class" in pairs[f"{prefix}-s{s}"]]
        return {"kind": kind, "band_classes": classes, "verdict": group_verdict(classes, kind),
                "pairs": {f"s{s}": pairs[f"{prefix}-s{s}"] for s in (0, 1)
                          if f"{prefix}-s{s}" in pairs}}

    crit = {
        "c1_main_equal_macs": {
            "note": "Δ = bpc(dense-B0) − bpc(moe-top1-E*)（对照 − 实验，正 = MoE 更好）；0.014 配对带",
            "E4": group("main-E4", "main"), "E16": group("main-E16", "main")},
        "c2_expert_count": {
            "note": "Δ = bpc(moe-top1-E4) − bpc(moe-top1-E16)（正 = 专家更多更好）；同一条 0.014 带",
            **{k: v for k, v in group("expert-count", "expert_count").items()}},
        "c3_attribution_eqparam": {
            "note": "Δ = bpc(dense-eqparam-E4) − bpc(moe-top1-E4)（正 = MoE 优于等参数 dense）；"
                    "SCHEMA §15 的归因对照（等参数、4× 激活算力）",
            **{k: v for k, v in group("attr-E4", "attr").items()}},
    }

    arm_table = {}
    dense_bpc = per_arm.get("dense-B0-s0", {}).get("bpc")
    for name, r in per_arm.items():
        md = meta(name)
        arm_table[name] = {
            "bpc": r["bpc"], "n_params_total": md.get("n_params", {}).get("total"),
            "tokens_per_param": md.get("tokens_per_param"),
            "wall_minutes": md.get("wall_minutes"), "peak_gpu_mem_mb": md.get("peak_gpu_mem_mb"),
            "gnorm_max": md.get("train_stats", {}).get("gnorm_max"),
            "grad_skips": md.get("train_stats", {}).get("grad_skips"),
            "stability_valid": md.get("stability_gate", {}).get("valid"),
            "routing": r.get("routing"),
        }
        if name.startswith(("moe-", "dense-eqparam")) and dense_bpc is not None \
                and name.endswith("-s0"):
            arm_table[name]["delta_vs_dense_bpc"] = round(dense_bpc - r["bpc"], 5)

    routing_summary = {n: {"entropy_norm_min_layer": r["routing"]["entropy_norm_min_layer"],
                           "max_frac_worst_layer": r["routing"]["max_frac_worst_layer"],
                           "global": r["routing"]["global"],
                           "collapse_flagged": r["routing"]["collapse_flagged"]}
                       for n, r in per_arm.items() if r.get("routing")}

    stability = {n: {"valid": meta(n).get("stability_gate", {}).get("valid"),
                     "gnorm_max": meta(n).get("train_stats", {}).get("gnorm_max"),
                     "grad_skips": meta(n).get("train_stats", {}).get("grad_skips")}
                 for n in per_arm}

    led = ledger()
    out = {
        "experiment": CFG["experiment"], "cell": CFG["cell"], "question": CFG["question"],
        "band_bpc": BAND,
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "sd_of_paired_delta": "comparator_minus_arm",
            "frac_batches_delta_gt_0": "comparator_minus_arm",
            "delta_consistency_abs": "comparator_minus_arm",
            "delta_vs_dense_bpc": "comparator_minus_arm",
        },
        "delta_direction_note": "全部配对 Δ = bpc(对照) − bpc(实验) ⇒ 正 = 实验臂更好",
        "recipe": RECIPE, "gating": CFG["gating"],
        "ledger": led,
        "per_batch_bpc": {n: r["bpc_by_batch"] for n, r in per_arm.items()},
        "mean_bpc": {n: r["bpc"] for n, r in per_arm.items()},
        "arm_table": arm_table,
        "arms_not_run": not_run,
        "arms_invalid_stability": invalid,
        "pairs": pairs,
        "criteria": crit,
        "c4_cost": {"expert_macs_equal_all_main_arms":
                    led["checks"]["expert_macs_equal_all_main_arms"],
                    "expert_macs_ratio_vs_dense": led["checks"]["expert_macs_ratio_vs_dense"],
                    "router_tax_e4_pct": led["checks"]["router_tax_e4_pct"],
                    "router_tax_e16_pct": led["checks"]["router_tax_e16_pct"],
                    "peak_gpu_mem_mb": {n: arm_table[n]["peak_gpu_mem_mb"] for n in arm_table},
                    "wall_minutes": {n: arm_table[n]["wall_minutes"] for n in arm_table},
                    "wall_in_criterion": False,
                    "note": "每 token MACs / 显存 / 墙钟 / token-per-param 都是申报列，不进判词"},
        "c6_routing": {"summary": routing_summary,
                       "collapse_rule": "任一层归一化熵 < 0.5 或最大占比 > 0.5",
                       "note": "路由行为是必须报的对照列；坍缩单独登记，不单独构成否证（均衡属 4.2）"},
        "c7_stability": stability,
        "main_verdict": {
            "quality_at_equal_macs": {"E4": crit["c1_main_equal_macs"]["E4"]["verdict"],
                                      "E16": crit["c1_main_equal_macs"]["E16"]["verdict"]},
            "expert_count": crit["c2_expert_count"]["verdict"],
            "attribution_vs_eqparam_dense": crit["c3_attribution_eqparam"]["verdict"],
            "param_caveat": {n: {"params_x_dense": led["arms"][n]["params_x_dense"],
                                 "tokens_per_param": led["arms"][n]["tokens_per_param"]}
                             for n in ("moe-top1-E4-s0", "moe-top1-E16-s0")},
            "n_seed": 2,
        },
        "train_declaration": {n: {"wall_minutes": arm_table[n]["wall_minutes"],
                                  "peak_gpu_mem_mb": arm_table[n]["peak_gpu_mem_mb"],
                                  "gnorm_max": arm_table[n]["gnorm_max"],
                                  "source": f"runs/{n}/results.json"} for n in arm_table},
    }
    bad = [p for p in pairs.values() if "delta_consistency_abs" in p
           and p["delta_consistency_abs"] >= float(CHECKS["delta_consistency_abs_max"])]
    if bad:
        raise SystemExit(f"Δ 与两端 bpc 对不上（> {CHECKS['delta_consistency_abs_max']}）：{bad}")
    write_results(os.path.join(HERE, "results.json"), out, source=__file__)
    print(json.dumps(out["main_verdict"], indent=2, ensure_ascii=False))
    print(json.dumps({k: v["verdict"] for k, v in crit.items()}, indent=2, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("ledger", "preflight", "train", "eval"), required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.stage in ("train", "eval"):
        if require_prereg(Path(HERE)) != 0:
            raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                             f"{os.path.relpath(HERE, ROOT)} --contract README.md "
                             "config.json run.py train_arms.sh")

    if args.stage == "ledger":
        write_results(os.path.join(HERE, "ledger.json"), ledger(), source=__file__)
        print(json.dumps(ledger()["checks"], indent=2, ensure_ascii=False))
    elif args.stage == "preflight":
        res = preflight()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if not res["pass"]:
            raise SystemExit("preflight 未过 ⇒ 实现问题，先修接线（README §8 第 1 条）")
    elif args.stage == "train":
        if args.arm not in ARMS:
            raise SystemExit(f"unknown arm {args.arm}; 可选 {sorted(ARMS)}")
        train_arm(args.arm, args.steps or None, args.out_root)
    else:
        evaluate()


if __name__ == "__main__":
    main()
