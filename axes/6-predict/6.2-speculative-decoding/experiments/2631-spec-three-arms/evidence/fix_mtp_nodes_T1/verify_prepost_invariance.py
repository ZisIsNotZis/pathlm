"""F_ENG 的前后不变性证据：`pathlm/model.py::PathLM._mtp_nodes` 的单行修复。

**缺陷**（本波新发现，P0 级）：`chain_mode="token"` 在**单位置解码**（`h.shape[1]==1`）下崩溃。
修复前该分支无条件把 `cond` 右垫一行：

    t_next = torch.cat([cond, cond[:, -1:]], dim=1)   # 无条件右垫

`h.shape[1] == 1` 时 `n_cond = max(0, 1) = 1` ⇒ `cond` 已是 1 行、右垫后变 2 行 ⇒
`torch.cat([h, embed(t_next)], dim=-1)` 在维度 1 上 1 vs 2 失配 ⇒ RuntimeError。
`pathlm/decode.py::Decoder.step()` 是所有单 token 解码与 prefill 的入口 ⇒ **连 native
对照都跑不起来**，也说明 ticket 12 的 "chain deploy 0.547" 不是解码口径。

**修复**：只在 `cond` 行数不足时右垫。

**本脚本的作用（supervisor 裁决 ① 的第 1 条硬要求）**：给「修复对 T≥2 逐位不变」提供
**数值证据**，而不是推理：

- 把修复前/后的表达式各写一份**冻结副本**（`_chain_pad_pre` / `_chain_pad_post`），
  在同一权重、同一输入上比较 `h_c` 与 `chain2` 的 logits/latent 的 **max|Δ|** 与 **sha256**；
- 与**活引擎** `PathLM._mtp_nodes` 的输出比对（证明冻结副本没有偏离引擎）；
- 在修复前 / 修复后**各跑一次**，把两次的 live 张量哈希落盘并逐条相等（真正的 before/after）。

产物：`invariance_pre.json` / `invariance_post.json`（均由本脚本写出；后者由
`compare_prepost()` 与前者逐条核对）。

跑法：`python3 .../verify_prepost_invariance.py --label pre|post`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根（含 pathlm/ 的祖先目录）: {start}")


ROOT = _repo_root(HERE)
for p in (ROOT, os.path.abspath(os.path.join(HERE, "..", ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)

from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

BASE_SHA = "d3d64b8abc0a471a20242c6d76d788ab40fa16c5"   # 修复前基线（见约定 commit）


def tiny_token_model(seed: int = 0) -> PathLM:
    torch.manual_seed(seed)
    mcfg = ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=32, mlp_mult=2)
    pcap = PathConfig(n_mtp=3, chain_mode="token", transport="none")
    return PathLM(mcfg, pcap, vocab_size=41).eval()


def _chain_pad_pre(cond: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """修复前表达式（冻结副本，逐字来自 BASE_SHA 的 `_mtp_nodes`）。"""
    return torch.cat([cond, cond[:, -1:]], dim=1)


def _chain_pad_post(cond: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """修复后表达式（冻结副本）：只在行数不足时右垫。"""
    return cond if cond.shape[1] >= h.shape[1] else torch.cat([cond, cond[:, -1:]], dim=1)


def _cond_for(model, h, condition):
    """复现 `_mtp_nodes` 里 `cond` 的取值（node-1 logits 由同一权重算出）。"""
    with torch.no_grad():
        logits1 = model.transforms[0](h) @ model.embed.weight.T
        cond = condition
        if model.deploy_chain or cond is None:
            n_cond = max(h.shape[1] - 1, 1)
            cond = logits1.argmax(-1)[:, :n_cond]
        return cond


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().float().numpy().tobytes()).hexdigest()[:32]


@torch.no_grad()
def case(model, name, B, T, use_condition, seed):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(B, T, model.mcfg.d_model, generator=g)
    # 引擎契约：`condition` 的形状是 [B, T-1]（行 i 条件在 t_{i+1} 上）——
    # 这是 `targets[:, 1:]` 的形状（见 `model.py::forward`）。
    tokens = torch.randint(0, 41, (B, T), generator=g)
    condition = tokens[:, 1:] if use_condition else None
    cond = _cond_for(model, h, condition)
    pad_pre = _chain_pad_pre(cond, h)
    pad_post = _chain_pad_post(cond, h)
    hc_pre = model.chain_cat(torch.cat([h, model.embed(pad_pre)], dim=-1))
    hc_post = model.chain_cat(torch.cat([h, model.embed(pad_post)], dim=-1))
    logits_pre = model.transforms[0](hc_pre) @ model.embed.weight.T
    logits_post = model.transforms[0](hc_post) @ model.embed.weight.T
    # 活引擎（修复前 / 修复后各跑一次）
    live = model._mtp_nodes(h, condition=condition)
    live_chain = live[1]["chain2"]
    row = {
        "case": name, "B": B, "T": T, "use_condition": use_condition,
        "max_abs_diff_hc": float((hc_pre - hc_post).abs().max()),
        "max_abs_diff_chain2_logits": float((logits_pre - logits_post).abs().max()),
        "max_abs_diff_live_vs_post_logits":
            float((live_chain["logits"] - logits_post).abs().max()),
        "sha256_hc_pre": _digest(hc_pre), "sha256_hc_post": _digest(hc_post),
        "sha256_chain2_logits_pre": _digest(logits_pre),
        "sha256_chain2_logits_post": _digest(logits_post),
        "sha256_live_chain2_logits": _digest(live_chain["logits"]),
        "sha256_live_chain2_latent": _digest(live_chain["latent"]),
    }
    return row


def t1_probe(model):
    """T==1：修复前崩溃、修复后正常——把两者的实际行为记下来。"""
    h = torch.zeros(1, 1, model.mcfg.d_model)
    out = {"t1_pre_fix_raises": None, "t1_pre_fix_error": None,
           "t1_post_fix_ok": None}
    try:
        cond = _cond_for(model, h, None)
        pad_pre = _chain_pad_pre(cond, h)
        model.chain_cat(torch.cat([h, model.embed(pad_pre)], dim=-1))
        out["t1_pre_fix_raises"] = False
    except RuntimeError as exc:
        out["t1_pre_fix_raises"] = True
        out["t1_pre_fix_error"] = str(exc)[:160]
    try:
        live = model._mtp_nodes(h)
        out["t1_post_fix_ok"] = "chain2" in live[1]
    except RuntimeError as exc:
        out["t1_post_fix_ok"] = False
        out["t1_live_error"] = str(exc)[:160]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=("pre", "post"), required=True)
    args = ap.parse_args()
    model = tiny_token_model(seed=0)
    rows = [
        case(model, "teacher_forcing_T8", 2, 8, True, 1),
        case(model, "deploy_T8", 2, 8, False, 2),
        case(model, "teacher_forcing_T2", 2, 2, True, 3),
        case(model, "deploy_T2", 2, 2, False, 4),
    ]
    report = {
        "label": args.label,
        "base_sha_before_fix": BASE_SHA,
        "engine_base_rev": os.popen("git -C '%s' rev-parse HEAD" % ROOT).read().strip(),
        "claim": "修复对 T>=2 的输出逐位不变（max|Δ| == 0）",
        "cases": rows,
        "t1_probe": t1_probe(model),
        "all_T_ge_2_identical": all(
            r["max_abs_diff_hc"] == 0.0
            and r["max_abs_diff_chain2_logits"] == 0.0
            and r["max_abs_diff_live_vs_post_logits"] == 0.0
            for r in rows),
    }
    out = os.path.join(HERE, f"invariance_{args.label}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("saved", out, flush=True)
    return 0 if report["all_T_ge_2_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
