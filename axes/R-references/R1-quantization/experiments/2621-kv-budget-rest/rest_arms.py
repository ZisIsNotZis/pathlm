"""2621 · KV 预算图**剩余缺失组合** —— 本批新臂的唯一语义层。

契约：同目录 `README.md`（判据同 V1，逐字抄录在 §4）。
本模块是**新文件**（不改 `gaps_arms.py` / `2619/run.py` / `tools/evalkit/*`，SCHEMA §22 第 7 条：
新逻辑进新模块）。

新增**一类**语义 `combo_mask`：**可见性掩码 × KV 量化在同一前向上同时开放**。它覆盖 W1 的
三个优先缺口：

1. **窗口 × int8**：`window=d, anchors=0, mode="int8"`（见 `win32_int8` / `win64_int8`）；
2. **锚点 × int4**：`window=w, anchors=a, mode="int4"`（见 `win16_anchor16_int4` /
   `win32_anchor32_int4`）；
3. **头共享 × 窗口**：`base_model="gqa", mode="fp16"`（= 纯掩码 × GQA 模型；量化档是恒等，
   只由掩码改可见集）。

`quant_head`（GQA/MQA × int4）不是新语义：它是 `gaps_arms.quant_head` 换 `mode`，
本模块只做分派（`mode` / `value_bytes` 由 config 给），不复制实现。

横轴公式与 `2619`/`2620` **逐字同一式**（判据 C1）：

    bytes_per_position = 2(K,V) · n_layers · n_kv_heads_eff · head_dim
                         · value_bytes · (positions_kept / T)

本类相对窗口+锚点档只**同时**改两个乘子：`value_bytes`（量化）与 `positions_kept`
（窗口/锚点）。`n_kv_heads_eff` 只在 `base_model="gqa"`（头共享）时取 2。

单测：`tests/test_kv_budget_rest.py`（每个语义开关都有可失败单测 + 变异验证）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import GQABlock, kv_head_liveness, window_anchor_keep_mask
from pathlm.model import Block
from tools.evalkit.kv_quant import (_merge, _new_stats, quantize_kv,
                                    summarize_stats)
from tools.evalkit.retention import mask_stats, recent_keep_mask

#: 每值字节数（解析口径）：与 `tools/evalkit/kv_quant.py` 的存储精度一一对应。
VALUE_BYTES = {"fp32": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}

#: 本模块负责的 kind（其余 kind 由 run.py 复用 2619 / gaps_arms 的分派）。
COMBO_KINDS = ("combo_mask",)


# ------------------------------------------------------------------ 横轴公式


def effective_n_kv_heads(spec: dict, n_heads: int) -> int:
    """横轴公式里的 `n_kv_heads_eff`：`base_model="gqa"` 时取 `n_kv_heads`，否则 = `n_heads`。"""
    if spec.get("base_model") == "gqa":
        nkv = spec.get("n_kv_heads")
        if nkv is None:
            raise ValueError(f"base_model='gqa' 的臂必须给 n_kv_heads（收到 {spec}）")
        return int(nkv)
    return n_heads


def positions_kept(spec: dict, T: int) -> int:
    """被保留的位置数（**解析口径**）：窗口档 = `window`，窗口+锚点档 = `window + anchors`。"""
    if spec["kind"] not in COMBO_KINDS:
        raise ValueError(f"rest_arms 不处理 kind={spec['kind']!r}")
    kept = int(spec["window"]) + int(spec.get("anchors", 0))
    if not 0 < kept <= T:
        raise ValueError(
            f"window({spec['window']}) + anchors({spec.get('anchors', 0)}) = {kept} 越界 "
            f"(0, T={T}]：锚点与窗口区间重叠或越界时『位置预算』不是相加的，本图不定义该点")
    return kept


def bytes_per_position(spec: dict, n_layers: int, n_heads: int, head_dim: int,
                       T: int) -> dict:
    """**唯一**的横轴公式（判据 C1，与 2619/2620 逐字同一式）。

    `value_bytes` 必须与该臂的 `mode` 一致（不一致 = 标签与事实不符 ⇒ 直接报错）。
    """
    vb = float(spec["value_bytes"])
    if "mode" in spec and spec["mode"] in VALUE_BYTES:
        want = VALUE_BYTES[spec["mode"]]
        if abs(vb - want) > 1e-9:
            raise ValueError(
                f"臂 {spec.get('arm', '?')}: value_bytes={vb} 与 mode={spec['mode']}"
                f"（应 {want}）不一致")
    nkv = effective_n_kv_heads(spec, n_heads)
    kept = positions_kept(spec, T)
    per_pos = 2 * n_layers * nkv * head_dim * vb * (kept / T)
    return {"bytes_per_position": round(per_pos, 6),
            "factor_value_bytes": vb,
            "factor_n_kv_heads_eff": nkv,
            "factor_positions_kept": kept,
            "factor_positions_kept_over_T": round(kept / T, 6),
            "T": T}


def keep_mask_of(spec: dict, T: int, device="cpu") -> torch.Tensor:
    """可见性掩码（本类恒有掩码）：最近 `window` 个位置 ∪ 最前 `anchors` 个位置
    （= `depth_ar.window_anchor_keep_mask`，训练/评测同一件掩码；`anchors=0` 退化为
    `retention.recent_keep_mask`，单测钉住）。"""
    if spec["kind"] not in COMBO_KINDS:
        raise ValueError(f"rest_arms 不处理 kind={spec['kind']!r}")
    return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                   device=device)


# ------------------------------------------------------------------ 前向（掩码 × 量化）


@torch.no_grad()
def masked_kv_quant_logits(model, tokens: torch.Tensor,
                           keep_mask: torch.Tensor | None, mode: str):
    """dense 主干在全深度前向下的 logits `[B,T,V]`：注意力可见集由 `keep_mask` 给定，
    K/V 走 `mode` 精度的往返。

    这是「可见性掩码」与「KV 量化」两个因子的**同一前向**。两个退化路径必须逐位还原既有
    仪器（单测钉住）：

    - `mode="fp32"` ⇒ 与 `tools/evalkit/retention.masked_logits(model, x, keep_mask)`
      逐位相等（量化恒等）；
    - `keep_mask=None` ⇒ 与 `tools/evalkit/kv_quant.kv_quant_logits(model, x, mode)`
      逐位相等（纯因果）。

    支持 stock `Block` 与 `GQABlock`（头共享 × 窗口），其余层类型直接报错。
    """
    if mode not in VALUE_BYTES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {tuple(VALUE_BYTES)}")
    B, T = tokens.shape
    attn_mask = None
    if keep_mask is not None:
        if keep_mask.shape == (T, T):
            attn_mask = keep_mask.view(1, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        elif keep_mask.shape == (B, T, T):
            attn_mask = keep_mask.view(B, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        else:
            raise ValueError(f"keep_mask must be [{T},{T}] or [{B},{T},{T}], "
                             f"got {tuple(keep_mask.shape)}")
    h = model._input_states(tokens)
    acc = _new_stats()
    for blk in model.blocks:
        if type(blk) is Block:
            B_, T_, d = h.shape
            q, k, v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)

            def shape(t, _B=B_, _T=T_, _d=d, _nh=blk.n_heads):
                return t.view(_B, _T, _nh, _d // _nh).transpose(1, 2)

            ks, ki = quantize_kv(shape(k), mode)
            _merge(acc, ki)
            vs, vi = quantize_kv(shape(v), mode)
            _merge(acc, vi)
            a = F.scaled_dot_product_attention(shape(q), ks, vs, attn_mask=attn_mask,
                                               is_causal=attn_mask is None)
            a = a.transpose(1, 2).reshape(B_, T_, d)
            h = h + blk.proj(a)
            h = h + blk.mlp(blk.ln2(h))
        elif type(blk) is GQABlock:
            B_, T_, d = h.shape
            qkv = blk.qkv(blk.ln1(h))
            q, kv = qkv[..., :d], qkv[..., d:]
            k, v = kv[..., :blk.kv_dim], kv[..., blk.kv_dim:]

            def shape(t, nh, _B=B_, _T=T_, _dh=blk.head_dim):
                return t.view(_B, _T, nh, _dh).transpose(1, 2)

            ks, ki = quantize_kv(shape(k, blk.n_kv_heads), mode)
            _merge(acc, ki)
            vs, vi = quantize_kv(shape(v, blk.n_kv_heads), mode)
            _merge(acc, vi)
            a = F.scaled_dot_product_attention(shape(q, blk.n_heads), ks, vs,
                                               attn_mask=attn_mask,
                                               is_causal=attn_mask is None,
                                               enable_gqa=True)
            a = a.transpose(1, 2).reshape(B_, T_, d)
            h = h + blk.proj(a)
            h = h + blk.mlp(blk.ln2(h))
        else:
            raise ValueError(
                f"combo_mask 只定义在 stock Block / GQABlock 上，遇到 {type(blk).__name__}")
    return model.readout(h), acc


@torch.no_grad()
def reduction_self_check(model, tokens: torch.Tensor, T: int, device: str,
                         window: int = 32, anchors: int = 8) -> dict:
    """combo_mask 前向的**两条退化路径**与既有仪器逐位对照（SCHEMA §9 实现层 equality）。

    - `mode="fp32"`（量化恒等）⇒ 与 `retention.masked_logits` 逐位相等（掩码路径同源）；
    - 掩码 = 全因果 ⇒ 与 `kv_quant.kv_quant_logits` 逐位相等（量化路径同源）；
    - 掩码 = None 且 `mode="fp32"` ⇒ 与 `kv_quant.kv_quant_logits(..., "fp32")` 逐位相等。

    返回每个判定的 bool（供 `results.json::instrument` 留证）。模型须是 stock `Block` 主干。
    """
    from tools.evalkit.kv_quant import kv_quant_logits
    from tools.evalkit.retention import masked_logits

    x = tokens[:4].to(device)
    keep_win = window_anchor_keep_mask(T, window, anchors, device=device)
    keep_full = window_anchor_keep_mask(T, T, 0, device=device)
    a = masked_kv_quant_logits(model, x, keep_win, "fp32")[0]
    b = masked_logits(model, x, keep_win)
    c = masked_kv_quant_logits(model, x, keep_full, "int8")[0]
    d = kv_quant_logits(model, x, "int8")[0]
    e = masked_kv_quant_logits(model, x, None, "fp32")[0]
    f = kv_quant_logits(model, x, "fp32")[0]
    return {
        "mask_fp32_vs_masked_logits_bitwise": bool(torch.equal(a, b)),
        "mask_full_quant_int8_vs_kv_quant_logits_bitwise": bool(torch.equal(c, d)),
        "causal_fp32_vs_kv_quant_fp32_bitwise": bool(torch.equal(e, f)),
    }


def make_logits_fn(model, spec: dict, T: int, device: str):
    """把一个 `combo_mask` 臂变成 `tokens -> logits`（与 2619/2620 同一接口）。"""
    if spec["kind"] not in COMBO_KINDS:
        raise ValueError(f"rest_arms 不处理 kind={spec['kind']!r}；请交给 2619/gaps_arms 的分派")
    keep = keep_mask_of(spec, T, device)
    mode = spec.get("mode", "fp32")

    def fn(x, _m=model, _k=keep, _mode=mode):
        return masked_kv_quant_logits(_m, x, _k, _mode)[0]
    return fn


def activity_of(spec: dict, model, T: int, device: str, probe_tokens: torch.Tensor) -> dict:
    """SCHEMA §10 活性量：证明 `combo_mask` 臂里**两个因子都真的活着**。

    - 掩码因子：`mask_stats` 报被挡掉的因果 (query, key) 对（`dropped_frac > 0`），
      并单列锚点贡献的可见对数（`base_model="gqa"` 的臂 `anchors=0`，该项为 0）；
    - 量化因子：`summarize_stats` 报 `rel_err` / `changed_frac`（`mode="fp16"` 的纯掩码
      臂该项应为 0 ⇒ 这正是「该臂没有量化因子」的证据）；
    - 头因子（`base_model="gqa"`）：`kv_head_liveness` 用**真实张量形状**证明 KV 头数。
    """
    if spec["kind"] not in COMBO_KINDS:
        raise ValueError(f"rest_arms 不处理 kind={spec['kind']!r}")
    keep = keep_mask_of(spec, T, device)
    mode = spec.get("mode", "fp32")
    _, stats = masked_kv_quant_logits(model, probe_tokens, keep, mode)
    anchors = int(spec.get("anchors", 0))
    out = {"kind": "combo_mask", "base_model": spec.get("base_model", "trunk"),
           "mode": mode, "window": int(spec["window"]), "anchors": anchors,
           "positions_kept_analytic": positions_kept(spec, T),
           **mask_stats(keep),
           "quant": summarize_stats(stats),
           "head": kv_head_liveness(model, probe_tokens)}
    out["anchor_contribution_pairs"] = (
        int(keep[:, :anchors].sum().item()) if anchors > 0 else 0)
    return out
