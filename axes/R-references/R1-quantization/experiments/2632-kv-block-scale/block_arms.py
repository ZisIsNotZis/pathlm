"""2632 · 标尺方案感知的 KV 量化前向（本实验新臂的唯一语义层）。

契约：同目录 `config.json`（判据线**运行前**写入）与 `README.md`。

本模块只做一件事：把 `tools/evalkit/kv_quant.py::quantize_kv` 的 **标尺方案**
（`per_value` / `block` / `block_tok` / `kivi`）接进评测前向，并同时支持

- stock `Block`（trunk 族：`int8` / `int4` / `win32_int8` / `win64_int8` /
  `win16_anchor16_int4` / `win32_anchor32_int4`）与
- `GQABlock`（头共享族：`gqa_int8` / `mqa_int8` / `gqa_int4` / `mqa_int4`），

外加可选的可见性掩码（`window` / `window+anchor`）。

`scheme="per_value"` 时必须与 `2631` 的生产前向（`rest_arms.masked_kv_quant_logits`
/ `gaps_arms.kv_quant_head_logits` / `kv_quant.kv_quant_logits`）**逐位一致**——这是
「唯一改动的因子只有标尺方案」的实现层 equality 证据（单测 + `run.py` 仪器自检钉住）。

新逻辑集中在本模块（SCHEMA §22 第 7 条：新逻辑进新模块）；被复用的 `tools/evalkit/kv_quant.py`
只增加了 `scheme` 关键字分支（默认 `per_value`，对既有调用向后兼容）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import GQABlock, kv_head_liveness, window_anchor_keep_mask
from pathlm.model import Block
from tools.evalkit.kv_quant import (_merge, _new_stats, quantize_kv, summarize_stats)
from tools.evalkit.retention import mask_stats, recent_keep_mask

#: 本模块负责的臂 kind（量化档的三种拓扑来源）。
QUANT_KINDS = ("quant", "quant_head", "combo_mask")


def _keep_mask_of(spec: dict, T: int, device):
    """可见性掩码：`None` = 全因果（纯量化臂）；否则按 `window` / `anchors` 给。"""
    kind = spec["kind"]
    if kind == "combo_mask":
        return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                       device=device)
    if kind in ("quant", "quant_head"):
        return None
    raise ValueError(f"block_arms 不处理 kind={kind!r}")


@torch.no_grad()
def masked_kv_quant_logits(model, tokens: torch.Tensor, keep_mask, mode: str,
                           scheme: str = "per_value", group: int = 32,
                           kivi_group: int = 32):
    """dense 主干全深度前向的 logits `[B,T,V]`：注意力可见集由 `keep_mask` 给定（None = 全因果），
    K/V 走 `mode` 精度、`scheme` 标尺方案的往返。

    支持 stock `Block` 与 `GQABlock`。返回 `(logits, stats)`。`scheme="per_value"` 且
    `keep_mask=None` 时与 `tools.evalkit.kv_quant.kv_quant_logits(model, x, mode)` 逐位相等；
    `scheme="per_value"` + 掩码时与 `rest_arms.masked_kv_quant_logits` 逐位相等（单测钉住）。
    """
    B, T = tokens.shape
    attn_mask = None
    if keep_mask is not None:
        if keep_mask.shape == (T, T):
            attn_mask = keep_mask.view(1, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        elif keep_mask.shape == (B, T, T):
            attn_mask = keep_mask.view(B, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        else:
            raise ValueError(f"keep_mask 必须是 [{T},{T}] 或 [{B},{T},{T}]，收到 "
                             f"{tuple(keep_mask.shape)}")
    h = model._input_states(tokens)
    acc = _new_stats()
    for blk in model.blocks:
        if type(blk) is Block:
            B_, T_, d = h.shape
            q, k, v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)

            def shape(t, _B=B_, _T=T_, _d=d, _nh=blk.n_heads):
                return t.view(_B, _T, _nh, _d // _nh).transpose(1, 2)

            ks, ki = quantize_kv(shape(k), mode, scheme, group, kivi_group, is_key=True)
            _merge(acc, ki)
            vs, vi = quantize_kv(shape(v), mode, scheme, group, kivi_group, is_key=False)
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

            ks, ki = quantize_kv(shape(k, blk.n_kv_heads), mode, scheme, group,
                                 kivi_group, is_key=True)
            _merge(acc, ki)
            vs, vi = quantize_kv(shape(v, blk.n_kv_heads), mode, scheme, group,
                                 kivi_group, is_key=False)
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
                f"block_arms 只定义在 stock Block / GQABlock 上，遇到 {type(blk).__name__}")
    return model.readout(h), acc


def make_logits_fn(model, spec: dict, T: int, device, scheme: str = "per_value",
                   group: int = 32, kivi_group: int = 32):
    """把一个量化臂变成 `tokens -> logits`（与 2619/2621 同一接口）。"""
    if spec["kind"] not in QUANT_KINDS:
        raise ValueError(f"block_arms 只处理量化臂 {QUANT_KINDS}，收到 kind={spec['kind']!r}")
    mode = spec["mode"]
    keep = _keep_mask_of(spec, T, device)

    def fn(x, _m=model, _k=keep, _mode=mode, _s=scheme, _g=group, _kg=kivi_group):
        return masked_kv_quant_logits(_m, x, _k, _mode, _s, _g, _kg)[0]
    return fn


def activity_of(spec: dict, model, T: int, device, probe_tokens: torch.Tensor,
                scheme: str = "per_value", group: int = 32, kivi_group: int = 32) -> dict:
    """SCHEMA §10 活性量：量化真的改了数值（`rel_err` / `changed_frac` / `scale_*`），
    并报真实 K/V 张量形状（头共享）与掩码挡掉的因果对（窗口/锚点）。"""
    keep = _keep_mask_of(spec, T, device)
    _, stats = masked_kv_quant_logits(model, probe_tokens, keep, spec["mode"], scheme,
                                      group, kivi_group)
    out = {"kind": spec["kind"], "mode": spec["mode"], "scheme": scheme,
           "group": group if scheme in ("block", "block_tok") else None,
           "kivi_group": kivi_group if scheme == "kivi" else None,
           "quant": summarize_stats(stats)}
    if keep is not None:
        out.update(mask_stats(keep))
        out["positions_kept_analytic"] = int(keep[:, -1].sum().item())
    if spec["kind"] == "quant_head":
        out.update(kv_head_liveness(model, probe_tokens))
    return out
