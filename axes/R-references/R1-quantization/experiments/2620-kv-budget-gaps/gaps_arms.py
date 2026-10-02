"""2620 · 补齐 U3 预算图的三个最优先缺失组合 —— **新臂的唯一语义层**。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 `.scratch/37-gaps/issues/01-gaps.md`
的「契约 V1」，执行期间未改）。

本模块只加**两个拓扑新组合**的语义（SCHEMA §22 第 7 条：新逻辑进新模块，不改
`2619` 与 `tools/evalkit/` 里的既有实现）：

1. `quant_head`：头共享 checkpoint（GQA / MQA）× KV int8 量化。两个因子作用在
   **同一个乘式**的不同位置——`n_kv_heads_eff` 与 `value_bytes`——所以它们可以在
   同一横轴上叠加，是 U3 的第一推荐缺口。
2. `window_anchor`：窗口 + 锚点槽（前 `anchors` 个绝对位置**恒可见**）。锚点**占用
   位置预算**，因此横轴仍是 `2·n_layers·n_kv_heads_eff·head_dim·value_bytes·(kept/T)`，
   其中 `kept = window + anchors`。

其余臂（`quant` / `head` / `window`）直接沿用 `2619` 的语义，本模块只做必要的
分派，不改那些实现。

单测：`tests/test_kv_budget_gaps.py`（每个语义开关都有可失败单测 + 变异验证）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import GQABlock, window_anchor_keep_mask
from pathlm.model import Block
from tools.evalkit.kv_quant import (_merge, _new_stats, kv_quant_logits,
                                     quantize_kv, summarize_stats)
from tools.evalkit.retention import masked_logits, mask_stats, recent_keep_mask

#: 每值字节数（解析口径）：与 `tools/evalkit/kv_quant.py` 的存储精度一一对应。
VALUE_BYTES = {"fp32": 2.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}

#: 本模块负责的**新组合** kind（其余 kind 交给 run.py 复用 2619 的分派）。
NEW_KINDS = ("quant_head", "window_anchor")


@torch.no_grad()
def kv_quant_head_logits(model, tokens: torch.Tensor, mode: str):
    """`quant_head` 臂的前向：头共享模型（GQA/MQA）× KV `mode` 精度往返。

    **为什么不能用 `tools.evalkit.kv_quant.kv_quant_logits`**：那个函数显式只接受
    stock `Block`（它的 docstring：内容 top-k 等自定义层会让「唯一因素 = KV 精度」
    不成立）——头共享模型用的是 `GQABlock`，会被它直接 `raise`。因此本模块复刻
    同一段 dense 前向，只把 `Block` 换成 `GQABlock`（K/V 半边宽 = `n_kv_heads·dh`，
    SDPA 走 `enable_gqa=True`，**不物化**广播到 n_heads）。量化算子（`quantize_kv`）
    与活性累加器（`_new_stats` / `_merge`）直接复用既有实现 ⇒ 与 2619 的量化
    语义逐位同一件仪器。

    `mode="fp32"` 时必须与头共享模型的释放版 dense 前向逐位相等（单测钉住），
    所以这不是「另一个模型」。
    """
    h = model._input_states(tokens)
    acc = _new_stats()
    for blk in model.blocks:
        if type(blk) is GQABlock:
            B, T, d = h.shape
            qkv = blk.qkv(blk.ln1(h))
            q, kv = qkv[..., :d], qkv[..., d:]
            k, v = kv[..., :blk.kv_dim], kv[..., blk.kv_dim:]

            def shape(t, nh):
                return t.view(B, T, nh, blk.head_dim).transpose(1, 2)

            qs = shape(q, blk.n_heads)
            ks, ki = quantize_kv(shape(k, blk.n_kv_heads), mode)
            _merge(acc, ki)
            vs, vi = quantize_kv(shape(v, blk.n_kv_heads), mode)
            _merge(acc, vi)
            a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True,
                                               enable_gqa=True)
            a = a.transpose(1, 2).reshape(B, T, d)
            h = h + blk.proj(a)
            h = h + blk.mlp(blk.ln2(h))
        elif type(blk) is Block:      # 兜底：MHA 臂不应走 quant_head，但保持可算
            from tools.evalkit.kv_quant import _block_forward_quant
            h = _block_forward_quant(blk, h, mode, acc)
        else:
            raise ValueError(
                f"quant_head 只定义在 stock Block / GQABlock 上，遇到 {type(blk).__name__}")
    return model.readout(h), summarize_stats(acc)


def effective_n_kv_heads(spec: dict, n_heads: int) -> int:
    """横轴公式里的 `n_kv_heads_eff`：只有头共享族改它，其余族 = `n_heads`。"""
    return spec["n_kv_heads"] if spec["kind"] in ("head", "quant_head") else n_heads


def positions_kept(spec: dict, T: int) -> int:
    """被保留的位置数（**解析口径**）：窗口档 = `window`，窗口+锚点档 = `window+anchors`。"""
    kind = spec["kind"]
    if kind == "window":
        return int(spec["d"])
    if kind == "window_anchor":
        kept = int(spec["window"]) + int(spec["anchors"])
        if kept > T:
            raise ValueError(
                f"window({spec['window']}) + anchors({spec['anchors']}) = {kept} > T={T}："
                "锚点与窗口区间重叠时『位置预算』不是相加的，本图不定义该点")
        return kept
    return T


def keep_mask_of(spec: dict, T: int, device="cpu") -> torch.Tensor | None:
    """可见性掩码（`None` = 全因果）。

    - `window`：只保留最近 `d` 个位置（= `retention.recent_keep_mask`，与 2619 同）；
    - `window_anchor`：最近 `window` 个位置 ∪ 最前 `anchors` 个位置
      （= `depth_ar.window_anchor_keep_mask`，训练/评测同一件掩码）。
    """
    kind = spec["kind"]
    if kind == "window":
        return recent_keep_mask(T, spec["d"], device=device)
    if kind == "window_anchor":
        return window_anchor_keep_mask(T, spec["window"], spec["anchors"], device=device)
    return None


def make_logits_fn(model, spec: dict, T: int, device: str):
    """把一个臂变成 `tokens -> logits`（与 2619 的 `make_logits_fn` 同一接口）。

    本模块负责 `quant_head`（头共享 + 量化）与 `window_anchor`（窗口 + 锚点）；
    其余 kind 由 `run.py` 复用 2619 的分派（本函数不处理）。
    """
    kind = spec["kind"]
    if kind == "quant_head":
        mode = spec["mode"]

        def fn(x, _m=model, _mode=mode):
            return kv_quant_head_logits(_m, x, _mode)[0]
        return fn
    if kind == "window_anchor":
        keep = keep_mask_of(spec, T, device)

        def fn(x, _m=model, _k=keep):
            return masked_logits(_m, x, _k)
        return fn
    raise ValueError(f"gaps_arms 不处理 kind={kind!r}；请交给 2619 的分派")


def activity_of(spec: dict, model, T: int, device: str, probe_tokens: torch.Tensor) -> dict:
    """SCHEMA §10 活性量：证明新组合臂里**两个因子都真的活着**。

    - `quant_head`：既报头共享的**实际张量形状**（`kv_head_liveness`），又报量化的
      `rel_err` / `changed_frac`（真改了数值）；
    - `window_anchor`：报掩码实际挡掉了多少因果 (query,key) 对（`mask_stats`），
      并单列锚点贡献的位置数。
    """
    from pathlm.depth_ar import kv_head_liveness

    kind = spec["kind"]
    if kind == "quant_head":
        _, stats = kv_quant_head_logits(model, probe_tokens, spec["mode"])
        head = kv_head_liveness(model, probe_tokens)
        return {"kind": "quant_head", "mode": spec["mode"],
                "n_kv_heads": spec["n_kv_heads"],
                "quant": summarize_stats(stats), **head}
    if kind == "window_anchor":
        keep = keep_mask_of(spec, T, device)
        ms = mask_stats(keep)
        return {"kind": "window_anchor", "window": int(spec["window"]),
                "anchors": int(spec["anchors"]), **ms,
                "anchor_contribution_pairs": int(keep[:, :spec["anchors"]].sum().item())
                if spec["anchors"] > 0 else 0}
    raise ValueError(f"gaps_arms 不处理 kind={kind!r}")


def bytes_per_position(spec: dict, n_layers: int, n_heads: int, head_dim: int,
                       T: int) -> dict:
    """**唯一**的横轴公式（判据 C1，与 2619 逐字同一式）：

        bytes_per_position = 2(K,V) · n_layers · n_kv_heads_eff · head_dim
                             · value_bytes · (positions_kept / T)

    新组合只是在同一乘式里**同时**改两个因子：
      - `quant_head`  ⇒ `n_kv_heads_eff`（2 / 1）与 `value_bytes`（1.0）一起改；
      - `window_anchor` ⇒ `positions_kept = window + anchors`。
    `value_bytes` 必须与该臂的 `mode` 一致（不一致 = 标签与事实不符 ⇒ 直接报错）。
    """
    kind = spec["kind"]
    vb = float(spec["value_bytes"])
    if "mode" in spec and spec["mode"] in VALUE_BYTES:
        want = VALUE_BYTES[spec["mode"]]
        if abs(vb - want) > 1e-9:
            raise ValueError(
                f"臂 {spec.get('arm', spec.get('role', '?'))}: value_bytes={vb} 与 "
                f"mode={spec['mode']}（应 {want}）不一致")
    nkv = effective_n_kv_heads(spec, n_heads)
    kept = positions_kept(spec, T)
    per_pos = 2 * n_layers * nkv * head_dim * vb * (kept / T)
    return {"bytes_per_position": round(per_pos, 6),
            "factor_value_bytes": vb,
            "factor_n_kv_heads_eff": nkv,
            "factor_positions_kept": kept,
            "factor_positions_kept_over_T": round(kept / T, 6),
            "T": T}
