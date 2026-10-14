"""2653 · fp8 / 大 head_dim 的语义层（本实验新臂的唯一新逻辑）。

契约：同目录 `README.md`（运行前预登记）+ `config.json`（判据线运行前写入）。

本模块只加两件在 `2619` / `2621` / `2632` 里没有的东西（SCHEMA §22 第 7 条：新逻辑进新模块，
不改 `tools/evalkit/kv_quant.py`、不改 `2632/block_arms.py`）：

1. **fp8 的 KV 往返**（`fp8_roundtrip` / `fp8_kv_logits`）：用 `torch.float8_e4m3fn` /
   `torch.float8_e5m2` 把 K/V 存成 8 位浮点再反读回 fp32。两种口径：
   - `scale_mode="none"`：**原样存放、无标尺**（fp8 的卖点，1.0 B/值）；
   - `scale_mode="tokenscale"`：每 `(batch, head, token)` 沿 `head_dim` 取 amax、存一把 fp16
     标尺（`1 + 2/head_dim` B/值）。标尺只由当前 token 自己的值决定 ⇒ **严格前缀确定**
     （与 `quantize_kv(scheme="block", group=head_dim)` 同一因果性）。
   活性量除 `n_values / n_scales / rel_err / changed_frac` 外，多报 `saturated_frac`
   （`|x| > 格式 max` 的占比）——这正是「fp8 要不要标尺」的直接证据。

2. **块标尺的解析字节账 + head_dim/G 阶梯**（`bytes_per_value_with_scale` / `head_dim_ladder` /
   `deployment_ledger`）：把「值 + 标尺」讲到每值字节这一层，并与 `5.1` 的 int8 G=32
   （1.0625 B/值、6528 B/位置）逐项对账。

统一前向：`kv_logits` 把 int8/int4（走 `quantize_kv` 的 `scheme` 分支）与 fp8（走本模块的往返）
接到**同一条** dense 前向里，`mode="fp32"` 时与释放版前向逐位相等（跑 `run.py` 的仪器自检）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.model import Block
from tools.evalkit.kv_quant import (_merge, _new_stats, quantize_kv,   # noqa: F401
                                     summarize_stats)

#: fp8 数值格式：FP8 Formats for Deep Learning（Micikevicius et al. 2022，arXiv:2209.05433）。
FP8_FORMATS = ("fp8_e4m3", "fp8_e5m2")
_FP8_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2}
#: 各格式的可表示上限（E4M3：±448、无 inf；E5M2：±57344，含 inf/nan）。
FP8_MAX = {k: float(torch.finfo(v).max) for k, v in _FP8_DTYPE.items()}
#: 实兑口径（登记在 results.json）：CPU 上 torch 的 float8 转换核（RN 取整；超范围**饱和**到 max）。
FP8_IMPL = ("torch CPU cast x.to(torch.float8_e4m3fn|e5m2).to(fp32)"
            "（round-to-nearest，无标尺档不缩放）。实测溢出语义："
            "E4M3（fn，无 inf）超 ±448 **饱和**到 ±448；E5M2 超 ±57344 **溢出为 inf**。"
            "本实验另报 `saturated_frac`（|x| > 格式 max 的占比），不把溢出后的数值当结论。")

#: 本模块负责的 KV 存放精度（8 位 = int8 / fp8 两族）。
KV_MODES = ("fp32", "fp16", "int8", "int4") + FP8_FORMATS
_INT_MODES = ("fp32", "fp16", "int8", "int4")

#: `5.1` 的块标尺口径锚（对账用；`head_dim`=32、G=32）。
ANCHOR = {"values_per_position": 6144.0, "fp16_bytes_per_position": 12288.0,
          "int8_block_g32_bytes_per_position": 6528.0,
          "fp8_no_scale_bytes_per_position": 6144.0}


# --------------------------------------------------------------- fp8 往返


@torch.no_grad()
def fp8_roundtrip(x: torch.Tensor, fmt: str, scale_mode: str = "none"):
    """`x`（fp32，布局 `[..., H, T, head_dim]`）→ fp8 存储 → 反读回 fp32。

    返回 `(dq, scale, n_saturated)`：`scale` 为 `tokenscale` 档的 fp16 标尺
    （形状 `[..., H, T, 1]`；`none` 档为 None），`n_saturated` 为 `|x| > 格式 max` 的值个数。
    """
    if fmt not in FP8_FORMATS:
        raise ValueError(f"未知 fp8 格式 {fmt!r}；已知 {FP8_FORMATS}")
    if scale_mode not in ("none", "tokenscale"):
        raise ValueError(f"未知 scale_mode {scale_mode!r}；已知 none / tokenscale")
    dt = _FP8_DTYPE[fmt]
    n_sat = int((x.abs() > FP8_MAX[fmt]).sum().item())
    if scale_mode == "none":
        return x.to(dt).to(x.dtype), None, n_sat
    # 严格前缀确定：标尺只由当前 token 自己的 head_dim 个值决定（token / batch 轴不参与归约）。
    amax = x.abs().amax(dim=-1, keepdim=True)
    s = (amax / FP8_MAX[fmt]).clamp_min(torch.finfo(torch.float16).tiny)
    return ((x / s).to(dt).to(x.dtype)) * s, s, n_sat


# --------------------------------------------------------------- 统计累加器


def _new_stats_fp8() -> dict:
    acc = _new_stats()
    acc["n_saturated"] = 0
    return acc


def _merge_fp8(dst: dict, src: dict) -> None:
    _merge(dst, src)
    dst["n_saturated"] = dst.get("n_saturated", 0) + src.get("n_saturated", 0)


def summarize_fp8(acc: dict) -> dict:
    out = summarize_stats(acc)
    out["n_saturated"] = acc.get("n_saturated", 0)
    out["saturated_frac"] = round(acc.get("n_saturated", 0) / max(acc["n_values"], 1), 9)
    return out


@torch.no_grad()
def quantize_fp8_kv(x: torch.Tensor, fmt: str, scale_mode: str = "none"):
    """fp8 的 KV 往返 + 与本仓其它量化档同形的活性量（`n_values / n_scales / rel_err / ...`）。"""
    acc = _new_stats_fp8()
    dq, scale, n_sat = fp8_roundtrip(x, fmt, scale_mode)
    acc["n_values"] = int(x.numel())
    acc["n_saturated"] = n_sat
    if scale is not None:
        acc["n_scales"] = int(scale.numel())
        acc["scale_min"] = float(scale.min())
        acc["scale_max"] = float(scale.max())
    err = (dq.float() - x.float()).abs()
    acc["sum_abs_err"] = float(err.sum())
    acc["max_abs_err"] = float(err.max()) if x.numel() else 0.0
    acc["sum_abs_x"] = float(x.abs().sum())
    acc["n_changed"] = int((dq != x).sum())
    return dq, acc


# --------------------------------------------------------------- dense 前向（与 kv_quant 同构）


@torch.no_grad()
def _kv_roundtrip(x: torch.Tensor, mode: str, scheme: str, group: int, scale_mode: str,
                  is_key: bool):
    if mode in _INT_MODES:
        # fp16 / fp32 档没有标尺（quantize_kv 要求 scheme='per_value'）。
        sch = scheme if mode in ("int8", "int4") else "per_value"
        return quantize_kv(x, mode, sch, group, 32, is_key=is_key)
    return quantize_fp8_kv(x, mode, scale_mode)


@torch.no_grad()
def kv_logits(model, tokens: torch.Tensor, mode: str, scheme: str = "block", group: int = 32,
              scale_mode: str = "none"):
    """dense 主干全深度前向的 logits `[B,T,V]`，KV 走 `mode` 精度的往返。

    `mode ∈ KV_MODES`。int8/int4 走 `tools.evalkit.kv_quant.quantize_kv` 的 `scheme` 分支
    （`scheme="per_value"` 时与既有生产前向逐位一致）；fp8 走本模块的往返。
    `mode="fp32"` ⇒ 不改数值，与释放版前向逐位相等。返回 `(logits, stats)`。
    """
    if mode not in KV_MODES:
        raise ValueError(f"未知 mode {mode!r}；已知 {KV_MODES}")
    B, T = tokens.shape
    h = model._input_states(tokens)
    acc = _new_stats_fp8()
    for blk in model.blocks:
        if type(blk) is not Block:
            raise ValueError(f"本实验只定义在 stock dense 注意力上，遇到 {type(blk).__name__}")
        d = h.shape[-1]
        q, k, v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)

        def shape(t, _B=B, _T=T, _d=d, _nh=blk.n_heads):
            return t.view(_B, _T, _nh, _d // _nh).transpose(1, 2)

        qs = shape(q)
        ks, ki = _kv_roundtrip(shape(k), mode, scheme, group, scale_mode, is_key=True)
        _merge_fp8(acc, ki)
        vs, vi = _kv_roundtrip(shape(v), mode, scheme, group, scale_mode, is_key=False)
        _merge_fp8(acc, vi)
        a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, d)
        h = h + blk.proj(a)
        h = h + blk.mlp(blk.ln2(h))
    return model.readout(h), summarize_fp8(acc)


def make_logits_fn(model, spec: dict, T: int, dev: str):
    """把一个臂变成 `tokens -> logits`（本实验四臂共用的唯一接口）。"""
    mode = spec["mode"]
    scheme = spec.get("scheme", "block")
    group = int(spec.get("group", 32))
    scale_mode = spec.get("scale_mode", "none")

    def fn(x, _m=model, _mode=mode, _s=scheme, _g=group, _sm=scale_mode):
        return kv_logits(_m, x, _mode, _s, _g, _sm)[0]
    return fn


@torch.no_grad()
def activity_of(spec: dict, model, probe_tokens: torch.Tensor):
    """SCHEMA §10 活性量：KV 数值真的被改（否则质量读数无意义）。"""
    mode = spec["mode"]
    _, stats = kv_logits(model, probe_tokens, mode, spec.get("scheme", "block"),
                         int(spec.get("group", 32)), spec.get("scale_mode", "none"))
    return {"kind": "quant", "mode": mode, "scheme": spec.get("scheme"),
            "group": spec.get("group"), "scale_mode": spec.get("scale_mode", "none"),
            "implementation": FP8_IMPL if mode in FP8_FORMATS else "tools/evalkit/kv_quant.quantize_kv",
            **stats}


# --------------------------------------------------------------- 解析字节账


def values_per_position(n_layers: int, n_kv_heads: int, head_dim: int, kept: int, T: int) -> float:
    """每位置 KV 标量数 = 2(K,V) · L · n_kv_heads · head_dim · (kept/T)。"""
    return 2.0 * n_layers * n_kv_heads * head_dim * (kept / T)


def bytes_per_value_with_scale(group: int) -> float:
    """块标尺（沿 head_dim，每 G 个值一把 fp16 标尺）的每值字节 = 1 (int8) + 2/G。"""
    if group < 1:
        raise ValueError(f"group 必须 ≥1，收到 {group}")
    return 1.0 + 2.0 / group


def head_dim_ladder(head_dims=(32, 64), groups=(8, 16, 32, 64), value_bytes: float = 1.0,
                    scale_bytes: float = 2.0, fp16_bytes_per_position: float = 12288.0) -> list[dict]:
    """`head_dim` × G 的解析阶梯：每值字节 `value + scale/G`、每位置字节、相对 fp16 省幅。

    G 必须整除 head_dim（块沿 head_dim，否则块不整齐、每值摊到的标尺字节不是 scale/G）。
    `head_dim` 越大 G 的上限越大 ⇒ 省幅趋近 `1 − value/2 = 50%`（int8）。
    """
    rows = []
    for hd in head_dims:
        for g in groups:
            legal = (hd % g == 0)
            bpe = value_bytes + scale_bytes / g
            bpp = bpe * fp16_bytes_per_position / 2.0        # fp16 ⇒ 2 B/值、12288 B/位置
            rows.append({"head_dim": hd, "group": g, "legal": bool(legal),
                         "reason": "" if legal else f"G={g} 不整除 head_dim={hd}",
                         "scale_bytes_per_value": round(scale_bytes / g, 6),
                         "bytes_per_value": round(bpe, 6),
                         "bytes_per_position": round(bpp, 3),
                         "ratio_vs_fp16": round(bpp / fp16_bytes_per_position, 6),
                         "reduction_vs_fp16": round(1 - bpp / fp16_bytes_per_position, 6),
                         "prefix_invariant": bool(legal)})
    return rows


def head_dim_cap(head_dim: int) -> dict:
    """`head_dim` 上块标尺的封顶：最大合法 G = head_dim ⇒ 每值 1 + 2/head_dim B。"""
    bpe = 1.0 + 2.0 / head_dim
    return {"head_dim": head_dim, "max_group": head_dim, "bytes_per_value": round(bpe, 6),
            "bytes_per_position": round(bpe * 6144.0, 3),
            "reduction_vs_fp16": round(1 - bpe / 2.0, 6)}


def deployment_ledger(cfg: dict) -> dict:
    """逐臂的部署字节账：每个数字给出解析式与**逐项因子**，并与 `5.1` 的 6528 / 12288 对账。"""
    m = cfg["model"]
    L, nkv, hd, T = m["n_layers"], m["n_heads"], cfg["head_dim"], m["seq_len"]
    vpp = values_per_position(L, nkv, hd, T, T)
    if abs(vpp - ANCHOR["values_per_position"]) > 1e-9:
        raise ValueError(f"每位置值数 {vpp} 与 5.1 的 {ANCHOR['values_per_position']} 不符")
    fp16_bpp = vpp * 2.0
    if abs(fp16_bpp - ANCHOR["fp16_bytes_per_position"]) > 1e-9:
        raise ValueError(f"fp16 每位置字节 {fp16_bpp} 与 5.1 的 12288 不符")
    int8_g32 = vpp * bytes_per_value_with_scale(32)
    if abs(int8_g32 - ANCHOR["int8_block_g32_bytes_per_position"]) > 1e-9:
        raise ValueError(f"int8 块标尺 G=32 每位置字节 {int8_g32} 与 5.1 的 6528 不符")

    def entry(mode: str, scale_mode: str, group):
        if mode in ("fp16", "fp32"):
            value_bpe, scale_bpe = 2.0, 0.0
        elif mode in FP8_FORMATS:
            scale_bpe = 2.0 / hd if scale_mode == "tokenscale" else 0.0
            value_bpe = 1.0
        else:                                     # int8 / int4 + 块标尺
            value_bpe, scale_bpe = 1.0, 2.0 / (group or 32)
        bpe = value_bpe + scale_bpe
        bpp = vpp * bpe
        return {"mode": mode, "scale_mode": scale_mode, "group": group if scale_bpe else None,
                "values_per_position": vpp,
                "value_bytes_per_value": round(value_bpe, 6),
                "scale_bytes_per_value": round(scale_bpe, 6),
                "bytes_per_value": round(bpe, 6),
                "bytes_per_position": round(bpp, 3),
                "ratio_vs_fp16": round(bpp / fp16_bpp, 6),
                "reduction_vs_fp16": round(1 - bpp / fp16_bpp, 6),
                "formula": ("2*L*n_kv_heads*head_dim*(kept/T) * value_bytes"
                            " [+ values_per_position * 2/G] for int8-block / fp8-token-scale")}

    arms = {name: entry(s["mode"], s.get("scale_mode", "none"), s.get("group"))
            for name, s in cfg["arms"].items()}
    side = {name: entry(s["mode"], s.get("scale_mode", "none"), s.get("group"))
            for name, s in cfg["side_arms"].items()}
    return {"values_per_position": vpp, "fp16_bytes_per_position": fp16_bpp,
            "anchors_5_1": ANCHOR, "per_arm": arms, "per_side_arm": side,
            "head_dim_cap": {str(hd_): head_dim_cap(hd_) for hd_ in (32, 64)},
            "head_dim_ladder": head_dim_ladder(),
            "note": ("fp8 免标尺 ⇒ 1.0 B/值 = 6144 B/位置（= 5.1 的解析价签）；"
                     "int8 块标尺 ⇒ 1 + 2/G，G=32 时 1.0625 B/值 = 6528 B/位置。"
                     "head_dim=32 时 G 的上限 = 32 ⇒ 省幅封顶 46.875%。")}
