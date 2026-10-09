"""评测期 KV cache 量化参照线（ticket 31 契约 K）——不改权重、不训练。

契约：`axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/README.md`

这件工具回答一个问题：**把 KV cache 存成 int8 / int4，质量掉多少、内存省多少。**
它就是参照线——**任何「省内存 / 丢信息」的主张都必须先打败这条线**。

实现口径（评测期，单因素 = KV cache 的数值精度）：

- 只改注意力里 **K / V 两个张量**的存储精度（query 保持 fp32），
  不碰权重、不训练、不新增任何参数（`params_added = 0`）。
- 三档 + 一个参考档：
  - `fp32`：不做任何处理（**释放版前向**，用于实现等值自检，逐位一致）；
  - `fp16`：K/V 存成 fp16 再读回（**基线**，工业默认 KV cache 精度）；
  - `int8` / `int4`：**对称 per-channel** 量化（见下）。
- **per-channel 定义**：注意力里 K/V 的布局是 `[B, H, T, head_dim]`
  （`pathlm.model.Block` 的 `transpose(1, 2)`）。通道轴 = `head_dim`，
  **每个 `(batch, head, token, channel)` 一个 scale**（见下「因果/流式口径」）。
- **对称**：零点固定为 0，`qmax = 2^(bits-1) - 1`（int8 → 127，int4 → 7），
  `scale = amax / qmax`，`q = clamp(round(x/scale), -qmax, qmax)`，
  反量化 `x̂ = q · scale`。往返误差上界 `|x̂ - x| ≤ 0.5 · scale`（取整误差，
  running-max 不越界，不触发额外的 clip 误差）；单测钉住。

**因果/流式口径（2026-10-02 修复 A2；旧实现是缺陷）**：
旧实现沿 `batch + token` 轴对**全序列**取 amax ⇒ scale 含未来 token 与同批其他请求的
信息，过去位置的反量化值会随未来 token 或同批邻居改变（`is_causal=True` 挡不住）。
现在的实现逐请求、因果、流式：位置 `t` 的 scale 只由**同一请求 ≤ t 的位置**决定
（对 token 轴取 running max，`cummax`），`batch` 轴不参与归约。三条性质明写如下：

| 性质 | 本实现 |
|---|---|
| 旧值是否重编码 | **否**。位置 `t` 的 scale 在写入时定为 `max_{t'≤t}|x_{t'}|/qmax`，写入后不再改动；没有分块/回填重编码（因此没有「等待整块到齐再回头重写」的窗口） |
| scale 存储开销 | **每值一个 fp16 scale**（`[B,H,T,head_dim]` 与 K/V 元素一一对应）⇒ 每值 +2 字节：int8 净 **3 B/值**、int4 净 **2.5 B/值**（fp16 基线 2 B/值）。**这是因果性的代价**：严格逐 token 因果 ⇒ scale 逐 token 变 ⇒ 无法摊薄。`kv_cache_footprint` 的 `scale_bytes_streaming_per_position` / `*_streaming_bytes_by_seq_len` 按此口径给数；`per_position_bytes` / `total_bytes_by_seq_len` 仍是**值字节**旧口径（与已入账 results.json 逐位兼容），部署价签须用 streaming 口径重算（R1） |
| 是否有残余高精度窗口 | **没有**。所有 K/V 位置都走量化往返，不额外保留 fp16 尾窗/锚点；本文件的组合入口（窗口 / 锚点 / 头共享）只改**可见位置掩码**与头数，量化算子同源 |

口径提醒：真要「省内存」的部署设计需用**分块 scale + 有界重编码窗口**（每 `G` 个 token 摊一个
scale，块关闭时回填一次），代价是块内过去值会随同块后续 token 变——本参照线**不**采该口径，
因为评测要的是「过去位置严格前缀确定」的可比价签。

**标尺方案（`SCALE_SCHEMES`，`quantize_kv(scheme=...)`）**：上表里的「每值 +2 字节标尺」是
**逐值标尺**（`per_value`）的价钱——严格前缀确定的**最贵**存法。它可以更便宜而不损伤因果：
把标尺沿**非 token 轴**（`head_dim`）摊薄（`block`：每 `group` 个相邻通道共享一把 ⇒ `2/group`
B/值，`group = head_dim` 时 int8 净 **1.0625 B/值**），或 KIVI 式（`kivi`：K per-channel 沿
token 分组、V per-token）。`per_value` 保留为**对照臂**。沿 token 轴摊薄（`kivi` 的 K 侧 /
块式 token 分组）只能用「块内过去值可被改写」换，违反前缀确定 ≤ `group − 1` 个位置。

前向实现：按 `DepthARModel.forward` 的 **dense 分支**逐行复刻
（`_input_states` → 逐层 `Block` → tied fp32 读出），只在 `Block` 的注意力
之前对 K/V 做一次量化-反量化往返。`fp32` 档必须与释放版前向**逐位相等**
（单测 + `run.py` 仪器自检），因此这不是「另一个模型」。

活性量（SCHEMA §10）：每个配置落 `n_values / mean_abs_err / rel_err /
max_abs_err / changed_frac`，证明「量化真的改了数值、且降了精度」，
而不是只改了个标签。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.model import Block

LN2 = 0.6931471805599453  # nats → bits

KV_MODES = ("fp32", "fp16", "int8", "int4")
_INT_BITS = {"int8": 8, "int4": 4}
_BYTES_PER_VALUE = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}

#: 标尺（scale）方案。标尺是每值存储精度的**元数据**；它存哪、摊多薄决定量化能不能省内存。
#:
#: - `per_value`：位置 t 的标尺 = 同请求 ≤ t 的 running max / qmax（逐值一个 fp16 标尺）。
#:   严格前缀确定，但**标尺 2 B/值** ⇒ int8 净 3 B/值（比 fp16 的 2 B/值更费）。本文件原实现。
#: - `block`：沿 `head_dim` 每 `group` 个**相邻**值共享一把标尺（`amax/qmax`）。标尺只由当前
#:   token 自己的值决定 ⇒ **严格前缀确定**，且摊薄到 `2/group` B/值（`group = head_dim` 时
#:   即 per-token 标尺，int8 = 1 + 2/32 = 1.0625 B/值）。这是 llama.cpp `q8_0` 式块量化。
#: - `kivi`：KIVI 式 = **K per-channel**（token 轴每 `kivi_group` 个位置共享一把通道标尺）、
#:   **V per-token**（每 token 沿 `head_dim` 一把标尺）。K 侧沿 token 分组 ⇒ 块内过去值会随
#:   同组后续位置改写（违反前缀确定 ≤ `kivi_group − 1` 个位置）；V 侧严格因果。
#: - `block_tok`：沿 **token** 轴每 `group` 个相邻位置共享一把标尺（每通道一把，K/V 同）。
#:   这就是 `2631/budget_caliber` 里只做过**解析**的方案 (b)；块内不前缀确定（≤ `group − 1`）。
SCALE_SCHEMES = ("per_value", "block", "block_tok", "kivi")

#: 每把标尺的存储字节数（fp16 标尺 = 2 B；与 `2631/budget_caliber.SCALE_BYTES` 同口径）。
SCALE_BYTES = 2.0


# --------------------------------------------------------------- 量化 / 反量化


def _new_stats() -> dict:
    return {"n_values": 0, "sum_abs_err": 0.0, "max_abs_err": 0.0,
            "sum_abs_x": 0.0, "n_changed": 0, "n_clipped": 0,
            "scale_min": None, "scale_max": None}


def _require_group(group: int, span: int, axis: str) -> None:
    """块标尺必须整除它所在的那条轴（否则块不整齐，每值摊到的标尺字节不是 `2/group`）。"""
    if group < 1 or span % group:
        raise ValueError(f"{axis} 块标尺要求 group 整除 {axis}={span}，收到 group={group}")


def _head_dim_block_quant(x: torch.Tensor, qmax: int, group: int):
    """沿 `head_dim`（最后一维）每 `group` 个相邻值共享一把标尺。

    `x` 布局 `[..., H, T, head_dim]`：标尺只对该 token 的 `group` 个通道取 amax，
    **token 轴不参与归约** ⇒ 严格前缀确定。返回 `(dq, scale, ratio)`，后两者在
    块视图 `[..., H, T, head_dim/group, group]` 里（供活性量统计用）。
    """
    hd = x.shape[-1]
    _require_group(group, hd, "head_dim")
    xr = x.reshape(*x.shape[:-1], hd // group, group)
    s = (xr.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)
    ratio = xr / s
    q = torch.round(ratio).clamp(-qmax, qmax)
    return (q * s).reshape(x.shape), s, ratio


def _token_block_quant(x: torch.Tensor, qmax: int, group: int):
    """沿 **token** 轴每 `group` 个相邻位置共享一把标尺（每通道一把，= KIVI 的 K 侧）。

    `x` 布局 `[..., H, T, head_dim]`：标尺 = 该 token 块在**同一通道**上的 amax。
    块内过去值会被同块（最多 `group − 1` 个）后续位置改写 ⇒ 违反前缀确定。
    """
    T = x.shape[-2]
    hd = x.shape[-1]
    _require_group(group, T, "T")
    xr = x.reshape(*x.shape[:-2], T // group, group, hd)
    s = (xr.abs().amax(dim=-2, keepdim=True) / qmax).clamp_min(1e-12)
    ratio = xr / s
    q = torch.round(ratio).clamp(-qmax, qmax)
    return (q * s).reshape(x.shape), s, ratio


def _int_quantize_schemed(x: torch.Tensor, qmax: int, scheme: str,
                          group: int, kivi_group: int, is_key: bool | None):
    """按 `scheme` 选标尺轴，做一次对称取整往返。返回 `(dq, scale, ratio)`。

    `dq` 与 `x` 同形；`scale` / `ratio` 在各自的块视图里（`ratio = x/scale`，
    供 `n_clipped` 统计）。`kivi` 需要 `is_key`：K 走 token 块（per-channel）、
    V 走 `head_dim` 整块（per-token）。
    """
    if scheme == "per_value":
        # 因果/流式：token 轴 = -2，只对 **该位置及其之前** 的 token 取 running max
        # （`cummax`），batch 轴不参与归约（逐请求）。不重编码、无 fp16 尾窗。
        scale = (x.abs().cummax(dim=x.dim() - 2).values / qmax).clamp_min(1e-12)
        ratio = x / scale
        q = torch.round(ratio).clamp(-qmax, qmax)
        return q * scale, scale, ratio
    if scheme == "block":
        return _head_dim_block_quant(x, qmax, group)
    if scheme == "block_tok":
        # 沿 token 轴分组（= 2631/budget_caliber 解析里的方案 (b)），块内不前缀确定。
        return _token_block_quant(x, qmax, group)
    if scheme == "kivi":
        if is_key is None:
            raise ValueError("kivi 标尺需要 is_key 区分 K（per-channel）/ V（per-token）")
        if is_key:
            return _token_block_quant(x, qmax, kivi_group)
        # V per-token：整条 head_dim 共享一把标尺（= block，group = head_dim）。
        return _head_dim_block_quant(x, qmax, x.shape[-1])
    raise ValueError(f"unknown scale scheme {scheme!r}; expected one of {SCALE_SCHEMES}")


def scheme_scale_bytes_per_value(mode: str, scheme: str, head_dim: int,
                                 group: int = 32, kivi_group: int = 32) -> float:
    """每值摊到的标尺字节数（解析口径）——量化能不能省内存全看这一个数字。

    - `per_value`：每值一把 fp16 标尺 ⇒ 2.0（严格前缀确定的最贵存法）。
    - `block`：每 `group` 个相邻 `head_dim` 值一把 ⇒ `2/group`（需 `group | head_dim`）。
    - `kivi`：K 侧 `2/kivi_group`（沿 token 分组）、V 侧 `2/head_dim`（沿通道）；半边各占
      一半值 ⇒ 取两者的平均。

    非量化档（fp32 / fp16）没有标尺 ⇒ 0。
    """
    if mode not in _BYTES_PER_VALUE and mode != "fp32":
        raise ValueError(f"unknown mode {mode!r}")
    if scheme not in SCALE_SCHEMES:
        raise ValueError(f"unknown scale scheme {scheme!r}; expected one of {SCALE_SCHEMES}")
    if mode in ("fp32", "fp16"):
        return 0.0
    if scheme == "per_value":
        return SCALE_BYTES
    if scheme == "block":
        _require_group(group, head_dim, "head_dim")
        return SCALE_BYTES / group
    if scheme == "block_tok":
        return SCALE_BYTES / group
    if kivi_group < 1 or head_dim < 1:
        raise ValueError(f"kivi 参数非法：kivi_group={kivi_group}, head_dim={head_dim}")
    return (SCALE_BYTES / kivi_group + SCALE_BYTES / head_dim) / 2.0


def _merge(dst: dict, src: dict) -> None:
    for k in ("n_values", "n_changed", "n_clipped"):
        dst[k] += src[k]
    for k in ("sum_abs_err", "sum_abs_x"):
        dst[k] += src[k]
    dst["max_abs_err"] = max(dst["max_abs_err"], src["max_abs_err"])
    for k in ("scale_min", "scale_max"):
        if src[k] is not None:
            dst[k] = src[k] if dst[k] is None else (
                min(dst[k], src[k]) if k == "scale_min" else max(dst[k], src[k]))


def summarize_stats(acc: dict) -> dict:
    """把累加器折成活性量（SCHEMA §10）。`rel_err` = Σ|err| / Σ|x|（相对），
    `changed_frac` = 被量化改变（含反量化后不等）的数值占比。"""
    n = max(acc["n_values"], 1)
    out = {
        "n_values": acc["n_values"],
        "mean_abs_err": round(acc["sum_abs_err"] / n, 9),
        "max_abs_err": round(acc["max_abs_err"], 9),
        "rel_err": round(acc["sum_abs_err"] / max(acc["sum_abs_x"], 1e-12), 9),
        "changed_frac": round(acc["n_changed"] / n, 9),
        "clipped_frac": round(acc["n_clipped"] / n, 9),
    }
    if acc["scale_min"] is not None:
        out["scale_min"] = acc["scale_min"]
        out["scale_max"] = acc["scale_max"]
    return out


@torch.no_grad()
def quantize_kv(x: torch.Tensor, mode: str, scheme: str = "per_value",
                group: int = 32, kivi_group: int = 32, is_key: bool | None = None):
    """评测期 KV cache 往返：`x` (fp32, 布局 `[..., H, T, head_dim]`) →
    存储（`mode` 精度）→ 反量化回 fp32。

    `scheme` 选标尺的共享方案（见 `SCALE_SCHEMES`）：默认 `"per_value"`（逐值标尺，本文件
    原实现、与既有 `2619/2620/2621/2631` 逐位一致）；`"block"` = 沿 `head_dim` 每 `group`
    个值共享一把（严格前缀确定且摊薄）；`"kivi"` = K per-channel（`kivi_group` 个 token
    一把）/ V per-token（需 `is_key` 区分）。

    返回 `(dequantized, stats)`；`stats` 用 `summarize_stats` 折成活性量。
    `"fp32"` 档是逐位恒等（不改数值）；`"fp16"` 档没有标尺（`scheme` 必须为默认值）。
    """
    if mode not in KV_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {KV_MODES}")
    acc = _new_stats()
    n = x.numel()
    if mode == "fp32":
        return x, acc
    if mode == "fp16":
        if scheme != "per_value":
            raise ValueError(f"fp16 档没有标尺，scheme 必须为 'per_value'，收到 {scheme!r}")
        dq = x.to(torch.float16).to(x.dtype)
    else:
        bits = _INT_BITS[mode]
        qmax = 2 ** (bits - 1) - 1
        if x.dim() < 3:
            raise ValueError(
                f"KV 布局至少 3 维 [..., H, T, head_dim]，收到 {tuple(x.shape)}")
        dq, scale, ratio = _int_quantize_schemed(x, qmax, scheme, group, kivi_group, is_key)
        acc["n_clipped"] = int((ratio.abs() > qmax).sum().item())
        acc["scale_min"] = float(scale.min())
        acc["scale_max"] = float(scale.max())
    err = (dq.float() - x.float()).abs()
    acc["n_values"] = n
    acc["sum_abs_err"] = float(err.sum())
    acc["max_abs_err"] = float(err.max()) if n else 0.0
    acc["sum_abs_x"] = float(x.abs().sum())
    acc["n_changed"] = int((dq != x).sum())
    return dq, acc


# --------------------------------------------------------------- 前向（dense 复刻）


def _block_forward_quant(blk: Block, x: torch.Tensor, mode: str,
                         acc: dict, scheme: str = "per_value", group: int = 32,
                         kivi_group: int = 32) -> torch.Tensor:
    """`pathlm.model.Block.forward` 的 dense 注意力分支（`attn_mask=None`、
    `dist_pen=0`、`kv_from=None`、`gate_vec=None`），在 SDPA 之前对 K/V 做
    量化-反量化往返。`scheme` / `group` / `kivi_group` 透传给 `quantize_kv`。"""
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)

    def shape(t):
        return t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)

    qs, ks, vs = shape(q), shape(k), shape(v)
    ks, ki = quantize_kv(ks, mode, scheme, group, kivi_group, is_key=True)
    _merge(acc, ki)
    vs, vi = quantize_kv(vs, mode, scheme, group, kivi_group, is_key=False)
    _merge(acc, vi)
    a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
    a = a.transpose(1, 2).reshape(B, T, d)
    x = x + blk.proj(a)
    x = x + blk.mlp(blk.ln2(x))
    return x


@torch.no_grad()
def kv_quant_logits(model, tokens: torch.Tensor, mode: str, scheme: str = "per_value",
                    group: int = 32, kivi_group: int = 32):
    """dense 主干在全深度前向下的 logits `[B,T,V]`，KV 走 `mode` 精度的往返。

    返回 `(logits, stats)`。与 `DepthARModel.forward` 的 dense 分支逐位一致
    （`mode="fp32"` 时，单测钉住）。只接受 stock `Block` 层——内容 top-k 等
    自定义层会让「唯一因素 = KV 精度」不成立。`scheme` 见 `SCALE_SCHEMES`。
    """
    h = model._input_states(tokens)
    acc = _new_stats()
    for blk in model.blocks:
        if type(blk) is not Block:
            raise ValueError(
                f"KV 量化参照线只定义在 stock dense 注意力上，遇到 {type(blk).__name__}")
        h = _block_forward_quant(blk, h, mode, acc, scheme, group, kivi_group)
    return model.readout(h), acc


@torch.no_grad()
def release_logits(model, tokens: torch.Tensor) -> torch.Tensor:
    """释放版 dense 前向的最终深度 logits——实现等值（§9）的对照锚。"""
    _, aux = model(tokens, exit_depths=None)
    return aux["depth_logits"][-1]


@torch.no_grad()
def kv_quant_bpc(model, eval_arr, mode: str, n_batches: int = 24,
                 batch_size: int = 16, window_seed: int = 2,
                 device: str = "cuda", on_batch=None):
    """clean bpc（越低越好）：因果移位 CE 的 token 平均，K/V 走 `mode` 精度。

    窗口采样与 `pathlm.depth_ar.depth_curve` / 释放版基线同一顺序
    （同一 `window_seed`、同一 `pathlm.data.batch`）⇒ **同一数据流、同一位置集**。

    `on_batch(i, stats, bpc_so_far)` 可选回调：用于**周期性落盘活性量**。
    返回 `(bpc, stats)`，`stats` 是全部 batch 汇总后的活性量。
    """
    from pathlm.data import batch

    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats, n_tok = 0.0, 0
    acc = _new_stats()
    model.eval()
    for i in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits, bstats = kv_quant_logits(model, x, mode)
        logits = logits[:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                tgt.reshape(-1), reduction="sum").item()
        n_tok += tgt.numel()
        _merge(acc, bstats)
        if on_batch is not None:
            on_batch(i, summarize_stats(bstats), nats / n_tok / LN2)
    return nats / n_tok / LN2, summarize_stats(acc)


@torch.no_grad()
def release_bpc(model, eval_arr, n_batches: int = 24, batch_size: int = 16,
                window_seed: int = 2, device: str = "cuda") -> float:
    """释放版 dense 前向的 clean bpc（`is_causal=True` 路径）——与 `fp32` 档
    相等（仪器自检），也与 2610 `depth_curve` 的锚 bpc 同序。"""
    from pathlm.data import batch

    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats, n_tok = 0.0, 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits = release_logits(model, x)[:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                tgt.reshape(-1), reduction="sum").item()
        n_tok += tgt.numel()
    return nats / n_tok / LN2


# --------------------------------------------------------------- 内存口径


def kv_cache_footprint(cfg, mode: str, seq_lens=(256, 4096, 131072)) -> dict:
    """KV cache 的**解析**占用（部署关心的量；评测期实现会立刻反量化，
    所以 GPU 实测峰值不会下降——见 README §7/§8 的口径标注）。

    `per_position_bytes` = 2(K,V) × n_layers × n_heads × head_dim × 每值字节
    （**只算值**，与已入账 results.json 的旧口径逐位兼容）。

    scale 开销给两套：
    - `scale_bytes_total`：旧的**非因果**口径（每 `(head, channel)` 一个静态 fp16
      scale、与序列长度无关），仅为兼容已入账结果保留；
    - `scale_bytes_streaming_per_position` + `*_streaming_bytes_by_seq_len`：
      本文件因果/流式实现的真实开销（每值一个 fp16 scale）；**部署价签用这一套**。
    """
    if mode not in _BYTES_PER_VALUE and mode != "fp32":
        raise ValueError(f"unknown mode {mode!r}")
    n_layers, n_heads = cfg.n_layers, cfg.n_heads
    hd = cfg.d_model // n_heads
    vals_per_pos = 2 * n_layers * n_heads * hd            # K + V 的标量数/位置
    bpe = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}[mode]
    per_pos = vals_per_pos * bpe
    scale_bytes = 0 if mode in ("fp32", "fp16") else 2 * n_layers * n_heads * hd * 2
    streaming = mode not in ("fp32", "fp16")
    # 因果/流式：每个标量值一个 fp16 scale（token 轴一变，scale 就变，无法摊薄）。
    scale_stream_per_pos = 2.0 * vals_per_pos if streaming else 0.0
    return {"mode": mode, "dtype_bytes_per_value": bpe,
            "n_layers": n_layers, "n_heads": n_heads, "head_dim": hd,
            "values_per_position": vals_per_pos,
            "per_position_bytes": per_pos,
            "scale_bytes_total": scale_bytes,
            "scale_bytes_streaming_per_position": scale_stream_per_pos,
            "streaming_bytes_per_position": per_pos + scale_stream_per_pos,
            "total_bytes_by_seq_len": {str(t): per_pos * t + scale_bytes
                                       for t in seq_lens},
            "streaming_bytes_by_seq_len": {
                str(t): (per_pos + scale_stream_per_pos) * t for t in seq_lens}}


def cache_reduction(base: dict, other: dict) -> dict:
    """相对基线的内存下降（`other` 相对 `base`）。"""
    b = base["per_position_bytes"]
    o = other["per_position_bytes"]
    return {"mode": other["mode"], "base_mode": base["mode"],
            "per_position_bytes": o, "base_per_position_bytes": b,
            "reduction_frac": round(1 - o / b, 6),
            "ratio": round(b / o, 4)}


def scheme_footprint(cfg, mode: str, scheme: str, group: int = 32,
                     kivi_group: int = 32, seq_lens=(256,)) -> dict:
    """**部署口径**的 KV 占用（值 + 标尺），按标尺方案逐项算清。

    `per_position_bytes` = `2(K,V)·L·H·head_dim·(每值字节)` 只算值；
    `deployment_bytes_per_position` = 值 + 标尺（标尺字节由
    `scheme_scale_bytes_per_value` 给，与 `quantize_kv` 的实际标尺轴一一对应）。
    这个函数就是「不得再把标尺当免费」的那个计算。
    """
    if mode not in _BYTES_PER_VALUE and mode != "fp32":
        raise ValueError(f"unknown mode {mode!r}")
    n_layers, n_heads = cfg.n_layers, cfg.n_heads
    hd = cfg.d_model // n_heads
    vals_per_pos = 2 * n_layers * n_heads * hd
    bpe = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int4": 0.5}[mode]
    scale_bpe = scheme_scale_bytes_per_value(mode, scheme, hd, group, kivi_group)
    value_bytes = vals_per_pos * bpe
    scale_bytes = vals_per_pos * scale_bpe
    total = value_bytes + scale_bytes
    fp16_total = kv_cache_footprint(cfg, "fp16", seq_lens)  # 基线（无标尺）
    return {"mode": mode, "scheme": scheme, "group": group, "kivi_group": kivi_group,
            "head_dim": hd, "values_per_position": vals_per_pos,
            "value_bytes_per_position": value_bytes,
            "scale_bytes_per_value": scale_bpe,
            "scale_bytes_per_position": scale_bytes,
            "per_position_bytes": total,
            "fp16_value_bytes_per_position": fp16_total["per_position_bytes"],
            "reduction_vs_fp16": round(1 - total / fp16_total["per_position_bytes"], 6),
            "ratio_vs_value_caliber": round(total / value_bytes, 6),
            "total_bytes_by_seq_len": {str(t): total * t for t in seq_lens}}
