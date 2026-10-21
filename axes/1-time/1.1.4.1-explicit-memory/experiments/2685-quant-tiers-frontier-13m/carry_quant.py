"""2685 · `CarryChannelBlock` 的**推理期**量化前向（KV 精度 × 状态精度）。

契约：同目录 `README.md`（判据线）与 `config.json`（臂 / 格 / 字节口径在 `analytic`）。

存在理由：`2632/block_arms.py` 的量化前向只定义在 stock `Block` / `GQABlock` 上，
`CarryChannelBlock` 不在它的支持面内 ⇒ 显式状态族此前只能以 fp16 档上图，而纯窗族
已经在它自己最省的 int8 档上（`2680` / `2681`）。本模块补的正是这一格：让 carry 点
与纯窗点落在**同一条每位置部署字节轴**上。

单因素（评测期，0 训练 token，不改权重、不改训练）：

- **KV 精度**：注意力里的 K / V 走 `quantize_kv` 的 `block` 标尺（沿 `head_dim`，G=32）
  往返，与 `2632` 同源同参（`scheme` / `group` / 位宽都由 `config.json::analytic` 给）。
- **状态精度**：`CarryChannel` 的状态 `S ∈ R^{dh×dh}` **在块边界 round-trip 一次**——
  `S_stored = quant(S)`，下一块用的是反量化回来的 `S`。这是「状态也按同一档存储」
  的部署语义：块内（intra-chunk）计算仍是 fp32，只有**跨块存下来的状态**降精度。
  `state.mode="none"` 时不做任何往返 ⇒ 逐位等于已入库仪器（`retention.masked_logits`）。

不改三处共享文件（`pathlm/**`、`train_depth_ar.py`、`tools/**`）：块内数学从
`pathlm.depth_ar` 只读导入（`_carry_cast` / `_carry_chunk_plain` / `_carry_chunk_delta`
是 `carry_scan_chunked` 的同一实现），量化算子从 `tools.evalkit.kv_quant` 只读导入。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import (CarryChannelBlock, _carry_cast, _carry_chunk_delta,
                             _carry_chunk_plain)
from pathlm.model import Block
from tools.evalkit.kv_quant import _merge, _new_stats, quantize_kv

#: 不量化档的显式名字（`quantize_kv` 的 `fp32` 是逐位恒等的空转，这里用 `none` 区分
#: 「本实验没有对这一项做任何往返」，避免读成「量化到了 fp32」）。
NO_QUANT = "none"


def roundtrip(x: torch.Tensor, spec: dict, is_key: bool | None, acc: dict,
              what: str) -> torch.Tensor:
    """按 `spec` 做一次量化往返；`mode="none"` 时恒等返回（不累加活性量）。

    `spec` 来自 `config.json::quant.cells[*]` 的 `kv` / `state` 子对象：
    `{"mode": "none"|"int8"|"int4", "scheme": "block", "group": 32}`。
    """
    mode = spec["mode"]
    if mode == NO_QUANT:
        return x
    if mode not in ("int8", "int4"):
        raise ValueError(f"{what}: 只支持 none / int8 / int4，收到 {mode!r}")
    dq, st = quantize_kv(x, mode, spec["scheme"], int(spec["group"]),
                         int(spec["group"]), is_key)
    _merge(acc, st)
    return dq


def quant_state(S: torch.Tensor, spec: dict, acc: dict) -> torch.Tensor:
    """状态块的存储往返：块边界上把 `S` 存成 `spec.mode` 再读回（`none` 为恒等）。"""
    return roundtrip(S, spec, None, acc, "state")


def carry_scan_quant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     beta: torch.Tensor, rule: str, chunk: int, decay: float,
                     state_spec: dict, acc: dict) -> torch.Tensor:
    """`carry_scan_chunked` 的同序实现，只在块边界插一次状态存储往返。

    `state_spec.mode="none"` 时与 `pathlm.depth_ar.carry_scan_chunked` 逐位相同
    （同 `_carry_cast`、同 `_carry_chunk_*`、同块序）——本文件的两个单测钉住这一条，
    因此「carry fp16 格」可以是新代码对已入库仪器的复现门。

    最后的块之后不再往返：终态没有读者，往返只为下一块服务（少一次无用的量化）。
    """
    q, k, v, beta = (_carry_cast(t) for t in (q, k, v, beta))
    B, H, T, dh = q.shape
    C = min(int(chunk), T)
    S = torch.zeros(B, H, dh, dh, dtype=q.dtype, device=q.device)
    ys = []
    for c0 in range(0, T, C):
        c1 = min(c0 + C, T)
        chunk_fn = _carry_chunk_plain if rule == "plain" else _carry_chunk_delta
        S_out, yc = chunk_fn(q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1],
                             beta[:, :, c0:c1], S, decay)
        ys.append(yc)
        S = quant_state(S_out, state_spec, acc) if c1 < T else S_out
    return torch.cat(ys, dim=2)


def carry_mem_quant(channel, h0: torch.Tensor, state_spec: dict,
                    acc: dict) -> torch.Tensor:
    """`CarryChannel.write_read` 的同序实现：写入投影 + 状态精度可换的扫描 + 读出。"""
    B, T, d = h0.shape
    hf = h0.float()

    def shape(t):
        return t.view(B, T, channel.n_heads, channel.head_dim).transpose(1, 2)

    kk = shape(hf @ channel.w_k.float().T)
    vv = shape(hf @ channel.w_v.float().T)
    qq = shape(hf @ channel.w_q.float().T)
    beta = torch.sigmoid(
        torch.einsum("btd,hd->bth", hf, channel.beta_w.float())
        + channel.beta_b.float()).transpose(1, 2).contiguous()
    mem = carry_scan_quant(qq, kk, vv, beta, channel.rule, channel.chunk,
                           channel.decay, state_spec, acc)
    return mem.transpose(1, 2).reshape(B, T, d).to(h0.dtype)


def _carry_block_forward(blk: CarryChannelBlock, x: torch.Tensor, mem: torch.Tensor,
                         kv_spec: dict, acc: dict, attn_mask) -> torch.Tensor:
    """`CarryChannelBlock.forward` 的 dense 分支（无 `kv_from` / `gate_vec` / `dist_pen`），
    K/V 走 `kv_spec` 的往返、读出与块内数学逐行照抄原实现。"""
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)  # noqa: E731
    qs = shape(q)
    ks = roundtrip(shape(k), kv_spec, True, acc, "kv")
    vs = roundtrip(shape(v), kv_spec, False, acc, "kv")
    a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                       is_causal=attn_mask is None)
    gain = blk.carry_scale.float().repeat_interleave(blk.carry_head_dim).view(1, 1, d)
    a = a.transpose(1, 2).reshape(B, T, d) + (mem * gain).to(a.dtype)
    x = x + blk.proj(a)
    x = x + blk.mlp(blk.ln2(x))
    return x


def _stock_block_forward(blk: Block, x: torch.Tensor, kv_spec: dict, acc: dict,
                         attn_mask) -> torch.Tensor:
    """stock `Block` 的 dense 分支（carry 臂与纯窗臂若要跑同一入口时用；
    纯窗臂在 `run.py` 里直接调 `2632` 的仪器，不经过这里）。"""
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)  # noqa: E731
    qs = shape(q)
    ks = roundtrip(shape(k), kv_spec, True, acc, "kv")
    vs = roundtrip(shape(v), kv_spec, False, acc, "kv")
    a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                       is_causal=attn_mask is None)
    a = a.transpose(1, 2).reshape(B, T, d)
    x = x + blk.proj(a)
    x = x + blk.mlp(blk.ln2(x))
    return x


def num_state_values(model) -> int:
    """这批臂的显式状态一共多少个数值（部署字节的分母）。

    逐层模式：`n_layers × 每层 H·dh·dh`；单共享槽：`1 × H·dh·dh`。两处都用块的
    `carry_heads` / `carry_head_dim` 实读（不是配置抄写）。
    """
    blocks = [b for b in model.blocks if isinstance(b, CarryChannelBlock)]
    if not blocks:
        return 0
    per = blocks[0].carry_heads * blocks[0].carry_head_dim * blocks[0].carry_head_dim
    return per * (len(blocks) if getattr(blocks[0], "_carry_per_layer", False) else 1)


@torch.no_grad()
def quant_carry_logits(model, tokens: torch.Tensor, keep_mask, kv_spec: dict,
                       state_spec: dict):
    """carry 主干在全深度前向下的 logits `[B,T,V]`（`keep_mask` 给可见集，None = 全因果）。

    `kv_spec.mode="none"` 且 `state_spec.mode="none"` 时与 `retention.masked_logits`
    逐位相等（单测 + `run.py` 的复现门钉住），因此这条入口的忠实性可被测。
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
    shared = None
    if model.cfg.carry_rule is not None and not model.cfg.carry_per_layer:
        shared = carry_mem_quant(model.carry, model.carry.gate(h), state_spec, acc)
    for blk in model.blocks:
        if isinstance(blk, CarryChannelBlock):
            if blk._carry_per_layer:
                mem = carry_mem_quant(blk._carry_channel, blk._carry_channel.gate(h),
                                      state_spec, acc)
            else:
                mem = shared
            h = _carry_block_forward(blk, h, mem, kv_spec, acc, attn_mask)
        elif type(blk) is Block:
            h = _stock_block_forward(blk, h, kv_spec, acc, attn_mask)
        else:
            raise ValueError(f"carry 量化前向只定义在 CarryChannelBlock / Block 上，"
                             f"遇到 {type(blk).__name__}")
    return model.readout(h), acc


def make_logits_fn(model, keep_mask, kv_spec: dict, state_spec: dict):
    """把一格 (KV 精度, 状态精度) 变成 `tokens -> logits`（与 `2680`/`2681` 同接口）。"""

    def fn(x, _m=model, _k=keep_mask, _kv=kv_spec, _st=state_spec):
        return quant_carry_logits(_m, x, _k, _kv, _st)[0]
    return fn
