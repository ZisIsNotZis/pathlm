"""2624 · 评测期 KV 量化的「carry × {MHA,GQA}」前向 —— 本实验唯一的新语义层。

契约：同目录 `README.md`（§4 判据线逐字抄录 Z3；§4b 记录 DESIGN 裁决）。
本模块回答一件事：**对一个「delta 写通道 + 头共享（GQA）」已被训练用起来的模型，
评测期把注意力 KV cache 降成 int8，召回会不会崩？** 同时把 MHA 基线放在**同一件
量化仪器**下，好让「头共享」与「int8」两个因素各自有干净的对照。

实现口径（SCHEMA §22 第 7 条：新逻辑进新模块，`pathlm/` 与 `tools/evalkit/` 只读）：

- 量化算子**复用** `tools/evalkit/kv_quant.py::quantize_kv`（对称 per-channel：
  通道轴 = `head_dim`、`(head, channel)` 一个 fp16 scale、沿 batch/token 取 amax，
  `qmax = 2^(bits-1)-1`）⇒ 与 2614/2619/2620/2621 是**同一件仪器**。本模块只把它的
  调用点接到两种**带写入通道**的块上：
    * `CarryChannelBlock`（MHA + carry，2618/W3 的块）——与 2621 `carry_quant.py`
      同一接线；本模块在此块上与它逐位一致（单测钉住）；
    * `ComposableBlock`（GQA × carry，ENG1 后才有）——2621 的模块**显式拒绝**该块
      （它只定义在 `CarryChannelBlock` 上），故需要这一层新接线：逐行复刻
      `ComposableBlock.forward`（GQA 头投影 `enable_gqa` + carry 读出 + 正交
      `attn_mask`），唯一的差别 = SDPA 之前对 K/V 做一次量化往返。
- `mode="fp32"` 必须与释放版前向（`retention.masked_logits`）**逐位相等**（单测 +
  `run.py` 仪器自检钉住，两种块类各一条）⇒ 这不是「另一个模型」，只是给同一张图
  加了存储精度这一维。

活性量（SCHEMA §10）：量化侧落 `n_values / mean_abs_err / rel_err / max_abs_err /
changed_frac`；通道侧落 `read_share = ‖γ⊙mem‖ / ‖a‖`（与训练日志每 200 步同一量）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import CarryChannelBlock, ComposableBlock
from tools.evalkit.kv_quant import (_merge, _new_stats, quantize_kv,
                                    summarize_stats)
from tools.evalkit.longrange_tasks import _summarize

#: 本实验的 KV 存储精度档。`fp32` = 释放版（实现等值锚），`fp16` = 工业默认
#: 全精度基线（与 2614/2619/2620/2621 的基线同口径），`int8` = 压缩档。
QUANT_MODES = ("fp32", "fp16", "int8")

#: 本模块支持的块类（都带写入通道）。
CARRY_BLOCK_TYPES = (CarryChannelBlock, ComposableBlock)


def _attn_mask_of(keep_mask, B: int, n_heads: int, device):
    """把 `[T,T]` / `[B,T,T]` 的可见集掩码 expand 成 SDPA 的 `[B,H,T,T]` bool。

    与 `tools/evalkit/retention.masked_logits` 同款（CUDA SDPA 内核不广播 bool
    掩码的 head 维），保证「训练/评测同一件仪器」的可见性口径不变。
    """
    if keep_mask is None:
        return None
    km = keep_mask.to(device)
    T = km.shape[-1]
    if km.shape == (T, T):
        return km.view(1, 1, T, T).expand(B, n_heads, T, T)
    if km.shape == (B, T, T):
        return km.view(B, 1, T, T).expand(B, n_heads, T, T)
    raise ValueError(f"keep_mask 必须是 [T,T] 或 [B,T,T]，收到 {tuple(km.shape)}")


def _quant_carry_block(blk, x: torch.Tensor, mode: str, attn_mask, acc: dict | None):
    """`CarryChannelBlock.forward` / `ComposableBlock.forward` 的逐行复刻，唯一的
    差别 = SDPA 之前对 K/V 做一次 `quantize_kv` 往返。`acc` 非空时累加量化活性量。"""
    is_composable = isinstance(blk, ComposableBlock)
    enable_gqa = bool(getattr(blk, "enable_gqa", False))
    carry_on = (blk.carry_on if is_composable else True)
    mem = blk._carry_box.get("mem") if carry_on else None
    if carry_on and mem is None:
        raise RuntimeError(
            "quant_carry 需要先经 `DepthARModel._input_states` 算好单共享记忆槽")
    B, T, d = x.shape
    head_dim = d // blk.n_heads
    qkv = blk.qkv(blk.ln1(x))

    def shape(t: torch.Tensor, nh: int) -> torch.Tensor:
        return t.view(B, T, nh, head_dim).transpose(1, 2)

    if not enable_gqa:
        q, k, v = qkv.chunk(3, dim=-1)
        qs, ks, vs = shape(q, blk.n_heads), shape(k, blk.n_heads), shape(v, blk.n_heads)
    else:
        q, kv = qkv[..., :d], qkv[..., d:]
        k, v = kv[..., :blk.kv_dim], kv[..., blk.kv_dim:]
        qs = shape(q, blk.n_heads)
        ks, vs = shape(k, blk.n_kv_heads), shape(v, blk.n_kv_heads)
    ks, ki = quantize_kv(ks, mode)
    if acc is not None:
        _merge(acc, ki)
    vs, vi = quantize_kv(vs, mode)
    if acc is not None:
        _merge(acc, vi)
    a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                       is_causal=attn_mask is None,
                                       enable_gqa=enable_gqa)
    a = a.transpose(1, 2).reshape(B, T, d)
    read_share = 0.0
    if carry_on:
        gain = blk.carry_scale.float().repeat_interleave(
            blk.carry_head_dim).view(1, 1, d)
        scaled = mem * gain
        a = a + scaled.to(a.dtype)
        # 与训练日志每 200 步落盘的 read_share 同一表达式（分母含读出本身）
        read_share = float(scaled.detach().float().norm()
                           / a.detach().float().norm().clamp_min(1e-6))
    x = x + blk.proj(a)
    x = x + blk.mlp(blk.ln2(x))
    return x, read_share


@torch.no_grad()
def quant_carry_logits(model, tokens: torch.Tensor, mode: str,
                       keep_mask: torch.Tensor | None = None,
                       acc: dict | None = None):
    """dense 主干在全深度前向下的 logits `[B,T,V]`，注意力 K/V 走 `mode` 精度。

    - **只接受带写入通道的块**（`CarryChannelBlock` / `ComposableBlock`）：其它块类
      会让「唯一因素 = KV 精度」不成立 ⇒ 直接报错；
    - `keep_mask=None` 时走释放版的 `is_causal=True` 路径；传
      `window_anchor_keep_mask(T, train_window, train_anchors)` 即与训练/评测同一掩码；
    - `acc` 非空时累加量化活性量（`_new_stats` 累加器，用 `summarize_stats` 折）。
    返回 `(logits, tele)`，`tele["read_share_by_layer"]` 是每层的通道读出占比。
    """
    if mode not in QUANT_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {QUANT_MODES}")
    if model.cfg.carry_rule is None:
        raise ValueError("quant_carry 只定义在带写入通道（carry_rule）的模型上")
    bad = [type(blk).__name__ for blk in model.blocks
           if not isinstance(blk, CARRY_BLOCK_TYPES)]
    if bad:
        raise ValueError(
            "quant_carry 只定义在 CarryChannelBlock / ComposableBlock 上，遇到 "
            + ", ".join(sorted(set(bad))))
    B = tokens.shape[0]
    attn_mask = _attn_mask_of(keep_mask, B, model.cfg.n_heads, tokens.device)
    h = model._input_states(tokens)            # 算好单共享槽并放进 `_carry_box`
    reads: list[float] = []
    for blk in model.blocks:
        h, rs = _quant_carry_block(blk, h, mode, attn_mask, acc)
        reads.append(rs)
    return model.readout(h), {"read_share_by_layer": reads,
                              "read_share": round(sum(reads) / len(reads), 6)}


# --------------------------------------------------------------- 评测腿


@torch.no_grad()
def quant_accuracy(model, task, mode: str, keep_mask: torch.Tensor | None = None,
                   device=None, max_chunk: int = 64, acc: dict | None = None) -> dict:
    """模型在任务上的 top-1 准确率 + 距离曲线，K/V 走 `mode` 精度。

    与 `tools.evalkit.longrange_tasks.accuracy` **同一算法**（同分块、同 gather、
    同 `_summarize`），只把「模型 → logits」换成 `quant_carry_logits`。
    """
    dev = device if device is not None else next(model.parameters()).device
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        if keep_mask is None:
            km = None
        elif keep_mask.dim() == 3:
            km = keep_mask[lo:hi]
        else:
            km = keep_mask
        logits, _ = quant_carry_logits(model, task.tokens[lo:hi].to(dev), mode, km, acc)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(dev))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(dev)).to(torch.long).cpu()
    return _summarize(hits, task.dists)


def _agg_row(model, batches, mode, keep, device, max_chunk, corrupt_bounds, nb,
             window):
    """对一个任务批序列聚合：主腿 + 毁源对照 + 局部先验 + 距离分布。"""
    from tools.evalkit.carry_channel import wilson_ci
    from tools.evalkit.longrange_tasks import corrupt_span, local_prior_accuracy
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
    local_prior = chance = None
    for tb, bad in batches:
        main = quant_accuracy(model, tb, mode, keep_mask=keep, device=device,
                              max_chunk=max_chunk)
        c = quant_accuracy(model, bad, mode, keep_mask=keep, device=device,
                           max_chunk=max_chunk)
        agg["n"] += main["n"]; agg["n_hits"] += main["n_hits"]
        agg["cn"] += c["n"]; agg["chits"] += c["n_hits"]
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        for d, v in main["by_distance"].items():
            slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
            slot["n"] += v["n"]
            slot["n_hits"] += round(v["accuracy"] * v["n"])
        if local_prior is None:
            local_prior = local_prior_accuracy(tb, k=8)["accuracy"]
            chance = round(tb.chance(), 8)
    dists = agg["dists"]
    by_dist = {d: {"accuracy": round(s["n_hits"] / s["n"], 6), "n": s["n"]}
               for d, s in sorted(agg["by_dist"].items(), key=lambda kv: int(kv[0]))}
    return {
        "n_batches": nb, "n": agg["n"], "n_hits": agg["n_hits"],
        "accuracy": round(agg["n_hits"] / agg["n"], 6),
        "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
        "chance": chance, "local_prior": local_prior,
        "by_distance": by_dist,
        "corrupt_source": {"span": list(corrupt_bounds), "n": agg["cn"],
                           "n_hits": agg["chits"],
                           "accuracy": round(agg["chits"] / agg["cn"], 6),
                           "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
        "query_key_distance": {
            "min": min(dists), "max": max(dists),
            "median": sorted(dists)[len(dists) // 2], "window": window,
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


@torch.no_grad()
def quant_recall_row(model, T, window, anchors, bs, nb, sym, fill, cue, K, mode,
                     device, corrupt_bounds=(0, 16), max_chunk=64, seed_base=1000):
    """一个评测行在**单查询**关联回忆上的读数（判决腿；2623/W3 同一件仪器）。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.longrange_tasks import assoc_recall_batch, corrupt_span
    keep = (window_anchor_keep_mask(T, window, anchors, device=device)
            if window > 0 else None)
    batches = []
    for i in range(nb):
        tb = assoc_recall_batch(bs, T, K, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(
                                    seed_base + 7 * K + i))
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        batches.append((tb, bad))
    row = _agg_row(model, batches, mode, keep, device, max_chunk, corrupt_bounds,
                   nb, window)
    row["K"] = K
    row["window"] = window
    row["anchors"] = anchors
    return row


@torch.no_grad()
def quant_multi_recall_row(model, T, window, anchors, bs, nb, sym, fill, cue, K,
                           Q, gap, mode, device, corrupt_bounds=(0, 16),
                           max_chunk=64, seed_base=2000):
    """一个评测行在**多查询**关联回忆（训练型任务）上的读数（in-distribution）。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.longrange_tasks import (assoc_recall_multi_batch,
                                               corrupt_span)
    keep = (window_anchor_keep_mask(T, window, anchors, device=device)
            if window > 0 else None)
    batches = []
    for i in range(nb):
        tb = assoc_recall_multi_batch(bs, T, K, Q, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue, gap=gap,
                                      generator=torch.Generator().manual_seed(
                                          seed_base + 13 * K + Q + i))
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        batches.append((tb, bad))
    row = _agg_row(model, batches, mode, keep, device, max_chunk, corrupt_bounds,
                   nb, window)
    row["K"] = K
    row["Q"] = Q
    row["window"] = window
    row["anchors"] = anchors
    return row


def quant_liveness_of(acc: dict | None) -> dict | None:
    """把量化累加器折成活性量（`fp32` 恒等档返回 None）。"""
    return summarize_stats(acc) if acc is not None and acc["n_values"] else None
