"""2621 · 搬运通道在 KV 精度压缩下的召回 —— **本实验唯一的新语义层**（评测期）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 `.scratch/38-auto-parallel/issues/
01-auto-parallel.md` 的「W3」，执行期间未改；§4b 记录并声明与契约原设计的偏离）。

本模块只回答一件事：**对一个「delta 写通道已被训练用起来」的模型，评测期把注意力
KV cache 降成 int8，召回会不会崩、字节/位置省多少？**

实现口径（SCHEMA §22 第 7 条：新逻辑进新模块，`pathlm/` 与 `tools/evalkit/` 只读）：

- 量化算子**复用** `tools/evalkit/kv_quant.py::quantize_kv`（对称 per-channel：
  通道轴 = `head_dim`、`(head, channel)` 一个 fp16 scale、沿 batch/token 取 amax，
  `qmax = 2^(bits-1)-1`）⇒ 与 2614/2619/2620 的量化是**同一件仪器**；本模块只把
  它的调用点接到 `CarryChannelBlock` 的注意力 K/V 上。因此**量化语义不由本模块定义**，
  本模块定义的是「**carry-aware 的量化前向**」这层接线。
- `mode="fp32"` 必须与模型自带的释放版前向（内置窗口掩码）**逐位相等**（单测 +
  `run.py` 仪器自检钉住）⇒ 这不是「另一个模型」，只是给同一张图加了存储精度这一维。
- 前向逐行复刻 `pathlm.depth_ar.CarryChannelBlock.forward`：`_input_states` 先算好
  单共享记忆槽（放进 `_carry_box`），每层 `a = SDPA(q,k,v; 窗口掩码)` 之前对 K/V 做
  一次量化-反量化往返，之后照旧加通道读出 `γ ⊙ mem`、`proj`、`mlp`。

活性量（SCHEMA §10）：
- 量化侧：每个 mode 落 `n_values / mean_abs_err / rel_err / max_abs_err / changed_frac`
  （证明「量化真的改了数值、且降了精度」，不是只改了个标签）；
- 通道侧：每层 `read_share = ‖γ⊙mem‖ / ‖a‖`（与训练日志每 200 步同一量）——
  压缩注意力后通道读出的相对占比是否变化，是「通道仍被读」的直接读数。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pathlm.depth_ar import CarryChannelBlock
from tools.evalkit.kv_quant import (_merge, _new_stats, kv_cache_footprint,
                                    quantize_kv, summarize_stats)
from tools.evalkit.longrange_tasks import _summarize

LN2 = 0.6931471805599453  # nats → bits

#: 本实验的 KV 存储精度档。`fp32` = 释放版（实现等值锚），`fp16` = 工业默认
#: 全精度基线（与 2614/2619/2620 的基线同口径），`int8` = 压缩档。
QUANT_MODES = ("fp32", "fp16", "int8")


# --------------------------------------------------------------- 前向（carry-aware）


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


def _quant_carry_block(blk: CarryChannelBlock, x: torch.Tensor, mode: str,
                       attn_mask, acc: dict | None):
    """`CarryChannelBlock.forward` 的逐行复刻，唯一的差别 = SDPA 之前对 K/V 做
    一次 `quantize_kv` 往返。`acc` 非空时把量化活性量累加进去。"""
    mem = blk._carry_box.get("mem")
    if mem is None:
        raise RuntimeError(
            "carry_quant 需要先经 `DepthARModel._input_states` 算好单共享记忆槽")
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)

    def shape(t):
        return t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)

    qs, ks, vs = shape(q), shape(k), shape(v)
    ks, ki = quantize_kv(ks, mode)
    if acc is not None:
        _merge(acc, ki)
    vs, vi = quantize_kv(vs, mode)
    if acc is not None:
        _merge(acc, vi)
    a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                       is_causal=attn_mask is None)
    gain = blk.carry_scale.float().repeat_interleave(
        blk.carry_head_dim).view(1, 1, d)
    scaled = mem * gain
    a = a.transpose(1, 2).reshape(B, T, d) + scaled.to(a.dtype)
    # 与训练日志每 200 步落盘的 read_share 同一表达式（分母含读出本身）
    read_share = float(scaled.detach().float().norm()
                       / a.detach().float().norm().clamp_min(1e-6))
    x = x + blk.proj(a)
    x = x + blk.mlp(blk.ln2(x))
    return x, read_share


@torch.no_grad()
def carry_quant_logits(model, tokens: torch.Tensor, mode: str,
                       keep_mask: torch.Tensor | None = None,
                       acc: dict | None = None):
    """dense 主干在全深度前向下的 logits `[B,T,V]`，注意力 K/V 走 `mode` 精度的往返。

    - **只接受带写入通道的模型**（每层都是 `CarryChannelBlock`）：内容 top-k / 线性
      注意力 / GQA 等自定义层会让「唯一因素 = KV 精度」不成立 ⇒ 直接报错；
    - `keep_mask=None` 时走释放版的 `is_causal=True` 路径；传
      `window_anchor_keep_mask(T, train_window, train_anchors)` 即与训练/评测同一掩码；
    - `acc` 非空时累加量化活性量（`_new_stats` 累加器，用 `summarize_stats` 折）。
    返回 `(logits, tele)`，`tele["read_share_by_layer"]` 是每层的通道读出占比。
    """
    if mode not in QUANT_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {QUANT_MODES}")
    if model.cfg.carry_rule is None:
        raise ValueError("carry_quant 只定义在带写入通道（carry_rule）的模型上")
    bad = [type(blk).__name__ for blk in model.blocks
           if type(blk) is not CarryChannelBlock]
    if bad:
        raise ValueError(
            "carry_quant 只定义在 CarryChannelBlock 上，遇到 " + ", ".join(sorted(set(bad))))
    B, T = tokens.shape
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
    同 `_summarize`），只把「模型 → logits」换成 `carry_quant_logits` ⇒ 两腿同一口径。
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
        logits, _ = carry_quant_logits(model, task.tokens[lo:hi].to(dev), mode,
                                       km, acc)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(dev))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(dev)).to(torch.long).cpu()
    return _summarize(hits, task.dists)


@torch.no_grad()
def quant_arm_battery(model, task, window: int, mode: str, anchors: int = 0,
                      device: str = "cuda", corrupt_span_bounds: tuple | None = None,
                      max_chunk: int = 64, acc: dict | None = None) -> dict:
    """一个 mode 在一个任务批上的读数：主臂 + 毁源对照臂 + 通道活性。

    字段与 `tools.evalkit.carry_channel.arm_battery` 对齐（判据 ①③ 的「同一件仪器」
    要求），只是把 `accuracy` 换成 `quant_accuracy`。可见性掩码由
    `window_anchor_keep_mask(task.T, window, anchors)` 构造（训练/评测一致）。
    """
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.carry_channel import wilson_ci
    from tools.evalkit.longrange_tasks import (corrupt_span, local_prior_accuracy)
    keep = window_anchor_keep_mask(task.T, window, anchors)
    acc_main = acc if acc is not None else _new_stats()
    main = quant_accuracy(model, task, mode, keep_mask=keep, device=device,
                          max_chunk=max_chunk, acc=acc_main)
    out = {"task": task.name, "mode": mode, "n": main["n"],
           "n_hits": main["n_hits"], "accuracy": main["accuracy"],
           "accuracy_ci": wilson_ci(main["n_hits"], main["n"]),
           "by_distance": main["by_distance"],
           "chance_uniform_answer_set": round(task.chance(), 8),
           "n_answers": task.n_answers,
           "local_prior": local_prior_accuracy(task, k=8)["accuracy"],
           "quant_liveness": (summarize_stats(acc_main) if mode != "fp32" else None),
           "keep_mask": {"train_window": window, "train_anchors": anchors,
                         "n_visible_pairs": int(keep.sum().item()),
                         "causal_pairs": task.T * (task.T + 1) // 2}}
    if corrupt_span_bounds is not None:
        lo, hi = corrupt_span_bounds
        bad = corrupt_span(task, lo, hi,
                           generator=torch.Generator().manual_seed(0))
        acc_c = _new_stats()
        c = quant_accuracy(model, bad, mode, keep_mask=keep, device=device,
                           max_chunk=max_chunk, acc=acc_c)
        out["corrupt_source"] = {"span": [lo, hi], "n": c["n"],
                                 "n_hits": c["n_hits"], "accuracy": c["accuracy"],
                                 "accuracy_ci": wilson_ci(c["n_hits"], c["n"]),
                                 "by_distance": c["by_distance"],
                                 "quant_liveness": (summarize_stats(acc_c)
                                                    if mode != "fp32" else None)}
    return out


@torch.no_grad()
def carry_quant_bpc(model, eval_arr, mode: str, keep_mask: torch.Tensor | None = None,
                    n_batches: int = 24, batch_size: int = 16, window_seed: int = 2,
                    device: str = "cuda", on_batch=None):
    """clean bpc（越低越好）：因果移位 CE 的 token 平均，K/V 走 `mode` 精度。

    窗口采样与 `tools.evalkit.kv_quant.kv_quant_bpc` / `2619.bpc_of` 同一顺序
    （同一 `window_seed`、同一 `pathlm.data.batch`）⇒ 与 2619/2620 的 bpc 腿同一条
    数据流、同一位置集。`on_batch(i, batch_stats, bpc_so_far)` 可选回调（周期性落盘）。

    返回 `(bpc, quant_liveness)`。
    """
    from pathlm.data import batch
    gen = torch.Generator().manual_seed(window_seed)
    T = model.cfg.seq_len
    nats, n_tok = 0.0, 0
    acc = _new_stats()
    for i in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        bacc = _new_stats()
        logits, _ = carry_quant_logits(model, x, mode, keep_mask, bacc)
        _merge(acc, bacc)
        logits = logits[:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                tgt.reshape(-1), reduction="sum").item()
        n_tok += tgt.numel()
        if on_batch is not None:
            on_batch(i, summarize_stats(bacc) if mode != "fp32" else None,
                     nats / n_tok / LN2)
    return nats / n_tok / LN2, (summarize_stats(acc) if mode != "fp32" else None)


# --------------------------------------------------------------- 字节口径（解析）


def bytes_table(cfg) -> dict:
    """解析 KV 每位置字节（唯一横轴口径，与 `kv_cache_footprint` 同式）。

    每位置字节 = `2(K,V) × n_layers × n_kv_heads_eff × head_dim × 每值字节`；
    int 档的 scale 开销 = 每 `(head, channel)` 一个 fp16，与 T 无关，单列。
    """
    rows = {}
    for mode in QUANT_MODES:
        rows[mode] = kv_cache_footprint(cfg, mode)
    base = rows["fp16"]["per_position_bytes"]
    for mode, r in rows.items():
        r["reduction_vs_fp16"] = round(1 - r["per_position_bytes"] / base, 6)
    return {"per_mode": rows, "baseline_mode": "fp16",
            "baseline_per_position_bytes": base,
            "note": ("MHA（n_heads=%d）解析：2×layers×heads×head_dim×value_bytes；"
                     "int8 = −50%%；GQA/MQA 因子**未测**（见 README §4b）"
                     % cfg.n_heads)}
