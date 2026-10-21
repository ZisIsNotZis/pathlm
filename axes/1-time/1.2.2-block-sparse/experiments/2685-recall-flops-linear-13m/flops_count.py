"""2685 · 每 token 等效注意力 FLOPs：在**产生读数的同一次前向里**按实际掩码计数。

`1.2.2` 的里程碑要「每 token 等效 FLOPs」，而旧交付只有**结构上界**（解析量）。本模块把它换成
**计数**：跑一次真实前向，用模块前置钩子取每个注意力块**实际收到**的注意力掩码与张量形状，按下面
两条式子累加 MAC（乘加），再除以 token 数。

- softmax / 掩码块（`Block`、`BlockStripeModel`）：每对 (query, key) 每个头要 `2·d_head` 个 MAC
  （QKᵀ 一个 + 加权求和一个）⇒ 每层 `2 · kept_pairs · d_head`，其中 `kept_pairs` 是该前向**实际**
  看到的掩码的非零数（`attn_mask=None` ⇒ 因果 `T(T+1)/2` 每行）。
- 线性核块（`LinearAttnBlock`，φ=elu+1 的 O(T) 分块实装）：按 `linear_attn_chunked` 的算式逐项数
  ——`S_chunk`(k v) 与 `num_inter`(q S) 各 `B·H·T·dh²`、`intra`(q kᵀ) 与 `num_intra`(intra v) 各
  `B·H·T·C·dh`、`z` 求和与 `den_inter` 各 `B·H·T·dh`。这些项与 `pathlm/depth_ar.py::linear_attn_chunked`
  的 einsum 一一对应，所以是**可复算**的。

两个数分开报，因为它们回答不同问题：

- `equiv_macs_per_token`：**等效**量（按实际可见键数算）——「这类掩码在真稀疏核上应当花的算力」；
- `executed_macs_per_token`：**本条实装真正算掉的**量——稠密 bool 掩码下掩码臂仍算整张因果矩阵，
  所以它恒等于全因果的量（定律 11 的「省幅未兑现」在这里就是一个数）。

本模块只读模型与前向，不改任何共享文件。
"""

from __future__ import annotations

import torch

from tools.evalkit.retention import masked_logits

#: 引擎里线性核块的类名（`pathlm.depth_ar.LinearAttnBlock`）：按名字判，避免为计数引入新的导入面。
LINEAR_BLOCK_CLASS = "LinearAttnBlock"


def _causal_pairs(T: int) -> int:
    return T * (T + 1) // 2


def _mask_pairs(am: torch.Tensor | None, B: int, H: int, T: int) -> int:
    """该前向实际收到的掩码里 (batch, head, query, key) 的非零数。"""
    if am is None:
        return B * H * _causal_pairs(T)
    a = am.detach().to("cpu", torch.bool)
    if a.dim() == 2:
        return B * H * int(a.sum())
    if a.dim() == 4:
        if a.shape[0] == 1 or a.shape[1] == 1:
            return int(a.expand(B, H, T, T).sum())
        return int(a.sum())
    raise ValueError(f"意外形状的注意力掩码：{tuple(am.shape)}")


def _linear_macs(B: int, H: int, T: int, dh: int, chunk: int) -> int:
    """`linear_attn_chunked` 的 MAC 数（逐项对应它的 einsum / sum）。"""
    C = min(int(chunk), T)
    return B * H * (T * dh * dh          # S_chunk  = einsum('bhncd,bhnce->bhnde')
                    + T * dh * dh        # num_inter = einsum('bhncd,bhnde->bhnce')
                    + T * dh             # z_chunk  = kc.sum(3)
                    + T * dh             # den_inter = einsum('bhncd,bhnd->bhnc')
                    + T * C * dh         # intra     = einsum('bhnid,bhnjd->bhnij')
                    + T * C * dh)        # num_intra = einsum('bhnij,bhnjd->bhnid')


def count_attention(model, tokens: torch.Tensor, keep_mask: torch.Tensor | None = None,
                    batch: int = 4) -> dict:
    """跑一次真实前向并数注意力 MAC：返回逐层记录 + 每 token 等效/实耗两个数。"""
    T = model.cfg.seq_len
    H = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = d_model // H
    blocks = list(model.blocks)
    seen: list[dict] = []

    def pre_hook(mod, args, kwargs):
        x = args[0]
        seen.append({"cls": type(mod).__name__, "B": int(x.shape[0]),
                     "T": int(x.shape[1]), "attn_mask": kwargs.get("attn_mask")})

    handles = [b.register_forward_pre_hook(pre_hook, with_kwargs=True) for b in blocks]
    try:
        x = tokens[:batch].to(next(model.parameters()).device)
        with torch.no_grad():
            masked_logits(model, x, keep_mask)
    finally:
        for h in handles:
            h.remove()
    if len(seen) != len(blocks):
        raise RuntimeError(f"钩子只抓到 {len(seen)}/{len(blocks)} 个块：计数不完整")

    per_layer, equiv, executed, n_tok = [], 0, 0, 0
    kept_pairs_total = 0
    query_head_total = 0
    for rec in seen:
        B, Tl = rec["B"], rec["T"]
        n_tok += B * Tl
        if rec["cls"] == LINEAR_BLOCK_CLASS:
            macs = _linear_macs(B, H, Tl, d_head, model.cfg.linear_chunk)
            per_layer.append({"class": rec["cls"], "method": "linear_attn_chunked",
                              "macs": macs, "kept_pairs": None})
            equiv += macs
            executed += macs              # 线性块算的就是它自己的 O(T) 算式
        else:
            pairs = _mask_pairs(rec["attn_mask"], B, H, Tl)
            kept_pairs_total += pairs
            query_head_total += B * H * Tl
            per_layer.append({"class": rec["cls"], "method": "softmax_masked",
                              "macs": 2 * pairs * d_head, "kept_pairs": pairs})
            equiv += 2 * pairs * d_head
            executed += 2 * B * H * _causal_pairs(Tl) * d_head   # 稠密 bool 掩码：整张因果矩阵
    return {
        "n_layers": len(per_layer), "n_heads": H, "d_head": d_head, "T": T,
        "batch": int(min(batch, tokens.shape[0])),
        "equiv_macs_per_token": round(equiv / n_tok, 3),
        "executed_macs_per_token": round(executed / n_tok, 3),
        "equiv_flops_per_token": round(2 * equiv / n_tok, 3),
        "executed_flops_per_token": round(2 * executed / n_tok, 3),
        "kept_keys_per_row_mean": (round(kept_pairs_total / query_head_total, 3)
                                   if query_head_total else None),
        "per_layer": per_layer,
        "formula": {
            "softmax_masked": ("macs = 2 · kept_pairs · d_head（kept_pairs = 该前向实际掩码的非零数；"
                               "None 掩码 ⇒ 因果 T(T+1)/2 每行）"),
            "linear_attn_chunked": ("macs = B·H·(2·T·dh² + 2·T·C·dh + 2·T·dh)（逐项对应 "
                                    "linear_attn_chunked 的 einsum；C = min(linear_chunk, T)）"),
            "per_token": "每层 macs / (B·T) 后对层求和；FLOPs = 2·MACs",
            "executed_vs_equiv": ("executed = 本条实装真正算掉的量：稠密 bool 掩码下掩码臂仍算整张因果"
                                  "矩阵 ⇒ executed 与全因果相同；equiv = 真稀疏核上应花的量"),
        },
    }
