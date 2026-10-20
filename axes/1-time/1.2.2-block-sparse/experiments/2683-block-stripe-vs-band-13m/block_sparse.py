"""2683 · 块条带（block stripe）固定掩码：`1.2.2` 的第三个固定非均匀模式。

`keep[i, j] = (j ≤ i) ∧ ( (i − j) < LOCAL ∨ j ∈ 对齐块(⌊i/B⌋ − LAG) )`，对角恒开。

语义（一句话）：每个查询看**最近 `LOCAL` 个位置**，外加**一块** `B` 个键的**块对齐**远块——
那块落在自己所在块往前 `LAG` 个块的位置：行内偏移 `r = i mod B` 时，远块的距离区间 =
`[25 + r, 32 + r]`，即 `[25, 39]` 整体。与「均匀条带」的差别只在 8/24 个键：最近的 16 个不变，
把「距离 16 到 24+r 的 `9 + r` 个键」换成「一块连续 8 键」。这就是 BigBird 式**块稀疏**的确定性成员
（local window + 按块取可见块），不是随机块、也不是内容选择：掩码在训练前就固定、逐行可算。
块结构就体现在这里：可见集的**同一种模式逐块平移**（每 8 行平移一次），而条带臂是逐行平移。

**逐行可见键数等于 `min(24, i+1)`（与条带臂逐行相等，由构造保证、不是凑出来的）**。
记 远块 `= [B·A, B·A + B − 1]`（`A = max(0, ⌊i/B⌋ − LAG)`）、本地带 `= [i − LOCAL + 1, i]`，分三段核：

- 稳态（`⌊i/B⌋ ≥ LAG`）：远块末尾 `= B·⌊i/B⌋ − B·LAG + B − 1 ≤ i − 25 < i − 16` ⇒ 两段不相交
  ⇒ 计数 `= 16 + 8 = 24 = min(24, i+1)`。
- 头部 `16 ≤ i ≤ 23`（远块被 clamp 到 0 号块）：本地带从 `i − 15 ≤ 8` 起、与 0 号块相接
  ⇒ 并集恰为 `[0, i]` ⇒ 计数 `= i + 1`。
- 更头部 `i < 16`：远块 ⊆ 本地带 ⇒ 计数 `= i + 1`。
- 因果：远块先与下三角合取，绝不放出未来键（`i < B·A + B − 1` 的头部行会被截断）。

结论：`n_visible_pairs` 与条带臂同为 5868（`T=256`）⇒ 两臂的每位置字节完全相同，差不能记在预算上。

这份构造是本实验**唯一**的干预点：它替换训练前向里的窗口掩码（`train_block.py` 用子类覆写
`_window_mask`），评测侧用同一个函数构造显式掩码（`run.py`），因此训练/评测的可见集同源。
"""

from __future__ import annotations

import hashlib

import torch

from pathlm.depth_ar import DepthARModel, window_anchor_keep_mask

#: 本地带宽度（最近多少个位置恒可见）。
LOCAL = 16
#: 块大小（远块 = 一块 `BLOCK` 个键，块对齐）。
BLOCK = 8
#: 远块滞后多少个块（`LAG=4` 是本模式能与本地带不相交的最小滞后 ⇒ 远块尽量近、信息密度最高，
#: 是对「非均匀性值不值」这个主张**最有利**的设定）。
LAG = 4

#: 每行可见键数（稳态），也是与条带臂对齐的那个数。
K_EFF = LOCAL + BLOCK


def block_stripe_keep_mask(T: int, local: int = LOCAL, block: int = BLOCK,
                           lag: int = LAG, device: str = "cpu") -> torch.Tensor:
    """块条带的因果保留掩码 `[T, T]`（bool，True = 进入 softmax）。

    `keep[i, j] = (j ≤ i) ∧ ( (i − j) < local ∨ j ∈ [a, a + block) )`，
    `a = max(0, ⌊i / block⌋ − lag) · block`。对角恒开（同
    `pathlm.depth_ar.window_anchor_keep_mask` 的约定：避免整行被掩导致 softmax 全 −inf）。
    """
    if local < 1 or block < 1 or lag < 0:
        raise ValueError(f"local/block 必须 >= 1、lag >= 0，收到 {local}/{block}/{lag}")
    idx = torch.arange(T, device=device)
    i, j = torch.meshgrid(idx, idx, indexing="ij")
    causal = j <= i
    local_ok = (i - j) < local
    far_start = (i // block - lag).clamp_min(0) * block
    far_ok = (j >= far_start) & (j < far_start + block)
    keep = causal & (local_ok | far_ok)
    return keep | torch.eye(T, dtype=torch.bool, device=device)


def pattern_fingerprint(keep: torch.Tensor) -> str:
    """掩码的 sha256（`bool` 字节 + 形状）：训练期与评测期用的是不是同一张掩码，比这个就够。"""
    raw = keep.detach().to("cpu", torch.bool).contiguous().numpy().tobytes()
    h = hashlib.sha256()
    h.update(str(tuple(keep.shape)).encode())
    h.update(raw)
    return h.hexdigest()


def per_row_counts(keep: torch.Tensor) -> list[int]:
    return [int(v) for v in keep.sum(-1).tolist()]


def kept_distances(keep: torch.Tensor) -> dict:
    """被保留的 (query, key) 对按距离 `i − j` 的计数分布（人话：预算花在哪几段距离上）。"""
    T = keep.shape[0]
    idx = torch.arange(T)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    kept = dist[keep.detach().cpu()]
    return {int(d): int(c) for d, c in sorted(zip(*kept.unique(return_counts=True)))}


def self_check(T: int = 256) -> dict:
    """构造自证：因果子集、逐行计数（对条带臂逐行相等）、距离结构、指纹。

    与 `run.py` 的对齐核查同源（同一函数），这里只给「掩码本身对不对」；
    两臂之间的对齐在 `run.py` 里用**同一批**掩码逐行比。
    """
    keep = block_stripe_keep_mask(T)
    band = window_anchor_keep_mask(T, K_EFF, 0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    counts = per_row_counts(keep)
    band_counts = per_row_counts(band)
    expected = [min(K_EFF, i + 1) for i in range(T)]
    dist = kept_distances(keep)
    r = torch.arange(T) % BLOCK
    far_lo = (r + LOCAL + 9).tolist()          # 行内偏移 r ⇒ 远块起点距离 = r + 25
    far_hi = (r + LOCAL + 9 + BLOCK - 1).tolist()
    return {
        "pattern": "块条带：最近 16 + 滞后 4 块的一块对齐 8 键（LAG=4, BLOCK=8）",
        "local": LOCAL, "block": BLOCK, "lag_blocks": LAG, "k_eff": K_EFF,
        "n_visible_pairs": int(keep.sum()),
        "n_visible_pairs_band24": int(band.sum()),
        "causal_pairs": T * (T + 1) // 2,
        "visible_frac": round(float(keep.sum()) / (T * (T + 1) / 2), 6),
        "mask_offending_future_pairs": int((keep & ~causal).sum()),
        "per_row_counts_equal_band24": counts == band_counts,
        "per_row_counts_equal_min_k": counts == expected,
        "mismatch_rows": sum(1 for a, b in zip(counts, band_counts) if a != b),
        "kept_distance_histogram": dist,
        "kept_distance_note": (
            "逐行的距离结构 = 本地带 [0, 15] + 远块 [25 + r, 32 + r]（r = 行内偏移，0–7），"
            "即逐行的洞宽 = 9 + r；上表是把 256 行汇总后的计数，所以 16–39 都出现（每距离 8 次）。"),
        "far_distance_by_row_offset": {str(k): [a, b] for k, (a, b) in
                                       enumerate(zip(far_lo[:BLOCK], far_hi[:BLOCK]))},
        "kept_distance_min": min(dist), "kept_distance_max": max(dist),
        "hole_distances": [d for d in range(min(dist), max(dist) + 1) if d not in dist],
        "fingerprint_sha256": pattern_fingerprint(keep),
        "alive": int(keep.sum()) < T * (T + 1) // 2,
    }


#: 训练期掩码调用计数（`BlockStripeModel._window_mask` 累加）：= 0 即「掩码没进前向」的死干预。
MASK_CALLS = {"layers": 0}


class BlockStripeModel(DepthARModel):
    """`DepthARModel` + 块条带掩码（**唯一**覆写 = `_window_mask`，本实验唯一的干预点）。

    `cfg.train_window > 0` 才走到 `_window_mask`（共享前向的分支条件）⇒
    `window_mask_probe` 的「掩码关」那一支仍是真的无掩码前向，探针照旧有效；
    本覆写只用 `T`，忽略 `train_window / train_anchors` 的数值（那两个数只作预算申报）。
    """

    def _window_mask(self, B: int, T: int, device) -> torch.Tensor:
        keep = block_stripe_keep_mask(T, device=str(device))
        MASK_CALLS["layers"] += int(B) * len(self.blocks)
        return keep.view(1, 1, T, T).expand(B, self.cfg.n_heads, T, T)


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(256), ensure_ascii=False, indent=2))
