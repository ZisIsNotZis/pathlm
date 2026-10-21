"""2683（`1.2.2` 的块条带掩码开关）的可失败单测：开关写错必须红。

钉住本实验**唯一干预点**的四条语义面，每一条都配一个已知写错的构造作反向样例：

1. **等价键数 / 行和**：逐行可见键数恒 = `min(24, i+1)`，且与均匀条带臂逐行相等
   （反向样例：把远块宽度改成 4 ⇒ 稳态只有 20 个键 ⇒ 红）。
2. **因果性**：`keep ⊆ causal`，`j > i` 一个都不开（反向样例：去掉下三角合取 ⇒ 红）。
3. **距离结构**：稳态行的远块距离区间恒 = `[25 + r, 32 + r]`（`r = i mod 8`）、最近 16 个距离全开
   （反向样例：把 `LAG` 从 4 改成 3 ⇒ 远块整体前移一格 ⇒ 红；这一条是「块粒度」本身的构造，
   行和看不见它，所以必须单独钉）。
4. **落盘一致性**：掩码指纹 = `results.json` 与训练期 `results.json::block_sparse_mask` 里的指纹
   （反向样例：任何一处改动都会让指纹变 ⇒ 红）。

外加判据三态 `classify()` 的边界（−0.014 / 0 / +0.014 三个点各判一次）：判据线被事后改动必红。

全程 CPU、秒级，不取 GPU 锁。

Run: python3 -m pytest tests/test_block_stripe_mask.py -q
"""

import importlib.util
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import window_anchor_keep_mask  # noqa: E402

EXP = os.path.join(ROOT, "axes", "1-time", "1.2.2-block-sparse",
                   "experiments", "2683-block-stripe-vs-band-13m")
T = 256
K_EFF = 24
K_BAND = 0.014                                               # harness/methodology.md §2 的配对带宽


def _load(modname, fname):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


block_sparse = _load("block_sparse_2683_test", "block_sparse.py")
run = _load("run_2683_test", "run.py")


def _counts(keep):
    return [int(v) for v in keep.sum(-1).tolist()]


def _distances(keep):
    """逐行被保留的距离集合 `{i: set(i-j)}`。"""
    idx = torch.arange(keep.shape[0])
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    return {i: set(dist[i][keep[i]].tolist()) for i in range(keep.shape[0])}


# ---------------------------------------------------------------- ① 等价键数 / 行和

def test_per_row_counts_equal_min_k_and_band():
    keep = block_sparse.block_stripe_keep_mask(T)
    band = window_anchor_keep_mask(T, K_EFF, 0)
    expected = [min(K_EFF, i + 1) for i in range(T)]
    assert _counts(keep) == expected
    assert _counts(keep) == _counts(band)                    # 与主判对照臂逐行等键
    assert int(keep.sum()) == int(band.sum()) == 5868        # n_visible_pairs 相等


def test_row_count_invariant_catches_wrong_block_width():
    """反向样例：远块宽度写错（4 而不是 8）⇒ 行和不再等键 ⇒ 上面那条不变量必须红。

    写错后 `far_start` 也按 4 对齐，远块会与本地带部分相交 ⇒ 稳态只剩 17 个键。
    """
    keep = block_sparse.block_stripe_keep_mask(T, local=16, block=4, lag=4)
    expected = [min(K_EFF, i + 1) for i in range(T)]
    assert _counts(keep) != expected
    assert _counts(keep)[100] == 17


def test_row_count_invariant_catches_overlapping_local_band():
    """反向样例：本地带 24 + 远块 8 ⇒ 远块不落在本地带里 ⇒ 稳态 32 个键 ⇒ 红。"""
    keep = block_sparse.block_stripe_keep_mask(T, local=24, block=8, lag=4)
    assert _counts(keep)[100] == 32
    assert _counts(keep) != [min(K_EFF, i + 1) for i in range(T)]


# ---------------------------------------------------------------- ② 因果性

def test_keep_is_causal_subset():
    keep = block_sparse.block_stripe_keep_mask(T)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert int((keep & ~causal).sum()) == 0
    assert bool(keep.diagonal().all())                        # 对角恒开（不留全 −inf 行）


# ---------------------------------------------------------------- ③ 距离结构（块粒度本身）

def test_far_block_distance_structure():
    """稳态行：最近 16 个距离全开 + 远块恰为 `[25 + r, 32 + r]`（r = i mod 8）。"""
    dist = _distances(block_sparse.block_stripe_keep_mask(T))
    for i in range(32, T):                                    # ⌊i/8⌋ ≥ LAG ⇒ 远块不再被 clamp
        r = i % 8
        want = set(range(0, 16)) | set(range(25 + r, 33 + r))
        assert dist[i] == want, f"row {i} (r={r}) 的距离结构不是本地 16 + 远块 8"
    for i in range(16, 24):                                   # 远块 clamp 到 0 号块 ⇒ 与本地带相接
        assert dist[i] == set(range(0, i + 1))
    for i in range(24, 32):                                   # ⌊i/8⌋ < LAG ⇒ 远块停在 0 号块，中间留洞
        assert dist[i] == set(range(0, 16)) | set(range(i - 7, i + 1))
    for i in range(0, 16):                                    # 更头部：远块 ⊆ 本地带
        assert dist[i] == set(range(0, i + 1))


def test_distance_invariant_catches_wrong_lag():
    """反向样例：`LAG=3` 行和照样是 24（16 + 8），只有距离结构变了 ⇒ 必须由这条抓住。"""
    keep = block_sparse.block_stripe_keep_mask(T, lag=3)
    assert _counts(keep) == [min(K_EFF, i + 1) for i in range(T)]   # 行和看不出来
    dist = _distances(keep)
    for i in (40, 100, 200):
        r = i % 8
        assert dist[i] != set(range(0, 16)) | set(range(25 + r, 33 + r))
        assert dist[i] == set(range(0, 16)) | set(range(17 + r, 25 + r))  # 远块整体前移一块


# ---------------------------------------------------------------- ④ 落盘一致性

def test_fingerprint_matches_landed_readings():
    keep = block_sparse.block_stripe_keep_mask(T)
    fp = block_sparse.pattern_fingerprint(keep)
    res = json.load(open(os.path.join(EXP, "results.json"), encoding="utf-8"))
    self_check = res["visible_keys_per_row"]["block_stripe_self_check"]
    assert self_check["fingerprint_sha256"] == fp
    assert self_check["per_row_counts_equal_band24"] is True
    assert self_check["n_visible_pairs"] == 5868
    assert self_check["mask_offending_future_pairs"] == 0
    for arm in ("block8l4_s0", "block8l4_s1"):
        train = json.load(open(os.path.join(EXP, "train", arm, "results.json"),
                               encoding="utf-8"))
        assert train["block_sparse_mask"]["fingerprint_sha256"] == fp
        assert train["block_sparse_mask"]["mask_calls_during_training"] > 0
        assert res["train_eval_mask_same"][arm]["same_mask"] is True


def test_fingerprint_is_not_constant():
    """反向样例：改一个键就改指纹 ⇒ 上一条的「指纹相等」不是空断言。"""
    keep = block_sparse.block_stripe_keep_mask(T)
    tweaked = keep.clone()
    tweaked[100, 50] = ~tweaked[100, 50]
    assert block_sparse.pattern_fingerprint(tweaked) != block_sparse.pattern_fingerprint(keep)


# ---------------------------------------------------------------- 判据三态边界（落盘判词 vs 预登记规则）

def _three_way(delta: float, band: float = K_BAND) -> str:
    """README §3 判据① 的预登记三态（`Δ = bpc_对照 − bpc_实验`，正 = 块条带更好）。"""
    if delta >= band:
        return "块条带占优"
    if delta <= -band:
        return "块条带更差"
    return "无质量增量"


def test_three_way_boundaries_and_landed_verdict():
    assert K_BAND == 0.014                                   # 预登记带宽本身（run.py 的 BAND）
    assert run.BAND == K_BAND
    cases = {-2 * K_BAND: "块条带更差", -K_BAND: "块条带更差", 0.0: "无质量增量",
             +K_BAND: "块条带占优", 2 * K_BAND: "块条带占优"}
    for delta, want in cases.items():
        assert _three_way(delta) == want
    # 落盘判词必须与「按落盘 Δ 重算的三态」逐 word 一致 ⇒ 事后改 Δ 或改带都会红
    res = json.load(open(os.path.join(EXP, "results.json"), encoding="utf-8"))
    verdict = res["main_verdict"]
    assert verdict["band_bpc"] == K_BAND and verdict["n_seed"] == 2
    for seed, key in ((0, "win24a0_s0"), (1, "win24a0_s1")):
        delta = res["main_block_stripe_vs_band"][str(seed)]["delta_bpc"]
        assert _three_way(delta) in verdict[f"seed{seed}"]
    # 反向样例：把同一个 Δ 换成越带的数，三态必须换档（不是常量函数）
    assert _three_way(res["main_block_stripe_vs_band"]["0"]["delta_bpc"]) == "无质量增量"
    assert _three_way(+0.02) == "块条带占优" and _three_way(-0.02) == "块条带更差"
