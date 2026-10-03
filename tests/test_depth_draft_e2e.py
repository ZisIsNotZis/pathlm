"""ticket 38 W2 · 深度草稿端到端解码的语义钉（可失败单测）。

对应契约 `axes/C-vertical/C1-layers-per-token/experiments/2621-depth-draft-e2e/README.md`。
每个语义开关都有可失败断言：批量追加 == 增量逐位（KV 语义）、spec 流 == 普通贪心流
（自 spec 不改质量）、`graded` 流 == `single` 流、拒绝路径（草稿故意错）仍保真、
接受账本 `emitted == accepted + 1`、`graded` 在拒绝时确实省 layer-position。

运行：`python3 -m pytest tests/test_depth_draft_e2e.py -q`
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "axes", "C-vertical", "C1-layers-per-token", "experiments",
    "2621-depth-draft-e2e"))

import spec_decode as S  # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARDecoder, DepthARModel,  # noqa: E402
                             token_input)


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=32, n_layers=4, n_heads=2, seq_len=64, mlp_mult=2)
    return DepthARModel(cfg, vocab_size=41).eval()


PROMPT = [3, 11, 7, 2, 19, 5, 23, 1]


def test_committed_cache_matches_depthar_decoder():
    """批量 `run_layers` 的逐层 KV 与末位状态，必须与 `DepthARDecoder.step`
    （逐位、forced_depth=L）在同一容差内一致——两条路径共用同一套 KV 语义。"""
    m = tiny_model()
    toks = PROMPT
    dec = DepthARDecoder(m)
    h_last = lg_last = None
    for i, t in enumerate(toks):
        h_last, lg_last = dec.step(token_input(m, t, i), m.cfg.n_layers)
    cache = S.CommittedCache(m)
    inp = torch.cat([token_input(m, t, i) for i, t in enumerate(toks)], dim=0)
    states, kv = S.run_layers(m, cache, inp, start_layer=0)
    lg_batch = m.readout(states[-1][-1:])
    assert (lg_batch.reshape(-1) - lg_last.reshape(-1)).abs().max().item() < 2e-4
    assert (states[-1][-1].reshape(-1) - h_last.reshape(-1)).abs().max().item() < 2e-4
    depth_h = {j: states[j][-1:] for j in range(len(states))}
    cache.commit(kv, len(toks), depth_h, 0)
    for k0 in range(m.cfg.n_layers):
        kb, vb = cache.layer_kv(k0)
        assert (kb - dec.K[k0]).abs().max().item() < 2e-4
        assert (vb - dec.V[k0]).abs().max().item() < 2e-4


def test_single_spec_matches_plain_greedy():
    """`single` 臂发出的 token 流 == 同 checkpoint 的普通全栈贪心流。"""
    m = tiny_model()
    gen_s, st = S.spec_generate(m, PROMPT, n_new=24, gamma=3, draft_depth=2,
                                scheme="single")
    gen_p = S.plain_greedy(m, PROMPT, n_new=24)
    assert st["n_cycles"] > 0
    assert gen_s == gen_p


def test_graded_spec_matches_single_stream():
    """`graded`（d2 → d3 → 全栈）与 `single`（一次全栈）发出的流逐 token 相同。"""
    m = tiny_model(seed=1)
    gen_s, _ = S.spec_generate(m, PROMPT, n_new=24, gamma=3, draft_depth=2,
                               scheme="single")
    gen_g, _ = S.spec_generate(m, PROMPT, n_new=24, gamma=3, draft_depth=2,
                               mid_depth=3, scheme="graded")
    assert gen_g == gen_s


def test_accept_prefix_boundaries():
    """接受判定的三个边界：全接受 / 首位拒绝 / 中位拒绝。"""
    assert S.accept_prefix([1, 2, 3], [1, 2, 3], 3) == 3
    assert S.accept_prefix([9, 2, 3], [1, 2, 3], 3) == 0
    assert S.accept_prefix([1, 9, 3], [1, 2, 3], 3) == 1
    # 只比较前 gamma 个（草稿比目标长一位是正常情形）
    assert S.accept_prefix([1, 2, 3, 4], [1, 2, 3, 4], 3) == 3


def test_cycle_accounting_emitted_is_accepted_plus_one():
    """每轮 `emitted == accepted + 1`；总生成长度 == n_new；提交位置单调递增。"""
    m = tiny_model(seed=2)
    gen, st = S.spec_generate(m, PROMPT, n_new=24, gamma=3, draft_depth=2,
                              scheme="single")
    assert len(gen) == 24
    prev = len(PROMPT)
    for c in st["cycles"]:
        assert c["emitted"] == c["accepted"] + 1
        assert c["committed"] == prev + c["emitted"]
        prev = c["committed"]
    assert st["n_drafts"] == 3 * st["n_cycles"]


def test_draft_depth_knob_is_live():
    """草稿深度开关不是死的：depth-2 出口与全栈出口的 logits 必须不同
    （否则「草稿深度」这个干预量在测一个恒等映射）。"""
    m = tiny_model(seed=3)
    cache, _ = S.prefill(m, PROMPT)
    toks, hdd, _ = S._draft_phase(m, cache, 2, 2)
    h_in = torch.cat(hdd, dim=0)
    states, _ = S.run_layers(m, cache, h_in, start_layer=2)
    lg_shallow = m.readout(hdd[0]).reshape(-1)
    lg_full = m.readout(states[-1][0:1]).reshape(-1)
    assert (lg_shallow - lg_full).abs().max().item() > 1e-3


@pytest.fixture
def wrong_draft(monkeypatch):
    """把草稿的 argmax 换成次优 token ⇒ 确定性拒绝（不引入生产代码钩子）。"""
    orig = S.Draft.step

    def step(self, inp):
        h, lg = orig(self, inp)
        idx = lg.topk(2, dim=-1).indices
        lg2 = lg.clone()
        lg2[..., idx[..., 0]] = -1e9
        return h, lg2

    monkeypatch.setattr(S.Draft, "step", step)
    return None


def test_rejection_path_keeps_stream_faithful(wrong_draft):
    """草稿全错时：accepted == 0，但发出的流仍 == 普通贪心流（修正路径正确）。"""
    m = tiny_model(seed=4)
    gen, st = S.spec_generate(m, PROMPT, n_new=16, gamma=3, draft_depth=2,
                              scheme="single")
    gen_p = S.plain_greedy(m, PROMPT, n_new=16)
    assert st["n_accepted"] == 0
    assert all(c["accepted"] == 0 and c["emitted"] == 1 for c in st["cycles"])
    assert gen == gen_p


def test_graded_saves_layerpos_on_rejection(wrong_draft):
    """拒绝发生时，`graded` 的等效 layer-position 必须严格少于 `single`。"""
    m = tiny_model(seed=5)
    _, s1 = S.spec_generate(m, PROMPT, n_new=12, gamma=3, draft_depth=2,
                            scheme="single")
    _, sg = S.spec_generate(m, PROMPT, n_new=12, gamma=3, draft_depth=2,
                            mid_depth=3, scheme="graded")
    assert s1["n_cycles"] == sg["n_cycles"]
    assert sg["eq_layerpos"] < s1["eq_layerpos"]
    for cs, cg in zip(s1["cycles"], sg["cycles"]):
        assert cg["eq_layerpos"] < cs["eq_layerpos"]


def test_spec_cache_kv_matches_reference_batch_forward():
    """spec 提交前缀的逐层 KV 必须与「对已提交序列做一次普通批量前向」的 KV 一致
    （容差内）。这条测试钉住一个重要设计：被接受位置的 block `0..dd-1` KV 来自
    草稿临时缓存、`dd..L-1` 来自验证批——两者拼起来必须等于全栈语义下的 KV。"""
    m = tiny_model(seed=7)
    gen, st, cache = S.spec_run(m, PROMPT, n_new=16, gamma=3, draft_depth=2,
                                scheme="single")
    seq = PROMPT + gen[:-1]          # 最后一个生成 token 的 KV 尚未提交
    n = min(len(seq), cache.n)
    ref, _ = S.prefill(m, seq[:n])
    assert cache.n >= len(PROMPT)
    for k0 in range(m.cfg.n_layers):
        a, b = cache.K[k0][:, :, :n], ref.K[k0][:, :, :n]
        assert (a - b).abs().max().item() < 5e-4, f"layer {k0} KV 不一致"


def test_invalid_configs_raise():
    m = tiny_model(seed=6)
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=4, draft_depth=0)
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=4, draft_depth=m.cfg.n_layers)
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=4, draft_depth=2, scheme="graded",
                        mid_depth=None)
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=4, draft_depth=2, scheme="graded",
                        mid_depth=2)
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=4, scheme="nope")
    with pytest.raises(ValueError):
        S.spec_generate(m, PROMPT, n_new=9999)
