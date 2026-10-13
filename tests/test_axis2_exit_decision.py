"""逐 token 退出决策（轴 2 · 2.1.1 WP1）的可失败单测。

钉住四件事：
① **生产解码器真的会逐 token 退出**（`DepthARDecoder.step_auto`），且与批量逐层实现
   `exit_policy.score_stream`、以及并行 `DepthARModel.forward(exit_depths=D)` 三者同刻度；
② 在线策略（top-1 阈值）在两种实现下给出**同一套退出深度**（策略必须是
   (depth, logits) 的纯函数，否则增量/批量会分叉）；
③ 阈值边界（min/max 深度、τ=1.0 永不过早退、τ=0 → min_depth 立即退）；
④ 口径是**因果**的（后面的 token 不影响前面位置的退出读出）与 no-fill / proj-fill 语义。

Run: python -m pytest tests/test_axis2_exit_decision.py -q
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             DepthARDecoder, token_input, generate)
from pathlm.exit_policy import (Top1ProbPolicy, PrescribedDepths,  # noqa: E402
                                NeverExit, score_stream, stream_bpc,
                                generate_with_exit, exit_calibration,
                                risk_coverage, depth_histogram,
                                layerpass_ledger)


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12, **kw)
    return DepthARModel(cfg, vocab_size=10)


def tiny_tokens(m, T=12, seed=7):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, m.vocab_size, (1, T), generator=g)


def _parallel_logits(m, x, depths):
    """并行 forward 的逐位置**退出深度**读出（参照面）。"""
    _, aux = m(x, exit_depths=depths)
    rows = [aux["depth_logits"][int(depths[0, i])][0, i] for i in range(x.shape[1])]
    return torch.stack(rows, dim=0)          # [T,V]


# ---------- ① 三者同刻度 ----------

def test_prescribed_stream_matches_parallel_forward():
    """预设退出深度下，批量逐层口径必须逐位等于并行 forward 的退出深度读出。"""
    m = tiny_model()
    x = tiny_tokens(m)
    depths = torch.tensor([[0, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 3]])
    out = score_stream(m, x, PrescribedDepths(depths))
    assert torch.equal(out["depths"], depths)
    ref = _parallel_logits(m, x, depths)
    assert torch.allclose(out["logits"][0], ref, atol=2e-4)


def test_step_auto_matches_prescribed_schedule():
    """**生产解码器**逐位置决策：与批量口径给出同一套深度与同一套读出。"""
    m = tiny_model()
    x = tiny_tokens(m)
    depths = torch.tensor([[0, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 3]])
    out = score_stream(m, x, PrescribedDepths(depths))
    dec = DepthARDecoder(m)
    got_depths, got_logits = [], []
    for i in range(x.shape[1]):
        _, logits, dep = dec.step_auto(token_input(m, int(x[0, i]), dec.n),
                                       PrescribedDepths(depths).at(0, i))
        got_depths.append(dep)
        got_logits.append(logits.reshape(-1))
    assert got_depths == depths[0].tolist()
    assert torch.allclose(torch.stack(got_logits), out["logits"][0], atol=2e-4)
    assert dec.depths == depths[0].tolist()
    assert dec.n == x.shape[1]


def test_top1_policy_decoder_matches_batched():
    """**在线**策略（top-1 阈值）下，增量解码链与批量逐层给出同一套退出深度。

    这是「策略是 (depth, logits) 纯函数」的端到端证据；分叉说明某一边用了全局信息。
    """
    m = tiny_model()
    x = tiny_tokens(m)
    pol = Top1ProbPolicy(threshold=0.5, min_depth=1, max_depth=m.cfg.n_layers)
    out = score_stream(m, x, pol)
    dec = DepthARDecoder(m)
    got_depths, got_logits = [], []
    for i in range(x.shape[1]):
        _, logits, dep = dec.step_auto(token_input(m, int(x[0, i]), dec.n), pol)
        got_depths.append(dep)
        got_logits.append(logits.reshape(-1))
    assert got_depths == out["depths"][0].tolist()
    assert torch.allclose(torch.stack(got_logits), out["logits"][0], atol=2e-4)


# ---------- ② 阈值/边界语义 ----------

def test_never_exit_is_full_depth():
    m = tiny_model()
    x = tiny_tokens(m)
    out = score_stream(m, x, NeverExit())
    _, aux = m(x, exit_depths=None)
    assert set(out["depths"][0].tolist()) == {m.cfg.n_layers}
    assert torch.allclose(out["logits"][0], aux["depth_logits"][-1][0], atol=2e-5)


def test_threshold_one_never_exits_early():
    """τ=1.0：softmax top-1 严格 < 1 ⇒ 只有到顶（max_depth）才停，等价全深度。"""
    m = tiny_model()
    x = tiny_tokens(m)
    L = m.cfg.n_layers
    out = score_stream(m, x, Top1ProbPolicy(1.0, min_depth=1, max_depth=L))
    assert set(out["depths"][0].tolist()) == {L}
    out2 = score_stream(m, x, Top1ProbPolicy(1.0, min_depth=1, max_depth=None))
    assert set(out2["depths"][0].tolist()) == {L}


def test_threshold_zero_exits_at_min_depth():
    m = tiny_model()
    x = tiny_tokens(m)
    L = m.cfg.n_layers
    pol = Top1ProbPolicy(0.0, min_depth=2, max_depth=L)
    out = score_stream(m, x, pol)
    assert set(out["depths"][0].tolist()) == {2}
    # 批量口径的读出 = 并行 forward 在深度 2 的读出
    ref = _parallel_logits(m, x, torch.full_like(out["depths"], 2))
    assert torch.allclose(out["logits"][0], ref, atol=2e-4)


def test_min_depth_is_respected_for_high_threshold():
    """高阈值 ⇒ 退得深；min_depth 是硬下界（不得超过）。"""
    m = tiny_model()
    x = tiny_tokens(m)
    L = m.cfg.n_layers
    out = score_stream(m, x, Top1ProbPolicy(0.999, min_depth=1, max_depth=L))
    assert int(out["depths"].min()) >= 1
    assert int(out["depths"].max()) <= L


def test_depth0_prescribed_is_input_readout():
    """depth 0 = 输入态直连读出（不进任何层），锁定 `_apply(0, ...)` 这条路径。"""
    m = tiny_model()
    x = tiny_tokens(m)
    d0 = torch.zeros_like(x)
    out = score_stream(m, x, PrescribedDepths(d0))
    h0 = m._input_states(x)
    ref = m.readout(h0, depth=0)
    assert torch.allclose(out["logits"][0], ref, atol=2e-5)


# ---------- ③ 因果性与语义变体 ----------

def test_stream_is_causal():
    """位置 i 的退出读出与深度只依赖 tokens[:i+1]（后面的 token 改了也不变）。"""
    m = tiny_model()
    x = tiny_tokens(m)
    pol = Top1ProbPolicy(0.5, min_depth=1, max_depth=m.cfg.n_layers)
    a = score_stream(m, x, pol)
    x2 = x.clone()
    x2[0, -1] = (int(x2[0, -1]) + 1) % m.vocab_size
    b = score_stream(m, x2, pol)
    assert torch.equal(a["depths"][0, :-1], b["depths"][0, :-1])
    assert torch.allclose(a["logits"][0, :-1], b["logits"][0, :-1], atol=2e-5)


def test_nofill_stream_matches_parallel_forward():
    """no-fill 语义（早退位置在深层 KV 缺席）下，批量口径同样对齐并行 forward。"""
    m = tiny_model(fill_kv=False)
    x = tiny_tokens(m)
    depths = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]])
    out = score_stream(m, x, PrescribedDepths(depths))
    assert torch.equal(out["depths"], depths)
    ref = _parallel_logits(m, x, depths)
    assert torch.allclose(out["logits"][0], ref, atol=2e-4)


def test_projfill_identity_matches_fill():
    """proj-fill 的 adapter 恒等初始化 ⇒ 起步与 fill 逐位一致（单因素对照的前提）。"""
    m = tiny_model(proj_fill=True)
    m2 = tiny_model()
    x = tiny_tokens(m)
    depths = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]])
    a = score_stream(m, x, PrescribedDepths(depths))
    b = score_stream(m2, x, PrescribedDepths(depths))
    assert torch.allclose(a["logits"], b["logits"], atol=1e-5)


def test_dense_model_is_rejected():
    m = tiny_model(dense=True)
    x = tiny_tokens(m)
    with pytest.raises(ValueError):
        score_stream(m, x, NeverExit())


# ---------- ④ 算力账 / bpc / 校准表 ----------

def test_stream_bpc_and_ledger():
    m = tiny_model()
    x = tiny_tokens(m)
    L = m.cfg.n_layers
    out = stream_bpc(m, x, Top1ProbPolicy(0.5, min_depth=1, max_depth=L))
    assert out["bpc"] > 0
    assert abs(out["layer_passes_per_token"] - out["depths"].float().mean().item()) < 1e-9
    led = layerpass_ledger(out)
    assert led["layer_passes_per_token"] == pytest.approx(out["layer_passes_per_token"], abs=1e-6)
    assert led["skip_frac"] == pytest.approx(1 - out["layer_passes_per_token"] / L, abs=1e-6)
    hist = out["depths_hist"]
    assert abs(sum(h["frac"] for h in hist) - 1.0) < 1e-9
    # 按退出深度分解的计数之和 = 预测位置总数
    # by_exit_depth 覆盖预测行（T−1），depths_hist / 算力账覆盖全部位置（T）
    assert sum(r["n"] for r in out["by_exit_depth"]) == out["depths"].numel() - out["depths"].shape[0]
    assert 0.0 <= out["exit_acc_tf"] <= 1.0 and 0.0 <= out["exit_acc_true"] <= 1.0


def test_risk_coverage_and_calibration_shapes():
    m = tiny_model()
    x = tiny_tokens(m)
    L = m.cfg.n_layers
    cal = exit_calibration(m, x, n_bins=5)
    assert len(cal["rows"]) == L           # depth 0..L-1（depth L 无「提前退出」语义）
    for r in cal["rows"]:
        assert 0.0 <= r["ece_tf"] <= 1.0 and 0.0 <= r["ece_true"] <= 1.0
        assert len(r["bins_tf"]) == 5 and len(r["bins_true"]) == 5
    rc = risk_coverage(m, x, thresholds=(0.5, 0.9))
    assert len(rc) == (L - 1) * 2          # depth 1..L-1 × 阈值
    for r in rc:
        assert 0.0 <= r["coverage"] <= 1.0


# ---------- ⑤ 自由生成端到端（生产解码器） ----------

def test_generate_with_never_exit_matches_full_depth_generate():
    """NeverExit 的自由生成必须与既有全深度贪心生成逐 token 相同。"""
    m = tiny_model()
    torch.manual_seed(11)
    prompt = torch.randint(0, m.vocab_size, (6,))
    a, _ = generate(m, prompt, 5, commit="soft", forced_depth=None)
    b, st = generate_with_exit(m, prompt, 5, NeverExit(), commit="soft")
    assert a == b
    assert set(st["depths"]) == {m.cfg.n_layers}


def test_generate_with_exit_legal_and_bounded():
    m = tiny_model()
    torch.manual_seed(12)
    prompt = torch.randint(0, m.vocab_size, (6,))
    L = m.cfg.n_layers
    gen, st = generate_with_exit(m, prompt, 5, Top1ProbPolicy(0.0, min_depth=1,
                                                             max_depth=L))
    assert len(gen) == 5
    assert all(0 <= t < m.vocab_size for t in gen)
    assert set(st["depths"]) == {1}
    assert len(st["depths"]) == 5
    assert st["layer_passes_per_token"] <= L
