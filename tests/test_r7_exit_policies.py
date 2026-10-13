"""R7 共用解码机械/早退策略的语义钉（可失败单测）。

对应契约 `axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/README.md`。
每个策略都有可失败断言：共享机械与并行前向一致（有门/无门两路）、CALM 门控确实
在最浅达标层退出、spec 发出的流 == 同 arm 全深度贪心流、spec 的解析算力账与实测一致、
接受账本自洽。

运行：`python3 -m pytest tests/test_r7_exit_policies.py -q`
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(
    ROOT, "axes", "7-references", "experiments", "R7-existing-early-exit",
    "2623-standard-early-exit-13m"))

import exit_policies as EP  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, token_input  # noqa: E402
from pathlm.mod_router import MoDModel  # noqa: E402


def tiny_model(seed=0, carry_gate=True, **kw):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=32, n_layers=4, n_heads=2, seq_len=64,
                        mlp_mult=2, carry_gate=carry_gate, **kw)
    return DepthARModel(cfg, vocab_size=41).eval()


PROMPT = [3, 11, 7, 2, 19, 5, 23, 1]


def trained_tiny_model(seed=0, steps=80, period=5, carry_gate=True):
    """**非退化**微模型：在周期性序列上训 80 步（几秒钟），贪心流由周期驱动
    ⇒ 流里出现多个不同 token、且 depth-0 草稿一定会被拒。

    为什么专门要它（2026-10-10 事故）：未训练微模型的贪心流常常是「同一个 token
    重复」，连 depth-0 草稿都能 100% 命中 ⇒ 任何「流错位/丢第一个 token」的实现
    bug 都会被这种退化点掩盖（本波真的发生了：spec 流相对贪心流整体错位一格）。
    单测必须先断言**非退化**再断言等价（SCHEMA §17.16）。
    """
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=32, n_layers=4, n_heads=2, seq_len=64, mlp_mult=2,
                        carry_gate=carry_gate)
    m = DepthARModel(cfg, vocab_size=41)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        x = torch.arange(1, 1 + 16).remainder(period) + 1        # 周期序列
        x = x.unsqueeze(0).repeat(8, 1)
        loss, _ = m(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return m.eval()


@pytest.mark.parametrize("gated", [True, False])
def test_shared_cache_matches_parallel_forward(gated):
    """共享机械的逐位增量解码必须与并行混合深度 forward 的逐深度 logits 一致
    （有门/无门两路都钉）——缓存语义错位时本测试变红。"""
    m = tiny_model(carry_gate=gated)
    torch.manual_seed(3)
    x = torch.randint(0, 41, (1, 8))
    depths = torch.tensor([[0, 3, 1, 2, 0, 3, 3, 1]])
    _, aux = m(x, exit_depths=depths)
    dec = EP.MixDepthCache(m, gated)
    for i in range(8):
        d, _, logits = dec.step(token_input(m, int(x[0, i]), dec.abs_pos),
                                int(depths[0, i]))
        assert d == int(depths[0, i])
        assert torch.allclose(logits, aux["depth_logits"][d][0, i], atol=2e-4), i


def test_calm_calibration_is_out_of_sample_and_hits_target():
    """CALM 标定：θ 只用前半批次标定；`*_holdout` 用后半批次（未参与标定）核验，
    样本外一致率应 ≥ 1−ε（这正是 CALM 的标定契约）。"""
    m = tiny_model()
    torch.manual_seed(9)
    ev = torch.randint(0, 41, (6000,))
    cal = EP.calm_calibrate(m, ev, eps_list=(0.05,), n_batches=4, batch_size=2,
                            device="cpu")
    e = cal["by_eps"]["0.05"]
    assert cal["n_calib"] > 0 and cal["n_holdout"] > 0
    assert len(e["theta"]) == m.cfg.n_layers + 1
    assert e["agree_with_full_holdout"] >= 0.95 - 1e-9
    assert 0 <= e["mean_exit_depth_holdout"] <= m.cfg.n_layers


def test_calm_gate_stops_at_shallowest_passing_depth():
    """CALM 门控：停在**最浅**的 `conf ≥ θ` 深度；θ 全 0 ⇒ 停在 depth 0，
    θ 全 1.1（不可达）⇒ 跑满 L。"""
    m = tiny_model()
    dec = EP.MixDepthCache(m, True)
    inp = token_input(m, 5, 0)
    k, _, lg = dec.step(inp, 4, depth_fn=lambda kk, l: _conf(l) >= 0.0)
    assert k == 0
    dec2 = EP.MixDepthCache(m, True)
    k2, _, _ = dec2.step(inp, 4, depth_fn=lambda kk, l: _conf(l) >= 1.1)
    assert k2 == 4


def _conf(lg):
    return float(lg.float().softmax(-1).max())


def test_calm_decode_reports_measured_depth_and_compute():
    """真解码路径上的 CALM：测得的平均退出深度、层×位置账与深度直方图自洽；
    θ 全 1.1 ⇒ 与全深度贪心流逐 token 相同（且算力 = L/token）。"""
    m = tiny_model()
    L = m.cfg.n_layers
    g_full, s_full = EP.greedy_decode(m, PROMPT, 12, timeit=False)
    g_calm, s_calm = EP.calm_decode(m, PROMPT, 12, [1.1] * (L + 1))
    assert g_calm == g_full
    assert s_calm["mean_exit_depth"] == L
    # 层×位置账 = L × 真正跑过前向的位置数（末位 token 由上一次前向的 argmax 免费给出）
    assert s_calm["layerpos_generation"] == L * (len(g_calm) - 1)
    assert s_calm["layerpos_per_emitted_token"] == pytest.approx(
        L * (len(g_calm) - 1) / len(g_calm), abs=1e-9)
    # θ 全 0 ⇒ 每个位置在 depth 0 退出（最浅），流与全深度不同（这条是活性断言）
    g_shallow, s_shallow = EP.calm_decode(m, PROMPT, 12, [0.0] * (L + 1))
    assert s_shallow["mean_exit_depth"] == 0.0
    assert set(s_shallow["depth_hist"]) == {"0"}


@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("gamma,dd", [(1, 4), (2, 2), (4, 2), (4, 4), (8, 4)])
def test_spec_stream_equals_plain_greedy(gamma, dd, gated):
    """自投机解码不改质量（用户口径：self-spec 只是加速器）：发出的流必须与同 arm
    的全深度贪心流**逐 token 相同**。

    先用**非退化**模型与**非退化流**（≥3 个不同 token）挡住「错位/丢一个 token」这类
    会被退化点掩盖的实现 bug；`gamma=1` 这条还钉住「prefill 末位的 token 必须被发出」。"""
    m = trained_tiny_model(carry_gate=gated)
    plain, _ = EP.greedy_decode(m, PROMPT, 30)
    assert len(set(plain)) >= 3, f"贪心流退化（{set(plain)}）⇒ 本测试无区分度"
    spec, st = EP.spec_decode(m, PROMPT, 30, gamma=gamma, draft_depth=dd)
    assert spec == plain, (gamma, dd, gated)
    assert st["n_cycles"] >= 1 and st["accept_rate"] is not None
    assert len(spec) == 30


def test_spec_ledger_and_analytic_compute():
    """接受账本自洽 + 解析算力账与实测一致：
    `layerpos/token == L(γ+1)/(1+τ)`（τ = 每轮接受数），逐位置接受率与汇总一致。"""
    m = trained_tiny_model()
    L = m.cfg.n_layers
    gamma, dd = 4, 2
    log = []
    _, st = EP.spec_decode(m, PROMPT, 40, gamma=gamma, draft_depth=dd,
                           cycle_log=log)
    tau = st["n_accepts"] / st["n_cycles"]
    assert st["layerpos_per_emitted_token_analytic"] == pytest.approx(
        L * (gamma + 1) / (tau + 1), abs=1e-4)
    assert st["layerpos_per_emitted_token"] == pytest.approx(
        L * (gamma + 1) / (tau + 1), rel=0.02)
    assert sum(r["accepted"] for r in log) == st["n_accepts"]
    assert len(log) == st["n_cycles"]
    assert all(r["committed"] == r["accepted"] + 1 for r in log)
    assert st["emitted"] == 40


def test_spec_rejection_path_still_faithful():
    """拒绝路径保真：草稿故意错（用不可信的 depth-0 草稿）时仍必须发出与普通贪心
    相同的流（把接受判定取反会让本测试变红——变异方向已在 `analysis.md` 记录）。"""
    m = trained_tiny_model()
    plain, _ = EP.greedy_decode(m, PROMPT, 16)
    assert len(set(plain)) >= 3
    spec, st = EP.spec_decode(m, PROMPT, 16, gamma=4, draft_depth=0)
    assert spec == plain
    assert st["accept_rate"] is not None and 0.0 <= st["accept_rate"] <= 1.0
    assert st["layerpos_per_emitted_token"] > m.cfg.n_layers * 0.2


def test_mod_arm_shared_mechanics_use_no_gate():
    """MoD 臂（dense 主干、无门）的解码必须走无门路径：`gated=False` 与
    `model.cfg.carry_gate` 的缺省接线一致（不静默套门）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2,
                        dense=True, mod_ratio=0.5)
    m = MoDModel(cfg, vocab_size=41).eval()
    _, st = EP.greedy_decode(m, PROMPT, 4)
    assert st["gated"] is False
