"""2674-mdiffusion-13m · 采样器可失败单测（CPU、秒级）。

覆盖契约要求的 gate probe：**K=1 退化等价**、**步数/位置数守恒**、**无 M 位置真值泄漏**、**确定性**、
bpc 口径。另覆盖 `mdiff.py` 的 `mdiff_metrics` 与 `single_forward_fill_ce`。

每条仪器都自带**可失败性**证据：`test_no_true_leak...` 内联一个故意泄漏的采样器并断言其被抓红；
其余检查由 `mutate.py` 的变异证明（变异靶子见下）。

    python3 -m pytest test_mdiff.py -q
"""

import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mdiff  # noqa: E402
import padfill as P  # noqa: E402


def _tiny(seed: int = 0) -> P.PadFillLM:
    """小号交付结构（同 attn_mask 语义，只是更小更快），固定种子 ⇒ 确定性。"""
    torch.manual_seed(seed)
    return P.PadFillLM(attn_mask="full", d_model=32, n_layers=2, n_heads=4, mlp_mult=2)


class _Recorder(torch.nn.Module):
    """记录每次前向的输入（用于泄漏检查）；不改变模型行为。"""

    def __init__(self, m: torch.nn.Module):
        super().__init__()
        self.m = m
        self.inputs: list[torch.Tensor] = []

    @property
    def mask_token(self) -> int:
        return self.m.mask_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return self.m(x)


# ---- K=1 退化等价（契约 §2 的连接点）------------------------------------------
def test_K1_degenerates_to_single_forward():
    T = 16
    torch.manual_seed(3)
    true = torch.randint(0, P.VOCAB_SIZE - 1, (1, T))
    M = torch.ones(1, T, dtype=torch.bool)
    M[0, :3] = False
    m = _tiny()
    res = mdiff.sample_denoise_k(m, true, M, K=1)
    ref = mdiff.single_forward_fill_ce(m, true, M)
    assert res["n_forwards"] == 1
    # K=1：一次前向、全部 M 一步定格 ⇒ 逐位置 CE 与「一次前向填全部 M」严格相同
    assert torch.equal(res["fix_ce"][M], ref[M])


# ---- 步数 / 位置数守恒 ---------------------------------------------------------
def test_step_counts_conserve():
    T = 17
    torch.manual_seed(4)
    true = torch.randint(0, P.VOCAB_SIZE - 1, (1, T))
    M = torch.ones(1, T, dtype=torch.bool)
    n = int(M.sum())
    m = _tiny()
    for K in (1, 4, 8, 16):
        res = mdiff.sample_denoise_k(m, true, M, K=K)
        assert res["n_forwards"] == K
        assert sum(res["step_fix"]) == n                       # 定格总数 = |M|
        assert res["step_fix"][0] == math.ceil(n / K)          # 首步定格 ceil(|M|/K)
        assert int((res["fix_step"][M] >= 1).sum()) == n       # 每个 M 位置至少定格一次
        assert int((res["fix_step"][M] <= K).sum()) == n       # 且不迟于第 K 步
        # 每个 M 位置恰好定格一次：fix_step 在 M 上全为正
        assert bool((res["fix_step"][M] > 0).all())


# ---- 无 M 位置真值泄漏（模型输入对 M 真值不敏感）-----------------------------
def _leaky_denoise(model, true_tokens, M, K):
    """故意泄漏：把定格位置写成**真 token**（gate probe 靶子），应被下面的检查抓红。"""
    B, T = true_tokens.shape
    import torch.nn.functional as F
    x = true_tokens.clone()
    x[M] = model.mask_token
    fixed = torch.zeros(B, T, dtype=torch.bool)
    n_m = M.sum(dim=1)
    per_step = torch.ceil(n_m.float() / K).long()
    for k in range(1, K + 1):
        logits = model(x)
        logp = F.log_softmax(logits.float(), dim=-1)
        conf = logp.exp().max(dim=-1).values
        remaining = M & (~fixed)
        conf = conf.masked_fill(~remaining, -1.0)
        n_fix = remaining.sum(dim=1) if k == K else torch.clamp(per_step, max=remaining.sum(dim=1))
        order = torch.argsort(conf, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        ar = torch.arange(T).unsqueeze(0).expand(B, T)
        rank.scatter_(1, order, ar)
        sel = remaining & (rank < n_fix.unsqueeze(1))
        x = torch.where(sel, true_tokens, x)      # 泄漏
        fixed = fixed | sel
    return x


def _inputs_for(model, true, M, K):
    rec = _Recorder(model)
    mdiff.sample_denoise_k(rec, true, M, K=K)
    return rec.inputs


def test_no_true_leak_into_context():
    T = 16
    torch.manual_seed(5)
    true = torch.randint(0, P.VOCAB_SIZE - 1, (1, T))
    M = torch.ones(1, T, dtype=torch.bool)
    true2 = true.clone()
    true2[0, 9] = (true[0, 9] + 7) % (P.VOCAB_SIZE - 1)   # 只改一个 M 位置的真值
    m = _tiny()
    ins1 = _inputs_for(m, true, M, 4)
    ins2 = _inputs_for(m, true2, M, 4)
    assert len(ins1) == 4 and len(ins2) == 4
    for a, b in zip(ins1, ins2):
        assert torch.equal(a, b)                 # 输入对 M 位置真值不敏感 ⇒ 无泄漏
    assert bool((ins1[0][M] == m.mask_token).all())   # 第 1 步 M 全为 [mask]
    # gate probe：泄漏采样器必须被同一检查抓红
    leaky_inputs = []
    rec = _Recorder(m)
    _leaky_denoise(rec, true, M, 4)
    leaky_inputs = rec.inputs
    differs = any(not torch.equal(a, b) for a, b in
                  zip(leaky_inputs, _inputs_for(m, true2, M, 4)))
    assert differs, "gate probe 未变红：泄漏构造未被抓出"


# ---- 确定性 -------------------------------------------------------------------
def test_determinism_same_seed_same_values():
    T = 16
    torch.manual_seed(6)
    true = torch.randint(0, P.VOCAB_SIZE - 1, (1, T))
    M = torch.ones(1, T, dtype=torch.bool)
    m = _tiny()
    a = mdiff.sample_denoise_k(m, true, M, K=4)
    b = mdiff.sample_denoise_k(m, true, M, K=4)
    assert torch.equal(a["x_final"], b["x_final"])
    assert torch.equal(torch.nan_to_num(a["fix_ce"]), torch.nan_to_num(b["fix_ce"]))
    assert a["step_fix"] == b["step_fix"]


# ---- bpc / NELBO 口径 ---------------------------------------------------------
def test_mdiff_metrics_bpc_log2():
    """bpc = (ΣCE)/(n·ln2)；用构造值验证 log 底为 2（ln2，不是 ln10）。"""
    n = 4
    res = {"s_mdiff": torch.tensor([3.0 * n * math.log(2)]),
           "n_m": torch.tensor([n]),
           "step_ce_mean_bpc": [3.0], "step_n": [n], "step_fix": [n]}
    out = mdiff.mdiff_metrics(res)
    assert out["mdiff_M_bpc"] == pytest.approx(3.0)
    assert out["nelbo_step_mean_bpc"] == pytest.approx(3.0)
    assert out["last_step_bpc"] == pytest.approx(3.0)


# ---- run.py 纯函数（参考带 / Δ / 代价轴 / 禁墙钟）------------------------------
def test_run_reference_band_and_delta():
    import run
    assert run.REFERENCE_BAND == 0.014
    assert run.exceeds_reference_band(0.014) is False
    assert run.exceeds_reference_band(0.0140001) is True
    assert run.ci_excludes_zero(0.01, 0.02) is True
    assert run.ci_excludes_zero(-0.01, 0.02) is False
    n = 128
    s_ar = 10.0 * n * math.log(2)
    assert run.delta_bpc_block(s_ar + 0.5 * n * math.log(2), s_ar, n) == pytest.approx(0.5)


def test_run_cost_axis_matches_contract():
    import run
    c = run.cost_axis_ar()
    assert c["tokens_per_forward"] == 1.0
    assert c["sequential_forwards_per_block"] == float(P.SEQ_LEN - 1)
    assert c["equivalent_flops_per_token"] == float(P.N_LAYERS)
    cm = run.cost_axis_mdiff(K=4, n_m_total=int(0.25 * P.SEQ_LEN) * 64, blocks=64)
    assert cm["tokens_per_forward"] == pytest.approx(0.25 * P.SEQ_LEN / 4)
    assert cm["sequential_forwards_per_block"] == 4.0
    assert cm["equivalent_flops_per_token"] == 4.0 * P.N_LAYERS


def test_run_wall_clock_and_mean_over_seeds():
    import run
    assert run.wall_clock_fields({"steps": 1500, "arms": ["mdiff"], "params": 12.8}) == []
    assert run.wall_clock_fields({"wall_clock_s": 1.0}) == [".wall_clock_s"]
    assert run.mean_over_seeds([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_single_forward_fill_ce_shapes_and_mask_insensitivity():
    T = 12
    torch.manual_seed(7)
    true = torch.randint(0, P.VOCAB_SIZE - 1, (1, T))
    M = torch.ones(1, T, dtype=torch.bool)
    m = _tiny()
    ce = mdiff.single_forward_fill_ce(m, true, M)
    assert ce.shape == (1, T)
    true2 = true.clone()
    true2[0, 5] = (true[0, 5] + 3) % (P.VOCAB_SIZE - 1)
    ce2 = mdiff.single_forward_fill_ce(m, true2, M)
    # 输入对 M 真值不敏感（全 mask）；CE 只在被改位置不同
    assert torch.allclose(ce[0, 0:5], ce2[0, 0:5])
    assert ce[0, 5] != ce2[0, 5]
