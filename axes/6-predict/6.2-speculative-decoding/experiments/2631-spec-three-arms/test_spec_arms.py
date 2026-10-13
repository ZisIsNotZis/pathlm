"""R6 · Spec 三臂的本地可失败单测（放在实验目录内：本波 `tests/**` 属 L1 线）。

跑法（仓库根）：

    python3 -m pytest \
      axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/test_spec_arms.py -q

覆盖四类不变量：
① **臂定义**：三臂分别读的是哪个头（chain → `chain2`、direct → `node2`）；
② **combined 是概率混合**（不是 logits 线性组合），且 λ 真的改变选择；
③ 第二个草稿位（N+3）在三臂中**固定**（共享背景，不属于自变量）；
④ **账目恒等式**（emitted = 1 + accepted、τ、tok/fwd）与**逐位一致门禁**在真引擎上成立。
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import spec_arms as S  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.decode import decode  # noqa: E402
from pathlm.model import PathLM  # noqa: E402


# ---------------------------------------------------------------- fixtures

V = 4


def _onehot_like(idx: int, logit: float = 2.0) -> torch.Tensor:
    t = torch.zeros(1, 1, V)
    t[0, 0, idx] = logit
    return t


def fake_nodes():
    """可分辨的 MTP 节点：chain2 → token 1、node2 → token 2、node3 → token 3。"""
    chain2 = torch.tensor([0.0, 2.0, 0.0, 0.0]).view(1, 1, V)
    direct2 = torch.tensor([0.0, 0.0, 1.5, 0.0]).view(1, 1, V)
    return {
        0: {"logits": _onehot_like(0)},
        1: {"logits": _onehot_like(0), "chain2": {"logits": chain2}},
        2: {"logits": direct2},
        3: {"logits": _onehot_like(3)},
    }


def tiny_model(seed: int = 0) -> PathLM:
    torch.manual_seed(seed)
    mcfg = ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2)
    pcap = PathConfig(n_mtp=3, chain_mode="token", transport="none")
    return PathLM(mcfg, pcap, vocab_size=41).eval()


# ---------------------------------------------------------------- ① 臂定义

def test_chain_arm_reads_the_conditional_head():
    """`chain` 臂的分布必须**逐位等于** `node1.chain2` 的 logits（不是 node2）。"""
    nodes = fake_nodes()
    got = S.second_slot_logits(nodes, "chain")
    assert torch.allclose(got, nodes[1]["chain2"]["logits"][0, -1].float())
    assert int(got.argmax()) == 1, "chain 臂必须选 chain2 的 argmax"


def test_direct_arm_reads_the_direct_head():
    nodes = fake_nodes()
    got = S.second_slot_logits(nodes, "direct")
    assert torch.allclose(got, nodes[2]["logits"][0, -1].float())
    assert int(got.argmax()) == 2, "direct 臂必须选 node2 的 argmax"


def test_chain_and_direct_disagree_on_the_fixture():
    """自变量真的活着：两臂在同一节点上给出**不同**的草稿 token。"""
    nodes = fake_nodes()
    assert int(S.second_slot_logits(nodes, "chain").argmax()) != \
        int(S.second_slot_logits(nodes, "direct").argmax())


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        S.second_slot_logits(fake_nodes(), "latent_chain")


# ---------------------------------------------------------------- ② 概率混合

def test_combined_is_a_probability_mixture_not_a_logit_sum():
    """`combined` 必须是 `λ·softmax(chain2) + (1−λ)·softmax(node2)`：
    ① 逐元素相等；② 归一化；③ **不等于** logits 的线性组合。"""
    nodes = fake_nodes()
    lam = 0.5
    pc = nodes[1]["chain2"]["logits"][0, -1].float().softmax(-1)
    pd = nodes[2]["logits"][0, -1].float().softmax(-1)
    got = S.second_slot_logits(nodes, "combined", lam)
    assert torch.allclose(got, lam * pc + (1 - lam) * pd, atol=1e-6)
    assert abs(float(got.sum()) - 1.0) < 1e-5, "混合结果必须仍是概率分布"
    logit_sum = lam * nodes[1]["chain2"]["logits"][0, -1].float() \
        + (1 - lam) * nodes[2]["logits"][0, -1].float()
    assert not torch.allclose(got, logit_sum, atol=1e-3), "概率混合 ≠ logits 线性组合"


def test_combined_lambda_moves_the_choice():
    """λ 是这个臂的**真旋钮**：λ→1 时选 chain 的 token，λ→0 时选 direct 的 token。"""
    nodes = fake_nodes()
    assert int(S.second_slot_logits(nodes, "combined", 0.9).argmax()) == 1
    assert int(S.second_slot_logits(nodes, "combined", 0.1).argmax()) == 2
    assert S.COMBINED_LAM == 0.5, "预登记的混合系数是 0.5，不许事后调"


# ---------------------------------------------------------------- ③ 第二草稿位固定

def test_second_draft_slot_is_fixed_across_arms():
    """第 N+3 位（`max_drafts` 的第 2 项）在三臂中必须**逐位相同**——它是共享背景。"""
    nodes = fake_nodes()
    third = {arm: S.make_draft_fn(arm)(nodes, S.GAMMA)[1] for arm in S.ARMS}
    assert set(third.values()) == {3}
    first = {arm: S.make_draft_fn(arm)(nodes, S.GAMMA)[0] for arm in S.ARMS}
    assert first["chain"] != first["direct"], "第 N+2 位必须随臂变"


def test_make_draft_fn_first_slot_equals_the_arm_distribution():
    """`make_draft_fn` 的第 1 项必须**就是**该臂分布的 argmax（臂接线不能串线）。"""
    nodes = fake_nodes()
    for arm in S.ARMS:
        want = int(S.second_slot_logits(nodes, arm).argmax())
        assert S.make_draft_fn(arm)(nodes, S.GAMMA)[0] == want, f"{arm} 臂接线串了"


def test_gamma_matches_the_verification_width():
    """γ=2 ⇒ 每次验证喂 K=1+γ=3 个 token（width-3 前向），算力账用实测 w3/w1。"""
    assert S.GAMMA == 2


# ---------------------------------------------------------------- ④ 账目 + 真引擎门禁

def test_round_metrics_identities():
    accepts = [2, 0, 1, 2, 0, 1, 1]
    n_new = sum(1 + a for a in accepts)
    m = S.round_metrics(accepts, n_new)
    assert m["n_rounds"] == len(accepts)
    assert m["alpha1"] == pytest.approx(5 / 7)
    assert m["alpha2_given_alpha1"] == pytest.approx(2 / 5)
    assert m["tau"] == pytest.approx(7 / 7)
    assert m["tokens_per_forward"] == pytest.approx(n_new / len(accepts))
    assert m["tokens_per_forward"] == pytest.approx(1 + m["tau"]), \
        "tok/fwd == 1 + τ 是每轮发出结构的恒等式"


def test_arm_decode_stream_is_bit_identical_to_native():
    """**门禁**：argmax 路线下三臂的发出流必须与原生贪心流逐位相同。"""
    m = tiny_model(seed=0)
    prompt = torch.tensor([3, 11, 7, 2, 19, 5, 23, 1])
    pcap = m.pcap
    native, _ = decode(m, prompt, 24, pcap)
    for arm in S.ARMS:
        gen, st = S.arm_decode(m, prompt, 24, pcap, arm)
        assert gen == native, f"{arm} 臂的发出流与原生贪心不一致"
        assert st["n_forward"] >= 1
        for a in st["accepts"]:
            assert 0 <= a <= S.GAMMA


def test_arm_decode_rejects_short_n_mtp():
    """γ=2 需要 node3；`n_mtp<=2` 必须**报错**而不是静默退化到 direct。"""
    torch.manual_seed(0)
    m = PathLM(ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2),
               PathConfig(n_mtp=2, chain_mode="token", transport="none"),
               vocab_size=41).eval()
    with pytest.raises(ValueError):
        S.arm_decode(m, torch.tensor([1, 2, 3]), 8, m.pcap, "chain")


def test_arms_read_different_distributions_in_the_real_engine():
    """活性量（SCHEMA §10）：在**真模型**上，三臂第 N+2 位读到的分布彼此不同，
    且 chain 臂确实**以第一个草稿 token 为条件**（改 node1 的条件 ⇒ chain2 变）。"""
    m = tiny_model(seed=1)
    prompt = torch.tensor([3, 11, 7, 2, 19, 5, 23, 1])
    with torch.no_grad():
        for tok in prompt.tolist()[:-1]:
            pass
        from pathlm.decode import Decoder
        d = Decoder(m)
        for tok in prompt.tolist():
            d.step(tok)
        nodes = d.last_nodes
        pc = S.second_slot_logits(nodes, "chain")
        pd = S.second_slot_logits(nodes, "direct")
        assert not torch.allclose(pc, pd), "chain 与 direct 读的必须是**不同**的头"

        # 条件活性：换掉条件 token 的 embedding 再算 chain2，结果必须变。
        h = torch.zeros(1, 1, m.mcfg.d_model)
        cond_a = m.chain_cat(torch.cat([h, m.embed(torch.tensor([[1]]))], dim=-1))
        cond_b = m.chain_cat(torch.cat([h, m.embed(torch.tensor([[2]]))], dim=-1))
        assert not torch.allclose(m.transforms[0](cond_a), m.transforms[0](cond_b)), \
            "chain2 必须真的以 N+1 的 token 为条件（否则退化成无条件头）"


def test_chain_arm_is_not_the_latent_T_composition():
    """反例警示的**实现层**钉子：`chain` 臂读的 `chain2` 与 `T1∘T1`（潜在复用）
    不是同一个张量（后者是 `chain_mode="latent"` 的语义，本实验不训）。"""
    m = tiny_model(seed=2)
    with torch.no_grad():
        h = torch.randn(1, 1, m.mcfg.d_model)
        nodes = m._mtp_nodes(h)                       # token 模式 ⇒ chain2 走条件分支
        t_compose = m.transforms[0](m.transforms[0](h))   # T1∘T1（latent 语义）
        assert not torch.allclose(nodes[1]["chain2"]["latent"], t_compose, atol=1e-5), \
            "token 条件链 ≠ T∘T 潜在复用"


def test_mtp_head_macs_is_arm_independent():
    """算力账的头部项与臂无关（四臂共享同一 forward 的同一组头）。"""
    cfg = ModelConfig(d_model=64, n_layers=4, n_heads=2, seq_len=64, mlp_mult=2)
    a = S.mtp_head_macs(cfg, vocab_size=41)
    b = S.mtp_head_macs(cfg, vocab_size=41)
    assert a == b
    assert a["head_macs_per_position"] > 0
    assert a["head_over_trunk_frac"] > 0
    two = S.mtp_head_macs(cfg, vocab_size=41, n_conditioning_heads=2)
    assert two["head_macs_per_position"] > a["head_macs_per_position"], \
        "多一级条件头必须更贵（解析项要能分辨）"
