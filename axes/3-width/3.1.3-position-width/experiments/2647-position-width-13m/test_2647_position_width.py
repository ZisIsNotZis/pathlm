"""2647 · 同层内位置相关宽度 —— 等值门与变异单测（契约 §7，能变红）。

跑法：`python3 -m pytest -q test_2647_position_width.py`
（CPU、无数据依赖：合成长度 256 的随机 token 流；不取 GPU 锁）

门清单（与 `README.md` §7 一一对应）：

* T1 `w ≡ 256` ⇒ 前向与 stock **逐位相同**（同一 `torch.equal` logits）；
* T2 attention 调用与 stock **逐位相同**（`attn_mask is None` / `is_causal is True` /
  同形状 q/k/v）——「只改宽度、不改可见集合」的证据；
* T3 平均激活通道数 / 等效 MACs 与**独立重推**的解析值一致，且实测每位置参与前向的
  通道数/隐藏单元数 = 段宽（结构量有实测对应）；
* T4 **变异**：把窄段掩码取错维度（保留尾部 `[d−w:]` 而不是前缀 `[:w]`）⇒ 与正确参考
  **不相等**（否则 T1/T3 恒真）；
* T5 **活性**：窄段的块内增量对自身**尾部**通道扰动逐位不变，对**前缀**通道扰动必然改变
  （正对照 ⇒ T5 可红）。
"""

import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run as R                                                  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel          # noqa: E402
from pathlm.model import Block                                   # noqa: E402

T = R.SEQ_LEN
B = 2


def _tokens(seed: int = 5) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, R.VOCAB, (B, T), generator=g)


def _logits(model: DepthARModel, tokens: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tokens)[1]["depth_logits"][-1]


def _stock(seed: int = 0) -> DepthARModel:
    return R.build_model("FULL", R.VOCAB, seed, patch=False)


# ---------------------------------------------------------------- T1 等值门

def test_t1_full_width_forward_bitwise_equals_stock():
    """窄段不缩（w ≡ 256）时前向与 stock 逐位相同 ⇒ 干预在「关闭」位点是恒等。"""
    tokens = _tokens()
    stock = _stock()
    full = R.build_model("FULL", R.VOCAB, 0)
    assert full._pw_spec.all_full
    a = _logits(stock, tokens)
    b = _logits(full, tokens)
    assert torch.equal(a, b), f"max|Δ|={(a - b).abs().max().item()}"
    # 干预开着但某个位置不缩的臂，也必须与 stock 逐位相同
    model = R.build_model("PN", R.VOCAB, 0)
    assert model._pw_spec.narrow_mask(tokens.new_zeros(B, T, 1)).sum() == 128


def test_t1_patched_model_has_no_extra_or_removed_params():
    """总参数逐臂/逐门位点相同（不新增、不裁剪）。"""
    stock = _stock()
    ps = sum(p.numel() for p in stock.parameters())
    for arm in R.ALL_ARMS:
        m = R.build_model(arm, R.VOCAB, 0)
        assert sum(p.numel() for p in m.parameters()) == ps
        assert set(m.state_dict()) == set(stock.state_dict())
    assert ps == R.ANCHOR_PARAMS


# ---------------------------------------------------------------- T2 可见集合

def test_t2_attention_call_bitwise_identical_to_stock(monkeypatch):
    """PN（窄段真的缩）与 stock 的 SDPA 调用参数逐位相同：可见集合未被改动。"""
    calls = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        calls.append({"q_shape": tuple(q.shape), "k_shape": tuple(k.shape),
                      "v_shape": tuple(v.shape),
                      "attn_mask": None if kwargs.get("attn_mask") is None
                      else tuple(kwargs["attn_mask"].shape),
                      "is_causal": kwargs.get("is_causal")})
        return real(q, k, v, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    tokens = _tokens()
    _logits(_stock(), tokens)
    stock_calls = list(calls)
    calls.clear()
    _logits(R.build_model("PN", R.VOCAB, 0), tokens)
    pn_calls = list(calls)
    assert len(stock_calls) == R.N_LAYERS == len(pn_calls)
    assert stock_calls == pn_calls
    for c in stock_calls:
        assert c["attn_mask"] is None and c["is_causal"] is True
        assert c["q_shape"] == c["k_shape"] == c["v_shape"]
        assert c["q_shape"] == (B, R.N_HEADS, T, R.D_MODEL // R.N_HEADS)


def test_t2_narrow_segment_changes_logits():
    """只改宽度也必须真的改变前向（否则「无差异」结论没有意义：干预是活的）。"""
    tokens = _tokens()
    base = _logits(_stock(), tokens)
    pn = _logits(R.build_model("PN", R.VOCAB, 0), tokens)
    np_ = _logits(R.build_model("NP", R.VOCAB, 0), tokens)
    assert not torch.equal(base, pn)
    assert not torch.equal(base, np_)
    assert not torch.equal(pn, np_)


# ---------------------------------------------------------------- T3 代价解析

def _expected_macs(widths, d=R.D_MODEL, T=R.SEQ_LEN, L=R.N_LAYERS, V=R.VOCAB):
    """独立重推：每层每 token `4w² + 2Tw`（注意力）+ `8w²`（FFN），按位置平均。"""
    per_pos = [4 * w * w + 2 * T * w + 8 * w * w for w in widths]
    return L * (sum(per_pos) / len(per_pos)) + d * V


def test_t3_analytic_cost_matches_independent_derivation():
    for arm in R.ALL_ARMS:
        c = R.analytic_cost(arm)
        widths = R.WIDTH_BY_POS[arm]
        assert c["macs_per_token"]["total"] == pytest.approx(
            _expected_macs(widths), abs=1e-6)
        assert c["avg_activated_channels_per_token"] == pytest.approx(
            sum(widths) / len(widths))
    assert R.analytic_full_macs() == R.ANCHOR_MACS            # 口径与锚对账
    pn = R.analytic_cost("PN")
    assert pn["avg_activated_channels_per_token"] == 192.0
    assert pn["macs_per_token"]["total"] == R.analytic_cost("NP")["macs_per_token"]["total"]
    assert pn["macs_rel_to_full"] < 1.0
    assert pn["macs_rel_to_full"] == pytest.approx(0.645405, abs=1e-6)


def test_t3_measured_channels_match_segment_width():
    """实测：进入 qkv 的非零通道数 / mlp[0] 输出的非零隐藏单元数 = 段宽。"""
    arr = (np.arange(200_000) % R.VOCAB).astype(np.uint16)
    for arm, exp_ch, exp_hid in (
            ("PN", {"0..127": 256, "128..255": 128},
             {"0..127": 1024, "128..255": 512}),
            ("NP", {"0..127": 128, "128..255": 256},
             {"0..127": 512, "128..255": 1024})):
        m = R.build_model(arm, R.VOCAB, 0)
        prof = R.activation_profile(m, arr, n_batches=1, batch_size=2)
        assert prof["measured_nonzero_channels_into_qkv_by_segment"] == exp_ch
        assert prof["measured_nonzero_hidden_units_by_segment"] == exp_hid
        assert prof["nonzero_matches_spec"] is True
        assert all(v > 0 for v in
                   prof["block_delta_absmean_last_layer_by_segment"].values())


# ---------------------------------------------------------------- T4 变异（能变红）

class _SuffixMaskSpec(R.PositionWidthSpec):
    """变异：把参与前向的通道取成**尾部** `[d−w:]` 而不是前缀 `[:w]`。"""

    def keep_channels(self, x):
        idx = torch.arange(self.d, device=x.device)
        w = torch.as_tensor(self.width_by_pos, device=x.device)
        return (idx.view(1, 1, -1) >= (self.d - w).view(1, -1, 1))

    def keep_hidden(self, x):
        idx = torch.arange(self.hidden, device=x.device)
        w = torch.as_tensor(self.width_by_pos, device=x.device)
        return idx.view(1, 1, -1) >= (self.hidden - R.MLP_MULT * w).view(1, -1, 1)


def test_t4_wrong_axis_mask_is_detected():
    """掩码取错维度 ⇒ 与正确参考不等（门会红）、且与 stock 不等。"""
    tokens = _tokens()
    stock = _logits(_stock(), tokens)
    ref = _logits(R.build_model("PN", R.VOCAB, 0), tokens)
    mutant = R.build_model("PN", R.VOCAB, 0, patch=False)
    R.patch_position_width(mutant, "PN", spec=_SuffixMaskSpec("PN"))
    mut = _logits(mutant, tokens)
    assert not torch.equal(mut, ref)
    assert not torch.equal(mut, stock)
    # 同一变异的 FULL 位点也必须在「非恒等」方向之外：FULL 无窄段 ⇒ 仍走 stock
    full_mut = R.build_model("FULL", R.VOCAB, 0, patch=False)
    R.patch_position_width(full_mut, "FULL", spec=_SuffixMaskSpec("FULL"))
    assert torch.equal(_logits(full_mut, tokens), stock)


# ---------------------------------------------------------------- T5 活性（能变红）

def test_t5_narrow_segment_is_invariant_to_its_own_tail_channels():
    """窄段的块内增量对自身尾部通道扰动逐位不变；对前缀扰动必然改变（正对照）。"""
    arr = (np.arange(200_000) % R.VOCAB).astype(np.uint16)
    for arm in ("PN", "NP"):
        m = R.build_model(arm, R.VOCAB, 0)
        probe = R.narrow_isolation_probe(m, arr, n_batches=1, batch_size=2)
        assert probe["enabled"] is True
        assert probe["tail_invariant"] is True, probe
        assert probe["positive_control_prefix_changes"] is True, probe
    # 全宽臂上没有窄段 ⇒ 探针明确报「未启用」（不伪装成通过）
    assert R.narrow_isolation_probe(R.build_model("FULL", R.VOCAB, 0),
                                    arr, n_batches=1,
                                    batch_size=2)["enabled"] is False


def test_t6_train_mode_gradients_reach_every_parameter():
    """训练模式前向 + 反向：loss 有限；`blocks.*` 参数全部有非零梯度，且**零梯度参数集合
    与 stock 逐位相同**（`gates.*` 在 dense 形态下 stock 本来就没有梯度——预先存在，不是本干预
    引入的死参数）。"""

    def zero_grads(m):
        m.train()
        loss, _aux = m(toks_source, targets=toks_source)
        assert torch.isfinite(loss)
        loss.backward()
        return {n for n, p in m.named_parameters()
                if p.grad is None or not torch.isfinite(p.grad).all()
                or float(p.grad.abs().max()) == 0.0}

    toks_source = _tokens(3)
    ref_zero = zero_grads(_stock())
    for arm in R.ALL_ARMS:
        z = zero_grads(R.build_model(arm, R.VOCAB, 0))
        assert z == ref_zero, (arm, sorted(z ^ ref_zero))
        assert not any(n.startswith("blocks.") for n in z), (arm, sorted(z))


def test_t5_causal_and_readout_path_present():
    """因果性不变 + 干预挂在真实 `Block.forward` 位置上（挂点自检）。"""
    m = R.build_model("PN", R.VOCAB, 0)
    assert all(isinstance(b, Block) for b in m.blocks)
    assert all(hasattr(b, "_pw_spec") for b in m.blocks)
    assert m._pw_spec.width_by_pos[:R.SPLIT] == [256] * R.SPLIT
    c = R.causal_check(m, _tokens().to(next(m.parameters()).device))
    assert c["causal"] is True
    with pytest.raises(ValueError):
        # 非 dense / 带掩码路径必须显式拒绝，而不是静默走错
        R.position_width_forward(m.blocks[0], m._pw_spec, torch.zeros(B, T, R.D_MODEL),
                                 attn_mask=torch.zeros(B, R.N_HEADS, T, T, dtype=torch.bool))
    cfg = DepthARConfig(d_model=R.D_MODEL, n_layers=R.N_LAYERS, n_heads=R.N_HEADS,
                        seq_len=T, mlp_mult=R.MLP_MULT, dense=True)
    assert DepthARModel(cfg, R.VOCAB) is not None
