"""冻结主干 + 跳跃头（H4 线 A）的可失败单测。Run: python -m pytest tests/ -q

钉住四件在本项目里曾出错/易错的事：
- **shift 方向**：位置 i 的跳跃头读出预测 t_{i+2}（本项目 0 级 review 缺陷：
  off-by-one）。test_shift2_targets_are_t_plus_2 用合成 logits 同时验证
  shift=2 的 CE≈0 与 shift=1 的 CE 很大（方向错了必红）。
- **tied 读出语义**：JumpHead 的 logits == transform(h) @ Eᵀ，与主干
  readout(transform(h)) 逐位一致（同一概率几何）。
- **主干表征等价**：trunk_hidden 与 DepthARModel dense forward 的末态逐位一致。
- **冻结断言**：主干 requires_grad=False 且反传后 .grad is None；
  变异验证（放开主干梯度）时 assert_frozen 必须抓红。
"""

import os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import DepthARConfig, DepthARModel
from train_jumphead import (HeadBank, JumpHead, ar1_second_step_readout,
                            assert_frozen, freeze_trunk, init_heads, shift_pairs,
                            trunk_hidden, trunk_layer_inputs)


def tiny_trunk(vocab: int = 12) -> DepthARModel:
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12, dense=True)
    return DepthARModel(cfg, vocab_size=vocab)


def test_shift2_targets_are_t_plus_2():
    """位置 i 的读出预测 t_{i+2}：shift=2 的配对 CE≈0，shift=1 必须远大于它。"""
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15]])
    V = 16
    # 合成 logits：行 i 把 tokens[i+2] 打成高置信——只有 shift=2 才对齐
    lg = torch.zeros(1, tokens.shape[1], V)
    idx = torch.arange(tokens.shape[1] - 2)
    lg[0, idx, tokens[0, 2:]] = 20.0
    l2, y2 = shift_pairs(lg, tokens, 2)
    assert l2.shape[:2] == (1, 4) and y2.shape == (1, 4)
    assert torch.equal(y2, tokens[:, 2:]), "shift=2 的目标必须是 t_{i+2}"
    assert F.cross_entropy(l2.reshape(-1, V), y2.reshape(-1)).item() < 1e-3
    l1, y1 = shift_pairs(lg, tokens, 1)
    assert torch.equal(y1, tokens[:, 1:]), "shift=1 的目标必须是 t_{i+1}"
    # 方向错（shift=1）时同一批 logits 必然很差——这是 off-by-one 的抓红条件
    assert F.cross_entropy(l1.reshape(-1, V), y1.reshape(-1)).item() > 0.5


def test_jumphead_tied_readout_matches_trunk_readout():
    """JumpHead 复用主干表：logits == transform(h) @ Eᵀ == trunk.readout(transform(h))。"""
    trunk = tiny_trunk()
    head = JumpHead(trunk.cfg.d_model)
    torch.manual_seed(1)
    h = torch.randn(2, 5, trunk.cfg.d_model)
    with torch.no_grad():
        z = head.transform(h)
        logits = head(h, trunk.embed.weight)
        expected = trunk.readout(z)
    assert logits.shape == (2, 5, trunk.vocab_size)
    assert torch.allclose(logits, expected, atol=1e-5), "tied 读出必须与主干 readout 逐位一致"
    # 表变了读出必须跟着变（确认真的 tied，而非自带头）
    with torch.no_grad():
        trunk.embed.weight[0] += 1.0
        logits2 = head(h, trunk.embed.weight)
    assert not torch.allclose(logits, logits2)


def test_trunk_hidden_matches_forward_states():
    """trunk_hidden 必须等于 dense forward 的最终状态（不重写一套层语义）。"""
    trunk = tiny_trunk()
    trunk.eval()
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    with torch.no_grad():
        h = trunk_hidden(trunk, x)
        _, aux = trunk(x, exit_depths=None)
    assert torch.allclose(h, aux["states"][-1], atol=1e-6)


def test_freeze_trunk_has_no_grad_after_backward():
    """冻结断言：requires_grad=False 且一次反传后主干 .grad is None。"""
    trunk = tiny_trunk()
    freeze_trunk(trunk)
    ok, info = assert_frozen(trunk)
    assert ok, info
    bank = HeadBank(trunk.cfg.d_model)
    init_heads(bank)
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    with torch.no_grad():
        h = trunk_hidden(trunk, x)
    l2, y2 = shift_pairs(bank.t2(h, trunk.embed.weight), x, 2)
    loss = F.cross_entropy(l2.reshape(-1, trunk.vocab_size), y2.reshape(-1))
    loss.backward()
    ok, info = assert_frozen(trunk)
    assert ok, f"冻结主干在反传后必须仍无梯度，offenders={info}"
    # 头自己有梯度（冻结的是主干，不是头）
    assert any(p.grad is not None for p in bank.parameters())


def test_unfrozen_trunk_is_caught_by_freeze_assertion():
    """变异验证：故意不冻结主干（放开梯度），assert_frozen 必须抓红。"""
    trunk = tiny_trunk()
    # 变异：跳过 freeze_trunk —— 主干参数保持 requires_grad=True
    ok, info = assert_frozen(trunk)
    assert not ok and info["requires_grad_true"], "未冻结的主干必须被冻结断言抓住"
    bank = HeadBank(trunk.cfg.d_model)
    init_heads(bank)
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    h = trunk_hidden(trunk, x)                      # 无 no_grad：梯度穿主干
    l2, y2 = shift_pairs(bank.t2(h, trunk.embed.weight), x, 2)
    F.cross_entropy(l2.reshape(-1, trunk.vocab_size), y2.reshape(-1)).backward()
    ok, info = assert_frozen(trunk)
    assert not ok, "变异后 assert_frozen 必须为 False（否则冻结断言失效）"
    assert info["grad_not_none"], "放开梯度后主干应有 .grad，冻结断言必须看到"


# ------------------------------------------------- ticket 29-A：诚实 1-step AR 对照


def _bruteforce_ar1_row(trunk, x: torch.Tensor, a1: torch.Tensor, i: int) -> torch.Tensor:
    """逐行参考实现：只把位置 i+1 换成 a1[:,i]，其余保留真值，标准前向读位置 i+1。"""
    xp = x.clone()
    xp[:, i + 1] = a1[:, i]
    with torch.no_grad():
        return trunk.readout(trunk_hidden(trunk, xp))[:, i + 1]


def test_trunk_layer_inputs_final_matches_trunk_hidden():
    """trunk_layer_inputs 的末态必须逐位等于 trunk_hidden（不重写层语义）。"""
    trunk = tiny_trunk()
    trunk.eval()
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    with torch.no_grad():
        inputs, final = trunk_layer_inputs(trunk, x)
        h = trunk_hidden(trunk, x)
    assert len(inputs) == len(trunk.blocks)
    assert torch.allclose(final, h, atol=1e-6)


def test_ar1_matches_bruteforce_rowloop():
    """AR-1 快路径必须与「逐行各跑一次标准前向」逐位等价（只替换位置 i+1）。

    这是 correctness 钉子：mask / 前缀 K/V / 自注意力若写错，逐行对不上必红。
    """
    trunk = tiny_trunk()
    trunk.eval()
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    with torch.no_grad():
        a1 = trunk.readout(trunk_hidden(trunk, x)).argmax(-1)
        fast = ar1_second_step_readout(trunk, x, a1)
    assert fast.shape == (2, x.shape[1] - 2, trunk.vocab_size)
    for i in range(x.shape[1] - 2):
        bf = _bruteforce_ar1_row(trunk, x, a1, i)
        assert torch.allclose(bf, fast[:, i], atol=1e-4), \
            f"AR-1 快路径与逐行前向在行 {i} 不一致"


def test_ar1_prefix_keeps_ground_truth_not_whole_context():
    """回归：AR-1 只替换紧邻位置 i+1，前缀 0..i 保持真值。

    旧 ticket 28 deploy 把**整段** xp[:,1:] 都换成自预测——行 0 恰好只替换了
    位置 1（两者一致），但行 ≥1 的前缀被污染（两者必须不同）。任何把前缀
    也替换掉的实现都会在这里被抓红。"""
    trunk = tiny_trunk()
    trunk.eval()
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    T = x.shape[1]
    with torch.no_grad():
        L = trunk.readout(trunk_hidden(trunk, x))
        a1 = L.argmax(-1)
        fast = ar1_second_step_readout(trunk, x, a1)
        xp = x.clone()
        xp[:, 1:] = L[:, :-1].argmax(-1)          # 旧 bug：整段替换
        buggy = trunk.readout(trunk_hidden(trunk, xp))[:, 1:T - 1]
    assert torch.allclose(fast[:, 0], buggy[:, 0], atol=1e-4), "行 0 只替换位置 1，二者应一致"
    assert not torch.allclose(fast[:, 1], buggy[:, 1], atol=1e-4), \
        "行 ≥1 前缀被污染：整段替换与只替换紧邻位置必须不同"


def test_ar1_is_live_to_fed_token():
    """活性：第二次前向确实吃了 a1（改变 fed token 必须改变输出），排除死通路。"""
    trunk = tiny_trunk()
    trunk.eval()
    x = torch.randint(0, trunk.vocab_size, (2, 12))
    with torch.no_grad():
        a1 = trunk.readout(trunk_hidden(trunk, x)).argmax(-1)
        a1b = (a1 + 1) % trunk.vocab_size
        o1 = ar1_second_step_readout(trunk, x, a1)
        o2 = ar1_second_step_readout(trunk, x, a1b)
    assert not torch.allclose(o1, o2), "改变 fed token 输出必须变化（否则 AR-1 通路是死的）"


# --------------------------------- 契约 30-G：自门控多跳组合（H3）可失败单测
#
# 钉住四件：① direct/composed 共用同一个 T 且参数量逐位相同；② composed 就是
# T∘T；③ gated 的混合公式与逐通道 sigmoid；④ 门的单边零起步（g≡0.5、fc2=0、
# fc1 随机）与首步逃逸（变异：两侧全零 ⇒ 永远停在 0 = 鞍点）。

import numpy as np

from train_jumphead import (COMPOSE_MODES, CompositionHead, evaluate_composition,
                            init_composition_head, tied_logits)


def _composition_heads(d: int = 16) -> dict:
    heads = {m: CompositionHead(d, m) for m in COMPOSE_MODES}
    for h in heads.values():
        init_composition_head(h)
    return heads


def test_composed_is_same_T_applied_twice_and_params_match_direct():
    """composed = 同一个 T 作用两次（T∘T），参数量与 direct 逐位相同。"""
    d = 16
    direct, composed = CompositionHead(d, "direct"), CompositionHead(d, "composed")
    init_composition_head(direct)
    init_composition_head(composed)
    composed.T.load_state_dict(direct.T.state_dict())        # 同一个 T
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(direct) == n(composed), "composed 必须是同一个 T 作用两次，不是两个变换"
    torch.manual_seed(3)
    h = torch.randn(2, 4, d)
    E = torch.randn(9, d)
    with torch.no_grad():
        z1, z2 = composed.branches(h)
        assert torch.allclose(z1, direct.T(h))
        assert torch.allclose(z2, direct.T(direct.T(h))), "composed 必须是 T(T(h))"
        z, g = direct.mixture(h)
        assert g is None and torch.allclose(z, direct.T(h)), "direct 只作用一次"
        assert not torch.allclose(direct(h, E), composed(h, E)), "两臂读出必须不同"


def test_gated_mix_formula_is_per_channel_sigmoid_of_G():
    """gated: h_mix = g⊙T(T(h)) + (1−g)⊙T(h)，g = sigmoid(G(h)) 逐通道 ∈ (0,1)。"""
    d = 16
    head = CompositionHead(d, "gated")
    init_composition_head(head)
    torch.manual_seed(5)
    with torch.no_grad():                     # 打破单边零，验证非平凡 g 下公式仍成立
        head.G.fc2.weight.normal_(std=0.5)
        head.G.fc2.bias.normal_(std=0.5)
    h = torch.randn(2, 5, d)
    with torch.no_grad():
        z, g = head.mixture(h)
        z1, z2 = head.branches(h)
        g_exp = torch.sigmoid(head.G(h))
    assert torch.allclose(g, g_exp)
    assert torch.allclose(z, g_exp * z2 + (1.0 - g_exp) * z1)
    assert float(g_exp.min()) > 0.0 and float(g_exp.max()) < 1.0
    assert not torch.allclose(g, torch.full_like(g, 0.5))


def _abs_sum(p: torch.Tensor) -> float:
    """权重绝对值之和（detach，避免 float(Tensor) 的告警）。"""
    return float(p.detach().abs().sum())


def _one_step(head: CompositionHead, h: torch.Tensor, y: torch.Tensor,
              E: torch.Tensor, lr: float = 1e-2) -> float:
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss = F.cross_entropy(head(h, E).reshape(-1, E.shape[0]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.detach())


def test_gate_single_sided_zero_starts_at_half_then_escapes():
    """§10 单边零：起步 G≡0 ⇒ g≡0.5（两臂 50/50）、fc2=0、fc1 随机；首步必须逃逸。

    变异（输入侧也置零）⇒ 梯度恒为 0、fc2 永远停在 0——这正是要避免的鞍点。
    """
    d = 16
    torch.manual_seed(7)
    head = CompositionHead(d, "gated")
    init_composition_head(head)
    h, y, E = torch.randn(4, 6, d), torch.randint(0, 7, (4, 6)), torch.randn(7, d)
    with torch.no_grad():
        g = torch.sigmoid(head.G(h))
        assert torch.allclose(g, torch.full_like(g, 0.5)), "单边零起步必须 g≡0.5"
        z, _ = head.mixture(h)
        z1, z2 = head.branches(h)
        assert torch.allclose(z, 0.5 * (z1 + z2)), "起步 = 两臂 50/50 混合"
    assert _abs_sum(head.G.fc2.weight) == 0.0, "输出侧必须置零（单边零）"
    assert _abs_sum(head.G.fc1.weight) > 0.0, "输入侧必须随机（不是全零初始化）"
    _one_step(head, h, y, E)
    assert _abs_sum(head.G.fc2.weight) > 0.0, "输出侧必须在首步逃逸（否则是鞍点）"

    bad = CompositionHead(d, "gated")
    init_composition_head(bad)
    with torch.no_grad():
        bad.G.fc1.weight.zero_(); bad.G.fc1.bias.zero_()
        bad.G.fc2.weight.zero_(); bad.G.fc2.bias.zero_()
    _one_step(bad, h, y, E)
    assert _abs_sum(bad.G.fc2.weight) == 0.0, \
        "两侧全零在乘性结构上是鞍点（梯度恒 0，永不逃逸）"


def test_gate_perturbation_changes_readout():
    """活性：扰动门参数必须改变读出（门通路不是死的）。"""
    d = 16
    torch.manual_seed(9)
    head = CompositionHead(d, "gated")
    init_composition_head(head)
    h, E = torch.randn(2, 4, d), torch.randn(9, d)
    with torch.no_grad():
        o0 = head(h, E)
        head.G.fc2.weight.normal_(std=1.0)
        o1 = head(h, E)
    assert not torch.allclose(o0, o1), "门参数变化必须改变输出（否则门是死通路）"


def test_eq_control_matches_gated_params_bitwise():
    """§15：direct_eq（去掉门与组合）与 gated 参数量逐位相同；direct 更少。"""
    d = 256
    heads = _composition_heads(d)
    n = {m: sum(p.numel() for p in h.parameters()) for m, h in heads.items()}
    assert n["direct"] == n["composed"] == 4 * d * d + 3 * d
    assert n["gated"] == n["direct_eq"] == 6 * d * d + 5 * d
    assert n["gated"] - n["direct"] == 2 * d * d + 2 * d, "门是 gated 唯一多出的参数"
    h, E = torch.randn(2, 3, d), torch.randn(9, d)
    with torch.no_grad():
        outs = {m: hh(h, E) for m, hh in heads.items()}
    assert not torch.allclose(outs["direct"], outs["direct_eq"])


def test_composition_eval_shared_positions_and_g_report():
    """评测管线（CPU 小模型）：所有臂吃同一位置集，gated 另报 g 分布与分支命中。"""
    trunk = tiny_trunk()
    trunk.eval()
    heads = _composition_heads(trunk.cfg.d_model)
    rng = np.random.default_rng(0)
    eval_arr = rng.integers(0, trunk.vocab_size, size=4096).astype(np.int64)
    ecfg = {"seed": 7, "n_batches": 2, "batch_size": 2}
    out = evaluate_composition(trunk, heads, eval_arr, ecfg, device="cpu")
    assert out["n_positions"] == 2 * 2 * (trunk.cfg.seq_len - 2)
    assert set(out["arms"]) == set(COMPOSE_MODES)
    for m in COMPOSE_MODES:
        assert 0.0 <= out["arms"][m]["t2_hit"] <= 1.0
        assert out["arms"][m]["t2_hit"] + out["arms"][m]["accept_vs_ar1"] >= 0.0
    assert "gated_g" in out and len(out["gated_g"]["bins"]) == 10
    assert out["shared"]["oracle_t2_hit"] >= out["shared"]["ar1_deploy_t2_hit"] - 0.5
