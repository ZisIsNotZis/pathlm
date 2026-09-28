"""深度自适应 AR（ticket 23）的可失败单测。Run: python -m pytest tests/ -q

重点钉住混合深度 KV 填充语义（判据 5）：位置 i 的 KV 在其退出深度之后的
所有层 = 层的 kv 投影作用于其退出深度表征 h_{d_i}[i]；以及全深度并行监督、
carry 门恒等起步、提交方式、增量解码与并行 forward 的逐位一致性。
"""

import os, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             commit_input, generate, token_input, workspace_probe,
                             tf_acceptance)
from pathlm.model import Block


def tiny_model(**kw):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12, **kw)
    return DepthARModel(cfg, vocab_size=10)


def test_block_kv_from_is_noop_when_equal():
    """kv_from=x 必须严格等价于不传（全深度路径的退化一致性）。"""
    torch.manual_seed(1)
    blk = Block(type("C", (), {"d_model": 16, "n_heads": 2, "mlp_mult": 4,
                               "dropout": 0.0})())
    x = torch.randn(2, 6, 16)
    out_plain, _ = blk(x)
    out_kv, _ = blk(x, kv_from=x)
    assert torch.allclose(out_plain, out_kv, atol=1e-6)


def test_block_gate_scales_whole_delta():
    """gate_vec 语义：out = x + g ⊙ (block_delta)，门乘注意力+MLP 整个 delta。"""
    torch.manual_seed(1)
    blk = Block(type("C", (), {"d_model": 16, "n_heads": 2, "mlp_mult": 4,
                               "dropout": 0.0})())
    x = torch.randn(2, 6, 16)
    g = torch.rand(2, 6, 16)
    out_gated, _ = blk(x, gate_vec=g)
    out_plain, _ = blk(x)
    expected = x + g * (out_plain - x)
    assert torch.allclose(out_gated, expected, atol=1e-6)
    # g=0 → 严格恒等
    out_zero, _ = blk(x, gate_vec=torch.zeros_like(x))
    assert torch.allclose(out_zero, x, atol=1e-6)


def test_full_depth_states_match_gated_chain():
    """全深度 forward 的 states 必须等于手写带门链（门+残差复合语义）。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    h = aux["states"][0]
    for k in range(m.cfg.n_layers):
        out, _ = m.blocks[k](h)          # 全深度：kv_in == h，plain 即等价
        g = torch.sigmoid(m.gates[k](h))
        h = h + g * (out - h)
        assert torch.allclose(aux["states"][k + 1], h, atol=1e-5), f"depth {k}"


def test_mixed_depth_kv_fill_semantics():
    """核心语义钉（判据 5）：混合深度前向 = 手写规则——
    位置 i 在层 k ≤ d_i 正常精化；k > d_i 冻结在 h_{d_i}[i]；
    层 k 的 KV 源 = 活跃行当前表征 / 已退出行冻结表征（经 Block 的
    ln1+qkv k/v 半边）。任何偏离（冻结 KV 复用、缺 ln1、状态继续更新）
    都会 here 失配。"""
    m = tiny_model()
    torch.manual_seed(3)
    x = torch.randint(0, 10, (1, 12))
    depths = torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 3, 2]])
    _, aux = m(x, exit_depths=depths)
    st = aux["states"]
    L = m.cfg.n_layers
    # 冻结不变量：k > d_i 的状态 == 退出深度状态
    for i in range(12):
        for k in range(int(depths[0, i]) + 1, L + 1):
            assert torch.allclose(st[k][0, i], st[int(depths[0, i])][0, i], atol=1e-6)
    # 手写混合深度规则逐层重建，必须逐位复现
    h = st[0]
    frozen = st[0]
    for k in range(1, L + 1):
        active = depths >= k
        kv_in = torch.where(active.unsqueeze(-1), h, frozen)
        g = torch.sigmoid(m.gates[k - 1](h))
        out, _ = m.blocks[k - 1](h, kv_from=kv_in)
        h = torch.where(active.unsqueeze(-1), h + g * (out - h), frozen)
        frozen = torch.where((depths == k).unsqueeze(-1), h, frozen)
        assert torch.allclose(st[k], h, atol=1e-5), f"mixed-depth layer {k}"


def test_carry_gate_identity_extreme():
    """carry 起步的精确极限：门偏置 -30 → σ≈0 → 全栈恒等，每深度状态
    等于输入嵌入态（判据：门全关时不塌不炸）。"""
    m = tiny_model(gate_bias_init=-30.0)
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    for k in range(1, m.cfg.n_layers + 1):
        assert torch.allclose(aux["states"][k], aux["states"][0], atol=1e-5), k


def test_depth0_readout_is_tied_embedding():
    """depth 0 读出 = 输入态直连 tied-unembed（无独立头），fp32。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    from pathlm.model import cap_norm
    h0 = cap_norm(m.embed(x) + m.pos_embed.weight[:12], m.cfg.norm_cap)
    expected = h0.float() @ m.embed.weight.float().T
    assert aux["depth_logits"][0].dtype == torch.float32
    assert torch.allclose(aux["depth_logits"][0], expected, atol=1e-5)
    # tied：模型里没有独立 [V,d] 头
    heads = [n for n, p in m.named_parameters()
             if p.shape == (m.vocab_size, m.cfg.d_model) and "embed" not in n]
    assert not heads, f"untied head found: {heads}"


def test_exit_loss_masks_rows_below_depth():
    """全深度监督掩码：depth-k CE 只统计 d_i ≥ k 的预测行；全 0 退出时
    更深深度贡献 0，总损失 == depth-0 CE。"""
    m = tiny_model()
    x = torch.randint(0, 10, (2, 12))
    depths = torch.zeros(2, 12, dtype=torch.long)
    loss, aux = m(x, exit_depths=depths)
    h0 = m._input_states(x)
    ce0 = F.cross_entropy((h0.float() @ m.embed.weight.float().T)[:, :-1].reshape(-1, 10),
                          x[:, 1:].reshape(-1))
    assert torch.allclose(aux["depth_ce"][0], ce0, atol=1e-6)
    for k in range(1, m.cfg.n_layers + 1):
        assert aux["depth_ce"][k].item() == 0.0
    assert torch.allclose(loss, ce0 / (m.cfg.n_layers + 1), atol=1e-6)
    # 掩码精确性：depths 只有列 0 为 L，其余 0 → depth-L CE 只含列 0 的行
    depths = torch.zeros(2, 12, dtype=torch.long)
    depths[:, 0] = m.cfg.n_layers
    _, aux = m(x, exit_depths=depths)
    hL = m(x, exit_depths=torch.full_like(depths, m.cfg.n_layers))[1]["states"][-1]
    # 用混合前向自己的最终状态算列 0 行的 CE
    lg = (hL.float() @ m.embed.weight.float().T)[:, :-1]
    ce_rows = F.cross_entropy(lg.reshape(-1, 10), x[:, 1:].reshape(-1), reduction="none")
    mask = (depths[:, :-1] == m.cfg.n_layers).reshape(-1).float()
    expected = (ce_rows * mask).sum() / mask.sum()
    assert torch.allclose(aux["depth_ce"][-1], expected, atol=1e-6)


def test_sample_exit_depths_uniform_support():
    """采样器支撑 = {0..L}（含 0 与 L），形状正确。"""
    g = torch.Generator().manual_seed(0)
    d = sample_exit_depths(64, 64, 4, g)
    assert d.shape == (64, 64)
    assert int(d.min()) == 0 and int(d.max()) == 4
    counts = torch.bincount(d.flatten(), minlength=5).float()
    frac = counts / counts.sum()
    assert (frac - 0.2).abs().max() < 0.02  # 均匀起步


def test_commit_modes():
    """soft = Σp·E[t]（新鲜 token gauge：cap+pos）；hard = E[argmax]；
    latent = 直通状态 + pos。"""
    m = tiny_model()
    torch.manual_seed(5)
    logits = torch.randn(1, 1, 10)
    h = torch.randn(1, 1, 16)
    pe = m.pos_embed.weight[7:8]
    from pathlm.model import cap_norm
    soft = commit_input(m, logits, h, "soft", pos=7)
    p = logits.float().softmax(-1)
    e = p @ m.embed.weight.float()
    assert torch.allclose(soft, cap_norm(e + pe, m.cfg.norm_cap), atol=1e-6)
    hard = commit_input(m, logits, h, "hard", pos=7)
    eh = m.embed.weight.float()[logits.argmax(-1)]
    assert torch.allclose(hard, cap_norm(eh + pe, m.cfg.norm_cap), atol=1e-6)
    latent = commit_input(m, logits, h, "latent", pos=7)
    assert torch.allclose(latent, h + pe, atol=1e-6)


def test_decoder_matches_parallel_forward():
    """增量解码（逐层 KV cache + 冻结填充）必须与并行混合深度 forward 在
    每个 position 的退出深度 logits 逐位一致——缓存语义的端到端钉。"""
    from pathlm.depth_ar import DepthARDecoder
    m = tiny_model()
    torch.manual_seed(7)
    x = torch.randint(0, 10, (1, 12))
    depths = torch.tensor([[0, 3, 1, 2, 0, 3, 3, 1, 2, 0, 1, 3]])
    _, aux = m(x, exit_depths=depths)
    dec = DepthARDecoder(m)
    for i in range(12):
        _, logits = dec.step(token_input(m, int(x[0, i]), dec.n), int(depths[0, i]))
        e = int(depths[0, i])
        assert torch.allclose(logits, aux["depth_logits"][e][0, i], atol=2e-4), \
            f"position {i} exit {e}"
    assert dec.n == 12


def test_generate_legal_and_finite():
    """贪心生成冒烟：id 在词表内、数量正确、无 NaN。"""
    m = tiny_model()
    torch.manual_seed(9)
    prompt = torch.randint(0, 10, (6,))
    gen, stats = generate(m, prompt, 5, commit="soft", forced_depth=1)
    assert len(gen) == 5 and stats["commit"] == "soft"
    assert all(0 <= t < 10 for t in gen)
    assert all(t == t for t in gen)  # NaN 检查（int 通道必然有限；dtype 防回归）


def test_dense_arm_is_plain_stack():
    """dense 对照 arm：无门遥测、states = L+1 层堆叠、仅一个 CE 分量。"""
    m = tiny_model(dense=True)
    x = torch.randint(0, 10, (2, 12))
    loss, aux = m(x)
    assert aux["gate_mean"] == [] and len(aux["depth_ce"]) == 1
    assert len(aux["states"]) == m.cfg.n_layers + 1
    # dense 最终状态 = 无门裸堆叠
    h = aux["states"][0]
    for blk in m.blocks:
        h, _ = blk(h)
    assert torch.allclose(aux["states"][-1], h, atol=1e-6)
    expected = F.cross_entropy((h.float() @ m.embed.weight.float().T)[:, :-1].reshape(-1, 10),
                               x[:, 1:].reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_workspace_probe_contract():
    """探针契约：d>V 时 null 维 = d−rank 且数值对照 ≈0（null 正交）；
    d<V 时 null 维结构性为 0。"""
    torch.manual_seed(0)
    m = tiny_model()  # V=10, d=16 → null 6 维
    arr = torch.randint(0, 10, (200,)).numpy()
    wp = workspace_probe(m, arr, n_batches=1, batch_size=2)
    assert wp["readout_rank"] == 10 and wp["null_dim"] == 6
    assert wp["control_same_row_max_dlogit"] < 1e-5
    # d<V：8 维模型、词表 20
    torch.manual_seed(0)
    cfg2 = DepthARConfig(d_model=8, n_layers=2, n_heads=2, seq_len=12)
    m3 = DepthARModel(cfg2, vocab_size=20)
    wp3 = workspace_probe(m3, arr, n_batches=1, batch_size=2)
    assert wp3["null_dim"] == 0 and wp3["functional_dirs"] == 0


def test_tf_acceptance_shapes():
    """TF 接受率：深度越深对最终 argmax 的一致率应不降（健全性下界：
    depth L-1 vs L 通常最高），且数值在 [0,1]。"""
    m = tiny_model()
    torch.manual_seed(2)
    arr = torch.randint(0, 10, (300,)).numpy()
    acc = tf_acceptance(m, arr, n_batches=2, batch_size=4)
    assert len(acc) == m.cfg.n_layers
    assert all(0.0 <= a["accept"] <= 1.0 for a in acc)


# ---------- ticket 23b：浅层 aux 加权 ----------


def test_shallow_weight_one_is_v1_uniform():
    """w=1 与缺省严格等值，且 == 逐深度 CE 的手算等权平均（W0 复现点）。"""
    m = tiny_model()
    torch.manual_seed(11)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(1))
    l_def, aux = m(x, exit_depths=depths)
    l_one, _ = m(x, exit_depths=depths, shallow_weight=1.0)
    assert torch.allclose(l_def, l_one, atol=1e-7)
    expected = sum(aux["depth_ce"]) / len(aux["depth_ce"])
    assert torch.allclose(l_def, expected, atol=1e-6)


def test_shallow_weight_zero_is_final_depth_only():
    """w=0：总损失 == 最终深度 CE（浅层读出 CE 完全退出损失）。"""
    m = tiny_model()
    torch.manual_seed(12)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(2))
    loss, aux = m(x, exit_depths=depths, shallow_weight=0.0)
    assert torch.allclose(loss, aux["depth_ce"][-1], atol=1e-6)


def test_shallow_weight_interp_matches_manual_formula():
    """w=0.1：loss == (w·Σ浅层 CE + ce_L) / (w·L + 1)（归一化保量级）。"""
    w = 0.1
    m = tiny_model()
    torch.manual_seed(13)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(3))
    loss, aux = m(x, exit_depths=depths, shallow_weight=w)
    ces = aux["depth_ce"]
    L = m.cfg.n_layers
    expected = (w * sum(ces[:L]) + ces[L]) / (w * L + 1)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_shallow_weight_zero_grad_matches_final_ce_backward():
    """梯度通道钉（w=0）：参数梯度 == 手动对 depth_ce[-1] 反传（浅层加权
    若被静默忽略或错乘到最终深度，此处失配）。"""
    m1, m2 = tiny_model(), tiny_model()  # 同 seed → 同初始化
    torch.manual_seed(14)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m1.cfg.n_layers,
                                torch.Generator().manual_seed(4))
    loss1, _ = m1(x, exit_depths=depths, shallow_weight=0.0)
    loss1.backward()
    aux2 = m2(x, exit_depths=depths)[1]
    aux2["depth_ce"][-1].backward()
    for name in ("embed.weight", "blocks.0.qkv.weight", "gates.0.weight"):
        g1 = dict(m1.named_parameters())[name].grad
        g2 = dict(m2.named_parameters())[name].grad
        assert torch.allclose(g1, g2, atol=1e-7), name


def test_aux_weight_schedule_constants_and_anneal():
    """train_depth_ar.aux_weight_at：常数臂恒 aux_weight；退火臂最后 20%
    步从 1.0 线性降到 0.1，起点连续、中点 0.55、终点精确 0.1（W3）。"""
    import train_depth_ar as t
    assert t.aux_weight_at(0, 3600, 0.1, None) == 0.1
    assert t.aux_weight_at(1234, 3600, 1.0, 0.1) == 1.0      # 退火前
    assert t.aux_weight_at(2880, 3600, 1.0, 0.1) == 1.0      # 起点（0.8·3600）
    assert abs(t.aux_weight_at(3600, 3600, 1.0, 0.1) - 0.1) < 1e-12  # 终点
    assert abs(t.aux_weight_at(3240, 3600, 1.0, 0.1) - 0.55) < 1e-9  # 中点
