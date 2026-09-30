"""深度自适应 AR（ticket 23）的可失败单测。Run: python -m pytest tests/ -q

重点钉住混合深度 KV 填充语义（判据 5）：位置 i 的 KV 在其退出深度之后的
所有层 = 层的 kv 投影作用于其退出深度表征 h_{d_i}[i]；以及全深度并行监督、
carry 门恒等起步、提交方式、增量解码与并行 forward 的逐位一致性。
"""

import os, sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             ParallelBlock, TOPOLOGIES, topology_liveness,
                             commit_input, generate, token_input, workspace_probe,
                             tf_acceptance, readout_workspace_energy,
                             spec_cost_model)
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


# ---------- ticket 23c：三消融开关（H1 dense+aux / H3 stop-grad 浅头） ----------


def test_shallow_stopgrad_trunk_grad_equals_final_only():
    """H3 梯度通道钉：shallow_stopgrad=True 时 trunk 梯度 == 仅最终深度 CE
    反传（w=0 臂同模型同输入）——浅层 CE 不穿 trunk；同时 tied U 头仍从浅层
    CE 收梯度（embed 梯度必不同，证明浅头仍在训练而非被丢弃）。
    detach 被静默忽略或错加到最终深度时，此测试变红。"""
    m1, m2 = tiny_model(), tiny_model()  # 同 seed → 同初始化
    torch.manual_seed(21)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m1.cfg.n_layers,
                                torch.Generator().manual_seed(5))
    loss1, _ = m1(x, exit_depths=depths, shallow_weight=1.0, shallow_stopgrad=True)
    loss1.backward()
    # 参照臂：手动反传 depth_ce[-1]/(L+1)——与 stop-grad 臂同归一化（w=0 臂的
    # 分母是 1，不能直接用）；trunk 梯度应逐位一致
    aux2 = m2(x, exit_depths=depths)[1]
    (aux2["depth_ce"][-1] / (m1.cfg.n_layers + 1)).backward()
    p1, p2 = dict(m1.named_parameters()), dict(m2.named_parameters())
    for name in ("blocks.0.qkv.weight", "blocks.2.mlp.0.weight", "gates.1.weight"):
        assert torch.allclose(p1[name].grad, p2[name].grad, atol=1e-7), name
    # 浅层 CE 仍训练读出头：stop-grad 只断状态通道，tied U 的梯度分量仍在
    assert not torch.allclose(p1["embed.weight"].grad,
                              p2["embed.weight"].grad, atol=1e-7)


def test_shallow_stopgrad_loss_value_unchanged():
    """H3 只改梯度通道不改损失值：同输入下 stop-grad 与否的 loss 与逐深度
    CE 逐位一致（评测路径 / 日志数字不受开关影响）。"""
    m = tiny_model()
    torch.manual_seed(22)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(6))
    l_plain, aux = m(x, exit_depths=depths)
    l_sg, aux_sg = m(x, exit_depths=depths, shallow_stopgrad=True)
    assert torch.allclose(l_plain, l_sg, atol=1e-6)
    assert torch.allclose(torch.stack(aux["depth_ce"]),
                          torch.stack(aux_sg["depth_ce"]), atol=1e-6)


def test_dense_aux_full_depth_supervision():
    """H1 dense+aux：dense 臂开启全深度并行监督——L+1 个 CE 分量、损失 =
    (w·Σ浅层 CE + ce_L)/(w·L+1)；缺省 dense 臂仍只 1 个分量。且 aux 梯度
    确实穿 trunk（w=1 vs w=0 的 block 梯度必不同）。dense_aux 被忽略时变红。"""
    m1, m2 = tiny_model(dense=True), tiny_model(dense=True)
    torch.manual_seed(23)
    x = torch.randint(0, 10, (2, 12))
    loss1, aux1 = m1(x, dense_aux=True)
    L = m1.cfg.n_layers
    assert len(aux1["depth_ce"]) == L + 1
    expected = sum(aux1["depth_ce"]) / (L + 1)
    assert torch.allclose(loss1, expected, atol=1e-6)
    # 缺省不变：dense 仅最终深度 CE（v1 dense-B 语义）
    loss_def, aux_def = m1(x)
    assert len(aux_def["depth_ce"]) == 1
    assert torch.allclose(loss_def, aux1["depth_ce"][-1], atol=1e-6)
    # aux 梯度穿 trunk：同初始化下 w=1 与 w=0 的 block 梯度不同
    loss1.backward()
    loss2, _ = m2(x, dense_aux=True, shallow_weight=0.0)
    loss2.backward()
    p1, p2 = dict(m1.named_parameters()), dict(m2.named_parameters())
    assert not torch.allclose(p1["blocks.0.qkv.weight"].grad,
                              p2["blocks.0.qkv.weight"].grad, atol=1e-7)


# ---------- ticket 23e：填充策略对决（no-fill ragged / proj-fill） ----------


def _manual_ragged_layer(m, k0: int, h: torch.Tensor,
                         depths: torch.Tensor) -> torch.Tensor:
    """手写 ragged 层规则（no-fill 参照实现）：层 k0+1 的注意力只在
    因果键 ∩ 已达层 k0+1 的行（d_j ≥ k0+1）上归一化，键值取当前表征
    （无冻结填充）；输出带门精化。"""
    blk = m.blocks[k0]
    B, T, d = h.shape
    qkv = blk.qkv(blk.ln1(h))
    q, k, v = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
    nh, dh = blk.n_heads, d // blk.n_heads
    shape = lambda t: t.view(B, T, nh, dh).transpose(1, 2)
    qs, ks, vs = shape(q), shape(k), shape(v)
    allow = torch.tril(torch.ones(T, T, dtype=torch.bool)) \
        & (depths[:, None, :] >= k0 + 1)
    allow = allow | torch.eye(T, dtype=torch.bool)
    scores = qs @ ks.transpose(-2, -1) / dh ** 0.5
    scores = scores.masked_fill(~allow.unsqueeze(1), float("-inf"))
    a = (scores.softmax(-1) @ vs).transpose(1, 2).reshape(B, T, d)
    y = h + blk.proj(a)
    y = y + blk.mlp(blk.ln2(y))
    g = torch.sigmoid(m.gates[k0](h))
    return h + g * (y - h)


def test_nofill_states_match_manual_ragged_attention():
    """no-fill ragged 语义钉（对照 test_mixed_depth_kv_fill_semantics）：
    fill_kv=False 的混合深度前向 = 手写 ragged 规则逐层重建——已退出行
    在更深层的 KV 中缺席而非冻结填充；实现若静默回落 fill 或掩码漏因果/
    漏 d_j 筛选/掩码 batch-head 维错位，此处失配。B=4 ≠ n_heads=2 且各行
    深度模式不同（掩码逐 batch 内容钉住，防 [B,T,T]/[1,B,T,T] 侥幸通过）。"""
    m = tiny_model(fill_kv=False)
    torch.manual_seed(3)
    x = torch.randint(0, 10, (4, 12))
    depths = torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 3, 2],
                           [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0]])
    st = m(x, exit_depths=depths)[1]["states"]
    L = m.cfg.n_layers
    # 冻结不变量与 fill 臂相同（退出行状态定格）
    for b in range(4):
        for i in range(12):
            for k in range(int(depths[b, i]) + 1, L + 1):
                assert torch.allclose(st[k][b, i], st[int(depths[b, i])][b, i],
                                      atol=1e-6)
    h, frozen = st[0], st[0]
    for k in range(1, L + 1):
        active = depths >= k
        out = _manual_ragged_layer(m, k - 1, h, depths)
        h = torch.where(active.unsqueeze(-1), out, frozen)
        frozen = torch.where((depths == k).unsqueeze(-1), h, frozen)
        assert torch.allclose(st[k], h, atol=1e-5), f"ragged layer {k}"


def test_nofill_exited_key_absent_fill_contrast():
    """缺席语义的双向钉：d_j=0 位置换 token，no-fill 下任何 i≠j 行的
    depth-1 logits 逐位不变（键缺席）；同扰动同初始化的 fill 臂 i>j 行
    必变（键在）。只改行 j 自身读出（其 depth-1 = 冻结 depth-0 态）不算。"""
    torch.manual_seed(7)
    m_ragged = tiny_model(fill_kv=False)
    m_fill = tiny_model(fill_kv=True)          # 同 seed → 同初始化
    x = torch.randint(0, 10, (1, 12))
    depths = torch.full((1, 12), 3, dtype=torch.long)
    depths[0, 2] = 0
    x2 = x.clone()
    x2[0, 2] = (x[0, 2] + 1) % 10
    rows = [i for i in range(11) if i != 2]
    with torch.no_grad():
        lg_r1 = m_ragged(x, exit_depths=depths)[1]["depth_logits"][1]
        lg_r2 = m_ragged(x2, exit_depths=depths)[1]["depth_logits"][1]
        assert torch.allclose(lg_r1[0, rows], lg_r2[0, rows], atol=1e-6), \
            "no-fill：早退键必须缺席"
        lg_f1 = m_fill(x, exit_depths=depths)[1]["depth_logits"][1]
        lg_f2 = m_fill(x2, exit_depths=depths)[1]["depth_logits"][1]
        assert not torch.allclose(lg_f1[0, 3:], lg_f2[0, 3:], atol=1e-6), \
            "fill 对照臂：填充键必被读到（对照失效=实验设计错误）"


def test_nofill_full_depth_equals_fill_and_none():
    """全深度退化：dᵢ=L 时 ragged 掩码 == 纯因果，no-fill 与 fill 及
    exit_depths=None 的 states/loss 一致——评测电池（全深度前向）数字
    不受填充策略开关影响。"""
    m1 = tiny_model(fill_kv=False)
    m2 = tiny_model(fill_kv=True)              # 同 seed → 同初始化
    torch.manual_seed(8)
    x = torch.randint(0, 10, (2, 12))
    depths = torch.full((2, 12), m1.cfg.n_layers, dtype=torch.long)
    l1, a1 = m1(x, exit_depths=depths)
    l2, a2 = m2(x, exit_depths=depths)
    assert torch.allclose(l1, l2, atol=1e-5)
    for s1, s2 in zip(a1["states"], a2["states"]):
        assert torch.allclose(s1, s2, atol=1e-5)
    l0, a0 = m1(x, exit_depths=None)
    assert torch.allclose(l0, l1, atol=1e-5)
    assert torch.allclose(a0["states"][-1], a1["states"][-1], atol=1e-5)


def test_nofill_all_zero_depths_finite_no_nan():
    """极端 ragged（全 dᵢ=0）：每查询只余对角键（开对角防全掩码 NaN），
    loss/逐深度 CE 有限且 depth-k>0 CE == 0（监督掩码语义与 fill 一致）。"""
    m = tiny_model(fill_kv=False)
    torch.manual_seed(9)
    x = torch.randint(0, 10, (2, 12))
    depths = torch.zeros(2, 12, dtype=torch.long)
    loss, aux = m(x, exit_depths=depths)
    assert torch.isfinite(loss)
    assert all(bool(torch.isfinite(c)) for c in aux["depth_ce"])
    for k in range(1, m.cfg.n_layers + 1):
        assert aux["depth_ce"][k].item() == 0.0


def test_proj_fill_identity_init_equals_fill():
    """proj-fill 恒等初始化：起步 states/loss 与 fill 逐位一致（remap 从
    fill 出发学，w 对齐 W0 复现点）；恒等复制不消耗随机数（同 seed 下其余
    参数初始化逐位相同）。proj_fill+no-fill 组合必须被拒绝。"""
    m1 = tiny_model(fill_kv=True, proj_fill=True)
    m2 = tiny_model(fill_kv=True)
    torch.manual_seed(10)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m1.cfg.n_layers,
                                torch.Generator().manual_seed(11))
    l1, a1 = m1(x, exit_depths=depths)
    l2, a2 = m2(x, exit_depths=depths)
    assert torch.allclose(l1, l2, atol=1e-6)
    for s1, s2 in zip(a1["states"], a2["states"]):
        assert torch.allclose(s1, s2, atol=1e-6)
    # 非法组合
    m3 = tiny_model(fill_kv=False, proj_fill=True)
    try:
        m3(x, exit_depths=depths)
        raise AssertionError("proj_fill+fill_kv=False 必须拒绝")
    except ValueError:
        pass


# ---------- ticket 24：最终深度权重（契约「权重均匀 + 最终深度 2×」） ----------


def test_final_weight_one_restores_v1():
    """final_weight=1.0 必须逐位还原 v1 等权损失（ticket 23 复现点不被
    参数化改动扰动）；final_weight 只进损失加权，不改 states。"""
    m = tiny_model()
    torch.manual_seed(12)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(13))
    l_v1, a_v1 = m(x, exit_depths=depths)
    l_fw1, a_fw1 = m(x, exit_depths=depths, final_weight=1.0)
    assert torch.allclose(l_v1, l_fw1, atol=0.0)
    for s1, s2 in zip(a_v1["states"], a_fw1["states"]):
        assert torch.allclose(s1, s2, atol=0.0)


def test_final_weight_two_matches_hand_weighting():
    """final_weight=2.0：loss == (Σ浅层 ce + 2·最终 ce) / (L + 2)
    （量级保持归一），且梯度可流（backward 有限）。"""
    m = tiny_model()
    torch.manual_seed(14)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(15))
    loss, aux = m(x, exit_depths=depths, final_weight=2.0)
    ces = torch.stack(aux["depth_ce"])
    expected = (ces[:-1].sum() + 2.0 * ces[-1]) / (m.cfg.n_layers + 2.0)
    assert torch.allclose(loss, expected, atol=1e-6)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9)
    assert torch.isfinite(gnorm)


# ---------- ticket 25 B 臂：结构切分（readout_dims） ----------


def test_readout_dims_slice_excludes_workspace_from_logits():
    """B 臂切片钉（ticket 25）：readout_dims=r 时 (a) logits == 手算
    h[..., :r] @ E[:, :r].T；(b) workspace 维（r..d）扰动不进该状态 logits
    （逐位不变）；(c) workspace 维仍经 trunk 残差参与计算——mid-depth 状态
    workspace 扰动经后半栈改变 depth-L logits。切片被静默忽略、清零或
    trunk 误读切片时，本测试变红。"""
    r = 10
    m = tiny_model(readout_dims=r)
    assert m.embed.weight.shape == (10, 16)      # embedding 端不动（全维）
    torch.manual_seed(24)
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    h1 = aux["states"][1]
    E = m.embed.weight.float()
    manual = h1[..., :r].float() @ E[:, :r].T
    assert torch.allclose(aux["depth_logits"][1], manual, atol=1e-5)
    # (b) workspace 扰动：本状态读出逐位不变
    delta = torch.zeros_like(h1)
    delta[..., r:] = torch.randn_like(h1[..., r:])
    assert torch.equal(m.readout(h1), m.readout(h1 + delta))
    # (c) workspace 经 trunk 参与：后半栈重跑后 depth-L 读出必变
    def restack(h):
        hh = h
        for k0 in range(1, m.cfg.n_layers):
            hh = m._gated_step(k0, hh)
        return m.readout(hh)
    assert not torch.allclose(restack(h1), restack(h1 + delta), atol=1e-6)
    # 非法 r 必须拒绝
    try:
        tiny_model(readout_dims=17)
        raise AssertionError("readout_dims > d_model 必须拒绝")
    except ValueError:
        pass


def test_readout_dims_default_restores_full_readout_bitwise():
    """缺省（None）与显式 r=d 都必须逐位还原 v1 全维度读出（logits 与 loss
    torch.equal；ticket-23 复现点不被参数化改动扰动）。"""
    m = tiny_model()
    m_r = tiny_model(readout_dims=16)
    m_r.load_state_dict(m.state_dict())
    torch.manual_seed(25)
    x = torch.randint(0, 10, (2, 12))
    _, a1 = m(x, exit_depths=None)
    _, a2 = m_r(x, exit_depths=None)
    for l1, l2 in zip(a1["depth_logits"], a2["depth_logits"]):
        assert torch.equal(l1, l2)
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(26))
    l1, _ = m(x, exit_depths=depths)
    l2, _ = m_r(x, exit_depths=depths)
    assert torch.equal(l1, l2)


def test_workspace_probe_readout_dims_structural_null():
    """workspace 探针 readout_dims 感知：有效读出表 = E[:, :r]，结构性
    workspace 维（r..d）计入 null。d=16/V=10/r=8：切片满秩 8 →
    null = 16−8 = 8（含 8 个结构维）；缺省 r=16：null = 16−10 = 6。
    控制量（当前行读出不变）两配置下都 ≈0。"""
    torch.manual_seed(0)
    m8 = tiny_model(readout_dims=8)
    arr = torch.randint(0, 10, (200,)).numpy()
    wp = workspace_probe(m8, arr, n_batches=1, batch_size=2)
    assert wp["readout_rank"] == 8 and wp["null_dim"] == 8
    assert wp["control_same_row_max_dlogit"] < 1e-5
    wp_def = workspace_probe(tiny_model(), arr, n_batches=1, batch_size=2)
    assert wp_def["readout_rank"] == 10 and wp_def["null_dim"] == 6


def test_readout_workspace_energy_probe_contract():
    """B 臂范数比探针：读出/workspace 能量占比互补（和为 1）且在 [0,1]；
    readout_dims=None 时读出占比恒 1。"""
    arr = torch.randint(0, 10, (200,)).numpy()
    rwe = readout_workspace_energy(tiny_model(readout_dims=8), arr,
                                   n_batches=1, batch_size=2)
    assert rwe["readout_dims"] == 8 and rwe["d_model"] == 16
    assert len(rwe["readout_energy_frac_by_depth"]) == 4  # L+1 = 3+1
    for f, wf in zip(rwe["readout_energy_frac_by_depth"],
                     rwe["workspace_energy_frac_by_depth"]):
        assert abs(f + wf - 1.0) < 1e-6 and 0.0 <= f <= 1.0
    rwe_full = readout_workspace_energy(tiny_model(), arr,
                                        n_batches=1, batch_size=2)
    assert all(f == 1.0 for f in rwe_full["readout_energy_frac_by_depth"])


# ---------- ticket 25 C 臂：--exit-anneal 冷却退火调度 ----------


def test_exit_anneal_schedule_values():
    """C 臂调度值（48k 口径）：75% 均匀、末 25% 线性；aux 1.0→0.3、final
    2.0→1.0 同窗（起点连续/中点精确/终点精确）；关断恒等（aux 恒 1.0、
    final 恒 2.0 = ticket-24 语义）。"""
    import train_depth_ar as t
    steps, start = 48000, 36000          # (1−0.25)·48000
    for s in (0, 20000, start):
        assert t.aux_weight_at(s, steps, 1.0, 0.3, 0.25) == 1.0
        assert t.final_weight_at(s, steps, 2.0, 1.0, 0.25) == 2.0
    mid = (start + steps) // 2           # 42000 → 窗口中点
    assert abs(t.aux_weight_at(mid, steps, 1.0, 0.3, 0.25) - 0.65) < 1e-9
    assert abs(t.final_weight_at(mid, steps, 2.0, 1.0, 0.25) - 1.5) < 1e-9
    assert abs(t.aux_weight_at(steps, steps, 1.0, 0.3, 0.25) - 0.3) < 1e-12
    assert abs(t.final_weight_at(steps, steps, 2.0, 1.0, 0.25) - 1.0) < 1e-12
    # 关断恒等
    assert t.aux_weight_at(47999, steps, 1.0, None) == 1.0
    assert t.final_weight_at(47999, steps, 2.0, None) == 2.0


def test_exit_anneal_endpoints_match_loss_formula():
    """C 臂端点接线钉：step 0 调度值 (1.0, 2.0) 逐位等于 A 配方
    （shallow_weight=1.0, final_weight=2.0）；末步调度值 (0.3, 1.0) 的损失
    == 手工公式 (0.3·Σ浅层 ce + ce_L)/(0.3·L + 1)（归一保量级）。"""
    import train_depth_ar as t
    m = tiny_model()
    torch.manual_seed(27)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(28))
    w0 = t.aux_weight_at(0, 48000, 1.0, 0.3, 0.25)
    fw0 = t.final_weight_at(0, 48000, 2.0, 1.0, 0.25)
    l_start, _ = m(x, exit_depths=depths, shallow_weight=w0, final_weight=fw0)
    l_a, _ = m(x, exit_depths=depths, shallow_weight=1.0, final_weight=2.0)
    assert torch.equal(l_start, l_a)
    w_end = t.aux_weight_at(48000, 48000, 1.0, 0.3, 0.25)
    fw_end = t.final_weight_at(48000, 48000, 2.0, 1.0, 0.25)
    l_end, aux = m(x, exit_depths=depths, shallow_weight=w_end,
                   final_weight=fw_end)
    ces = torch.stack(aux["depth_ce"])
    expected = (0.3 * ces[:-1].sum() + ces[-1]) / (0.3 * m.cfg.n_layers + 1.0)
    assert torch.allclose(l_end, expected, atol=1e-6)


def test_spec_cost_model_formula():
    """等效验证加速公式（契约主口径）：speedup = (1+a_k)/(1+k/L)；k=4,
    a=0.9726, L=12 → (1.9726)/(1.3333) ≈ 1.479。"""
    tf = [{"depth": k, "accept": a} for k, a in ((4, 0.9726), (6, 0.9946))]
    out = spec_cost_model(tf, n_layers=12, draft_depths=(4, 6))
    assert abs(out[0]["equivalent_validation_speedup"]
               - (1 + 0.9726) / (1 + 4 / 12)) < 1e-3
    assert abs(out[1]["equivalent_validation_speedup"]
               - (1 + 0.9946) / (1 + 6 / 12)) < 1e-3
    assert out[0]["tf_accept"] == 0.9726
    assert out[0]["cost_per_cycle"] == 1.3333


# ---------- W4 主臂：gated 残差读出（readout_gated） ----------


def test_readout_gated_off_matches_v1_bitwise():
    """① --readout-gated 关闭逐位还原 A 臂：logits 与 v1 表达式
    h.float() @ Eᵀ torch.equal（全深度与随机退出深度两条路径都钉）。"""
    m = tiny_model()
    torch.manual_seed(21)
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        assert torch.equal(lg, s.float() @ m.embed.weight.float().T)
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(22))
    _, aux_d = m(x, exit_depths=depths)
    for s, lg in zip(aux_d["states"], aux_d["depth_logits"]):
        assert torch.equal(lg, s.float() @ m.embed.weight.float().T)


def test_readout_gated_zero_weights_identity_limit():
    """② W1=W2=0 时 gated 读出严格等于 tied 线性 h·Eᵀ（恒等极限，逐位）：
    起步 == 基线 A，不引入任何初始税。"""
    m = tiny_model()                     # A 臂
    g = tiny_model(readout_gated=True)
    # 复制 A 臂权重（g 多出 readout_w1/w2，保持其零初始化）
    missing = g.load_state_dict(m.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"readout_w1.weight",
                                         "readout_w2.weight"}
    assert not missing.unexpected_keys
    torch.manual_seed(23)
    x = torch.randint(0, 10, (2, 12))
    for model in (m, g):
        _, aux = model(x, exit_depths=None)
        for s, lg in zip(aux["states"], aux["depth_logits"]):
            assert torch.equal(lg, s.float() @ model.embed.weight.float().T)
    # A 臂与 gated（零权重）逐深度 logits 逐位一致
    _, a1 = m(x, exit_depths=None)
    _, a2 = g(x, exit_depths=None)
    for l1, l2 in zip(a1["depth_logits"], a2["depth_logits"]):
        assert torch.equal(l1, l2)


def test_readout_gated_formula_and_switch_guard():
    """③ 主臂读数公式 = (h + SiLU(W1 h) ⊙ (W2 h))·Eᵀ；且两处变异必抓红：
    (a) 忽略开关（永远返回 tied）——W1/W2 非零时 logits 必须偏离 tied；
    (b) 去掉非线性（silu→identity / 门不生效）——两处非零仍退化恒等或
    与公式不符（此处以精确公式对照）。"""
    g = tiny_model(readout_gated=True)
    torch.manual_seed(24)
    with torch.no_grad():
        nn.init.normal_(g.readout_w1.weight, std=0.3)
        nn.init.normal_(g.readout_w2.weight, std=0.3)
    h = torch.randn(2, 12, 16)
    hf = h.float()
    expected = (hf + F.silu(g.readout_w1(hf)) * g.readout_w2(hf)) \
        @ g.embed.weight.float().T
    got = g.readout(h)
    assert torch.allclose(got, expected, atol=0)
    # (a) 开关必须生效：非零 W1/W2 下读出必须偏离 tied
    assert not torch.allclose(got, hf @ g.embed.weight.float().T, atol=1e-6)
    # (b) 单侧为零仍退化恒等（门 ⊙ 值，任一为 0 则该分支不贡献）
    with torch.no_grad():
        g.readout_w2.weight.zero_()
    assert torch.equal(g.readout(h), hf @ g.embed.weight.float().T)
    with torch.no_grad():
        nn.init.normal_(g.readout_w2.weight, std=0.3)
        g.readout_w1.weight.zero_()
    assert torch.equal(g.readout(h), hf @ g.embed.weight.float().T)


def test_readout_gated_reaches_workspace_dims():
    """gated 变换在全维度 h 上做：readout_dims=r 时，workspace 维 h[r:] 经
    W1/W2 仍能改变 logits（结构性切分下的耦合被解开）；tied 线性（W=0）
    下 workspace 维对 logits 零影响（对照）。"""
    g = tiny_model(readout_gated=True, readout_dims=8)
    torch.manual_seed(25)
    with torch.no_grad():
        nn.init.normal_(g.readout_w1.weight, std=0.3)
        nn.init.normal_(g.readout_w2.weight, std=0.3)
    h = torch.randn(1, 4, 16)
    perturb = h.clone()
    perturb[..., 8:] += 0.5 * h[..., 8:].norm(dim=-1, keepdim=True)
    assert not torch.allclose(g.readout(h), g.readout(perturb), atol=1e-6)
    with torch.no_grad():                # W=0 对照：workspace 维不可见
        g.readout_w1.weight.zero_()
        g.readout_w2.weight.zero_()
    assert torch.equal(g.readout(h), g.readout(perturb))


def test_readout_gated_init_does_not_disturb_other_params():
    """init 规范（写死）：W1 ~ N(0,0.02)、W2 = 0。W1/W2 建在参数 init 循环
    之后 ⇒ 同 seed 下 gated 臂与 A 臂**其余全部参数逐位一致**（单因素：只加
    读出几何）。W2 必须全零（恒等起步），W1 必须非零（非零初始，避免鞍点）。"""
    a = tiny_model()
    g = tiny_model(readout_gated=True)
    sa, sg = a.state_dict(), g.state_dict()
    assert set(sg) - set(sa) == {"readout_w1.weight", "readout_w2.weight"}
    for k in sa:
        assert torch.equal(sa[k], sg[k]), k
    assert torch.count_nonzero(sg["readout_w2.weight"]) == 0
    assert torch.count_nonzero(sg["readout_w1.weight"]) > 0


def test_readout_gated_init_escapes_saddle():
    """干预活性门（P3 强制项）：零初始 W1=W2=0 是鞍点——实测 200 步
    |W1|=|W2|=0（永不可学）。本 init（W1~N(0,0.02)、W2=0）必须：(a) 起步
    严格恒等（readout == tied，逐位）；(b) 一步 backward 后 readout_w2.weight
    的梯度非零（∂L/∂W2 = SiLU(W1h)⊗h ≠ 0）⇒ 可逃逸鞍点。"""
    g = tiny_model(readout_gated=True)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, g.cfg.n_layers,
                                torch.Generator().manual_seed(29))
    # (a) 起步恒等极限（W1 随机、W2=0 ⇒ transform = SiLU(W1h)⊙0 = 0）
    _, aux = g(x, exit_depths=depths)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        assert torch.equal(lg, s.float() @ g.embed.weight.float().T)
    # (b) 非鞍点：W2 梯度非零；W1 在原点方向仍为零梯（靠 W2 破零）
    loss, _ = g(x, exit_depths=depths)
    loss.backward()
    g2 = g.readout_w2.weight.grad
    assert g2 is not None and torch.count_nonzero(g2) > 0
    assert torch.count_nonzero(g.readout_w1.weight.grad) == 0


# ---------- W4 契约 B 对照臂：线性等参数读出（readout_linear_resid） ----------


def test_readout_linear_resid_off_matches_v1_bitwise():
    """① --readout-linear-resid 关闭逐位还原 A 臂：logits 与 v1 表达式
    h.float() @ Eᵀ torch.equal（全深度与随机退出深度两条路径都钉）。"""
    m = tiny_model()
    torch.manual_seed(31)
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        assert torch.equal(lg, s.float() @ m.embed.weight.float().T)
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(32))
    _, aux_d = m(x, exit_depths=depths)
    for s, lg in zip(aux_d["states"], aux_d["depth_logits"]):
        assert torch.equal(lg, s.float() @ m.embed.weight.float().T)


def test_readout_linear_resid_identity_start():
    """② init 规范（写死）：Q ~ N(0,0.02)、P = 0（单边零）⇒ 起步 transform
    = P(Qh) = 0，读出与 tied 线性逐位一致（严格恒等起步，不引入初始税）。
    P、Q 建在参数 init 循环之后 ⇒ 同 seed 下与 A 臂**其余全部参数逐位一致**
    （单因素：只加读出几何）。"""
    a = tiny_model()
    r = tiny_model(readout_linear_resid=True)
    sa, sr = a.state_dict(), r.state_dict()
    assert set(sr) - set(sa) == {"readout_p.weight", "readout_q.weight"}
    for k in sa:
        assert torch.equal(sa[k], sr[k]), k
    assert torch.count_nonzero(sr["readout_p.weight"]) == 0
    assert torch.count_nonzero(sr["readout_q.weight"]) > 0
    torch.manual_seed(33)
    x = torch.randint(0, 10, (2, 12))
    for model in (a, r):
        _, aux = model(x, exit_depths=None)
        for s, lg in zip(aux["states"], aux["depth_logits"]):
            assert torch.equal(lg, s.float() @ model.embed.weight.float().T)
    _, a1 = a(x, exit_depths=None)
    _, a2 = r(x, exit_depths=None)
    for l1, l2 in zip(a1["depth_logits"], a2["depth_logits"]):
        assert torch.equal(l1, l2)


def test_readout_linear_resid_formula_linearity_and_switch_guard():
    """③ 对照臂读数公式 = (h + P·(Q h))·Eᵀ，且**严格线性**（与 gated 臂的
    区别就是无非线性）：readout(a·h1 + b·h2) == a·readout(h1) + b·readout(h2)。
    变异必抓红：(a) 忽略开关（P/Q 非零时仍返回 tied）；(b) 误把 gated 的非线性
    接上来（exact 公式对照）。"""
    r = tiny_model(readout_linear_resid=True)
    torch.manual_seed(34)
    with torch.no_grad():
        nn.init.normal_(r.readout_p.weight, std=0.3)
        nn.init.normal_(r.readout_q.weight, std=0.3)
    h = torch.randn(2, 12, 16)
    hf = h.float()
    expected = (hf + r.readout_p(r.readout_q(hf))) @ r.embed.weight.float().T
    got = r.readout(h)
    assert torch.allclose(got, expected, atol=0)
    # (a) 开关必须生效：P/Q 非零下读出必须偏离 tied
    assert not torch.allclose(got, hf @ r.embed.weight.float().T, atol=1e-6)
    # (b) 纯线性：叠加律成立（非线性读出会破坏它，gated 臂即反例）
    h1, h2 = torch.randn(2, 12, 16), torch.randn(2, 12, 16)
    got_mix = r.readout(0.7 * h1 - 1.3 * h2)
    mix = 0.7 * r.readout(h1) - 1.3 * r.readout(h2)
    assert torch.allclose(got_mix, mix, atol=1e-5)
    # 对照：同一叠加测试在 gated 臂上必须失败（证明该测试真的能区分非线性）
    g = tiny_model(readout_gated=True)
    torch.manual_seed(34)
    with torch.no_grad():
        nn.init.normal_(g.readout_w1.weight, std=0.3)
        nn.init.normal_(g.readout_w2.weight, std=0.3)
    g_mix = g.readout(0.7 * h1 - 1.3 * h2)
    g_lin = 0.7 * g.readout(h1) - 1.3 * g.readout(h2)
    assert not torch.allclose(g_mix, g_lin, atol=1e-4)


def test_readout_linear_resid_init_escapes_saddle():
    """干预活性门（SCHEMA §10 强制项）：P=Q=0 是鞍点（同 gated 零初始）。
    本 init（Q~N(0,0.02)、P=0）必须：(a) 起步严格恒等；(b) 一步 backward 后
    ∂L/∂P = grad ⊗ (Q h) ≠ 0 ⇒ 可逃逸鞍点；(c) 一步 optimizer 后 P 非零，
    且第二/三步 ∂L/∂Q ≠ 0（嵌套线性因子：Q 靠 P 离零后获得梯度）⇒ 全参数
    梯度可流。"""
    r = tiny_model(readout_linear_resid=True)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, r.cfg.n_layers,
                                torch.Generator().manual_seed(35))
    # (a) 起步恒等（P=0 ⇒ transform = 0）
    _, aux = r(x, exit_depths=depths)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        assert torch.equal(lg, s.float() @ r.embed.weight.float().T)
    opt = torch.optim.AdamW(r.parameters(), lr=1e-3)
    # (b) 起步 ∂L/∂P ≠ 0（非鞍点）；∂L/∂Q 在 P=0 处仍为 0（嵌套因子的固有
    # 现象——靠 P 先离零开通 Q 的梯度通道，故训练不会停住）
    loss, _ = r(x, exit_depths=depths)
    loss.backward()
    assert torch.count_nonzero(r.readout_p.weight.grad) > 0
    assert torch.count_nonzero(r.readout_q.weight.grad) == 0
    opt.step()
    opt.zero_grad(set_to_none=True)
    assert torch.count_nonzero(r.readout_p.weight) > 0
    # (c) P 离零后 ∂L/∂Q ≠ 0 ⇒ 全参数梯度可流
    loss, _ = r(x, exit_depths=depths)
    loss.backward()
    assert torch.count_nonzero(r.readout_q.weight.grad) > 0


def test_readout_linear_resid_equal_params_to_gated():
    """参数量判据（契约 B 核心）：线性对照臂与 gated 臂的**新增参数量完全
    相等**（各 2·d²）——只有「非线性」这一处功能差异。"""
    a = tiny_model()
    g = tiny_model(readout_gated=True)
    r = tiny_model(readout_linear_resid=True)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(g) - n(a) == n(r) - n(a) == 2 * 16 * 16


def test_readout_gated_and_linear_resid_mutually_exclusive():
    """单因素纪律：两种读出几何干预不得同时开启（否则混淆参数量/非线性）。"""
    with pytest.raises(ValueError):
        tiny_model(readout_gated=True, readout_linear_resid=True)


# ---------------- 层间拓扑（ticket 30 契约 E）----------------


def _manual_attn_delta(blk, ln, x):
    """手写注意力子层输出增量 proj(a(ln x))（因果 SDPA 路径），用于公式钉死。"""
    B, T, d = x.shape
    q, k, v = blk.qkv(ln(x)).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)
    a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v), is_causal=True)
    return blk.proj(a.transpose(1, 2).reshape(B, T, d))


def test_topology_default_off_matches_v1_bitwise():
    """缺省拓扑（不传 topology）必须与显式 sequential 逐位一致（v1 还原）。"""
    m_default = tiny_model(dense=True)
    m_seq = tiny_model(dense=True, topology="sequential")
    assert sum(p.numel() for p in m_default.parameters()) == \
        sum(p.numel() for p in m_seq.parameters())
    x = torch.randint(0, 10, (2, 12))
    assert torch.equal(m_default(x)[1]["states"][-1], m_seq(x)[1]["states"][-1])
    assert torch.equal(m_default(x)[0], m_seq(x)[0])


def test_parallel_block_formula_and_stock_path_guard():
    """并行子层公式逐位钉死：h' = h + attn(ln_a h) + mlp(ln_b h)，两个子层读
    同一块输入 h（不是链式）；并钉住它不接受退出/填充路径的参数。"""
    from pathlm.model import ModelConfig
    torch.manual_seed(3)
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, mlp_mult=4, seq_len=12)
    x = torch.randn(2, 6, 16)
    for share_ln in (True, False):
        b = ParallelBlock(cfg, share_ln=share_ln)
        out, dist = b(x)
        assert dist is None
        ln_b = b.ln1 if share_ln else b.ln2
        expect = x + _manual_attn_delta(b, b.ln1, x) + b.mlp(ln_b(x))
        assert torch.allclose(out, expect, atol=0, rtol=0)
        # mlp 分支读的是**块输入** x（链式实现读的是 h + attn(ln1 h)，会不同）
        staged = b.mlp(ln_b(x + _manual_attn_delta(b, b.ln1, x)))
        assert not torch.allclose(out, x + _manual_attn_delta(b, b.ln1, x) + staged)
        for kw in ({"kv_from": x}, {"gate_vec": torch.ones_like(x)},
                   {"dist_pen": 0.1}, {"attn_mask": torch.ones(1, 1, 6, 6,
                                                                dtype=torch.bool)}):
            with pytest.raises(ValueError):
                b(x, **kw)


def test_parallel_param_count_and_init_parity():
    """单因素 + SCHEMA §15：parallel-dual-ln 与 sequential 参数量**逐位相等**、
    且同 seed 下全部公共参数逐位相同（子模块创建顺序一致 ⇒ 随机数流对齐）；
    parallel（共享 pre-LN）按定义少 L·2d 个 LN 参数。"""
    seq = tiny_model(dense=True)
    dual = tiny_model(dense=True, topology="parallel-dual-ln")
    shared = tiny_model(dense=True, topology="parallel")
    n = lambda m: sum(p.numel() for p in m.parameters())
    d, L = 16, 3
    assert set(TOPOLOGIES) == {"sequential", "parallel", "parallel-dual-ln",
                              "dense"}
    assert n(dual) == n(seq)
    assert n(seq) - n(shared) == L * 2 * d
    # 包装层命名差（blocks.k.block.* vs blocks.k.*）在对照时按同一逻辑名映射
    def logical(sd):
        out = {}
        for k, v in sd.items():
            parts = k.split(".")
            if parts[0] == "blocks" and parts[2] == "block":
                parts = parts[:2] + parts[3:]
            out[".".join(parts)] = v
        return out
    sd_seq, sd_dual, sd_shared = logical(seq.state_dict()), \
        logical(dual.state_dict()), logical(shared.state_dict())
    assert set(sd_seq) - set(sd_dual) == set()      # dual 不丢任何公共参数
    for k, v in sd_seq.items():
        assert torch.equal(v, sd_dual[k]), k
        if k in sd_shared:                          # 共享 LN 臂只有 ln1/ln2 之差
            assert torch.equal(v, sd_shared[k]), k
    assert set(sd_seq) - set(sd_shared) == {
        f"blocks.{k}.ln2.{p}" for k in range(L) for p in ("weight", "bias")}


def test_parallel_wiring_changes_computation():
    """并行接线必须是**活**的（不是顺序残差的等价物）：把块内权重放大到
    delta 与 h 同量级后（初始化尺度下 delta 很小，接线差异会被淹没），
    同一输入下并行臂与 sequential 输出显著不同；共享 LN 与双 LN 变体也不相同。
    另记初始化尺度下的接线差异（很小）作为参考量。"""
    seq = tiny_model(dense=True)
    dual = tiny_model(dense=True, topology="parallel-dual-ln")
    shared = tiny_model(dense=True, topology="parallel")
    x = torch.randint(0, 10, (2, 12))
    rel = lambda u, v: float(((u - v).norm() / v.norm()).detach())
    s0, a0, b0 = (m(x)[1]["states"][-1].detach() for m in (seq, dual, shared))
    init_rel_dual, init_rel_shared = rel(a0, s0), rel(b0, s0)
    assert init_rel_dual > 0 and init_rel_shared > 0     # 接线确实改变计算

    def amplify(m, seed):
        """同 seed 下三个臂得到**同一套**块权重（每臂各自重置生成器）。"""
        rng = torch.Generator().manual_seed(seed)
        for blk in m.blocks:
            b = getattr(blk, "block", blk)
            for mod in (b.qkv, b.proj, b.mlp[0], b.mlp[2]):
                nn.init.normal_(mod.weight, std=0.3, generator=rng)

    for m in (seq, dual, shared):
        amplify(m, 11)
    s, a, b = (m(x)[1]["states"][-1].detach() for m in (seq, dual, shared))
    assert rel(a, s) > 0.05, (init_rel_dual, rel(a, s))
    assert rel(b, s) > 0.05, (init_rel_shared, rel(b, s))
    # 初始化时 ln1 == ln2（都是默认 ones/zeros）⇒ 双 LN 与共享 LN 变体此时重合；
    # ln2 被扰动（训练后必然如此）后两者必须分道扬镳。
    assert torch.allclose(a, b, atol=1e-7)

    def perturb_ln2(m, seed):
        rng = torch.Generator().manual_seed(seed)
        for blk in m.blocks:
            b_ = getattr(blk, "block", blk)
            if hasattr(b_, "ln2"):
                nn.init.normal_(b_.ln2.weight, std=0.3, generator=rng)
                nn.init.normal_(b_.ln2.bias, std=0.1, generator=rng)

    perturb_ln2(seq, 12)
    perturb_ln2(dual, 12)
    a2 = dual(x)[1]["states"][-1].detach()
    assert rel(a2, b) > 0.05, rel(a2, b)
    assert rel(a2, s0) > 0                                                 # 且仍非基线
    assert not torch.equal(a2, seq(x)[1]["states"][-1].detach())


def test_dense_identity_start_bitwise_equals_sequential():
    """dense 拓扑的层权 identity 起步（只取当前层）必须与 sequential **逐位**
    等价（严格恒等起步、零初始税）；参数量差 = L(L+1)/2 个标量（§15 记账）。"""
    seq = tiny_model(dense=True)
    den = tiny_model(dense=True, topology="dense")
    L = seq.cfg.n_layers
    assert sum(p.numel() for p in den.parameters()) - \
        sum(p.numel() for p in seq.parameters()) == L * (L + 1) // 2
    den.load_state_dict(seq.state_dict(), strict=False)
    a = den.layer_mix.detach().clone()
    for k in range(1, L + 1):
        assert float(a[k * (k - 1) // 2 + k - 1]) == 1.0
        assert float(a[k * (k - 1) // 2: k * (k + 1) // 2].sum()) == 1.0
    x = torch.randint(0, 10, (2, 12))
    _, aux_seq = seq(x)
    _, aux_den = den(x)
    assert torch.equal(aux_den["states"][-1], aux_seq["states"][-1])
    assert torch.equal(aux_den["depth_logits"][-1], aux_seq["depth_logits"][-1])
    # 层权被故意改动后必须偏离 sequential（证明该测试真的在测层权）
    with torch.no_grad():
        den.layer_mix[1] = 0.5
    assert not torch.equal(den(x)[1]["states"][-1], aux_seq["states"][-1])


def test_dense_layer_mix_formula():
    """dense 拓扑的层输入 = Σ_j α_kj · h_j（含 depth-0 输入态），用同一套块权重
    手算对照：α 设为 [0.3, 0.7] 时必须等于 blk1(0.3·h0 + 0.7·h1)。"""
    torch.manual_seed(7)
    seq = tiny_model(dense=True)
    den = tiny_model(dense=True, topology="dense")
    den.load_state_dict(seq.state_dict(), strict=False)
    with torch.no_grad():
        den.layer_mix[0] = 1.0          # 层 1 仍只取 h0
        den.layer_mix[1], den.layer_mix[2] = 0.3, 0.7   # 层 2：0.3·h0 + 0.7·h1
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        _, aux_seq = seq(x, exit_depths=None)
        _, aux_den = den(x, exit_depths=None)
        h0 = aux_seq["states"][0]
        h1 = seq.blocks[0](h0)[0]
        manual = seq.blocks[1](0.3 * h0 + 0.7 * h1)[0]
    assert torch.allclose(aux_den["states"][1], h1, atol=0, rtol=0)
    assert torch.allclose(aux_den["states"][2], manual, atol=1e-6)
    assert not torch.allclose(aux_den["states"][2], aux_seq["states"][2], atol=1e-6)


def test_dense_layer_mix_activity_escapes_identity():
    """干预活性（SCHEMA §10）：identity 起步点非鞍点——一步 backward 后
    ∂L/∂α 非零，一步 optimizer 后 α 离开 identity（否则干预是死的）。"""
    den = tiny_model(dense=True, topology="dense")
    x = torch.randint(0, 10, (2, 12))
    loss, _ = den(x, exit_depths=None)
    loss.backward()
    g = den.layer_mix.grad
    assert g is not None and torch.count_nonzero(g) > 0
    init = den.layer_mix.detach().clone()
    opt = torch.optim.AdamW(den.parameters(), lr=1e-3)
    opt.step()
    assert (den.layer_mix.detach() - init).abs().max() > 0
    live = topology_liveness(den)
    assert live["alive"] and live["max_dev_from_identity_init"] > 0
    assert live["params_added"] == den.cfg.n_layers * (den.cfg.n_layers + 1) // 2


def test_topology_guard_requires_dense_and_known_value():
    """单因素守卫：拓扑干预只在 dense=True 形态下开放（退出/填充语义下的并行
    接线未定义）；未知拓扑值直接报错；_gated_step（探针/解码路径）拒绝非
    sequential 拓扑，避免用错语义。"""
    for topo in ("parallel", "parallel-dual-ln", "dense"):
        with pytest.raises(ValueError):
            tiny_model(topology=topo)
    with pytest.raises(ValueError):
        tiny_model(dense=True, topology="paralell")
    m = tiny_model(dense=True, topology="dense")
    with pytest.raises(ValueError):
        m._gated_step(0, torch.zeros(1, 1, 16))


def test_topology_liveness_params_added_matches_param_delta():
    """活性块（SCHEMA §10）的 `params_added` 必须与**实测参数量增量**逐位一致，
    否则「活性量」与「参数账」会在两个文件间自相矛盾（本实验踩过：把
    per-block 的 −2d 当成 total 的 −L·2d）。"""
    ms = {t: tiny_model(dense=True, topology=t) for t in TOPOLOGIES}
    n = lambda m: sum(p.numel() for p in m.parameters())
    base = n(ms["sequential"])
    d, L = 16, 3
    for t, m in ms.items():
        live = topology_liveness(m)
        assert live["params_added"] == n(m) - base, (t, live["params_added"],
                                                     n(m) - base)
    assert topology_liveness(ms["parallel"])["params_added"] == -L * 2 * d
    assert topology_liveness(ms["dense"])["params_added"] == L * (L + 1) // 2
    # 两个可逐位核对的布尔量：等参数与起步等价性（不得互相代替）
    assert topology_liveness(ms["parallel-dual-ln"])["params_identical_to_sequential"]
    assert not topology_liveness(ms["parallel-dual-ln"])[
        "start_bitwise_equals_sequential"]
    assert topology_liveness(ms["dense"])["start_bitwise_equals_sequential"]
    assert not topology_liveness(ms["dense"])["params_identical_to_sequential"]
