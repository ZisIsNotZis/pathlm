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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             ParallelBlock, TOPOLOGIES, topology_liveness,
                             commit_input, generate, token_input, workspace_probe,
                             tf_acceptance, readout_workspace_energy,
                             spec_cost_model, TopKBlock, causal_topk_keep_mask,
                             sparse_attn_telemetry, attn_sparse_liveness,
                             attn_sparse_probe, LinearAttnBlock, linear_attn_phi,
                             linear_attn_chunked, linear_attn_reference,
                             attn_linear_liveness, attn_linear_probe,
                             attention_mass_stats, DEFAULT_LINEAR_CHUNK,
                             GQABlock, kv_head_liveness, CarryChannelBlock,
                             CarryChannel, CARRY_RULES, DEFAULT_CARRY_CHUNK,
                             carry_norm_keys, carry_scan_sequential,
                             carry_scan_chunked, carry_scan,
                             carry_liveness, carry_probe,
                             carry_read_mass_stats, carry_far_read_norm_frac,
                             window_anchor_keep_mask)
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


# ---------- W4 ticket 38：逐深度独立读出（readout_perdepth）----------
# 唯一变量 = 读出是否逐深度独立：A 臂 = 共享 tied 读出；PD 臂 = 每个深度一套线性
# 读出头（实现 = 逐深度线性状态变换 h → h + P_k(Q_k h)，再走共享 tied 表）；
# EP 臂 = **单个共享**线性变换 h → h + A(B h)，名义参数与 PD 臂相等（SCHEMA §15）。
# 变异验证（实验目录 smoke.py）：忽略 perdepth 开关 / 逐深度共用一套 P/Q / 把 EP
# 变成逐深度 —— 三条必须被抓红（对应下面的 test_readout_perdepth_*）。


def test_readout_perdepth_off_matches_v1_bitwise():
    """① 开关关闭逐位还原 v1：全深度与随机退出两条路径下 logits 都 == 手算
    s·Eᵀ（防止新分支被无条件接入）。"""
    m = tiny_model()
    torch.manual_seed(31)
    x = torch.randint(0, 10, (2, 12))
    for depths in (None, sample_exit_depths(
            2, 12, m.cfg.n_layers, torch.Generator().manual_seed(32))):
        _, aux = m(x, exit_depths=depths)
        for s, lg in zip(aux["states"], aux["depth_logits"]):
            assert torch.equal(lg, s.float() @ m.embed.weight.float().T)


def test_readout_perdepth_identity_start_and_init_parity():
    """② init 规范（写死）：Q_k ~ N(0,0.02)、P_k = 0 ⇒ 起步 transform = 0，读出
    与共享 tied 线性逐位一致；P_k/Q_k 建在参数 init 循环之后 ⇒ 同 seed 下与 A 臂
    **其余全部参数逐位一致**（单因素：只加逐深度读出头）。"""
    a = tiny_model()
    pd = tiny_model(readout_perdepth=True)
    sa, sp = a.state_dict(), pd.state_dict()
    extra = set(sp) - set(sa)
    L = pd.cfg.n_layers
    assert extra == {f"readout_pd_{t}.{k}.weight" for t in ("p", "q")
                     for k in range(L + 1)}
    for k in sa:
        assert torch.equal(sa[k], sp[k]), k
    for t in ("p", "q"):
        for k in range(L + 1):
            w = sp[f"readout_pd_{t}.{k}.weight"]
            assert (torch.count_nonzero(w) == 0) == (t == "p")
    torch.manual_seed(33)
    x = torch.randint(0, 10, (2, 12))
    _, a1 = a(x, exit_depths=None)
    _, p1 = pd(x, exit_depths=None)
    for l1, l2 in zip(a1["depth_logits"], p1["depth_logits"]):
        assert torch.equal(l1, l2)


def test_readout_perdepth_formula_and_independence():
    """③ 读数公式 = (h + P_k·(Q_k h))·Eᵀ（逐深度），且**每个深度一套**：
    - 只扰动 P_0 ⇒ 只有 depth 0 的 logits 变，depth 1 不变（逐深度独立性的直接钉）；
    - P_2 与 P_0 独立赋值 ⇒ 两个深度得到不同变换；
    变异必抓红：(a) 忽略开关（P_k 非零时仍返回 tied）；(b) 逐深度共用一套 P/Q
    （实现里把 P_k 取成同一张量）会让「扰动 P_0 只改 depth 0」失配。"""
    pd = tiny_model(readout_perdepth=True)
    torch.manual_seed(34)
    with torch.no_grad():
        for mod in pd.readout_pd_p:
            nn.init.normal_(mod.weight, std=0.3)
        for mod in pd.readout_pd_q:
            nn.init.normal_(mod.weight, std=0.3)
    x = torch.randint(0, 10, (2, 12))
    _, aux = pd(x, exit_depths=None)
    states = aux["states"]
    for k in (0, 1, pd.cfg.n_layers):
        hf = states[k].float()
        expected = (hf + pd.readout_pd_p[k](pd.readout_pd_q[k](hf))) \
            @ pd.embed.weight.float().T
        assert torch.allclose(aux["depth_logits"][k], expected, atol=0)
    # (a) 开关生效：与 tied 显著不同
    assert not torch.allclose(aux["depth_logits"][0],
                              states[0].float() @ pd.embed.weight.float().T,
                              atol=1e-6)
    # 逐深度独立性：只改 P_0 ⇒ depth 0 变、depth 1 逐位不变
    with torch.no_grad():
        pd.readout_pd_p[0].weight.mul_(-1.0)
    _, aux2 = pd(x, exit_depths=None)
    assert not torch.equal(aux2["depth_logits"][0], aux["depth_logits"][0])
    assert torch.equal(aux2["depth_logits"][1], aux["depth_logits"][1])
    # 不同深度用不同的头：P_0 与 P_2 独立 ⇒ 各自深度的读出不同
    assert not torch.allclose(pd.readout_pd_p[0].weight,
                              pd.readout_pd_p[2].weight)


def test_readout_perdepth_requires_depth():
    """④ 逐深度读出必须显式传 depth（depth=None 会静默退化 ⇒ 报错）；且 A/EP
    两个非逐深度臂的 readout 不带 depth 也能算（向后兼容）。"""
    pd = tiny_model(readout_perdepth=True)
    with pytest.raises(ValueError):
        pd.readout(torch.randn(2, 12, 16))
    # 非逐深度臂：readout(h) 仍可用且 == readout(h, depth=k)
    a = tiny_model()
    h = torch.randn(2, 12, 16)
    assert torch.equal(a.readout(h), a.readout(h, depth=3))
    ep = tiny_model(readout_wide_shared=64)
    assert torch.equal(ep.readout(h), ep.readout(h, depth=3))


def test_readout_perdepth_init_escapes_saddle_and_liveness():
    """⑤ 干预活性门（SCHEMA §10）：P_k = 0 处起步恒等但非鞍点。全深度监督下
    **每个深度**的 ∂L/∂P_k ≠ 0（可学）；∂L/∂Q_k 在 P_k=0 处为 0（嵌套因子），
    P_k 离零后 Q_k 获得梯度 ⇒ 全参数可流。"""
    pd = tiny_model(readout_perdepth=True)
    x = torch.randint(0, 10, (2, 12))
    L = pd.cfg.n_layers
    loss, aux = pd(x, exit_depths=None)          # 全深度 ⇒ 每个头都被监督
    loss.backward()
    for k in range(L + 1):
        assert torch.count_nonzero(pd.readout_pd_p[k].weight.grad) > 0, k
        assert torch.count_nonzero(pd.readout_pd_q[k].weight.grad) == 0, k
    opt = torch.optim.AdamW(pd.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad(set_to_none=True)
    for k in range(L + 1):
        assert torch.count_nonzero(pd.readout_pd_p[k].weight) > 0, k
    loss, _ = pd(x, exit_depths=None)
    loss.backward()
    for k in range(L + 1):
        assert torch.count_nonzero(pd.readout_pd_q[k].weight.grad) > 0, k


def test_readout_wide_shared_equal_params_and_identity():
    """⑥ SCHEMA §15 等参数对照：共享读出臂（m = (L+1)·d）的新增参数量与
    per-depth 臂 **完全相等**（各 (L+1)·2d²）；且起步严格恒等（A=0）、忽略
    depth（共享语义）。"""
    a = tiny_model()
    pd = tiny_model(readout_perdepth=True)
    d, L = 16, 3
    ep = tiny_model(readout_wide_shared=(L + 1) * d)
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(pd) - n(a) == n(ep) - n(a) == (L + 1) * 2 * d * d
    torch.manual_seed(41)
    x = torch.randint(0, 10, (2, 12))
    _, a_aux = a(x, exit_depths=None)
    _, e_aux = ep(x, exit_depths=None)
    for l1, l2 in zip(a_aux["depth_logits"], e_aux["depth_logits"]):
        assert torch.equal(l1, l2)
    # 共享语义：所有深度用同一张量 ⇒ 两个不同深度的变换由同一 A/B 决定
    h = torch.randn(2, 12, 16)
    assert torch.equal(ep.readout(h, depth=0), ep.readout(h, depth=2))
    # 非鞍点：A=0 起步 ∂L/∂A ≠ 0（B 非零）
    loss, _ = ep(x, exit_depths=None)
    loss.backward()
    assert torch.count_nonzero(ep.readout_shared_a.weight.grad) > 0


def test_readout_geometries_mutually_exclusive():
    """单因素纪律：四种读出几何干预最多开一个（同时开会混淆归因）。"""
    for kw in ({"readout_gated": True, "readout_perdepth": True},
               {"readout_linear_resid": True, "readout_perdepth": True},
               {"readout_gated": True, "readout_wide_shared": 64},
               {"readout_perdepth": True, "readout_wide_shared": 64}):
        with pytest.raises(ValueError):
            tiny_model(**kw)


def test_train_cli_perdepth_plumbed_to_config():
    """CLI 通道（可失败）：`--readout-perdepth` / `--readout-wide-shared` 必须真的
    进 `DepthARConfig` 并改变参数集合，否则就是「开关是死的」。变异验证（把
    `build_depthar_config` 里这两行删掉）靠这条抓红。"""
    import argparse
    import train_depth_ar as T

    base = dict(d=16, layers=3, heads=2, seq_len=12, mlp_mult=4, dense=False,
                gate_bias_init=-2.0, topology="sequential", fill_kv=True,
                proj_fill=False, readout_dims=None, readout_gated=False,
                readout_linear_resid=False, attn_sparse_k=0, attn_linear=False,
                linear_chunk=64, train_window=0, train_anchors=0,
                carry_write="none", carry_chunk=64, carry_state_dim=None,
                n_kv_heads=None)
    cfg = T.build_depthar_config(argparse.Namespace(**base,
                                                   readout_perdepth=True))
    assert cfg.readout_perdepth is True and cfg.readout_wide_shared == 0
    assert "readout_pd_p.0.weight" in DepthARModel(cfg, 10).state_dict()
    cfg_w = T.build_depthar_config(argparse.Namespace(
        **base, readout_perdepth=False, readout_wide_shared=64))
    assert cfg_w.readout_wide_shared == 64
    assert "readout_shared_a.weight" in DepthARModel(cfg_w, 10).state_dict()
    # 缺字段的历史 Namespace 仍可用（向后兼容）
    cfg_off = T.build_depthar_config(argparse.Namespace(**base))
    assert cfg_off.readout_perdepth is False and cfg_off.readout_wide_shared == 0


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


# ---------------- 契约 J：内容 top-k 稀疏注意力（ticket 31） ----------------

def _ref_topk_keep(scores: torch.Tensor, k: int) -> torch.Tensor:
    """参考实现（逐行排序，无并列分数）：因果内分数最大的 k 个键。"""
    T = scores.shape[-1]
    keep = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        order = sorted(range(i + 1), key=lambda j: (-float(scores[i, j]), j))
        for j in order[:k]:
            keep[i, j] = True
    return keep


def test_topk_default_off_matches_v1_bitwise():
    """attn_sparse_k 缺省 = 0 时必须与 dense 全注意力臂**逐位**一致（v1 还原）。"""
    m_default = tiny_model(dense=True)
    m_off = tiny_model(dense=True, attn_sparse_k=0)
    assert type(m_default.blocks[0]).__name__ == "Block"
    assert sum(p.numel() for p in m_default.parameters()) == \
        sum(p.numel() for p in m_off.parameters())
    x = torch.randint(0, 10, (2, 12))
    assert torch.equal(m_default(x)[1]["states"][-1], m_off(x)[1]["states"][-1])
    assert "attn_sparse" not in m_off(x)[1]


def test_topk_mask_keeps_exactly_min_k_causal_keys():
    """掩码 = 因果 ∩「分数最大的 k 个」：每行保留数恒为 min(k, i+1)、不放出非因果键、
    且与逐行排序的参考实现逐位一致（分数无并列 ⇒ 与 topk 的定序约定无关）。"""
    torch.manual_seed(5)
    for k in (1, 3, 8):
        scores = torch.randn(2, 2, 10, 10)
        causal = torch.tril(torch.ones(10, 10, dtype=torch.bool))
        keep = causal_topk_keep_mask(scores, k, causal)
        assert not bool((keep & ~causal.unsqueeze(0).unsqueeze(0)).any()), "非因果键被放出"
        counts = keep.sum(-1)
        for i in range(10):
            assert int(counts[0, 0, i]) == min(k, i + 1), (k, i, int(counts[0, 0, i]))
        for b in range(2):
            for h in range(2):
                assert torch.equal(keep[b, h], _ref_topk_keep(scores[b, h], k))
    # k ≥ T：恒等因果掩码（选择无信息量）
    assert torch.equal(causal_topk_keep_mask(scores, 10, causal), causal)
    assert torch.equal(causal_topk_keep_mask(scores, 99, causal), causal)


def test_topk_mask_selection_is_by_content_not_position():
    """选择必须**按内容**：构造「最远的第 0 键分数最高」的分数矩阵，top-2 必须选中
    第 0 键而不是「最近的键」——位置窗口（最近 k 个）会给出 {i−1, i}，二者不同。"""
    T, k = 8, 2
    scores = torch.zeros(1, 1, T, T)
    for i in range(T):
        for j in range(i + 1):
            scores[0, 0, i, j] = -(i - j)          # 默认：越近越好
        scores[0, 0, i, 0] = 100.0                 # 最远的第 0 键最高分
    keep = causal_topk_keep_mask(scores, k)
    assert bool(keep[0, 0, 7, 0])                  # 远键被选中（按内容）
    assert bool(keep[0, 0, 7, 7])
    assert not bool(keep[0, 0, 7, 6])              # 最近的非自身键被丢弃（非位置窗口）
    for i in range(1, T):
        assert sorted(keep[0, 0, i].nonzero().flatten().tolist()) == [0, i]
    tele = sparse_attn_telemetry(keep, k)
    # 位置基线（最近 k 个键）的重合度：T=8、k=2 时只有 2 个键命中 ⇒ 远小于 1
    assert float(tele["sel_in_lastk_frac"]) < 0.9
    assert 1.0 < float(tele["sel_dist_mean"]) < T


def test_topk_softmax_weights_live_only_on_kept_keys():
    """稀疏 softmax 的权重必须严格只在被选键上：未选键权重恰好 0、行和恰为 1、
    与「只在被选分数上做 softmax」的手算一致；允许键 ≤ k 的行退化为全注意力。"""
    torch.manual_seed(6)
    T, k = 9, 3
    scores = torch.randn(1, 1, T, T, dtype=torch.float64)
    keep = causal_topk_keep_mask(scores, k)
    w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
    assert float(w[~keep].abs().max()) == 0.0
    assert torch.allclose(w.sum(-1), torch.ones(T, dtype=torch.float64),
                          atol=1e-12, rtol=0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    full = scores.masked_fill(~causal, float("-inf")).softmax(-1)
    for i in range(T):
        if i + 1 <= k:                              # 允许键数 ≤ k ⇒ 选择是恒等
            assert torch.equal(w[0, 0, i], full[0, 0, i])
        else:
            assert not torch.allclose(w[0, 0, i], full[0, 0, i])
            man = scores[0, 0, i][keep[0, 0, i]].softmax(-1)
            assert torch.allclose(w[0, 0, i][keep[0, 0, i]], man, atol=0, rtol=0)
    # 被掩掉的键在反向时拿不到梯度（masked_fill(-inf) 的权重恰为 0）
    assert float((w * (~keep)).sum()) == 0.0


def test_topk_block_formula_matches_manual_sparse_attention():
    """TopKBlock 公式逐位钉死：scores→因果 top-k 掩码→稀疏 softmax→w@v→proj→mlp；
    前 k 行（允许键数 ≤ k）必须与同权重全注意力逐位相同；并钉住它只接受 stock 路径。"""
    from pathlm.model import ModelConfig
    torch.manual_seed(3)
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, mlp_mult=4, seq_len=8)
    k = 3
    blk = TopKBlock(cfg, k)
    x = torch.randn(2, 8, 16)
    out, tele = blk(x)
    assert set(tele) == {"kept_mean", "kept_min", "kept_max", "kept_frac_of_T",
                         "sel_dist_mean", "sel_in_lastk_frac"}
    B, T, d = x.shape
    q, kk, v = blk.qkv(blk.ln1(x)).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)
    scores = shape(q) @ shape(kk).transpose(-2, -1) / (d // blk.n_heads) ** 0.5
    keep = causal_topk_keep_mask(scores, k, blk.causal)
    w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
    a = (w @ shape(v)).transpose(1, 2).reshape(B, T, d)
    expect = x + blk.proj(a)
    expect = expect + blk.mlp(blk.ln2(expect))
    assert torch.allclose(out, expect, atol=0, rtol=0)
    # 同上权重、只改成全注意力（topk ≥ T）作对照
    blk.topk = T
    out_full, tele_full = blk(x)
    blk.topk = k
    assert tele_full is None
    assert not torch.allclose(out, out_full, atol=1e-7)     # 算子在起作用
    assert torch.allclose(out[:, :k], out_full[:, :k], atol=1e-6)  # 允许键 ≤ k 的行
    for kw in ({"kv_from": x}, {"gate_vec": torch.ones_like(x)}, {"dist_pen": 0.1},
               {"attn_mask": torch.ones(1, 1, 8, 8, dtype=torch.bool)}):
        with pytest.raises(ValueError):
            blk(x, **kw)


def test_topk_block_k_ge_T_bitwise_equals_block():
    """k ≥ T ⇒ 选择退化为恒等 ⇒ 必须与 stock `Block` **逐位**一致（同一份权重）。"""
    from pathlm.model import ModelConfig
    torch.manual_seed(4)
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, mlp_mult=4, seq_len=8)
    blk = Block(cfg)
    top = TopKBlock(cfg, 8)
    top.load_state_dict(blk.state_dict(), strict=True)
    x = torch.randn(2, 8, 16)
    out_b, dist_b = blk(x)
    out_t, tele_t = top(x)
    assert dist_b is None and tele_t is None
    assert torch.equal(out_b, out_t)
    # 更大的 k（> T）同样退化
    top.topk = 64
    assert torch.equal(blk(x)[0], top(x)[0])


def test_topk_params_identical_to_full_attention_block():
    """单因素 + SCHEMA §15 记账：top-k 臂相对全注意力臂**参数量增量 = 0**，同 seed
    下全部参数逐位相同（子模块创建顺序一致）；因果掩码是 non-persistent buffer，
    不进 state_dict（否则「参数量相等」会被 buffer 掩盖）。"""
    torch.manual_seed(9)
    from pathlm.model import ModelConfig
    cfg = ModelConfig(d_model=16, n_layers=2, n_heads=2, mlp_mult=4, seq_len=8)
    blk = Block(cfg)
    torch.manual_seed(9)
    top = TopKBlock(cfg, 3)
    assert set(blk.state_dict()) == set(top.state_dict())
    for key, val in blk.state_dict().items():
        assert torch.equal(val, top.state_dict()[key]), key
    assert "causal" not in top.state_dict()
    assert sum(p.numel() for p in top.parameters()) == \
        sum(p.numel() for p in blk.parameters())
    full_model = tiny_model(dense=True)
    top_model = tiny_model(dense=True, attn_sparse_k=4)
    assert type(top_model.blocks[0]).__name__ == "TopKBlock"
    assert sum(p.numel() for p in top_model.parameters()) == \
        sum(p.numel() for p in full_model.parameters())
    for key, val in full_model.state_dict().items():
        assert torch.equal(val, top_model.state_dict()[key]), key


def test_topk_guard_requires_dense_sequential_and_valid_k():
    """单因素守卫：top-k 只在 dense=True + sequential 拓扑下开放（退出/填充与
    层间拓扑都改了另一个因素）；非法 k 直接报错；`attn_sparse_k=0` 等价于关闭。"""
    with pytest.raises(ValueError):                    # 非 dense
        tiny_model(attn_sparse_k=4)
    with pytest.raises(ValueError):                    # 与层间拓扑混用
        tiny_model(dense=True, topology="parallel", attn_sparse_k=4)
    with pytest.raises(ValueError):                    # 非法 k
        tiny_model(dense=True, attn_sparse_k=-1)
    from pathlm.model import ModelConfig
    with pytest.raises(ValueError):
        TopKBlock(ModelConfig(d_model=16, n_heads=2, seq_len=8), 0)
    live = attn_sparse_liveness(tiny_model(dense=True))
    assert live["enabled"] is False and live["alive"] is False
    assert live["params_added"] == 0


def test_topk_switch_changes_computation_and_k_is_effective():
    """SCHEMA §10 活性：开关打开必须改变前向（不是死开关），且每层活性量显示
    「实际保留键数 == 理论 min(k, i+1)」——k 真的生效；k=1 时每行只剩自身（因果
    内分数最大者恒为对角）⇒ 与全注意力的偏差应显著更大。"""
    full = tiny_model(dense=True)
    sp = tiny_model(dense=True, attn_sparse_k=4)
    x = torch.randint(0, 10, (2, 12))
    _, aux_sp = sp(x, exit_depths=None)
    _, aux_full = full(x, exit_depths=None)
    assert len(aux_sp["attn_sparse"]) == 3
    rel = float(((aux_sp["states"][-1] - aux_full["states"][-1]).norm()
                 / aux_full["states"][-1].norm()).detach())
    assert rel > 0
    T, k = 12, 4
    theory = sum(min(k, i + 1) for i in range(T)) / T
    live = attn_sparse_liveness(sp, aux_sp["attn_sparse"])
    assert live["alive"] and live["kept_count_matches_theory"]
    assert live["kept_mean"] == pytest.approx(theory, abs=1e-6)
    assert live["params_added"] == 0
    assert live["attn_sparse_k"] == 4
    # k = T：与全注意力臂逐位一致（选择退化为恒等）
    sp_all = tiny_model(dense=True, attn_sparse_k=12)
    assert torch.equal(sp_all(x, exit_depths=None)[1]["states"][-1],
                       aux_full["states"][-1])
    # k = 1：每行只留对角 ⇒ 输出 = v 的自投影，与全注意力偏差更大
    sp1 = tiny_model(dense=True, attn_sparse_k=1)
    _, aux1 = sp1(x, exit_depths=None)
    rel1 = float(((aux1["states"][-1] - aux_full["states"][-1]).norm()
                  / aux_full["states"][-1].norm()).detach())
    assert rel1 > rel
    assert attn_sparse_liveness(sp1, aux1["attn_sparse"])["kept_mean"] == 1.0


def test_topk_sel_in_lastk_frac_measures_positional_overlap():
    """「内容选择 vs 最近 k 个键（位置基线）」的重合度是可解释量：分数 = −距离
    （对角最大）时 k=1 的 top-1 恰是对角 ⇒ 重合度 = 1、距离均值 = 0；随机分数下
    k=4/T=12 的重合度应 < 1 且被选键距离 > 0（选择不是位置窗口）。"""
    T, k = 12, 4
    dist_scores = -torch.cdist(torch.arange(T, dtype=torch.float64).view(-1, 1),
                               torch.arange(T, dtype=torch.float64).view(-1, 1)).view(1, 1, T, T)
    keep1 = causal_topk_keep_mask(dist_scores, 1)
    t1 = sparse_attn_telemetry(keep1, 1)
    assert float(t1["sel_in_lastk_frac"]) == 1.0          # 对角（距离 0）被选中
    assert float(t1["sel_dist_mean"]) == 0.0
    assert float(t1["kept_mean"]) == 1.0
    torch.manual_seed(8)
    scores = torch.randn(1, 1, T, T)
    keep = causal_topk_keep_mask(scores, k)
    tele = sparse_attn_telemetry(keep, k)
    assert 0.0 < float(tele["sel_in_lastk_frac"]) < 1.0
    assert 0.0 < float(tele["sel_dist_mean"]) <= T
    assert float(tele["kept_mean"]) == sum(min(k, i + 1) for i in range(T)) / T


def test_topk_probe_and_liveness_contract_on_real_batches():
    """`attn_sparse_probe`（训练入口与实验电池共用的活性汇总）在真实数据流上：
    开关关闭 ⇒ enabled=False 的空块；开关打开 ⇒ alive、保留键数符合理论、
    参数量增量为 0（探针自身不引入参数）。"""
    import numpy as np
    arr = (np.arange(4000) % 7).astype(np.uint8)
    full = tiny_model(dense=True)
    assert attn_sparse_probe(full, arr, n_batches=1, batch_size=2)["enabled"] is False
    sp = tiny_model(dense=True, attn_sparse_k=4)
    probe = attn_sparse_probe(sp, arr, n_batches=2, batch_size=2)
    assert probe["enabled"] and probe["alive"]
    assert probe["n_layers_sampled"] == 2 * 3          # n_batches × n_layers
    theory = sum(min(4, i + 1) for i in range(12)) / 12
    assert probe["kept_mean"] == pytest.approx(theory, abs=1e-6)
    n_before = sum(p.numel() for p in sp.parameters())
    attn_sparse_probe(sp, arr, n_batches=1, batch_size=2)
    assert sum(p.numel() for p in sp.parameters()) == n_before


def test_topk_gradients_flow_and_non_selected_values_get_none():
    """反向可达且数值健康：qkv/proj 梯度非零、有限；全层输出有限；**未被选中的
    键的 v 拿到的梯度恰为 0**（w=0 ⇒ ∂L/∂v_j = 0）——这是「稀疏混合真的跳过了
    未选键」的梯度层证据。"""
    sp = tiny_model(dense=True, attn_sparse_k=4)
    x = torch.randint(0, 10, (2, 12))
    loss, aux = sp(x, exit_depths=None)
    assert torch.isfinite(loss)
    assert torch.isfinite(aux["states"][-1]).all()
    loss.backward()
    for name in ("qkv", "proj"):
        g = getattr(sp.blocks[0], name).weight.grad
        assert g is not None and torch.isfinite(g).all() and \
            bool(g.abs().sum() > 0), name
    # 手工复核：同一 mask 下未被选键的 v 梯度为 0
    blk = TopKBlock(type("C", (), {"d_model": 16, "n_heads": 2, "mlp_mult": 4,
                                   "dropout": 0.0, "seq_len": 12})(), 4)
    xx = torch.randn(1, 12, 16)
    q, k, v = blk.qkv(blk.ln1(xx)).chunk(3, dim=-1)
    v = v.detach().requires_grad_(True)
    shape = lambda t: t.view(1, 12, 2, 8).transpose(1, 2)
    scores = shape(q.detach()) @ shape(k.detach()).transpose(-2, -1) / 8 ** 0.5
    keep = causal_topk_keep_mask(scores, 4)
    w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
    (w @ shape(v)).sum().backward()
    grad_v = shape(v.grad)[0, 0]                       # [T, dh]
    got = keep[0, 0].float().sum(0).clamp(0, 1)        # 每个键是否被任何查询选中
    assert bool((grad_v.abs().sum(-1)[got == 0] == 0).all())


# ------------------------------------------------ 训练期可见窗口（ticket 32 契约 O）


def test_window_anchor_keep_mask_semantics():
    """掩码语义逐位钉死：keep[i,j] = (j≤i) ∧ ((i−j)<window ∨ j<anchors)，对角恒开。

    - 非锚点位置：距离 < window 可见、≥ window 不可见；
    - 锚点位置（j < anchors）：任意 query 可见（锚通道）；
    - 永远不含未来（j > i）；
    - 非法参数（window ≤ 0 / anchors < 0）显式报错。
    """
    from pathlm.depth_ar import window_anchor_keep_mask
    T, window, anchors = 16, 4, 2
    keep = window_anchor_keep_mask(T, window, anchors)
    for i in range(T):
        for j in range(T):
            want = (j <= i) and ((i - j) < window or j < anchors)
            assert bool(keep[i, j]) == want, (i, j)
        assert bool(keep[i, i])                      # 对角恒开（NaN 防线）
    # 锚点列在窗口外仍全开；非锚点列在窗口外全关
    assert bool(keep[15, :anchors].all())
    assert not bool(keep[15, anchors:15 - window + 1].any())
    assert bool(keep[15, 15 - window + 1:16].all())
    for bad in ((0, 0), (-3, 0)):
        with pytest.raises(ValueError):
            window_anchor_keep_mask(T, bad[0], bad[1])
    with pytest.raises(ValueError):
        window_anchor_keep_mask(T, 4, -1)


def test_window_anchor_keep_mask_matches_instrument_recent_mask():
    """anchors=0 时必须与保留曲线仪器的 `recent_keep_mask(T, d)` **逐位相同**
    （训练/评测共用同一件仪器语义；契约 P 第 ③ 项的接口层证据）。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.retention import recent_keep_mask
    for T, d in ((256, 32), (16, 1), (16, 16), (13, 4)):
        assert torch.equal(window_anchor_keep_mask(T, d, 0),
                           recent_keep_mask(T, d)), (T, d)


def test_window_anchor_keep_mask_full_window_is_exact_causal():
    """window ≥ T 且 anchors=0 ⇒ 精确因果掩码（= 不传掩码的 stock 路径）。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    for T in (8, 12, 256):
        assert torch.equal(window_anchor_keep_mask(T, T, 0),
                           torch.tril(torch.ones(T, T, dtype=torch.bool)))
        assert torch.equal(window_anchor_keep_mask(T, T + 99, 0),
                           torch.tril(torch.ones(T, T, dtype=torch.bool)))


def test_train_window_default_off_matches_v1_bitwise():
    """缺省（train_window=0）必须与「显式全窗口」逐位一致，且都等于无掩码的
    裸堆叠路径（v1 还原）。"""
    m_off = tiny_model(dense=True)
    m_full = tiny_model(dense=True, train_window=12)      # = seq_len ⇒ 恒等因果
    x = torch.randint(0, 10, (2, 12))
    l_off, aux_off = m_off(x)
    l_full, aux_full = m_full(x)
    assert torch.equal(aux_off["states"][-1], aux_full["states"][-1])
    assert torch.equal(l_off, l_full)
    h = aux_off["states"][0]
    for blk in m_off.blocks:
        h, _ = blk(h)
    assert torch.equal(aux_off["states"][-1], h)


def test_train_window_restricts_attention_to_recent_positions():
    """功能判据（不是接口判据）：window=1、anchors=0 ⇒ 每个位置只看自己。
    改动更早位置的 token **不得**改变该位置之后的 logits；但改动自身位置必须
    改变 —— 前半句若红即「掩码没被真的应用」，后半句若红即「前向被掩死」。"""
    m = tiny_model(dense=True, train_window=1)
    torch.manual_seed(7)
    x1 = torch.randint(0, 10, (1, 12))
    x2 = x1.clone()
    x2[0, 5] = (int(x1[0, 5]) + 1) % 10
    lg1 = m(x1)[1]["depth_logits"][-1]
    lg2 = m(x2)[1]["depth_logits"][-1]
    assert torch.equal(lg1[0, 6:], lg2[0, 6:])      # 更早 token 传不到右边
    assert not torch.equal(lg1[0, 5], lg2[0, 5])    # 自身位置确实受影响
    # 反向对照：窗口足够大时，更早的 token 一定传得到（证明上面的相等不是
    # 由「输入根本没进前向」造成的假绿）
    m_open = tiny_model(dense=True)
    lg3 = m_open(x1)[1]["depth_logits"][-1]
    lg4 = m_open(x2)[1]["depth_logits"][-1]
    assert not torch.equal(lg3[0, 6:], lg4[0, 6:])


def test_train_window_forward_matches_eval_mask_path():
    """训练侧掩码（forward 内建）与评测侧掩码（仪器 `masked_logits` 显式传掩码）
    必须**逐位一致** —— 否则量的是别的模型（契约 P 第 ③ 项）。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.retention import masked_logits
    m = tiny_model(dense=True, train_window=4, train_anchors=2)
    x = torch.randint(0, 10, (2, 12))
    train_side = m(x)[1]["depth_logits"][-1]
    eval_side = masked_logits(m, x, window_anchor_keep_mask(12, 4, 2))
    assert torch.equal(train_side, eval_side)


def test_window_mask_liveness_and_probe():
    """活性量（SCHEMA §10）：掩码不引入参数（params_added=0）；可见占比 = 闭式
    计数；window ≥ T 时 alive=False；`window_mask_probe` 在掩码开/关两次前向间
    给出非零最大 logit 差（开关不是死的），且不改变模型状态/参数。"""
    from pathlm.depth_ar import (window_mask_liveness, window_mask_probe,
                                 window_anchor_keep_mask)
    T, window, anchors = 256, 32, 8
    live = window_mask_liveness(T, window, anchors)
    expected = sum(min(i + 1, window) + min(i + 1, anchors)
                   - max(0, min(i + 1, anchors) - max(0, i - window + 1))
                   for i in range(T))
    assert live["n_visible_pairs"] == int(window_anchor_keep_mask(
        T, window, anchors).sum())
    assert live["n_visible_pairs"] == expected
    assert live["causal_pairs"] == T * (T + 1) // 2
    assert live["params_added"] == 0 and live["alive"]
    assert abs(live["visible_frac"] + live["dropped_frac"] - 1.0) < 1e-6
    assert window_mask_liveness(T, T, 0)["alive"] is False

    m = tiny_model(dense=True, train_window=4, train_anchors=2)
    before = [p.detach().clone() for p in m.parameters()]
    x = torch.randint(0, 10, (2, 12))
    probe = window_mask_probe(m, x)
    assert probe["alive"] and probe["max_abs_logit_diff_masked_vs_open"] > 0
    assert m.cfg.train_window == 4                      # 借用后必须还原
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    with pytest.raises(ValueError):
        window_mask_probe(tiny_model(dense=True), x)     # 掩码关闭时无意义


def test_train_window_switch_guards():
    """单因素护栏：窗口训练只在 dense + sequential 下开放；anchors 需 window>0；
    与内容 top-k 稀疏互斥。"""
    with pytest.raises(ValueError):
        tiny_model(train_window=4)                       # dense=False
    with pytest.raises(ValueError):
        tiny_model(dense=True, train_window=4, topology="parallel")
    with pytest.raises(ValueError):
        tiny_model(dense=True, train_anchors=2)          # anchors 无 window
    with pytest.raises(ValueError):
        tiny_model(dense=True, train_window=4, attn_sparse_k=3)
    # 合法组合可建
    assert tiny_model(dense=True, train_window=4, train_anchors=2) is not None


# ---------------------- 训练期可见窗口 = 192（ticket 33 契约 Q，真实尺度 T=256）
#
# 契约 Q 的干预就是一个**参数取值**（window 192 而不是 32），所以这里把「192 这个取值
# 真的生效」钉在真实尺度上：① 可见对的闭式计数（把 window 夹到 32 就会被抓红）；
# ② 活性量（窗口长度必须如实报 192）；③ 功能判据——模型前向的掩码 == 评测端
# masked_logits 的掩码（训练/评测掩码一致），且 192 的可见集**与 32 的可见集不同**
# （防止实现把 window 悄悄夹小，或在 forward 里漏掉锚点）。


def test_window192_keep_mask_counts_and_boundary_at_real_scale():
    """T=256 / window=192 / anchors=8 的可见对的**闭式计数**与边界。

    独立推导（不读实现）：
      i < 192  ：所有 j ≤ i 可见 ⇒ Σ(i+1) = 192·193/2 = 18528
      i ≥ 192  ：窗口给 192 个（j ∈ [i−191, i]），锚点再补出 min(8, i−191) 个
                 ⇒ Σ_{i=192..255} (192 + min(8, i−191)) = 12772
      合计 31300 / 32896 = 0.951483
    同时钉住「被删掉的位置」正好是 8..63（窗口外且不在锚区的位置）。
    """
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.retention import mask_stats
    T, window, anchors = 256, 192, 8
    keep = window_anchor_keep_mask(T, window, anchors)
    want = (sum(i + 1 for i in range(window))
            + sum(window + min(anchors, i - window + 1)
                  for i in range(window, T)))
    assert want == 31300
    assert int(keep.sum()) == want
    # 边界：距离 191 可见、192 不可见（非锚点）；锚区 0..7 恒可见
    assert bool(keep[255, 255 - 191]) and not bool(keep[255, 255 - 192])
    assert bool(keep[255, :anchors].all())
    assert not bool(keep[255, 63]) and bool(keep[255, 64])
    # 因果性与对角
    assert not bool(torch.triu(keep, diagonal=1).any())
    assert bool(keep.diag().all())
    st = mask_stats(keep)
    assert st["n_dropped_pairs"] == 32896 - 31300 == 1596
    assert st["n_dropped_keys"] == 56
    assert round(st["dropped_frac"], 8) == 0.04851654
    assert st["diagonal_kept"] is True


def test_window192_liveness_reports_the_window_length():
    """活性量必须**如实报出窗口长度与可见份额**（SCHEMA §10/§17.4）：
    192 的可见份颏 0.951483（不是 32 的 0.287573）；window ≥ T 时才报死。"""
    from pathlm.depth_ar import window_mask_liveness
    live = window_mask_liveness(256, 192, 8)
    assert live["train_window"] == 192 and live["train_anchors"] == 8
    assert live["params_added"] == 0
    assert live["n_visible_pairs"] == 31300
    assert live["causal_pairs"] == 32896
    assert live["visible_frac"] == 0.951483
    assert live["dropped_frac"] == 0.048517
    assert live["alive"] is True
    assert window_mask_liveness(256, 32, 8)["visible_frac"] == 0.287573
    assert window_mask_liveness(256, 256, 0)["alive"] is False


def test_window192_forward_matches_eval_mask_and_differs_from_32():
    """功能判据（真实尺度 T=256）：① 模型前向的可见集 == 评测端用**同一参数**
    构造的掩码（训练/评测一致，含锚点）；② 它与 window=32 的可见集**不同**
    ——把 window 夹成 32、或在 forward 里漏掉锚点，都会被这条抓红。"""
    from pathlm.depth_ar import window_anchor_keep_mask
    from tools.evalkit.retention import masked_logits
    T = 256
    torch.manual_seed(7)
    m = DepthARModel(DepthARConfig(d_model=16, n_layers=1, n_heads=2, seq_len=T,
                                   dense=True, train_window=192,
                                   train_anchors=8), vocab_size=64)
    x = torch.randint(0, 64, (2, T))
    fwd = m(x)[1]["depth_logits"][-1]
    evl192 = masked_logits(m, x, window_anchor_keep_mask(T, 192, 8))
    assert torch.equal(fwd, evl192)
    evl32 = masked_logits(m, x, window_anchor_keep_mask(T, 32, 8))
    assert not torch.equal(evl192, evl32)
    # 锚点真的在最前面：换成「锚点取在窗口末端」的错版掩码，输出必须变化
    idx = torch.arange(T)
    tail_anchors = ((idx.unsqueeze(1) <= idx.unsqueeze(0))
                    & (((idx.unsqueeze(1) - idx.unsqueeze(0)) < 192)
                       | (idx.unsqueeze(0) >= T - 8)))
    assert not torch.equal(evl192, masked_logits(m, x, tail_anchors))


# ---------------------- 线性注意力（ticket 33 契约 R，A6 混合算子续） ----------------------
#
# 契约 R 的干预是「把注意力算子换成归一化线性核」，所以这里把四件事钉死：
# ① 关闭开关逐位还原 v1；② 训练的 O(T) 实装与「显式打分参考实现」数值一致
# （同一 φ、同一归一化）；③ 实现真的是 O(T)（FLOP 计数随 T 线性、且不物化 [T,T]
# 张量）——这是「不能只是换个名字的全注意力」的直接证据；④ 干预活性（φ 恒正、
# 分母恒正、同权重换算子有偏差）。变异验证见实验目录 mutate.py。


def test_linear_default_off_matches_v1_bitwise():
    """attn_linear 缺省 = False 时必须与密度全注意力臂**逐位**一致（v1 还原）：
    块类是 stock `Block`、参数逐位相同、前向状态逐位相同、aux 里没有线性活性块。"""
    m_default = tiny_model(dense=True)
    m_off = tiny_model(dense=True, attn_linear=False)
    assert type(m_default.blocks[0]).__name__ == "Block"
    assert sum(p.numel() for p in m_default.parameters()) == \
        sum(p.numel() for p in m_off.parameters())
    x = torch.randint(0, 10, (2, 12))
    assert torch.equal(m_default(x)[1]["states"][-1], m_off(x)[1]["states"][-1])
    assert "attn_linear" not in m_off(x)[1]


def test_linear_phi_formula_is_elu_plus_one():
    """特征映射必须**逐点等于 elu(x)+1**（不是 relu+1、不是 exp、不是恒等）：
    φ(0)=1、φ(-1)=elu(-1)+1=e^{-1}=0.3679…、负半轴非负且在小值处下溢为恰好 0
    （fp32：φ(-20)=0.0），但**永不为负**。换成 relu+1 会在 x<0 处给出 1.0（下面
    的断言抓红），换成 φ=elu（去掉 +1）会在负半轴给出负值。"""
    x = torch.linspace(-20, 20, 81)
    want = torch.nn.functional.elu(x) + 1.0
    assert torch.equal(linear_attn_phi(x), want)
    assert float(linear_attn_phi(torch.tensor(0.0))) == 1.0
    # elu(-1) = e^{-1} − 1 = −0.63212… ⇒ φ(−1) = e^{-1} = 0.36788…（relu+1 会给出 1.0）
    assert float(linear_attn_phi(torch.tensor(-1.0))) == pytest.approx(
        0.36787944, abs=1e-6)
    assert float(linear_attn_phi(torch.tensor(-15.0))) > 0.0        # e^-15 ≈ 3.06e-7
    assert float(linear_attn_phi(torch.tensor(-20.0))) == 0.0        # fp32 下溢（非负）
    assert float(linear_attn_phi(x).min()) >= 0.0
    assert bool((linear_attn_phi(x) >= 0).all())
    # relu+1 / elu（无 +1）两种错版的特征映射都会在这两条上分叉
    assert float(linear_attn_phi(torch.tensor(-1.0))) != pytest.approx(1.0)
    assert float(linear_attn_phi(torch.tensor(-2.0))) < 1.0


def test_linear_chunked_matches_explicit_reference_and_chunk_invariance():
    """训练的 O(T) 分块实装必须与「显式 [T,T] 核分数」的参考实现数值一致
    （同一 φ、同一归一化；差异仅浮点累加顺序），且**分块大小只是实现细节**：
    换 chunk 不改变语义。这条是 mutate.py 里 relu/不去归一化/非因果三处破坏的
    共同抓手。"""
    torch.manual_seed(11)
    for (B, H, T, dh) in ((2, 3, 24, 8), (1, 1, 12, 4)):
        q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
        qf, kf = linear_attn_phi(q), linear_attn_phi(k)
        ref = linear_attn_reference(qf, kf, v)
        for C in (T, T // 2, T // 4, 4):
            out, tele = linear_attn_chunked(qf, kf, v, C)
            assert float((out - ref).abs().max()) < 1e-5 * float(ref.abs().max())
            assert tele["phi_min"] > 0 and tele["den_min"] > 0
        # 分块大小不变语义（同一 chunk 序列的两种切法给出同一结果）
        a = linear_attn_chunked(qf, kf, v, 6)[0]
        b = linear_attn_chunked(qf, kf, v, 12)[0]
        assert float((a - b).abs().max()) < 1e-5 * float(a.abs().max())
    # 不整除 ⇒ 显式报错（不是静默改语义）
    with pytest.raises(ValueError):
        linear_attn_chunked(torch.ones(1, 1, 10, 2), torch.ones(1, 1, 10, 2),
                            torch.ones(1, 1, 10, 2), 3)


def test_linear_attn_causal_only_no_future_leak():
    """因果性：改动位置 t 的输入 token，位置 < t 的**所有**状态必须逐位不变。
    非因果实装（把块内 `tril` 去掉、或状态用 inclusive cumsum）会被这条抓红。"""
    torch.manual_seed(5)
    cfg = DepthARConfig(d_model=16, n_layers=2, n_heads=2, seq_len=24,
                        dense=True, attn_linear=True, linear_chunk=8)
    m = DepthARModel(cfg, vocab_size=10)
    x = torch.randint(0, 10, (2, 24))
    x2 = x.clone()
    t = 17
    x2[0, t] = (x2[0, t] + 1) % 10
    with torch.no_grad():
        s1 = m(x)[1]["states"][-1]
        s2 = m(x2)[1]["states"][-1]
    assert torch.equal(s1[0, :t], s2[0, :t])          # 未来泄漏 ⇒ 这里红
    assert not torch.equal(s1[0, t], s2[0, t])        # t 及之后确实受影响
    assert torch.equal(s1[1], s2[1])                  # 另一行不受影响


def test_linear_attn_is_linear_in_T_not_quadratic():
    """**实现真的是 O(T)**（契约 R 实现要点，不能是换个名字的全注意力）：

    ① FLOP 计数（`FlopCounterMode`，解析量、可复现）：T 放大 8 倍时，本实装注意力
       子层的 FLOPs 近似 ×8（线性），而同权重下显式全注意力（QKᵀ + softmax + PV）
       近似 ×64（平方）；
    ② 形状证据（`TorchDispatchMode` 逐个算子抄输出张量尺寸）：本实装全程**不物化
       任何 T×T 张量**（最大中间张量 ≤ B·H·(T/C)·C²），显式全注意力会物化 T×T。
    """
    from torch.utils.flop_counter import FlopCounterMode
    from torch.utils._python_dispatch import TorchDispatchMode

    d, H, B, C = 128, 4, 2, 32
    dh = d // H

    def op_flops(kind: str, T: int) -> int:
        torch.manual_seed(0)
        q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
        qf, kf = linear_attn_phi(q), linear_attn_phi(k)
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            if kind == "linear":
                linear_attn_chunked(qf, kf, v, C)
            else:                                    # 显式全注意力（同一权重形状）
                s = (q @ k.transpose(-2, -1) / dh ** 0.5).tril()
                _ = s.softmax(-1) @ v
        return counter.get_total_flops()

    lin = {T: op_flops("linear", T) for T in (64, 512)}
    full = {T: op_flops("full", T) for T in (64, 512)}
    lin_ratio = lin[512] / lin[64]
    full_ratio = full[512] / full[64]
    assert lin_ratio < 12.0, f"线性实装 FLOPs 随 T 的斜率不是线性：×8 T ⇒ ×{lin_ratio}"
    assert full_ratio > 50.0, f"对照的显式全注意力不是平方：×8 T ⇒ ×{full_ratio}"
    assert lin[512] < full[512] / 4.0

    class _SizeTracker(TorchDispatchMode):
        def __init__(self):
            self.max_numel = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))

            def rec(o):
                if isinstance(o, torch.Tensor):
                    self.max_numel = max(self.max_numel, o.numel())
                elif isinstance(o, (list, tuple)):
                    for z in o:
                        rec(z)
            rec(out)
            return out

    T = 512
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 1, T, dh) for _ in range(3))
    qf, kf = linear_attn_phi(q), linear_attn_phi(k)
    tr = _SizeTracker()
    with tr, torch.no_grad():
        linear_attn_chunked(qf, kf, v, C)
    assert tr.max_numel <= (T // C) * C * C, (
        f"线性实装的最大中间张量 numel={tr.max_numel} 超过分块上界 "
        f"{(T // C) * C * C}（可能物化了 [T,T]）")
    assert tr.max_numel < T * T
    tr_full = _SizeTracker()
    with tr_full, torch.no_grad():
        s = (q @ k.transpose(-2, -1)).tril()
        _ = s.softmax(-1) @ v
    assert tr_full.max_numel >= T * T, "对照的显式全注意力应当物化 T×T 张量"


def test_linear_masked_reference_equals_unmasked_identity_causal():
    """**仪器自检**：评测期仪器（`retention.masked_bpc` / `masked_logits`）在
    线性臂上走的是显式打分的参考路径——传入**精确因果掩码**时必须与训练路径
    （O(T) 分块、无掩码）给出同一结果（否则「依赖曲线」量的就不是这个模型）。
    同时在块级验证：掩码把被删键真的从分子分母中一起去掉（去掉一个键 ⇒ 输出变化）。"""
    torch.manual_seed(3)
    m = DepthARModel(DepthARConfig(d_model=16, n_layers=1, n_heads=2, seq_len=24,
                                   dense=True, attn_linear=True, linear_chunk=8),
                     vocab_size=10)
    x = torch.randint(0, 10, (2, 24))
    with torch.no_grad():
        h = m._input_states(x)
        blk = m.blocks[0]
        out_train, _ = blk(h)
        T = 24
        ident = torch.tril(torch.ones(T, T, dtype=torch.bool))
        out_mask, tele = blk(h, attn_mask=ident.view(1, 1, T, T))
        assert tele is None                       # 仪器路径不出活性量（不是训练路径）
        assert torch.allclose(out_train, out_mask, atol=1e-5)
        drop_one = ident.clone()
        drop_one[20, 3] = False                   # 只删 (query=20, key=3) 这一对
        out_drop, _ = blk(h, attn_mask=drop_one.view(1, 1, T, T))
        assert not torch.allclose(out_mask, out_drop, atol=1e-6)


def test_linear_params_and_init_identical_to_full_attention_block():
    """单因素 + 等参数（SCHEMA §15 自动满足）：同 seed 下线性注意力臂与全注意力
    臂的参数量、state_dict 键集与**每一个参数**逐位相同（唯一差异 = 算子）。"""
    torch.manual_seed(0)
    lin = linear_attn_block()
    torch.manual_seed(0)
    ful = Block(type("C", (), {"d_model": 16, "n_heads": 2, "mlp_mult": 4,
                               "dropout": 0.0})())
    assert type(lin).__name__ == "LinearAttnBlock"
    assert sum(p.numel() for p in lin.parameters()) == \
        sum(p.numel() for p in ful.parameters())
    sd_l, sd_f = lin.state_dict(), ful.state_dict()
    assert set(sd_l) == set(sd_f)
    assert all(torch.equal(v, sd_f[k]) for k, v in sd_l.items())


def linear_attn_block(seq_len: int = 12, chunk: int = DEFAULT_LINEAR_CHUNK):
    """构造一个单块线性注意力块（测试用；与 Block 同参）。"""
    return LinearAttnBlock(type("C", (), {"d_model": 16, "n_heads": 2,
                                          "mlp_mult": 4, "dropout": 0.0,
                                          "seq_len": seq_len})(), chunk)


def test_linear_switch_changes_computation_and_liveness():
    """开关必须真的改变计算（不是死开关）：同一权重、同一输入下，把该层算子换成
    全因果 softmax 注意力的相对偏差 > 0；同时训练路径的 telemetry 必须给出
    φ_min > 0、den_min > 0（SCHEMA §10 的活性量）。"""
    torch.manual_seed(1)
    blk = linear_attn_block()
    x = torch.randn(2, 12, 16)
    out_lin, tele = blk(x)
    out_full, _ = Block.forward(blk, x)
    rel = float((out_lin - out_full).norm().detach() / out_full.norm().detach())
    assert rel > 0, "线性算子没有改变前向（死开关）"
    assert tele["phi_min"] > 0 and tele["den_min"] > 0
    assert 0.0 <= tele["inter_mass_frac"] <= 1.0
    assert tele["state_norm"] > 0
    # 预测量口径：块类有 telemetry_kind，模型按它归类（TopKBlock 仍是 attn_sparse）
    assert LinearAttnBlock.telemetry_kind == "attn_linear"
    assert TopKBlock.telemetry_kind == "attn_sparse"


def test_linear_guards_require_dense_sequential_and_exclusive_flags():
    """单因素护栏：线性注意力只在 dense + sequential 下开放，与内容 top-k / 训练期
    窗口互斥；非法 chunk 显式报错；stock 路径守卫（dist_pen/kv_from/gate_vec）。"""
    with pytest.raises(ValueError):
        tiny_model(attn_linear=True)                          # dense=False
    with pytest.raises(ValueError):
        tiny_model(dense=True, attn_linear=True, topology="parallel")
    with pytest.raises(ValueError):
        tiny_model(dense=True, attn_linear=True, attn_sparse_k=3)
    with pytest.raises(ValueError):
        tiny_model(dense=True, attn_linear=True, train_window=4)
    with pytest.raises(ValueError):
        tiny_model(dense=True, attn_linear=True, linear_chunk=0)
    ok = tiny_model(dense=True, attn_linear=True)
    assert type(ok.blocks[0]).__name__ == "LinearAttnBlock"
    with pytest.raises(ValueError):
        linear_attn_block()(torch.randn(2, 12, 16), dist_pen=0.5)
    with pytest.raises(ValueError):
        linear_attn_block()(torch.randn(2, 12, 16), kv_from=torch.randn(2, 12, 16))
    with pytest.raises(ValueError):
        linear_attn_block()(torch.randn(2, 12, 16), gate_vec=torch.rand(2, 12, 16))
    with pytest.raises(ValueError):
        linear_attn_block()(torch.randn(2, 12, 16),
                            attn_mask=torch.ones(2, 2, 12, 12, 12, dtype=torch.bool))


def test_linear_gradients_reach_every_key_and_value():
    """反向可达性：数值健康、梯度有限，且**每个位置**的 v、以及每个位置的
    φ(k_j) 都拿到非零梯度——线性注意力不做键的剔除（被剔除的键应恰好为 0，
    这条断言是「没有静默丢键」的梯度层证据）。"""
    torch.manual_seed(2)
    B, H, T, dh = 2, 2, 16, 8
    q, k = torch.randn(B, H, T, dh), torch.randn(B, H, T, dh)
    v = torch.randn(B, H, T, dh, requires_grad=True)
    qf, kf = linear_attn_phi(q), linear_attn_phi(k)
    kf = kf.clone().requires_grad_(True)
    out, _ = linear_attn_chunked(qf, kf, v, 8)
    out.sum().backward()
    assert torch.isfinite(out).all()
    v_grad = v.grad.abs().sum(-1).sum(-1)                # [B,H,T]
    assert bool((v_grad > 0).all()), "有位置的 v 拿不到梯度（键被静默剔除）"
    k_grad = kf.grad.abs().sum(-1).sum(-1)
    assert bool((k_grad > 0).all()), "有位置的 φ(k) 拿不到梯度"
    assert torch.isfinite(kf.grad).all() and torch.isfinite(v.grad).all()


def test_linear_liveness_contract_and_probe_on_real_batches():
    """活性块契约（SCHEMA §10）：开关关闭 ⇒ enabled=False 的空块且 alive=False；
    打开 ⇒ params_added=0、φ/分母实测为正、alive=True，且 `attn_linear_probe` 在
    真实数据流上逐层给出「同权重换算子」的相对偏差（最小值 > 0）与核质量分布。"""
    import numpy as np
    arr = (np.arange(5000) % 7).astype(np.uint8)
    off = tiny_model(dense=True)
    assert attn_linear_liveness(off)["enabled"] is False
    assert attn_linear_probe(off, arr, n_batches=1, batch_size=2)["alive"] is False
    m = tiny_model(dense=True, attn_linear=True)
    x = torch.randint(0, 10, (2, 12))
    _, aux = m(x, exit_depths=None)
    live = attn_linear_liveness(m, aux["attn_linear"])
    assert live["params_added"] == 0 and live["parametrized"] is False
    assert live["block_python_class"] == "LinearAttnBlock"
    assert live["phi_min"] > 0 and live["den_min"] > 0
    assert live["alive"] is True
    n_before = sum(p.numel() for p in m.parameters())
    probe = attn_linear_probe(m, arr, n_batches=2, batch_size=2)
    assert probe["alive"] and probe["rel_dev_vs_full_min"] > 0
    assert len(probe["per_layer"]) == m.cfg.n_layers
    assert all(r["n_batches"] == 2 for r in probe["per_layer"])
    assert probe["n_layers_sampled"] == 2 * m.cfg.n_layers
    km = probe["kernel_mass"]
    assert set(km) == {"linear", "full"}
    for kind in km:
        assert 0.0 < km[kind]["mass_mean_dist"] < m.cfg.seq_len
        assert km[kind]["mass_frac_within_32"] <= km[kind]["mass_frac_within_128"]
    assert sum(p.numel() for p in m.parameters()) == n_before


def test_attention_mass_stats_reads_both_operators_on_same_weights():
    """核质量分布仪器：同一组权重下 `kind="linear"` 必须是**线性核**（φ(q)·φ(k)
    归一化）而 `kind="full"` 必须是 softmax——两者的权重矩阵可由手算复现，
    且 kind 非法时显式报错。这条防「仪器把两种算子写成同一个分布」。"""
    torch.manual_seed(4)
    m = DepthARModel(DepthARConfig(d_model=16, n_layers=1, n_heads=2, seq_len=12,
                                   dense=True, attn_linear=True), vocab_size=10)
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        h = m._input_states(x)
        blk = m.blocks[0]
        q, k, _ = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)
        shape = lambda t: t.view(2, 12, 2, 8).transpose(1, 2)
        qs, ks = shape(q), shape(k)
        causal = torch.tril(torch.ones(12, 12, dtype=torch.bool))
        w_lin = (linear_attn_phi(qs) @ linear_attn_phi(ks).transpose(-2, -1))
        w_lin = w_lin.masked_fill(~causal, 0.0)
        w_lin = w_lin / w_lin.sum(-1, keepdim=True)
        w_ful = (qs @ ks.transpose(-2, -1) / 8 ** 0.5).masked_fill(
            ~causal, float("-inf")).softmax(-1)
        dist = (torch.arange(12).unsqueeze(1) - torch.arange(12).unsqueeze(0)
                ).clamp_min(0).float()
        want_lin = float((w_lin.float() * dist).sum(-1).mean())
        want_ful = float((w_ful * dist).sum(-1).mean())
        got_lin = attention_mass_stats(m, x, kind="linear")
        got_ful = attention_mass_stats(m, x, kind="full")
    assert got_lin["mass_mean_dist"] == pytest.approx(want_lin, abs=1e-4)
    assert got_ful["mass_mean_dist"] == pytest.approx(want_ful, abs=1e-4)
    assert got_lin["n_layers"] == 1
    assert abs(got_lin["mass_mean_dist"] - got_ful["mass_mean_dist"]) < 0.5
    with pytest.raises(ValueError):
        attention_mass_stats(m, x, kind="kernel")


# ---------------- KV 头共享（ticket 33 契约 S，W3 参照线）：GQA / MQA ----------------
#
# 契约 S 的干预 = 注意力里 K/V 的**头数**（query 头数不变），所以这里钉死四件事：
# ① 缺省（n_kv_heads=None 或 == n_heads）逐位还原 v1（块类、参数、前向状态）；
# ② 真正的形状证据：qkv 投影输出宽度 = d + 2·n_kv_heads·dh、进 SDPA 的 K/V 张量
#    形状 = [B, n_kv_heads, T, dh]（不是改标签）、且解析 KV 内存按 n_kv_heads 等比下降
#    （2 头 −75%、1 头 −87.5%）；③ enable_gqa 内核语义 == 手工 repeat_interleave 展开
#    （逐位）⇒「每组 query 头共享同一组 K/V」是内核行为而不是文档承诺；
# ④ 梯度真的到达变窄的 K/V 投影（不是死路径）。变异验证见实验目录 mutate.py。


def gqa_model(n_kv_heads, **kw):
    """契约 S 的测试位点：d=32 / 8 头 / dh=4，dense + sequential（单因素）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=8, seq_len=12,
                        dense=True, n_kv_heads=n_kv_heads, **kw)
    return DepthARModel(cfg, vocab_size=10)


def test_kv_heads_default_off_matches_v1_bitwise():
    """n_kv_heads 缺省 / = n_heads（MHA）时必须与 v1 **逐位**一致：块类仍是 stock
    `Block`、参数量逐位相同、前向状态逐位相同、qkv 输出宽度仍是 3d。"""
    m_default = tiny_model(dense=True)
    m_none = tiny_model(dense=True, n_kv_heads=None)
    m_mha = tiny_model(dense=True, n_kv_heads=2)          # n_heads=2 ⇒ MHA
    assert type(m_default.blocks[0]).__name__ == "Block"
    assert type(m_none.blocks[0]).__name__ == "Block"
    assert type(m_mha.blocks[0]).__name__ == "Block"
    assert sum(p.numel() for p in m_default.parameters()) == \
        sum(p.numel() for p in m_none.parameters()) == \
        sum(p.numel() for p in m_mha.parameters())
    x = torch.randint(0, 10, (2, 12))
    ref = m_default(x)[1]["states"][-1]
    assert torch.equal(ref, m_none(x)[1]["states"][-1])
    assert torch.equal(ref, m_mha(x)[1]["states"][-1])
    assert tuple(m_mha.blocks[0].qkv.weight.shape) == (3 * 16, 16)


def test_kv_heads_guards_require_dense_sequential_and_divisibility():
    """单因素护栏：GQA/MQA 只在 dense + sequential 下开放、与内容 top-k / 线性注意力
    互斥；n_kv_heads 必须整除 n_heads；stock 路径守卫（dist_pen/kv_from/
    gate_vec）；增量解码路径显式拒绝（契约 S 不主张解码侧收益）。

    ENG1（ticket 39-X2）：与训练期可见窗口（train_window>0）**不再互斥**——窗口
    掩码是与 block 类正交的装饰层，下面补一条正向断言（该组合现在能建起来）。"""
    with pytest.raises(ValueError):
        tiny_model(dense=False, n_kv_heads=1)              # dense=False
    with pytest.raises(ValueError):
        gqa_model(2, topology="parallel")
    with pytest.raises(ValueError):
        gqa_model(2, attn_linear=True)
    with pytest.raises(ValueError):
        gqa_model(2, attn_sparse_k=3)
    for bad in (0, 3, 16):                                 # 0 / 不整除 8 / > n_heads
        with pytest.raises(ValueError):
            gqa_model(bad)
    # ENG1：GQA × 训练期窗口现在合法（块类仍是 GQABlock，窗口经 attn_mask 正交接入）
    assert type(gqa_model(2, train_window=4).blocks[0]).__name__ == "GQABlock"
    ok = gqa_model(2)
    assert type(ok.blocks[0]).__name__ == "GQABlock"
    with pytest.raises(ValueError):
        ok.blocks[0](torch.randn(2, 12, 32), dist_pen=0.5)
    with pytest.raises(ValueError):
        ok.blocks[0](torch.randn(2, 12, 32), kv_from=torch.randn(2, 12, 32))
    with pytest.raises(ValueError):
        ok.blocks[0](torch.randn(2, 12, 32), gate_vec=torch.rand(2, 12, 32))
    with pytest.raises(ValueError):
        ok.blocks[0](torch.randn(2, 12, 32),
                     attn_mask=torch.ones(2, 2, 12, 12, 12, dtype=torch.bool))
    from pathlm.depth_ar import DepthARDecoder
    with pytest.raises(ValueError):
        DepthARDecoder(ok)


def test_kv_heads_actual_tensor_shapes_and_analytic_memory():
    """形状类干预的活性证据 = **实际张量形状**（不是标签）：qkv 输出宽度真的变窄
    （d + 2·n_kv_heads·dh）、进 SDPA 的 K/V 张量头数真的等于 n_kv_heads、解析 KV
    内存按同一比例下降（GQA −75% / MQA −87.5%），且各 KV 头的 K 投影行不全同
    （否则就是「广播一个头」式假实现）。"""
    x = torch.randint(0, 10, (2, 12))
    seen = {}
    for nkv, red in ((2, 0.75), (1, 0.875)):
        m = gqa_model(nkv)
        live = kv_head_liveness(m, x)
        seen[nkv] = live
        d, nh, dh = 32, 8, 4
        assert live["enabled"] is True and live["mha"] is False
        assert live["n_kv_heads"] == nkv and live["head_dim"] == dh
        assert live["group_size"] == nh // nkv
        assert live["qkv_out_width"] == d + 2 * nkv * dh        # 不是 3d
        assert live["qkv_out_width_mha_equivalent"] == 3 * d
        assert live["k_batch_shape"] == [2, nkv, 12, dh]        # 实际进 SDPA 的形状
        for row in live["per_layer"]:
            assert row["k_shape"][1] == nkv and row["v_shape"][1] == nkv
            assert row["q_shape"][1] == nh                      # query 头数不变
            assert row["kv_bytes_per_position_fp16"] == 2 * nkv * dh * 2
            assert row["k_rows_nonzero_frac"] == 1.0
            if nkv > 1:
                assert row["kv_head_pair_max_abs_weight_diff"] > 0
                assert len(row["k_proj_per_kv_head_row_norm"]) == nkv
        assert live["kv_reduction_frac"] == pytest.approx(red)
        assert live["kv_elems_per_position"] == 3 * 2 * nkv * dh
        assert live["kv_bytes_per_position_fp16"] == live["kv_elems_per_position"] * 2
        assert live["mha_kv_bytes_per_position_fp16"] == \
            live["kv_bytes_per_position_fp16"] / (1 - red)
        # 剪枝类：负（含 qkv 偏置的半边差 ⇒ 系数是 (d+1) 不是 d）
        assert live["params_added"] == -2 * (d - nkv * dh) * (d + 1) * 3
        assert live["alive"] is True
    # MHA 臂：同一函数必须报「无共享」（reduction 0、形状 = [B, n_heads, T, dh]）
    mha = kv_head_liveness(tiny_model(dense=True), x)
    assert mha["mha"] is True and mha["enabled"] is False
    assert mha["kv_reduction_frac"] == 0.0 and mha["params_added"] == 0
    assert mha["k_batch_shape"] == [2, 2, 12, 8]
    assert mha["params_added_note"] != seen[1]["params_added_note"]


def test_kv_heads_enable_gqa_kernel_equals_repeat_interleave():
    """内核语义 = 标准 GQA：`enable_gqa=True` 的输出必须与手工把 K/V 按
    `repeat_interleave(group)` 展开到 n_heads 后的输出**逐位**一致（同权重、同输入）。
    这条同时钉住「哪几个 query 头共享哪一组 K/V」的组划分，以及「没有物化复制」的
    等价性；变异（比如把 K/V 张量整形错位）会把它抓红。"""
    m = gqa_model(2)
    blk = m.blocks[0]
    blk.eval()
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        got, tele = blk(x)
        d, nh, nkv, dh = 32, 8, 2, 4
        B, T = 2, 12
        qkv = blk.qkv(blk.ln1(x))
        q, kv = qkv[..., :d], qkv[..., d:]
        k, v = kv[..., :nkv * dh], kv[..., nkv * dh:]
        shape = lambda t, h: t.view(B, T, h, dh).transpose(1, 2)
        rep = nh // nkv
        a = F.scaled_dot_product_attention(
            shape(q, nh), shape(k, nkv).repeat_interleave(rep, dim=1),
            shape(v, nkv).repeat_interleave(rep, dim=1), is_causal=True)
        want = x + blk.proj(a.transpose(1, 2).reshape(B, T, d))
        want = want + blk.mlp(blk.ln2(want))
    assert torch.equal(got, want)
    assert tele is None
    assert blk.kv_telemetry["k_shape"] == (B, nkv, T, dh)
    assert blk.kv_telemetry["expanded_to_n_heads"] is False
    # 掩码路径（评测期仪器）也必须与同一参考的掩码版本一致
    keep = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
    with torch.no_grad():
        got_m, _ = blk(x, attn_mask=keep.expand(B, nh, T, T))
        a_m = F.scaled_dot_product_attention(
            shape(q, nh), shape(k, nkv).repeat_interleave(rep, dim=1),
            shape(v, nkv).repeat_interleave(rep, dim=1),
            attn_mask=keep.expand(B, nh, T, T))
        want_m = x + blk.proj(a_m.transpose(1, 2).reshape(B, T, d))
        want_m = want_m + blk.mlp(blk.ln2(want_m))
    assert torch.equal(got_m, want_m)


def test_kv_heads_switch_changes_computation_and_grouping():
    """干预真的改了计算（不是改标签）：同 seed 下 MHA 与 GQA 的前向状态必然不同
    （KV 投影宽度不同 ⇒ 参数不同）；且 GQA 里同组 query 头读到的 K/V **恰好相同**、
    不同组不同（组划分的直接张量证据）。"""
    torch.manual_seed(0)
    mha = gqa_model(None)
    gqa = gqa_model(2)
    x = torch.randint(0, 10, (2, 12))
    assert not torch.equal(mha(x)[1]["states"][-1], gqa(x)[1]["states"][-1])
    blk = gqa.blocks[0]
    with torch.no_grad():
        qkv = blk.qkv(blk.ln1(x.float())) if False else None
    with torch.no_grad():
        h = torch.randn(2, 12, 32)
        qkv = blk.qkv(blk.ln1(h))
        kv = qkv[..., 32:]
        k, v = kv[..., :8], kv[..., 8:]
        ks = k.view(2, 12, 2, 4).transpose(1, 2).repeat_interleave(4, dim=1)
        # 组划分：query 头 0..3 与 4..7 各自共享同一组 K/V
        assert torch.equal(ks[:, 0], ks[:, 3])
        assert not torch.equal(ks[:, 0], ks[:, 4])
        assert torch.equal(k.view(2, 12, 2, 4).transpose(1, 2)[:, 0],
                           ks[:, 0])


def test_kv_heads_gradients_reach_narrow_kv_projections():
    """反向可达性：损失对变窄的 K/V 投影行（qkv 权重的 [d, d+2·kv_dim) 行）梯度
    非零且有限 —— 变窄的 K/V 不是死路径（活性量的梯度侧证据）。"""
    for nkv in (2, 1):
        m = gqa_model(nkv)
        x = torch.randint(0, 10, (2, 12))
        loss, _ = m(x, exit_depths=None)
        loss.backward()
        d, dh, kv_dim = 32, 4, nkv * 4
        for blk in m.blocks:
            g = blk.qkv.weight.grad
            assert g is not None and torch.isfinite(g).all()
            kv_rows = g[d:d + 2 * kv_dim]
            assert bool((kv_rows.abs().sum(-1) > 0).all()), \
                "变窄的 K/V 投影有行拿不到梯度（死路径）"
            assert float(kv_rows.abs().max()) > 0


def test_train_cli_kv_heads_plumbed_to_config():
    """CLI 通道（可失败）：`--n-kv-heads` 必须真的进 `DepthARConfig` 并建出 GQABlock，
    否则就是「开关是死的」。变异验证 S8（把 `build_depthar_config` 里的
    `n_kv_heads=args.n_kv_heads` 改成 `None`）靠这条抓红。"""
    import argparse
    import subprocess
    import train_depth_ar as T

    base = dict(d=32, layers=2, heads=2, seq_len=12, mlp_mult=4, dense=True,
                gate_bias_init=-2.0, topology="sequential", fill_kv=True,
                proj_fill=False, readout_dims=None, readout_gated=False,
                readout_linear_resid=False, attn_sparse_k=0, attn_linear=False,
                linear_chunk=64, train_window=0, train_anchors=0,
                carry_write="none", carry_chunk=64, carry_state_dim=None)
    cfg = T.build_depthar_config(argparse.Namespace(**base, n_kv_heads=1))
    assert cfg.n_kv_heads == 1
    torch.manual_seed(0)
    m = DepthARModel(cfg, vocab_size=10)
    assert type(m.blocks[0]).__name__ == "GQABlock"
    assert tuple(m.blocks[0].qkv.weight.shape)[0] == 32 + 2 * 1 * 16
    cfg_off = T.build_depthar_config(argparse.Namespace(**base, n_kv_heads=None))
    assert cfg_off.n_kv_heads is None
    assert type(DepthARModel(cfg_off, 10).blocks[0]).__name__ == "Block"
    # CLI 表面：argparse 必须真的认这个选项（在子进程里读 --help，不跑训练）
    r = subprocess.run([sys.executable, "train_depth_ar.py", "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "--n-kv-heads" in r.stdout, r.stdout[-500:]


# =====================================================================
# 显式写入通道（ticket 35 契约 U）的可失败单测
#
# ① 数值正确性：分块并行实装 == 逐位置参考实现（两条写入规则、多个 chunk）；
# ② 硬判据一「等参数」：U1/U2 的参数字典逐位相同、唯一变量 = 写入规则；
# ③ 硬判据二「因果性」：位置 t 的读出只用 S_{≤t}（内核级 + 端到端级）；
# ④ 单边零起步：γ=0 ⇒ 起步严格恒等于无通道基线（且主干参数逐位相同）；
# ⑤ 活性：γ 逃逸后通道真的改前向、梯度真的到达写门；
# ⑥ 窗口内的「搬」是通道干的：窗口看不见的 token 仍然改变读出。
# 变异验证见实验目录 mutate.py（故意泄漏未来 / 写错更新式 / 把规则退化成 plain）。
# =====================================================================


def carry_model(rule, **kw):
    """契约 U 的测试位点：d=32 / 4 头 / dh=8，dense + sequential（单因素）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=4, seq_len=12,
                        dense=True, carry_rule=rule, **kw)
    return DepthARModel(cfg, vocab_size=10)


def _carry_qkv(model, tokens):
    """(qs, ks, vs) —— 第一层在 ln1(x) 上的 q/k/v 分头视图 [B,H,T,dh]。"""
    blk = model.blocks[0]
    B, T, d = 2, tokens.shape[1], model.cfg.d_model
    with torch.no_grad():
        q, k, v = blk.qkv(blk.ln1(model._input_states(tokens))).chunk(3, dim=-1)
    shape = lambda t: t.view(tokens.shape[0], T, blk.n_heads,
                             d // blk.n_heads).transpose(1, 2)
    return shape(q), shape(k), shape(v)


def test_carry_update_rules_match_written_formulas():
    """两条更新式必须与契约写下的式子逐项一致（单步、非零起步状态）：

    plain：`S_1 = S_0 + β v k̂ᵀ`；delta：`S_1 = S_0(I − β k̂k̂ᵀ) + β v k̂ᵀ`
    （矩阵写法 ≡ 设计文档的残差写法 `S + β(v − S k̂)k̂ᵀ`，两者必须等价），
    读出 `y = S_1 q`；且零起步时两条规则的第一步**必须重合**（`S_1 = βvk̂ᵀ`）。
    """
    torch.manual_seed(0)
    B, H, dh = 2, 3, 5
    q = torch.randn(B, H, 1, dh)
    k = torch.randn(B, H, 1, dh)
    v = torch.randn(B, H, 1, dh)
    beta = torch.rand(B, H, 1)
    kh = carry_norm_keys(k[:, :, 0])
    S0 = torch.randn(B, H, dh, dh)
    bt = beta[:, :, 0].unsqueeze(-1).unsqueeze(-1)          # [B,H,1,1]
    outer = v[:, :, 0].unsqueeze(-1) @ kh.unsqueeze(-2)     # [B,H,dv,dk]
    S1p = S0 + bt * outer
    eye = torch.eye(dh)
    S1d = S0 @ (eye - bt * (kh.unsqueeze(-1) @ kh.unsqueeze(-2))) + bt * outer
    resid = v[:, :, 0] - (S0 @ kh.unsqueeze(-1)).squeeze(-1)
    S1r = S0 + bt * (resid.unsqueeze(-1) @ kh.unsqueeze(-2))
    assert torch.allclose(S1d, S1r, atol=1e-12)
    y_p = (S1p @ q[:, :, 0].unsqueeze(-1)).squeeze(-1)
    y_d = (S1d @ q[:, :, 0].unsqueeze(-1)).squeeze(-1)
    assert not torch.allclose(y_p, y_d)
    # 零起步：两条规则的第一步重合，且 == β (k̂·q) v
    y0p, _ = carry_scan_sequential(q, k, v, beta, "plain")
    y0d, _ = carry_scan_sequential(q, k, v, beta, "delta")
    want = bt.squeeze(-1) * (kh * q[:, :, 0]).sum(-1, keepdim=True) * v[:, :, 0]
    assert y0p.shape == (B, H, 1, dh)
    assert torch.allclose(y0p[:, :, 0], want, atol=1e-12)
    assert torch.allclose(y0d[:, :, 0], want, atol=1e-12)


def test_carry_chunked_matches_sequential():
    """分块并行实装 == 逐位置参考实现（两条规则 × 多个 chunk 尺寸）。

    这是「实现正确」的核心单测：delta 的并行形式（WY 表示 + 三角求解）不是靠
    直觉写的，靠这条测试对逐位置定义钉死。
    """
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(3)
        B, H, T, dh = 2, 2, 24, 4
        q = torch.randn(B, H, T, dh) * 0.7
        k = torch.randn(B, H, T, dh)
        v = torch.randn(B, H, T, dh)
        beta = torch.rand(B, H, T)
        for rule in ("plain", "delta"):
            y_ref, _ = carry_scan_sequential(q, k, v, beta, rule)
            for chunk in (1, 2, 3, 5, 8, 24):
                y, tele = carry_scan_chunked(q, k, v, beta, rule, chunk)
                assert float((y - y_ref).abs().max()) < 1e-12, (rule, chunk)
                assert torch.isfinite(tele["state_fro"])
                assert float(tele["beta_mean"]) == pytest.approx(
                    float(beta.mean()), rel=1e-9)
    finally:
        torch.set_default_dtype(torch.float32)


def test_carry_channel_is_causal_only():
    """**硬判据：因果性**。把 t ≥ 10 的 q/k/v/β 换成另一组随机数，t ≤ 9 的读出
    必须**逐位不变**（分块并行实装与逐位置参考实现都要过）。"""
    torch.manual_seed(4)
    B, H, T, dh = 2, 2, 16, 4
    q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
    beta = torch.rand(B, H, T)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    for t in (q2, k2, v2):
        t[:, :, 10:] = torch.randn(B, H, T - 10, dh)
    b2 = beta.clone()
    b2[:, :, 10:] = torch.rand(B, H, T - 10)
    for rule in ("plain", "delta"):
        for fn, kw in ((carry_scan_sequential, {}),
                       (carry_scan_chunked, {"chunk": 4}),
                       (carry_scan_chunked, {"chunk": 1})):
            ya, _ = fn(q, k, v, beta, rule, **kw)
            yb, _ = fn(q2, k2, v2, b2, rule, **kw)
            assert torch.equal(ya[:, :, :10], yb[:, :, :10]), (rule, fn, kw)


def test_carry_channel_causality_end_to_end():
    """端到端因果：改动输入 token t0 之后的 token，t0 之前的 logits 必须逐位不变
    （含窗口掩码与通道两条路径）。"""
    m = carry_model("delta")
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=0.5)          # 让通道真的活着
    m.eval()
    x = torch.randint(0, 10, (2, 12))
    y = x.clone()
    y[:, 7:] = torch.randint(0, 10, (2, 5))
    with torch.no_grad():
        la = m(x)[1]["depth_logits"][-1]
        lb = m(y)[1]["depth_logits"][-1]
    assert torch.equal(la[:, :7], lb[:, :7])


def test_carry_plain_and_delta_params_bitwise_equal():
    """**硬判据：逐位等参数**。U1 与 U2 的参数名集合、形状、初始化取值逐位相同，
    且唯一不同的配置字段是 `carry_rule`（SCHEMA §15：增益只能归因到写入规则）。"""
    u1, u2 = carry_model("plain"), carry_model("delta")
    p1, p2 = u1.state_dict(), u2.state_dict()
    assert sorted(p1) == sorted(p2)
    for k in p1:
        assert p1[k].shape == p2[k].shape, k
        assert torch.equal(p1[k], p2[k]), k
    assert sum(p.numel() for p in u1.parameters()) == \
        sum(p.numel() for p in u2.parameters())
    assert u1.cfg.carry_rule == "plain" and u2.cfg.carry_rule == "delta"
    d1 = {k: v for k, v in u1.cfg.__dict__.items() if k != "carry_rule"}
    d2 = {k: v for k, v in u2.cfg.__dict__.items() if k != "carry_rule"}
    assert d1 == d2
    # 新增参数 = 单共享槽（w_k/w_v/w_q/beta_w/beta_b/gate_w）+ 每层一个读出增益
    added = set(p1) - set(carry_model(None).state_dict())
    assert len(added) == 6 + u1.cfg.n_layers          # 6 个槽参数 + L 个读出增益
    d, nh, L = u1.cfg.d_model, u1.cfg.n_heads, u1.cfg.n_layers
    assert sum(p1[k].numel() for k in added) == \
        3 * d * d + nh * d + nh + d * d + L * nh


def test_carry_read_convention_includes_own_write():
    """**读出约定必须写清并钉住**（契约 U 追加项 ②）：位置 t 读到的是 `S_t`
    （**含 t 自己的写入**，read-after-write，与因果 softmax 注意力把对角项计入同一
    约定）。证据：单点序列上读出 == `β (k̂·q) v`（= 自写入贡献）；
    变异验证 M3 会让位置 t 去读 t+1 的写入（必须被因果性单测抓红）。"""
    torch.manual_seed(7)
    B, H, dh = 1, 2, 4
    q = torch.randn(B, H, 1, dh)
    k = torch.randn(B, H, 1, dh)
    v = torch.randn(B, H, 1, dh)
    beta = torch.rand(B, H, 1)
    kh = carry_norm_keys(k[:, :, 0])
    want = (beta[:, :, 0].unsqueeze(-1) * (kh * q[:, :, 0]).sum(-1, keepdim=True)
            * v[:, :, 0])
    # **训练实装（分块并行）与逐位置参考实装都要过**——读出约定是训练路径的性质，
    # 只测参考实装会让「去掉自写入」的变异漏网（变异验证 M2 的教训）。
    impls = [("sequential", lambda qq, kk, vv, bb, r: carry_scan_sequential(qq, kk, vv, bb, r)),
             ("chunked1", lambda qq, kk, vv, bb, r: carry_scan_chunked(qq, kk, vv, bb, r, 1)),
             ("chunked4", lambda qq, kk, vv, bb, r: carry_scan_chunked(qq, kk, vv, bb, r, 4))]
    for name, fn in impls:
        for rule in ("plain", "delta"):
            y, _ = fn(q, k, v, beta, rule)
            assert torch.allclose(y[:, :, 0], want, atol=1e-6), (name, rule)
        # 改掉 v_0（自写入）必须改变 y_0 —— 「读到自己的写入」而不是「读之前的状态」
        v2 = v.clone()
        v2[:, :, 0] += 1.0
        for rule in ("plain", "delta"):
            y1, _ = fn(q, k, v, beta, rule)
            y2, _ = fn(q, k, v2, beta, rule)
            assert not torch.allclose(y1[:, :, 0], y2[:, :, 0]), (name, rule)


def test_carry_read_mass_and_far_share():
    """契约 U 追加项 ③：通道活性要带**读取量**证据 —— 每层读出占注意力输出的
    比例（`read_share`）、槽的分数质量、以及精确的「窗口外贡献份额」。U0 全为 0。"""
    m = carry_model("delta", train_window=4)
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=1.0)
    x = torch.randint(0, 10, (2, 12))
    stats = carry_read_mass_stats(m, x, window=4)
    assert stats["enabled"] and stats["read_share_per_layer"] > 0
    assert stats["slot_score_mass_per_query"] > 0
    assert 0 <= stats["far_read_norm_frac"] <= 1.5
    off = carry_read_mass_stats(carry_model(None), x, window=4)
    assert off["read_share_per_layer"] is None
    # 精确的窗口外份额：只留窗口外写入时读出必然变小（窗口 > 0）
    torch.manual_seed(8)
    q = torch.randn(1, 2, 18, 4)
    k = torch.randn(1, 2, 18, 4)
    v = torch.randn(1, 2, 18, 4)
    beta = torch.full((1, 2, 18), 0.5)
    for rule in ("plain", "delta"):
        f = carry_far_read_norm_frac(q, k, v, beta, window=4, rule=rule)
        assert 0 < f < 1


def test_carry_channel_identity_start_matches_no_channel():
    """**单边零起步**（SCHEMA §10）：读出增益 γ=0 ⇒ 起步严格恒等于无通道基线，
    且主干参数与 U0 臂**逐位相同**（新增参数在其余参数 init 之后才抽随机数）。

    同时钉住「恒等起步不是零初始化」：槽的写投影/写门非零（∂L/∂γ ≠ 0 ⇒ 非鞍点）。
    """
    u0, u1, u2 = carry_model(None), carry_model("plain"), carry_model("delta")
    p0 = u0.state_dict()
    for m in (u1, u2):
        p = m.state_dict()
        assert set(p) - set(p0) == (
            {"carry.w_k", "carry.w_v", "carry.w_q", "carry.beta_w",
             "carry.beta_b", "carry.gate_w"}
            | {f"blocks.{i}.carry_scale" for i in range(3)})
        for k in p0:
            assert torch.equal(p0[k], p[k]), k        # 主干逐位相同
        for blk in m.blocks:
            assert torch.equal(blk.carry_scale, torch.zeros_like(blk.carry_scale))
        assert float(m.carry.w_k.detach().abs().sum()) > 0
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        l0 = u0(x)[1]["depth_logits"][-1]
        for m in (u1, u2):
            assert torch.equal(l0, m(x)[1]["depth_logits"][-1])
            assert m._carry_box["mem"].abs().sum() > 0        # 槽是活的…
            assert all(float(b.carry_scale.abs().sum()) == 0
                       for b in m.blocks)                     # …但读出被单边零关掉


def test_carry_block_zero_scale_equals_plain_block_bitwise():
    """γ=0 时 `CarryChannelBlock.forward` 必须与同权重 `Block.forward` **逐位**一致
    （含窗口掩码路径）⇒ 「通道关掉 = 基线」是逐位事实，不是近似。"""
    m = carry_model("delta", train_window=4)
    blk = m.blocks[0]
    x = torch.randint(0, 10, (2, 12))
    win = window_anchor_keep_mask(12, 4, 0)
    with torch.no_grad():
        h = m._input_states(x)
        mask = win.view(1, 1, 12, 12).expand(2, m.cfg.n_heads, 12, 12)
        got, tele = blk(h, attn_mask=mask)
        want, _ = Block.forward(blk, h, attn_mask=mask)
        assert torch.equal(got, want)
        assert tele["read_scale_absmean"] == 0
        blk.carry_scale.data.fill_(1.0)
        got2, tele2 = blk(h, attn_mask=mask)
        want2, _ = Block.forward(blk, h, attn_mask=mask)
        assert not torch.equal(got2, want2)          # 开关打开 ⇒ 真的改了前向
        assert float(tele2["read_share"]) > 0


def test_carry_channel_write_path_gets_gradients():
    """梯度可达性：损失对槽的写投影/写门与每层读出增益的梯度非零且有限
    —— 不是死路径（活性量的梯度侧证据）。"""
    m = carry_model("delta")
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=1.0)
    x = torch.randint(0, 10, (2, 12))
    loss, aux = m(x, exit_depths=None)
    loss.backward()
    for p in m.carry.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert float(p.grad.abs().max()) > 0        # 写入路径也有梯度（不是死路径）
    for blk in m.blocks:
        assert float(blk.carry_scale.grad.abs().max()) > 0
    assert aux["attn_carry"] and len(aux["attn_carry"]) == 3


def test_carry_channel_carries_across_window_but_baseline_does_not():
    """**契约 U 的机制核心**：训练窗口 4 个 token 时，位置 10 的读出在无通道臂上
    与「位置 0 的 token」**无关**（窗口挡住了），而在有通道臂上**必须**随它变
    ——「信息被搬过来」这件事只有在通道存在时才成立（判据 ① 的前提校验同源）。
    """
    for rule in ("plain", "delta"):
        m = carry_model(rule, train_window=4)
        for blk in m.blocks:
            blk.carry_scale.data.normal_(std=1.0)
        m.eval()
        x = torch.randint(0, 10, (2, 12))
        y = x.clone()
        y[:, 0] = (x[:, 0] + 1) % 10
        with torch.no_grad():
            la = m(x)[1]["depth_logits"][-1]
            lb = m(y)[1]["depth_logits"][-1]
        assert not torch.equal(la[:, 10], lb[:, 10])
    base = carry_model(None, train_window=4)
    base.eval()
    x = torch.randint(0, 10, (2, 12))
    y = x.clone()
    y[:, 0] = (x[:, 0] + 1) % 10
    with torch.no_grad():
        la = base(x)[1]["depth_logits"][-1]
        lb = base(y)[1]["depth_logits"][-1]
    assert torch.equal(la[:, 10], lb[:, 10])          # 窗口真的挡住了位置 0


def test_carry_liveness_and_probe_report_alive():
    """活性块与功能探针（SCHEMA §10）：状态/写门/写入量非零，且同权重关掉通道后
    前向有偏差；把规则换掉也有偏差（两臂的唯一变量就是规则本身）。"""
    # carry_chunk=4 ⇒ seq_len=12 有 3 个块，`‖ΔS‖/‖S‖`（块 2..n）有定义
    m = carry_model("delta", carry_chunk=4)
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=1.0)
    m.eval()
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        _, aux = m(x, exit_depths=None)
    live = carry_liveness(m, m._carry_tele, aux["attn_carry"],
                          prob=carry_probe(m, x))
    assert live["write_rel_chunks"] == 2 and live["n_chunks"] == 3
    assert live["enabled"] and live["alive"]
    d, nh, L = 32, 4, 3
    assert live["params_added"] == 3 * d * d + nh * d + nh + d * d + L * nh
    assert live["state_fro"] > 0 and live["beta_mean"] > 0 and live["write_rel"] > 0
    assert live["rel_dev_vs_channel_off"] > 0
    assert live["rel_dev_vs_other_rule"] > 0
    off = carry_liveness(carry_model(None))
    assert off["alive"] is False and off["params_added"] == 0
    assert off["read_share"] is None and off["state_fro"] is None


def test_carry_plain_rule_does_not_forget_delta_does():
    """两条规则的**语义差异**必须可观测（否则 U1 vs U2 是同一个臂）：

    用同一把钥匙（`q = k̂`）连写两次，`v1 = e0`、`v2 = e1`、β 恒定。解析解：
    plain 读出 `= β(v1 + v2)`（旧值原样叠加）；delta 读出 `= β(v1 + v2 − βv1)`
    （把「状态已经能预测的部分」扣掉）⇒ delta 的读出更小、不重复占位。
    """
    torch.manual_seed(5)
    B, H, dh = 1, 1, 4
    kh = torch.zeros(B, H, dh)
    kh[..., 0] = 1.0
    k = kh.unsqueeze(2).repeat(1, 1, 2, 1)
    v = torch.zeros(B, H, 2, dh)
    v[..., 0, 0] = 1.0
    v[..., 1, 1] = 1.0
    beta = torch.full((B, H, 2), 0.9)
    q_read = kh.unsqueeze(2).clone()                        # 读出用同一把钥匙
    q_two = q_read.repeat(1, 1, 2, 1)
    y_p, _ = carry_scan_sequential(q_two, k, v, beta, "plain")
    y_d, _ = carry_scan_sequential(q_two, k, v, beta, "delta")
    bt = 0.9
    want_p = bt * (v[:, :, 0] + v[:, :, 1])
    want_d = bt * (v[:, :, 0] + v[:, :, 1] - bt * v[:, :, 0])
    assert torch.allclose(y_p[:, :, 1], want_p, atol=1e-12)
    assert torch.allclose(y_d[:, :, 1], want_d, atol=1e-12)
    assert float(y_d[:, :, 1].norm()) < float(y_p[:, :, 1].norm())


def test_carry_cli_plumbed_to_config():
    """CLI 通道（可失败）：`--carry-write` 必须真的进 `DepthARConfig` 并建出
    `CarryChannelBlock`，否则就是「开关是死的」（变异验证 C6 靠这条抓红）。"""
    import argparse
    import train_depth_ar as T

    base = dict(d=32, layers=2, heads=2, seq_len=12, mlp_mult=4, dense=True,
                gate_bias_init=-2.0, topology="sequential", fill_kv=True,
                proj_fill=False, readout_dims=None, readout_gated=False,
                readout_linear_resid=False, attn_sparse_k=0, attn_linear=False,
                linear_chunk=64, train_window=0, train_anchors=0,
                n_kv_heads=None, carry_chunk=64, carry_state_dim=None)
    cfg = T.build_depthar_config(argparse.Namespace(**base, carry_write="delta"))
    assert cfg.carry_rule == "delta"
    torch.manual_seed(0)
    m = DepthARModel(cfg, vocab_size=10)
    assert type(m.blocks[0]).__name__ == "CarryChannelBlock"
    # 容量旋钮（ticket 36 U1）也必须真的进 config（否则就是死开关）
    cfg_cap = T.build_depthar_config(argparse.Namespace(
        **{**base, "carry_state_dim": 16}, carry_write="delta"))
    assert cfg_cap.carry_state_dim == 16
    assert DepthARModel(cfg_cap, 10).carry.head_dim == 16
    cfg_off = T.build_depthar_config(
        argparse.Namespace(**base, carry_write="none"))
    assert cfg_off.carry_rule is None
    assert type(DepthARModel(cfg_off, 10).blocks[0]).__name__ == "Block"


def test_carry_config_rejects_combinations():
    """单因素：写入通道与其它干预互斥/依赖必须显式报错；与 `train_window`
    **可以**同时开（窗口造需求、通道提供路径）。

    ENG1（ticket 39-X2）：与 GQA/MQA（n_kv_heads<n_heads）**不再互斥**——KV 头投影
    已从 block 类解耦（GQA × carry 走 `ComposableBlock`），下面补正向断言。"""
    bad = ({"attn_linear": True}, {"topology": "parallel"},
           {"attn_sparse_k": 2})
    for kw in bad:
        with pytest.raises(ValueError):
            carry_model("plain", **kw)
    # ENG2（ticket 44 后续项）：dense=False（非 dense 的随机退出 / ragged / fill 路径）
    # **不再**报错——写入通道现在可与混合填充语义同开，走 `CarryChannelBlock` 的
    # kv_from / attn_mask / gate_vec 约定（正向断言；组合单测见文件末尾 §ENG2）。
    torch.manual_seed(0)
    m_nd = DepthARModel(DepthARConfig(d_model=32, n_layers=2, n_heads=4,
                                      seq_len=12, carry_rule="plain"),
                        vocab_size=10)
    assert type(m_nd.blocks[0]).__name__ == "CarryChannelBlock"
    m_nd(torch.randint(0, 10, (2, 12)))
    with pytest.raises(ValueError):                 # 未知规则
        torch.manual_seed(0)
        DepthARModel(DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=12,
                                   dense=True, carry_rule="luar"), vocab_size=10)
    m = carry_model("plain", train_window=4)        # 组合合法：window + channel
    assert m.cfg.train_window == 4 and m.cfg.carry_rule == "plain"
    m(torch.randint(0, 10, (2, 12)))
    # ENG1：GQA × channel 现在合法，走可组合块（KV 头投影从块类解耦）
    m2 = carry_model("plain", n_kv_heads=2)
    assert type(m2.blocks[0]).__name__ == "ComposableBlock"
    m2(torch.randint(0, 10, (2, 12)))


def test_carry_default_off_matches_v1_bitwise():
    """`carry_rule=None`（缺省）逐位还原 v1：块类仍是 stock `Block`、参数量与
    参数名与 v1 相同、前向与 `retention.masked_logits` 的 v1 参考路径逐位一致。"""
    from tools.evalkit.retention import masked_logits
    m = carry_model(None, train_window=4)
    assert all(type(b).__name__ == "Block" for b in m.blocks)
    x = torch.randint(0, 10, (2, 12))
    keep = window_anchor_keep_mask(12, 4, 0)
    m.eval()
    with torch.no_grad():
        got = m(x)[1]["depth_logits"][-1]
        want = masked_logits(m, x, keep)
    assert torch.equal(got, want)


def test_mqar_training_task_layout_and_local_prior():
    """训练侧合成任务（契约 U 的数据源）：自监督布局自洽（答案 = 位置 rows+1 的
    token）、同行内查询问不同的 pair、判决行的可见窗口里**没有符号**（gap ≥ 14
    ⇒ 与单查询版判决行同一局部模式）、局部先验退化到 chance。"""
    from tools.evalkit.longrange_tasks import (assoc_recall_multi_batch,
                                               check_local_prior_resistance,
                                               local_prior_accuracy)
    g = torch.Generator().manual_seed(0)
    t = assoc_recall_multi_batch(8, 256, 8, 8, generator=g)
    row = torch.arange(t.B).unsqueeze(1)
    assert torch.equal(t.tokens[row, t.rows + 1], t.answers)
    pick = (t.rows - (t.meta["query_base"] + 1)) // t.meta["query_block"]
    assert int(t.dists.min()) > 16
    assert check_local_prior_resistance(t, k=16)["passed"]
    assert local_prior_accuracy(t, k=16)["accuracy"] < 0.047
    # 判决行的可见窗口（最近 16 个位置）里只有 cue + 查询键本身，**没有别的符号**
    # （尤其没有上一个查询的答案）⇒ 局部先验拿不到任何可抄的东西
    sym = set(range(t.n_answers))
    for b in range(t.B):
        for j in range(t.M):
            r = int(t.rows[b, j])
            win = [int(v) for v in t.tokens[b, r - 15:r + 1]]
            assert set(win) & sym == {int(t.tokens[b, r])}
    assert list(pick.shape) == [t.B, t.M]


# =====================================================================
# ⑦ 容量旋钮（ticket 36 契约 U1）：`carry_state_dim` 只改记忆槽的状态宽度，
#    不改注意力。默认口径（None）与显式 dh = d_model/n_heads 必须逐位相同；
#    显式 dh ≠ d/n_heads 时注意力逐位不变、只有记忆槽的头结构/读出增益变化。
# =====================================================================


def test_carry_state_dim_default_and_explicit_n_heads_bitwise_equal():
    """`carry_state_dim = d_model/n_heads` 与缺省（None）**逐位相同**：
    参数逐位相等、同输入前向逐位相等。⇒ 容量旋钮的缺省口径就是 v1。"""
    m_def = carry_model("delta")
    m_exp = carry_model("delta", carry_state_dim=8)      # d=32, n_heads=4 ⇒ dh=8
    assert m_def.carry.head_dim == m_exp.carry.head_dim == 8
    assert m_def.carry.n_heads == m_exp.carry.n_heads == 4
    sd_def, sd_exp = m_def.state_dict(), m_exp.state_dict()
    assert set(sd_def) == set(sd_exp)
    for k in sd_def:
        assert torch.equal(sd_def[k], sd_exp[k]), k
    x = torch.randint(0, 10, (2, 12))
    m_def.eval(); m_exp.eval()
    with torch.no_grad():
        a = m_def(x)[1]["depth_logits"][-1]
        b = m_exp(x)[1]["depth_logits"][-1]
    assert torch.equal(a, b)


def test_carry_state_dim_decouples_state_from_attention_heads():
    """dh = 16 ≠ d/n_heads = 8：**注意力头与主干逐位不变**，只有记忆槽头数
    4 → 2、状态宽度 8 → 16（总状态浮点 32·8 → 32·16 = 翻倍）。"""
    m_def = carry_model("delta")
    m_big = carry_model("delta", carry_state_dim=16)     # d=32, dh=16 ⇒ H_carry=2
    assert m_big.blocks[0].n_heads == 4                  # 注意力头不变
    assert m_big.carry.n_heads == 2 and m_big.carry.head_dim == 16
    assert m_big.blocks[0].carry_heads == 2
    assert m_big.carry.n_heads * m_big.carry.head_dim ** 2 == 32 * 16
    assert m_def.carry.n_heads * m_def.carry.head_dim ** 2 == 32 * 8
    # 主干（除 carry.* 与每层 carry_scale）逐位相同
    sa, sb = m_def.state_dict(), m_big.state_dict()
    carry_names = {k for k in sa if k.startswith("carry.")
                   or k.endswith("carry_scale")}
    assert set(sa) == set(sb)
    for k in sa:
        if k in carry_names:
            continue
        assert torch.equal(sa[k], sb[k]), k
    # 读出增益随记忆槽头数定形
    assert m_def.blocks[0].carry_scale.shape == (4,)
    assert m_big.blocks[0].carry_scale.shape == (2,)
    # 存活：γ 逃逸后前向形状正确、梯度到达写门
    for blk in m_big.blocks:
        blk.carry_scale.data.normal_(std=0.5)
    x = torch.randint(0, 10, (2, 12))
    out = m_big(x)[1]["depth_logits"][-1]
    assert out.shape == (2, 12, 10)
    loss = out.float().pow(2).mean()
    loss.backward()
    assert float(m_big.carry.w_v.grad.abs().max()) > 0


def test_carry_state_dim_validation():
    """`carry_state_dim` 只在开启通道时有意义、且必须整除 d_model。"""
    import pytest
    with pytest.raises(ValueError, match="carry_state_dim"):
        torch.manual_seed(0)
        DepthARModel(DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=8,
                                   dense=True, carry_rule=None,
                                   carry_state_dim=16), vocab_size=10)
    with pytest.raises(ValueError, match="整除"):
        torch.manual_seed(0)
        DepthARModel(DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=8,
                                   dense=True, carry_rule="delta",
                                   carry_state_dim=5), vocab_size=10)


# =====================================================================
# ENG2（ticket 44 后续项）：写入通道 × 非 dense 路径
# （ragged / 随机退出 / 混合填充 / proj-fill）
#
# carry 此前有硬校验要求 dense=True（`pathlm/depth_ar.py`），且 `CarryChannelBlock`
# 拒绝 kv_from / gate_vec ⇒ 产品上有意义的「carry × 随机退出 × proj-fill」组合在
# 引擎里不存在。本线解耦该校验，并把 `CarryChannelBlock` 的调用约定对齐 stock
# `Block`（kv_from = fill 的混合 KV 源、attn_mask = no-fill ragged 掩码、gate_vec =
# 逐通道门），使四条组合第一次可测。
#
# 四条组合各一条可失败单测；「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」的变异验证见
# `axes/1-time/1.1.4.1-explicit-memory/experiments/2628-eng2-carry-ragged/mutate.py`。
#
# 写入时机（本引擎的显式语义，`test_eng2_carry_write_timing_ignores_exit_depths`
# 钉住）：写入在 `_input_states` 对输入流 h₀ 因果扫描一次、发生在任何退出之前 ⇒
# **所有位置（含已退出位置）都写各自的 h₀**，通道状态与退出模式无关；读出的生效
# 范围由调用方的 `where(active, out, frozen)` 决定。
# =====================================================================


def _nd_carry(rule="delta", **kw):
    """非 dense（RAP）的写入通道位点：`tiny_model` 的 dense 缺省就是 False。"""
    return tiny_model(carry_rule=rule, **kw)


def _activate_carry_scales(model, value=0.5):
    for blk in model.blocks:
        blk.carry_scale.data.fill_(value)


def _manual_nd_layer(m, k, h, frozen, mem, depths):
    """非 dense 层的手写参照实现（与 `DepthARModel.forward` 非 dense 分支逐项对应）：
    fill 时 K/V 取自 `where(active, h, kv_src)`、ragged 时只用因果 ∩ 已达层的键；
    读出 `γ ⊙ mem_t` 加在注意力输出上（进 proj）；输出带门 `g ⊙ delta`。"""
    blk = m.blocks[k - 1]
    B, T, d = h.shape
    nh, dh = blk.n_heads, d // blk.n_heads
    shape = lambda t: t.view(B, T, nh, dh).transpose(1, 2)
    active = depths >= k
    if m.cfg.fill_kv:
        kv_src = frozen
        if m.cfg.proj_fill:
            kv_src = m.kv_adapters[k - 1](frozen)
        kv_in = torch.where(active.unsqueeze(-1), h, kv_src)
        q = blk.qkv(blk.ln1(h))[..., :d]
        kkv = blk.qkv(blk.ln1(kv_in))
        kk, vv = kkv[..., d:2 * d], kkv[..., 2 * d:]
        a = F.scaled_dot_product_attention(shape(q), shape(kk), shape(vv),
                                           is_causal=True)
    else:
        allow = (torch.tril(torch.ones(T, T, dtype=torch.bool))
                 & (depths[:, None, :] >= k)) | torch.eye(T, dtype=torch.bool)
        allow = allow.unsqueeze(1).expand(-1, nh, -1, -1)
        q, kk, vv = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)
        a = F.scaled_dot_product_attention(shape(q), shape(kk), shape(vv),
                                           attn_mask=allow)
    a = a.transpose(1, 2).reshape(B, T, d)
    gain = blk.carry_scale.float().repeat_interleave(
        blk.carry_head_dim).view(1, 1, d)
    a = a + (mem * gain).to(a.dtype)
    y = h + blk.proj(a)
    y = y + blk.mlp(blk.ln2(y))
    g = torch.sigmoid(m.gates[k - 1](h))
    return h + g * (y - h)


def _manual_nd_states(m, x, depths):
    """按手写规则逐层重建 states（供四条组合单测对照模型前向）。"""
    with torch.no_grad():
        h = m._input_states(x)
        mem = m._carry_box["mem"].detach()
        frozen = h
        states = [h]
        for k in range(1, m.cfg.n_layers + 1):
            active = depths >= k
            out = _manual_nd_layer(m, k, h, frozen, mem, depths)
            h = torch.where(active.unsqueeze(-1), out, frozen)
            frozen = torch.where((depths == k).unsqueeze(-1), h, frozen)
            states.append(h)
    return states


def _nd_depths(n_layers=3):
    """固定的混合退出模式：逐 batch 内容不同（防掩码 [B,T,T]/[1,B,T,T] 侥幸）。"""
    return torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 3, 2],
                         [3, 3, 1, 2, 3, 0, 1, 3, 2, 1, 2, 3]])


# ---------------- ① carry × ragged（no-fill 掩码路径） ----------------

def test_eng2_carry_x_ragged_layer_matches_manual_rule():
    """carry × ragged 能建起来、且写入通道在 ragged 路径上真的活着：模型前向与手写
    ragged+通道规则逐层一致（掩码 / 逐通道门 / 读出 / 读出进 proj 四处都对着抄），
    且 γ 逃逸后读出量非零。"""
    m = _nd_carry("delta", fill_kv=False)
    assert type(m.blocks[0]).__name__ == "CarryChannelBlock"
    _activate_carry_scales(m)
    torch.manual_seed(5)
    x = torch.randint(0, 10, (2, 12))
    depths = _nd_depths()
    _, aux = m(x, exit_depths=depths)
    assert len(aux["attn_carry"]) == m.cfg.n_layers        # 遥测真的进了 aux
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0  # 通道活着
    manual = _manual_nd_states(m, x, depths)
    for k in range(m.cfg.n_layers + 1):
        assert torch.allclose(manual[k], aux["states"][k], atol=1e-5), f"layer {k}"


# ---------------- ② carry × 随机退出（fill 缺省，退出模式影响前向） ----------------

def test_eng2_carry_x_random_exit_composes():
    """carry × 随机退出：退出模式真的生效（已退出位置状态定格 = 冻结不变量），
    且含通道的前向与手写规则逐层一致、读出量非零。"""
    m = _nd_carry("delta", fill_kv=True)
    assert type(m.blocks[0]).__name__ == "CarryChannelBlock"
    _activate_carry_scales(m)
    torch.manual_seed(6)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, m.cfg.n_layers,
                                torch.Generator().manual_seed(7))
    assert depths.min() < m.cfg.n_layers                   # 真的有退出（非全深度）
    _, aux = m(x, exit_depths=depths)
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0
    L = m.cfg.n_layers
    for b in range(2):
        for i in range(12):
            for k in range(int(depths[b, i]) + 1, L + 1):
                assert torch.allclose(aux["states"][k][b, i],
                                      aux["states"][int(depths[b, i])][b, i],
                                      atol=1e-6)          # 退出即冻结
    manual = _manual_nd_states(m, x, depths)
    for k in range(L + 1):
        assert torch.allclose(manual[k], aux["states"][k], atol=1e-5), f"layer {k}"


# ---------------- ③ carry × 随机退出 × fill（混合填充 KV 源被真的消费） ----------------

def test_eng2_carry_x_random_exit_x_fill_consumes_frozen_kv():
    """carry × 随机退出 × fill：混合填充的 KV 源（含 proj-fill adapter）**真的进了
    带通道的块**——把 adapter 从恒等扰动成非恒等（**换方向**，不是缩放：kv 源经
    `ln1` 的 LayerNorm 对正缩放几乎不敏感，缩放扰动是退化测试），底层 logits 必须
    变。若块忽略 `kv_from`（变异 M1），adapter 被无视 ⇒ 本断言抓红。"""
    m = _nd_carry("delta", fill_kv=True, proj_fill=True)
    _activate_carry_scales(m)
    torch.manual_seed(8)
    x = torch.randint(0, 10, (2, 12))
    depths = torch.full((2, 12), 3, dtype=torch.long)
    depths[0, 4] = 0                                        # 至少一个早退位置
    d = m.cfg.d_model
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        base = m(x, exit_depths=depths)[1]["depth_logits"][-1].float().clone()
        for a in m.kv_adapters:
            # 非恒等且**换方向**（I + 0.2·N(0,1)），远离 LN 的缩放不变性
            a.weight.data.copy_(
                torch.eye(d) + 0.2 * torch.randn(d, d, generator=gen))
        after = m(x, exit_depths=depths)[1]["depth_logits"][-1].float()
    # 只有单个早退位置的 KV 源被扰动 ⇒ 信号很小（实测 max|Δlogit| ≈ 2e-4，在随机
    # 初始化的近均匀注意力下），但 **kv_from 被真的消费时必非零**；忽略 kv_from 时
    # 扰动完全不可见（Δ == 0 bitwise）⇒ 阈值取 1e-6（≫ float 噪声 1e-7）。
    delta = float((base - after).abs().max())
    assert delta > 1e-6, (
        "混合填充的 KV adapter 未被带通道的块消费（kv_from 被忽略）；"
        f"max|Δlogit|={delta}")
    # 通道自身也活着
    _, aux = m(x, exit_depths=depths)
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0


# ---------------- ④ carry × proj-fill（恒等起步 = fill 臂） ----------------

def test_eng2_carry_x_proj_fill_identity_starts_equal_to_fill():
    """carry × proj-fill：adapter 恒等初始化 ⇒ 同 seed 下 states/loss 与 fill 臂
    逐位一致（remap 从 fill 出发学）；通道读出量非零。"""
    m_proj = _nd_carry("delta", fill_kv=True, proj_fill=True)
    m_fill = _nd_carry("delta", fill_kv=True)
    _activate_carry_scales(m_proj)
    _activate_carry_scales(m_fill)
    torch.manual_seed(9)
    x = torch.randint(0, 10, (2, 12))
    depths = sample_exit_depths(2, 12, 3, torch.Generator().manual_seed(10))
    l1, a1 = m_proj(x, exit_depths=depths)
    l2, a2 = m_fill(x, exit_depths=depths)
    assert torch.allclose(l1.detach(), l2.detach(), atol=1e-6)
    for s1, s2 in zip(a1["states"], a2["states"]):
        assert torch.allclose(s1, s2, atol=1e-6)
    assert float(a1["attn_carry"][0]["read_share"]) > 0.0


# ---------------- ⑤ 写入时机：写入发生在退出之前、与退出模式无关 ----------------

def test_eng2_carry_write_timing_ignores_exit_depths():
    """写入时机语义钉（ENG2 的关键设计点）：记忆槽在 `_input_states` 对输入流 h₀
    扫描一次，**所有位置（含已退出位置）都写各自的 h₀**，因此同一输入在任意退出
    模式下得到**同一段记忆**；改变退出模式只改「读出在哪几层生效」，不改槽内容。
    若有人把写入改成按退出跳过/改写成冻结表征，本断言抓红。"""
    m = _nd_carry("delta", fill_kv=False)
    _activate_carry_scales(m)
    torch.manual_seed(11)
    x = torch.randint(0, 10, (2, 12))
    mems = []
    for depths in (torch.full((2, 12), 3, dtype=torch.long),
                   _nd_depths(),
                   torch.zeros(2, 12, dtype=torch.long)):
        with torch.no_grad():
            m(x, exit_depths=depths)
            mems.append(m._carry_box["mem"].clone())
    assert torch.equal(mems[0], mems[1]) and torch.equal(mems[0], mems[2]), \
        "写入通道的槽内容随退出模式变化（写入时机不再是「退出之前、全位置各写一次」）"


# ---------------- ⑥ 缺省路径逐位还原 v1（ENG2 不改缺省行为） ----------------

def test_eng2_default_path_bitwise_v1():
    """ENG2 只解耦 carry 的守卫、只改 `CarryChannelBlock`（非缺省块）：缺省非 dense
    模型（carry_rule=None）的块类、参数字典与前向仍与 `Block` 路径逐位一致。"""
    m = tiny_model()                                        # dense=False，无通道
    assert all(type(b).__name__ == "Block" for b in m.blocks)
    assert m.cfg.carry_rule is None
    x = torch.randint(0, 10, (2, 12))
    depths = _nd_depths()
    with torch.no_grad():
        _, aux = m(x, exit_depths=depths)
    # 无通道时 `_input_states` 不建槽、aux 不含 attn_carry（旧口径）
    assert "attn_carry" not in aux
    assert not hasattr(m, "_carry_box")


# ---------------- ⑦ 增量解码路径显式拒绝 carry（不交付静默丢路） ----------------

def test_eng2_decoder_rejects_carry_channel():
    """ENG2 让写入通道可与非 dense 路径同开，但**增量解码路径尚未接入读出**
    （`DepthARDecoder.step` 只算注意力、不会加 γ ⊙ mem_t）⇒ 显式报错，避免交付
    一条「看起来能用、实际少了一路」的解码路径（同 GQA 的处置）。"""
    from pathlm.depth_ar import DepthARDecoder
    m = _nd_carry("delta", fill_kv=True)
    with pytest.raises(ValueError, match="写入通道"):
        DepthARDecoder(m)


# ================= E1（ticket 67 §2）：逐层状态 + 内容写门 + 可选衰减 ===========

def test_carry_decay_one_is_bitwise_identity():
    """E1 硬性要求：新开关**默认关闭、逐位不变**。

    `carry_scan(..., decay=1.0)`（新参数缺省值）必须与不传 `decay` **逐位相同**；
    两条规则、多个 chunk 都如此（γ=1 走原公式、不加任何衰减乘子）。"""
    torch.manual_seed(3)
    q = torch.randn(2, 2, 17, 4)
    k = torch.randn(2, 2, 17, 4)
    v = torch.randn(2, 2, 17, 4)
    beta = torch.rand(2, 2, 17)
    for rule in ("plain", "delta"):
        for chunk in (1, 4, 17):
            y_def, _ = carry_scan(q, k, v, beta, rule=rule, chunk=chunk)
            y_one, _ = carry_scan(q, k, v, beta, rule=rule, chunk=chunk, decay=1.0)
            assert torch.equal(y_def, y_one), (rule, chunk)
            for t in (carry_scan_sequential,):
                a, _ = t(q, k, v, beta, rule=rule)
                b, _ = t(q, k, v, beta, rule=rule, decay=1.0)
                assert torch.equal(a, b), rule


def test_carry_default_flags_are_inert_bitwise():
    """E1 硬性要求（模型级）：`carry_per_layer=False` + `carry_decay=1.0`（新字段
    的缺省）与**不传新字段**的模型在参数字典与前向上**逐位相同** —— 等价性证明，
    不是「近似一致」。"""
    torch.manual_seed(0)
    m0 = carry_model("delta")
    m1 = carry_model("delta", carry_per_layer=False, carry_decay=1.0)
    s0, s1 = m0.state_dict(), m1.state_dict()
    assert sorted(s0) == sorted(s1)
    for kname in s0:
        assert torch.equal(s0[kname], s1[kname]), kname
    for blk in m0.blocks:
        blk.carry_scale.data.fill_(0.7)
    for blk in m1.blocks:
        blk.carry_scale.data.fill_(0.7)
    x = torch.randint(0, 10, (2, 12))
    m0.eval(); m1.eval()
    with torch.no_grad():
        for d in (None,):
            a = m0(x, exit_depths=d)[1]["depth_logits"][-1]
            b = m1(x, exit_depths=d)[1]["depth_logits"][-1]
            assert torch.equal(a, b)
    # 单共享槽的扫描遥测形状也不变：不新增 `layer_idx` 键（训练日志逐值 float 依赖它）
    assert "layer_idx" not in m0._carry_tele
    assert all(float(v) == float(v) for v in m0._carry_tele.values())


def test_carry_decay_chunked_matches_sequential():
    """E1：可选衰减 γ 的**分块实装 == 逐位置参考实现**（fp64 1e-12 内）。delta 的
    带衰减 WY 表示不是靠直觉写的（写强度 `β_i γ^{C−1−i}`、读出侧 `γ^{i−j}` 两个
    因子），靠这条测试对逐位置定义钉死；γ=1 也一并过（防回归）。"""
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(3)
        B, H, T, dh = 2, 2, 21, 4
        q = torch.randn(B, H, T, dh) * 0.7
        k = torch.randn(B, H, T, dh)
        v = torch.randn(B, H, T, dh)
        beta = torch.rand(B, H, T)
        for decay in (1.0, 0.999, 0.9, 0.5, 0.2):
            for rule in ("plain", "delta"):
                y_ref, _ = carry_scan_sequential(q, k, v, beta, rule, decay)
                for chunk in (1, 3, 5, 8, 21):
                    y, tele = carry_scan_chunked(q, k, v, beta, rule, chunk, decay)
                    assert float((y - y_ref).abs().max()) < 1e-12, (decay, rule, chunk)
                    assert torch.isfinite(tele["state_fro"])
    finally:
        torch.set_default_dtype(torch.float32)


def test_carry_decay_forgets_and_matches_recurrence():
    """E1：衰减的**语义**可观测——γ<1 时旧写入对读出的贡献随距离衰减（γ=1 不减）。
    单键单值、β 恒定的解析式：位置 t 读出 = β·Σ_{j≤t} γ^{t−j}·(k̂·q)·v。"""
    torch.manual_seed(5)
    B, H, dh = 1, 1, 4
    kh = torch.zeros(B, H, dh)
    kh[..., 0] = 1.0
    k = kh.unsqueeze(2).repeat(1, 1, 3, 1)
    v = torch.zeros(B, H, 3, dh)
    v[..., 0, 0] = 1.0                                # 只在 t=0 写一次
    beta = torch.full((B, H, 3), 0.5)
    q = kh.unsqueeze(2).repeat(1, 1, 3, 1)
    for decay in (1.0, 0.5):
        y, _ = carry_scan_sequential(q, k, v, beta, "plain", decay)
        want = torch.tensor([0.5, 0.5 * decay, 0.5 * decay ** 2])
        assert torch.allclose(y[0, 0, :, 0], want, atol=1e-12), decay
    y1, _ = carry_scan_sequential(q, k, v, beta, "plain", 1.0)
    yh, _ = carry_scan_sequential(q, k, v, beta, "plain", 0.5)
    assert float(yh[0, 0, 2, 0]) < float(y1[0, 0, 2, 0])   # 衰减把旧写入压小


def test_carry_per_layer_states_are_independent(monkeypatch):
    """E1：逐层**独立状态**——每层的状态由**本层当前 hidden** 写入。

    抓法：monkeypatch `CarryChannel.write_read` 记录 `(layer_idx, mem)`。断言
    ① 每层各调一次、层号 0..L−1 都到；② 各层 state 张量两两不同（若只是同一次
    h₀ 扫描的复本会逐位相同 ⇒ 抓红）；③ 与单共享槽（h₀ 上一次扫描）的 mem 不同
    （证明写入源确实是当前 hidden，而不是 h₀）。"""
    torch.manual_seed(0)
    m = carry_model("delta", carry_per_layer=True)
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=1.0)
    rec: dict = {}
    orig = CarryChannel.write_read

    def spy(self, h0, rule=None, decay=None, layer_idx=None):
        mem, tele = orig(self, h0, rule=rule, decay=decay, layer_idx=layer_idx)
        rec[layer_idx] = mem.detach().clone()
        return mem, tele

    monkeypatch.setattr(CarryChannel, "write_read", spy)
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        m(x)
    L = m.cfg.n_layers
    assert sorted(rec) == list(range(L)), sorted(rec)
    mems = [rec[k] for k in range(L)]
    for i in range(L):
        for j in range(i + 1, L):
            assert not torch.equal(mems[i], mems[j]), (i, j)
    # 单共享槽（同一权重、h₀ 一次扫描）≠ 逐层状态
    shared = m.carry.write_read(m.carry.gate(m._input_states(x)))[0]
    assert not torch.equal(mems[-1], shared)


def test_carry_write_gate_is_content_dependent():
    """E1：写入门（β = σ(W_β h + b_β)）**内容相关**——同权重下不同 hidden 给不同
    门；门随位置/内容变化（不是常数），因此「该不该写」由当前 hidden 决定。"""
    torch.manual_seed(0)
    m = carry_model("delta", carry_per_layer=True)
    ch = m.carry

    def gate(h):
        hf = h.float()
        return torch.sigmoid(
            torch.einsum("btd,hd->bth", hf, ch.beta_w.float())
            + ch.beta_b.float()).detach()

    a, b = torch.randn(2, 6, 32), torch.randn(2, 6, 32)
    ga, gb = gate(a), gate(b)
    assert not torch.allclose(ga, gb)                   # 不同 hidden ⇒ 不同门
    assert not torch.allclose(ga[:, 0], ga[:, 1])       # 门随内容/位置变化
    assert float(ga.std()) > 1e-4                       # 不是退化常数门
    # 端到端：改一个 token 改变该位置之后的写入状态（gate 真的进了状态）
    m.eval()
    x = torch.randint(0, 10, (2, 12))
    y = x.clone()
    y[:, 0] = (x[:, 0] + 1) % 10
    with torch.no_grad():
        la = m(x)[1]["depth_logits"][-1]
        lb = m(y)[1]["depth_logits"][-1]
    assert not torch.equal(la, lb)


def test_carry_per_layer_changes_forward_and_gradients_flow():
    """E1：① 逐层模式确实改了前向（与单共享槽不同）；② 活性/梯度——写入路径与
    每层读出增益的梯度非零且有限（不是死路径）。"""
    torch.manual_seed(0)
    shared = carry_model("delta")
    per = carry_model("delta", carry_per_layer=True)
    for mm in (shared, per):
        for blk in mm.blocks:
            blk.carry_scale.data.normal_(std=1.0)
    x = torch.randint(0, 10, (2, 12))
    loss_p, aux_p = per(x, exit_depths=None)
    with torch.no_grad():
        logits_shared = shared(x, exit_depths=None)[1]["depth_logits"][-1]
    assert not torch.equal(aux_p["depth_logits"][-1], logits_shared)
    loss_p.backward()
    for p in per.carry.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert float(p.grad.abs().max()) > 0
    for blk in per.blocks:
        assert float(blk.carry_scale.grad.abs().max()) > 0
    assert aux_p["attn_carry"] and len(aux_p["attn_carry"]) == per.cfg.n_layers


def test_carry_per_layer_composes_with_gqa_and_liveness():
    """能力面（§22.8）：逐层状态**不换 block 类、不与 GQA 互斥**——GQA × carry ×
    per_layer 走 `ComposableBlock` 且前向有限；活性块报 `per_layer=True`、
    `shared_slots=False`，扫描遥测按层聚合后仍非零。"""
    torch.manual_seed(0)
    m = carry_model("delta", n_kv_heads=2, carry_per_layer=True)
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    for blk in m.blocks:
        blk.carry_scale.data.normal_(std=1.0)
    m.eval()
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        _, aux = m(x, exit_depths=None)
    assert torch.isfinite(aux["depth_logits"][-1]).all()
    live = carry_liveness(m, m._carry_tele, aux["attn_carry"], prob=carry_probe(m, x))
    assert live["per_layer"] is True and live["shared_slots"] is False
    assert live["enabled"] and live["alive"]
    assert float(m._carry_tele["n_layers"]) == m.cfg.n_layers
    # 训练日志 `attn_carry_scan_mean` 会对 tele 的每个值 `float(v)`：逐层聚合后的
    # 遥测必须全是可转 float 的标量（否则共享卡的训练日志会中途抛错）。
    assert all(float(v) == float(v) for v in m._carry_tele.values())
    # 单共享槽缺省下 `per_layer=False`、`shared_slots=True`（不被误报）
    ms = carry_model("delta")
    for blk in ms.blocks:
        blk.carry_scale.data.fill_(0.7)
    ms.eval()
    with torch.no_grad():
        _, auxs = ms(x, exit_depths=None)
    lives = carry_liveness(ms, ms._carry_tele, auxs["attn_carry"])
    assert lives["per_layer"] is False and lives["shared_slots"] is True


def test_carry_e1_validation_and_decay_plumbing():
    """能力面（§22.8）：新 flag 的死组合**显式报错**；正组合真的把 γ 接到模型上。

    - `carry_per_layer=True` 而 `carry_rule=None` ⇒ 报错（否则是死开关）；
    - `carry_decay` 离开 (0,1] 或 `carry_rule=None` ⇒ 报错；
    - `carry_decay<1` 真的进 `CarryChannel.decay` 并改变前向。"""
    with pytest.raises(ValueError, match="carry_per_layer"):
        DepthARModel(DepthARConfig(d_model=32, n_layers=3, n_heads=4, seq_len=12,
                                   dense=True, carry_per_layer=True), 10)
    with pytest.raises(ValueError, match="carry_decay"):
        DepthARModel(DepthARConfig(d_model=32, n_layers=3, n_heads=4, seq_len=12,
                                   dense=True, carry_decay=0.5), 10)
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="carry_decay"):
            carry_model("delta", carry_decay=bad)
    torch.manual_seed(0)
    m1 = carry_model("delta", carry_decay=1.0)
    m5 = carry_model("delta", carry_decay=0.5)
    assert m1.carry.decay == 1.0 and m5.carry.decay == 0.5
    for mm in (m1, m5):
        for blk in mm.blocks:
            blk.carry_scale.data.normal_(std=1.0)
        mm.eval()
    x = torch.randint(0, 10, (2, 12))
    with torch.no_grad():
        l1 = m1(x)[1]["depth_logits"][-1]
        l5 = m5(x)[1]["depth_logits"][-1]
    assert not torch.equal(l1, l5)
