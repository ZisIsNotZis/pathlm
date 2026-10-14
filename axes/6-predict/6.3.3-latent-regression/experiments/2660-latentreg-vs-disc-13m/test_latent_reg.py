"""单测（CPU，不取 GPU 锁）。契约 §6 的必要自检，全部须能失败（见 mutate.py）。

    python3 -m pytest test_latent_reg.py -q
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import latent_reg as L  # noqa: E402
from latent_reg import (PathConfig, PathLM, ModelConfig, _wrap_mtp_nodes,  # noqa: E402
                        ar_t1_block_hits, cluster_bootstrap_paired, freeze,
                        make_heads, train_arm, valid_positions, wilson_ci)


def tiny_model(seed: int = 0) -> PathLM:
    torch.manual_seed(seed)
    m = PathLM(ModelConfig(d_model=32, n_layers=2, n_heads=4, seq_len=64),
               PathConfig(n_mtp=3, chain_mode="token"), vocab_size=L.VOCAB_SIZE)
    return freeze(m.eval())


# 1 --------------------------------------------------------------------------
def test_feature_wrapper_does_not_change_backbone():
    """包装 `_mtp_nodes` 前后，同输入的主干 node logits 逐位相同。"""
    m = tiny_model(1)
    tok = torch.randint(0, L.VOCAB_SIZE, (2, 16))
    with torch.no_grad():
        h = L.trunk_h(m, tok)
        ref = m._mtp_nodes(h)
        restore, buf = _wrap_mtp_nodes(m)
        try:
            got = m._mtp_nodes(h)
            assert buf["h"] is h, "wrapper 未抓到 h"
        finally:
            restore()
    for k in ref:
        assert torch.equal(ref[k]["latent"], got[k]["latent"]), f"node{k} latent 变了"
        assert torch.equal(ref[k]["logits"], got[k]["logits"]), f"node{k} logits 变了"
    # restore 后行为恢复
    with torch.no_grad():
        again = m._mtp_nodes(h)
    assert torch.equal(ref[0]["logits"], again[0]["logits"])


# 2 --------------------------------------------------------------------------
def test_target_offset_j1_is_t_plus_2():
    """off-by-one 钉死：head1 (j=1) 预测 t_{i+2}；t+1 情形必须不同。"""
    torch.manual_seed(0)
    V, d, T = L.VOCAB_SIZE, 64, 6
    E = torch.randn(V, d)
    tok = torch.tensor([[5, 4, 3, 2, 1, 0]])
    heads = make_heads(2, 0, d, "cpu")
    for hd in heads:  # MLP=0 ⇒ TransformHead 退化为 identity ⇒ logits = h_i @ E^T
        nn.init.zeros_(hd.mlp[0].weight); nn.init.zeros_(hd.mlp[0].bias)
        nn.init.zeros_(hd.mlp[2].weight); nn.init.zeros_(hd.mlp[2].bias)
    pos = valid_positions(T, 2)
    assert list(pos) == [0, 1, 2]

    def hits_for(offset: int) -> int:
        H = torch.zeros(1, T, d)
        for i in pos:
            H[0, i] = E[tok[0, i + offset]] * 1e3  # 行 i 指向 t_{i+offset}
        return int(L.eval_hits(heads, H, tok, E, k=2, device="cpu")["_hits"][0, 0])

    assert hits_for(2) == len(pos), "head1 应在 t+2 全中（若记成 t+1 则为 0）"
    assert hits_for(1) < len(pos), "位移口径未能区分 t+1 与 t+2"
    assert valid_positions(T, 2)[-1] == T - 2 - 2  # 最大 i 满足 i+1+k <= T-1


# 3 --------------------------------------------------------------------------
def test_tied_readout_is_shared_embed():
    """两臂 logits 必须与主干 `model.embed.weight` 机器可核地同源。"""
    m = tiny_model(2)
    tok = torch.randint(0, L.VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        h = L.trunk_h(m, tok)
    heads = make_heads(2, 0, m.mcfg.d_model, "cpu")
    zs, logits = L.head_forward(heads, h, m.embed.weight, 2)
    for j in range(2):
        assert torch.equal(logits[j], zs[j] @ m.embed.weight.T), "读出不是同一份 embed"
    assert m.embed.weight.requires_grad is False
    with torch.no_grad():
        alt = L.head_forward(heads, h, m.embed.weight, 2)[1][0]
    assert torch.equal(logits[0], alt)


# 4 --------------------------------------------------------------------------
def test_freeze_assertion_and_grad_none():
    """主干 `requires_grad=False`；一次 backward 后 `.grad is None`。"""
    m = tiny_model(3)
    tok = torch.randint(0, L.VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        h = L.trunk_h(m, tok)
    heads = make_heads(2, 0, m.mcfg.d_model, "cpu")
    zs, logits = L.head_forward(heads, h, m.embed.weight, 2)
    loss = logits[0].sum() + logits[1].sum()
    loss.backward()
    assert all(not p.requires_grad for p in m.parameters()), "主干未冻结"
    assert all(p.grad is None for p in m.parameters()), "冻结主干出现梯度（变体应抓红）"
    # 头有梯度（证明梯度确实流过，而不是 loss 与主干无关）
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in heads.parameters())


# 5 --------------------------------------------------------------------------
def test_lambda0_arms_equivalent_and_deterministic():
    """λ=0 等价自检：disc_k2 与 lat_k2_lam0 规格相同（同代码路径）⇒ 同种子逐位相同。"""
    import run as R
    assert R.ARMS["disc_k2"]["lam"] == 0.0
    assert R.ARMS["lat_k2_lam0"]["lam"] == 0.0
    assert R.ARMS["disc_k2"]["k"] == R.ARMS["lat_k2_lam0"]["k"] == 2

    torch.manual_seed(0)
    nb, T, d = 2, 8, 16
    H = torch.randn(nb, T, d)
    tok = torch.randint(0, L.VOCAB_SIZE, (nb, T))
    E = torch.randn(L.VOCAB_SIZE, d)
    h1, _ = train_arm(H, tok, E, k=2, lam=0.0, seed=0, steps=5, batch=32, device="cpu")
    h2, _ = train_arm(H, tok, E, k=2, lam=0.0, seed=0, steps=5, batch=32, device="cpu")
    for j in range(2):
        for (n1, p1), (n2, p2) in zip(h1[j].state_dict().items(), h2[j].state_dict().items()):
            assert n1 == n2 and torch.equal(p1, p2), f"λ=0 训练非确定：{n1}"
    # λ>0 与 λ=0 必须不同（否则 λ 没接线）
    h3, _ = train_arm(H, tok, E, k=2, lam=1.0, seed=0, steps=5, batch=32, device="cpu")
    assert not torch.equal(h1[0].mlp[0].weight, h3[0].mlp[0].weight), "λ 未接线"


# 5b -------------------------------------------------------------------------
def test_head_files_all_tensors_equal_covers_every_tensor():
    """L0 权重同一性必须覆盖**整份 state_dict**（两个头的全部张量），不是单个张量。"""
    import tempfile
    d = 8
    h1 = make_heads(2, 0, d, "cpu")
    h2 = make_heads(2, 0, d, "cpu")            # 同 seed ⇒ 同初始化
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "a.pt")
        p2 = os.path.join(td, "b.pt")
        L.save_heads(h1, p1)
        L.save_heads(h2, p2)
        assert L.head_files_all_tensors_equal(p1, p2) is True
        # 只改 head1 的**最后一个**张量（非 head0.mlp.0.weight）⇒ 仍须报不同
        sd = torch.load(p2, map_location="cpu", weights_only=True)
        last = sorted(sd["head1"])[-1]
        sd["head1"][last] = sd["head1"][last] + 1.0
        torch.save(sd, p2)
        assert L.head_files_all_tensors_equal(p1, p2) is False, "未覆盖全部张量（只比了一个）"


# 6 --------------------------------------------------------------------------
def test_no_wall_clock_fields_in_results():
    """results.json 不得有墙钟字段（机器可核）。"""
    import run as R
    assert R.has_wall_clock({"hit": 1, "ce": 2.0}) == []
    assert R.has_wall_clock({"wall_seconds": 1}) != []
    assert R.has_wall_clock({"nested": {"elapsed_ms": 1}}) != []
    p = os.path.join(HERE, "results.json")
    if os.path.exists(p):
        import json
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        bad = [b for b in R.has_wall_clock(res) if not b.endswith("_wall_clock_fields")]
        assert not bad, f"results.json 含墙钟字段：{bad}"


# 6b -------------------------------------------------------------------------
def test_mse_target_position_is_next_next_hidden():
    """MSE 目标位移钉死：head0 的 MSE 目标必须是 `h_{i+2}`（而非 `h_{i+1}`）。

    构造 H：位置 i+1 与 i+2 的隐状态固定且可分 ⇒ 训头（λ=10）后 head0 输出 z 应更靠近 h_{i+2}。
    若把 MSE 目标错位到 h_{i+1}（变异 `mse_target_off_by_one`），本测试必须抓红。
    """
    torch.manual_seed(0)
    nb, T, d = 2, 6, 4
    base = [torch.tensor([1., 0, 0, 0]), torch.tensor([0., 1, 0, 0]),
            torch.tensor([0., 0, 1, 0]), torch.tensor([0., 0, 0, 1]),
            torch.tensor([1., 1, 0, 0]), torch.tensor([0., 1, 1, 0])]
    H = torch.zeros(nb, T, d)
    for b in range(nb):
        for t in range(T):
            H[b, t] = base[t] * (1.0 if b == 0 else -1.0)
    tok = torch.randint(0, L.VOCAB_SIZE, (nb, T))
    E = torch.randn(L.VOCAB_SIZE, d)
    pos = valid_positions(T, 2)
    heads, _ = train_arm(H, tok, E, k=2, lam=10.0, seed=0, steps=300,
                         batch=nb * len(pos), device="cpu")
    with torch.no_grad():
        z = heads[0](H)                       # head0（j=1 ⇒ 目标 h_{i+2}）的输出
    d_correct = sum(((z[:, i] - H[:, i + 2]) ** 2).sum().item() for i in pos)
    d_shifted = sum(((z[:, i] - H[:, i + 1]) ** 2).sum().item() for i in pos)
    assert d_correct < d_shifted, (
        f"MSE 目标不在 h_{{i+2}}：dist_correct={d_correct:.4f} >= "
        f"dist_shifted={d_shifted:.4f}（目标位移）")


# 7 --------------------------------------------------------------------------
def test_stats_helpers():
    """Wilson 与聚类配对 bootstrap 的符号/覆盖健全性。"""
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi
    a = np.array([10, 10, 10, 10])
    b = np.array([4, 4, 4, 4])
    n = np.array([20, 20, 20, 20])
    d = cluster_bootstrap_paired(a, b, n)
    assert d["delta"] == 0.3 and d["excludes_zero"] is True
    d2 = cluster_bootstrap_paired(a, a, n)
    assert d2["delta"] == 0.0 and d2["excludes_zero"] is False
    tok = torch.tensor([[1, 2, 3, 4, 5, 6]])
    n1 = np.tile(np.array([[2, 3, 4, 5, 6]]), (1, 1))  # 每行预测下一 token ⇒ 全中
    hits = ar_t1_block_hits(tok, n1, k=2)
    assert hits[0] == len(valid_positions(6, 2))


# 8 --------------------------------------------------------------------- 路径 (ii)
def test_draftverify_head_reads_trunk_latent_and_repro_direct1():
    """路径 (ii) 接线：N+2 草稿取自 `nodes[0]["latent"]`（主干最终隐状态）经 head0 + tied 读出；
    N+3 固定取自引擎 `nodes[3]`；`repro_direct1` = 引擎默认（2640 direct1 定义）。

    变异 `draftverify_head_wrong_position`（改取 `nodes[1]["latent"]`）必须让本测试抓红。
    """
    import draftverify as D
    torch.manual_seed(0)
    V, d = L.VOCAB_SIZE, 8
    E = torch.zeros(V, d)
    E[7, 0] = E[9, 1] = E[5, 2] = E[6, 3] = 1.0
    fake = nn.Module()
    fake.embed = nn.Embedding(V, d)
    with torch.no_grad():
        fake.embed.weight.copy_(E)
    head = L.TransformHead(d)
    for lin in (head.mlp[0], head.mlp[2]):      # 归零 ⇒ TransformHead = identity ⇒ z = h
        nn.init.zeros_(lin.weight)
        nn.init.zeros_(lin.bias)

    def onehot(token: int) -> torch.Tensor:
        lg = torch.full((1, 1, V), -1e9)
        lg[0, 0, token] = 0.0
        return lg

    nodes = {
        0: {"latent": torch.tensor([[[1., 0, 0, 0, 0, 0, 0, 0]]])},   # tied 读出 ⇒ 7
        1: {"latent": torch.tensor([[[0., 1, 0, 0, 0, 0, 0, 0]]])},   # 接错才得 9
        2: {"logits": onehot(5)},
        3: {"logits": onehot(6)},
    }
    fn = D.make_draft_fn_head(head, fake)
    drafts = fn(nodes, D.GAMMA)
    assert drafts == [7, 6], f"路径 (ii) 接线错：{drafts}（应 N+2=head0 ⇒ 7，N+3=node3 ⇒ 6）"
    assert drafts[0] != 9, "N+2 草稿误取 nodes[1]（草稿头接错位置）"
    # repro_direct1 = 引擎默认 = 2640 direct1 = [argmax(nodes[2]), argmax(nodes[3])]
    assert D.make_draft_fn_direct1(nodes, D.GAMMA) == [5, 6]


# 9 --------------------------------------------------------------------------
def test_draftverify_stats_definitions():
    """路径 (ii) 指标定义：α₁=接受≥1 的轮占比、τ=平均接受数、α₂|₁；配对 bootstrap 符号正确。

    变异 `draftverify_alpha1_loosened`（`a >= 1` → `a >= 2`）必须让本测试抓红。
    """
    import draftverify as D
    g = [[2, 2], [1, 0]]
    assert D.stat_alpha1(g) == 3 / 4, "α₁ 口径错（应为接受≥1 的轮占比）"
    assert D.stat_tau(g) == 5 / 4, "τ 口径错（应为平均接受数）"
    assert D.stat_alpha2(g) == 2 / 3, "α₂|₁ 口径错"
    d = D.paired_diff([[1, 1]] * 4, [[0, 0]] * 4, D.stat_alpha1, B=200, seed=0)
    assert d["point"] == 1.0 and d["excludes_zero"] is True
    d0 = D.paired_diff([[1, 0]] * 4, [[1, 0]] * 4, D.stat_alpha1, B=200, seed=0)
    assert d0["point"] == 0.0 and d0["excludes_zero"] is False


# 10 -------------------------------------------------------------------------
def test_draftverify_repro_reference_present():
    """保真硬门的参照（2640 direct1）确实存在且等于钉死的 α₁/τ —— 参照被改动时必须报。"""
    import draftverify as D
    with open(D.REF_2640, encoding="utf-8") as f:
        import json
        ref = json.load(f)["checkpoints"][D.REF_CKPT]["arms"][D.REF_ARM]
    assert abs(ref["alpha1"] - D.REF_ALPHA1) < 1e-6, "2640 direct1 α₁ 与钉死值不符"
    assert abs(ref["tau"] - D.REF_TAU) < 1e-6, "2640 direct1 τ 与钉死值不符"
    assert len(ref["per_prompt"]) == D.N_PROMPTS


# 11 -------------------------------------------------------------------------
def test_draftverify_prompts_match_2640_protocol():
    """prompt 构造与 2640 同口径：`val[i*5000 : i*5000+32]`。"""
    import draftverify as D
    val = np.arange(12 * D.STRIDE + D.PROMPT_LEN, dtype=np.int64)
    ps = D.make_prompts(val, 3)
    for i, p in enumerate(ps):
        assert p.dtype == torch.int64 and p.shape[0] == D.PROMPT_LEN
        assert int(p[0]) == i * D.STRIDE and int(p[-1]) == i * D.STRIDE + D.PROMPT_LEN - 1
