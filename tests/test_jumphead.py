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
from train_jumphead import (HeadBank, JumpHead, assert_frozen, freeze_trunk,
                            init_heads, shift_pairs, trunk_hidden)


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
