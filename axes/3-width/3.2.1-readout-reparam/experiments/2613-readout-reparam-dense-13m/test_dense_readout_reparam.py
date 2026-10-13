"""2613 · 契约 F 的语义开关在**普通 dense 形态**下的可失败单测。

Run: python3 -m pytest axes/W-width/W4-readout-geometry/experiments/2613-readout-reparam-dense-13m/test_dense_readout_reparam.py -q

钉住三件事（本实验新增的语义接缝 = 「读出重参数化作用在 dense 单出口路径上」）：
1. 开关关闭时 dense 路径逐位还原 v1（`logits = h.float() @ Eᵀ`）；
2. 恒等起步：dense+linres 的 readout_p=0 ⇒ 全深度 logits 与同 seed 的 dense 基线
   逐位一致，且 state_dict 只多 readout_p/q 两个张量（单因素：只加读出几何）；
3. dense 只监督最终深度 CE；P/Q 在一步梯度后离开零点（非鞍点，干预是活的）。
"""

import os, sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel

VOCAB = 10


def tiny_dense(**kw) -> DepthARModel:
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=16, n_layers=3, n_heads=2, seq_len=12,
                        dense=True, **kw)
    return DepthARModel(cfg, VOCAB)


def _x():
    torch.manual_seed(41)
    return torch.randint(0, VOCAB, (2, 12))


def test_dense_readout_off_matches_v1_bitwise():
    """① 开关关闭：dense 路径每层 logits 与 v1 表达式逐位相等。"""
    m = tiny_dense()
    x = _x()
    _, aux = m(x, exit_depths=None)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        assert torch.equal(lg, s.float() @ m.embed.weight.float().T)


def test_dense_linres_off_state_dict_and_init_identity():
    """② 恒等起步：dense+linres 与同 seed dense 基线的公共参数逐位相同，
    P=0 时全深度 logits 逐位一致（不引入任何初始税）。"""
    a = tiny_dense()
    r = tiny_dense(readout_linear_resid=True)
    sa, sr = a.state_dict(), r.state_dict()
    assert set(sr) - set(sa) == {"readout_p.weight", "readout_q.weight"}
    for k in sa:
        assert torch.equal(sa[k], sr[k]), k
    assert torch.count_nonzero(sr["readout_p.weight"]) == 0
    assert torch.count_nonzero(sr["readout_q.weight"]) > 0
    x = _x()
    _, a_aux = a(x, exit_depths=None)
    _, r_aux = r(x, exit_depths=None)
    for la, lr in zip(a_aux["depth_logits"], r_aux["depth_logits"]):
        assert torch.equal(la, lr)
    assert a_aux["states"][-1].data_ptr() != r_aux["states"][-1].data_ptr()


def test_dense_linres_formula_and_activity():
    """③ dense 读出变换的精确公式：logits = (h + P(Qh)) @ Eᵀ；给 P 非零值后
    计算确实改变（开关真的作用在 dense 单出口路径上）。"""
    r = tiny_dense(readout_linear_resid=True)
    torch.manual_seed(42)
    with torch.no_grad():
        r.readout_p.weight.copy_(torch.randn_like(r.readout_p.weight) * 0.1)
    x = _x()
    _, aux = r(x, exit_depths=None)
    for s, lg in zip(aux["states"], aux["depth_logits"]):
        h = s.float()
        ref = (h + r.readout_p(r.readout_q(h))) @ r.embed.weight.float().T
        assert torch.allclose(lg, ref, atol=1e-6)
    a = tiny_dense()
    _, a_aux = a(x, exit_depths=None)
    # 最终深度读出确实被改写（活性），且不改写 trunk 状态
    assert not torch.allclose(aux["depth_logits"][-1], a_aux["depth_logits"][-1])


def test_dense_linres_params_match_2d2_and_gated():
    """④ 参数量账：dense+linres 比 dense 基线多 2d²，与 dense+gated 逐位相同。"""
    a = tiny_dense()
    r = tiny_dense(readout_linear_resid=True)
    g = tiny_dense(readout_gated=True)
    pa = sum(p.numel() for p in a.parameters())
    pr = sum(p.numel() for p in r.parameters())
    pg = sum(p.numel() for p in g.parameters())
    d = a.cfg.d_model
    assert pr - pa == 2 * d * d
    assert pr == pg


def test_dense_only_final_readout_supervised_and_saddle_escape():
    """⑤ dense 形态只监督最终深度 CE（n_ce=1）；一步 AdamW 后 P 离开零点
    （单边零非鞍点 ⇒ 干预是活的）。"""
    r = tiny_dense(readout_linear_resid=True)
    x = _x()
    loss, aux = r(x, exit_depths=None)
    tgt = x[:, 1:]
    ref = torch.nn.functional.cross_entropy(
        aux["depth_logits"][-1][:, :-1].reshape(-1, VOCAB).float(),
        tgt.reshape(-1))
    assert torch.allclose(loss, ref, atol=1e-6)
    opt = torch.optim.AdamW(r.parameters(), lr=1e-3)
    loss.backward()
    assert r.readout_p.weight.grad is not None
    assert r.readout_p.weight.grad.abs().sum() > 0
    opt.step()
    assert torch.count_nonzero(r.readout_p.weight) > 0
