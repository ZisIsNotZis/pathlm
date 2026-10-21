"""2611 的**实现等价性测试**：重绑后的 `TopKMoEFFN.forward` 必须与重绑前的 `forward_naive`
逐位相同（`max_abs_diff == 0`，不是「接近 0」），且 `E=1/top-1` 灌入 stock `mlp` 权重后与
dense 臂逐位相同。

这是「换实现不换科学设定」的证据（契约见同目录 `README.md` §10）：本文件里的断言在设计上
**可失败**——`test_equality_assertion_can_fail` 故意把一个专家权重改动 1e-3，断言逐位相等
不再成立，从而证明前面几条不是恒真断言。

跑法：
    python3 -m pytest axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/test_moe_forward_equivalence.py -q
"""

from __future__ import annotations

import importlib.util
import os
import warnings

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_2611", os.path.join(HERE, "run.py"))
run2611 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run2611)

HAS_CUDA = torch.cuda.is_available()
SHAPES = ((2, 17), (3, 32), (4, 64))


def _ffn(n_experts: int, top_k: int, hidden: int, d: int = 64, seed: int = 0):
    torch.manual_seed(seed)
    return run2611.TopKMoEFFN(d, n_experts, top_k, hidden).eval()


def _pair_out(m, x):
    with torch.no_grad():
        a = m(x)
        ua = m.last_usage.clone()
        b = m.forward_naive(x)
        ub = m.last_usage.clone()
    return a, ua, b, ub


@pytest.mark.parametrize("n_experts,top_k,hidden", [(1, 1, 96), (4, 1, 96), (16, 1, 96)])
def test_grouped_forward_is_bitwise_equal_to_naive(n_experts, top_k, hidden):
    """k=1（本实验全部臂的设定）：逐位相等，max_abs_diff 恰为 0。"""
    m = _ffn(n_experts, top_k, hidden)
    for shape in SHAPES:
        torch.manual_seed(1234 + shape[1])
        x = torch.randn(*shape, 64)
        fast, uf, naive, un = _pair_out(m, x)
        assert torch.equal(fast, naive), (
            f"E={n_experts} k={top_k} shape={shape}: max_abs_diff="
            f"{float((fast - naive).abs().max())}（必须恰为 0）")
        assert torch.equal(uf, un), f"E={n_experts} 使用率计数不一致：{uf} vs {un}"


def test_grouped_forward_matches_naive_for_top2_within_fp32_rounding():
    """k=2（本实验不用）：浮点等价，**不是**位级一致。

    原因写在这里而不是藏起来：k>1 时同一 token 的 k 份贡献由 `index_add` 累加，CUDA 上原子累加
    顺序不保证；`forward_naive` 按专家升序逐次写回。两条路径的乘加操作数相同，差异只在求和顺序。
    """
    m = _ffn(4, 2, 96)
    for shape in SHAPES:
        torch.manual_seed(4321 + shape[1])
        x = torch.randn(*shape, 64)
        fast, _, naive, _ = _pair_out(m, x)
        dmax = float((fast - naive).abs().max())
        assert dmax < 1e-6, (f"k=2 shape={shape}: max_abs_diff={dmax} 超出 fp32 舍入"
                             "（k=2 的累加顺序不保证，见 README §10）")


@pytest.mark.skipif(not HAS_CUDA, reason="bf16 autocast 分支只在 CUDA 上跑")
@pytest.mark.parametrize("n_experts,top_k,hidden", [(4, 1, 1536), (16, 1, 1536)])
def test_grouped_forward_is_bitwise_equal_to_naive_bf16(n_experts, top_k, hidden):
    m = _ffn(n_experts, top_k, hidden, d=256).cuda()
    for shape in ((2, 128), (8, 512)):
        torch.manual_seed(7 * shape[1])
        x = torch.randn(*shape, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                fast = m(x)
                naive = m.forward_naive(x)
        assert torch.equal(fast, naive), (
            f"bf16 E={n_experts} shape={shape}: max_abs_diff="
            f"{float((fast - naive).abs().max())}（必须恰为 0）")


def test_identity_point_e1_top1_matches_dense_mlp():
    """判据⑤：E=1/top-1 灌入 stock mlp 权重 ⇒ 与 dense 臂**逐位**一致（归一化后权重恒 1）。"""
    torch.manual_seed(0)
    cfg = run2611.model_cfg(run2611.BASE_MLP)
    dense = run2611.DepthARModel(cfg, 206).eval()
    moe1 = run2611.MoEDepthARModel(cfg, 206, 1, 1, run2611.BASE_MLP * run2611.D, renorm=True).eval()
    extra = run2611.copy_shared_from(dense, moe1)
    assert extra == [], f"MoE 臂多了非 mlp 参数：{extra}"
    for blk_d, blk_m in zip(dense.blocks, moe1.blocks):
        blk_m.mlp.experts[0].load_state_dict(blk_d.mlp.state_dict())
    torch.manual_seed(123)
    x = torch.randint(0, 206, (2, 64))
    with torch.no_grad():
        lg_d = dense(x)[1]["depth_logits"][-1]
        lg_m = moe1(x)[1]["depth_logits"][-1]
    assert torch.equal(lg_d, lg_m), (
        f"max_abs_diff={float((lg_d - lg_m).abs().max())}（必须恰为 0）")


def test_equality_assertion_can_fail():
    """反恒真断言：把一个专家权重改动 1e-3，逐位相等必须**不再成立**。"""
    m = _ffn(4, 1, 96)
    torch.manual_seed(99)
    x = torch.randn(2, 32, 64)
    with torch.no_grad():
        fast = m(x)
        naive = m.forward_naive(x)
    assert torch.equal(fast, naive)
    busy = int(m.last_usage.argmax())       # 扰动**被用到**的那个专家，否则这条断言会恒真
    assert int(m.last_usage[busy]) > 0
    with torch.no_grad():
        m.experts[busy][0].weight[0, 0] += 1e-3
        fast2 = m(x)
        naive2 = m.forward_naive(x)
    assert not torch.equal(fast2, fast), "改动权重后输出没变 ⇒ 那条逐位等式是恒真的"
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # 不允许用「数值警告」冒充断言失败
        assert float((fast2 - fast).abs().max()) > 0.0
    assert torch.equal(fast2, naive2), "扰动后两条路径仍应彼此逐位相同（k=1）"
