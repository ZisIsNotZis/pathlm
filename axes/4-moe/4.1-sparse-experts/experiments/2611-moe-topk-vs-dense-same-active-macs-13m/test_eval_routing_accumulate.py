"""2611 的**设备对齐测试**：评测期的专家使用率累加器必须能吃 CUDA 上的 `last_usage`。

存在的理由是一条真实缺陷：`eval_arm` 用 CPU 的 float64 累加器跨 20 批累加，而 `last_usage` 跟着
模型在 CUDA 上，`acc[i] += m.last_usage.double()` 于是在第一个 MoE 臂就
`RuntimeError: Expected all tensors to be on the same device`——8 臂训完、`--stage eval` 才炸，
实验根 `results.json` 写不出来。这条测试直接复现那个场景（小块模型上 CUDA、前向一次、再累加），
所以它在缺陷存在时**必红**，在修复后必绿。

无 CUDA 时跳过：本实验的评测只在 CUDA 上跑，CPU 上两边的张量本来就同设备，测不出这个缺陷。

跑法：
    python3 -m pytest axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/test_eval_routing_accumulate.py -q
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_2611_accum", os.path.join(HERE, "run.py"))
run2611 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run2611)

HAS_CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not HAS_CUDA, reason="设备对齐缺陷只在 CUDA 上重现（CPU 上累加器与 last_usage 同设备）")
def test_accumulate_usage_accepts_cuda_last_usage():
    """CUDA 模型的一次前向之后，`accumulate_usage` 能把它写进 CPU 累加器，且数值对得上。"""
    torch.manual_seed(0)
    d, e, h_e = 8, 4, 4
    moe = run2611.TopKMoEFFN(d, e, 1, h_e).cuda().eval()
    x = torch.randn(2, 5, d, device="cuda")
    with torch.no_grad():
        moe(x)
    acc = [torch.zeros(e, dtype=torch.float64) for _ in range(1)]
    run2611.accumulate_usage(acc, _OneLayer(moe))          # 缺陷存在时这里抛 RuntimeError
    assert acc[0].device.type == "cpu"
    assert int(acc[0].sum().item()) == 2 * 5               # 2×5 个 token 各选一个专家


@pytest.mark.skipif(not HAS_CUDA, reason="同上的设备对齐测试")
def test_accumulate_usage_adds_up_across_batches():
    """第二次累加是相加而不是覆盖（评测期要跨 20 批）。"""
    torch.manual_seed(0)
    d, e, h_e = 8, 4, 4
    moe = run2611.TopKMoEFFN(d, e, 1, h_e).cuda().eval()
    acc = [torch.zeros(e, dtype=torch.float64) for _ in range(1)]
    for _ in range(3):
        with torch.no_grad():
            moe(torch.randn(2, 4, d, device="cuda"))
        run2611.accumulate_usage(acc, _OneLayer(moe))
    assert int(acc[0].sum().item()) == 3 * 2 * 4


class _OneLayer:
    """`accumulate_usage` 只用到 `model.moe_modules()`；单层替身让测试不必建整个模型。"""

    def __init__(self, moe):
        self._moe = moe

    def moe_modules(self):
        return [self._moe]
