"""2654 · `layerdrop-pt` 逐 token 跳层接线回归测试（CPU、秒级）。

P0 根因：`ARMS["layerdrop-pt"]` 曾缺 `"p"` 键，`brand_layers` 的 `spec.get("p", 0.0)`
把 p 缺省成 0.0 ⇒ `SkipBlock` 的 `drop` 模式恒不跳过（`r < 0.0` 恒 False）⇒ 实测
layer-passes = 8.0 ≠ 设计 7.2（冒烟红在 `run.py` 的 `assert layer_passes_match_design`）。
本测试钉两件事：① 该臂 spec 显式声明 p；② 用该 p 套出的 `SkipBlock` 做一次前向时，
逐层跳过占比 ≈ p（大 batch ⇒ 均值稳落 [0.05, 0.15]）。

**变异必红**：把 spec 的 `"p"` 改成 `0.0`（或让 `brand_layers` 拿不到 p），本文件两条
断言都会失败 —— 第一层靠 `spec["p"] == LAYERDROP_P`，第二层靠 `SkipBlock.p` 与跳过占比。
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run as R  # noqa: E402


def test_layerdrop_arm_spec_declares_p():
    spec = R.ARMS["layerdrop-pt"]
    assert spec["mode"] == "drop"
    assert "p" in spec, "layerdrop-pt 必须显式声明逐 token 跳层概率 p 键"
    assert spec["p"] == R.LAYERDROP_P


def test_layerdrop_branded_blocks_carry_p_and_skip_about_p():
    spec = R.ARMS["layerdrop-pt"]
    model = R.build_model(spec, vocab_size=206, gen_seed=7_007_007)
    blocks = list(model.blocks)
    assert len(blocks) == spec["layers"]
    # 接线断言：每个 SkipBlock 真的拿到 p（缺 p 时会退化成 spec.get("p", 0.0) = 0.0）
    assert all(b.p == R.LAYERDROP_P for b in blocks), [b.p for b in blocks]

    model.eval()
    x = torch.randint(0, 206, (64, 32))
    with torch.no_grad():
        model(x, exit_depths=None)

    means = [b.last_skip_mean for b in blocks]
    for i, m in enumerate(means):
        assert 0.05 <= m <= 0.15, f"layer {i} 跳过占比 {m} 不在 [0.05, 0.15]"
    avg = sum(means) / len(means)
    assert 0.05 <= avg <= 0.15, f"平均跳过占比 {avg} 不在 [0.05, 0.15]"
    layer_passes = len(blocks) - sum(means)
    assert abs(layer_passes - spec["lp_design"]) < 0.5, (
        f"实测 layer-passes {layer_passes} 与设计 {spec['lp_design']} 不符")
