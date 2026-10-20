#!/usr/bin/env python3
"""2681 · 本目录自包含单测（**不改共享 `tests/`**——边界纪律）。

钉住判据①–⑦ 里**能手算**的那几条（复核只需复核文字，不必复算算术）：
  ① 字节账：每位置每键值数 = n_layers × 2 × d_model；fp16 = 2 B/值、int8 块标尺 G=32
     = 1 B/值 + 2 B/32 值 ⇒ win60@int8 = 391,680 B、win64@int8 = 417,792 B，
     `int8/fp16` 之比精确 0.53125，且 `win60` 进得了 `≤393,216 B` 的支配集、`win64` 进不了；
  ② 支配定义：被支配 ⇔ 存在 `total_bytes` 不更多且 `bpc ≤ bpc − band` 的点（带内不算支配）；
  ③ 主判阈值：`1.527595 − 0.014 = 1.513595`（carry 点与判据带都写在 `config.json` 里，
     本文件只验算术，不复述读数）；
  ④ 判据⑤「那一格」的选取：网格里 int8 且 `total_bytes ≤ 393,216` 的最大窗口就是 k=60
     （k=61 的 398,208 B 已越界）——防止事后把 64 当成同字节格。

运行：`python3 -m pytest test_2681_frontier.py -q`（纯 CPU，不取 GPU 锁、不读 `results.json`）。
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run as R                                              # noqa: E402


def _cfg():
    return json.loads(open(os.path.join(HERE, "config.json"), encoding="utf-8").read())


def test_byte_ledger_matches_analytic():
    cfg = _cfg()
    ax = cfg["cost_axis"]
    values = cfg["model"]["n_layers"] * 2 * cfg["model"]["d_model"]
    assert values == 6144
    assert values * 2 == ax["fp16_bytes_per_key"] == 12288
    assert values * (1 + 2 / cfg["quant"]["group"]) == ax["int8_block32_bytes_per_key"] == 6528
    assert ax["int8_block32_bytes_per_key"] * 60 == 391_680
    assert ax["int8_block32_bytes_per_key"] * 64 == 417_792
    assert abs(6528 / 12288 - cfg["criterion"]["bytes_ratio_int8_vs_fp16"]) == 0.0


def test_k60_is_the_largest_int8_window_under_carry_bytes():
    cfg = _cfg()
    ax = cfg["cost_axis"]
    carry = cfg["criterion"]["carry_point"]["total_bytes"]
    assert carry == 393_216
    assert ax["int8_block32_bytes_per_key"] * 60 <= carry
    assert ax["int8_block32_bytes_per_key"] * 61 > carry
    assert ax["int8_block32_bytes_per_key"] * 64 > carry          # win64 不在支配集里
    # 在盘的两条旧 int8 格也必须 cheaper-or-equal
    assert ax["int8_block32_bytes_per_key"] * 32 <= carry
    assert ax["int8_block32_bytes_per_key"] * 128 > carry


def test_domination_needs_more_than_one_band():
    band = 0.014
    pts = [
        {"name": "cheap_bad", "total_bytes": 100, "bpc": 2.000},
        {"name": "cheap_slightly_better", "total_bytes": 100, "bpc": 1.990},
        {"name": "dear", "total_bytes": 200, "bpc": 1.600},
        {"name": "dearer_better", "total_bytes": 300, "bpc": 1.500},
    ]
    dom = R.domination(pts, band)
    # 1.990 比 1.600 好，但字节更多 ⇒ 不构成支配（支配要求字节不更多）
    assert dom["dear"]["dominated_by"] == []
    # 1.990 vs 2.000 只差 0.010 < 0.014 ⇒ 带内不判支配
    assert dom["cheap_bad"]["dominated_by"] == []
    assert dom["cheap_slightly_better"]["dominated_by"] == []

    pts2 = [
        {"name": "cheap_good", "total_bytes": 100, "bpc": 1.500},
        {"name": "dear_bad", "total_bytes": 200, "bpc": 1.520},
    ]
    dom2 = R.domination(pts2, band)
    assert dom2["dear_bad"]["dominated_by"] == ["cheap_good"]
    assert dom2["cheap_good"]["dominated_by"] == []


def test_same_bytes_exact_tie_is_not_domination():
    band = 0.014
    pts = [
        {"name": "a", "total_bytes": 393_216, "bpc": 1.527595},
        {"name": "b", "total_bytes": 393_216, "bpc": 1.527595},
    ]
    dom = R.domination(pts, band)
    assert dom["a"]["dominated_by"] == [] and dom["b"]["dominated_by"] == []


def test_carry_threshold_arithmetic():
    crit = _cfg()["criterion"]
    assert crit["carry_point"]["bpc"] == 1.527595
    assert round(crit["carry_point"]["bpc"] - crit["cross_cell_band"], 6) == 1.513595
    assert crit["domination_threshold_bpc"] == 1.513595
    assert crit["equality_gate_tolerance_bpc"] == 1e-6


def test_every_quant_arm_is_a_window_arm():
    """carry 族本轮不测 int8（`CarryChannelBlock` 不在量化前向支持范围内）⇒
    `quant=True` 的臂必须都是 stock 窗口臂，否则判据② 会拿错块类做量化。"""
    cfg = _cfg()
    for arm in cfg["arms"]:
        if arm["quant"]:
            assert arm["kind"] == "window", arm["name"]
        if arm["kind"] == "carry":
            assert arm["name"] in cfg["carry_specs"]
            assert arm["quant"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
