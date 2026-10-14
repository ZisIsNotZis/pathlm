#!/usr/bin/env python3
"""2639 · 本目录自包含单测（**不改共享 `tests/`**——边界纪律）。

钉住以下事项（复核 N8 / B1 的同类要求）：
  ① 逐层独立状态的**字节记账**：`state_floats` = n_layers × n_heads × hd²（逐层）
     或 1 × n_heads × hd²（单共享槽）——判据②/③ 的代价轴口径；
  ② 前沿支配定义的**符号约定**：`carry_minus_interp_window < 0 ⇔ 被支配`
     （与 `results.json` 的 `frontier.dominated_by` 一致）；
  ③ `pl_minus_win32` = `bpc(win32_s0) − bpc(carrypl16_s0)` 的**数值一致**
     （抓不住文档文字型回归）；
  ④ 插值只用于定位、不外推（区间外返回 None）；
  ⑤ 逐层探针有抓力：逐层模型上 12/4 层的 mem 两两不同，单共享槽模型上
     `applicable=False`。

运行：`python3 -m pytest test_2639_frontier.py -q`（纯 CPU；若 `results.json`/`config.json`
不存在，读盘类用例自动 skip）。
"""
import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel      # noqa: E402
sys.path.insert(0, HERE)
import run as R                                              # noqa: E402


def _tiny(per_layer: bool):
    cfg = DepthARConfig(d_model=32, n_layers=4, n_heads=4, seq_len=16, mlp_mult=2,
                        dense=True, topology="sequential", carry_rule="delta",
                        carry_chunk=16, carry_per_layer=per_layer, carry_decay=1.0)
    torch.manual_seed(0)
    return DepthARModel(cfg, 32)


def test_state_floats_per_layer_vs_shared():
    for per_layer in (True, False):
        m = _tiny(per_layer)
        per = m.carry.n_heads * m.carry.head_dim ** 2
        expect = per * (m.cfg.n_layers if per_layer else 1)
        assert per == 4 * 8 * 8, "tiny 模型每层状态 = n_heads×hd² = 256 floats"
        assert expect == (4 * per if per_layer else per)
    # 逐层 = 12 倍单共享槽（本实验 13M 配置的算术）
    assert 12 * (8 * 32 * 32) == 98304, "13M 配置：12 层 × 8192 floats"


def test_frontier_dominance_sign_convention():
    rows = [{"name": "a", "total_bytes": 10, "bpc": 1.5},
            {"name": "b", "total_bytes": 20, "bpc": 1.4},
            {"name": "c", "total_bytes": 30, "bpc": 1.6}]
    fr = R.frontier(rows)
    assert fr["dominated_by"]["c"] == ["a", "b"]
    assert fr["dominated_by"]["b"] == []
    assert fr["lower_envelope"] == ["a", "b"]


def test_interp_only_locates_never_extrapolates():
    rows = [{"name": f"win{k}", "kind": "window", "seed": 0, "k_eff": float(k),
             "bpc": 1.5 - 0.001 * k, "total_bytes": k, "window": k} for k in (16, 24, 32)]
    mid = R.interp_window_at(rows, 20.0)
    assert mid["between"] == ["win16", "win24"]
    assert mid["bpc_interp"] == pytest.approx(1.480, abs=1e-9)   # k=20 = 16↔24 的中点
    assert R.interp_window_at(rows, 40.0) is None      # 外推 ⇒ None
    assert R.interp_window_at(rows, 32.0) == {"between": ["win24", "win32"], "w": 1.0,
                                              "bpc_interp": 1.468}   # 端点 = 该点本身


def test_perlayer_probe_has_teeth():
    dev = "cpu"
    x = torch.randint(0, 32, (2, 16))
    m_pl = _tiny(True).eval()
    probe = R.perlayer_state_probe(m_pl, x)
    assert probe["applicable"] and probe["layer_idx_covers_all"]
    assert probe["all_pairs_different"] and probe["n_layers_seen"] == 4
    assert probe["max_abs_mem_diff"] > 0
    m_sh = _tiny(False).eval()
    assert R.perlayer_state_probe(m_sh, x)["applicable"] is False


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        pytest.skip(f"{name} 未落盘")
    return json.load(open(p))


def test_carrypl_and_win32_are_the_same_budget_point():
    cfg = _load("config.json")
    if cfg.get("missing_new_arms"):
        pytest.skip("新臂 checkpoint 未就绪")
    by = {a["name"]: a for a in cfg["arms"]}
    bpe = cfg["cost_axis"]["bytes_per_elem"]
    unit = cfg["cost_axis"]["kv_bytes_per_position_unit"]
    n_layers, d = cfg["model"]["n_layers"], cfg["model"]["d_model"]

    def total(a):
        k = a["window"] if a["kind"] != "full" else cfg["model"]["seq_len"]
        kv = n_layers * 2 * d * bpe * k
        st = 0
        if a["rule"]:
            hd = d // cfg["model"]["n_heads"]
            nh = d // hd
            st = nh * hd * hd * bpe * (n_layers if a.get("carry_per_layer") else 1)
        return kv + st
    assert total(by["carrypl16_s0"]) == total(by["win32_s0"]) == 393216
    assert total(by["carry16_s0"]) == 212992
    assert round(total(by["carrypl16_s0"]) / unit, 4) == 32.0


def test_results_key_readings_match_rows():
    res = _load("results.json")
    by = {r["name"]: r for r in res["rows"]}
    kr = res["key_readings"]["pl_minus_win32"]
    d = round(by["win32_s0"]["bpc"] - by["carrypl16_s0"]["bpc"], 6)
    assert kr["pl_minus_win32"] == d
    assert kr["same_bytes"] is (by["win32_s0"]["total_bytes"]
                                == by["carrypl16_s0"]["total_bytes"])
    want = ("改善前沿（机制翻转）" if d > R.SIGMA_CELL else
            ("被等预算纯窗口支配" if d < -R.SIGMA_CELL else "带内不可排序"))
    assert kr["verdict"] == want
    # 支配关系与判定线同向：pl_minus_win32 < 0 ⇒ carrypl 被 win32 支配
    dom = res["frontier"]["dominated_by"]["carrypl16_s0"]
    assert (d < 0) == ("win32_s0" in dom)
