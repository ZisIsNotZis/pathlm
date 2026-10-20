"""Δ 方向是机器可读字段（`docs/research-schema.md` §2.3）——四件事必须能红。

1. **未知枚举值必须报错**：`delta_direction` 写个人话上去，`check_tree` 必须 fail。
2. **手改生成物仍被抓红**：把 `docs/cost-table.md` 复制一份、改掉一行的「Δ 方向」，
   `build_cost_table.py --check` 必须非零退出（真仓库文件不动）。
3. **方向与符号自洽**：`metric` 说「实验 − 对照；正 = 更差」而 `delta_direction` 写成
   `comparator_minus_arm` 的行，必须 fail；反过来写对就必须过。
4. **判据串与 `delta_direction` 一致**：`results.json` 的 `criterion` 串声称一个方向、
   顶层字段声称另一个方向时必须 fail（`2653` 这种带 `criterion` 串的文件即可用）。
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_tree  # noqa: E402

COST_TABLE = os.path.join(ROOT, "docs", "cost-table.md")
BUILD = os.path.join(ROOT, "tools", "build_cost_table.py")


def _row(**over):
    row = {"law": 1, "change": "层间线性加权和", "touches_trunk": True,
           "metric": "clean bpc Δ（实验 − 对照；正 = 更差）", "delta": "+0.0458",
           "comparator": "stock 串行子层", "scale": "13M / 12k 步", "boundary": "只在本尺度",
           "source": "axes", "delta_direction": "arm_minus_comparator"}
    row.update(over)
    return row


def _errs(row):
    return check_tree.check_laws_and_costs({"laws": [1], "cost_rows": [row]},
                                           "axes/x/project.yaml")


def test_unknown_enum_value_fails():
    """① 未知枚举值必须报错（枚举只有两个，写别的都是非法）。"""
    errs = _errs(_row(delta_direction="正 = 更差"))
    assert any("delta_direction" in e and "非法" in e for e in errs), errs
    errs = _errs(_row(delta_direction=None))
    assert any("缺 delta_direction" in e for e in errs), errs


def test_direction_and_sign_must_agree():
    """③ 声明的方向与 `metric` 的方向短语/算式矛盾 ⇒ fail；一致 ⇒ 过。"""
    ok = _errs(_row())
    assert ok == [], ok

    # metric 的短语说「正 = 更差」，字段却声称规范方向
    bad = _errs(_row(delta_direction="comparator_minus_arm"))
    assert any("矛盾" in e for e in bad), bad

    # 短语与字段对上了，但算式次序写反了（数字没变，只是标签对不上）⇒ 也要抓
    bad2 = _errs(_row(metric="clean bpc Δ（对照 − 实验；正 = 更差）",
                      delta_direction="arm_minus_comparator"))
    assert any("算式" in e and "矛盾" in e for e in bad2), bad2

    # 「负 = 更好」等价于「正 = 更差」：不得因为符号词不同就判错
    ok2 = _errs(_row(metric="clean bpc Δ（实验 − 对照；负 = 更好）",
                     delta_direction="arm_minus_comparator"))
    assert ok2 == [], ok2


def test_hand_edited_direction_column_is_caught(tmp_path, monkeypatch):
    """② 手改生成物的「Δ 方向」列仍被抓红：`--check` 必须非零退出。"""
    import build_cost_table

    hand = tmp_path / "cost-table.md"
    shutil.copyfile(build_cost_table.OUT, hand)
    text = hand.read_text(encoding="utf-8")
    assert "正 = 更差" in text, "生成物里必须有「Δ 方向」列"
    hand.write_text(text.replace("正 = 更差", "正 = 改善", 1), encoding="utf-8")
    monkeypatch.setattr(build_cost_table, "OUT", hand)
    assert build_cost_table.main(["--check"]) == 1


def test_results_criterion_direction_must_match():
    """④ `results.json` 的判据串方向与 `delta_direction` 不一致 ⇒ fail。"""
    good = {"delta_direction": "comparator_minus_arm",
            "criterion": "P1 Δ = bpc_stock − bpc_该臂（正 = 该臂更好）；方差带 0.014"}
    assert check_tree.check_results_delta_direction("axes/x/results.json", good) == []

    bad = {"delta_direction": "arm_minus_comparator",
           "criterion": "P1 Δ = bpc_stock − bpc_该臂（正 = 该臂更好）；方差带 0.014"}
    errs = check_tree.check_results_delta_direction("axes/x/results.json", bad)
    assert any("矛盾" in e for e in errs), errs

    missing = {"criterion": "P1 Δ = bpc_stock − bpc_该臂（正 = 该臂更好）"}
    errs = check_tree.check_results_delta_direction("axes/x/results.json", missing)
    assert any("缺 delta_direction" in e for e in errs), errs


def test_real_2653_criterion_declares_registered_direction():
    """真仓库里的 `2653` 判据串声明的方向，必须与它 `results.json` 里写的字段一致。"""
    path = os.path.join(ROOT, "axes/2-depth/2.2.2-attention-residual/experiments/"
                              "2653-interlayer-attn-13m/results.json")
    data = json.load(open(path, encoding="utf-8"))
    assert data["delta_direction"] == "comparator_minus_arm"
    assert check_tree.check_results_delta_direction(
        os.path.relpath(path, ROOT), data) == []
