"""Δ 方向是机器可读字段（`harness/research-schema.md` §2.3）——五件事必须能红。

1. **同一文件两种方向合法且逐字段校验**：`delta_directions` 是**字段级映射**，
   逐臂质量腿与代价腿可以各用一套方向；把某个字段的方向写成与它所在判据块相反，必须 fail。
2. **`delta` 字段漏声明必须失败**：映射里少一个 `delta*` 读数就 fail（不是静默跳过）。
3. **未知枚举必须失败**：写个人话进去、写成旧的文件级单值、缺 `undeclared` 的原因，
   都必须 fail。
4. **`undeclared` 被统计而不是静默通过**：合法通过，但字段进 `UNDECLARED_FIELDS`，
   并由 `delta_direction_summary()` 计入打印。
5. **`cost_rows` 仍是必填单值**：方向与 `metric` 的短语/算式矛盾 ⇒ fail；手改生成物仍被抓红。
"""

import json
import os
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_tree  # noqa: E402

COST_TABLE = os.path.join(ROOT, "atlas", "cost-table.md")
CMP = "comparator_minus_arm"
ARM = "arm_minus_comparator"


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


def _check(data):
    return check_tree.check_results_delta_directions("axes/x/results.json", data)


@pytest.fixture(autouse=True)
def _clean_counters():
    check_tree.UNDECLARED_FIELDS.clear()
    check_tree.UNJUDGED.clear()
    yield
    check_tree.UNDECLARED_FIELDS.clear()
    check_tree.UNJUDGED.clear()


def test_mixed_directions_in_one_file_are_legal_and_checked_per_field():
    """① 质量腿与代价腿方向相反是合法的；逐字段校验仍生效。"""
    data = {
        "criteria": {
            "criterion_1_quality": {"delta_convention": "Δ = bpc_对照 − bpc_实验（正 = 实验臂更好）",
                                    "delta_bpc": -0.012},
            "criterion_2_cost": {"delta_convention": "Δ = 层通路 − 对照（正 = 更慢）",
                                 "delta_layer_passes": 0.31},
        },
        "delta_directions": {"delta_bpc": CMP, "delta_layer_passes": ARM},
    }
    assert _check(data) == []

    # 把质量腿的方向写成代价腿那套 ⇒ 必须 fail（同文件两种方向不等于随便写）
    bad = json.loads(json.dumps(data))
    bad["delta_directions"]["delta_bpc"] = ARM
    errs = _check(bad)
    assert any("delta_bpc" in e and "矛盾" in e for e in errs), errs

    # 反过来把代价腿写成质量腿那套，也必须 fail
    bad2 = json.loads(json.dumps(data))
    bad2["delta_directions"]["delta_layer_passes"] = CMP
    assert any("delta_layer_passes" in e and "矛盾" in e for e in _check(bad2))


def test_every_delta_field_must_be_declared():
    """② 映射漏一个 `delta*` 读数就 fail；路径键优先、键名键兜底。"""
    data = {"results": [{"arm": "a", "delta_bpc": -0.01},
                        {"arm": "b", "delta_bpc": 0.02}],
            "delta_directions": {"delta_bpc": CMP}}
    assert _check(data) == []

    missing = {"results": [{"arm": "a", "delta_bpc": -0.01, "delta_pp": 1.5}],
               "delta_directions": {"delta_bpc": CMP}}
    errs = _check(missing)
    assert any("delta_pp" in e and "没有声明" in e for e in errs), errs

    # 整个映射都不写：报了 Δ 读数就不许沉默
    errs = _check({"results": [{"delta_bpc": -0.01}]})
    assert any("缺 delta_directions" in e for e in errs), errs

    # 声明了对不上任何字段的键 = 写错名字，等于没声明
    errs = _check({"delta_bpc": 0.01, "delta_directions": {"delta_bpcc": CMP}})
    assert any("对不上任何 Δ 字段" in e for e in errs), errs


def test_unknown_enum_and_legacy_single_value_fail():
    """③ 未知枚举、旧的文件级单值、缺原因都 fail。"""
    errs = _check({"delta_bpc": 0.01,
                   "delta_directions": {"delta_bpc": "正 = 更差"}})
    assert any("非法" in e for e in errs), errs

    errs = _check({"delta_bpc": 0.01,
                   "delta_directions": {"delta_bpc": ARM},
                   "delta_direction": CMP})
    assert any("单值已废弃" in e for e in errs), errs

    errs = _check({"delta_bpc": 0.01,
                   "delta_directions": {"delta_bpc": "undeclared"}})
    assert any("delta_direction_note" in e for e in errs), errs


def test_undeclared_is_counted_not_silently_passed():
    """④ `undeclared` 合法但必须被统计（缺口看得见，不是假装没有）。"""
    data = {"capability_probe": {"param_delta": 4240},
            "delta_directions": {"param_delta": "undeclared"},
            "delta_direction_note": "参数量差，没有质量/代价的比较方向。"}
    assert _check(data) == []
    assert ("axes/x/results.json", "capability_probe.param_delta") in check_tree.UNDECLARED_FIELDS


def test_real_mixed_file_declares_both_legs():
    """真仓库里那类「同一文件两套约定」的文件：逐字段声明后必须通过，且两腿方向不同。"""
    rel = ("axes/5-quant/5.1-kv-quant/experiments/2620-kv-budget-gaps/results.json")
    data = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
    mapping = data["delta_directions"]
    assert mapping["delta_schema"] == CMP           # Δ_schema = 对照 − 实验
    assert mapping["quant_increment_recall_delta"] == ARM   # 臂 − 参照臂
    assert mapping["max_abs_delta_bpc_threshold"] == "undeclared"  # 幅度门限，没有比较方向
    assert check_tree.check_results_delta_directions(rel, data) == []


def test_cost_row_direction_still_required_single_value():
    """⑤ `cost_rows` 的方向是一条事实：仍是必填单值，且与 metric 自洽。"""
    assert _errs(_row()) == []
    errs = _errs(_row(delta_direction="正 = 更差"))
    assert any("非法" in e for e in errs), errs
    errs = _errs(_row(delta_direction="undeclared"))
    assert any("非法" in e for e in errs), errs
    errs = _errs(_row(delta_direction=None))
    assert any("缺 delta_direction" in e for e in errs), errs
    bad = _errs(_row(delta_direction="comparator_minus_arm"))
    assert any("矛盾" in e for e in bad), bad


def test_hand_edited_direction_column_is_caught(tmp_path, monkeypatch):
    """手改生成物的「Δ 方向」列仍被抓红：`--check` 必须非零退出。"""
    import build_cost_table

    hand = tmp_path / "cost-table.md"
    shutil.copyfile(build_cost_table.OUT, hand)
    text = hand.read_text(encoding="utf-8")
    assert "正 = 更差" in text, "生成物里必须有「Δ 方向」列"
    hand.write_text(text.replace("正 = 更差", "正 = 改善", 1), encoding="utf-8")
    monkeypatch.setattr(build_cost_table, "OUT", hand)
    assert build_cost_table.main(["--check"]) == 1
