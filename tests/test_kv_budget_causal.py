"""2631 · 部署口径（值 + 标尺）字节核算的可失败单测。

契约：`axes/R-references/R1-quantization/experiments/2631-kv-budget-causal/`
（`config.json::causal_caliber` + `README.md` + `analysis.md`）。
语义层：`.../2631-kv-budget-causal/budget_caliber.py`。

钉住的四件事：

1. **严格逐值标尺（方案 a）的真实占用**：因果/流式量化每值配一个 fp16 标尺
   ⇒ int8 = 1 + 2 = **3 B/值**、int4 = 0.5 + 2 = 2.5 B/值（fp16 基线 2 B/值）
   ⇒ **int8 / int4 都比 fp16 更占内存**（这就是 `table.md` 值字节列看不见、
   `analysis.md` 要消掉的那处矛盾）。
2. **分块标尺（方案 b）的摊薄与代价**：`G=1` 退回方案 (a)；摊薄随 G 单调；
   违反前缀确定性的距离 = `G − 1` 个位置；staging 是**另算**的加法项、与 T 无关。
3. **量化因子**：在同一拓扑上再叠量化后，方案 (a) 下占用倍率 **恒 > 1**（负收益），
   方案 (b) G=32 下 **< 1**（有收益但要让出前缀确定性）。
4. **与判据 C1 对账**：本模块算出的值字节必须与 `results.json` 里已记录的
   `bytes_per_position` 逐点一致（对不上直接报错）；段模型与 `positions_kept` 一致。

破坏上述语义的变异（把 (a) 的标尺开销当 0、让 (b) 的摊薄分母恒为 1、
让 `mode_of` 不校验 `value_bytes`、把 staging 混进每位置列）会让对应单测抓红。
"""

import copy
import importlib.util
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "axes", "R-references", "R1-quantization",
                   "experiments", "2631-kv-budget-causal")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bc = _load(os.path.join(EXP, "budget_caliber.py"), "budget_caliber_2631")
CFG = json.load(open(os.path.join(EXP, "config.json")))
REC = json.load(open(os.path.join(EXP, "results.json")))
POINTS = {p["arm"]: p for p in REC["points"]}
SPECS = CFG["arms"]
GROUP = bc.DEFAULT_GROUP


def dep(arm, scheme, group=GROUP, point=None, cfg=None):
    return bc.deployment_bytes(point if point is not None else POINTS[arm],
                               SPECS[arm], cfg if cfg is not None else CFG,
                               scheme, group)


# ---------------------------------------------------------------- 1. 方案 (a)

def test_per_value_scale_makes_int8_three_bytes_and_int4_two_and_a_half():
    """严格逐值标尺：int8 = 1 + 2 = 3 B/值、int4 = 0.5 + 2 = 2.5 B/值。"""
    assert bc.scale_bytes_per_value("int8", "a_per_value") == 2.0
    assert bc.bytes_per_value("int8", "a_per_value") == 3.0
    assert bc.bytes_per_value("int4", "a_per_value") == 2.5
    assert bc.bytes_per_value("fp16", "a_per_value") == 2.0


def test_int8_and_int4_occupy_more_than_the_fp16_baseline_under_scheme_a():
    """矛盾消解的核心：方案 (a) 下 int8 = 1.5× fp16、int4 = 1.25× fp16 的每位置占用。"""
    fp16 = dep("fp16", "a_per_value")["bytes_per_position"]
    int8 = dep("int8", "a_per_value")["bytes_per_position"]
    int4 = dep("int4", "a_per_value")["bytes_per_position"]
    assert fp16 == 12288.0
    assert int8 == 18432.0 and int8 / fp16 == 1.5
    assert int4 == 15360.0 and int4 / fp16 == 1.25
    assert int8 > fp16 and int4 > fp16           # 「int8 省一半」在 (a) 下不成立
    # 值字节列（旧横轴）确实是**低估**：它只有 6144 / 3072
    assert POINTS["int8"]["bytes_per_position"] == 6144.0
    assert POINTS["int4"]["bytes_per_position"] == 3072.0


def test_quantization_factor_is_above_one_for_every_quantized_arm_under_scheme_a():
    """同一拓扑上再叠量化：方案 (a) 下每个量化臂的字节倍率都 > 1（内存负收益）。"""
    refs = CFG["quant_increment_ref"]
    checked = 0
    for arm in REC["deployment_caliber"]["quantized_arms"]:
        ref_arm = refs.get(arm) or CFG["family_baseline"][SPECS[arm]["family"]]
        ratio = (dep(arm, "a_per_value")["bytes_per_position"]
                 / dep(ref_arm, "a_per_value")["bytes_per_position"])
        assert ratio > 1.0, f"{arm} 在 (a) 口径下反而更省（{ratio}）"
        checked += 1
    assert checked == 10, f"量化臂数应为 10，实测 {checked}"


def test_quantization_factor_is_below_one_for_every_quantized_arm_under_scheme_b():
    """方案 (b) G=32：量化因子回到 < 1（省），代价是块内前缀确定性（见下一条）。"""
    refs = CFG["quant_increment_ref"]
    for arm in REC["deployment_caliber"]["quantized_arms"]:
        ref_arm = refs.get(arm) or CFG["family_baseline"][SPECS[arm]["family"]]
        ratio = (dep(arm, "b_grouped")["bytes_per_position"]
                 / dep(ref_arm, "b_grouped")["bytes_per_position"])
        assert ratio < 1.0, f"{arm} 在 (b) 口径下不省（{ratio}）"


# ---------------------------------------------------------------- 2. 方案 (b)

def test_b_group_size_one_equals_per_value_scheme():
    """G=1 ⇒ 每个位置自己一个标尺 ⇒ 方案 (b) 逐位等于方案 (a)（两方案同源的自洽性）。"""
    for arm in ("int8", "int4", "gqa_int8", "win32_int8"):
        a = dep(arm, "a_per_value")
        b = dep(arm, "b_grouped", group=1)
        assert b["bytes_per_position"] == pytest.approx(a["bytes_per_position"])
        assert b["scale_bytes_per_value"] == pytest.approx(a["scale_bytes_per_value"])


def test_grouped_scale_amortizes_monotonically_in_group_size():
    """块越大，标尺越便宜（严格单调递减），且永不小于逐值标尺的一半字节。"""
    overheads = [bc.scale_bytes_per_value("int8", "b_grouped", g)
                 for g in bc.GROUP_LADDER]
    assert overheads == sorted(overheads, reverse=True)
    assert overheads[0] == 2.0 / 8 and overheads[-1] == 2.0 / 128
    assert bc.bytes_per_value("int8", "b_grouped", GROUP) == 1.0625
    assert bc.bytes_per_value("int4", "b_grouped", GROUP) == 0.5625
    assert bc.bytes_per_value("int8", "b_grouped", GROUP) < bc.bytes_per_value(
        "int8", "a_per_value")


def test_group_violation_distance_is_group_minus_one():
    """(b) 让出的性质：块边界处过去值受未来影响的**最大距离** = G − 1 个位置。"""
    assert bc.group_violation_distance(1) == 0            # G=1 严格前缀确定
    assert bc.group_violation_distance(GROUP) == 31
    assert bc.group_violation_distance(128) == 127
    with pytest.raises(ValueError):
        bc.group_violation_distance(0)


def test_staging_is_separate_and_independent_of_seq_len():
    """回填 staging 是**另算**的加法项（不进每位置列），且与 T 无关。"""
    st = bc.staging_bytes_per_request(CFG, SPECS["int8"], POINTS["fp16"], GROUP)
    assert st["mean_positions_held"] == pytest.approx(15.5)
    assert st["bytes_per_request"] == pytest.approx(15.5 * 12288)
    bigger = copy.deepcopy(CFG)
    bigger["model"]["seq_len"] = 4096
    st2 = bc.staging_bytes_per_request(bigger, SPECS["int8"], POINTS["fp16"], GROUP)
    assert st2["bytes_per_request"] == pytest.approx(st["bytes_per_request"])
    # 每位置列里不含 staging（否则 T 越大它就越小，不是「每位置」的量）
    assert dep("int8", "b_grouped")["bytes_per_position"] == pytest.approx(6528.0)


def test_short_segments_amortize_less_than_a_full_group():
    """窗口/锚点段比 G 短时摊薄分母更小 ⇒ 标尺更贵（乐观口径的适用边界）。"""
    short = dep("win16_anchor16_int4", "b_grouped")
    assert [s["group_used"] for s in short["per_segment"]] == [16, 16]
    assert short["scale_bytes_per_value"] == pytest.approx(2.0 / 16)
    full = dep("win32_anchor32_int4", "b_grouped")
    assert [s["group_used"] for s in full["per_segment"]] == [32, 32]
    assert full["scale_bytes_per_value"] == pytest.approx(2.0 / 32)


# ---------------------------------------------------------------- 3. 非量化臂

def test_nonquantized_arms_have_no_scale_in_either_scheme():
    """fp16 / 窗口 / 头共享档没有标尺：两套方案都等于值字节（旧横轴不受影响）。"""
    for arm, p in POINTS.items():
        if bc.mode_of(SPECS[arm]) in ("int8", "int4"):
            continue
        a, b = dep(arm, "a_per_value"), dep(arm, "b_grouped")
        assert a["needs_scale"] is False and a["scale_bytes_per_position"] == 0.0
        assert a["bytes_per_position"] == pytest.approx(p["bytes_per_position"])
        assert b["bytes_per_position"] == pytest.approx(p["bytes_per_position"])


def test_mode_of_rejects_value_bytes_mode_mismatch():
    """`value_bytes` 与 `mode` 不一致 = 标签与事实不符 ⇒ 直接报错（沿用 R1 线规则）。"""
    bad = {"kind": "quant", "mode": "int8", "value_bytes": 0.5}
    with pytest.raises(ValueError):
        bc.mode_of(bad)
    assert bc.mode_of({"kind": "quant", "mode": "int8", "value_bytes": 1.0}) == "int8"
    assert bc.mode_of({"kind": "window", "d": 16, "value_bytes": 2.0}) == "fp16"


# ---------------------------------------------------------------- 4. 判据 C1 对账

def test_segments_reproduce_the_recorded_axis_factor_for_every_arm():
    """段模型必须逐臂复现判据 C1 记录的 `positions_kept`（口径漂移即报错）。"""
    for arm, p in POINTS.items():
        segs = bc.kept_segments(SPECS[arm], CFG["model"]["seq_len"])
        assert sum(segs) == p["factor_positions_kept"], arm
        assert sum(segs) / CFG["model"]["seq_len"] == pytest.approx(
            p["factor_positions_kept_over_T"])
        d = dep(arm, "a_per_value")
        assert d["value_bytes_per_position"] == pytest.approx(p["bytes_per_position"])


def test_deployment_bytes_refuses_to_report_when_axis_factor_drifts():
    """单点反例：把记录的 `positions_kept` 改掉 ⇒ 直接报错，而不是悄悄出一个数。"""
    tampered = copy.deepcopy(POINTS["int8"])
    tampered["factor_positions_kept"] = 255
    with pytest.raises(ValueError):
        bc.deployment_bytes(tampered, SPECS["int8"], CFG, "a_per_value")


def test_results_json_carries_both_schemes_for_all_27_points():
    """落盘可回溯：`results.json` 里 27 点都带两套方案的字节与定义块。"""
    assert len(REC["points"]) == 27
    assert set(REC["deployment_caliber"]["schemes"]) == {"a_per_value", "b_grouped"}
    assert REC["deployment_caliber"]["b_violation_distance_positions"] == 31
    for p in REC["points"]:
        db = p["deployment_bytes"]
        assert db["a_per_value"]["bytes_per_position"] > 0
        assert db["b_grouped"]["bytes_per_position"] > 0
        assert db["b_grouped"]["bytes_per_position"] <= db["a_per_value"][
            "bytes_per_position"]


def test_table_md_reports_the_two_deployment_columns():
    """判据 ②：`table.md` 必须真的带两套方案的字节列（含 int8 的两个数）。"""
    md = open(os.path.join(EXP, "table.md")).read()
    assert "部署字节 (a) 逐值标尺" in md and "部署字节 (b) 分块标尺" in md
    row = [ln for ln in md.splitlines() if ln.startswith("| `int8` |")][0]
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "6144"        # 值字节（旧横轴）
    assert cells[3] == "18432"       # (a) 逐值标尺 = 1B + 2B 标尺
    assert cells[4] == "6528"        # (b) 分块标尺 G=32 = 1B + 2/32B
    assert "分析" not in cells[4]
