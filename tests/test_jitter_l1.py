"""契约 2628（G5 机制定论抖动**补全**，l1 臂，ticket 44-X1 后续）的单因素护栏单测。

对象 = 本实验的落盘产物/判据逻辑：
  ① `configs13m/l1.json` 与 2626（V2）的 `l1.json` **逐字节相同**——单因素 = 只**重跑**
     `l1` 一臂（不改配置、不改 seed、不改步数）；
  ② `config.json` 的抖动判决阈值与 README §4 预登记一致（合并极差 ≥0.0144 降级 / <0.005 维持）；
  ③ `jitter_l1.py` 的净效应算术、两臂合并极差（取大）与三态判决（含边界），可失败性由
     `mutate.py` 变异验证（M5 语义猴补丁抓红 + 逐字节还原）。
"""

import ast
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

HERE = os.path.join(ROOT, "axes", "G-policy", "G5-parallel-lines",
                    "experiments", "2628-jitter-l1")
sys.path.insert(0, HERE)

from jitter_l1 import (DOWNGRADE_GE, EFFECT_V2, REPR01_JITTER_2627,  # noqa: E402
                       UPGRADE_LT, arm_jitter, combined_range, combos_range,
                       jitter_verdict, mechanism_net, worst_case_additive_range)

V2 = os.path.join(ROOT, "axes", "G-policy", "G5-parallel-lines",
                  "experiments", "2626-mechanism-convergence")


def _cfg() -> dict:
    return json.load(open(os.path.join(HERE, "config.json")))


def test_l1_config_byte_identical_to_v2():
    """单因素：本轮 l1 配置与 2626 逐字节相同（唯一变量 = 多跑了一遍）。"""
    a = open(os.path.join(V2, "configs13m", "l1.json"), "rb").read()
    b = open(os.path.join(HERE, "configs13m", "l1.json"), "rb").read()
    assert a == b, "2628 的 l1 配置必须与 2626 逐字节相同"


def test_l1_config_pins_single_factor():
    c = json.load(open(os.path.join(HERE, "configs13m", "l1.json")))
    assert c["path"]["p_skip"] == 0.1
    assert c["path"]["w_repr_decorr"] == 0.0      # l1 臂 = 路径机制、无去相关（对照基准）
    assert c["path"]["w_diversity"] == 0.0
    assert c["train"]["steps"] == 24000
    assert c["train"]["seed"] == 0
    assert c["train"]["batch_size"] == 32


def test_config_thresholds_match_preregistered_rule():
    cfg = _cfg()
    assert cfg["effect_v2"] == 0.0144
    assert cfg["repr01_jitter_2627"] == 0.0035
    assert cfg["jitter_rule"]["downgrade_if_range_ge"] == 0.0144
    assert cfg["jitter_rule"]["upgrade_if_range_lt"] == 0.005
    assert cfg["variance_floor"] == 0.014
    assert EFFECT_V2 == 0.0144
    assert DOWNGRADE_GE == 0.0144
    assert UPGRADE_LT == 0.005
    assert REPR01_JITTER_2627 == 0.0035


def test_arms_and_sources():
    cfg = _cfg()
    s = cfg["scales"]["13m"]
    assert s["arms"] == ["l1_A", "l1_B", "repr01_A", "repr01_B"]
    assert s["mechanism_baseline"] == "l1_A"
    assert s["l1_pair"] == ["l1_A", "l1_B"]
    assert s["repr_pair"] == ["repr01_A", "repr01_B"]
    assert s["det_arm"] == "l1_B"
    assert cfg["arms"]["13m/l1_A"]["source"] == "2626"
    assert cfg["arms"]["13m/l1_B"]["source"] == "2628"
    assert cfg["arms"]["13m/repr01_A"]["source"] == "2626"
    assert cfg["arms"]["13m/repr01_B"]["source"] == "2627"
    assert "2626-mechanism-convergence" in cfg["arms"]["13m/l1_A"]["config_file"]
    assert "2628-jitter-l1" in cfg["arms"]["13m/l1_B"]["config_file"]
    assert "2627-mechanism-jitter" in cfg["arms"]["13m/repr01_B"]["config_file"]


def test_mechanism_net_and_arm_jitter():
    assert abs(mechanism_net(1.4089, 1.3945) - 0.0144) < 1e-12
    assert abs(mechanism_net(1.0, 1.1) - (-0.1)) < 1e-12
    assert mechanism_net(2.0, 2.0) == 0.0
    assert abs(arm_jitter(1.4089, 1.4089) - 0.0) < 1e-12
    assert abs(arm_jitter(1.4089, 1.3910) - 0.0179) < 1e-12
    assert abs(arm_jitter(1.3910, 1.4089) - 0.0179) < 1e-12   # 极差对称


def test_combined_range_is_max_of_two_arms():
    # README §4 预登记：合并极差 = 两臂各自极差的较大者
    assert combined_range(0.002, 0.0035) == 0.0035
    assert combined_range(0.008, 0.0035) == 0.008
    assert combined_range(0.0035, 0.0035) == 0.0035
    assert combined_range(0.0, 0.0) == 0.0


def test_secondary_lines_additive_and_combos():
    # 并列披露（非 primary）：最坏情形加和、四组合极差恒等于两臂之和
    assert abs(worst_case_additive_range(0.002, 0.0035) - 0.0055) < 1e-12
    combos = [1.4089 - 1.3945, 1.4089 - 1.3910, 1.4020 - 1.3945, 1.4020 - 1.3910]
    assert abs(combos_range(combos) - (abs(1.4020 - 1.4089) + abs(1.3910 - 1.3945))) < 1e-12
    assert combos_range([]) == 0.0


def test_jitter_verdict_three_states_and_boundaries():
    # 合并极差 < 0.005 ⇒ 维持稳健（并升格口径为两臂完整口径）
    assert jitter_verdict(0.0)["state"] == "maintained_robust"
    assert jitter_verdict(0.0049)["state"] == "maintained_robust"
    # 边界：0.005 已不算「≪ 效应」⇒ 仍贴门槛
    assert jitter_verdict(0.005)["state"] == "on_threshold"
    assert jitter_verdict(0.01)["state"] == "on_threshold"
    assert jitter_verdict(0.0143)["state"] == "on_threshold"
    # 边界：合并极差 ≥ 效应本身（0.0144）⇒ 降级不可排序
    assert jitter_verdict(0.0144)["state"] == "downgraded_to_unorderable"
    assert jitter_verdict(0.02)["state"] == "downgraded_to_unorderable"


def test_jitter_verdict_exposes_thresholds_and_ratio():
    v = jitter_verdict(0.0072)     # = 0.0144 / 2
    assert v["downgrade_ge"] == 0.0144
    assert v["upgrade_lt"] == 0.005
    assert v["effect"] == 0.0144
    assert abs(v["range_over_effect"] - 0.5) < 1e-12


# ------------------------------------------------- F_JITTER：分量 note 与数值一致

def test_component_note_formula_matches_recorded_components():
    """分量 note 必须写明 `net = (repr01_gain − l1_gain) − clean_tax`，且该式用
    `results_13m.json` 的 K 曲线读数重算后等于 `effect_v2`（净效应）。

    修复前 `run.py` 与 `results_13m.json` 的 note 只写「集成增益 − clean 税（相对 l1）」，
    漏掉「− l1_gain」⇒ 本测试对旧文本应红（复核 B8）。"""
    R = json.load(open(os.path.join(HERE, "results_13m.json")))
    src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    note_run = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                    if isinstance(node, ast.Assign)
                    and getattr(node.targets[0], "id", None) == "COMPONENT_NOTE")
    note_json = R["jitter"]["component_decomposition"]["note"]
    for text, where in ((note_json, "results_13m.json"), (note_run, "run.py")):
        assert "net = (repr01_gain − l1_gain) − clean_tax" in text, \
            f"{where} 的分量 note 未写明正确公式（应为 net = (repr01_gain − l1_gain) − clean_tax）"
    assert note_json == note_run, "note 在 run.py 与 results_13m.json 两处不一致"

    kc = R["k_curve"]
    gain = lambda arm: kc[arm]["1"]["bpc"] - kc[arm]["8"]["bpc"]           # noqa: E731
    clean_tax = lambda arm: kc[arm]["1"]["bpc"] - kc["l1_A"]["1"]["bpc"]   # noqa: E731
    net = round((gain("repr01_A") - gain("l1_A")) - clean_tax("repr01_A"), 6)
    assert net == R["jitter"]["effect_v2"] == R["jitter"]["net_combos"]["l1_A__repr01_A"], \
        f"按 note 的公式重算 net={net}，与 effect_v2={R['jitter']['effect_v2']} 不符"
