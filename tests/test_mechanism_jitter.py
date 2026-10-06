"""契约 2627（G5 机制定论抖动补点，ticket 44 X1）的单因素护栏单测。

对象 = 本实验的落盘产物/判据逻辑：
  ① `configs13m/repr01.json` 与 2626（V2）的 `repr01.json` **逐字节相同**——单因素 = 只**重跑**
     一臂（不改配置、不改 seed、不改步数）；
  ② `config.json` 的抖动判决阈值与 README §4 预登记一致（极差 ≥0.0144 降级 / <0.005 升格）；
  ③ `jitter.py` 的净效应算术与三态判决（含边界），可失败性由 `mutate.py` 变异验证
     （M4 语义猴补丁抓红 + 逐字节还原）。
"""

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

HERE = os.path.join(ROOT, "axes", "G-policy", "G5-parallel-lines",
                    "experiments", "2627-mechanism-jitter")
sys.path.insert(0, HERE)

from jitter import (DOWNGRADE_GE, EFFECT_V2, UPGRADE_LT,  # noqa: E402
                    jitter_verdict, mechanism_net)

V2 = os.path.join(ROOT, "axes", "G-policy", "G5-parallel-lines",
                  "experiments", "2626-mechanism-convergence")


def _cfg() -> dict:
    return json.load(open(os.path.join(HERE, "config.json")))


def test_repr01_config_byte_identical_to_v2():
    """单因素：本轮 repr01 配置与 2626 逐字节相同（唯一变量 = 多跑了一遍）。"""
    a = open(os.path.join(V2, "configs13m", "repr01.json"), "rb").read()
    b = open(os.path.join(HERE, "configs13m", "repr01.json"), "rb").read()
    assert a == b, "2627 的 repr01 配置必须与 2626 逐字节相同"


def test_repr01_config_pins_single_factor():
    c = json.load(open(os.path.join(HERE, "configs13m", "repr01.json")))
    assert c["path"]["p_skip"] == 0.1
    assert c["path"]["w_repr_decorr"] == 0.1
    assert c["path"]["w_diversity"] == 0.0
    assert c["train"]["steps"] == 24000
    assert c["train"]["seed"] == 0
    assert c["train"]["batch_size"] == 32


def test_config_thresholds_match_preregistered_rule():
    cfg = _cfg()
    assert cfg["effect_v2"] == 0.0144
    assert cfg["jitter_rule"]["downgrade_if_range_ge"] == 0.0144
    assert cfg["jitter_rule"]["upgrade_if_range_lt"] == 0.005
    assert cfg["variance_floor"] == 0.014
    assert EFFECT_V2 == 0.0144
    assert DOWNGRADE_GE == 0.0144
    assert UPGRADE_LT == 0.005


def test_arms_and_sources():
    cfg = _cfg()
    s = cfg["scales"]["13m"]
    assert s["arms"] == ["l1_A", "repr01_A", "repr01_B"]
    assert s["mechanism_baseline"] == "l1_A"
    assert s["repr_pair"] == ["repr01_A", "repr01_B"]
    assert s["det_arm"] == "repr01_B"
    assert cfg["arms"]["13m/l1_A"]["source"] == "2626"
    assert cfg["arms"]["13m/repr01_A"]["source"] == "2626"
    assert cfg["arms"]["13m/repr01_B"]["source"] == "2627"
    assert "2626-mechanism-convergence" in cfg["arms"]["13m/repr01_A"]["config_file"]
    assert "2627-mechanism-jitter" in cfg["arms"]["13m/repr01_B"]["config_file"]


def test_mechanism_net_arithmetic():
    assert abs(mechanism_net(1.4089, 1.3945) - 0.0144) < 1e-12
    assert abs(mechanism_net(1.0, 1.1) - (-0.1)) < 1e-12
    assert mechanism_net(2.0, 2.0) == 0.0


def test_jitter_verdict_three_states_and_boundaries():
    # 极差 < 0.005 ⇒ 升格稳健
    assert jitter_verdict(0.0)["state"] == "upgraded_to_robust"
    assert jitter_verdict(0.0049)["state"] == "upgraded_to_robust"
    # 边界：0.005 已不算「≪ 效应」⇒ 仍贴门槛
    assert jitter_verdict(0.005)["state"] == "on_threshold"
    assert jitter_verdict(0.01)["state"] == "on_threshold"
    assert jitter_verdict(0.0143)["state"] == "on_threshold"
    # 边界：极差 ≥ 效应本身（0.0144）⇒ 降级不可排序
    assert jitter_verdict(0.0144)["state"] == "downgraded_to_unorderable"
    assert jitter_verdict(0.02)["state"] == "downgraded_to_unorderable"


def test_jitter_verdict_exposes_thresholds_and_ratio():
    v = jitter_verdict(0.0072)     # = 0.0144 / 2
    assert v["downgrade_ge"] == 0.0144
    assert v["upgrade_lt"] == 0.005
    assert v["effect"] == 0.0144
    assert abs(v["range_over_effect"] - 0.5) < 1e-12
