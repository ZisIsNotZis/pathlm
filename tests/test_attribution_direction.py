"""2683 的契约外归因读数（`attribution.json`）的**方向**可失败单测。

P1-1（复核）指出的缺陷：`delta_direction_note` 把 `band24_on_block_ckpt` 的方向写反了——Δ 记的是
`bpc_own − bpc_反事实`（`comparator_minus_arm`，正 = 反事实更好），落地值却是 −0.398799 / −0.939319
（反事实更差），而注文写着「负且越带 ⇒ 只换推理掩码就能补回」。只读 `attribution.json` 的读者会
得到与 `analysis.md` 相反的解释。

本文件钉住修法：解释不再手写，由 `attribution.py::interpretation()` 按 Δ 的符号查表生成，并在落盘前
校验「声明（跑之前写死）与实测相符」，不符即 `SystemExit`。四件事必须能红：

1. 符号 ↔ 方向（`observed_effect`）：正 = 反事实更好、负 = 更差、带内 = 无区分力；
2. **把声明写反** ⇒ `interpretation()` 拒绝落盘（这条就是复核要的那条断言）；
3. 同一 Δ 换符号 ⇒ 人话随之换（不是常量函数，「补回」只出现在正方向）；
4. **落盘的** `attribution.json` 每条读数都带与自身 Δ 一致的 `interpretation`，且顶层注不写任何
   单条读数的结论（注文与数值相反这件事在文件里无处可写）。

全程 CPU、毫秒级，不取 GPU 锁。

Run: python3 -m pytest tests/test_attribution_direction.py -q
"""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXP = os.path.join(ROOT, "axes", "1-time", "1.2.2-block-sparse",
                   "experiments", "2683-block-stripe-vs-band-13m")


def _load_attribution():
    spec = importlib.util.spec_from_file_location(
        "attribution_2683_test", os.path.join(EXP, "attribution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


attr = _load_attribution()
BAND = attr.BAND
READINGS = ("far_block_off", "band24_on_block_ckpt")


def test_observed_effect_matches_delta_direction():
    """`comparator_minus_arm`：正 = 反事实更好。"""
    assert attr.DELTA_DIRECTION == "comparator_minus_arm"
    assert attr.observed_effect(+2 * BAND) == "better"
    assert attr.observed_effect(+BAND) == "better"
    assert attr.observed_effect(BAND / 2) == "in_band"
    assert attr.observed_effect(0.0) == "in_band"
    assert attr.observed_effect(-BAND) == "worse"
    assert attr.observed_effect(-2 * BAND) == "worse"


def test_reversed_declaration_is_refused():
    """把任一条读数的方向声明写反 ⇒ 落盘前必须拒绝（复核 P1-1 要的那条断言）。"""
    for name in READINGS:
        declared = attr.DECLARED_EFFECT[name]
        wrong = "better" if declared == "worse" else "worse"
        step = 2 * BAND if wrong == "better" else -2 * BAND      # 实测方向 = wrong
        with pytest.raises(SystemExit):
            attr.interpretation(name, step)                      # 声明仍是 declared ⇒ 不符
        saved = attr.DECLARED_EFFECT[name]
        attr.DECLARED_EFFECT[name] = wrong                       # 反过来：声明 = 实测 ⇒ 通过
        try:
            got = attr.interpretation(name, step)
        finally:
            attr.DECLARED_EFFECT[name] = saved
        assert got["observed_effect"] == wrong


def test_text_follows_sign_not_a_constant():
    """`band24_on_block_ckpt`：负（反事实更差）= 崩；正 = 补回。旧注文把两者写反了。"""
    neg = attr.effect_text("band24_on_block_ckpt", -0.398799)
    pos = attr.effect_text("band24_on_block_ckpt", +0.398799)
    assert "崩" in neg and "分布外" in neg
    assert "补回" in pos and "补回" not in neg
    assert attr.observed_effect(-0.398799) == "worse"
    assert attr.observed_effect(+0.398799) == "better"
    assert attr.in_band(-0.398799).startswith("越带（负")


def test_landed_readings_carry_consistent_interpretation():
    """落盘文件自洽：每条读数的 `interpretation` 与自身 Δ 的符号一致，顶层注不写单条结论。"""
    data = json.load(open(os.path.join(EXP, "attribution.json"), encoding="utf-8"))
    note = data["delta_direction_note"]
    assert "comparator_minus_arm" in note and "interpretation" in note
    assert "补回" not in note                                     # 旧缺陷的原句就在这里
    assert data["delta_directions"]["delta_bpc"] == attr.DELTA_DIRECTION
    for seed, block in data["per_seed"].items():
        assert block["own_matches_judged_reading"] is True, seed
        for name in READINGS:
            reading = block[name]
            want = attr.observed_effect(reading["delta_bpc"])
            assert reading["interpretation"]["observed_effect"] == want, (seed, name)
            assert reading["interpretation"]["declared_effect"] == attr.DECLARED_EFFECT[name]
            assert reading["interpretation"]["text"] == attr.EFFECT_TEXT[name][want]
            assert reading["in_band"] == attr.in_band(reading["delta_bpc"])
