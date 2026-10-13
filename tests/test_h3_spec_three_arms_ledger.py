"""H3 `2631-spec-three-arms` 每 token 算力账 `R=(1+γ)/(1+τ)` 的可失败断言。

背景：2631 早先把**墙钟** width-3/width-1 前向比 `c₃` 当代算力比，产出
`equivalent_compute_ratio = c₃/(tok/fwd)`（0.48/0.49/0.56，看着像「2× 省算力」）。
H1 复核裁定这是**符号级错误**——一次宽度-(1+γ) 前向按层-位置 FLOPs ≈ `(1+γ)×`
一次逐 token 前向，故算力比是解析量 `(1+γ)`，与墙钟无关。

本测试把两件事钉成可失败断言：
1. 生成器 `run.py::arm_ledger` 与落盘 `results.json` 的 `equivalent_compute_R`
   逐字段一致（改坏生成器或改坏产物即红）；
2. 旧口径只在 `equivalent_compute_ratio_withdrawn` 下、且必须与 R 背离（`<1` vs `>1`），
   防止旧值再次被当成算力。
"""

import importlib.util
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(
    ROOT, "axes", "6-predict", "6.2-speculative-decoding", "experiments",
    "2631-spec-three-arms")
RUN = os.path.join(EXP, "run.py")
RESULTS = os.path.join(EXP, "results.json")


def _load_run():
    spec = importlib.util.spec_from_file_location("h3_2631_run", RUN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # run.py 自己把仓库根与实验目录插进 sys.path
    return m


def test_committed_results_use_R_and_withdraw_old_ledger():
    """9 个 (checkpoint × 臂) 必须：R == arm_ledger(...)，旧字段已移出，且旧值<R 背离。"""
    m = _load_run()
    res = json.load(open(RESULTS, encoding="utf-8"))
    gamma = res["config"]["gamma"]
    seen = 0
    for ck, v in res["checkpoints"].items():
        c3 = v["micro"]["c3_vs_w1"]
        for arm, s in v["acceptance"].items():
            exp = m.arm_ledger(gamma, s["tokens_per_forward"], c3)
            assert s["equivalent_compute_R"] == exp["equivalent_compute_R"], (ck, arm)
            assert "equivalent_compute_ratio" not in s, f"{ck}/{arm} 旧字段仍在前台"
            assert (s["equivalent_compute_ratio_withdrawn"]["value"]
                    == exp["equivalent_compute_ratio_withdrawn"]["value"]), (ck, arm)
            # 符号错误活证：旧账 <1（像「省」）而 R>1（实为「亏」）
            assert s["equivalent_compute_ratio_withdrawn"]["value"] < 1.0, (ck, arm)
            assert s["equivalent_compute_R"] > 1.0, (ck, arm)
            seen += 1
    assert seen == 9


def test_R_is_analytic_and_independent_of_c3():
    """同 tok/fwd、不同 c₃ ⇒ R 相同而旧账不同（证明 R 不含 c₃）。"""
    m = _load_run()
    a = m.arm_ledger(2, 2.5, 1.25)
    b = m.arm_ledger(2, 2.5, 5.28)
    assert a["equivalent_compute_R"] == b["equivalent_compute_R"] == round(3 / 2.5, 4)
    assert a["equivalent_compute_R"] > 1.0
    assert (a["equivalent_compute_ratio_withdrawn"]["value"]
            != b["equivalent_compute_ratio_withdrawn"]["value"])


def test_R_bounds_hold():
    """τ=γ ⇒ R=1（持平）；τ<γ ⇒ R>1（亏）；τ=0 ⇒ R=1+γ。"""
    m = _load_run()
    assert m.arm_ledger(2, 3.0, 1.0)["equivalent_compute_R"] == 1.0
    assert m.arm_ledger(2, 2.0, 1.0)["equivalent_compute_R"] == 1.5
    assert m.arm_ledger(2, 1.0, 1.0)["equivalent_compute_R"] == 3.0


def test_compute_ledger_correction_block_present_and_explicit():
    res = json.load(open(RESULTS, encoding="utf-8"))
    c = res["compute_ledger_correction"]
    assert "R = (1+gamma)/(1+tau)" in c["rule"]
    assert c["new_field"] == "equivalent_compute_R"
