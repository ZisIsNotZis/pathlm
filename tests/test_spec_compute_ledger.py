"""H1 算力账（预登记主判据）的单测：`R = (1+γ)/(1+τ)`。

对应 `axes/H-horizon/H1-outputs-per-pass/experiments/2520-spec-at-scale/`：
spec 一次前向把 `1+γ` 个位置送进主干、产出 `1+τ` 个 token，故每 token 算力
（以 native 逐 token 的 1 次 width-1 前向为单位）为 `R=(1+γ)/(1+τ)`。
旧口径 `c₃/(tok/fwd)` 把**墙钟**前向比 `c₃` 当算力比，是符号错误——本测试把
两条口径钉成可失败断言，防止 `c₃` 再次被代进算力。
"""

import importlib.util
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
LEDGER = os.path.join(
    ROOT, "axes", "H-horizon", "H1-outputs-per-pass", "experiments",
    "2520-spec-at-scale", "compute_ledger.py")


def _mod():
    spec = importlib.util.spec_from_file_location("h1_compute_ledger", LEDGER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_R_formula_matches_analytic_ratio():
    """100M enwik8 k=2 实点：γ=2、tok/fwd=2.3952 ⇒ R = 3/2.3952 ≈ 1.2525（亏损）。"""
    out = _mod().compute_ledger(gamma=2, tok_per_fwd=2.3952, speedup=1.223,
                                c_wall=5.276)
    # 函数按 4 位小数的报告精度四舍五入 ⇒ 容差取 1e-4
    assert abs(out["per_token_compute_ratio__analytic"] - 3 / 2.3952) < 1e-4
    assert abs(out["tau_accepted"] - 1.3952) < 1e-9
    assert out["verdict"] == "loss"


def test_old_wallclock_ledger_is_separate_and_may_disagree():
    """旧账 = c₃/tokfwd 与 R 分开：c₃=1.283（13M）时旧账 <1 而 R>1 —— 背离即符号错误。"""
    out = _mod().compute_ledger(gamma=2, tok_per_fwd=2.3952, speedup=1.677,
                                c_wall=1.283)
    assert abs(out["old_wallclock_ledger__c_wall_over_tokfwd"]
               - 1.283 / 2.3952) < 1e-4
    assert out["old_wallclock_ledger__c_wall_over_tokfwd"] < 1.0
    assert out["per_token_compute_ratio__analytic"] > 1.0


def test_R_bounds_tau_lt_gamma_implies_loss():
    """解析上界：τ<γ ⇒ R>1；τ=0 ⇒ R=1+γ。"""
    mod = _mod()
    for g in (1, 2):
        # 物理可达域：τ=tok−1 ≤ γ ⇒ tok ≤ 1+γ（接受数不可能超过草稿数）
        for tok in (1.0, 1.0 + 0.4 * g, 1.0 + g):
            R = mod.compute_ledger(g, tok, 1.0, None)["per_token_compute_ratio__analytic"]
            assert 1.0 - 1e-9 <= R <= (1 + g) + 1e-9
    # τ=γ ⇒ R=1（持平，非亏损）；τ<γ 且 R>1.05 ⇒ loss
    assert mod.compute_ledger(2, 3.0, 1.0, None)["per_token_compute_ratio__analytic"] == 1.0
    assert mod.compute_ledger(2, 3.0, 1.0, None)["verdict"] == "flat"
    assert mod.compute_ledger(2, 2.5, 1.0, None)["verdict"] == "loss"


def test_rows_from_parses_rerun_and_pairs_width(tmp_path):
    """run.py 产物（顶层即单 checkpoint + 各臂）要被解析，且墙钟前向比与宽度配对：
    γ=2 ⇒ width-3（w3_vs_w1），γ=1 ⇒ width-2（w2_vs_w1）——不得用 width-3 比值除
    width-2 的 tok/fwd（P2.2 的口径瑕疵）。"""
    blob = {
        "checkpoint": "INT3_100M_s0", "n_mtp": 3, "reps": 5,
        "micro": {"w2_vs_w1": 2.3, "w3_vs_w1": 2.439},
        "enwik8_k1": {"tokens_per_forward": 1.8182,
                      "speedup_ratio_of_best": 0.931},
        "enwik8_k2": {"tokens_per_forward": 2.3952,
                      "speedup_ratio_of_best": 1.07,
                      "speedup_reps_paired": [1.0, 1.06, 1.045]},
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(blob))
    rows = {r["k_drafts"]: r for r in _mod().rows_from(str(p))}
    assert set(rows) == {1, 2}
    assert rows[2]["c_wall_forward_ratio"] == 2.439   # 宽度 3 = 1+γ(γ=2)
    assert rows[1]["c_wall_forward_ratio"] == 2.3     # 宽度 2 = 1+γ(γ=1)，不是 width-3
    assert abs(rows[2]["per_token_compute_ratio_R"] - 3 / 2.3952) < 1e-4
    assert rows[2]["R_verdict"] == "loss"
    assert rows[2]["old_ledger_cwall_over_tokfwd"] == round(2.439 / 2.3952, 4)
    assert rows[1]["old_ledger_cwall_over_tokfwd"] == round(2.3 / 1.8182, 4)
