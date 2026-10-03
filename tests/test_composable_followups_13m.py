"""2624-composable-followups-13m · 三个补点的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义逻辑（`points.py`）+ 评测仪器的一致性（`run.py::accuracy_detailed`
必须与既有仪器 `longrange_tasks.accuracy` 逐位相同，SCHEMA §9 对照复用规则 1）。

三组断言，各对准契约里的一条判据：

- **① 容量括号 + 插值**：两个 MHA 点**非退化**（参数量不同，§17.16）、括号**夹住** GQA、
  插值只在括号内（**禁外推**，§17.11）、退化括号必须报错；
- **② 无泄漏 Q=7**：`unique_unqueried_last_block` 在 Q=K=8 时为真（末块是唯一没被问过
  的那一对 ⇒ 排除法伪影）、在 Q=K−1=7 时为假（**伪影条件被消除**）；逐块分解覆盖全部
  查询；坐标换算 `pick=(rows−dists)/2`、`j=(rows−base−1)/(3+gap)` 与生成器自洽；
- **③ 6k → 10k 同相对刻度**：两对逐字段同配方、只差步数、两对用**同一**刻度（§17.16）。

变异验证（每处抓红 + 逐字节还原）见同目录 `mutate.py`。
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2624-composable-followups-13m")
sys.path.insert(0, EXP)

from pathlm.depth_ar import DepthARConfig, DepthARModel            # noqa: E402
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    accuracy, assoc_recall_multi_batch)
import points as P                                                 # noqa: E402


def _load_run_module():
    """按**唯一模块名**装载本实验的 `run.py`（避免与其它实验的 `run.py` 撞名）。"""
    spec = importlib.util.spec_from_file_location("exp2624_run",
                                                  os.path.join(EXP, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUN = _load_run_module()

#: 13M 规格（与训练同配方）；参数量的断言用 d=256/layers=12。
BIG = {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
       "dense": True, "topology": "sequential"}
T, K, GAP = 256, 8, 14


def _batch(Q, bs=4, T_=T, K_=K, gap=GAP, seed=0):
    return assoc_recall_multi_batch(bs, T_, K_, Q, gap=gap,
                                    generator=torch.Generator().manual_seed(seed))


def _build(name, vocab=10):
    torch.manual_seed(0)
    cfg = DepthARConfig(**{**BIG, **P.model_config_kwargs(name)})
    return DepthARModel(cfg, vocab_size=vocab)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------ 臂表本身

def test_arm_registry_is_the_contract_set():
    assert P.ARM_NAMES == ("stock", "gqa", "gqa_carry", "gqa_carry_window",
                           "mha_mlp5", "gqa_10k", "gqa_carry_10k")
    assert P.BASELINE_ARM == "stock"
    assert P.Q_TRAIN == 8 and P.Q_LEAKFREE == 7


def test_registered_param_counts_match_built_models():
    """臂表里写死的参数量 = 真建出来的参数量（① 的括号几何依赖它）。"""
    for name in P.ARM_NAMES:
        # vocab=206（与训练/评测同一词表；embedding 参与参数量账）
        assert _n_params(_build(name, vocab=206)) == P.ARMS[name]["params"], name
    # 训练配方的其余字段逐字段相同（唯一变量 = mlp_mult / steps 与 flags）
    sa, sb = P.arm_spec("stock"), P.arm_spec("mha_mlp5")
    assert sa["flags"] == sb["flags"] and sa["steps"] == sb["steps"]
    assert (sa["mlp_mult"], sb["mlp_mult"]) == (6, 5)
    assert sb["params"] < sa["params"]            # 括号下端确实更低


# ------------------------------------------------ ① 容量括号 + 插值

def test_capacity_bracket_span_contains_gqa_and_is_nondegenerate():
    lo, hi = P.bracket_points()
    assert (lo, hi) == ("mha_mlp5", "stock")
    geo = P.bracket_span(P.ARMS[lo]["params"], P.ARMS[hi]["params"],
                         P.ARMS[P.P1_TARGET]["params"])
    assert geo["target_inside"] is True
    assert geo["lo"] == 11_960_832 and geo["hi"] == 13_536_768
    assert geo["target"] == 12_352_512
    assert 0.0 < geo["target_offset_frac"] < 1.0


def test_capacity_slope_rejects_degenerate_bracket():
    """同参数量两点 ⇒ 退化括号，斜率必须报错（§17.16）。"""
    try:
        P.capacity_slope(100, 100, 1.0, 2.0)
    except ValueError:
        return
    raise AssertionError("退化括号（x_lo == x_hi）必须报 ValueError")


def test_interpolation_values_and_no_extrapolation():
    # 手算：括号 [10, 20]，读数 [1.0, 3.0] ⇒ 斜率 0.2；插值到 15 = 2.0
    assert P.capacity_slope(10, 20, 1.0, 3.0) == 0.2
    assert abs(P.interp_at_target(15, 10, 20, 1.0, 3.0) - 2.0) < 1e-12
    assert abs(P.residual_vs_interp(2.5, 2.0) - 0.5) < 1e-12
    for x in (9.9, 20.1):
        try:
            P.interp_at_target(x, 10, 20, 1.0, 3.0)
        except ValueError:
            continue
        raise AssertionError(f"x={x} 在括号外，插值必须报错（§17.11 禁外推）")


def test_flat_mha_bracket_makes_gqa_residual_equal_to_gap():
    """MHA 两点读数相同时（容量斜率 0），GQA 的残差 = GQA − MHA（=几何差）。"""
    slope = P.capacity_slope(11_960_832, 13_536_768, 0.30, 0.30)
    assert slope == 0.0
    y_i = P.interp_at_target(12_352_512, 11_960_832, 13_536_768, 0.30, 0.30)
    assert abs(y_i - 0.30) < 1e-12
    assert abs(P.residual_vs_interp(0.31, y_i) - 0.01) < 1e-12


# ------------------------------------------------ ② 无泄漏 Q=7

def test_pick_and_block_coordinates_match_generator():
    tb = _batch(P.Q_TRAIN)
    pick = P.pick_index(tb.rows, tb.dists)
    blk = P.block_index(tb.rows, tb.T, P.Q_TRAIN, GAP)
    assert pick.shape == tb.rows.shape and blk.shape == tb.rows.shape
    assert int(pick.min()) >= 0 and int(pick.max()) < K
    assert set(blk.flatten().tolist()) == set(range(P.Q_TRAIN))
    for b in range(tb.B):
        assert sorted(pick[b].tolist()) == list(range(K))       # Q=K ⇒ 是 K 的一个排列
    # 逐块分解覆盖全部查询（每块每行恰好一个）
    counts = {j: int((blk == j).sum().item()) for j in range(P.Q_TRAIN)}
    assert set(counts.values()) == {tb.B}


def test_q8_last_block_is_unique_unqueried_q7_is_not():
    """本线的核心可失败断言：伪影条件在 Q=K 成立、在 Q=K−1 不成立。"""
    tb8 = _batch(P.Q_TRAIN)
    leak8 = P.unique_unqueried_last_block(tb8.rows, tb8.dists, tb8.T,
                                          P.Q_TRAIN, K, GAP)
    assert all(leak8), "Q=K=8 时末块问的是唯一没被前面问过的那一对 ⇒ 应由排除法白拿"
    tb7 = _batch(P.Q_LEAKFREE)
    leak7 = P.unique_unqueried_last_block(tb7.rows, tb7.dists, tb7.T,
                                          P.Q_LEAKFREE, K, GAP)
    assert not any(leak7), "Q=K−1=7 时末块仍剩 2 个候选 ⇒ 伪影条件必须被消除"
    # Q=7 的 7 个查询互不相同（生成器保证）且末块不是全部 K 对
    pick7 = P.pick_index(tb7.rows, tb7.dists)
    for b in range(tb7.B):
        row = pick7[b].tolist()
        assert len(set(row)) == P.Q_LEAKFREE


def test_leakfree_leg_fits_and_shape_is_consistent():
    Q = P.leak_free_leg_gap(T, K, GAP)
    assert Q == P.Q_LEAKFREE
    tb = _batch(Q)
    assert tb.M == Q
    assert tb.rows.shape == (tb.B, Q) and tb.answers.shape == (tb.B, Q)
    assert tb.chance() == 1.0 / 64
    # 距离必须远大于任何局部窗口（SCHEMA §19：局部先验抗性）
    assert int(tb.dists.min()) > 16


# ------------------------------------------------ ③ 6k → 10k 同相对刻度

def test_scale_pairs_same_recipe_same_ratio():
    pairs = P.scale_pair_check()
    assert [(p["base_arm"], p["long_arm"]) for p in pairs] == [
        ("gqa", "gqa_10k"), ("gqa_carry", "gqa_carry_10k")]
    assert {p["ratio"] for p in pairs} == {P.scale_ratio()}
    assert abs(P.scale_ratio() - 10000 / 6000) < 1e-4
    # 两对的步数比**精确**相同（同相对刻度，不是四舍五入后相同）
    assert len({tuple(p["steps"]) for p in pairs}) == 1
    for a, b in P.P3_PAIRS:
        sa, sb = P.arm_spec(a), P.arm_spec(b)
        assert sa["flags"] == sb["flags"] and sa["mlp_mult"] == sb["mlp_mult"]
        assert sa["steps"] == 6000 and sb["steps"] == 10000


def test_increment_helpers_and_orderability():
    assert P.increment_pp(0.10, 0.25527) == 15.527
    assert P.increment_pp(0.30, 0.30) == 0.0
    assert P.orderable(1.4) and not P.orderable(1.399)
    inc = P.carry_increment({"gqa": 0.0, "gqa_carry": 0.15674,
                            "gqa_10k": 0.02, "gqa_carry_10k": 0.19}, 10000)
    assert inc["target_pp"] == 17.0 and inc["steps"] == 10000
    try:
        P.carry_increment({"gqa": 0.0, "gqa_carry": 0.1}, 12000)
    except ValueError:
        return
    raise AssertionError("未登记的 token 预算必须报错（防跨刻度混读）")


# ------------------------------------------------ 仪器一致性（§9）

def test_accuracy_detailed_matches_existing_instrument_bitwise():
    """**同一件仪器**：`run.accuracy_detailed` 的聚合（含 by_distance）必须与
    `longrange_tasks.accuracy` 逐位相同——否则按块分解读的不是同一批数据。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=64,
                        mlp_mult=2, dense=True, topology="sequential")
    model = DepthARModel(cfg, vocab_size=206).eval()
    for Q in (P.Q_TRAIN, P.Q_LEAKFREE):
        gap = 2                                     # T=64 上放得下 2K + Q(3+gap)
        tb = assoc_recall_multi_batch(4, 64, K, Q, gap=gap,
                                      generator=torch.Generator().manual_seed(7))
        ref = accuracy(model, tb, device="cpu")
        hits = RUN.accuracy_detailed(model, tb, device="cpu")
        assert int(hits.sum().item()) == ref["n_hits"]
        assert hits.numel() == ref["n"]
        by_dist = RUN._by_dist_from_hits(hits, tb.dists)
        assert by_dist == ref["by_distance"]
        prof = RUN._block_profile(hits, tb, tb.T, Q, gap)
        assert sum(v["n"] for v in prof.values()) == ref["n"]
        assert sum(v["n_hits"] for v in prof.values()) == ref["n_hits"]
