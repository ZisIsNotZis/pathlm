"""契约 V3（2620 去相关的规模验证）的语义开关单测。

对象（每个都必须是**可失败**的：变异体见实验目录 `mutate.py`）：
1. 阶梯 `stock → l1 → divl1` 只改两个声明键（`p_skip` / `w_diversity`）；
2. 确定性路径 ⇒ K 次前向的 logits **逐位相同**（集成增益恰好 0 的来源）；
   而 `p_skip>0` ⇒ 路径真的不同（机制活着）；
3. 集成是**各路径 softmax 的均值**（不是取第一条、也不是取最后一条）；
4. 成对 JS：相同路径 = 0、不同路径 > 0、bits = nats/ln2；
5. 活性量落盘：`activity_line` 每 200 步带 `diversity_js` / `skip_frac`；
   `liveness_record` 对 w_diversity=0 的对照臂报 `alive=False`（不是死开关，是阶梯底）；
6. 主判据（读法 B）的**预登记形式**与**化简形式**代数等价（§12 算术对账）。

全部在 CPU 上跑（本仓测试的既有约定；GPU 工作要取锁，测试不参与 GPU 争用）。
"""

import json
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from pathlm.config import ModelConfig, PathConfig, sample_path       # noqa: E402
from pathlm.model import PathLM                                       # noqa: E402
from tools.evalkit.ensemble_probe import (                            # noqa: E402
    deterministic_paths, mixture_ce, pairwise_js, path_ensemble_bpc,
    reading_A, reading_B)
from train_m1 import activity_line, liveness_record                    # noqa: E402

HERE = os.path.join(ROOT, "archive", "dropped", "axes", "G-policy", "G5-parallel-lines",
                    "experiments", "2620-decorrelation-at-scale")
LN2 = 0.6931471805599453


def _tiny_model(seed: int = 0, **path_over) -> PathLM:
    torch.manual_seed(seed)
    mcfg = ModelConfig(d_model=32, n_layers=4, n_heads=4, seq_len=16, mlp_mult=2)
    path = {"n_mtp": 1, **path_over}
    return PathLM(mcfg, PathConfig(**path), vocab_size=64).eval()


def _fake_arr(n: int = 4096, vocab: int = 64, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=n).astype(np.uint16)


# ------------------------------------------------------------------ 1. 阶梯
def test_ladder_configs_differ_only_in_declared_knobs():
    """`stock→l1` 只改 p_skip；`l1→divl1` 只改 w_diversity；model/train 段逐字段相同。"""
    cfgs = {a: json.load(open(os.path.join(HERE, "configs", f"{a}.json")))
            for a in ("stock", "l1", "divl1")}
    assert cfgs["stock"]["model"] == cfgs["l1"]["model"] == cfgs["divl1"]["model"]
    assert cfgs["stock"]["train"] == cfgs["l1"]["train"] == cfgs["divl1"]["train"]
    assert cfgs["stock"]["train"]["seed"] == cfgs["l1"]["train"]["seed"] \
        == cfgs["divl1"]["train"]["seed"], "seed 必须逐臂一致（口径重叠要显式排除）"

    def path_delta(a, b):
        pa, pb = cfgs[a]["path"], cfgs[b]["path"]
        assert set(pa) == set(pb), "两臂的 path 键集必须一致（否则默认值会偷偷泄漏）"
        return {k: (pa[k], pb[k]) for k in pa if pa[k] != pb[k]}

    assert path_delta("stock", "l1") == {"p_skip": (0.0, 0.1)}
    assert path_delta("l1", "divl1") == {"w_diversity": (0.0, 0.05)}
    # 每个 PathConfig 字段都必须被配置文件**显式钉住**（SCHEMA §D2：不许静默吃默认值）。
    # 例外：`w_repr_decorr` 由 ticket 39（2622 去相关 v2）引入，2620 三臂早于它、
    # 正确地吃它的 0.0 默认值；其余字段一律必须显式。
    from dataclasses import fields
    all_keys = {f.name for f in fields(PathConfig)}
    for a in ("stock", "l1", "divl1"):
        assert all_keys - set(cfgs[a]["path"]) == {"w_repr_decorr"}, \
            f"{a} 未钉住全部 path 键"


# ------------------------------------------------------- 2. 确定性路径 = 逐位
def test_deterministic_paths_make_k_forwards_bitwise_identical():
    """确定性路径对照的实现接缝：**在同一条 p_skip>0 的模型上**进入
    `deterministic_paths` 后，K 次前向的 logits 必须逐位相同（这正是集成增益恰好 0
    的来源）；离开上下文后随机性回来。用 p_skip>0 的模型才抓得住「上下文是空操作」的变异。"""
    m = _tiny_model(p_skip=0.4)
    x = torch.randint(0, 64, (2, 16))
    rng = random.Random(0)
    with torch.no_grad():
        with deterministic_paths(m):
            outs = []
            for _ in range(4):
                path = sample_path(m.pcap, rng, m.mcfg.n_layers)
                assert len(path.layer_order) == m.mcfg.n_layers, \
                    "确定性路径必须保留全部层（不丢层）"
                outs.append(m(x, [path], x)[1]["rounds"][0][1]["logits"])
        after = []
        for _ in range(4):
            path = sample_path(m.pcap, rng, m.mcfg.n_layers)
            after.append(m(x, [path], x)[1]["rounds"][0][1]["logits"])
    assert all(torch.equal(outs[0], o) for o in outs[1:]), \
        "确定性路径下 K 条路径必须逐位相同（这正是集成增益恰好 0 的来源）"
    assert any(not torch.equal(after[0], o) for o in after[1:]), \
        "退出上下文后路径随机性必须回来（否则对照把模型改坏了）"
    assert (m.pcap.p_skip, m.pcap.p_redo, m.pcap.shuffle_locality) == (0.4, 0.0, 0.0), \
        "上下文退出必须还原路径旋钮"


def test_skip_paths_are_really_different_when_p_skip_positive():
    """机制活性：p_skip>0 时路径真的不同（否则「集成增益 0」会被误读成机制无效）。"""
    m = _tiny_model(p_skip=0.3)
    x = torch.randint(0, 64, (2, 16))
    rng = random.Random(1)
    with torch.no_grad():
        outs, orders = [], []
        for _ in range(6):
            path = sample_path(m.pcap, rng, m.mcfg.n_layers)
            orders.append(tuple(path.layer_order))
            outs.append(m(x, [path], x)[1]["rounds"][0][1]["logits"])
    assert len(set(orders)) > 1, "p_skip>0 却没采出不同层序 ⇒ 采样是死的"
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), \
        "不同层序必须给出不同 logits"


# ------------------------------------------------------- 3. 集成 = 路径均值
def test_mixture_ce_is_the_mean_over_paths():
    torch.manual_seed(0)
    a = torch.rand(2, 5, 8)
    a = a / a.sum(-1, keepdim=True)
    b = torch.rand(2, 5, 8)
    b = b / b.sum(-1, keepdim=True)
    tgt = torch.randint(0, 8, (2, 5))
    mix = (a + b) / 2
    got = mixture_ce(mix, tgt)
    assert got["tokens"] == 10
    # 与「取第一条」「取最后一条」都不同（除非两条恰好相同）
    assert not math.isclose(mixture_ce(a, tgt)["nats"], got["nats"], rel_tol=1e-6)
    assert not math.isclose(mixture_ce(b, tgt)["nats"], got["nats"], rel_tol=1e-6)
    # 若两条完全相同，集成必须退化为单条（确定性路径对照的算术本质）
    assert mixture_ce(a, tgt)["nats"] == pytest.approx(mixture_ce((a + a) / 2, tgt)["nats"])


def test_ensemble_bpc_equals_single_path_when_paths_are_deterministic():
    """端到端（CPU）：p_skip=0 时 K=4 集成与 K=1 逐位一致。"""
    m = _tiny_model(p_skip=0.0)
    arr = _fake_arr()
    k1 = path_ensemble_bpc(m, arr, 1, n_batches=2, batch_size=2, vocab_size=64)
    k4 = path_ensemble_bpc(m, arr, 4, n_batches=2, batch_size=2, vocab_size=64)
    assert k4["paths_bitwise_identical"] is True
    assert k4["n_batches_paths_identical"] == k4["n_batches"]
    assert k4["bpc"] == pytest.approx(k1["bpc"], abs=1e-6)
    assert k4["acc"] == k1["acc"]


# ------------------------------------------------------------- 4. 成对 JS
def test_pairwise_js_zero_for_identical_positive_for_distinct():
    torch.manual_seed(0)
    p = torch.rand(3, 4, 6)
    p = p / p.sum(-1, keepdim=True)
    same = pairwise_js(torch.stack([p, p, p]))
    assert same["mean_nats"] == pytest.approx(0.0, abs=1e-9)
    assert same["mean_bits"] == pytest.approx(0.0, abs=1e-9)
    assert same["n_pairs"] == 3
    q = torch.full_like(p, 1.0 / p.shape[-1])
    diff = pairwise_js(torch.stack([p, q]))
    assert diff["mean_nats"] > 0.0
    assert diff["mean_bits"] == pytest.approx(diff["mean_nats"] / LN2)
    assert pairwise_js(p.unsqueeze(0))["mean_nats"] == 0.0   # K=1 没有成对项


# --------------------------------------------------------- 5. 活性量落盘
def test_activity_line_carries_diversity_and_skip_frac():
    class P:
        def __init__(self, layers):
            self.layer_order = layers

    loss = torch.tensor(3.5, requires_grad=True)
    paths = [P([0, 1, 2, 3]), P([0, 2, 3])]
    with_div = activity_line(200, loss, 0.001, 1.5, torch.tensor(2.0), paths,
                             {"diversity": torch.tensor(0.25)}, 4)
    assert with_div["diversity_js"] == 0.25
    assert with_div["skip_frac"] == 0.875          # (4+3)/(2*4)
    assert with_div["n_paths"] == 2
    assert with_div["step"] == 200 and with_div["loss"] == 3.5
    no_div = activity_line(200, loss, 0.001, 1.5, torch.tensor(2.0), paths, {}, 4)
    assert no_div["diversity_js"] is None, "对照臂也必须落活性字段（值为 None）"
    assert no_div["skip_frac"] == 0.875


def test_liveness_record_marks_control_arm_as_control_not_dead_switch():
    pc = PathConfig(p_skip=0.1, w_diversity=0.0)
    off = liveness_record(pc, [], [1.0, 1.0], [0.5], 0, 1)
    assert off["alive"] is False and off["diversity_js_last"] is None
    assert off["skip_frac_mean"] == 1.0 and off["w_diversity"] == 0.0
    on = liveness_record(PathConfig(p_skip=0.1, w_diversity=0.05),
                         [0.1, 0.3, 0.2], [0.9], [0.7], 2, 2)
    assert on["alive"] is True
    assert on["diversity_js_first"] == 0.1 and on["diversity_js_last"] == 0.2
    assert on["diversity_js_mean"] == pytest.approx(0.2)
    assert on["gate_skips"] == 2


# --------------------------------------------------- 6. 主判据的代数等价
def test_reading_b_verbatim_form_equals_reduced_form():
    """预登记形式 `[l1@K1 − divl1@K8] − [l1@K1 − stock@K1]` == `stock@K1 − divl1@K8`。"""
    l1, stock, div = 1.69, 1.51, 1.62
    r = reading_B(l1, stock, div)
    assert r["algebra_consistent"] is True
    assert r["verbatim"] == pytest.approx(r["reduced"])
    assert r["reduced"] == pytest.approx(stock - div)
    assert r["passed_point_estimate"] is (r["reduced"] > 0)
    # 13M 既有锚点：读法 B 不过、读法 A 过（这正是 §17.14 事故的两种读法）
    r13 = reading_B(1.6926, 1.5074, 1.6227)
    assert r13["passed_point_estimate"] is False
    assert reading_A(1.6926, 1.6227)["passed_point_estimate"] is True
