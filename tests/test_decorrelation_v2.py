"""契约 2622（去相关 v2，ticket 39 X1）的语义开关单测。

对象（每个都必须是**可失败**的：变异体见实验目录 `mutate.py`）：
1. 阶梯 `stock → l1 → repr` 只改两个声明键（`p_skip` / `w_repr_decorr`）；
2. `repr_decorr_penalty`：同一路径 = 1.0、独立路径 ≈ 0、尺度不变、梯度只流向当前路径；
3. 引擎接线：`w_repr_decorr>0` 才落 `aux["repr_decorr"]`；
4. `sample_rounds` 在 `w_repr_decorr>0` 时多采一条并行路径；
5. `validate_pcap` 拒绝两条去相关罚则同时开（单因素）；
6. pre-step gate 只在 **非有限** 梯度上跳过（ticket 39 的 gate 混杂修复）；
7. 活性量落盘：`activity_line` 带 `repr_decorr`；`liveness_record` 报 repr 统计；
8. 主判据任务形式 `repr@K8 − stock@K1 < 0` 与 `reading_B` 的化简形式代数等价。

全部在 CPU 上跑（本仓测试的既有约定；GPU 工作要取锁，测试不参与 GPU 争用）。
"""

import json
import os
import random
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from dataclasses import fields                                       # noqa: E402
from pathlm.config import ModelConfig, PathConfig                    # noqa: E402
from pathlm.model import PathLM, repr_decorr_penalty                 # noqa: E402
from tools.evalkit.ensemble_probe import (                           # noqa: E402
    deterministic_paths, path_ensemble_bpc, reading_B)
from train_m1 import (activity_line, gate_skip, liveness_record,     # noqa: E402
                      sample_rounds, validate_pcap)

HERE = os.path.join(ROOT, "archive", "dropped", "axes", "G-policy", "G5-parallel-lines",
                    "experiments", "2622-decorrelation-v2")


def _tiny_model(seed: int = 0, **path_over) -> PathLM:
    torch.manual_seed(seed)
    mcfg = ModelConfig(d_model=32, n_layers=4, n_heads=4, seq_len=16, mlp_mult=2)
    return PathLM(mcfg, PathConfig(n_mtp=1, **path_over), vocab_size=64).eval()


def _fake_arr(n: int = 4096, vocab: int = 64, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, vocab, size=n).astype(np.uint16)


# ------------------------------------------------------------------ 1. 阶梯
@pytest.mark.parametrize("scale,path_keys", [("13m", ("stock", "l1", "repr", "repr01")),
                                             ("50m", ("stock", "l1", "repr"))])
def test_ladder_configs_differ_only_in_declared_knobs(scale, path_keys):
    cdir = os.path.join(HERE, f"configs{scale}")
    cfgs = {a: json.load(open(os.path.join(cdir, f"{a}.json"))) for a in path_keys}
    ref = cfgs["stock"]
    for a in path_keys:
        assert cfgs[a]["model"] == ref["model"], f"{a} model 段不同"
        assert cfgs[a]["train"] == ref["train"], f"{a} train 段不同"
    assert (cfgs["stock"]["train"]["seed"] == cfgs["l1"]["train"]["seed"]
            == cfgs["repr"]["train"]["seed"]), "seed 必须逐臂一致"

    def path_delta(a, b):
        pa, pb = cfgs[a]["path"], cfgs[b]["path"]
        assert set(pa) == set(pb), "两臂的 path 键集必须一致（否则默认值会偷偷泄漏）"
        return {k: (pa[k], pb[k]) for k in pa if pa[k] != pb[k]}

    assert path_delta("stock", "l1") == {"p_skip": (0.0, 0.1)}
    assert path_delta("l1", "repr") == {"w_repr_decorr": (0.0, 1.0)}
    if scale == "13m":
        assert path_delta("l1", "repr01") == {"w_repr_decorr": (0.0, 0.1)}
        assert path_delta("repr", "repr01") == {"w_repr_decorr": (1.0, 0.1)}
    all_keys = {f.name for f in fields(PathConfig)}
    for a in path_keys:
        assert all_keys - set(cfgs[a]["path"]) == set(), f"{a} 未钉住全部 path 键"


# ------------------------------------------------- 2. repr_decorr_penalty 本体
def test_repr_penalty_identical_is_one_and_independent_is_zero():
    torch.manual_seed(0)
    a = torch.randn(512, 16)
    b = torch.randn(512, 16)
    assert float(repr_decorr_penalty(a, a)) == pytest.approx(1.0, abs=1e-5)
    assert float(repr_decorr_penalty(a, b)) < 0.02, "独立路径的相关罚则必须接近 0"


def test_repr_penalty_is_scale_invariant_per_feature():
    """Pearson 相关对每维正尺度不变 ⇒ 罚则不能被「缩小隐状态」这个退化解刷低。"""
    torch.manual_seed(0)
    a = torch.randn(256, 8)
    b = torch.randn(256, 8) + a          # 造一点真实相关
    base = float(repr_decorr_penalty(a, b))
    assert base > 0.05
    assert float(repr_decorr_penalty(a, b * 3.0)) == pytest.approx(base, rel=1e-4)
    assert float(repr_decorr_penalty(a * 0.25, b)) == pytest.approx(base, rel=1e-4)


def test_repr_penalty_gradient_flows_only_to_current_path():
    torch.manual_seed(0)
    a = torch.randn(128, 8)
    b = torch.randn(128, 8, requires_grad=True)
    pen = repr_decorr_penalty(a, b)
    pen.backward()
    assert b.grad is not None and torch.isfinite(b.grad).all() and b.grad.abs().sum() > 0


# ----------------------------------------------------- 3. 引擎接线 / 4. 并行路径
def test_model_wires_repr_penalty_only_when_enabled():
    x = torch.randint(0, 64, (2, 16))
    on = _tiny_model(p_skip=0.3, w_repr_decorr=0.5)
    paths = sample_rounds(on.pcap, random.Random(0), on.mcfg.n_layers)
    assert len(paths) == 2, "w_repr_decorr>0 必须多采一条并行路径"
    _, aux = on(x, paths, x)
    assert "repr_decorr" in aux and float(aux["repr_decorr"]) > 0

    off = _tiny_model(p_skip=0.3, w_repr_decorr=0.0)
    paths_off = sample_rounds(off.pcap, random.Random(0), off.mcfg.n_layers)
    assert len(paths_off) == 1, "w_repr_decorr=0 不得多采路径（对照臂）"
    _, aux_off = off(x, paths_off, x)
    assert "repr_decorr" not in aux_off


def test_sample_rounds_single_factor_parallel_path():
    pc = PathConfig(p_skip=0.1, w_repr_decorr=1.0)
    assert len(sample_rounds(pc, random.Random(0), 4)) == 2
    pc2 = PathConfig(p_skip=0.1, w_diversity=0.05)
    assert len(sample_rounds(pc2, random.Random(0), 4)) == 2


# ------------------------------------------------------------- 5. 单因素护栏
def test_validate_pcap_rejects_two_decorrelation_terms():
    validate_pcap(PathConfig(p_skip=0.1, w_repr_decorr=1.0))          # 单开 OK
    validate_pcap(PathConfig(p_skip=0.1, w_diversity=0.05))           # 单开 OK
    validate_pcap(PathConfig())                                      # 全关 OK
    with pytest.raises(ValueError):
        validate_pcap(PathConfig(p_skip=0.1, w_diversity=0.05, w_repr_decorr=1.0))
    with pytest.raises(ValueError):
        validate_pcap(PathConfig(reentry_mix=True))                   # 无 loop 的 no-op


# ----------------------------------------------------- 6. gate 只在非有限时跳
def test_gate_skip_only_on_non_finite():
    assert gate_skip(torch.tensor(float("inf"))) is True
    assert gate_skip(torch.tensor(float("nan"))) is True
    assert gate_skip(torch.tensor(1e9)) is False, "有限的大梯度由 clip 处理，不跳步"
    assert gate_skip(torch.tensor(1.0)) is False


# --------------------------------------------------------- 7. 活性量落盘
def test_activity_line_and_liveness_carry_repr_decorr():
    class P:
        def __init__(self, layers):
            self.layer_order = layers

    loss = torch.tensor(3.5, requires_grad=True)
    paths = [P([0, 1, 2, 3]), P([0, 2, 3])]
    line = activity_line(200, loss, 0.001, 1.5, torch.tensor(2.0), paths,
                         {"repr_decorr": torch.tensor(0.42)}, 4)
    assert line["repr_decorr"] == 0.42 and line["skip_frac"] == 0.875
    line_off = activity_line(200, loss, 0.001, 1.5, torch.tensor(2.0), paths, {}, 4)
    assert line_off["repr_decorr"] is None

    pc = PathConfig(p_skip=0.1, w_repr_decorr=1.0)
    on = liveness_record(pc, [], [0.9], [0.7], 0, 2, repr_hist=[0.9, 0.5, 0.3])
    assert on["alive"] is True and on["w_repr_decorr"] == 1.0
    assert on["repr_decorr_first"] == 0.9 and on["repr_decorr_last"] == 0.3
    assert on["n_repr_observations"] == 3
    off = liveness_record(PathConfig(p_skip=0.1), [], [1.0], [0.5], 0, 1, repr_hist=[])
    assert off["alive"] is False and off["repr_decorr_last"] is None
    assert off["w_repr_decorr"] == 0.0


# --------------------------------------------------- 8. 主判据代数等价
def test_main_criterion_task_form_equals_neg_reading_b():
    """任务预登记形式 `bpc_repr@K8 − bpc_stock@K1 < 0`
    等价于 `reading_B(l1@K1, stock@K1, repr@K8)` 的化简式取负。"""
    l1, stock, repr_ = 1.69, 1.51, 1.62
    rb = reading_B(l1, stock, repr_)
    assert rb["algebra_consistent"] is True
    assert rb["reduced"] == pytest.approx(stock - repr_)
    task_form = repr_ - stock
    assert task_form == pytest.approx(-rb["reduced"])
    assert rb["passed_point_estimate"] is False          # 该点 repr 集成反而更差
    assert (task_form < 0) is False


# --------------------------------------------- 9. 确定性路径对照（repr 臂）
def test_deterministic_paths_are_bitwise_identical_for_repr_arm():
    m = _tiny_model(p_skip=0.4, w_repr_decorr=0.5)
    arr = _fake_arr()
    with deterministic_paths(m):
        k4 = path_ensemble_bpc(m, arr, 4, n_batches=2, batch_size=2, vocab_size=64)
        k1 = path_ensemble_bpc(m, arr, 1, n_batches=2, batch_size=2, vocab_size=64)
    assert k4["paths_bitwise_identical"] is True
    assert k4["n_batches_paths_identical"] == k4["n_batches"]
    assert k4["bpc"] == pytest.approx(k1["bpc"], abs=1e-6)
    assert (m.pcap.p_skip, m.pcap.p_redo, m.pcap.shuffle_locality) == (0.4, 0.0, 0.0)
