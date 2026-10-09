"""契约 2626（G5 机制收敛收尾，ticket 43 V2）的单因素护栏单测。

对象 = 本实验的配置阶梯与评测配置（新增/被比较的落盘产物）：
  stock → l1 → repr01 三条臂**只改两个声明键**（`p_skip` / `w_repr_decorr`），
  其余 `model` / `train` / `path` 键**逐字段相同（含 seed）**；
  三臂都跑 24000 步（4× X1 的 token）；`run.py` 的轨迹点 = ckpt@6k/12k/18k + 终点。
可失败性由 `mutate.py` 逐字节变异验证（抓红 + 还原）。
"""

import json
import os
import sys
from dataclasses import fields

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from pathlm.config import PathConfig                                  # noqa: E402

HERE = os.path.join(ROOT, "archive", "dropped", "axes", "G-policy", "G5-parallel-lines",
                    "experiments", "2626-mechanism-convergence")
X1 = os.path.join(ROOT, "archive", "dropped", "axes", "G-policy", "G5-parallel-lines",
                  "experiments", "2622-decorrelation-v2")
ARMS = ("stock", "l1", "repr01")
TOTAL_STEPS = 24000


def _load(arm: str) -> dict:
    return json.load(open(os.path.join(HERE, "configs13m", f"{arm}.json")))


def _path_delta(a: str, b: str) -> dict:
    pa, pb = _load(a)["path"], _load(b)["path"]
    assert set(pa) == set(pb), "两臂的 path 键集必须一致（否则默认值会偷偷泄漏）"
    return {k: (pa[k], pb[k]) for k in pa if pa[k] != pb[k]}


def test_ladder_configs_differ_only_in_declared_knobs():
    cfgs = {a: _load(a) for a in ARMS}
    ref = cfgs["stock"]
    for a in ARMS:
        assert cfgs[a]["model"] == ref["model"], f"{a} model 段不同"
        assert cfgs[a]["train"] == ref["train"], f"{a} train 段不同"
    assert len({cfgs[a]["train"]["seed"] for a in ARMS}) == 1, "seed 必须逐臂一致"

    assert _path_delta("stock", "l1") == {"p_skip": (0.0, 0.1)}
    assert _path_delta("l1", "repr01") == {"w_repr_decorr": (0.0, 0.1)}

    all_keys = {f.name for f in fields(PathConfig)}
    for a in ARMS:
        assert all_keys - set(cfgs[a]["path"]) == set(), f"{a} 未钉住全部 path 键"


def test_all_arms_train_24000_steps_seed0():
    for a in ARMS:
        tr = _load(a)["train"]
        assert tr["steps"] == TOTAL_STEPS, f"{a} 步数必须是 24000（4× X1 token）"
        assert tr["seed"] == 0


def test_run_config_trajectory_and_arms():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    s = cfg["scales"]["13m"]
    assert s["steps_full"] == TOTAL_STEPS
    assert s["arms"] == ["stock", "l1", "repr01"]
    assert s["traj_ckpts"] == [6000, 12000, 18000]
    assert all(0 < st < TOTAL_STEPS for st in s["traj_ckpts"])
    assert s["primary_repr"] == "repr01"
    assert s["arms_meta"]["repr01"]["role"].startswith("路径机制")


def test_recipe_identical_to_x1_except_steps():
    """2626 三臂与 X1（2622）同名臂**同配方**：除 `train.steps`（6000→24000）外逐字段相同。"""
    for a in ARMS:
        cur = _load(a)
        x1 = json.load(open(os.path.join(X1, "configs13m", f"{a}.json")))
        assert set(cur) == set(x1)
        assert cur["model"] == x1["model"], f"{a} model 段与 X1 不同"
        assert cur["path"] == x1["path"], f"{a} path 段与 X1 不同"
        cur_train = {k: v for k, v in cur["train"].items() if k != "steps"}
        x1_train = {k: v for k, v in x1["train"].items() if k != "steps"}
        assert cur_train == x1_train, f"{a} train 段（除 steps）与 X1 不同"
        assert x1["train"]["steps"] == 6000 and cur["train"]["steps"] == TOTAL_STEPS
