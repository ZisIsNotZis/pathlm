"""2664 的 checkpoint 指纹门：**复用臂降级、自训臂必崩**，两边都不能被写成同一处置。

为什么这条单测单独存在：`config.json` 是冻结契约，里面 `gqa2_eq/runs[1].sha256` 记成了一串
转写错字（`…b58012`，盘上是 `…b58024`）。修契约会推翻整格「契约先于读数」的定序证据，
所以门改成「复用臂降级为记录、自训臂仍然硬拦」——这条区分一旦被写错，本实验会静默接受
一个**来源不明**的 checkpoint，或者反过来把合法的复用臂拦死。两种写错的构造都钉在这里：

1. `ckpt_sha256_check` 的三态：相符 ⇒ `ok` 且不 `fatal`；复用臂不符 ⇒ 不 `ok` 但不 `fatal`；
   自训臂不符 ⇒ `fatal`。反向样例：把 `reuse:` 判成 `train:` ⇒ 复用臂变 `fatal` ⇒ 红。
2. `load_arm_model` 的实际处置：自训臂不符 ⇒ `SystemExit`（且在装模型之前就停）；复用臂
   不符时**不**由这条门中止（本测不装模型，只看门本身放行）。
3. `run.py::ckpt_sha256_check` 的两个哈希都报出来：`declared` 是契约值、`actual` 是盘上值，
   读数里两个都要能核对（只报一个就等于把不符藏起来）。

全程 CPU、秒级，不取 GPU 锁。

Run: python3 -m pytest tests/test_2664_ckpt_sha256_gate.py -q
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

EXP = os.path.join(ROOT, "axes", "5-quant", "5.4-low-rank", "experiments",
                   "2664-lowrank-vs-quant-13m")
DECLARED = "b03d13091dcb6f96cbadbe27c43da5f4d7a682a2f0bc757f7d48095b01b58012"
ACTUAL = "b03d13091dcb6f96cbadbe27c43da5f4d7a682a2f0bc757f7d48095b01b58024"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run2664 = _load("run_2664_gate_test", "run.py")


def _run(source, sha=DECLARED, ckpt="unused.pt"):
    return {"run": "x", "seed": 0, "source": source, "sha256": sha, "ckpt": ckpt}


def test_ckpt_sha256_check_three_states():
    ok = run2664.ckpt_sha256_check(_run("reuse:2656", sha=ACTUAL), ACTUAL)
    assert ok["ok"] and not ok["fatal"], "指纹相符时既不该 red、也不该 fatal"

    reuse = run2664.ckpt_sha256_check(_run("reuse:2656"), ACTUAL)
    assert not reuse["ok"], "复用臂指纹不符必须记成 sha256_verified=false（不许冒充已校验）"
    assert not reuse["fatal"], "复用臂指纹不符不得中止（契约是冻结的，转写错字不动契约）"
    assert (reuse["declared"], reuse["actual"]) == (DECLARED, ACTUAL), "两个哈希都要报出来"

    trained = run2664.ckpt_sha256_check(_run("train:mla"), ACTUAL)
    assert not trained["ok"] and trained["fatal"], "自训臂指纹不符必须硬拦（拿错了文件）"


def test_no_declared_sha_makes_no_claim():
    bare = run2664.ckpt_sha256_check(_run("train:mla", sha=None), ACTUAL)
    assert bare["ok"] and not bare["fatal"], "没记指纹的计划臂（未训）不该被这条门判红"


def test_reverse_sample_would_be_red():
    """反向样例：把复用臂当成自训臂处置 ⇒ 与真实分支的结论相反 ⇒ 本测会红。"""
    reuse = run2664.ckpt_sha256_check(_run("reuse:2656"), ACTUAL)
    trained = run2664.ckpt_sha256_check(_run("train:mla"), ACTUAL)
    assert reuse["fatal"] != trained["fatal"], "两种来源的处置必须不同，否则这条门没有区分力"


def test_load_arm_model_aborts_before_loading_for_trained_arm():
    """自训臂在装模型之前就被拦；复用臂走到装模型那一步（本测不装，故不调用）。"""
    arm = {"label": "gqa2_eq", "structure": "gqa", "latent": 0, "n_kv_heads": 2,
           "mlp_hidden_sched": None, "params_analytic": 1}
    ckpt = os.path.join(EXP, "runs", "mha-s1", "model.pt")
    if not os.path.exists(ckpt):
        pytest.skip("本实验自己训出来的 checkpoint 不在这台机器上（权重不入库）")
    trained = {"run": "mha-s1", "seed": 1, "source": "train:engine",
               "sha256": DECLARED, "ckpt": os.path.relpath(ckpt, ROOT)}
    with pytest.raises(SystemExit) as err:
        run2664.load_arm_model(arm, trained, "cpu")
    assert "sha256 不符" in str(err.value), f"自训臂不符必须以 SystemExit 报出：{err.value}"
