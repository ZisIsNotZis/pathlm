"""F3（复核 B4）：变异框架的分类、判红与非零退出。

钉住四件事：
1. `classify_pytest` 把 **测试失败 / 收集错误 / 环境错误 / 通过** 分得开（合成 + 真实调用）；
2. 收集错误与环境错误**不算抓红**，且 `pytest_run` 直接抛 `ClassifiedError`；
3. **未抓到 ⇒ 非零退出**（端到端：生成的实验脚本跑一个抓不到的变异 ⇒ rc != 0；
   抓得到的变异 ⇒ rc == 0）；
4. **全仓统一形制**：每个 `*/mutate.py` 都接 `tools.evalkit.mutation`，且不再直接调
   `subprocess.run(pytest)`。
"""

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from tools.evalkit import mutation                              # noqa: E402


# --------------------------------------------------------------- 1. 分类

@pytest.mark.parametrize("rc,stdout,stderr,want", [
    (0, "15 passed in 1.0s", "", mutation.PASSED),
    (1, "1 failed, 14 passed in 1.0s", "", mutation.TEST_FAILURE),
    (1, "ERROR tests/test_x.py", "errors during collection", mutation.COLLECTION_ERROR),
    (2, "!!! Interrupted: 1 error during collection !!!", "", mutation.COLLECTION_ERROR),
    (5, "no tests ran in 0.01s", "", mutation.COLLECTION_ERROR),
    (4, "ERROR: usage: pytest [options]", "", mutation.ENVIRONMENT_ERROR),
    (3, "INTERNALERROR> Traceback", "", mutation.ENVIRONMENT_ERROR),
    (127, "", "command not found", mutation.ENVIRONMENT_ERROR),
])
def test_classify_synthetic(rc, stdout, stderr, want):
    assert mutation.classify_pytest(rc, stdout, stderr) == want


def test_classify_real_pytest_runs(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (tmp_path / "test_bad.py").write_text("def test_bad():\n    assert False\n", encoding="utf-8")
    (tmp_path / "test_import_error.py").write_text(
        "import definitely_not_installed_xyz  # noqa: F401\n\n\ndef test_x():\n    assert True\n",
        encoding="utf-8")

    ok = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", "test_ok.py"], cwd=tmp_path)
    assert ok.returncode == 0

    bad = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", "test_bad.py"], cwd=tmp_path)
    assert bad.returncode == 1                       # 只有 test_failure 才返回非零

    with pytest.raises(mutation.ClassifiedError) as ei:
        mutation.pytest_run([sys.executable, "-m", "pytest", "-q", "test_import_error.py"],
                            cwd=tmp_path)
    assert ei.value.status == mutation.COLLECTION_ERROR

    with pytest.raises(mutation.ClassifiedError) as ei2:
        mutation.pytest_run([sys.executable, "-m", "pytest", "--no-such-flag"], cwd=tmp_path)
    assert ei2.value.status == mutation.ENVIRONMENT_ERROR

    with pytest.raises(mutation.ClassifiedError) as ei3:
        mutation.pytest_run(["/definitely/not/python", "-m", "pytest"], cwd=tmp_path)
    assert ei3.value.status == mutation.ENVIRONMENT_ERROR


def test_inprocess_outcome_classification():
    def _boom():
        raise AssertionError("抓红")

    def _ok():
        return None

    def _env():
        raise ModuleNotFoundError("no module named 'torch'")

    assert mutation.inprocess_outcome(_boom) == (True, "抓红", mutation.TEST_FAILURE)
    assert mutation.inprocess_outcome(_ok)[0] is False
    with pytest.raises(mutation.ClassifiedError) as ei:
        mutation.inprocess_outcome(_env)
    assert ei.value.status == mutation.ENVIRONMENT_ERROR


# --------------------------------------------------------------- 3. 非零退出（端到端）

SCRIPT = textwrap.dedent('''
    import json, os, sys
    ROOT = {root!r}
    sys.path.insert(0, ROOT)
    from tools.evalkit import mutation
    HERE = os.path.dirname(os.path.abspath(__file__))
    TARGET = os.path.join(HERE, "target_mod.py")
    original = open(TARGET, encoding="utf-8").read()
    report = {{"mutations": []}}
    ok = True
    try:
        for mid, old, new, tests in MUTATIONS:
            assert original.count(old) == 1, mid
            open(TARGET, "w", encoding="utf-8").write(original.replace(old, new))
            r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", *tests], cwd=HERE)
            caught = r.returncode != 0
            ok = ok and caught
            report["mutations"].append({{"id": mid, "caught_red": caught}})
            open(TARGET, "w", encoding="utf-8").write(original)
    finally:
        open(TARGET, "w", encoding="utf-8").write(original)
    report["all_caught"] = ok
    mutation.exit_if_uncaught(ok, report, os.path.join(HERE, "mutation_report.json"))
''')


def _make_experiment(tmp_path, mutations_src: str):
    (tmp_path / "target_mod.py").write_text("def add_one(x):\n    return x + 1\n",
                                            encoding="utf-8")
    (tmp_path / "test_target.py").write_text(
        "from target_mod import add_one\n\n\ndef test_add():\n    assert add_one(1) == 2\n",
        encoding="utf-8")
    (tmp_path / "gen_mutate.py").write_text(
        "MUTATIONS = " + mutations_src + "\n" + SCRIPT.format(root=ROOT), encoding="utf-8")


def test_uncaught_mutation_exits_nonzero(tmp_path):
    """变异抓不到（把 1 改成 1，等价恒等）⇒ 必须非零退出。"""
    _make_experiment(tmp_path, repr([("M_UNCAUGHT", "return x + 1", "return x + 1",
                                      ["test_target.py"])]))
    r = subprocess.run([sys.executable, "gen_mutate.py"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode != 0, f"未抓到却没有非零退出：{r.stdout}{r.stderr}"
    assert "FAIL" in (r.stdout + r.stderr)


def test_caught_mutation_exits_zero_and_restores(tmp_path):
    _make_experiment(tmp_path, repr([("M_CAUGHT", "return x + 1", "return x + 99",
                                      ["test_target.py"])]))
    r = subprocess.run([sys.executable, "gen_mutate.py"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"{r.stdout}{r.stderr}"
    assert open(tmp_path / "target_mod.py", encoding="utf-8").read() == \
        "def add_one(x):\n    return x + 1\n", "变异后必须逐字节还原"


# --------------------------------------------------------------- 4. 全仓统一形制

def test_all_repo_mutate_scripts_use_the_harness():
    scripts = mutation.hunt_repo_mutate_scripts(ROOT)
    assert len(scripts) >= 39, f"只找到 {len(scripts)} 个 mutate.py"
    bad_harness, bad_subprocess, bad_exit = [], [], []
    for rel in scripts:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        if "from tools.evalkit import mutation" not in src:
            bad_harness.append(rel)
        if "subprocess.run(" in src:
            bad_subprocess.append(rel)
        if not any(k in src for k in ("mutation.exit_if_uncaught(", "mutation.pytest_main(",
                                      "raise SystemExit", "sys.exit(0 if", "return 0 if",
                                      "assert report[", "assert all_")):
            bad_exit.append(rel)
    assert bad_harness == [], f"未接入统一框架：{bad_harness}"
    assert bad_subprocess == [], f"仍直接调 subprocess.run：{bad_subprocess}"
    assert bad_exit == [], f"缺少「未抓到 ⇒ 非零退出」路径：{bad_exit}"
