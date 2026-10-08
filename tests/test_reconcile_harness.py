"""F4（复核 B3）：对账的容差语义、期望值来源，以及「篡改原始计数 ⇒ 对账失败」。

三组断言：
1. `tools.evalkit.reconcile.Ledger` 的容差/来源规则（绝对 vs 相对、golden 要写 why、
   derived 要写 sources、缺容差直接报错）；
2. `2621-depth-draft-e2e/reconcile_numbers.py` 在**未篡改**的输入上 rc == 0，且期望值
   以**重算**为主（`n_derived ≫ n_golden`，golden 只允许历史记录常数）；
3. 篡改一个**原始计数**（`sum_n_accepted` / `cycle_log` 的一行 / `per_prompt.n_drafts`）
   ⇒ 对账必须红（rc != 0）并点名对应检查。
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from tools.evalkit.reconcile import (GoldenWithoutWhy, Ledger,   # noqa: E402
                                     MissingProvenance, MissingTolerance)

EXP = os.path.join(ROOT, "axes", "C-vertical", "C1-layers-per-token",
                   "experiments", "2621-depth-draft-e2e")
SCRIPT = os.path.join(EXP, "reconcile_numbers.py")


# ------------------------------------------------------------- 1. Ledger 规则

def test_tolerance_semantics_abs_and_rel(tmp_path):
    led = Ledger("t", sources=("raw.json",))
    assert led.check("abs ok", 100.0, 100.5, abs_tol=0.6) is True
    assert led.check("abs too tight", 100.0, 100.5, abs_tol=0.4) is False
    assert led.check("rel ok", 100.0, 100.5, rel_tol=5e-3) is True      # 0.5025 ≥ 0.5
    assert led.check("rel too tight", 100.0, 100.5, rel_tol=4e-3) is False
    assert led.finish(str(tmp_path / "r.json")) == 1
    rep = json.load(open(tmp_path / "r.json", encoding="utf-8"))
    assert rep["n_failed"] == 2 and "abs too tight" in rep["failed"]


def test_old_ambiguous_tol_is_rejected():
    """既不给 abs_tol 也不给 rel_tol ⇒ 直接报错（废止 `tol·max(1,|want|)`）。"""
    led = Ledger("t", sources=("raw.json",))
    with pytest.raises(MissingTolerance):
        led.check("no tol", 1.0, 2.0)


def test_provenance_required():
    bare = Ledger("t")
    with pytest.raises(MissingProvenance):
        bare.check("no source", 1.0, 1.0, abs_tol=0.1)
    with pytest.raises(MissingProvenance):
        bare.derived("derived without sources", 1.0, lambda: 1.0, sources=())
    with pytest.raises(GoldenWithoutWhy):
        bare.golden("golden without why", 1.0, 1.0, abs_tol=0.1)
    # 给了 why / sources ⇒ 放行，且 golden 与 derived 分开计数
    bare.golden("g", 1.0, 1.0, abs_tol=0.1, why="历史记录值")
    bare.derived("d", 2.0, lambda: 2.0, sources=("raw.json::x",), abs_tol=1e-9)
    rep = bare.summary()
    assert (rep["n_golden"], rep["n_derived"]) == (1, 1)


# ------------------------------------------------------- 2/3. 真实脚本 + 篡改

def _copy_inputs(tmp_path):
    for name in ("results.json", "cycle_log.jsonl"):
        shutil.copy(os.path.join(EXP, name), os.path.join(tmp_path, name))
    return tmp_path


def _run(rdir):
    return subprocess.run([sys.executable, SCRIPT, str(rdir)],
                          cwd=ROOT, capture_output=True, text=True)


def test_baseline_reconcile_passes_and_is_mostly_derived(tmp_path):
    r = _run(_copy_inputs(tmp_path))
    assert r.returncode == 0, f"{r.stdout[-2000:]}{r.stderr[-2000:]}"
    rep = json.load(open(tmp_path / "reconcile_report.json", encoding="utf-8"))
    assert rep["n_checks"] > 500 and rep["n_failed"] == 0
    assert rep["n_derived"] > 10 * max(rep["n_golden"], 1), \
        "期望值必须以重算为主，不能把 golden 常数当唯一证据"


def test_tampered_raw_count_makes_reconcile_fail(tmp_path):
    """篡改一个原始计数（Σaccepted +1）⇒ 对账必须红，且点名对应检查。"""
    p = _copy_inputs(tmp_path)
    R = json.load(open(os.path.join(p, "results.json"), encoding="utf-8"))
    R["arms"]["single_g4"]["sum_n_accepted"] += 1
    json.dump(R, open(os.path.join(p, "results.json"), "w"), ensure_ascii=False)
    r = _run(p)
    assert r.returncode != 0, "篡改原始计数却没有对账失败"
    assert "acceptance_pooled" in r.stdout
    assert "FAIL" in r.stdout


def test_tampered_cycle_log_makes_reconcile_fail(tmp_path):
    p = _copy_inputs(tmp_path)
    path = os.path.join(p, "cycle_log.jsonl")
    lines = open(path, encoding="utf-8").read().splitlines()
    row = json.loads(lines[0])
    row["emitted"] = row["accepted"] + 2          # 账本恒等式 emitted == accepted+1 被破坏
    lines[0] = json.dumps(row, ensure_ascii=False)
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    r = _run(p)
    assert r.returncode != 0
    assert "emitted == accepted+1" in r.stdout


def test_tampered_drafts_break_gamma_identity(tmp_path):
    p = _copy_inputs(tmp_path)
    R = json.load(open(os.path.join(p, "results.json"), encoding="utf-8"))
    R["arms"]["single_g4"]["per_prompt"][0]["n_drafts"] += 2
    json.dump(R, open(os.path.join(p, "results.json"), "w"), ensure_ascii=False)
    r = _run(p)
    assert r.returncode != 0
    assert "n_drafts == gamma×cycles" in r.stdout
