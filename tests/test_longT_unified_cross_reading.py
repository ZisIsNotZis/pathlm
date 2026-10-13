"""2636 · 判据 C5（跨口径差）的可失败自测：该块必须与**两个分量**、**两个门限**一致。

背景：跨口径差（T=256 → 0.012210、T=512 → 0.010573）此前只存在于 `analysis.md` 的两个
分量手算里，**不是字段、不被对账覆盖** ⇒ 下次改门限时下游比较句会静默失配（SCHEMA §24
第 3、11 类）。本文件把 `reconcile_numbers.py` **当交付入口**（文档化的调用方式）来跑：

1. 未篡改的输入 ⇒ rc 0 且 `n_fail == 0`，且 C5 的检查真的出现在报告里；
2. 篡改 `cross_diff`（与两个分量脱钩）⇒ 必须红并点名该检查；
3. 只改 `cell_criterion_noise`（模拟「下调/改门限却没同步下游比较句」）⇒ 必须红。
"""

import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EXP = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                   "experiments", "2636-longT-unified")
SCRIPT = os.path.join(EXP, "reconcile_numbers.py")
INPUTS = ("results.json", "eval.json", "analysis.md")


def _copy_inputs(tmp_path):
    for name in INPUTS:
        shutil.copy(os.path.join(EXP, name), os.path.join(tmp_path, name))
    os.makedirs(os.path.join(tmp_path, "smoke"), exist_ok=True)
    shutil.copy(os.path.join(EXP, "smoke", "smoke.json"),
                os.path.join(tmp_path, "smoke", "smoke.json"))
    return tmp_path


def _run(rdir):
    return subprocess.run([sys.executable, SCRIPT, str(rdir)],
                          cwd=ROOT, capture_output=True, text=True)


def _set_cross_field(rdir, T, key, value):
    p = os.path.join(rdir, "results.json")
    R = json.load(open(p, encoding="utf-8"))
    R["criteria"]["criterion_5_cross_reading"]["per_T"][str(T)][key] = value
    json.dump(R, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def test_baseline_cross_reading_reconciles_clean(tmp_path):
    r = _run(_copy_inputs(tmp_path))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.load(open(os.path.join(tmp_path, "reconcile_report.json"), encoding="utf-8"))
    assert rep["n_fail"] == 0
    labels = [c["label"] for c in rep["checks"]]
    assert "cross@256 cross_diff == Δ_step − Δ_token" in labels
    assert "cross@256 cell_criterion_noise == C4 Δ 跨 seed 极差" in labels


def test_tampered_cross_diff_makes_reconcile_fail(tmp_path):
    """差值块与两个分量脱钩 ⇒ 对账必须红。"""
    p = _copy_inputs(tmp_path)
    _set_cross_field(p, 256, "cross_diff", 0.0)
    r = _run(p)
    assert r.returncode != 0, "篡改跨口径差却没有对账失败"
    assert "cross@256 cross_diff" in r.stdout
    assert "MISMATCH" in r.stdout


def test_changed_threshold_without_updating_sentence_fails(tmp_path):
    """只改门限（`cell_criterion_noise`）而块里的比较结论没跟着改 ⇒ 必须红。"""
    p = _copy_inputs(tmp_path)
    _set_cross_field(p, 256, "cell_criterion_noise", 0.014)
    r = _run(p)
    assert r.returncode != 0, "门限被改却没有对账失败"
    assert "cross@256 cell_criterion_noise" in r.stdout
    assert "cross@256 exceeds_cell_criterion_noise" in r.stdout
