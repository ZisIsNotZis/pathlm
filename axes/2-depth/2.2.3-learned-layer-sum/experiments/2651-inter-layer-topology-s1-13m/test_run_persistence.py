"""2651 · `run.py` 修复与增量落盘的回归测试（CPU，无 GPU）。

钉住三件上一轮真实崩过 / 本轮改过的事：

1. `train_arm` 收尾**不得**再去读导师产物 `runs/<arm>/results.json` 的 `bpc` 键
   （上一轮 `run.py:405 KeyError: 'bpc'` 的根因；该文件本就没有 `bpc`）；
2. `run_battery` 在 `depth_curve` 末项缺 `bpc` 时**显式报错**（带自述消息），
   不静默填 0；`clean_bpc` 的唯一来源 = `depth_curve[-1]['bpc']`；
3. `dump_results` / `append_run_log` 真的把**逐臂**结果落盘（`completed_arms` /
   `pending_arms` 如实反映当前进度）。

用法：`python3 -m pytest test_run_persistence.py -q`（在仓库根运行）。
"""

import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("run2651", os.path.join(HERE, "run.py"))
run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run)


def test_train_arm_no_longer_reads_results_json_bpc():
    """静态回归：`run.py` 全文不得再出现 `res['bpc']` / `res["bpc"]`
    （导师产物 `results.json` 没有 `bpc` 键；该取值是上一轮崩溃点）。"""
    src = open(os.path.join(HERE, "run.py")).read()
    assert "res['bpc']" not in src, "train_arm 又在读导师产物的 bpc 键（上一轮崩溃点）"
    assert 'res["bpc"]' not in src, "train_arm 又在读导师产物的 bpc 键（上一轮崩溃点）"


def test_run_battery_missing_bpc_key_raises_explicit_error(monkeypatch):
    """`depth_curve` 末项缺 `bpc` ⇒ 显式 KeyError（用自己的消息，不是裸 `'bpc'`），
    不静默填 0。消息断言可失败：若本轮的显式守卫被删掉，消息会退回裸 `'bpc'`。"""
    monkeypatch.setattr(run, "depth_curve",
                        lambda *a, **k: [{"depth": 8, "acc": 0.5}])
    with pytest.raises(KeyError) as ei:
        run.run_battery(None, None, 1)
    assert "depth_curve 末项缺 'bpc' 键" in str(ei.value)


def test_dump_results_reports_completed_and_pending(tmp_path, monkeypatch):
    """增量落盘：只完成一臂时 `results.json` 已可读，且 completed/pending 如实。"""
    monkeypatch.setattr(run, "HERE", str(tmp_path))
    out = {"arms": {"sequential": {"clean_bpc": 1.5, "tf@4": 0.45,
                                   "params": 6962688}}}
    run.dump_results(out, note="增量：完成 sequential 后落盘")
    path = tmp_path / "results.json"
    assert path.exists(), "dump_results 未落盘"
    got = json.loads(path.read_text())
    assert got["completed_arms"] == ["sequential"]
    assert set(got["pending_arms"]) == {"parallel", "parallel-dual-ln", "dense"}
    assert got["partial_note"] == "增量：完成 sequential 后落盘"
    assert got["arms"]["sequential"]["clean_bpc"] == 1.5


def test_append_run_log_appends_one_line(tmp_path, monkeypatch):
    """逐臂流水：append 语义（不覆盖），每臂一行。"""
    log = tmp_path / "run_log.txt"
    monkeypatch.setattr(run, "RUN_LOG", str(log))
    run.append_run_log("[arm] sequential clean_bpc=1.5")
    run.append_run_log("[arm] parallel clean_bpc=1.57")
    lines = log.read_text().splitlines()
    assert lines == ["[arm] sequential clean_bpc=1.5",
                     "[arm] parallel clean_bpc=1.57"]
