"""F_KEY（复核 B12）的框架单测：**缺证据不得静默跳过**。

钉住三件事：
1. 本地原始证据缺失 ⇒ `pytest.xfail`（出现在汇总里），**不是** `pytest.skip`、也不是通过；
2. 本地证据与入 git 的锚点不一致（被换过/重跑过）⇒ **失败**；
3. 锚点文件/条目缺失、入 git 的产物缺失 ⇒ **失败**（不是 skip）。
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import evidence_anchor                                          # noqa: E402


def test_missing_local_evidence_xfails_not_skips():
    anchors = evidence_anchor.load_anchors()
    with pytest.raises(pytest.xfail.Exception):
        evidence_anchor.local_step0("runs/does-not-exist/train_log.jsonl",
                                    "stability_preflight_step0", anchors)


def test_tampered_local_evidence_fails(tmp_path, monkeypatch):
    """本地首行换了 ⇒ sha256 不符 ⇒ 失败（不是 skip、也不是通过）。"""
    p = tmp_path / "train_log.jsonl"
    p.write_text('{"step": 0, "loss": 9.9, "gnorm": 1.1}\n', encoding="utf-8")
    monkeypatch.setattr(evidence_anchor, "ROOT", str(tmp_path))
    anchors = evidence_anchor.load_anchors()
    with pytest.raises(pytest.fail.Exception):
        evidence_anchor.local_step0("train_log.jsonl", "stability_preflight_step0", anchors)


def test_missing_anchor_file_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence_anchor, "ANCHOR_FILE", str(tmp_path / "nope.json"))
    with pytest.raises(pytest.fail.Exception):
        evidence_anchor.load_anchors()


def test_missing_anchor_entry_fails():
    anchors = evidence_anchor.load_anchors()
    with pytest.raises(pytest.fail.Exception):
        evidence_anchor.anchor(anchors, "no-such-anchor")


def test_tracked_artifact_missing_fails(tmp_path):
    with pytest.raises(pytest.fail.Exception):
        evidence_anchor.require_tracked(str(tmp_path / "missing_config.json"), "入 git 的配置")


def test_anchor_file_is_self_consistent():
    """入 git 的锚点本身必须自洽（两侧 step0 的比对字段相等）。"""
    a = evidence_anchor.load_anchors()
    for left, right in (("stability_preflight_step0", "engine_2623_full1024_step0"),
                        ("clean_full1024_step0", "w1_qknorm_2625_step0")):
        x, y = evidence_anchor.anchor(a, left), evidence_anchor.anchor(a, right)
        assert x["step"] == y["step"] == 0
        assert x["loss"] == y["loss"] and x["gnorm"] == y["gnorm"]
        assert len(x["first_line_sha256"]) == 64 and len(y["first_line_sha256"]) == 64


def test_fkey_files_have_no_silent_skip():
    """F_KEY 点名的两个文件里不得再出现静默 `pytest.skip`。"""
    for name in ("test_longT_stability.py", "test_longT_clean.py"):
        text = open(os.path.join(ROOT, "tests", name), encoding="utf-8").read()
        assert "pytest.skip(" not in text, f"{name} 仍有静默 skip（F_KEY 未清）"
