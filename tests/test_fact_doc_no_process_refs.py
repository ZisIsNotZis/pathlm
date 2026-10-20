"""事实文档不带变更叙事，也不带内部工单号（`AGENTS.md` §2.1）。

`project.yaml` 只写当前状态；过程史（谁在哪一轮改了什么）的家是
`.scratch/doc-log/axes/<轴>/<格子>/CHANGELOG.md`。这里钉四件事，每件都能红：

1. `change_log` / `actor` 键出现即报错（该形态已退役）；
2. 字符串值里的 `ticket NN` / `NN-review` / `WP-xxxx` 出现即报错；
3. 报错信息把读者指向 `.scratch/doc-log/**`，而不是只说「非法」；
4. 角色名 `axis1-reviewer` 这类合法字符串**不**被误抓（`-review` 只认两位数字前缀）。
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_tree  # noqa: E402


def test_change_log_key_is_refused():
    errs = check_tree.check_no_change_narrative(
        {"change_log": [{"date": "2026-01-01", "actor": "w", "change": "x"}]}, "axes/a/b.yaml")
    assert any("change_log" in e and "已退役" in e for e in errs), errs
    assert any(".scratch/doc-log" in e for e in errs), errs


def test_actor_key_is_refused():
    errs = check_tree.check_no_change_narrative({"work_packages": [{"actor": "w"}]}, "axes/a/b.yaml")
    assert any("actor" in e and "已退役" in e for e in errs), errs


def test_process_ids_in_text_are_refused():
    doc = {"gap": "按 55-review W-08 改口径", "gaps": ["ticket 62 §10.3 待办", "WP-2652 收口"]}
    errs = check_tree.check_no_change_narrative(doc, "axes/a/b.yaml")
    assert len(errs) == 3, errs
    assert any("'55-review'" in e for e in errs), errs
    assert any("'ticket 62'" in e for e in errs), errs
    assert any("'WP-2652'" in e for e in errs), errs


def test_role_names_are_not_false_positives():
    doc = {"role_map": {"reviewer": "axis1-reviewer（fresh 上下文）"},
           "conclusion_review": "按复核意见改读数口径"}
    assert check_tree.check_no_change_narrative(doc, "axes/a/b.yaml") == []
