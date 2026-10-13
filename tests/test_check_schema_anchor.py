"""G1 复核：`check_tree.py` 必须校验证据锚点 `@<sha>` 是**真实提交**。

迁移后 `tools/check_schema.py` 转调 `tools/check_tree.py`；校验对象从 `catalog.yaml`
改为 `axes/**/project.yaml`（SSOT）。本测试钉住四件事（都可失败）：
1. 构造假 sha ⇒ 锚点校验**失败并逐条列出**；
2. 真实提交锚点（HEAD）⇒ 通过；
3. git 不可用 ⇒ **显式报错**（不得静默跳过、不得当作通过）；
4. `main()` 确实把 `project.yaml` 里的锚点交给 `check_anchors` / `check_anchor_syntax`。

G1b（锚点形态不可绕过）：非十六进制锚点不得被 `ANCHOR_RE.findall` 静默吞掉——
`@FINAL` / `@HEAD~1` / `@ticketNN` / 裸 `@` 必须**报错并逐条列出**（`check_anchor_syntax`）。
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_tree  # noqa: E402
import check_schema  # noqa: E402  —— 兼容入口必须仍可导入并导出锚点工具


def test_forwarder_reexports_anchor_helpers():
    """旧入口 check_schema 必须仍导出锚点工具（且是同一个函数对象）。"""
    assert check_schema.check_anchors is check_tree.check_anchors
    assert check_schema.check_anchor_syntax is check_tree.check_anchor_syntax
    assert check_schema.GitUnavailable is check_tree.GitUnavailable


def test_fake_anchor_fails_and_is_listed():
    errs = check_tree.check_anchors({"deadbeef1234": ["project:1.1.1"]})
    assert len(errs) == 1, errs
    assert "@deadbeef1234" in errs[0]
    assert "project:1.1.1" in errs[0]


def test_real_commit_anchor_ok():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert check_tree.check_anchors({head[:9]: ["project:HEAD"]}) == []


def test_git_unavailable_is_explicit_error(monkeypatch):
    def boom():
        raise check_tree.GitUnavailable("git: command not found")
    monkeypatch.setattr(check_tree, "reachable_commits", boom)
    errs = check_tree.check_anchors({"deadbeef1234": ["project:X"]})
    assert len(errs) == 1 and "git 不可用" in errs[0], errs


def test_invalid_non_hex_anchors_are_listed_and_fail():
    """`@FINAL` / `@HEAD~1` / `@ticketNN` ⇒ 报错并逐条列出（此前静默跳过）。"""
    evidences = {"axes/X/X-g/project.yaml @FINAL": ["axes/X/X-g"],
                 "axes/X/X-g/analysis.md @HEAD~1": ["axes/X/X-g"],
                 "axes/X/X-g/results.json @ticketNN": ["project:X-g"]}
    errs = check_tree.check_anchor_syntax(evidences)
    assert len(errs) == 3, errs
    for token in ("@FINAL", "@HEAD~1", "@ticketNN"):
        hit = [e for e in errs if token in e]
        assert len(hit) == 1, (token, errs)
    assert "project:X-g" in [e for e in errs if "@ticketNN" in e][0]


def test_valid_commit_prefix_passes_syntax_check():
    """真实 sha 前缀（HEAD 缩写）⇒ 形态校验通过。"""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert check_tree.check_anchor_syntax({f"axes/X/X-g/analysis.md @{head[:9]}": ["axes/X/X-g"]}) == []


def test_bare_at_sign_is_flagged():
    """裸 `@`（其后没有任何提交前缀）也必须报错，不得静默跳过。"""
    errs = check_tree.check_anchor_syntax({"axes/X/X-g/analysis.md @": ["axes/X/X-g"]})
    assert len(errs) == 1 and "@" in errs[0], errs


def test_main_feeds_evidence_into_syntax_check(monkeypatch):
    """`main()` 必须把 project.yaml 的 evidence 串交给 `check_anchor_syntax`。"""
    seen: dict = {}

    def fake_syntax(evidences):
        seen["evidences"] = evidences
        return []

    monkeypatch.setattr(check_tree, "check_anchor_syntax", fake_syntax)
    check_tree.main()
    ev = seen.get("evidences")
    assert ev, "main() 没有把任何 evidence 串交给 check_anchor_syntax"
    # 示例 1.1.1 的证据串确实被收集
    assert ".scratch/07-x1-x3/evidence @ce2610a0d" in ev


def test_main_feeds_anchors_into_check(monkeypatch):
    """main() 必须把 project.yaml 里的锚点交给 check_anchors（不是只看路径存在）。"""
    seen: dict = {}

    def fake_check(anchors):
        seen["anchors"] = anchors
        return []

    monkeypatch.setattr(check_tree, "check_anchors", fake_check)
    check_tree.main()
    assert seen.get("anchors"), "main() 没有把任何证据锚点交给 check_anchors"
    assert "ce2610a0d" in seen["anchors"] or any(
        a.startswith("ce2610a0d") for a in seen["anchors"])
