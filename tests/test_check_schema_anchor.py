"""G1 复核：`check_schema.py` 必须校验证据锚点 `@<sha>` 是**真实提交**。

钉住四件事（都可失败）：
1. 构造假 sha ⇒ 锚点校验**失败并逐条列出**；
2. 真实提交锚点（HEAD）⇒ 通过；
3. git 不可用 ⇒ **显式报错**（不得静默跳过、不得当作通过）；
4. `main()` 确实把 `catalog.yaml` / 课题页里的锚点交给 `check_anchors`。
"""

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_schema                                             # noqa: E402


def test_fake_anchor_fails_and_is_listed():
    errs = check_schema.check_anchors({"deadbeef1234": ["catalog:A2-full-attention"]})
    assert len(errs) == 1, errs
    assert "@deadbeef1234" in errs[0]
    assert "catalog:A2-full-attention" in errs[0]


def test_real_commit_anchor_ok():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert check_schema.check_anchors({head[:9]: ["catalog:HEAD"]}) == []


def test_git_unavailable_is_explicit_error(monkeypatch):
    def boom():
        raise check_schema.GitUnavailable("git: command not found")
    monkeypatch.setattr(check_schema, "reachable_commits", boom)
    errs = check_schema.check_anchors({"deadbeef1234": ["catalog:X"]})
    assert len(errs) == 1 and "git 不可用" in errs[0], errs


def test_main_feeds_anchors_into_check(monkeypatch):
    """main() 必须把目录里的锚点交给 check_anchors（不是只看路径存在）。"""
    real_run = check_schema.subprocess.run
    seen: dict = {}

    def fake_check(anchors):
        seen["anchors"] = anchors
        return []

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and \
                os.path.basename(str(cmd[1])) in ("build_wiki.py", "backfill_legacy.py"):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(check_schema, "check_anchors", fake_check)
    monkeypatch.setattr(check_schema.subprocess, "run", fake_run)
    check_schema.main()
    assert seen.get("anchors"), "main() 没有把任何证据锚点交给 check_anchors"
    assert "e38349f" in seen["anchors"]           # A2 的证据锚点确实被收集
    assert any(loc.startswith("axes/") for loc in seen["anchors"]["e38349f"])
