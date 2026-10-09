"""G1 复核：`check_schema.py` 必须校验证据锚点 `@<sha>` 是**真实提交**。

钉住四件事（都可失败）：
1. 构造假 sha ⇒ 锚点校验**失败并逐条列出**；
2. 真实提交锚点（HEAD）⇒ 通过；
3. git 不可用 ⇒ **显式报错**（不得静默跳过、不得当作通过）；
4. `main()` 确实把 `catalog.yaml` / 课题页里的锚点交给 `check_anchors`。

G1b（锚点形态不可绕过）：非十六进制锚点不得被 `ANCHOR_RE.findall` 静默吞掉——
`@FINAL` / `@HEAD~1` / `@ticketNN` / 裸 `@` 必须**报错并逐条列出**（`check_anchor_syntax`），
`main()` 必须把它接在可达性校验前面。
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


def test_invalid_non_hex_anchors_are_listed_and_fail():
    """`@FINAL` / `@HEAD~1` / `@ticketNN` ⇒ 报错并逐条列出（此前静默跳过）。"""
    evidences = {"axes/X/X-g/README.md @FINAL": ["axes/X/X-g"],
                 "axes/X/X-g/analysis.md @HEAD~1": ["axes/X/X-g"],
                 "axes/X/X-g/results.json @ticketNN": ["catalog:X-g"]}
    errs = check_schema.check_anchor_syntax(evidences)
    assert len(errs) == 3, errs
    for token in ("@FINAL", "@HEAD~1", "@ticketNN"):
        hit = [e for e in errs if token in e]
        assert len(hit) == 1, (token, errs)
    assert "catalog:X-g" in [e for e in errs if "@ticketNN" in e][0]


def test_valid_commit_prefix_passes_syntax_check():
    """真实 sha 前缀（HEAD 缩写）⇒ 形态校验通过。"""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()
    assert check_schema.check_anchor_syntax({f"axes/X/X-g/analysis.md @{head[:9]}": ["axes/X/X-g"]}) == []


def test_bare_at_sign_is_flagged():
    """裸 `@`（其后没有任何提交前缀）也必须报错，不得静默跳过。"""
    errs = check_schema.check_anchor_syntax({"axes/X/X-g/analysis.md @": ["axes/X/X-g"]})
    assert len(errs) == 1 and "@" in errs[0], errs


def test_main_feeds_evidence_into_syntax_check(monkeypatch):
    """`main()` 必须把 catalog/课题页的 evidence 串交给 `check_anchor_syntax`。"""
    real_run = check_schema.subprocess.run
    seen: dict = {}

    def fake_syntax(evidences):
        seen["evidences"] = evidences
        return []

    def fake_run(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and len(cmd) > 1 and \
                os.path.basename(str(cmd[1])) in ("build_wiki.py", "backfill_legacy.py"):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(check_schema, "check_anchor_syntax", fake_syntax)
    monkeypatch.setattr(check_schema.subprocess, "run", fake_run)
    check_schema.main()
    ev = seen.get("evidences")
    assert ev, "main() 没有把任何 evidence 串交给 check_anchor_syntax"
    assert ".scratch/04-m1-runs/evidence/B0 @e38349f" in ev      # A2 的证据串确实被收集
    assert "axes/A-mixing/A2-full-attention" in ev[".scratch/04-m1-runs/evidence/B0 @e38349f"]


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
