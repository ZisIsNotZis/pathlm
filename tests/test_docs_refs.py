"""文档引用必须可解析（`tools/check_docs.py` 的入口绑定）。

钉住四件事：
1. 全树（仓库根 `*.md` / `docs/**` / `axes/**`）的 `文件:行` 与裸路径引用**零违规**——不经提示就跑；
2. 注入一条悬空引用（`docs/不存在.md:3`）时检查**必须变红**，删掉后恢复绿（gate probe：证明它抓得住）；
3. 行号越界也算违规（引用对、行号错的形态）；
4. 新增一个事实文档但**不接进文档地图**（`AGENTS.md` §2.1 / `docs/design.md` §4）时检查**必须变红**。
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import check_docs  # noqa: E402


def test_all_doc_references_resolve(capsys):
    """全树零违规，且扫描面不为空（`扫描 0 个文档` 也算失败）。"""
    rc = check_docs.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 处违规" in out, out
    scanned = int(out.split("扫描 ")[1].split(" 个文档")[0])
    checked = int(out.split("检查 ")[1].split(" 处引用")[0])
    assert scanned > 100 and checked > 100, out


def test_dangling_reference_turns_red(tmp_path, monkeypatch, capsys):
    """注入悬空引用 ⇒ 非零退出并报出「目标不存在」；移除后恢复绿。"""
    doc = ROOT / "docs" / "methodology.md"
    original = doc.read_text(encoding="utf-8")
    bad = "docs/不存在.md:3"
    try:
        doc.write_text(original + f"\n悬空引用探针：`{bad}`。\n", encoding="utf-8")
        rc = check_docs.main(["--check"])
        out = capsys.readouterr().out
        assert rc == 1, out
        assert "docs/不存在.md:3（目标不存在）" in out, out
    finally:
        doc.write_text(original, encoding="utf-8")
    rc = check_docs.main(["--check"])
    assert rc == 0, capsys.readouterr().out


def test_out_of_range_line_turns_red(tmp_path):
    """路径存在但行号越界 ⇒ 违规（`行号越界`），不是通过。"""
    probes = [
        check_docs.Reference("docs/methodology.md", 1, "docs/plan.md", 99999, 99999),
        check_docs.Reference("docs/methodology.md", 1, "docs/plan.md", 5, 3),
    ]
    suffix_index = check_docs._build_suffix_index()
    problems, declared_absent = check_docs.check(probes, suffix_index)
    assert len(problems) == 2, problems
    assert all("行号越界" in p or "行号区间非法" in p for p in problems), problems
    assert declared_absent == []


def test_absent_marker_is_declared_not_silent(tmp_path):
    """文档自述「未产出」的引用计入已声明缺失，不静默、也不算违规。"""
    ref = check_docs.Reference("axes/6-predict/6.2-speculative-decoding/experiments/"
                               "2675-second-tree-13m/README.md", 88,
                               "test_second_tree.py")
    suffix_index = check_docs._build_suffix_index()
    problems, declared_absent = check_docs.check([ref], suffix_index)
    assert problems == []
    assert declared_absent == [str(ref)]


def test_unmapped_fact_doc_turns_red(capsys):
    """新增事实文档却不接进文档地图 ⇒ 非零退出并点名漏了谁；删掉后恢复绿。"""
    probe = ROOT / "docs" / "未入地图的事实文档.md"
    try:
        probe.write_text("# 未入地图的临时文档\n", encoding="utf-8")
        rc = check_docs.main(["--check"])
        out = capsys.readouterr().out
        assert rc == 1, out
        assert "文档地图：`docs/未入地图的事实文档.md`" in out, out
    finally:
        probe.unlink()
    assert check_docs.main(["--check"]) == 0, capsys.readouterr().out
