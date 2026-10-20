"""文档引用必须可解析（`tools/check_docs.py` 的入口绑定）。

钉住七件事：
1. 全树（仓库根 `*.md` / `docs/**` / `axes/**`）的 `文件:行`、裸路径与锚点引用**零违规**——不经提示就跑；
2. 注入一条悬空引用（`docs/不存在.md:3`）时检查**必须变红**，删掉后恢复绿（gate probe：证明它抓得住）；
3. 行号越界也算违规（引用对、行号错的形态）；
4. 新增一个事实文档但**不接进文档地图**（`AGENTS.md` §2.1 / `docs/design.md` §4）时检查**必须变红**；
5. 键锚点 `路径::键` 必须解析到**唯一**一处（键不存在 / 命中多处 / JSON 解析失败都是违规）——
   `::status` 这种单段锚点落在**顶层**键上，不会命中 `work_packages[*].status`；
6. 用 `路径:行` 指向 `.yaml` / `.json` 是**禁止**项（行号会随文件增行而漂到别的字段上）；
7. 标题锚点 `路径#标题` / `路径::标题` 必须命中唯一一条标题。
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
    problems, declared_absent, declared_anchor = check_docs.check(probes, suffix_index)
    assert len(problems) == 2, problems
    assert all("行号越界" in p or "行号区间非法" in p for p in problems), problems
    assert declared_absent == [] and declared_anchor == []


def test_absent_marker_is_declared_not_silent(tmp_path):
    """文档自述「未产出」的引用计入已声明缺失，不静默、也不算违规。"""
    ref = check_docs.Reference("axes/6-predict/6.2-speculative-decoding/experiments/"
                               "2675-second-tree-13m/README.md", 88,
                               "test_second_tree.py")
    suffix_index = check_docs._build_suffix_index()
    problems, declared_absent, declared_anchor = check_docs.check([ref], suffix_index)
    assert problems == []
    assert declared_absent == [str(ref)]
    assert declared_anchor == []


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


def _check_one(ref):
    """单条引用过一遍 `check_docs.check`（返回 (违规, 已声明缺失, 已声明悬空锚点)）。"""
    return check_docs.check([ref], check_docs._build_suffix_index())


def test_yaml_key_anchor_resolves_top_level_field():
    """`项目.yaml::status` 落在**顶层** `status` 上，不落到 `work_packages[*].status`（后者是「已完成」）。"""
    rel = "axes/2-depth/2.2.2-attention-residual/project.yaml"
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    lineno, error = check_docs.resolve_yaml_key("\n".join(lines), "status")
    assert error is None, error
    assert lines[lineno - 1] == 'status: "被否证"', lines[lineno - 1]
    problems, _, _ = _check_one(check_docs.Reference("STATUS.md", 1, rel,
                                                     anchor="status", anchor_kind="key"))
    assert problems == [], problems


def test_unknown_key_anchor_turns_red():
    """键不存在 ⇒ 违规，信息里点名那个键。"""
    ref = check_docs.Reference("STATUS.md", 1,
                               "axes/2-depth/2.2.2-attention-residual/project.yaml",
                               anchor="statusX", anchor_kind="key")
    problems, _, _ = _check_one(ref)
    assert len(problems) == 1, problems
    assert "键不存在：`statusX`" in problems[0], problems


def test_duplicate_key_at_same_level_turns_red():
    """同层同名键命中多处 ⇒ 报「键不唯一」并提示用 `a.b` 收紧（不替它猜一个）。"""
    text = "status: A\nowner:\n  name: 1\n  name: 2\n"
    assert check_docs.resolve_yaml_key(text, "status") == (1, None)
    lineno, error = check_docs.resolve_yaml_key(text, "owner.name")
    assert lineno is None, lineno
    assert "键不唯一" in error and "a.b" in error, error


def test_top_level_key_ignores_nested_same_name():
    """`::status` 落在顶层键上：同名嵌套键（`work_packages[*].status`）不参与竞争、不误伤。"""
    ref = check_docs.Reference("STATUS.md", 1,
                               "axes/2-depth/2.2.3-learned-layer-sum/project.yaml",
                               anchor="status", anchor_kind="key")
    problems, _, _ = _check_one(ref)
    assert problems == [], problems


def test_json_key_anchor_and_bad_json_turn_red():
    """JSON 键锚点：存在的键通过，不存在的键与坏 JSON 都违规。"""
    good = check_docs.Reference("STATUS.md", 1,
                                "axes/2-depth/2.2.2-attention-residual/experiments/"
                                "2653-interlayer-attn-13m/smoke_report.json",
                                anchor="B_liveness_gate.per_layer", anchor_kind="key")
    problems, _, _ = _check_one(good)
    assert problems == [], problems
    missing = check_docs.Reference("STATUS.md", 1, "axes/2-depth/2.2.2-attention-residual/"
                                   "experiments/2653-interlayer-attn-13m/smoke_report.json",
                                   anchor="B_liveness_gate.per_layerX", anchor_kind="key")
    problems, _, _ = _check_one(missing)
    assert len(problems) == 1 and "键不存在" in problems[0], problems
    assert check_docs.resolve_json_key("{not json", "a")[1].startswith("JSON 解析失败")


def test_line_reference_to_yaml_json_is_forbidden():
    """`project.yaml:56` ⇒ 违规，信息里写清「改用键锚点」。"""
    ref = check_docs.Reference("STATUS.md", 1,
                               "axes/2-depth/2.2.2-attention-residual/project.yaml", 56, 56)
    problems, _, _ = _check_one(ref)
    assert len(problems) == 1, problems
    assert "禁止行号引用" in problems[0] and "::键" in problems[0], problems
    json_ref = check_docs.Reference("STATUS.md", 1, "axes/2-depth/2.2.2-attention-residual/"
                                    "experiments/2653-interlayer-attn-13m/results.json", 12, 12)
    problems, _, _ = _check_one(json_ref)
    assert len(problems) == 1 and "禁止行号引用" in problems[0], problems


def test_markdown_heading_anchor_turns_red():
    """标题锚点：存在的标题通过，不存在的标题违规（`#` 与 `::` 同一条规矩）。"""
    good = check_docs.Reference("README.md", 1, "docs/cost-table.md",
                                anchor="定律-7--主干被改税就来旁路判断头可以免费",
                                anchor_kind="head")
    problems, _, _ = _check_one(good)
    assert problems == [], problems
    bad = check_docs.Reference("README.md", 1, "docs/cost-table.md",
                               anchor="定律-12--不存在的标题", anchor_kind="head")
    problems, _, _ = _check_one(bad)
    assert len(problems) == 1 and "标题不存在" in problems[0], problems


def test_declared_unresolved_key_anchor_is_counted_not_failing():
    """`UNRESOLVED_KEY_ANCHORS` 里声明的悬空锚点只计数，不在表里的同类锚点仍是违规。"""
    declared = check_docs.UNRESOLVED_KEY_ANCHORS[0]
    source, rest = declared.split(" → ", 1)
    target, anchor = rest.split("::", 1)
    ref = check_docs.Reference(source, 1, target, anchor=anchor, anchor_kind="key")
    problems, _, declared_anchor = _check_one(ref)
    assert problems == [] and declared_anchor == [ref.key], (problems, declared_anchor)
