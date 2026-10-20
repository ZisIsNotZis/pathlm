"""迁移台账 `docs/ledger.md` 与它的机检（`tools/check_ledger.py`）。

钉住五件事，每件都可失败：
1. **覆盖**：迁移前的每个元级 / 根级事实文档在台账里恰好一行（漏一个就红）；
2. **新家真在**：每行的新家是仓库里真实存在的文件（写成将来的路径就红）；
3. **删除有理由**：处置为「删除」的行必须写清理由（空理由 / 一句话就红）；
4. **新增有登记**：本次新增的 `docs/*.md` / `atlas/*.md` / `harness/*.md` 都在台账里出现过；
5. **反向对照**：一份完整台账必须通过（否则上面四条红是假的）。
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_ledger  # noqa: E402


def _rows():
    rows, errors = check_ledger.parse(
        check_ledger.LEDGER.read_text(encoding="utf-8"))
    assert not errors, errors
    return rows


def _row(rows, source):
    hits = [r for r in rows if r.source == source]
    assert len(hits) == 1, f"台账里 `{source}` 有 {len(hits)} 行"
    return hits[0]


def _replace(rows, source, disposition):
    out = []
    for r in rows:
        out.append(check_ledger.Row(r.lineno, r.source, disposition, r.raw)
                   if r.source == source else r)
    return out


def test_ledger_is_complete_and_resolvable():
    rows = _rows()
    assert check_ledger.check(rows) == [], check_ledger.check(rows)
    assert check_ledger.main(["--check"]) == 0


def test_every_base_file_has_exactly_one_row():
    rows = _rows()
    seen = [r.source for r in rows if not r.is_new]
    missing = [f for f in check_ledger.BASE_FILES if f not in seen]
    assert missing == [], f"台账漏了原文：{missing}"
    assert len(seen) == len(set(seen)), "某个原文登记了两次"
    assert sorted(seen) == sorted(check_ledger.BASE_FILES), (
        "台账里有搬迁前清单之外的原文（笔误？还是该写成「本次新增」？）")


def test_dropping_a_row_is_red():
    rows = [r for r in _rows() if r.source != "docs/playbook.md"]
    problems = check_ledger.check(rows)
    assert any("docs/playbook.md" in p and "漏了" in p for p in problems), problems


def test_missing_new_home_is_red():
    rows = _replace(_rows(), "docs/triage.md", "迁到 `atlas/no-such-file.md`")
    problems = check_ledger.check(rows)
    assert any("atlas/no-such-file.md" in p and "不存在" in p for p in problems), problems


def test_directory_as_new_home_is_red():
    """新家是目录（不是文件）也算没落地——目录里没有那份事实。"""
    rows = _replace(_rows(), "docs/design.md", "迁到 `atlas`")
    problems = check_ledger.check(rows)
    assert any("atlas" in p and "不存在" in p for p in problems), problems


def test_deletion_without_reason_is_red():
    rows = _replace(_rows(), "STATUS.md", "删除")
    problems = check_ledger.check(rows)
    assert any("STATUS.md" in p and "理由" in p for p in problems), problems


def test_invented_source_is_red():
    rows = _rows() + [check_ledger.Row(0, "docs/no-such-source.md", "迁到 `atlas/design.md`", "")]
    problems = check_ledger.check(rows)
    assert any("docs/no-such-source.md" in p and "不在搬迁前" in p for p in problems), problems


def test_unregistered_new_doc_is_red():
    rows = [r for r in _rows() if "docs/catalog.md" not in r.disposition]
    problems = check_ledger.check(rows)
    assert any("docs/catalog.md" in p and "没有登记" in p for p in problems), problems


def test_new_doc_globs_cover_the_three_layers():
    docs = check_ledger.new_docs()
    for rel in ("docs/catalog.md", "docs/ledger.md", "atlas/README.md", "harness/README.md"):
        assert rel in docs, f"{rel} 没被新增文档的 glob 覆盖到：{docs}"


def test_second_section_is_not_parsed_as_ledger_rows():
    """§2 记的是「与旧规则冲突的处置」，不是逐文件台账，不许被当成原文行。"""
    text = check_ledger.LEDGER.read_text(encoding="utf-8")
    rows, errors = check_ledger.parse(text)
    assert not errors, errors
    assert not any(r.source.startswith("`") for r in rows)
    assert not any("| 位置 |" in r.raw for r in rows)
