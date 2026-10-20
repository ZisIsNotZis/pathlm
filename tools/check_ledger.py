#!/usr/bin/env python3
"""check_ledger.py — 迁移台账必须无遗漏、可回溯（可失败）。

**为什么需要它**：一次分层搬迁里，最贵的错误不是搬错一处，而是**某个文件悄悄消失**——
读者在旧位置找不到它，也没人说得清它去了哪里。所以搬迁必须留下台账：**一源文件一行**，
原文 → 新家，或者「删除 + 理由」。本工具把「台账说全了没有」变成机检：

  1. **覆盖**：迁移前树上的每一个元级 / 根级事实文档（`BASE_FILES`）在台账里**恰好一行**；
  2. **不含糊**：台账里不许出现 `BASE_FILES` 之外当作「原文」的路径（那是一个笔误或者一个
     无人认领的文件）；
  3. **新家真在**：每一行的新家必须是**仓库里真实存在的文件**（不是目录、不是将来的路径）；
  4. **删除有理由**：处置为「删除」的行必须写清为什么删（不写理由的删除不许通过）；
  5. **新增有登记**：`docs/*.md` / `atlas/*.md` / `harness/*.md` 里不在 `BASE_FILES` 的文档，
     必须在台账里出现过（作为新家或作为「本次新增」的一行）——新文件也得有出处。

`BASE_FILES` 是**迁移前那一版树的文件清单**（提交 `45869db` 时点的 `docs/**` 与根级 `*.md`）。
它钉在代码里而不是每次从 git 读，是因为 CI 是浅克隆、拿不到那个提交；
它是「搬迁必须交代的账」这一事实本身，不是台账的副本——台账记的是**去向**，本表记的是**欠账**。

用法：
    python3 tools/check_ledger.py            # 打印全部违规；有违规即非零退出
    python3 tools/check_ledger.py --check    # 同上（CI / pytest 入口）
    python3 tools/check_ledger.py --help     # 打印本文件头
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "docs" / "ledger.md"

#: 迁移前那一版树的文件清单（`45869db` 的 `docs/**` 与根级 `*.md`）：每个文件都要有归宿。
BASE_FILES: tuple[str, ...] = (
    "AGENTS.md",
    "CONTRIBUTING.md",
    "FINDINGS.md",
    "LICENSE.md",
    "README.md",
    "SCHEMA.md",
    "STATUS.md",
    "docs/cost-table.md",
    "docs/design.md",
    "docs/engine_capability.md",
    "docs/methodology.md",
    "docs/paper-plan.md",
    "docs/plan.md",
    "docs/playbook.md",
    "docs/repro.md",
    "docs/research-schema.md",
    "docs/structure.md",
    "docs/triage.md",
)

#: 本次新增的文档会被这条 glob 找出来（它们不必有「原文」，但必须有一行登记）。
NEW_DOC_GLOBS = ("docs/*.md", "atlas/*.md", "harness/*.md")

#: 台账里「本次新增」这一列的写法（没有原文的行）。
NEW_MARKER = "（本次新增）"

#: 本工具只判台账的第一节（原文 / 处置 / 说明）；后面各节写的是方法、规则改动
#: 与不判清单，不是逐文件台账，不按这张表解析。
SECTION_START = "## 1"
SECTION_END = "## 2"

#: 处置为删除时必须以这个词开头。
DELETE_WORD = "删除"

#: 一行台账最少这么长才算「写了理由」（删一句话不等于写了理由）。
MIN_REASON = 8

_PATH_RE = re.compile(r"`([^`]+)`")


class Row:
    __slots__ = ("lineno", "source", "disposition", "raw")

    def __init__(self, lineno: int, source: str, disposition: str, raw: str) -> None:
        self.lineno = lineno
        self.source = source
        self.disposition = disposition
        self.raw = raw

    @property
    def is_new(self) -> bool:
        return self.source == NEW_MARKER

    @property
    def is_delete(self) -> bool:
        return self.disposition.startswith(DELETE_WORD)

    @property
    def homes(self) -> list[str]:
        """处置列里出现的仓库内路径（删除行没有新家，但可能点名「并入」的那一处）。"""
        return _PATH_RE.findall(self.disposition)

    @property
    def reason(self) -> str:
        text = self.disposition
        if text.startswith(DELETE_WORD):
            text = text[len(DELETE_WORD):].lstrip("：: ")
        return text.strip()


def _section(text: str) -> list[tuple[int, str]]:
    """台账的逐文件一节：`SECTION_START` 到其后第一个 `SECTION_END` → [(行号, 行)]。"""
    out: list[tuple[int, str]] = []
    inside = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith(SECTION_START):
            inside = True
        elif inside and line.startswith(SECTION_END):
            break
        if inside:
            out.append((lineno, line))
    return out


def parse(text: str) -> tuple[list[Row], list[str]]:
    """把台账正文解析成行 → (行, 结构错误)。只认三列的 Markdown 表。"""
    rows: list[Row] = []
    errors: list[str] = []
    for lineno, line in _section(text):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 3:
            errors.append(f"docs/ledger.md:{lineno} 表格必须有且只有三列（原文 / 处置 / 说明），"
                          f"实际 {len(cells)} 列")
            continue
        if set(cells[0]) <= set("-: ") or cells[0] in ("原文",):
            continue                    # 表头与分隔行
        source = cells[0].strip("`").strip()
        rows.append(Row(lineno, source, cells[1], stripped))
    return rows, errors


def _is_file(rel: str) -> bool:
    return (ROOT / rel).is_file()


def new_docs() -> list[str]:
    """`NEW_DOC_GLOBS` 覆盖的、且在 `BASE_FILES` 之外的文档（仓库相对路径）。"""
    out: set[str] = set()
    for pattern in NEW_DOC_GLOBS:
        for path in ROOT.glob(pattern):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                if rel not in BASE_FILES:
                    out.add(rel)
    return sorted(out)


def check(rows: list[Row]) -> list[str]:
    """台账是否说全了 → 违规列表（空 = 通过）。"""
    problems: list[str] = []
    seen: dict[str, Row] = {}
    for row in rows:
        if row.is_new:
            continue
        if row.source not in BASE_FILES:
            problems.append(f"docs/ledger.md:{row.lineno} 原文 `{row.source}` 不在搬迁前的文件清单里"
                            f"（笔误？还是该登记为「{NEW_MARKER}」？）")
            continue
        if row.source in seen:
            problems.append(f"docs/ledger.md:{row.lineno} 原文 `{row.source}` 重复登记"
                            f"（首次在第 {seen[row.source].lineno} 行；一源文件一行）")
            continue
        seen[row.source] = row

    for rel in BASE_FILES:
        if rel not in seen:
            problems.append(f"台账漏了原文 `{rel}`：搬迁前的每个文件都要有一行（迁到 / 删除 + 理由）")

    for row in rows:
        if row.is_delete:
            if len(row.reason) < MIN_REASON:
                problems.append(f"docs/ledger.md:{row.lineno} `{row.source}` 处置为删除但没有理由"
                                f"（要写清为什么删、事实去了哪里）")
            continue
        homes = row.homes
        if not homes:
            problems.append(f"docs/ledger.md:{row.lineno} `{row.source}` 的处置里没有新家路径"
                            f"（写成 `\\`新家\\`` 或 `{DELETE_WORD}：理由`）")
            continue
        for home in homes:
            if not _is_file(home):
                problems.append(f"docs/ledger.md:{row.lineno} `{row.source}` 的新家 `{home}` 不存在")

    registered = {home for row in rows for home in row.homes}
    for rel in new_docs():
        if rel not in registered:
            problems.append(f"`{rel}` 是本次新增的文档，但台账里没有登记"
                            f"（加一行，原文写「{NEW_MARKER}」）")
    return problems


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip())
        return 0
    if not LEDGER.is_file():
        print(f"FAIL：`docs/ledger.md` 不存在", file=sys.stderr)
        return 1
    rows, errors = parse(LEDGER.read_text(encoding="utf-8"))
    problems = errors + check(rows)
    for p in problems:
        print(p)
    n_new = sum(1 for r in rows if r.is_new)
    n_del = sum(1 for r in rows if r.is_delete)
    print(f"台账：{len(rows) - n_new} 个原文（其中删除 {n_del} 个，其余迁到新家）"
          f"+ 本次新增 {n_new} 个；搬迁前清单 {len(BASE_FILES)} 个文件；{len(problems)} 处违规。")
    if problems:
        print("FAIL：迁移台账不完整或不可回溯。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
