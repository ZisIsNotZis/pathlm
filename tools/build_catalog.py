#!/usr/bin/env python3
"""build_catalog.py — 从 `axes/**/project.yaml` 生成 `docs/catalog.md`（元级研究目录）。

`docs/catalog.md` 是**元级唯一的研究目录**：一行一个研究项，七列
（id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针），行数恒等于 `project.yaml` 的个数。
它回答的只有一个问题：**树上有哪些研究项、每项的契约与状态是什么、去哪里读全文**——
实验读数、判据推导、逐格分析一概不进来（那是研究级 `axes/**` 与 `atlas/**` 的家）。

**本文件是生成物（render）**：每一格都逐字取自该研究项的 `project.yaml`，
`目的` / `验收` / `期望产出` 三列超长时按 `CELL_WIDTHS` 截断并加 `…`（全文看 `project.yaml`，
本表只做定位）。生成器不重算、不换算口径、不新造一个字。

用法：
    python3 tools/build_catalog.py          # 写 docs/catalog.md
    python3 tools/build_catalog.py --check  # 与磁盘逐字节比对；手改或源变化未重生 ⇒ 非零退出
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"
OUT = ROOT / "docs" / "catalog.md"

#: 七列（顺序即契约里的顺序，多一列少一列都算改契约）。
COLUMNS = ("id", "标题", "目的", "验收", "期望产出", "状态", "指针")

#: 每列在表里的截断宽度（字符）。`id` / `状态` / `指针` 不截断——它们是定位与判读用的，
#: 截断会让目录读不成目录；`标题` 也不截（研究项名的 exact wording 不得改）。
CELL_WIDTHS = {"目的": 60, "验收": 70, "期望产出": 55}

BANNER = ("<!-- 本文件由 `tools/build_catalog.py` 从 `axes/**/project.yaml` 生成，禁止手改；"
          "改事实请改 project.yaml -->")

INTRO = [
    "# 研究目录（一行一个研究项）",
    "",
    "> 本文件由 `tools/build_catalog.py` 从 `axes/**/project.yaml` 生成，**禁止手改**；"
    "改事实请改 `project.yaml`（字段契约见 `harness/research-schema.md`）。",
    "> 七列：**id** / **标题** / **目的** / **验收**（= `milestone.acceptance`）/ "
    "**期望产出**（= `milestone.output`）/ **状态**（= `status`）/ **指针**（该研究的 SSOT 入口）。",
    "> 超长的 `目的` / `验收` / `期望产出` 按固定宽度截断并加 `…`：**全文只在家里**"
    "（`project.yaml`），本表只做定位，不复制读数——实验读数、判据推导、逐格分析在"
    "`axes/**/experiments/**` 与 `atlas/**`。",
    "> 研究项事实的 SSOT 是同目录的 `project.yaml`（本表七列就是由它逐字生成的）；"
    "`RESEARCH.md` 只在该研究**已有研究级 Agent 接手**时才存在，写的是派发口径而不是事实，本表不转发它。",
    "",
]

#: 一个研究项必须带的字段（缺一列就生成不出目录，宁可失败也不补空值）。
REQUIRED = ("id", "title", "purpose", "status", "dir")
REQUIRED_MILESTONE = ("acceptance", "output")


def _cell(value: object, width: int | None) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())                      # 折行 / 多空格 → 单空格
    text = text.replace("|", "\\|")                    # 表格管道符转义
    if width is not None and len(text) > width:
        text = text[: width - 1] + "…"
    return text or "—"


def id_key(item_id: str) -> tuple:
    """研究项 id 的自然序（`1.1.4.10` 排在 `1.1.4.9` 之后）。"""
    out: list[tuple[int, object]] = []
    for part in str(item_id).split("."):
        out.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(out)


def collect() -> tuple[list[dict], list[str]]:
    """扫描全部 `project.yaml` → (行字典, 错误)。顺序 = id 自然序。"""
    rows: list[dict] = []
    errors: list[str] = []
    root = AXES.parent                      # 仓库根（测试会把 AXES 指到临时树）
    for path in sorted(AXES.rglob("project.yaml")):
        rel = path.relative_to(root).as_posix()
        doc = _load(path, rel, errors)
        if doc is None:
            continue
        missing = [k for k in REQUIRED if not doc.get(k)]
        milestone = doc.get("milestone") or {}
        if not isinstance(milestone, dict):
            errors.append(f"{rel}: `milestone` 必须是映射（缺它就取不到验收与期望产出）")
            milestone = {}
        missing += [f"milestone.{k}" for k in REQUIRED_MILESTONE if not milestone.get(k)]
        if missing:
            errors.append(f"{rel}: 缺字段 {', '.join(missing)}——目录生成不出来，"
                          f"不补空值（`@absent`）")
            continue
        declared_dir = str(doc.get("dir") or "").strip().rstrip("/")
        actual_dir = path.parent.relative_to(root).as_posix()
        if declared_dir and declared_dir != actual_dir:
            errors.append(f"{rel}: `dir` 声明为 `{declared_dir}`，实际在 `{actual_dir}`"
                          f"——目录与事实源必须一致")
            continue
        rows.append({
            "id": str(doc["id"]),
            "title": str(doc["title"]),
            "purpose": str(doc["purpose"]),
            "acceptance": str(milestone["acceptance"]),
            "output": str(milestone["output"]),
            "status": str(doc["status"]),
            "pointer": f"{actual_dir}/project.yaml",
            "source": rel,
        })
    ids = [r["id"] for r in rows]
    for dup in sorted({i for i in ids if ids.count(i) > 1}, key=id_key):
        errors.append(f"研究项 id 重复：`{dup}`（id 是目录的键，必须唯一）")
    rows.sort(key=lambda r: id_key(r["id"]))
    return rows, errors


def _load(path: Path, rel: str, errors: list[str]) -> dict | None:
    try:
        import yaml
    except ImportError:                                # pragma: no cover - 环境缺 PyYAML
        errors.append("缺 PyYAML：`tools/bootstrap.sh` 的依赖门会装它")
        return None
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"{rel}: 读不出来（{exc}）")
        return None
    if not isinstance(doc, dict):
        errors.append(f"{rel}: 顶层不是一个映射")
        return None
    return doc


def render(rows: list[dict]) -> str:
    body: list[str] = [BANNER, ""] + list(INTRO)
    body.append("| " + " | ".join(COLUMNS) + " |")
    body.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        body.append("| " + " | ".join((
            _cell(r["id"], None),
            _cell(r["title"], None),
            _cell(r["purpose"], CELL_WIDTHS["目的"]),
            _cell(r["acceptance"], CELL_WIDTHS["验收"]),
            _cell(r["output"], CELL_WIDTHS["期望产出"]),
            _cell(r["status"], None),
            f"`{_cell(r['pointer'], None)}`",
        )) + " |")
    body.append("")
    body.append(f"共 **{len(rows)}** 行 = `axes/**/project.yaml` 的个数"
                f"（`tools/build_catalog.py` 生成时统计，不手写）。")
    body.append("")
    return "\n".join(body)


def _disp(path: Path) -> str:
    """仓库相对路径（测试会把 `OUT` 指到临时目录，那时退回绝对路径）。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip())
        return 0
    check = "--check" in argv
    rows, errors = collect()
    if errors:
        print(f"FAIL：{len(errors)} 项问题，未生成目录", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    text = render(rows)
    disp = _disp(OUT)

    if check:
        disk = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if disk is None:
            print(f"FAIL：{disp} 不存在；先运行 `python3 tools/build_catalog.py`", file=sys.stderr)
            return 1
        if disk != text:
            diff = list(difflib.unified_diff(
                disk.splitlines(), text.splitlines(),
                fromfile=f"{disp}（磁盘）", tofile=f"{disp}（按 project.yaml 重算）",
                lineterm="", n=1))
            print(f"FAIL：{disp} 与 project.yaml 不一致（差异 {len(diff)} 行，前 20 行如下）；"
                  f"本文件是生成物，改事实请改 project.yaml 后重跑生成器：", file=sys.stderr)
            for line in diff[:20]:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"OK：{disp} 与 {len(rows)} 个研究项的 project.yaml 逐字节一致"
              f"（七列：{' / '.join(COLUMNS)}）。")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"已生成 {disp}：{len(rows)} 行（= {len(rows)} 个 project.yaml）。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
