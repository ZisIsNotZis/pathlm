#!/usr/bin/env python3
"""summary.py — 一条命令打印研究树状态全表（SSOT 状态表；无网页、无缓存、无 fancy）。

扫描 `axes/**/project.yaml`，输出列：
    id / 轴 / 起 / 止 / 完成? / 状态 / 结论 / 结论对不对 / 实验 / 论证 / 遗漏

用法：
    python3 tools/summary.py          # 打印全表
    python3 tools/summary.py --check  # 校验字段/证据（转调 check_tree.py），非零退出

表尾除了「共 N 个研究项」还打印一行辅助实验的条数：轴 7 / 轴 8 的实验直接挂在轴下、没有 `project.yaml`，
不在研究项表内（见 `docs/structure.md` §16）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"
sys.path.insert(0, str(ROOT / "tools"))

import check_tree  # noqa: E402

#: 完成? 灯（状态 → 标记）
DONE_LAMP = {"已完成": "✅", "被否证": "❌", "进行中": "🔄", "未开始": "⬜"}

COLUMNS = ("id", "轴", "起", "止", "完成?", "状态", "结论", "结论对不对",
           "实验", "论证", "遗漏")
#: 表格列宽（长字段截断；完整内容看各自 project.yaml）
WIDTHS = (10, 12, 12, 12, 6, 8, 40, 24, 24, 28, 24)


def _cell(v, width: int) -> str:
    if isinstance(v, (list, tuple)):
        v = f"{len(v)} 条" if v else "—"
    s = str(v or "—").replace("\n", " ").strip() or "—"
    if len(s) > width:
        s = s[: width - 1] + "…"
    return s


def _aux_experiments() -> list[Path]:
    """辅助实验目录：直接挂在轴下（所在轴没有 `project.yaml`），见 `docs/structure.md` §16。"""
    out: list[Path] = []
    for exp in sorted(AXES.rglob("experiments")):
        if not exp.is_dir() or (exp.parent / "project.yaml").is_file():
            continue
        out.extend(sorted(d for d in exp.iterdir() if d.is_dir()))
    return out


def _row(doc: dict) -> list[str]:
    exps = doc.get("experiments") or []
    return [
        doc.get("id", "?"),
        doc.get("axis", "?"),
        doc.get("started") or "—",
        doc.get("ended") or "—",
        DONE_LAMP.get(doc.get("status"), "?"),
        doc.get("status", "?"),
        doc.get("conclusion") or "—",
        doc.get("conclusion_review") or "—",
        f"{len(exps)} 条" if exps else "—",
        doc.get("argument") or "—",
        doc.get("gaps") or "—",
    ]


def main(argv: list[str]) -> int:
    if "--check" in argv:
        return check_tree.main()

    errors, entries = check_tree.validate_all()
    if errors:
        # 校验有错也照常出表，但把错误打到 stderr 并提示。
        print(f"（校验发现 {len(errors)} 项问题；运行 `tools/check_tree.py` 看明细）",
              file=sys.stderr)

    header = "  ".join(_cell(c, w).ljust(w) for c, w in zip(COLUMNS, WIDTHS))
    print(header)
    print("-" * len(header))
    for e in entries:
        cells = _row(e["doc"])
        print("  ".join(_cell(c, w).ljust(w) for c, w in zip(cells, WIDTHS)))
    n_done = sum(1 for e in entries if e["doc"].get("status") == "已完成")
    n_fals = sum(1 for e in entries if e["doc"].get("status") == "被否证")
    print("-" * len(header))
    print(f"共 {len(entries)} 个研究项 ｜ ✅ 已完成 {n_done} ｜ ❌ 被否证 {n_fals}")
    aux = _aux_experiments()
    if aux:
        print(f"另有 {len(aux)} 个辅助实验不在上表内（轴 7 参照线 / 轴 8 工程纪律随附的实验，无 project.yaml）："
              f"见 docs/structure.md 的「辅助实验」一节")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
