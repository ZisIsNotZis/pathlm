#!/usr/bin/env python3
"""build_cost_table.py — 从 `axes/**/project.yaml` 生成 `docs/cost-table.md`（代价表）。

本文件是**生成物**（render）：定律 ↔ 证据 ↔ 代价数字的映射写在每个研究项的
`laws` / `cost_rows` 字段里（契约见 `docs/research-schema.md` §2.1 / §2.2），
定律正文只有 `docs/playbook.md` 一个家。本脚本只把它们拼成一张可引用的表，
**不重算任何数字、不改口径**。

用法：
    python3 tools/build_cost_table.py          # 写 docs/cost-table.md；未覆盖的定律打到 stderr
    python3 tools/build_cost_table.py --check  # 与磁盘逐字节比对；不一致 ⇒ 非零退出 + 差异摘要
                                               # 定律 1..11 里有定律无 cost_rows 覆盖 ⇒ 非零退出
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"
OUT = ROOT / "docs" / "cost-table.md"
PLAYBOOK = ROOT / "docs" / "playbook.md"

sys.path.insert(0, str(ROOT / "tools"))

import check_tree  # noqa: E402

#: 定律总数（编号域 = `docs/playbook.md` 的「定律 1..11」）。
LAWS_TOTAL = check_tree.LAWS_TOTAL
#: 定律 8–11 不是「代价」，而是**用表前必须过的口径与前置条件**。
CALIBER_LAWS = (8, 9, 10, 11)
#: 表头（列名自解释；顺序即契约里的顺序）。
COLUMNS = ("定律", "改动类型", "触主干", "观测代价", "规模与配方", "适用边界", "出处", "研究项")

BANNER = (
    "<!-- 本文件由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 生成，禁止手改；"
    "改事实请改 project.yaml -->"
)
INTRO = [
    "# 代价表：定律 × 是否触主干 × 观测代价 × 边界 × 出处",
    "",
    "> 本文件由 `tools/build_cost_table.py` 从 `axes/**/project.yaml` 的 `laws` / `cost_rows` 生成，"
    "禁止手改；改事实请改 `project.yaml`（契约见 `docs/research-schema.md` §2.1 / §2.2）。",
    "> 定律正文与适用边界只有 `docs/playbook.md` 一个家；本表只把「定律 ↔ 证据 ↔ 代价数字」的对应写出来。",
    "> 每个数字都逐位照抄「出处」那一栏指向的文件，**未重算、未换算口径**；"
    "指标的单位与增长方向写在「观测代价」里，因为不同落盘文件的 Δ 方向并不统一。",
    "> 列义：**改动类型** = 改了什么（人话）；**触主干** = 是否改主干前向 / 残差 / 层间聚合"
    "（旁路判断头、读出、训练配方一律为「否」）；**观测代价** = 实测 `delta` + 指标 + 对照；"
    "**规模与配方** = 这个数字所处的规模与训练配方；**适用边界** = 它不能被读成什么；"
    "**研究项** = 事实源所在的 `project.yaml`。",
    "",
]

LAW_HEADING_RE = re.compile(r"^##\s*定律\s*(\d+)\s*·\s*(.+?)\s*$")


def law_titles() -> dict[int, str]:
    """从 `docs/playbook.md` 的二级标题取定律编号 → 标题（只取标题，不搬正文）。"""
    titles: dict[int, str] = {}
    for line in PLAYBOOK.read_text(encoding="utf-8").splitlines():
        m = LAW_HEADING_RE.match(line)
        if m:
            titles[int(m.group(1))] = m.group(2)
    return titles


def _disp(path: Path) -> str:
    """打印用路径：仓库内给相对路径，仓库外（测试用临时副本）给绝对路径。"""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _cell(text, width: int | None = None) -> str:
    s = " ".join(str(text).split()) or "—"
    s = s.replace("|", "\\|")
    if width and len(s) > width:
        s = s[: width - 1] + "…"
    return s


def _id_key(item_id: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 0 for p in str(item_id).split("."))


def collect() -> tuple[list[dict], list[str]]:
    """读全部 `project.yaml`，返回 (代价事实行, 校验错误)。行里带上研究项坐标。"""
    rows: list[dict] = []
    errors: list[str] = []
    files = check_tree.collect_files()
    if not files:
        return [], ["axes/** 下没有找到任何 project.yaml"]
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        try:
            doc = check_tree.yaml.load(path.read_text(encoding="utf-8"),
                                      Loader=check_tree.DuplicateKeyLoader)
        except Exception as e:  # noqa: BLE001 —— 解析失败要如实报出
            errors.append(f"{rel}: YAML 解析失败：{e}")
            continue
        if not isinstance(doc, dict):
            errors.append(f"{rel}: 顶层不是 map")
            continue
        # 只校验本表吃的那两个字段：坏行绝不进表（未知定律 / 悬空 source 一律 fail）。
        errors.extend(check_tree.check_laws_and_costs(doc, rel))
        for row in doc.get("cost_rows") or []:
            if not isinstance(row, dict):
                continue
            rows.append({**row, "_id": doc.get("id", "?"),
                         "_title": doc.get("title", ""),
                         "_dir": doc.get("dir") or path.parent.relative_to(ROOT).as_posix(),
                         "_file": rel})
    rows.sort(key=lambda r: (r.get("law"), _id_key(r["_id"]), str(r.get("change"))))
    return rows, errors


def _table(rows: list[dict]) -> list[str]:
    out = ["| " + " | ".join(COLUMNS) + " |",
           "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for r in rows:
        item = f"[{r['_id']}](../{r['_dir']}/project.yaml)"
        if r.get("_title"):
            item += f" {r['_title']}"
        cells = (
            f"{r['law']}",
            r["change"],
            "是" if r["touches_trunk"] else "否",
            f"`{r['delta']}` {r['metric']}；对照 = {r['comparator']}",
            r["scale"],
            r["boundary"],
            f"`{r['source']}`",
            item,
        )
        out.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return out


def render(rows: list[dict], titles: dict[int, str]) -> str:
    by_law: dict[int, list[dict]] = {}
    for r in rows:
        by_law.setdefault(int(r["law"]), []).append(r)

    body: list[str] = [BANNER, ""] + list(INTRO)
    body.append("## 按定律分组的代价事实")
    body.append("")
    for law in range(1, LAWS_TOTAL + 1):
        if law in CALIBER_LAWS:
            continue  # 8–11 单列在表后（它们是口径，不是代价）
        title = titles.get(law, "")
        body.append(f"### 定律 {law}" + (f" · {title}" if title else ""))
        body.append("")
        body.append(f"> 定律正文与适用边界见 `docs/playbook.md` 的「定律 {law}」；本节只列证据与数字。")
        body.append("")
        if by_law.get(law):
            body.extend(_table(by_law[law]))
        else:
            body.append(f"（本定律目前没有任何 `cost_rows` 覆盖——见脚本 stderr 的显式报告。）")
        body.append("")

    body.append("## 定律 8–11 · 用表前必须过的口径与前置条件（不是「代价」）")
    body.append("")
    body.append(
        "这四条不是代价行，而是**引用上面任何数字之前必须先过的口径与前置条件**："
        "该不该报「翻转」、判据怎么设计、机制有没有变化空间、算力与内存按什么口径算。"
        "正文、适用边界与判定条件都在 `docs/playbook.md`，本节**不复述**，只指回它，"
        "并列出这些口径之下已经测到的具体事实。")
    body.append("")
    for law in CALIBER_LAWS:
        title = titles.get(law, "")
        body.append(f"### 定律 {law}" + (f" · {title}" if title else ""))
        body.append("")
        body.append(f"- 指回：`docs/playbook.md` 的「定律 {law}」（正文与边界不在本文件重复）。")
        body.append("")
        if by_law.get(law):
            body.extend(_table(by_law[law]))
        else:
            body.append("（该口径下目前没有任何 `cost_rows`——见脚本 stderr 的显式报告。）")
        body.append("")

    covered = sorted({int(r["law"]) for r in rows})
    missing = [lw for lw in range(1, LAWS_TOTAL + 1) if lw not in covered]
    items = sorted({r["_id"] for r in rows}, key=_id_key)
    body.append("## 覆盖情况（本表生成时统计）")
    body.append("")
    body.append(f"- 代价事实 **{len(rows)}** 条，来自 **{len(items)}** 个研究项"
                f"（{len(check_tree.collect_files())} 个 `project.yaml` 全量扫描）。")
    body.append(f"- 已被 `cost_rows` 覆盖的定律：{('、'.join(str(x) for x in covered)) or '（无）'}。")
    body.append(f"- 尚无覆盖的定律：{('、'.join(str(x) for x in missing)) or '（无）'}。")
    body.append("")
    return "\n".join(body)


def _uncovered(rows: list[dict]) -> list[int]:
    covered = {int(r["law"]) for r in rows}
    return [lw for lw in range(1, LAWS_TOTAL + 1) if lw not in covered]


def _duplicate_facts(rows: list[dict]) -> list[str]:
    seen: dict[tuple[str, str], str] = {}
    dups: list[str] = []
    for r in rows:
        key = (str(r["source"]), str(r["delta"]))
        if key in seen:
            dups.append(f"`{r['source']}` 的 `{r['delta']}` 同时出现在 {seen[key]} 与 {r['_id']}")
        else:
            seen[key] = r["_id"]
    return dups


def main(argv: list[str]) -> int:
    check = "--check" in argv
    rows, errors = collect()
    if errors:
        print(f"FAIL：cost_rows / laws 有 {len(errors)} 项问题，未生成表", file=sys.stderr)
        for e in errors:
            print("  -", e, file=sys.stderr)
        return 1

    titles = law_titles()
    missing_titles = [lw for lw in range(1, LAWS_TOTAL + 1) if lw not in titles]
    if missing_titles:
        print(f"FAIL：docs/playbook.md 里找不到定律标题：{missing_titles}", file=sys.stderr)
        return 1

    text = render(rows, titles)
    uncovered = _uncovered(rows)
    for dup in _duplicate_facts(rows):
        print(f"警告：同一条代价事实写了两行 —— {dup}", file=sys.stderr)
    if uncovered:
        print(f"未覆盖的定律（没有任何 cost_rows 支撑）：{uncovered}"
              f"；定律正文见 docs/playbook.md", file=sys.stderr)

    if check:
        disk = OUT.read_text(encoding="utf-8") if OUT.exists() else None
        if disk is None:
            print(f"FAIL：{_disp(OUT)} 不存在；先运行 tools/build_cost_table.py",
                  file=sys.stderr)
            return 1
        if disk != text:
            diff = list(difflib.unified_diff(disk.splitlines(), text.splitlines(),
                                             fromfile=f"{OUT.name}（磁盘）",
                                             tofile=f"{OUT.name}（按 project.yaml 重算）",
                                             lineterm="", n=1))
            print(f"FAIL：{_disp(OUT)} 与 project.yaml 不一致"
                  f"（差异 {len(diff)} 行，前 20 行如下）：", file=sys.stderr)
            for line in diff[:20]:
                print("  " + line, file=sys.stderr)
            return 1
        if uncovered:
            print("FAIL：有定律没有任何 cost_rows 覆盖（见上）", file=sys.stderr)
            return 1
        print(f"OK：{_disp(OUT)} 与 {len(rows)} 条代价事实逐字节一致。")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"已生成 {_disp(OUT)}：{len(rows)} 条代价事实 / "
          f"{len({r['_id'] for r in rows})} 个研究项。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
