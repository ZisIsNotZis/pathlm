#!/usr/bin/env python3
"""build_wiki.py — 由 catalog.yaml 生成/刷新 Research Wiki 的索引与课题页。

**已退役（ticket 56）**：新结构下研究项 SSOT = `axes/**/project.yaml`（见
`docs/research-schema.md` / `docs/structure.md`）；本文件不再维护、不再生成网页，
迁移完成后整体移入 `archive/wiki-legacy/`。保留在仓仅供迁移窗口读旧内容。

设计原则（见 SCHEMA.md）：
  * catalog.yaml 是「目录完备」的唯一来源（枚举了哪些格子必须存在）。
  * 课题页 = front-matter（生成）+ 九节正文（生成）+ 手写块（保留）。
  * 自动内容一律夹在 `<!-- AUTO:begin --> ... <!-- AUTO:end -->` 之间；
    手写内容夹在 `<!-- MANUAL:begin --> ... <!-- MANUAL:end -->` 之间，重跑不丢——
    `--check` 也是先嫁接 MANUAL 再比对，故「只手写块有差异」不算不一致。
  * 既有页面的 MANUAL 区块缺失 / 畸形 / 内含 AUTO 标记 ⇒ 直接报错（不静默覆盖手写内容、
    不许手写内容伪装成自动块）。
  * 索引（根 INDEX.md、各轴 README 的格子表、各格子 experiments/ 清单）全部生成，
    人永远不手改自动块。

用法：
    python3 tools/build_wiki.py            # 生成/刷新（幂等）
    python3 tools/build_wiki.py --check     # 只校验是否与 catalog 一致（CI 用）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.yaml"

STATUS_BADGE = {
    "judged": "✅ judged",
    "todo": "⬜ todo",
}

AUTO_BEGIN = "<!-- AUTO:begin -->"
AUTO_END = "<!-- AUTO:end -->"
MANUAL_BEGIN = "<!-- MANUAL:begin -->"
MANUAL_END = "<!-- MANUAL:end -->"


class WikiError(Exception):
    """既有页面违反生成契约（MANUAL 区块缺失/畸形/越界），必须报错而不是静默覆盖。"""


def badge(status: str) -> str:
    """状态徽章；未知状态直接报错，不静默回落。"""
    if status not in STATUS_BADGE:
        raise SystemExit(f"catalog.yaml 出现未知 status：{status!r}（合法值：{sorted(STATUS_BADGE)}）")
    return STATUS_BADGE[status]

REQUIRED_GRID_KEYS = ("id", "axis", "title", "claim", "options", "judge", "cost", "status")


def load() -> dict:
    with CATALOG.open(encoding="utf-8") as f:
        cat = yaml.safe_load(f)
    # 扁平 grids 列表按 axis 归组（catalog 只维护一份 grid 定义）
    for a in cat["axes"]:
        a["grids"] = [g for g in cat["grids"] if g["axis"] == a["id"]]
    return cat


def fm(grid: dict) -> str:
    """课题页 front-matter（机器可读）。"""
    local = grid.get("local") or {}
    lines = [
        "---",
        f"id: {grid['id']}",
        f"axis: {grid['axis']}",
        f"grid_dir: {grid['axis_dir']}/{grid['id']}",
        "title: " + json.dumps(grid["title"], ensure_ascii=False),
        f"status: {grid['status']}",
        "judge: " + json.dumps(grid["judge"].strip(), ensure_ascii=False),
        "cost: " + json.dumps(grid["cost"], ensure_ascii=False),
        "tax: " + json.dumps((local.get("tax") or "—"), ensure_ascii=False),
        "benefit: " + json.dumps((local.get("benefit") or "—"), ensure_ascii=False),
        "verdict: " + json.dumps((grid.get("verdict") or "—"), ensure_ascii=False),
        "deps: " + json.dumps(grid.get("deps") or [], ensure_ascii=False),
        "evidence: " + json.dumps(local.get("evidence") or [], ensure_ascii=False),
        "---",
    ]
    return "\n".join(lines)


def bullets(items: list[str], empty: str = "（无）") -> str:
    if not items:
        return empty
    return "\n".join(f"{i + 1}. {t}" for i, t in enumerate(items))


def page(grid: dict, axis: dict) -> str:
    local = grid.get("local") or {}
    failure = local.get("failure") or []
    options = grid.get("options") or []
    verdict = grid.get("verdict")
    industry = (grid.get("industry") or "").strip()
    notes = (grid.get("notes") or "").strip()

    body = [fm(grid), "", "<!-- AUTO:begin -->", ""]
    body += [f"# {grid['id']} · {grid['title']}", ""]
    body += [f"> 轴：**{axis['id']} {axis['title']}** ｜ 状态：**{badge(grid['status'])}**"]
    if verdict:
        body += ["", f"> **判读**：{verdict.strip()}"]
    body += ["", "## 1. 命题", "", grid["claim"].strip(), ""]

    body += ["## 2. 选项枚举（本格全部可能取值）", "", bullets([f"`{o}`" if not o.startswith("**") else o for o in options]), ""]

    body += ["## 3. 状态（我们做了什么 / 行业做了什么）", ""]
    if local:
        body += ["**本地已测**：" + (local.get("benefit") or "—").strip(), ""]
    else:
        body += ["**本地**：零经验（无实验）。", ""]
    body += ["**行业口径**：" + (industry if industry else "见 `docs/playbook.md` 与参照线 `R-references/`。"), ""]

    pa_c = (grid.get("prior_art_conclusion") or "").strip()
    pa_n = (grid.get("prior_art_note") or "").strip()
    pa_s = (grid.get("prior_art_source") or "").strip()
    if pa_c or pa_n:
        body += ["## 3b. 既有做法（ticket 36 分诊，SCHEMA §21）", ""]
        if pa_c:
            body += [f"**业界结论（非本地实测）**：{pa_c}", ""]
        if pa_n:
            body += [f"**既有做法参照**：{pa_n}", ""]
        if pa_s:
            body += [f"来源：`{pa_s}`（`记忆·未核`，联网不可用，待核）", ""]

    body += ["## 4. 结论与数字", ""]
    body += [f"- **税**：{(local.get('tax') or '—')}", f"- **收益**：{(local.get('benefit') or '—')}"]
    if verdict:
        body += [f"- **判读**：{verdict.strip()}"]
    body += [""]

    body += ["## 5. 失败路径与原因", "", bullets(failure), ""]

    body += ["## 6. 判据与判决实验设计", "", f"- **判据**：{grid['judge'].strip()}"]
    if grid["status"] == "todo":
        body += [
            "",
            "**下一步判决实验**（未做时必填）：",
            "",
            f"- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）",
            f"- 判据线：{grid['judge'].strip()}",
            f"- 成本：{grid['cost']}",
        ]
    body += [""]

    body += ["## 7. 成本估计", "", f"- {grid['cost']}", ""]

    body += ["## 8. 依赖与耦合声明（只记录，不做组合实验）", ""]
    body += [bullets([f"`{d}`" for d in (grid.get("deps") or [])], "（无已知依赖）"), ""]

    body += ["## 9. 复现手段", ""]
    ev = local.get("evidence") or []
    body += [bullets(ev, "（本地未做；复现命令待本格实验目录建立后补）"), ""]
    if notes:
        body += ["## 备注", "", notes, ""]
    body += [AUTO_END, "", MANUAL_BEGIN, MANUAL_END, ""]
    return "\n".join(body)


def axis_page(axis: dict) -> str:
    grids = axis["grids"]
    out = ["<!-- AUTO:begin -->", "", f"# {axis['id']} · {axis['title']}", ""]
    out += [f"- **原语**：{axis.get('primitive', '—')}", f"- **命题**：{axis['claim'].strip()}", ""]
    out += ["## 定义", "", axis["definition"].strip(), "", "## 子维度", "", axis["subdims"].strip(), ""]
    out += ["## 本地总体结论", "", axis["local_summary"].strip(), ""]
    pa = (ROOT / "axes" / axis["dir"] / "PRIOR_ART.md")
    if pa.exists():
        out += [f"> **既有做法情报**：[PRIOR_ART.md](PRIOR_ART.md)（含机器可读分诊表；来源标 `记忆·未核`，联网不可用）", ""]
    out += ["## 格子索引", "", "| 格子 | 状态 | 判读 | 税 |", "|---|---|---|---|"]
    for g in grids:
        local = g.get("local") or {}
        v = (g.get("verdict") or "—").strip().replace("\n", " ")
        if len(v) > 90:
            v = v[:88] + "…"
        tax = (local.get("tax") or "—").strip().replace("\n", " ")
        if len(tax) > 40:
            tax = tax[:38] + "…"
        out += [f"| [`{g['id']}`]({g['id']}/README.md) | {badge(g['status'])} | {v} | {tax} |"]
    out += ["", AUTO_END, "", MANUAL_BEGIN, MANUAL_END, ""]
    return "\n".join(out)


def experiments_page(grid: dict, existing: list[str]) -> str:
    out = ["<!-- AUTO:begin -->", "", f"# {grid['id']} · 实验目录", "",
           "本格每次判决占一个子目录（自包含）：`README.md`(契约) / 代码 / `results.json` / "
           "`analysis.md`(判读) / `MODEL.md`(权重哈希，权重本身不入库)。", "",
           "命名：`YYMM-<干预>-<规模>-<引擎>`，例：`2601-summary-slot-13m-depthar`。", ""]
    if existing:
        out += ["| 实验 | 来源 |", "|---|---|"]
        out += [f"| [`{n}`]({n}/README.md) | 目录内 README |" for n in sorted(existing)]
    else:
        out += ["（本格暂无实验目录）"]
    out += ["", AUTO_END, "", MANUAL_BEGIN, MANUAL_END, ""]
    return "\n".join(out)


def manual_region(old: str, path: Path) -> str:
    """取出既有页面里唯一的 MANUAL 区块（含标记）。

    缺失 / 重复 / 顺序颠倒 / 区块内含 AUTO 标记 ⇒ 报错；绝不静默丢弃手写内容，
    也绝不允许手写内容伪装成 AUTO 块（见 SCHEMA：MANUAL 只能存在于自己的区间内）。
    """
    nb, ne = old.count(MANUAL_BEGIN), old.count(MANUAL_END)
    if nb == 0 and ne == 0:
        raise WikiError(f"{path}：既有页面没有 MANUAL 区块，拒绝整页覆盖（会丢手写内容）")
    if nb != 1 or ne != 1:
        raise WikiError(f"{path}：MANUAL 区块畸形（begin×{nb} / end×{ne}，必须各恰好 1 个）")
    s, e = old.index(MANUAL_BEGIN), old.index(MANUAL_END)
    if e < s:
        raise WikiError(f"{path}：MANUAL 区块顺序颠倒（end 在 begin 之前）")
    region = old[s:e + len(MANUAL_END)]
    if AUTO_BEGIN in region or AUTO_END in region:
        raise WikiError(f"{path}：MANUAL 区块里出现 AUTO 标记，手写内容不得越界篡改自动块")
    return region


def graft(path: Path, old: str, content: str) -> str:
    """生成内容（AUTO）保留，手工内容（MANUAL）从既有页嫁接过来。"""
    region = manual_region(old, path)
    s, e = content.index(MANUAL_BEGIN), content.index(MANUAL_END) + len(MANUAL_END)
    return content[:s] + region + content[e:]


def write(path: Path, content: str, check: bool, changed: list[Path]) -> None:
    if path.exists():
        old = path.read_text(encoding="utf-8")
        # --check 也必须嫁接后再比对，否则手写块永远报「不一致」
        new = graft(path, old, content)
        if old == new:
            return
        if check:
            changed.append(path)
            return
        path.write_text(new, encoding="utf-8")
        changed.append(path)
    else:
        if check:
            changed.append(path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        changed.append(path)


def index_page(cat: dict) -> str:
    proj = cat["project"]
    counts: dict[str, int] = {}
    for a in cat["axes"]:
        for g in a["grids"]:
            counts[g["status"]] = counts.get(g["status"], 0) + 1
    total = sum(counts.values())
    out = ["<!-- AUTO:begin -->", "", "# INDEX · 设计空间状态灯", "",
           f"共 **{total}** 个格子 ｜ " + " ｜ ".join(f"{badge(k)} {v}" for k, v in sorted(counts.items())),
           "", "> 全部内容由 `tools/build_wiki.py` 从 `catalog.yaml` 生成；**不要手改本文件**。", ""]
    for a in cat["axes"]:
        out += [f"## {a['id']} · {a['title']}（{len(a['grids'])} 格）", "",
                f"[轴说明](axes/{a['dir']}/README.md) ｜ {a['claim'].strip()}", "",
                "| 格子 | 状态 | 判读 |", "|---|---|---|"]
        for g in a["grids"]:
            v = (g.get("verdict") or "—").strip().replace("\n", " ")
            if len(v) > 100:
                v = v[:98] + "…"
            out += [f"| [`{g['id']}`](axes/{a['dir']}/{g['id']}/README.md) · {g['title']} | {badge(g['status'])} | {v} |"]
        out += [""]
    out += ["<!-- AUTO:end -->", "", "<!-- MANUAL:begin -->", "<!-- MANUAL:end -->", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验是否与 catalog 一致")
    args = ap.parse_args()

    cat = load()
    changed: list[Path] = []
    try:
        for a in cat["axes"]:
            adir = ROOT / "axes" / a["dir"]
            for g in a["grids"]:
                g["axis_dir"] = a["dir"]
                gdir = adir / g["id"]
                write(gdir / "README.md", page(g, a), args.check, changed)
                expdir = gdir / "experiments"
                existing = ([d.name for d in expdir.iterdir() if d.is_dir()]
                            if expdir.exists() else [])
                write(expdir / "README.md", experiments_page(g, existing), args.check, changed)
            write(adir / "README.md", axis_page(a), args.check, changed)
        write(ROOT / "INDEX.md", index_page(cat), args.check, changed)
    except WikiError as e:
        print(f"FAIL：{e}", file=sys.stderr)
        return 1

    if args.check:
        if changed:
            print("与 catalog 不一致（需重跑 tools/build_wiki.py）：")
            for p in changed:
                print("  -", p.relative_to(ROOT))
            return 1
        print("wiki 与 catalog 一致。")
        return 0
    print(f"更新 {len(changed)} 个文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
