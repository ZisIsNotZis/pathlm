#!/usr/bin/env python3
"""`catalog.yaml` 的**严格结构审计**（补充 `check_schema.py` 不覆盖的三件事）。

**已退役（ticket 56）**：catalog.yaml 已退役（SSOT = `axes/**/project.yaml`）；
本文件不再维护，迁移完成后随 catalog 一并移入 `archive/wiki-legacy/`。

0b. **已废止数字黑名单（`local.key_numbers_forbidden`）**：声明过的**废弃值**不得出现在
   三类表面（账本 verdict+benefit、生成页、实验产物）任何一处——「改数字只改一半」会立刻红。
0. **跨表面一致性（`local.key_numbers` 声明式）**：一个格子的**关键数字**必须在**三类表面**
   逐字出现——① `catalog.yaml` 的该格 `verdict`、② 由它生成的课题页
   `axes/<grid_dir>/README.md`、③ 该格的**实验产物文本**（`experiments/**` 下的
   md/json/jsonl/log）。这是「只改了实验文档、漏了账本与生成页」那类事故（`2.53%` vs 产物
   `2.253%`）的机检：三表面任一缺失即失败。**未声明 `key_numbers` 的格子跳过本条。**
1. **重复键**：默认 `yaml.SafeLoader` 会静默取后一个值 ⇒ 用自定义 loader 让重复键直接报错；
2. **`prior_art_source` 的归属**：每格的 `prior_art_source` 必须指向**与本格轴同名**的
   `axes/<X>-*/PRIOR_ART.md`（`A-mixing→A`、`C-vertical→C`、`H-horizon→H`、`W-width→W`、
   `F-iteration→F`、`M-carrier→M`）——格内键写错位置时会“漂”到上一格，这条能抓住；
3. **游离键**：两格之间的 `# ====` 注释块前后不得出现格内的键（`    <key>:` 行）——那是
   “字段写到了上一格里”的结构性缺陷。

用法：`python3 tools/check_catalog_strict.py`（退出码 0 = 全过）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.yaml"

#: `axes/<dir>/PRIOR_ART.md` → 轴字母
AXIS_OF_PRIOR_ART_DIR = {
    "A-mixing": "A", "C-vertical": "C", "H-horizon": "H", "W-width": "W",
    "F-iteration": "F", "M-carrier": "M",
}
PRIOR_ART_RE = re.compile(r"axes/([A-Za-z0-9._-]+)/PRIOR_ART\.md")
GRID_ID_RE = re.compile(r"^  - id: (\S+)")
KEY_LINE_RE = re.compile(r"^    ([a-z_][a-z0-9_]*):")
COMMENT_RE = re.compile(r"^  #")


class StrictLoader(yaml.SafeLoader):
    """重载 mapping 构造：键重复直接抛错（默认是静默覆盖）。"""


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.Node):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping", node.start_mark,
                f"重复键 {key!r}（catalog.yaml 不允许；默认 loader 会静默取后者）",
                key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def check_duplicate_keys() -> list[str]:
    text = CATALOG.read_text()
    try:
        yaml.load(text, Loader=StrictLoader)
        return []
    except yaml.constructor.ConstructorError as e:
        return [f"重复键：{e}"]


def check_prior_art_axis() -> tuple[list[str], list[dict]]:
    cfg = yaml.safe_load(CATALOG.read_text())
    rows, errors = [], []
    for g in cfg["grids"]:
        src = g.get("prior_art_source")
        if not src:
            continue
        m = PRIOR_ART_RE.search(str(src))
        if not m:
            errors.append(f"{g['id']}: prior_art_source 不含 axes/<dir>/PRIOR_ART.md：{src}")
            continue
        want = AXIS_OF_PRIOR_ART_DIR.get(m.group(1))
        ok = (want == g["axis"])
        rows.append({"grid": g["id"], "axis": g["axis"], "dir": m.group(1),
                     "expected_axis": want, "ok": bool(ok)})
        if not ok:
            errors.append(
                f"{g['id']}（axis={g['axis']}）的 prior_art_source 指向 {m.group(1)}"
                f"（应为 axis {g['axis']} 的 PRIOR_ART）⇒ 字段很可能写进了上一格")
    return errors, rows


def _iter_text_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in ("__pycache__", ".git") for part in p.parts):
            continue
        if p.suffix in (".pt", ".npz", ".png", ".pyc"):
            continue
        yield p


def check_key_numbers() -> tuple[list[str], list[dict]]:
    """跨表面一致性（三类表面：账本 verdict / 生成页 / 实验产物文本）。

    `local.key_numbers: ["2.253%", "2.32×", …]` 声明的每个**关键数字**都必须在
    ① 该格 catalog `verdict`、② 生成的课题页、③ 该格实验产物的**至少一个**文本文件里出现。
    """
    cfg = yaml.safe_load(CATALOG.read_text())
    # 轴目录名（与 `tools/build_wiki.py` 同一来源：`axes.<axis>.dir`）
    axis_dir = {a["id"]: a["dir"] for a in (cfg.get("axes") or [])}
    rows, errors = [], []
    for g in cfg["grids"]:
        toks = (g.get("local") or {}).get("key_numbers") or []
        if not toks:
            continue
        grid_dir = f"{axis_dir[g['axis']]}/{g['id']}"
        page = ROOT / "axes" / grid_dir / "README.md"
        page_text = page.read_text() if page.exists() else ""
        exp_dir = ROOT / "axes" / grid_dir / "experiments"
        prod_text = ""
        if exp_dir.exists():
            prod_text = "\n".join(
                p.read_text(errors="ignore") for p in _iter_text_files(exp_dir))
        for tok in toks:
            in_verdict = tok in (g.get("verdict") or "")
            in_page = tok in page_text
            in_prod = tok in prod_text
            rows.append({"grid": g["id"], "token": tok, "verdict": in_verdict,
                         "page": in_page, "products": in_prod})
            if not (in_verdict and in_page and in_prod):
                missing = [n for n, v in (("账本 verdict", in_verdict),
                                          ("生成页", in_page),
                                          ("实验产物", in_prod)) if not v]
                errors.append(f"{g['id']}: 关键数字 {tok!r} 缺表面 {missing}"
                              f"（三类表面必须逐字一致；见 SCHEMA §24 第 13 类）")
    return errors, rows


def check_forbidden_numbers() -> tuple[list[str], list[dict]]:
    """已废止数字不得出现在三类表面（账本 / 生成页 / 实验产物）。"""
    cfg = yaml.safe_load(CATALOG.read_text())
    axis_dir = {a["id"]: a["dir"] for a in (cfg.get("axes") or [])}
    rows, errors = [], []
    for g in cfg["grids"]:
        toks = (g.get("local") or {}).get("key_numbers_forbidden") or []
        if not toks:
            continue
        grid_dir = f"{axis_dir[g['axis']]}/{g['id']}"
        page = ROOT / "axes" / grid_dir / "README.md"
        page_text = page.read_text() if page.exists() else ""
        exp_dir = ROOT / "axes" / grid_dir / "experiments"
        prod_text = ""
        if exp_dir.exists():
            prod_text = "\n".join(p.read_text(errors="ignore") for p in _iter_text_files(exp_dir))
        ledger_text = "\n".join(str((g.get("local") or {}).get(k) or "")
                                for k in ("tax", "benefit")) + "\n" + (g.get("verdict") or "")
        for tok in toks:
            hits = [n for n, t in (("账本", ledger_text), ("生成页", page_text),
                                   ("实验产物", prod_text)) if tok in t]
            rows.append({"grid": g["id"], "token": tok, "hits": hits})
            if hits:
                errors.append(f"{g['id']}: 已废止数字 {tok!r} 仍出现在 {hits}"
                              f"（改数字必须扫全部表面；见 SCHEMA §24 第 13 类）")
    return errors, rows


def check_stray_keys() -> list[str]:
    """结构检查：**注释块之后、下一个 `- id:` 之前**不得出现格内键。

    这正是「字段写到了上一格里」的形态：`# ==== X` 之后紧跟一行 `    prior_art_source:`
    再出现 `- id: X` ⇒ YAML 把该键归给 **X 的上一格**（默认 loader 静默）。
    """
    lines = CATALOG.read_text().split("\n")
    errors: list[str] = []
    after_comment = False
    pending_comment = ""
    for i, line in enumerate(lines, start=1):
        if COMMENT_RE.match(line) and "====" in line:
            after_comment = True
            pending_comment = line.strip()[:60]
            continue
        if GRID_ID_RE.match(line):
            after_comment = False
            pending_comment = ""
            continue
        if after_comment and (m := KEY_LINE_RE.match(line)):
            errors.append(f"{i}: 游离键 {m.group(1)!r}（在注释块 {pending_comment!r} 之后、"
                          f"下一个 `- id:` 之前）⇒ YAML 会把它归给「上一格」")
    return errors


def main() -> int:
    errors: list[str] = []
    errors += check_duplicate_keys()
    axis_errors, rows = check_prior_art_axis()
    errors += axis_errors
    errors += check_stray_keys()
    kn_errors, kn_rows = check_key_numbers()
    errors += kn_errors
    fb_errors, fb_rows = check_forbidden_numbers()
    errors += fb_errors
    print("== 每格 prior_art_source 归属核对 ==")
    for r in rows:
        print(f"  [{'OK ' if r['ok'] else 'BAD'}] {r['grid']:28s} axis={r['axis']} "
              f"→ {r['dir']} (expect {r['expected_axis']})")
    n_src = len(rows)
    print(f"\n共 {n_src} 格带 prior_art_source；轴不匹配 {len(axis_errors)} 条；"
          f"重复键 {len([e for e in errors if e.startswith('重复键')])} 条；"
          f"游离键 {len([e for e in errors if '游离键' in e])} 条")
    if kn_rows:
        print("== 跨表面一致性（local.key_numbers：账本 verdict / 生成页 / 实验产物）==")
        for r in kn_rows:
            mark = "OK " if (r["verdict"] and r["page"] and r["products"]) else "BAD"
            print(f"  [{mark}] {r['grid']:22s} {r['token']:10s} verdict={r['verdict']} "
                  f"page={r['page']} products={r['products']}")
        print(f"  共 {len(kn_rows)} 个声明数字，缺表面 {len(kn_errors)} 条")
    if fb_rows:
        print("== 已废止数字黑名单（local.key_numbers_forbidden）==")
        for r in fb_rows:
            print(f"  [{'BAD' if r['hits'] else 'OK '}] {r['grid']:22s} {r['token']:10s} "
                  f"仍出现于={r['hits'] or '—'}")
        print(f"  共 {len(fb_rows)} 个废弃值，命中 {len(fb_errors)} 条")
    for e in errors:
        print("  ✗", e)
    print("\n=== catalog 严格审计：" + ("全过 ===" if not errors else f"{len(errors)} 条问题 ==="))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
