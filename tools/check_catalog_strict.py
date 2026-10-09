#!/usr/bin/env python3
"""`catalog.yaml` 的**严格结构审计**（补充 `check_schema.py` 不覆盖的三件事）。

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
    print("== 每格 prior_art_source 归属核对 ==")
    for r in rows:
        print(f"  [{'OK ' if r['ok'] else 'BAD'}] {r['grid']:28s} axis={r['axis']} "
              f"→ {r['dir']} (expect {r['expected_axis']})")
    n_src = len(rows)
    print(f"\n共 {n_src} 格带 prior_art_source；轴不匹配 {len(axis_errors)} 条；"
          f"重复键 {len([e for e in errors if e.startswith('重复键')])} 条；"
          f"游离键 {len([e for e in errors if '游离键' in e])} 条")
    for e in errors:
        print("  ✗", e)
    print("\n=== catalog 严格审计：" + ("全过 ===" if not errors else f"{len(errors)} 条问题 ==="))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
