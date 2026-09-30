#!/usr/bin/env python3
"""backfill_legacy.py — 把 archive/legacy-experiments.yaml 渲染成各格子下的实验目录。

一次性回填工具（可重复运行、幂等）。渲染目标：
    axes/<轴目录>/<primary grid>/experiments/<name>/README.md

映射表同时承担「完备性检查」：primary/also 里的格子 id 不存在 → 直接报错。

用法：
    python3 tools/backfill_legacy.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.yaml"
LEGACY = ROOT / "archive" / "legacy-experiments.yaml"


def grid_index(cat: dict) -> dict[str, str]:
    """grid id -> 轴目录名"""
    return {g["id"]: a["dir"] for a in cat["axes"] for g in cat["grids"] if g["axis"] == a["id"]}


def render(exp: dict, grid_dir: str) -> str:
    out = ["---", f"name: {exp['name']}", f"ticket: {exp['ticket']}", f"date: \"{exp['date']}\"",
           f"grid: {exp['primary']}", f"engine: \"{exp.get('engine', '—')}\"",
           f"budget: \"{exp.get('budget', '—')}\"", "---", "",
           f"# {exp['name']}（ticket {exp['ticket']}）", "",
           f"> 主格：`{exp['primary']}`｜关联格：{', '.join('`'+a+'`' for a in exp.get('also', [])) or '（无）'}"
           f"｜引擎：{exp.get('engine', '—')}｜预算：{exp.get('budget', '—')}", ""]
    out += ["## 命题", "", exp["claim"], "", "## 干预", "", exp.get("intervention", "—"), "",
            "## 结果", ""]
    out += [f"- {r}" for r in exp.get("results", [])] or ["- （无）"]
    out += ["", "## 失败路径与原因", ""]
    fails = exp.get("failure", [])
    out += [f"{i+1}. {f}" for i, f in enumerate(fails)] if fails else ["（无）"]
    out += ["", "## 证据与复现", ""]
    out += [f"- `{e}`" for e in exp.get("evidence", [])] or ["- （无）"]
    out += ["", "> 数字口径警告见 `archive/README.md`；本条目由 `tools/backfill_legacy.py` 生成。", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    gdirs = grid_index(cat)
    legacy = yaml.safe_load(LEGACY.read_text(encoding="utf-8"))["experiments"]

    changed, errors = [], []
    for e in legacy:
        for gid in [e["primary"], *e.get("also", [])]:
            if gid not in gdirs:
                errors.append(f"{e['name']}: 未知格子 {gid}")
        for ev in e.get("evidence", []):
            p = str(ev).split(" @")[0].strip()
            if p and not (ROOT / p).exists():
                errors.append(f"{e['name']}: 证据路径不存在 {p}")
        if errors:
            continue
        path = ROOT / "axes" / gdirs[e["primary"]] / e["primary"] / "experiments" / e["name"] / "README.md"
        content = render(e, gdirs[e["primary"]])
        if path.exists() and path.read_text(encoding="utf-8") == content:
            continue
        if args.check:
            changed.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        changed.append(path)

    if errors:
        print("FAIL：")
        for x in errors:
            print("  -", x)
        return 1
    if args.check and changed:
        print("与映射表不一致（需重跑 tools/backfill_legacy.py）：")
        for p in changed:
            print("  -", p.relative_to(ROOT))
        return 1
    print(f"{'校验一致' if args.check else '写入'}：{len(legacy)} 个实验条目，{len(changed)} 个文件{'需更新' if args.check else '已更新'}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
