#!/usr/bin/env python3
"""check_schema.py — Research Wiki 的 CI 校验（可失败）。

校验项：
  1. 目录完备：catalog 枚举的每一格都有实体目录与 README.md（缺 → fail）。
  2. 模板完备：每页 front-matter 的必需字段齐全、九节齐全。
  3. 证据完备：status=judged 必须有 judge 与 evidence；status=todo 无本地证据，
     但 verdict 必须写明「待立项」或「尚缺什么」（待补做）。
     evidence 里 `path @commit` 的 path 必须在磁盘上存在（归档/证据锚点不悬空）。
  4. 无孤儿：axes/ 下的格子目录若不在 catalog 中 → fail（防止绕开枚举加内容）。
  5. 生成一致：INDEX.md / 轴 README / 课题页自动块与 catalog 一致
     （等价于 `build_wiki.py --check`）。

用法：python3 tools/check_schema.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "catalog.yaml"
SECTIONS = ["## 1. 命题", "## 2. 选项枚举", "## 3. 状态", "## 4. 结论与数字",
            "## 5. 失败路径与原因", "## 6. 判据与判决实验设计", "## 7. 成本估计",
            "## 8. 依赖与耦合声明", "## 9. 复现手段"]
REQUIRED = ("id", "axis", "grid_dir", "title", "status", "judge", "cost",
            "tax", "benefit", "verdict", "deps", "evidence")
# 状态只回答「做完没有」：judged=已有明确结论（含失败结论）；todo=尚无明确结论。
VALID_STATUS = ("judged", "todo")


def main() -> int:
    errors: list[str] = []
    with CATALOG.open(encoding="utf-8") as f:
        cat = yaml.safe_load(f)

    expected: set[str] = set()
    n_judged = 0
    for a in cat["axes"]:
        for g in [x for x in cat["grids"] if x["axis"] == a["id"]]:
            for k in ("id", "axis", "title", "claim", "options", "judge", "cost", "status"):
                if k not in g:
                    errors.append(f"catalog: {g.get('id', '?')} 缺字段 {k}")
            rel = f"{a['dir']}/{g['id']}"
            expected.add(rel)
            page = ROOT / "axes" / rel / "README.md"
            if not page.exists():
                errors.append(f"缺页：axes/{rel}/README.md")
                continue
            text = page.read_text(encoding="utf-8")
            m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
            if not m:
                errors.append(f"{rel}: 缺 front-matter")
                continue
            meta = yaml.safe_load(m.group(1))
            for k in REQUIRED:
                if k not in meta:
                    errors.append(f"{rel}: front-matter 缺 {k}")
            for s in SECTIONS:
                if s not in text:
                    errors.append(f"{rel}: 缺小节 {s}")
            st = meta.get("status")
            if st not in VALID_STATUS:
                errors.append(f"{rel}: 非法 status={st!r}（合法值：{VALID_STATUS}）")
            elif st == "judged":
                n_judged += 1
                if not meta.get("judge") or meta["judge"].strip() in ("—", "-"):
                    errors.append(f"{rel}: status=judged 但无 judge（判据）")
                if not meta.get("evidence"):
                    errors.append(f"{rel}: status=judged 但无 evidence")
                elif meta.get("verdict", "—") in ("—", "-", None):
                    errors.append(f"{rel}: status=judged 但无 verdict（判读）")
            elif st == "todo":
                v = meta.get("verdict") or ""
                if not any(t in v for t in ("待立项", "尚缺")):
                    errors.append(f"{rel}: status=todo 但 verdict 未写明「待立项」或「尚缺什么」")
            for ev in meta.get("evidence") or []:
                p = str(ev).split(" @")[0].strip()
                if p and not (ROOT / p).exists():
                    errors.append(f"{rel}: evidence 路径不存在：{p}")

    # 孤儿目录
    for ax in sorted((ROOT / "axes").glob("*")):
        if not ax.is_dir():
            continue
        for gd in sorted(ax.iterdir()):
            if not gd.is_dir():
                continue
            if f"{ax.name}/{gd.name}" not in expected:
                errors.append(f"孤儿格子目录（不在 catalog 中）：axes/{ax.name}/{gd.name}")

    # 生成一致性（课题页/索引/轴页）
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "build_wiki.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("生成物与 catalog 不一致：\n" + r.stdout.strip())

    # 历史实验映射表（格子 id 与证据锚点）
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "backfill_legacy.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        errors.append("legacy-experiments 映射表不一致：\n" + (r.stdout or r.stderr).strip())

    if errors:
        print(f"FAIL：{len(errors)} 项")
        for e in errors:
            print("  -", e)
        return 1
    print(f"OK：{len(expected)} 格全部有页并且模板/证据齐备（其中 {n_judged} 格已判）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
