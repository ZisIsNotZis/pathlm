#!/usr/bin/env python3
"""check_schema.py — Research Wiki 的 CI 校验（可失败）。

校验项：
  1. 目录完备：catalog 枚举的每一格都有实体目录与 README.md（缺 → fail）。
  2. 模板完备：每页 front-matter 的必需字段齐全、九节齐全。
  3. 证据完备：status=judged 必须有 judge 与 evidence；status=todo 无本地证据，
     但 verdict 必须写明「待立项」或「尚缺什么」（待补做）。
     evidence 里 `path @commit` 的 path 必须在磁盘上存在（归档/证据锚点不悬空）。
     并且所有 `@<sha>` 锚点必须是**真实提交**（refs 可达历史中的 commit）——「路径存在」
     不等于「锚点有效」；无效锚点逐条列出并 fail。git 不可用 ⇒ 显式报错，不得静默跳过。
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


ANCHOR_RE = re.compile(r"@([0-9a-f]{7,40})")


class GitUnavailable(RuntimeError):
    """git 不可用（未安装 / 不是仓库）——锚点校验必须显式报错，不得静默跳过。"""


def reachable_commits() -> set[str]:
    """一次取回所有 ref 可达的完整提交 hash（批处理；不对每个锚点单独 spawn git）。"""
    r = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise GitUnavailable((r.stderr or r.stdout or "git rev-list --all 失败").strip())
    return set(r.stdout.split())


def check_anchors(anchors: dict[str, list[str]]) -> list[str]:
    """校验所有 `@<sha>` 锚点都指向真实提交（可达 ref 历史中的 commit）。

    anchors：{锚点 hex 串 -> [出现位置,...]}。返回错误清单（空 = 全绿）。
    git 不可用 ⇒ 返回显式错误（不得静默跳过）。
    """
    if not anchors:
        return []
    try:
        reach = reachable_commits()
    except GitUnavailable as e:
        return [f"git 不可用，无法校验证据锚点（不得静默跳过）：{e}"]
    # 批匹配：锚点必须是对某个可达提交 hash 的前缀（7–40 位缩写均可）。
    bad = sorted(a for a in anchors if not any(s.startswith(a) for s in reach))
    if not bad:
        return []
    # 对差分集一次性确认：区分「对象不存在」与「对象存在但不在历史中」。
    detail: dict[str, str] = {}
    r = subprocess.run(["git", "cat-file", "--batch-check"], cwd=ROOT,
                       input="\n".join(bad) + "\n", capture_output=True, text=True)
    lines = r.stdout.splitlines()
    if r.returncode == 0 and len(lines) == len(bad):
        for a, line in zip(bad, lines):        # batch-check 按输入顺序逐行输出
            parts = line.split()
            detail[a] = "missing" if len(parts) >= 2 and parts[1] == "missing" else "unreachable"
    kind = {"missing": "对象不存在", "unreachable": "对象存在但不在任何 ref 的历史中"}
    return [f"证据锚点无效：@{a}（{kind.get(detail.get(a), '不是可达的真实提交')}；出现于 "
            f"{'、'.join(anchors[a])}）" for a in bad]


def main() -> int:
    errors: list[str] = []
    with CATALOG.open(encoding="utf-8") as f:
        cat = yaml.safe_load(f)

    expected: set[str] = set()
    n_judged = 0
    anchors: dict[str, list[str]] = {}      # 收集全部 `@<sha>` 锚点 -> 出现位置
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
                for anc in ANCHOR_RE.findall(str(ev)):
                    anchors.setdefault(anc, []).append(f"axes/{rel}")
            # catalog 是 SSOT：即使页面未重生成，也校验 catalog 里的证据锚点。
            for ev in ((g.get("local") or {}).get("evidence") or []):
                for anc in ANCHOR_RE.findall(str(ev)):
                    anchors.setdefault(anc, []).append(f"catalog:{g.get('id', '?')}")
            for anc in ANCHOR_RE.findall(str(g.get("prior_art_source") or "")):
                anchors.setdefault(anc, []).append(f"catalog:{g.get('id', '?')}")

    # 孤儿目录
    for ax in sorted((ROOT / "axes").glob("*")):
        if not ax.is_dir():
            continue
        for gd in sorted(ax.iterdir()):
            if not gd.is_dir():
                continue
            if f"{ax.name}/{gd.name}" not in expected:
                errors.append(f"孤儿格子目录（不在 catalog 中）：axes/{ax.name}/{gd.name}")

    # 证据锚点必须是真实提交（批处理；不逐个 spawn git）
    errors.extend(check_anchors(anchors))

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
