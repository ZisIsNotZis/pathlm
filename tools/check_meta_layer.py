#!/usr/bin/env python3
"""check_meta_layer.py — 元级文档里不许出现研究级读数（可失败）。

**为什么需要它**：元级（仓库根 + `docs/`）要回答的只有「树上有哪些研究项、各层拥有什么事实、
谁可以改、下一步派什么活」。一次实验读数一旦写进元级，元级读者每次读状态都要穿过它，
而它已经有家（`axes/**/experiments/**/analysis.md`、`atlas/cost-table.md`）——
同一事实两个家，元级那个必然先过期（`@ssot` / `@derived`）。

**扫描面**（元级**作者写**的文档，不是生成物）：

  - 仓库根 `README.md`、`AGENTS.md`
  - `docs/**/*.md`

**不扫**（每条都有理由，理由本身就是它不被扫的原因）：

  - `CONTRIBUTING.md`：流程文档，不属元级事实层（本文件只判元级事实文档）；
  - `atlas/**`、`harness/**`、`axes/**`：研究级与工级，读数本来就该在那儿；
  - `archive/`、`.scratch/`：只读旧资产与日志，不在 SSOT 义务内；
  - **生成物投影**（`GENERATED` 表）：它的正文逐字来自上游事实源，改了也会被生成器的
    `--check` 判红，所以它不是「元级作者写进去的读数」；本检只判**人能改**的文档。

**判什么**（每条都给出具体命中的那一行，不是「疑似」）：

  | 规则 | 命中的形态（例） | 它为什么是研究级 |
  |---|---|---|
  | `bpc 读数` | `1.4836 bpc`、`bpc 1.4836` | 实验结果，家是实验的 `analysis.md` |
  | `Δ 读数` | `Δ +0.000101`、`Δ = 0.014` | 同上，且方向由各研究的 `delta_direction` 定义 |
  | `显存/内存读数` | `1.14 GiB`、`0.98 GiB` | 实测资源占用，家是 `results.json` |
  | `墙钟/延迟读数` | `23.2 ms`、`墙钟 41 分钟` | 实测时间，口径按 `harness/methodology.md` |
  | `倍数/比例读数` | `2.58×`、`1.41×` | 由两个读数派生的比值，两个家都不在元级 |
  | `百分比读数` | `46.9%`、`+2.592pp` | 同上 |
  | `逐实验目录路径` | `axes/…/experiments/<实验名>/…` | 逐格事实的地址，元级只认研究项一级 |
  | `逐格判词计数` | `已完成 13`、`被否证 5` | 由 46 个 `project.yaml` 派生的计数，家是生成目录 |

**不判**（honest limits，判了只会给假安全感）：

  1. **不带数字的术语**：`bpc`、`Δ`、`显存`、`墙钟` 作为词出现在**定义句**里是必须的
     （否则读者不知道禁令指的是什么），本检只在它们带上数字时判红；
  2. 整数规格（`24 GB` 的单卡显存、`190 MB` 的数据文件）——那是环境与仓库体量，不是实验读数，
     所以显存规则只收**带小数**的读数；
  3. 判据**常数**（`0.014 bpc 带`）：它是在测之前定的门槛（家 `harness/methodology.md`），
     不是测出来的东西——但本检**没有**能力把它和读数区分开，所以这类文字不进元级作者文档，
     只可能出现在生成物投影里（见上）；
  4. 一段读数式散文（「掉了一点质量」）里没有可机判的形态：那要复核者读，本检不假装抓得住。

**白名单**（`ALLOWLIST`，只放**定义性说明**）：一条 `(文件, 片段, 理由)`，命中的行只要包含该片段
就放行。它只收「这行是在**定义**规则、不是在**报告**读数」的行；要放一条读数进去，等于承认
那条读数该留在元级，那就应当改规则，而不是加白名单。

用法：
    python3 tools/check_meta_layer.py            # 打印全部命中；有命中即非零退出
    python3 tools/check_meta_layer.py --check    # 同上（CI / pytest 入口；与其它 tools 口径一致）
    python3 tools/check_meta_layer.py --help     # 打印本文件头（扫描面、规则、不判清单）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 扫描面：元级作者写的文档（仓库相对路径的 glob）。
META_GLOBS = ("README.md", "AGENTS.md", "docs/**/*.md")

#: 生成物投影：正文逐字来自上游事实源（由各自的生成器 `--check` 保证不漂），
#: 不是「元级作者写进去的读数」，所以不判（理由必须写清，别处不许照抄这条豁免）。
GENERATED = {
    "docs/catalog.md": "`tools/build_catalog.py` 从 `axes/**/project.yaml` 逐字生成",
}

#: 判红规则：(规则名, 正则, 一句话说明)。
RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("bpc 读数",
     re.compile(r"(?:[+\-−±]?\d[\d,]*(?:\.\d+)?\s*bpc)|(?:bpc\s*[=≈~]?\s*[+\-−±]?\d)"),
     "实验结果读数：家是实验的 analysis.md / atlas/cost-table.md"),
    ("Δ 读数",
     re.compile(r"Δ\s*[=≈:：]?\s*[+\-−±]?\d(?:\.\d+)?"),
     "配对差值：方向由各研究的 delta_direction 定义，家不在元级"),
    ("显存/内存读数",
     re.compile(r"\d+\.\d+\s*(?:GiB|GB|MiB|MiB|MiB|MB)\b"),
     "实测资源占用：家是 results.json"),
    ("墙钟/延迟读数",
     re.compile(r"\d+(?:\.\d+)?\s*(?:ms|毫秒)\b|墙钟[^。\n]{0,16}\d(?:\.\d+)?"),
     "实测时间：口径见 harness/methodology.md，家不在元级"),
    ("倍数/比例读数",
     re.compile(r"\d+\.\d+\s*×"),
     "由两个读数派生的比值：两个家都不在元级"),
    ("百分比读数",
     re.compile(r"\d+(?:\.\d+)?\s*(?:%|pp|百分点)"),
     "派生比例：家不在元级"),
    ("逐实验目录路径",
     re.compile(r"axes/[\w./\-]*experiments/"),
     "逐格事实的地址：元级只认研究项一级（axes/<轴>/<研究项>/）"),
    ("逐格判词计数",
     re.compile(r"(?:已完成|被否证|进行中|未开始)\s*\d+|\d+\s*(?:个研究项)?[^。\n|]{0,4}(?:已完成|被否证)\s*\d"),
     "由 46 个 project.yaml 派生的计数：家是生成目录 docs/catalog.md"),
)

#: 白名单：只放**定义性说明**（这行在定义规则，不在报告读数）。`(文件, 片段, 理由)`。
ALLOWLIST: tuple[tuple[str, str, str], ...] = ()


class Violation:
    __slots__ = ("path", "lineno", "rule", "why", "hit", "line")

    def __init__(self, path: str, lineno: int, rule: str, why: str, hit: str, line: str) -> None:
        self.path = path
        self.lineno = lineno
        self.rule = rule
        self.why = why
        self.hit = hit
        self.line = line

    def __str__(self) -> str:
        return (f"{self.path}:{self.lineno} [{self.rule}] 命中 `{self.hit}`"
                f"（{self.why}）：{self.line.strip()[:160]}")


def meta_files() -> list[str]:
    """扫描面里的文件（仓库相对路径，排序；生成物投影不在此列时也不扫）。"""
    out: set[str] = set()
    for pattern in META_GLOBS:
        for path in ROOT.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel in GENERATED:
                continue
            out.add(rel)
    return sorted(out)


def _fenced_lines(lines: list[str]) -> set[int]:
    """围栏代码块里的行号（示例代码不是散文；判它只会误报）。"""
    out: set[int] = set()
    inside = False
    for i, line in enumerate(lines, 1):
        if re.match(r"^\s*(```|~~~)", line):
            inside = not inside
            out.add(i)
            continue
        if inside:
            out.add(i)
    return out


def _allowed(rel: str, line: str) -> bool:
    return any(rel == path and snippet in line for path, snippet, _ in ALLOWLIST)


def scan_text(rel: str, text: str) -> list[Violation]:
    """一段元级正文里的全部命中（供测试直接喂文本）。"""
    lines = text.splitlines()
    fenced = _fenced_lines(lines)
    out: list[Violation] = []
    for lineno, line in enumerate(lines, 1):
        if lineno in fenced or not line.strip():
            continue
        if _allowed(rel, line):
            continue
        for rule, pattern, why in RULES:
            m = pattern.search(line)
            if m:
                out.append(Violation(rel, lineno, rule, why, m.group(0), line))
                break                       # 一行只报一条（同一行多命中不多算违规）
    return out


def scan() -> tuple[list[Violation], list[str]]:
    """扫全部分元级文档 → (命中, 扫过的文件)。"""
    violations: list[Violation] = []
    files = meta_files()
    for rel in files:
        try:
            text = (ROOT / rel).read_text(encoding="utf-8")
        except OSError as exc:                                  # pragma: no cover - 环境异常
            raise SystemExit(f"FAIL：读不出 `{rel}`（{exc}）")
        violations.extend(scan_text(rel, text))
    return violations, files


def main(argv: list[str]) -> int:
    if any(a in ("-h", "--help") for a in argv):
        print(__doc__.strip())
        return 0
    violations, files = scan()
    for v in violations:
        print(v)
    exempt = "、".join(f"`{k}`（{v}）" for k, v in GENERATED.items())
    print(f"扫描元级 {len(files)} 个文档（{', '.join(files) or '—'}）："
          f"{len(violations)} 处研究级读数；"
          f"生成物投影不判：{exempt or '—'}；"
          f"白名单 {len(ALLOWLIST)} 条（只放定义性说明）。")
    if violations:
        print("FAIL：元级文档里出现了研究级事实；把读数搬回它自己的家（研究级 / 工级），"
              "元级只留指针。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
