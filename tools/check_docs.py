#!/usr/bin/env python3
"""check_docs.py — 文档里的引用必须解析到真实存在的位置（可失败）。

本仓大量事实靠「`文件:行`」与「`文件`」指针互相引用，指针一旦悬空，读者只能选择相信。
本工具把它们变成机检：

  扫描范围（**只扫这些**）：
    - 仓库根 `*.md`
    - `docs/**/*.md`
    - `axes/**/*.md`
  **不扫** `archive/` 与 `.scratch/`：前者是旧项目只读资产，后者是日志与原始证据
  （`AGENTS.md` §2.2 / §2.3）——它们不在 SSOT 义务内，不要求引用自洽。作为**目标**它们仍可被引用，
  只要目标真实存在即通过。

  检查三类问题：
    ① `路径:行` 与 `路径:起始行-结束行`：文件必须存在，行号必须在 `1..行数` 内且 `起始 ≤ 结束`。
    ② 裸路径（`*.md` / `*.yaml` / `*.json` / `*.py`）：文件必须存在。
    ③ 文档地图完整性：每个事实文档（`docs/*.md` + 仓库根 `*.md`）必须在 `AGENTS.md` §2.1 与
       `docs/design.md` §4 的地图里**各有一行**（新事实文档不接进地图 = 读者找不到它）。

  解析基准（按序，先命中先用）：仓库根 → 本文件所在目录 → 仓库内任意以该路径结尾的文件
  （仓内引用常把实验目录写在路径里、把上层省略，如 `2634-.../analysis.md`、`2619/run.py`）。
  前两个基准是「强候选」，命中即只按它判；都没有时才退到后缀匹配。

  跳过（写死，见 `_skip_token`）：
    - 围栏代码块（``` 之间）里的内容：示例代码，不是引用；
    - URL（`http://` / `https://`）；
    - 占位符片段：`<...>`、`{...}`、`...`、`…`、`path/to`、`xxx`，以及紧贴它们的片段
      （`results_<arm>_s0.json` 是一个示例表达式，不是一条引用）；
    - 绝对路径（`/home/...`）：不是仓内引用；
    - 以不入库目录或第三方源码根开头的路径（`runs/`、`checkpoints/`、`.tmp/`、`data/`、`wandb/`、
      `torch/`、`site-packages/`）：干净 clone 里本就不存在；
    - `PENDING_TARGETS` 里**已声明待建**的交付物（数量打在汇总行里，落盘后必须从该表删除）；
    - **已声明缺失的引用**：引用所在行（**引用之外**的正文）自述该目标「不存在 / 未产出 / 将来才写」
      （`ABSENT_MARKERS`）——`@absent`：缺失是一种状态，不发明路径去凑；这类引用计数后打印，不静默跳过。

  本检**不判**（honest limits，判了也只会给假安全感）：
    1. 引用是否指对了那段**内容**——只校目标存在，校不了语义；
    2. 行号是否仍指向**语意相同**的位置——只校 `1..行数`，「整体错一行」这类漂移抓不到；
    3. `path::§N` / `path#§标题` 一类自造锚点的语义——只校 `:` / `#` 之前的路径存在；
    4. 裸文件名（无目录段的 `analysis.md`、`results.json`）在仓内多处同名时，行号只要求**其中一处**
       成立；`analysis.md:206` 到底指哪个 `analysis.md`，本检判不了；
    5. `见上一节` / `本文件` 这类不带路径的指代；
    6. 跨行断开的路径（上一行以 `/` 结尾、路径接在下一行）只看得到后半句；
    7. 引用所在行自述「不存在 / 未产出 / 将来才写」时，本检**信这句声明**、不计违规：它只证明「文档说了
       它不在」，不证明那个文件应该存在；
    8. 文档地图只校「文件名在地图里出现过」，校不了那一行说的是不是它、与别处是否重复——那是复核的活。

用法：
    python3 tools/check_docs.py            # 打印全部违规；有违规即非零退出
    python3 tools/check_docs.py --check    # 同上（CI / pytest 入口；与其它 tools 口径一致）
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 扫描源文件：仓库根 `*.md`、`docs/**/*.md`、`axes/**/*.md`（`archive/`、`.scratch/` 不扫）。
SOURCE_GLOBS = ("*.md", "docs/**/*.md", "axes/**/*.md")

#: 后缀索引 `os.walk` 时要剪掉的目录：与仓库内容无关的缓存/产物（`runs/`、`checkpoints/` 不入库）。
INDEX_PRUNE = frozenset({
    ".git", "__pycache__", ".venv", "node_modules", "runs", "checkpoints",
    ".tmp", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".cache", "data",
})

#: 裸路径**要**检查的扩展名（`docs/research-schema.md` §1 之外的文件如 `.pt`/`.npz` 不检）。
BARE_EXTS = (".md", ".yaml", ".json", ".py")

#: 不入库目录 / 第三方源码根：指向它们的引用不检（干净 clone 里不存在，且不属本仓事实）。
SKIP_PREFIXES = (
    "runs/", "checkpoints/", "wandb/", ".tmp/", "data/",
    "torch/", "site-packages/", "numpy/",
)

#: 占位符片段：与引用 token 重叠时整条跳过（`results_<arm>_s0.json`、`{a.json,b.json}`、`x/.../y.json`）。
PLACEHOLDER_RE = re.compile(r"<[^<>\n]{0,80}>|\{[^{}\n]{0,120}\}|\.\.\.|…|path/to|\bxx+\b")

#: 一条引用候选：可选 `:行` / `:起始-结束`。前面不能紧跟路径字符（否则是长路径的尾巴）。
REF_RE = re.compile(r"(?<![\w/.\-])([\w][\w./\-]*\.[A-Za-z0-9]+)(?::(\d+)(?:-(\d+))?)?")

#: 引用可带的自造锚点后缀（只做剥离，不判语义）：`::§N`、`#§标题`。
ANCHOR_RE = re.compile(r"(?:::§[^`\s）)]*|#§[^`\s）)]*)$")

#: 文档地图必须覆盖的文件：`docs/*.md` 事实文档 + 仓库根的事实文档。
#: `axes/**` 的实验级文件由 `AGENTS.md` §2.1「单个实验」那一行统一覆盖，不逐个进图。
DOC_MAP_GLOBS = ("docs/*.md", "*.md")

#: 两张地图（文件，起点标题，下一节标题）：每个事实文档必须在两处**各有一行**。
DOC_MAP_SECTIONS = (
    ("AGENTS.md", "## 2. 文档地图", "## 3. 编排制度"),
    ("docs/design.md", "## 4. 仓库布局", "## 5. 代码布局"),
)

#: 围栏代码块开头。
FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: `docs/plan.md` §2 点名而**尚未落盘**的工作包交付物：声明「缺失」而不是假装存在。
#: 它们是计划层的未来路径，落盘后必须从这里删除（`@migrate_on_touch`：碰到就清）。
PENDING_TARGETS = (
    "tools/inventory_runs.py",      # 复现清单
    "docs/repro.md",                 # 复现清单
    "docs/triage.md",                # 树分诊
)

#: 「已声明缺失」标记：引用所在行出现这些词，即文档自己声明了目标不存在或将才写出。
#: 命中时不计违规、只计数（`@absent`：缺失是一种状态；本检不替文档发明一个路径）。
ABSENT_MARKERS = (
    "尚未", "不存在", "未产出", "未生成", "未落盘", "未出现", "未跑",
    "可补", "则写", "会写", "即写",
)

URL_PREFIXES = ("http://", "https://", "ftp://", "www.")


class Reference:
    """文档里的一处引用。"""

    __slots__ = ("source", "lineno", "target", "start", "end")

    def __init__(self, source: str, lineno: int, target: str,
                 start: int | None = None, end: int | None = None) -> None:
        self.source = source      # 引用所在的文件（仓库相对路径）
        self.lineno = lineno      # 引用所在的**行**
        self.target = target      # 被引用的路径（已剥锚点）
        self.start = start        # `路径:行` 的行号；裸路径为 None
        self.end = end            # `路径:起始-结束` 的结束行；无区间时等于 start

    def __str__(self) -> str:
        loc = self.target if self.start is None else f"{self.target}:{self.start}"
        if self.end is not None and self.end != self.start:
            loc = f"{self.target}:{self.start}-{self.end}"
        return f"{self.source}:{self.lineno} → {loc}"


def source_files() -> list[Path]:
    """要扫描的源文件（去重、排序；`archive/` 与 `.scratch/` 天然不在 glob 里）。"""
    files: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file():
                files.add(p)
    return sorted(files)


def _build_suffix_index() -> dict[str, list[str]]:
    """仓库内所有文件的路径后缀 → 文件列表（`a/b/c.md` 登记 `a/b/c.md`、`b/c.md`、`c.md`）。"""
    index: dict[str, list[str]] = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in INDEX_PRUNE)
        for name in filenames:
            rel = Path(dirpath, name).relative_to(ROOT).as_posix()
            parts = rel.split("/")
            for i in range(len(parts)):
                index.setdefault("/".join(parts[i:]), []).append(rel)
    return index


def _skip_token(token: str, span: tuple[int, int],
                placeholders: list[tuple[int, int]]) -> bool:
    """这条 token 是否该跳过（不是仓内引用 / 不是可判的引用）。"""
    if token.startswith(URL_PREFIXES):
        return True
    for lo, hi in placeholders:
        if span[0] <= hi and lo <= span[1]:     # 与占位符片段重叠或紧贴（`<arm>_s0.json`）
            return True
    if token in PENDING_TARGETS:                 # 已声明待建（不是悬空指针）
        return True
    return token.startswith(SKIP_PREFIXES)


def extract(path: Path) -> list[Reference]:
    """从一个文档里抽出全部引用（跳过围栏代码块、占位符、URL 等，见 `_skip_token`）。"""
    rel = path.relative_to(ROOT).as_posix()
    refs: list[Reference] = []
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip():
            continue
        placeholders = [m.span() for m in PLACEHOLDER_RE.finditer(line)]
        for m in REF_RE.finditer(line):
            token = m.group(1)
            if _skip_token(token, m.span(1), placeholders):
                continue
            target = ANCHOR_RE.sub("", token)
            if not target:
                continue
            start = int(m.group(2)) if m.group(2) else None
            if start is None:
                if Path(target).suffix.lower() not in BARE_EXTS:
                    continue                          # 裸路径只查 md/yaml/json/py
                refs.append(Reference(rel, lineno, target))
            else:
                end = int(m.group(3)) if m.group(3) else start
                refs.append(Reference(rel, lineno, target, start, end))
    return refs


def resolve(ref: Reference, suffix_index: dict[str, list[str]]) -> list[str]:
    """把一条引用解析成仓库相对路径候选（空列表 = 解析不到）。"""
    strong = []
    for base in (ROOT, (ROOT / ref.source).parent):
        cand = (base / ref.target)
        if cand.is_file():
            strong.append(cand.relative_to(ROOT).as_posix())
    if strong:
        return strong
    return sorted(suffix_index.get(ref.target, []))


def _line_text(ref: Reference) -> str:
    """引用所在的整行文本（判「已声明缺失」用）。"""
    try:
        lines = (ROOT / ref.source).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return lines[ref.lineno - 1] if ref.lineno <= len(lines) else ""


def _declares_absent(ref: Reference) -> bool:
    """引用**之外**的正文是否自述该目标缺失（引用的文件名本身不算声明）。"""
    line = _line_text(ref).replace(ref.target, "")
    return any(m in line for m in ABSENT_MARKERS)


def _line_count(rel: str) -> int:
    try:
        return len((ROOT / rel).read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def _section_text(rel: str, start_marker: str, end_marker: str) -> str:
    """取一个 Markdown 文件里从 `start_marker` 到此后第一个 `end_marker` 的正文（找不到起点返回空）。"""
    text = (ROOT / rel).read_text(encoding="utf-8")
    i = text.find(start_marker)
    if i < 0:
        return ""
    j = text.find(end_marker, i + len(start_marker))
    return text[i:] if j < 0 else text[i:j]


def doc_map_files() -> list[str]:
    """必须进图的事实文档（仓库相对路径，排序）。"""
    files: set[str] = set()
    for pattern in DOC_MAP_GLOBS:
        for p in ROOT.glob(pattern):
            if p.is_file():
                files.add(p.relative_to(ROOT).as_posix())
    return sorted(files)


def check_doc_map() -> list[str]:
    """每个事实文档都要在两张文档地图里各有一行（否则 `AGENTS.md` §2.1 的地图会静默漏文件）。"""
    problems: list[str] = []
    sections = []
    for rel, start_marker, end_marker in DOC_MAP_SECTIONS:
        body = _section_text(rel, start_marker, end_marker)
        if not body:
            problems.append(f"文档地图：`{rel}` 里找不到「{start_marker}」这一节")
        sections.append((rel, body))
    for rel in doc_map_files():
        name = rel.split("/")[-1]
        for map_rel, body in sections:
            if body and name not in body:
                problems.append(f"文档地图：`{rel}` 不在 `{map_rel}` 的地图里（存在的事实文档必须有一行）")
    return problems


def check(refs: list[Reference], suffix_index: dict[str, list[str]]
          ) -> tuple[list[str], list[str]]:
    """返回 (违规描述列表, 已声明缺失的引用列表)，形如 `本文件:行号 → 目标`。"""
    problems: list[str] = []
    declared_absent: list[str] = []
    for ref in refs:
        candidates = resolve(ref, suffix_index)
        if not candidates:
            if _declares_absent(ref):
                declared_absent.append(str(ref))
            else:
                problems.append(f"{ref}（目标不存在）")
            continue
        if ref.start is None:
            continue
        if any((ROOT / c).is_dir() for c in candidates):
            problems.append(f"{ref}（目标是目录）")
            continue
        if ref.start < 1 or ref.end < 1 or ref.start > ref.end:
            problems.append(f"{ref}（行号区间非法）")
            continue
        counts = [_line_count(c) for c in candidates]
        if not any(ref.end <= n for n in counts):
            best = max(counts)
            problems.append(f"{ref}（行号越界：{ref.start}-{ref.end} 超出 1..{best}）")
    return problems, declared_absent


def main(argv: list[str]) -> int:      # `--check` 与默认同一行为，只为与其它 tools 对齐
    del argv
    files = source_files()
    suffix_index = _build_suffix_index()
    refs: list[Reference] = []
    for path in files:
        refs.extend(extract(path))
    line_refs = sum(1 for r in refs if r.start is not None)
    problems, declared_absent = check(refs, suffix_index)
    map_problems = check_doc_map()
    for p in problems + map_problems:
        print(p)
    print(f"扫描 {len(files)} 个文档，检查 {len(refs)} 处引用"
          f"（{line_refs} 处 路径:行 / {len(refs) - line_refs} 处 裸路径）："
          f"{len(problems) + len(map_problems)} 处违规；"
          f"{len(declared_absent)} 处已声明缺失（文档自述不存在/未产出，不计违规）；"
          f"待建交付物不检：{', '.join(PENDING_TARGETS)}。")
    print(f"文档地图：{len(doc_map_files())} 个事实文档在 "
          f"{' 与 '.join(rel + ' 的「' + start + '」' for rel, start, _ in DOC_MAP_SECTIONS)} "
          f"里各有一行；地图缺项 {len(map_problems)} 个。")
    return 1 if (problems or map_problems) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
