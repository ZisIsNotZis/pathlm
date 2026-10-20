#!/usr/bin/env python3
"""check_docs.py — 文档里的引用必须解析到真实存在的位置（可失败）。

本仓大量事实靠「`文件`」指针、`文件::键` / `文件#标题` 锚点与「`文件:行`」互相引用，指针一旦悬空、
或漂到邻行的另一个字段上，读者只能选择相信。本工具把它们变成机检：

  扫描范围（**只扫这些**）：
    - 仓库根 `*.md`
    - `docs/**/*.md`
    - `axes/**/*.md`
  **不扫** `archive/` 与 `.scratch/`：前者是旧项目只读资产，后者是日志与原始证据
  （`AGENTS.md` §2.2 / §2.3）——它们不在 SSOT 义务内，不要求引用自洽。作为**目标**它们仍可被引用，
  只要目标真实存在即通过。

  三种定位锚点（指向**字段**用锚点，指向**某个具体叙事行**才用行号）：
    - 键锚点（`.yaml` / `.json`）：`路径::键` / `路径::a.b` / `路径::a[0].b`；
    - 标题锚点（`.md`）：`路径#§标题` / `路径#标题`；
    - 行号（`.md` / `.py` / 其它源码）：`路径:行` / `路径:起始-结束`。

  检查六类问题：
    ① `路径:行` 与 `路径:起始行-结束行`：文件必须存在，行号必须在 `1..行数` 内且 `起始 ≤ 结束`。
    ② **禁止**用 `路径:行` 指向 `.yaml` / `.json`：行号会随文件增行而整体漂到别的字段上，
       改用键锚点（报错信息直接写这条规矩）。
    ③ 键锚点（`.yaml` / `.json`）必须解析到**唯一**一处：键不存在、命中多处且收紧不掉、
       JSON 解析失败、下标越界，都是违规。
    ④ 标题锚点（`.md`）必须命中**唯一**一条标题：目标文件里要有一条标题的文本包含该串
       （仓内两种写法都收：`路径#标题` 与 `路径::标题`——同一件事，同一条规矩）。
    ⑤ 裸路径（`*.md` / `*.yaml` / `*.json` / `*.py`）：文件必须存在。
    ⑥ 文档地图完整性：每个事实文档（`docs/*.md` + 仓库根 `*.md`）必须在 `AGENTS.md` §2.1 与
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
    - `UNRESOLVED_KEY_ANCHORS` 里**已声明的悬空键锚点**（开工前的既有状态：锚点检查绑定的是
      本表之外的新引用，见 `@migrate_on_touch`；计数后打印，修好一条就从该表删一条）；
    - **已声明缺失的引用**：引用所在行（**引用之外**的正文）自述该目标「不存在 / 未产出 / 将来才写」
      （`ABSENT_MARKERS`）——`@absent`：缺失是一种状态，不发明路径去凑；这类引用计数后打印，不静默跳过。

  本检**不判**（honest limits，判了也只会给假安全感）：
    1. 引用是否指对了那段**内容**——只校目标与锚点存在、唯一，校不了语义：`::status` 命中的
       那一行确实叫 `status`，但本检不知道你**想不想**指它；
    2. 行号是否仍指向**语意相同**的位置——只校 `1..行数`，「整体错一行」这类漂移抓不到。
       这类漂移的根因不是校得不够严，而是**行号这种定位方式天生会漂**，所以指向 `.yaml` / `.json`
       的行号被 ② 整类禁掉：指向字段用键锚点，行号只留给「某个具体叙事行」；
    3. 通配 / 占位锚点（`arms.*.train_tokens`、`arms.<arm>.intervention`、`arms[*`）——它们描述的
       是一个**模式**而不是一处位置，本检只计数、打进汇总行，不判；
    4. `.yaml` 的行内列表（`laws: [10]`）不支持下标锚点（`::laws[0]`）：行内写法没有「可指向的行」，
       只支持块式列表（`- ` 起头的那些）；
    5. 裸文件名（无目录段的 `analysis.md`、`results.json`）在仓内多处同名时，行号只要求**其中一处**
       成立；`analysis.md:206` 到底指哪个 `analysis.md`，本检判不了；
    6. `见上一节` / `本文件` 这类不带路径的指代；
    7. 跨行断开的路径（上一行以 `/` 结尾、路径接在下一行）只看得到后半句；
    8. 引用所在行自述「不存在 / 未产出 / 将来才写」时，本检**信这句声明**、不计违规：它只证明「文档说了
       它不在」，不证明那个文件应该存在；
    9. 文档地图只校「文件名在地图里出现过」，校不了那一行说的是不是它、与别处是否重复——那是复核的活；
   10. `UNRESOLVED_KEY_ANCHORS` 里声明的悬空键锚点只计数、不判：那句话该不该指别处、该指哪个字段，
       只有读得懂那句话的人能定（`@absent`：缺失是一种状态，不替它发明一个字段）。

用法：
    python3 tools/check_docs.py            # 打印全部违规；有违规即非零退出
    python3 tools/check_docs.py --check    # 同上（CI / pytest 入口；与其它 tools 口径一致）
    python3 tools/check_docs.py --help     # 打印本文件头（扫描范围、锚点规则、不判清单）
"""

from __future__ import annotations

import json
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

#: 一条引用候选：`路径` + 可选一种定位后缀（`::键` / `#标题` / `:行` / `:起始-结束`）。
#: 前面不能紧跟路径字符（否则是长路径的尾巴）。
REF_RE = re.compile(
    r"(?<![\w/.\-])([\w][\w./\-]*\.[A-Za-z0-9]+)"
    r"(?:::([^\s`()），,、。；]+)|#([^\s`()），,、。；]+)|:(\d+)(?:-(\d+))?)?"
)

#: 键锚点：`.yaml` / `.json` 里指向一个字段的路径，形如 `status` / `a.b` / `a[0].b`。
#: 不匹配这把字符表的锚点（`arms.*.x`、`arms.<arm>.x`、`arms[*`）= 模式锚点，只计数、不判。
KEY_ANCHOR_RE = re.compile(r"^[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+|\[\d+\])*$")

#: 键锚点的一段：`键` + 可选若干 `[下标]`。
KEY_SEGMENT_RE = re.compile(r"^([A-Za-z0-9_-]+)((?:\[\d+\])*)$")
KEY_INDEX_RE = re.compile(r"\[(\d+)\]")

#: 键锚点只能指向结构化文件，标题锚点只能指向 Markdown。
KEY_ANCHOR_EXTS = (".yaml", ".json")
HEAD_ANCHOR_EXTS = (".md",)

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
    "docs/triage.md",                # 树分诊
)

#: 「已声明缺失」标记：引用所在行出现这些词，即文档自己声明了目标不存在或将才写出。
#: 命中时不计违规、只计数（`@absent`：缺失是一种状态；本检不替文档发明一个路径）。
ABSENT_MARKERS = (
    "尚未", "不存在", "未产出", "未生成", "未落盘", "未出现", "未跑",
    "可补", "则写", "会写", "即写",
)

#: 已声明的悬空**键锚点**：锚点指向的字段在目标文件里根本不存在，属开工前的既有状态。
#: 形式 = `源文档 → 目标::锚点`（锚在源文档上，行号挪动不影响本表）。
#: `@migrate_on_touch`：锚点检查绑定的是**本表之外的新引用**，否则一个在有存量缺陷的树上
#: 永远打不开的门就是没打开的门。修好一条就从本表删一条（工具会打印已失效的条目）；本表只缩不涨。
UNRESOLVED_KEY_ANCHORS = (
    "axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/analysis.md → results.json::criteria.criterion_1_quality.tax_bpc_native",
    "axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean/analysis.md → results.json::criterion_2_stability_gate",
    "axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean/analysis.md → results.json::gnorm_monitor",
    "axes/1-time/1.1.1-hard-window/experiments/2628-longT-2048/analysis.md → results.json::gnorm_monitor",
    "axes/1-time/1.1.4.1-explicit-memory/experiments/2620-capacity-converged/stability.md → results.json::wall_minutes",
    "axes/1-time/1.1.4.1-explicit-memory/experiments/2622-eng1-composable/MODEL.md → results.json::kv_head_liveness.block_python_class",
    "axes/1-time/1.1.4.1-explicit-memory/experiments/2628-eng2-carry-ragged/MODEL.md → results.json::carry_liveness",
    "axes/1-time/1.1.4.1-explicit-memory/experiments/2628-eng2-carry-ragged/README.md → results.json::carry_liveness",
    "axes/1-time/1.1.4.1-explicit-memory/experiments/2629-carry-ragged-projfill/CALIBER_DECISION.md → results.json::tf_enwik8.by_depth",
    "axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2615-window-trained-retention-13m/analysis.md → mutation_report.json::M4",
    "axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/MODEL.md → config.json::checkpoint.sha256",
    "axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/README.md → results.json::verdict",
    "axes/1-time/1.2.2-block-sparse/experiments/2637-linear-attn-quality/analysis.md → results.json::gnorm_monitor",
    "axes/2-depth/2.2.3-learned-layer-sum/experiments/2613-inter-layer-topology-13m/MODEL.md → smoke/smoke.json::identity_start_bitwise_vs_sequential",
    "axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/README.md → results.json::s0_anchor_remeasured",
    "axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/stability.md → results.json::main_arm",
    "axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m/README.md → results.json::A_arm",
    "axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/analysis.md → results.json::quant_rows",
    "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/README.md → config.json::no_single_point_extrapolation",
    "axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/README.md → results.json::resolved",
    "axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/controls/README.md → results.json::bit_identical_all_arms",
    "axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/analysis.md → results.json::criteria",
    "axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/analysis.md → results.json::criteria",
    "axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/analysis.md → probes_output.json::P3_free_gen_overall.n",
    "axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/README.md → results.json::criteria.V5_cost_ledger",
    "axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/analysis.md → results.json::criteria.V5_cost_ledger",
    "axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/MODEL.md → results.json::wall_minutes",
    "axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/analysis.md → results.json::wall_minutes",
    "axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/analysis.md → results.json::wall_minutes_total_eval",
)

URL_PREFIXES = ("http://", "https://", "ftp://", "www.")


class Reference:
    """文档里的一处引用。"""

    __slots__ = ("source", "lineno", "target", "start", "end", "anchor", "anchor_kind")

    def __init__(self, source: str, lineno: int, target: str,
                 start: int | None = None, end: int | None = None,
                 anchor: str | None = None, anchor_kind: str | None = None) -> None:
        self.source = source          # 引用所在的文件（仓库相对路径）
        self.lineno = lineno          # 引用所在的**行**
        self.target = target          # 被引用的路径（已剥锚点）
        self.start = start            # `路径:行` 的行号；裸路径为 None
        self.end = end                # `路径:起始-结束` 的结束行；无区间时等于 start
        self.anchor = anchor          # 锚点文本（`::` 之后 / `#` 之后）；无锚点为 None
        self.anchor_kind = anchor_kind  # `key`（yaml/json 字段）/ `head`（md 标题）/ None

    def __str__(self) -> str:
        if self.anchor_kind == "key":
            return f"{self.source}:{self.lineno} → {self.target}::{self.anchor}"
        if self.anchor_kind == "head":
            return f"{self.source}:{self.lineno} → {self.target}#{self.anchor}"
        loc = self.target if self.start is None else f"{self.target}:{self.start}"
        if self.end is not None and self.end != self.start:
            loc = f"{self.target}:{self.start}-{self.end}"
        return f"{self.source}:{self.lineno} → {loc}"

    @property
    def key(self) -> str:
        """本引用在 `UNRESOLVED_KEY_ANCHORS` 里的声明形式（不含行号）。"""
        return f"{self.source} → {self.target}::{self.anchor}"


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
            target = m.group(1)
            if _skip_token(target, m.span(1), placeholders):
                continue
            key_anchor, head_anchor, start, end = m.group(2), m.group(3), m.group(4), m.group(5)
            known_ext = Path(target).suffix.lower() in BARE_EXTS
            if key_anchor is not None:
                anchor = key_anchor.rstrip(".")
                if anchor and known_ext:               # 锚点只判 md/yaml/json/py 目标
                    refs.append(Reference(rel, lineno, target,
                                          anchor=anchor, anchor_kind="key"))
                continue
            if head_anchor is not None:
                if known_ext:
                    refs.append(Reference(rel, lineno, target,
                                          anchor=head_anchor, anchor_kind="head"))
                continue
            if start is None:
                if not known_ext:
                    continue                          # 裸路径只查 md/yaml/json/py
                refs.append(Reference(rel, lineno, target))
            else:
                refs.append(Reference(rel, lineno, target, int(start),
                                      int(end) if end else int(start)))
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


def _read(rel: str) -> str | None:
    try:
        return (ROOT / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _segment(part: str) -> tuple[str | None, list[int]]:
    """键锚点的一段 → (键名, 下标列表)；不合语法返回 (None, [])。"""
    m = KEY_SEGMENT_RE.match(part)
    if not m:
        return None, []
    return m.group(1), [int(x) for x in KEY_INDEX_RE.findall(m.group(2))]


def _yaml_block_end(lines: list[str], start: int, stop: int, indent: int) -> int:
    """从 `start` 起，第一条缩进 ≤ `indent` 的非空行所在位置（YAML 里就是本键的块结束）。"""
    for i in range(start, stop):
        line = lines[i]
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent:
            return i
    return stop


def resolve_yaml_key(text: str, anchor: str) -> tuple[int | None, str | None]:
    r"""键锚点 → (命中行号 1-based, 错误描述)。

    逐层收窄：第 i 段只在上一段的块内找 `^\s*键:`，取**该层最小缩进**的那些行；
    一个单段锚点因此落在顶层键上（`::status` 不会命中 `work_packages[*].status`），
    同层仍有多处即报「键不唯一」并提示用 `a.b` 收紧。
    """
    lines = text.splitlines()
    lo, hi, floor = 0, len(lines), -1
    parts = anchor.split(".")
    lineno = None
    for i, part in enumerate(parts):
        key, indices = _segment(part)
        if key is None:
            return None, f"键锚点不合语法：`{part}`"
        pattern = re.compile(r"^(\s*)" + re.escape(key) + r"\s*:")
        hits = []
        for n in range(lo, hi):
            m = pattern.match(lines[n])
            if m and len(m.group(1)) > floor:
                hits.append((n, len(m.group(1))))
        if not hits:
            return None, f"键不存在：`{'.'.join(parts[:i + 1])}`"
        top = min(ind for _, ind in hits)
        hits = [h for h in hits if h[1] == top]
        if len(hits) > 1:
            return None, (f"键不唯一：`{'.'.join(parts[:i + 1])}`（命中 "
                          f"{', '.join(str(h[0] + 1) for h in hits)} 行，用 `a.b` 收紧）")
        lineno, indent = hits[0]
        lo, hi = lineno + 1, _yaml_block_end(lines, lineno + 1, hi, indent)
        floor = indent
        for n in indices:
            items = []
            for j in range(lo, hi):
                m = re.match(r"^(\s*)-(\s|$)", lines[j])
                if m:
                    items.append((j, len(m.group(1))))
            if not items:
                return None, (f"列表不可按下标定位：`{part}`"
                              f"（行内写法（`key: [..]`）没有可指向的行）")
            first = min(ind for _, ind in items)
            items = [it for it in items if it[1] == first]
            if n >= len(items):
                return None, f"下标越界：`{part}`（列表长度 {len(items)}）"
            lineno, indent = items[n]
            lo, hi = lineno + 1, _yaml_block_end(lines, lineno + 1, hi, indent)
            floor = indent
    return lineno + 1, None


def resolve_json_key(text: str, anchor: str) -> tuple[int | None, str | None]:
    """键锚点 → (None, 错误描述)；JSON 按结构解析，成功时行号不可知（返回 None）。"""
    try:
        data = json.loads(text)
    except ValueError as exc:
        return None, f"JSON 解析失败（{exc}）"
    parts = anchor.split(".")
    node = data
    for i, part in enumerate(parts):
        key, indices = _segment(part)
        if key is None:
            return None, f"键锚点不合语法：`{part}`"
        if not isinstance(node, dict) or key not in node:
            return None, f"键不存在：`{'.'.join(parts[:i + 1])}`"
        node = node[key]
        for n in indices:
            if not isinstance(node, list):
                return None, f"不是列表：`{'.'.join(parts[:i + 1])}`"
            if n >= len(node):
                return None, f"下标越界：`{part}[{n}]`（列表长度 {len(node)}）"
            node = node[n]
    return None, None


def _heading_lines(text: str) -> list[tuple[int, str]]:
    """Markdown 标题行（跳过围栏代码块里的 `# 注释`）：(行号, 标题文本)。"""
    out: list[tuple[int, str]] = []
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if m:
            out.append((lineno, m.group(1)))
    return out


def _slug(text: str) -> str:
    """标题 slug 的宽松归一口径：只留字母 / 数字 / 汉字，丢掉空白、连字符、下划线与标点。

    这样 `#§2.1 口径` 与 `#2-1-口径` / `#口径` 都能对上写它的那一行标题，同时不会因为
    「标点写成全角还是半角」判一条正确的引用为悬空。
    """
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.strip().lower())


def resolve_head_anchor(text: str, anchor: str) -> tuple[int | None, str | None]:
    """标题锚点 → (命中行号, 错误描述)：标题文本（归一后）包含该串，且只命中一条。"""
    want = _slug(anchor.lstrip("§"))
    if not want:
        return None, f"标题锚点不合语法：`{anchor}`"
    hits = [(lineno, head) for lineno, head in _heading_lines(text) if want in _slug(head)]
    if not hits:
        return None, f"标题不存在：`{anchor}`（目标文件里没有标题包含这一串）"
    if len(hits) > 1:
        return None, (f"标题不唯一：`{anchor}`（命中 {len(hits)} 条，"
                      f"第 {', '.join(str(h[0]) for h in hits)} 行）")
    return hits[0][0], None


def anchor_kind_judged(ref: Reference) -> str | None:
    """这条锚点用哪个解析器判：`key`（`.yaml` / `.json` 字段）/ `head`（`.md` 标题）/ None（不判）。

    没有解析器的锚点不判、只计数：`.py::符号`（符号表要另开解析器）和点号被当成后缀的
    「研究项 id::字段」写法（`5.4::sota_baseline` —— 它指的是**格子**不是文件，本来就不是路径引用）。
    """
    ext = Path(ref.target).suffix.lower()
    if ext in KEY_ANCHOR_EXTS:
        return "key"
    if ext in HEAD_ANCHOR_EXTS:
        return "head"
    return None


def _anchor_problem(ref: Reference, candidates: list[str], kind: str) -> str | None:
    """一条带锚点的引用在候选文件里是否解析得到（`None` = 通过）。"""
    if kind == "key" and not KEY_ANCHOR_RE.match(ref.anchor or ""):
        return None                     # 模式锚点（通配 / 占位）：只计数，不判
    errors: list[str] = []
    for candidate in dict.fromkeys(candidates):      # 同一个文件可能被两条解析基准各命中一次
        text = _read(candidate)
        if text is None:
            continue
        if kind == "key":
            if Path(candidate).suffix.lower() == ".json":
                error = resolve_json_key(text, ref.anchor or "")[1]
            else:
                error = resolve_yaml_key(text, ref.anchor or "")[1]
        else:
            error = resolve_head_anchor(text, ref.anchor or "")[1]
        if error is None:
            return None
        errors.append(f"{candidate}: {error}")
    return f"{ref}（{'；'.join(errors)}）"


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
          ) -> tuple[list[str], list[str], list[str]]:
    """返回 (违规列表, 已声明缺失的引用, 命中 `UNRESOLVED_KEY_ANCHORS` 的悬空键锚点)。

    每条引用形如 `本文件:行号 → 目标::锚点`（见 `Reference.__str__`）。
    """
    problems: list[str] = []
    declared_absent: list[str] = []
    declared_anchor: list[str] = []
    for ref in refs:
        candidates = resolve(ref, suffix_index)
        if not candidates:
            if _declares_absent(ref):
                declared_absent.append(str(ref))
            else:
                problems.append(f"{ref}（目标不存在）")
            continue
        if ref.start is not None and Path(ref.target).suffix.lower() in KEY_ANCHOR_EXTS:
            problems.append(
                f"{ref}（禁止行号引用 `.yaml` / `.json`：行号会随文件增行而整体漂到别的字段上，"
                f"改用键锚点 `{ref.target}::键`，嵌套字段用 `::a.b`）")
            continue
        if any((ROOT / c).is_dir() for c in candidates):
            problems.append(f"{ref}（目标是目录）")
            continue
        if ref.anchor_kind is not None:
            if ref.anchor_kind == "head" and Path(ref.target).suffix.lower() not in HEAD_ANCHOR_EXTS:
                problems.append(f"{ref}（标题锚点只能指向 `.md`）")
                continue
            kind = anchor_kind_judged(ref)
            if kind is None:
                continue                # 没有解析器的锚点（`.py::符号`）：不判，只计数
            problem = _anchor_problem(ref, candidates, kind)
            if problem is None:
                continue
            if _declares_absent(ref):
                declared_absent.append(str(ref))
            elif ref.key in UNRESOLVED_KEY_ANCHORS:
                declared_anchor.append(ref.key)
            else:
                problems.append(problem)
            continue
        if ref.start is None:
            continue
        if ref.start < 1 or ref.end < 1 or ref.start > ref.end:
            problems.append(f"{ref}（行号区间非法）")
            continue
        counts = [_line_count(c) for c in candidates]
        if not any(ref.end <= n for n in counts):
            best = max(counts)
            problems.append(f"{ref}（行号越界：{ref.start}-{ref.end} 超出 1..{best}）")
    return problems, declared_absent, declared_anchor


def main(argv: list[str]) -> int:      # `--check` 与默认同一行为，只为与其它 tools 对齐
    if any(arg in ("-h", "--help") for arg in argv):
        print(__doc__.strip())
        return 0
    files = source_files()
    suffix_index = _build_suffix_index()
    refs: list[Reference] = []
    for path in files:
        refs.extend(extract(path))
    line_refs = [r for r in refs if r.start is not None]
    key_refs = [r for r in refs if r.anchor_kind == "key"
                and Path(r.target).suffix.lower() in KEY_ANCHOR_EXTS]
    head_refs = [r for r in refs if r.anchor_kind is not None
                 and anchor_kind_judged(r) == "head"]
    symbol_refs = [r for r in refs if r.anchor_kind is not None
                   and anchor_kind_judged(r) is None]
    bare_refs = [r for r in refs if r.anchor_kind is None and r.start is None]
    pattern_refs = [r for r in key_refs if not KEY_ANCHOR_RE.match(r.anchor or "")]
    banned = [r for r in line_refs if Path(r.target).suffix.lower() in KEY_ANCHOR_EXTS]
    problems, declared_absent, declared_anchor = check(refs, suffix_index)
    map_problems = check_doc_map()
    for p in problems + map_problems:
        print(p)
    print(f"扫描 {len(files)} 个文档，检查 {len(refs)} 处引用"
          f"（{len(line_refs)} 处 路径:行 / {len(key_refs)} 处 键锚点 / "
          f"{len(head_refs)} 处 标题锚点 / {len(symbol_refs)} 处 符号锚点 / "
          f"{len(bare_refs)} 处 裸路径）："
          f"{len(problems) + len(map_problems)} 处违规；"
          f"{len(declared_absent)} 处已声明缺失（文档自述不存在/未产出，不计违规）；"
          f"待建交付物不检：{', '.join(PENDING_TARGETS)}。")
    stale = [k for k in UNRESOLVED_KEY_ANCHORS if k not in set(declared_anchor)]
    print(f"锚点：键锚点 {len(key_refs)} 处（`.yaml` / `.json`，键必须存在且唯一），"
          f"其中 {len(pattern_refs)} 处为模式锚点（通配 / 占位，不判）；"
          f"标题锚点 {len(head_refs)} 处（`.md`，标题文本必须唯一命中）；"
          f"符号锚点 {len(symbol_refs)} 处（`.py::符号`，无解析器，不判）；"
          f"指向 `.yaml` / `.json` 的 路径:行 引用 {len(banned)} 处（**禁止**项）；"
          f"已声明悬空键锚点 {len(declared_anchor)} 处（既有，不计违规；"
          f"失效声明 {len(stale)} 条，应从 `UNRESOLVED_KEY_ANCHORS` 删除）。")
    by_ext: dict[str, int] = {}
    for ref in line_refs:
        suffix = Path(ref.target).suffix.lower() or "（无后缀）"
        by_ext[suffix] = by_ext.get(suffix, 0) + 1
    breakdown = " / ".join(f"{ext} {n} 处" for ext, n in sorted(by_ext.items()))
    print(f"仍保留的 路径:行 引用 {len(line_refs)} 处：{breakdown or '无'}"
          f"（行号只留给「某个具体叙事行」，不用于指向字段）。")
    print(f"文档地图：{len(doc_map_files())} 个事实文档在 "
          f"{' 与 '.join(rel + ' 的「' + start + '」' for rel, start, _ in DOC_MAP_SECTIONS)} "
          f"里各有一行；地图缺项 {len(map_problems)} 个。")
    return 1 if (problems or map_problems) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
