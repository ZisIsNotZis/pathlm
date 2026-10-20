#!/usr/bin/env python3
"""inventory_runs.py — 复现清单：每个实验有哪些 run、权重与日志还在不在、能复跑到什么程度。

本文件是**生成物**（render）：`atlas/repro.md` 由它从**已入库路径**（`git ls-files`）算出，
禁止手改。它回答论文与外部读者问的一个问题——「这个数字能不能重跑出来」——用的不是散文，
而是逐个实验的 run 目录、已入库产物、重跑命令与三态判定。

用法：
    python3 tools/inventory_runs.py                 # 写 atlas/repro.md
    python3 tools/inventory_runs.py --check         # 与磁盘逐字节比对 + 每条路径存在 + 每条 not_reproducible 写明缺什么
    python3 tools/inventory_runs.py --json          # 把同一份清单打到 stdout（供机器读）
    python3 tools/inventory_runs.py --local         # 追加「本机视图」：本机存在但未入库的权重（字节数 + sha256）与 runs/ 目录
    python3 tools/inventory_runs.py --root DIR --check   # 对另一个树跑同一套检查（tests/ 用）

**为什么只列已入库路径**：`.gitignore` 排除了 `*.pt`（权重）、`runs/`、`checkpoints/`、`data/`、`.tmp/`。
这些是本机产物，任何别的克隆都没有；把它们写进生成物，生成物就与别的机器不一致，`--check` 也就
没法在任何克隆上通过。所以权重与 `runs/` 的存在性只能按需用 `--local` 现场量，不进本文件的表。

**三态（判定规则见 `_verdict`，每条都带依据，不猜）**：
    reproducible      克隆内可重算：入口脚本不触训练数据 / 引擎 / CUDA，数字来自仓内已入库的
                      `results*.json`，重跑是纯 CPU 重算，不依赖已删产物。
    rebuildable_only  只能重训得到「同类」结果：脚本触训练或 CUDA，权重被 `.gitignore` 排除、
                      语料不入库也不自动下载 ⇒ 换一台机器只能重训一个同规模同配方的模型，
                      不会得到同样的字节。
    not_reproducible  连「同类结果」都拿不到：缺入口脚本 / 缺落盘数字 / 引用了仓内与外部都拿不到的路径。
                      **每条都必须写明缺什么**（`--check` 拒绝缺 `missing` 的行）。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "atlas" / "repro.md"

BANNER = (
    "<!-- 本文件由 `tools/inventory_runs.py` 从已入库路径（`git ls-files`）生成，禁止手改；"
    "改事实请改实验目录下的文件 -->"
)

#: 判定三态（枚举即契约，别处不得另起名字）。
VERDICTS = ("reproducible", "rebuildable_only", "not_reproducible")
VERDICT_LABELS = {
    "reproducible": "克隆内可重算",
    "rebuildable_only": "只能重训得到同类结果",
    "not_reproducible": "不可重建",
}

#: 一个 run 目录（或一组同名产物）里视为「run 产物」的文件名。
RUN_ATTR_FILES = ("results.json", "train_log.jsonl", "gate_log.jsonl",
                  "train_summary.json", "probe.jsonl")
PREFIX_ATTR_RE = re.compile(
    r"^(?P<run>.+?)-(?P<kind>train_log\.jsonl|gate_log\.jsonl|train_summary\.json|probe\.jsonl)$")
RUN_RESULTS_RE = re.compile(r"^results_(?P<run>.+)\.json$")

#: 入口脚本的优先序：先精确名，再按后缀挑第一个。
ENTRY_NAMES = ("run.py", "run_all.py", "run_all.sh", "supervise.sh", "train_arms.sh",
               "train_all.sh", "run_batch.sh", "eval.sh", "eval_run.sh")

#: 「触训练 / 触语料 / 触 CUDA」的静态证据：命中即 `rebuildable_only`（不能字节级重跑）。
TRAIN_TOKENS = (
    "train_depth_ar", "train_m0", "train_m1", "train_jumphead",
    "pathlm.data", "load_enwik8", "torch.cuda", 'DEVICE = "cuda"', "DEVICE = 'cuda'",
)
TRAIN_TOKEN_RE = re.compile("|".join(re.escape(t) for t in TRAIN_TOKENS))

#: 代码里出现的路径类 token（判定 R3 用）。只认**带文件扩展名**的写法：散文里
#: `harness/methodology`、`tools/evalkit/retention.masked_logits`、`runs/hd64-sN` 这类截断 /
#: 符号引用不算路径，否则会把「转述」当成「悬空依赖」。相对写法（`../<另一实验>/train/…`）
#: 也认：跨实验复用别人的权重正是最常见的不可复现来源。
PATH_TOKEN_RE = re.compile(
    r"(?<![\w/.-])((?:\.\.?/)?(?:axes|\.scratch|tools|configs|pathlm|docs)/[\w./+-]+)")
#: 任何路径里出现 `train/` / `runs/` / `checkpoints/` 这一段（含相对写法与 shell 数组项）：
#: 在本仓布局里这三段就是训练产物=权重所在（`train/<arm>/`、`runs/<arm>-s<seed>/`）。
ARTIFACT_DIR_RE = re.compile(r"(?<![\w/.-])((?:\.\./)?[\w.+-]+/(?:train|runs|checkpoints)/[\w./+-]*)")
FILE_EXTS = (".json", ".jsonl", ".pt", ".npz", ".py", ".sh", ".md", ".txt", ".yaml",
             ".yml", ".log", ".csv", ".bin", ".pkl", ".npy", ".png", ".json.gz")
#: `.gitignore` 排除的输入类前缀：引用它们不构成「悬空」，只意味着「要重训/自取才有」。
IGNORED_PREFIXES = ("runs/", "checkpoints/", "data/", ".tmp/")

#: `results*.json` 里视为「声明墙钟」的键名（任一深度命中，按此优先序取第一个）。
WALL_KEYS = ("wall_minutes", "wall_total_min", "train_wall_minutes", "wall_seconds",
             "wall_minutes_total_eval", "gpu_wall_budget_min", "wall_min", "wall_median_s")

#: 「语料/权重不入库」的机器证据：`.gitignore`
GITIGNORE_ARTIFACTS = (
    ("`*.pt`", "训练权重（每个 run 的 `model.pt`）", "重训，或按 `MODEL.md` 的哈希核对"),
    ("`runs/`", "训练产物目录（`model.pt` / `train_log.jsonl` / `gate_log.jsonl`）",
     "重训；已入库的 `results*.json` 与 `*-train_log.jsonl` 覆盖其判据部分"),
    ("`checkpoints/`", "训练中检点", "重训"),
    ("`data/`", "语料缓存（`enwik8_full.npz`、`m0_subset.npz`）",
     "自取 enwik8 放到 `.tmp/enwik8`，首次运行生成缓存（`README.md` 的数据一节）"),
    ("`.tmp/`", "临时工作区（`enwik8` 原始字节等）", "同上"),
)

#: 语料前置条件（每个重训实验都要付的成本，逐字引 `README.md`）。
CORPUS_NOTE = "语料 enwik8（`data/`、`.tmp/` 均不入库且不自动下载，见 `README.md`）"
WEIGHT_NOTE = "权重不入库（`.gitignore` 的 `*.pt`）"


# --------------------------------------------------------------------------- 读树


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=True).stdout


def tracked_files(root: Path) -> list[str]:
    """已入库路径（仓库相对，排序）。没有 git 的树按 `.gitignore` 的派生物规则现场筛。"""
    if (root / ".git").exists():
        return sorted(p for p in _git(root, "ls-files", "-z").split("\0") if p)
    files: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(seg in ("runs", "checkpoints", "data", ".tmp", "__pycache__", ".git",
                       ".pytest_cache", ".venv") for seg in parts[:-1]):
            continue
        if p.suffix in (".pt", ".npz", ".pyc", ".log", ".pid"):
            continue
        files.append(rel)
    return files


def experiment_dirs(root: Path, tracked: list[str]) -> list[str]:
    """`axes/**/experiments/<exp>`（以入库文件的父目录为准，故空目录不会被多算一个实验）。"""
    dirs: set[str] = set()
    for rel in tracked:
        parts = rel.split("/")
        if parts[0] != "axes" or "experiments" not in parts:
            continue
        i = parts.index("experiments")
        if len(parts) > i + 2:          # `experiments/README.md` 是索引页，不是一个实验
            dirs.add("/".join(parts[: i + 2]))
    return sorted(dirs)


def _read(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


# --------------------------------------------------------------------------- 抽字段


@dataclass
class Run:
    """一个 run = 一组同名的已入库产物（`train/gqa/results.json` 或 `win128-train_log.jsonl`）。"""

    name: str
    artifacts: tuple[str, ...]
    train_steps: int | None = None

    @property
    def has_results(self) -> bool:
        return any(is_results_file(a) for a in self.artifacts)

    @property
    def has_gate_log(self) -> bool:
        return any(a.endswith("gate_log.jsonl") for a in self.artifacts)

    def has_kind(self, suffix: str) -> bool:
        return any(a.endswith(suffix) for a in self.artifacts)


def is_results_file(rel: str) -> bool:
    """`results.json` 或 `results_<arm>_s<seed>.json` 都算「落盘判据」。"""
    name = rel.rsplit("/", 1)[-1]
    return name == "results.json" or bool(RUN_RESULTS_RE.match(name))


@dataclass
class Experiment:
    id: str
    path: str                       # 仓库相对目录
    item: str                       # 所属研究项目录
    item_title: str
    files: tuple[str, ...]
    has_project: bool = False
    runs: list[Run] = field(default_factory=list)
    entry: str | None = None
    command: str = ""
    command_source: str = ""
    wall: str = ""
    verdict: str = ""
    basis: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def has_readme(self) -> bool:
        return "README.md" in self.files

    @property
    def has_analysis(self) -> bool:
        return "analysis.md" in self.files

    @property
    def has_results(self) -> bool:
        return any(f == "results.json" or RUN_RESULTS_RE.match(f) for f in self.files)


def _rel_files(tracked: list[str], exp: str) -> list[str]:
    prefix = exp + "/"
    return [t[len(prefix):] for t in tracked if t.startswith(prefix)]


def _train_steps(root: Path, rel: str) -> int | None:
    """`train_log.jsonl` 的步数：优先读末行的 `step`，读不到就退回行数。"""
    text = _read(root, rel)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
        if isinstance(last, dict) and isinstance(last.get("step"), int):
            return last["step"]
    except (ValueError, TypeError):
        pass
    return len(lines)


def collect_runs(root: Path, exp: str, exp_files: list[str]) -> list[Run]:
    """按「同名前缀」或「同一子目录」把已入库产物归成 run。"""
    groups: dict[str, list[str]] = {}
    for rel in exp_files:
        parent, _, name = rel.rpartition("/")
        if name in RUN_ATTR_FILES:
            groups.setdefault(parent or ".", []).append(rel)
            continue
        if parent:
            continue
        m = PREFIX_ATTR_RE.match(name)
        if m:
            groups.setdefault(m.group("run"), []).append(rel)
            continue
        m = RUN_RESULTS_RE.match(name)
        if m:
            groups.setdefault(m.group("run"), []).append(rel)
    runs = []
    for name, rels in groups.items():
        run = Run(name=name, artifacts=tuple(sorted(rels)))
        log = next((r for r in run.artifacts if r.endswith("train_log.jsonl")), None)
        if log:
            run.train_steps = _train_steps(root, f"{exp}/{log}")
        runs.append(run)
    runs.sort(key=lambda r: (r.name != ".", r.name))
    return runs


def _command(exp_files: list[str], readme: str, entry: str | None) -> tuple[str, str]:
    """重跑命令：先读 README 的「复现 / 重跑」代码块，再退到入口脚本，最后明确写「未声明」。

    代码块里的冒烟命令（`smoke`）只验证机制、不入判据，所以先取**非冒烟**的第一条；
    只有整节都是冒烟时才拿它。
    """
    heading_re = re.compile(r"^#{1,6}\s*(.*)$")
    cmd_re = re.compile(r"^(python3|bash|sh|pytest|\./|tools/|ARMS=|STEPS=)")
    in_repro = False
    in_fence = False
    first = ""
    for line in readme.splitlines():
        stripped = line.strip()
        m = heading_re.match(stripped)
        if m and not in_fence:
            in_repro = bool(re.search(r"复现|重跑|运行方式|用法|reproduce|re-run", m.group(1), re.I))
            continue
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence and in_repro and cmd_re.match(stripped) and not stripped.startswith("#"):
            command = stripped.rstrip("\\").strip()
            if "smoke" not in command:
                return command, "README.md 的复现小节"
            first = first or command
    if first:
        return first, "README.md 的复现小节（只有冒烟命令）"
    if entry:
        runner = "python3" if entry.endswith(".py") else "bash"
        return f"{runner} {entry}", f"入口脚本 `{entry}`，README 未给复现小节"
    return "命令未在仓内声明", ""


def _wall(readme_files: dict[str, str]) -> str:
    """声明的墙钟：先找 `results*.json` 的墙钟键，再找 `analysis.md` 的「墙钟」句。"""
    for rel, text in readme_files.items():
        if not (rel == "results.json" or RUN_RESULTS_RE.match(rel)):
            continue
        try:
            doc = json.loads(text)
        except ValueError:
            continue
        found = _find_wall(doc, "")
        if found:
            return found
    analysis = readme_files.get("analysis.md", "")
    m = re.search(r"墙钟[^\n]{0,60}?(\d[\d.]*)\s*(min|分钟|分|秒|s\b|小时|h\b)", analysis)
    if m:
        return f"`analysis.md`「…{m.group(0)[:40]}…」"
    return ""


def _find_wall(node, prefix: str) -> str:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in WALL_KEYS and isinstance(value, (int, float)):
                return f"`{path}` = {value}"
            hit = _find_wall(value, path)
            if hit:
                return hit
    elif isinstance(node, list):
        for i, value in enumerate(node[:5]):
            hit = _find_wall(value, f"{prefix}[{i}]")
            if hit:
                return hit
    return ""


def _entry(exp_files: list[str]) -> str | None:
    for name in ENTRY_NAMES:
        if name in exp_files:
            return name
    py = [f for f in exp_files if f.endswith(".py") and not f.startswith("test_")
          and "mutate" not in f and not f.startswith("smoke")]
    return py[0] if py else None


def _train_evidence(root: Path, exp: str, exp_files: list[str]) -> tuple[str, str] | None:
    """返回 (文件, 命中 token) 或 None：脚本里是否触训练 / 语料 / CUDA。"""
    for rel in exp_files:
        if not rel.endswith((".py", ".sh")):
            continue
        m = TRAIN_TOKEN_RE.search(_read(root, f"{exp}/{rel}"))
        if m:
            return rel, m.group(0)
    return None


def _code_paths(root: Path, exp: str, exp_files: list[str], tree: "Tree") -> tuple[list[str], list[str]]:
    """代码（`.py` / `.sh`）里引用的路径 token，分成 (仓内悬空, 要自产/自取)。

    只扫代码、不扫 Markdown：散文里的 `harness/methodology`、`retention.masked_logits`、
    `runs/hd64-sN` 是转述或占位，不是依赖。分类：
      * `.gitignore` 排除的输入前缀（`runs/` `data/` …）、任何 `.pt` 权重、**本机存在但未入库**的
        路径 ⇒ 「要自产/自取」（可重训）；
      * 其余（应在库却不在库也不在本机）⇒ 「仓内悬空」（不可重建）。
    """
    dangling: list[str] = []
    needs: list[str] = []
    for rel in exp_files:
        if not rel.endswith((".py", ".sh")):
            continue
        text = _read(root, f"{exp}/{rel}")
        for m in ARTIFACT_DIR_RE.finditer(text):
            token = m.group(1).rstrip("/")
            if "*" not in token and token not in needs:
                needs.append(token)
        for m in PATH_TOKEN_RE.finditer(text):
            token = m.group(1).rstrip("/")
            if "*" in token or not token.endswith(FILE_EXTS):
                continue
            norm = (Path(exp) / token).as_posix() if token.startswith(("./", "../")) else token
            norm = _normpath(norm)
            if token.endswith(".pt") or norm.startswith(IGNORED_PREFIXES):
                if token not in needs:
                    needs.append(token)
                continue
            if tree.has(norm):
                continue
            if (root / norm).exists() and token not in needs:
                needs.append(token)          # 本机有、但不在库：别人拿不到
                continue
            if token not in dangling:
                dangling.append(token)
    return dangling, needs


def _normpath(path: str) -> str:
    """去掉 `./` 与 `../` 的纯文本归一（不碰磁盘）。"""
    parts: list[str] = []
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


class Tree:
    """已入库路径集合 + 它们的所有祖先目录（判「这条路径在不在库」）。"""

    def __init__(self, tracked: list[str]):
        self.files = set(tracked)
        self.dirs: set[str] = set()
        for rel in tracked:
            parts = rel.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                self.dirs.add("/".join(parts[:i]))

    def has(self, rel: str) -> bool:
        return rel in self.files or rel in self.dirs


def _verdict(root: Path, tree: "Tree", exp: Experiment) -> None:
    """三态判定。规则按顺序判，命中即停；每条判定都把依据写进 `basis`。"""
    ev = _train_evidence(root, exp.path, list(exp.files))
    if exp.entry is None:
        exp.verdict = "not_reproducible"
        exp.basis.append(f"仓内无入口脚本（只有 {', '.join(sorted(exp.files)) or '空目录'}）")
        exp.missing.append("入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑")
        return
    if not exp.has_results:
        exp.verdict = "not_reproducible"
        exp.basis.append(f"有入口脚本 `{exp.entry}`，但仓内无 `results*.json`")
        exp.missing.append("落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准")
        return
    dangling, needs = _code_paths(root, exp.path, list(exp.files), tree)
    if dangling:
        exp.verdict = "not_reproducible"
        exp.basis.append(f"代码引用的仓内路径不存在：{', '.join(dangling[:3])}")
        exp.missing.append("输入产物：" + "、".join(f"`{d}`" for d in dangling[:5]))
        return
    if ev is not None or needs:
        exp.verdict = "rebuildable_only"
        if ev is not None:
            rel, token = ev
            exp.basis.append(f"`{rel}` 触训练 / 语料 / CUDA（命中 `{token}`）")
        if needs:
            exp.basis.append("代码引用不入库的产物：" + "、".join(f"`{n}`" for n in needs[:3])
                             + "（克隆里不存在，需先重训/自取）")
        return
    exp.verdict = "reproducible"
    exp.basis.append(f"`{exp.entry}` 不触训练数据 / 引擎 / CUDA，也不引用未入库产物")
    exp.basis.append(f"数字来自仓内 `results*.json`，输入全部入库（{len(exp.files)} 个文件）")


def collect(root: Path = ROOT) -> list[Experiment]:
    tracked = tracked_files(root)
    tree = Tree(tracked)
    exps: list[Experiment] = []
    for path in experiment_dirs(root, tracked):
        files = tuple(sorted(_rel_files(tracked, path)))
        item = path.rsplit("/experiments/", 1)[0]
        item_title = _item_title(root, item)
        exp = Experiment(id=path.rsplit("/", 1)[-1], path=path, item=item, item_title=item_title,
                         files=files, has_project=(root / item / "project.yaml").is_file())
        exp.runs = collect_runs(root, path, list(files))
        texts = {rel: _read(root, f"{path}/{rel}") for rel in files
                 if rel in ("README.md", "analysis.md", "results.json") or RUN_RESULTS_RE.match(rel)}
        exp.entry = _entry(list(files))
        exp.command, exp.command_source = _command(list(files), texts.get("README.md", ""), exp.entry)
        exp.wall = _wall(texts)
        _verdict(root, tree, exp)
        exps.append(exp)
    return exps


def _item_title(root: Path, item: str) -> str:
    p = root / item / "project.yaml"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^title:\s*(.+)$", text, re.M)
    return m.group(1).strip().strip('"\'') if m else ""


# --------------------------------------------------------------------------- 渲染


def _cell(text: object, width: int | None = None) -> str:
    s = " ".join(str(text).split()) or "—"
    s = s.replace("|", "\\|")
    if width and len(s) > width:
        s = s[: width - 1] + "…"
    return s


def _flag(value: bool) -> str:
    return "✓" if value else "—"


INTRO = [
    "# 复现清单：哪些 run 的权重与日志还在、能复跑到什么程度",
    "",
    "> 本文件由 `tools/inventory_runs.py` 生成，禁止手改；`--check` 与手改过的副本逐字节比对。"
    "它回答一个问题：**这个数字，别人拿这个克隆能重跑出来吗**。",
    "> 数据源 = `git ls-files`：**表里只出现已入库的路径**，所以每一条都在任何克隆里存在。"
    "`.gitignore` 排除的本机产物（`*.pt` 权重、`runs/`、`checkpoints/`、`data/`、`.tmp/`）"
    "不进表，它们是本机的、不是这个仓库的承诺，见「这个克隆重建不了什么」。",
    "> 「权重在库」列因此恒为「否」：已入库的 `.pt` 有 **{WEIGHTS_IN_REPO}** 个（`.gitignore` 排除了 `*.pt`）。"
    "要量本机权重（字节数 + sha256），"
    "跑 `python3 tools/inventory_runs.py --local`——那是本机视图，不是本文件的事实。",
    "> 判定是**静态证据**：路径存在性 + 脚本是否触训练 / 语料 / CUDA + 数字是否落盘。"
    "脚本没有被逐个执行 ⇒ 「克隆内可重算」= 结构上具备重算条件，**不**等于「已跑通」。"
    "要更强的保证只能逐个跑，那不属于本工作包。",
    "",
    "## 三态定义与判定规则（逐条，判定必须落在其中一条上）",
    "",
    "| 判定 | 含义 | 判据（按顺序判，命中即停） |",
    "|---|---|---|",
    "| `reproducible`（克隆内可重算） | 数字能在克隆内重算出来，不需要外部语料、GPU 训练或已删产物 | "
    "有入口脚本、有 `results*.json`、无悬空引用、脚本**不触**训练 / 语料 / CUDA，"
    "也**不引用**不入库的产物 |",
    "| `rebuildable_only`（只能重训得到同类结果） | 换一台机器只能重训一个同规模同配方的模型，"
    "不会得到同样的字节 | 同上，但脚本**触**训练 / 语料 / CUDA，或引用了不入库的产物"
    "（`*.pt`、`train/` · `runs/` 下的权重与日志）⇒ 重跑是重造而不是重算 |",
    "| `not_reproducible`（不可重建） | 连「同类结果」都拿不到 | 缺入口脚本 / 缺 `results*.json` / "
    "引用了既不在库也不在本机的路径——**每条都必须写明缺什么** |",
    "",
    "「声明墙钟」逐字取自该实验的 `results*.json`（墙钟键）或 `analysis.md` 的墙钟句，本文件不重算、不换算口径。",
    "「重跑命令」只抄实验自己 `README.md` 复现小节里的第一条非冒烟命令，或退到它的入口脚本；"
    "仓内确定不了就写「命令未在仓内声明」，不替它编一条。",
    f"`rebuildable_only` 的**共因**（每行都适用，故不逐行重复）：{WEIGHT_NOTE}；{CORPUS_NOTE}。",
    "",
]


def render(exps: list[Experiment], local: dict[str, list[str]] | None = None,
           tracked: list[str] | None = None) -> str:
    counts = {v: sum(1 for e in exps if e.verdict == v) for v in VERDICTS}
    runs_total = sum(len(e.runs) for e in exps)
    if tracked is None:
        tracked = tracked_files(ROOT)
    weights_in_repo = [t for t in tracked if t.endswith(".pt")]
    intro = [line.replace("{WEIGHTS_IN_REPO}", str(len(weights_in_repo))) for line in INTRO]
    body: list[str] = [BANNER, ""] + intro

    body.append("## 汇总（生成时统计）")
    body.append("")
    aggregates = sum(1 for e in exps for r in e.runs if r.name == ".")
    body.append(f"- 规模：实验 **{len(exps)}** 个（`axes/**/experiments/*`），run 条目 **{runs_total}** 个"
                f"（{runs_total - aggregates} 个是 run 子目录或前缀组，{aggregates} 个是**实验级汇总（不是 run 目录）**，"
                f"即实验自己那一行「.」；有 run 产物的实验 {sum(1 for e in exps if e.runs)} 个）。")
    body.append("")
    body.append("- 三态分布（按**实验**计数，与上一段的 run 条目是两种计数）：" + "、".join(
        f"`{v}`（{VERDICT_LABELS[v]}）**{counts[v]}** 个" for v in VERDICTS) + "。")
    body.append(f"- 入库权重：**{len(weights_in_repo)}** 个 `.pt`（`*.pt` 被 `.gitignore` 排除）；"
                f"有落盘 `results*.json` 的实验 {sum(1 for e in exps if e.has_results)} 个。")
    body.append("")

    body.append("## 这个克隆重建不了什么")
    body.append("")
    body.append("下面这些路径被 `.gitignore` 排除，**不在任何克隆里**；它们是派生物，重建方式写在第三列：")
    body.append("")
    body.append("| `.gitignore` 的路径 | 它是什么派生物 | 怎么重建 |")
    body.append("|---|---|---|")
    for pattern, what, how in GITIGNORE_ARTIFACTS:
        body.append(f"| {pattern} | {_cell(what)} | {_cell(how)} |")
    body.append("")
    body.append("- 数据下载：本仓**不下载任何数据集**。`README.md` 的数据一节要求自取 enwik8 "
                "放到 `.tmp/enwik8`，首次运行时生成 `data/enwik8_full.npz`。")
    body.append("- 权重：每个 run 的 `model.pt` 都不入库；`MODEL.md`（有该文件的实验）只登记哈希清单，"
                "不是权重本身。")
    body.append("- GPU 产物：`runs/**`、`checkpoints/**` 里的训练日志、探针 jsonl、检点都在被忽略的目录里；"
                "判据用到的部分已另行入库（见下表的 run 明细）。")
    body.append("")

    body.append("## 总表：每个实验")
    body.append("")
    body.append("| 实验 | 研究项 | README | analysis | results | run | 权重在库 | 重跑命令 | 声明墙钟 | 判定 | 依据 |")
    body.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in exps:
        item = (f"[{e.item.split('/')[-1]}](../{e.item}/project.yaml)" if e.has_project
                else e.item.split("/")[-1])
        cells = (
            f"[{e.id}](../{e.path}/)" if e.has_readme else e.id,
            item,
            _flag(e.has_readme), _flag(e.has_analysis), _flag(e.has_results),
            str(len(e.runs)),
            "否（`*.pt` 不入库）" if e.has_results else "—",
            f"`{e.command}`" + (f"（{e.command_source}）" if e.command_source and
                               e.command != "命令未在仓内声明" else ""),
            e.wall or "—",
            f"`{e.verdict}`",
            "；".join(e.basis),
        )
        body.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    body.append("")

    body.append("## 不可重建明细（`not_reproducible`：缺什么）")
    body.append("")
    bad = [e for e in exps if e.verdict == "not_reproducible"]
    if bad:
        body.append("| 实验 | 缺什么 | 依据 |")
        body.append("|---|---|---|")
        for e in bad:
            body.append(f"| [{e.id}](../{e.path}/) | {_cell('；'.join(e.missing))} | "
                        f"{_cell('；'.join(e.basis))} |")
    else:
        body.append("（无：每个实验都至少能重训出同类结果。）")
    body.append("")

    body.append("## run 明细：每个 run 的已入库产物")
    body.append("")
    body.append("一个 run = 一组同名的**已入库**产物：某个子目录里的判据 json，或顶层的一组同名日志。"
                "`model.pt` 一律不入库，故不设该列。各列口径：")
    body.append("")
    body.append("```")
    body.append("results        判据 json 已入库（该 run 的落盘数字）")
    body.append("gate_log       等值门日志已入库（逐位一致那类判据）")
    body.append("训练步数       train_log.jsonl 末行的 step（没有该日志则为 —）")
    body.append("train_summary  训练汇总已入库（步时、显存峰值）")
    body.append("probe          探针 jsonl 已入库（活性 / 生成质量类判据）")
    body.append("```")
    body.append("")
    body.append("| 实验 | run | results | gate_log | 训练步数 | train_summary | probe |")
    body.append("|---|---|---|---|---|---|---|")
    for e in exps:
        if not e.runs:
            body.append(f"| [{e.id}](../{e.path}/) | — | — | — | — | — | 仓内没有 run 产物 |")
            continue
        for run in e.runs:
            body.append(
                f"| {_cell(e.id)} | {_cell(run.name)} | {_flag(run.has_results)} | "
                f"{_flag(run.has_gate_log)} | "
                f"{run.train_steps if run.train_steps is not None else '—'} | "
                f"{_flag(run.has_kind('train_summary.json'))} | {_flag(run.has_kind('probe.jsonl'))} |")
    body.append("")

    if local is not None:
        body.append("## 本机视图（**不是**本文件的事实；由 `--local` 现量）")
        body.append("")
        body.append("| 实验 | 本机权重文件 | 字节数 | sha256 |")
        body.append("|---|---|---|---|")
        for exp_id in sorted(local):
            for path, size, digest in local[exp_id]:
                body.append(f"| {_cell(exp_id)} | `{path}` | {size} | `{digest}` |")
        body.append("")
    return "\n".join(body)


# --------------------------------------------------------------------------- 校验


def validate(exps: list[Experiment], root: Path = ROOT) -> list[str]:
    """`--check` 的判据：三态合法、依据在位、不可重建必写缺什么、每条路径真存在。"""
    problems: list[str] = []
    if not exps:
        return ["axes/**/experiments 下没有扫到任何实验（生成物与磁盘不一致，或扫描根选错了）"]
    for e in exps:
        where = f"{e.path}"
        if e.verdict not in VERDICTS:
            problems.append(f"{where}: 判定 `{e.verdict}` 不在三态枚举里")
        if not e.basis:
            problems.append(f"{where}: 判定 `{e.verdict}` 没有给出依据")
        if e.verdict == "not_reproducible" and not [m for m in e.missing if m.strip()]:
            problems.append(f"{where}: 判定 `not_reproducible` 但没写明缺什么")
        if not (root / e.path).is_dir():
            problems.append(f"{where}: 实验目录不存在")
        for rel in e.files:
            if not (root / e.path / rel).is_file():
                problems.append(f"{where}: 清单里的路径不存在 —— {e.path}/{rel}")
    return problems


def local_view(root: Path) -> dict[str, list[tuple[str, int, str]]]:
    """本机存在但未入库的权重：路径、字节数、sha256（`--local` 用，不进生成物）。"""
    out: dict[str, list[tuple[str, int, str]]] = {}
    tracked = tracked_files(root)
    for path in experiment_dirs(root, tracked):
        tracked_rel = set(_rel_files(tracked, path))
        hits: list[tuple[str, int, str]] = []
        for p in sorted((root / path).rglob("*.pt")):
            rel = p.relative_to(root / path).as_posix()
            if rel in tracked_rel:
                continue
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            hits.append((rel, p.stat().st_size, digest))
        if hits:
            out[path.rsplit("/", 1)[-1]] = hits
    return out


def as_json(exps: list[Experiment]) -> dict:
    return {
        "experiments": [
            {
                "id": e.id,
                "path": e.path,
                "item": e.item,
                "docs": {"readme": e.has_readme, "analysis": e.has_analysis,
                         "results": e.has_results},
                "runs": [
                    {"name": r.name, "artifacts": list(r.artifacts),
                     "train_steps": r.train_steps, "gate_log": r.has_gate_log,
                     "results": r.has_results, "weights_in_repo": False}
                    for r in e.runs
                ],
                "command": e.command,
                "command_source": e.command_source,
                "wall_clock": e.wall or None,
                "verdict": e.verdict,
                "basis": e.basis,
                "missing": e.missing,
            }
            for e in exps
        ],
        "counts": {
            "experiments": len(exps),
            "runs": sum(len(e.runs) for e in exps),
            **{v: sum(1 for e in exps if e.verdict == v) for v in VERDICTS},
        },
    }


def _disp(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str], root: Path | None = None) -> int:
    root = Path(root) if root else ROOT
    out = OUT if root == ROOT else root / "atlas" / "repro.md"
    check = "--check" in argv
    want_json = "--json" in argv
    want_local = "--local" in argv
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1])
        out = root / "atlas" / "repro.md"

    exps = collect(root)
    tracked = tracked_files(root)
    problems = validate(exps, root)
    local = local_view(root) if want_local else None

    if want_json:
        payload = as_json(exps)
        if local is not None:
            payload["local_weights"] = {
                exp: [{"path": p, "bytes": s, "sha256": h} for p, s, h in rows]
                for exp, rows in local.items()}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if problems else 0

    if local is not None:
        print(render(exps, local, tracked), end="")
        return 1 if problems else 0

    import difflib
    text = render(exps, tracked=tracked)
    if check:
        if problems:
            print(f"FAIL：复现清单有 {len(problems)} 项问题", file=sys.stderr)
            for p in problems[:20]:
                print("  -", p, file=sys.stderr)
            return 1
        disk = out.read_text(encoding="utf-8") if out.exists() else None
        if disk is None:
            print(f"FAIL：{_disp(out)} 不存在；先运行 tools/inventory_runs.py", file=sys.stderr)
            return 1
        if disk != text:
            diff = list(difflib.unified_diff(disk.splitlines(), text.splitlines(),
                                             fromfile=f"{out.name}（磁盘）",
                                             tofile=f"{out.name}（按已入库路径重算）",
                                             lineterm="", n=1))
            print(f"FAIL：{_disp(out)} 与磁盘上的实验目录不一致"
                  f"（差异 {len(diff)} 行，前 20 行如下）：", file=sys.stderr)
            for line in diff[:20]:
                print("  " + line, file=sys.stderr)
            return 1
        counts = {v: sum(1 for e in exps if e.verdict == v) for v in VERDICTS}
        print(f"OK：{_disp(out)} 与磁盘一致 —— {len(exps)} 个实验 / "
              f"{sum(len(e.runs) for e in exps)} 个 run；"
              + "、".join(f"{v} {counts[v]}" for v in VERDICTS) + "。")
        return 0

    if problems:
        print(f"FAIL：复现清单有 {len(problems)} 项问题，未生成 {_disp(out)}", file=sys.stderr)
        for p in problems[:20]:
            print("  -", p, file=sys.stderr)
        return 1
    out.write_text(text, encoding="utf-8")
    counts = {v: sum(1 for e in exps if e.verdict == v) for v in VERDICTS}
    print(f"已生成 {_disp(out)}：{len(exps)} 个实验 / {sum(len(e.runs) for e in exps)} 个 run；"
          + "、".join(f"{v} {counts[v]}" for v in VERDICTS) + "。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
