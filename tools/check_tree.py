#!/usr/bin/env python3
"""check_tree.py — 研究树（`axes/**/project.yaml`）的格式与证据校验（可失败）。

新结构下 `axes/**/project.yaml` 是研究项的 SSOT（契约见 `harness/research-schema.md`，
布局见 `docs/structure.md` §12）。旧生成式 wiki（catalog / build_wiki / INDEX）已退役，
`tools/check_schema.py` 转调本模块。

校验项：
  1. 字段齐全：`harness/research-schema.md` §2 列出的字段一个都不能少。
  2. 关键字段非空：`purpose` / `innovation` / `validation` / `objective_criteria` /
     `falsification` 不得为空串/空列表。
  3. `milestone` 单数：必须是 map 且有 `output` + `acceptance`（不是 list）。
  4. `role_map` 四角色齐：owner / writer / reviewer / integrator。
  5. `work_packages`：每条含 id/what/goal/means/acceptance/evidence/review/status。
  6. `status` 合法：未开始 / 进行中 / 已完成 / 被否证。
  7. `cost_model` 禁墙钟/日期（只许等效理论量：FLOPs / layer-passes / tokens-per-forward）。
  8. `status ∈ {已完成, 被否证}` ⇒ 必须有 `conclusion` + 非空且存在的 `experiments`；
     `status = 被否证` ⇒ 还必须有非空 `failure_analysis`（失败归因：设计 / 结构 / innovation，
     见 `harness/research-schema.md` §7）。
  9. `validation` 必须含「对照/基线/stock/SOTA」之一 + 「判据/该涨/该跌」之一 +
     「口径/bpc/...」之一。
 10. `id` 全局唯一；`dir` 必须等于文件实际所在目录。
 11. 证据锚点：`experiments` 路径存在；WP `evidence` 允许未来路径（只校锚点 `@<sha>` 可达）。
 12. `laws` / `cost_rows`（`harness/research-schema.md` §2.1 / §2.2）：定律编号限 1–11 且不重复，
     `cost_rows[].law` 必须是本项 `laws` 的成员，`touches_trunk` 必须是 bool，`delta` 必须是数字，
     `source` 指向的路径必须存在。
 13. 文档引用可解析：转调 `tools/check_docs.py`（文档里的裸路径必须存在，`::键` / `#标题` 锚点
     必须唯一命中，指向 `.yaml` / `.json` 的行号引用是禁止项；该文件的扫描范围与「不判」清单写在
     它自己的文件头）。一条命令覆盖树校验 + 引用校验。
 14. 预登记定序：转调 `tools/prereg_bind.py` 的 `check()`——有 `results.json` 的实验必须有
     `prereg_binding.json`；契约文件当前 sha256 必须与绑定时的相同；`bound_at` 必须严格早于读数
     的产物时间；两文件都已入库时，绑定的提交必须早于读数的提交；`bound_at` 必须落在绑定文件那次
     提交的 ±`BIND_WINDOW_MINUTES` 分钟窗内。`bound_at = "unbound"`（历史产物，无定序证据）只计数、
     不判失败，但实验名打进汇总行（见 `harness/methodology.md` §8）。

用法：python3 tools/check_tree.py   （退出码 0 = 全过；1 = 有错）
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_docs  # noqa: E402  —— 文档引用校验只此一份实现
import prereg_bind  # noqa: E402  —— 预登记绑定校验只此一份实现

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"

VALID_STATUS = ("未开始", "进行中", "已完成", "被否证")
DONE_STATUS = ("已完成", "被否证")

REQUIRED = (
    "id", "axis", "title", "dir", "purpose", "innovation", "expected_effect",
    "sota_baseline", "gap", "method", "validation", "objective_criteria",
    "falsification", "cost_model", "role_map", "milestone", "work_packages",
    "roadmap", "status", "failure_analysis", "conclusion_review", "laws",
    "cost_rows", "experiments", "argument", "gaps", "started", "ended",
)
#: 变更记录形态的键（`change_log` 与其条目里的 `actor`）：`project.yaml` 是事实文档，只写当前状态；
#: 过程史的家是 `.scratch/doc-log/**`（`AGENTS.md` §2.1 / §2.2）。出现即 fail。
RETIRED_KEYS = ("change_log", "actor")
#: 内部工单号 / 复核编号的形态：`ticket 62`、`55-review P10`、`WP-2652`。
#: `-review` 只认**两位数字**前缀，否则角色名 `axis1-reviewer` 会被误抓。
PROCESS_REF_RE = re.compile(r"ticket\s*\d+|\d{2}-review|WP-[A-Za-z0-9]+")
#: 过程史的家（错误信息里指过去的那一条路径）。
DOC_LOG = ".scratch/doc-log/axes/<轴>/<格子>/CHANGELOG.md"
#: 「必填」= 必须**非空**的字符串字段（不允许空串/空列表蒙混，见 harness/research-schema.md §2）。
NONEMPTY = ("purpose", "innovation", "validation", "objective_criteria", "falsification")
WP_REQUIRED = ("id", "what", "goal", "means", "acceptance", "evidence", "review", "status")
ROLE_REQUIRED = ("owner", "writer", "reviewer", "integrator")

#: cost_model 里**禁止**出现的墙钟/日期指标（只许等效理论量）。
WALLCLOCK_RE = re.compile(
    r"墙钟|wall[-_ ]?(?:clock|time)|tok(?:en)?s?/s|tokens? per second|"
    r"毫秒|秒|分钟|小时|latency|\bms/step\b|elapsed|\d{4}-\d{2}-\d{2}"
)

#: 合法提交前缀锚点：`@<7–40 位十六进制>`。
ANCHOR_RE = re.compile(r"@([0-9a-f]{7,40})")
#: evidence 里任何 `@` 之后到空白为止的片段（抓 `@FINAL` 这类非法锚点）。
AT_TOKEN_RE = re.compile(r"@(\S*)")

#: `laws` 的编号域 = `atlas/playbook.md` 的「定律 1..11」（见 `harness/research-schema.md` §2.1）。
LAWS_TOTAL = 11
#: `cost_rows` 一条必须齐备的键（见 `harness/research-schema.md` §2.2）。
COST_ROW_FIELDS = ("law", "change", "touches_trunk", "metric", "delta",
                   "comparator", "scale", "boundary", "source")
#: `delta` 必须是带符号数字（含千分位/小数；单位与方向由 `delta_direction` + `metric` 共同声明）。
NUMBER_RE = re.compile(r"[+-]?\d+(?:[\d,]*)?(?:\.\d+)?")

#: Δ 方向的**唯一枚举**（`harness/research-schema.md` §2.3）：名字即算式，读表的人不可能读反。
#: `undeclared` 是**声明「缺失」本身**——它不进自洽校验，但必须被统计打印，缺口不许静默通过。
DELTA_DIRECTIONS = ("comparator_minus_arm", "arm_minus_comparator", "undeclared")
#: `cost_rows` 是一条必填事实，不存在「没方向」这种状态 ⇒ 那里只许两个真方向。
COST_ROW_DIRECTIONS = ("comparator_minus_arm", "arm_minus_comparator")
#: 枚举 → 展示层中文短语（`atlas/cost-table.md` 的「Δ 方向」列用这一份，不另抄）。
DELTA_DIRECTION_LABELS = {
    "comparator_minus_arm": "正 = 改善",
    "arm_minus_comparator": "正 = 更差",
}
#: `results.json` 的方向声明：**字段级映射**（一个文件的不同 Δ 字段可以有不同方向）。
DELTA_DIRECTIONS_KEY = "delta_directions"
#: 出现 `undeclared` 时必须写下的一句话原因（可选键，别处不写）。
DELTA_DIRECTION_NOTE_KEY = "delta_direction_note"
#: 旧的**文件级单值**字段：新形状下必须消失，留着即 fail（防止两种形状并存）。
LEGACY_DELTA_DIRECTION_KEY = "delta_direction"
#: 这些键是声明本身，不是 Δ 读数，不能被当成待声明字段。
DELTA_META_KEYS = (DELTA_DIRECTIONS_KEY, DELTA_DIRECTION_NOTE_KEY,
                   LEGACY_DELTA_DIRECTION_KEY)
#: 方向短语里的「更好 / 更差」词表（`metric` 与 `results.json` 的判据串共用）。
_BETTER_WORDS = ("更好", "改善", "变好", "更可读", "更优", "有收益")
_WORSE_WORDS = ("更差", "变差", "损伤", "税", "失效", "更费", "更慢", "恶化",
                "负迁移", "更早报废", "吃亏")
#: `正 = …` / `负 = …` / `>1 = …` 三种方向短语（全角等号也收）。
_POS_CLAIM_RE = re.compile(r"正\s*[=＝]\s*([^；;，,）)|]{1,20})")
_NEG_CLAIM_RE = re.compile(r"负\s*[=＝]\s*([^；;，,）)|]{1,20})")
_RATIO_CLAIM_RE = re.compile(r">\s*1\s*[=＝]\s*([^；;，,）)|]{1,20})")
#: `Δ = A − B` 形式的算式声明（`−` / `-` / `–` 都认）。
_FORMULA_RE = re.compile(
    r"Δ\s*[（(]?\s*(?:[=＝]\s*)?([^；;，,（）()]{1,40}?)\s*[−–-]\s*([^；;，,（）()]{1,40})")
#: 算式里代表「对照」与「臂」的标记词（只认明确的角色词，含糊的一律不判）。
_COMPARATOR_MARKS = ("对照", "基线", "stock", "B0", "b0", "全因果", "fp16", "sequential",
                     "base", "同形态基线", "窗口", "纯窗")
_ARM_MARKS = ("实验", "本臂", "该臂", "臂", "改动", "窄", "量化")
#: `results.json` 里一个 Δ 读数 = 键名含 `delta` / `Δ` 且**值里有数字**的节点（标量或它的整棵子树）。
#: 只看键名、不看后代路径：`delta_spread: {min, max}` 是**一个**读数（整棵子树的统计），不是十几个。
DELTA_NODE_RE = re.compile(r"(?i)delta|Δ")
#: 「本行报的是绝对值/比值，符号不可判」的标记（只校验枚举合法，不校验符号自洽）。
_UNSIGNED_METRIC_MARKS = ("非负", "|Δ|", "绝对值")
#: `source` 可带的定位后缀：`:12` / `:12-14` / `#§标题`。
SOURCE_SUFFIX_RE = re.compile(r"(?::\d+(?:-\d+)?)?(?:#§.*)?$")


class DuplicateKeyLoader(yaml.SafeLoader):
    """让重复键直接报错（默认 SafeLoader 静默取后一个值）。"""


def _no_duplicates(loader: DuplicateKeyLoader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"重复键 {key!r}", key_node.start_mark)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


DuplicateKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates)


class GitUnavailable(RuntimeError):
    """git 不可用（未安装 / 不是仓库）——锚点校验必须显式报错，不得静默跳过。"""


def reachable_commits() -> set[str]:
    """一次取回所有 ref 可达的完整提交 hash（批处理）。"""
    r = subprocess.run(["git", "rev-list", "--all"], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise GitUnavailable((r.stderr or r.stdout or "git rev-list --all 失败").strip())
    return set(r.stdout.split())


def check_anchors(anchors: dict[str, list[str]]) -> list[str]:
    """校验所有 `@<sha>` 锚点都指向真实提交。git 不可用 ⇒ 显式报错。"""
    if not anchors:
        return []
    try:
        reach = reachable_commits()
    except GitUnavailable as e:
        return [f"git 不可用，无法校验证据锚点（不得静默跳过）：{e}"]
    bad = sorted(a for a in anchors if not any(s.startswith(a) for s in reach))
    if not bad:
        return []
    detail: dict[str, str] = {}
    r = subprocess.run(["git", "cat-file", "--batch-check"], cwd=ROOT,
                       input="\n".join(bad) + "\n", capture_output=True, text=True)
    lines = r.stdout.splitlines()
    if r.returncode == 0 and len(lines) == len(bad):
        for a, line in zip(bad, lines):
            parts = line.split()
            detail[a] = "missing" if len(parts) >= 2 and parts[1] == "missing" else "unreachable"
    kind = {"missing": "对象不存在", "unreachable": "对象存在但不在任何 ref 的历史中"}
    return [f"证据锚点无效：@{a}（{kind.get(detail.get(a), '不是可达的真实提交')}；出现于 "
            f"{'、'.join(anchors[a])}）" for a in bad]


def check_anchor_syntax(evidences: dict[str, list[str]]) -> list[str]:
    """凡 evidence 里出现 `@` 且其后不是合法提交前缀 ⇒ 逐条报错。"""
    errors = []
    for ev, locs in evidences.items():
        for m in AT_TOKEN_RE.finditer(ev):
            if not ANCHOR_RE.fullmatch(m.group(0)):
                errors.append(
                    f"证据锚点非法：`{m.group(0)}` 不是合法提交前缀"
                    f"（须形如 @<7–40 位十六进制 sha>；出现于 {'、'.join(locs)}）")
    return errors


def source_path(src) -> str:
    """`source` 去掉 `:行` / `#§标题` 定位后缀后的仓库内路径。"""
    s = str(src).strip()
    return SOURCE_SUFFIX_RE.sub("", s).strip()


def _claim_direction(text: str) -> tuple[str | None, str]:
    """从一句方向短语（`正 = X` / `负 = X` / `>1 = X`）判方向。返回 (方向 | None, 原词)。

    None = 这一句不能判（既不是更好词也不是更差词）。
    """
    t = " ".join(str(text).split())
    better = any(w in t for w in _BETTER_WORDS)
    worse = any(w in t for w in _WORSE_WORDS)
    if better == worse:
        return None, t  # 都没命中，或同时命中（歧义）⇒ 不判
    if better:
        return "comparator_minus_arm", t
    return "arm_minus_comparator", t


def direction_claims(text: str, ratios: bool = True) -> list[tuple[str, str]]:
    """把一段文字里所有的方向声明抽成 [(方向, 原词)]；`负 = 更好` 一类的反向说法已翻转。

    `ratios=False` 时丢掉 `>1 = …` 这一类：它描述的是一个比值字段的符号，
    不是 Δ 字段的朝向，拿它去判同块里的 Δ 字段会把方向判反。
    """
    out: list[tuple[str, str]] = []
    s = " ".join(str(text).split())
    for m in _POS_CLAIM_RE.finditer(s):
        d, _ = _claim_direction(m.group(1))
        if d:
            out.append((d, m.group(0)))
    for m in _NEG_CLAIM_RE.finditer(s):
        d, _ = _claim_direction(m.group(1))
        if d:  # 「负 = 更好」⇒ 正 = 更差 ⇒ arm − comparator
            out.append(("arm_minus_comparator" if d == "comparator_minus_arm"
                        else "comparator_minus_arm", m.group(0)))
    if ratios:
        for m in _RATIO_CLAIM_RE.finditer(s):
            d, _ = _claim_direction(m.group(1))
            if d:
                out.append((d, m.group(0)))
    return out


def formula_direction(text: str) -> tuple[str | None, str]:
    """从 `Δ = A − B` 判算式次序；只认角色明确的写法，含糊则返回 (None, 原句)。"""
    s = " ".join(str(text).split())
    for m in _FORMULA_RE.finditer(s):
        a, b = m.group(1).strip(), m.group(2).strip()

        def role(tok: str) -> str | None:
            is_c = any(w in tok for w in _COMPARATOR_MARKS)
            is_a = any(w in tok for w in _ARM_MARKS)
            if is_c == is_a:  # 都不是 / 两者都是 ⇒ 角色不明
                return None
            return "comparator" if is_c else "arm"

        ra, rb = role(a), role(b)
        if ra == "comparator" and rb == "arm":
            return "comparator_minus_arm", m.group(0)
        if ra == "arm" and rb == "comparator":
            return "arm_minus_comparator", m.group(0)
    return None, s


#: 「机械判不了」的行（`metric` 里没有方向短语）——由 `main()` 打到 stderr，不算 fail。
UNJUDGED: list[str] = []


def check_cost_row_direction(row: dict, where: str) -> list[str]:
    """`cost_rows[].delta_direction` 必填 + 枚举合法 + 与 `metric` 的方向/算式自洽。

    能机械判的：枚举值、`metric` 的 `正/负/>1 = …` 短语、`Δ = A − B` 算式次序。
    判不了的：`metric` 自己写成「非负 / |Δ| / 绝对值」（行本身就不带符号）——只校枚举。
    """
    errors: list[str] = []
    declared = row.get("delta_direction")
    if declared is None:
        return [f"{where} 缺 delta_direction（合法值：{COST_ROW_DIRECTIONS}）"]
    if declared not in COST_ROW_DIRECTIONS:
        return [f"{where}.delta_direction={declared!r} 非法（合法值：{COST_ROW_DIRECTIONS}）"]
    metric = str(row.get("metric") or "")
    if any(w in metric for w in _UNSIGNED_METRIC_MARKS):
        return errors  # 本行报的是绝对值/记号本身非负 ⇒ 符号不可机判
    claims = direction_claims(metric)
    if not claims:
        UNJUDGED.append(f"{where}: metric 里没有「正 = …」「负 = …」「>1 = …」方向短语，"
                        f"符号无法机判（delta_direction 仍必填）：{metric!r}")
    for d, raw in claims:
        if d != declared:
            errors.append(f"{where}: metric 的方向声明「{raw}」与 delta_direction="
                          f"{declared!r} 矛盾（按该声明应为 {d!r}）")
    fd, raw = formula_direction(metric)
    if fd and fd != declared:
        errors.append(f"{where}: metric 的算式「{raw}」算出来是 {fd!r}，"
                      f"与 delta_direction={declared!r} 矛盾（数字没变，只是标签对不上）")
    elif fd and not claims and not errors:
        pass
    return errors


def _walk_strings(obj, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """递归抽出 JSON 里所有字符串（带路径），供方向声明扫描用。"""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk_strings(v, path + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_strings(v, path + (str(i),)))
    elif isinstance(obj, str):
        out.append((".".join(path), obj))
    return out


def _walk_keys(obj, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """递归抽出所有 map 的键（带路径），供「已退役字段」扫描用。"""
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((".".join(path + (str(k),)), str(k)))
            out.extend(_walk_keys(v, path + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk_keys(v, path + (str(i),)))
    return out


def check_no_change_narrative(doc: dict, rel: str) -> list[str]:
    """事实文档不带变更记录，也不带内部工单号 / 复核编号（`AGENTS.md` §2.1）。

    判据只有两条：出现 `change_log` / `actor` 键，或任何字符串值里出现 `ticket NN` /
    `NN-review` / `WP-xxxx`。两条都是**类**级的：读者的两三个工具调用里，过程编号只会让他
    去找一个仓外不存在的东西，而当前事实已经写在同一行的正文里。
    """
    errors: list[str] = []
    for path, key in _walk_keys(doc):
        if key in RETIRED_KEYS:
            errors.append(
                f"{rel}: {path} 是变更记录字段（{key}），已退役——事实文档只写当前状态；"
                f"过程史写进 {DOC_LOG}")
    for path, s in _walk_strings(doc):
        m = PROCESS_REF_RE.search(s)
        if m:
            errors.append(
                f"{rel}: {path} 含内部工单号 / 复核编号 {m.group(0)!r}"
                f"（事实文档不记过程编号，正文写实质；过程史与编号写进 {DOC_LOG}）")
    return errors


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _has_number(obj) -> bool:
    """子树里有没有数字（决定一个 `delta*` 键是「读数」还是「一句话解释」）。"""
    if _is_number(obj):
        return True
    if isinstance(obj, dict):
        return any(_has_number(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_number(v) for v in obj)
    return False


def delta_nodes(data) -> list[tuple[str, str]]:
    """本 `results.json` 的全部 Δ 读数：[(dotted path, 末位键名)]。

    口径（`harness/research-schema.md` §2.3）：键名含 `delta` / `Δ`、且值里带数字；
    声明字段自己（`delta_directions` / `delta_direction_note`）不算读数。
    同一个键名在同一文件里出现多次（如逐臂 `delta_bpc`）就是一个待声明项，
    要按路径分开时就写路径键。
    """
    out: list[tuple[str, str]] = []

    def walk(obj, path: tuple[str, ...]) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = path + (str(k),)
                if (k not in DELTA_META_KEYS and DELTA_NODE_RE.search(str(k))
                        and _has_number(v)):
                    out.append((".".join(p), str(k)))
                walk(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, path + (str(i),))

    if isinstance(data, dict):
        walk(data, ())
    return out


def resolve_delta_direction(mapping, path: str, leaf: str):
    """字段的方向：先看完整路径、再看末位键名；两处都没有返回 `None`（= 漏声明）。"""
    if not isinstance(mapping, dict):
        return None
    if path in mapping:
        return mapping[path]
    return mapping.get(leaf)


CRITERION_KEY_RE = re.compile(r"(?i)criteri(on|a)")


#: 「声明了缺失」的字段，供 `main()` 统计与打印（不等于 fail，但不许静默）。


#: 「声明了缺失」的字段，供 `main()` 统计与打印（不等于 fail，但不许静默）。
UNDECLARED_FIELDS: list[tuple[str, str]] = []


def criterion_block_claims(data, path: str) -> tuple[set[str], list[str]]:
    """该字段所隶的最近一个判据块（键名含 criterion/criteria）里的方向声明。

    返回 (`声明到的方向集合`, `原词列表`)。字段不在任何判据块里 ⇒ 空集。
    """
    segs = path.split(".")
    for cut in range(len(segs) - 1, -1, -1):  # 从最近的一个判据键往根走
        if CRITERION_KEY_RE.search(segs[cut]):
            prefix = segs[:cut]
            node = data
            for s in prefix:
                node = node[int(s)] if isinstance(node, list) else node[s]
            block = node.get(segs[cut]) if isinstance(node, dict) else None
            dirs: set[str] = set()
            raws: list[str] = []
            for _w, s in _walk_strings(block):
                for d, raw in direction_claims(s, ratios=False):
                    dirs.add(d)
                    raws.append(raw)
                fd, raw = formula_direction(s)
                if fd:
                    dirs.add(fd)
                    raws.append(f"算式「{raw}」")
            return dirs, raws
    return set(), []


def check_results_delta_directions(rel: str, data) -> list[str]:
    """`results.json` 的 `delta_directions`：映射形状 + 枚举 + 字段全覆盖 + 与判据串不矛盾。

    口径见 `harness/research-schema.md` §2.3：文件里每个 `delta*` 读数都必须出现（路径或键名），
    同一文件的不同字段可以不同方向；`undeclared` 声明「缺失」本身，它不参与自洽校验，
    但必须配一句 `delta_direction_note` 并被统计打印。
    """
    if not isinstance(data, dict):
        return []
    errors: list[str] = []
    if LEGACY_DELTA_DIRECTION_KEY in data:
        errors.append(f"{rel}: 顶部 delta_direction 单值已废弃，换成字段级映射 "
                      f"{DELTA_DIRECTIONS_KEY}（§2.3）")
    nodes = delta_nodes(data)
    mapping = data.get(DELTA_DIRECTIONS_KEY)
    if mapping is None:
        if nodes:
            errors.append(f"{rel}: 报了 {len(nodes)} 个 Δ 读数却缺 {DELTA_DIRECTIONS_KEY}"
                          f"（每个 delta 字段都要声明，判不了就写 undeclared）")
        return errors
    if not isinstance(mapping, dict):
        return errors + [f"{rel}: {DELTA_DIRECTIONS_KEY} 必须是映射"
                         f"（字段名/路径 → 枚举），现在是 {type(mapping).__name__}"]
    bad_shape = [k for k in mapping if not isinstance(k, str)]
    if bad_shape:
        errors.append(f"{rel}: {DELTA_DIRECTIONS_KEY} 的键必须是字段名或路径：{bad_shape}")
    for key, val in mapping.items():
        if val not in DELTA_DIRECTIONS:
            errors.append(f"{rel}: {DELTA_DIRECTIONS_KEY}[{key!r}]={val!r} 非法"
                          f"（合法值：{DELTA_DIRECTIONS}）")
    covered: set[str] = set()
    undeclared_here: list[str] = []
    for path, leaf in nodes:
        if path in mapping:
            covered.add(path)
        elif leaf in mapping:
            covered.add(leaf)
        else:
            errors.append(f"{rel}:{path}: 是 Δ 读数但 {DELTA_DIRECTIONS_KEY} 里没有声明"
                          f"（写字段名 {leaf!r} 或完整路径）")
            continue
        declared = resolve_delta_direction(mapping, path, leaf)
        if declared == "undeclared":
            undeclared_here.append(path)
            continue
        if declared not in DELTA_DIRECTIONS:
            continue  # 已在上面报过非法值
        dirs, raws = criterion_block_claims(data, path)
        if not dirs:
            continue
        if len(dirs) > 1:
            UNJUDGED.append(f"{rel}:{path}: 所在判据块同时声明了 {sorted(dirs)} 两种方向，"
                            f"该字段的方向机判不了（待人裁决）")
            continue
        only = next(iter(dirs))
        if only != declared:
            errors.append(f"{rel}:{path}: 判据串的方向「{raws[0]}」与 "
                          f"{DELTA_DIRECTIONS_KEY} 声明的 {declared!r} 矛盾"
                          f"（按该串应为 {only!r}）")
    dangling = [k for k in mapping if k not in covered]
    if dangling:
        errors.append(f"{rel}: {DELTA_DIRECTIONS_KEY} 里有对不上任何 Δ 字段的键：{dangling}"
                      f"（写错名字就等于没声明）")
    if undeclared_here:
        UNDECLARED_FIELDS.extend((rel, p) for p in undeclared_here)
        note = data.get(DELTA_DIRECTION_NOTE_KEY)
        if not isinstance(note, str) or not note.strip():
            errors.append(f"{rel}: {len(undeclared_here)} 个字段标了 undeclared，却缺 "
                          f"{DELTA_DIRECTION_NOTE_KEY}（为什么判不了，一句话）")
    return errors


def delta_direction_summary() -> str:
    """一行统计：多少文件报 Δ、多少已全字段声明、多少 undeclared（缺口要看得见）。"""
    files = declared_files = undeclared_files = 0
    declared_readings = 0
    undeclared_keys: set[tuple[str, str]] = set()
    undeclared_readings = 0
    for path in collect_results_files():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 —— 解析失败由 validate_results_files 报
            continue
        nodes = delta_nodes(data)
        if not nodes:
            continue
        files += 1
        mapping = data.get(DELTA_DIRECTIONS_KEY) if isinstance(data, dict) else None
        dirs = [resolve_delta_direction(mapping, p, l) for p, l in nodes]
        if not isinstance(mapping, dict) or "undeclared" in dirs:
            undeclared_files += 1
            for (p, l), d in zip(nodes, dirs):
                if d != "undeclared":
                    continue
                undeclared_readings += 1
                undeclared_keys.add((path.relative_to(ROOT).as_posix(),
                                     p if isinstance(mapping, dict) and p in mapping else l))
        else:
            declared_files += 1
            declared_readings += len(nodes)
    return (f"Δ 方向：{files} 个 results.json 报了 Δ 读数；{declared_files} 个文件全字段已声明"
            f"（{declared_readings} 个读数）；{undeclared_files} 个文件含 undeclared"
            f"（{len(undeclared_keys)} 条声明 / {undeclared_readings} 个读数）。")


RESULTS_FILES = "results.json"

def collect_results_files() -> list[Path]:
    if not AXES.is_dir():
        return []
    return sorted(AXES.glob("**/results.json"))


def check_laws_and_costs(doc: dict, rel: str) -> list[str]:
    """校验 `laws` / `cost_rows`（`harness/research-schema.md` §2.1 / §2.2）。"""
    errors: list[str] = []
    laws = doc.get("laws")
    if not isinstance(laws, list):
        errors.append(f"{rel}: laws 必须是 list（无证据支撑写 []）")
        laws = []
    seen: set[int] = set()
    for i, law in enumerate(laws):
        if isinstance(law, bool) or not isinstance(law, int):
            errors.append(f"{rel}: laws[{i}]={law!r} 必须是整数定律编号（1–{LAWS_TOTAL}）")
            continue
        if not 1 <= law <= LAWS_TOTAL:
            errors.append(f"{rel}: laws[{i}]={law} 是未知定律编号"
                          f"（合法值 1–{LAWS_TOTAL}，见 atlas/playbook.md）")
        if law in seen:
            errors.append(f"{rel}: laws 里定律 {law} 重复")
        seen.add(law)

    rows = doc.get("cost_rows")
    if not isinstance(rows, list):
        errors.append(f"{rel}: cost_rows 必须是 list（没有可引用数字写 []）")
        return errors
    for i, row in enumerate(rows):
        where = f"{rel}: cost_rows[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{where} 不是 map")
            continue
        for k in COST_ROW_FIELDS:
            if k not in row:
                errors.append(f"{where} 缺 {k}")
            elif not _nonempty(row[k]):
                errors.append(f"{where} 的 {k} 不得为空")
        law = row.get("law")
        if isinstance(law, bool) or not isinstance(law, int):
            errors.append(f"{where}.law={law!r} 必须是整数定律编号（1–{LAWS_TOTAL}）")
        else:
            if not 1 <= law <= LAWS_TOTAL:
                errors.append(f"{where}.law={law} 是未知定律编号（合法值 1–{LAWS_TOTAL}，"
                              f"见 atlas/playbook.md）")
            elif isinstance(doc.get("laws"), list) and law not in doc["laws"]:
                errors.append(f"{where}.law={law} 不在本项 laws={doc['laws']} 里"
                              f"（代价事实必须归到本项自己声明的定律）")
        if "touches_trunk" in row and not isinstance(row["touches_trunk"], bool):
            errors.append(f"{where}.touches_trunk={row['touches_trunk']!r} 必须是 true/false")
        delta = row.get("delta")
        if delta is not None:
            raw = str(delta).strip()
            if isinstance(delta, bool) or not NUMBER_RE.fullmatch(raw):
                errors.append(f"{where}.delta={delta!r} 必须是带符号数字"
                              f"（单位由 metric 声明、方向由 delta_direction 声明）")
        errors.extend(check_cost_row_direction(row, where))
        src = row.get("source")
        if src is not None:
            p = source_path(src)
            if not p:
                errors.append(f"{where}.source 为空")
            elif not (ROOT / p).exists():
                errors.append(f"{where}.source 指向不存在的路径：{p}")
    return errors


def _nonempty(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, dict, tuple)):
        return len(v) > 0
    return True


def _ev_path(s: str) -> str:
    return str(s).split(" @")[0].strip()


def validate_file(path: Path) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """校验单个 project.yaml。返回 (errors, anchors, evidences)。"""
    rel = path.relative_to(ROOT).as_posix()
    errors: list[str] = []
    anchors: dict[str, list[str]] = {}
    evidences: dict[str, list[str]] = {}
    try:
        doc = yaml.load(path.read_text(encoding="utf-8"), Loader=DuplicateKeyLoader)
    except Exception as e:  # noqa: BLE001 —— 解析失败要如实报出
        return [f"{rel}: YAML 解析失败：{e}"], anchors, evidences
    if not isinstance(doc, dict):
        return [f"{rel}: 顶层不是 map"], anchors, evidences

    for k in REQUIRED:
        if k not in doc:
            errors.append(f"{rel}: 缺字段 {k}")

    # 关键字段非空（必填 = 校验器 fail，空串/空列表不得蒙混）
    for k in NONEMPTY:
        if k in doc and not _nonempty(doc.get(k)):
            errors.append(f"{rel}: {k} 不得为空（必填字段）")

    # dir 与实际位置一致
    actual = path.parent.relative_to(ROOT).as_posix()
    if doc.get("dir") and str(doc["dir"]).rstrip("/") != actual:
        errors.append(f"{rel}: dir={doc['dir']!r} 与实际目录 {actual!r} 不一致")

    # status
    st = doc.get("status")
    if st not in VALID_STATUS:
        errors.append(f"{rel}: 非法 status={st!r}（合法值：{VALID_STATUS}）")

    # milestone 单数
    ms = doc.get("milestone")
    if not isinstance(ms, dict):
        errors.append(f"{rel}: milestone 必须是 map（有且仅有一个：output + acceptance），"
                      f"实际 {type(ms).__name__}")
    else:
        for k in ("output", "acceptance"):
            if not _nonempty(ms.get(k)):
                errors.append(f"{rel}: milestone 缺 {k}（单 milestone 必须有 output + acceptance）")

    # role_map
    rm = doc.get("role_map")
    if not isinstance(rm, dict):
        errors.append(f"{rel}: role_map 必须是 map")
    else:
        for k in ROLE_REQUIRED:
            if not _nonempty(rm.get(k)):
                errors.append(f"{rel}: role_map 缺角色 {k}")

    # work_packages
    wps = doc.get("work_packages")
    if not isinstance(wps, list):
        errors.append(f"{rel}: work_packages 必须是 list")
    else:
        for i, wp in enumerate(wps):
            if not isinstance(wp, dict):
                errors.append(f"{rel}: work_packages[{i}] 不是 map")
                continue
            for k in WP_REQUIRED:
                if k not in wp:
                    errors.append(f"{rel}: work_packages[{i}] 缺 {k}")
            if wp.get("status") is not None and wp["status"] not in VALID_STATUS:
                errors.append(f"{rel}: work_packages[{i}] 非法 status={wp['status']!r}")

    # cost_model 禁墙钟/日期
    cm = doc.get("cost_model")
    if cm is not None:
        m = WALLCLOCK_RE.search(str(cm))
        if m:
            errors.append(f"{rel}: cost_model 含墙钟/日期指标 {m.group(0)!r}"
                          f"（只许 FLOPs / layer-passes / tokens-per-forward 等等效理论量）")

    # validation 三要素（对照 + 判据 + 口径）
    val = str(doc.get("validation") or "")
    if not any(w in val for w in ("对照", "基线", "stock", "SOTA", "sota", "baseline")):
        errors.append(f"{rel}: validation 缺「与 SOTA/stock 基线对照」")
    if not any(w in val for w in ("判据", "该涨", "该跌", "必须不变", "criteria", "objective")):
        errors.append(f"{rel}: validation 缺「客观判据」（该涨/该跌/必须不变）")
    if not any(w in val for w in ("口径", "bpc", "FLOPs", "layer-passes",
                                  "tokens-per-forward", "perplexity", "acc", "一致率")):
        errors.append(f"{rel}: validation 缺口径")

    # 变更叙事与内部工单号（AGENTS.md §2.1：事实文档只写当前状态）
    errors.extend(check_no_change_narrative(doc, rel))

    # laws / cost_rows（harness/research-schema.md §2.1 / §2.2）
    errors.extend(check_laws_and_costs(doc, rel))

    # 已完成/被否证 ⇒ conclusion + experiments；被否证 ⇒ failure_analysis 非空
    if st in DONE_STATUS:
        if not _nonempty(doc.get("conclusion")):
            errors.append(f"{rel}: status={st} 但无 conclusion")
        if not _nonempty(doc.get("experiments")):
            errors.append(f"{rel}: status={st} 但无 experiments（结论必须有证据）")
        if st == "被否证" and not _nonempty(doc.get("failure_analysis")):
            errors.append(f"{rel}: status=被否证 但 failure_analysis 为空"
                          f"（必须区分 设计问题 / 结构问题 / innovation 本身失败 并给证据）")

    # experiments 路径存在 + 收集锚点
    for ev in (doc.get("experiments") or []):
        p = _ev_path(ev)
        if p and not (ROOT / p).exists():
            errors.append(f"{rel}: experiments 路径不存在：{p}")
        for anc in ANCHOR_RE.findall(str(ev)):
            anchors.setdefault(anc, []).append(rel)
        evidences.setdefault(str(ev), []).append(rel)
    for i, wp in enumerate(wps or []):
        if isinstance(wp, dict):
            evs = wp.get("evidence") or []
            if isinstance(evs, str):
                evs = [evs]
            for ev in evs:
                # WP 的 evidence 是「证据将落在哪」，可能是**未来路径** ⇒ 只收集锚点，
                # 不强制路径已存在（路径存在性只在 status=已完成/被否证 的顶层 `experiments` 上硬查）。
                for anc in ANCHOR_RE.findall(str(ev)):
                    anchors.setdefault(anc, []).append(rel)
                evidences.setdefault(str(ev), []).append(rel)

    return errors, anchors, evidences


def collect_files() -> list[Path]:
    if not AXES.is_dir():
        return []
    return sorted(AXES.glob("**/project.yaml"))


def validate_all() -> tuple[list[str], list[dict]]:
    """校验全部 project.yaml。返回 (errors, entries)。entries 供 summary.py 制表。"""
    errors: list[str] = []
    entries: list[dict] = []
    UNJUDGED.clear()
    anchors: dict[str, list[str]] = {}
    evidences: dict[str, list[str]] = {}
    seen_ids: dict[str, str] = {}
    files = collect_files()
    if not files:
        return ["axes/** 下没有找到任何 project.yaml"], []
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        errs, anch, evid = validate_file(path)
        errors.extend(errs)
        for k, v in anch.items():
            anchors.setdefault(k, []).extend(v)
        for k, v in evid.items():
            evidences.setdefault(k, []).extend(v)
        try:
            doc = yaml.load(path.read_text(encoding="utf-8"), Loader=DuplicateKeyLoader)
        except Exception:  # noqa: BLE001 —— 已在 validate_file 报错
            continue
        if isinstance(doc, dict):
            gid = doc.get("id")
            if gid:
                if gid in seen_ids:
                    errors.append(f"{rel}: id={gid!r} 与 {seen_ids[gid]} 重复")
                else:
                    seen_ids[gid] = rel
            entries.append({"path": rel, "doc": doc})
    errors.extend(check_anchor_syntax(evidences))
    errors.extend(check_anchors(anchors))
    errors.extend(validate_results_files())
    return errors, entries


def validate_prereg() -> tuple[list[str], dict]:
    """预登记绑定（`tools/prereg_bind.py::check`：存在 / 契约未改 / 定序）。"""
    return prereg_bind.check()


def unjudged_notes() -> list[str]:
    """上一次 `validate_all()` 收集到的「符号无法机判」的行（不算 fail，但必须打印）。"""
    return list(UNJUDGED)


def validate_results_files() -> list[str]:
    """校验全部 `axes/**/results.json` 的 Δ 方向声明（`harness/research-schema.md` §2.3）。"""
    errors: list[str] = []
    for path in collect_results_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 —— 解析失败要如实报出
            errors.append(f"{rel}: JSON 解析失败：{e}")
            continue
        errors.extend(check_results_delta_directions(rel, data))
    return errors


def main() -> int:
    errors, entries = validate_all()
    prereg_errors, prereg_stats = validate_prereg()
    for note in unjudged_notes():
        print(f"不判：{note}", file=sys.stderr)
    print(delta_direction_summary(), file=sys.stderr)
    print(prereg_bind.summary_line(prereg_stats), file=sys.stderr)
    errors.extend(prereg_errors)
    if errors:
        print(f"FAIL：{len(errors)} 项")
        for e in errors:
            print("  -", e)
        return 1
    print(f"OK：{len(entries)} 个 project.yaml 全部符合 harness/research-schema.md。")
    # 文档引用（`tools/check_docs.py` 是唯一实现，这里只转调）
    return check_docs.main([])


if __name__ == "__main__":
    sys.exit(main())
