#!/usr/bin/env python3
"""check_tree.py — 研究树（`axes/**/project.yaml`）的格式与证据校验（可失败）。

新结构下 `axes/**/project.yaml` 是研究项的 SSOT（契约见 `docs/research-schema.md`，
布局见 `docs/structure.md` §12）。旧生成式 wiki（catalog / build_wiki / INDEX）已退役，
`tools/check_schema.py` 转调本模块。

校验项：
  1. 字段齐全：`docs/research-schema.md` §2 列出的字段一个都不能少。
  2. 关键字段非空：`purpose` / `innovation` / `validation` / `objective_criteria` /
     `falsification` 不得为空串/空列表。
  3. `milestone` 单数：必须是 map 且有 `output` + `acceptance`（不是 list）。
  4. `role_map` 四角色齐：owner / writer / reviewer / integrator。
  5. `work_packages`：每条含 id/what/goal/means/acceptance/evidence/review/status。
  6. `status` 合法：未开始 / 进行中 / 已完成 / 被否证。
  7. `cost_model` 禁墙钟/日期（只许等效理论量：FLOPs / layer-passes / tokens-per-forward）。
  8. `status ∈ {已完成, 被否证}` ⇒ 必须有 `conclusion` + 非空且存在的 `experiments`；
     `status = 被否证` ⇒ 还必须有非空 `failure_analysis`（失败归因：设计 / 结构 / innovation，
     见 `docs/research-schema.md` §7）。
  9. `validation` 必须含「对照/基线/stock/SOTA」之一 + 「判据/该涨/该跌」之一 +
     「口径/bpc/...」之一。
 10. `id` 全局唯一；`dir` 必须等于文件实际所在目录。
 11. 证据锚点：`experiments` 路径存在；WP `evidence` 允许未来路径（只校锚点 `@<sha>` 可达）。
 12. `laws` / `cost_rows`（`docs/research-schema.md` §2.1 / §2.2）：定律编号限 1–11 且不重复，
     `cost_rows[].law` 必须是本项 `laws` 的成员，`touches_trunk` 必须是 bool，`delta` 必须是数字，
     `source` 指向的路径必须存在。

用法：python3 tools/check_tree.py   （退出码 0 = 全过；1 = 有错）
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"

VALID_STATUS = ("未开始", "进行中", "已完成", "被否证")
DONE_STATUS = ("已完成", "被否证")

REQUIRED = (
    "id", "axis", "title", "dir", "purpose", "innovation", "expected_effect",
    "sota_baseline", "gap", "method", "validation", "objective_criteria",
    "falsification", "cost_model", "role_map", "milestone", "work_packages",
    "roadmap", "status", "failure_analysis", "conclusion_review", "laws",
    "cost_rows", "experiments", "argument", "gaps", "started", "ended", "change_log",
)
#: 「必填」= 必须**非空**的字符串字段（不允许空串/空列表蒙混，见 docs/research-schema.md §2）。
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

#: `laws` 的编号域 = `docs/playbook.md` 的「定律 1..11」（见 `docs/research-schema.md` §2.1）。
LAWS_TOTAL = 11
#: `cost_rows` 一条必须齐备的键（见 `docs/research-schema.md` §2.2）。
COST_ROW_FIELDS = ("law", "change", "touches_trunk", "metric", "delta",
                   "comparator", "scale", "boundary", "source")
#: `delta` 必须是带符号数字（含千分位/小数；单位与方向由 `delta_direction` + `metric` 共同声明）。
NUMBER_RE = re.compile(r"[+-]?\d+(?:[\d,]*)?(?:\.\d+)?")

#: Δ 方向的**唯一枚举**（`docs/research-schema.md` §2.3）：名字即算式，读表的人不可能读反。
DELTA_DIRECTIONS = ("comparator_minus_arm", "arm_minus_comparator")
#: 枚举 → 展示层中文短语（`docs/cost-table.md` 的「Δ 方向」列用这一份，不另抄）。
DELTA_DIRECTION_LABELS = {
    "comparator_minus_arm": "正 = 改善",
    "arm_minus_comparator": "正 = 更差",
}
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
#: `results.json` 里出现这些键名即视为「本文件报了 Δ 读数」（`workspace.delta` 这类配置项不算）。
RESULTS_DELTA_KEY_RE = re.compile(
    r"(?i)(^|_)(delta_(?:bpc|pp|vs_|mean|halves|window_se|min_pp)|_delta_bpc)(_|$|\.)|Δ")
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


def direction_claims(text: str) -> list[tuple[str, str]]:
    """把一段文字里所有的方向声明抽成 [(方向, 原词)]；`负 = 更好` 一类的反向说法已翻转。"""
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
        return [f"{where} 缺 delta_direction（合法值：{DELTA_DIRECTIONS}）"]
    if declared not in DELTA_DIRECTIONS:
        return [f"{where}.delta_direction={declared!r} 非法（合法值：{DELTA_DIRECTIONS}）"]
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


def results_delta_scope(data) -> bool:
    """本 `results.json` 是否报出 Δ（= 需要 `delta_direction` 的机械入口）。"""
    if "Δ" in json.dumps(data, ensure_ascii=False):
        return True
    return any(RESULTS_DELTA_KEY_RE.search(k) for k, _ in _all_keys(data))


def _all_keys(obj, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(("".join(path) + str(k), str(k)))
            out.extend(_all_keys(v, path + (str(k), ".")))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_all_keys(v, path + (str(i), ".")))
    return out


CRITERION_KEY_RE = re.compile(r"(?i)criteri(on|a)")


def criterion_strings(obj, path: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    """只取「预登记判据」那一类字符串（键名含 criterion/criteria）——方向的权威处。

    别的散文（判词、边界、结论）不参与方向机检：它们常混着别的量纲的读数，
    拿它们投票只会把方向判错。这条边界在 `docs/research-schema.md` §2.3 写明。
    """
    out: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if CRITERION_KEY_RE.search(str(k)):
                out.extend(_walk_strings(v, path + (str(k),)))
            else:
                out.extend(criterion_strings(v, path + (str(k),)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(criterion_strings(v, path + (str(i),)))
    return out


def results_scope_claims(data) -> list[tuple[str, str, str]]:
    """`results.json` 里可机检的方向声明：[(路径, 方向, 原词)]。

    口径：判据串（键名含 criterion/criteria）里的 `正/负/>1 = …` 与 `Δ = A − B`；
    判据串里没有时退回全文件的算式声明。两处都没有 ⇒ 无可检的声明。
    """
    out: list[tuple[str, str, str]] = []
    for where, s in criterion_strings(data):
        for d, raw in direction_claims(s):
            out.append((where, d, raw))
        fd, raw = formula_direction(s)
        if fd:
            out.append((where, fd, f"算式「{raw}」"))
    if out:
        return out
    for where, s in _walk_strings(data):
        fd, raw = formula_direction(s)
        if fd:
            out.append((where, fd, f"算式「{raw}」"))
    return out


def check_results_delta_direction(rel: str, data) -> list[str]:
    """`results.json` 的 `delta_direction`：有可检的方向声明就必须声明，且必须一致。"""
    if not isinstance(data, dict):
        return []
    errors: list[str] = []
    declared = data.get("delta_direction")
    claims = results_scope_claims(data)
    dirs = {d for _w, d, _r in claims}
    if len(dirs) > 1:
        UNJUDGED.append(f"{rel}: 文件里两条判据的 Δ 方向相反，一条 delta_direction 表达不了"
                        f"（待人裁决，不猜）：{sorted(dirs)}")
        return errors
    if claims and declared is None:
        errors.append(f"{rel}: 判据串声明了 Δ 方向却缺 delta_direction"
                      f"（合法值：{DELTA_DIRECTIONS}）")
    if declared is not None and declared not in DELTA_DIRECTIONS:
        return errors + [f"{rel}: delta_direction={declared!r} 非法（合法值：{DELTA_DIRECTIONS}）"]
    for where, d, raw in claims:
        if declared is not None and d != declared:
            errors.append(f"{rel}:{where}: 判据串的方向「{raw}」与 delta_direction="
                          f"{declared!r} 矛盾（按该串应为 {d!r}）")
    return errors


RESULTS_FILES = "results.json"

def collect_results_files() -> list[Path]:
    if not AXES.is_dir():
        return []
    return sorted(AXES.glob("**/results.json"))


def check_laws_and_costs(doc: dict, rel: str) -> list[str]:
    """校验 `laws` / `cost_rows`（`docs/research-schema.md` §2.1 / §2.2）。"""
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
                          f"（合法值 1–{LAWS_TOTAL}，见 docs/playbook.md）")
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
                              f"见 docs/playbook.md）")
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

    # laws / cost_rows（docs/research-schema.md §2.1 / §2.2）
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


def unjudged_notes() -> list[str]:
    """上一次 `validate_all()` 收集到的「符号无法机判」的行（不算 fail，但必须打印）。"""
    return list(UNJUDGED)


def validate_results_files() -> list[str]:
    """校验全部 `axes/**/results.json` 的 Δ 方向声明（`docs/research-schema.md` §2.3）。"""
    errors: list[str] = []
    for path in collect_results_files():
        rel = path.relative_to(ROOT).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 —— 解析失败要如实报出
            errors.append(f"{rel}: JSON 解析失败：{e}")
            continue
        errors.extend(check_results_delta_direction(rel, data))
    return errors


def main() -> int:
    errors, entries = validate_all()
    for note in unjudged_notes():
        print(f"不判（符号无法机判）：{note}", file=sys.stderr)
    if errors:
        print(f"FAIL：{len(errors)} 项")
        for e in errors:
            print("  -", e)
        return 1
    print(f"OK：{len(entries)} 个 project.yaml 全部符合 docs/research-schema.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
