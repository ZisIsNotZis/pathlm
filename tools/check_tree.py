#!/usr/bin/env python3
"""check_tree.py — 研究树（`axes/**/project.yaml`）的格式与证据校验（可失败）。

新结构下 `axes/**/project.yaml` 是研究项的 SSOT（契约见 `docs/research-schema.md`，
布局见 `docs/structure.md` §12）。旧生成式 wiki（catalog / build_wiki / INDEX）已退役，
`tools/check_schema.py` 转调本模块。

校验项：
  1. 字段齐全：`docs/research-schema.md` §2 列出的字段一个都不能少。
  2. `milestone` 单数：必须是 map 且有 `output` + `acceptance`（不是 list）。
  3. `role_map` 四角色齐：owner / writer / reviewer / integrator。
  4. `work_packages`：每条含 id/what/goal/means/acceptance/evidence/review/status。
  5. `status` 合法：未开始 / 进行中 / 已完成 / 被否证。
  6. `cost_model` 禁墙钟/日期（只许等效理论量：FLOPs / layer-passes / tokens-per-forward）。
  7. `status ∈ {已完成, 被否证}` ⇒ 必须有 `conclusion` + 非空且存在的 `experiments`。
  8. `validation` 必须含「对照/基线/stock/SOTA」之一 + 「口径/bpc/FLOPs/...」之一。
  9. `id` 全局唯一；`dir` 必须等于文件实际所在目录。
 10. 证据锚点：`experiments/evidence` 里的 `path @<sha>` 路径存在，锚点是真实可达提交。

用法：python3 tools/check_tree.py   （退出码 0 = 全过；1 = 有错）
"""

from __future__ import annotations

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
    "roadmap", "status", "conclusion_review", "experiments", "argument", "gaps",
    "started", "ended", "change_log",
)
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
    if not any(w in val for w in ("口径", "bpc", "FLOPs", "layer-passes",
                                  "tokens-per-forward", "perplexity", "acc", "一致率")):
        errors.append(f"{rel}: validation 缺口径")

    # 已完成/被否证 ⇒ conclusion + experiments
    if st in DONE_STATUS:
        if not _nonempty(doc.get("conclusion")):
            errors.append(f"{rel}: status={st} 但无 conclusion")
        if not _nonempty(doc.get("experiments")):
            errors.append(f"{rel}: status={st} 但无 experiments（结论必须有证据）")

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
    return errors, entries


def main() -> int:
    errors, entries = validate_all()
    if errors:
        print(f"FAIL：{len(errors)} 项")
        for e in errors:
            print("  -", e)
        return 1
    print(f"OK：{len(entries)} 个 project.yaml 全部符合 docs/research-schema.md。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
