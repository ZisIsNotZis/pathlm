#!/usr/bin/env python3
"""prereg_bind.py — 预登记绑定：把「契约先于读数」变成可机检的事实。

`README.md` 里写死的判据线只有**证明它写在读数之前**才有约束力。契约文件与 `results.json`
落在同一个提交里时，仓内没有任何独立定序证据 ⇒「契约先于读数」**不可判**（同一缺口已被
独立复核在 `1.2.3` / `3.1.5` / `2653` 三处判到）。本工具把契约文件的 `sha256` 与绑定时刻
写进实验目录下的 `prereg_binding.json`；训练 / 评测入口开跑前 `--require` 它；
`tools/check_tree.py` 转调本文件的 `check()` 做全树机检。

用法：
    python3 tools/prereg_bind.py <实验目录>             # 绑定（幂等：已绑定且契约未变则不动 bound_at）
    python3 tools/prereg_bind.py <实验目录> --require   # 只核绑定存在且契约未变（训练 / 评测入口用）
    python3 tools/prereg_bind.py <实验目录> --rebind    # 契约改过之后显式重绑（bound_at 改写为现在）
    python3 tools/prereg_bind.py --backfill-unbound     # 历史产物：写 bound_at="unbound"（显式缺失，不伪造时间）
    python3 tools/prereg_bind.py --check                # 全树机检（`check_tree.py` 用的同一实现）

绑定记录 `prereg_binding.json`（实验根目录，契约文件旁边）：

| 键 | 值 |
|---|---|
| `experiment` | 实验目录（相对仓库根） |
| `bound_at` | ISO 时间（带时区），或 `"unbound"` = 历史产物、无定序证据 |
| `unbound_reason` | 仅 `unbound` 时出现：为什么没有定序证据 |
| `contract_files` | 契约文件名 → 绑定时的 `sha256`（`unbound` 时不存在：没有「绑定时」可言） |
| `readings_existing_at_bind` | 绑定那一刻已存在的 `results.json`（路径 → 当时的 `sha256`）：它们是**绑定之前**的读数，定序不可判；判定时要求读数被重跑（`sha256` 变了）且产物时间晚于 `bound_at` |
| `tool` | 产生这条记录的工具 |

机检四项（`check()`）：**有 `results.json` 的实验必须有绑定记录**；契约文件当前 `sha256`
必须与绑定时的相同（不同 ⇒ 报「契约在绑定后被改过」并列出差异文件）；`bound_at` 必须
**严格早于** 判序锚的产物时间（判序锚 = 实验根 `results.json`；该实验只有嵌套读数时取其中
最新的一份），且该读数不得是绑定时已存在的那一份（`sha256` 必须变）；两份文件都已入库时，
绑定记录的提交必须**早于**读数的提交（同一提交 ⇒ 仓内仍无独立定序证据，判失败）。

`"unbound"` 是**显式缺失**：`--check` 只计数、不判失败并打印计数。它不得用来给新实验免检——
新实验一律先绑定再训练。同一实验目录里追加第二批读数时，绑定记下当时已有的读数
（`readings_existing_at_bind`），因此给旧读数「补绑」不会伪装成定序证据：旧读数要重新产出才过门。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"
BINDING = "prereg_binding.json"
RESULTS = "results.json"
#: 契约文件：`README.md` 必须在；`config.json` 存在即纳入（它也是判据的预登记面）。
CONTRACT_REQUIRED = "README.md"
CONTRACT_OPTIONAL = ("config.json",)
#: `bound_at` 的显式缺失值（历史产物）。
UNBOUND = "unbound"
UNBOUND_REASON = "历史产物，无定序证据"
TOOL = "tools/prereg_bind.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def experiment_dir(arg: str) -> Path:
    """把命令行给的方向解析成实验目录（必须是 `axes/**/experiments/` 下、含 `README.md` 的目录）。"""
    path = Path(arg)
    if not path.is_absolute():
        path = (Path.cwd() / path)
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"不是目录：{arg}")
    if not path.is_relative_to(AXES.resolve()):
        raise SystemExit(f"不在 axes/ 下：{arg}")
    if not (path / CONTRACT_REQUIRED).exists():
        raise SystemExit(f"缺契约文件 {CONTRACT_REQUIRED}：{path}"
                         f"（实验目录 = 存契约的目录；跑法请从实验根调用本工具）")
    return path


def contract_names(exp: Path, declared: dict | None = None,
                   given: list[str] | None = None) -> list[str]:
    """本次绑定要覆盖的契约文件清单。已存在的记录用它的清单（重核不该悄悄换集合）。"""
    if given:
        names = list(given)
    elif declared:
        names = list(declared)
    else:
        names = [CONTRACT_REQUIRED] + [n for n in CONTRACT_OPTIONAL if (exp / n).exists()]
    if CONTRACT_REQUIRED not in names:
        names.insert(0, CONTRACT_REQUIRED)
    return names


def load(exp: Path) -> dict | None:
    p = exp / BINDING
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{rel(p)}: JSON 解析失败：{e}")


def contract_diff(exp: Path, declared: dict) -> list[str]:
    """契约文件当前 sha256 与绑定不同的那些文件（含已删除的契约）。"""
    changed = []
    for name, want in declared.items():
        p = exp / name
        if not p.exists() or sha256(p) != want:
            changed.append(name)
    return sorted(changed)


def write_record(exp: Path, record: dict) -> Path:
    p = exp / BINDING
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def bind(exp: Path, given: list[str] | None = None, rebind: bool = False) -> int:
    existing = load(exp)
    if existing is not None and not rebind:
        if existing.get("bound_at") == UNBOUND:
            print(f"已有 unbound 记录（{UNBOUND_REASON}）：{rel(exp / BINDING)}", file=sys.stderr)
            print("它证明不了定序；若这是重跑的实验，用 --rebind 把 bound_at 改写为现在"
                  "（旧读数的定序随之失去，须重跑读数）", file=sys.stderr)
            return 1
        changed = contract_diff(exp, existing.get("contract_files") or {})
        if changed:
            print(f"契约在绑定后被改过：{', '.join(changed)}（{rel(exp / BINDING)}）", file=sys.stderr)
            print("判据「先于读数」对旧绑定不再成立：改回契约，或用 --rebind 显式重绑并重跑读数",
                  file=sys.stderr)
            return 1
        print(f"已绑定，契约未变：{rel(exp / BINDING)}（bound_at {existing['bound_at']}）")
        return 0

    names = contract_names(exp, (existing or {}).get("contract_files"), given)
    missing = [n for n in names if not (exp / n).exists()]
    if missing:
        print(f"缺契约文件：{', '.join(missing)}（{rel(exp)}）", file=sys.stderr)
        return 1

    prior = {rel(p): sha256(p) for p in sorted(exp.glob(f"**/{RESULTS}"))}
    record = {
        "experiment": rel(exp),
        "bound_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "contract_files": {n: sha256(exp / n) for n in names},
        "readings_existing_at_bind": prior,
        "tool": TOOL,
    }
    p = write_record(exp, record)
    print(f"已绑定：{rel(p)}  bound_at={record['bound_at']}  契约={', '.join(names)}")
    if prior:
        print(f"注意：本目录已有 {len(prior)} 份绑定前的读数（已记入 readings_existing_at_bind）"
              "⇒ 它们的定序不可判；本绑定只给**重跑后**的读数定序", file=sys.stderr)
    return 0


def require(exp: Path) -> int:
    """训练 / 评测入口用：绑定缺失即拒绝启动；契约被改过即报出差异文件。"""
    existing = load(exp)
    if existing is None:
        print(f"先绑定契约：python3 {TOOL} {rel(exp)}", file=sys.stderr)
        print(f"（判据「先于读数」靠 {BINDING} 定序；训练之前必须绑定）", file=sys.stderr)
        return 1
    if existing.get("bound_at") == UNBOUND:
        print(f"注意：{rel(exp / BINDING)} 是 unbound（{UNBOUND_REASON}）⇒ 本实验的定序不可判；"
              "新读数请先 --rebind", file=sys.stderr)
        return 0
    changed = contract_diff(exp, existing.get("contract_files") or {})
    if changed:
        print(f"契约在绑定后被改过：{', '.join(changed)}（{rel(exp / BINDING)}）", file=sys.stderr)
        print("拒绝启动：先确认契约是最终版，再 --rebind 并重跑读数", file=sys.stderr)
        return 1
    print(f"预登记已绑定：{rel(exp / BINDING)}（bound_at {existing['bound_at']}）")
    return 0


def experiments() -> dict[Path, list[Path]]:
    """有结果的实验 → 它的全部 `results.json`。

    一个 `results.json` 所属的**实验** = 它向上最近的、含 `README.md`（契约）的目录：
    臂级 / 冒烟读数（`runs/<arm>/results.json`）归属同一个契约，不各自单独绑定。
    """
    owners: dict[Path, list[Path]] = {}
    for p in sorted(AXES.glob("**/results.json")):
        owners.setdefault(owner_of(p), []).append(p)
    return owners


def owner_of(p: Path) -> Path:
    """`results.json` → 它所属的实验目录（向上最近的含契约的目录）。"""
    cur = p.parent
    while cur != AXES.resolve() and AXES.resolve() in cur.parents:
        if (cur / CONTRACT_REQUIRED).exists():
            return cur
        cur = cur.parent
    return p.parent                               # 没有契约目录可归（不该发生）：按自身目录计


def order_anchor(results: list[Path]) -> Path:
    """判序锚：实验根 `results.json`；只有嵌套读数时取其中最新的一份。"""
    owner = owner_of(results[0])
    root = owner / RESULTS
    if root in results:
        return root
    return max(results, key=lambda p: p.stat().st_mtime)


def last_commit(args: list[str]) -> str | None:
    """`git log -1 --format=%cI -- <路径>`：返回 ISO 提交时间；未入库 / 无 git 时返回 None。"""
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def uncommitted(path: Path) -> bool:
    """该路径相对 HEAD 有无未提交改动（未跟踪 / 已修改都算）。"""
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", rel(path)],
                             cwd=ROOT, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return True
    return bool(out.stdout.strip())


def check() -> tuple[list[str], dict]:
    """全树机检。返回 (errors, stats)；stats 供打印计数（含 unbound 的历史计数）。"""
    errors: list[str] = []
    stats = {"experiments": 0, "results_files": 0, "bound": 0, "unbound": 0,
             "missing": 0, "ordered": 0, "predate": 0,
             "git_ordered": 0, "git_pending": 0, "git_same_commit": 0}
    for exp, results in sorted(experiments().items()):
        stats["experiments"] += 1
        stats["results_files"] += len(results)
        name = rel(exp)
        record = load(exp)
        if record is None:
            stats["missing"] += 1
            errors.append(f"{name}: 有 {RESULTS} 但缺 {BINDING} ⇒ 预登记未绑定"
                          f"（python3 {TOOL} {name}，历史产物用 --backfill-unbound）")
            continue
        bound_at = record.get("bound_at")
        if bound_at == UNBOUND:
            stats["unbound"] += 1            # 显式缺失：只计数，不判失败
            continue
        if not isinstance(bound_at, str):
            errors.append(f"{name}: {BINDING} 的 bound_at 不是 ISO 时间也不是 {UNBOUND!r}")
            continue
        try:
            bound_dt = datetime.fromisoformat(bound_at)
        except ValueError:
            errors.append(f"{name}: {BINDING} 的 bound_at 不是 ISO 时间：{bound_at!r}")
            continue
        if bound_dt.tzinfo is None:
            errors.append(f"{name}: {BINDING} 的 bound_at 缺时区：{bound_at!r}")
            continue
        stats["bound"] += 1

        declared = record.get("contract_files") or {}
        if not declared:
            errors.append(f"{name}: {BINDING} 没有 contract_files ⇒ 绑定没覆盖任何契约文件")
        changed = contract_diff(exp, declared)
        if changed:
            errors.append(f"{name}: 契约在绑定后被改过：{', '.join(changed)}"
                          f"（{BINDING} 记的是绑定时的 sha256；判据「先于读数」对旧绑定不再成立）")

        anchor = order_anchor(results)
        anchor_rel = rel(anchor)
        prior = record.get("readings_existing_at_bind") or {}
        if prior.get(anchor_rel) == sha256(anchor):
            # 登记过的状态：这批读数在绑定之前就已存在且未被重跑 ⇒ 它的定序不可判（与 unbound 同级）。
            # 不是失败：记录本身写明了这件事，但计数与打印要让它看得见。
            stats["predate"] += 1
            continue
        anchor_mtime = datetime.fromtimestamp(anchor.stat().st_mtime).astimezone()
        if bound_dt >= anchor_mtime:
            errors.append(f"{name}: bound_at {bound_at} 不早于判序锚 {anchor_rel} 的产物时间 "
                          f"{anchor_mtime.isoformat(timespec='seconds')} ⇒ 定序不成立")
        stats["ordered"] += 1

        binding_rel = rel(exp / BINDING)
        if uncommitted(exp / BINDING) or uncommitted(anchor):
            stats["git_pending"] += 1        # 未提交：仓内还没有定序，等提交时再看
            continue
        bind_commit = last_commit(["log", "-1", "--format=%cI", "--", binding_rel])
        anchor_commit = last_commit(["log", "-1", "--format=%cI", "--", anchor_rel])
        if not bind_commit or not anchor_commit:
            stats["git_pending"] += 1
            continue
        if bind_commit >= anchor_commit:
            stats["git_same_commit"] += 1
            errors.append(f"{name}: 绑定与读数在仓内没有先后：{BINDING} 的提交 "
                          f"{bind_commit} 不早于 {anchor_rel} 的提交 {anchor_commit}"
                          "（同一提交 ⇒「契约先于读数」仍不可判）")
        else:
            stats["git_ordered"] += 1
    return errors, stats


def summary_line(stats: dict) -> str:
    return (f"预登记绑定：{stats['experiments']} 个有读数的实验（{stats['results_files']} 份 results.json）"
            f"—— 已绑定 {stats['bound']}（其中读数晚于绑定 {stats['ordered']}、"
            f"读数早于绑定即定序不可判 {stats['predate']}）、"
            f"unbound（历史产物，只计数不判失败）{stats['unbound']}、缺绑定 {stats['missing']}；"
            f"仓内定序：分提交有先后 {stats['git_ordered']}、未提交 {stats['git_pending']}、"
            f"同一提交 {stats['git_same_commit']}")


def backfill_unbound(dry_run: bool = False) -> int:
    """历史产物：为有读数、无绑定记录的实验写 `bound_at: "unbound"`（不填看起来真的时间）。"""
    written = 0
    for exp, results in sorted(experiments().items()):
        if (exp / BINDING).exists():
            continue
        record = {"experiment": rel(exp), "bound_at": UNBOUND,
                  "unbound_reason": UNBOUND_REASON, "tool": TOOL}
        written += 1
        if dry_run:
            print(f"[dry-run] 将写 {rel(exp / BINDING)}")
            continue
        write_record(exp, record)
        print(f"已写 {rel(exp / BINDING)}（unbound，{len(results)} 份 results.json）")
    print(f"回填 {written} 条 unbound 记录（历史产物：不填时间，不伪装定序）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="预登记绑定：契约定序的可机检证据")
    ap.add_argument("experiment", nargs="?", help="实验目录（含 README.md / results.json 的那一层）")
    ap.add_argument("--require", action="store_true", help="只核绑定存在且契约未变（入口用）")
    ap.add_argument("--rebind", action="store_true", help="显式重绑：bound_at 改写为现在")
    ap.add_argument("--contract", nargs="+", metavar="文件", help="契约文件清单（缺省 README.md[+config.json]）")
    ap.add_argument("--backfill-unbound", action="store_true", help="为历史产物写 unbound 记录")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要写的记录")
    ap.add_argument("--check", action="store_true", help="全树机检")
    args = ap.parse_args(argv)

    if args.check:
        errors, stats = check()
        print(summary_line(stats))
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        if errors:
            print(f"FAIL：{len(errors)} 项", file=sys.stderr)
            return 1
        print("OK：预登记绑定全部可判。")
        return 0
    if args.backfill_unbound:
        return backfill_unbound(args.dry_run)
    if not args.experiment:
        ap.error("要给实验目录，或用 --check / --backfill-unbound")
    exp = experiment_dir(args.experiment)
    return require(exp) if args.require else bind(exp, args.contract, args.rebind)


if __name__ == "__main__":
    sys.exit(main())
