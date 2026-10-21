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
    python3 tools/prereg_bind.py <实验目录> --contract README.md run.py   # 契约文件清单（默认 README.md[+config.json]）
    python3 tools/prereg_bind.py --backfill-unbound     # 历史产物：写 bound_at="unbound"（显式缺失，不伪造时间）
    python3 tools/prereg_bind.py --check                # 全树机检（`check_tree.py` 用的同一实现）

契约文件清单要覆盖**判据本身落在哪**：判据线写在 `README.md` 只约束文字；`BAND`、配对规则这类
**应用判据的代码**若写在实验自己的 `run.py` 里，`--contract README.md run.py` 才把它一起绑住
（或者把它移进 `config.json`，反正默认清单会自动纳入）。只绑 `README.md` 时，`run.py` 事后改判据线
不会让绑定失效——sha256 变了才叫「契约被改过」。

`bound_at` 必须落在**绑定文件入库提交**的时间窗内（`BIND_WINDOW_MINUTES`）：只有文件 mtime 与提交
先后两道证据时，「先跑读数 → 事后补一条带旧 `bound_at` 的绑定 → 再重跑读数」能凑出合法的先后差，
时间窗把这条路堵住（绑定完就提交，别隔很久）。

绑定记录 `prereg_binding.json`（实验根目录，契约文件旁边）：

| 键 | 值 |
|---|---|
| `experiment` | 实验目录（相对仓库根） |
| `bound_at` | ISO 时间（带时区），或 `"unbound"` = 历史产物、无定序证据 |
| `unbound_reason` | 仅 `unbound` 时出现：为什么没有定序证据 |
| `contract_files` | 契约文件名 → 绑定时的 `sha256`（`unbound` 时不存在：没有「绑定时」可言） |
| `readings_existing_at_bind` | 绑定那一刻已存在的读数（路径 → 当时的 `sha256`，`results*.json` 全覆盖）：它们是**绑定之前**的读数，定序不可判；判定时要求读数被重跑（`sha256` 变了）且产物时间晚于 `bound_at` |
| `tool` | 产生这条记录的工具 |

机检（`check()`）：**有读数的实验必须有绑定记录**；契约文件当前 `sha256` 必须与绑定时的相同
（不同 ⇒ 报「契约在绑定后被改过」并列出差异文件）；**每一份读数都要过定序与代码哈希两道**——
读数不止实验根的聚合 `results.json`，还有臂级 / 子目录的 `results_<臂>_<种子>.json`
（`READINGS_GLOB`）；对每份读数，其 `bound_at` 必须**严格早于**该读数的产物时间（绑定时已存在
且逐字节未变的读数记 `predate`：定序不可判，不冒充「先于」），且它记录的代码哈希必须等于
`contract_files` 里对应文件的绑定值（读数没记任何代码哈希、或记的哈希无一被绑定覆盖，都判红——
这就是「臂级读数在绑定之前产生、用另一个版本的 `run.py` 跑出来」那条漏检）；两份文件都已入库时，
绑定记录的提交必须**早于**判序锚（实验根 `results.json`；只有嵌套读数时取最新一份）的提交
（同一提交 ⇒ 仓内仍无独立定序证据，判失败）；`bound_at` 必须落在绑定文件入库提交时间的
`BIND_WINDOW_MINUTES` 窗内（超窗 ⇒ 这条 `bound_at` 不是绑定那次提交写下的，仓内先后能凑但不能成立）。

读数里记代码哈希的字段名各入口不同（`code_hash_run_py` / `code.<文件>` / `code_hashes.<文件>` /
`code_hashes_by_arm.<臂>`），适配只发生在 `reading_code_hashes()` 一处：新入口照那张表写。

已声明存量基线 `harness/prereg_debt.tsv`：口径一扩就会在存量上亮一堆红（本工具开新口径时是 81 条），
而一条对存量永远失败的闸门等于永远开不了（`@migrate_on_touch`）。所以存量逐条声明在那个文件里，
每条带**路径 / 类别 / 签名 / owner / 原因**：基线内且签名未变 ⇒ 计「已声明存量」不判失败；
**基线外的、或文件变了而问题仍在的 ⇒ 红**。基线只减不增（`tests/test_prereg_bind.py` 卡条数上限），
修好一条就删一条（汇总行会点名「已不再违规、该删」的条目）。

`"unbound"` 是**显式缺失**：`--check` 只计数、不判失败，并把它们的实验名打进汇总行（读者要看得出
**哪几个**实验定序不可判）。它不得用来给新实验免检——新实验一律先绑定再训练，两道口都会拦：
`--require`（`train_arms.sh` / `run.py` / 共享训练器 `train_depth_ar.py`）与 `--backfill-unbound`
（只接受入库提交早于本工具引入提交的读数；更晚的、或没入库因而无从定序的，一律拒绝回填并非零退出）。
同一实验目录里追加第二批读数时，绑定记下当时已有的读数（`readings_existing_at_bind`），因此给旧读数
「补绑」不会伪装成定序证据：旧读数要重新产出才过门。

训练器的门在这里而不只在各实验自己的入口：`train_depth_ar.py` 本身就是产物写入点
（`<out-root>/<run>/results.json`），直跑它能绕过实验入口的两道门。`owner_for_output()` 判定一个
out-root 是否归属 `axes/**` 的某个实验目录（含 `README.md` 的那一层）；`.scratch/` / `/tmp/` 下的
out-root 不归属任何实验，不受此门约束。

诚实边界：这些判据看的是**字节与提交时间**，不读训练日志。读数被手工改过时 `sha256` 也会变，
机制把它读成「重跑过」——它证明的是字节变了，而不是真跑了一遍。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AXES = ROOT / "axes"
BINDING = "prereg_binding.json"
RESULTS = "results.json"
#: 一个实验的**全部**读数：聚合的 `results.json` 与臂级 `results_<臂>[_<种子>].json`。
#: 定序与代码哈希对每一份都成立——只看聚合文件会给臂级读数留后门（臂级读数可以先于绑定产生、
#: 甚至由另一个版本的 `run.py` 跑出来，而 `results.json` 仍然全绿）。
READINGS_GLOB = "results*.json"
#: 契约文件：`README.md` 必须在；`config.json` 存在即纳入（它也是判据的预登记面）。
CONTRACT_REQUIRED = "README.md"
CONTRACT_OPTIONAL = ("config.json",)
#: `bound_at` 的显式缺失值（历史产物）。
UNBOUND = "unbound"
#: `bound_at` 与「绑定文件入库提交时间」允许的最大偏差（绑定完就提交，超窗即 `bound_at` 是回填的旧时间）。
BIND_WINDOW_MINUTES = 10
UNBOUND_REASON = "历史产物，无定序证据"
TOOL = "tools/prereg_bind.py"
#: 读数里记代码哈希的字段（适配表：各实验入口写得不一样，读法只在 `reading_code_hashes()` 一处）。
HASH_FIELDS = ("code_hash_run_py", "code", "code_hashes", "code_hashes_by_arm")
#: sha256 的字面形状：只认它，不把 `code` 里的备注当成哈希。
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: 跨实验借用同名文件的标注后缀：`realtext_probe.py(2662)` = 2662 那份。
BORROWED = re.compile(r"\(\d+\)$")
#: 已声明存量的基线：一个文件一行（`路径\t类别\t签名\towner\t原因`）。
#: 它把「新口径下已存在的违规」从「永久红」改成「已声明存量」，从而让本检**今天就能当门用**；
#: 基线**只减不增**（新条目要上层批准，`tests/test_prereg_bind.py` 卡住条数上限），
#: 且锚的是**当时那份文件的签名**：文件变了而问题仍在 ⇒ 红（防“签一次就长期免检”）。
BASELINE = "harness/prereg_debt.tsv"
BASELINE_COLUMNS = ("路径", "类别", "签名", "owner", "原因")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def violation(path: str, kind: str, signature: str, message: str) -> dict:
    """一条违规：`(路径, 类别)` 是基线的键，`签名` 声明时那份文件的内容（变了就要重新判），
    `message` 是给人看的那句（总是点名到文件）。"""
    return {"path": path, "kind": kind, "signature": signature, "message": message}


def file_signature(path: Path) -> str:
    """基线的签名：文件内容的 sha256 前 16 位（够分辨，又不至于每行都长到读不进眼）。"""
    return sha256(path)[:16] if path.exists() else "missing"


def load_baseline() -> dict[tuple[str, str], dict]:
    """读已声明存量基线：`{(路径, 类别): {签名, owner, 原因}}`。

    格式：一个文件一行，五列制表符分隔（`路径 类别 签名 owner 原因`）；`#` 开头与空行是注释。
    列数不对、原因 / owner 为空都直接报错：一条没有理由的免检就是一个没人认领的漏洞。
    """
    p = ROOT / BASELINE
    if not p.exists():
        return {}
    rows: dict[tuple[str, str], dict] = {}
    for no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != len(BASELINE_COLUMNS):
            raise SystemExit(f"{BASELINE}:{no}: 应该是 {len(BASELINE_COLUMNS)} 列"
                             f"（{', '.join(BASELINE_COLUMNS)}），实际 {len(cols)} 列")
        path, kind, signature, owner, reason = (c.strip() for c in cols)
        if not (path and kind and signature and owner and reason):
            raise SystemExit(f"{BASELINE}:{no}: 每列都要有值（一条没有理由的免检就是一个漏洞）")
        rows[(path, kind)] = {"signature": signature, "owner": owner, "reason": reason}
    return rows


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

    prior = {rel(p): sha256(p) for p in sorted(exp.glob(f"**/{READINGS_GLOB}"))}
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
    """有读数的实验 → 它的全部读数（`READINGS_GLOB`：聚合 `results.json` + 臂级 `results_*.json`）。

    一份读数所属的**实验** = 它向上最近的、含 `README.md`（契约）的目录：
    臂级 / 冒烟读数（`runs/<arm>/results.json`、`results_<臂>_<种子>.json`）归属同一个契约，
    不各自单独绑定。
    """
    owners: dict[Path, list[Path]] = {}
    for p in sorted(AXES.glob(f"**/{READINGS_GLOB}")):
        owners.setdefault(owner_of(p), []).append(p)
    return owners


def owner_of(p: Path) -> Path:
    """读数 → 它所属的实验目录（向上最近的含契约的目录）。"""
    cur = p.parent
    while cur != AXES.resolve() and AXES.resolve() in cur.parents:
        if (cur / CONTRACT_REQUIRED).exists():
            return cur
        cur = cur.parent
    return p.parent                               # 没有契约目录可归（不该发生）：按自身目录计


def owner_for_output(out_dir: str | Path) -> Path | None:
    """产物目录（`<out-root>/<run>`）→ 它所属的实验目录；不属于任何实验 ⇒ None。

    训练器写产物前用它定门：out-root 落在 `axes/**` 的某个实验目录下（向上最近的含 `README.md`
    的那一层）⇒ 该实验的契约必须先绑定。`.scratch/`、`/tmp/` 这类 out-root 不属于任何实验。
    """
    axes = AXES.resolve()
    start = Path(out_dir)
    if not start.is_absolute():
        start = Path.cwd() / start
    start = start.resolve()
    for d in (start, *start.parents):
        if not d.is_relative_to(axes):
            return None                           # 走出 axes/ 就再没有实验目录可归属
        if (d / CONTRACT_REQUIRED).exists():
            return d
    return None


def code_path(exp: Path, name: str) -> str | None:
    """读数里记的文件名 → 它在仓库里的相对路径；磁盘上找不到就返回 None。

    读数写路径的方式不统一：`run.py`（相对实验目录）、`axes/.../run.py`（仓库相对）、
    `run_py_sha256`（变量名字式的 `run.py`）、`realtext_probe.py(2662)`（跨实验借的同名文件）。
    解析结果与 `contract_files` 的路径能直接对齐，比对才不靠字符串猜。
    """
    name = BORROWED.sub("", name)
    if name.endswith("_py_sha256"):
        name = name[: -len("_py_sha256")] + ".py"
    for cand in (exp / name, ROOT / name):
        if cand.is_file():
            return rel(cand)
    return None


def reading_code_hashes(exp: Path, reading: dict
                        ) -> tuple[dict[str, list[tuple[str, str]]], list[str]]:
    """一份读数记录的代码哈希：`({仓库相对路径 → [(记它的字段, sha256), …]}, 落不到本仓文件的字段)`。

    字段适配只在这一处（各实验入口写得不一样，新入口照这张表写，读法不散在调用点）：
    - `code_hash_run_py`：单份字符串，产出该读数的 `run.py`
    - `code` / `code_hashes`：{文件路径 或 `<名>_py_sha256` → sha256}
    - `code_hashes_by_arm`：{臂 → 产出该臂那一份的 sha256}（逐臂版本不同的读数靠它记；
      键是臂名不是文件名，所以它一律指本实验的 `run.py`）

    同一文件可以有多份（如逐臂记录的旧版本）：一份不符就要报，不能只留最后一条。
    第二项是「记了哈希、但它指的文件在本仓找不到」的字段名：那是另一回事（绑定同样盖不住它），
    不能报成「没记代码哈希」。
    """
    found: dict[str, list[tuple[str, str]]] = {}
    unplaced: list[str] = []

    def add(path: str | None, field: str, sha: str) -> None:
        if path is None:
            unplaced.append(f"{field}→{sha[:12]}…")
        else:
            found.setdefault(path, []).append((field, sha))

    for field in HASH_FIELDS:
        value = reading.get(field)
        if isinstance(value, str):             # `code_hash_run_py`：本实验 run.py 的 sha256
            if SHA256_RE.match(value):
                add(code_path(exp, "run.py"), field, value)
        elif isinstance(value, dict):
            for key, sha in value.items():
                if not isinstance(sha, str) or not SHA256_RE.match(sha):
                    continue
                name = "run.py" if field == "code_hashes_by_arm" else str(key)
                add(code_path(exp, name), f"{field}.{key}", sha)
    return found, unplaced


def bound_code_hashes(exp: Path, contract: dict) -> dict[str, str]:
    """绑定的契约文件 → {仓库相对路径: 绑定时的 sha256}（磁盘上已不在的文件略过）。"""
    bound: dict[str, str] = {}
    for name, sha in contract.items():
        p = code_path(exp, name)
        if p is not None:
            bound[p] = sha
    return bound


def order_anchor(results: list[Path]) -> Path:
    """仓内定序的锚：实验根 `results.json`；只有嵌套读数时取其中最新的一份。

    它只管「绑定与读数的**提交**有没有先后」（聚合文件是实验的主读数）；每份读数自己的
    产物时间与代码哈希由 `reading_violations()` 逐份判定。
    """
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


def introduced_at() -> str | None:
    """引入本工具的入库提交时间（「哪些读数早于机制」的时间原点）；未入库 / 无 git ⇒ None。"""
    return last_commit(["log", "--diff-filter=A", "-1", "--format=%cI", "--", TOOL])


def reading_violations(path: Path, exp: Path, bound_at: str, bound_dt: datetime,
                       bound_hashes: dict[str, str], prior: dict, stats: dict) -> list[dict]:
    """一份读数的两道：它记的代码哈希必须等于绑定值；它的产物时间必须晚于 `bound_at`。"""
    r = rel(path)
    sig = file_signature(path)
    try:
        reading = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        stats["unhashed"] += 1
        return [violation(r, "no-hash", sig, f"{r}: 读数读不出 JSON（{e}）⇒ 产出它的代码无从核对")]
    if not isinstance(reading, dict):
        stats["unhashed"] += 1
        return [violation(r, "no-hash", sig, f"{r}: 读数不是 JSON 对象 ⇒ 产出它的代码无从核对")]

    out: list[dict] = []
    recorded, unplaced = reading_code_hashes(exp, reading)
    if not recorded and not unplaced:
        stats["unhashed"] += 1
        fields = " / ".join(f"`{f}`" for f in HASH_FIELDS)
        out.append(violation(r, "no-hash", sig,
                             f"{r}: 该读数未记录代码哈希（{fields} 都没有）⇒ 产出它的代码无从核对"))
    covered = [p for p in sorted(recorded) if p in bound_hashes]
    for code in covered:
        want = bound_hashes[code]
        for field, got in recorded[code]:
            if got == want:
                continue
            stats["mismatch"] += 1
            out.append(violation(
                r, "hash-mismatch", sig,
                f"{r}: 读数里 {field} 记的 {code} 哈希 {got[:12]}… 与绑定值 {want[:12]}… 不符"
                " ⇒ 产出这份读数的代码不是绑定那一刻的代码"
                "（代码在绑定后被改过，或这份读数早于当前绑定）"))
    if recorded and not covered:
        stats["uncovered"] += 1
        out.append(violation(
            r, "uncovered", sig,
            f"{r}: 读数记的代码哈希（{', '.join(sorted(recorded))}）无一被 {BINDING} 的 "
            "contract_files 覆盖 ⇒ 产出它的代码与契约没有可比对的对象"
            "（重绑时用 --contract 把该文件一起绑住并重跑读数）"))
    if unplaced and not covered:
        stats["uncovered"] += 1
        out.append(violation(
            r, "uncovered", sig,
            f"{r}: 读数记的代码哈希指不到本仓的任何一个文件（{', '.join(unplaced)}）⇒ 产出它的代码"
            "无从核对（文件名写错 / 文件已不在；改成真的路径，或重绑时一起绑住）"))
    if prior.get(r) == sha256(path):
        # 登记过的状态：这份读数在绑定之前就已存在且未被重跑 ⇒ 定序不可判（与 unbound 同级）。
        # 不是失败：记录本身写明了这件事，但计数与打印要让它看得见。
        stats["predate"] += 1
        stats["predate_names"].append(r)
        return out
    mtime = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
    if bound_dt >= mtime:
        out.append(violation(
            r, "order", sig,
            f"{r}: bound_at {bound_at} 不早于该读数的产物时间 "
            f"{mtime.isoformat(timespec='seconds')} ⇒ 定序不成立"))
    else:
        stats["ordered"] += 1
    return out


def _bad_bound_at(bound_at) -> str | None:
    """`bound_at` 的形态问题（不是 ISO 时间 / 缺时区）⇒ 一句人话；没问题 ⇒ None。"""
    if not isinstance(bound_at, str):
        return f"bound_at 不是 ISO 时间也不是 {UNBOUND!r}"
    try:
        parsed = datetime.fromisoformat(bound_at)
    except ValueError:
        return f"bound_at 不是 ISO 时间：{bound_at!r}"
    return None if parsed.tzinfo is not None else f"bound_at 缺时区：{bound_at!r}"


def collect() -> tuple[list[dict], dict]:
    """全树机检的**违规收集**：`(违规列表, stats)`；违规按 `(路径, 类别, 签名)` 结构化。

    分流交给 `apply_baseline()`：基线条目且在声明的签名上未变 ⇒ 只计数；
    基线外的、或文件已变而问题仍在 ⇒ 红。
    """
    found: list[dict] = []
    stats = {"experiments": 0, "readings": 0, "bound": 0, "unbound": 0,
             "unbound_names": [], "missing": 0, "ordered": 0, "predate": 0,
             "predate_names": [], "unhashed": 0, "mismatch": 0, "uncovered": 0,
             "declared": 0, "declared_stale": [], "debt": 0,
             "git_ordered": 0, "git_pending": 0, "git_same_commit": 0}
    for exp, readings in sorted(experiments().items()):
        stats["experiments"] += 1
        stats["readings"] += len(readings)
        name = rel(exp)
        bind_sig = file_signature(exp / BINDING)
        record = load(exp)
        if record is None:
            stats["missing"] += 1
            found.append(violation(name, "missing-binding", bind_sig,
                                   f"{name}: 有读数但缺 {BINDING} ⇒ 预登记未绑定"
                                   f"（python3 {TOOL} {name}，历史产物用 --backfill-unbound）"))
            continue
        bound_at = record.get("bound_at")
        if bound_at == UNBOUND:
            stats["unbound"] += 1            # 显式缺失：只计数，不判失败
            stats["unbound_names"].append(name)   # 读者要看得见是哪几个实验定序不可判
            continue
        bad = _bad_bound_at(bound_at)
        if bad is not None:
            found.append(violation(name, "bound-at", bind_sig, f"{name}: {bad}"))
            continue
        bound_dt = datetime.fromisoformat(bound_at)
        stats["bound"] += 1

        declared = record.get("contract_files") or {}
        if not declared:
            found.append(violation(name, "no-contract", bind_sig,
                                   f"{name}: {BINDING} 没有 contract_files ⇒ 绑定没覆盖任何契约文件"))
        changed = contract_diff(exp, declared)
        if changed:
            found.append(violation(name, "contract-edited", bind_sig,
                                   f"{name}: 契约在绑定后被改过：{', '.join(changed)}"
                                   f"（{BINDING} 记的是绑定时的 sha256；判据「先于读数」对旧绑定不再成立）"))

        # 定序与代码哈希对**每一份**读数都成立：臂级 / 子目录的读数一样能绕过绑定。
        prior = record.get("readings_existing_at_bind") or {}
        bound_hashes = bound_code_hashes(exp, declared)
        for reading in readings:
            found.extend(reading_violations(reading, exp, bound_at, bound_dt,
                                            bound_hashes, prior, stats))

        anchor = order_anchor(readings)
        anchor_rel = rel(anchor)
        binding_rel = rel(exp / BINDING)
        if uncommitted(exp / BINDING) or uncommitted(anchor):
            stats["git_pending"] += 1        # 未提交：仓内还没有定序，等提交时再看
            continue
        bind_commit = last_commit(["log", "-1", "--format=%cI", "--", binding_rel])
        anchor_commit = last_commit(["log", "-1", "--format=%cI", "--", anchor_rel])
        if not bind_commit or not anchor_commit:
            stats["git_pending"] += 1
            continue
        # bound_at 必须落在「绑定文件入库提交」的时间窗内：否则「先跑读数 → 事后补一条带旧
        # bound_at 的绑定 → 再重跑读数」能凑出合法的先后差（文件 mtime 与提交先后都对得上，
        # 只有 bound_at 是回填的旧时间）。
        skew = abs(bound_dt - datetime.fromisoformat(bind_commit))
        if skew > timedelta(minutes=BIND_WINDOW_MINUTES):
            found.append(violation(name, "bind-window", bind_sig,
                                   f"{name}: bound_at {bound_at} 不在 {BINDING} 的提交 {bind_commit} 的 "
                                   f"±{BIND_WINDOW_MINUTES} 分钟窗内（差 {skew}）⇒ 这条 bound_at 不是绑定那次"
                                   "提交写下的；先跑读数、事后补一条带旧 bound_at 的绑定、再重跑读数，能凑出"
                                   "合法的先后差，时间窗把它堵住"))
        if bind_commit >= anchor_commit:
            stats["git_same_commit"] += 1
            found.append(violation(name, "same-commit", bind_sig,
                                   f"{name}: 绑定与读数在仓内没有先后：{BINDING} 的提交 "
                                   f"{bind_commit} 不早于 {anchor_rel} 的提交 {anchor_commit}"
                                   "（同一提交 ⇒「契约先于读数」仍不可判）"))
        else:
            stats["git_ordered"] += 1

    return found, stats


def check() -> tuple[list[str], dict]:
    """全树机检：收集违规，再用 `harness/prereg_debt.tsv` 分流成「红」与「已声明存量」。"""
    found, stats = collect()
    errors, stock = apply_baseline(found)
    stats["declared"] = len(stock)
    stats["declared_stale"] = stale_entries(found)
    stats["debt"] = len(load_baseline())
    return errors, stats


def apply_baseline(found: list[dict]) -> tuple[list[str], list[tuple[str, str]]]:
    """把违规分成「红」与「已声明存量」：基线内且文件签名未变 ⇒ 存量；否则红。"""
    debt = load_baseline()
    errors: list[str] = []
    stock: list[tuple[str, str]] = []
    for v in found:
        key = (v["path"], v["kind"])
        row = debt.get(key)
        if row is None:
            errors.append(v["message"])
        elif row["signature"] != v["signature"]:
            stock_sign = row["signature"]
            errors.append(f"{v['message']}（基线条目 {BASELINE} 声明时签名 {stock_sign}，"
                          f"现在是 {v['signature']}：文件已变而问题仍在 ⇒ 重跑读数或修好，"
                          "再删掉基线条目；基线只减不增）")
        else:
            stock.append(key)
    return errors, stock


def stale_entries(found: list[dict]) -> list[tuple[str, str]]:
    """基线里已不再违规的条目（修好了就该删；留着不判失败，但汇总行会点名）。"""
    hit = {(v["path"], v["kind"]) for v in found}
    return sorted(k for k in load_baseline() if k not in hit)


def summary_line(stats: dict) -> str:
    names = stats.get("unbound_names") or []
    listed = f"：{'、'.join(names)}" if names else ""
    predate = stats.get("predate_names") or []
    shown = "、".join(predate[:20]) + (f" … 共 {len(predate)} 份" if len(predate) > 20 else "")
    predate_listed = f"（{shown}）" if predate else ""
    stale = stats.get("declared_stale") or []
    stale_shown = "、".join(f"{p}（{k}）" for p, k in stale[:10])
    stale_listed = (f"；基线条目已不再违规、该删 {len(stale)} 条（{stale_shown}"
                    f"{' …' if len(stale) > 10 else ''}）") if stale else ""
    return (f"预登记绑定：{stats['experiments']} 个有读数的实验（{stats['readings']} 份 "
            f"{READINGS_GLOB}，含臂级读数）"
            f"—— 已绑定 {stats['bound']}（其中读数晚于绑定 {stats['ordered']}、"
            f"绑定时已存在即定序不可判 {stats['predate']}{predate_listed}）、"
            f"unbound（历史产物，只计数不判失败，定序不可判）{stats['unbound']}{listed}、"
            f"缺绑定 {stats['missing']}；"
            f"代码哈希：未记录 {stats['unhashed']}、与绑定不符 {stats['mismatch']}、"
            f"无一被绑定覆盖 {stats['uncovered']}；"
            f"已声明存量 {stats['declared']}/{stats['debt']} 条（{BASELINE}，只减不增）{stale_listed}；"
            f"仓内定序：分提交有先后 {stats['git_ordered']}、未提交 {stats['git_pending']}、"
            f"同一提交 {stats['git_same_commit']}")


def backfill_unbound(dry_run: bool = False) -> int:
    """历史产物：为有读数、无绑定记录的实验写 `bound_at: "unbound"`（不填看起来真的时间）。

    只接受**入库提交早于本工具引入提交**的读数：晚于它的说明这实验是本轮新做的，它必须走
    「先绑定再训练」而不是被洗成「历史产物」。读数未入库 / 无 git ⇒ 定序不可判，同样拒绝：
    它没被更新，只是没证据（先回填再补读数，就是绕过「必须有序」的那条路）。
    """
    intro = introduced_at()
    if intro is None:
        print(f"拒绝回填：取不到 {TOOL} 的引入提交时间（未入库 / 无 git）⇒ "
              "无法判定读数是否早于机制引入", file=sys.stderr)
        return 1
    written = refused = 0
    for exp, readings in sorted(experiments().items()):
        if (exp / BINDING).exists():
            continue
        late = []
        for r in readings:
            commit = last_commit(["log", "-1", "--format=%cI", "--", rel(r)])
            if commit is None or commit >= intro:
                late.append((rel(r), commit))
        if late:
            refused += 1
            print(f"拒绝回填 {rel(exp)}：{len(late)} 份读数不能证明早于 {TOOL} 的引入提交 "
                  f"{intro} ⇒ 它不是历史产物，先绑定再训练（python3 {TOOL} {rel(exp)}）",
                  file=sys.stderr)
            for path, commit in late:
                print(f"  - {path}（入库提交 {commit or '未入库'}）", file=sys.stderr)
            continue
        record = {"experiment": rel(exp), "bound_at": UNBOUND,
                  "unbound_reason": UNBOUND_REASON, "tool": TOOL}
        written += 1
        if dry_run:
            print(f"[dry-run] 将写 {rel(exp / BINDING)}")
            continue
        write_record(exp, record)
        print(f"已写 {rel(exp / BINDING)}（unbound，{len(readings)} 份读数）")
    print(f"回填 {written} 条 unbound 记录（历史产物：不填时间，不伪装定序），拒绝 {refused} 条")
    return 1 if refused else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="预登记绑定：契约定序的可机检证据")
    ap.add_argument("experiment", nargs="?", help="实验目录（含 README.md 的那一层，读数在它下面）")
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
