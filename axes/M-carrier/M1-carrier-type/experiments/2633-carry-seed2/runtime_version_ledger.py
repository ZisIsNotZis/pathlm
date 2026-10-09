#!/usr/bin/env python3
"""2633-carry-seed2 · **运行期版本台账**（复核 P1-3）：回答「每一臂到底跑的是哪一版代码」。

为什么需要：`equality_vs_2631` 的「实现层 equality」若只比文件哈希，会被**其它并行线的
未提交改动**打断；而 `engine_equivalence` 证明的是「两版在**本实验用到的配置上**行为等价」。
两者都不能回答「**我的 seed=2 三臂与复用的 seed0/1 六臂是否用同一版代码训练**」。本脚本把
这个问题的证据落成产物：

- **臂的启动时刻**：从本波训练总驱动的日志 `train_runner.log` 里解析
  （`=== [<arm>] 训练开始 <ISO 时间>`；该日志是三个实验共享的波级产物）；
- **漂移落地时刻**：工作树里各引擎文件与 **2631 运行时版本**不同时，取该文件的 **mtime**
  （最后一次修改时间）作为「漂移不早于/不晚于」的界；
- **基准**：`2631 运行时版本` = 按 2631 `results.json::code_hashes` 记录的 sha256 在 git 历史里
  定位到的那个 commit（`engine_equivalence.resolve_recorded_commit`）；**字节可变**（并行线的
  默认关闭加性改动），判据是行为等价；
- 逐臂分类：`start < mtime` ⇒ 该臂跑的是 `HEAD` 版；否则跑的是工作树版；
- 并用 `engine_equivalence.json::coverage` 标出**哪一层**的等价证据覆盖了「工作树版」的臂。

**边界（如实写出来）**：mtime 只给「最后一次修改」的时刻——若某文件在臂启动前已被**多次**
修改、且最后一次修改在臂启动之后，则「start < mtime」**不能**排除「臂跑的是更早的某个
工作树版本」。因此对本实验真正起作用的是 **`engine_equivalence` 的行为等价**：
只要两版行为逐位相同，「哪一版」就不影响结论。

输出 `runtime_version_ledger.json`。用法：`python3 runtime_version_ledger.py`
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
#: 臂启动时刻的日志源：本波总驱动 + 2631（复用臂）自己的驱动日志
ARM_LOGS = (
    os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                 "2634-carry-realtext-13m", "train_runner.log"),
    os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                 "2631-carry-reseed", "train_arms.runlog"),
)
EVID = os.path.join(HERE, "evidence", "drift_snapshot")
CODE_FILES = ("pathlm/depth_ar.py", "train_depth_ar.py",
              "tools/evalkit/longrange_tasks.py", "tools/evalkit/carry_channel.py",
              "tools/evalkit/retention.py")
ARM_START_RE = re.compile(r"=== \[(\S+)\] 训练开始 (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _head_blob(rel: str) -> str | None:
    r = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def _previous_ledger() -> dict:
    """上一版已入账的台账（`git show HEAD:<本文件同名产物>`）——用于取**历史观测到的 mtime**。

    为什么需要它：并行线把漂移**提交**之后，工作树的 mtime 变成 checkout 时间，
    「漂移什么时候落到盘上」这个事实只留在**当时落盘的台账**里。取不到就返回空字典。
    """
    rel = os.path.relpath(os.path.join(HERE, "runtime_version_ledger.json"), ROOT)
    out = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return {}
    try:
        return json.loads(out.stdout)
    except Exception:
        return {}


def _drift_landing_commits(rel: str, rt_commit: str) -> list[dict]:
    """`rt_commit..HEAD` 里改过 `rel` 的提交（正序）——漂移在 git 里的落地时刻。"""
    out = subprocess.run(["git", "-C", ROOT, "log", "--reverse", "--format=%cI %h %s",
                          f"{rt_commit}..HEAD", "--", rel],
                         capture_output=True, text=True).stdout.strip()
    rows = []
    for line in out.split("\n") if out else []:
        if not line.strip():
            continue
        ts, short, *subject = line.split(" ", 2)
        rows.append({"committed_at": ts, "commit": short,
                     "subject": (subject[0] if subject else "")[:80]})
    return rows


def _arm_start_times() -> dict[str, str]:
    """从全部日志源汇总臂启动时刻（后出现的源只补缺，不覆盖）。"""
    out: dict[str, str] = {}
    for p in ARM_LOGS:
        if not os.path.exists(p):
            continue
        txt = open(p, errors="ignore").read()
        for m in ARM_START_RE.finditer(txt):
            out.setdefault(m.group(1), m.group(2))
    return out


def main() -> int:
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    eq = json.load(open(os.path.join(HERE, "equality_report.json")))
    ee_path = os.path.join(HERE, "engine_equivalence.json")
    ee = json.load(open(ee_path)) if os.path.exists(ee_path) else {}
    starts = _arm_start_times()
    arms = [a["name"] for a in cfg["arms"]]

    import engine_equivalence as _ee
    rec = _ee._recorded_code_hashes()
    prev = _previous_ledger()          # 历史观测到的 mtime（git 提交后工作树 mtime 会被 checkout 覆盖）
    files = {}
    for rel in CODE_FILES:
        path = os.path.join(ROOT, rel)
        work = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else None
        located = _ee.resolve_recorded_commit(rel, rec[rel]) if rec.get(rel) else None
        rt = (located or {}).get("commit")
        prev_mtime = ((prev.get("code_files") or {}).get(rel) or {}).get(
            "working_tree_mtime")
        now_mtime = (time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
                     if os.path.exists(path) else None)
        commits = _drift_landing_commits(rel, rt) if rt else []
        first_commit_ts = commits[0]["committed_at"][:19] if commits else None
        cands = [c for c in (prev_mtime, first_commit_ts) if c]
        files[rel] = {
            "runtime_version_sha256": rec.get(rel),
            "runtime_version_commit": rt,
            "runtime_version_subject": (located or {}).get("commit_subject"),
            "working_tree_sha256": _sha(work) if work is not None else None,
            "working_tree_mtime_now": now_mtime,
            "working_tree_mtime_observed_earlier": prev_mtime,
            "drift_landed_no_later_than": (min(cands) if cands else None),
            "drift_commits_after_runtime_version": commits,
            "working_tree_equals_runtime_version": bool(
                work is not None and rec.get(rel) == _sha(work)),
            "drifted_now": bool(work is not None and rec.get(rel) != _sha(work)),
        }
    per_arm = {}
    for name in arms:
        t = starts.get(name)
        row = {"started_at": t, "files": {}}
        for rel, f in files.items():
            if not f["drifted_now"]:
                row["files"][rel] = {"version": "runtime_version(==working_tree)",
                                     "evidence": ("工作树字节 == 2631 运行时版本"
                                                  f"（{f['runtime_version_commit'][:8]}）")}
                continue
            if t is None:
                row["files"][rel] = {"version": "unknown(缺启动时刻)",
                                     "evidence": "train_runner.log 里没有该臂的启动行"}
                continue
            landing = f["drift_landed_no_later_than"]
            before = bool(landing) and t < landing
            row["files"][rel] = {
                "version": "runtime_version" if before else "working_tree_at_run_time",
                "evidence": (f"start {t} {'<' if before else '>='} 漂移落地上界 {landing}"
                             f"（取历史观测 mtime {f['working_tree_mtime_observed_earlier']} "
                             f"与首个漂移提交 {f['drift_commits_after_runtime_version'][:1]} 的较早者）；"
                             f"判据是行为等价（见 engine_equivalence，coverage={ee.get('coverage')}，"
                             f"基准=2631 运行时版本）"),
            }
        per_arm[name] = row
    drift_now = [r for r, f in files.items() if f["drifted_now"]]
    out = {
        "purpose": ("逐臂记录「跑的是 HEAD 版还是工作树版」，并把漂移快照归档；与 "
                    "engine_equivalence 的行为等价证据配套（复核 P1-3）"),
        "arm_log_sources": [os.path.relpath(p, ROOT) for p in ARM_LOGS
                            if os.path.exists(p)],
        "code_files": files,
        "drifted_at_evidence_time": drift_now,
        "per_arm": per_arm,
        "engine_equivalence_coverage": ee.get("coverage"),
        "engine_equivalence_all_layers_ok": ee.get("all_layers_ok"),
        "comparison_base": "2631 运行时版本（按 2631/results.json::code_hashes 定位）",
        "residual_uncertainty": ("mtime 只给最后一次修改时刻 ⇒ 「start < mtime」不能排除"
                                 "「臂跑的是更早的某个工作树版本」；本实验靠 "
                                 "`engine_equivalence` 的**行为等价**（前向逐位 logits + "
                                 "训练脚本同一 run 的日志点逐字段 loss 序列）覆盖这一不确定性"),
        "verdict": ("all_arms_behaviorally_equivalent"
                    if ee.get("all_layers_ok") else
                    "pathlm_only__trainer_layer_unverified"),
    }
    os.makedirs(EVID, exist_ok=True)
    diff = subprocess.run(["git", "-C", ROOT, "diff", "--"] + list(CODE_FILES),
                          capture_output=True, text=True).stdout
    stamp = time.strftime("%Y%m%dT%H%M%S")
    diff_path = os.path.join(EVID, f"drift_{stamp}.diff")
    with open(diff_path, "w") as f:
        f.write(diff)
    out["drift_snapshot"] = {
        "diff_path": os.path.relpath(diff_path, ROOT),
        "sha256": _sha(diff),
        "n_bytes": len(diff),
        "note": ("工作树相对 HEAD 的 diff（证据时刻快照；含**其它并行线**的未提交改动），"
                 "用于回溯「当时的漂移是什么」"),
    }
    with open(os.path.join(HERE, "runtime_version_ledger.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("drifted at evidence time:", drift_now or "无")
    for name in arms:
        v = sorted({list(x.values())[0] for x in per_arm[name]["files"].values()
                    if isinstance(x, dict) and "version" in x})
        print(f"  {name}: {v}")
    print("coverage:", out["engine_equivalence_coverage"], "| verdict:", out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
