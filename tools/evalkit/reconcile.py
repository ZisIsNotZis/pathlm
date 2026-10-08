"""对账脚本的共享骨架（F4 / 复核 B3）。

**问题（B3）**：各实验的 `reconcile_numbers.py` 把容差写成一个含糊的 `tol`，并按
`tol · max(1, |want|)` 使用——对 FLOPs 量级（`want ≈ 1.4e7`）等于**放行 100% 误差**；
`2621-depth-draft-e2e` 的 `tol=1.0` 就落在这个坑里。另一半问题是**期望值的来源**：
把历史记录值（golden 常数）当唯一证据，等于「报告与报告自己对账」。

**两条硬规则（本模块强制）**：

1. **容差必须显式且分开写**：`abs_tol`（绝对）+ `rel_tol`（相对 `|want|`），
   `允许误差 = abs_tol + rel_tol·|want|`；`check(...)` 两者都缺省 ⇒ `ValueError`。
2. **期望必须可溯**：
   - 由 raw 计数/日志**重算** ⇒ `derived(...)` / `check(..., derived_from=(...))`；
   - 确实是历史记录常数 ⇒ `golden(..., why=...)`，并在 `finish()` 的报告里计数。
   `finish()` 打印 `derived / golden` 两类计数，golden 缺 `why` 直接失败。

用法：

    ledger = Ledger("2621-depth-draft-e2e", sources=("results.json", "cycle_log.jsonl"))
    ledger.derived("acceptance_pooled == Σaccepted/Σdrafts", a["acceptance_pooled"],
                   lambda: a["sum_n_accepted"] / a["sum_n_drafts"],
                   sources=("results.json::arms.*.sum_n_accepted",), abs_tol=2e-4)
    ledger.eq("gamma == 4", arms["single_g4"]["config"]["gamma"], 4)
    ledger.golden("proxy tf_enwik8@4 == 0.9694", tf4, 0.9694, abs_tol=0.02,
                  why="23c-H2-w0-12k 的历史探针记录值")
    sys.exit(ledger.finish(os.path.join(HERE, "reconcile_report.json")))
"""

from __future__ import annotations

import json


class GoldenWithoutWhy(ValueError):
    """golden 类期望没写来源说明——不允许。"""


class MissingTolerance(ValueError):
    """既没给 `abs_tol` 也没给 `rel_tol`——不允许（旧 `tol` 语义已废止）。"""


class MissingProvenance(ValueError):
    """期望既不是重算的，也没声明来自哪些 raw 数据。"""


class Ledger:
    def __init__(self, title: str, sources=()):
        self.title = title
        self.sources = tuple(sources)
        self.checks: list[dict] = []
        self.fails: list[str] = []

    # ---------------------------------------------------------------- 内部
    @staticmethod
    def _tol(abs_tol, rel_tol, want) -> float:
        if abs_tol is None and rel_tol is None:
            raise MissingTolerance(
                "对账容差必须显式：给 abs_tol（绝对）或 rel_tol（相对 |want|）——"
                "旧的 `tol`（= tol·max(1,|want|)）语义已废止")
        return (abs_tol or 0.0) + (rel_tol or 0.0) * abs(want)

    def _record(self, name, got, want, ok, kind, tol, why, sources):
        row = {"name": name, "got": got, "want": want, "ok": bool(ok),
               "kind": kind, "tol": tol, "why": why, "sources": list(sources)}
        self.checks.append(row)
        if not ok:
            self.fails.append(name)
        print(f"[{'PASS' if ok else 'FAIL'}] ({kind}) {name}: got={got} want={want} tol={tol:g}")
        return ok

    # ---------------------------------------------------------------- API
    def check(self, name, got, want, *, abs_tol=None, rel_tol=None,
              derived_from=(), why=""):
        """数值核对：`got` 与 `want` 的差必须 ≤ `abs_tol + rel_tol·|want|`。

        `derived_from` / `why` 至少给一个（`derived_from` 声明期望由哪些 raw 字段重算，
        `why` 用于历史记录常数）。两者都空且 `self.sources` 也空 ⇒ `MissingProvenance`。
        """
        if not derived_from and not why and not self.sources:
            raise MissingProvenance(f"{name}: 期望值没有来源（既非重算、也无 why）")
        tol = self._tol(abs_tol, rel_tol, want)
        ok = abs(float(got) - float(want)) <= tol
        kind = "golden" if (why and not derived_from) else "derived"
        return self._record(name, got, want, ok, kind, tol, why,
                            derived_from or self.sources)

    def eq(self, name, got, want, *, derived_from=(), why=""):
        """精确相等（整数计数、字符串、布尔）。"""
        return self.check(name, got, want, abs_tol=0.0, rel_tol=0.0,
                          derived_from=derived_from, why=why)

    def derived(self, name, got, compute, *, sources, abs_tol=None, rel_tol=None):
        """期望值由 `compute()` 从 raw 数据**重算**（`sources` 必填）。"""
        if not sources:
            raise MissingProvenance(f"{name}: derived 必须声明 sources（raw 计数/日志）")
        return self.check(name, got, compute(), abs_tol=abs_tol, rel_tol=rel_tol,
                          derived_from=sources)

    def golden(self, name, got, want, *, why: str = "", abs_tol=None, rel_tol=None):
        """历史记录常数（golden）核对——必须写清 `why`。"""
        if not why:
            raise GoldenWithoutWhy(f"{name}: golden 期望必须写 why（来源说明）")
        return self.check(name, got, want, abs_tol=abs_tol, rel_tol=rel_tol, why=why)

    def summary(self, include_checks: bool = False) -> dict:
        """落盘用的小结：计数 + 失败清单（**不含**全部 checks，避免报告膨胀；
        逐条 PASS/FAIL 仍打印到 stdout）。"""
        n_golden = sum(1 for c in self.checks if c["kind"] == "golden")
        out = {"title": self.title, "sources": list(self.sources),
               "n_checks": len(self.checks), "n_failed": len(self.fails),
               "n_derived": len(self.checks) - n_golden, "n_golden": n_golden,
               "failed": self.fails}
        if include_checks:
            out["checks"] = self.checks
        return out

    def finish(self, out_path: str | None = None) -> int:
        rep = self.summary()
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=1, ensure_ascii=False)
        print(f"\n=== {self.title}: {rep['n_checks'] - rep['n_failed']}/{rep['n_checks']} "
              f"PASS（derived {rep['n_derived']} / golden {rep['n_golden']}）===")
        for name in rep["failed"]:
            print(f"  ✗ {name}")
        return 1 if rep["n_failed"] else 0
