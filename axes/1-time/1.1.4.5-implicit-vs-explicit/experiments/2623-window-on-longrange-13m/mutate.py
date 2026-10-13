#!/usr/bin/env python3
"""2623 · 变异验证（新语义层 `lr_eval.py` 的每个开关「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象（本实验唯一新增的语义层，四处）：
  ① `native_keep_mask` 对窗口臂**真的返回窗口算子**（不是释放版）；
  ② 该算子是**真的 16 窗**（不是恒等/全因果）；
  ③ `distance_source_inside_window` 的窗内外判据真的按 `d < window`；
  ④ `aggregate_recall` 的聚合口径（n = 逐批求和，不是批数）。

只动本实验目录内的 `lr_eval.py`（不碰 `pathlm/`、`tools/evalkit/`，SCHEMA §22）。
用法：python3 mutate.py            # 输出 mutation_report.json（退出码 0 表示全抓红且全还原）
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
TARGET = os.path.join(HERE, "lr_eval.py")
TFILE = "tests/test_a9_window_longrange.py"

MUTATIONS = [
    ("M1", "窗口臂被当成释放版（native_keep_mask 对 window>0 也返回 None）",
     """    if train_window <= 0:
        if train_anchors > 0:
            raise ValueError("anchors 只在 train_window > 0 时有意义")
        return None
    return window_anchor_keep_mask(T, train_window, train_anchors, device)""",
     """    if train_window <= 0:
        if train_anchors > 0:
            raise ValueError("anchors 只在 train_window > 0 时有意义")
        return None
    return None""",
     ["test_native_keep_mask_win16_is_window_operator"]),
    ("M2", "窗口算子退化为恒等（窗口参数被写死成 ≥T）",
     "    return window_anchor_keep_mask(T, train_window, train_anchors, device)",
     "    return window_anchor_keep_mask(T, 9999, train_anchors, device)",
     ["test_native_keep_mask_win16_is_window_operator",
      "test_window16_at_T256_is_nondegenerate"]),
    ("M3", "距离腿窗内外判据失效（恒判「在窗内」）",
     "    return distance < window",
     "    return True",
     ["test_distance_source_inside_window"]),
    ("M4", "聚合口径错（n 用批数而非逐批求和）",
     '    n = sum(int(r["n"]) for r in rows)',
     '    n = len(rows)',
     ["test_aggregate_recall_wilson_and_totals"]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    orig = open(TARGET, "rb").read()
    report = {"target": os.path.relpath(TARGET, ROOT),
              "sha256_original": sha256_bytes(orig), "mutations": []}
    for mid, desc, old, new, tests in MUTATIONS:
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段出现 {src.count(old)} 次，无法唯一定位")
        open(TARGET, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(TARGET, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-k",
               " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=900)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(TARGET, "wb").write(orig)                         # 逐字节还原
        restored = sha256_bytes(open(TARGET, "rb").read()) == report["sha256_original"]
        report["mutations"].append({
            "id": mid, "desc": desc, "expected_tests": tests,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"all_caught_red={report['all_caught_red']} "
          f"all_restored_bitwise={report['all_restored_bitwise']}")
    if not (report["all_caught_red"] and report["all_restored_bitwise"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
