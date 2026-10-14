#!/usr/bin/env python3
"""变异验证（CPU，不取锁）：每条变异必须让对应单测**抓红**，随后**逐字节还原**。

    python3 mutate.py            # 全部变异
    python3 mutate.py -k target  # 只跑名字含 target 的变异

报告写 `mutation_report.json`。变异采用「文本替换 + 逐字节还原」，还原后校验每个文件的 sha256
与变异前一致；任何一条「没抓红」或「还原不一致」都非零退出。报告**不含逐次运行耗时**（照抄 2660
的 `_strip_timing`：逐次计时会抖动，写进入库事实文件会让任何重跑都把工作树弄脏）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, "test_freerun.py")

# (name, file, old, new, pytest -k selector)
MUTATIONS = [
    ("readout_not_shared", "freerun.py",
     "    return z @ E.T",
     "    return z @ (E * 1.0000001).T",
     "readout"),
    ("paired_bootstrap_sign_flipped", "freerun.py",
     "    num = a[idx].sum(1) - b[idx].sum(1)",
     "    num = b[idx].sum(1) - a[idx].sum(1)",
     "ci_sign"),
    ("ci_percentiles_swapped", "freerun.py",
     "    lo = float(np.percentile(d, 2.5))\n    hi = float(np.percentile(d, 97.5))",
     "    lo = float(np.percentile(d, 97.5))\n    hi = float(np.percentile(d, 2.5))",
     "ci_perc"),
    ("head_order_reversed", "freerun.py",
     "            z = heads[j](h_i)",
     "            z = heads[k - 1 - j](h_i)",
     "head_order"),
    ("wall_clock_undetected", "freerun.py",
     "WALL_SUBSTR = (\"wall\", \"elapsed\", \"clock\", \"duration\", \"seconds\", \"milliseconds\",\n"
     "               \"latency\", \"throughput\")",
     "WALL_SUBSTR = ()",
     "wall_clock"),
    # 复核 N1 新增读数（真值锚 / 退化统计）的变异：各自必须被对应单测抓红。
    ("teacher_forced_shift_offbyone", "freerun.py",
     "    sel = logits[:, pl - 1:pl - 1 + n, :]",
     "    sel = logits[:, pl - 2:pl - 2 + n, :]",
     "teacher_forced"),
    ("degeneracy_max_run_offbyone", "freerun.py",
     "        max_run = max(max_run, run)",
     "        max_run = max(max_run, run - 1)",
     "degeneracy"),
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _strip_timing(summary: str) -> str:
    """去掉 pytest 摘要尾部的耗时（如 `in 2.19s`）：逐次运行的计时会抖动。"""
    return re.sub(r"\s+in\s+[0-9.]+s\s*$", "", summary.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", default=None)
    a = ap.parse_args()
    muts = [m for m in MUTATIONS if a.k is None or a.k in m[0]]

    report = {"n_mutations": len(muts), "results": [], "all_caught": True, "all_restored": True}
    for name, fname, old, new, sel in muts:
        path = os.path.join(HERE, fname)
        orig_bytes = open(path, "rb").read()
        orig_sha = sha(path)
        text = orig_bytes.decode()
        assert text.count(old) == 1, f"{name}: 匹配 {text.count(old)} 次（须恰好 1）"
        try:
            open(path, "w").write(text.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q", "-k", sel],
                               capture_output=True, text=True, cwd=HERE)
            caught = r.returncode != 0
        finally:
            open(path, "wb").write(orig_bytes)
        restored = sha(path) == orig_sha
        report["results"].append({
            "name": name, "file": fname, "test_selector": sel,
            "caught_red": bool(caught), "restored_byte_exact": bool(restored),
            "pytest_tail": [_strip_timing(l) for l in
                            ((r.stdout or "")[-400:].strip().splitlines()[-1:] if r.stdout else [])],
        })
        report["all_caught"] &= bool(caught)
        report["all_restored"] &= bool(restored)
        print(f"{'RED ' if caught else 'MISS'} {name:34s} restored={restored}")

    # 全绿复核：还原后单测必须全过
    r = subprocess.run([sys.executable, "-m", "pytest", TEST, "-q"],
                       capture_output=True, text=True, cwd=HERE)
    report["post_restore_all_green"] = r.returncode == 0
    report["post_restore_tail"] = [_strip_timing(l) for l in (r.stdout or "").strip().splitlines()[-1:]]
    with open(os.path.join(HERE, "mutation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("n_mutations", "all_caught", "all_restored",
                                             "post_restore_all_green")}, ensure_ascii=False))
    ok = report["all_caught"] and report["all_restored"] and report["post_restore_all_green"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
