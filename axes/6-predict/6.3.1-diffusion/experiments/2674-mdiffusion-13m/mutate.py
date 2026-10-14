#!/usr/bin/env python3
"""变异验证（CPU，不取锁）：每条变异必须让对应单测**抓红**，随后**逐字节还原**。

    python3 mutate.py            # 全部变异
    python3 mutate.py -k leak    # 只跑名字含 leak 的变异

报告写 `mutation_report.json`。变异采用「文本替换 + 逐字节还原」，还原后校验文件的 sha256 与变异前
一致；任何一条「没抓红」或「还原不一致」都非零退出。报告**不含逐次运行耗时**（逐次运行的计时会抖动，
写进入库事实文件会让任何人重跑 `mutate.py` 都把工作树弄脏）。

靶子 = 本实验的仪器 `mdiff.py`：泄漏门（定格用真值）、步数守恒（ceil→floor）、bpc 口径（log 底）、
K=1 退化等价（第 0 步不置 mask）。
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
TESTS = [os.path.join(HERE, "test_mdiff.py")]

# (name, file, old, new, pytest -k selector)
MUTATIONS = [
    ("denoise_fixes_true_token", "mdiff.py",
     "x = torch.where(sel, pred, x)",
     "x = torch.where(sel, true_tokens, x)",
     "no_true_leak"),
    ("denoise_floor_instead_of_ceil", "mdiff.py",
     "per_step_fix = torch.ceil(n_m.float() / K).long()",
     "per_step_fix = torch.floor(n_m.float() / K).long()",
     "step_counts_conserve"),
    ("bpc_wrong_log_base", "mdiff.py",
     "total_s / (total_n * LN2) if total_n else float(\"nan\")",
     "total_s / (total_n * math.log(10)) if total_n else float(\"nan\")",
     "bpc_log2"),
    ("K1_step0_not_masked", "mdiff.py",
     "    x[M] = mask_token",
     "    x[M] = true_tokens[M]",
     "K1_degenerates"),
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _strip_timing(summary: str) -> str:
    """去掉 pytest 摘要尾部的耗时（如 `in 2.19s`）。"""
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
        caught = False
        tail: list[str] = []
        try:
            open(path, "w").write(text.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-k", sel],
                               capture_output=True, text=True, cwd=HERE)
            caught = r.returncode != 0
            if r.stdout:
                tail = [_strip_timing(l) for l in r.stdout.strip().splitlines()[-1:]]
        finally:
            open(path, "wb").write(orig_bytes)
        restored = sha(path) == orig_sha
        report["results"].append({
            "name": name, "file": fname, "test_selector": sel,
            "caught_red": bool(caught), "restored_byte_exact": bool(restored),
            "pytest_tail": tail,
        })
        report["all_caught"] &= bool(caught)
        report["all_restored"] &= bool(restored)
        print(f"{'RED ' if caught else 'MISS'} {name:34s} restored={restored}")

    # 全绿复核：还原后单测必须全过
    r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q"],
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
