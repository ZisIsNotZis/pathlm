#!/usr/bin/env python3
"""变异验证（CPU，不取锁）：每条变异必须让对应单测**抓红**，随后**逐字节还原**。

    python3 mutate.py            # 全部变异
    python3 mutate.py -k target  # 只跑名字含 target 的变异

报告写 `mutation_report.json`。变异采用「文本替换 + 逐字节还原」，还原后校验每个文件的 sha256
与变异前一致；任何一条「没抓红」或「还原不一致」都非零退出。报告**不含逐次运行耗时**
（沿用 2660 的 `_strip_timing`：逐次运行的计时会抖动，写进入库事实文件会让任何人重跑
`mutate.py` 都把工作树弄脏）。

靶子：本实验的仪器 —— 库 `padfill.py`（mask 语义 / tied 读出 / bpc 口径）与驱动 `run.py`
（参考带边界 / 禁墙钟自检）。
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
TESTS = [os.path.join(HERE, "test_padfill.py"), os.path.join(HERE, "test_run_s2.py")]

# (name, file, old, new, pytest -k selector)
MUTATIONS = [
    ("leak_mask_substitution", "padfill.py",
     '    if mode == "mask":\n'
     '        x = true_tokens.clone()\n'
     '        x[M] = mask_token\n'
     '        return x',
     '    if mode == "mask":\n'
     '        return true_tokens.clone()',
     "L0_bidir"),
    ("readout_not_tied", "padfill.py",
     "        return h @ self.tok_embed.weight.t()  # tied 读出：E 是唯一读出基",
     "        return h @ (self.tok_embed.weight * 1.0000001).t()  # tied 读出：E 是唯一读出基",
     "tied_readout"),
    ("bpc_wrong_log_base", "padfill.py",
     "    return float(nll_sum) / (n * math.log(2))",
     "    return float(nll_sum) / (n * math.log(10))",
     "bpc"),
    ("reference_band_loosened", "run.py",
     "    return abs(delta_bpc) > band",
     "    return abs(delta_bpc) >= band",
     "reference_band"),
    ("wall_clock_check_disabled", "run.py",
     '    """L5 禁墙钟自检：复用 `padfill.has_wall_clock`（key 命中墙钟语义即报）。"""\n'
     "    return P.has_wall_clock(obj)",
     '    """L5 禁墙钟自检：复用 `padfill.has_wall_clock`（key 命中墙钟语义即报）。"""\n'
     "    return []",
     "wall_clock"),
    ("ar_target_off_by_one", "run.py",
     "    return F.cross_entropy(logits.reshape(-1, V), win[:, 1:].reshape(-1))",
     "    return F.cross_entropy(logits[:, :-1].reshape(-1, V), win[:, 1:].reshape(-1))",
     "ar_target"),
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
        try:
            open(path, "w").write(text.replace(old, new))
            r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "-k", sel],
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
        print(f"{'RED ' if caught else 'MISS'} {name:32s} restored={restored}")

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
