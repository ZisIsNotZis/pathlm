"""WP 6.2.3 · 2650-verify-gate-13m 的变异验证（3 条预登记语义变异；CPU，不取 GPU 锁）。

对象 = 本实验通用 K 行树验证主循环 `verify_gate.py::decode_gate` 的三处语义开关
（2640 的 `spec_tree.py` **只读、不参与变异**）：

- **M1 行↔候选映射错位**：`rows[i]` 里的候选改成 `cands[(i+1) % K]`（旋转一格），
  而接受判据/发出仍按原 `cands` 索引 ⇒ 接受行写回的 K/V 与实际候选错位
  ⇒ 缓存污染、发出流偏离 native。期望抓红：`test_gate_streams_are_bit_identical_to_native_greedy`。
- **M2 接受行缓存回写错行**：`row_used` 恒取第 0 行（不按接受行索引）⇒ 接受行不是第 0 行时
  写回错误 K/V。期望抓红：`test_gate_streams_are_bit_identical_to_native_greedy`。
- **M3 接受判据放宽为「无条件接受」**：`acc_i` 恒取第 0 行（不做「候选 == 真值 argmax」判定）
  ⇒ 发出流不再是 native 贪心流。期望抓红：同上。

每条变异跑完**逐字节还原**（sha256 前后相等）。只有 `test_failure` 算抓红（分类与判红由
`tools/evalkit/mutation.py` 统一）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.evalkit import mutation  # noqa: E402

VG = os.path.join(HERE, "verify_gate.py")
TESTS_REL = os.path.relpath(os.path.join(HERE, "test_verify_gate.py"), ROOT)
LOSSLESS_TEST = "test_gate_streams_are_bit_identical_to_native_greedy"

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测名)
MUTATIONS = [
    ("M1", "行↔候选映射错位（rows 的候选旋转一格，接受行写回错位）", VG,
     "        rows = [[v1, c, d3] for c in cands]",
     "        rows = [[v1, cands[(i + 1) % len(cands)], d3]\n"
     "                for i in range(len(cands))]",
     [LOSSLESS_TEST]),
    ("M2", "接受行缓存回写错行（row_used 恒取第 0 行）", VG,
     "        row_used = 0 if acc_i is None else acc_i",
     "        row_used = 0",
     ["test_gate_writeback_uses_the_accepted_row_not_row_zero"]),
    ("M3", "接受判据放宽为「无条件接受」（acc_i 恒取第 0 行）", VG,
     """        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[i, 0].argmax()) == c), None)""",
     "        acc_i = 0",
     [LOSSLESS_TEST]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_timing(summary: str) -> str:
    """去掉 pytest 摘要尾部的耗时（如 `in 2.19s`）：逐次运行的计时会抖动，
    写进入库事实文件会让任何人重跑 `mutate.py` 都把工作树弄脏。"""
    return re.sub(r"\s+in\s+[0-9.]+s\s*$", "", summary.strip())


def main() -> int:
    original = open(VG, "rb").read()
    base_digest = sha256_bytes(original)
    report = {"targets": {os.path.relpath(VG, ROOT): base_digest}, "mutations": []}
    src = original.decode()
    for mid, desc, target, old, new, tests in MUTATIONS:
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段在 {target} 出现 {src.count(old)} 次，无法唯一定位")
        open(target, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(target, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TESTS_REL, "-q", "-k",
               " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(target, "wb").write(original)                    # 逐字节还原
        restored = sha256_bytes(open(target, "rb").read()) == base_digest
        report["mutations"].append({
            "id": mid, "desc": desc, "target": os.path.relpath(target, ROOT),
            "expected_tests": tests, "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "sha256_restored": sha256_bytes(open(target, "rb").read()),
            "restored_bitwise": restored,
            "pytest_summary": _strip_timing(tail[0] if tail else "")})
        print(f"{mid}: caught_red={caught} restored={restored} | {tail[0] if tail else ''}",
              flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"] for m in report["mutations"])
    with open(os.path.join(HERE, "mutation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red", "all_restored_bitwise")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
