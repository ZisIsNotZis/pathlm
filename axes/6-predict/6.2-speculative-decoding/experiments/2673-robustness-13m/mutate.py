"""WP 6.2.4 · 2673-robustness-13m 的变异验证（CPU，不取锁）。

对象 = `spec_lossless.py` 的三处**拒绝采样语义开关**（`spec_tree.py` / `pathlm/**` 只读，不参与变异）：

- **M1 接受概率恒为 1**：`accept_prob` 直接返回 1.0（= L0 gate probe 的「接受恒真」改错）
  ⇒ 接受–重采的发出边缘分布 = 草稿 q ≠ 目标 p。期望抓红：`test_accept_resample_marginal_equals_target`。
- **M2 残差方向写反**：`residual_dist` 用 `max(0, q − p)` 而非 `max(0, p − q)` ⇒ 重采分布错。
  期望抓红：`test_accept_resample_marginal_equals_target`。
- **M3 写回位置数错**：carry 结转后每轮写回 `n_pos = 1`（不按接受数）⇒ 缓存与发出流错位。
  期望抓红：`test_emitted_logp_matches_fresh_forward`。
- **M4 缓存游标取错位置**：`dec.last_nodes` 恒取第 0 行/第 0 位 ⇒ 下一轮目标分布错。
  期望抓红：`test_emitted_logp_matches_fresh_forward`。

每条变异跑完**逐字节还原**（sha256 前后相等）。只有 `test_failure` 算抓红（分类与判红由
`tools/evalkit/mutation.py` 统一，未抓到 ⇒ 非零退出）。报告**不含逐次运行耗时**
（pytest 摘要尾部的 `in N.Ns` 已剥除）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.evalkit import mutation  # noqa: E402

TARGET = os.path.join(HERE, "spec_lossless.py")
TESTS_REL = os.path.relpath(os.path.join(HERE, "test_spec_lossless.py"), ROOT)
MARGINAL_TEST = "test_accept_resample_marginal_equals_target"
LOGP_TEST = "test_emitted_logp_matches_fresh_forward"

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测名)
MUTATIONS = [
    ("M1", "接受概率恒为 1（L0 gate probe 的「接受恒真」改错）",
     "    return 1.0 if r >= 1.0 else r",
     "    return 1.0",
     [MARGINAL_TEST]),
    ("M2", "残差方向写反（用 max(0,q−p) 而非 max(0,p−q)）",
     "    r = torch.clamp(p - q, min=0.0)",
     "    r = torch.clamp(q - p, min=0.0)",
     [MARGINAL_TEST]),
    ("M3", "写回位置数错（carry 后每轮恒写回 1 个位置）",
     "        n_pos = 1 + n_acc",
     "        n_pos = 1",
     [LOGP_TEST]),
    ("M4", "缓存游标取错位置（last_nodes 恒取第 0 位）",
     "        dec.last_nodes = T._take_row(T._slice_nodes(fnodes, n_pos - 1), 0)",
     "        dec.last_nodes = T._take_row(T._slice_nodes(fnodes, 0), 0)",
     [LOGP_TEST]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _strip_timing(summary: str) -> str:
    """去掉 pytest 摘要尾部的耗时（如 `in 2.19s`）：逐次运行的计时会抖动，
    写进入库报告会让任何人重跑 `mutate.py` 都把工作树弄脏。"""
    return re.sub(r"\s+in\s+[0-9.]+s\s*$", "", summary.strip())


def main() -> int:
    original = open(TARGET, "rb").read()
    base_digest = sha256_bytes(original)
    src = original.decode()
    report = {"targets": {os.path.relpath(TARGET, ROOT): base_digest}, "mutations": []}
    for mid, desc, old, new, tests in MUTATIONS:
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段出现 {src.count(old)} 次，无法唯一定位")
        open(TARGET, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(TARGET, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TESTS_REL, "-q", "-k", " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        caught = r.returncode != 0
        tail = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()][-1:]
        open(TARGET, "wb").write(original)                      # 逐字节还原
        restored = sha256_bytes(open(TARGET, "rb").read()) == base_digest
        report["mutations"].append({
            "id": mid, "desc": desc, "target": os.path.relpath(TARGET, ROOT),
            "expected_tests": tests, "caught_red": bool(caught),
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": _strip_timing(tail[0] if tail else "")})
        print(f"{mid}: caught_red={caught} restored={restored} | {tail[0] if tail else ''}",
              flush=True)
    report["n_mutations"] = len(report["mutations"])
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"] for m in report["mutations"])
    with open(os.path.join(HERE, "mutation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    mutation.exit_if_uncaught(
        report["all_caught_red"] and report["all_restored_bitwise"], report,
        os.path.join(HERE, "mutation_report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
