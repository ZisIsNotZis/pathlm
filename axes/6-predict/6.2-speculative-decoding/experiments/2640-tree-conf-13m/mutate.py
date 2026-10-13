"""2640 · 变异验证（3 条预登记语义变异；CPU，不取 GPU 锁）。

对象 = `spec_tree.py` 的三处语义开关：

- **M1 候选集只剩 1 个**：`tree_direct_k2` 的 K 从 2 改成 1 ⇒ 候选集 = `top-1` = `direct1` 的草稿，
  两臂同值。期望抓红：`test_candidate_counts_match_preregistration`。
- **M2 接受判据改成「无条件接受」**：`acc_i` 恒取第 0 行（不做「候选 == 真值 argmax」判定）
  ⇒ 发出流不再是 native 贪心流。期望抓红：无损性门禁
  `test_tree_streams_are_bit_identical_to_native_greedy`。
- **M3 `combo_pw` 权重换固定 0.5**：`combo_pw` 的草稿分布退回等权（= `combined_fixed`）⇒ 两臂完全
  相同。期望抓红：`test_combo_pw_differs_from_combined_fixed_on_the_fixture`。

每条变异跑完**逐字节还原**（sha256 前后相等）。只有 `test_failure` 算抓红（分类与判红由
`tools/evalkit/mutation.py` 统一）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tools.evalkit import mutation  # noqa: E402

ST = os.path.join(HERE, "spec_tree.py")
TESTS_REL = os.path.relpath(os.path.join(HERE, "test_spec_tree.py"), ROOT)

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测名)
MUTATIONS = [
    ("M1", "候选集只剩 1 个（tree_direct_k2 退化成 direct1，两臂同值）", ST,
     'TOPK_ARMS = {"tree_direct_k2": ("direct", 2), "tree_qpw_k2": ("pw", 2),',
     'TOPK_ARMS = {"tree_direct_k2": ("direct", 1), "tree_qpw_k2": ("pw", 2),',
     ["test_candidate_counts_match_preregistration"]),
    ("M2", "接受判据改成「无条件接受」（恒取第 0 行，不做候选==真值判定）", ST,
     """        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[i, 0].argmax()) == c), None)""",
     """        acc_i = 0""",
     ["test_tree_streams_are_bit_identical_to_native_greedy"]),
    ("M3", "combo_pw 权重换固定 0.5（退回等权，与 combined_fixed 完全相同）", ST,
     """    if arm == "combo_pw":
        return pw_mix(nodes)["q"]""",
     """    if arm == "combo_pw":
        return lam * probs["pc"] + (1.0 - lam) * probs["p2"]""",
     ["test_combo_pw_differs_from_combined_fixed_on_the_fixture"]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    original = open(ST, "rb").read()
    base_digest = sha256_bytes(original)
    report = {"targets": {os.path.relpath(ST, ROOT): base_digest}, "mutations": []}
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
            "pytest_summary": tail[0] if tail else ""})
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
