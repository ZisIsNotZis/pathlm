"""2623-composable-arms-13m · 变异验证（四臂阶梯的每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `arms.py`（四臂定义 / 块类选择规则 / native 可见性 / 阶梯
增量）。引擎三特性（GQA × carry × window）**本线不改**，其活性/变异已由
`2622-eng1-composable/mutate.py`（改 `pathlm/depth_ar.py`）钉住——这里只动 `arms.py`，
不碰任何共享引擎文件（SCHEMA §22：避免与并行线抢写）。

四处破坏点各对准一条断言：

- M1 `native_visibility`：把目标臂的窗口吞掉（窗口臂退化为全因果 ⇒ 与其它臂不可分）；
- M2 `staircase_increments`：把「逐叠加步增量」偷换成「净和」（§17.17 明令禁止只看净和）；
- M3 `expected_block_class`：GQA×carry 仍报 `GQABlock`（假装 ENG1 没解除组合）；
- M4 `native_visibility`：把窗口张到所有臂（基准臂也套窗口 ⇒ 基准被污染）。

每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

用法：python3 mutate.py            # 输出 mutation_report.json
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
TARGET = os.path.join(HERE, "arms.py")
TFILE = "tests/test_composable_arms_13m.py"

MUTATIONS = [
    ("M1", "native_visibility：目标臂的窗口被吞掉（窗口臂退化为全因果）",
     """    f = spec["flags"]
    return int(f.get("train_window", 0)), int(f.get("train_anchors", 0))
""",
     """    f = spec["flags"]
    return 0, 0
""",
     "test_window_arm_is_nondegenerate_and_not_identity"),
    ("M2", "staircase_increments：逐叠加步增量被偷换成净和（§17.17 禁止只看净和）",
     """    out = []
    for prev, cur in zip(ARMS, ARMS[1:]):
        a, b = acc[prev["name"]], acc[cur["name"]]
        out.append({"from": prev["name"], "to": cur["name"],
                    "added": cur["stacked"][-1],
                    "delta_pp": round(100.0 * (b - a), 3)})
    return out
""",
     """    net = round(100.0 * (acc[TARGET_ARM] - acc[BASELINE_ARM]), 3)
    out = []
    for prev, cur in zip(ARMS, ARMS[1:]):
        out.append({"from": prev["name"], "to": cur["name"],
                    "added": cur["stacked"][-1],
                    "delta_pp": net})
    return out
""",
     "test_staircase_increments_are_adjacent_diffs"),
    ("M3", "expected_block_class：GQA×carry 仍报 GQABlock（假装 ENG1 未解除组合）",
     """    if gqa and carry:
        return "ComposableBlock"
    if gqa:
        return "GQABlock"
""",
     """    if gqa and carry:
        return "GQABlock"
    if gqa:
        return "GQABlock"
""",
     "test_each_arm_builds_and_lands_expected_block_class"),
    ("M4", "native_visibility：窗口张到所有臂（基准臂被污染）",
     """    f = spec["flags"]
    return int(f.get("train_window", 0)), int(f.get("train_anchors", 0))
""",
     """    f = spec["flags"]
    return 16, 0
""",
     "test_non_window_arms_use_full_causal"),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    orig = open(TARGET, "rb").read()
    report = {"target": os.path.relpath(TARGET, ROOT),
              "sha256_original": sha256_bytes(orig), "mutations": []}
    for mid, desc, old, new, test in MUTATIONS:
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段出现 {src.count(old)} 次，无法唯一定位")
        open(TARGET, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(TARGET, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-p", "no:cacheprovider",
               "-k", test]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(TARGET, "wb").write(orig)                          # 逐字节还原
        restored = sha256_bytes(open(TARGET, "rb").read()) == report["sha256_original"]
        report["mutations"].append({
            "id": mid, "desc": desc, "expected_test": test,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    report["sha256_restored"] = sha256_bytes(open(TARGET, "rb").read())
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
