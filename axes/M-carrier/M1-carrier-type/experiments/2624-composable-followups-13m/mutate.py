"""2624-composable-followups-13m · 变异验证（三个补点的每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `points.py`（无泄漏谓词 / 容量括号与插值 / 同相对刻度 / 坐标换算）。
**本线不改引擎**（`pathlm/` 一行未动）⇒ 这里只动 `points.py`，不碰任何共享文件
（SCHEMA §22：避免与并行线抢写）。

五处破坏点各对准一条断言：

- M1 `Q_LEAKFREE`：无泄漏腿的 Q 被改回 K（伪影条件复活 ⇒ 「Q=7 无泄漏」这条断言必须红）；
- M2 `unique_unqueried_last_block`：谓词取反（把「唯一未被问过」判反）；
- M3 `interp_at_target`：去掉括号检查 ⇒ 允许**外推**（§17.11 明令禁止）；
- M4 `capacity_slope`：退化括号不再报错（§17.16 要求排除退化点）；
- M5 臂表 `gqa_10k`：步数从 10000 改回 6000（③ 的两点退化成单点 ⇒ 同相对刻度断言必须红）。

每处先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(HERE, "points.py")
TFILE = "tests/test_composable_followups_13m.py"

MUTATIONS = [
    ("M1", "Q_LEAKFREE：无泄漏腿的 Q 被改回 K（排除法伪影条件复活）",
     "Q_LEAKFREE = 7",
     "Q_LEAKFREE = 8",
     "test_q8_last_block_is_unique_unqueried_q7_is_not"),
    ("M2", "unique_unqueried_last_block：谓词取反（把「唯一未被问过」判反）",
     "        out.append(len(asked | {last}) == int(n_pairs))",
     "        out.append(len(asked | {last}) != int(n_pairs))",
     "test_q8_last_block_is_unique_unqueried_q7_is_not"),
    ("M3", "interp_at_target：去掉括号检查 ⇒ 允许外推（§17.11 禁止）",
     """    if not float(x_lo) <= float(x) <= float(x_hi):
        raise ValueError(f"目标参数量 {x} 不在括号 [{x_lo}, {x_hi}] 内 ⇒ 禁外推（§17.11）")
""",
     """    if not float(x_lo) <= float(x) <= float(x_hi):
        pass
""",
     "test_interpolation_values_and_no_extrapolation"),
    ("M4", "capacity_slope：退化括号（x_lo == x_hi）不再报错（§17.16 要求排除退化点）",
     """    if float(x_lo) == float(x_hi):
        raise ValueError("容量括号退化：两个 MHA 点的参数量相同 ⇒ 斜率无定义")
""",
     """    if float(x_lo) == float(x_hi):
        return 0.0
""",
     "test_capacity_slope_rejects_degenerate_bracket"),
    ("M5", "臂表 gqa_10k：步数从 10000 改回 6000（③ 的两点退化成单点）",
     '"gqa_10k": {"group": "p3_10k", "source": "2624", "steps": 10000, "mlp_mult": 6,',
     '"gqa_10k": {"group": "p3_10k", "source": "2624", "steps": 6000, "mlp_mult": 6,',
     "test_scale_pairs_same_recipe_same_ratio"),
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
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
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
