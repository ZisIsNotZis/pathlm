"""2627-carry-x-depth · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `carry_x_depth_arms.py`（2×2 因子定义 / depth 因子到 `--dense-aux` 的映射 /
块类映射 / 四臂同可见性 / 因子分解）。引擎侧 `dense_aux` 本身、以及 `dense 主干忽略
exit_depths` 这两条**能力面**事实，由 `tests/test_carry_x_depth_13m.py` 的两条引擎回归
钉住、并由 `tests/test_depth_ar.py`（ticket 23c）覆盖其引擎变异——本线**不改**任何共享
引擎文件（SCHEMA §22：一个 cwd 同时只有一个写入者）。

五处破坏点各对准一条断言：

- M1 `train_flags`：深度因子被吞掉（`depth`/`both` 不再带 `--dense-aux` ⇒ depth 臂名不副实）；
- M2 `model_config_kwargs`：carry 因子被吞掉（四臂都变成无 carry ⇒ 只剩 depth 单因子）；
- M3 `factorial_increments`：2×2 单向增量被偷换成净和（§17.17 明令禁止只看净和）；
- M4 `native_visibility`：窗口被吞掉（四臂退化为全因果 ⇒ 无长程需求）；
- M5 `expected_block_class`：carry 臂仍报 `Block`（假装 carry 因子没生效）。

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
TARGET = os.path.join(HERE, "carry_x_depth_arms.py")
TFILE = "tests/test_carry_x_depth_13m.py"

MUTATIONS = [
    ("M1", "train_flags：深度因子被吞掉（depth/both 不再带 --dense-aux）",
     """    return [DEPTH_TRAIN_FLAG] if spec["depth"] else []
""",
     """    return []
""",
     "test_depth_factor_maps_to_dense_aux_train_flag"),
    ("M2", "model_config_kwargs：carry 因子被吞掉（四臂都变成无 carry）",
     """    return {"carry_rule": (CARRY_RULE if spec["carry"] else None)}
""",
     """    return {"carry_rule": None}
""",
     "test_carry_factor_is_the_only_config_variable"),
    ("M3", "factorial_increments：2×2 单向增量被偷换成净和（§17.17 禁止只看净和）",
     """    b, c, d, bd = (metric[n] for n in ("base", "carry", "depth", "both"))
    return {
        "carry_at_depth_off": round(c - b, 6),
        "depth_at_carry_off": round(d - b, 6),
        "carry_at_depth_on": round(bd - d, 6),
        "depth_at_carry_on": round(bd - c, 6),
        "interaction": round((bd - c) - (d - b), 6),
        "net_vs_base": round(bd - b, 6),
    }
""",
     """    b, c, d, bd = (metric[n] for n in ("base", "carry", "depth", "both"))
    net = round(bd - b, 6)
    return {
        "carry_at_depth_off": net,
        "depth_at_carry_off": net,
        "carry_at_depth_on": net,
        "depth_at_carry_on": net,
        "interaction": net,
        "net_vs_base": net,
    }
""",
     "test_factorial_increments_are_factorial_not_staircase"),
    ("M4", "native_visibility：窗口被吞掉（四臂退化为全因果）",
     """    return 16, 0
""",
     """    return 0, 0
""",
     "test_native_visibility_is_window16_for_all_arms"),
    ("M5", "expected_block_class：carry 臂仍报 Block（假装 carry 因子没生效）",
     """    return "CarryChannelBlock" if spec["carry"] else "Block"
""",
     """    return "Block"
""",
     "test_expected_block_class_per_arm"),
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
