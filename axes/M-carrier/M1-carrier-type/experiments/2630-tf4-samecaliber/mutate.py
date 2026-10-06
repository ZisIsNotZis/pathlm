"""2630-tf4-samecaliber · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `samecaliber_arms.py`（两臂只差 carry / 两臂都训在 enwik8 上 /
随机退出 + proj-fill 是两臂同款形态件 / 块类映射 / 单因素增量 / tf@4 保持性规则）。引擎侧
能力面（carry × 非 dense 可建、`train_window` 与 `dense=False` 互斥、`proj_fill` 需
`fill_kv`）由 `tests/test_tf4_samecaliber_13m.py` 的三条回归钉住——本线**不改**任何共享
引擎文件（SCHEMA §22：一个 cwd 同时只有一个写入者）。

六处破坏点各对准一条断言：

- M1 `train_flags`：proj-fill 形态件被吞掉（两臂都退化成裸非 dense ⇒ 不是 100M 配方）；
- M2 `model_config_kwargs`：carry 因子被吞掉（两臂都变成无 carry ⇒ 单因素对照失效）；
- M3 `single_factor_increment`：增量被偷换成「基准 − 目标」（符号反转）；
- M4 `data_source`：训练数据被伪造成 synth（同口径前提被破坏）；
- M5 `expected_block_class`：carry 臂仍报 `Block`（假装 carry 因子没生效）；
- M6 `tf4_holds`：判据①的保持性规则被反转（≥ band 才算保持）。

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
TARGET = os.path.join(HERE, "samecaliber_arms.py")
TFILE = "tests/test_tf4_samecaliber_13m.py"

MUTATIONS = [
    ("M1", "train_flags：proj-fill 形态件被吞掉（两臂退化成裸非 dense）",
     """    flags = ["--proj-fill", "--exit-mode", EXIT_MODE]
""",
     """    flags = ["--exit-mode", EXIT_MODE]
""",
     "test_train_flags_share_recipe_and_add_carry"),
    ("M2", "model_config_kwargs：carry 因子被吞掉（两臂都变成无 carry）",
     """        "carry_rule": (CARRY_RULE if spec["carry"] else None),
""",
     """        "carry_rule": None,
""",
     "test_only_carry_varies_between_arms"),
    ("M3", "single_factor_increment：增量被偷换成基准−目标（符号反转）",
     """    b, c = metric[BASELINE_ARM], metric[TARGET_ARM]
    return {"carry_effect": round(c - b, 6), "net_vs_base": round(c - b, 6)}
""",
     """    b, c = metric[BASELINE_ARM], metric[TARGET_ARM]
    return {"carry_effect": round(b - c, 6), "net_vs_base": round(b - c, 6)}
""",
     "test_single_factor_increment_arithmetic"),
    ("M4", "data_source：训练数据被伪造成 synth（同口径前提被破坏）",
     """    return DATA_SOURCE
""",
     """    return "synth"
""",
     "test_both_arms_train_on_enwik8_not_synth"),
    ("M5", "expected_block_class：carry 臂仍报 Block（假装 carry 因子没生效）",
     """    return "CarryChannelBlock" if spec["carry"] else "Block"
""",
     """    return "Block"
""",
     "test_expected_block_class_per_arm"),
    ("M6", "tf4_holds：判据①的保持性规则被反转（≥ band 才算保持）",
     """    return abs(delta_pp) < band_pp
""",
     """    return abs(delta_pp) >= band_pp
""",
     "test_tf4_holds_rule"),
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
