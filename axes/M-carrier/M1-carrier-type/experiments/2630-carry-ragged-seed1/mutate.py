"""2630-carry-ragged-seed1 · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `ragged_seed1_arms.py`（跨 seed 统计 + W2 判据②③）。
**不**改任何共享引擎文件，**不**改 2629 的语义层（SCHEMA §22：一个 cwd 同时只有一个写入者）。

六处破坏点各对准一条断言：

- M1 `range_pp`：极差变成 min−max（符号反转）；
- M2 `sign_consistent`：恒真（不再检查符号一致性）；
- M3 `ood_effect_robust`：丢掉「每 seed 差值 CI 下界 > 0」的统计检验（只留同号）；
- M4 `indist_redundancy_stable`：方差带判据反转（超带才算「稳定」）；
- M5 `BASE_TO_S1`：seed=0 臂名映射被调换（把 carry 臂当基准）；
- M6 `sd_pp`：样本 sd 被换成总体 sd（除 n 而非 n−1）。

每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
TARGET = os.path.join(HERE, "ragged_seed1_arms.py")
TFILE = "tests/test_carry_ragged_seed1_13m.py"

MUTATIONS = [
    ("M1", "pp_range：极差被换成 min−max（符号反转）",
     "    return round(max(vals) - min(vals), 3)\n",
     "    return round(min(vals) - max(vals), 3)\n",
     "test_pp_range"),
    ("M2", "sign_consistent：恒真（不再检查符号一致性）",
     "    return all(v > 0 for v in vals) or all(v < 0 for v in vals)\n",
     "    return True\n",
     "test_sign_consistency"),
    ("M3", "ood_effect_robust：丢掉每 seed 差值 CI 下界 > 0 的统计检验",
     "    robust = len(seeds) >= 2 and positive and significant\n",
     "    robust = len(seeds) >= 2 and positive\n",
     "test_ood_robust_requires_significance"),
    ("M4", "indist_redundancy_stable：方差带判据反转（超带才算稳定）",
     "    within = all(abs(float(e)) <= band_pp for e in effects)\n",
     "    within = all(abs(float(e)) >= band_pp for e in effects)\n",
     "test_indist_redundancy_band"),
    ("M5", "BASE_TO_S1：seed=0 臂名映射被调换（carry 臂当基准）",
     'BASE_TO_S1 = {BASELINE_ARM: "projfill_s1", TARGET_ARM: "projfill_carry_s1"}\n',
     'BASE_TO_S1 = {BASELINE_ARM: "projfill_carry_s1", TARGET_ARM: "projfill_s1"}\n',
     "test_arm_name_mapping"),
    ("M6", "pp_sd：样本 sd 被换成总体 sd（除 n 而非 n−1）",
     "    var = sum((v - m) ** 2 for v in vals) / (n - 1)\n",
     "    var = sum((v - m) ** 2 for v in vals) / n\n",
     "test_pp_sd"),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _clear_bytecode():
    """删掉靶子模块的 `.pyc` + 禁止子进程写新 `.pyc`。

    必要性：`*.pyc` 的失效判据是（源码 mtime 秒级, 源码大小）；本脚本的多次
    变异写入常在**同一秒内**、且破坏点与原文**等长** ⇒ 子进程会误用上一次编译的
    `.pyc`，把变异**掩盖**（实测 M5 曾被掩盖成「通过」）。清 bytecode 后每次必编译。
    """
    for f in glob.glob(os.path.join(HERE, "__pycache__", "ragged_seed1_arms*.pyc")):
        os.remove(f)


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
        _clear_bytecode()
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-p", "no:cacheprovider",
               "-k", test]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800,
                           env=env)
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
    print(json.dumps({k: report[k] for k in ("all_caught_red", "all_restored_bitwise")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
