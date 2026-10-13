"""2631-carry-reseed · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `reseed_arms.py`（臂定义 + 聚类统计 + 跨种子判据）。
**不**改任何共享文件（`tools/**`、`tests/**`、`pathlm/**`、`train_depth_ar.py` 只读）。

八处破坏点各对准一条断言（`test_reseed_stats.py`）：

- M1 `pp_range`：极差换成 min−max（符号反转）；
- M2 `sign_consistent`：恒真（不再检查符号一致性）；
- M3 `pp_sd`：样本 sd 换成总体 sd（除 n 而非 n−1）；
- M4 `paired_clustered_bootstrap_diff_ci`：两臂改用**独立**抽样（不再配对）；
- M5 `corrupt_control_reading`：作废判定恒真（毁源后不掉也放过）；
- M6 `effect_stable`：丢掉「配对 CI 下界 > 0」这一半判据；
- M7 `arm_name`：臂名种子标签反转（接线错位）；
- M8 `model_config_kwargs`：所有规则都返回 delta（臂标签与实现脱钩）。

每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256；
未抓到 / 未还原 ⇒ **非零退出**（F3 / 复核 B4）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from tools.evalkit import mutation                                  # noqa: E402

TARGET = os.path.join(HERE, "reseed_arms.py")
TFILE = "axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed/test_reseed_stats.py"

MUTATIONS = [
    ("M1", "pp_range：极差被换成 min−max（符号反转）",
     "    return round(max(values) - min(values), 3)\n",
     "    return round(min(values) - max(values), 3)\n",
     "test_pp_range_mean_sd_and_sign"),
    ("M2", "sign_consistent：恒真（不再检查符号一致性）",
     "    return all(v > 0 for v in values) or all(v < 0 for v in values)\n",
     "    return True\n",
     "test_pp_range_mean_sd_and_sign"),
    ("M3", "pp_sd：样本 sd 被换成总体 sd（除 n 而非 n−1）",
     "    var = sum((v - m) ** 2 for v in values) / (n - 1)\n",
     "    var = sum((v - m) ** 2 for v in values) / n\n",
     "test_pp_range_mean_sd_and_sign"),
    ("M4", "配对聚类 bootstrap：两臂改用独立抽样（不再配对）",
     "        idx = rng.choices(range(m), k=m)                 # 两臂共用同一组索引 ⇒ 配对\n"
     "        ka = sum(map(hits_a.__getitem__, idx))\n"
     "        na = sum(map(n_a.__getitem__, idx))\n"
     "        kb = sum(map(hits_b.__getitem__, idx))\n"
     "        nb = sum(map(n_b.__getitem__, idx))\n",
     "        idx = rng.choices(range(m), k=m)\n"
     "        idx_b = rng.choices(range(m), k=m)              # 变异：不配对\n"
     "        ka = sum(map(hits_a.__getitem__, idx))\n"
     "        na = sum(map(n_a.__getitem__, idx))\n"
     "        kb = sum(map(hits_b.__getitem__, idx_b))\n"
     "        nb = sum(map(n_b.__getitem__, idx_b))\n",
     "test_paired_bootstrap_diff_sign_and_null"),
    ("M5", "corrupt_control_reading：作废判定恒真（毁源不掉也放过）",
     "        ok = (not high) or (cor_lo <= bound)\n",
     "        ok = True                                       # 变异：放过\n",
     "test_corrupt_control_flags_void_arm"),
    ("M6", "effect_stable：丢掉「配对 CI 下界 > 0」这一半判据",
     "                            and float(r[\"paired_boot_ci_pp\"][0]) > 0)}\n",
     "                            )}\n",
     "test_effect_stable_requires_threshold_and_positive_ci_lower"),
    ("M7", "arm_name：种子标签反转（接线错位）",
     "    return f\"{rule}_s{seed}\"\n",
     "    return f\"{rule}_s{1 - seed}\"\n",
     "test_arm_names_and_single_factor"),
    ("M8", "model_config_kwargs：所有规则都返回 delta（臂标签与实现脱钩）",
     "    return {\"carry_rule\": None if rule == \"none\" else rule}\n",
     "    return {\"carry_rule\": \"delta\"}\n",
     "test_model_config_kwargs_and_train_flags"),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _clear_bytecode():
    """删掉靶子模块的 `.pyc` 并禁止子进程写新 `.pyc`（mtime 秒级 + 等长改写会掩盖变异）。"""
    for f in glob.glob(os.path.join(HERE, "__pycache__", "reseed_arms*.pyc")):
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
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True,
                                timeout=1800, env=env)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(TARGET, "wb").write(orig)                       # 逐字节还原
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
    ok = report["all_caught_red"] and report["all_restored_bitwise"]
    mutation.exit_if_uncaught(ok, report, os.path.join(HERE, "mutation_report.json"),
                              message="有变异未被抓红或未逐字节还原")


if __name__ == "__main__":
    main()
