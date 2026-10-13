"""2625-ood-leg-variance · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层 `ood_variance_arms.py`（三臂定义 / 同批位置 / 方差统计 / 判据阈值）。
引擎三特性（GQA × carry × window）与复用仪器 `quant_carry.py` **本线不改**——这里只动
本实验自己的文件，不碰任何共享文件或他人文件（SCHEMA §22）。

六处破坏点各对准一条单测（每个先记原文 sha256，跑指定单测期望**红**，再逐字节还原）：

- M1 `CONFIG_FLAGS`：窗口 16 → 0（不再是目标配置）；
- M2 `expected_block_class`：GQA×carry 仍报 `GQABlock`（假装 ENG1 没解除组合）；
- M3 `downgrade_to_auxiliary`：`>=` → `>`（边界 3.0 判反，§17.18 的「≥」被破坏）；
- M4 `MULTI_SEED_BASE`：2000 → 2001（不再与 2624 仪器同批位置）；
- M5 `range_pp`：`max−min` → `min−max`（极差符号反）；
- M6 `EFFECT_THRESHOLD_PP`：3.0 → 5.0（判据阈值被悄悄放宽）。

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
TFILE = "tests/test_ood_leg_variance.py"

MUTATIONS = [
    ("M1", "ood_variance_arms.py", "CONFIG_FLAGS：窗口 16 → 0",
     'train_window": 16, "train_anchors": 0}',
     'train_window": 0, "train_anchors": 0}',
     "test_target_config_is_gqa_carry_window16"),
    ("M2", "ood_variance_arms.py", "expected_block_class：GQA×carry 仍报 GQABlock",
     """    if gqa and carry:
        return "ComposableBlock"
""",
     """    if gqa and carry:
        return "GQABlock"
""",
     "test_expected_block_class_is_composable_for_all_arms"),
    ("M3", "ood_variance_arms.py", "downgrade_to_auxiliary：边界 >= 判反",
     "    return float(range_pp_value) >= float(threshold_pp)",
     "    return float(range_pp_value) > float(threshold_pp)",
     "test_downgrade_threshold_is_inclusive_at_3pp"),
    ("M4", "ood_variance_arms.py", "MULTI_SEED_BASE：2000 → 2001（不再同批位置）",
     "MULTI_SEED_BASE = 2000",
     "MULTI_SEED_BASE = 2001",
     "test_eval_batch_seeds_identical_regardless_of_arm"),
    ("M5", "ood_variance_arms.py", "range_pp：max−min → min−max（符号反）",
     "    return round(100.0 * (max(vals) - min(vals)), 3)",
     "    return round(100.0 * (min(vals) - max(vals)), 3)",
     "test_range_mean_sd_pp"),
    ("M6", "ood_variance_arms.py", "EFFECT_THRESHOLD_PP：3.0 → 5.0（放宽判据）",
     "EFFECT_THRESHOLD_PP = 3.0",
     "EFFECT_THRESHOLD_PP = 5.0",
     "test_downgrade_threshold_is_inclusive_at_3pp"),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    report = {"targets": {}, "mutations": []}
    digests = {}
    for _, fname, _, _, _, _ in MUTATIONS:
        if fname not in digests:
            path = os.path.join(HERE, fname)
            orig = open(path, "rb").read()
            digests[fname] = (path, orig, sha256_bytes(orig))
            report["targets"][fname] = sha256_bytes(orig)
    for mid, fname, desc, old, new, test in MUTATIONS:
        path, orig, digest = digests[fname]
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 片段在 {fname} 出现 {src.count(old)} 次，无法唯一定位")
        open(path, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(path, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-p", "no:cacheprovider",
               "-k", test]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(path, "wb").write(orig)                            # 逐字节还原
        restored = sha256_bytes(open(path, "rb").read()) == digest
        report["mutations"].append({
            "id": mid, "file": fname, "desc": desc, "expected_test": test,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    report["sha256_restored"] = {f: sha256_bytes(open(os.path.join(HERE, f), "rb").read())
                                 for f in digests}
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
