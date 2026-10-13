"""2631 · 本实验两个语义层的**变异验证**（`causal_check.py` + `budget_caliber.py`）。

对两个模块做**故意破坏**，跑对应单测，要求全部抓红；每次破坏后立即逐字节还原
（原文/破坏文对照 + sha256 校验）。统一走 `tools.evalkit.mutation`（F3）：
只有 `test_failure` 算抓红，「测试没跑起来 / 环境错误」直接非零退出；未抓到 ⇒ 非零退出。

| 变异 | 目标 | 破坏内容 | 期望抓红的单测 |
|---|---|---|---|
| M1 | `causal_check.py` | 前缀不变性的 **peer 检查**写成恒 True（旁路不再被验） | `test_prefix_invariance_check_flags_noncausal_implementation` |
| M2 | `causal_check.py` | 方向标签符号反转（改善↔变差） | `test_direction_label_signs` |
| M3 | `causal_check.py` | `quantized_arms_of` 去掉 mode 过滤（把所有臂当量化臂） | `test_quantized_arms_of_selects_only_int_arms` |
| M4 | `causal_check.py` | 非量化臂漂移不判 False（可控性证据恒真） | `test_compare_detects_nonquant_drift` |
| M5 | `budget_caliber.py` | 逐值标尺（方案 a）当免费 ⇒ 矛盾又回来了 | `test_int8_and_int4_occupy_more_than_the_fp16_baseline_under_scheme_a` |
| M6 | `budget_caliber.py` | 分块摊薄被架空（分母恒 1 = 逐值标尺） | `test_short_segments_amortize_less_than_a_full_group`、`test_quantization_factor_is_below_one…scheme_b` |
| M7 | `budget_caliber.py` | `value_bytes` 与 `mode` 不一致不再报错 | `test_mode_of_rejects_value_bytes_mode_mismatch` |
| M8 | `budget_caliber.py` | 段模型吞掉锚点段（判据 C1 对账被绕过） | `test_segments_reproduce_the_recorded_axis_factor_for_every_arm` |

用法：`python3 mutate.py`（纯 CPU，<1 min）。
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402

TARGET = os.path.join(HERE, "causal_check.py")
TESTS = "axes/R-references/R1-quantization/experiments/2631-kv-budget-causal/test_causal_check.py"
CALIBER = os.path.join(HERE, "budget_caliber.py")
CALIBER_TESTS = "tests/test_kv_budget_causal.py"

MUTATIONS = [
    {
        "id": "M1",
        "target": TARGET,
        "tests_file": TESTS,
        "what": "前缀不变性的 peer 检查恒 True（A2 的跨请求旁路不再被验）",
        "old": '        "peer_request_logits_equal": bool(torch.equal(base[0], got_peer[0])),',
        "new": '        "peer_request_logits_equal": True,   # 变异：旁路检查被架空',
        "tests": ["test_prefix_invariance_check_flags_noncausal_implementation"],
    },
    {
        "id": "M2",
        "target": TARGET,
        "tests_file": TESTS,
        "what": "方向标签符号反转（改善↔变差）",
        "old": '    if delta_bpc < -tol:\n        return "改善（因果口径质量代价更小）"',
        "new": '    if delta_bpc < -tol:\n        return "变差（因果口径质量代价更大）"   # 变异：符号反转',
        "tests": ["test_direction_label_signs"],
    },
    {
        "id": "M3",
        "target": TARGET,
        "tests_file": TESTS,
        "what": "quantized_arms_of 去掉 mode 过滤（把所有臂当量化臂）",
        "old": '        if spec.get("mode") in ("int8", "int4"):',
        "new": '        if True:   # 变异：不区分量化臂',
        "tests": ["test_quantized_arms_of_selects_only_int_arms"],
    },
    {
        "id": "M4",
        "target": TARGET,
        "tests_file": TESTS,
        "what": "非量化臂漂移不判 False（可控性证据恒真）",
        "old": '        "nonquant_bitwise_preserved": all(abs(r["delta_bpc"]) <= tol for r in nonq),',
        "new": '        "nonquant_bitwise_preserved": True,   # 变异：漂移不报',
        "tests": ["test_compare_detects_nonquant_drift"],
    },
    {
        "id": "M5",
        "target": CALIBER,
        "tests_file": CALIBER_TESTS,
        "what": "逐值标尺（方案 a）当免费 ⇒ table.md 与 analysis.md 的矛盾又回来",
        "old": '    if scheme == "a_per_value":\n        return SCALE_BYTES',
        "new": '    if scheme == "a_per_value":\n        return 0.0   # 变异：逐值标尺当免费',
        "tests": ["test_per_value_scale_makes_int8_three_bytes_and_int4_two_and_a_half",
                  "test_int8_and_int4_occupy_more_than_the_fp16_baseline_under_scheme_a"],
    },
    {
        "id": "M6",
        "target": CALIBER,
        "tests_file": CALIBER_TESTS,
        "what": "分块摊薄被架空（分母恒 1 = 方案 b 退化成方案 a）",
        "old": '        group_used = 1 if scheme == "a_per_value" else min(group, seg)',
        "new": '        group_used = 1   # 变异：分块摊薄被架空',
        "tests": ["test_short_segments_amortize_less_than_a_full_group",
                  "test_quantization_factor_is_below_one_for_every_quantized_arm_under_scheme_b"],
    },
    {
        "id": "M7",
        "target": CALIBER,
        "tests_file": CALIBER_TESTS,
        "what": "value_bytes 与 mode 不一致不再报错（标签与事实不符也出数）",
        "old": '        if abs(vb - VALUE_BYTES[mode]) > 1e-9:',
        "new": '        if False:   # 变异：不校验标签',
        "tests": ["test_mode_of_rejects_value_bytes_mode_mismatch"],
    },
    {
        "id": "M8",
        "target": CALIBER,
        "tests_file": CALIBER_TESTS,
        "what": "段模型吞掉锚点段（与判据 C1 的 positions_kept 对不上也不报错）",
        "old": '        return ([a] if a > 0 else []) + [w]',
        "new": '        return [w]   # 变异：锚点段被吞掉',
        "tests": ["test_segments_reproduce_the_recorded_axis_factor_for_every_arm"],
    },
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_file(tests_file: str = TESTS) -> dict:
    r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", tests_file],
                            cwd=ROOT, capture_output=True, text=True)
    return {"returncode": r.returncode,
            "tail": r.stdout.strip().splitlines()[-1][:160]}


def run_files() -> dict:
    """两个模块各自的全文件单测结果（还原后必须都绿）。"""
    return {f: run_file(path) for f, path in (("causal", TESTS), ("caliber", CALIBER_TESTS))}


def run_tests(names: list[str], tests_file: str = TESTS) -> dict:
    out = {}
    for n in names:
        r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q",
                                 f"{tests_file}::{n}"],
                                cwd=ROOT, capture_output=True, text=True)
        out[n] = {"returncode": r.returncode,
                  "verdict": "FAILED（抓红）" if r.returncode != 0 else "passed（未抓到）",
                  "tail": r.stdout.strip().splitlines()[-1][:160]}
    return out


def main():
    originals = {TARGET: open(TARGET).read(), CALIBER: open(CALIBER).read()}
    sha_before = {p: sha(p) for p in originals}
    report = {"targets": {os.path.relpath(p, ROOT): {"sha256_before": sha_before[p]}
                          for p in originals},
              "mutations": []}
    try:
        for m in MUTATIONS:
            path, original = m["target"], originals[m["target"]]
            assert original.count(m["old"]) == 1, f"{m['id']}: 定位失败"
            open(path, "w").write(original.replace(m["old"], m["new"]))
            assert sha(path) != sha_before[path]
            res = run_tests(m["tests"], m["tests_file"])
            caught = all(v["returncode"] != 0 for v in res.values())
            report["mutations"].append({"id": m["id"], "target": os.path.relpath(path, ROOT),
                                        "what": m["what"], "tests": res,
                                        "all_expected_tests_caught": caught})
            print(f"[{m['id']}] {m['what']}: "
                  f"{'全部抓红 ✓' if caught else '未抓到 ✗'}", flush=True)
            open(path, "w").write(original)          # 立即还原
            assert sha(path) == sha_before[path]
        report["restored_sha256_equal"] = all(sha(p) == sha_before[p] for p in originals)
        report["after_restore"] = run_files()
        report["green_after_restore"] = all(v["returncode"] == 0
                                           for v in report["after_restore"].values())
    finally:
        for path, original in originals.items():
            open(path, "w").write(original)           # 任何异常也还原
    for p in originals:
        report["targets"][os.path.relpath(p, ROOT)]["sha256_after"] = sha(p)
    report["all_caught"] = all(m["all_expected_tests_caught"] for m in report["mutations"])
    print(json.dumps({"green_after_restore": report["green_after_restore"],
                      "restored_sha256_equal": report["restored_sha256_equal"],
                      "all_caught": report["all_caught"]}, indent=2))
    mutation.exit_if_uncaught(
        bool(report["all_caught"] and report["restored_sha256_equal"]),
        report, os.path.join(HERE, "mutation_report.json"),
        message="有变异未被抓红，或还原不是逐字节")


if __name__ == "__main__":
    main()
