"""2631 · `causal_check.py` 的新语义（前缀不变性仪器 + 新旧并列比较）的**变异验证**。

对 `causal_check.py` 做四处**故意破坏**，跑本目录单测，要求全部抓红；每次破坏后立即
逐字节还原（原文/破坏文对照 + sha256 校验）。统一走 `tools.evalkit.mutation`（F3）：
只有 `test_failure` 算抓红，「测试没跑起来 / 环境错误」直接非零退出；未抓到 ⇒ 非零退出。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | 前缀不变性的 **peer 检查**写成恒 True（旁路不再被验） | `test_prefix_invariance_check_flags_noncausal_implementation` |
| M2 | 方向标签符号反转（改善↔变差） | `test_direction_label_signs` |
| M3 | `quantized_arms_of` 去掉 mode 过滤（把所有臂当量化臂） | `test_quantized_arms_of_selects_only_int_arms` |
| M4 | 非量化臂漂移不判 False（可控性证据恒真） | `test_compare_detects_nonquant_drift` |

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

MUTATIONS = [
    {
        "id": "M1",
        "what": "前缀不变性的 peer 检查恒 True（A2 的跨请求旁路不再被验）",
        "old": '        "peer_request_logits_equal": bool(torch.equal(base[0], got_peer[0])),',
        "new": '        "peer_request_logits_equal": True,   # 变异：旁路检查被架空',
        "tests": ["test_prefix_invariance_check_flags_noncausal_implementation"],
    },
    {
        "id": "M2",
        "what": "方向标签符号反转（改善↔变差）",
        "old": '    if delta_bpc < -tol:\n        return "改善（因果口径质量代价更小）"',
        "new": '    if delta_bpc < -tol:\n        return "变差（因果口径质量代价更大）"   # 变异：符号反转',
        "tests": ["test_direction_label_signs"],
    },
    {
        "id": "M3",
        "what": "quantized_arms_of 去掉 mode 过滤（把所有臂当量化臂）",
        "old": '        if spec.get("mode") in ("int8", "int4"):',
        "new": '        if True:   # 变异：不区分量化臂',
        "tests": ["test_quantized_arms_of_selects_only_int_arms"],
    },
    {
        "id": "M4",
        "what": "非量化臂漂移不判 False（可控性证据恒真）",
        "old": '        "nonquant_bitwise_preserved": all(abs(r["delta_bpc"]) <= tol for r in nonq),',
        "new": '        "nonquant_bitwise_preserved": True,   # 变异：漂移不报',
        "tests": ["test_compare_detects_nonquant_drift"],
    },
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_file() -> dict:
    r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", TESTS],
                            cwd=ROOT, capture_output=True, text=True)
    return {"returncode": r.returncode,
            "tail": r.stdout.strip().splitlines()[-1][:160]}


def run_tests(names: list[str]) -> dict:
    out = {}
    for n in names:
        r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q",
                                 f"{TESTS}::{n}"],
                                cwd=ROOT, capture_output=True, text=True)
        out[n] = {"returncode": r.returncode,
                  "verdict": "FAILED（抓红）" if r.returncode != 0 else "passed（未抓到）",
                  "tail": r.stdout.strip().splitlines()[-1][:160]}
    return out


def main():
    original = open(TARGET).read()
    sha_before = sha(TARGET)
    report = {"target": os.path.relpath(TARGET, ROOT), "sha256_before": sha_before,
              "mutations": []}
    try:
        for m in MUTATIONS:
            assert original.count(m["old"]) == 1, f"{m['id']}: 定位失败"
            open(TARGET, "w").write(original.replace(m["old"], m["new"]))
            assert sha(TARGET) != sha_before
            res = run_tests(m["tests"])
            caught = all(v["returncode"] != 0 for v in res.values())
            report["mutations"].append({"id": m["id"], "what": m["what"],
                                        "tests": res, "all_expected_tests_caught": caught})
            print(f"[{m['id']}] {m['what']}: "
                  f"{'全部抓红 ✓' if caught else '未抓到 ✗'}", flush=True)
            open(TARGET, "w").write(original)          # 立即还原
            assert sha(TARGET) == sha_before
        report["restored_sha256_equal"] = sha(TARGET) == sha_before
        report["after_restore"] = run_file()
        report["green_after_restore"] = report["after_restore"]["returncode"] == 0
    finally:
        open(TARGET, "w").write(original)              # 任何异常也还原
    report["sha256_after"] = sha(TARGET)
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
