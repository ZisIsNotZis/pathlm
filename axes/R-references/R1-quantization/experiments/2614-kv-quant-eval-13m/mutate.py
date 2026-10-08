"""2614 · KV 量化参照线 —— 关键开关（量化/反量化语义）的**变异验证**。

对 `tools/evalkit/kv_quant.py` 做四处**故意破坏**，跑定点单测，要求全部抓红；
每次破坏后立即还原（原文/破坏文对照 + sha256 校验）。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | qmax 差一（`2^(bits-1)` 而非 `2^(bits-1)-1`）⇒ scale 与端点错 | `test_int_roundtrip_error_within_half_scale` |
| M2 | 因果 running max 退回**非因果全序列 amax**（复核 A2 的原始缺陷） | `test_prefix_invariance_future_token_does_not_change_past_values`、`test_prefix_invariance_other_requests_do_not_change_this_request` |
| M3 | fp16 档不真降精度（不 cast，直接返回 fp32） | `test_fp16_relative_bound_and_actually_changes_values` |
| M4 | 只取整不反量化（`q` 直接当 `x̂`） | `test_int_roundtrip_error_within_half_scale` |

用法：`python3 mutate.py`（纯 CPU，<1 min）。
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
TARGET = os.path.join(ROOT, "tools", "evalkit", "kv_quant.py")
TESTS = "tests/test_kv_quant.py"

MUTATIONS = [
    {
        "id": "M1",
        "what": "qmax 差一（2^(bits-1) 而非 2^(bits-1)-1）",
        "old": "        qmax = 2 ** (bits - 1) - 1",
        "new": "        qmax = 2 ** (bits - 1)   # 变异：qmax 差一",
        "tests": ["test_int_roundtrip_error_within_half_scale[int8-127]",
                  "test_int_roundtrip_error_within_half_scale[int4-7]"],
    },
    {
        "id": "M2",
        "what": "因果 running max 退回非因果全序列 amax（A2 的原始缺陷）",
        "old": "        running_amax = x.abs().cummax(dim=x.dim() - 2).values\n"
               "        scale = (running_amax / qmax).clamp_min(1e-12)",
        "new": ("        reduce = tuple(i for i in range(x.dim()) if i not in "
                "(x.dim() - 1, x.dim() - 3))   # 变异：退回非因果全序列 amax\n"
                "        scale = (x.abs().amax(dim=reduce, keepdim=True)"
                " / qmax).clamp_min(1e-12)"),
        "tests": ["test_prefix_invariance_future_token_does_not_change_past_values",
                  "test_prefix_invariance_other_requests_do_not_change_this_request"],
    },
    {
        "id": "M3",
        "what": "fp16 档不真降精度（不 cast）",
        "old": "        dq = x.to(torch.float16).to(x.dtype)",
        "new": "        dq = x   # 变异：fp16 档不降精度",
        "tests": ["test_fp16_relative_bound_and_actually_changes_values"],
    },
    {
        "id": "M4",
        "what": "只取整不反量化（q 直接当 x̂）",
        "old": "        dq = q * scale",
        "new": "        dq = q   # 变异：跳过反量化",
        "tests": ["test_int_roundtrip_error_within_half_scale[int8-127]",
                  "test_int_roundtrip_error_within_half_scale[int4-7]"],
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
                                        "tests": res,
                                        "all_expected_tests_caught": caught})
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
    # F3：未抓到 / 未逐字节还原 ⇒ 非零退出（旧版本此处只打印，退出码恒为 0）
    mutation.exit_if_uncaught(
        bool(report["all_caught"] and report["restored_sha256_equal"]),
        report, os.path.join(HERE, "mutation_report.json"),
        message="有变异未被抓红，或还原不是逐字节")


if __name__ == "__main__":
    main()
