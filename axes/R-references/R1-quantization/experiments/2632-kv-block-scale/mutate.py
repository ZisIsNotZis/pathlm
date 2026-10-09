"""2632 · 标尺方案语义层（`kv_quant.py` / `block_arms.py` / `scale_caliber.py`）的**变异验证**。

对三个模块做**故意破坏**，跑 `tests/test_kv_block_scale.py::<name>`，要求全部抓红；
每次破坏后立即逐字节还原（sha256 校验）。统一走 `tools.evalkit.mutation`（F3）：
只有 `test_failure` 算抓红，「测试没跑起来 / 环境错误」直接非零退出；未抓到 ⇒ 非零退出。

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

KVQ = os.path.join(ROOT, "tools/evalkit/kv_quant.py")
ARMS = os.path.join(HERE, "block_arms.py")
CAL = os.path.join(HERE, "scale_caliber.py")
TESTS = "tests/test_kv_block_scale.py"

MUTATIONS = [
    {
        "id": "M1", "target": KVQ, "what": "块标尺的 amax 归约轴写错（head_dim 块 → 块内错配）",
        "old": "    s = (xr.abs().amax(dim=-1, keepdim=True) / qmax).clamp_min(1e-12)",
        "new": "    s = (xr.abs().amax(dim=-2, keepdim=True) / qmax).clamp_min(1e-12)  # 变异：轴错",
        "tests": ["test_block_error_bound_is_half_of_the_block_scale"],
    },
    {
        "id": "M2", "target": KVQ, "what": "block 路由到 token 轴块（前缀确定被破坏）",
        "old": '    if scheme == "block":\n        return _head_dim_block_quant(x, qmax, group)',
        "new": '    if scheme == "block":\n        return _token_block_quant(x, qmax, group)  # 变异',
        "tests": ["test_block_prefix_invariance_future_token_and_peer_request"],
    },
    {
        "id": "M3", "target": KVQ, "what": "块标尺当免费（摊薄被架空）",
        "old": '    if scheme == "block":\n        _require_group(group, head_dim, "head_dim")\n        return SCALE_BYTES / group',
        "new": '    if scheme == "block":\n        _require_group(group, head_dim, "head_dim")\n        return 0.0  # 变异：标尺当免费',
        "tests": ["test_block_scale_bytes_per_value_is_two_over_group",
                  "test_deployment_bytes_reconcile_and_shrink"],
    },
    {
        "id": "M4", "target": KVQ, "what": "kivi 的 K 也走 per-token（K per-channel 语义丢失）",
        "old": '        if is_key:\n            return _token_block_quant(x, qmax, kivi_group)',
        "new": '        if is_key:\n            return _head_dim_block_quant(x, qmax, x.shape[-1])  # 变异',
        "tests": ["test_kivi_k_per_channel_is_not_prefix_invariant"],
    },
    {
        "id": "M5", "target": ARMS, "what": "前向忽略 scheme（标尺方案接不进去）",
        "old": '_s=scheme, _g=group, _kg=kivi_group):',
        "new": '_s="per_value", _g=group, _kg=kivi_group):  # 变异：忽略 scheme',
        "tests": ["test_make_logits_fn_honours_scheme_prefix_invariance"],
    },
    {
        "id": "M6", "target": CAL, "what": "值字节与判据 C1 对账被绕过（口径漂移也出数）",
        "old": '    if abs(value_part - float(point["bytes_per_position"])) > 1e-6:',
        "new": '    if False:  # 变异：不校验旧值字节',
        "tests": ["test_deployment_bytes_rejects_value_caliber_mismatch"],
    },
    {
        "id": "M7", "target": CAL, "what": "mode 与 value_bytes 标签不符不再报错",
        "old": '        if abs(vb - VALUE_BYTES[mode]) > 1e-9:',
        "new": '        if False:  # 变异：不校验标签',
        "tests": ["test_deployment_bytes_reject_mode_value_bytes_mismatch"],
    },
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(names):
    out = {}
    for n in names:
        r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q",
                                 f"{TESTS}::{n}"], cwd=ROOT, capture_output=True, text=True)
        out[n] = {"returncode": r.returncode,
                  "verdict": "FAILED（抓红）" if r.returncode != 0 else "passed（未抓到）",
                  "tail": r.stdout.strip().splitlines()[-1][:160]}
    return out


def run_file():
    r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", TESTS],
                            cwd=ROOT, capture_output=True, text=True)
    return {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-1][:160]}


def main():
    originals = {p: open(p).read() for p in (KVQ, ARMS, CAL)}
    sha_before = {p: sha(p) for p in originals}
    report = {"targets": {os.path.relpath(p, ROOT): {"sha256_before": sha_before[p]}
                          for p in originals}, "mutations": []}
    try:
        for m in MUTATIONS:
            path, original = m["target"], originals[m["target"]]
            assert original.count(m["old"]) == 1, f"{m['id']}: 定位失败"
            open(path, "w").write(original.replace(m["old"], m["new"]))
            assert sha(path) != sha_before[path]
            res = run_tests(m["tests"])
            caught = all(v["returncode"] != 0 for v in res.values())
            report["mutations"].append({"id": m["id"], "target": os.path.relpath(path, ROOT),
                                        "what": m["what"], "tests": res,
                                        "all_expected_tests_caught": caught})
            print(f"[{m['id']}] {m['what']}: {'全部抓红 ✓' if caught else '未抓到 ✗'}",
                  flush=True)
            open(path, "w").write(original)
            assert sha(path) == sha_before[path]
        report["restored_sha256_equal"] = all(sha(p) == sha_before[p] for p in originals)
        report["after_restore"] = run_file()
        report["green_after_restore"] = report["after_restore"]["returncode"] == 0
    finally:
        for path, original in originals.items():
            open(path, "w").write(original)
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
