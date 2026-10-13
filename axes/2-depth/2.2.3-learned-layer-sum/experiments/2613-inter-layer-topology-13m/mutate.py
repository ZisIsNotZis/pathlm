"""2613 · 层间拓扑 —— 关键开关的**变异验证**（docs/methodology.md §5）。

对 `pathlm/depth_ar.py` 做三处**故意破坏**，跑定点单测，要求全部抓红；
每次破坏后立即还原（脚本用原文/破坏文对照，还原后校验 sha256 一致）。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | `forward` 忽略 dense 拓扑开关（层间接线退回顺序残差） | `test_dense_layer_mix_formula`、`..._activity_escapes_identity` |
| M2 | `ParallelBlock` 实现成**链式**子层（把并行接线的 mlp 分支接到 `x+Δ_attn` 上） | `test_parallel_block_formula_and_stock_path_guard` |
| M3 | `_layer_mix_input` 忽略层权 α（直接返回上一层输出） | `test_dense_layer_mix_formula`、`..._activity_escapes_identity` |

用法：`python3 mutate.py`（纯 CPU，<1 min）。
"""

import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from tools.evalkit import mutation                     # noqa: E402
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")

MUTATIONS = [
    {
        "id": "M1",
        "what": "forward 忽略 dense 拓扑开关（退回顺序残差接线）",
        "old": "        elif self.cfg.topology == \"dense\":\n"
               "            # dense 拓扑：每层输入 = 对所有此前层输出的学习加权和（再进 Block）",
        "new": "        elif False:\n"
               "            # dense 拓扑：每层输入 = 对所有此前层输出的学习加权和（再进 Block）",
        "tests": ["test_dense_layer_mix_formula",
                  "test_dense_layer_mix_activity_escapes_identity"],
    },
    {
        "id": "M2",
        "what": "ParallelBlock 实现成链式子层（mlp 分支读 x+Δ_attn 而非块输入）",
        "old": "        h2 = h1 if self.share_ln else self.ln2(x)   # mlp 分支：同一块输入 x",
        "new": "        h2 = self.ln1(x + self.proj(a))   # 变异：链式（读 x + Δ_attn）",
        "tests": ["test_parallel_block_formula_and_stock_path_guard"],
    },
    {
        "id": "M3",
        "what": "_layer_mix_input 忽略层权 α（直接返回上一层输出 = 干预死）",
        "old": "        off = k * (k - 1) // 2\n"
               "        alpha = self.layer_mix[off:off + k].to(states[0].dtype)\n"
               "        out = alpha[0] * states[0]\n"
               "        for j in range(1, k):\n"
               "            out = out + alpha[j] * states[j]\n"
               "        return out",
        "new": "        return states[k - 1]   # 变异：忽略 α",
        "tests": ["test_dense_layer_mix_formula",
                  "test_dense_layer_mix_activity_escapes_identity"],
    },
]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(names: list[str]) -> dict:
    out = {}
    for n in names:
        r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", "-x",
                            f"tests/test_depth_ar.py::{n}"],
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
        # 还原后回归：全部新单测必须绿
        report["restored_sha256_equal"] = sha(TARGET) == sha_before
        report["after_restore"] = run_tests(
            ["test_topology_default_off_matches_v1_bitwise",
             "test_parallel_block_formula_and_stock_path_guard",
             "test_parallel_param_count_and_init_parity",
             "test_parallel_wiring_changes_computation",
             "test_dense_identity_start_bitwise_equals_sequential",
             "test_dense_layer_mix_formula",
             "test_dense_layer_mix_activity_escapes_identity",
             "test_topology_guard_requires_dense_and_known_value"])
        report["green_after_restore"] = all(
            v["returncode"] == 0 for v in report["after_restore"].values())
    finally:
        open(TARGET, "w").write(original)             # 任何异常也还原
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
