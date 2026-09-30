"""2614 · KV 消融保留曲线 —— 关键开关（删除位置集合）的**变异验证**。

对 `tools/evalkit/retention.py` 做四处**故意破坏**（都是「删错位置」），跑定点
单测，要求全部抓红；每次破坏后立即还原（原文/破坏文对照 + sha256 校验）。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | `distance_keep_mask` 距离差一（删距离 d+1 而不是 d） | `test_distance_mask_deletes_exactly_distance_d` |
| M2 | `prefix_keep_mask` 删**最新**的 N 个而不是最远的 N 个 | `test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal` |
| M3 | `segment_keep_mask` 忽略起点（永远删 `[0, L)`） | `test_segment_mask_deletes_contiguous_block_only` |
| M4 | 去掉对角线保留（前缀删除时把 query 自身也删掉） | `test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal` |

用法：`python3 mutate.py`（纯 CPU，<1 min）。
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "tools", "evalkit", "retention.py")
TESTS = "tests/test_retention.py"

MUTATIONS = [
    {
        "id": "M1",
        "what": "distance_keep_mask 距离差一（删距离 d+1）",
        "old": "    return (j <= i) & ((i - j) != d)",
        "new": "    return (j <= i) & ((i - j) != d + 1)   # 变异：删错距离",
        "tests": ["test_distance_mask_deletes_exactly_distance_d"],
    },
    {
        "id": "M2",
        "what": "prefix_keep_mask 删最新的 N 个（应为最远的 N 个）",
        "old": "    dropped_key = j < n_dropped                       # [1, T] 被删的绝对位置",
        "new": "    dropped_key = j >= T - n_dropped                 # 变异：删错一端",
        "tests": ["test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal"],
    },
    {
        "id": "M3",
        "what": "segment_keep_mask 忽略随机起点（永远删 [0, L)）",
        "old": "    in_segment = (j >= start) & (j < start + length)   # [1, T]",
        "new": "    in_segment = j < length                            # 变异：忽略 start",
        "tests": ["test_segment_mask_deletes_contiguous_block_only"],
    },
    {
        "id": "M4",
        "what": "去掉对角线保留（前缀删除时 query 自身也被删）",
        "old": "    dropped_key = j < n_dropped                       # [1, T] 被删的绝对位置\n"
               "    keep = (j <= i) & ~dropped_key\n"
               "    return keep | torch.eye(T, dtype=torch.bool, device=device)",
        "new": "    dropped_key = j < n_dropped                       # [1, T] 被删的绝对位置\n"
               "    keep = (j <= i) & ~dropped_key\n"
               "    return keep   # 变异：不保对角线",
        "tests": ["test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal"],
    },
]

GREEN_SET = ["test_distance_mask_deletes_exactly_distance_d",
             "test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal",
             "test_segment_mask_deletes_contiguous_block_only",
             "test_mask_stats_reports_dropped_pair_activity",
             "test_masked_logits_matches_manual_reference",
             "test_identity_causal_mask_bitwise_equals_release_forward",
             "test_deletion_is_live_changes_logits_and_raises_bpc"]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(names: list[str]) -> dict:
    out = {}
    for n in names:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
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
        report["after_restore"] = run_tests(GREEN_SET)
        report["green_after_restore"] = all(
            v["returncode"] == 0 for v in report["after_restore"].values())
    finally:
        open(TARGET, "w").write(original)              # 任何异常也还原
    report["sha256_after"] = sha(TARGET)
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"green_after_restore": report["green_after_restore"],
                      "restored_sha256_equal": report["restored_sha256_equal"],
                      "all_caught": all(m["all_expected_tests_caught"]
                                        for m in report["mutations"])}, indent=2))


if __name__ == "__main__":
    main()
