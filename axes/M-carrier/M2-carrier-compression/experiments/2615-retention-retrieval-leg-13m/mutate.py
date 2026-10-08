"""2615 · 检索腿 —— 关键开关的**变异验证**（ticket 32 契约 M）。

对 `tools/evalkit/retention.py` 的**新增**检索腿实现做四处**故意破坏**（三处是
「针被删错 / 针被放错 / 半删当成没删」，一处是「只保留最近 d」差一），跑定点
单测，要求全部抓红；每次破坏后立即还原并校验 sha256（原文/破坏文对照）。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | `recent_keep_mask` 差一（保留 `<= d` 而不是 `< d`） | `test_recent_keep_mask_keeps_only_last_d_keys` |
| M2 | `needle_fate` 用 `any` 代替 `all`：**半删的针被当成「保住了」** | `test_half_deleted_needle_counts_as_deleted_not_kept` |
| M3 | `phrase_needle_batch` 取消间距约束：**针的位置可以重叠（同一位置被放两次）** | `test_phrase_needle_layout_makes_the_plant_the_only_source` |
| M4 | `phrase_needle_batch` 只植入线索、不植入答案：副本里拿不到答案 | 同上 |

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
TARGET = os.path.join(ROOT, "tools", "evalkit", "retention.py")
TESTS = "tests/test_retention.py"

MUTATIONS = [
    {
        "id": "M1",
        "what": "recent_keep_mask 差一（保留距离 <= d，而不是 < d）",
        "old": "    keep = (j <= i) & ((i - j) < d)",
        "new": "    keep = (j <= i) & ((i - j) <= d)   # 变异：多留一圈",
        "tests": ["test_recent_keep_mask_keeps_only_last_d_keys"],
    },
    {
        "id": "M2",
        "what": "needle_fate 半删当成没删（any 代替 all）",
        "old": "        visible = (keep is None\n"
               "                   or all(bool(keep[q, p + j]) for j in range(phrase_len + 1)))",
        "new": "        visible = (keep is None\n"
               "                   or any(bool(keep[q, p + j]) for j in range(phrase_len + 1)))\n"
               "        # 变异：半删也算可见",
        "tests": ["test_half_deleted_needle_counts_as_deleted_not_kept"],
    },
    {
        "id": "M3",
        "what": "phrase_needle_batch 取消针位间距约束（针可以重叠 ⇒ 同一位置被放两次）",
        "old": "                if any(abs(p - u) < span for u in used):     # 位置互不重叠（间距 ≥ L+1）",
        "new": "                if any(abs(p - u) < 0 for u in used):       # 变异：允许重叠",
        "tests": ["test_phrase_needle_layout_makes_the_plant_the_only_source"],
    },
    {
        "id": "M4",
        "what": "phrase_needle_batch 只植入线索、不植入答案（副本里拿不到答案）",
        "old": "                if plant:\n                    row[p:p + span] = phrase",
        "new": "                if plant:\n                    row[p:p + span - 1] = phrase[:span - 1]   # 变异：丢答案字节",
        "tests": ["test_phrase_needle_layout_makes_the_plant_the_only_source"],
    },
]

GREEN_SET = ["test_distance_mask_deletes_exactly_distance_d",
             "test_prefix_mask_deletes_oldest_n_keys_and_keeps_diagonal",
             "test_segment_mask_deletes_contiguous_block_only",
             "test_mask_stats_reports_dropped_pair_activity",
             "test_masked_logits_matches_manual_reference",
             "test_identity_causal_mask_bitwise_equals_release_forward",
             "test_deletion_is_live_changes_logits_and_raises_bpc",
             "test_recent_keep_mask_keeps_only_last_d_keys",
             "test_phrase_needle_layout_makes_the_plant_the_only_source",
             "test_needle_fate_partitions_each_needle_exactly_once",
             "test_half_deleted_needle_counts_as_deleted_not_kept",
             "test_needle_recall_counts_every_needle_once_and_tracks_the_buckets"]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(names: list[str]) -> dict:
    out = {}
    for n in names:
        r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", f"{TESTS}::{n}"],
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
            report["mutations"].append({"id": m["id"], "what": m["what"], "tests": res,
                                        "all_expected_tests_caught": caught})
            print(f"[{m['id']}] {m['what']}: "
                  f"{'全部抓红 ✓' if caught else '未抓到 ✗'}", flush=True)
            open(TARGET, "w").write(original)          # 立即还原
            assert sha(TARGET) == sha_before
        report["restored_sha256_equal"] = sha(TARGET) == sha_before
        report["after_restore"] = run_tests(GREEN_SET)
        report["green_after_restore"] = all(v["returncode"] == 0
                                            for v in report["after_restore"].values())
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
