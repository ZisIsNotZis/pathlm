"""2615 · 三种压缩优先级对决 —— 关键开关的**变异验证**（ticket 32 契约 N）。

对 `tools/evalkit/retention.py` 的**本实验新增**实现做十处**故意破坏**，跑定点单测，
要求全部抓红；每次破坏后立即还原并校验 sha256（原文/破坏文对照）。

| 变异 | 破坏内容 | 期望抓红的单测 |
|---|---|---|
| M1 | `attention_stats` 的因果掩码换成全 1（统计量泄漏未来） | 手写参考 / 因果与归一化 |
| M2 | 注意力质量**不排除对角线**（自己读自己当成被读过） | 手写参考 / 因果与归一化 |
| M3 | 注意力质量**按头求和**而非平均 | 手写参考 / 因果与归一化 |
| M4 | 优先级排序方向翻反（删「被读得最多」的） | 规则语义（含平局次序） |
| M5 | `drop_mask_keep` 丢对角线（被删 query 整行全掩码） | 与 `prefix_keep_mask` 逐位一致 |
| M6 | `drop_mask_keep` 把删除集套在**查询轴**而非 key 轴 | 同上 |
| M7 | `_rank` 取排序**尾部**（`distance` 变成删最近的） | 预算/规则语义 |
| M8 | `_rank` 忽略候选池（诊断臂「只压正文窗口」失效） | 候选池预算测试 |
| M9 | `drop_mask_overlap` 把交集算成并集 | 重叠度测试 |
| M10 | `drop_mask_stats` 把「离得近」当代价（`T−1−j` → `j`） | 与 `mask_stats` 同口径 |

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
        "what": "attention_stats 用全 1 因果掩码（统计量看到未来）",
        "old": "    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)                  # [T, T]",
        "new": "    causal = torch.ones(T, T, dtype=torch.bool, device=dev)       # 变异：全可见",
        "tests": ["test_attention_stats_match_manual_reference",
                  "test_attention_stats_are_causal_and_normalized"],
    },
    {
        "id": "M2",
        "what": "注意力质量不排除对角线（自己读自己算被读过）",
        "old": "        wo = w * off_diag                                             # 排除自身读出",
        "new": "        wo = w                                                        # 变异：含对角线",
        "tests": ["test_attention_stats_match_manual_reference",
                  "test_attention_stats_are_causal_and_normalized"],
    },
    {
        "id": "M3",
        "what": "注意力质量按头求和而不是逐头平均",
        "old": "        w = w * nh_inv                                                # 逐头平均",
        "new": "        w = w * 1.0                                                   # 变异：按头求和",
        "tests": ["test_attention_stats_match_manual_reference",
                  "test_attention_stats_are_causal_and_normalized"],
    },
    {
        "id": "M4",
        "what": "优先级排序方向翻反（删被读得最多的位置）",
        "old": "        order = sorted(range(T), key=lambda j: (p[b][j], s[b][j], j))",
        "new": "        order = sorted(range(T), key=lambda j: (-p[b][j], -s[b][j], j))"
               "   # 变异：方向翻反",
        "tests": ["test_priority_rules_delete_by_their_own_key"],
    },
    {
        "id": "M5",
        "what": "drop_mask_keep 丢掉对角线（被删 query 整行全掩码）",
        "old": "    return keep | torch.eye(T, dtype=torch.bool, device=drop.device).unsqueeze(0)",
        "new": "    return keep                                                   # 变异：丢对角线",
        "tests": ["test_drop_mask_keep_matches_the_prefix_baseline_and_never_empties_a_row"],
    },
    {
        "id": "M6",
        "what": "drop_mask_keep 把删除集套在查询轴（删 query 而不是删 key）",
        "old": "    keep = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0) & ~drop.unsqueeze(1)",
        "new": "    keep = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0) & ~drop.unsqueeze(2)",
        "tests": ["test_drop_mask_keep_matches_the_prefix_baseline_and_never_empties_a_row"],
    },
    {
        "id": "M7",
        "what": "_rank 取排序尾部（distance 变成删最近的 N 个）",
        "old": "        for j in order[:n_drop]:",
        "new": "        for j in order[-n_drop:]:                                     # 变异：取尾部",
        "tests": ["test_priority_rules_spend_the_same_budget_and_are_deterministic",
                  "test_priority_rules_delete_by_their_own_key"],
    },
    {
        "id": "M8",
        "what": "_rank 忽略候选池（「只压正文窗口」的诊断臂失效）",
        "old": "        primary = torch.where(candidates.view(1, T), primary, inf)",
        "new": "        primary = primary                                             # 变异：忽略候选池",
        "tests": ["test_priority_candidates_keep_the_budget_and_only_shrink_the_pool"],
    },
    {
        "id": "M9",
        "what": "drop_mask_overlap 把交集算成并集",
        "old": "    inter = (a & b).sum(-1).double()",
        "new": "    inter = (a | b).sum(-1).double()                              # 变异：并集冒充交集",
        "tests": ["test_deleted_set_overlap_reports_intersection_and_jaccard"],
    },
    {
        "id": "M10",
        "what": "drop_mask_stats 把「离得近」当被删代价（T−1−j → j）",
        "old": "    pairs = (drop.to(torch.float64) * (T - 1 - idx).view(1, T)).sum(-1)   # [B]",
        "new": "    pairs = (drop.to(torch.float64) * idx.view(1, T)).sum(-1)   # 变异：代价反了",
        "tests": ["test_drop_mask_stats_matches_mask_stats_on_a_shared_drop_set"],
    },
]

GREEN_SET = ["test_attention_stats_match_manual_reference",
             "test_attention_stats_are_causal_and_normalized",
             "test_priority_rules_spend_the_same_budget_and_are_deterministic",
             "test_priority_rules_delete_by_their_own_key",
             "test_drop_mask_keep_matches_the_prefix_baseline_and_never_empties_a_row",
             "test_deleted_set_overlap_reports_intersection_and_jaccard",
             "test_priority_rules_move_the_outcome_liveness",
             "test_attention_stats_summary_reports_the_distribution",
             "test_per_sample_masks_reach_their_own_window_in_bpc_and_needles",
             "test_priority_candidates_keep_the_budget_and_only_shrink_the_pool",
             "test_drop_mask_stats_matches_mask_stats_on_a_shared_drop_set"]


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(names: list[str]) -> dict:
    out = {}
    for n in names:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", f"{TESTS}::{n}"],
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
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"green_after_restore": report["green_after_restore"],
                      "restored_sha256_equal": report["restored_sha256_equal"],
                      "all_caught": all(m["all_expected_tests_caught"]
                                        for m in report["mutations"])}, indent=2))


if __name__ == "__main__":
    main()
