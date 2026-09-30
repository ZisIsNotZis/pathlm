"""2614 · 关键开关的变异验证（SCHEMA §17 / methodology §5）：故意破坏必须抓红，
破坏后**逐字节还原**。

五处变异（都只动 `pathlm/depth_ar.py`，逐个独立施加、逐个还原）：

| # | 变异 | 破坏的语义 |
|---|---|---|
| 1 | `ignore-topk` | 块内不做选择（keep 退化为全因果掩码）= 干预关闭 |
| 2 | `positional-select` | 选择改成「最近 k 个键」（位置选择）而不是按内容 |
| 3 | `noncausal-leak` | 掩码用并集（`keep \\| causal`）⇒ 放出非因果键 |
| 4 | `unmasked-softmax` | softmax 不掩掉未选键 ⇒ 稀疏退化为稠密 |
| 5 | `k-ignored` | 建模时忽略 `attn_sparse_k`（块按 k=seq_len 建）⇒ k 没生效 |

每处：施加补丁 → 跑目标单测（必须红）→ 还原 → 校验 sha256 与原文件逐字节一致。
最后在未变异状态下跑一遍目标单测（必须绿）。

用法（<3 min，纯 CPU；**不要在与训练子进程并发的窗口内运行**——新起的训练子进程
会 import 被变异过的模块）：
    python3 mutate.py
"""

import hashlib, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")

T = "tests/test_depth_ar.py"
MUTATIONS = [
    {"id": "ignore-topk",
     "old": "            keep = causal_topk_keep_mask(scores, self.topk, self.causal[:T, :T])",
     "new": "            keep = self.causal[:T, :T]",
     "breaks": "块内不做内容选择（keep = 全因果掩码）⇒ 稀疏干预被关闭",
     "targets": [f"{T}::test_topk_block_formula_matches_manual_sparse_attention",
                 f"{T}::test_topk_switch_changes_computation_and_k_is_effective"]},
    {"id": "positional-select",
     "old": "    idx = allowed.topk(k, dim=-1).indices",
     "new": "    idx = torch.arange(len(causal) - k, len(causal),\n"
            "                       device=scores.device).expand(len(causal), k)",
     "breaks": "选择改成「最近的 k 个键」（位置选择），不是按内容 q·kᵀ",
     "targets": [f"{T}::test_topk_mask_selection_is_by_content_not_position",
                 f"{T}::test_topk_mask_keeps_exactly_min_k_causal_keys"]},
    {"id": "noncausal-leak",
     "old": "    keep = torch.zeros_like(allowed, dtype=torch.bool).scatter_(-1, idx, True)\n"
            "    return keep & causal",
     "new": "    keep = torch.zeros_like(allowed, dtype=torch.bool).scatter_(-1, idx, True)\n"
            "    return keep | causal",
     "breaks": "掩码并集 ⇒ 放出非因果（未来）键",
     "targets": [f"{T}::test_topk_mask_keeps_exactly_min_k_causal_keys",
                 f"{T}::test_topk_softmax_weights_live_only_on_kept_keys"]},
    {"id": "unmasked-softmax",
     "old": "            w = scores.masked_fill(~keep, float(\"-inf\")).softmax(-1)",
     "new": "            w = scores.softmax(-1)",
     "breaks": "softmax 不掩掉未选键 ⇒ 稀疏注意力退化为稠密",
     "targets": [f"{T}::test_topk_block_formula_matches_manual_sparse_attention"]},
    {"id": "k-ignored",
     "old": "            self.blocks = nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k)\n"
            "                                        for _ in range(cfg.n_layers))",
     "new": "            self.blocks = nn.ModuleList(TopKBlock(mcfg, mcfg.seq_len)\n"
            "                                        for _ in range(cfg.n_layers))",
     "breaks": "建模时忽略 attn_sparse_k（k 被设成 seq_len）⇒ k 不生效",
     "targets": [f"{T}::test_topk_switch_changes_computation_and_k_is_effective",
                 f"{T}::test_topk_probe_and_liveness_contract_on_real_batches"]},
]


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run_tests(targets: list[str]) -> tuple[int, str]:
    p = subprocess.run([sys.executable, "-m", "pytest", *targets, "-q",
                        "--no-header", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr)


def main():
    baseline = open(TARGET).read()
    base_sha = sha256(TARGET)
    report = {"target_file": "pathlm/depth_ar.py", "baseline_sha256": base_sha,
              "mutations": [], "restored_byte_identical": True}
    for mut in MUTATIONS:
        assert mut["old"] in baseline, f"变异 {mut['id']}：锚点文本不存在"
        patched = baseline.replace(mut["old"], mut["new"], 1)
        assert patched != baseline, mut["id"]
        open(TARGET, "w").write(patched)
        rc, out = run_tests(mut["targets"])
        failing = sorted({ln.split("::")[-1].split(" ")[0]
                          for ln in out.splitlines() if ln.startswith("FAILED")})
        entry = {"id": mut["id"], "breaks": mut["breaks"],
                 "targets": mut["targets"], "exit_code": rc,
                 "caught_red": rc != 0, "failing_tests": failing,
                 "stdout_tail": out.strip().splitlines()[-1] if out.strip() else ""}
        open(TARGET, "w").write(baseline)
        entry["restored_sha256"] = sha256(TARGET)
        entry["restored_byte_identical"] = entry["restored_sha256"] == base_sha
        if not entry["restored_byte_identical"]:
            report["restored_byte_identical"] = False
        report["mutations"].append(entry)
        print(f"[mutate] {mut['id']}: rc={rc} caught_red={entry['caught_red']} "
              f"failing={failing} restored={entry['restored_byte_identical']}",
              flush=True)
    rc, out = run_tests([f"{T}::test_topk_block_formula_matches_manual_sparse_attention",
                         f"{T}::test_topk_mask_keeps_exactly_min_k_causal_keys",
                         f"{T}::test_topk_mask_selection_is_by_content_not_position",
                         f"{T}::test_topk_softmax_weights_live_only_on_kept_keys",
                         f"{T}::test_topk_switch_changes_computation_and_k_is_effective",
                         f"{T}::test_topk_probe_and_liveness_contract_on_real_batches"])
    report["baseline_green_after_restore"] = rc == 0
    report["baseline_tests_stdout_tail"] = out.strip().splitlines()[-1]
    report["final_sha256"] = sha256(TARGET)
    report["all_mutations_caught_red"] = all(m["caught_red"]
                                             for m in report["mutations"])
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    assert report["all_mutations_caught_red"], "有变异未被抓红"
    assert report["restored_byte_identical"], "还原后与原文件不一致"
    assert report["baseline_green_after_restore"], "还原后基线单测不绿"
    print("[mutate] all mutations caught red; file restored byte-identical; "
          "baseline green", flush=True)


if __name__ == "__main__":
    main()
