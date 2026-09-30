"""2616 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：`pathlm/depth_ar.py` 的 KV 头共享开关（ticket 33 契约 S，本实验新实现），
外加 `train_depth_ar.py` 的 `--n-kv-heads` 通道。破坏点覆盖契约 S 复核清单里的
两类风险：**「只改标签没改张量」**（形状类干预的假实现）与**「开关是死的」**
（配置传不进来 / 守卫被拆掉）。

用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")
TRAINER = os.path.join(ROOT, "train_depth_ar.py")
TFILE = "tests/test_depth_ar.py"

#: depth_ar.py 里的锚点原文
QKV_WIDE = "        self.qkv = nn.Linear(d, d + 2 * self.kv_dim)"
GQA_ATTN = """        qs, ks, vs = shape(q, self.n_heads), shape(k, self.n_kv_heads), shape(v, self.n_kv_heads)
        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                           is_causal=attn_mask is None,
                                           enable_gqa=True)"""
KV_DIM = "        self.kv_dim = self.n_kv_heads * self.head_dim"
REDUCTION = '        "kv_reduction_frac": round(1 - nkv / nh, 6),'
LIVE_SHAPE = "        \"k_batch_shape\": layers[0][\"k_shape\"],"
#: S6 只改那一个条件表达式（保持语法合法，否则只能抓到 SyntaxError 这种弱红）
GUARD = "                if not cfg.dense:"
BLOCK_PICK = """        elif cfg.n_kv_heads is not None and cfg.n_kv_heads < cfg.n_heads:"""
TRAIN_ARG = """                        n_kv_heads=args.n_kv_heads)"""

#: (id, 说明, 目标文件, 原文片段, 破坏后片段, 期望抓红的单测, 是否期望抓红)
MUTATIONS = [
    ("S1", "只改标签：qkv 投影仍输出 3d（KV 头数没真的进张量宽度）",
     TARGET, QKV_WIDE, "        self.qkv = nn.Linear(d, 3 * d)",
     ["test_kv_heads_actual_tensor_shapes_and_analytic_memory",
      "test_kv_heads_enable_gqa_kernel_equals_repeat_interleave",
      "test_kv_heads_default_off_matches_v1_bitwise"], True),
    ("S2", "KV 头分区错位：按组广播改成「重复首个 KV 头」（= MQA 语义，形状合法但头没用上）",
     TARGET, GQA_ATTN,
     """        qs, ks, vs = shape(q, self.n_heads), shape(k, self.n_kv_heads), shape(v, self.n_kv_heads)
        ks = ks[:, :1].repeat(1, self.n_kv_heads, 1, 1)
        vs = vs[:, :1].repeat(1, self.n_kv_heads, 1, 1)
        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                           is_causal=attn_mask is None,
                                           enable_gqa=True)""",
     ["test_kv_heads_enable_gqa_kernel_equals_repeat_interleave"], True),
    ("S3", "头维度算错：kv_dim 用 n_heads·dh 而不是 n_kv_heads·dh（张量宽度与标签不符）",
     TARGET, KV_DIM,
     "        self.kv_dim = self.n_heads * self.head_dim",
     ["test_kv_heads_guards_require_dense_sequential_and_divisibility",
      "test_kv_heads_actual_tensor_shapes_and_analytic_memory"], True),
    ("S4", "解析内存下降比被写死成 0（省内存的主张失去算术支撑）",
     TARGET, REDUCTION, '        "kv_reduction_frac": 0.0,',
     ["test_kv_heads_actual_tensor_shapes_and_analytic_memory"], True),
    ("S5", "活性证据造假：k_batch_shape 直接按配置写 [B, n_heads, T, dh]（只看标签）",
     TARGET, LIVE_SHAPE,
     "        \"k_batch_shape\": [tokens.shape[0], nh, tokens.shape[1], dh],",
     ["test_kv_heads_actual_tensor_shapes_and_analytic_memory"], True),
    ("S6", "拆掉 dense 守卫（非 dense 形态也放行 GQA ⇒ 退出/填充语义未定义）",
     TARGET, GUARD, "                if False and not cfg.dense:",
     ["test_kv_heads_guards_require_dense_sequential_and_divisibility"], True),
    ("S7", "块选择条件失效：n_kv_heads < n_heads 时仍建 stock Block（开关是死的）",
     TARGET, BLOCK_PICK,
     "        elif False:",
     ["test_kv_heads_guards_require_dense_sequential_and_divisibility",
      "test_kv_heads_actual_tensor_shapes_and_analytic_memory",
      "test_kv_heads_default_off_matches_v1_bitwise"], True),
    ("S8", "训练器没把 --n-kv-heads 传给配置（CLI 开关是死的）",
     TRAINER, TRAIN_ARG, "                        n_kv_heads=None)",
     ["test_kv_heads_default_off_matches_v1_bitwise"], True),
]

#: S8 的抓红靠训练器 + 深度 AR 的接口一致性单测，见 tests/test_depth_ar.py::test_train_cli_kv_heads
MUTATION_EXTRA_TESTS = {
    "S8": ["test_train_cli_kv_heads_plumbed_to_config"],
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    report = {"targets": {}, "mutations": []}
    digests = {}
    for path in (TARGET, TRAINER):
        src = open(path).read()
        digests[path] = (src, sha256(src))
        report["targets"][os.path.relpath(path, ROOT)] = {"sha256_before": sha256(src)}
    for mid, desc, path, old, new, tests, expect_red in MUTATIONS:
        src, digest0 = digests[path]
        tests = MUTATION_EXTRA_TESTS.get(mid, tests)
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段在 {os.path.basename(path)} 里出现 "
                             f"{src.count(old)} 次，无法唯一定位")
        open(path, "w").write(src.replace(old, new))
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-k", " or ".join(tests)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        caught = r.returncode != 0
        red = sorted({ln.split("::")[1].split(" ")[0] for ln in r.stdout.splitlines()
                      if ln.startswith("FAILED ") and "::" in ln})
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        report["mutations"].append({
            "id": mid, "desc": desc,
            "target": os.path.relpath(path, ROOT),
            "expected_tests": tests, "expect_red": expect_red, "caught_red": caught,
            "outcome_ok": caught == expect_red,
            "tests_actually_red": red,
            "pytest_summary": tail[0] if tail else "",
        })
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'} — {desc} — "
              f"{tail[0] if tail else ''}", flush=True)
        open(path, "w").write(src)                                  # 逐字节还原
        if sha256(open(path).read()) != digest0:
            raise SystemExit(f"{mid}: {os.path.basename(path)} 还原失败（sha256 不一致）")
    for path, (_, digest0) in digests.items():
        report["targets"][os.path.relpath(path, ROOT)]["sha256_after"] = \
            sha256(open(path).read())
        report["targets"][os.path.relpath(path, ROOT)]["restored_bitwise"] = \
            report["targets"][os.path.relpath(path, ROOT)]["sha256_after"] == digest0
    report["all_restored_bitwise"] = all(v["restored_bitwise"]
                                        for v in report["targets"].values())
    report["all_expected_outcomes"] = all(m["outcome_ok"] for m in report["mutations"])
    report["all_caught"] = all(m["caught_red"] for m in report["mutations"]
                               if m["expect_red"])
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in
                      ("all_restored_bitwise", "all_caught",
                       "all_expected_outcomes")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
