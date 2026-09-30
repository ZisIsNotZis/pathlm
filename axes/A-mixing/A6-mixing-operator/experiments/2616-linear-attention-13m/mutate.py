"""2616 · 变异验证（契约 R：每个语义开关要有可失败单测，且变异必须被抓红）。

对 `pathlm/depth_ar.py`（线性注意力实现）做 6 处**故意破坏**，每次跑一组针对性单测，
要求**抓红**（pytest 返回非 0），随后从备份**逐字节还原**并核对 sha256。

| # | 变异 | 破坏的语义 | 应该抓红谁 |
|---|---|---|---|
| 1 | `noncausal-intra` | 块内不去三角（`tril`）⇒ 看到未来 | 因果性、与参考实现一致 |
| 2 | `inclusive-state` | 块间状态用 inclusive cumsum ⇒ 看到本块未来 | 因果性、与参考实现一致 |
| 3 | `phi-no-plus-one` | φ = elu（去掉 +1） | 特征映射公式、活性（φ>0/den>0） |
| 4 | `full-attention-rename` | 线性块直接调 SDPA（换个名字的全注意力） | O(T) FLOP 斜率、与参考实现一致、同权重换算子偏差 > 0 |
| 5 | `no-normalize` | 去掉归一化分母 | 与参考实现一致 |
| 6 | `instrument-ignores-mask` | 仪器的掩码路径忽略掩码 | 掩码真的删键（评测期依赖曲线的语义前提）|

用法（纯 CPU，<3 min；**不要与训练子进程并发**）：
    python3 mutate.py
"""

import hashlib, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")

MUTATIONS = [
    {"name": "noncausal-intra",
     "desc": "块内因果打分去掉 tril ⇒ 看到未来（非因果）",
     "old": '    intra = torch.einsum("bhnid,bhnjd->bhnij", qc, kc).tril()   # 块内因果核分数',
     "new": '    intra = torch.einsum("bhnid,bhnjd->bhnij", qc, kc)   # MUTATED: no tril',
     "tests": "-k 'linear_attn_causal or linear_chunked_matches'"},
    {"name": "inclusive-state",
     "desc": "块间状态用 inclusive cumsum ⇒ 本块状态泄漏到本块输出",
     "old": "    S_prev = torch.cat([zero_s, S_chunk.cumsum(2)[:, :, :-1]], dim=2)\n"
            "    z_prev = torch.cat([zero_z, z_chunk.cumsum(2)[:, :, :-1]], dim=2)",
     "new": "    S_prev = S_chunk.cumsum(2)   # MUTATED: inclusive\n"
            "    z_prev = z_chunk.cumsum(2)   # MUTATED: inclusive",
     "tests": "-k 'linear_attn_causal or linear_chunked_matches'"},
    {"name": "phi-no-plus-one",
     "desc": "φ = elu（去掉 +1）⇒ 特征映射定义被改",
     "old": "    return F.elu(x) + 1.0",
     "new": "    return F.elu(x)   # MUTATED: no +1",
     "tests": "-k 'linear_phi_formula or linear_switch_changes'"},
    {"name": "full-attention-rename",
     "desc": "线性块直接调 SDPA（换个名字的全注意力）",
     "old": "        if attn_mask is None:\n"
            "            a, tele = linear_attn_chunked(qf, kf, vs, self.chunk)",
     "new": "        if attn_mask is None:   # MUTATED: full attention renamed\n"
            "            a = F.scaled_dot_product_attention(shape(q), shape(k), vs,\n"
            "                                               is_causal=True)\n"
            "            tele = {\"phi_min\": 1.0, \"phi_max\": 1.0, \"den_min\": 1.0,\n"
            "                    \"inter_mass_frac\": 0.0, \"state_norm\": 0.0}",
     "tests": "-k 'linear_attn_is_linear_in_T or linear_chunked_matches or "
              "linear_switch_changes'"},
    {"name": "no-normalize",
     "desc": "去掉归一化分母（核分数不归一化）",
     "old": "    out = (num_inter + num_intra) / den_clamped.unsqueeze(-1)",
     "new": "    out = num_inter + num_intra   # MUTATED: no normalization",
     "tests": "-k 'linear_chunked_matches or linear_masked_reference'"},
    {"name": "instrument-ignores-mask",
     "desc": "仪器的掩码路径忽略掩码（依赖曲线不再量任何东西）",
     "old": "            a, tele = linear_attn_reference(qf, kf, vs, keep=attn_mask), None",
     "new": "            a, tele = linear_attn_reference(qf, kf, vs, keep=None), None"
            "   # MUTATED: mask ignored",
     "tests": "-k 'linear_masked_reference'"},
]


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    orig_bytes = open(TARGET, "rb").read()
    orig_sha = sha256(TARGET)
    backup = TARGET + ".mutbak"
    open(backup, "wb").write(orig_bytes)
    report = {"target": os.path.relpath(TARGET, ROOT), "original_sha256": orig_sha,
              "mutations": []}
    try:
        for mut in MUTATIONS:
            src = open(TARGET, "rb").read().decode()
            if mut["old"] not in src:
                raise SystemExit(f"变异 {mut['name']} 的锚点找不到（实现变了？）")
            open(TARGET, "wb").write(
                src.replace(mut["old"], mut["new"], 1).encode())
            cmd = [sys.executable, "-m", "pytest", "tests/test_depth_ar.py", "-q"] + \
                shlex.split(mut["tests"])   # 带引号的 -k 表达式必须整体传参
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            tail = (p.stdout or "").strip().splitlines()[-1:] or [""]
            entry = {"name": mut["name"], "desc": mut["desc"],
                     "pytest_args": mut["tests"], "returncode": p.returncode,
                     "n_tests_collected_line": (p.stdout or "").strip().splitlines()[-1:]
                     if p.returncode == 5 else None,
                     # exit 5 = 没收集到测试 ≠ 抓红；这两者必须分开
                     "result": ("GREEN(BAD)" if p.returncode == 0 else
                                ("NO_TESTS(BAD)" if p.returncode == 5 else "red")),
                     "failed_summary": tail[-1][:200],
                     "mutated_sha256": sha256(TARGET)}
            report["mutations"].append(entry)
            print(f"[mutate] {mut['name']}: {entry['result']} ({tail[-1][:90]})",
                  flush=True)
            # 每次变异后**逐字节还原**
            open(TARGET, "wb").write(orig_bytes)
            entry["restored_sha256"] = sha256(TARGET)
            entry["restored_byte_identical"] = entry["restored_sha256"] == orig_sha
            if not entry["restored_byte_identical"]:
                raise SystemExit(f"还原失败：{mut['name']}")
        # 还原后基线必须全绿
        p = subprocess.run([sys.executable, "-m", "pytest",
                            "tests/test_depth_ar.py", "-q"],
                           cwd=ROOT, capture_output=True, text=True)
        report["baseline_after_restore_returncode"] = p.returncode
        report["baseline_after_restore_tail"] = \
            (p.stdout or "").strip().splitlines()[-1:]
        report["restored_byte_identical"] = all(
            m["restored_byte_identical"] for m in report["mutations"])
        report["all_mutations_caught"] = all(
            m["result"] == "red" for m in report["mutations"])
        report["n_red"] = sum(1 for m in report["mutations"] if m["result"] == "red")
        report["final_sha256"] = sha256(TARGET)
    finally:
        open(TARGET, "wb").write(orig_bytes)
        os.remove(backup)
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "mutations"},
                     ensure_ascii=False), flush=True)
    assert report["all_mutations_caught"], "有变异没被抓红"
    assert report["restored_byte_identical"], "有变异没有逐字节还原"
    assert report["baseline_after_restore_returncode"] == 0, "还原后基线不绿"
    print("[mutate] 6/6 抓红 + 逐字节还原 + 还原后基线绿", flush=True)


if __name__ == "__main__":
    main()
