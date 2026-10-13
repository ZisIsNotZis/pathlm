"""2624-carry-under-compression-quality · 变异验证（每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

靶子 = 本实验自己的语义层：
- `quant_carry.py`（评测期量化前向 / 分量分解）；
- `compression_arms.py`（三行评测定义 / 块类选择规则）。

引擎三特性（GQA × carry × window）**本线不改**，其活性/变异已由
`2622-eng1-composable/mutate.py` 钉住——这里只动本实验自己的文件，不碰任何共享引擎
文件（SCHEMA §22：避免与并行线抢写）。

五处破坏点各对准一条断言（每个先记原文 sha256，跑指定单测期望**红**，再逐字节还原）：

- M1 `_quant_carry_block`：把 K/V 量化钉死成 `"fp32"`（假压缩 ⇒ 活性量归零）；
- M2 `_attn_mask_of`：把窗口掩码吞掉（`return None` ⇒ 前向与释放版不再逐位等值）；
- M3 `compress_increment`：把 int8 增量符号反掉；
- M4 `expected_block_class`：GQA×carry 仍报 `GQABlock`（假装 ENG1 没解除组合）；
- M5 `EVAL_ARMS`：把压缩行偷换成 fp16（假 int8 臂）。

用法：python3 mutate.py            # 输出 mutation_report.json
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
TFILE = "tests/test_carry_under_compression_quality.py"

MUTATIONS = [
    ("M1", "quant_carry.py", "_quant_carry_block：K/V 量化钉死成 fp32（假压缩）",
     """    ks, ki = quantize_kv(ks, mode)
    if acc is not None:
        _merge(acc, ki)
    vs, vi = quantize_kv(vs, mode)
""",
     """    ks, ki = quantize_kv(ks, "fp32")
    if acc is not None:
        _merge(acc, ki)
    vs, vi = quantize_kv(vs, "fp32")
""",
     "test_int8_changes_values_and_logits"),
    ("M2", "quant_carry.py", "_attn_mask_of：窗口掩码被吞掉（前向不再逐位等值）",
     """    if km.shape == (T, T):
        return km.view(1, 1, T, T).expand(B, n_heads, T, T)
""",
     """    if km.shape == (T, T):
        return None
""",
     "test_fp32_bitwise_matches_released_composable"),
    ("M3", "compression_arms.py", "compress_increment：int8 增量符号反掉",
     """    return {"gqa_vs_mha_pp": round(100.0 * (gqa - mha), 3),
            "int8_vs_fp16_pp": round(100.0 * (int8 - gqa), 3),
            "net_pp": round(100.0 * (int8 - mha), 3)}
""",
     """    return {"gqa_vs_mha_pp": round(100.0 * (gqa - mha), 3),
            "int8_vs_fp16_pp": round(100.0 * (gqa - int8), 3),
            "net_pp": round(100.0 * (int8 - mha), 3)}
""",
     "test_compress_increment_signs"),
    ("M4", "compression_arms.py", "expected_block_class：GQA×carry 仍报 GQABlock",
     """    if gqa and carry:
        return "ComposableBlock"
""",
     """    if gqa and carry:
        return "GQABlock"
""",
     "test_expected_block_classes"),
    ("M5", "compression_arms.py", "EVAL_ARMS：压缩行偷换成 fp16（假 int8 臂）",
     """    {"name": "carry_gqa_int8", "weights": "carry_gqa", "kv_mode": "int8"},
""",
     """    {"name": "carry_gqa_int8", "weights": "carry_gqa", "kv_mode": "fp16"},
""",
     "test_eval_rows_three_int8_shares_gqa_weights"),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    report = {"targets": {}, "mutations": []}
    digests = {}
    for _, fname, _, _, _, _ in MUTATIONS:
        if fname not in digests:
            path = os.path.join(HERE, fname)
            orig = open(path, "rb").read()
            digests[fname] = (path, orig, sha256_bytes(orig))
            report["targets"][fname] = sha256_bytes(orig)
    for mid, fname, desc, old, new, test in MUTATIONS:
        path, orig, digest = digests[fname]
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 片段在 {fname} 出现 {src.count(old)} 次，无法唯一定位")
        open(path, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(path, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-p", "no:cacheprovider",
               "-k", test]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(path, "wb").write(orig)                            # 逐字节还原
        restored = sha256_bytes(open(path, "rb").read()) == digest
        report["mutations"].append({
            "id": mid, "file": fname, "desc": desc, "expected_test": test,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    report["sha256_restored"] = {f: sha256_bytes(open(os.path.join(HERE, f), "rb").read())
                                 for f in digests}
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
