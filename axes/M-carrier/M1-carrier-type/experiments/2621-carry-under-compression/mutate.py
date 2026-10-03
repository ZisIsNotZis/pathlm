"""2621 · 变异验证（新语义层 `carry_quant.py` 的每个开关「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：本实验唯一的新语义层四处开关——① 量化真的发生；② 前向真的走 carry 读出；
③ `keep_mask`（窗口掩码）真的接进 SDPA；④ 字节口径的基线是 fp16。每个破坏点先记
原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

只动本实验目录内的 `carry_quant.py`（不碰 `pathlm/` 与 `tools/evalkit/`，SCHEMA §22）。
用法：python3 mutate.py            # 输出 mutation_report.json
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CQ = os.path.join(HERE, "carry_quant.py")
TFILE = "tests/test_carry_quant.py"

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测)
MUTATIONS = [
    ("M1", "量化没有真的发生（K/V 原样进 SDPA ⇒ int8 == fp32）",
     """    ks, ki = quantize_kv(ks, mode)
    if acc is not None:
        _merge(acc, ki)
    vs, vi = quantize_kv(vs, mode)
    if acc is not None:
        _merge(acc, vi)""",
     """    ks, ki = ks, _new_stats()
    if acc is not None:
        _merge(acc, ki)
    vs, vi = vs, _new_stats()
    if acc is not None:
        _merge(acc, vi)""",
     ["test_carry_quant_int8_changes_values_and_logits"]),
    ("M2", "carry 读出没有嫁接到注意力输出（通道成摆设）",
     """    scaled = mem * gain
    a = a.transpose(1, 2).reshape(B, T, d) + scaled.to(a.dtype)""",
     """    scaled = mem * gain
    a = a.transpose(1, 2).reshape(B, T, d)""",
     ["test_carry_quant_forward_reads_the_carry_channel"]),
    ("M3", "keep_mask（窗口掩码）被忽略（不再接进 SDPA）",
     """    if keep_mask is None:
        return None
    km = keep_mask.to(device)""",
     """    return None
    if keep_mask is None:
        return None
    km = keep_mask.to(device)""",
     ["test_carry_quant_window_mask_changes_logits_vs_full_causal"]),
    ("M4", "字节口径的基线被换成 fp32（int8 的下降比例算错）",
     """    base = rows[\"fp16\"][\"per_position_bytes\"]""",
     """    base = rows[\"fp32\"][\"per_position_bytes\"]""",
     ["test_bytes_table_int8_is_half_of_fp16"]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    orig = open(CQ, "rb").read()
    report = {"target": os.path.relpath(CQ, ROOT), "sha256_original": sha256_bytes(orig),
              "mutations": []}
    for mid, desc, old, new, tests in MUTATIONS:
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段出现 {src.count(old)} 次，无法唯一定位")
        open(CQ, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(CQ, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-k",
               " or ".join(tests)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=900)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(CQ, "wb").write(orig)                             # 逐字节还原
        restored = sha256_bytes(open(CQ, "rb").read()) == report["sha256_original"]
        report["mutations"].append({
            "id": mid, "desc": desc, "expected_tests": tests,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    report["sha256_restored"] = sha256_bytes(open(CQ, "rb").read())
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
