"""2622-eng1-composable · 变异验证（ENG1 可组合性的每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：本线唯一改动的共享文件 `pathlm/depth_ar.py`（SCHEMA §22：先声明锚点、只做局部
追加、改完回读校验）。四处破坏点各对准一条组合活性：

- M1 `GQA×carry`：可组合块的写入通道读出被丢掉（通道成摆设）；
- M2 `GQA×window`：KV 头投影的注意力忽略窗口掩码（掩码退化成恒等因果）；
- M3 `carry×window`：写入通道块忽略窗口掩码；
- M4 `三者齐开`：可组合块前向忽略 `attn_mask`（窗口对组合路径失效）。

每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。
只动 `pathlm/depth_ar.py`（本线持有代码锁；不碰 `train_depth_ar.py` 与其它模块）。

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
TARGET = os.path.join(ROOT, "pathlm", "depth_ar.py")
TFILE = "tests/test_eng1_composable.py"

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测)
MUTATIONS = [
    ("M1", "GQA×carry：可组合块的写入通道读出被丢掉（通道成摆设）",
     """            scaled = mem * gain
            a = a + scaled.to(a.dtype)
""",
     """            scaled = mem * gain
            a = a + scaled.to(a.dtype) * 0.0
""",
     ["test_eng1_gqa_x_carry_composes_and_both_active",
      "test_eng1_gqa_x_carry_x_window_all_compose"]),
    ("M2", "GQA×window：KV 头投影的注意力忽略窗口掩码（退化为恒等因果）",
     """            a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                               is_causal=attn_mask is None,
                                               enable_gqa=True)
""",
     """            a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None,
                                               is_causal=attn_mask is None,
                                               enable_gqa=True)
""",
     ["test_eng1_gqa_x_window_composes_and_window_active",
      "test_eng1_gqa_x_carry_x_window_all_compose"]),
    ("M3", "carry×window：写入通道块忽略窗口掩码",
     """        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                            is_causal=attn_mask is None)
""",
     """        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None,
                                            is_causal=attn_mask is None)
""",
     ["test_eng1_carry_x_window_composes_and_both_active"]),
    ("M4", "三者齐开：可组合块前向忽略 attn_mask（窗口对组合路径失效）",
     """        a = self._attention(self.ln1(x), attn_mask)
        tele = None
""",
     """        a = self._attention(self.ln1(x), None)
        tele = None
""",
     ["test_eng1_gqa_x_carry_x_window_all_compose"]),
]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    orig = open(TARGET, "rb").read()
    report = {"target": os.path.relpath(TARGET, ROOT),
              "sha256_original": sha256_bytes(orig), "mutations": []}
    for mid, desc, old, new, tests in MUTATIONS:
        src = orig.decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段出现 {src.count(old)} 次，无法唯一定位")
        open(TARGET, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(TARGET, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-p", "no:cacheprovider",
               "-k", " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(TARGET, "wb").write(orig)                          # 逐字节还原
        restored = sha256_bytes(open(TARGET, "rb").read()) == report["sha256_original"]
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
    report["sha256_restored"] = sha256_bytes(open(TARGET, "rb").read())
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
