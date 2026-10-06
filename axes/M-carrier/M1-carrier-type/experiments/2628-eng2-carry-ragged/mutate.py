"""2628-eng2-carry-ragged · 变异验证（ENG2 每处语义「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：本线唯一改动的共享文件 `pathlm/depth_ar.py`（SCHEMA §22：先声明锚点、只做局部
追加、改完回读校验）。五处破坏点各对准一条组合/一条语义：

- M1 `carry×fill/proj-fill`：带通道的块忽略 `kv_from`（混合深度 KV 源失效）；
- M2 `carry×随机退出`：带通道的块忽略 `gate_vec`（逐通道门失效）；
- M3 `carry×全部`：带通道的块把记忆读出乘 0（通道成摆设）；
- M4 `carry×ragged`：带通道的块忽略 `attn_mask`（ragged 掩码失效）；
- M5 `写入时机`：非 dense 前向把「已退出位置的写入」抹掉（让槽内容依赖退出模式）。

每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。
只动 `pathlm/depth_ar.py`（本线持有代码锁；不碰其它模块）。

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
TFILE = "tests/test_depth_ar.py"

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测)
MUTATIONS = [
    ("M1", "carry×fill/proj-fill：带通道的块忽略 kv_from（混合深度 KV 源失效）",
     """        if kv_from is not None:
            # 混合深度 KV 源（fill / proj-fill）：K/V 取自 kv_from 的 pre-LN，
            # query 仍取自 x——与 stock `Block` 逐字同款。
            kv = self.qkv(self.ln1(kv_from))
            k, v = kv[..., d:2 * d], kv[..., 2 * d:]
""",
     """        if kv_from is not None and False:
            # 混合深度 KV 源（fill / proj-fill）：K/V 取自 kv_from 的 pre-LN，
            # query 仍取自 x——与 stock `Block` 逐字同款。
            kv = self.qkv(self.ln1(kv_from))
            k, v = kv[..., d:2 * d], kv[..., 2 * d:]
""",
     ["test_eng2_carry_x_random_exit_x_fill_consumes_frozen_kv"]),
    ("M2", "carry×随机退出：带通道的块忽略 gate_vec（逐通道门失效）",
     """        if gate_vec is not None:
            x = x_in + gate_vec * (x - x_in)     # 与 `Block` 同款逐通道门
""",
     """        if gate_vec is not None and False:
            x = x_in + gate_vec * (x - x_in)     # 与 `Block` 同款逐通道门
""",
     ["test_eng2_carry_x_random_exit_composes",
      "test_eng2_carry_x_ragged_layer_matches_manual_rule"]),
    ("M3", "carry×全部：带通道的块把记忆读出乘 0（通道成摆设）",
     """        scaled = mem * gain
        a = a.transpose(1, 2).reshape(B, T, d) + scaled.to(a.dtype)
""",
     """        scaled = mem * gain
        a = a.transpose(1, 2).reshape(B, T, d) + scaled.to(a.dtype) * 0.0
""",
     ["test_eng2_carry_x_ragged_layer_matches_manual_rule",
      "test_eng2_carry_x_random_exit_composes"]),
    ("M4", "carry×ragged：带通道的块忽略 attn_mask（ragged 掩码失效）",
     """        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                            is_causal=attn_mask is None)
        # 读出拆到 [B,T,d] 空间再做加法：每层增益按**记忆槽头**展开到各自 dh 个
""",
     """        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=None,
                                            is_causal=True)
        # 读出拆到 [B,T,d] 空间再做加法：每层增益按**记忆槽头**展开到各自 dh 个
""",
     ["test_eng2_carry_x_ragged_layer_matches_manual_rule"]),
    ("M5", "写入时机：非 dense 前向把「已退出位置的写入」按 depth-0 退出抹掉"
           "（槽内容错误地依赖退出模式）",
     """            frozen = h
            for k in range(1, L + 1):
                active = exit_depths >= k                       # [B,T]
""",
     """            frozen = h
            if self.cfg.carry_rule is not None:
                self._carry_box["mem"] = self._carry_box["mem"] * (
                    exit_depths > 0).unsqueeze(-1)
            for k in range(1, L + 1):
                active = exit_depths >= k                       # [B,T]
""",
     ["test_eng2_carry_write_timing_ignores_exit_depths",
      "test_eng2_carry_x_ragged_layer_matches_manual_rule"]),
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
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
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
