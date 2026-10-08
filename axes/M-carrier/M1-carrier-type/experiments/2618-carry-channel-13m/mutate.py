"""2618 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：契约 U 的三处语义开关（写入式 / 读出约定 / 因果掩码 / 单边零起步 / 读出嫁接 /
CLI 通道 / 训练任务构造）。每个破坏点先记原文 sha256，跑指定单测（期望**红**），
再逐字节还原并核对 sha256。

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
DA = os.path.join(ROOT, "pathlm", "depth_ar.py")
TR = os.path.join(ROOT, "train_depth_ar.py")
LT = os.path.join(ROOT, "tools", "evalkit", "longrange_tasks.py")
TFILE = "tests/test_depth_ar.py"

#: (id, 说明, 目标文件, 原文片段, 破坏后片段, 期望抓红的单测)
MUTATIONS = [
    ("M1", "delta 块的因果掩码放宽一格（位置 t 读到 t+1 的写入 ⇒ 未来泄漏）", DA,
     """    up = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device), 1)
    keep = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device))""",
     """    up = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device), 1)
    keep = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device), -1)""",
     ["test_carry_channel_is_causal_only",
      "test_carry_chunked_matches_sequential"]),
    ("M2", "读出约定被改成「不含自写入」（S_{t−1} 而不是 S_t）", DA,
     """    keep = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device))
    Om = (Gq * bc.unsqueeze(-1)).masked_fill(~keep, 0.0)""",
     """    keep = torch.triu(torch.ones(C, C, dtype=torch.bool, device=qc.device), 1)
    Om = (Gq * bc.unsqueeze(-1)).masked_fill(~keep, 0.0)""",
     ["test_carry_read_convention_includes_own_write"]),
    ("M3", "delta 的写入式退化成 plain（两臂变成同一个臂）", DA,
     """        chunk_fn = (_carry_chunk_plain if rule == "plain" else _carry_chunk_delta)""",
     """        chunk_fn = _carry_chunk_plain""",
     ["test_carry_plain_rule_does_not_forget_delta_does",
      "test_carry_liveness_and_probe_report_alive"]),
    ("M4", "单边零起步被破坏（读出增益随机初始化 ⇒ 起步不再恒等于 U0）", DA,
     """                nn.init.zeros_(blk.carry_scale)    # **单边零**：起步恒等于无通道基线""",
     """                nn.init.normal_(blk.carry_scale, std=0.01)""",
     ["test_carry_channel_identity_start_matches_no_channel",
      "test_carry_block_zero_scale_equals_plain_block_bitwise"]),
    ("M5", "读出没有被嫁接到注意力输出（通道成摆设）", DA,
     """        a = a + scaled.to(a.dtype)""",
     """        a = a""",
     ["test_carry_block_zero_scale_equals_plain_block_bitwise",
      "test_carry_liveness_and_probe_report_alive",
      "test_carry_channel_carries_across_window_but_baseline_does_not"]),
    ("M6", "CLI 开关没有接到 config（开关是死的）", TR,
     """                         carry_rule=(None if args.carry_write == "none"
                                     else args.carry_write),""",
     """                         carry_rule=None,""",
     ["test_carry_cli_plumbed_to_config"]),
    ("M7", "训练任务构造的查询间隔 gap 变小（局部窗口里出现上一个答案 ⇒ 可抄）", LT,
     """                             gap: int = 14,""",
     """                             gap: int = 0,""",
     ["test_mqar_training_task_layout_and_local_prior"]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    originals = {p: open(p, "rb").read() for p in (DA, TR, LT)}
    report = {"targets": {os.path.relpath(p, ROOT): sha256_bytes(b)
                          for p, b in originals.items()},
              "mutations": []}
    for mid, desc, target, old, new, tests in MUTATIONS:
        src = originals[target].decode()
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段在 {target} 出现 {src.count(old)} 次，"
                             "无法唯一定位")
        open(target, "w").write(src.replace(old, new))
        mut_digest = sha256_bytes(open(target, "rb").read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q", "-k",
               " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        open(target, "wb").write(originals[target])            # 逐字节还原
        restored = sha256_bytes(open(target, "rb").read()) == \
            report["targets"][os.path.relpath(target, ROOT)]
        report["mutations"].append({
            "id": mid, "desc": desc,
            "target": os.path.relpath(target, ROOT), "expected_tests": tests,
            "caught_red": caught, "outcome_ok": caught,
            "sha256_mutated": mut_digest, "restored_bitwise": restored,
            "pytest_summary": tail[0] if tail else ""})
        print(f"{mid}: caught_red={caught} restored={restored} "
              f"| {tail[0] if tail else ''}", flush=True)
    report["all_caught_red"] = all(m["caught_red"] for m in report["mutations"])
    report["all_restored_bitwise"] = all(m["restored_bitwise"]
                                         for m in report["mutations"])
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("all_caught_red",
                                             "all_restored_bitwise")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
