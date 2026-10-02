"""2619 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象 = 本线新增的 5 处语义开关：目标偏移 / 目标行 / λ=0 守卫 / fp32 口径 / CLI 通道。
每个破坏点先记原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

**并发纪律**（本批 U1 线同时在改 `pathlm/depth_ar.py` / `train_depth_ar.py`）：
本脚本会临时改写共享文件 ⇒ **必须先取代码锁**（`/tmp/pathlm_code.lock`）再运行，
否则「逐字节还原」会把另一条线的未提交改动一起盖掉。

用法：
  n=0; until mkdir /tmp/pathlm_code.lock 2>/dev/null; do sleep 10; n=$((n+1)); [ $n -gt 60 ] && exit 9; done
  python3 mutate.py ; rmdir /tmp/pathlm_code.lock
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RS = os.path.join(ROOT, "pathlm", "recon_signal.py")
TR = os.path.join(ROOT, "train_depth_ar.py")
TFILE = "tests/test_recon_signal.py"

#: (id, 说明, 目标文件, 原文片段, 破坏后片段, 期望抓红的单测)
MUTATIONS = [
    ("M1", "目标偏移错一格（做成 tokens[i−W+1]：落在窗内 ⇒ 退化成平凡项）", RS,
     """    rows = torch.arange(window, T, device=tokens.device)
    return rows, tokens[:, :T - window]""",
     """    rows = torch.arange(window, T, device=tokens.device)
    return rows, tokens[:, 1:T - window + 1]""",
     ["test_recon_targets_are_exactly_window_masked",
      "test_recon_loss_matches_manual_ce"]),
    ("M2", "取错隐状态行（用前 T−W 行而不是 rows 指定的行）", RS,
     """    logits = recon_logits(head, hidden[:, rows])                 # [B, T−W, V] fp32""",
     """    logits = recon_logits(head, hidden[:, :rows.numel()])        # 变异：取错行""",
     ["test_recon_loss_matches_manual_ce",
      "test_recon_mutation_target_shift_changes_loss"]),
    ("M3", "λ=0 守卫被放宽（λ=0 也建图 ⇒ 不再是「原样返回」，逐位等价失效）", TR,
     """    if recon_lambda <= 0 or model.cfg.recon_window <= 0:
        return loss, None""",
     """    if recon_lambda < 0 or model.cfg.recon_window <= 0:
        return loss, None""",
     ["test_recon_lambda_zero_returns_same_loss_object",
      "test_recon_head_dead_at_lambda_zero_grads_flow_at_lambda_one"]),
    ("M4", "CLI 开关没接到 config（开关是死的）", TR,
     """                         recon_window=getattr(args, "recon_window", 0),""",
     """                         recon_window=0,""",
     ["test_recon_cli_plumbed_to_config"]),
    ("M5", "fp32 口径被破坏（头在 autocast 内出 bf16 logits）", RS,
     """    with torch.autocast(hidden_rows.device.type, enabled=False):
        return head(hidden_rows.float())""",
     """    with torch.autocast(hidden_rows.device.type, enabled=False):
        return head(hidden_rows.to(torch.bfloat16))""",
     ["test_recon_logits_is_fp32_under_bf16_autocast"]),
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    if not os.path.isdir("/tmp/pathlm_code.lock"):
        raise SystemExit("拒绝运行：没有持代码锁（/tmp/pathlm_code.lock）——"
                         "共享文件会被本脚本临时改写，见模块 docstring")
    originals = {p: open(p, "rb").read() for p in (RS, TR)}
    report = {"targets": {os.path.relpath(p, ROOT): sha256_bytes(b)
                          for p, b in originals.items()},
              "code_lock": "/tmp/pathlm_code.lock 存在（运行前检查通过）",
              "note": ("路径锚点：pathlm/recon_signal.py 全文件；train_depth_ar.py "
                       "只变异本线新增片段（add_recon_term 的 λ=0 守卫 / "
                       "build_depthar_config 的 recon_window 接线）"),
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
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
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
