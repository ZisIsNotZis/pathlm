"""2616 · 变异验证（每个语义开关必须「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）。

对象：`pathlm/depth_ar.py` 的训练期可见窗口开关（ticket 32 契约 O / ticket 33 契约 Q
复用同一实现；本实验**未改动** `pathlm/` 与 `train_depth_ar.py`，只把 window 取值
放到 192 —— 所以这里验证的正是「192 这个取值真的生效」）。
每个破坏点先记录原文 sha256，跑指定单测（期望**红**），再逐字节还原并核对 sha256。

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
TFILE = "tests/test_depth_ar.py"

MASK_LINE = "    keep = (j <= i) & (((i - j) < window) | (j < anchors))"
FWD_LINE = "                h, tele = blk(h, attn_mask=win_mask)"
WMASK_CALL = ("        keep = window_anchor_keep_mask(T, self.cfg.train_window,\n"
              "                                       self.cfg.train_anchors, device)")

#: (id, 说明, 原文片段, 破坏后片段, 期望抓红的单测, 是否期望抓红)
MUTATIONS = [
    ("Q1", "掩码丢掉「锚点恒可见」项（j < anchors 被删）",
     MASK_LINE,
     "    keep = (j <= i) & ((i - j) < window)",
     ["test_window192_keep_mask_counts_and_boundary_at_real_scale",
      "test_window_anchor_keep_mask_semantics"], True),
    ("Q2", "窗口边界 off-by-one（< 写成 ≤ ⇒ 多放一个位置：192→193）",
     MASK_LINE,
     "    keep = (j <= i) & (((i - j) <= window) | (j < anchors))",
     ["test_window192_keep_mask_counts_and_boundary_at_real_scale",
      "test_window_anchor_keep_mask_semantics"], True),
    ("Q3", "窗口长度被**夹到 32**（192 的取值失效 ⇒ 本实验的干预其实没生效）",
     MASK_LINE,
     "    keep = (j <= i) & (((i - j) < min(window, 32)) | (j < anchors))",
     ["test_window192_keep_mask_counts_and_boundary_at_real_scale",
      "test_window192_forward_matches_eval_mask_and_differs_from_32"], True),
    ("Q4", "forward 里掩码没被真的传给注意力（死开关）",
     FWD_LINE,
     "                h, tele = blk(h)",
     ["test_window192_forward_matches_eval_mask_and_differs_from_32",
      "test_train_window_restricts_attention_to_recent_positions"], True),
    ("Q5", "锚点取在窗口**末端**（j ≥ T−anchors）而不是最前面",
     MASK_LINE,
     "    keep = (j <= i) & (((i - j) < window) | (j >= T - anchors))",
     ["test_window192_keep_mask_counts_and_boundary_at_real_scale",
      "test_window_anchor_keep_mask_semantics"], True),
    ("Q6", "forward 侧掩码漏掉 anchors（训练掩码 ≠ 评测掩码）",
     WMASK_CALL,
     ("        keep = window_anchor_keep_mask(T, self.cfg.train_window,\n"
      "                                       0, device)"),
     ["test_window192_forward_matches_eval_mask_and_differs_from_32",
      "test_train_window_forward_matches_eval_mask_path"], True),
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    src = open(TARGET).read()
    digest0 = sha256(src)
    report = {"target": os.path.relpath(TARGET, ROOT), "sha256_before": digest0,
              "note": ("本实验未改动实现（`pathlm/depth_ar.py` 与 2615 记录同 sha256）；"
                       "变异验证的对象是既有实现，目的 = 证明「window=192 真的生效」"
                       "这件事被可失败单测钉住。"),
              "mutations": []}
    for mid, desc, old, new, tests, expect_red in MUTATIONS:
        if src.count(old) != 1:
            raise SystemExit(f"{mid}: 原文片段在文件里出现 {src.count(old)} 次，无法唯一定位")
        open(TARGET, "w").write(src.replace(old, new))
        mut_digest = sha256(open(TARGET).read())
        cmd = [sys.executable, "-m", "pytest", TFILE, "-q",
               "-k", " or ".join(tests)]
        r = mutation.pytest_run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
        caught = r.returncode != 0
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1:]
        # 逐条记录哪些测试真的红了（判据「可失败单测」的有效性证据）
        red = sorted({ln.split("::")[1].split(" ")[0] for ln in r.stdout.splitlines()
                      if ln.startswith("FAILED ") and "::" in ln})
        report["mutations"].append({
            "id": mid, "desc": desc, "expected_tests": tests,
            "expect_red": expect_red, "caught_red": caught,
            "outcome_ok": caught == expect_red, "sha256_mutated": mut_digest,
            "tests_actually_red": red,
            "pytest_summary": tail[0] if tail else "",
        })
        print(f"{mid}: {'抓红 OK' if caught else '未抓红'}"
              f"{'（预期如此）' if caught == expect_red and not expect_red else ''}"
              f" — {desc} — {tail[0] if tail else ''}", flush=True)
        open(TARGET, "w").write(src)                 # 逐字节还原
        if sha256(open(TARGET).read()) != digest0:
            raise SystemExit(f"{mid}: 还原失败（sha256 不一致）")
    report["sha256_after"] = sha256(open(TARGET).read())
    report["restored_bitwise"] = report["sha256_after"] == digest0
    report["all_expected_outcomes"] = all(m["outcome_ok"] for m in report["mutations"])
    report["all_caught"] = all(m["caught_red"] for m in report["mutations"]
                               if m["expect_red"])
    json.dump(report, open(os.path.join(HERE, "mutation_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("sha256_before", "sha256_after",
                                             "restored_bitwise", "all_caught",
                                             "all_expected_outcomes")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
