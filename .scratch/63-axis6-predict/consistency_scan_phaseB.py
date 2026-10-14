"""Phase B · 兄弟字段一致性扫描（axis6-lead）。

只读检查（不修改任何文件），输出一份可核清单：
1. `dir` 字段 vs 磁盘目录；`experiments` 列出的每个路径是否存在。
2. 6.2 / 6.3.3 的结论数字是否与实验落盘一致（逐项断言）。
3. experiments/README.md 的表行数 vs 磁盘实验目录数。
4. 跨文件冲突扫描：其它轴/文档里有无同时陈述本轴事实的地方（粗筛 + 人工核）。
跑法：python3 consistency_scan_phaseB.py
"""
from __future__ import annotations

import json
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT: list[str] = []


def say(tag: str, msg: str) -> None:
    OUT.append(f"[{tag}] {msg}")


def load_yaml(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    fails = 0
    items = ["6.2-speculative-decoding", "6.3.1-diffusion", "6.3.2-pad-fill",
             "6.3.3-latent-regression", "6.4-training-fill"]
    for it in items:
        py = os.path.join(ROOT, "axes", "6-predict", it, "project.yaml")
        d = load_yaml(py)
        dirp = os.path.join(ROOT, d["dir"])
        if not os.path.isdir(dirp):
            say("FAIL", f"{it}: dir 字段指向不存在的目录 {d['dir']}")
            fails += 1
        else:
            say("ok", f"{it}: dir 存在（{d['dir']}）")
        for e in d.get("experiments") or []:
            if e.startswith(".scratch"):
                continue
            if not os.path.exists(os.path.join(ROOT, e)):
                say("FAIL", f"{it}: experiments 条目不存在 {e}")
                fails += 1
        # experiments/README.md 行数 vs 磁盘子目录数
        expdir = os.path.join(dirp, "experiments")
        if os.path.isdir(expdir):
            subdirs = sorted(x for x in os.listdir(expdir)
                             if os.path.isdir(os.path.join(expdir, x)) and not x.startswith("__"))
            rdm = os.path.join(expdir, "README.md")
            if os.path.isfile(rdm):
                rows = [l for l in open(rdm, encoding="utf-8") if l.startswith("| [`")]
                if len(rows) != len(subdirs):
                    say("WARN", f"{it}: experiments/README.md 表 {len(rows)} 行 vs 磁盘 {len(subdirs)} 目录")
                else:
                    say("ok", f"{it}: experiments/README.md 表 {len(rows)} 行 == 磁盘 {len(subdirs)} 目录")
            elif subdirs:
                say("WARN", f"{it}: experiments/ 有 {len(subdirs)} 个子目录但没有 README.md 表")

    # ---- 6.2 结论数字 vs 2650 / 2640 落盘
    p2650 = os.path.join(ROOT, "axes", "6-predict", "6.2-speculative-decoding",
                         "experiments", "2650-verify-gate-13m", "results.json")
    p2640 = os.path.join(ROOT, "axes", "6-predict", "6.2-speculative-decoding",
                         "experiments", "2640-tree-conf-13m", "results.json")
    r2650, r2640 = json.load(open(p2650, encoding="utf-8")), json.load(open(p2640, encoding="utf-8"))

    def a1(blk, arm):
        acc = blk["arms"][arm]["per_prompt"]
        n = sum(len(p["accepts"]) for p in acc)
        return sum(1 for p in acc for x in p["accepts"] if x >= 1) / n

    checks = [
        ("6.2 (j) adv_random α₁ 13m_s0 = 0.0185", round(a1(r2650["checkpoints"]["13m_s0"], "adv_random_k4"), 4), 0.0185),
        ("6.2 (j) adv_bottomk4 α₁ = 0.0", round(a1(r2650["checkpoints"]["13m_s0"], "adv_bottomk4"), 4), 0.0),
        ("6.2 (j) aggr_topk8 α₁ = 0.9785", round(a1(r2650["checkpoints"]["13m_s0"], "aggr_topk8"), 4), 0.9785),
        ("6.2 (j) repro_topk4 α₁ = 0.9317（= 2640 tree_topk4）",
         round(a1(r2650["checkpoints"]["13m_s0"], "repro_topk4"), 4),
         round(a1(r2640["checkpoints"]["13m_s0"], "tree_topk4"), 4)),
        ("6.2 (j) single_k1_ctl α₁ = 0.8000（= combo_pw）",
         round(a1(r2650["checkpoints"]["13m_s0"], "single_k1_ctl"), 4),
         round(a1(r2640["checkpoints"]["13m_s0"], "combo_pw"), 4)),
    ]
    for label, got, want in checks:
        if abs(got - want) > 1e-9:
            say("FAIL", f"{label}: 落盘 {got} != 结论 {want}")
            fails += 1
        else:
            say("ok", f"{label}: 落盘 {got} == 结论")

    # ---- 6.3.3 结论数字 vs 2660 落盘
    p2660 = os.path.join(ROOT, "axes", "6-predict", "6.3.3-latent-regression",
                         "experiments", "2660-latentreg-vs-disc-13m", "results.json")
    r2660 = json.load(open(p2660, encoding="utf-8"))
    d1 = r2660["arms"]["disc_k2"]["0"]["hit_t2"]
    d1s1 = r2660["arms"]["disc_k2"]["1"]["hit_t2"]
    l1 = r2660["arms"]["lat_k2_lam1"]["0"]["hit_t2"]
    for label, got, want in [
        ("6.3.3 disc_k2 hit@t+2 = 0.4572", round(d1, 4), 0.4572),
        ("6.3.3 disc_k2 hit@t+2 (seed1) = 0.4573", round(d1s1, 4), 0.4573),
        ("6.3.3 lat_k2_lam1 hit@t+2 = 0.4880", round(l1, 4), 0.4880),
        ("6.3.3 L1 Δ(lam1) 跟种子合并 = +0.0313",
         round(r2660["L1"]["lat_k2_lam1"]["t+2"]["combined_mean_over_seeds"]["delta"], 4), 0.0313),
        ("6.3.3 L1 Δ(lam1) 种子0 = +0.0306", round(l1 - d1, 4), 0.0306),
    ]:
        if abs(got - want) > 6e-4:
            say("FAIL", f"{label}: 落盘 {got} != 结论 {want}")
            fails += 1
        else:
            say("ok", f"{label}: 落盘 {got} ≈ 结论")
    dv = r2660.get("draftverify", {})
    if dv:
        say("ok", f"6.3.3 draftverify 节存在（键 {sorted(dv.keys())[:6]}…）")
    else:
        say("FAIL", "6.3.3 results.json 缺 draftverify 节（路径 ii 读数未并入）")
        fails += 1

    # ---- 跨文件粗筛：其它轴是否同时陈述 6.2/6.3 的事实
    say("info", "跨轴粗筛：见 TICKET.md §B5 与报告「未决」节的 grep 结果（人工核）")
    print("\n".join(OUT))
    print(f"\n== 扫描结束：FAIL={fails} ==")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
