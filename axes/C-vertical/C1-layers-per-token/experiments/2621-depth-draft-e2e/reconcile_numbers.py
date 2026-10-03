"""ticket 38 W2 · 报告数字对账（SCHEMA §12 三道对账的可执行版）。

读 `results.json` + `cycle_log.jsonl`，**重新推导**分析报告里出现的每个数字，
逐条打印 PASS/FAIL；任一 FAIL 时非零退出。用法：

    python3 reconcile_numbers.py            # 默认读同目录 results.json
"""

import json
import os
import sys
from statistics import median

HERE = os.path.dirname(os.path.abspath(__file__))
FAILS = []
CHECKS = []


def check(name, got, want, tol=5e-4):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    CHECKS.append((name, got, want, ok))
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")


def check_eq(name, got, want):
    ok = got == want
    CHECKS.append((name, got, want, ok))
    if not ok:
        FAILS.append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got} want={want}")


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else HERE
    with open(os.path.join(rdir, "results.json")) as f:
        R = json.load(f)
    arms = R["arms"]

    # ---------- 1. 算术对账 ----------
    for name, a in arms.items():
        if "sum_n_drafts" not in a:
            continue
        check(f"{name}.acceptance_pooled == Σaccepted/Σdrafts",
              a["acceptance_pooled"], a["sum_n_accepted"] / a["sum_n_drafts"])
        # 逐 prompt 的接受数之和 == 汇总
        check_eq(f"{name}.sum_n_accepted == Σ per_prompt",
                 a["sum_n_accepted"], sum(p["n_accepted"] for p in a["per_prompt"]))
        check_eq(f"{name}.sum_n_drafts == Σ per_prompt",
                 a["sum_n_drafts"], sum(p["n_drafts"] for p in a["per_prompt"]))
        # 逐 prompt 汇总的 draft 数 == gamma × cycles（账本恒等式）
        for p in a["per_prompt"]:
            check_eq(f"{name}.prompt{p['prompt_idx']}.n_drafts == gamma×cycles",
                     p["n_drafts"], a["config"]["gamma"] * p["n_cycles"])
        # 接受率的上界：Σaccepted ≤ Σdrafts
        assert a["sum_n_accepted"] <= a["sum_n_drafts"]
        # 逐位置的接受数之和 == 汇总接受数
        by_pos = a["accept_by_pos_pooled"]
        est = sum(b["accept"] * b["n"] for b in by_pos)
        check(f"{name}.Σ_by_pos(accept×n) ≈ Σ accepted（±1 舍入）",
              round(est), round(a["sum_n_accepted"]), tol=1.0)
        # 墙钟：pooled tok/s == Σgen / Σwall
        check(f"{name}.tok_s_pooled == Σgen×reps/Σwall",
              a["tok_s_pooled"],
              a["sum_n_gen"] * a["n_repeats_per_prompt"] / a["wall_s_all"],
              tol=5e-3)
        # 每次重测的中位数 == 记录值
        for p in a["per_prompt"]:
            check(f"{name}.prompt{p['prompt_idx']}.wall_median_s",
                  p["wall_median_s"], median(p["wall_s"]), tol=1e-3)
            check(f"{name}.prompt{p['prompt_idx']}.tok_s",
                  p["tok_s"], p["n_gen"] / p["wall_median_s"], tol=5e-3)

    # ---------- 2. 任务对账（对照臂测的是同一件事）----------
    for name, a in arms.items():
        if "per_prompt" not in a:
            continue
        for p in a["per_prompt"]:
            assert p["wall_median_s"] > 0
    plain = arms["plain_full"]
    for name, a in arms.items():
        if name.startswith("plain"):
            continue
        check_eq(f"{name}.sum_n_gen == plain_full.sum_n_gen（同 prompt 同长度）",
                 a["sum_n_gen"], plain["sum_n_gen"])

    # ---------- 3. 口径对账（等效 FLOPs）----------
    d = R["ckpt"]["config"]["d_model"]
    m = R["ckpt"]["config"]["mlp_mult"]
    ppb = (4 + 2 * m) * d * d
    for name, a in arms.items():
        if "eq_flops" not in a:
            continue
        check_eq(f"{name}.params_per_block", a["params_per_block"], ppb)
        check_eq(f"{name}.eq_flops == eq_layerpos×2×ppb",
                 a["eq_flops"], a["eq_layerpos"] * 2 * ppb)
        check(f"{name}.eq_layerpos_per_token == gen_only/tokens",
              a["eq_layerpos_per_token"],
              a["eq_layerpos_gen_only"] / a["tokens"], tol=2e-3)
        check(f"{name}.eq_flops_per_token == 2×ppb×layerpos_per_token",
              a["eq_flops_per_token"],
              2 * ppb * a["eq_layerpos_per_token"], tol=2e-3)

    # ---------- 预算对账：等效 FLOPs 与墙钟两口径 ----------
    check_eq("single_g4.gamma == 4", arms["single_g4"]["config"]["gamma"], 4)
    check_eq("graded.mid_depth == 6", arms["graded"]["config"]["mid_depth"], 6)
    check_eq("graded.draft_depth == single.draft_depth",
             arms["graded"]["config"]["draft_depth"],
             arms["single_g4"]["config"]["draft_depth"])

    # ---------- ① 接受率三口径 ----------
    prox = R["proxies"]
    tf4 = [t["accept"] for t in prox["tf_enwik8"] if t["depth"] == 4][0]
    check("proxy.tf_enwik8.depth4 == 0.9694（历史记录值 23c-H2-w0-12k）",
          tf4, 0.9694, tol=0.02)
    ss = prox["tf_selfstream_pooled"]
    check("proxy.tf_selfstream.depth4 ∈ (0,1]", min(ss["depth4"], 1.0), ss["depth4"])
    pos1 = [b["accept"] for b in arms["single_g4"]["accept_by_pos_pooled"]
            if b["pos"] == 1]
    assert pos1, "缺位置 1 接受率"
    print(f"[info] 三口径：tf_enwik8@4={tf4} / tf_selfstream@4={ss['depth4']} / "
          f"real_pos1={pos1[0]} / real_pooled={arms['single_g4']['acceptance_pooled']}")

    # ---------- 逐 cycle 账本（活性量落盘）----------
    path = os.path.join(rdir, "cycle_log.jsonl")
    n_lines = 0
    with open(path) as f:
        for line in f:
            c = json.loads(line)
            check_eq(f"cycle[{n_lines}].emitted == accepted+1",
                     c["emitted"], c["accepted"] + 1)
            assert c["accepted"] <= c["gamma_effective"]
            n_lines += 1
    print(f"[info] cycle_log.jsonl 共 {n_lines} 行（每 cycle 一行 = 活性量落盘）")
    want_lines = sum(p["n_cycles"] for a in arms.values()
                     if "per_prompt" in a for p in a["per_prompt"]
                     if "n_cycles" in p)
    check_eq("cycle_log 行数 == Σ cycles（spec 臂）", n_lines, want_lines)

    # ---------- ②墙钟主口径：交错轮次块 ----------
    w = R["wall_clock_interleaved"]
    rows = w["raw_rows"]
    base = w["by_arm"]["plain_full"]["wall_median_s"]
    cache_base = w["by_arm"]["plain_cache_full"]["wall_median_s"]
    check_eq("wall.raw_rows 条数 == rounds×arms×prompts",
             len(rows), w["rounds"] * len(w["by_arm"]) * w["prompts"])
    for name, a in w["by_arm"].items():
        ws = [x["wall_s"] for x in rows if x["arm"] == name]
        check(f"wall.{name}.wall_median_s == median(raw)",
              a["wall_median_s"], median(ws), tol=1e-4)
        check(f"wall.{name}.tok_s == n_gen/median",
              a["tok_s"], a["n_gen"] / a["wall_median_s"], tol=5e-3)
        check(f"wall.{name}.speedup_vs_plain_full",
              a["speedup_vs_plain_full"], base / a["wall_median_s"], tol=5e-3)
        check(f"wall.{name}.speedup_vs_plain_cache_full",
              a["speedup_vs_plain_cache_full"],
              cache_base / a["wall_median_s"], tol=5e-3)
        check_eq(f"wall.{name}.n_obs == rounds×prompts",
                 a["n_obs"], w["rounds"] * w["prompts"])

    # ---------- 质量护栏 ----------
    qg = R["quality_guard"]
    check_eq("quality_guard.total_mismatch == Σ rows",
             qg["total_mismatch"],
             sum(r["spec_vs_decoder_mismatch"] for r in qg["rows"]))
    check_eq("quality_guard：spec 流与普通贪心流逐 token 相同",
             qg["total_mismatch"], 0)
    for r in qg["rows"]:
        assert r["plain_decoder_vs_cache_mismatch"] == 0, r

    # ---------- checkpoint 哈希对账 ----------
    with open(os.path.join(HERE, "MODEL.md")) as f:
        md = f.read()
    assert R["ckpt"]["sha256"] in md, "results.json 的 ckpt sha256 未写入 MODEL.md"
    print(f"[PASS] ckpt.sha256 与 MODEL.md 记录一致：{R['ckpt']['sha256'][:16]}…")

    print(f"\n=== {len(CHECKS) - len(FAILS)}/{len(CHECKS)} PASS ===")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("对账全过。")


if __name__ == "__main__":
    main()
