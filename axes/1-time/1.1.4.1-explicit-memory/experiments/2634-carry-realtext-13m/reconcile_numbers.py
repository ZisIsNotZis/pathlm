#!/usr/bin/env python3
"""2634-carry-realtext-13m · 数字对账（期望值一律由 raw 产物**重算**，不用历史常数）。

raw 来源：`results.json`、`needle_raw.json`、`train/<arm>/train_log.jsonl`、`config.json`。
对账口径：`tools/evalkit/reconcile.py::Ledger`（容差显式分开写；derived/golden 分类计数）。

用法：python3 reconcile_numbers.py      # 退出码 0 = 全过
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.evalkit.reconcile import Ledger                          # noqa: E402
import needle_stats as ns                                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,           # noqa: E402
                             window_anchor_keep_mask)


def _mean(vals):
    return sum(vals) / len(vals)


def main() -> int:
    R = json.load(open(os.path.join(HERE, "results.json")))
    raw = json.load(open(os.path.join(HERE, "needle_raw.json")))
    cfg = R["config"]
    T = cfg["model"]["seq_len"]
    ep = cfg["eval"]["probe"]
    ledger = Ledger("2634-carry-realtext-13m",
                    sources=("results.json", "needle_raw.json", "config.json",
                             "train/*/train_log.jsonl"))

    # ---- 仪器/布局不变量 ----
    keep = window_anchor_keep_mask(T, cfg["model"]["train_window"],
                                   cfg["model"]["train_anchors"])
    ledger.derived("窗口掩码可见对数 == keep.sum()",
                   R["selfchecks"]["window_mask_visible_pairs"],
                   lambda: int(keep.sum().item()),
                   sources=("pathlm.depth_ar.window_anchor_keep_mask(256,16,0)",),
                   abs_tol=1e-9)
    ledger.eq("布局前提 eval_lo − rf ≥ corrupt_hi",
              ep["eval_lo"] - ep["rf"] >= ep["corrupt_hi"], True)
    ledger.eq("rf == n_layers·(window−1)", ep["rf"],
              cfg["model"]["n_layers"] * (cfg["model"]["train_window"] - 1))
    ledger.eq("探针 token 数 == n_batches·batch_size·被评行数",
              R["far_source_probe"]["rt_none_s0"]["n_tokens"],
              ep["n_batches"] * ep["batch_size"] * (ep["eval_hi"] - ep["eval_lo"]))

    # ---- 参数量（由代码重建计，不用历史常数） ----
    for spec in cfg["arms"]:
        m = DepthARModel(DepthARConfig(**{
            **cfg["model"], "carry_rule": None if spec["rule"] == "none" else spec["rule"]}),
            cfg["data"]["vocab_size"])
        ledger.derived(f"{spec['name']} 参数量 == 重建模型参数量",
                       R["checkpoints"][spec["name"]]["n_params"],
                       lambda m=m: sum(p.numel() for p in m.parameters()),
                       sources=("pathlm.depth_ar.DepthARModel",), abs_tol=1e-9)

    # ---- 探针：Δbpc_far 必须等于两份行剖面均值之差 ----
    for name, row in R["far_source_probe"].items():
        ledger.derived(f"{name}: Δbpc_far == mean(corrupt) − mean(clean)",
                       row["delta_bpc_far"],
                       lambda row=row: round(_mean(row["ce_by_row_corrupt"])
                                             - _mean(row["ce_by_row_clean"]), 6),
                       # 逐行剖面在 results.json 里按 6 位小数落盘 ⇒ 由它重算的均值
                       # 与落盘值最多差 ~1e-6（11 行的舍入误差），容差取 2e-6
                       sources=("results.json::far_source_probe.*.ce_by_row_*",),
                       abs_tol=2e-6)
        ledger.derived(f"{name}: ce_far_clean == mean(ce_by_row_clean)",
                       row["ce_far_clean"],
                       lambda row=row: round(_mean(row["ce_by_row_clean"]), 6),
                       sources=("results.json::far_source_probe.*.ce_by_row_clean",),
                       abs_tol=2e-6)

    # ---- 判据①：无通道臂必须逐位不敏感 ----
    for sd in cfg["seeds"]:
        ledger.derived(f"判据① 无通道 seed{sd} 的被评行 max|Δlogit| == 0",
                       R["far_source_probe"][f"rt_none_s{sd}"][
                           "max_abs_logit_diff_on_eval_rows"],
                       lambda: 0.0,
                       sources=("results.json::far_source_probe.rt_none_s*/"
                                "max_abs_logit_diff_on_eval_rows",), abs_tol=1e-12)

    # ---- 质量腿效应与 σ_cell ----
    q = R["quality_leg"]
    for sd in cfg["seeds"]:
        ledger.derived(f"seed{sd}: Δbpc_win16 == bpc_none − bpc_delta",
                       R["effects_by_seed"][str(sd)]["bpc_win16"]["delta_bpc"],
                       lambda sd=sd: round(q[f"rt_none_s{sd}"]["bpc_win16"]
                                           - q[f"rt_delta_s{sd}"]["bpc_win16"], 6),
                       sources=("results.json::quality_leg.*.bpc_win16",), abs_tol=5e-7)
    ledger.derived("σ_cell(bpc) == 同配置换种子极差的最大值",
                   R["sigma_cell"]["bpc_win16"],
                   lambda: round(max(abs(q["rt_none_s0"]["bpc_win16"] - q["rt_none_s1"]["bpc_win16"]),
                                     abs(q["rt_delta_s0"]["bpc_win16"] - q["rt_delta_s1"]["bpc_win16"])), 6),
                   sources=("results.json::quality_leg.*.bpc_win16",), abs_tol=5e-7)
    ledger.derived("σ_cell(Δbpc_far) == 同配置换种子极差的最大值",
                   R["sigma_cell"]["delta_bpc_far"],
                   lambda: round(max(
                       abs(R["far_source_probe"]["rt_none_s0"]["delta_bpc_far"]
                           - R["far_source_probe"]["rt_none_s1"]["delta_bpc_far"]),
                       abs(R["far_source_probe"]["rt_delta_s0"]["delta_bpc_far"]
                           - R["far_source_probe"]["rt_delta_s1"]["delta_bpc_far"])), 6),
                   sources=("results.json::far_source_probe.*.delta_bpc_far",), abs_tol=5e-7)

    # ---- 判据②的布尔值与数字自洽 ----
    trans = {}
    for sd in cfg["seeds"]:
        row = R["far_source_probe"][f"rt_delta_s{sd}"]
        trans[sd] = bool(row["delta_bpc_far"] > 0
                         and row["delta_bpc_far"] > R["sigma_cell"]["delta_bpc_far"]
                         and row["delta_ci95_clustered"][0] > 0)
    ledger.eq("判据② == 逐种子 (Δ>0 ∧ Δ>σ_far ∧ CI下界>0)",
              R["criteria"]["c2_channel_transports_on_realtext"] == all(trans.values()), True,
              derived_from=("results.json::far_source_probe",))

    # ---- needle 腿：从 needle_raw.json 的逐针原始记录重算 ----
    arms = [a["name"] for a in cfg["arms"]]
    L = cfg["needle"]["phrase_len"]
    for name in arms:
        p = [r for r in raw[f"{name}|planted"] if r["bucket"] == "beyond"]
        c = [r for r in raw[f"{name}|control"] if r["bucket"] == "beyond"]
        rec = R["needle_leg"]["paired_beyond_vs_floor"][name]
        ledger.derived(f"{name}: beyond 桶针数 == 逐针记录里 bucket=='beyond' 的条数",
                       rec["n"], lambda p=p: len(p),
                       sources=("needle_raw.json",), abs_tol=1e-9)
        ledger.derived(f"{name}: planted 命中数 == Σ hit",
                       rec["planted_hits"], lambda p=p: sum(r["hit"] for r in p),
                       sources=("needle_raw.json",), abs_tol=1e-9)
        ledger.derived(f"{name}: control 命中数 == Σ hit",
                       rec["control_hits"], lambda c=c: sum(r["hit"] for r in c),
                       sources=("needle_raw.json",), abs_tol=1e-9)
        ledger.derived(f"{name}: Δpp == 100·(planted−control)/n",
                       rec["delta_pp"],
                       lambda p=p, c=c: round(100 * (sum(r["hit"] for r in p)
                                                     - sum(r["hit"] for r in c)) / len(p), 3),
                       sources=("needle_raw.json",), abs_tol=1e-3)
        ledger.eq(f"{name}: 控制臂与主臂针位逐一配对（同数、同 dist）",
                  [r["dist"] for r in p] == [r["dist"] for r in c], True)
    ledger.eq("needle 桶只有 window/beyond（无锚区）",
              sorted({r["bucket"] for n in arms
                      for r in raw[f"{n}|planted"]}) == ["beyond", "window"], True)

    # ---- 布局前提（rf 接线；复核 P1-1）----
    for name, row in R["far_source_probe"].items():
        ledger.derived(f"{name}: layout.required_min_eval_lo == corrupt_hi + rf",
                       row["layout"]["required_min_eval_lo"],
                       lambda row=row: row["layout"]["corrupt_prefix"][1]
                       + row["layout"]["receptive_field_positions"],
                       sources=("results.json::far_source_probe.*.layout",), abs_tol=1e-9)
        ledger.eq(f"{name}: rf == n_layers·(window−1)",
                  row["layout"]["receptive_field_positions"]
                  == cfg["model"]["n_layers"] * (cfg["model"]["train_window"] - 1), True,
                  derived_from=("config.json::model",))

    # ---- needle 腿：判据方向 + 同索引硬子集（复核 P1-2 / P1-4）----
    bplant = {n: [r for r in raw[f"{n}|planted"] if r["bucket"] == "beyond"] for n in arms}
    bctrl = {n: [r for r in raw[f"{n}|control"] if r["bucket"] == "beyond"] for n in arms}
    for name in arms:
        rec = R["needle_leg"]["judgement"]["per_arm"][name]
        ledger.derived(f"{name}: 判据④ delta_pp == 植入 − 地板（方向固化）",
                       rec["delta_pp"],
                       lambda name=name: round(100 * (
                           sum(r["hit"] for r in bplant[name]) / len(bplant[name])
                           - sum(r["hit"] for r in bctrl[name]) / len(bctrl[name])), 3),
                       sources=("needle_raw.json",), abs_tol=1e-3)
        ledger.eq(f"{name}: passes == (植入 − 地板 的 95% CI 下界 > 0)",
                  rec["passes"] == ns.beyond_above_floor(
                      sum(r["hit"] for r in bplant[name]), len(bplant[name]),
                      sum(r["hit"] for r in bctrl[name]), len(bctrl[name]))["passes"], True,
                  derived_from=("needle_stats.beyond_above_floor",))
    sets = ns.hard_index_sets(bctrl)
    ledger.derived("同索引硬子集：intersection == 四臂 control 全失败（同一组索引）",
                   R["needle_leg"]["hard_index_sets"]["n_intersection"],
                   lambda: len(sets["intersection"]), sources=("needle_raw.json",), abs_tol=1e-9)
    cross = R["needle_leg"]["hard_subset_same_index_cross_arm"]["intersection"]["delta_vs_none"]
    for key, val in cross.items():
        dname, none_name = key.split("_minus_")
        ledger.derived(f"同索引跨臂 {key}: delta_pp == 100·(植入Δ/n)（同一索引集）",
                       val["delta_pp"],
                       lambda d=dname, n_=none_name: round(100 * (
                           sum(bplant[d][i]["hit"] for i in sets["intersection"])
                           - sum(bplant[n_][i]["hit"] for i in sets["intersection"]))
                           / len(sets["intersection"]), 3),
                       sources=("needle_raw.json", "needle_stats.hard_index_sets"),
                       abs_tol=1e-3)

    # ---- §20 记账（锁口径 / 墙钟 / 峰值；复核 P2-3：此前 70 项对账不覆盖这一条）----
    import lock_accounting as _la
    la_path = os.path.join(HERE, "lock_accounting.json")
    if os.path.exists(la_path):
        la = json.load(open(la_path))
        cost = R["cost"]
        ledger.eq("§20：内嵌记账与权威产物逐字段相等",
                  cost.get("lock_accounting") == la, True,
                  derived_from=("lock_accounting.json", "results.json::cost"))
        ledger.derived("§20：评测墙钟 == lock_accounting.json::wall_s",
                       cost.get("lock_accounting", {}).get("wall_s"),
                       lambda: la["wall_s"], sources=("lock_accounting.json",), abs_tol=1e-9)
        ledger.eq("§20：lock_note 与记账同口径（口径一致性）",
                  (lambda: (_la.assert_consistent({"lock_note": cost.get("lock_note"),
                                                   "lock_accounting": la}), True)[1])(), True,
                  derived_from=("lock_accounting.json", "results.json::cost.lock_note"))
        ledger.eq("§20：lock_note 逐字 == 由记账派生",
                  cost.get("lock_note") == _la.derive_note(la), True,
                  derived_from=("lock_accounting.derive_note",))
        ledger.eq("§20：空闲余量门槛判定自洽",
                  la.get("headroom_ok") == (min(la.get("free_gpu_mem_mib_before", 0),
                                               la.get("free_gpu_mem_mib_after", 0))
                                            >= la.get("free_headroom_gate_mib", 0)), True,
                  derived_from=("lock_accounting.json",))

    # ---- 训练侧：末步与活性 ----
    steps = cfg["train_recipe"]["steps"]
    for name in arms:
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        lines = [json.loads(l) for l in open(os.path.join(ROOT, spec["train_log"]))
                 if l.strip()]
        ledger.derived(f"{name}: 训练日志末步 == {steps}", R["train_log_tail"][name]["last_step"],
                       lambda lines=lines: lines[-1]["step"], sources=(spec["train_log"],),
                       abs_tol=1e-9)
    carry_arms = [a["name"] for a in cfg["arms"] if a["rule"] != "none"]
    ledger.eq("通道臂评测期读出份额 > 0（干预活着）",
              all(R["liveness"][n]["read_share_per_layer"] > 0 for n in carry_arms), True)
    ledger.eq("无通道臂没有 carry 活性块（不是死的干预，而是没有该干预）",
              [n for n in arms if n not in R["liveness"]]
              == [a["name"] for a in cfg["arms"] if a["rule"] == "none"], True)

    return ledger.finish(os.path.join(HERE, "reconcile_report.json"))


if __name__ == "__main__":
    raise SystemExit(main())
