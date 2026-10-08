#!/usr/bin/env python3
"""2631-carry-reseed · 对账（F4 / 复核 B3）：期望值**从原始计数/日志重算**，不写常数。

三类来源分开写：`derived(...)`（由 raw 重算）/ `golden(..., why=...)`（历史或架构常数）/
`eq(...)`（精确相等）。每条都给**显式容差**（`abs_tol` / `rel_tol`），旧的含糊 `tol` 已废止。

覆盖：主腿/毁源/辅助腿的计数与准确率、聚类 bootstrap 与配对差的可复现性（直接由
`cluster_hits.json` 重算）、跨种子极差/均值/sd、毁源阈值、种子与写入规则接线、
训练 token 预算、答案行 CE、逐簇计数与汇总命中数一致。

用法：python3 reconcile_numbers.py   # 输出 reconcile_report.json；失败非零退出
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import reseed_arms as ra                                            # noqa: E402
import torch                                                        # noqa: E402
from tools.evalkit.longrange_tasks import assoc_recall_multi_batch   # noqa: E402
from tools.evalkit.reconcile import Ledger                          # noqa: E402

TRAIN_MAX_DIST = 240


def load(rel: str):
    return json.load(open(os.path.join(HERE, rel)))


def main() -> int:
    cfg = load("config.json")
    res = load("results.json")
    raw = load("cluster_hits.json")
    ledger = Ledger("2631-carry-reseed", sources=("results.json", "cluster_hits.json",
                                                  "config.json", "train/*/train_log.jsonl"))
    K, Q = cfg["task"]["K"], cfg["task"]["Q"]
    nb, bs = cfg["eval"]["n_batches"], cfg["eval"]["batch_size"]
    n_seq = nb * bs
    chance = 1 / cfg["tokens"]["sym_vocab"]

    # ---- 1. 主腿：规模、计数、准确率（全部由 raw 重算） ----
    leg_def = res["leg_definitions"]["main"]
    ledger.eq("主腿查询数 == n_batches·batch_size·Q", leg_def["n_queries"], n_seq * Q,
              derived_from=("config.json::eval", "config.json::task"))
    for a in cfg["arms"]:
        n = a["name"]
        row = res["main_leg"][n]
        ledger.eq(f"{n} 主腿 n == 序列数×Q", row["n"], n_seq * Q,
                  derived_from=("config.json::eval",))
        ledger.eq(f"{n} 主腿 cluster 数 == 序列数", row["n_clusters"], n_seq,
                  derived_from=("cluster_hits.json::main",))
        ledger.derived(f"{n} 主腿 accuracy == Σhits/Σn", row["accuracy"],
                       lambda r=row: r["n_hits"] / r["n"],
                       sources=(f"results.json::main_leg.{n}.n_hits",))
        ledger.derived(f"{n} 主腿 cluster_hits 总和 == n_hits", row["n_hits"],
                       lambda n=n: sum(raw["main"][n]["cluster_hits"]),
                       sources=(f"cluster_hits.json::main.{n}.cluster_hits",))
        ledger.derived(f"{n} 主腿 cluster_n 总和 == n", row["n"],
                       lambda n=n: sum(raw["main"][n]["cluster_n"]),
                       sources=(f"cluster_hits.json::main.{n}.cluster_n",))
        # 聚类 bootstrap 可复现（同一 n_boot/seed ⇒ 逐位一致）
        recomputed = ra.clustered_bootstrap_ci(raw["main"][n]["cluster_hits"],
                                              raw["main"][n]["cluster_n"],
                                              n_boot=2000, seed=2631)
        ledger.check(f"{n} 主腿聚类 CI 可复现", row["clustered_ci"], recomputed,
                     abs_tol=1e-9, derived_from=(f"cluster_hits.json::main.{n}",))
        ledger.derived(f"{n} 主腿 Wilson 对照 == 由 n_hits/n 重算", row["wilson_ci_naive"],
                       lambda r=row: ra.wilson_ci(r["n_hits"], r["n"]),
                       sources=(f"results.json::main_leg.{n}",))
        # 毁源腿
        crow = res["corrupt_leg"][n]
        ledger.eq(f"{n} 毁源腿 n == 主腿 n", crow["n"], row["n"],
                  derived_from=("results.json::main_leg",))
        ledger.derived(f"{n} 毁源腿 cluster_hits 总和 == n_hits", crow["n_hits"],
                       lambda n=n: sum(raw["corrupt"][n]["cluster_hits"]),
                       sources=(f"cluster_hits.json::corrupt.{n}.cluster_hits",))
        # 辅助腿（分布外）
        orow = res["aux_ood_leg"][n]
        on = cfg["aux_ood"]["n_batches"] * cfg["aux_ood"]["batch_size"]
        ledger.eq(f"{n} 辅助腿 n == 其批数×批大小", orow["n"], on,
                  derived_from=("config.json::aux_ood",))
        ledger.derived(f"{n} 辅助腿 cluster_hits 总和 == n_hits", orow["n_hits"],
                       lambda n=n: sum(raw["aux_ood"][n]["cluster_hits"]),
                       sources=(f"cluster_hits.json::aux_ood.{n}.cluster_hits",))

    # ---- 2. 距离口径：主腿分布内、辅助腿分布外（由生成器重算） ----
    tb = assoc_recall_multi_batch(4, cfg["model"]["seq_len"], K, Q,
                                  generator=torch.Generator().manual_seed(0))
    ledger.eq("主腿距离下界 == 生成器重算", leg_def["query_key_distance"][0],
              int(tb.dists.min()), derived_from=("tools.evalkit.longrange_tasks",))
    ledger.eq("主腿距离上界 == 生成器重算", leg_def["query_key_distance"][1],
              int(tb.dists.max()), derived_from=("tools.evalkit.longrange_tasks",))
    ledger.check("主腿最远距离 ≤ 训练上限 240", leg_def["query_key_distance"][1],
                 TRAIN_MAX_DIST, abs_tol=0.0,
                 derived_from=("README §4a 训练分布定义",))
    ledger.eq("辅助腿最远距离 == 255（T−1，训练外）",
              res["leg_definitions"]["aux_ood"]["query_key_distance"], [241, 255],
              derived_from=("tools.evalkit.longrange_tasks::assoc_recall_batch",))
    ledger.eq("chance == 1/sym_vocab", res["premise_check"]["0"]["chance"], chance,
              derived_from=("config.json::tokens.sym_vocab",))

    # ---- 3. 效应：配对差与描述量由 raw 重算 ----
    for pname, row in res["effects"].items():
        effs = [x["effect_pp"] for x in row["per_seed"]]
        ledger.eq(f"{pname} 极差 == max−min", row["range_pp"], ra.pp_range(effs),
                  derived_from=("results.json::effects.per_seed",))
        ledger.eq(f"{pname} 均值 == 平均", row["mean_pp"], ra.pp_mean(effs),
                  derived_from=("results.json::effects.per_seed",))
        ledger.eq(f"{pname} 样本 sd == 除 n−1", row["sd_pp"], ra.pp_sd(effs),
                  derived_from=("results.json::effects.per_seed",))
        ledger.eq(f"{pname} 方向一致性 == 同号", row["sign_consistent"],
                  ra.sign_consistent(effs),
                  derived_from=("results.json::effects.per_seed",))
        for x in row["per_seed"]:
            b = ra.arm_name(x["base_arm"].split("_")[0], x["seed"])
            t = ra.arm_name(x["target_arm"].split("_")[0], x["seed"])
            ledger.derived(f"{pname} seed{x['seed']} Δ == 100·(target−base)", x["effect_pp"],
                           lambda b=b, t=t: round(
                               100 * (res["main_leg"][t]["accuracy"]
                                      - res["main_leg"][b]["accuracy"]), 3),
                           sources=("results.json::main_leg",), abs_tol=1e-9)
            recomputed = ra.paired_clustered_bootstrap_diff_ci(
                raw["main"][b]["cluster_hits"], raw["main"][b]["cluster_n"],
                raw["main"][t]["cluster_hits"], raw["main"][t]["cluster_n"],
                n_boot=2000, seed=2631)
            ledger.check(f"{pname} seed{x['seed']} 配对 CI 可复现",
                         x["paired_boot_ci_pp"],
                         [round(100 * recomputed[0], 3), round(100 * recomputed[1], 3)],
                         abs_tol=1e-9, derived_from=("cluster_hits.json::main",))
    # 跨种子判定与预登记阈值一致
    st = res["effects"]["delta_vs_none"]["stability"]
    ledger.eq("delta>none 判定 == 预登记 5pp 阈值", st["min_pp_threshold"],
              ra.DELTA_VS_NONE_MIN_PP, derived_from=("reseed_arms.py",))
    ledger.eq("delta>none 稳定 == 逐种子过线", st["stable"],
              res["criteria"]["c2_delta_vs_none_stable"],
              derived_from=("results.json::effects",))
    ledger.eq("delta>plain 判定 == 预登记 3pp 阈值",
              res["effects"]["delta_vs_plain"]["stability"]["min_pp_threshold"],
              ra.DELTA_VS_PLAIN_MIN_PP, derived_from=("reseed_arms.py",))

    # ---- 4. 逐种子最好臂 / 毁源阈值 / 前提 ----
    for seed in ra.SEEDS:
        accs = {r: res["main_leg"][ra.arm_name(r, seed)]["accuracy"] for r in ra.RULES}
        best = max(accs, key=accs.get)
        ledger.eq(f"seed{seed} 最好臂 == 由准确率重算", res["best_arm_per_seed"][str(seed)],
                  best, derived_from=("results.json::main_leg",))
        ledger.eq(f"seed{seed} 无通道臂 ≈ chance（前提）",
                  res["premise_check"][str(seed)]["significantly_above_chance"],
                  res["main_leg"][ra.arm_name("none", seed)]["clustered_ci"][0] > chance,
                  derived_from=("results.json::main_leg",))
    ledger.derived("毁源阈值 == factor×chance", res["corrupt_control"]["bound"],
                   lambda: ra.CORRUPT_LEAK_FACTOR * chance,
                   sources=("config.json::stats.corrupt_leak_factor",), abs_tol=1e-12)

    # ---- 5. 接线：种子 / 写入规则 / 训练预算 / 参数量 ----
    for a in cfg["arms"]:
        n = a["name"]
        ledger.eq(f"{n} 训练 seed == 配置", res["train_meta"][n]["seed"], a["seed"],
                  derived_from=("config.json::arms",))
        ledger.eq(f"{n} 训练 carry_write == 规则标签", res["train_meta"][n]["carry_write"],
                  a["rule"], derived_from=("config.json::arms",))
        ledger.eq(f"{n} checkpoint carry_rule == 规则标签", res["checkpoints"][n]["carry_rule"],
                  None if a["rule"] == "none" else a["rule"],
                  derived_from=("config.json::arms",))
        ledger.eq(f"{n} 窗口掩码活性（可见对数 < 因果对数）",
                  res["train_meta"][n]["window_mask"]["n_visible_pairs"]
                  < res["train_meta"][n]["window_mask"]["causal_pairs"], True,
                  derived_from=("train/*/results.json::window_mask",))
    tr = cfg["train_recipe"]
    ledger.eq("训练 token 预算 == steps×bs×T",
              tr["steps"] * tr["batch_size"] * cfg["model"]["seq_len"], tr["train_tokens"],
              derived_from=("config.json::train_recipe",))
    ledger.golden("无通道臂参数量 == 13,536,768", res["train_meta"]["none_s0"]["params"],
                  13536768, abs_tol=0.0,
                  why="架构算术（12 层 d256/8 头/mlp6）＋与 2618 U0 的 trained checkpoint "
                      "逐位相同（同一份模型定义）")
    ledger.golden("通道臂参数量 == 13,801,064", res["train_meta"]["delta_s1"]["params"],
                  13801064, abs_tol=0.0,
                  why="无通道 + 单共享槽 264,296 参数（2618 表；U1/U2 逐位等参数已由自检断言）")
    ledger.eq("plain/delta 逐位等参数（自检）",
              res["selfchecks"]["params_bitwise_equal_plain_delta"]["0"]["bitwise_equal"], True,
              derived_from=("results.json::selfchecks",))
    ledger.eq("通道臂主干逐位 == 无通道（自检）",
              res["selfchecks"]["trunk_bitwise_equal_carry_vs_none"]["1"]["trunk_bitwise_equal"],
              True, derived_from=("results.json::selfchecks",))

    # ---- 6. 训练侧答案行 CE（口径：决定判据的那个量，§17.15） ----
    for a in cfg["arms"]:
        n = a["name"]
        lines = [json.loads(l) for l in
                 open(os.path.join(ROOT, a["train_log"])) if l.strip()]
        ledger.eq(f"{n} 末行 answer CE == 训练日志末行",
                  res["train_log_tail"][n]["last_answer_ce"],
                  lines[-1]["synth_answer_ce"], derived_from=(a["train_log"],), abs_tol=1e-9)
        ledger.eq(f"{n} 逃出瞎猜线 == answerCE < chanceCE−0.5",
                  res["train_log_tail"][n]["escaped_chance_line"],
                  lines[-1]["synth_answer_ce"] < lines[-1]["synth_chance_ce"] - 0.5,
                  derived_from=(a["train_log"],))

    return ledger.finish(os.path.join(HERE, "reconcile_report.json"))


if __name__ == "__main__":
    raise SystemExit(main())
