"""2624-composable-followups-13m · 对账（SCHEMA §12 三道 + §9 仪器一致性）。

`analysis.md` 里的每个数字都必须能在 `results.json`（及其引用的训练日志）里找到同值，
且算术自洽：

1. **算术对账**：准确率 = hits/n（含逐查询块）、增量 = 两点差、容量斜率与插值、
   bpc 差分、参数量差——全部从 `results.json` 的原始计数重算。
2. **任务对账**：同一条腿内各臂的 n / chance / local_prior 相同；毁源对照同批；
   **Q=8 腿的末块是「唯一没被问过的一对」（伪影条件成立）、Q=7 腿不是**
   （谓词直接在生成的 batch 上重算，不引用结果里的自述字段）；距离区间与训练分布分开。
3. **口径对账**：基准唯一（`stock`）；两个主判据（① 的插值基准、③ 的 6k 点）显式；
   ① 声明的「不是逐位等参数臂」必须为真；③ 两对同相对刻度；收敛门禁落在答案行 CE 上。
4. **仪器一致性（§9 对照复用规则 1）**：本次重评 2623 的四臂读数必须与
   `2623-composable-arms-13m/results.json` **逐位相同**（判决腿 / Q=8 分布内腿 /
   去伪影块 / clean bpc / 毁源）——否则本线与 2623 不是同一件仪器，跨实验读法作废。

退出码 0 = 全过；非 0 = 有对账失败（并打印失败清单）。
"""

import json
import math
import os
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from points import (ARMS, ARM_NAMES, BASELINE_ARM, P1_TARGET, P2_ARMS,  # noqa: E402
                    P3_PAIRS, Q_LEAKFREE, Q_TRAIN, arm_spec, capacity_slope,
                    interp_at_target, leak_free_leg_gap, pick_index,
                    unique_unqueried_last_block)
from tools.evalkit.longrange_tasks import assoc_recall_multi_batch  # noqa: E402


def main():
    R = json.load(open(os.path.join(HERE, "results.json")))
    REF = json.load(open(os.path.join(
        ROOT, "axes/1-time/1.1.4.1-explicit-memory/experiments",
        "2623-composable-arms-13m/results.json")))
    checks: list[dict] = []

    def ck(name, got, exp, tol=0.0):
        ok = (abs(float(got) - float(exp)) <= tol
              if isinstance(got, (int, float)) and isinstance(exp, (int, float))
              else got == exp)
        checks.append({"check": name, "want": exp, "got": got, "pass": bool(ok)})

    cols = list(R["checkpoints"])          # 本次实际评测的臂（= config 里的臂）

    # ---------------- ① 算术对账 ----------------
    for leg in ("main_leg", "cross_leg_full_causal", "train_task_leg_q8",
                "train_task_leg_q7_leakfree"):
        for n in cols:
            a = R[leg][n]
            ck(f"{leg}.{n}.acc==hits/n", a["accuracy"], a["n_hits"] / a["n"], 1e-6)
            ck(f"{leg}.{n}.ci_contains_acc",
               a["accuracy_ci"][0] <= a["accuracy"] <= a["accuracy_ci"][1], True)
            ck(f"{leg}.{n}.corrupt==hits/n", a["corrupt_source"]["accuracy"],
               a["corrupt_source"]["n_hits"] / a["corrupt_source"]["n"], 1e-6)
            if "by_query_block" not in a:
                continue                    # 单查询腿（判决腿 / 因果交叉腿）没有块结构
            # 逐查询块：n 之和 = 总 n；每块 acc = hits/n
            tot = sum(v["n"] for v in a["by_query_block"].values())
            ck(f"{leg}.{n}.blocks_n_sum", tot, a["n"])
            hit = sum(v["n_hits"] for v in a["by_query_block"].values())
            ck(f"{leg}.{n}.blocks_hits_sum", hit, a["n_hits"])
            for j, v in a["by_query_block"].items():
                ck(f"{leg}.{n}.block{j}.acc", v["accuracy"],
                   v["n_hits"] / max(v["n"], 1), 1e-6)
            # 逐块毁源同批同 n
            ck(f"{leg}.{n}.corrupt_blocks_same_n",
               [v["n"] for v in a["by_query_block"].values()],
               [v["n"] for v in a["by_query_block_corrupt"].values()])
            # by_distance 与总计数自洽
            dn = sum(v["n"] for v in a["by_distance"].values())
            ck(f"{leg}.{n}.by_dist_n_sum", dn, a["n"])

    # ① 的容量括号：斜率与插值从 results.json 的读数重算
    P1 = R["p1_gqa_capacity_bracket"]
    ck("p1.not_bitwise_equal_bracket", P1["declared_not_bitwise_equal_params"], True)
    lo, hi = P1["bracket_arms"]
    for metric in ("clean_bpc", "main_leg_acc", "q7_leakfree_acc",
                   "q7_mid_blocks_acc"):
        m = P1["metrics"][metric]
        y_lo, y_hi = m["mha_bracket"][lo], m["mha_bracket"][hi]
        x_lo, x_hi = ARMS[lo]["params"], ARMS[hi]["params"]
        x_t = ARMS[P1_TARGET]["params"]
        slope = capacity_slope(x_lo, x_hi, y_lo, y_hi)
        want_slope = round(slope * 1e6, 9) if metric == "clean_bpc" \
            else round(slope * 1e6 * 100, 3)
        ck(f"p1.{metric}.slope", m["mha_slope_per_1M_params"], want_slope, 1e-9)
        y_i = interp_at_target(x_t, x_lo, x_hi, y_lo, y_hi)
        ck(f"p1.{metric}.interp", m["mha_interp_at_gqa_params"], round(y_i, 6), 1e-6)
        obs = m["gqa_observed"]
        res = round(obs - y_i, 6) if metric == "clean_bpc" else round((obs - y_i) * 100, 3)
        ck(f"p1.{metric}.residual", m["gqa_residual_vs_interp"], res, 1e-9)
        # 读数必须来自本实验的同一批腿
        if metric == "clean_bpc":
            ck(f"p1.{metric}.obs==bpc_leg", obs, R["bpc_leg"][P1_TARGET]["bpc"], 1e-9)
        elif metric == "main_leg_acc":
            ck(f"p1.{metric}.obs==main_leg", obs,
               R["main_leg"][P1_TARGET]["accuracy"], 1e-9)
        elif metric == "q7_leakfree_acc":
            ck(f"p1.{metric}.obs==q7_leg", obs,
               R["train_task_leg_q7_leakfree"][P1_TARGET]["accuracy"], 1e-9)
        else:
            # 只取前 Q−1 个查询块（甩掉仍带块序梯度的末块）
            v = R["train_task_leg_q7_leakfree"][P1_TARGET]
            Ql_ = R["train_task_leg_q7_leakfree"][P1_TARGET]["Q"]
            n = sum(v["by_query_block"][str(j)]["n"] for j in range(Ql_ - 1))
            h = sum(v["by_query_block"][str(j)]["n_hits"] for j in range(Ql_ - 1))
            ck(f"p1.{metric}.obs==q7_mid_blocks", obs, round(h / n, 6), 1e-9)
        ck(f"p1.{metric}.bracket_inside", x_lo < x_t < x_hi, True)

    # ③ 的增量：从两条腿的读数重算
    P3 = R["p3_carry_6k_vs_10k"]
    # carry 增量 = (同一预算下的) 有 carry 臂 − 无 carry 臂；两点各用自己那一对
    CARRY_PAIRS = {6000: ("gqa", "gqa_carry"), 10000: ("gqa_10k", "gqa_carry_10k")}
    for steps, (a, b) in CARRY_PAIRS.items():
        inc = P3["carry_increment"][str(steps)]
        ck(f"p3.{steps}.target_pp", inc["target_pp"],
           round(100 * (R["main_leg"][b]["accuracy"] - R["main_leg"][a]["accuracy"]), 3),
           1e-9)
        ck(f"p3.{steps}.q7_pp", inc["q7_leakfree_pp"],
           round(100 * (R["train_task_leg_q7_leakfree"][b]["accuracy"] -
                        R["train_task_leg_q7_leakfree"][a]["accuracy"]), 3), 1e-9)
        ck(f"p3.{steps}.bpc_delta", inc["bpc_delta"],
           round(R["bpc_leg"][b]["bpc"] - R["bpc_leg"][a]["bpc"], 6), 1e-9)
    ck("p3.retention_delta", P3["increment_retention"]["delta_pp"],
       round(P3["carry_increment"]["10000"]["target_pp"] -
             P3["carry_increment"]["6000"]["target_pp"], 3), 1e-9)

    # 参数量账
    n_base = R["param_declaration"][BASELINE_ARM]["n_params"]
    for n in cols:
        d = R["param_declaration"][n]
        ck(f"param.{n}.delta", d["delta_vs_base"], d["n_params"] - n_base)
        ck(f"param.{n}.equals_python_truth", d["n_params"], ARMS[n]["params"])
    ck("param.window_adds_none", ARMS["gqa_carry_window"]["params"],
       ARMS["gqa_carry"]["params"])
    ck("param.gqa_minus_14pct_below_stock", n_base - ARMS["gqa"]["params"], 1_184_256)

    # bpc 差分
    for n in cols:
        ck(f"bpc.damage.{n}", R["bpc_damage_vs_stock"][n],
           round(R["bpc_leg"][n]["bpc"] - R["bpc_leg"][BASELINE_ARM]["bpc"], 6), 1e-9)

    # ---------------- ② 任务对账 ----------------
    for leg in ("main_leg", "train_task_leg_q8", "train_task_leg_q7_leakfree",
                "cross_leg_full_causal"):
        ck(f"task.{leg}.same_n", len({R[leg][n]["n"] for n in cols}), 1)
        ck(f"task.{leg}.same_chance", len({R[leg][n]["chance"] for n in cols}), 1)
        ck(f"task.{leg}.same_local_prior",
           len({R[leg][n]["local_prior"] for n in cols}), 1)
        ck(f"task.{leg}.corrupt_same_n",
           len({R[leg][n]["corrupt_source"]["n"] for n in cols}), 1)
        ck(f"task.{leg}.corrupt_span",
           {n: R[leg][n]["corrupt_source"]["span"] for n in cols},
           {n: [0, 2 * R[leg][n]["K"]] for n in cols})

    # 伪影谓词：不引用结果里的自述字段，直接在生成的 batch 上重算
    gap = R["config"]["train_task_leg"]["gap"]
    T = R["config"]["model"]["seq_len"]
    K = R["config"]["train"]["synth_pairs"]
    Q7 = leak_free_leg_gap(T, K, gap)
    ck("task.q7_leg_Q", R["train_task_leg_q7_leakfree"][BASELINE_ARM]["Q"], Q7)
    ck("task.q7_leg_Q_is_K_minus_1", Q7, K - 1)
    tb8 = assoc_recall_multi_batch(4, T, K, Q_TRAIN, gap=gap,
                                   generator=torch.Generator().manual_seed(11))
    tb7 = assoc_recall_multi_batch(4, T, K, Q7, gap=gap,
                                   generator=torch.Generator().manual_seed(11))
    ck("task.q8_last_block_leaky_predicate",
       all(unique_unqueried_last_block(tb8.rows, tb8.dists, T, Q_TRAIN, K, gap)), True)
    ck("task.q7_last_block_not_leaky_predicate",
       any(unique_unqueried_last_block(tb7.rows, tb7.dists, T, Q7, K, gap)), False)
    ck("task.result_q8_leaky_flag",
       R["train_task_leg_q8"][BASELINE_ARM]["leaky_last_block_all_rows"], True)
    ck("task.result_q7_leaky_flag",
       R["train_task_leg_q7_leakfree"][BASELINE_ARM]["leaky_last_block_all_rows"], False)
    # Q=8 末块四臂都 ≈100%（伪影的直接证据）；Q=7 末块明显低于 100%
    ck("task.q8_last_block_all_100pct",
       all(R["train_task_leg_q8"][n]["by_query_block"][str(Q_TRAIN - 1)]["accuracy"] == 1.0
           for n in cols), True)
    ck("task.q7_last_block_below_90pct",
       all(R["train_task_leg_q7_leakfree"][n]["by_query_block"][str(Q7 - 1)]["accuracy"] < 0.9
           for n in cols), True)
    ck("task.pick_coords_are_permutation",
       sorted(pick_index(tb8.rows, tb8.dists)[0].tolist()), list(range(K)))
    # 判决腿的距离全部 > 训练分布最大距离 240；分布内腿 ≤ 240
    ck("task.main.dist_beyond_train",
       all(R["main_leg"][n]["query_key_distance"]["min"] > 240 for n in cols), True)
    ck("task.q8.dist_max_240",
       all(R["train_task_leg_q8"][n]["query_key_distance"]["max"] == 240 for n in cols), True)
    ck("task.q7.dist_within_train",
       all(R["train_task_leg_q7_leakfree"][n]["query_key_distance"]["max"] <= 240
           for n in cols), True)

    # ---------------- ③ 口径对账 ----------------
    ck("caliber.baseline_is_stock", BASELINE_ARM, "stock")
    ck("caliber.three_points_present",
       all(R[k] is not None for k in ("p1_gqa_capacity_bracket", "p2_leakfree_q7",
                                      "p3_carry_6k_vs_10k")), True)
    ck("caliber.p2_arms_are_2623_checkpoints",
       all(ARMS[n]["source"] == "2623" and ARMS[n]["steps"] == 6000 for n in P2_ARMS), True)
    ck("caliber.p2_eval_only",
       all(R["checkpoints"][n]["steps"] == 6000 for n in P2_ARMS), True)
    ck("caliber.p3_same_ratio",
       sorted({p["ratio"] for p in P3["pairs"]}), [P3["scale_ratio"]])
    ck("caliber.p3_flags_identical_per_pair",
       [p["flags"] for p in P3["pairs"]],
       [arm_spec(a)["flags"] | {"mlp_mult": arm_spec(a)["mlp_mult"]}
        for a, _ in P3_PAIRS])
    ck("caliber.nonwindow_arms_native_full_causal",
       all(R["main_leg"][n]["native_window"] == 0
           for n in cols if arm_spec(n)["flags"].get("train_window", 0) == 0), True)
    ck("caliber.window_arm_native_window_16",
       R["main_leg"]["gqa_carry_window"]["native_window"], 16)
    ck("caliber.window_nondegenerate", R["window_nondegenerate"]["alive"], True)
    ck("caliber.gqa_kv_shape", R["kv_head_liveness"]["gqa"]["k_batch_shape"][1], 2)
    ck("caliber.gqa_reduction_75pct",
       R["kv_head_liveness"]["gqa"]["kv_reduction_frac"], 0.75)
    ck("caliber.destroy_all_valid",
       all(v["valid"] for v in R["destroy_source_validity"].values()), True)
    ck("caliber.bpc_all_worse_than_uniform",
       all(R["bpc_leg"][n]["bpc"] > math.log2(206) for n in cols), True)
    ck("caliber.convergence_on_answer_ce",
       all(R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"] is not None
           for n in cols), True)
    for n, sw in (("gqa", "kv_heads_series_present"),
                  ("gqa_10k", "kv_heads_series_present"),
                  ("gqa_carry", "attn_carry_series_present"),
                  ("gqa_carry_10k", "attn_carry_series_present"),
                  ("gqa_carry_window", "window_mask_series_present")):
        ck(f"caliber.activity.{n}.{sw}", R["train_activity"][n][sw] > 0, True)
    ck("caliber.eval_only", "train_wall_minutes" not in R["cost"], True)
    ck("caliber.peak_mem_under_5gb",
       (R["cost"]["peak_gpu_mem_mb"] or 0) < 5120.0, True)

    # ---------------- ④ 仪器一致性（§9）：与 2623 的四臂逐位相同 ----------------
    for n in P2_ARMS:
        ck(f"instr.{n}.main_acc", R["main_leg"][n]["accuracy"],
           REF["main_leg"][n]["accuracy"], 1e-9)
        ck(f"instr.{n}.main_hits", R["main_leg"][n]["n_hits"],
           REF["main_leg"][n]["n_hits"])
        ck(f"instr.{n}.main_corrupt", R["main_leg"][n]["corrupt_source"]["accuracy"],
           REF["main_leg"][n]["corrupt_source"]["accuracy"], 1e-9)
        ck(f"instr.{n}.cross_acc", R["cross_leg_full_causal"][n]["accuracy"],
           REF["cross_leg_full_causal"][n]["accuracy"], 1e-9)
        ck(f"instr.{n}.q8_acc", R["train_task_leg_q8"][n]["accuracy"],
           REF["train_task_leg"][n]["accuracy"], 1e-9)
        ck(f"instr.{n}.q8_no_last_acc", R["train_task_leg_q8_no_last_block"][n]["accuracy"],
           REF["train_task_leg_no_last_block"][n]["accuracy"], 1e-9)
        ck(f"instr.{n}.bpc", R["bpc_leg"][n]["bpc"], REF["bpc_leg"][n]["bpc"], 1e-9)
        ck(f"instr.{n}.distance_leg_d128", R["distance_leg"][n]["128"]["accuracy"],
           REF["distance_leg"][n]["128"]["accuracy"], 1e-9)
        ck(f"instr.{n}.activity_slope",
           R["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"],
           REF["train_activity"][n]["answer_ce_slope_nats_per_1k_lastquarter"], 1e-9)

    # ---------------- analysis.md 引用的关键数字 → results.json ----------------
    # 每条 = (名字, results.json 的取值, **analysis.md 里逐字出现的那个 token**, 单位刻度, 容差)。
    # 数值从 token 文本里**解析**（不是从 results.json 抄），再与 results.json 比；
    # 同时断言该 token **确实出现在 analysis.md 里** ⇒ 抓「文本与结果不符」。
    # —— analysis.md 引用逐条回填（数值 + 该数值**确实出现在 analysis.md 文本里**双检查）——
    R8 = R["train_task_leg_q8"]
    R8NL = R["train_task_leg_q8_no_last_block"]
    R7 = R["train_task_leg_q7_leakfree"]

    def num(token: str) -> float:
        """从 analysis.md 的 token 文本里解析第一个数值（支持 `−`/千分位/单位后缀）。"""
        m = re.search(r"[-−+]?\d[\d,]*\.?\d*", token)
        if not m:
            raise ValueError(f"token 里没有数值：{token!r}")
        return float(m.group().replace(",", "").replace("−", "-"))


    def q7_mid(arm: str) -> float:
        """Q=7 腿的**前 Q−1 个查询块**聚合（最保守的无泄漏口径）。"""
        v = R7[arm]
        Q = int(v["Q"])
        n = sum(v["by_query_block"][str(j)]["n"] for j in range(Q - 1))
        h = sum(v["by_query_block"][str(j)]["n_hits"] for j in range(Q - 1))
        return h / n

    T = R["p1_gqa_capacity_bracket"]["table"]
    M = R["p1_gqa_capacity_bracket"]["metrics"]
    P3 = R["p3_carry_6k_vs_10k"]
    AT = R["train_activity"]
    R8 = R["train_task_leg_q8"]
    R8NL = R["train_task_leg_q8_no_last_block"]
    R7 = R["train_task_leg_q7_leakfree"]
    cited = [
        ('5.1 mha_mlp5.params', T["mha_mlp5"]["n_params"], '11,960,832', 1.0, 0.0),
        ('5.1 mha_mlp5.bpc', T["mha_mlp5"]["bpc"], '23.246', 1.0, 0.0005),
        ('5.1 mha_mlp5.q7', T["mha_mlp5"]["q7_leakfree_acc"], '26.674%', 100.0, 6e-04),
        ('5.1 mha_mlp5.mid', q7_mid("mha_mlp5"), '22.493%', 100.0, 6e-04),
        ('5.1 gqa.params', T["gqa"]["n_params"], '12,352,512', 1.0, 0.0),
        ('5.1 gqa.bpc', T["gqa"]["bpc"], '37.145', 1.0, 0.0005),
        ('5.1 gqa.q7', T["gqa"]["q7_leakfree_acc"], '27.093%', 100.0, 6e-04),
        ('5.1 gqa.mid', q7_mid("gqa"), '22.786%', 100.0, 6e-04),
        ('5.1 stock.params', T["stock"]["n_params"], '13,536,768', 1.0, 0.0),
        ('5.1 stock.bpc', T["stock"]["bpc"], '13.069', 1.0, 0.0005),
        ('5.1 stock.q7', T["stock"]["q7_leakfree_acc"], '26.228%', 100.0, 6e-04),
        ('5.1 stock.mid', q7_mid("stock"), '22.591%', 100.0, 6e-04),
        ('5.1 gqa.main_leg', T["gqa"]["main_leg_acc"], '0.146%（3/2048）', 100.0, 6e-04),
        ('5.1 stock.main_leg', T["stock"]["main_leg_acc"], '0.000%（0/2048）', 100.0, 6e-04),
        ('5.1 mha.main_leg', T["mha_mlp5"]["main_leg_acc"], '0.000%（0/2048）', 100.0, 6e-04),
        ('5.1 interp.q7', M["q7_leakfree_acc"]["mha_interp_at_gqa_params"], '26.563%', 100.0, 6e-04),
        ('5.1 interp.q7mid', M["q7_mid_blocks_acc"]["mha_interp_at_gqa_params"], '22.518%', 100.0, 6e-04),
        ('5.1 interp.bpc', M["clean_bpc"]["mha_interp_at_gqa_params"], '20.717', 1.0, 0.0005),
        ('5.1 res.q7', M["q7_leakfree_acc"]["gqa_residual_vs_interp"], '+0.529pp', 1.0, 0.0005),
        ('5.1 res.q7mid', M["q7_mid_blocks_acc"]["gqa_residual_vs_interp"], '+0.269pp', 1.0, 0.0005),
        ('5.1 res.main', M["main_leg_acc"]["gqa_residual_vs_interp"], '+0.146pp', 1.0, 0.0005),
        ('5.1 res.bpc', M["clean_bpc"]["gqa_residual_vs_interp"], '+16.429', 1.0, 0.0005),
        ('5.1 slope.q7', M["q7_leakfree_acc"]["mha_slope_per_1M_params"], '−0.283 pp/1M 参数', 1.0, 0.0005),
        ('5.1 slope.q7mid', M["q7_mid_blocks_acc"]["mha_slope_per_1M_params"], '+0.062 pp/1M 参数', 1.0, 0.0005),
        ('5.1 slope.bpc', M["clean_bpc"]["mha_slope_per_1M_params"], '−6.458 bpc/1M 参数', 1.0, 0.0005),
        ('5.2 stock.q8', R8["stock"]["accuracy"], '35.010%', 100.0, 6e-04),
        ('5.2 stock.q8nl', R8NL["stock"]["accuracy"], '25.725%', 100.0, 6e-04),
        ('5.2 stock.q7', R7["stock"]["accuracy"], '26.228%', 100.0, 6e-04),
        ('5.2 stock.q7mid', q7_mid("stock"), '22.591%', 100.0, 6e-04),
        ('5.2 stock.q7last', R7["stock"]["by_query_block"]["6"]["accuracy"], '48.047%', 100.0, 6e-04),
        ('5.2 gqa.q8', R8["gqa"]["accuracy"], '35.840%', 100.0, 6e-04),
        ('5.2 gqa.q8nl', R8NL["gqa"]["accuracy"], '26.674%', 100.0, 6e-04),
        ('5.2 gqa.q7', R7["gqa"]["accuracy"], '27.093%', 100.0, 6e-04),
        ('5.2 gqa.q7mid', q7_mid("gqa"), '22.786%', 100.0, 6e-04),
        ('5.2 gqa.q7last', R7["gqa"]["by_query_block"]["6"]["accuracy"], '52.930%', 100.0, 6e-04),
        ('5.2 gqa_carry.q8', R8["gqa_carry"]["accuracy"], '36.230%', 100.0, 6e-04),
        ('5.2 gqa_carry.q8nl', R8NL["gqa_carry"]["accuracy"], '27.120%', 100.0, 6e-04),
        ('5.2 gqa_carry.q7', R7["gqa_carry"]["accuracy"], '27.288%', 100.0, 6e-04),
        ('5.2 gqa_carry.q7mid', q7_mid("gqa_carry"), '23.014%', 100.0, 6e-04),
        ('5.2 gqa_carry.q7last', R7["gqa_carry"]["by_query_block"]["6"]["accuracy"], '52.930%', 100.0, 6e-04),
        ('5.2 gqa_carry_window.q8', R8["gqa_carry_window"]["accuracy"], '34.766%', 100.0, 6e-04),
        ('5.2 gqa_carry_window.q8nl', R8NL["gqa_carry_window"]["accuracy"], '25.446%', 100.0, 6e-04),
        ('5.2 gqa_carry_window.q7', R7["gqa_carry_window"]["accuracy"], '26.088%', 100.0, 6e-04),
        ('5.2 gqa_carry_window.q7mid', q7_mid("gqa_carry_window"), '22.103%', 100.0, 6e-04),
        ('5.2 gqa_carry_window.q7last', R7["gqa_carry_window"]["by_query_block"]["6"]["accuracy"], '50.000%', 100.0, 6e-04),
        ('5.2 mha_mlp5.q8', R8["mha_mlp5"]["accuracy"], '35.620%', 100.0, 6e-04),
        ('5.2 mha_mlp5.q8nl', R8NL["mha_mlp5"]["accuracy"], '26.423%', 100.0, 6e-04),
        ('5.2 mha_mlp5.q7', R7["mha_mlp5"]["accuracy"], '26.674%', 100.0, 6e-04),
        ('5.2 mha_mlp5.q7mid', q7_mid("mha_mlp5"), '22.493%', 100.0, 6e-04),
        ('5.2 mha_mlp5.q7last', R7["mha_mlp5"]["by_query_block"]["6"]["accuracy"], '51.758%', 100.0, 6e-04),
        ('5.2 gqa_10k.q8', R8["gqa_10k"]["accuracy"], '35.767%', 100.0, 6e-04),
        ('5.2 gqa_10k.q8nl', R8NL["gqa_10k"]["accuracy"], '26.590%', 100.0, 6e-04),
        ('5.2 gqa_10k.q7', R7["gqa_10k"]["accuracy"], '27.009%', 100.0, 6e-04),
        ('5.2 gqa_10k.q7mid', q7_mid("gqa_10k"), '22.754%', 100.0, 6e-04),
        ('5.2 gqa_10k.q7last', R7["gqa_10k"]["by_query_block"]["6"]["accuracy"], '52.539%', 100.0, 6e-04),
        ('5.2 gqa_carry_10k.q8', R8["gqa_carry_10k"]["accuracy"], '36.108%', 100.0, 6e-04),
        ('5.2 gqa_carry_10k.q8nl', R8NL["gqa_carry_10k"]["accuracy"], '26.981%', 100.0, 6e-04),
        ('5.2 gqa_carry_10k.q7', R7["gqa_carry_10k"]["accuracy"], '27.316%', 100.0, 6e-04),
        ('5.2 gqa_carry_10k.q7mid', q7_mid("gqa_carry_10k"), '22.949%', 100.0, 6e-04),
        ('5.2 gqa_carry_10k.q7last', R7["gqa_carry_10k"]["by_query_block"]["6"]["accuracy"], '53.516%', 100.0, 6e-04),
        ('5.3 6000.delta', P3["carry_increment"]["6000"]["target_pp"], '15.527pp', 1.0, 0.0005),
        ('5.3 6000.nocarry', P3["carry_increment"]["6000"]["no_carry"], '0.146%', 100.0, 6e-04),
        ('5.3 6000.carry', P3["carry_increment"]["6000"]["carry"], '15.674%', 100.0, 6e-04),
        ('5.3 6000.q7delta', P3["carry_increment"]["6000"]["q7_leakfree_pp"], '+0.195pp', 1.0, 0.0005),
        ('5.3 6000.bpcdelta', P3["carry_increment"]["6000"]["bpc_delta"], '+21.163', 1.0, 0.0005),
        ('5.3 10000.delta', P3["carry_increment"]["10000"]["target_pp"], '14.697pp', 1.0, 0.0005),
        ('5.3 10000.nocarry', P3["carry_increment"]["10000"]["no_carry"], '2.148%', 100.0, 6e-04),
        ('5.3 10000.carry', P3["carry_increment"]["10000"]["carry"], '16.846%', 100.0, 6e-04),
        ('5.3 10000.q7delta', P3["carry_increment"]["10000"]["q7_leakfree_pp"], '+0.307pp', 1.0, 0.0005),
        ('5.3 10000.bpcdelta', P3["carry_increment"]["10000"]["bpc_delta"], '+20.018', 1.0, 0.0005),
        ('5.3 retention', P3["increment_retention"]["delta_pp"], '−0.830pp', 1.0, 0.0005),
        ('5.3 slope.gqa10k', AT["gqa_10k"]["answer_ce_slope_nats_per_1k_lastquarter"], '−0.03418', 1.0, 5e-06),
        ('5.3 slope.gqa', AT["gqa"]["answer_ce_slope_nats_per_1k_lastquarter"], '−0.07398', 1.0, 5e-06),
        ('5.3 slope.carry10k', AT["gqa_carry_10k"]["answer_ce_slope_nats_per_1k_lastquarter"], '−0.02397', 1.0, 5e-06),
        ('5.3 slope.carry6k', AT["gqa_carry"]["answer_ce_slope_nats_per_1k_lastquarter"], '−0.06019', 1.0, 5e-06),
        ('5.4 read_share.carry', AT["gqa_carry"]["last"]["attn_carry"]["read_share"], '`read_share=0.6050`', 1.0, 5e-05),
        ('5.4 read_share.carry10k', AT["gqa_carry_10k"]["last"]["attn_carry"]["read_share"], '`read_share=0.7014`', 1.0, 5e-05),
        ('5.4 read_scale.carry', AT["gqa_carry"]["last"]["attn_carry"]["read_scale_absmean"], '`read_scale_absmean=0.1073`', 1.0, 5e-05),
        ('5.4 beta.carry10k', AT["gqa_carry_10k"]["last"]["attn_carry"]["beta_mean"], '`beta_mean=0.4761`', 1.0, 5e-05),
        ('5.4 visible_frac', AT["gqa_carry_window"]["last"]["window_mask"]["visible_frac"], '`window_mask.visible_frac=0.1209`', 1.0, 5e-05),
        ('5.5 eval_min', R["cost"]["wall_minutes_total_eval"], '0.725 min', 1.0, 0.0005),
        ('5.5 peak_mem', R["cost"]["peak_gpu_mem_mb"], '702.2 MB', 1.0, 0.05),
    ]
    analysis_text = open(os.path.join(HERE, "analysis.md")).read()
    for _name, _got, _tok, _scale, _tol in cited:
        ck(f"cited {_name}", _scale * _got, num(_tok), _tol)
        ck(f"cited {_name}.in_analysis_text", _tok in analysis_text, True)
    ck("files.analysis_exists", os.path.exists(os.path.join(HERE, "analysis.md")), True)

    n_fail = sum(1 for c in checks if not c["pass"])
    report = {"n_checks": len(checks), "n_fail": n_fail, "all_pass": n_fail == 0,
              "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for c in checks:
        if not c["pass"]:
            print(f"FAIL {c['check']}: want {c['want']} got {c['got']}")
    print(f"reconcile: {len(checks)} checks, {n_fail} fail, all_pass={n_fail == 0}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
