#!/usr/bin/env python3
"""2641 · 把 `results.json` 打成 markdown 表（防手抄数字漂移）。

用法：python3 report_table.py            # 主表 + 判据块
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "results.json")))

print("### 臂表（全部数字来自 results_<arm>_s<seed>.json 的汇总）\n")
print("| 臂 | Σr | clean bpc | 每 token MACs（总） | 每 token MACs（FFN） | FFN 参数 | 总参数 | 末25% loss 斜率/1k步 |")
print("|---|---|---|---|---|---|---|---|")
for k, v in d["arms"].items():
    print(f"| {k} | {v['sum_r']} | {v['clean_bpc']} | {v['macs_per_token']} | "
          f"{v['macs_ffn']} | {v['params_ffn']} | {v['params_total']} | {v['slope_per_1k']} |")

print("\n### 判据\n")
print(f"- 方差带 = {d['variance_bound']}；同 config 跨 seed 散布（2613 参照）= "
      f"{d['seed_spread_reference']['same_config_seed_spread_2613']}")
for s, v in d["P1_P2_deltas_by_seed"].items():
    print(f"- seed {s}：Δ_up = bpc_U2 − bpc_PYR = {v['delta_up_U2_minus_PYR']}；"
          f"Δ_down = bpc_U2 − bpc_APYR = {v['delta_down_U2_minus_APYR']}")
print(f"- P1 判词：{d['P1_verdict_by_seed']}")
print(f"- P4 跨 seed 同号 = {d['P4_same_sign_across_seeds']}")
print(f"- P5 等值性门 = {d['P5_engine_equivalence_gate']}")
print(f"- 等 FFN 预算检查 = {d['equal_ffn_budget_check']}")
print(f"- P6 活性 = {d['P6_liveness']}")
