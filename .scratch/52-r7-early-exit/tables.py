"""R7 结果 → Markdown 表（供 analysis.md 引用；不写事实、只排版，避免手抄错数）。

用法：python3 .scratch/52-r7-early-exit/tables.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "..", "..", "axes", "R-references", "R7-existing-early-exit",
                   "experiments", "2623-standard-early-exit-13m")
RES = os.path.join(EXP, "results.json")
TRAIN = os.path.join(HERE, "evidence")


def train_row(name):
    p = os.path.join(TRAIN, name, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    res = json.load(open(RES))
    b = res["battery"]
    print("## 训练臂（13M，12k 步 = 98.3M tokens）\n")
    print("| 臂 | 门 | 配方 | 参数 | 训练墙钟 min | 训练峰值 MB | d8 bpc | tf@4 | tf@6 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, a in b.items():
        tr = train_row(name) or {}
        c = a["config"]
        tf = {r["depth"]: r["accept"] for r in a.get("tf_acceptance", [])}
        dc = {r["depth"]: r["bpc"] for r in a.get("depth_curve", [])}
        recipe = c["aux_mode"]
        mod = a.get("mod_eval")
        print(f"| {name} | {'无门' if not c['carry_gate'] else ('—' if c['dense'] else 'carry')} "
              f"| {recipe if not c['dense'] else 'CE only'}"
              f"{' + MoD r=%s' % c['mod_ratio'] if c['mod_ratio'] else ''} "
              f"| {a['params'] / 1e6:.3f}M | {tr.get('wall_minutes')} "
              f"| {tr.get('peak_gpu_mem_mb')} "
              f"| {mod['bpc'] if mod else dc.get(len(dc) - 1 if dc else 8)} "
              f"| {tf.get(4)} | {tf.get(6)} |")
    print("\n## 等价算力与 CALM / spec（真解码路径）\n")
    for name, a in b.items():
        if "mod_eval" in a:
            m = a["mod_eval"]
            print(f"- **{name}**（MoD）：bpc {m['bpc']}，每 token 块作用次数 "
                  f"{m['layer_apps_per_token_measured']}（解析 {m['layer_apps_per_token_analytic']}），"
                  f"放行比例均值 {m['routed_frac_mean']}，路由器参数 {m['router_params']}")
            continue
        cal = a["calm_calibration"]["by_eps"]
        print(f"\n### {name}")
        for eps, c in cal.items():
            print(f"- CALM ε={eps}: θ={c['theta']} → 样本外平均退出深度 "
                  f"{c['mean_exit_depth_holdout']}（跑满比例 {c['frac_exit_at_full_holdout']}），"
                  f"一致率 {c['agree_with_full_holdout']}")
        rows = a["decode"]
        print("- 真解码（4 prompt × %d token）：全深度算力 %.3f/token；固定退 4 层 %.3f/token "
              "（与全深度流一致率 %s）；CALM(ε=0.05) 平均深度 %s、一致率 %s"
              % (rows["n_new"],
                 sum(r["layerpos_per_token"] for r in rows["full"]) / 4,
                 sum(r["layerpos_per_token"] for r in rows["exit4"]) / 4,
                 sum(r["match_full_frac"] for r in rows["exit4"]) / 4,
                 sum(r["mean_exit_depth"] for r in rows["calm_eps0.05"]) / 4,
                 sum(r["match_full_frac"] for r in rows["calm_eps0.05"]) / 4))
        for g in ("1", "2", "4", "8"):
            p = rows.get(f"spec_g{g}_pooled")
            if p:
                lps = [r["layerpos_per_token"] for r in rows["spec"][g]]
                print(f"- LayerSkip 自草稿 γ={g}（草稿深度 4）：接受率 "
                      f"{p['accept_rate']} ± {p['binom_se']}（n={p['n_drafts']}），"
                      f"算力 {sum(lps) / len(lps):.3f}/token，"
                      f"发出流与全深度流一致率 "
                      f"{sum(r['match_full_frac'] for r in rows['spec'][g]) / len(lps)}")
    print("\n## 墙钟（交错轮次，比值 vs 同轮 stock/full）\n")
    print("| 臂\|策略 | 中位 tok/s | 极差 | 比值中位 | 比值极差 | 算力/token |")
    print("|---|---|---|---|---|---|")
    for key, s in sorted(res["wall_clock_interleaved"]["summary"].items()):
        print(f"| {key} | {s['tok_s_median']} | {s['tok_s_min']}–{s['tok_s_max']} "
              f"({s['tok_s_range_frac']}) | {s.get('ratio_median')} "
              f"| {s.get('ratio_min')}–{s.get('ratio_max')} "
              f"| {s['layerpos_per_token_median']} |")
    mf = res.get("mod_forward_throughput", {}).get("summary", {})
    print("\n## MoD 前向吞吐（batch 8 × seq 256，交错）\n")
    for k, v in mf.items():
        print(f"- {k}: 中位 {v['tok_per_s_median']} tok/s（{v['tok_per_s_min']}–"
              f"{v['tok_per_s_max']}），峰值 {v['peak_mb']} MB")


if __name__ == "__main__":
    main()
