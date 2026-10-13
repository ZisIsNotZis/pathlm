"""WP 6.2.2 · **离线重算判据**（不碰 GPU、不重跑读数）。

背景（判据代码缺陷，非读数缺陷）：run #1 落盘的 `results.json` 里
`checkpoints[*].criteria.P1/P2.tree_*.paired*` 全为 `null` —— 因为算配对差时按
`f"{a}_minus_{b}"` 直接取键，而 `paired_diffs` 只按臂**写入顺序**存无向键；树臂在
单臂之后写入 ⇒ 树臂 vs 最强单臂的真实键是相反方向（`<best>_minus_<tree>`），
直接取键必然 miss，于是 P1/P2 全判 false（`delta_vs_best_single` 明明为正）。

本脚本**只读** `results.json`（其中每个臂的 `per_prompt[*].accepts` 是逐 prompt
原始接受数）与 `../2632-alpha1-attribution/probes_output.json`（C1 探针），
按与 `run.py` 完全相同的公式离线重算：

- 臂级 α₁ / α₂|₁ / τ / tok-fwd：按 **prompt 聚类** bootstrap（B=2000，seed=2640，
  percentile 95%）；
- **配对**差：同一 prompt 重采样索引下 `stat(A) − stat(B)`（也对 seed=2640）；
- 最强单臂 = 同批 prompt 实测 α₁ 最大的单候选臂；
- P1/P2/P3/P4 + C1（口径见 `README.md` §4）。

结果写 `criteria_fixed.json`。脚本同时核对重算臂级 α₁/τ 与 `results.json` 原值
是否逐位相等（应相等 ⇒ 证明是判据取键缺陷，读数本身未变）。

用法：python3 recompute_criteria.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spec_tree as T  # noqa: E402

RESULTS = os.path.join(HERE, "results.json")
PROBE_JSON = os.path.abspath(os.path.join(HERE, "..", "2632-alpha1-attribution",
                                          "probes_output.json"))
OUT = os.path.join(HERE, "criteria_fixed.json")
BOOT = 2000
BOOT_SEED = 2640
DELTA = T.DELTA_SORTABLE


# ---------------------------------------------------------------- 统计（照抄 run.py 口径）

def _ci(vals: list) -> list:
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 6),
            round(float(np.percentile(vals, 97.5)), 6)]


def stat_alpha1(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(1 for g in groups for a in g if a >= 1) / max(tot, 1)


def stat_alpha2(groups: list) -> float:
    den = sum(1 for g in groups for a in g if a >= 1)
    return (sum(1 for g in groups for a in g if a >= 2) / den) if den else 0.0


def stat_tau(groups: list) -> float:
    tot = sum(len(g) for g in groups)
    return sum(a for g in groups for a in g) / max(tot, 1)


def stat_tpf(groups: list) -> float:
    nr = sum(len(g) for g in groups)
    return (nr + sum(a for g in groups for a in g)) / max(nr, 1)


def cluster_ci(groups: list, stat) -> list:
    if not groups:
        return [None, None]
    rng = np.random.default_rng(BOOT_SEED)
    n = len(groups)
    return _ci([stat([groups[i] for i in rng.integers(0, n, n)]) for _ in range(BOOT)])


def paired_diff(groups_a: list, groups_b: list, stat) -> dict:
    if not groups_a or len(groups_a) != len(groups_b):
        return {"point": None, "ci": [None, None]}
    rng = np.random.default_rng(BOOT_SEED)
    n = len(groups_a)
    vals = [stat([groups_a[i] for i in idx]) - stat([groups_b[i] for i in idx])
            for idx in (rng.integers(0, n, n) for _ in range(BOOT))]
    return {"point": round(stat(groups_a) - stat(groups_b), 6), "ci": _ci(vals),
            "n_prompts": n, "bootstrap_B": BOOT, "seed": BOOT_SEED}


def wilson_ci(p: float, n: int, z: float = 1.96) -> list:
    if not n:
        return [None, None]
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(centre - half, 4), round(centre + half, 4)]


# ---------------------------------------------------------------- C1 探针

def probe_recall(ckpt_name: str) -> dict | None:
    try:
        with open(PROBE_JSON, encoding="utf-8") as f:
            blob = json.load(f)[ckpt_name]
        return {"recall": float(blob["P3_free_gen_topk_vs_T2"]["2"]["direct"]),
                "n": int(blob["P3_free_gen_overall"]["n"]),
                "source": "2632-alpha1-attribution/probes_output.json::"
                          "P3_free_gen_topk_vs_T2['2']['direct']"}
    except Exception:
        return None


# ---------------------------------------------------------------- 主重算

def recompute_ckpt(name: str, rec: dict) -> dict:
    arm_sum = rec["arms"]
    groups = {a: [pp["accepts"] for pp in v["per_prompt"]]
              for a, v in arm_sum.items() if not v.get("is_control")}
    arms = {}
    for a, g in groups.items():
        tau = stat_tau(g)
        mean_k = arm_sum[a].get("mean_candidates_per_round") or 1.0
        arms[a] = {
            "alpha1": round(stat_alpha1(g), 6), "alpha1_ci": cluster_ci(g, stat_alpha1),
            "alpha2_given_alpha1": round(stat_alpha2(g), 6),
            "alpha2_given_alpha1_ci": cluster_ci(g, stat_alpha2),
            "tau": round(tau, 6), "tau_ci": cluster_ci(g, stat_tau),
            "tokens_per_forward": round(stat_tpf(g), 6),
            "tokens_per_forward_ci": cluster_ci(g, stat_tpf),
            "mean_candidates_per_round": mean_k,
            "n_rounds": sum(len(x) for x in g),
            "stream_identical_to_native": arm_sum[a].get("stream_identical_to_native"),
            "R_equivalent_flops": round(mean_k * (1 + T.GAMMA) / (1 + tau), 4),
        }

    # 逐位核对：重算的臂级读数 vs run #1 落盘值（应完全一致）
    drift = {}
    for a in groups:
        d = {"alpha1": round(arms[a]["alpha1"] - arm_sum[a]["alpha1"], 9),
             "tau": round(arms[a]["tau"] - arm_sum[a]["tau"], 9)}
        if d["alpha1"] or d["tau"]:
            drift[a] = d

    singles = [a for a in T.SINGLE_ARMS if a in arms]
    best = max(singles, key=lambda a: arms[a]["alpha1"]) if singles else None

    P1, P2 = {}, {}
    if best is not None:
        for a in T.TREE_ARMS:
            if a not in arms:
                continue
            pa1 = paired_diff(groups[a], groups[best], stat_alpha1)
            ptau = paired_diff(groups[a], groups[best], stat_tau)
            ptpf = paired_diff(groups[a], groups[best], stat_tpf)
            d1 = round(arms[a]["alpha1"] - arms[best]["alpha1"], 6)
            dtau = round(arms[a]["tau"] - arms[best]["tau"], 6)
            P1[a] = {"delta_vs_best_single": d1, "paired_alpha1": pa1,
                     "pass": bool(d1 > DELTA and (pa1["ci"][0] is not None
                                                  and pa1["ci"][0] > 0))}
            P2[a] = {"delta_tau_vs_best_single": dtau,
                     "delta_tpf_vs_best_single": round(
                         arms[a]["tokens_per_forward"] - arms[best]["tokens_per_forward"], 6),
                     "paired_tau": ptau, "paired_tpf": ptpf,
                     "pass": bool(dtau > DELTA and (ptau["ci"][0] is not None
                                                    and ptau["ci"][0] > 0))}

    out = {"strongest_single_arm": best, "arms": arms, "drift_vs_results_json": drift,
           "P1_alpha1_lift": P1, "P2_tau_lift": P2}

    pw, cf = arm_sum.get("combo_pw", {}), arm_sum.get("combined_fixed", {})
    if "combo_pw" in arms and "combined_fixed" in arms:
        p3 = paired_diff(groups["combo_pw"], groups["combined_fixed"], stat_alpha1)
        d3 = round(arms["combo_pw"]["alpha1"] - arms["combined_fixed"]["alpha1"], 6)
        out["P3_pw_vs_fixed"] = {
            "delta_alpha1": d3, "paired_alpha1": p3,
            "divergence_rate_pw_arm": pw.get("activity", {}).get("pw_divergence_rate_vs_fixed"),
            "divergence_rate_fixed_arm": cf.get("activity", {}).get("pw_divergence_rate_vs_fixed"),
            "pass": bool(d3 > DELTA and p3["ci"][0] is not None and p3["ci"][0] > 0),
            "note": "分歧率 ~0 ⇒ 两臂不可区分（无信息）；>0 ⇒ 两臂真的不同，"
                    "差别只是没超过接受率噪声带"}

    out["P4_lossless"] = {
        "all_arms_stream_identical": all(v.get("stream_identical_to_native", False)
                                         for v in arm_sum.values()),
        "mismatching_arms": [a for a, v in arm_sum.items()
                             if not v.get("stream_identical_to_native", False)]}

    probe = probe_recall(name)
    if probe is not None and "tree_direct_k2" in arms:
        a1 = arms["tree_direct_k2"]["alpha1"]
        lo, hi = arms["tree_direct_k2"]["alpha1_ci"]
        p_ci = wilson_ci(probe["recall"], probe.get("n") or 0)
        out["C1_cross_check_tree_direct_k2_vs_probe_recall2"] = {
            "measured_alpha1": a1, "measured_alpha1_ci": [lo, hi],
            "probe_recall_at_2": probe["recall"], "probe_n_rounds": probe.get("n"),
            "probe_wilson_ci95": p_ci,
            "probe_inside_ci": (lo is not None and lo <= probe["recall"] <= hi),
            "cis_overlap": (lo is not None and p_ci[1] is not None
                            and not (hi < p_ci[0] or lo > p_ci[1])),
            "measured_below_probe": (hi is not None and hi < probe["recall"]),
            "note": "measured_below_probe=True ⇒ 树实现可疑；探针 4 prompt vs 本臂 12 prompt，"
                    "只要求同量级"}

    # 旧（run #1 落盘）判据里的 P1/P2，用于「旧 vs 重算」对照
    old_crit = rec["criteria"]
    out["old_from_results_json"] = {
        "strongest_single_arm": old_crit.get("strongest_single_arm"),
        "P1_paired_all_null": all(
            (v.get("paired") or {}).get("ci", [None, None])[0] is None
            for v in old_crit.get("P1_alpha1_lift", {}).values()),
        "P1_pass": {a: v["pass"] for a, v in old_crit.get("P1_alpha1_lift", {}).items()},
        "P2_pass": {a: v["pass"] for a, v in old_crit.get("P2_tau_lift", {}).items()},
        "P1_delta_vs_best_single": {a: v.get("delta_vs_best_single")
                                    for a, v in old_crit.get("P1_alpha1_lift", {}).items()}}
    return out


def main() -> int:
    with open(RESULTS, encoding="utf-8") as f:
        blob = json.load(f)
    out = {
        "source": "results.json（run #1，2026-10-13 16:54 落盘）",
        "reason": "run #1 的 P1/P2 配对差取键有缺陷（树臂 vs 最强单臂的键方向相反 ⇒ null）；"
                  "本文件离线按同一 bootstrap 口径重算，读数本身未变",
        "method": {"bootstrap": {"B": BOOT, "seed": BOOT_SEED, "cluster": "prompt",
                                 "percentile": 95},
                   "paired": "同一 prompt 重采样索引差",
                   "delta_sortable": DELTA, "gamma": T.GAMMA},
        "checkpoints": {}}
    for name, rec in blob["checkpoints"].items():
        out["checkpoints"][name] = recompute_ckpt(name, rec)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 控制台汇总
    for name, r in out["checkpoints"].items():
        print(f"=== {name} best_single={r['strongest_single_arm']} "
              f"drift={r['drift_vs_results_json']}")
        for a in T.TREE_ARMS:
            if a in r["P1_alpha1_lift"]:
                p1, p2 = r["P1_alpha1_lift"][a], r["P2_tau_lift"][a]
                print(f"  {a:15s} dA1={p1['delta_vs_best_single']:+.4f} "
                      f"CI={p1['paired_alpha1']['ci']} P1={p1['pass']} | "
                      f"dtau={p2['delta_tau_vs_best_single']:+.4f} "
                      f"CI={p2['paired_tau']['ci']} P2={p2['pass']}")
        if "P3_pw_vs_fixed" in r:
            p3 = r["P3_pw_vs_fixed"]
            print(f"  P3 dA1={p3['delta_alpha1']:+.4f} CI={p3['paired_alpha1']['ci']} "
                  f"pass={p3['pass']} div={p3['divergence_rate_pw_arm']}")
        print(f"  P4={r['P4_lossless']}")
        if "C1_cross_check_tree_direct_k2_vs_probe_recall2" in r:
            c1 = r["C1_cross_check_tree_direct_k2_vs_probe_recall2"]
            print(f"  C1 measured={c1['measured_alpha1']} CI={c1['measured_alpha1_ci']} "
                  f"probe={c1['probe_recall_at_2']} inside={c1['probe_inside_ci']} "
                  f"overlap={c1['cis_overlap']} below={c1['measured_below_probe']}")
    print(f"[recompute] wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
