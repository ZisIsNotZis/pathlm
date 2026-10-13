"""axis-lead 独立复算（不经 worker 的判据代码）：从 run #1 快照的 per-prompt 接受数
离线重算 α₁ / α₂|₁ / τ + 按 prompt 聚类的 bootstrap CI + 树臂 vs 最强单臂的配对差。

这是 axis-lead 的**交叉检查**，不是实验产物本体；产物在
`axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/`。
口径：per-prompt 聚类（resample prompt，不把同一序列的多轮当独立样本）；
配对 = 同一重采样索引用于两臂；B=2000，seed=2640，percentile 95%。
用法：python3 recompute_check.py <results.json 路径> > recompute_check.json
"""

from __future__ import annotations

import json
import statistics
import sys

import numpy as np

B = 2000
SEED = 2640
TREE_ARMS = ("tree_union2", "tree_direct_k2", "tree_qpw_k2", "tree_topk4")
SINGLE_ARMS = ("direct1", "chain1", "combined_fixed", "combo_pw")


def arm_stats(accepts_by_prompt: list[list[int]]) -> dict:
    n = sum(len(a) for a in accepts_by_prompt)
    if n == 0:
        return {"alpha1": None, "tau": None, "alpha2_given_alpha1": None, "n_rounds": 0}
    ge1 = sum(1 for a in accepts_by_prompt for x in a if x >= 1)
    ge2 = sum(1 for a in accepts_by_prompt for x in a if x >= 2)
    tot = sum(sum(a) for a in accepts_by_prompt)
    return {"alpha1": ge1 / n, "tau": tot / n,
            "alpha2_given_alpha1": (ge2 / ge1) if ge1 else None, "n_rounds": n}


def boot_metric(per_prompt: list[list[int]], kind: str, idx: np.ndarray) -> float:
    sel = [per_prompt[i] for i in idx]
    st = arm_stats(sel)
    return st[kind]


def percentile_ci(xs: list[float]) -> list[float]:
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]


def main() -> int:
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rng = np.random.default_rng(SEED)
    out: dict = {"source": path, "note": "axis-lead 独立复算（per-prompt 接受数 → 指标 + 配对 bootstrap）",
                 "checkpoints": {}}
    for ck, c in data["checkpoints"].items():
        arms = c["arms"]
        per = {a: [r["accepts"] for r in arms[a]["per_prompt"]] for a in arms}
        n_prompts = len(next(iter(per.values())))
        rec: dict = {"n_prompts": n_prompts, "arm_reported_vs_recomputed": {}, "boot": {},
                     "strongest_single": None, "paired_vs_strongest": {}}
        for a, pp in per.items():
            if a == "native":
                continue
            rep = arms[a]
            rec["arm_reported_vs_recomputed"][a] = {
                "alpha1_reported": rep.get("alpha1"),
                "alpha1_recomputed": arm_stats(pp)["alpha1"],
                "tau_reported": rep.get("tau"),
                "tau_recomputed": arm_stats(pp)["tau"],
                "n_rounds_reported": rep.get("n_rounds"),
                "n_rounds_recomputed": arm_stats(pp)["n_rounds"],
                "identical_to_native": rep.get("stream_identical_to_native"),
                "mean_candidates_per_round": rep.get("mean_candidates_per_round"),
                "R_equivalent_flops": rep.get("R_equivalent_flops"),
                "R_B_verified_mean": rep.get("R_B_verified_mean"),
            }
        # bootstrap CI per arm + paired diffs
        boots = {a: {"alpha1": [], "tau": []} for a in per}
        for _ in range(B):
            idx = rng.integers(0, n_prompts, n_prompts)
            for a in per:
                if a == "native":
                    continue
                st = arm_stats([per[a][i] for i in idx])
                if st["alpha1"] is not None:
                    boots[a]["alpha1"].append(st["alpha1"])
                    boots[a]["tau"].append(st["tau"])
        for a in per:
            if a == "native":
                continue
            rec["boot"][a] = {"alpha1_ci": percentile_ci(boots[a]["alpha1"]),
                              "tau_ci": percentile_ci(boots[a]["tau"])}
        singles = [a for a in SINGLE_ARMS if a in per and arms[a].get("alpha1") is not None]
        best = max(singles, key=lambda a: arm_stats(per[a])["alpha1"])
        rec["strongest_single"] = {"arm": best, "alpha1": arm_stats(per[best])["alpha1"],
                                  "tau": arm_stats(per[best])["tau"]}
        for t in TREE_ARMS:
            if t not in per:
                continue
            da = [boots[t]["alpha1"][i] - boots[best]["alpha1"][i] for i in range(B)]
            dt = [boots[t]["tau"][i] - boots[best]["tau"][i] for i in range(B)]
            ci_a, ci_t = percentile_ci(da), percentile_ci(dt)
            rec["paired_vs_strongest"][t] = {
                "d_alpha1_point": arm_stats(per[t])["alpha1"] - arm_stats(per[best])["alpha1"],
                "d_alpha1_ci": ci_a, "pass_gt_0.014_and_ci_excl_0": bool(ci_a[0] > 0 and (arm_stats(per[t])["alpha1"] - arm_stats(per[best])["alpha1"]) > 0.014),
                "d_tau_point": arm_stats(per[t])["tau"] - arm_stats(per[best])["tau"],
                "d_tau_ci": ci_t, "pass_tau_ci_excl_0": bool(ci_t[0] > 0),
            }
        # P3: combo_pw vs combined_fixed
        if "combo_pw" in per and "combined_fixed" in per:
            da = [boots["combo_pw"]["alpha1"][i] - boots["combined_fixed"]["alpha1"][i] for i in range(B)]
            rec["P3_combo_pw_minus_combined_fixed"] = {
                "d_alpha1_point": arm_stats(per["combo_pw"])["alpha1"] - arm_stats(per["combined_fixed"])["alpha1"],
                "d_alpha1_ci": percentile_ci(da),
            }
        out["checkpoints"][ck] = rec
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
