"""Phase B · axis6-lead 的第三方复算（不复用 writer 的判据代码）。

输入（只读）：`2640-tree-conf-13m/results.json`（Phase A 的落盘读数）。
输出：`recompute_phaseB.json` —— 为 Phase B 的 WP 6.2.3（验证开销 / τ 饱和）提供
**独立于 worker 代码**的代价台账与饱和比；同时核对臂级 α₁/τ 是否与落盘逐位一致。

跑法：python3 recompute_phaseB.py
"""
from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "axes", "6-predict", "6.2-speculative-decoding",
                   "experiments", "2640-tree-conf-13m", "results.json")
GAMMA = 2
# 预登记（2650 README §4 V5 / V6）的 K 分层：单臂 K=1 → K=2 → K=4
TREE_BY_LEVEL = {1: None, 2: ["tree_direct_k2", "tree_qpw_k2", "tree_union2"], 4: ["tree_topk4"]}


def arm_stats(arm: dict) -> dict:
    """从 per_prompt[*].accepts 独立重算 α₁ / τ / K。"""
    acc = [p["accepts"] for p in arm["per_prompt"]]
    n_rounds = sum(len(a) for a in acc)
    n_ge1 = sum(1 for a in acc for x in a if x >= 1)
    n_tok = sum(len(a) + sum(a) for a in acc)   # 每轮 1 个 v1 + 接受的草稿数
    ks = [arm["mean_candidates_per_round"]] * len(acc)
    return {
        "n_rounds": n_rounds,
        "n_emitted": n_tok,
        "alpha1": n_ge1 / n_rounds,
        "tau": (n_tok - n_rounds) / n_rounds,
        "tokens_per_forward": n_tok / n_rounds,
        "K_mean": (sum(ks) / len(ks)) if ks else None,
    }


CYCLE = os.path.join(os.path.dirname(SRC), "cycle_log.jsonl")


def rank_decomposition() -> dict:
    """逐轮 `accepted_candidate_index` 分解：τ 增益里有多少来自「验证的第 2..K 个候选」。

    口径：`index == 0` ⇒ 只验第一个候选就够（K 行里的第 1 行）；`index >= 1` ⇒ 该轮的接受
    **只能**靠更深的候选（即「多候选验证」净贡献）；`None` ⇒ 该轮不接受草稿。
    这是 Phase A 落盘数据即可得到的结论，**不需要新 GPU**。
    """
    per: dict = {}
    with open(CYCLE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "accepted_candidate_index" not in r:
                continue
            key = (r["ckpt"], r["arm"])
            d = per.setdefault(key, {"n_rounds": 0, "rank0": 0, "rank_ge1": 0, "reject_all": 0,
                                    "k_sum": 0, "rank_hist": {}})
            d["n_rounds"] += 1
            d["k_sum"] += r["n_candidates"]
            i = r["accepted_candidate_index"]
            if i is None:
                d["reject_all"] += 1
            elif i == 0:
                d["rank0"] += 1
            else:
                d["rank_ge1"] += 1
                d["rank_hist"][str(i)] = d["rank_hist"].get(str(i), 0) + 1
    out = {}
    for (ck, arm), d in sorted(per.items()):
        n = d["n_rounds"]
        out[f"{ck}|{arm}"] = {
            "n_rounds": n,
            "K_mean": d["k_sum"] / n,
            "alpha1": (d["rank0"] + d["rank_ge1"]) / n,
            "p_rank0": d["rank0"] / n,
            "p_rank_ge1_only": d["rank_ge1"] / n,
            "p_reject_all": d["reject_all"] / n,
            "rank_hist_ge1": d["rank_hist"],
        }
    return out


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        res = json.load(f)
    out: dict = {
        "source": os.path.relpath(SRC, HERE),
        "gamma": GAMMA,
        "note": ("axis6-lead 第三方复算：只用 results.json::per_prompt[*].accepts 独立重算；"
                 "R = 3K·(1+γ)/(1+τ) 为 K 行平铺实现口径（每轮 3K 个主干位置），"
                 "节点共享理想树为 2K+1；R>1 是范式定义性质、不作判据。"),
        "checkpoints": {},
    }
    for ck, block in res["checkpoints"].items():
        arms = block["arms"]
        rows = {}
        for name, arm in arms.items():
            if arm.get("is_control"):
                rows[name] = {"is_control": True}
                continue
            st = arm_stats(arm)
            stored_a1, stored_tau = arm.get("alpha1"), arm.get("tau")
            st["drift_alpha1"] = (None if stored_a1 is None else round(st["alpha1"] - stored_a1, 12))
            st["drift_tau"] = (None if stored_tau is None else round(st["tau"] - stored_tau, 12))
            km = arm.get("mean_candidates_per_round") or 1.0
            st["R_impl_3K"] = 3 * km * (1 + GAMMA) / st["tokens_per_forward"]
            st["R_nodeshared_2K1"] = (2 * km + 1) * (1 + GAMMA) / st["tokens_per_forward"]
            st["tau_per_K"] = st["tau"] / km
            rows[name] = st
        # τ 上界与饱和
        oracle = rows.get("oracle_accept", {})
        tau_oracle = oracle.get("tau")
        singles = [n for n in ("direct1", "chain1", "combined_fixed", "combo_pw") if n in rows]
        best_single = max(singles, key=lambda n: rows[n]["alpha1"])
        ledger = {}
        if tau_oracle:
            for name, st in rows.items():
                if st.get("is_control") or name == "oracle_accept":
                    continue
                st["tau_over_oracle"] = st["tau"] / tau_oracle
            b = rows[best_single]
            b["tau_over_oracle"] = b["tau"] / tau_oracle
            # 边际 Δτ/ΔK：K=1(best single) → K=2(树,取实测最高 τ) → K=4
            lvl = {}
            k1 = {"name": best_single, "K": 1.0, "tau": b["tau"], "alpha1": b["alpha1"]}
            lvl[1] = k1
            for k in (2, 4):
                cands = [n for n in TREE_BY_LEVEL[k] or [] if n in rows]
                if not cands:
                    continue
                best = max(cands, key=lambda n: rows[n]["tau"])
                lvl[k] = {"name": best, "K": rows[best]["K_mean"],
                          "tau": rows[best]["tau"], "alpha1": rows[best]["alpha1"]}
            marg = {}
            keys = sorted(lvl)
            for a, b2 in zip(keys, keys[1:]):
                dk = lvl[b2]["K"] - lvl[a]["K"]
                marg[f"K{lvl[a]['K']:.2f}->K{lvl[b2]['K']:.2f}"] = {
                    "d_tau": lvl[b2]["tau"] - lvl[a]["tau"],
                    "d_K": dk,
                    "d_tau_per_d_K": (lvl[b2]["tau"] - lvl[a]["tau"]) / dk,
                }
            ledger = {"levels": lvl, "marginal": marg, "tau_oracle": tau_oracle,
                      "alpha1_oracle": oracle.get("alpha1")}
        out["checkpoints"][ck] = {
            "best_single_arm": best_single,
            "arms": rows,
            "tau_ceiling_ledger": ledger,
        }
    out["rank_decomposition"] = rank_decomposition()
    dest = os.path.join(HERE, "recompute_phaseB.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    # 控制台摘要
    for ck, blk in out["checkpoints"].items():
        led = blk["tau_ceiling_ledger"]
        print(f"[{ck}] best_single={blk['best_single_arm']} "
              f"tau_oracle={led.get('tau_oracle')}")
        for k, v in led.get("levels", {}).items():
            st = blk["arms"][v["name"]]
            print(f"   K={v['K']:.2f} {v['name']:<16} alpha1={v['alpha1']:.4f} "
                  f"tau={v['tau']:.4f} tau/oracle={st.get('tau_over_oracle', float('nan')):.3f} "
                  f"R3K={st['R_impl_3K']:.3f}")
        for k, v in led.get("marginal", {}).items():
            print(f"   marginal {k}: d_tau={v['d_tau']:+.4f} d_tau/d_K={v['d_tau_per_d_K']:+.4f}")
        bad = {n: (s["drift_alpha1"], s["drift_tau"]) for n, s in blk["arms"].items()
               if isinstance(s, dict) and (s.get("drift_alpha1") or s.get("drift_tau"))}
        print(f"   drift vs results.json: {bad if bad else 'none (bitwise)'}")
    print(f"wrote {os.path.relpath(dest, os.getcwd())}")


if __name__ == "__main__":
    main()
