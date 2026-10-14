"""Phase B · axis6-lead 的第三方复算 #2：独立核 WP 6.2.3（2650）的 V0/V2/V3 与代价台账。

输入（只读）：`2650-verify-gate-13m/results.json` + `2640-tree-conf-13m/results.json`。
输出：`recompute_phaseB_2650.json`。**不复用** worker 的 `verify_gate.py` / `recompute_v5.py`。

跑法：python3 recompute_phaseB_2650.py
"""
from __future__ import annotations

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.join(HERE, "..", "..", "axes", "6-predict", "6.2-speculative-decoding", "experiments")
P2650 = os.path.join(E6, "2650-verify-gate-13m", "results.json")
P2640 = os.path.join(E6, "2640-tree-conf-13m", "results.json")
GAMMA = 2
VOCAB = 206


def accs(block: dict, arm: str) -> list[list[int]]:
    return [p["accepts"] for p in block["arms"][arm]["per_prompt"]]


def stats(acc: list[list[int]]) -> dict:
    n = sum(len(a) for a in acc)
    ge1 = sum(1 for a in acc for x in a if x >= 1)
    tok = sum(len(a) + sum(a) for a in acc)
    return {"n_rounds": n, "alpha1": ge1 / n, "tau": (tok - n) / n,
            "tokens_per_forward": tok / n}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main() -> None:
    r2650 = json.load(open(P2650, encoding="utf-8"))
    r2640 = json.load(open(P2640, encoding="utf-8"))
    out: dict = {"sources": [os.path.relpath(P2650, os.getcwd()), os.path.relpath(P2640, os.getcwd())]}

    # ---- V0 / V3：逐位比对（2640 为基准）
    v0 = {}
    for ck in ("13m_s0", "13m_s1"):
        b2650, b2640 = r2650["checkpoints"][ck], r2640["checkpoints"][ck]
        for mine, ref in (("repro_topk4", "tree_topk4"), ("repro_combo_pw", "combo_pw"),
                          ("single_k1_ctl", "combo_pw"), ("dup_k4", "combo_pw")):
            if mine in b2650["arms"] and ref in b2640["arms"]:
                a, b = accs(b2650, mine), accs(b2640, ref)
                v0[f"{ck}|{mine}=={ref}"] = {"bitwise_equal": a == b,
                                             "alpha1_mine": stats(a)["alpha1"],
                                             "alpha1_ref": stats(b)["alpha1"]}
            else:
                v0[f"{ck}|{mine}=={ref}"] = {"bitwise_equal": None, "note": "arm 未跑"}
    out["V0_V3_bitwise"] = v0

    # ---- V2：对抗臂 + Wilson CI + 均匀随机假接受率
    rng_p = 4 / VOCAB
    v2 = {}
    for ck in ("13m_s0", "13m_s1"):
        b = r2650["checkpoints"][ck]
        for arm in ("adv_random_k4", "adv_bottomk4"):
            if arm in b["arms"]:
                a = accs(b, arm)
                st = stats(a)
                k = round(st["alpha1"] * st["n_rounds"])
                lo, hi = wilson(k, st["n_rounds"])
                v2[f"{ck}|{arm}"] = {**st, "wilson95": [lo, hi],
                                     "uniform_false_accept_rate_4_206": rng_p,
                                     "stream_identical_to_native": b["arms"][arm]["stream_identical_to_native"],
                                     "n_forward_eq_rounds": (b["arms"][arm]["n_forward_total"]
                                                             == b["arms"][arm]["n_rounds"])}
    out["V2_adversarial"] = v2

    # ---- 代价台账（K 同族 = pw_mix.q）：K=1/4/8 取 2650，K=2 取 2640 tree_qpw_k2（指针）
    ck = "13m_s0"
    b2650, b2640 = r2650["checkpoints"][ck], r2640["checkpoints"][ck]
    rows = []
    spec = [(1, b2650, "single_k1_ctl"), (2, b2640, "tree_qpw_k2"),
            (4, b2650, "repro_topk4"), (4, b2640, "tree_topk4"),
            (8, b2650, "aggr_topk8")]
    for K, blk, arm in spec:
        if arm not in blk["arms"]:
            continue
        st = stats(accs(blk, arm))
        rows.append({"K": K, "arm": arm, "source": "2650" if blk is b2650 else "2640",
                     "tau": st["tau"], "alpha1": st["alpha1"],
                     "R_3K_over_1ptau": 3 * K / st["tokens_per_forward"],
                     "R_2640conv_K_times_3_over_1ptau": K * (1 + GAMMA) / st["tokens_per_forward"],
                     "seq_fwd_per_token": 1 / st["tokens_per_forward"]})
    tau_oracle = stats(accs(b2640, "oracle_accept"))["tau"]
    for r in rows:
        r["tau_over_oracle"] = r["tau"] / tau_oracle
        r["tau_per_K"] = r["tau"] / r["K"]
    marg = []
    same_family = [r for r in rows if r["source"] == "2650" or r["arm"] == "tree_qpw_k2"]
    fam = sorted({(r["K"], r["arm"]): r for r in same_family}.values(), key=lambda x: x["K"])
    for a, b2 in zip(fam, fam[1:]):
        dk = b2["K"] - a["K"]
        marg.append({"from": f"K={a['K']}({a['arm']})", "to": f"K={b2['K']}({b2['arm']})",
                     "d_tau": b2["tau"] - a["tau"], "d_tau_per_d_K": (b2["tau"] - a["tau"]) / dk})
    thresh = 0.5 * marg[0]["d_tau_per_d_K"] if marg else None
    out["cost_ledger_13m_s0"] = {"tau_oracle": tau_oracle, "rows": rows, "marginal_same_family": marg,
                                 "threshold_half_of_first_step": thresh,
                                 "k4_to_k8_worth_it": (marg[-1]["d_tau_per_d_K"] >= thresh if (marg and thresh) else None)}

    dest = os.path.join(HERE, "recompute_phaseB_2650.json")
    json.dump(out, open(dest, "w", encoding="utf-8"), ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(out["V0_V3_bitwise"], ensure_ascii=False, indent=1))
    print(json.dumps(out["V2_adversarial"], ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out["cost_ledger_13m_s0"].items() if k != "rows"},
                     ensure_ascii=False, indent=1))
    for r in rows:
        print(f"  K={r['K']} {r['arm']:<16} ({r['source']}) tau={r['tau']:.4f} a1={r['alpha1']:.4f} "
              f"R={r['R_3K_over_1ptau']:.3f} tau/oracle={r['tau_over_oracle']:.3f}")
    print("wrote", os.path.relpath(dest, os.getcwd()))


if __name__ == "__main__":
    main()
