"""2672 汇聚器（接任段）：把 6 个快照 run（snap<TAG>-<arm>-s<seed>，3 臂 × seed{0,1}）
与既有 12k 档参照读数、等参数 rescore 表、config 对平清单汇成 `results.json`
（判据区**无墙钟字段**，L5 自检沿用 2660 的 `has_wall_clock`）。

用法：python3 aggregate.py    （TAG 由环境变量 TAG 覆盖，默认 2k）
"""

import json
import os
import re

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/z/vibe/pathlm"
TAG = os.environ.get("TAG", "2k")
SNAP_STEPS = int(os.environ.get("SNAP_STEPS", "2000"))
ARMS = ["fill", "nofill", "projfill"]
E12 = f"{ROOT}/.scratch/23-depth-ar/evidence"
REF12 = {
    "fill": f"{E12}/23c-H2-w0-12k",
    "nofill": f"{E12}/23e-nofill-12k",
    "projfill": f"{E12}/23e-projfill-12k",
}
WALL_RE = re.compile(r"wall|minutes|elapsed|clock|seconds|_s$", re.IGNORECASE)


def snap_dir(arm, seed):
    return f"{D}/snap{TAG}-{arm}-s{seed}"


def load(rd):
    p = os.path.join(rd, "results.json")
    if not rd or not os.path.exists(p):
        return None
    return json.load(open(p))


def from_curve(meta, depth, field):
    for r in meta["depth_curve"]:
        if r["depth"] == depth:
            return r[field]
    return None


def tf_at(meta, depth):
    for r in meta["tf_acceptance"]:
        if r["depth"] == depth:
            return r["accept"]
    return None


def point(meta):
    return {
        "steps": meta["train"]["steps"],
        "params": meta["params"],
        "train_tokens": meta["train_tokens"],
        "train_seed": meta["train"]["seed"],
        "fill_kv": meta.get("fill_kv", True),
        "proj_fill": meta.get("proj_fill", False),
        "depth8_bpc": from_curve(meta, 8, "bpc"),
        "depth4_bpc": from_curve(meta, 4, "bpc"),
        "depth8_acc": from_curve(meta, 8, "acc"),
        "depth4_acc": from_curve(meta, 4, "acc"),
        "tf4_accept": tf_at(meta, 4),
        "tf_acceptance": meta["tf_acceptance"],
        "depth_curve": meta["depth_curve"],
        "avg_train_gate_mean": meta["avg_train_gate_mean"],
        "avg_train_depth_ce": meta["avg_train_depth_ce"],
    }


def fill_fraction(n_layers):
    """由 exit_mode=uniform（d_j~U{0..L}）导出的填充位置占比：第 k 层已退出位置
    = P(d_j<k) = k/(L+1)。**非落盘测量字段**（引擎不记录该量），据采样律导出。"""
    per = [round(k / (n_layers + 1), 4) for k in range(n_layers)]
    return {"per_layer": per, "mean": round(sum(per) / len(per), 4),
            "provenance": "derived from exit_mode=uniform sampling law (not a logged field)"}


def rescore_rows(generation):
    by = {(r["depth"], r["commit"]): r for r in generation}
    depths = sorted({r["depth"] for r in generation})
    out = {"rescore_bpc": {}, "delta_soft_minus_hard": {},
           "delta_latent_minus_hard": {}, "legal": {}, "distinct2": {}}
    for dp in depths:
        h = by[(dp, "hard")]
        s = by[(dp, "soft")]
        l = by[(dp, "latent")]
        out["rescore_bpc"][str(dp)] = {c: by[(dp, c)]["rescore_bpc"]
                                      for c in ("soft", "hard", "latent")}
        out["delta_soft_minus_hard"][str(dp)] = round(s["rescore_bpc"] - h["rescore_bpc"], 4)
        out["delta_latent_minus_hard"][str(dp)] = round(l["rescore_bpc"] - h["rescore_bpc"], 4)
        out["legal"][str(dp)] = {c: by[(dp, c)]["legal"] for c in ("soft", "hard", "latent")}
        out["distinct2"][str(dp)] = {c: by[(dp, c)]["distinct2"]
                                     for c in ("soft", "hard", "latent")}
    peak = {c: max(r["rescore_bpc"] for r in generation if r["commit"] == c)
            for c in ("soft", "hard", "latent")}
    out["rescore_peak"] = peak
    out["delta_peak_soft_minus_hard"] = round(peak["soft"] - peak["hard"], 4)
    out["delta_peak_latent_minus_hard"] = round(peak["latent"] - peak["hard"], 4)
    return out


def paired_bootstrap_ci(vals, seed=1234, n=20000):
    """配对 bootstrap 95% CI（重采样单位 = seed 复本）。n<2 ⇒ None；n=2 ⇒ 区间退化
    （端点≈两点值），显著性以「两 seed 的符号是否一致」为准（契约 §4.2）。"""
    vals = list(vals)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    a = np.array(vals, dtype=float)
    means = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4),
            round(float(np.percentile(means, 97.5)), 4)]


def has_wall_clock(obj, path="", hits=None):
    hits = [] if hits is None else hits
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).endswith("_wall_clock_fields"):
                continue
            if WALL_RE.search(str(k)):
                hits.append(f"{path}.{k}")
            has_wall_clock(v, f"{path}.{k}", hits)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            has_wall_clock(v, f"{path}[{i}]", hits)
    return hits


def main():
    pk = {}
    for tag, seed in (("seed0", 0), ("seed1", 1)):
        pk[tag] = {}
        for arm in ARMS:
            m = load(snap_dir(arm, seed))
            if m:
                pk[tag][arm] = point(m)
    ref = {}
    for arm in ARMS:
        m = load(REF12[arm])
        if m:
            ref[arm] = point(m)

    def delta(d, a, b, field):
        if a in d and b in d:
            return round(d[a][field] - d[b][field], 4)
        return None

    deltas = {}
    for tag, d in pk.items():
        deltas[tag] = {
            "fill_minus_nofill_depth8": delta(d, "fill", "nofill", "depth8_bpc"),
            "fill_minus_nofill_depth4": delta(d, "fill", "nofill", "depth4_bpc"),
            "projfill_minus_fill_depth8": delta(d, "projfill", "fill", "depth8_bpc"),
            "projfill_minus_fill_depth4": delta(d, "projfill", "fill", "depth4_bpc"),
        }

    # 复本一致性（seed 作为复本轴）：符号是否一致 + 极差
    repro = {}
    for name, field in [("Delta(fill - nofill) depth8", "fill_minus_nofill_depth8"),
                        ("Delta(fill - nofill) depth4", "fill_minus_nofill_depth4"),
                        ("Delta(projfill - fill) depth8", "projfill_minus_fill_depth8"),
                        ("Delta(projfill - fill) depth4", "projfill_minus_fill_depth4")]:
        v0, v1 = deltas["seed0"].get(field), deltas["seed1"].get(field)
        vals = [v for v in (v0, v1) if v is not None]
        if len(vals) < 2:
            status, sign_ok, rng = "insufficient_replication (n<2)", None, None
        else:
            sign_ok = (v0 > 0) == (v1 > 0)
            status = "reproducible" if sign_ok else "inconsistent"
            rng = round(max(vals) - min(vals), 4)
        repro[name] = {
            "seed0": v0, "seed1": v1, "n_seeds": len(vals), "status": status,
            "sign_consistent": sign_ok, "range": rng, "values": vals,
            "scale_note": "同一 6 个 run 统一快照刻度；与 12k 档不可比",
        }

    # 活性列 + 可读性
    activity, readability = {}, {}
    for tag, seed in (("seed0", 0), ("seed1", 1)):
        for arm in ARMS:
            m = load(snap_dir(arm, seed))
            if not m:
                continue
            activity[f"{arm}-s{seed}"] = {
                "avg_train_gate_mean": m["avg_train_gate_mean"],
                "avg_train_depth_ce": m["avg_train_depth_ce"],
                "exit_mode": m["exit_mode"],
                "fill_fraction": fill_fraction(m["config"]["n_layers"]),
            }
            readability[f"{arm}-s{seed}"] = {
                "tf_acceptance": m["tf_acceptance"],
                "depth_curve_acc": [{"depth": r["depth"], "acc": r["acc"]}
                                    for r in m["depth_curve"]],
            }

    # rescore 表（快照刻度 6 个 + 12k 参照 3 个）
    rescore, missing_ck = {}, []
    sources = [(f"{D}/rescore-snap{TAG}.json", SNAP_STEPS, "snap"),
               (f"{D}/rescore-seed0.json", 12000, "ref12k")]
    for path, steps, kind in sources:
        if not os.path.exists(path):
            continue
        for name, rec in json.load(open(path)).items():
            cls = ("projfill" if rec.get("proj_fill")
                   else ("nofill" if rec.get("fill_kv") is False else "fill"))
            rescore[name] = {
                "kind": kind, "scale_steps": steps, "seed": rec["seed"],
                "arm_class": cls, "depths": rec["depths"], "commits": rec["commits"],
                "stored_match": rec.get("stored_match"),
                **rescore_rows(rec["generation"]),
            }
    for seed in (0, 1):
        for arm in ARMS:
            if f"snap{TAG}-{arm}-s{seed}" not in rescore:
                missing_ck.append(f"snap{TAG}-{arm}-s{seed}")

    # 显著性：逐臂以 seed 复本做配对 bootstrap（n=2 退化，须显式声明）
    rescore_ci = {}
    for arm in ARMS:
        vals, labels = [], []
        for name, r in sorted(rescore.items()):
            if r["arm_class"] == arm and r["kind"] == "snap":
                vals.append(r["delta_peak_soft_minus_hard"]); labels.append(name)
        ci = paired_bootstrap_ci(vals)
        signs = {v > 0 for v in vals}
        rescore_ci[arm] = {
            "checkpoints": labels,
            "delta_peak_soft_minus_hard": vals,
            "paired_bootstrap_95ci": ci,
            "resampling_unit": "seed 复本（同一 checkpoint 内 soft/hard 配对）",
            "ci_excludes_0": bool(ci is not None and (ci[0] > 0 or ci[1] < 0)),
            "sign_consistent": (len(signs) == 1 if len(signs) else None),
            "ci_degenerate_note": "n=2 ⇒ bootstrap 区间端点≈两点值；显著性以两 seed 符号一致 + CI 不含 0 为准（契约 §4.2）",
            "n_checkpoints": len(vals),
        }

    # config 对平清单（本目录下所有 config-parity-*.json）
    parity = {}
    for fn in sorted(os.listdir(D)):
        if fn.startswith("config-parity-") and fn.endswith(".json"):
            parity[fn] = json.load(open(os.path.join(D, fn)))

    missing_runs = [f"snap{TAG}-{arm}-s{seed}"
                    for seed in (0, 1) for arm in ARMS
                    if not os.path.exists(os.path.join(snap_dir(arm, seed), "results.json"))]

    n_seeds_ok = {a: sum(1 for s in (0, 1) if a in pk[f"seed{s}"]) for a in ARMS}
    criteria = {
        "L0": {
            "same_scale_seed_pairs": {
                k: v["passed"] for k, v in parity.items()
                if k.startswith(f"config-parity-snap{TAG}-") and "-s0-vs-s1" in k
            },
            "cross_scale_vs_12k_all_explained": {
                k: v["passed"] for k, v in parity.items() if k.endswith("-vs-12k.json")
            },
            "params": {f"{tag}-{a}": pk[tag][a]["params"] for tag in pk for a in pk[tag]},
            "note": "同刻度种子对只差 train.seed（allowed 为空）；对 12k 档的差异仅 train.steps（逐条记入 allowed_diffs）",
        },
        "L1": {"n_seeds_per_arm": n_seeds_ok, "reproducibility": repro},
        "L2": {"rescore_ci": rescore_ci,
               "declaration": "同一 checkpoint ⇒ 参数逐位相同，容量因素被剥离"},
        "L3": {"activity_reported": sorted(activity)},
        "L4": {"readability_reported": sorted(readability)},
        "L5": {},
    }

    results = {
        "experiment": "2672-fill-n2-13m",
        "arms": ARMS,
        "scale": {
            "steps": SNAP_STEPS, "batch_size": 32, "seq_len": 256, "lr": 1e-3,
            "train_tokens_per_run": SNAP_STEPS * 32 * 256,
            "scale_note": f"{TAG} 快照口径：与既有 12k 档不可比，只读臂间增量与排序",
            "why_this_step_count": (
                "锁饥饿：本段时间盒内 GPU 锁被其它 worker（2671-padfill-13m）连续占用，"
                "按 23e 实测（12000 步 ≈ 9.9 min）可用 GPU 预算 ≈10 min ⇒ 取 6 run 总量 ≤10 min "
                "的最大 500 倍数 = 2000（≥1500 下限、臂间同刻度，契约 §2 预登记允许缩步数）"),
        },
        "l5_note": "判据区不含墙钟字段；墙钟只进 run_log.txt / 各 run 的 train.log",
        "point_values": pk,
        "reference_12k": {
            "note": "既有 12k 档读数（seed 0，三个臂）——**与上面的快照刻度不可比**，仅作方向参照",
            "seed0": ref,
        },
        "deltas": deltas,
        "reproducibility": repro,
        "activity": activity,
        "readability": readability,
        "rescore": {
            "declaration": "同一 checkpoint ⇒ 参数逐位相同，容量因素被剥离",
            "ci_unit_note": "rescore 峰值/量属率类口径 ⇒ 配对 bootstrap 为主判，不跨量纲借用 0.014 bpc 带",
            "checkpoints": rescore,
            "missing_checkpoints": missing_ck,
        },
        "rescore_ci": rescore_ci,
        "config_parity": parity,
        "missing_runs": missing_runs,
        "criteria": criteria,
    }

    hits = has_wall_clock(results)
    results["l5_pass"] = (len(hits) == 0)
    results["l5_wall_clock_fields"] = hits

    with open(os.path.join(D, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[aggregate] wrote results.json; l5_pass={results['l5_pass']} hits={hits}")
    print(f"[aggregate] points: seed0={sorted(pk['seed0'])} seed1={sorted(pk['seed1'])}")
    print(f"[aggregate] missing_runs={missing_runs} missing_checkpoints={missing_ck}")
    print(f"[aggregate] deltas={json.dumps(deltas, ensure_ascii=False)}")
    print(f"[aggregate] repro={json.dumps(repro, ensure_ascii=False)}")
    print(f"[aggregate] rescore_ci={json.dumps(rescore_ci, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
