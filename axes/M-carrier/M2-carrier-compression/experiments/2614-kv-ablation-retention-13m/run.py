"""2614 · KV 消融保留曲线（ticket 31 契约 I）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
仪器实现：`tools/evalkit/retention.py`（评测期删 KV，不改权重、不训练）。
被测量模型：`2610-frozen-trunk-jumphead-13m/trunk.pt`（复用，运行时断言 sha256）。

跑三件事：① 仪器自检（身份因果掩码 == 释放版前向，逐位）；② 三条删除曲线
（距离 / 前缀截断 / 连续段）+ 判据 (a)(b)(c)(d)；③ 复现性与噪声底。

用法：
  python3 run.py                 # 全电池（约 1–2 min，共享卡）
  python3 run.py --smoke         # 冒烟：2 个窗口批次 + 少量配置，实测步时/显存
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                      # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402
from tools.evalkit.retention import (distance_keep_mask,      # noqa: E402
                                     prefix_keep_mask, segment_keep_mask,
                                     mask_stats, masked_bpc, row_bpc,
                                     safe_radius, monotonicity)

RELEASE_ANCHOR_BPC = 1.4836   # 2610/MODEL.md 记的 13.5M 主干 clean bpc（口径见 README）


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trunk(cfg: dict, device: str):
    """加载被测量主干，并**断言 sha256**（复用他人 checkpoint 的强制项）。"""
    ckpt = os.path.join(ROOT, cfg["checkpoint"]["path"])
    got = sha256_file(ckpt)
    want = cfg["checkpoint"]["sha256"]
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    size = os.path.getsize(ckpt)
    model = DepthARModel(DepthARConfig(**cfg["model"]), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": cfg["checkpoint"]["path"], "sha256": got,
                   "size_bytes": size, "sha256_verified": True}


def segment_starts(T: int, lengths: list[int], n_starts: int, seed: int) -> dict:
    """固定 seed 抽随机起点（可复现）：每个长度 n_starts 个起点。"""
    gen = torch.Generator().manual_seed(seed)
    return {L: [int(v) for v in torch.randint(0, T - L + 1, (n_starts,), generator=gen)]
            for L in lengths}


def build_mask(kind: str, T: int, **kw):
    if kind == "distance":
        return distance_keep_mask(T, kw["d"], device="cuda")
    if kind == "prefix":
        return prefix_keep_mask(T, kw["n_dropped"], device="cuda")
    if kind == "segment":
        return segment_keep_mask(T, kw["start"], kw["length"], device="cuda")
    raise ValueError(kind)


def damage_row(bpc: float, bpc_ref: float, keep: torch.Tensor, **ids) -> dict:
    """一行结果：Δ = bpc_对照 − bpc_实验（正=改善，SCHEMA §17.3）；
    damage = bpc_实验 − bpc_对照（正=变差）——单调性判据跑在 damage 上。"""
    return {"bpc": round(bpc, 6),
            "delta_bpc": round(bpc_ref - bpc, 6),
            "damage_bpc": round(bpc - bpc_ref, 6),
            **ids, **mask_stats(keep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None,
                    help="落盘目录；缺省 = 本目录（--smoke 时 = 本目录/smoke，"
                         "避免冒烟覆盖正式 results.json）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：2 个窗口批次 + 少量配置，实测步时与显存（不外推时间）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE

    cfg = json.load(open(args.config))
    ecfg = cfg["eval"]
    if args.smoke:
        ecfg["n_batches"] = 2
        cfg["curves"]["distance"]["d"] = [1, 8, 64]
        cfg["curves"]["prefix"]["n_dropped"] = [8, 64]
        cfg["curves"]["segment"]["n_starts"] = 1
        cfg["noise_floor"]["window_seeds"] = [3, 4]
        cfg["reproducibility"]["rerun"] = False
    os.makedirs(args.out_dir, exist_ok=True)

    dev = ecfg["device"]
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    model, ck = load_trunk(cfg, dev)
    T = cfg["model"]["seq_len"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log_f = open(os.path.join(args.out_dir, "retention_log.jsonl"), "w")
    neval = {"n": 0}

    def bpc_of(keep, window_seed=None, tag=""):
        g = {"n_batches": ecfg["n_batches"], "batch_size": ecfg["batch_size"],
             "window_seed": ecfg["window_seed"] if window_seed is None else window_seed,
             "device": dev}
        t = time.time()
        v = masked_bpc(model, eval_arr, keep, **g)
        neval["n"] += 1
        log_f.write(json.dumps({"tag": tag, "window_seed": g["window_seed"],
                                "bpc": round(v, 6),
                                "sec": round(time.time() - t, 4)}, ensure_ascii=False) + "\n")
        log_f.flush()
        return v

    # ---------- 仪器自检：身份掩码 ≡ 释放版前向 ----------
    bpc_release = bpc_of(None, tag="full_release_path")
    ident = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    bpc_identity = bpc_of(ident, tag="full_identity_mask")
    instrument = {
        "bpc_full_release_path": round(bpc_release, 6),
        "bpc_full_identity_mask": round(bpc_identity, 6),
        "abs_diff_release_vs_identity": abs(bpc_release - bpc_identity),
        "bitwise_equal": bpc_release == bpc_identity,
        "release_anchor_bpc_2610": RELEASE_ANCHOR_BPC,
        "abs_diff_vs_anchor": round(abs(bpc_release - RELEASE_ANCHOR_BPC), 6),
    }
    print(f"[自检] 释放版 {bpc_release:.6f} / 身份掩码 {bpc_identity:.6f} "
          f"(逐位 {'=' if instrument['bitwise_equal'] else '≠'}); "
          f"锚 {RELEASE_ANCHOR_BPC}", flush=True)

    # ---------- 三条曲线 ----------
    curves = {"full": {"bpc": round(bpc_release, 6), **mask_stats(ident)}}

    dist_rows, dist_bpc = [], {}
    for d in cfg["curves"]["distance"]["d"]:
        keep = build_mask("distance", T, d=d)
        v = bpc_of(keep, tag=f"distance:d={d}")
        dist_bpc[d] = v
        dist_rows.append(damage_row(v, bpc_release, keep, d=d))
    curves["distance"] = dist_rows

    pre_rows, pre_bpc = [], {}
    base_rows = row_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                        batch_size=ecfg["batch_size"], window_seed=ecfg["window_seed"],
                        device=dev)
    pre_diag = []
    for n in cfg["curves"]["prefix"]["n_dropped"]:
        keep = build_mask("prefix", T, n_dropped=n)
        v = bpc_of(keep, tag=f"prefix:N={n}")
        pre_bpc[n] = v
        pre_rows.append(damage_row(v, bpc_release, keep, n_dropped=n))
        # 诊断：把前缀曲线的损伤拆成「被删前缀内的 query」与「前缀外的 query」。
        # 前缀删除会连带抽掉「位置 i < N 的 query 自己的全部历史」（它们只剩
        # 对角）——那是冷启动伪影，不是「丢老前缀」的代价；两列必须分开看。
        rows_n = row_bpc(model, eval_arr, keep, n_batches=ecfg["n_batches"],
                         batch_size=ecfg["batch_size"], window_seed=ecfg["window_seed"],
                         device=dev)
        neval["n"] += 1
        mean_rows = sum(rows_n) / len(rows_n)
        assert abs(mean_rows - v) < 1e-6, (mean_rows, v)   # 算术对账（行均值 == token 均值）
        qmin = cfg["curves"]["prefix_diagnostics"]["query_min"]
        d_lo = sum(rows_n[:n]) / max(n, 1) - sum(base_rows[:n]) / max(n, 1)
        d_hi = (sum(rows_n[n:]) / (T - 1 - n) - sum(base_rows[n:]) / (T - 1 - n)
                if n < T - 1 else 0.0)
        n2 = min(2 * n, T - 1)
        d_near = ((sum(rows_n[n:n2]) / max(n2 - n, 1)
                   - sum(base_rows[n:n2]) / max(n2 - n, 1)) if n2 > n else 0.0)
        d_far = ((sum(rows_n[n2:]) / max(T - 1 - n2, 1)
                  - sum(base_rows[n2:]) / max(T - 1 - n2, 1)) if n2 < T - 1 else 0.0)
        pre_diag.append({
            "n_dropped": n,
            "damage_bpc": round(v - bpc_release, 6),
            "damage_rows_inside_prefix": round(d_lo, 6),
            "damage_rows_boundary": round(d_near, 6),
            "damage_rows_far": round(d_far, 6),
            "contribution_inside": round(n / (T - 1) * d_lo, 6),
            "contribution_boundary": round((n2 - n) / (T - 1) * d_near, 6),
            "contribution_far": round((T - 1 - n2) / (T - 1) * d_far, 6),
            "damage_qmin128": round(sum(rows_n[qmin:]) / (T - 1 - qmin)
                                    - sum(base_rows[qmin:]) / (T - 1 - qmin), 6),
        })
        log_f.write(json.dumps({"tag": f"row_bpc:prefix:N={n}",
                                "mean_bpc": round(mean_rows, 6),
                                "sec": 0}, ensure_ascii=False) + "\n")
    curves["prefix"] = pre_rows
    curves["prefix_diagnostics"] = {
        "query_min": cfg["curves"]["prefix_diagnostics"]["query_min"],
        "rows": pre_diag,
        "note": ("damage_rows_inside_prefix = 位置 i < N 的 query 的平均 CE 差（这些"
                 "query 被删前缀后只剩对角 = 冷启动伪影）；damage_rows_boundary = "
                 "紧邻前缀的 query（N ≤ i < 2N，它们的历史长度 ≤ 2N）；"
                 "damage_rows_far = i ≥ 2N；damage_qmin128 = 只统计长上下文 query"
                 "（i ≥ 128）的配对损伤。三个 band 的贡献之和 == 总 damage（对账）。")}

    starts = segment_starts(T, cfg["curves"]["segment"]["lengths"],
                            cfg["curves"]["segment"]["n_starts"],
                            cfg["curves"]["segment"]["start_seed"])
    seg_rows, seg_summary = [], []
    for L, ss in starts.items():
        dams = []
        for s in ss:
            keep = build_mask("segment", T, start=s, length=L)
            v = bpc_of(keep, tag=f"segment:L={L}@{s}")
            row = damage_row(v, bpc_release, keep, length=L, start=s)
            seg_rows.append(row)
            dams.append(row["damage_bpc"])
        seg_summary.append({"length": L, "n_starts": len(dams),
                            "mean_bpc": round(sum(d["bpc"] for d in seg_rows if d["length"] == L) / len(dams), 6),
                            "mean_delta_bpc": round(sum(-d for d in dams) / len(dams), 6),
                            "mean_damage_bpc": round(sum(dams) / len(dams), 6),
                            "min_damage_bpc": round(min(dams), 6),
                            "max_damage_bpc": round(max(dams), 6)})
    curves["segment_by_start"] = seg_rows
    curves["segment_by_length"] = seg_summary

    # ---------- 复现性（同 seed 重跑）----------
    repro = {"threshold": cfg["reproducibility"]["max_delta_diff_threshold"],
             "rerun": bool(cfg["reproducibility"]["rerun"]), "per_config": []}
    if cfg["reproducibility"]["rerun"]:
        def redo(kind, idkey, **kw):
            keep = build_mask(kind, T, **kw)
            v2 = bpc_of(keep, tag=f"rerun:{kind}:{idkey}")
            v1 = {"distance": dist_bpc, "prefix": pre_bpc}[kind][kw[idkey]]
            return {"config": f"{kind}:{idkey}={kw[idkey]}",
                    "damage_run1": round(v1 - bpc_release, 6),
                    "damage_run2": round(v2 - bpc_release, 6),
                    "abs_diff": round(abs(v2 - v1), 9)}
        for d in cfg["curves"]["distance"]["d"]:
            repro["per_config"].append(redo("distance", "d", d=d))
        for n in cfg["curves"]["prefix"]["n_dropped"]:
            repro["per_config"].append(redo("prefix", "n_dropped", n_dropped=n))
        for L, ss in starts.items():
            for s in ss:
                keep = build_mask("segment", T, start=s, length=L)
                v1 = [r for r in seg_rows if r["length"] == L and r["start"] == s][0]["bpc"]
                v2 = bpc_of(keep, tag=f"rerun:segment:L={L}@{s}")
                repro["per_config"].append({"config": f"segment:L={L}@{s}",
                                            "damage_run1": round(v1 - bpc_release, 6),
                                            "damage_run2": round(v2 - bpc_release, 6),
                                            "abs_diff": round(abs(v2 - v1), 9)})
        repro["max_abs_damage_diff"] = max(r["abs_diff"] for r in repro["per_config"])
        repro["pass"] = repro["max_abs_damage_diff"] < repro["threshold"]

    # ---------- 噪声底（不同评测窗口 seed 下的配对 damage 散布）----------
    nf = {"window_seeds": cfg["noise_floor"]["window_seeds"], "configs": []}
    for c in cfg["noise_floor"]["configs"]:
        kw = {k: v for k, v in c.items() if k not in ("id", "kind")}
        keep = build_mask(c["kind"], T, **kw)
        dams = []
        for sd in nf["window_seeds"]:
            ref = bpc_of(ident, window_seed=sd, tag=f"nf_ref:{c['id']}")
            v = bpc_of(keep, window_seed=sd, tag=f"nf:{c['id']}@{sd}")
            dams.append(v - ref)
        mean = sum(dams) / len(dams)
        var = sum((d - mean) ** 2 for d in dams) / max(len(dams) - 1, 1)
        nf["configs"].append({"id": c["id"], "n": len(dams),
                              "mean_damage_bpc": round(mean, 6),
                              "std_damage_bpc": round(var ** 0.5, 6),
                              "damage_bpc": [round(d, 6) for d in dams]})
    stds = sorted(c["std_damage_bpc"] for c in nf["configs"])
    nf["max_std_damage_bpc"] = stds[-1] if stds else None
    nf["median_std_damage_bpc"] = stds[len(stds) // 2] if stds else None

    # ---------- 判据 ----------
    band = cfg["variance_band_bpc"]
    tol = 2 * (nf["max_std_damage_bpc"] or 0.0)     # 噪声容差 = 2σ（跨窗口 seed）
    mono = {
        "distance": monotonicity(dist_rows, "d", direction="decreasing"),
        "prefix": monotonicity(pre_rows, "n_dropped", direction="increasing"),
        "segment": monotonicity(seg_summary, "length",
                                field="mean_damage_bpc", direction="increasing"),
    }
    mono_noise = {
        "distance": monotonicity(dist_rows, "d", direction="decreasing", tol=tol),
        "prefix": monotonicity(pre_rows, "n_dropped", direction="increasing", tol=tol),
        "segment": monotonicity(seg_summary, "length", field="mean_damage_bpc",
                                direction="increasing", tol=tol),
    }
    all_mono = all(v["monotone"] for v in mono.values())
    all_mono_noise = all(v["monotone"] for v in mono_noise.values())
    safe = safe_radius(dist_rows, band=band)
    # 判据 (c) 的字面读法（|Δ| < band 的最大 d）在单调曲线下是退化的（集合的最大值
    # 恒为最大测试距离）。同时给出**可操作读法**：从哪个距离起，更远的 KV 全部
    # 可丢（即「保留最近 d−1 个位置、其余全丢」的代价 < band）。
    dist_sorted = sorted(dist_rows, key=lambda r: r["d"])
    op_d = None
    for k, row in enumerate(dist_sorted):
        if all(abs(r["damage_bpc"]) < band for r in dist_sorted[k:]):
            op_d = row["d"]
            break
    criteria = {
        "band_bpc": band,
        "noise_tol_bpc_2sigma": round(tol, 6),
        "a_three_curves_monotone": all_mono,
        "a_monotone_with_noise_tolerance": all_mono_noise,
        "a_detail": mono,
        "a_detail_with_tolerance": mono_noise,
        # 契约 (a) 括号里对距离曲线的字面读法（「距离越远损伤不应更小」）与物理
        # 预期相反；这里把它也跑一遍、如实报结果，不改判据线。
        "a_distance_literal_direction": monotonicity(
            dist_rows, "d", direction="increasing"),
        "b_reproducibility_max_abs_diff": repro.get("max_abs_damage_diff"),
        "b_reproducibility_pass": repro.get("pass"),
        "c_safe_radius_d_literal_max": safe,
        "c_safe_radius_d_operational": {
            "value": op_d, "band": band,
            "meaning": (f"距离 ≥ {op_d} 的 KV 全部可丢，累计代价 < {band} bpc"
                        if op_d is not None else "无距离可安全丢弃")},
        "c_prefix_curve": {str(r["n_dropped"]): r["damage_bpc"] for r in pre_rows},
        "d_instrument_trusted": all_mono_noise,
        "verdict": ("仪器可信：三条曲线在噪声容差内单调" if all_mono_noise else
                    "仪器不可信：存在非单调且超出噪声容差的曲线 ⇒ 先修仪器并记 ERRATA"),
    }

    alive = max(abs(r["damage_bpc"]) for r in dist_rows + pre_rows + seg_rows) > band
    intervention = {
        "what": "评测期删除 KV：把指定位置的 key 从 query 的可见集中移除（注意力掩码）",
        "params_added": 0, "weights_changed": False,
        "alive": alive,
        "max_abs_damage_bpc": round(max(abs(r["damage_bpc"])
                                        for r in dist_rows + pre_rows + seg_rows), 6),
        "identity_mask_is_noop": instrument["bitwise_equal"],
        "activity_note": ("活性量 = 每个配置被删的因果 (query,key) 对数 / 占比 "
                          "（见 curves[*].n_dropped_pairs / dropped_frac）；"
                          "本实验无训练步，故无「每 200 步」训练活性日志"),
    }

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    results = {
        "contract": ("axes/M-carrier/M2-carrier-compression/experiments/"
                     "2614-kv-ablation-retention-13m/README.md"),
        "checkpoint": ck,
        "config": cfg,
        "instrument": instrument,
        "intervention": intervention,
        "curves": curves,
        "reproducibility": repro,
        "noise_floor": nf,
        "criteria": criteria,
        "code": {
            "run_py_sha256": sha256_file(os.path.abspath(__file__)),
            "retention_py_sha256": sha256_file(
                os.path.join(ROOT, "tools", "evalkit", "retention.py")),
            "tests_sha256": sha256_file(os.path.join(ROOT, "tests", "test_retention.py")),
        },
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "peak_mem_mib": round(peak, 1), "n_evals": neval["n"],
                 "device": torch.cuda.get_device_name(0)},
    }
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    log_f.close()
    print(json.dumps({k: v for k, v in results.items() if k not in ("config",)},
                     indent=2, ensure_ascii=False)[:4000])
    print(f"\n落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB，"
          f"{neval['n']} 次评测", flush=True)


if __name__ == "__main__":
    main()
