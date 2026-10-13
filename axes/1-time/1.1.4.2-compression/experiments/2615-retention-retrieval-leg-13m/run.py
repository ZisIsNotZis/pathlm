"""2615 · 保留曲线补「检索腿」（ticket 32 契约 M）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
仪器：`tools/evalkit/retention.py`（依赖腿 = ticket 31-I 的三条曲线；检索腿 = 本实验新增：
`recent_keep_mask` / `phrase_needle_batch` / `needle_fate` / `needle_recall`）。
被测量模型：`2610-frozen-trunk-jumphead-13m/trunk.pt`（复用，运行时断言 sha256）。

跑四件事：
  ① 仪器自检：身份因果掩码 == 释放版前向（逐位）+ 检索腿配对自检（plant=False 与
     plant=True 同一 generator 状态 ⇒ meta 逐字段相同）；
  ② 构造诊断：契约「方法」原文指定的 needle v2 构造在本 checkpoint 上的基线召回
     （不足 50% 即为构造问题 ⇒ 记 ERRATA 并改用 needle v3，见 README §3）；
  ③ 两条腿 × 全部保留配置（全保留 / 删最远 N / 只保留最近 d / 连续段删除）；
  ④ 判据：主判据「两腿必须给出不同信息」+ 基线召回归档 + 地板（无针控制臂）。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡）
  python3 run.py --smoke         # 冒烟：少批 + 少配置，实测步时/显存（不外推时间）
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

from pathlm.data import load_enwik8_full, needle_batch            # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel           # noqa: E402
from tools.evalkit.retention import (prefix_keep_mask,            # noqa: E402
                                     segment_keep_mask, recent_keep_mask,
                                     mask_stats, masked_bpc, row_bpc,
                                     masked_logits, phrase_needle_batch,
                                     needle_recall, needle_summary)

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
    """固定 seed 抽随机起点（可复现）：每个长度 n_starts 个起点（与 2614 同法）。"""
    gen = torch.Generator().manual_seed(seed)
    return {L: [int(v) for v in torch.randint(0, T - L + 1, (n_starts,), generator=gen)]
            for L in lengths}


def build_configs(cfg: dict, T: int, device: str) -> list[dict]:
    """全部保留配置。`keep=None` = 释放版全注意力（全保留基线）。"""
    out = [{"id": "full", "kind": "full", "keep": None, "params": {}}]
    for N in cfg["configs"]["prefix"]["n_dropped"]:
        out.append({"id": f"prefix:N={N}", "kind": "prefix", "params": {"n_dropped": N},
                    "keep": prefix_keep_mask(T, N, device)})
    for d in cfg["configs"]["recent"]["d"]:
        out.append({"id": f"recent:d={d}", "kind": "recent", "params": {"d": d},
                    "keep": recent_keep_mask(T, d, device)})
    starts = segment_starts(T, cfg["configs"]["segment"]["lengths"],
                            cfg["configs"]["segment"]["n_starts"],
                            cfg["configs"]["segment"]["start_seed"])
    for L, ss in starts.items():
        for s in ss:
            out.append({"id": f"segment:L={L}@{s}", "kind": "segment",
                        "params": {"length": L, "start": s},
                        "keep": segment_keep_mask(T, s, L, device)})
    return out


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def v2_construction_diagnosis(model, eval_arr, ncfg, dev, log_f):
    """契约「方法」原文指定的 v2 构造（随机字节对针）在本 checkpoint 上的基线召回。"""
    gen = torch.Generator().manual_seed(ncfg["seed"])
    T = model.cfg.seq_len
    buckets = {"anchor": [], "in_window": []}
    with torch.no_grad():
        for bi in range(ncfg["n_batches"]):
            x, _, meta = needle_batch(ncfg["batch_size"], T, 205, 205, gen, data=eval_arr,
                                      anchors=ncfg["anchors"],
                                      anchor_frac=ncfg["anchor_frac"],
                                      n_needles=ncfg["n_needles"])
            x = x.to(dev)
            logits = masked_logits(model, x, None).cpu()
            for b, rows in enumerate(meta):
                for q_row, p, _d in rows:
                    hit = int(logits[b, q_row].argmax(-1).item() == int(x[b, q_row + 1].cpu()))
                    buckets["anchor" if p < ncfg["anchors"] else "in_window"].append(hit)
            n_so_far = len(buckets["in_window"]) + len(buckets["anchor"])
            log_f.write(json.dumps({"tag": "v2_diagnosis_batch", "batch": bi,
                                    "n_cum": n_so_far,
                                    "recall_cum": round(sum(buckets["in_window"]
                                                           + buckets["anchor"]) / max(n_so_far, 1), 6)},
                                   ensure_ascii=False) + "\n")
            log_f.flush()
    allv = buckets["in_window"] + buckets["anchor"]
    return {"construction": "pathlm.data.needle_batch（v2：随机字节对针）",
            "n": len(allv), "recall": round(sum(allv) / max(len(allv), 1), 6),
            "buckets": {k: {"recall": round(sum(v) / len(v), 6) if v else None, "n": len(v)}
                        for k, v in buckets.items()},
            "meets_50pct_floor": (sum(allv) / max(len(allv), 1)) >= 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None,
                    help="落盘目录；缺省 = 本目录（--smoke 时 = 本目录/smoke）")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE

    cfg = json.load(open(args.config))
    ecfg = cfg["eval"]
    ncfg = cfg["needle"]
    if args.smoke:
        ecfg["n_batches"] = 2
        ncfg["n_batches"] = 2
        cfg["v2_construction_diagnosis"]["n_batches"] = 1
        cfg["configs"]["prefix"]["n_dropped"] = [8, 64]
        cfg["configs"]["recent"]["d"] = [32, 256]
        cfg["configs"]["segment"]["n_starts"] = 1
        cfg["reproducibility"]["rerun"] = False
        cfg["noise_floor"]["window_seeds"] = [3, 4]
        cfg["noise_floor"]["needle_seeds"] = [3]
    os.makedirs(args.out_dir, exist_ok=True)

    dev = ecfg["device"]
    _, eval_arr, _ = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                      os.path.join(ROOT, cfg["data"]["path"]))
    model, ck = load_trunk(cfg, dev)
    T = cfg["model"]["seq_len"]
    mask_tok, n_real = cfg["data"]["mask_token"], cfg["data"]["n_real_tokens"]
    L, anchors = ncfg["phrase_len"], ncfg["anchors"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log_f = open(os.path.join(args.out_dir, "retention_log.jsonl"), "w")
    neval = {"n": 0}

    # ---------- ① 仪器自检 ----------
    ident = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    bpc_release = masked_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                             batch_size=ecfg["batch_size"],
                             window_seed=ecfg["window_seed"], device=dev)
    neval["n"] += 1
    bpc_identity = masked_bpc(model, eval_arr, ident, n_batches=ecfg["n_batches"],
                              batch_size=ecfg["batch_size"],
                              window_seed=ecfg["window_seed"], device=dev)
    neval["n"] += 1
    instrument = {"bpc_full_release_path": round(bpc_release, 6),
                  "bpc_full_identity_mask": round(bpc_identity, 6),
                  "bitwise_equal": bpc_release == bpc_identity,
                  "abs_diff_vs_anchor": round(abs(bpc_release - RELEASE_ANCHOR_BPC), 6),
                  "release_anchor_bpc_2610": RELEASE_ANCHOR_BPC}
    print(f"[自检] 释放版 {bpc_release:.6f} / 身份掩码 {bpc_identity:.6f} "
          f"(逐位 {'=' if instrument['bitwise_equal'] else '≠'})", flush=True)
    # 检索腿配对自检：同一 generator 状态 ⇒ plant=True 与 plant=False 的 meta 逐字段相同
    g_probe = torch.Generator().manual_seed(ncfg["seed"])
    st = g_probe.get_state()
    _x1, _m1 = phrase_needle_batch(2, T, eval_arr, n_real, mask_tok, g_probe,
                                   anchors=anchors, anchor_frac=ncfg["anchor_frac"],
                                   n_needles=ncfg["n_needles"], phrase_len=L)
    g_probe.set_state(st)
    _x0, _m0 = phrase_needle_batch(2, T, eval_arr, n_real, mask_tok, g_probe,
                                   anchors=anchors, anchor_frac=ncfg["anchor_frac"],
                                   n_needles=ncfg["n_needles"], phrase_len=L, plant=False)
    instrument["paired_control_same_meta"] = (_m0 == _m1)
    instrument["plant_cells"] = int((_x1 != _x0).sum())
    instrument["needle_layout"] = {"n_needles": ncfg["n_needles"], "phrase_len": L,
                                   "anchors": anchors, "anchor_frac": ncfg["anchor_frac"],
                                   "q0": T - ncfg["n_needles"] * (L + 1),
                                   "answer_rows_first_row": [m["q_row"] for m in _m1[0]]}

    # ---------- ② 构造诊断（v2） ----------
    v2 = v2_construction_diagnosis(model, eval_arr, cfg["v2_construction_diagnosis"], dev, log_f)
    print(f"[构造诊断] needle v2 基线召回 = {v2['recall']:.4f} (n={v2['n']}) "
          f"→ 达到 50% 门槛: {v2['meets_50pct_floor']}", flush=True)

    # ---------- ③ 两条腿 ----------
    base_rows = row_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                        batch_size=ecfg["batch_size"], window_seed=ecfg["window_seed"],
                        device=dev)
    neval["n"] += 1
    configs = build_configs(cfg, T, dev)
    qmin_long = cfg["diagnostics"]["query_min_long"]
    qmin_needle = cfg["diagnostics"]["query_min_needle"]

    def dep_row(keep):
        """依赖腿一行：bpc + 活性量 + 长上下文口径的配对损伤（同一窗口序）。"""
        v = masked_bpc(model, eval_arr, keep, n_batches=ecfg["n_batches"],
                       batch_size=ecfg["batch_size"], window_seed=ecfg["window_seed"],
                       device=dev)
        neval["n"] += 1
        rows = row_bpc(model, eval_arr, keep, n_batches=ecfg["n_batches"],
                       batch_size=ecfg["batch_size"], window_seed=ecfg["window_seed"],
                       device=dev)
        neval["n"] += 1
        assert abs(sum(rows) / len(rows) - v) < 1e-6, (sum(rows) / len(rows), v)
        d = lambda lo: (sum(rows[lo:]) / (T - 1 - lo)
                        - sum(base_rows[lo:]) / (T - 1 - lo))
        return {"bpc": round(v, 6), "delta_bpc": round(bpc_release - v, 6),
                "damage_bpc": round(v - bpc_release, 6),
                "damage_qmin128": round(d(qmin_long), 6),
                "damage_qmin200": round(d(qmin_needle), 6)}

    # 检索腿：同一批针（plant=True / plant=False 配对），逐配置跑一遍
    needle_gen = torch.Generator().manual_seed(ncfg["seed"])
    needle_records = {c["id"]: [] for c in configs}
    control_records = []
    n_needles_total = 0
    log_f.write(json.dumps({"tag": "needle_start", "n_batches": ncfg["n_batches"],
                            "batch_size": ncfg["batch_size"], "phrase_len": L,
                            "anchors": anchors, "anchor_frac": ncfg["anchor_frac"],
                            "seed": ncfg["seed"]}, ensure_ascii=False) + "\n")
    for bi in range(ncfg["n_batches"]):
        state = needle_gen.get_state()
        x, meta = phrase_needle_batch(ncfg["batch_size"], T, eval_arr, n_real, mask_tok,
                                      needle_gen, anchors=anchors,
                                      anchor_frac=ncfg["anchor_frac"],
                                      n_needles=ncfg["n_needles"], phrase_len=L)
        needle_gen.set_state(state)
        x0, meta0 = phrase_needle_batch(ncfg["batch_size"], T, eval_arr, n_real, mask_tok,
                                        needle_gen, anchors=anchors,
                                        anchor_frac=ncfg["anchor_frac"],
                                        n_needles=ncfg["n_needles"], phrase_len=L,
                                        plant=False)
        assert meta0 == meta, "配对控制臂的针位必须与主臂逐一相同"
        n_needles_total += sum(len(m) for m in meta)
        ctrl = needle_recall(model, x0, meta, None, anchors=anchors, phrase_len=L, device=dev)
        neval["n"] += 1
        control_records.extend(ctrl["per_needle"])
        base = needle_recall(model, x, meta, None, anchors=anchors, phrase_len=L, device=dev)
        neval["n"] += 1
        needle_records["full"].extend(base["per_needle"])
        for c in configs:
            if c["kind"] == "full":
                continue
            rec = needle_recall(model, x, meta, c["keep"], anchors=anchors,
                                phrase_len=L, device=dev)
            neval["n"] += 1
            needle_records[c["id"]].extend(rec["per_needle"])
        # §10 活性量周期落盘：本实验无训练步，等价物 = 每批判定即写一行
        log_f.write(json.dumps({"tag": "needle_batch", "batch": bi, "n": base["n"],
                                "recall_full": base["recall"],
                                "recall_no_needle": ctrl["recall"],
                                "buckets_full": base["buckets"]},
                               ensure_ascii=False) + "\n")
        log_f.flush()

    # ---------- 逐配置汇总 ----------
    rows_out = []
    for c in configs:
        dep = dep_row(c["keep"])
        rec = needle_summary(needle_records[c["id"]])
        stats = mask_stats(c["keep"]) if c["keep"] is not None else {
            "n_dropped_pairs": 0, "causal_pairs": T * (T + 1) // 2, "dropped_frac": 0.0,
            "n_dropped_keys": 0, "diagonal_kept": True}
        rows_out.append({"id": c["id"], "kind": c["kind"], **c["params"], **dep, **stats,
                         "retention": rec})
        log_f.write(json.dumps({"tag": "config", **rows_out[-1]}, ensure_ascii=False) + "\n")
        log_f.flush()
        print(f"  {c['id']:<20s} Δbpc={dep['delta_bpc']:+.5f} "
              f"(qmin200 {dep['damage_qmin200']:+.5f}) | recall={rec['recall']:.4f} "
              f"n={rec['n']} {rec['buckets']}", flush=True)

    base_rec = needle_summary(needle_records["full"])
    floor_rec = needle_summary(control_records)
    q0 = T - ncfg["n_needles"] * (L + 1)
    # 配对分析（同一批、同一序 ⇒ 逐下标对应同一根针）：
    #   ① 删掉的针（visible=False）在「全保留」下的召回 = 上限，在配置下 = 下限；
    #   ② 难度控制：「无针控制臂也答不出」的针（线索不可预测 ⇒ 答案只能来自记忆）。
    deleted_of = {c["id"]: [i for i, p in enumerate(needle_records[c["id"]])
                             if not p["visible"]] for c in configs}
    hard_idx = [i for i, p in enumerate(control_records) if p["hit"] == 0]
    hard = {
        "n_hard": len(hard_idx),
        "frac_hard": round(len(hard_idx) / max(len(control_records), 1), 6),
        "baseline_recall_hard": round(sum(needle_records["full"][i]["hit"]
                                          for i in hard_idx) / max(len(hard_idx), 1), 6),
        "control_recall_hard": 0.0,
        "per_config": {c["id"]: round(sum(needle_records[c["id"]][i]["hit"]
                                           for i in hard_idx) / max(len(hard_idx), 1), 6)
                       for c in configs},
        "note": ("难度控制子集 = 无针控制臂答不出的针（答案不能从线索先验得到）。"
                 "契约的 50% 门槛按**全针**基线判：本子集基线低于 50% ⇒ 仅作诊断，"
                 "不作为主判据口径（SCHEMA §17.10 的「带病构造不得判读」）。"),
    }

    # ---------- ④ 判据 ----------
    band = cfg["variance_band_bpc"]
    crit_pp = cfg["main_criterion_recall_drop_pp"]
    for r in rows_out:
        r["recall_drop_vs_baseline"] = round(base_rec["recall"] - r["retention"]["recall"], 6)
        n_anchor = r["retention"]["buckets"].get("anchor", {"n": 0})["n"]
        n_window = r["retention"]["buckets"].get("window", {"n": 0})["n"]
        non_anchor = r["retention"]["n"] - n_anchor
        r["deleted_frac_non_anchor"] = round(1 - n_window / max(non_anchor, 1), 6)
        r["bucket_recall"] = {k: v["recall"] for k, v in r["retention"]["buckets"].items()}
        didx = deleted_of[r["id"]]
        r["n_deleted"] = len(didx)
        r["deleted_share"] = round(len(didx) / max(r["retention"]["n"], 1), 6)
        r["paired_baseline_recall_deleted"] = round(
            sum(needle_records["full"][i]["hit"] for i in didx) / max(len(didx), 1), 6)
        r["paired_recall_deleted"] = round(
            sum(needle_records[r["id"]][i]["hit"] for i in didx) / max(len(didx), 1), 6)
        r["paired_drop_deleted"] = round(
            r["paired_baseline_recall_deleted"] - r["paired_recall_deleted"], 6)
        # 删了「查询块」的配置改变了任务本身（线索没了）⇒ 标记，判据另算
        r["overlaps_query_block"] = bool(r.get("kind") == "segment"
                                          and r.get("start", 0) + r.get("length", 0) > q0)
        r["hard_recall"] = hard["per_config"][r["id"]]
        r["hard_drop_vs_baseline"] = round(hard["baseline_recall_hard"]
                                            - hard["per_config"][r["id"]], 6)

    def qualifying(key: str, drop_field: str = "recall_drop_vs_baseline",
                   exclude_query_block: bool = False):
        rows = [r for r in rows_out if not (exclude_query_block and r["overlaps_query_block"])]
        hits = [r for r in rows if abs(r[key]) < band and r[drop_field] > crit_pp]
        low = [r[drop_field] for r in rows if abs(r[key]) < band]
        return {"caliber": key, "drop_field": drop_field,
                "exclude_query_block": exclude_query_block,
                "n_qualifying": len(hits),
                "configs": [{"id": r["id"], key: r[key], drop_field: r[drop_field],
                             "n_deleted": r["n_deleted"], "deleted_share": r["deleted_share"]}
                            for r in hits],
                "n_low_damage_configs": len(low),
                "max_recall_drop_overall": round(max(r[drop_field] for r in rows), 6),
                "max_recall_drop_at_low_damage": round(max(low or [0.0]), 6)}

    criteria = {
        "band_bpc": band, "main_criterion_recall_drop_pp": crit_pp,
        "baseline_recall_full": base_rec["recall"],
        "baseline_recall_ok_ge_50pct": base_rec["recall"] >= 0.5,
        "no_needle_floor_recall": floor_rec["recall"],
        "no_needle_floor_by_bucket": {k: v["recall"] for k, v in floor_rec["buckets"].items()},
        "instrument_gap_baseline_minus_floor": round(base_rec["recall"] - floor_rec["recall"], 6),
        "max_paired_drop_deleted_needles": round(max(r["paired_drop_deleted"]
                                                     for r in rows_out if r["n_deleted"] > 0), 6),
        "main_delta_bpc_full_query": qualifying("delta_bpc"),
        "main_delta_bpc_qmin128": qualifying("damage_qmin128"),
        "main_delta_bpc_qmin200": qualifying("damage_qmin200"),
        "main_delta_bpc_full_query_no_query_block_overlap": qualifying(
            "delta_bpc", exclude_query_block=True),
        "main_on_paired_deleted_subset_full_query": qualifying(
            "delta_bpc", drop_field="paired_drop_deleted"),
        "main_on_hard_needle_subset_full_query": qualifying(
            "delta_bpc", drop_field="hard_drop_vs_baseline"),
        "main_criterion_met_full_query": qualifying("delta_bpc")["n_qualifying"] > 0,
        "v2_construction_baseline_recall": v2["recall"],
        "v2_construction_meets_floor": v2["meets_50pct_floor"],
        "two_legs_move_together": {
            "corr_delta_bpc_vs_recall_drop": round(_pearson(
                [r["delta_bpc"] for r in rows_out],
                [r["recall_drop_vs_baseline"] for r in rows_out]), 6),
            "note": ("Pearson r（配置级）：Δbpc（负 = 变差）与召回下降（正）若为**负相关**，"
                     "说明两腿同向（删得越多两边都越差）。"),
        },
    }

    # ---------- 复现性 + 噪声底 ----------
    repro = {"threshold": 0.005, "rerun": bool(cfg["reproducibility"]["rerun"]),
             "per_config": []}
    if cfg["reproducibility"]["rerun"]:
        picks = [c for c in configs if c["kind"] in ("full", "prefix", "recent")][
            :cfg["reproducibility"]["n_configs"]]
        for c in picks:
            dep2 = dep_row(c["keep"])
            v1 = [r for r in rows_out if r["id"] == c["id"]][0]
            repro["per_config"].append({"config": c["id"], "damage_run1": v1["damage_bpc"],
                                        "damage_run2": dep2["damage_bpc"],
                                        "abs_diff": round(abs(dep2["bpc"] - v1["bpc"]), 9)})
        repro["max_abs_damage_diff"] = max(r["abs_diff"] for r in repro["per_config"])
        repro["pass"] = repro["max_abs_damage_diff"] < repro["threshold"]
    nf = {"needle_seeds": cfg["noise_floor"]["needle_seeds"],
          "window_seeds": cfg["noise_floor"]["window_seeds"], "configs": []}
    for cid in ("full", "recent:d=128", "recent:d=192"):
        c = [x for x in configs if x["id"] == cid]
        if not c:
            continue
        c = c[0]
        recs, ns = [], []
        for sd in nf["needle_seeds"]:
            g = torch.Generator().manual_seed(sd)
            per = []
            for _ in range(max(1, ncfg["n_batches"] // 8)):
                x, meta = phrase_needle_batch(ncfg["batch_size"], T, eval_arr, n_real,
                                              mask_tok, g, anchors=anchors,
                                              anchor_frac=ncfg["anchor_frac"],
                                              n_needles=ncfg["n_needles"], phrase_len=L)
                r = needle_recall(model, x, meta, c["keep"], anchors=anchors,
                                  phrase_len=L, device=dev)
                neval["n"] += 1
                per.extend(r["per_needle"])
            s = needle_summary(per)
            recs.append(s["recall"])
            ns.append(s["n"])
        mean = sum(recs) / len(recs)
        var = sum((v - mean) ** 2 for v in recs) / max(len(recs) - 1, 1)
        nf["configs"].append({"id": cid, "n_seeds": len(recs), "n_per_seed": ns[0],
                              "recall": [round(v, 6) for v in recs],
                              "mean_recall": round(mean, 6),
                              "std_recall": round(var ** 0.5, 6)})
    stds = [c["std_recall"] for c in nf["configs"]]
    nf["max_std_recall"] = max(stds) if stds else None
    wf = {"configs": []}
    for cid in ("recent:d=128", "prefix:N=32"):
        c = [x for x in configs if x["id"] == cid]
        if not c:
            continue
        c = c[0]
        dams = []
        for sd in nf["window_seeds"]:
            ref = masked_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                             batch_size=ecfg["batch_size"], window_seed=sd, device=dev)
            v = masked_bpc(model, eval_arr, c["keep"], n_batches=ecfg["n_batches"],
                           batch_size=ecfg["batch_size"], window_seed=sd, device=dev)
            neval["n"] += 2
            dams.append(v - ref)
        mean = sum(dams) / len(dams)
        var = sum((d - mean) ** 2 for d in dams) / max(len(dams) - 1, 1)
        wf["configs"].append({"id": cid, "n": len(dams), "mean_damage_bpc": round(mean, 6),
                              "std_damage_bpc": round(var ** 0.5, 6),
                              "damage_bpc": [round(d, 6) for d in dams]})
    wf["max_std_damage_bpc"] = max((c["std_damage_bpc"] for c in wf["configs"]), default=None)

    alive = (max(r["n_dropped_pairs"] for r in rows_out) > 0
             and base_rec["recall"] > floor_rec["recall"])
    intervention = {
        "what": ("评测期改注意力可见集（依赖腿）+ 在窗口里植入/不植入真实文本短语针"
                 "（检索腿）；不改权重、不训练"),
        "params_added": 0, "weights_changed": False, "alive": alive,
        "max_abs_damage_bpc": round(max(abs(r["damage_bpc"]) for r in rows_out), 6),
        "retention_gap": round(base_rec["recall"] - floor_rec["recall"], 6),
        "activity_note": ("活性量 = 每个配置的被删因果 (query,key) 对数 / 占比（mask_stats），"
                          "以及逐批判定的针命中与分桶（needle_batch 行）；本实验无训练步，"
                          "等价物是**每批判定即写** retention_log.jsonl，见 SCHEMA §17.4。"),
    }

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    results = {
        "contract": ("axes/1-time/1.1.4.2-compression/experiments/"
                     "2615-retention-retrieval-leg-13m/README.md"),
        "checkpoint": ck, "config": cfg, "instrument": instrument,
        "v2_construction_diagnosis": v2, "intervention": intervention,
        "hard_needle_subset": hard,
        "baseline": {"bpc": round(bpc_release, 6), "retention": base_rec},
        "no_needle_control": {
            "retention": floor_rec,
            "note": ("plant=False，同一 generator 状态 ⇒ 针位与线索与主臂逐一配对；"
                     "= 检索腿地板「记忆不存在」")},
        "configs": rows_out, "criteria": criteria, "reproducibility": repro,
        "noise_floor": {"needle_recall": nf, "dependency_damage": wf},
        "code": {"run_py_sha256": sha256_file(os.path.abspath(__file__)),
                 "retention_py_sha256": sha256_file(
                     os.path.join(ROOT, "tools", "evalkit", "retention.py")),
                 "tests_sha256": sha256_file(
                     os.path.join(ROOT, "tests", "test_retention.py"))},
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "peak_mem_mib": round(peak, 1), "n_evals": neval["n"],
                 "n_needles": n_needles_total, "n_configs": len(rows_out),
                 "device": torch.cuda.get_device_name(0)},
    }
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    log_f.close()
    print(json.dumps({"criteria": criteria,
                      "cost": results["cost"]}, indent=2, ensure_ascii=False))
    print(f"\n落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB，"
          f"{neval['n']} 次评测，{n_needles_total} 根针", flush=True)


if __name__ == "__main__":
    main()
