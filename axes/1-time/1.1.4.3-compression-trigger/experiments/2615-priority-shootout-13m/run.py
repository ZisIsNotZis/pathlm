"""2615 · 三种压缩优先级对决（ticket 32 契约 N）—— 实验入口（纯推理）。

契约：同目录 `README.md`（判据线**逐字**抄录，未改一字）。
仪器：`tools/evalkit/retention.py`（本实验新增「优先级规则」一节：
`attention_stats` / `priority_drop_mask` / `drop_mask_keep` / `drop_mask_overlap` /
`attention_stats_summary` / `drop_mask_stats`，并让既有前向助手支持逐样本掩码）。
被测量模型：`2610-frozen-trunk-jumphead-13m/trunk.pt`（复用，运行时断言 sha256）。

跑六件事：
  ① 仪器自检：身份因果掩码 == 释放版前向（逐位）+ 针的配对控制臂 meta 逐字段相同 +
     契约 M 的 `prefix` 对照**在同一进程内**重算并与 M 记录比对（SCHEMA §9 复用的等值性）；
  ② 优先级规则的**活性**自检：三条规则在真实统计量下删出的集合确实不同（重叠度 < 1）；
  ③ 注意力质量的**分布**（契约要求：哪些位置被读得多）；
  ④ 依赖腿：同一窗口序上跑全部 24 个配置（每批一次统计前向 + 每配置一次前向）；
  ⑤ 保留腿：同一批针（契约 M 的器材，seed=3）+ 无针地板 + 逐针配对/分桶/难度控制；
  ⑥ 判据：逐 N 比较两条腿 + 噪声底 + 复现性 + 成本实测。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡）
  python3 run.py --smoke         # 冒烟：2 批 + 2 针批，实测步时/显存（**不外推时间**）
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full, batch                        # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel                # noqa: E402
from tools.evalkit.retention import (prefix_keep_mask, mask_stats,     # noqa: E402
                                     masked_logits, masked_bpc,
                                     attention_stats, attention_stats_summary,
                                     priority_drop_mask, drop_mask_keep,
                                     drop_mask_overlap, drop_mask_stats,
                                     phrase_needle_batch, needle_score,
                                     needle_summary, LN2)

M_RESULTS = ("axes/1-time/1.1.4.2-compression/experiments/"
             "2615-retention-retrieval-leg-13m/results.json")
RELEASE_ANCHOR_BPC = 1.483569   # 契约 M / 2614 记录的 13.5M 主干 clean bpc（同窗口序）


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


# ------------------------------------------------------------------ 臂定义


def build_arms(cfg: dict) -> list[dict]:
    """24 个配置（SCHEMA §17.8 的臂数 = 判据臂 9 + 基线 1 + 敏感性 14）。"""
    p = cfg["priority"]
    arms = [{"id": "full", "rule": None, "n_drop": 0, "k": None, "key": None,
             "candidates": None, "kind": "baseline"}]
    for N in p["n_drop"]:                                    # 判据臂：3 规则 × 3 档
        arms.append({"id": f"distance:N={N}", "rule": "distance", "n_drop": N,
                     "k": None, "key": None, "candidates": None, "kind": "primary"})
        arms.append({"id": f"lru:k={p['primary_k']}:N={N}", "rule": "lru", "n_drop": N,
                     "k": p["primary_k"], "key": None, "candidates": None,
                     "kind": "primary"})
        arms.append({"id": f"content:N={N}", "rule": "content", "n_drop": N,
                     "k": None, "key": None, "candidates": None, "kind": "primary"})
    for k in p["lru_k_sensitivity"]:                         # 敏感性：契约未定义的 k
        for N in p["n_drop"]:
            arms.append({"id": f"lru:k={k}:N={N}", "rule": "lru", "n_drop": N,
                         "k": k, "key": None, "candidates": None, "kind": "sensitivity_k"})
    if p["lastused_sensitivity"]:                            # 敏感性：LRU 语义的另一种落地
        for N in p["n_drop"]:
            arms.append({"id": f"lastused:N={N}", "rule": "lastused", "n_drop": N,
                         "k": None, "key": None, "candidates": None,
                         "kind": "sensitivity_rule"})
    for N in p["content_keynorm_n_drop"]:                    # 敏感性：content 的另一种读数
        arms.append({"id": f"content_keynorm:N={N}", "rule": "content", "n_drop": N,
                     "k": None, "key": "key_norm", "candidates": None,
                     "kind": "sensitivity_key"})
    return arms


def build_context_restricted_arms(cfg: dict, T: int, q0: int) -> list[dict]:
    """诊断臂：可删位置限定在**正文窗口** `[0, q0)`（针的线索块不参与压缩）。

    针的尾块 `[q0, T)` 是 harness 的提问块（部署里是在解码的 query，不是被压缩的
    KV）；若某条规则把提示线索删掉，召回会掉，却与「记忆被压掉」无关。这一组臂
    把候选池缩到正文窗口，回答「召回差是否来自删掉了线索块」。
    """
    cand = (torch.arange(T) < q0)   # 设备无关：`_rank` 内部搬到统计量所在设备
    out = []
    for rule, k in cfg["context_restricted"]["rules"]:
        for N in cfg["priority"]["n_drop"]:
            kid = f"{rule}" + (f":k={k}" if k else "")
            out.append({"id": f"{kid}:ctx:N={N}", "rule": rule, "n_drop": N, "k": k,
                        "key": None, "candidates": cand, "kind": "context_restricted"})
    return out


def build_pair_curve_arms(cfg: dict) -> list[dict]:
    """诊断臂：把预算换成「被删因果 (query,key) 对数」时，三条规则的完整损伤曲线。"""
    out = []
    for rule, k in cfg["pair_budget"]["rules"]:
        kid = f"{rule}" + (f":k={k}" if k else "")
        for N in cfg["pair_budget"]["n_grid"]:
            out.append({"id": f"pair:{kid}:N={N}", "rule": rule, "n_drop": N, "k": k,
                        "key": None, "candidates": None, "kind": "pair_curve"})
    return out


def arms_of(arms, kind):
    return [a for a in arms if a["kind"] == kind]


# ------------------------------------------------------------------ 一条腿：依赖


@torch.no_grad()
def dependency_run(model, eval_arr, arms, cfg, win_seed: int, n_batches: int,
                   log_f=None) -> dict:
    """同一窗口序上跑完整电池：每批一次统计前向 + 每配置一次前向。

    返回 `{"arms": {id: {...}}, "drops": {id: [B_total, T] bool}, "stats": {...}}`。
    每配置的 bpc 与其行级 CE 来自**同一次前向**（同位置集、同数据流）。
    """
    ecfg, T = cfg["eval"], cfg["model"]["seq_len"]
    bs = ecfg["batch_size"]
    device = ecfg["device"]
    recent_ks = tuple(cfg["priority"]["recent_ks"])
    ids = [a["id"] for a in arms]
    nats = {i: 0.0 for i in ids}
    rows = {i: torch.zeros(T - 1, dtype=torch.float64) for i in ids}
    n_tok = 0
    drop_cat = {i: [] for i in ids}
    stat_cat = {k: [] for k in ("mass_total", "mass_total_incl_diag", "cos_query",
                                "key_norm", "soft_recency")}
    stat_rec = {k: [] for k in recent_ks}
    gen = torch.Generator().manual_seed(win_seed)
    for bi in range(n_batches):
        x, _ = batch(eval_arr, bs, T, gen)
        x = x.to(device)
        st = attention_stats(model, x, recent_ks=recent_ks)
        for k in stat_cat:
            stat_cat[k].append(st[k].cpu())
        for k in recent_ks:
            stat_rec[k].append(st["mass_recent"][k].cpu())
        for a in arms:
            drop = None
            if a["rule"] is None:
                logits = masked_logits(model, x, None)
            else:
                drop = priority_drop_mask(st, a["n_drop"], a["rule"], k=a["k"],
                                          key=a["key"], candidates=a.get("candidates"))
                logits = masked_logits(model, x, drop_mask_keep(drop))
                drop_cat[a["id"]].append(drop.cpu())
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                 x[:, 1:].reshape(-1), reduction="none")
            nats[a["id"]] += float(ce.double().sum().item())
            rows[a["id"]] += ce.double().view(x.shape[0], -1).sum(0).cpu()
        n_tok += x.shape[0] * (T - 1)
        if log_f is not None and (bi % 4 == 0 or bi == n_batches - 1):
            log_f.write(json.dumps({
                "tag": "dep_batch", "batch": bi, "n_batches": n_batches,
                "win_seed": win_seed, "bpc_full": round(nats["full"] / n_tok / LN2, 6),
                "n_dropped_keys_mean": {
                    a["id"]: round(sum(float(v.sum()) for v in drop_cat[a["id"]])
                                   / max(n_batches * bs, 1), 3)
                    for a in arms if a["rule"] is not None},
            }, ensure_ascii=False) + "\n")
            log_f.flush()
    out = {}
    for a in arms:
        bpc = nats[a["id"]] / n_tok / LN2
        row_ce = (rows[a["id"]] / (n_batches * bs) / LN2)
        out[a["id"]] = {"bpc": bpc, "row_ce": row_ce.tolist(),
                        "row_ce_mean": float(row_ce.mean().item())}
    return {"arms": out,
            "drops": {i: (torch.cat(v) if v else None) for i, v in drop_cat.items()},
            "stats": {**stat_cat, "mass_recent": stat_rec},
            "n_windows": n_batches * bs, "n_tok": n_tok}


# ------------------------------------------------------------------ 一条腿：保留


def _cue_flags(keep, meta, phrase_len: int) -> list[bool]:
    """线索（`[mask, cue…]`）是否对答案行**全部可见**（被删 = 线索断）。"""
    out = []
    for b, rows in enumerate(meta):
        for m in rows:
            base = m["q_row"] - phrase_len
            out.append(keep is None or all(
                bool(keep[b, m["q_row"], base + 1 + t]) for t in range(phrase_len)))
    return out


@torch.no_grad()
def needle_run(model, eval_arr, arms, cfg, seed: int, n_batches: int,
               ncfg: dict, log_f=None, with_control: bool = True) -> dict:
    """同一批针上跑完整电池（主臂 / 地板臂配对；逐针命中 + 分桶 + 线索状态）。"""
    T = cfg["model"]["seq_len"]
    device = cfg["eval"]["device"]
    bs = ncfg["batch_size"]
    L, anchors, K = ncfg["phrase_len"], ncfg["anchors"], ncfg["n_needles"]
    mask_tok, n_real = cfg["data"]["mask_token"], cfg["data"]["n_real_tokens"]
    recent_ks = tuple(cfg["priority"]["recent_ks"])
    ids = [a["id"] for a in arms]
    recs = {i: [] for i in ids}
    cues = {i: [] for i in ids}
    control, control_cues = [], []
    gen = torch.Generator().manual_seed(seed)
    for bi in range(n_batches):
        state = gen.get_state()
        x, meta = phrase_needle_batch(bs, T, eval_arr, n_real, mask_tok, gen,
                                      anchors=anchors, anchor_frac=ncfg["anchor_frac"],
                                      n_needles=K, phrase_len=L)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(bs, T, eval_arr, n_real, mask_tok, gen,
                                        anchors=anchors, anchor_frac=ncfg["anchor_frac"],
                                        n_needles=K, phrase_len=L, plant=False)
        assert meta0 == meta, "配对控制臂的针位必须与主臂逐一相同"
        x = x.to(device)
        if with_control:
            lg0 = masked_logits(model, (x0.to(device)), None).cpu()
            control.extend(needle_score(lg0, meta, None, anchors=anchors, phrase_len=L))
            control_cues.extend(_cue_flags(None, meta, L))
        st = attention_stats(model, x, recent_ks=recent_ks)
        for a in arms:
            if a["rule"] is None:
                keep = None
                logits = masked_logits(model, x, None)
            else:
                drop = priority_drop_mask(st, a["n_drop"], a["rule"], k=a["k"],
                                          key=a["key"], candidates=a.get("candidates"))
                keep = drop_mask_keep(drop)
                logits = masked_logits(model, x, keep)
            recs[a["id"]].extend(needle_score(logits.cpu(), meta, keep,
                                              anchors=anchors, phrase_len=L))
            cues[a["id"]].extend(_cue_flags(keep, meta, L))
        if log_f is not None:
            log_f.write(json.dumps({
                "tag": "needle_batch", "batch": bi, "seed": seed,
                "recall_full": round(sum(p["hit"] for p in recs["full"])
                                     / max(len(recs["full"]), 1), 6),
                "recall_lru": round(sum(p["hit"] for p in recs[f"lru:k=64:N=32"])
                                    / max(len(recs[f"lru:k=64:N=32"]), 1), 6),
                "n_so_far": len(recs["full"])}, ensure_ascii=False) + "\n")
            log_f.flush()
    return {"records": recs, "cues": cues, "control": control,
            "control_cues": control_cues, "n_batches": n_batches, "bs": bs}


def merge_records(acc: dict, part: dict) -> dict:
    """把多次 needle_run（噪声底的不同 seed）的逐针记录并起来。"""
    out = {"records": {k: list(v) for k, v in acc["records"].items()},
           "cues": {k: list(v) for k, v in acc["cues"].items()},
           "control": list(acc["control"]), "control_cues": list(acc["control_cues"])}
    for k, v in part["records"].items():
        out["records"].setdefault(k, []).extend(v)
    for k, v in part["cues"].items():
        out["cues"].setdefault(k, []).extend(v)
    out["control"].extend(part["control"])
    out["control_cues"].extend(part["control_cues"])
    return out


def _rate(hits: list[int]) -> float | None:
    return round(sum(hits) / len(hits), 6) if hits else None


def retention_row(records: list[dict], cues: list[bool], control: list[dict],
                  control_cues: list[bool], base_recall: float) -> dict:
    """一条臂的保留腿汇总（总召回 / 分桶 / 配对 / 线索状态 / 难度控制）。"""
    s = needle_summary(records)
    hard_idx = [i for i, p in enumerate(control) if p["hit"] == 0]
    plant_kept = [p["hit"] for p in records if p["visible"]]
    plant_broken = [p["hit"] for p in records if not p["visible"]]
    cue_ok = [p["hit"] for p, c in zip(records, cues) if c]
    cue_bad = [p["hit"] for p, c in zip(records, cues) if not c]
    return {
        **s,
        "buckets": {k: {kk: (round(vv, 6) if kk == "recall" else vv)
                        for kk, vv in v.items()} for k, v in s["buckets"].items()},
        "recall_drop_vs_baseline": round(base_recall - s["recall"], 6),
        "n_plant_kept": len(plant_kept), "recall_plant_kept": _rate(plant_kept),
        "n_plant_broken": len(plant_broken), "recall_plant_broken": _rate(plant_broken),
        "n_cue_intact": len(cue_ok), "recall_cue_intact": _rate(cue_ok),
        "n_cue_broken": len(cue_bad), "recall_cue_broken": _rate(cue_bad),
        "n_hard": len(hard_idx),
        "hard_recall": _rate([records[i]["hit"] for i in hard_idx]),
        "n_control_cue_broken": sum(1 for c in control_cues if not c),
    }


# ------------------------------------------------------------------ 预算口径（诊断）


def _interp(xs: list[float], ys: list[float], x: float):
    """折线插值（xs 递增）；超出范围时按端点夹住并标记（不外推）。"""
    if not xs:
        return None, "no_data"
    if x <= xs[0]:
        return ys[0], "below_range_clamped"
    if x >= xs[-1]:
        return ys[-1], "above_range_clamped"
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1]), "interpolated"
    return ys[-1], "above_range_clamped"


def pair_budget_analysis(run: dict, rows_all: dict, cfg: dict,
                         base_bpc: float) -> dict:
    """把「删除预算」从**位置个数**换成**被删因果对数**后的损伤曲线（诊断）。

    契约 N 的预算定义是位置个数（三者相同，已如实核对）；但位置的位置分布决定了
    实际掉掉的可见性对数（`n_dropped_pairs` 三规则不同）⇒ 必须回答「看到的差异
    是不是只是对数差异」。对每条规则扫一条 (对数, 损伤) 曲线，再按对数插值出
    「同对数下 distance 的损伤」，给出 `gain_matched_pairs`。
    """
    base = base_bpc
    curves, table = {}, []
    for a in run["arms"]:
        if a == "full":
            continue
        rule = run["arm_spec"][a]
        key = f"{rule['rule']}" + (f":k={rule['k']}" if rule["k"] else "")
        st = drop_mask_stats(run["drops"][a])
        curves.setdefault(key, []).append(
            {"N": rule["n_drop"], "n_dropped_keys_mean": st["n_dropped_keys_mean"],
             "n_dropped_pairs_mean": st["n_dropped_pairs_mean"],
             "damage_bpc": round(run["arms"][a]["bpc"] - base, 6)})
    for k in curves:
        curves[k].sort(key=lambda r: r["n_dropped_pairs_mean"])
    ref = curves["distance"]
    xs = [r["n_dropped_pairs_mean"] for r in ref]
    ys = [r["damage_bpc"] for r in ref]
    for a, row in rows_all.items():
        if "activity" not in row or row["n_drop"] == 0:
            continue
        pairs = row["activity"]["n_dropped_pairs_mean"]
        dam, mode = _interp(xs, ys, pairs)
        table.append({"id": a, "n_dropped_keys_mean":
                      row["activity"]["n_dropped_keys_mean"],
                      "n_dropped_pairs_mean": round(pairs, 3),
                      "damage_bpc": row["damage_bpc"],
                      "distance_damage_at_same_pairs": None if dam is None
                      else round(dam, 6),
                      "gain_vs_distance_matched_pairs": None if dam is None
                      else round(dam - row["damage_bpc"], 6),
                      "match": mode})
    return {"curves": curves, "pair_matched": table, "reference_rule": "distance",
            "note": ("诊断口径（**不改判据**）：契约的预算 = 位置个数；这里额外按"
                     "**被删因果对数**对齐，回答「差异是不是只是对数差异」。"
                     "折线插值在 distance 曲线上取同对数点，超界夹住并标记。")}


# ------------------------------------------------------------------ 判据


def evaluate_criteria(rows: dict, cfg: dict) -> dict:
    """逐字执行契约 N 的判据（预登记；阈值与析取关系不做事后调整）。"""
    pp = cfg["criteria"]["recall_gain_pp"]
    band = cfg["criteria"]["damage_gain_bpc"]
    k = cfg["priority"]["primary_k"]
    base_recall = rows["full"]["retention"]["recall"]
    base_bpc = rows["full"]["bpc"]
    per_n = []
    for N in cfg["priority"]["n_drop"]:
        d = rows[f"distance:N={N}"]
        cands = {r: rows[f"{r}:" + (f"k={k}:" if r == "lru" else "") + f"N={N}"]
                 for r in ("lru", "content")}
        rec = {"N": N,
               "recall": {"distance": d["retention"]["recall"],
                          **{r: c["retention"]["recall"] for r, c in cands.items()}},
               "damage_bpc": {"distance": d["damage_bpc"],
                              **{r: c["damage_bpc"] for r, c in cands.items()}},
               "recall_gain_vs_distance": {r: round(c["retention"]["recall"]
                                                    - d["retention"]["recall"], 6)
                                           for r, c in cands.items()},
               "damage_gain_vs_distance": {r: round(d["damage_bpc"] - c["damage_bpc"], 6)
                                           for r, c in cands.items()},
               "distance_reverse_gain": {r: round(c["damage_bpc"] - d["damage_bpc"], 6)
                                         for r, c in cands.items()},
               "same_budget_n_dropped_keys": {
                   "distance": d["activity"]["n_dropped_keys_mean"],
                   **{r: c["activity"]["n_dropped_keys_mean"]
                      for r, c in cands.items()}},
               "n_dropped_pairs": {
                   "distance": d["activity"]["n_dropped_pairs_mean"],
                   **{r: c["activity"]["n_dropped_pairs_mean"]
                      for r, c in cands.items()}},
               "recall_ceiling_full": base_recall,
               }
        rec["passes_recall_leg"] = any(g >= pp for g in
                                       rec["recall_gain_vs_distance"].values())
        rec["passes_dependency_leg"] = any(g >= band for g in
                                           rec["damage_gain_vs_distance"].values())
        # 算术对账（SCHEMA §12.1）：召回判据在该 N 上是否**可达**
        rec["recall_criterion_arithmetically_reachable"] = bool(
            (d["retention"]["recall"] + pp) <= base_recall + 1e-9)
        per_n.append(rec)
    informative = any(p["passes_recall_leg"] or p["passes_dependency_leg"]
                      for p in per_n)
    return {"recall_threshold_pp": pp, "damage_threshold_bpc": band,
            "baseline_bpc_full": base_bpc, "baseline_recall_full": base_recall,
            "primary_k": k, "per_N": per_n,
            "verdict": "priority_rules_informative" if informative
                       else "no_visible_benefit",
            "verdict_text": ("优先级规则有信息量（前提成立，值得往下做）" if informative else
                             "优先级规则不带来可见收益（LRU 前提在该规模被否证）"),
            "primary_only_note": ("敏感性臂（lru 的其他 k、lastused、content(key_norm)）"
                                 "不参与判据，只用于检验结论对契约留白的稳健性")}


def overlap_report(drops: dict, cfg: dict) -> list[dict]:
    """三种规则实际删掉集合的重叠度（契约要求；逐窗口 → 均值）。"""
    k = cfg["priority"]["primary_k"]
    out = []
    for N in cfg["priority"]["n_drop"]:
        pairs = [("distance", "lru:k=%d" % k), ("distance", "content"),
                 ("lru:k=%d" % k, "content"), ("distance", "lastused")]
        for a, b in pairs:
            ia, ib = f"{a}:N={N}", f"{b}:N={N}"
            if ia in drops and ib in drops:
                ov = drop_mask_overlap(drops[ia], drops[ib])
                out.append({"N": N, "a": ia, "b": ib, **ov})
    return out


# ------------------------------------------------------------------ 主流程


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
    ecfg, ncfg = cfg["eval"], cfg["needle"]
    if args.smoke:
        ecfg["n_batches"] = 2
        ncfg["n_batches"] = 2
        cfg["reproducibility"]["rerun"] = False
        cfg["noise_floor"]["window_seeds"] = [3]
        cfg["noise_floor"]["needle_seeds"] = [3]
        cfg["noise_floor"]["needle_batches_per_seed"] = 1
    os.makedirs(args.out_dir, exist_ok=True)

    dev, T = ecfg["device"], cfg["model"]["seq_len"]
    _, eval_arr, _ = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                      os.path.join(ROOT, cfg["data"]["path"]))
    model, ck = load_trunk(cfg, dev)
    arms = build_arms(cfg)
    L, anchors = ncfg["phrase_len"], ncfg["anchors"]
    recent_ks = tuple(cfg["priority"]["recent_ks"])
    qmin_long, qmin_needle = cfg["diagnostics"]["query_min_long"], \
        cfg["diagnostics"]["query_min_needle"]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_fwd = {"n": 0}
    log_f = open(os.path.join(args.out_dir, "retention_log.jsonl"), "w")

    # ---------- ① 仪器自检 ----------
    bpc_release = masked_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                             batch_size=ecfg["batch_size"],
                             window_seed=ecfg["window_seed"], device=dev)
    n_fwd["n"] += ecfg["n_batches"]
    ident = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    bpc_ident = masked_bpc(model, eval_arr, ident, n_batches=ecfg["n_batches"],
                           batch_size=ecfg["batch_size"],
                           window_seed=ecfg["window_seed"], device=dev)
    n_fwd["n"] += ecfg["n_batches"]
    instrument = {"smoke": bool(args.smoke),
                  "smoke_note": ("冒烟（2 批）下 bpc/复用对照不可判读，只用于实测步时与显存"
                                 "（SCHEMA §17.5：不外推时间）" if args.smoke else
                                 "全电池：bpc 与复用对照按 24×16 窗口口径判读"),
                  "bpc_full_release_path": round(bpc_release, 6),
                  "bpc_full_identity_mask": round(bpc_ident, 6),
                  "bitwise_equal": bpc_release == bpc_ident,
                  "abs_diff_vs_anchor_2610_M": round(abs(bpc_release
                                                         - RELEASE_ANCHOR_BPC), 6),
                  "release_anchor_bpc": RELEASE_ANCHOR_BPC}
    print(f"[自检] 释放版 {bpc_release:.6f} / 身份掩码 {bpc_ident:.6f} "
          f"(逐位 {'=' if instrument['bitwise_equal'] else '≠'})，"
          f"vs M/2614 锚 {RELEASE_ANCHOR_BPC}", flush=True)

    # 契约 M 的复用对照：同一进程内重算 prefix N∈{8,32,64}，与 M 记录比对（SCHEMA §9）
    reuse = {"m_results_path": M_RESULTS, "available": os.path.exists(
        os.path.join(ROOT, M_RESULTS)), "points": []}
    if reuse["available"]:
        m = json.load(open(os.path.join(ROOT, M_RESULTS)))
        mck = m["checkpoint"]
        reuse["m_checkpoint_sha256"] = mck["sha256"]
        reuse["checkpoint_same"] = mck["sha256"] == ck["sha256"]
        mprefix = {r["id"]: r["bpc"] for r in m["configs"] if r["kind"] == "prefix"}
        for N in cfg["priority"]["n_drop"]:
            got = masked_bpc(model, eval_arr, prefix_keep_mask(T, N, dev),
                             n_batches=ecfg["n_batches"],
                             batch_size=ecfg["batch_size"],
                             window_seed=ecfg["window_seed"], device=dev)
            n_fwd["n"] += ecfg["n_batches"]
            want = mprefix.get(f"prefix:N={N}")
            reuse["points"].append({"config": f"prefix:N={N}", "m_recorded_bpc": want,
                                    "this_run_bpc": round(got, 6),
                                    "abs_diff": None if want is None
                                    else round(abs(got - want), 9)})
        reuse["max_abs_diff"] = max(p["abs_diff"] for p in reuse["points"]
                                    if p["abs_diff"] is not None)
        reuse["pass_1e_6"] = reuse["max_abs_diff"] < 1e-6
        print(f"[复用对照] 契约 M 的 prefix 三点最大 |Δbpc| = {reuse['max_abs_diff']:.2e}"
              f" ⇒ {'一致' if reuse['pass_1e_6'] else '不一致'}", flush=True)

    # ---------- ④ 依赖腿 ----------
    t_dep = time.time()
    dep = dependency_run(model, eval_arr, arms, cfg, ecfg["window_seed"],
                         ecfg["n_batches"], log_f)
    n_fwd["n"] += ecfg["n_batches"] * (len(arms) + 1)
    wall_dep = time.time() - t_dep
    # 行级 CE 的均值必须等于 bpc（算术对账：同一批 token 的平均）
    row_arith = {i: round(abs(dep["arms"][i]["row_ce_mean"] - dep["arms"][i]["bpc"]), 12)
                 for i in dep["arms"]}
    assert max(row_arith.values()) < 1e-9, row_arith
    stat_cat = {k: torch.cat(v) for k, v in dep["stats"].items() if k != "mass_recent"}
    stat_rec = {k: torch.cat(v) for k, v in dep["stats"]["mass_recent"].items()}
    stats_for_summary = {**stat_cat, "mass_recent": stat_rec,
                         "meta": {"n_windows": dep["n_windows"], "n_layers":
                                  cfg["model"]["n_layers"], "T": T,
                                  "recent_ks": list(recent_ks),
                                  "layer_mean_mass_recent_bins": {}}}
    # 共享掩码路径 vs 逐样本掩码路径：distance 臂的 bpc 必须一致（同一删除集、同一数据流）
    if reuse["available"]:
        reuse["shared_vs_per_sample_masks"] = [
            {"config": f"distance:N={N}", "shared_mask_bpc":
             next(p["this_run_bpc"] for p in reuse["points"]
                  if p["config"] == f"prefix:N={N}"),
             "per_sample_mask_bpc": round(dep["arms"][f"distance:N={N}"]["bpc"], 6),
             "abs_diff": round(abs(next(p["this_run_bpc"] for p in reuse["points"]
                                        if p["config"] == f"prefix:N={N}")
                                   - dep["arms"][f"distance:N={N}"]["bpc"]), 9)}
            for N in cfg["priority"]["n_drop"]]
        reuse["max_shared_vs_per_sample_diff"] = max(
            p["abs_diff"] for p in reuse["shared_vs_per_sample_masks"])
    dist = attention_stats_summary(stats_for_summary, ks=recent_ks)
    print(f"[依赖腿] {len(arms)} 配置 / {dep['n_windows']} 窗口，"
          f"墙钟 {wall_dep:.1f}s；注意力质量最大位置 = "
          f"{[h['pos'] for h in dist['mass_total']['top_positions'][:5]]}", flush=True)

    # 交付标准 §2 的 C 组：逐配置汇总
    dep_rows = {}
    for a in arms:
        r = dep["arms"][a["id"]]
        rows = r["row_ce"]
        mean_from = lambda lo: sum(rows[lo:]) / (T - 1 - lo)
        base_rows = dep["arms"]["full"]["row_ce"]
        base_mean_from = lambda lo: sum(base_rows[lo:]) / (T - 1 - lo)
        act = (drop_mask_stats(dep["drops"][a["id"]]) if a["rule"] is not None
               else {"n_dropped_keys_mean": 0.0, "n_dropped_pairs_mean": 0.0,
                     "dropped_pair_frac_mean": 0.0})
        dep_rows[a["id"]] = {
            "bpc": round(r["bpc"], 6),
            "delta_bpc": round(dep["arms"]["full"]["bpc"] - r["bpc"], 6),
            "damage_bpc": round(r["bpc"] - dep["arms"]["full"]["bpc"], 6),
            "damage_qmin128": round(mean_from(qmin_long) - base_mean_from(qmin_long), 6),
            "damage_qmin200": round(mean_from(qmin_needle)
                                    - base_mean_from(qmin_needle), 6),
            "row_ce_mean_check": row_arith[a["id"]],
            "activity": act}

    # 判据臂的活性 + 重叠度（契约要求）
    ov = overlap_report(dep["drops"], cfg)
    k0 = cfg["priority"]["primary_k"]
    live_pairs = [o for o in ov if o["N"] in cfg["priority"]["n_drop"]]
    rules_live = all(o["mean_jaccard"] < 1.0 for o in live_pairs
                     if o["b"].startswith(("lru", "content")))
    intervention_dep = {
        "what": ("评测期只改**删除哪些位置**（优先级规则）；删除预算（位置个数）逐档相同；"
                 "不改权重、不训练、不新增参数"),
        "params_added": 0, "weights_changed": False,
        "rules_live_sets_differ": rules_live,
        "max_mean_jaccard_primary": max(o["mean_jaccard"] for o in live_pairs),
        "n_dropped_keys_mean": {a["id"]: dep_rows[a["id"]]["activity"]["n_dropped_keys_mean"]
                                for a in arms if a["rule"] is not None},
        "activity_note": ("活性量 = 每个配置的逐窗口被删位置数 / 被删因果 (query,key) 对数"
                          "（`drop_mask_stats`，与 `mask_stats` 同一口径的闭式）；本实验无训练步，"
                          "等价物是**每批评定即写** retention_log.jsonl（SCHEMA §17.4）。"),
    }

    # ---------- ④b 诊断：预算换成「被删因果对数」 ----------
    pc_arms = build_pair_curve_arms(cfg)
    pc_run = dependency_run(model, eval_arr, pc_arms, cfg, ecfg["window_seed"],
                            ecfg["n_batches"])
    pc_run["arm_spec"] = {a["id"]: a for a in pc_arms}
    n_fwd["n"] += ecfg["n_batches"] * (len(pc_arms) + 1)
    print(f"[预算诊断] {len(pc_arms)} 个曲线点（N 网格 "
          f"{cfg['pair_budget']['n_grid']}）跑完", flush=True)

    # ---------- ⑤ 保留腿 ----------
    t_needle = time.time()
    nd = needle_run(model, eval_arr, arms, cfg, ncfg["seed"], ncfg["n_batches"], ncfg,
                    log_f)
    n_fwd["n"] += ncfg["n_batches"] * (len(arms) + 1)
    wall_needle = time.time() - t_needle
    base_rec = needle_summary(nd["records"]["full"])
    q0_ctx = T - ncfg["n_needles"] * (L + 1)
    ctx_arms = build_context_restricted_arms(cfg, T, q0_ctx)
    ctx_run = needle_run(model, eval_arr, ctx_arms, cfg, ncfg["seed"], ncfg["n_batches"],
                         ncfg, with_control=False)
    n_fwd["n"] += ncfg["n_batches"] * (len(ctx_arms) + 1)
    print(f"[预算诊断] 正文限制臂 {len(ctx_arms)} 个（候选池 [0,{q0_ctx})）跑完",
          flush=True)
    floor_rec = needle_summary(nd["control"])
    floor = {"recall": floor_rec["recall"],
             "buckets": {k2: v["recall"] for k2, v in floor_rec["buckets"].items()},
             "n": floor_rec["n"],
             "note": ("plant=False，同一 generator 状态 ⇒ 针位与线索与主臂逐一配对；"
                      "= 检索腿地板「记忆不存在」")}
    print(f"[保留腿] 全保留召回 {base_rec['recall']:.4f} (n={base_rec['n']})，"
          f"无针地板 {floor_rec['recall']:.4f}，墙钟 {wall_needle:.1f}s", flush=True)

    hard_idx = [i for i, p in enumerate(nd["control"]) if p["hit"] == 0]
    retention_rows = {a["id"]: retention_row(nd["records"][a["id"]], nd["cues"][a["id"]],
                                             nd["control"], nd["control_cues"],
                                             base_rec["recall"]) for a in arms}
    # 契约判据里「保留腿召回」的三种口径：总体 / 只算被该规则删掉的针（配对）/ 只算难度针
    for a in arms:
        r = retention_rows[a["id"]]
        if a["rule"] is None:
            continue
        deleted = [i for i, p in enumerate(nd["records"][a["id"]]) if not p["visible"]]
        r["n_deleted_needles"] = len(deleted)
        r["paired_baseline_recall_deleted"] = _rate([nd["records"]["full"][i]["hit"]
                                                     for i in deleted])
        r["paired_recall_deleted"] = _rate([nd["records"][a["id"]][i]["hit"]
                                            for i in deleted])
        r["paired_drop_deleted"] = (
            None if r["paired_baseline_recall_deleted"] is None else
            round(r["paired_baseline_recall_deleted"] - r["paired_recall_deleted"], 6))
    # 三条判据臂在同一 N 上「都被删掉」的针（共同可见子集）——把「删得多」与「删得坏」分开
    common = {}
    for N in cfg["priority"]["n_drop"]:
        cids = [f"distance:N={N}", f"lru:k={k0}:N={N}", f"content:N={N}"]
        kept = [i for i in range(base_rec["n"])
                if all(nd["records"][c][i]["visible"] for c in cids)]
        common[f"N={N}"] = {
            "n": len(kept),
            "recall": {c: _rate([nd["records"][c][i]["hit"] for i in kept])
                       for c in cids},
            "note": ("三条规则在**同一批针**上都判「可见」的子集（同一位置集、同一任务）"
                     "⇒ 比较的是「谁把记忆弄坏了」，不是「谁删掉的针多」"),
        }

    # ---------- 逐臂汇总（两条腿合并） ----------
    rows_all = {}
    for a in arms:
        r = {**a, **dep_rows[a["id"]]}
        r["retention"] = retention_rows[a["id"]]
        rows_all[a["id"]] = r

    # 限正文的臂只有保留腿（依赖腿没有 harness 尾块，限制无意义）
    de0 = set()
    for a in ctx_arms:
        ret = retention_row(ctx_run["records"][a["id"]], ctx_run["cues"][a["id"]],
                            ctx_run["control"], ctx_run["control_cues"],
                            base_rec["recall"])
        deleted = [i for i, p in enumerate(ctx_run["records"][a["id"]])
                   if not p["visible"]]
        ret["n_deleted_needles"] = len(deleted)
        ret["paired_baseline_recall_deleted"] = _rate(
            [nd["records"]["full"][i]["hit"] for i in deleted])
        ret["paired_recall_deleted"] = _rate([ctx_run["records"][a["id"]][i]["hit"]
                                              for i in deleted])
        rows_all[a["id"]] = {**{k: v for k, v in a.items() if k != "candidates"},
                             "candidates": f"[0, {q0_ctx})",
                             "retention": ret, "dependency": "not_measured"}

    # ---------- ⑥ 判据 + 敏感性 ----------
    pair_budget = pair_budget_analysis(pc_run, rows_all, cfg,
                                      dep["arms"]["full"]["bpc"])
    criteria = evaluate_criteria(rows_all, cfg)
    # 对账义务（SCHEMA §12）：契约判据按**位置数**对齐；但删同样多的位置 ≠ 掉同样多的
    # 可见性对数 ⇒ 把「按对数重对齐后」的结论与判据原文放在同一个文件里。
    matched = {p["id"]: p["gain_vs_distance_matched_pairs"]
               for p in pair_budget["pair_matched"]
               if p["id"].split(":")[0] in ("lru", "content", "lastused")}
    criteria["reconciliation_errata"] = {
        "verdict_after_reconciliation": (
            "informative_stands" if any((v is not None and v >= criteria[
                "damage_threshold_bpc"]) for v in matched.values())
            else "retracted_no_visible_benefit"),
        "matched_pair_gains_vs_distance": matched,
        "threshold_bpc": criteria["damage_threshold_bpc"],
        "why": ("删除**位置数**相同 ≠ 删除**可见性对数**相同（位置 j 的代价是 T−1−j 对）："
                "N=8 时 distance 掉 2012 对、lru 1562 对、content 1859 对。按同一对数重对齐后"
                "lru 由正转负、content 落回噪声带 ⇒ 契约字面达成的优势**不足以**支撑「LRU 前提"
                "成立」（SCHEMA §12；详见 analysis.md §6.4 的对账结论）。判据线本身未改。"),
    }
    sens = []
    for a in arms:
        if a["kind"] not in ("sensitivity_k", "sensitivity_rule", "sensitivity_key"):
            continue
        d = rows_all[f"distance:N={a['n_drop']}"]
        ovd = drop_mask_overlap(dep["drops"][f"distance:N={a['n_drop']}"],
                                dep["drops"][a["id"]])
        sens.append({"id": a["id"],
                     "damage_bpc": rows_all[a["id"]]["damage_bpc"],
                     "recall": rows_all[a["id"]]["retention"]["recall"],
                     "damage_gain_vs_distance": round(d["damage_bpc"]
                                                      - rows_all[a["id"]]["damage_bpc"], 6),
                     "recall_gain_vs_distance": round(
                         rows_all[a["id"]]["retention"]["recall"]
                         - d["retention"]["recall"], 6),
                     "mean_jaccard_vs_distance": ovd["mean_jaccard"]})

    # ---------- 噪声底 ----------
    nf = {"needle": cfg["noise_floor"]["needle_seeds"],
          "window": cfg["noise_floor"]["window_seeds"], "configs": []}
    for cid in cfg["noise_floor"]["dependency_configs"]:
        arm = [a for a in arms if a["id"] == cid]
        if not arm:
            continue
        dam = []
        for sd in nf["window"]:
            run = dependency_run(model, eval_arr, [arms[0], arm[0]], cfg, sd,
                                 ecfg["n_batches"])
            n_fwd["n"] += ecfg["n_batches"] * 3
            dam.append(run["arms"][cid]["bpc"] - run["arms"]["full"]["bpc"])
        mean = sum(dam) / len(dam)
        var = sum((v - mean) ** 2 for v in dam) / max(len(dam) - 1, 1)
        nf["configs"].append({"id": cid, "leg": "dependency", "n_seeds": len(dam),
                              "damage_bpc": [round(v, 6) for v in dam],
                              "mean_damage_bpc": round(mean, 6),
                              "std_damage_bpc": round(var ** 0.5, 6)})
    sub = [a for a in arms if a["id"] in cfg["noise_floor"]["needle_configs"]] + [arms[0]]
    for sd in nf["needle"]:
        part = needle_run(model, eval_arr, sub, cfg, sd,
                          cfg["noise_floor"]["needle_batches_per_seed"], ncfg,
                          with_control=False)
        n_fwd["n"] += cfg["noise_floor"]["needle_batches_per_seed"] * (len(sub) + 1)
        for a in sub:
            s = needle_summary(part["records"][a["id"]])
            e = next((c for c in nf["configs"] if c["id"] == a["id"]
                      and c["leg"] == "needle"), None)
            if e is None:
                e = {"id": a["id"], "leg": "needle", "n_seeds": 0, "recall": [],
                     "mean_recall": 0.0, "std_recall": 0.0, "n_per_seed": s["n"]}
                nf["configs"].append(e)
            e["recall"].append(round(s["recall"], 6))
            e["n_seeds"] = len(e["recall"])
            m = sum(e["recall"]) / len(e["recall"])
            v = sum((x - m) ** 2 for x in e["recall"]) / max(len(e["recall"]) - 1, 1)
            e["mean_recall"], e["std_recall"] = round(m, 6), round(v ** 0.5, 6)
    nf["max_std_damage_bpc"] = max((c["std_damage_bpc"] for c in nf["configs"]
                                    if c["leg"] == "dependency"), default=None)
    nf["max_std_recall"] = max((c["std_recall"] for c in nf["configs"]
                                if c["leg"] == "needle"), default=None)

    # ---------- 复现性 ----------
    repro = {"threshold": cfg["reproducibility"]["threshold"],
             "rerun": bool(cfg["reproducibility"]["rerun"]), "per_config": []}
    if cfg["reproducibility"]["rerun"]:
        picks = [f"distance:N={cfg['priority']['n_drop'][-1]}",
                 f"lru:k={k0}:N={cfg['priority']['n_drop'][-1]}",
                 f"content:N={cfg['priority']['n_drop'][-1]}"]
        for cid in picks:
            arm = [a for a in arms if a["id"] == cid][0]
            run = dependency_run(model, eval_arr, [arms[0], arm], cfg,
                                 ecfg["window_seed"], ecfg["n_batches"])
            n_fwd["n"] += ecfg["n_batches"] * 3
            d2 = run["arms"][cid]["bpc"] - run["arms"]["full"]["bpc"]
            repro["per_config"].append(
                {"config": cid, "damage_run1": rows_all[cid]["damage_bpc"],
                 "damage_run2": round(d2, 6),
                 "abs_diff": round(abs(d2 - rows_all[cid]["damage_bpc"]), 9)})
        repro["max_abs_damage_diff"] = max(p["abs_diff"] for p in repro["per_config"])
        repro["pass"] = repro["max_abs_damage_diff"] < repro["threshold"]

    # ---------- 落盘 ----------
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    intervention = {**intervention_dep,
                    "n_windows_dependency": dep["n_windows"],
                    "n_needles_per_arm": base_rec["n"],
                    "baseline_recall_full": base_rec["recall"],
                    "no_needle_floor_recall": floor_rec["recall"],
                    "retention_gap": round(base_rec["recall"] - floor_rec["recall"], 6),
                    "alive": bool(intervention_dep["rules_live_sets_differ"]
                                  and base_rec["recall"] > floor_rec["recall"])}
    results = {
        "contract": ("axes/1-time/1.1.4.3-compression-trigger/experiments/"
                     "2615-priority-shootout-13m/README.md"),
        "criteria_verbatim": ("判据：若 lru 或 content 在同等删除预算下保留腿召回高 ≥10pp "
                              "或依赖腿伤害低 ≥0.014 ⇒ 优先级规则有信息量（前提成立）；"
                              "若三者差异都低于阈值 ⇒ 优先级规则无可见收益（LRU 前提在该规模"
                              "被否证）。必须报注意力质量的分布与三种规则实际删掉集合的重叠度。"),
        "checkpoint": ck, "config": cfg, "arms": arms,
        "instrument": instrument, "reuse_check_contract_M": reuse,
        "intervention": intervention,
        "attention_quality": {**dist, "overlap_deleted_sets": ov},
        "baseline": {"bpc": round(dep["arms"]["full"]["bpc"], 6),
                     "retention": base_rec},
        "no_needle_control": floor,
        "hard_needle_subset": {
            "n_hard": len(hard_idx),
            "frac_hard": round(len(hard_idx) / max(len(nd["control"]), 1), 6),
            "baseline_recall_hard": _rate([nd["records"]["full"][i]["hit"]
                                           for i in hard_idx]),
            "control_recall_hard": 0.0,
            "per_config": {a["id"]: retention_rows[a["id"]]["hard_recall"]
                           for a in arms},
            "note": ("难度控制子集 = 无针控制臂答不出的针（答案不能从线索先验得到）；"
                     "仅作诊断，不作为主判据口径（SCHEMA §17.10 的「带病构造不得判读」）。")},
        "common_visible_subset": common,
        "arms_table": rows_all,
        "criteria": criteria,
        "pair_budget_diagnostic": pair_budget,
        "context_restricted_needle_leg": {
            "candidates": f"[0, {q0_ctx})（针的线索块 [q0, T) 不参与压缩）",
            "q0": q0_ctx,
            "rows": {a["id"]: rows_all[a["id"]]["retention"] for a in ctx_arms},
            "note": ("诊断：若某规则的召回损失主要来自「删掉了 harness 的线索块」，"
                     "把它排除后差距应显著缩小。")},
        "sensitivity": sens,
        "noise_floor": nf,
        "reproducibility": repro,
        "code": {"run_py_sha256": sha256_file(os.path.abspath(__file__)),
                 "retention_py_sha256": sha256_file(
                     os.path.join(ROOT, "tools", "evalkit", "retention.py")),
                 "tests_sha256": sha256_file(
                     os.path.join(ROOT, "tests", "test_retention.py"))},
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "dependency_wall_seconds": round(wall_dep, 2),
                 "needle_wall_seconds": round(wall_needle, 2),
                 "peak_mem_mib": round(peak, 1), "n_batches": ecfg["n_batches"],
                 "n_needle_batches": ncfg["n_batches"], "n_configs": len(arms),
                 "n_forwards": n_fwd["n"], "ms_per_forward": round(
                     1000 * wall / max(n_fwd["n"], 1), 3),
                 "n_windows": dep["n_windows"], "n_needles_per_arm": base_rec["n"],
                 "n_configs_pair_curve": len(pc_arms),
                 "n_configs_context_restricted": len(ctx_arms),
                 "n_arm_evaluations_dependency_leg": len(arms) + len(pc_arms),
                 "n_arm_evaluations_needle_leg": len(arms) + len(ctx_arms),
                 "n_reruns": len(repro["per_config"]),
                 "device": torch.cuda.get_device_name(0)},
    }
    log_f.close()
    results["cost"]["n_log_lines"] = sum(1 for _ in open(
        os.path.join(args.out_dir, "retention_log.jsonl")))
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"criteria": {"verdict": criteria["verdict"],
                                   "per_N": criteria["per_N"]},
                      "cost": results["cost"]}, indent=2, ensure_ascii=False))
    print(f"\n落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB，"
          f"{n_fwd['n']} 次前向，{len(arms)} 个配置", flush=True)


if __name__ == "__main__":
    main()
