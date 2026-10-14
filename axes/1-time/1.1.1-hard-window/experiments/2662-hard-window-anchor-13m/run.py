#!/usr/bin/env python3
"""2662-hard-window-anchor-13m · 评测入口（纯推理）：锚点通道自包含复算。

契约：同目录 `README.md`（§4 判据线预登记，未改一字）。本入口做：

  ① 自检：逐臂掩码非退化（visible_pairs == 期望）+ 参数量；**因果性**（只改后半段 token
     ⇒ 前半段 logits 逐位不变）；锚点臂掩码反事实探针；
  ② 质量腿（价格）：每臂在其训练可见性下的 clean bpc；`price_bpc = bpc(win16a) − bpc(win16)`
     （同 seed）；复用臂 bpc 与 2634 落盘值逐位比对；本格自测带 σ_cell = 本实验换种子极差；
  ③ **锚点召回腿（主判据）**：2634 needle v3 仪器（真实文本短语针），归属桶一律 `anchors=8`
     ⇒ 逐臂报 `anchor` / `beyond` / `window` 桶「植入 vs 配对无针地板」+ Newcombe 两比例差 CI
     + 按针配对 bootstrap CI；同索引硬子集上的跨臂比较；
  ④ 毁源敏感度（负对照）：两变体——变体 A `far_all` 被毁 `[0,64)`；变体 B `beyond_anchor`
     被毁 `[8,64)`（保住锚点槽 `[0,8)`）。锚点臂对变体 A **必然**敏感（锚点槽在被毁区内，
     `premise_holds=False`），对变体 B 应逐位不敏感 ⇒「锚点不搬运锚点槽之外的远处内容」；
  ⑤ 代价：每位置等效 KV 字节（fp16）= 12288×k_eff；活性（掩码可见对数 + 反事实探针 + 训练日志）。

用法：
  python3 run.py            # 全电池
  python3 run.py --smoke    # 冒烟：少批
  python3 run.py --config /tmp/xxx.json --out-dir /tmp/yyy   # CPU dry-run 等
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             window_anchor_keep_mask, window_mask_probe)
from tools.evalkit.retention import (needle_recall, needle_summary,  # noqa: E402
                                     phrase_needle_batch)
from tools.evalkit.carry_channel import two_proportion              # noqa: E402
import realtext_probe as rp                                        # noqa: E402
import needle_stats as ns                                          # noqa: E402

# 跨格参考带（C1 裁决：本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用）。
CROSS_CELL_BAND = 0.014
#: 期望的掩码可见对数（window_anchor_keep_mask(256, w, a)）。
EXPECTED_VISIBLE = {(16, 0): 3976, (16, 8): 5868}
#: 2634 落盘的 win16 clean bpc（复用臂必须逐位复现；判据⑥）。
REF_2634_WIN16_BPC = {"win16_s0": 1.62811, "win16_s1": 1.629093}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _strict(obj):
    """非有限浮点（NaN/Inf）→ None，保证落盘严格 JSON（禁裸 NaN/Infinity）。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


def model_kwargs(spec: dict, mcfg: dict) -> dict:
    kw = dict(mcfg)
    for k in ("note",):
        kw.pop(k, None)
    kw["train_window"] = int(spec["window"]) if spec["kind"] != "full" else 0
    kw["train_anchors"] = int(spec.get("anchors", 0)) if spec["kind"] != "full" else 0
    kw["carry_rule"] = None if spec["rule"] in (None, "none") else spec["rule"]
    return kw


def keep_mask_for(spec: dict, T: int, dev: str):
    if spec["kind"] == "full":
        return None
    return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                   device=dev)


def load_arm(spec: dict, mcfg: dict, vocab: int, dev: str):
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{spec['name']}: checkpoint sha256 mismatch:\n "
                         f"want {want}\n got  {got}")
    cfg = DepthARConfig(**model_kwargs(spec, mcfg))
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "kind": spec["kind"], "window": spec["window"],
                   "anchors": spec.get("anchors", 0), "rule": spec["rule"],
                   "seed": spec["seed"], "reused_from": spec.get("reused_from"),
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """**严格因果性探针（必须不变项）**：只改后半段 token，前半段 logits 必须逐位不变。"""
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            a = rp.masked_logits(model, x, keep_mask).float()
            b = rp.masked_logits(model, x2, keep_mask).float()
    finally:
        model.train(was_training)
    d = (a[:, :half] - b[:, :half]).abs().max()
    return {"perturbed_from": half,
            "max_abs_logit_diff_prefix": round(float(d), 9),
            "causal_no_future_leak": bool(float(d) == 0.0),
            "changed_suffix_max_abs_logit_diff": round(
                float((a[:, half:] - b[:, half:]).abs().max()), 6),
            "note": ("只改后半段 token ⇒ 前半段 logits 逐位不变 ⇔ 严格因果（无未来泄漏）；"
                     "后半段必须真的变（否则探针没抓力）")}


def bucket_records(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["bucket"], []).append(r)
    return out


def paired_reading(planted: list[int], control: list[int]) -> dict:
    """「植入 vs 配对无针地板」：Newcombe 两比例差 CI + 按针配对 bootstrap CI。"""
    if not planted:
        return {"n": 0, "note": "空桶"}
    kp, kc, n = sum(planted), sum(control), len(planted)
    tp = two_proportion(kc, n, kp, n)             # diff = 植入 − 地板
    pb = ns.paired_bootstrap_diff_pp(planted, control)
    lo = float(tp["diff_ci_95"][0])
    return {"n": n, "planted_hits": int(kp), "control_hits": int(kc),
            "planted_recall": round(kp / n, 6), "control_recall": round(kc / n, 6),
            "delta_pp": round(100 * (kp - kc) / n, 3),
            "newcombe_diff_ci95": tp["diff_ci_95"],
            "newcombe_diff_ci95_pp": [round(100 * v, 3) for v in tp["diff_ci_95"]],
            "newcombe_ci_overlap": tp["ci_overlap"], "z": tp["z"], "p_value": tp["p_value"],
            "paired_bootstrap_pp": pb["diff_ci_95_pp"],
            "paired_bootstrap_mean_pp": pb["mean_diff_pp"],
            "passes_newcombe": bool(lo > 0),
            "passes_paired_bootstrap": bool(pb["passes_lower_gt_0"]),
            "passes_both": bool(lo > 0 and pb["passes_lower_gt_0"]),
            "direction_note": "正 = 植入高于地板（判据②要的方向）"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke", "eval") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    if args.smoke:
        for k in ("bpc", "probe"):
            cfg["eval"][k]["n_batches"] = 2
            cfg["eval"][k]["batch_size"] = 4
        cfg["needle"]["n_batches"] = 2
        cfg["needle"]["batch_size"] = 4
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    eb, ep2 = cfg["eval"]["bpc"], cfg["eval"]["probe"]
    vcfg = cfg["needle"]
    window, anchors = int(cfg["model"]["train_window"]), 8
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]
    unit = cfg["model"]["n_layers"] * 2 * cfg["model"]["d_model"] * 2   # fp16 每位置 KV 字节
    log: list[dict] = []
    t0 = time.time()
    on_cuda = str(dev).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段做评测（路径锚到仓库根）")
    _, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                          os.path.join(ROOT, "data/enwik8_full.npz"))
    assert vocab == cfg["data"]["vocab_size"], f"vocab {vocab} != {cfg['data']['vocab_size']}"

    note("加载各臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:12]}… kind={spec['kind']} "
             f"w={spec['window']} anchors={spec['anchors']} seed={spec['seed']} "
             f"params={ck[spec['name']]['n_params']}")

    # ---------------- ① 自检：掩码非退化 + 参数量 + 因果性 + 掩码反事实 ----------------
    note("自检①：掩码非退化 / 参数量 / 因果性 / 锚点掩码反事实")
    selfchecks = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0), "n_params": ck[name]["n_params"]}
        if spec["kind"] != "full":
            km = keep_mask_for(spec, T, dev)
            s = int(km.sum().item())
            exp = EXPECTED_VISIBLE[(int(spec["window"]), int(spec.get("anchors", 0)))]
            row["mask"] = {"visible_pairs": s, "expected": exp, "ok": s == exp,
                           "causal_pairs": T * (T + 1) // 2, "nondegenerate": s < T * (T + 1) // 2}
            assert s == exp, f"{name}: visible pairs {s} != {exp}"
            assert s < T * (T + 1) // 2, f"{name}: 掩码是恒等算子（退化）"
        selfchecks[name] = row
    if "win16a_s0" in ck and "win16_s0" in ck:
        assert ck["win16a_s0"]["n_params"] == ck["win16_s0"]["n_params"], \
            "锚点臂与纯窗臂参数量应逐位相同（掩码不引入参数）"

    g_probe = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, vocab, (2, T), generator=g_probe).to(dev)
    for spec in cfg["arms"]:
        name = spec["name"]
        selfchecks[name]["causality"] = causality_probe(arms[name], x_probe,
                                                        keep_mask_for(spec, T, dev))
        if spec["kind"] != "full":
            selfchecks[name]["mask_counterfactual"] = window_mask_probe(arms[name], x_probe)
        note(f"  {name}: causal_prefix_diff="
             f"{selfchecks[name]['causality']['max_abs_logit_diff_prefix']} "
             f"mask_counterfactual="
             f"{selfchecks[name].get('mask_counterfactual', {}).get('max_abs_logit_diff_masked_vs_open')}")

    # ---------------- ② 质量腿（价格）----------------
    note(f"质量腿②：clean bpc（每臂自己的训练可见性，{eb['n_batches']}×{eb['batch_size']}，"
         f"window_seed={eb['window_seed']}）")
    quality = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        bpc = rp.win16_bpc(arms[name], eval_arr, km, n_batches=eb["n_batches"],
                           batch_size=eb["batch_size"], window_seed=eb["window_seed"],
                           device=dev)
        quality[name] = round(bpc, 6)
        note(f"  {name}: bpc={bpc:.6f}")

    # 本格自测带 σ_cell = 本实验换种子极差（2 样本 ⇒ 描述量）
    sp = {}
    for kind, a, b in (("win16a", "win16a_s0", "win16a_s1"), ("win16", "win16_s0", "win16_s1")):
        if a in quality and b in quality:
            sp[kind] = round(abs(quality[a] - quality[b]), 6)
    sigma_cell = max(sp.values()) if sp else None
    price = {}
    for s in cfg["seeds"]:
        a, w = f"win16a_s{s}", f"win16_s{s}"
        if a not in quality or w not in quality:
            continue
        d = round(quality[a] - quality[w], 6)
        price[str(s)] = {"bpc_win16a": quality[a], "bpc_win16": quality[w],
                         "price_bpc": d,
                         "sortable_vs_sigma_cell": bool(abs(d) > sigma_cell),
                         "sortable_vs_0.014": bool(abs(d) > CROSS_CELL_BAND),
                         "sign": "锚点是质量税（更差）" if d > 0 else
                                 ("锚点是质量收益（更好）" if d < 0 else "相等")}
    reuse_equiv = {"note": "复用臂（eval-only）clean bpc 必须与 2634 落盘值逐位相同（判据⑥）"}
    for nm, ref in REF_2634_WIN16_BPC.items():
        got = quality[nm]
        reuse_equiv[nm] = {"bpc": got, "bpc_2634": ref, "equal": got == ref,
                           "reused_from": ck[nm]["reused_from"]}

    # ---------------- ③ 锚点召回腿（主判据，needle v3）----------------
    ncfg = dict(n_needles=vcfg["n_needles"], phrase_len=vcfg["phrase_len"],
                anchors=vcfg["anchors"], anchor_frac=vcfg["anchor_frac"])
    bucket_anchors = int(vcfg["bucket_anchors"])
    L = vcfg["phrase_len"]
    note(f"锚点召回腿③：needle v3（{vcfg['n_batches']}×{vcfg['batch_size']}，"
         f"{vcfg['n_needles']} 针/行，seed={vcfg['seed']}，归属桶 anchors={bucket_anchors}）")
    gen = torch.Generator().manual_seed(vcfg["seed"])
    arm_names = [s["name"] for s in cfg["arms"]]
    records = {f"{n}|{kind}": [] for n in arm_names for kind in ("planted", "control")}
    n_needles_total = 0
    for bi in range(vcfg["n_batches"]):
        state = gen.get_state()
        x, meta = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                      mask_tok, gen, **ncfg, plant=True)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                        mask_tok, gen, **ncfg, plant=False)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        n_needles_total += sum(len(m) for m in meta)
        for name in arm_names:
            spec = next(s for s in cfg["arms"] if s["name"] == name)
            km = keep_mask_for(spec, T, dev)
            for kind, xx in (("planted", x), ("control", x0)):
                rec = needle_recall(arms[name], xx, meta, km,
                                    anchors=bucket_anchors, phrase_len=L, device=dev)
                records[f"{name}|{kind}"].extend(rec["per_needle"])
        log.append({"t": round(time.time() - t0, 2), "msg": f"needle batch {bi}"})

    planted_b = {n: bucket_records(records[f"{n}|planted"]) for n in arm_names}
    control_b = {n: bucket_records(records[f"{n}|control"]) for n in arm_names}
    needle = {"conditions": {}, "n_needles_total": n_needles_total,
              "seed": vcfg["seed"], "n_batches": vcfg["n_batches"],
              "bucket_anchors": bucket_anchors,
              "bucket_rule": ("归属桶一律 anchors=%d：anchor = 针植入 p<%d（锚点槽恒可见区）；"
                              "window = 非锚针且短语全可见；beyond = 非锚针且有位置不可见。"
                              "五臂同归属规则 ⇒ 同索引可比" % (bucket_anchors, bucket_anchors))}
    bands = [(0, 32), (32, 64), (64, 128), (128, 256)]

    def band_of(d):
        for lo, hi in bands:
            if lo <= d < hi:
                return f"[{lo},{hi})"
        return f"[{bands[-1][1]},)"

    for key, recs in records.items():
        s = needle_summary(recs)
        by_band: dict[str, list[int]] = {}
        for r in recs:
            if r["bucket"] == "beyond":
                by_band.setdefault(band_of(r["dist"]), []).append(r["hit"])
        needle["conditions"][key] = {
            "recall": s["recall"], "n": s["n"], "buckets": s["buckets"],
            "beyond_by_distance_band": {b: {"recall": round(sum(v) / len(v), 6),
                                            "n": len(v)} for b, v in sorted(by_band.items())}}

    # 逐臂 × 逐桶：植入 vs 配对地板（Newcombe + 配对 bootstrap）
    per_arm_bucket = {}
    for n in arm_names:
        row = {}
        for b in ("anchor", "window", "beyond"):
            p = [r["hit"] for r in planted_b[n].get(b, [])]
            c = [r["hit"] for r in control_b[n].get(b, [])]
            assert len(p) == len(c), f"{n}/{b}: 植入与地板针数不一致"
            row[b] = paired_reading(p, c)
        per_arm_bucket[n] = row
    needle["per_arm_bucket_planted_vs_floor"] = per_arm_bucket

    # 锚点桶内：按「短语是否全可见」拆分（诊断，锚点臂内部说明）
    anchor_split = {}
    for n in arm_names:
        vis = [r["hit"] for r in planted_b[n].get("anchor", []) if r.get("visible")]
        nvis = [r["hit"] for r in planted_b[n].get("anchor", []) if not r.get("visible")]
        anchor_split[n] = {
            "fully_visible": {"n": len(vis),
                              "recall": (round(sum(vis) / len(vis), 6) if vis else None)},
            "partially_visible": {"n": len(nvis),
                                  "recall": (round(sum(nvis) / len(nvis), 6) if nvis else None)},
            "note": ("锚点槽 [0,8) 只有 8 格，而短语占 L+1=7 格 ⇒ 只有 p∈{0,1} 的针"
                     "「短语全可见」，p∈{2..7} 只部分可见；此拆分只作诊断")}
    needle["anchor_bucket_visibility_split"] = anchor_split

    # 同索引硬子集（跨臂可比）：对 anchor / beyond 两桶各算一次
    needle["hard_subset_same_index"] = {}
    for b in ("anchor", "beyond"):
        ctrl_by_arm = {n: control_b[n].get(b, []) for n in arm_names}
        if not all(ctrl_by_arm.values()):
            needle["hard_subset_same_index"][b] = {"note": "该桶在至少一臂为空，跳过同索引"}
            continue
        idx_sets = ns.hard_index_sets(ctrl_by_arm)
        planted_by_arm = {n: planted_b[n].get(b, []) for n in arm_names}
        same = {}
        for n in arm_names:
            same[n] = {
                "own_recall": round(sum(r["hit"] for r in planted_by_arm[n]) / len(planted_by_arm[n]), 6),
                "intersection": ns.recall_on_indices(planted_by_arm[n], idx_sets["intersection"]),
                "union": ns.recall_on_indices(planted_by_arm[n], idx_sets["union"])}
        cross = {}
        for s in cfg["seeds"]:
            a, w = f"win16a_s{s}", f"win16_s{s}"
            cross[f"{a}_minus_{w}"] = {}
            for set_name in ("intersection", "union"):
                idx = idx_sets[set_name]
                pa = ns.recall_on_indices(planted_by_arm[a], idx)
                pw = ns.recall_on_indices(planted_by_arm[w], idx)
                cross[f"{a}_minus_{w}"][set_name] = ns.beyond_above_floor(
                    pa["hits"], pa["n"], pw["hits"], pw["n"])
        needle["hard_subset_same_index"][b] = {
            "n_intersection": idx_sets["n_intersection"], "n_union": idx_sets["n_union"],
            "n_indexed": idx_sets["n_indexed"], "per_arm": same,
            "cross_arm_anchor_vs_window": cross}

    # 判据②③ 汇总
    def _passes(arm, b, key="passes_both"):
        return bool(per_arm_bucket[arm][b].get(key, False)) if per_arm_bucket[arm][b].get("n") else False

    crit = {
        "c2_anchor_bucket_above_floor": {
            "win16a_s0": per_arm_bucket["win16a_s0"]["anchor"],
            "win16a_s1": per_arm_bucket["win16a_s1"]["anchor"],
            "win16a_both_seeds_pass_both_ci": bool(
                _passes("win16a_s0", "anchor") and _passes("win16a_s1", "anchor")),
            "intervention_arms_all_pass_newcombe": bool(
                _passes("win16a_s0", "anchor", "passes_newcombe")
                and _passes("win16a_s1", "anchor", "passes_newcombe")),
        },
        "c3_beyond_bucket_stays_at_floor": {
            n: {"delta_pp": per_arm_bucket[n]["beyond"].get("delta_pp"),
                "newcombe_diff_ci95_pp": per_arm_bucket[n]["beyond"].get("newcombe_diff_ci95_pp"),
                "ci_contains_0": (None if not per_arm_bucket[n]["beyond"].get("newcombe_diff_ci95_pp")
                                  else bool(per_arm_bucket[n]["beyond"]["newcombe_diff_ci95_pp"][0] <= 0
                                            <= per_arm_bucket[n]["beyond"]["newcombe_diff_ci95_pp"][1]))}
            for n in ("win16a_s0", "win16a_s1", "win16_s0", "win16_s1")},
    }
    note("判据② anchor 桶："
         f"s0={crit['c2_anchor_bucket_above_floor']['win16a_s0']['delta_pp']}pp "
         f"CI{crit['c2_anchor_bucket_above_floor']['win16a_s0']['newcombe_diff_ci95_pp']} | "
         f"s1={crit['c2_anchor_bucket_above_floor']['win16a_s1']['delta_pp']}pp "
         f"CI{crit['c2_anchor_bucket_above_floor']['win16a_s1']['newcombe_diff_ci95_pp']} "
         f"⇒ 两 seed 全过={crit['c2_anchor_bucket_above_floor']['win16a_both_seeds_pass_both_ci']}")
    for n in ("win16_s0", "win16_s1"):
        r = per_arm_bucket[n]["anchor"]
        note(f"  负对照 {n}: anchor 桶 {r.get('delta_pp')}pp CI{r.get('newcombe_diff_ci95_pp')}")

    # ---------------- ④ 毁源敏感度（两变体）----------------
    pkw = rp.probe_kwargs(cfg)
    assert pkw["rf"] == cfg["model"]["n_layers"] * (window - 1) == 180
    note(f"毁源敏感度④：变体 A far_all 被毁 [0,64)；变体 B beyond_anchor 被毁 [8,64)；"
         f"被评行 [{pkw['eval_lo']},{pkw['eval_hi']})，rf={pkw['rf']}")
    probe = {}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        arms_kw = {k: pkw[k] for k in ("n_batches", "batch_size", "window_seed",
                                      "corrupt_hi", "eval_lo", "eval_hi", "corrupt_seed")}
        arms_kw["rf"] = pkw["rf"]
        row = {}
        # 变体 A：被毁 [0,64)。窗口臂要求不重叠；锚点臂必然重叠（锚点槽 [0,8) 恒可见）
        is_anchor = int(spec.get("anchors", 0)) > 0
        row["far_all"] = rp.far_source_probe(
            arms[name], eval_arr, km, **arms_kw, corrupt_lo=0,
            require_no_overlap=not is_anchor, device=dev)
        # 变体 B：被毁 [8,64)（保住锚点槽完整）
        row["beyond_anchor"] = rp.far_source_probe(
            arms[name], eval_arr, km, **arms_kw, corrupt_lo=int(spec.get("anchors", 0)),
            require_no_overlap=True, device=dev)
        probe[name] = row
        note(f"  {name}: far_all Δbpc={row['far_all']['delta_bpc_far']:+.6f} "
             f"max|Δlogit|={row['far_all']['max_abs_logit_diff_on_eval_rows']:.3e} "
             f"premise={row['far_all']['premise_holds']} | "
             f"beyond_anchor Δbpc={row['beyond_anchor']['delta_bpc_far']:+.6f} "
             f"max|Δlogit|={row['beyond_anchor']['max_abs_logit_diff_on_eval_rows']:.3e} "
             f"premise={row['beyond_anchor']['premise_holds']}")

    crit["c4_far_source"] = {
        n: {"far_all": {"delta_bpc_far": probe[n]["far_all"]["delta_bpc_far"],
                        "max_abs_logit_diff_on_eval_rows":
                            probe[n]["far_all"]["max_abs_logit_diff_on_eval_rows"],
                        "premise_holds": probe[n]["far_all"]["premise_holds"]},
            "beyond_anchor": {"delta_bpc_far": probe[n]["beyond_anchor"]["delta_bpc_far"],
                              "max_abs_logit_diff_on_eval_rows":
                                  probe[n]["beyond_anchor"]["max_abs_logit_diff_on_eval_rows"],
                              "premise_holds": probe[n]["beyond_anchor"]["premise_holds"]}}
        for n in probe}
    crit["c4_beyond_anchor_bitwise_insensitive"] = bool(all(
        probe[n]["beyond_anchor"]["max_abs_logit_diff_on_eval_rows"] == 0.0 for n in probe))
    crit["c4_far_all_window_bitwise_insensitive"] = bool(all(
        probe[n]["far_all"]["max_abs_logit_diff_on_eval_rows"] == 0.0
        for n in probe if int(next(s for s in cfg["arms"] if s["name"] == n)["anchors"]) == 0))
    crit["c5_causality_no_future_leak"] = bool(all(
        selfchecks[n]["causality"]["causal_no_future_leak"] for n in arm_names))
    crit["c6_mask_nondegenerate_and_reuse_equal"] = bool(
        all(v.get("mask", {}).get("nondegenerate", True) for v in selfchecks.values())
        and reuse_equiv["win16_s0"]["equal"] and reuse_equiv["win16_s1"]["equal"])

    # ---------------- ⑤ 代价 + 活性 ----------------
    rows = []
    for spec in cfg["arms"]:
        name = spec["name"]
        k_eff = T if spec["kind"] == "full" else int(spec["window"]) + int(spec.get("anchors", 0))
        kv = unit * k_eff
        rows.append({"name": name, "kind": spec["kind"], "seed": spec["seed"],
                     "window": spec["window"], "anchors": spec.get("anchors", 0),
                     "k_eff": k_eff, "kv_bytes_per_position": kv,
                     "bpc": quality[name], "n_params": ck[name]["n_params"]})
    liveness = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0),
               "mask": selfchecks[name].get("mask"),
               "mask_counterfactual": selfchecks[name].get("mask_counterfactual"),
               "causality": selfchecks[name]["causality"]}
        tdir = os.path.dirname(resolve(spec["path"]))
        tlog = os.path.join(tdir, "train_log.jsonl")
        tres = os.path.join(tdir, "results.json")
        if os.path.exists(tlog):
            lines = [json.loads(x) for x in open(tlog) if x.strip()]
            row["train_window_mask"] = lines[-1].get("window_mask")
            row["train_last_loss"] = lines[-1].get("loss")
        if os.path.exists(tres):
            tr = json.load(open(tres))
            row["train_wall_minutes"] = tr.get("wall_minutes")
            row["train_peak_gpu_mem_mb"] = tr.get("peak_gpu_mem_mb")
            row["train_window_liveness"] = tr.get("window_liveness")
        liveness[name] = row
        note(f"  {name}: visible_pairs={row['mask']['visible_pairs'] if row['mask'] else 'n/a'} "
             f"counterfactual={row['mask_counterfactual']['max_abs_logit_diff_masked_vs_open'] if row['mask_counterfactual'] else 'n/a'} "
             f"train_window_mask_alive="
             f"{(row.get('train_window_mask') or {}).get('alive')}")

    results = {
        "config": cfg, "checkpoints": ck, "selfchecks": selfchecks,
        "quality_leg": {"bpc": quality, "price": price,
                        "sigma_cell_own_cell": sigma_cell,
                        "sigma_cell_per_kind": sp,
                        "bands": {"sigma_cell": sigma_cell, "cross_cell_band": CROSS_CELL_BAND,
                                  "note": "本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用（C1）"},
                        "reuse_equivalence": reuse_equiv,
                        "gap_win16_full": round(quality["win16_s0"] - quality["full"], 6),
                        "gap_win16a_full": round(quality["win16a_s0"] - quality["full"], 6)},
        "needle_leg": needle,
        "far_source_probe": probe,
        "criteria": crit,
        "rows": rows,
        "liveness": liveness,
        "code_hashes": {
            rel: sha256_file(os.path.join(ROOT, rel)) for rel in
            ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/retention.py",
             "tools/evalkit/carry_channel.py", "pathlm/data.py")
            if os.path.exists(os.path.join(ROOT, rel))},
        "cost": {
            "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
            "eval_peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                     if on_cuda else None),
            "n_eval_windows_bpc": eb["n_batches"] * eb["batch_size"],
            "n_probe_windows": ep2["n_batches"] * ep2["batch_size"],
            "n_needle_batches": vcfg["n_batches"],
            "note": "墙钟只作过程指标，不进判据",
        },
    }
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "realtext_probe.py", "needle_stats.py", "make_config.py", "config.json",
         "train_arms.sh") if os.path.exists(os.path.join(HERE, rel))})

    results = _strict(results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(args.out_dir, "needle_raw.json"), "w") as f:
        json.dump(_strict(records), f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "run_log.txt"), "w") as f:
        for r in log:
            f.write(f"[{r['t']:7.1f}s] {r['msg']}\n")
    note(f"落盘：{args.out_dir}/{{results.json,needle_raw.json,run_log.jsonl,run_log.txt}}")


if __name__ == "__main__":
    main()
