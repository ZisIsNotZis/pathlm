#!/usr/bin/env python3
"""2634-carry-realtext-13m · 评测入口（纯推理）：真实文本口径的判据①–⑤。

契约：同目录 `README.md`（§4 判据线预登记，未改一字）。本入口做：

  ① 仪器自检：参数量/块类断言；窗口掩码非退化；同 seed 重建 none/delta 的**主干逐位相等**
     与「单边零起步逐位恒等」；needle 构造检查（配对控制臂的针位逐一相同）；
  ② 质量腿：`bpc_win16`（窗口 16 掩码）+ `bpc_full`（release 口径诊断）+ 通道消融后的 bpc；
  ③ **主判据**：毁源式窗外源敏感度探针 `Δbpc_far`（`realtext_probe.far_source_probe`），
     前提① = 无通道臂被评行 logits 逐位不变；
  ④ 毁源对照（离散答案）：needle v3 真实文本短语针，`beyond` 桶（窗外）vs **配对无针地板**；
  ⑤ 本格自测噪声界 σ_cell（同配置换种子的极差）+ 禁外推声明。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~5 min）
  python3 run.py --smoke    # 冒烟：少批
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
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             carry_probe, carry_read_mass_stats,
                             window_anchor_keep_mask)
from tools.evalkit.retention import (needle_recall, needle_summary,  # noqa: E402
                                     phrase_needle_batch)
from tools.evalkit.carry_channel import (loss_slope, param_equality,  # noqa: E402
                                         trunk_equality)
import realtext_probe as rp                                        # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def model_config_kwargs(rule: str) -> dict:
    return {"carry_rule": None if rule == "none" else rule}


def load_arm(name: str, spec: dict, mcfg: dict, vocab: int, device: str):
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n want {want}\n got  {got}")
    cfg = DepthARConfig(**{**mcfg, **model_config_kwargs(spec["rule"])})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "rule": spec["rule"], "seed": spec["seed"],
                   "carry_rule": cfg.carry_rule,
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


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
            cfg["eval"][k]["n_batches"] = 1
            cfg["eval"][k]["batch_size"] = 8
        cfg["needle"]["n_batches"] = 1
        cfg["needle"]["batch_size"] = 8
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]
    vcfg = cfg["needle"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8（尾段 10M 字符做评测；训练侧由 train_depth_ar.py 自行加载）")
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    assert vocab == cfg["data"]["vocab_size"], f"vocab {vocab} != {cfg['data']['vocab_size']}"

    note("加载四臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(
            spec["name"], spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:16]}… carry={spec['rule']} "
             f"params={ck[spec['name']]['n_params']} "
             f"block={ck[spec['name']]['block_class']}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors}}

    # ---------------- ① 仪器自检 ----------------
    note("自检①：参数量/主干逐位/单边零起步/窗口掩码非退化")
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    n_visible = int(keep.sum().item())
    assert n_visible < T * (T + 1) // 2, "窗口掩码在 T=256 上是恒等算子（退化点）"
    g = torch.Generator().manual_seed(7)
    x_probe = torch.randint(0, vocab, (2, T), generator=g).to(dev)
    eq, trunk, id_start = {}, {}, {}
    for seed in cfg["seeds"]:
        built = {}
        for rule in ("none", "delta"):
            torch.manual_seed(seed)
            built[rule] = DepthARModel(
                DepthARConfig(**{**cfg["model"], **model_config_kwargs(rule)}), vocab)
            built[rule].eval()
        eq[seed] = param_equality(built["none"], built["delta"])
        trunk[seed] = trunk_equality(built["delta"], built["none"])
        with torch.no_grad():
            l0 = built["none"](x_probe.cpu())[1]["depth_logits"][-1]
            id_start[seed] = {"delta_matches_none_bitwise": torch.equal(
                l0, built["delta"](x_probe.cpu())[1]["depth_logits"][-1])}
    assert ck["rt_none_s0"]["n_params"] == ck["rt_none_s1"]["n_params"]
    assert ck["rt_delta_s0"]["n_params"] == ck["rt_delta_s1"]["n_params"]
    results["selfchecks"] = {
        "window_mask_visible_pairs": n_visible, "window_mask_nondegenerate": True,
        "params_bitwise_equal_none_delta": eq,
        "trunk_bitwise_equal_delta_vs_none": trunk,
        "identity_start_logits_bitwise": id_start,
        "n_params": {n: ck[n]["n_params"] for n in arms},
    }
    note(f"  可见对数 {n_visible}；none/delta 主干逐位={trunk[0]['trunk_bitwise_equal']}"
         f"（+{trunk[0]['n_params_added']} 参数）；单边零起步={id_start[0]}")

    # ---------------- ② 质量腿 + 通道消融 ----------------
    eb, ep = cfg["eval"]["bpc"], cfg["eval"]["probe"]
    note(f"质量腿：bpc_win16（窗口 {window}/{anchors}，{eb['n_batches']}×{eb['batch_size']}，"
         f"~{eb['n_batches']*eb['batch_size']*(T-1)} token）；release 口径只作诊断")
    quality = {}
    for name, model in arms.items():
        row = {}
        row["bpc_win16"] = round(rp.win16_bpc(model, eval_arr, keep, **{
            k: eb[k] for k in ("n_batches", "batch_size", "window_seed")}, device=dev), 6)
        row["bpc_full_release_diagnostic"] = round(rp.win16_bpc(model, eval_arr, None, **{
            k: eb[k] for k in ("n_batches", "batch_size", "window_seed")}, device=dev), 6)
        with rp.channel_off(model):
            row["bpc_win16_channel_off"] = round(rp.win16_bpc(model, eval_arr, keep, **{
                k: eb[k] for k in ("n_batches", "batch_size", "window_seed")},
                device=dev), 6)
        quality[name] = row
        note(f"  {name}: bpc_win16={row['bpc_win16']} "
             f"release(diag)={row['bpc_full_release_diagnostic']} "
             f"通道消融={row['bpc_win16_channel_off']}")

    # ---------------- ③ 主判据：毁源式窗外源敏感度探针 ----------------
    note(f"判据①③：Δbpc_far 探针（毁前缀 [0,{ep['corrupt_hi']})，被评行 "
         f"[{ep['eval_lo']},{ep['eval_hi']})，{ep['n_batches']}×{ep['batch_size']}）")
    probe = {}
    for name, model in arms.items():
        row = rp.far_source_probe(model, eval_arr, keep, **{
            k: ep[k] for k in ("n_batches", "batch_size", "window_seed", "corrupt_hi",
                               "eval_lo", "eval_hi", "corrupt_seed")}, device=dev)
        with rp.channel_off(model):
            row["with_channel_off"] = rp.far_source_probe(model, eval_arr, keep, **{
                k: ep[k] for k in ("n_batches", "batch_size", "window_seed", "corrupt_hi",
                                   "eval_lo", "eval_hi", "corrupt_seed")}, device=dev)
        probe[name] = row
        note(f"  {name}: Δbpc_far={row['delta_bpc_far']:+.6f} "
             f"CI{row['delta_ci95_clustered']} clean={row['ce_far_clean']:.4f} "
             f"max|Δlogit|被评行={row['max_abs_logit_diff_on_eval_rows']:.3e} | "
             f"通道消融后 Δ={row['with_channel_off']['delta_bpc_far']:+.6f}")

    # ---------------- ④ 毁源对照（离散答案）：needle v3 ----------------
    ncfg = dict(n_needles=vcfg["n_needles"], phrase_len=vcfg["phrase_len"],
                anchors=vcfg["anchors"], anchor_frac=vcfg["anchor_frac"])
    L = vcfg["phrase_len"]
    note(f"判据④：真实文本短语针（needle v3，{vcfg['n_batches']}×{vcfg['batch_size']}，"
         f"{vcfg['n_needles']} 针/行，seed={vcfg['seed']}）")
    gen = torch.Generator().manual_seed(vcfg["seed"])
    records = {f"{n}|{kind}": [] for n in arms for kind in ("planted", "control")}
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
        for name, model in arms.items():
            for kind, xx in (("planted", x), ("control", x0)):
                rec = needle_recall(model, xx, meta, keep,
                                    anchors=vcfg["bucket_anchors"], phrase_len=L, device=dev)
                records[f"{name}|{kind}"].extend(rec["per_needle"])
        log.append({"t": round(time.time() - t0, 2), "msg": f"needle batch {bi}"})
    needle = {"conditions": {}, "n_needles_total": n_needles_total,
              "seed": vcfg["seed"], "n_batches": vcfg["n_batches"],
              "bucket_rule": ("anchors=0 ⇒ 桶只有 window（短语全可见）与 beyond（有位置不可见）；"
                              "本实验锚点 0，窗 16 ⇒ beyond 占绝大多数")}
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
                                            "n": len(v)} for b, v in sorted(by_band.items())},
        }
    # 地板之上的增量（配对：同批、同针位、同生成器序列）
    needle["paired_beyond_vs_floor"] = {}
    for name in arms:
        p = [r for r in records[f"{name}|planted"] if r["bucket"] == "beyond"]
        c = [r for r in records[f"{name}|control"] if r["bucket"] == "beyond"]
        assert len(p) == len(c)
        kp, kc, n = sum(r["hit"] for r in p), sum(r["hit"] for r in c), len(p)
        hard = [i for i, r in enumerate(c) if r["hit"] == 0]
        needle["paired_beyond_vs_floor"][name] = {
            "n": n, "planted_hits": kp, "control_hits": kc,
            "planted_recall": round(kp / n, 6) if n else None,
            "control_recall": round(kc / n, 6) if n else None,
            "delta_pp": round(100 * (kp - kc) / n, 3) if n else None,
            "hard_subset": {
                "n": len(hard),
                "planted_recall": round(sum(p[i]["hit"] for i in hard) / max(len(hard), 1), 6),
                "note": "控制臂答不出的针（答案只能来自记忆）"},
        }
    for name in arms:
        r = needle["paired_beyond_vs_floor"][name]
        note(f"  {name}: beyond 植入 {r['planted_recall']} vs 地板 {r['control_recall']} "
             f"⇒ Δ={r['delta_pp']}pp（n={r['n']}，硬子集 n={r['hard_subset']['n']}）")

    # ---------------- ⑤ 跨臂效应 + 本格噪声界 ----------------
    note("跨臂效应与 σ_cell（同配置换种子极差；2 样本 ⇒ 描述量、非方差上界）")
    def spread(vals):
        return {"per_seed": [round(v, 6) for v in vals],
                "range": round(max(vals) - min(vals), 6)}
    sigma_bpc = max(abs(quality["rt_none_s0"]["bpc_win16"]
                        - quality["rt_none_s1"]["bpc_win16"]),
                    abs(quality["rt_delta_s0"]["bpc_win16"]
                        - quality["rt_delta_s1"]["bpc_win16"]))
    sigma_far = max(abs(probe["rt_none_s0"]["delta_bpc_far"]
                        - probe["rt_none_s1"]["delta_bpc_far"]),
                    abs(probe["rt_delta_s0"]["delta_bpc_far"]
                        - probe["rt_delta_s1"]["delta_bpc_far"]))
    effects = {}
    for seed in cfg["seeds"]:
        qn = quality[f"rt_none_s{seed}"]["bpc_win16"]
        qd = quality[f"rt_delta_s{seed}"]["bpc_win16"]
        effects[str(seed)] = {
            "bpc_win16": {"none": qn, "delta": qd, "delta_bpc": round(qn - qd, 6)},
            "delta_bpc_far": {
                "none": probe[f"rt_none_s{seed}"]["delta_bpc_far"],
                "delta": probe[f"rt_delta_s{seed}"]["delta_bpc_far"]},
        }
    trans = {str(sd): bool(
        probe[f"rt_delta_s{sd}"]["delta_bpc_far"] > 0
        and probe[f"rt_delta_s{sd}"]["delta_bpc_far"] > sigma_far
        and probe[f"rt_delta_s{sd}"]["delta_ci95_clustered"][0] > 0)
        for sd in cfg["seeds"]}
    premise_ok = all(
        probe[f"rt_none_s{sd}"]["max_abs_logit_diff_on_eval_rows"] == 0.0
        for sd in cfg["seeds"])
    quality_gain = {str(sd): bool(effects[str(sd)]["bpc_win16"]["delta_bpc"] > sigma_bpc)
                    for sd in cfg["seeds"]}
    needle_transport = {n: bool(
        needle["paired_beyond_vs_floor"][n]["delta_pp"] > 0) for n in arms}
    criteria = {
        "c1_premise_nonchannel_insensitive": bool(premise_ok),
        "c2_channel_transports_on_realtext": bool(all(trans.values())),
        "c3_quality_leg_channel_better": bool(all(quality_gain.values())),
        "c4_needle_beyond_above_floor": bool(any(needle_transport.values())),
        "c4_note": ("该腿的判据是「离散答案探针在窗外桶高于配对地板」；"
                    "若各臂都 ≈ 地板 ⇒ 该探针在此尺度无区分力（如实报，不作机制结论）"),
        "c5_no_extrapolation": True,
    }
    note(f"判据① 前提（无通道臂被评行逐位不敏感）：{premise_ok}")
    note(f"判据② delta 臂 Δbpc_far>0 且>σ_far={sigma_far:.6f}：{trans}")
    note(f"判据③ 质量腿 Δbpc_win16>σ_cell={sigma_bpc:.6f}：{effects} ⇒ {quality_gain}")
    note(f"判据④ needle beyond 高于地板：{needle_transport}")

    # ---------------- 活性 / 训练侧读数 ----------------
    note("活性量与训练侧读数（答案行 CE 斜率不适用：真实文本无答案行 ⇒ 报 loss 斜率）")
    slopes, tail, train_meta, read_mass = {}, {}, {}, {}
    for name, model in arms.items():
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        res_p = json.load(open(resolve(spec["train_results_json"])))
        lines = [json.loads(l) for l in open(resolve(spec["train_log"])) if l.strip()]
        slopes[name] = loss_slope(lines)
        tail[name] = {"last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
                      "last_read_share": lines[-1].get("attn_carry")}
        train_meta[name] = {"wall_minutes": res_p.get("wall_minutes"),
                            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
                            "params": res_p.get("params"),
                            "seed": res_p.get("train", {}).get("seed"),
                            "carry_write": res_p.get("carry_write"),
                            "window_mask": res_p.get("window_mask")}
        if ck[name]["carry_rule"]:
            read_mass[name] = carry_read_mass_stats(model, x_probe, window=window)
            res_p_live = res_p.get("carry_liveness")
            train_meta[name]["carry_liveness_training"] = res_p_live
            read_mass[name]["eval_recheck"] = {
                "note": "评测期反事实：关通道 / 换写入规则是否改变前向（见 2618 §活性口径）",
                "probe": carry_probe(model, x_probe)}
        note(f"  {name}: wall={train_meta[name]['wall_minutes']}min "
             f"peak={train_meta[name]['peak_gpu_mem_mb']}MB "
             f"loss末={tail[name]['last_loss']:.4f} "
             f"read_share={read_mass.get(name, {}).get('read_share_per_layer')}")

    results.update({
        "quality_leg": quality, "far_source_probe": probe,
        "needle_leg": needle, "effects_by_seed": effects,
        "sigma_cell": {"bpc_win16": sigma_bpc, "delta_bpc_far": sigma_far,
                       "note": "同配置换种子的极差（2 样本 ⇒ 描述量，不是方差上界）"},
        "criteria": criteria,
        "noise_floor_note": ("本格自测；SCHEMA §2 的 0.014 是 bpc 口径的历史方差带，"
                             "本实验以 own-cell 极差为准并并列报出"),
        "liveness": read_mass, "loss_slope": slopes, "train_log_tail": tail,
        "train_meta": train_meta,
    })
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: train_meta[n]["wall_minutes"] for n in arms},
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: train_meta[n]["peak_gpu_mem_mb"] for n in arms},
        "n_eval_windows_bpc": eb["n_batches"] * eb["batch_size"],
        "n_probe_windows": ep["n_batches"] * ep["batch_size"],
        "n_needle_batches": vcfg["n_batches"],
        "lock_note": "训练按臂释放锁（§20.1，臂间 ≥60s）；评测整程 <10 min 持锁",
    }
    results["code_hashes"] = {
        rel: sha256_file(os.path.join(ROOT, rel)) for rel in
        ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/longrange_tasks.py",
         "tools/evalkit/carry_channel.py", "tools/evalkit/retention.py",
         "pathlm/data.py") if os.path.exists(os.path.join(ROOT, rel))}
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "realtext_probe.py", "config.json")})

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "needle_raw.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"落盘：{args.out_dir}/results.json + needle_raw.json")


if __name__ == "__main__":
    main()
