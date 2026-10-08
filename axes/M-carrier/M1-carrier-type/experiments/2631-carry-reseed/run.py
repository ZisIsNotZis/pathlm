#!/usr/bin/env python3
"""2631-carry-reseed · 评测入口（纯推理）：判据①②③④ + 六问所需的全部读数。

契约：同目录 `README.md`（§4 判据线**逐字**抄录，未改一字）。本入口做：

  ① 仪器自检：释放版 == 显式因果掩码（逐位）；模型自带窗口掩码前向 ==
     `retention.masked_logits` + `window_anchor_keep_mask`（训练/评测同一件仪器；
     **非退化点**：窗口 16 在 T=256 上不是恒等算子）；三臂同 seed 重建的**逐位等参数**
     （plain vs delta）与**主干逐位相等**（通道臂 vs 无通道臂）；单边零起步逐位恒等；
  ② **主腿 = 训练分布内多查询关联回忆**（`assoc_recall_multi_batch`，K=8/Q=8/gap=14，
     距离 107–240 ≤ 训练上限 240）；六臂在**同一批位置**上评测（配对）；
     **主 CI = 按序列聚类的 bootstrap**（复核 B2：同一序列的 Q 个查询共享 KV/记忆状态，
     不是独立二项样本），每条查询的 Wilson CI 只作偏窄对照；
  ③ 毁源对照臂（判据④）：源片段换全新随机符号后再测一遍；
  ④ 跨种子合并（判据①②③）：逐种子效应 + 极差 + 方向一致性 + 预登记阈值（5pp/3pp）；
  ⑤ **辅助腿**（不作判据）：单查询关联回忆，距离 241–255（训练分布外，§17.18）；
  ⑥ 活性量（`carry_liveness` / `carry_read_mass_stats`）、答案行 CE 末窗斜率（§17.15）、
     loss 斜率（§16）、解析/实测显存分开报。

用法：
  python3 run.py                # 全电池（纯推理，共享卡 ~3–5 min）
  python3 run.py --smoke        # 冒烟：少批（实测耗时/显存，不外推）
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

from pathlm.depth_ar import (DepthARConfig, DepthARModel,            # noqa: E402
                             carry_liveness, carry_probe, carry_read_mass_stats,
                             window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import (                          # noqa: E402
    assoc_recall_batch, assoc_recall_multi_batch, corrupt_span,
    local_prior_accuracy)
from tools.evalkit.retention import masked_logits                    # noqa: E402
from tools.evalkit.carry_channel import (analytic_memory, loss_slope,  # noqa: E402
                                         param_equality, trunk_equality)
import reseed_arms as ra                                             # noqa: E402

TRAIN_MAX_DIST = 240          # 训练生成器（K=8/Q=8/gap=14）能出现的最远查询距离


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, spec: dict, mcfg: dict, vocab: int, device: str):
    """加载一臂权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n"
                         f"  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, **ra.model_config_kwargs(spec["rule"])})
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


@torch.no_grad()
def per_cluster_hits(model, task, keep, device: str, max_chunk: int = 32):
    """一臂在一件任务批上的**逐序列**命中数（判据①聚类统计的原始计数）。

    返回 `(cluster_hits, cluster_n)`：每个元素是一条序列（簇）的命中数 / 提问数。
    """
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        km = keep.to(device) if keep.dim() == 2 else keep[lo:hi].to(device)
        logits = masked_logits(model, task.tokens[lo:hi].to(device), km)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(device))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(device)).to(torch.long).cpu()
    return hits.sum(1).tolist(), [int(task.M)] * int(task.B)


def eval_leg(model, tasks, keep, device: str, max_chunk: int = 32) -> dict:
    """一臂在一条腿（多批任务）上的读数：聚类统计 + 距离分布 + 局部先验。"""
    ch: list[int] = []
    cn: list[int] = []
    dists: list[int] = []
    lp = None
    for tb in tasks:
        h, n = per_cluster_hits(model, tb, keep, device, max_chunk)
        ch.extend(h)
        cn.extend(n)
        dists.extend(int(v) for v in tb.dists.flatten().tolist())
        if lp is None:
            lp = local_prior_accuracy(tb, k=8)["accuracy"]
    row = ra.arm_reading(ch, cn, n_boot=2000, seed=2631)
    row["query_key_distance"] = {
        "min": min(dists), "max": max(dists),
        "median": sorted(dists)[len(dists) // 2], "n": len(dists),
        "frac_outside_window": round(sum(1 for d in dists if d >= 16) / len(dists), 6)}
    row["local_prior"] = lp
    row["n_batches"] = len(tasks)
    return row


def strip_raw(row: dict) -> dict:
    """落 `results.json` 前剥掉逐簇原始列表（它们进 `cluster_hits.json` 单独存证）。"""
    return {k: v for k, v in row.items() if k not in ("cluster_hits", "cluster_n")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["eval"]["n_batches"] = 1
        cfg["eval"]["batch_size"] = 8
        cfg["aux_ood"]["n_batches"] = 1
        cfg["aux_ood"]["batch_size"] = 8
        cfg["sensitivity_K4"]["n_batches"] = 1
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    K, Q, gap = cfg["task"]["K"], cfg["task"]["Q"], cfg["task"]["gap"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    # ---------------- 加载六臂（sha256 断言） ----------------
    note("加载六臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(
            spec["name"], spec, cfg["model"], cfg["data"]["vocab_size"], dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:16]}… "
             f"carry={spec['rule']} params={ck[spec['name']]['n_params']}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors}}

    # ---------------- ① 仪器自检 ----------------
    note("自检①：释放版(is_causal) vs 显式因果掩码；模型自带窗口掩码 vs 掩码仪器")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T),
                            generator=g).to(dev)
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))
    with torch.no_grad():
        a1 = masked_logits(arms["none_s0"], x_probe, None)
        a2 = masked_logits(arms["none_s0"], x_probe, exact)
    release_vs_exact = float((a1 - a2).abs().max())
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    with torch.no_grad():
        b1 = arms["none_s0"](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms["none_s0"], x_probe, keep)
    win_selfcheck = float((b1 - b2).abs().max())
    n_visible = int(keep.sum().item())
    assert n_visible < T * (T + 1) // 2, "窗口掩码在 T=256 上是恒等算子（退化点）"
    note(f"  max|Δlogit| 释放 vs 显式 = {release_vs_exact:.3e}；"
         f"窗口前向 vs 掩码仪器 = {win_selfcheck:.3e}；可见对数 {n_visible}")

    eq, trunk, id_start = {}, {}, {}
    for seed in ra.SEEDS:
        built = {}
        for rule in ra.RULES:
            torch.manual_seed(seed)
            built[rule] = DepthARModel(
                DepthARConfig(**{**cfg["model"], **ra.model_config_kwargs(rule)}),
                cfg["data"]["vocab_size"])
            built[rule].eval()
        eq[seed] = param_equality(built["plain"], built["delta"])
        eq[seed]["built_for_check"] = "同 seed 重新构造（训练后的权重当然不同）"
        trunk[seed] = trunk_equality(built["plain"], built["none"])
        trunk[seed]["built_for_check"] = "同 seed 重新构造"
        with torch.no_grad():
            l0 = built["none"](x_probe.cpu())[1]["depth_logits"][-1]
            id_start[seed] = {
                "plain_matches_none_bitwise": torch.equal(
                    l0, built["plain"](x_probe.cpu())[1]["depth_logits"][-1]),
                "delta_matches_none_bitwise": torch.equal(
                    l0, built["delta"](x_probe.cpu())[1]["depth_logits"][-1])}
        note(f"  seed={seed}: plain/delta 逐位等参数={eq[seed]['bitwise_equal']} "
             f"（{eq[seed]['n_params_a']}）；通道臂主干==无通道={trunk[seed]['trunk_bitwise_equal']}"
             f"（+{trunk[seed]['n_params_added']} 参数）；单边零起步={id_start[seed]}")
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": release_vs_exact,
        "model_window_forward_vs_mask_instrument_max_abs_logit": win_selfcheck,
        "window_mask_visible_pairs": n_visible,
        "window_mask_nondegenerate": True,
        "params_bitwise_equal_plain_delta": eq,
        "trunk_bitwise_equal_carry_vs_none": trunk,
        "identity_start_logits_bitwise": id_start,
    }

    # ---------------- ②③ 主腿（分布内多查询）+ 毁源对照 ----------------
    e = cfg["eval"]
    bs, nb = e["batch_size"], e["n_batches"]
    note(f"判据①：主腿 = 分布内多查询关联回忆（K={K}/Q={Q}/gap={gap}，"
         f"{nb} 批 × {bs} 序列 × {Q} 查询 = {nb * bs * Q} 次提问）")
    main_tasks = [assoc_recall_multi_batch(
        bs, T, K, Q, sym_vocab=sym, filler_token=fill, cue_token=cue,
        generator=torch.Generator().manual_seed(e["gen_seed"] + i))
        for i in range(nb)]
    cor_tasks = [corrupt_span(tb, 0, 2 * K,
                              generator=torch.Generator().manual_seed(
                                  e["corrupt_seed"] + i))
                 for i, tb in enumerate(main_tasks)]
    d_min = min(int(tb.dists.min()) for tb in main_tasks)
    d_max = max(int(tb.dists.max()) for tb in main_tasks)
    assert d_max <= TRAIN_MAX_DIST, f"主腿最远距离 {d_max} > 训练上限 {TRAIN_MAX_DIST}"
    note(f"  主腿查询距离 {d_min}–{d_max}（训练上限 {TRAIN_MAX_DIST} ⇒ **分布内**）；"
         f"chance={main_tasks[0].chance():.6f}；"
         f"局部先验={local_prior_accuracy(main_tasks[0], k=8)['accuracy']:.6f}")

    legs: dict[str, dict] = {"main": {}, "corrupt": {}}
    raw: dict[str, dict] = {"main": {}, "corrupt": {}}
    for name, model in arms.items():
        legs["main"][name] = eval_leg(model, main_tasks, keep, dev)
        legs["corrupt"][name] = eval_leg(model, cor_tasks, keep, dev)
        raw["main"][name] = {"cluster_hits": legs["main"][name]["cluster_hits"],
                             "cluster_n": legs["main"][name]["cluster_n"]}
        raw["corrupt"][name] = {"cluster_hits": legs["corrupt"][name]["cluster_hits"],
                                "cluster_n": legs["corrupt"][name]["cluster_n"]}
        note(f"  {name}: 主 {legs['main'][name]['accuracy']:.4f} "
             f"聚类CI{legs['main'][name]['clustered_ci']} "
             f"(Wilson 偏窄{legs['main'][name]['wilson_ci_naive']}) | "
             f"毁源 {legs['corrupt'][name]['accuracy']:.4f}")

    # ---------------- ⑤ 辅助腿（分布外，不作判据） ----------------
    o = cfg["aux_ood"]
    note(f"辅助腿（分布外，§17.18，不作判据）：单查询关联回忆 K={K}，"
         f"{o['n_batches']} 批 × {o['batch_size']} 序列")
    ood_tasks = [assoc_recall_batch(o["batch_size"], T, K, sym_vocab=sym,
                                    filler_token=fill, cue_token=cue,
                                    generator=torch.Generator().manual_seed(
                                        o["gen_seed"] + i))
                 for i in range(o["n_batches"])]
    od = min(int(tb.dists.min()) for tb in ood_tasks)
    ox = max(int(tb.dists.max()) for tb in ood_tasks)
    assert od > TRAIN_MAX_DIST, f"辅助腿距离 {od} 未越界（不能当 OOD 腿）"
    ood_raw = {}
    for name, model in arms.items():
        legs.setdefault("aux_ood", {})[name] = eval_leg(model, ood_tasks, keep, dev)
        ood_raw[name] = {"cluster_hits": legs["aux_ood"][name]["cluster_hits"],
                         "cluster_n": legs["aux_ood"][name]["cluster_n"]}
        note(f"  {name}: 越界腿 {legs['aux_ood'][name]['accuracy']:.4f} "
             f"聚类CI{legs['aux_ood'][name]['clustered_ci']}")
    raw["aux_ood"] = ood_raw

    # ---------------- K 敏感性（非判据：K 变化即难度变化） ----------------
    s = cfg["sensitivity_K4"]
    sQ = int(s.get("Q", Q))
    sens_tasks = [assoc_recall_multi_batch(s["batch_size"], T, s["K"], sQ,
                                           sym_vocab=sym,
                                           filler_token=fill, cue_token=cue,
                                           generator=torch.Generator().manual_seed(
                                               s["gen_seed"] + i))
                  for i in range(s["n_batches"])]
    legs["sens_K4"] = {}
    raw["sens_K4"] = {}
    for name, model in arms.items():
        legs["sens_K4"][name] = eval_leg(model, sens_tasks, keep, dev)
        raw["sens_K4"][name] = {"cluster_hits": legs["sens_K4"][name]["cluster_hits"],
                                "cluster_n": legs["sens_K4"][name]["cluster_n"]}
    note("  K=%d/Q=%d 敏感性（非判据）：" % (s["K"], sQ) + " ".join(
        f"{n}:{legs['sens_K4'][n]['accuracy']:.4f}" for n in arms))

    # ---------------- 活性 / loss / 显存 ----------------
    note("活性量与训练侧读数")
    read_mass, liveness, slopes, tail, train_meta = {}, {}, {}, {}, {}
    for name, model in arms.items():
        read_mass[name] = carry_read_mass_stats(model, x_probe, window=window)
        liveness[name] = carry_liveness(model, None, None,
                                        prob=carry_probe(model, x_probe)
                                        if ck[name]["carry_rule"] else None)
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        res_p = json.load(open(resolve(spec["train_results_json"])))
        lines = [json.loads(l) for l in open(resolve(spec["train_log"]))
                 if l.strip()]
        slopes[name] = loss_slope(lines)
        ans = [(r["step"], r["synth_answer_ce"]) for r in lines
               if r.get("synth_answer_ce") is not None]
        ans_slope = loss_slope([{"step": st, "loss": v} for st, v in ans],
                               lo=ans[len(ans) // 2][0], hi=ans[-1][0])
        chance_ce = lines[-1].get("synth_chance_ce")
        tail[name] = {
            "last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
            "last_answer_ce": lines[-1].get("synth_answer_ce"),
            "chance_ce": chance_ce,
            "escaped_chance_line": bool(
                lines[-1].get("synth_answer_ce") is not None and chance_ce is not None
                and lines[-1]["synth_answer_ce"] < chance_ce - 0.5),
            "answer_ce_last_window_slope": ans_slope.get("slope"),
            "answer_ce_window_steps": ans_slope.get("window_steps"),
            "last_read_share": lines[-1].get("attn_carry"),
        }
        train_meta[name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "seed": res_p.get("train", {}).get("seed"),
            "carry_write": res_p.get("carry_write"),
            "window_mask": res_p.get("window_mask"),
            "synth_task_config": res_p.get("synth_task_config"),
        }
        note(f"  {name}: wall={train_meta[name]['wall_minutes']}min "
             f"peak={train_meta[name]['peak_gpu_mem_mb']}MB "
             f"answerCE末={tail[name]['last_answer_ce']} "
             f"(chance {tail[name]['chance_ce']}) 逃出瞎猜线="
             f"{tail[name]['escaped_chance_line']} "
             f"read_share={read_mass[name]['read_share_per_layer']}")

    # ---------------- 判据①②③：逐种子效应 + 跨种子合并 ----------------
    note("判据①②③：逐种子效应 + 跨种子极差/方向一致性")
    pairs = [("delta_vs_none", "none", "delta", ra.DELTA_VS_NONE_MIN_PP),
             ("delta_vs_plain", "plain", "delta", ra.DELTA_VS_PLAIN_MIN_PP),
             ("plain_vs_none", "none", "plain", None)]
    effects = {}
    for pname, base_rule, tgt_rule, thr in pairs:
        per_seed = []
        for seed in ra.SEEDS:
            b = ra.arm_name(base_rule, seed)
            t = ra.arm_name(tgt_rule, seed)
            eff = ra.effect(legs["main"][b], legs["main"][t], n_boot=2000, seed=2631)
            eff["seed"] = seed
            eff["base_arm"], eff["target_arm"] = b, t
            per_seed.append(eff)
        row = {"per_seed": per_seed,
               "range_pp": ra.pp_range([x["effect_pp"] for x in per_seed]),
               "mean_pp": ra.pp_mean([x["effect_pp"] for x in per_seed]),
               "sd_pp": ra.pp_sd([x["effect_pp"] for x in per_seed]),
               "sign_consistent": ra.sign_consistent([x["effect_pp"] for x in per_seed])}
        if thr is not None:
            row["stability"] = ra.effect_stable(per_seed, thr)
        effects[pname] = row
        note(f"  {pname}: " + " | ".join(
            f"seed{x['seed']} Δ={x['effect_pp']:+.3f}pp "
            f"配对CI{x['paired_boot_ci_pp']}" for x in per_seed)
            + f" ⇒ 极差 {row['range_pp']}pp，同号={row['sign_consistent']}")

    arm_acc_by_seed = {rule: [legs["main"][ra.arm_name(rule, sd)]["accuracy"]
                              for sd in ra.SEEDS] for rule in ra.RULES}
    acc_spread = {rule: {"per_seed": vals, "range_pp": ra.pp_range(vals),
                         "mean_pp": ra.pp_mean(vals), "sd_pp": ra.pp_sd(vals)}
                  for rule, vals in arm_acc_by_seed.items()}
    best_per_seed = {seed: max(ra.RULES,
                               key=lambda r: legs["main"][ra.arm_name(r, seed)]["accuracy"])
                     for seed in ra.SEEDS}
    chance = main_tasks[0].chance()
    corrupt_ctrl = ra.corrupt_control_reading(
        {n: {"main": legs["main"][n], "corrupt": legs["corrupt"][n]} for n in arms},
        chance)
    premise = {str(seed): {
        "arm": ra.arm_name("none", seed),
        "accuracy": legs["main"][ra.arm_name("none", seed)]["accuracy"],
        "clustered_ci": legs["main"][ra.arm_name("none", seed)]["clustered_ci"],
        "chance": chance,
        "significantly_above_chance":
            legs["main"][ra.arm_name("none", seed)]["clustered_ci"][0] > chance}
        for seed in ra.SEEDS}

    criteria = {
        "c1_premise_none_near_chance":
            not any(v["significantly_above_chance"] for v in premise.values()),
        "c2_delta_vs_none_stable":
            effects["delta_vs_none"]["stability"]["stable"],
        "c3_delta_vs_plain_stable":
            effects["delta_vs_plain"]["stability"]["stable"],
        "c4_corrupt_control_ok": corrupt_ctrl["ok"],
        "c5_no_extrapolation": True,
    }
    note(f"判据① 前提：none 臂 ≈ chance ⇒ {criteria['c1_premise_none_near_chance']}")
    note(f"判据② delta>none（≥5pp 且配对 CI 下界>0，逐种子）："
         f"{criteria['c2_delta_vs_none_stable']}")
    note(f"判据③ delta>plain（≥3pp 且配对 CI 下界>0，逐种子）："
         f"{criteria['c3_delta_vs_plain_stable']}")
    note(f"判据④ 毁源对照：{corrupt_ctrl['ok']}（作废臂 {corrupt_ctrl['void_arms']}）")
    note(f"判据③ 逐种子最好臂：{best_per_seed}")

    results.update({
        "main_leg": {k: strip_raw(v) for k, v in legs["main"].items()},
        "corrupt_leg": {k: strip_raw(v) for k, v in legs["corrupt"].items()},
        "aux_ood_leg": {k: strip_raw(v) for k, v in legs["aux_ood"].items()},
        "sens_K4_leg": {k: strip_raw(v) for k, v in legs["sens_K4"].items()},
        "leg_definitions": {
            "main": {"generator": "assoc_recall_multi_batch", "K": K, "Q": Q,
                     "gap": gap, "batch_size": bs, "n_batches": nb,
                     "n_queries": nb * bs * Q,
                     "query_key_distance": [d_min, d_max],
                     "in_training_distribution": d_max <= TRAIN_MAX_DIST,
                     "gen_seed": e["gen_seed"]},
            "corrupt": {"span": [0, 2 * K], "gen_seed": e["corrupt_seed"],
                        "note": "源片段换全新随机符号，其余逐位不变"},
            "aux_ood": {"generator": "assoc_recall_batch", "K": K,
                        "batch_size": o["batch_size"], "n_batches": o["n_batches"],
                        "query_key_distance": [od, ox],
                        "in_training_distribution": False,
                        "role": "辅助腿（§17.18；不作判据）"},
            "sens_K4": {"generator": "assoc_recall_multi_batch", "K": s["K"],
                        "Q": int(s.get("Q", Q)),
                        "role": "敏感性（K/Q 变化即难度变化；不作判据）"},
        },
        "effects": effects, "arm_accuracy_spread_across_seeds": acc_spread,
        "best_arm_per_seed": {str(k): v for k, v in best_per_seed.items()},
        "premise_check": premise, "corrupt_control": corrupt_ctrl,
        "criteria": criteria,
        "liveness": {k: {"carry_liveness": liveness[k], "read_mass": read_mass[k]}
                     for k in arms},
        "loss_slope": slopes, "train_log_tail": tail, "train_meta": train_meta,
        "memory_analytic": analytic_memory(
            {**cfg["model"], "n_params": train_meta["none_s0"]["params"]},
            bs, window, anchors),
    })

    # ---------------- 复现性：同一批位置再评一遍（纯推理确定性） ----------------
    note("复现性：主腿第一批再评一遍（同权重同批位置）")
    repro = {}
    for name in ("none_s0", "delta_s1"):
        h1, _ = per_cluster_hits(arms[name], main_tasks[0], keep, dev)
        h2, _ = per_cluster_hits(arms[name], main_tasks[0], keep, dev)
        repro[name] = {"n": main_tasks[0].B * main_tasks[0].M,
                       "hits_1": sum(h1), "hits_2": sum(h2),
                       "bitwise_same": h1 == h2}
    results["reproducibility"] = {
        "deterministic_reeval": repro,
        "note": "同权重、同批位置把主腿第一批再评一遍（纯推理确定性）；"
                "训练侧的跨种子散布才是判据②的方差来源"}

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: train_meta[n]["wall_minutes"] for n in arms},
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: train_meta[n]["peak_gpu_mem_mb"] for n in arms},
        "n_forward_batches": (nb * 2 + o["n_batches"] + s["n_batches"]) * len(arms),
    }
    results["code_hashes"] = {
        rel: sha256_file(os.path.join(ROOT, rel)) for rel in
        ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/longrange_tasks.py",
         "tools/evalkit/carry_channel.py", "tools/evalkit/retention.py")
        if os.path.exists(os.path.join(ROOT, rel))}
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "reseed_arms.py", "config.json")})

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "cluster_hits.json"), "w") as f:
        json.dump(raw, f)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"落盘：{args.out_dir}/results.json + cluster_hits.json")


if __name__ == "__main__":
    main()
