#!/usr/bin/env python3
"""2633-carry-seed2 · 评测入口（纯推理）：三 seed × 三臂的判据①②③④ + 判据⑥（样本量估计）。

契约：同目录 `README.md`（§4 判据线逐字沿用 2631，未改一字）。本入口做：

  ① 仪器轻自检：九臂参数量/通道规则断言；窗口掩码非退化；同 seed 重建的 plain/delta
     **逐位等参数**（复用 `tools/evalkit/carry_channel.param_equality`）；
     （完整仪器自检在 2631，实现未变 ⇒ 直接引用其 `results.json::selfchecks`）
  ② 主腿 = **训练分布内**多查询关联回忆（与 2631 同一生成器、同一批位置 ⇒ 九臂配对）；
     主 CI = 按序列聚类的 bootstrap；Wilson 只作偏窄对照（复核 B2）；
  ③ 毁源对照臂（判据④）；
  ④ 跨种子合并（判据①②③）：逐种子效应 + 极差 + 方向一致性 + 预登记阈值（5pp/3pp）；
  ⑤ 辅助腿（分布外 241–255）与 K=4/Q=4 敏感性（不作判据）；
  ⑥ **判据⑥**：若判据③不成立 ⇒ 用 plain 臂跨种子散布做**种子层面**样本量估计 +
     「塌回地板」比例的 Wilson CI；
  ⑦ 训练侧读数（答案行 CE 末窗斜率 §17.15、loss 斜率 §16、活性量 §10）。

**复用**：主腿/毁源的评测核（`per_cluster_hits` / `eval_leg` / `load_arm`）与统计层
（`reseed_arms`）都从 2631 只读导入——同一函数、同一口径，不重写（README §9）。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~4 min）
  python3 run.py --smoke    # 冒烟：少批
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RES = os.path.abspath(os.path.join(HERE, "..", "2631-carry-reseed"))
for _p in (ROOT, HERE, RES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.depth_ar import DepthARConfig, DepthARModel, window_anchor_keep_mask  # noqa: E402
from tools.evalkit.longrange_tasks import (                          # noqa: E402
    assoc_recall_batch, assoc_recall_multi_batch, corrupt_span,
    local_prior_accuracy)
from tools.evalkit.carry_channel import (analytic_memory, loss_slope,   # noqa: E402
                                         param_equality, trunk_equality)
import reseed_arms as ra                                             # noqa: E402
import seed3_stats as s3                                             # noqa: E402

TRAIN_MAX_DIST = 240          # 训练生成器（K=8/Q=8/gap=14）能出现的最远查询距离


def _load_base_run():
    """只读导入 2631 的评测核（同一函数、同一口径）。"""
    spec = importlib.util.spec_from_file_location(
        "carry_reseed_base_run", os.path.join(RES, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# §20 记账的**唯一实现**在本目录 `lock_accounting.py`（驱动事后覆写 accounting 与 note 时
# 必须走同一处，否则会造出「同文件两处相反」的产物——复核 P1）。这里只做导入。
from lock_accounting import derive_note as _lock_note, assert_consistent as _assert_lock_consistency  # noqa: E402


def _lock_accounting() -> dict | None:
    """读评测驱动写的 `lock_accounting.json`（§20 记账的结构化产物；缺则 None）。"""
    p = os.path.join(HERE, "lock_accounting.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke", "eval") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    import equality_vs_2631 as eqrep
    base_run = _load_base_run()
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

    note("§9 复用证据：基准 = 2631 运行时版本（按记录 sha256 定位），判据 = 两层行为等价")
    equality = eqrep.compare(cfg)
    eqrep.assert_report_schema(equality)      # 消费方显式再核一次（接线可见；缺键即报错）
    note(f"  实现层行为等价：{equality['implementation_equivalent_to_2631_runtime']}；"
         f"复用范围：{equality['reuse_scope']}；"
         f"配置逐字段不同项：{equality['differing_fields'] or '无'}")

    # ---------------- 加载九臂（sha256 断言） ----------------
    note("加载九臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = base_run.load_arm(
            spec["name"], spec, cfg["model"], cfg["data"]["vocab_size"], dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:16]}… "
             f"carry={spec['rule']} params={ck[spec['name']]['n_params']}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors},
               # 指针块（不是整块内嵌）：权威产物 = equality_report.json（复核 P2-1）
               "reuse_equality_vs_2631": eqrep.pointer_block(equality)}
    # **写盘路径也设防**：内嵌指针块的标量必须与权威产物逐项一致（此前只有 pytest 调用，
    # 写盘路径不设防——复核 P2-1）
    eqrep.assert_pointer_consistent(results["reuse_equality_vs_2631"])

    # ---------------- ① 仪器轻自检 ----------------
    note("自检①：参数量/块类/窗口掩码非退化；同 seed 重建 plain/delta 逐位等参数")
    n_params = {n: ck[n]["n_params"] for n in arms}
    assert n_params["none_s0"] == n_params["none_s1"] == n_params["none_s2"]
    for seed in s3.SEEDS3:
        assert n_params[s3.arm_name("plain", seed)] == n_params[s3.arm_name("delta", seed)]
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    n_visible = int(keep.sum().item())
    assert n_visible < T * (T + 1) // 2, "窗口掩码在 T=256 上是恒等算子（退化点）"
    eq, trunk = {}, {}
    for seed in s3.SEEDS3:
        built = {}
        for rule in ra.RULES:
            torch.manual_seed(seed)
            built[rule] = DepthARModel(
                DepthARConfig(**{**cfg["model"], **ra.model_config_kwargs(rule)}),
                cfg["data"]["vocab_size"])
            built[rule].eval()
        eq[seed] = param_equality(built["plain"], built["delta"])
        trunk[seed] = trunk_equality(built["plain"], built["none"])
    results["selfchecks"] = {
        "window_mask_visible_pairs": n_visible,
        "window_mask_nondegenerate": True,
        "n_params": n_params,
        "params_bitwise_equal_plain_delta": eq,
        "trunk_bitwise_equal_carry_vs_none": trunk,
        "instrument_selfcheck_reference": (
            "2611/2631 的完整仪器自检（释放版 vs 显式因果掩码、单边零起步逐位恒等、"
            "评测掩码 == 训练掩码）——实现未变，见 "
            "axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed/results.json"
            "::selfchecks"),
    }
    note(f"  可见对数 {n_visible}；plain/delta 逐位等参数={eq[0]['bitwise_equal']} "
         f"({eq[0]['n_params_a']})；通道臂主干==无通道={trunk[0]['trunk_bitwise_equal']}")

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
    chance = main_tasks[0].chance()
    note(f"  主腿查询距离 {d_min}–{d_max}（训练上限 {TRAIN_MAX_DIST} ⇒ **分布内**）；"
         f"chance={chance:.6f}；局部先验="
         f"{local_prior_accuracy(main_tasks[0], k=8)['accuracy']:.6f}")

    legs: dict[str, dict] = {"main": {}, "corrupt": {}}
    raw: dict[str, dict] = {"main": {}, "corrupt": {}}
    for name, model in arms.items():
        legs["main"][name] = base_run.eval_leg(model, main_tasks, keep, dev)
        legs["corrupt"][name] = base_run.eval_leg(model, cor_tasks, keep, dev)
        raw["main"][name] = {"cluster_hits": legs["main"][name]["cluster_hits"],
                             "cluster_n": legs["main"][name]["cluster_n"]}
        raw["corrupt"][name] = {"cluster_hits": legs["corrupt"][name]["cluster_hits"],
                                "cluster_n": legs["corrupt"][name]["cluster_n"]}
        note(f"  {name}: 主 {legs['main'][name]['accuracy']:.4f} "
             f"聚类CI{legs['main'][name]['clustered_ci']} | "
             f"毁源 {legs['corrupt'][name]['accuracy']:.4f}")

    # ---------------- ⑤ 辅助腿（分布外）+ K 敏感性 ----------------
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
    legs["aux_ood"], raw["aux_ood"] = {}, {}
    for name, model in arms.items():
        legs["aux_ood"][name] = base_run.eval_leg(model, ood_tasks, keep, dev)
        raw["aux_ood"][name] = {"cluster_hits": legs["aux_ood"][name]["cluster_hits"],
                                "cluster_n": legs["aux_ood"][name]["cluster_n"]}
    note("  越界腿：" + " ".join(f"{n}:{legs['aux_ood'][n]['accuracy']:.4f}"
                                 for n in arms))
    s = cfg["sensitivity_K4"]
    sQ = int(s.get("Q", Q))
    sens_tasks = [assoc_recall_multi_batch(s["batch_size"], T, s["K"], sQ,
                                           sym_vocab=sym, filler_token=fill,
                                           cue_token=cue,
                                           generator=torch.Generator().manual_seed(
                                               s["gen_seed"] + i))
                  for i in range(s["n_batches"])]
    legs["sens_K4"], raw["sens_K4"] = {}, {}
    for name, model in arms.items():
        legs["sens_K4"][name] = base_run.eval_leg(model, sens_tasks, keep, dev)
        raw["sens_K4"][name] = {"cluster_hits": legs["sens_K4"][name]["cluster_hits"],
                                "cluster_n": legs["sens_K4"][name]["cluster_n"]}
    note("  K=%d/Q=%d 敏感性（非判据）：" % (s["K"], sQ) + " ".join(
        f"{n}:{legs['sens_K4'][n]['accuracy']:.4f}" for n in arms))

    # ---------------- 训练侧读数 ----------------
    note("训练侧读数（答案行 CE 末窗斜率 / loss 斜率 / 活性量）")
    slopes, tail, train_meta = {}, {}, {}
    for name in arms:
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        res_p = json.load(open(resolve(spec["train_results_json"])))
        lines = [json.loads(l) for l in open(resolve(spec["train_log"])) if l.strip()]
        slopes[name] = loss_slope(lines)
        ans = [(r["step"], r["synth_answer_ce"]) for r in lines
               if r.get("synth_answer_ce") is not None]
        ans_slope = (loss_slope([{"step": st, "loss": v} for st, v in ans],
                                lo=ans[len(ans) // 2][0], hi=ans[-1][0])
                     if len(ans) >= 2 else {})
        train_meta[name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "seed": res_p.get("train", {}).get("seed"),
            "carry_write": res_p.get("carry_write"),
            "window_mask": res_p.get("window_mask"),
            "synth_task_config": res_p.get("synth_task_config"),
            # SCHEMA §10 的活性块：训练期的真实 `_carry_tele`（无通道臂为 enabled=False）
            "carry_liveness_training": res_p.get("carry_liveness"),
            "origin": ("2633（本实验新训）" if name in cfg["new_in_2633"] else "2631（复用）"),
        }
        tail[name] = {
            "last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
            "last_answer_ce": lines[-1].get("synth_answer_ce"),
            "chance_ce": lines[-1].get("synth_chance_ce"),
            "escaped_chance_line": bool(
                lines[-1].get("synth_answer_ce") is not None
                and lines[-1].get("synth_chance_ce") is not None
                and lines[-1]["synth_answer_ce"] < lines[-1]["synth_chance_ce"] - 0.5),
            "answer_ce_last_window_slope": ans_slope.get("slope"),
            "last_read_share": lines[-1].get("attn_carry"),
        }
        note(f"  {name}[{train_meta[name]['origin']}]: "
             f"wall={train_meta[name]['wall_minutes']}min "
             f"peak={train_meta[name]['peak_gpu_mem_mb']}MB "
             f"answerCE末={tail[name]['last_answer_ce']} (chance "
             f"{tail[name]['chance_ce']}) 逃出瞎猜线={tail[name]['escaped_chance_line']}")

    # ---------------- 判据①②③：逐种子效应 + 三 seed 合并 ----------------
    note("判据①②③：逐种子效应 + 跨种子极差/方向一致性（三 seed）")
    pairs = [("delta_vs_none", "none", "delta", ra.DELTA_VS_NONE_MIN_PP),
             ("delta_vs_plain", "plain", "delta", ra.DELTA_VS_PLAIN_MIN_PP),
             ("plain_vs_none", "none", "plain", None)]
    effects = {}
    for pname, base_rule, tgt_rule, thr in pairs:
        per_seed = []
        for seed in s3.SEEDS3:
            b = s3.arm_name(base_rule, seed)
            t = s3.arm_name(tgt_rule, seed)
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

    arm_acc_by_seed = {rule: [legs["main"][s3.arm_name(rule, sd)]["accuracy"]
                              for sd in s3.SEEDS3] for rule in ra.RULES}
    acc_spread = {rule: {"per_seed": vals,
                         "per_seed_pp": [round(100 * v, 3) for v in vals],
                         "range_pp": ra.pp_range([100 * v for v in vals]),
                         "mean_pp": ra.pp_mean([100 * v for v in vals]),
                         "sd_pp": ra.pp_sd([100 * v for v in vals]),
                         "mean_ci95_t": s3.mean_ci_t([100 * v for v in vals])}
                  for rule, vals in arm_acc_by_seed.items()}
    best_per_seed = {seed: max(ra.RULES,
                               key=lambda r: legs["main"][s3.arm_name(r, seed)]["accuracy"])
                     for seed in s3.SEEDS3}
    corrupt_ctrl = ra.corrupt_control_reading(
        {n: {"main": legs["main"][n], "corrupt": legs["corrupt"][n]} for n in arms},
        chance)
    premise = {str(seed): {
        "arm": s3.arm_name("none", seed),
        "accuracy": legs["main"][s3.arm_name("none", seed)]["accuracy"],
        "clustered_ci": legs["main"][s3.arm_name("none", seed)]["clustered_ci"],
        "chance": chance,
        "significantly_above_chance":
            legs["main"][s3.arm_name("none", seed)]["clustered_ci"][0] > chance}
        for seed in s3.SEEDS3}

    # ---------------- 判据⑥：样本量估计 + 塌回地板比例 ----------------
    note("判据⑥：种子层面样本量估计 + plain「塌回地板」比例")
    acc_pp = {rule: [100 * v for v in vals] for rule, vals in arm_acc_by_seed.items()}
    floor_line_pp = 4 * chance * 100                       # 4×chance = 6.25pp 地板线
    collapse = {rule: s3.collapse_wilson([v < floor_line_pp for v in acc_pp[rule]])
                for rule in ra.RULES}
    dp = effects["delta_vs_plain"]
    sample_size = {
        "target_min_pp": ra.DELTA_VS_PLAIN_MIN_PP,
        "plain_spread": dp["range_pp"],
        "plain_sd_pp": acc_spread["plain"]["sd_pp"],
        "delta_sd_pp": acc_spread["delta"]["sd_pp"],
        "seeds_needed_using_plain_sd": s3.seeds_needed_for_mean_effect(
            acc_spread["plain"]["sd_pp"], ra.DELTA_VS_PLAIN_MIN_PP),
        "seeds_needed_using_delta_sd": s3.seeds_needed_for_mean_effect(
            acc_spread["delta"]["sd_pp"], ra.DELTA_VS_PLAIN_MIN_PP),
        "note": ("「种子层面」均值检验的样本量（80% 功效、α=0.05 双侧、正态近似，"
                 "σ 由 n=3 的样本 sd 估计 ⇒ 自身很不可靠，只作规划量级）；"
                 "与它**并列**报「塌回地板比例」的 Wilson CI——后者才是 plain 双稳态的"
                 "正确统计量（判据③不成立的机制是 plain 会塌，不是均值偏移）。"),
    }
    note(f"  plain 跨种子 {acc_spread['plain']['per_seed_pp']}pp；"
         f"样本 sd={acc_spread['plain']['sd_pp']}pp ⇒ 判 3pp 均值差需 ≈"
         f"{sample_size['seeds_needed_using_plain_sd']} 种子；"
         f"delta 侧 ≈{sample_size['seeds_needed_using_delta_sd']} 种子")
    note(f"  塌回地板（<{floor_line_pp:.2f}pp）："
         + " ".join(f"{r}:{collapse[r]['rate']:.3f}{collapse[r]['wilson_ci']}"
                    for r in ra.RULES))

    criteria = {
        "c1_premise_none_near_chance":
            not any(v["significantly_above_chance"] for v in premise.values()),
        "c2_delta_vs_none_stable":
            effects["delta_vs_none"]["stability"]["stable"],
        "c3_delta_vs_plain_stable":
            effects["delta_vs_plain"]["stability"]["stable"],
        "c4_corrupt_control_ok": corrupt_ctrl["ok"],
        "c5_no_extrapolation": True,
        "c6_sample_size_reported": True,
    }
    note(f"判据① 前提 none≈chance 三臂 ⇒ {criteria['c1_premise_none_near_chance']}")
    note(f"判据② delta>none（≥5pp 且配对 CI 下界>0，逐种子，三 seed）："
         f"{criteria['c2_delta_vs_none_stable']}")
    note(f"判据③ delta>plain（≥3pp 且配对 CI 下界>0，逐种子，三 seed）："
         f"{criteria['c3_delta_vs_plain_stable']}")
    note(f"判据④ 毁源对照：{corrupt_ctrl['ok']}（作废臂 {corrupt_ctrl['void_arms']}）")
    note(f"逐种子最好臂：{best_per_seed}")

    results.update({
        "main_leg": {k: base_run.strip_raw(v) for k, v in legs["main"].items()},
        "corrupt_leg": {k: base_run.strip_raw(v) for k, v in legs["corrupt"].items()},
        "aux_ood_leg": {k: base_run.strip_raw(v) for k, v in legs["aux_ood"].items()},
        "sens_K4_leg": {k: base_run.strip_raw(v) for k, v in legs["sens_K4"].items()},
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
                        "Q": sQ, "role": "敏感性（K/Q 变化即难度变化；不作判据）"},
        },
        "effects": effects, "arm_accuracy_spread_across_seeds": acc_spread,
        "best_arm_per_seed": {str(k): v for k, v in best_per_seed.items()},
        "premise_check": premise, "corrupt_control": corrupt_ctrl,
        "sample_size_analysis": sample_size, "collapse_to_floor": collapse,
        "criteria": criteria,
        "loss_slope": slopes, "train_log_tail": tail, "train_meta": train_meta,
        "memory_analytic": analytic_memory(
            {**cfg["model"], "n_params": train_meta["none_s0"]["params"]},
            bs, window, anchors),
    })

    # ---------------- 复现性：同一批位置再评一遍 ----------------
    note("复现性：主腿第一批再评一遍（同权重同批位置）")
    repro = {}
    for name in ("none_s2", "delta_s2"):
        h1, _ = base_run.per_cluster_hits(arms[name], main_tasks[0], keep, dev)
        h2, _ = base_run.per_cluster_hits(arms[name], main_tasks[0], keep, dev)
        repro[name] = {"n": main_tasks[0].B * main_tasks[0].M,
                       "hits_1": sum(h1), "hits_2": sum(h2), "bitwise_same": h1 == h2}
    results["reproducibility"] = {"deterministic_reeval": repro}

    acct = _lock_accounting()
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: train_meta[n]["wall_minutes"] for n in arms},
        "new_arms_train_wall_minutes": {n: train_meta[n]["wall_minutes"]
                                        for n in cfg["new_in_2633"]},
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: train_meta[n]["peak_gpu_mem_mb"] for n in arms},
        "n_forward_batches": (nb * 2 + o["n_batches"] + s["n_batches"]) * len(arms),
        "lock_note": _lock_note(acct),
        "lock_accounting": acct,
    }
    _assert_lock_consistency(results["cost"])      # 不一致 ⇒ 落盘前就炸（复核 P1 护栏）
    results["code_hashes"] = {
        rel: sha256_file(os.path.join(ROOT, rel)) for rel in
        ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/longrange_tasks.py",
         "tools/evalkit/carry_channel.py", "tools/evalkit/retention.py")
        if os.path.exists(os.path.join(ROOT, rel))}
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "seed3_stats.py", "equality_vs_2631.py", "config.json")})
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(RES, rel)) for rel in
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
