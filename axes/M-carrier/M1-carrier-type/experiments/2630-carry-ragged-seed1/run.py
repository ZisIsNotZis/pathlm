#!/usr/bin/env python3
"""2630-carry-ragged-seed1 · seed=1 复现 + 跨 seed（n=2）合并 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 W2）。本入口做两件事：

  **A. seed=1 两臂的评测**（与 2629 同一件仪器、同一批位置）：复用 2629 `run.py`
     的评测函数（`recall_multi` / `recall_single_ood` / `per_depth_agreement` /
     `clean_bpc` / `read_train_log` / `drop_last_block`），只换权重与臂名 ⇒ 两 seed
     在**同一批位置 + 同一退出模式**上可比（成对）。
      ① 主判据 = 训练分布内多查询 assoc-mqar 召回（随机退出 = 训练同款退出模式，§17.18）；
         并列口径 = 全深度（部署全栈模式，训练时未见 ⇒ 严格 OOD）；
      ② 越界腿 = 单查询关联回忆（距离 241–255 > 训练上限 240；**外推参照，非判据**）；
      ③ 出口资产 tf@4 / tf@6（enwik8 口径 + synth 训练分布内口径）；
      ④ clean bpc（跨分布；非退化锚 + 质量代价项）；
      ⑤ 参数量 + 活性量（每 200 步）+ 收敛门禁（§17.15，答案行 CE 末窗斜率）。

  **B. 跨 seed 合并（n=2）**：读 2629（seed=0）的 `results.json`，与本次 seed=1 读数
     合并，给 W2 判据①②③：每个 seed 的 carry 效应（含 `two_proportion` 差值 CI）、
     极差/均值/sd、以及两条强断言判据（`ood_effect_robust` / `indist_redundancy_stable`）。

用法：
  python3 run.py                # 全电池（纯推理）
  python3 run.py --smoke        # 冒烟：少批（实测耗时/显存，不外推）
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARConfig, DepthARModel  # noqa: E402
from tools.evalkit.carry_channel import two_proportion, wilson_ci  # noqa: E402
import ragged_seed1_arms as sa  # noqa: E402  (把 2629 目录加进 sys.path)
from ragged_projfill_arms import (model_config_kwargs,  # noqa: E402
                                  expected_block_class)


def _load_module(name: str, path: str):
    """载入 2629 的 `run.py` 作为**同件评测仪器**（不复制，保持可追溯）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: 2629（seed=0）的评测仪器；其模块级 `from ragged_projfill_arms import ...` 由
#: `ragged_seed1_arms` 已把 2629 目录加进 `sys.path` 满足。
R2629 = _load_module("ragged2629_run", os.path.join(sa.SEED0_DIR, "run.py"))
RAND, FULL = R2629.RAND, R2629.FULL

FULL_NAME = {sa.seed1_to_seed0(n): n for n in sa.ARM_NAMES_S1}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, mcfg: dict, vocab: int, device: str, ckpt: dict):
    """加载 seed=1 一臂的权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    spec = R2629.arm_spec(sa.seed1_to_seed0(name))
    path = resolve(os.path.join(ckpt["dir"], name, "model.pt"))
    got = sha256_file(path)
    want = ckpt.get("sha256", {}).get(name)
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{name}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, **model_config_kwargs(spec)})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "seed": sa.SEED,
                   "block_class": type(model.blocks[0]).__name__,
                   "block_class_expected": expected_block_class(spec),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "carry_rule": spec["carry"],
                   "dense": bool(model.cfg.dense),
                   "proj_fill": bool(model.cfg.proj_fill),
                   "fill_kv": bool(model.cfg.fill_kv)}


def _per_seed_effect(base_hits, base_n, tgt_hits, tgt_n) -> dict:
    """单因素效应（目标 − 基准）+ `two_proportion` 差值 CI（跨 seed 判据的输入）。"""
    tp = two_proportion(base_hits, base_n, tgt_hits, tgt_n)
    return {"effect_pp": tp["delta_pp"],
            "diff_ci_lo": tp["diff_ci_95"][0] * 100.0,
            "diff_ci_hi": tp["diff_ci_95"][1] * 100.0,
            "p_value": tp["p_value"], "ci_overlap": tp["ci_overlap"],
            "base_hits": base_hits, "base_n": base_n,
            "target_hits": tgt_hits, "target_n": tgt_n}


def _counts_from_row(row: dict) -> tuple[int, int]:
    return int(row["n_hits"]), int(row["n"])


def cross_seed_block(seed1: dict, cfg: dict) -> dict:
    """W2 判据①②③：把 seed=0（2629）与 seed=1（本次）合并成 n=2。

    只读**原始计数**（hits/n、tf 一致数/n）重算效应——不引用 2629 的派生字段，
    以免口径漂移（§14）。
    """
    s0 = json.load(open(sa.SEED0_RESULTS))
    T = cfg["model"]["seq_len"]
    n_tf_enwik8 = cfg["tf_leg"]["enwik8_n_batches"] * cfg["tf_leg"]["enwik8_batch_size"] * (T - 1)

    # seed → 各腿的 (基准 hits/n, 目标 hits/n)
    def seed_row(seed: int):
        if seed == 0:
            base, tgt = sa.BASELINE_ARM, sa.TARGET_ARM
            in_rand = s0["train_task_leg"][RAND]
            in_full = s0["train_task_leg"][FULL]
            nlb = s0["train_task_leg_no_last_block"][RAND]
            ood_rand = s0["main_leg_single_query_ood"][RAND]
            ood_full = s0["main_leg_single_query_ood"][FULL]
            tf_synth = s0["tf_synth_indist"]
            tf_en = s0["tf_enwik8"]
        else:
            base, tgt = FULL_NAME[sa.BASELINE_ARM], FULL_NAME[sa.TARGET_ARM]
            in_rand = seed1["train_task_leg"][RAND]
            in_full = seed1["train_task_leg"][FULL]
            nlb = seed1["train_task_leg_no_last_block"][RAND]
            ood_rand = seed1["main_leg_single_query_ood"][RAND]
            ood_full = seed1["main_leg_single_query_ood"][FULL]
            tf_synth = seed1["tf_synth_indist"]
            tf_en = seed1["tf_enwik8"]
        out = {
            "recall_indist_random_exit": (
                _counts_from_row(in_rand[base]), _counts_from_row(in_rand[tgt])),
            "recall_indist_random_exit_no_last_block": (
                _counts_from_row(nlb[base]), _counts_from_row(nlb[tgt])),
            "recall_full_depth_deployment": (
                _counts_from_row(in_full[base]), _counts_from_row(in_full[tgt])),
            "ood_single_query_random_exit": (
                _counts_from_row(ood_rand[base]), _counts_from_row(ood_rand[tgt])),
            "ood_single_query_full_depth": (
                _counts_from_row(ood_full[base]), _counts_from_row(ood_full[tgt])),
        }
        # 出口资产：tf@4 一致率（synth 用 n_tokens；enwik8 用 批×批大小×(T−1)）
        n_synth = int(tf_synth[base]["n_tokens"])
        out["tf4_synth_indist"] = (
            (round(tf_synth[base]["accept"]["4"] * n_synth), n_synth),
            (round(tf_synth[tgt]["accept"]["4"] * n_synth), n_synth))
        out["tf4_enwik8"] = (
            (round(tf_en[base]["tf4"] * n_tf_enwik8), n_tf_enwik8),
            (round(tf_en[tgt]["tf4"] * n_tf_enwik8), n_tf_enwik8))
        return out

    legs = {}
    for leg_name in seed_row(0):
        per_seed = {}
        for seed in (0, 1):
            (bh, bn), (th, tn) = seed_row(seed)[leg_name]
            per_seed[seed] = _per_seed_effect(bh, bn, th, tn)
        legs[leg_name] = per_seed

    indist_primary = sa.indist_redundancy_stable(legs["recall_indist_random_exit"])
    ood_primary = sa.ood_effect_robust(legs["ood_single_query_full_depth"])
    return {
        "n_seeds": 2, "seeds": [0, 1],
        "seed0_source": os.path.relpath(sa.SEED0_RESULTS, ROOT),
        "seed0_source_sha256": sha256_file(sa.SEED0_RESULTS),
        "seed1_source": "本实验 results.json（同配置 seed=1）",
        "legs": legs,
        "criterion_1_per_seed_effects": {
            leg: {str(s): legs[leg][s]["effect_pp"] for s in (0, 1)} for leg in legs},
        "criterion_1_range_pp": {leg: sa.pp_range(
            [legs[leg][0]["effect_pp"], legs[leg][1]["effect_pp"]]) for leg in legs},
        "criterion_2_ood_robust_full_depth": ood_primary,
        "criterion_2_ood_robust_random_exit": sa.ood_effect_robust(
            legs["ood_single_query_random_exit"]),
        "criterion_3_indist_redundancy_stable": indist_primary,
        "criterion_3_no_last_block": sa.indist_redundancy_stable(
            legs["recall_indist_random_exit_no_last_block"]),
        "criterion_3_full_depth": sa.indist_redundancy_stable(
            legs["recall_full_depth_deployment"]),
        "no_extrapolation": ("两点（seed=0/1）/ 13M / 6k 快照；§17.11 禁外推——"
                             "不推断其它 seed、其它尺度、其它退出模式、或 window×carry 格"),
    }


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
        cfg["train_task_leg"]["n_batches"] = 1
        cfg["train_task_leg"]["batch_size"] = 16
        cfg["tf_leg"]["enwik8_n_batches"] = 1
        cfg["tf_leg"]["synth_n_batches"] = 1
        cfg["bpc_leg"]["n_batches"] = 1
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    K = 8
    L = cfg["model"]["n_layers"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], \
        cfg["tokens"]["cue_token"]
    mcfg = cfg["model"]
    tt = cfg["train_task_leg"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    free_before = None
    if torch.cuda.is_available():
        free_before = torch.cuda.mem_get_info()[0] // 2 ** 20

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    results: dict = {"config": cfg, "seed": sa.SEED}

    note("CPU 能力面自检（两臂可建 / 块类 / 主干逐位一致，复用 2629 语义层）")
    cap = R2629.capability_probe()
    results["capability_probe"] = cap
    note(f"  combo_buildable={cap['combo_buildable']} "
         f"trunk_bit_identical={cap['trunk_bit_identical']} "
         f"param_delta={cap['param_delta']}")

    note("加载 seed=1 两臂（sha256 断言）")
    arms, ck = {}, {}
    for name in sa.ARM_NAMES_S1:
        arms[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"], dev,
                                        cfg["checkpoint"])
        note(f"  {name:20s} {ck[name]['sha256'][:16]}… block={ck[name]['block_class']} "
             f"params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    base_name, tgt_name = FULL_NAME[sa.BASELINE_ARM], FULL_NAME[sa.TARGET_ARM]
    n_base = ck[base_name]["n_params"]
    results["param_declaration"] = {
        name: {"n_params": ck[name]["n_params"],
               "delta_vs_base": ck[name]["n_params"] - n_base,
               "delta_pct_vs_base": round(100.0 * (ck[name]["n_params"] - n_base)
                                          / n_base, 3)}
        for name in sa.ARM_NAMES_S1}
    results["architecture_equality"] = {
        "block_class": {n: ck[n]["block_class"] for n in sa.ARM_NAMES_S1},
        "block_class_expected": {n: ck[n]["block_class_expected"] for n in sa.ARM_NAMES_S1},
        "dense": {n: ck[n]["dense"] for n in sa.ARM_NAMES_S1},
        "proj_fill": {n: ck[n]["proj_fill"] for n in sa.ARM_NAMES_S1},
        "fill_kv": {n: ck[n]["fill_kv"] for n in sa.ARM_NAMES_S1},
    }

    # ---------------- ① 主判据：训练分布内多查询召回（随机退出）+ 毁源 ----------------
    note("判据①：多查询 assoc-mqar 召回 —— in_distribution_random_exit + full_depth（seed=1）")
    main_leg = {RAND: {}, FULL: {}}
    for mode in (RAND, FULL):
        for name in sa.ARM_NAMES_S1:
            row = R2629.recall_multi(arms[name], T, sym, fill, cue, K, tt["n_queries"],
                                     tt["gap"], dev, mode, tt["batch_size"],
                                     tt["n_batches"], 2000, L)
            main_leg[mode][name] = row
            note(f"  [{mode:28s}] {name:20s} acc={row['accuracy']:.4f} "
                 f"CI={row['accuracy_ci']} chance={row['chance']:.4f} "
                 f"lp={row['local_prior']:.4f} "
                 f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["train_task_leg"] = main_leg
    results["train_task_leg_no_last_block"] = {
        mode: {name: R2629.drop_last_block(main_leg[mode][name])
               for name in sa.ARM_NAMES_S1} for mode in (RAND, FULL)}

    # ---------------- 越界腿（单查询，距离 241–255 = 训练分布外） ----------------
    note("越界腿：单查询关联回忆（距离 241–255 = 训练分布外）—— 外推参照、非判据")
    ood = {RAND: {}, FULL: {}}
    for mode in (RAND, FULL):
        for name in sa.ARM_NAMES_S1:
            row = R2629.recall_single_ood(arms[name], T, sym, fill, cue, K, dev, mode,
                                          tt["batch_size"], tt["n_batches"], 4000, L)
            ood[mode][name] = row
            note(f"  [{mode:28s}] {name:20s} acc={row['accuracy']:.4f} "
                 f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["main_leg_single_query_ood"] = ood

    # ---------------- ② 出口资产 tf@4 / tf@6 ----------------
    note("判据②：出口资产 tf@4 / tf@6（enwik8 口径 + synth 训练分布内口径）")
    tcfg = cfg["tf_leg"]
    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    tf_enwik8, tf_synth = {}, {}
    for name in sa.ARM_NAMES_S1:
        rows = R2629.tf_acceptance(arms[name], eval_arr,
                                   n_batches=tcfg["enwik8_n_batches"],
                                   batch_size=tcfg["enwik8_batch_size"], device=dev)
        tf_enwik8[name] = {"by_depth": rows,
                           "tf4": next(r["accept"] for r in rows if r["depth"] == 4),
                           "tf6": next(r["accept"] for r in rows if r["depth"] == 6),
                           "caliber": "enwik8 teacher-forcing 代理（对齐 100M 0.9726）"}
        row = R2629.per_depth_agreement(arms[name], T, sym, fill, cue, K,
                                        tt["n_queries"], tt["gap"], dev,
                                        depths=tuple(tcfg["depths"]),
                                        bs=tcfg["synth_batch_size"],
                                        nb=tcfg["synth_n_batches"])
        tf_synth[name] = row
        note(f"  {name:20s} tf@4(enwik8)={tf_enwik8[name]['tf4']:.4f} "
             f"tf@6(enwik8)={tf_enwik8[name]['tf6']:.4f} "
             f"tf@4(synth)={row['accept']['4']:.4f} tf@6(synth)={row['accept']['6']:.4f}")
    results["tf_enwik8"] = tf_enwik8
    results["tf_synth_indist"] = tf_synth

    # ---------------- clean bpc（非退化锚 + 质量代价项） ----------------
    note("clean bpc（enwik8 全因果；跨分布 + 合成训练 ⇒ 非退化锚，不作召回判据）")
    bpc = {}
    for name in sa.ARM_NAMES_S1:
        b = R2629.clean_bpc(arms[name], eval_arr, cfg["bpc_leg"]["batch_size"],
                            cfg["bpc_leg"]["n_batches"], 2, dev)
        bpc[name] = {"bpc": round(b, 6), "n_batches": cfg["bpc_leg"]["n_batches"],
                     "batch_size": cfg["bpc_leg"]["batch_size"]}
        note(f"  {name:20s} clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    results["bpc_damage_vs_base"] = {
        n: round(bpc[n]["bpc"] - bpc[base_name]["bpc"], 6) for n in sa.ARM_NAMES_S1}
    note(f"  enwik8 vocab={bpc_vocab}（模型 vocab={cfg['data']['vocab_size']}）")

    # ---------------- 单因素增量（§17.17；seed=1 内） ----------------
    # 2629 的 `single_factor_increment` 以 seed=0 臂名（projfill/projfill_carry）为键
    # ⇒ 调用前先把 seed=1 臂名映射回 seed=0 名（复用同一件算术，不复制）。
    def inc(leg):
        return R2629.single_factor_increment_pp(
            {sa.seed1_to_seed0(n): leg[n]["accuracy"] for n in sa.ARM_NAMES_S1})

    results["increments"] = {
        "recall_random_exit": inc(main_leg[RAND]),
        "recall_random_exit_no_last_block": inc(
            results["train_task_leg_no_last_block"][RAND]),
        "recall_full_depth": inc(main_leg[FULL]),
        "single_query_ood_random_exit": inc(ood[RAND]),
        "single_query_ood_full_depth": inc(ood[FULL]),
        "tf4_synth_indist": R2629.single_factor_increment_pp(
            {sa.seed1_to_seed0(n): tf_synth[n]["accept"]["4"]
             for n in sa.ARM_NAMES_S1}),
        "tf4_enwik8": R2629.single_factor_increment_pp(
            {sa.seed1_to_seed0(n): tf_enwik8[n]["tf4"] for n in sa.ARM_NAMES_S1}),
        "bpc": R2629.single_factor_increment(
            {sa.seed1_to_seed0(n): bpc[n]["bpc"] for n in sa.ARM_NAMES_S1}),
    }
    results["increments"]["pairwise_recall_random_exit"] = two_proportion(
        main_leg[RAND][base_name]["n_hits"], main_leg[RAND][base_name]["n"],
        main_leg[RAND][tgt_name]["n_hits"], main_leg[RAND][tgt_name]["n"])
    results["increments"]["pairwise_recall_full_depth"] = two_proportion(
        main_leg[FULL][base_name]["n_hits"], main_leg[FULL][base_name]["n"],
        main_leg[FULL][tgt_name]["n_hits"], main_leg[FULL][tgt_name]["n"])
    results["increments"]["pairwise_ood_full_depth"] = two_proportion(
        ood[FULL][base_name]["n_hits"], ood[FULL][base_name]["n"],
        ood[FULL][tgt_name]["n_hits"], ood[FULL][tgt_name]["n"])

    # ---------------- 毁源对照有效性 ----------------
    destroy = {mode: {} for mode in (RAND, FULL)}
    for mode in (RAND, FULL):
        for name in sa.ARM_NAMES_S1:
            lo = main_leg[mode][name]["corrupt_source"]["accuracy_ci"][0]
            chance = main_leg[mode][name]["chance"]
            destroy[mode][name] = {
                "corrupt_accuracy": main_leg[mode][name]["corrupt_source"]["accuracy"],
                "corrupt_ci": main_leg[mode][name]["corrupt_source"]["accuracy_ci"],
                "chance": chance, "valid": R2629.destroy_source_valid(lo, chance),
                "rule": "毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废"}
    results["destroy_source_validity"] = destroy
    note(f"  毁源对照有效（CI 下界 ≤ 4×chance）：RAND="
         f"{ {n: destroy[RAND][n]['valid'] for n in sa.ARM_NAMES_S1} }")

    # ---------------- 活性量 + 收敛门禁 ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    results["train_activity"] = {
        name: R2629.read_train_log(name, train_dir, steps) for name in sa.ARM_NAMES_S1}
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）： " +
         ", ".join(f"{n}={results['train_activity'][n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in sa.ARM_NAMES_S1))

    # ---------------- 出口资产保持性（seed=1 内，与 2629 同判据） ----------------
    tf4_base = tf_synth[base_name]["accept"]["4"]
    tf4_carry = tf_synth[tgt_name]["accept"]["4"]
    tf4_e_base = tf_enwik8[base_name]["tf4"]
    tf4_e_carry = tf_enwik8[tgt_name]["tf4"]
    results["exit_asset_holds"] = {
        "primary_caliber": "synth_assoc-mqar_in_distribution",
        "tf4_indist_base": tf4_base, "tf4_indist_carry": tf4_carry,
        "delta_pp_indist": round(100 * (tf4_carry - tf4_base), 4),
        "tf4_enwik8_base": tf4_e_base, "tf4_enwik8_carry": tf4_e_carry,
        "delta_pp_enwik8": round(100 * (tf4_e_carry - tf4_e_base), 4),
        "reproduces_100m_magnitude": bool(min(tf4_base, tf4_carry) >= 0.90),
        "holds_indist": abs(100 * (tf4_carry - tf4_base)) <= 1.4,
        "holds_enwik8_proxy": abs(100 * (tf4_e_carry - tf4_e_base)) <= 1.4,
        "holds": abs(100 * (tf4_carry - tf4_base)) <= 1.4,
        "rule": ("主口径 = 训练分布内（synth）tf@4：两臂均 ≥0.90 ⇒ 100M 量级（0.9726）在本"
                 "尺度/步数下复现；|Δ| ≤ 1.4pp ⇒ 「出口资产在 carry 存在时保持」。enwik8 口径"
                 "是**跨分布代理**（本线模型在合成任务上训练），单独报、不作主口径。"),
    }
    note(f"  出口资产 tf@4（主口径 in-dist）: base={tf4_base:.4f} carry={tf4_carry:.4f} ⇒ "
         f"holds={results['exit_asset_holds']['holds']}；"
         f"enwik8 代理: base={tf4_e_base:.4f} carry={tf4_e_carry:.4f} "
         f"(Δ={results['exit_asset_holds']['delta_pp_enwik8']:+.2f}pp)")

    # ---------------- W2：跨 seed 合并（n=2） ----------------
    note("W2 判据①②③：与 2629（seed=0）合并为 n=2")
    results["cross_seed"] = cross_seed_block(results, cfg)
    cs = results["cross_seed"]
    note(f"  分布内召回（随机退出）每 seed 效应 pp："
         f"{cs['criterion_1_per_seed_effects']['recall_indist_random_exit']}，"
         f"极差 {cs['criterion_1_range_pp']['recall_indist_random_exit']}pp，"
         f"冗余稳定={cs['criterion_3_indist_redundancy_stable']['stable']}")
    note(f"  越界腿（全深度）每 seed 效应 pp："
         f"{cs['criterion_1_per_seed_effects']['ood_single_query_full_depth']}，"
         f"极差 {cs['criterion_1_range_pp']['ood_single_query_full_depth']}pp，"
         f"稳健={cs['criterion_2_ood_robust_full_depth']['robust']}")

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 3),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "free_mb_before": free_before,
        "free_mb_after": (torch.cuda.mem_get_info()[0] // 2 ** 20
                          if torch.cuda.is_available() else None),
        "pure_inference": True,
    }
    results["provenance"] = {"run_log": log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("MAINLEG(random_exit):",
          {n: main_leg[RAND][n]["accuracy"] for n in sa.ARM_NAMES_S1})
    print("OOD(full_depth):", {n: ood[FULL][n]["accuracy"] for n in sa.ARM_NAMES_S1})
    print("CROSS_SEED indist eff:", cs["criterion_1_per_seed_effects"]["recall_indist_random_exit"])
    print("CROSS_SEED ood eff:", cs["criterion_1_per_seed_effects"]["ood_single_query_full_depth"])


if __name__ == "__main__":
    main()
