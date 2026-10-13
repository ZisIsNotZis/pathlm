#!/usr/bin/env python3
"""2623-composable-arms-13m · ENG1 解锁的组合阶梯 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 Y1 契约，§4c 记录 6k 快照 token 预算
的裁决）。本入口只做**评测**（不训练、不改权重），把四臂（`stock` / `+GQA` /
`+GQA+carry` / `+GQA+carry+window`）在两条腿上量出来：

  ① 合成关联回忆（主腿，SCHEMA §19：不用 enwik8 当长程主任务）——每臂在**自己训练
     时的可见性**下评测（训练/评测同一掩码），并**另跑**一套全因果交叉口径；
  ② clean bpc（enwik8 评测尾段，非退化锚——合成训练 ⇒ 真实文本上的通用 LM 能力本就
     未建立，此腿只证明「叠加特性没把主干弄坏」，**不作质量判据**）；
  ③ 逐叠加步的增量（§17.17 分量分解：净和 + 相邻步两列）；
  ④ 毁源对照臂（把源 KV 段落毁掉 ⇒ 读数若仍高则作废）；
  ⑤ 参数量差（GQA −8.75% 等，按 §15 声明是否混淆归因）；
  ⑥ 活性量（KV 头形状 / carry read_share / window mask，来自每臂 200 步训练日志）；
  ⑦ 收敛门禁（§17.15：落在决定判据的 `synth_answer_ce` 上）。

用法：
  python3 run.py                # 全电池（纯推理，共享卡 ~4–8 min）
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
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,   # noqa: E402
                             kv_head_liveness, window_anchor_keep_mask,
                             window_mask_liveness)
from tools.evalkit.longrange_tasks import (                 # noqa: E402
    accuracy, assoc_recall_batch, assoc_recall_multi_batch, corrupt_span,
    local_prior_accuracy, var_recall_batch)
from tools.evalkit.carry_channel import wilson_ci, two_proportion  # noqa: E402
from tools.evalkit.retention import masked_bpc              # noqa: E402
from arms import (ARMS, ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                  arm_spec, expected_block_class, model_config_kwargs,
                  native_visibility, net_vs_baseline, staircase_increments)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, mcfg: dict, vocab: int, device: str, ckpt: dict):
    """加载一臂的权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    spec = arm_spec(name)
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
                   "block_class": type(model.blocks[0]).__name__,
                   "block_class_expected": expected_block_class(spec),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "flags": model_config_kwargs(spec)}


def _keep(T: int, window: int, anchors: int, device: str):
    return (window_anchor_keep_mask(T, window, anchors, device=device)
            if window > 0 else None)


def _agg_dist(agg: dict, main: dict) -> None:
    """把一次 `accuracy()` 的 by_distance 累加进 `agg['by_dist']`。"""
    for d, v in main["by_distance"].items():
        slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
        slot["n"] += v["n"]
        slot["n_hits"] += round(v["accuracy"] * v["n"])


def _by_dist_out(agg: dict) -> dict:
    return {d: {"accuracy": round(s["n_hits"] / s["n"], 6), "n": s["n"]}
            for d, s in sorted(agg["by_dist"].items(), key=lambda kv: int(kv[0]))}


def recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K,
                device, corrupt_bounds=(0, 16), max_chunk=64, seed_base=1000):
    """一臂在一个任务配置上的关联回忆：主腿 + 毁源对照 + 局部先验 + 距离分布。"""
    keep = _keep(T, window, anchors, device)
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
    local_prior = chance = None
    for i in range(nb):
        tb = assoc_recall_batch(bs, T, K, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(
                                    seed_base + 7 * K + i))
        main = accuracy(model, tb, keep_mask=keep, device=device, max_chunk=max_chunk)
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        c = accuracy(model, bad, keep_mask=keep, device=device, max_chunk=max_chunk)
        agg["n"] += main["n"]; agg["n_hits"] += main["n_hits"]
        agg["cn"] += c["n"]; agg["chits"] += c["n_hits"]
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        _agg_dist(agg, main)
        if local_prior is None:
            local_prior = local_prior_accuracy(tb, k=8)["accuracy"]
            chance = round(tb.chance(), 8)
    dists = agg["dists"]
    return {
        "K": K, "n_batches": nb, "n": agg["n"], "n_hits": agg["n_hits"],
        "accuracy": round(agg["n_hits"] / agg["n"], 6),
        "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
        "chance": chance, "local_prior": local_prior,
        "by_distance": _by_dist_out(agg),
        "corrupt_source": {"span": list(corrupt_bounds), "n": agg["cn"],
                           "n_hits": agg["chits"],
                           "accuracy": round(agg["chits"] / agg["cn"], 6),
                           "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
        "query_key_distance": {
            "min": min(dists), "max": max(dists),
            "median": sorted(dists)[len(dists) // 2],
            "window": window,
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


def multi_recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K, Q,
                      gap, device, corrupt_bounds=(0, 16), seed_base=2000):
    """**训练型任务**（多查询 assoc-mqar）上的召回：判据 「合成召回」的 in-distribution 口径。

    与判决腿（单查询 `assoc_recall_batch`）是**同一件任务**，只是保留训练时的多查询
    布局（Q 个查询块）。距离与训练同分布。
    """
    keep = _keep(T, window, anchors, device)
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
    local_prior = chance = None
    for i in range(nb):
        tb = assoc_recall_multi_batch(bs, T, K, Q, sym_vocab=sym,
                                      filler_token=fill, cue_token=cue, gap=gap,
                                      generator=torch.Generator().manual_seed(
                                          seed_base + 13 * K + Q + i))
        main = accuracy(model, tb, keep_mask=keep, device=device, max_chunk=64)
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        c = accuracy(model, bad, keep_mask=keep, device=device, max_chunk=64)
        agg["n"] += main["n"]; agg["n_hits"] += main["n_hits"]
        agg["cn"] += c["n"]; agg["chits"] += c["n_hits"]
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        _agg_dist(agg, main)
        if local_prior is None:
            local_prior = local_prior_accuracy(tb, k=8)["accuracy"]
            chance = round(tb.chance(), 8)
    dists = agg["dists"]
    return {
        "K": K, "Q": Q, "n_batches": nb, "n": agg["n"], "n_hits": agg["n_hits"],
        "accuracy": round(agg["n_hits"] / agg["n"], 6),
        "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
        "chance": chance, "local_prior": local_prior,
        "by_distance": _by_dist_out(agg),
        "corrupt_source": {"span": list(corrupt_bounds), "n": agg["cn"],
                           "n_hits": agg["chits"],
                           "accuracy": round(agg["chits"] / agg["cn"], 6),
                           "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
        "query_key_distance": {
            "min": min(dists), "max": max(dists),
            "median": sorted(dists)[len(dists) // 2], "window": window,
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


def read_train_log(name: str, train_dir: str, steps: int) -> dict:
    """每臂 200 步训练日志的活性量 + 收敛门禁（§17.15：答案行 CE 末窗斜率）。"""
    path = os.path.join(train_dir, name, "train_log.jsonl")
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    r200 = [r for r in rows if r["step"] % 200 == 0]
    last = rows[-1]
    # 末窗 [3/4·steps, steps] 的答案行 CE 斜率（nats / 1k 步；负 = 仍在下降）
    lo = 0.75 * steps
    win = [r for r in rows if r["step"] >= lo and r.get("synth_answer_ce") is not None]
    slope = None
    if len(win) >= 2:
        (s0, c0), (s1, c1) = (win[0]["step"], win[0]["synth_answer_ce"]), \
                             (win[-1]["step"], win[-1]["synth_answer_ce"])
        slope = round((c1 - c0) / max(s1 - s0, 1) * 1000, 5)
    return {
        "n_log_lines": len(rows),
        "n_rows_step_multiple_of_200": len(r200),
        "final_step": last["step"], "final_loss": last["loss"],
        "final_synth_answer_ce": last.get("synth_answer_ce"),
        "synth_chance_ce": last.get("synth_chance_ce"),
        "answer_ce_slope_nats_per_1k_lastquarter": slope,
        "last": {"kv_heads": last.get("kv_heads"),
                 "attn_carry": last.get("attn_carry"),
                 "window_mask": last.get("window_mask")},
        "kv_heads_series_present": sum(1 for r in r200 if r.get("kv_heads")),
        "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
        "window_mask_series_present": sum(1 for r in r200 if r.get("window_mask")),
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
        cfg["eval"]["n_batches"] = 1
        cfg["eval"]["batch_size"] = 16
        cfg["cross_leg"]["n_batches"] = 1
        cfg["cross_leg"]["batch_size"] = 16
        cfg["train_task_leg"]["n_batches"] = 1
        cfg["train_task_leg"]["batch_size"] = 16
        cfg["bpc_leg"]["n_batches"] = 1
        cfg["distance_leg"]["n_batches"] = 1
    dev = cfg["device"]
    T, K = cfg["model"]["seq_len"], cfg["eval"]["K_main"]
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], \
        cfg["tokens"]["cue_token"]
    mcfg = cfg["model"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    results: dict = {"config": cfg}

    # ---------------- 加载四臂 ----------------
    note("加载四臂（sha256 断言）")
    arms, ck = {}, {}
    for name in ARM_NAMES:
        arms[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"],
                                        dev, cfg["checkpoint"])
        note(f"  {name:18s} {ck[name]['sha256'][:16]}… block={ck[name]['block_class']} "
             f"params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    # ---------------- ① 仪器自检 ----------------
    note("自检①：块类 / 参数量 / 窗口掩码非退化（§17.16）")
    n_base = ck[BASELINE_ARM]["n_params"]
    param_decl = {}
    for name in ARM_NAMES:
        n = ck[name]["n_params"]
        param_decl[name] = {"n_params": n,
                            "delta_vs_base": n - n_base,
                            "delta_pct_vs_base": round(100.0 * (n - n_base) / n_base, 3)}
    results["param_declaration"] = param_decl
    wliv = None
    gwi = arm_spec(TARGET_ARM)["flags"]
    if gwi.get("train_window", 0) > 0:
        wliv = window_mask_liveness(T, gwi["train_window"], gwi.get("train_anchors", 0))
    results["window_nondegenerate"] = wliv
    note(f"  参数量：{ {n: param_decl[n]['delta_pct_vs_base'] for n in ARM_NAMES} }%；"
         f"窗口非退化={wliv['alive'] if wliv else None}")

    # KV 头活性（形状类，取自真实前向）：GQA 臂的 k_shape 必须等于 2
    g = torch.Generator().manual_seed(1234)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    kv_live = {}
    for name in ARM_NAMES:
        out = kv_head_liveness(arms[name], x_probe)
        kv_live[name] = {"n_heads_cfg": out["n_heads"], "n_kv_heads_cfg": out["n_kv_heads"],
                         "k_shape": out["k_batch_shape"], "qkv_out_width": out["qkv_out_width"],
                         "block_class": out["block_python_class"],
                         "kv_reduction_frac": out["kv_reduction_frac"],
                         "alive": out["alive"]}
    results["kv_head_liveness"] = kv_live
    note(f"  KV 头形状：{ {n: kv_live[n]['k_shape'] for n in ARM_NAMES} }")

    # ---------------- ② 主腿（合成长程任务，native 可见性）+ 毁源对照 ----------------
    note("判据①③④：合成关联回忆（native 可见性）+ 毁源对照臂")
    main_leg, cross_leg, distance_leg = {}, {}, {}
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        row = recall_eval(arms[name], T, w, a, bs, nb, sym, fill, cue, K, dev)
        main_leg[name] = {**row, "native_window": w, "native_anchors": a}
        note(f"  {name:18s} window={w:2d} acc={row['accuracy']:.4f} "
             f"CI={row['accuracy_ci']} chance={row['chance']:.4f} "
             f"lp={row['local_prior']:.4f} corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["main_leg"] = main_leg

    note("交叉口径：全因果（所有臂都 keep_mask=None）")
    cbs, cnb = cfg["cross_leg"]["batch_size"], cfg["cross_leg"]["n_batches"]
    for name in ARM_NAMES:
        row = recall_eval(arms[name], T, 0, 0, cbs, cnb, sym, fill, cue, K, dev)
        cross_leg[name] = row
        note(f"  {name:18s} full-causal acc={row['accuracy']:.4f} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["cross_leg_full_causal"] = cross_leg

    # ---------------- ②b 训练型任务（多查询）召回：in-distribution 口径 ----------------
    note("训练型任务（多查询 assoc-mqar）召回 —— in-distribution 口径")
    mtcfg = cfg["train_task_leg"]
    train_task_leg = {}
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        row = multi_recall_eval(arms[name], T, w, a, mtcfg["batch_size"],
                                mtcfg["n_batches"], sym, fill, cue, K,
                                mtcfg["n_queries"], mtcfg["gap"], dev)
        train_task_leg[name] = row
        note(f"  {name:18s} window={w:2d} acc={row['accuracy']:.4f} "
             f"CI={row['accuracy_ci']} corrupt={row['corrupt_source']['accuracy']:.4f}"
             f" dist={row['query_key_distance']['min']}-{row['query_key_distance']['max']}")
    results["train_task_leg"] = train_task_leg
    # 去伪影：Q=K ⇒ 最后一块可由「排除法」白拿（不是长程查找，见 analysis.md）。
    # 最后一块的距离恰为 226–240（前一块 ≤ 223），故按 by_distance 过滤 d≤223 即去伪影。
    no_last = {}
    for name in ARM_NAMES:
        n = h = 0
        for d, v in train_task_leg[name]["by_distance"].items():
            if int(d) <= 223:
                n += v["n"]; h += round(v["accuracy"] * v["n"])
        no_last[name] = {"n": n, "n_hits": h, "accuracy": round(h / n, 6),
                         "accuracy_ci": wilson_ci(h, n),
                         "dropped": "距离 226–240（= Q=K 最后一块，排除法伪影）"}
    results["train_task_leg_no_last_block"] = no_last

    # ---------------- ③ 距离腿（辅助） ----------------
    note("辅助腿：变距离回忆距离曲线（native 可见性）")
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        keep = _keep(T, w, a, dev)
        curve = {}
        for d in cfg["distance_leg"]["distances"]:
            accs, ns = 0, 0
            for i in range(cfg["distance_leg"]["n_batches"]):
                tb = var_recall_batch(bs, T, d, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      generator=torch.Generator().manual_seed(5000 + i))
                r = accuracy(arms[name], tb, keep_mask=keep, device=dev)
                accs += r["n_hits"]; ns += r["n"]
            curve[str(d)] = {"accuracy": round(accs / ns, 6), "n": ns}
        distance_leg[name] = curve
    results["distance_leg"] = distance_leg

    # ---------------- ④ 两腿之二：clean bpc（enwik8 评测尾段） ----------------
    note("判据②：clean bpc（enwik8 评测尾段；非退化锚，非质量判据）")
    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    bpc = {}
    for name in ARM_NAMES:
        b = masked_bpc(arms[name], eval_arr, keep_mask=None,
                       n_batches=cfg["bpc_leg"]["n_batches"],
                       batch_size=cfg["bpc_leg"]["batch_size"],
                       window_seed=cfg["bpc_leg"]["window_seed"], device=dev)
        bpc[name] = {"bpc": round(b, 6), "n_batches": cfg["bpc_leg"]["n_batches"],
                     "batch_size": cfg["bpc_leg"]["batch_size"]}
        note(f"  {name:18s} clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    bpc_base = bpc[BASELINE_ARM]["bpc"]
    results["bpc_damage_vs_stock"] = {n: round(bpc[n]["bpc"] - bpc_base, 6)
                                      for n in ARM_NAMES}
    note(f"  enwik8 vocab={bpc_vocab}（模型 vocab={cfg['data']['vocab_size']}）")

    # ---------------- ⑤ 增量（§17.17 分量分解） ----------------
    acc_native = {n: main_leg[n]["accuracy"] for n in ARM_NAMES}
    acc_cross = {n: cross_leg[n]["accuracy"] for n in ARM_NAMES}
    acc_train = {n: train_task_leg[n]["accuracy"] for n in ARM_NAMES}
    acc_train_nl = {n: results["train_task_leg_no_last_block"][n]["accuracy"]
                    for n in ARM_NAMES}
    results["increments"] = {
        "native": {"staircase": staircase_increments(acc_native),
                   "net_vs_baseline_pp": net_vs_baseline(acc_native)},
        "full_causal": {"staircase": staircase_increments(acc_cross),
                        "net_vs_baseline_pp": net_vs_baseline(acc_cross)},
        "train_task": {"staircase": staircase_increments(acc_train),
                       "net_vs_baseline_pp": net_vs_baseline(acc_train)},
        "train_task_no_last_block": {
            "staircase": staircase_increments(acc_train_nl),
            "net_vs_baseline_pp": net_vs_baseline(acc_train_nl)},
        "pairwise": {},
    }
    for a, b in (("stock", "gqa"), ("gqa", "gqa_carry"),
                 ("gqa_carry", "gqa_carry_window"),
                 ("stock", "gqa_carry_window")):
        results["increments"]["pairwise"][f"{b}_vs_{a}"] = two_proportion(
            main_leg[a]["n_hits"], main_leg[a]["n"],
            main_leg[b]["n_hits"], main_leg[b]["n"])
        results["increments"].setdefault("pairwise_train_task", {})[
            f"{b}_vs_{a}"] = two_proportion(
                train_task_leg[a]["n_hits"], train_task_leg[a]["n"],
                train_task_leg[b]["n_hits"], train_task_leg[b]["n"])
        results["increments"].setdefault("pairwise_train_task_no_last_block", {})[
            f"{b}_vs_{a}"] = two_proportion(
                results["train_task_leg_no_last_block"][a]["n_hits"],
                results["train_task_leg_no_last_block"][a]["n"],
                results["train_task_leg_no_last_block"][b]["n_hits"],
                results["train_task_leg_no_last_block"][b]["n"])

    # ---------------- ⑥ 判据 ③ 毁源对照的有效性 ----------------
    destroy = {}
    for name in ARM_NAMES:
        ci = main_leg[name]["corrupt_source"]["accuracy_ci"]
        chance = main_leg[name]["chance"]
        destroy[name] = {"corrupt_accuracy": main_leg[name]["corrupt_source"]["accuracy"],
                         "corrupt_ci": ci, "chance": chance,
                         "valid": ci[0] <= 4 * chance,
                         "rule": "毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废"}
    results["destroy_source_validity"] = destroy
    note(f"  毁源对照有效（CI 下界 ≤ 4×chance）："
         f"{ {n: destroy[n]['valid'] for n in ARM_NAMES} }")

    # ---------------- ⑦ 活性量 + 收敛门禁（from 训练日志） ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    activity = {}
    for name in ARM_NAMES:
        activity[name] = read_train_log(name, train_dir, steps)
    results["train_activity"] = activity
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）： " +
         ", ".join(f"{n}={activity[n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in ARM_NAMES))

    # ---------------- 成本 / 口径 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 3),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "pure_inference": True,
    }
    results["provenance"] = {"run_log": log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("provenance", "train_activity")},
                     indent=2, ensure_ascii=False)[:4000])
    print("MAINLEG:", {n: main_leg[n]["accuracy"] for n in ARM_NAMES})
    print("INCREMENTS:", results["increments"]["native"])


if __name__ == "__main__":
    main()
