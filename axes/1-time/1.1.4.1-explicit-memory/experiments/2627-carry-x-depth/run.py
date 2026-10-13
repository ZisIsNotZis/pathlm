#!/usr/bin/env python3
"""2627-carry-x-depth · delta 搬运通道 × dense 全深度监督 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 X2 契约；§4b/§4c 记录 §22.8 能力面
分析与主 Agent 裁决：深度因子落到 `--dense-aux`）。本入口只做**评测**（不训练、不改
权重），把四臂（`base` / `+carry` / `+depth` / `carry×depth`）在两条腿上量出来：

  ① 合成关联回忆（**主判据 = 训练型多查询 assoc-mqar**，距离 ≤240 = 训练分布内，
     SCHEMA §17.18）——每臂在自己训练时的可见性（window16）下评测，并另跑全因果交叉口径；
  ② 出口资产 `tf@4` / `tf@6`（depth-k 出口 argmax 与全栈 argmax 的一致率）：
     enwik8 口径（对齐 100M 的 0.9726）+ synth 口径（训练分布内）；
  ③ 2×2 因子分解（§17.17：四个单向增量 + 交互项 + 净和）；
  ④ 毁源对照臂（把源 KV 段毁掉 ⇒ 读数若仍高则作废）；
  ⑤ 参数量 + 活性量（carry read_share；depth = 出口 CE 条数 13 vs 1 与逐深度 CE）；
  ⑥ clean bpc（enwik8 尾段，非退化锚 + dense-aux 质量代价项，不作召回判据）；
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

from pathlm.depth_ar import DepthARConfig, DepthARModel, tf_acceptance  # noqa: E402
from tools.evalkit.longrange_tasks import (                 # noqa: E402
    accuracy, assoc_recall_batch, assoc_recall_multi_batch, corrupt_span,
    local_prior_accuracy, var_recall_batch)
from tools.evalkit.carry_channel import wilson_ci, two_proportion  # noqa: E402
from tools.evalkit.retention import masked_bpc                       # noqa: E402
from carry_x_depth_arms import (ARMS, ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                  arm_spec, expected_block_class, factorial_increments,
                  factorial_increments_pp, model_config_kwargs,
                  native_visibility, train_flags)


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
                   "carry_rule": spec["carry"], "depth_grad_supervision": spec["depth"]}


def _keep(T: int, window: int, anchors: int, device: str):
    if window <= 0:
        return None
    from pathlm.depth_ar import window_anchor_keep_mask
    return window_anchor_keep_mask(T, window, anchors, device=device)


def _agg_dist(agg: dict, main: dict) -> None:
    for d, v in main["by_distance"].items():
        slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
        slot["n"] += v["n"]
        slot["n_hits"] += round(v["accuracy"] * v["n"])


def _by_dist_out(agg: dict) -> dict:
    return {d: {"accuracy": round(s["n_hits"] / s["n"], 6), "n": s["n"]}
            for d, s in sorted(agg["by_dist"].items(), key=lambda kv: int(kv[0]))}


def recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K,
                device, corrupt_bounds=(0, 16), max_chunk=64, seed_base=1000):
    """单查询关联回忆：主腿 + 毁源对照 + 局部先验 + 距离分布。

    判决行 = `T-1` ⇒ 距离 241–255，**训练分布外**（§17.18）——本实验只作外推参照。"""
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
            "median": sorted(dists)[len(dists) // 2], "window": window,
            "in_train_distribution": max(dists) <= 240,
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


def multi_recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K, Q,
                      gap, device, corrupt_bounds=(0, 16), seed_base=2000):
    """**主判据**：训练型多查询 assoc-mqar 上的召回（距离 ≤240，训练分布内，§17.18）。"""
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
            "in_train_distribution": max(dists) <= 240,
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


@torch.no_grad()
def per_depth_agreement(model, T, window, anchors, bs, nb, sym, fill, cue, K, Q,
                        gap, device, depths=(4, 6), seed_base=3000):
    """训练分布内的出口资产：depth-k 出口 argmax 与全栈 argmax 的一致率（tf@k）。

    与 `tf_acceptance`（enwik8，teacher-forcing 代理）同一算法，只是把数据流换成
    训练分布内的多查询 assoc-mqar，并走**训练同款**窗口掩码。逐位置（含答案行）统计。
    """
    from pathlm.depth_ar import window_anchor_keep_mask
    keep = (window_anchor_keep_mask(T, window, anchors, device=device)
            if window > 0 else None)
    L = model.cfg.n_layers
    agree = {k: 0 for k in depths}
    n_tok = 0
    for i in range(nb):
        tb = assoc_recall_multi_batch(bs, T, K, Q, sym_vocab=sym,
                                      filler_token=fill, cue_token=cue, gap=gap,
                                      generator=torch.Generator().manual_seed(
                                          seed_base + i))
        x = tb.tokens.to(device)
        B = x.shape[0]
        attn_mask = None
        if keep is not None:
            attn_mask = keep.view(1, 1, T, T).expand(B, model.cfg.n_heads, T, T)
        h = model._input_states(x)
        states = [h]
        for blk in model.blocks:
            h, _ = blk(h, attn_mask=attn_mask)
            states.append(h)
        full = model.readout(states[L], depth=L)[:, :-1].argmax(-1)
        for k in depths:
            draft = model.readout(states[k], depth=k)[:, :-1].argmax(-1)
            agree[k] += (draft == full).sum().item()
        n_tok += full.numel()
    return {"depths": list(depths), "n_tokens": n_tok,
            "accept": {str(k): round(agree[k] / n_tok, 4) for k in depths},
            "caliber": "synth-assoc-mqar（训练分布内），逐位置 argmax 一致率",
            "n_batches": nb, "batch_size": bs}


def read_train_log(name: str, train_dir: str, steps: int) -> dict:
    """每臂 200 步训练日志的活性量 + 收敛门禁（§17.15：答案行 CE 末窗斜率）。"""
    path = os.path.join(train_dir, name, "train_log.jsonl")
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    r200 = [r for r in rows if r["step"] % 200 == 0]
    last = rows[-1]
    lo = 0.75 * steps
    win = [r for r in rows if r["step"] >= lo and r.get("synth_answer_ce") is not None]
    slope = None
    if len(win) >= 2:
        (s0, c0), (s1, c1) = (win[0]["step"], win[0]["synth_answer_ce"]), \
                             (win[-1]["step"], win[-1]["synth_answer_ce"])
        slope = round((c1 - c0) / max(s1 - s0, 1) * 1000, 5)
    # 深度因子活性：出口 CE 条数（1 = 只最终深度；L+1 = 全深度监督）+ 逐深度 CE 末值
    dc = last.get("depth_ce")
    return {
        "n_log_lines": len(rows),
        "n_rows_step_multiple_of_200": len(r200),
        "final_step": last["step"], "final_loss": last["loss"],
        "final_synth_answer_ce": last.get("synth_answer_ce"),
        "synth_chance_ce": last.get("synth_chance_ce"),
        "answer_ce_slope_nats_per_1k_lastquarter": slope,
        "n_exit_ce": (len(dc) if dc else None),
        "final_depth_ce": dc,
        "depth_supervision_live": (len(dc) == 13 if dc else None),
        "last": {"attn_carry": last.get("attn_carry"),
                 "window_mask": last.get("window_mask")},
        "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
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
        cfg["tf_leg"]["enwik8_n_batches"] = 1
        cfg["tf_leg"]["synth_n_batches"] = 1
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
        note(f"  {name:6s} {ck[name]['sha256'][:16]}… block={ck[name]['block_class']} "
             f"params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    # ---------------- 仪器自检：架构一致性 / 参数量 ----------------
    note("自检：架构逐字段一致（depth 因子不进配置）/ 参数量")
    n_base = ck[BASELINE_ARM]["n_params"]
    results["param_declaration"] = {
        name: {"n_params": ck[name]["n_params"],
               "delta_vs_base": ck[name]["n_params"] - n_base,
               "delta_pct_vs_base": round(100.0 * (ck[name]["n_params"] - n_base)
                                          / n_base, 3)}
        for name in ARM_NAMES}
    results["architecture_equality"] = {
        "base_eq_depth": ck["base"]["n_params"] == ck["depth"]["n_params"],
        "carry_eq_both": ck["carry"]["n_params"] == ck["both"]["n_params"],
        "block_class": {n: ck[n]["block_class"] for n in ARM_NAMES},
        "block_class_expected": {n: ck[n]["block_class_expected"] for n in ARM_NAMES},
    }
    note(f"  参数量：{ {n: results['param_declaration'][n]['delta_pct_vs_base'] for n in ARM_NAMES} }%；"
         f"base==depth={results['architecture_equality']['base_eq_depth']} "
         f"carry==both={results['architecture_equality']['carry_eq_both']}")

    # ---------------- ① 主判据：训练型多查询召回（分布内 §17.18）+ 毁源 ----------------
    note("判据①（主）：训练型多查询 assoc-mqar 召回（window16，距离 ≤240 = 分布内）")
    mtcfg = cfg["train_task_leg"]
    train_task_leg = {}
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        row = multi_recall_eval(arms[name], T, w, a, mtcfg["batch_size"],
                                mtcfg["n_batches"], sym, fill, cue, K,
                                mtcfg["n_queries"], mtcfg["gap"], dev)
        train_task_leg[name] = row
        note(f"  {name:6s} acc={row['accuracy']:.4f} CI={row['accuracy_ci']} "
             f"chance={row['chance']:.4f} lp={row['local_prior']:.4f} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f} "
             f"dist={row['query_key_distance']['min']}-{row['query_key_distance']['max']}")
    results["train_task_leg"] = train_task_leg
    # 去伪影：Q=K ⇒ 最后一块可由排除法白拿（距离恰为 226–240）；按 by_distance 过滤 d≤223。
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

    # ---------------- 单查询腿（训练分布外，外推参照）----------------
    note("单查询关联回忆（判决行 T-1，距离 241–255 = **训练分布外**）——外推参照")
    main_leg = {}
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        row = recall_eval(arms[name], T, w, a, bs, nb, sym, fill, cue, K, dev)
        main_leg[name] = {**row, "native_window": w, "native_anchors": a}
        note(f"  {name:6s} acc={row['accuracy']:.4f} CI={row['accuracy_ci']} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f} dist_max="
             f"{row['query_key_distance']['max']} in_dist="
             f"{row['query_key_distance']['in_train_distribution']}")
    results["main_leg_single_query_ood"] = main_leg

    # ---------------- 交叉口径：全因果 ----------------
    note("交叉口径：全因果（所有臂都 keep_mask=None）")
    cbs, cnb = cfg["cross_leg"]["batch_size"], cfg["cross_leg"]["n_batches"]
    cross_leg = {}
    for name in ARM_NAMES:
        row = multi_recall_eval(arms[name], T, 0, 0, cbs, cnb, sym, fill, cue, K,
                                mtcfg["n_queries"], mtcfg["gap"], dev)
        cross_leg[name] = row
        note(f"  {name:6s} full-causal acc={row['accuracy']:.4f} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["cross_leg_full_causal"] = cross_leg

    # ---------------- ② 出口资产 tf@4 / tf@6 ----------------
    note("判据②：出口资产 tf@4 / tf@6（enwik8 口径 + synth 训练分布内口径）")
    tcfg = cfg["tf_leg"]
    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    tf_enwik8, tf_synth = {}, {}
    for name in ARM_NAMES:
        w, a = native_visibility(arm_spec(name))
        rows = tf_acceptance(arms[name], eval_arr, n_batches=tcfg["enwik8_n_batches"],
                             batch_size=tcfg["enwik8_batch_size"], device=dev)
        # `tf_acceptance` 的 "depth" = 已执行块数（depth 0 = 输入直连读出）；
        # tf@4 ⇔ entry depth==4，tf@6 ⇔ depth==6（口径对齐 100M 的 tf@4=0.9726）。
        tf_enwik8[name] = {"by_depth": rows,
                           "tf4": next(r["accept"] for r in rows if r["depth"] == 4),
                           "tf6": next(r["accept"] for r in rows if r["depth"] == 6),
                           "caliber": "enwik8 teacher-forcing 代理（对齐 100M 0.9726）"}
        row = per_depth_agreement(arms[name], T, w, a, tcfg["synth_batch_size"],
                                  tcfg["synth_n_batches"], sym, fill, cue, K,
                                  mtcfg["n_queries"], mtcfg["gap"], dev,
                                  depths=tuple(tcfg["depths"]))
        tf_synth[name] = row
        note(f"  {name:6s} tf@4(enwik8)={tf_enwik8[name]['tf4']:.4f} "
             f"tf@4(synth)={row['accept']['4']:.4f} tf@6(synth)={row['accept']['6']:.4f}")
    results["tf_enwik8"] = tf_enwik8
    results["tf_synth_indist"] = tf_synth

    # ---------------- 辅助腿：变距离回忆距离曲线 ----------------
    note("辅助腿：变距离回忆距离曲线（window16）")
    distance_leg = {}
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

    # ---------------- ⑥ clean bpc（非退化锚 + 质量代价项）----------------
    note("判据⑥：clean bpc（enwik8 尾段；非退化锚 + dense-aux 质量代价项，非召回判据）")
    bpc = {}
    for name in ARM_NAMES:
        b = masked_bpc(arms[name], eval_arr, keep_mask=None,
                       n_batches=cfg["bpc_leg"]["n_batches"],
                       batch_size=cfg["bpc_leg"]["batch_size"],
                       window_seed=cfg["bpc_leg"]["window_seed"], device=dev)
        bpc[name] = {"bpc": round(b, 6), "n_batches": cfg["bpc_leg"]["n_batches"],
                     "batch_size": cfg["bpc_leg"]["batch_size"]}
        note(f"  {name:6s} clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    bpc_base = bpc[BASELINE_ARM]["bpc"]
    results["bpc_damage_vs_base"] = {n: round(bpc[n]["bpc"] - bpc_base, 6)
                                     for n in ARM_NAMES}
    note(f"  enwik8 vocab={bpc_vocab}（模型 vocab={cfg['data']['vocab_size']}）")

    # ---------------- ③ 2×2 因子分解（§17.17）----------------
    def fact(leg: dict) -> dict:
        return factorial_increments_pp({n: leg[n]["accuracy"] for n in ARM_NAMES})

    results["increments"] = {
        "train_task_indist": fact(train_task_leg),
        "train_task_indist_no_last_block": factorial_increments_pp(
            {n: results["train_task_leg_no_last_block"][n]["accuracy"]
             for n in ARM_NAMES}),
        "cross_full_causal": fact(cross_leg),
        "single_query_ood": fact(main_leg),
        "tf4_synth_indist": factorial_increments_pp(
            {n: tf_synth[n]["accept"]["4"] for n in ARM_NAMES}),
        "bpc_pp": factorial_increments(
            {n: bpc[n]["bpc"] for n in ARM_NAMES}),
        "pairwise_indist": {},
    }
    for a, b in (("base", "carry"), ("base", "depth"), ("depth", "both"),
                 ("carry", "both"), ("base", "both")):
        results["increments"]["pairwise_indist"][f"{b}_vs_{a}"] = two_proportion(
            train_task_leg[a]["n_hits"], train_task_leg[a]["n"],
            train_task_leg[b]["n_hits"], train_task_leg[b]["n"])

    # ---------------- ④ 毁源对照有效性 ----------------
    destroy = {}
    for name in ARM_NAMES:
        ci = train_task_leg[name]["corrupt_source"]["accuracy_ci"]
        chance = train_task_leg[name]["chance"]
        destroy[name] = {"corrupt_accuracy": train_task_leg[name]["corrupt_source"]["accuracy"],
                         "corrupt_ci": ci, "chance": chance,
                         "valid": ci[0] <= 4 * chance,
                         "rule": "毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废"}
    results["destroy_source_validity"] = destroy
    note(f"  毁源对照有效（CI 下界 ≤ 4×chance）："
         f"{ {n: destroy[n]['valid'] for n in ARM_NAMES} }")

    # ---------------- ⑤ 活性量 + 收敛门禁 ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    activity = {}
    for name in ARM_NAMES:
        activity[name] = read_train_log(name, train_dir, steps)
    results["train_activity"] = activity
    note("深度因子活性（出口 CE 条数 13=depth 开 / 1=depth 关）： " +
         ", ".join(f"{n}={activity[n]['n_exit_ce']}" for n in ARM_NAMES))
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）： " +
         ", ".join(f"{n}={activity[n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in ARM_NAMES))

    # ---------------- 判据⑤：可叠加性判定 ----------------
    inc = results["increments"]["train_task_indist"]
    inter = inc["interaction"]
    results["additivity_verdict"] = {
        "interaction_pp_indist": inter,
        "carry_effect_pp": inc["carry_at_depth_off"],
        "depth_effect_pp": inc["depth_at_carry_off"],
        "rule": ("交互项 < 0 且 |交互项| > 1.4pp 方差带 ⇒ 记「两资产不可叠加」"
                 "（both < carry + depth − base）"),
        "verdict": ("可加（交互项落在方差带内）" if abs(inter) <= 1.4
                    else ("不可叠加（负交互项，互相干扰）" if inter < 0
                          else "超可加（正交互项）")),
    }
    note(f"  可叠加性判定：交互项 {inter:+.3f}pp ⇒ {results['additivity_verdict']['verdict']}")

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 3),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "pure_inference": True,
    }
    results["provenance"] = {"run_log": log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("MAINLEG(train_task_indist):",
          {n: train_task_leg[n]["accuracy"] for n in ARM_NAMES})
    print("TF4 synth:",
          {n: tf_synth[n]["accept"]["4"] for n in ARM_NAMES})
    print("INCREMENTS(pp, train_task_indist):", results["increments"]["train_task_indist"])


if __name__ == "__main__":
    main()
