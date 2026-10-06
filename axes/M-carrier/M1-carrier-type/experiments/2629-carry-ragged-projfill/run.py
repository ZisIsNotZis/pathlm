#!/usr/bin/env python3
"""2629-carry-ragged-projfill · delta 搬运通道 × 随机退出 × proj-fill —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录任务下发文本）。本入口只做**评测**
（不训练、不改权重），把两臂（`projfill` = 随机退出 + proj-fill；`projfill_carry` =
同上 + delta 写入通道）在下列腿上量出来：

  ① 主判据 = 训练分布内多查询 assoc-mqar 召回（**随机退出** = 训练同款退出模式，
     SCHEMA §17.18）——每臂在自己的训练分布下评测，成对（两臂同一批 + 同一退出模式），
     附 chance / Wilson CI / 局部先验 / 逐距离 / 毁源对照；
     并列口径 = **全深度**（部署全栈模式，退出深度恒 = L）；
  ② 出口资产 `tf@4` / `tf@6`（depth-k 出口 argmax 与全栈 argmax 的一致率）：
     enwik8 口径（对齐 100M 的 0.9726）+ synth 口径（训练分布内，**真实非 dense 前向**）；
  ③ 单因素增量（§17.17：净和 == 单因素增量，仍显式给出）——召回 / tf@4 / bpc；
  ④ 毁源对照臂（把源 KV 段毁掉 ⇒ 读数若仍高于 4×chance 则作废）；
  ⑤ 参数量 + 活性量（每 200 步训练日志：carry read_share / read_scale 等）；
  ⑥ clean bpc（enwik8 全因果；跨分布 + 合成训练 ⇒ 非退化锚 + 质量代价项，不作召回判据）；
  ⑦ 收敛门禁（§17.15：落在决定判据的 `synth_answer_ce` 上）。

**非 dense 评测的口径声明**：两臂都是非 dense（RAP 形态：逐通道门 + 混合填充 +
随机退出），因此**不能**用 `retention.masked_logits`（那是 dense 主干的读法，会静默
丢掉门与填充）——本入口统一走模型自己的 `forward(tokens, exit_depths=...)`，取
`aux["depth_logits"][-1]`（最终深度读出）作为 logits。这是本实验与 2627（dense 臂）
在仪器上的关键区别，写在这里以免后人误用 `accuracy()`。

用法：
  python3 run.py                # 全电池（纯推理）
  python3 run.py --smoke        # 冒烟：少批（实测耗时/显存，不外推）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             sample_exit_depths, tf_acceptance)
from tools.evalkit.longrange_tasks import (              # noqa: E402
    assoc_recall_batch, assoc_recall_multi_batch, corrupt_span,
    local_prior_accuracy)
from tools.evalkit.carry_channel import two_proportion, wilson_ci  # noqa: E402
from ragged_projfill_arms import (ARMS, ARM_NAMES, BASELINE_ARM, TARGET_ARM,  # noqa: E402
                                  arm_spec, capability_probe, destroy_source_valid,
                                  expected_block_class, model_config_kwargs,
                                  native_visibility, single_factor_increment,
                                  single_factor_increment_pp)

FULL = "full_depth_deployment"
RAND = "in_distribution_random_exit"


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
                   "carry_rule": spec["carry"],
                   "dense": bool(model.cfg.dense),
                   "proj_fill": bool(model.cfg.proj_fill),
                   "fill_kv": bool(model.cfg.fill_kv)}


@torch.no_grad()
def nd_logits(model, tokens: torch.Tensor, exit_depths):
    """非 dense 前向的最终深度 logits `[B,T,V]`（含逐通道门与混合填充）。

    统一走模型自己的 `forward`（**不是** `retention.masked_logits`）：后者是 dense
    主干的读法，会把非 dense 的门 / 填充 / 退出语义静默丢掉。"""
    _, aux = model(tokens, exit_depths=exit_depths)
    return aux["depth_logits"][-1]


def _depths(mode: str, B: int, T: int, L: int, seed: int, device: str):
    if mode == FULL:
        return torch.full((B, T), L, dtype=torch.long, device=device)
    return sample_exit_depths(B, T, L, torch.Generator(device=device).manual_seed(seed),
                              device=device)


def recall_multi(model, T, sym, fill, cue, K, Q, gap, device, mode, bs, nb,
                 seed_base, L, corrupt_bounds=(0, 16)):
    """多查询 assoc-mqar 召回（主判据）。两臂**成对**：同一批 + 同一退出模式。"""
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
    local_prior = chance = None
    for i in range(nb):
        tb = assoc_recall_multi_batch(
            bs, T, K, Q, sym_vocab=sym, filler_token=fill, cue_token=cue, gap=gap,
            generator=torch.Generator().manual_seed(seed_base + 13 * K + Q + i))
        x = tb.tokens.to(device)
        depths = _depths(mode, bs, T, L, seed_base + i, device)
        logits = nd_logits(model, x, depths)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, tb.rows.to(device))
        hits = (sel == tb.answers.to(device)).to(torch.long).cpu()
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        blogits = nd_logits(model, bad.tokens.to(device), depths)
        bsel = torch.gather(blogits.argmax(-1), 1, bad.rows.to(device))
        bhits = (bsel == bad.answers.to(device)).to(torch.long).cpu()
        agg["n"] += hits.numel(); agg["n_hits"] += int(hits.sum())
        agg["cn"] += bhits.numel(); agg["chits"] += int(bhits.sum())
        for j in range(Q):
            for b in range(bs):
                d = int(tb.dists[b, j])
                slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
                slot["n"] += 1
                slot["n_hits"] += int(hits[b, j])
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        if local_prior is None:
            local_prior = local_prior_accuracy(tb, k=8)["accuracy"]
            chance = round(tb.chance(), 8)
    dists = agg["dists"]
    by_dist = {str(d): {"accuracy": round(s["n_hits"] / s["n"], 6), "n": s["n"]}
               for d, s in sorted(agg["by_dist"].items())}
    return {
        "mode": mode, "K": K, "Q": Q, "n_batches": nb, "n": agg["n"],
        "n_hits": agg["n_hits"], "accuracy": round(agg["n_hits"] / agg["n"], 6),
        "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
        "chance": chance, "local_prior": local_prior,
        "by_distance": by_dist,
        "corrupt_source": {"span": list(corrupt_bounds), "n": agg["cn"],
                           "n_hits": agg["chits"],
                           "accuracy": round(agg["chits"] / agg["cn"], 6),
                           "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
        "query_key_distance": {
            "min": min(dists), "max": max(dists), "median": sorted(dists)[len(dists) // 2],
            "window": 0, "in_train_distribution": max(dists) <= 240,
            "frac_outside_window": 0.0, "n": len(dists)},
    }


def drop_last_block(leg: dict) -> dict:
    """去掉距离 226–240（= Q=K 时最后一块可由排除法白拿）后的召回。

    与 2623/2627 同一去伪影口径（§17.18 的难度档控制）。"""
    n = h = 0
    for d, v in leg["by_distance"].items():
        if int(d) <= 223:
            n += v["n"]; h += round(v["accuracy"] * v["n"])
    return {"n": n, "n_hits": h, "accuracy": round(h / n, 6),
            "accuracy_ci": wilson_ci(h, n),
            "dropped": "距离 226–240（= Q=K 最后一块，排除法伪影）"}


def recall_single_ood(model, T, sym, fill, cue, K, device, mode, bs, nb, seed_base,
                      L, corrupt_bounds=(0, 16)):
    """单查询关联回忆（判决行 T-1，距离 241–255 = **训练分布外**）——外推参照，非判据。"""
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": []}
    chance = None
    for i in range(nb):
        tb = assoc_recall_batch(bs, T, K, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(
                                    seed_base + 7 * K + i))
        depths = _depths(mode, bs, T, L, seed_base + i, device)
        logits = nd_logits(model, tb.tokens.to(device), depths)
        sel = torch.gather(logits.argmax(-1), 1, tb.rows.to(device))
        hits = (sel == tb.answers.to(device)).to(torch.long).cpu()
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        blogits = nd_logits(model, bad.tokens.to(device), depths)
        bsel = torch.gather(blogits.argmax(-1), 1, bad.rows.to(device))
        bhits = (bsel == bad.answers.to(device)).to(torch.long).cpu()
        agg["n"] += hits.numel(); agg["n_hits"] += int(hits.sum())
        agg["cn"] += bhits.numel(); agg["chits"] += int(bhits.sum())
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        if chance is None:
            chance = round(tb.chance(), 8)
    return {"mode": mode, "K": K, "n_batches": nb, "n": agg["n"],
            "n_hits": agg["n_hits"], "accuracy": round(agg["n_hits"] / agg["n"], 6),
            "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]), "chance": chance,
            "corrupt_source": {"span": list(corrupt_bounds), "n": agg["cn"],
                               "n_hits": agg["chits"],
                               "accuracy": round(agg["chits"] / agg["cn"], 6),
                               "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
            "distance_range": [min(agg["dists"]), max(agg["dists"])],
            "in_train_distribution": max(agg["dists"]) <= 240}


@torch.no_grad()
def per_depth_agreement(model, T, sym, fill, cue, K, Q, gap, device, depths=(4, 6),
                        bs=32, nb=16, seed_base=3000):
    """训练分布内的出口资产：depth-k 出口 argmax 与全栈 argmax 的一致率（tf@k）。

    走**真实非 dense 前向**（`exit_depths=None` = 全深度，即部署全栈模式），用
    `aux["states"]`（已含逐通道门）的各深度表征做 tied 读出。逐位置（含答案行）统计。"""
    L = model.cfg.n_layers
    agree = {k: 0 for k in depths}
    n_tok = 0
    for i in range(nb):
        tb = assoc_recall_multi_batch(
            bs, T, K, Q, sym_vocab=sym, filler_token=fill, cue_token=cue, gap=gap,
            generator=torch.Generator().manual_seed(seed_base + i))
        _, aux = model(tb.tokens.to(device), exit_depths=None)
        states = aux["states"]
        full = model.readout(states[L], depth=L)[:, :-1].argmax(-1)
        for k in depths:
            draft = model.readout(states[k], depth=k)[:, :-1].argmax(-1)
            agree[k] += int((draft == full).sum())
        n_tok += full.numel()
    return {"depths": list(depths), "n_tokens": n_tok,
            "accept": {str(k): round(agree[k] / n_tok, 4) for k in depths},
            "caliber": "synth-assoc-mqar（训练分布内，真实非 dense 前向的全深度 states），"
                       "逐位置 argmax 一致率",
            "n_batches": nb, "batch_size": bs}


@torch.no_grad()
def clean_bpc(model, eval_arr, bs, nb, seed, device):
    """clean bpc：enwik8 全因果下的每字符比特（跨分布；只用词表与 LM 质量口径）。"""
    from pathlm.data import batch
    V = model.vocab_size
    gen = torch.Generator().manual_seed(seed)
    T = model.cfg.seq_len
    tot_ce = 0.0
    tot_n = 0
    for _ in range(nb):
        x, _ = batch(eval_arr, bs, T, gen)
        x = x.to(device)
        logits = nd_logits(model, x, None)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, V).float(),
                             x[:, 1:].reshape(-1), reduction="sum")
        tot_ce += float(ce)
        tot_n += x.shape[0] * (T - 1)
    return tot_ce / tot_n / math.log(2)


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
    return {
        "n_log_lines": len(rows),
        "n_rows_step_multiple_of_200": len(r200),
        "final_step": last["step"], "final_loss": last["loss"],
        "final_synth_answer_ce": last.get("synth_answer_ce"),
        "synth_chance_ce": last.get("synth_chance_ce"),
        "answer_ce_slope_nats_per_1k_lastquarter": slope,
        "last_attn_carry": last.get("attn_carry"),
        "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
        "n_exit_ce": (len(last.get("depth_ce")) if last.get("depth_ce") else None),
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

    results: dict = {"config": cfg}

    note("CPU 能力面自检（两臂可建 / 块类 / 主干逐位一致）")
    cap = capability_probe()
    results["capability_probe"] = cap
    note(f"  combo_buildable={cap['combo_buildable']} "
         f"trunk_bit_identical={cap['trunk_bit_identical']} "
         f"param_delta={cap['param_delta']}")

    note("加载两臂（sha256 断言）")
    arms, ck = {}, {}
    for name in ARM_NAMES:
        arms[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"],
                                        dev, cfg["checkpoint"])
        note(f"  {name:16s} {ck[name]['sha256'][:16]}… block={ck[name]['block_class']} "
             f"params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    n_base = ck[BASELINE_ARM]["n_params"]
    results["param_declaration"] = {
        name: {"n_params": ck[name]["n_params"],
               "delta_vs_base": ck[name]["n_params"] - n_base,
               "delta_pct_vs_base": round(100.0 * (ck[name]["n_params"] - n_base)
                                          / n_base, 3)}
        for name in ARM_NAMES}
    results["architecture_equality"] = {
        "block_class": {n: ck[n]["block_class"] for n in ARM_NAMES},
        "block_class_expected": {n: ck[n]["block_class_expected"] for n in ARM_NAMES},
        "dense": {n: ck[n]["dense"] for n in ARM_NAMES},
        "proj_fill": {n: ck[n]["proj_fill"] for n in ARM_NAMES},
        "fill_kv": {n: ck[n]["fill_kv"] for n in ARM_NAMES},
    }

    # ---------------- ① 主判据：训练分布内多查询召回（随机退出）+ 毁源 ----------------
    note("判据①（主）：多查询 assoc-mqar 召回 —— in_distribution_random_exit + full_depth")
    main_leg = {RAND: {}, FULL: {}}
    for mode in (RAND, FULL):
        for name in ARM_NAMES:
            row = recall_multi(arms[name], T, sym, fill, cue, K, tt["n_queries"],
                               tt["gap"], dev, mode, tt["batch_size"],
                               tt["n_batches"], 2000, L)
            main_leg[mode][name] = row
            note(f"  [{mode:28s}] {name:16s} acc={row['accuracy']:.4f} "
                 f"CI={row['accuracy_ci']} chance={row['chance']:.4f} "
                 f"lp={row['local_prior']:.4f} "
                 f"corrupt={row['corrupt_source']['accuracy']:.4f} "
                 f"dist={row['query_key_distance']['min']}-"
                 f"{row['query_key_distance']['max']}")
    results["train_task_leg"] = main_leg
    results["train_task_leg_no_last_block"] = {
        mode: {name: drop_last_block(main_leg[mode][name]) for name in ARM_NAMES}
        for mode in (RAND, FULL)}

    # ---------------- 单查询腿（训练分布外，外推参照）----------------
    note("单查询关联回忆（判决行 T-1，距离 241–255 = **训练分布外**）——外推参照")
    ood = {RAND: {}, FULL: {}}
    for mode in (RAND, FULL):
        for name in ARM_NAMES:
            row = recall_single_ood(arms[name], T, sym, fill, cue, K, dev, mode,
                                    tt["batch_size"], tt["n_batches"], 4000, L)
            ood[mode][name] = row
            note(f"  [{mode:28s}] {name:16s} acc={row['accuracy']:.4f} "
                 f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["main_leg_single_query_ood"] = ood

    # ---------------- ② 出口资产 tf@4 / tf@6 ----------------
    note("判据②：出口资产 tf@4 / tf@6（enwik8 口径 + synth 训练分布内口径）")
    tcfg = cfg["tf_leg"]
    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    tf_enwik8, tf_synth = {}, {}
    for name in ARM_NAMES:
        rows = tf_acceptance(arms[name], eval_arr,
                             n_batches=tcfg["enwik8_n_batches"],
                             batch_size=tcfg["enwik8_batch_size"], device=dev)
        # `tf_acceptance` 的 "depth" = 已执行块数（depth 0 = 输入直连读出）；
        # tf@4 ⇔ depth==4（口径对齐 100M 的 tf@4=0.9726）。
        tf_enwik8[name] = {"by_depth": rows,
                           "tf4": next(r["accept"] for r in rows if r["depth"] == 4),
                           "tf6": next(r["accept"] for r in rows if r["depth"] == 6),
                           "caliber": "enwik8 teacher-forcing 代理（对齐 100M 0.9726）"}
        row = per_depth_agreement(arms[name], T, sym, fill, cue, K, tt["n_queries"],
                                  tt["gap"], dev, depths=tuple(tcfg["depths"]),
                                  bs=tcfg["synth_batch_size"],
                                  nb=tcfg["synth_n_batches"])
        tf_synth[name] = row
        note(f"  {name:16s} tf@4(enwik8)={tf_enwik8[name]['tf4']:.4f} "
             f"tf@6(enwik8)={tf_enwik8[name]['tf6']:.4f} "
             f"tf@4(synth)={row['accept']['4']:.4f} tf@6(synth)={row['accept']['6']:.4f}")
    results["tf_enwik8"] = tf_enwik8
    results["tf_synth_indist"] = tf_synth

    # ---------------- ⑥ clean bpc（非退化锚 + 质量代价项）----------------
    note("判据⑥：clean bpc（enwik8 全因果；跨分布 + 合成训练 ⇒ 非退化锚，不作召回判据）")
    bpc = {}
    for name in ARM_NAMES:
        b = clean_bpc(arms[name], eval_arr, cfg["bpc_leg"]["batch_size"],
                      cfg["bpc_leg"]["n_batches"], 2, dev)
        bpc[name] = {"bpc": round(b, 6), "n_batches": cfg["bpc_leg"]["n_batches"],
                     "batch_size": cfg["bpc_leg"]["batch_size"]}
        note(f"  {name:16s} clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    results["bpc_damage_vs_base"] = {n: round(bpc[n]["bpc"] - bpc[BASELINE_ARM]["bpc"], 6)
                                     for n in ARM_NAMES}
    note(f"  enwik8 vocab={bpc_vocab}（模型 vocab={cfg['data']['vocab_size']}）")

    # ---------------- ③ 单因素增量（§17.17）----------------
    def inc(leg):
        return single_factor_increment_pp({n: leg[n]["accuracy"] for n in ARM_NAMES})

    results["increments"] = {
        "recall_random_exit": inc(main_leg[RAND]),
        "recall_random_exit_no_last_block": single_factor_increment_pp(
            {n: results["train_task_leg_no_last_block"][RAND][n]["accuracy"]
             for n in ARM_NAMES}),
        "recall_full_depth": inc(main_leg[FULL]),
        "single_query_ood": inc(ood[FULL]),
        "tf4_synth_indist": single_factor_increment_pp(
            {n: tf_synth[n]["accept"]["4"] for n in ARM_NAMES}),
        "tf4_enwik8": single_factor_increment_pp(
            {n: tf_enwik8[n]["tf4"] for n in ARM_NAMES}),
        "bpc": single_factor_increment({n: bpc[n]["bpc"] for n in ARM_NAMES}),
    }
    results["increments"]["pairwise_recall_random_exit"] = two_proportion(
        main_leg[RAND][BASELINE_ARM]["n_hits"], main_leg[RAND][BASELINE_ARM]["n"],
        main_leg[RAND][TARGET_ARM]["n_hits"], main_leg[RAND][TARGET_ARM]["n"])
    results["increments"]["pairwise_recall_full_depth"] = two_proportion(
        main_leg[FULL][BASELINE_ARM]["n_hits"], main_leg[FULL][BASELINE_ARM]["n"],
        main_leg[FULL][TARGET_ARM]["n_hits"], main_leg[FULL][TARGET_ARM]["n"])

    # ---------------- ④ 毁源对照有效性 ----------------
    destroy = {mode: {} for mode in (RAND, FULL)}
    for mode in (RAND, FULL):
        for name in ARM_NAMES:
            lo = main_leg[mode][name]["corrupt_source"]["accuracy_ci"][0]
            chance = main_leg[mode][name]["chance"]
            destroy[mode][name] = {
                "corrupt_accuracy": main_leg[mode][name]["corrupt_source"]["accuracy"],
                "corrupt_ci": main_leg[mode][name]["corrupt_source"]["accuracy_ci"],
                "chance": chance, "valid": destroy_source_valid(lo, chance),
                "rule": "毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废"}
    results["destroy_source_validity"] = destroy
    note(f"  毁源对照有效（CI 下界 ≤ 4×chance）：RAND="
         f"{ {n: destroy[RAND][n]['valid'] for n in ARM_NAMES} }")

    # ---------------- ⑤ 活性量 + 收敛门禁 ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    results["train_activity"] = {name: read_train_log(name, train_dir, steps)
                                 for name in ARM_NAMES}
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）： " +
         ", ".join(f"{n}={results['train_activity'][n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in ARM_NAMES))

    # ---------------- 判据⑤：出口资产在 carry 存在时是否保持 ----------------
    tf4_base = tf_synth[BASELINE_ARM]["accept"]["4"]
    tf4_carry = tf_synth[TARGET_ARM]["accept"]["4"]
    tf4_e_base = tf_enwik8[BASELINE_ARM]["tf4"]
    tf4_e_carry = tf_enwik8[TARGET_ARM]["tf4"]
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
                 "尺度/步数下复现；|Δ| ≤ 1.4pp ⇒ 「出口资产在 carry 存在时保持」。enwik8 口径是"
                 "**跨分布代理**（本线模型在合成任务上训练），单独报、不作主口径。"),
        "note": ("两个口径不一致时（本线 enwik8 代理 |Δ| > 1.4pp）必须在报告里分开写、"
                 "不得只报有利的那个；跨分布口径的下降标为待确认（单 seed）。"),
    }
    note(f"  出口资产 tf@4（主口径 in-dist）: base={tf4_base:.4f} carry={tf4_carry:.4f} ⇒ "
         f"holds={results['exit_asset_holds']['holds']}；"
         f"enwik8 代理: base={tf4_e_base:.4f} carry={tf4_e_carry:.4f} "
         f"(Δ={results['exit_asset_holds']['delta_pp_enwik8']:+.2f}pp)")

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
          {n: main_leg[RAND][n]["accuracy"] for n in ARM_NAMES})
    print("MAINLEG(full_depth):",
          {n: main_leg[FULL][n]["accuracy"] for n in ARM_NAMES})
    print("TF4 enwik8:", {n: tf_enwik8[n]["tf4"] for n in ARM_NAMES})
    print("INCREMENTS(pp, recall_random_exit):", results["increments"]["recall_random_exit"])


if __name__ == "__main__":
    main()
