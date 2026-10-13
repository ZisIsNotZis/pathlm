#!/usr/bin/env python3
"""2624-composable-followups-13m · 三个补点的评测入口（纯推理，不改权重）。

契约：同目录 `README.md`（§4 逐字抄录 Z2 契约；§4b 记 ① 的裁决：容量括号 + 插值）。
三个补点：

  ① **GQA 的 §15 对照（容量括号 + 插值）**：`stock`（13,536,768 / MHA）与
     `mha_mlp5`（11,960,832 / MHA）夹住 `gqa`（12,352,512） ⇒ 报 MHA 容量斜率，
     把 GQA 与插值到 12.35M 的 MHA 比（**显式声明：不是逐位等参数臂**）。
  ② **无泄漏 Q=7 分布内腿**：多查询 assoc-mqar 在 Q=K=8 时末块带排除法伪影
     （2623 实测末块 100%）；并报 **Q=7**（Q=K−1）无泄漏版 + **Q=8 的逐块分解**
     直接暴露伪影。**只做评测，不重训**（README §4c）。
  ③ **carry 的 6k → 10k**：`gqa`/`gqa_carry` 各有一个 10k 版本（逐字段同配方，
     唯一差别 = 步数 ⇒ §17.16 同相对刻度），看 carry 的增益是否随训练保持。

每条腿都带：毁源对照臂（§4 判据 c3）、chance、局部先验（SCHEMA §19）、
Wilson 95% CI、距离分布、活性量（每 200 步，取自训练日志）与收敛门禁（§17.15）。

用法：
  python3 run.py            # 全电池（纯推理，峰值 ≪5GB，共享卡 ~3–5 min）
  python3 run.py --smoke    # 冒烟：少批（实测耗时/显存，不外推）
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import fields as _dc_fields

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
from tools.evalkit.retention import masked_bpc, masked_logits  # noqa: E402
from tools.evalkit.carry_channel import wilson_ci, two_proportion  # noqa: E402
from points import (ARMS, ARM_NAMES, BASELINE_ARM, PP_BAND, P1_BRACKET,   # noqa: E402
                    P1_TARGET, P2_ARMS, P3_PAIRS, Q_LEAKFREE, Q_TRAIN,
                    arm_spec, block_index, bracket_points, bracket_span,
                    capacity_slope, carry_increment, increment_pp,
                    interp_at_target, leak_free_leg_gap, model_config_kwargs,
                    orderable, pick_index, residual_vs_interp, scale_pair_check,
                    scale_ratio, unique_unqueried_last_block)


# ------------------------------------------------------------------ 工具

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def ckpt_dir(name: str) -> str:
    """臂 → 权重所在目录（`source` 指向的实验目录）。"""
    s = arm_spec(name)
    if s["source"] == "2623":
        return resolve("axes/1-time/1.1.4.1-explicit-memory/experiments/"
                       "2623-composable-arms-13m/train")
    return os.path.join(HERE, "train")


def expected_window(name: str) -> tuple[int, int]:
    """该臂训练/评测的可见性（训练/评测同一掩码；窗口臂 16/0，其余全因果）。"""
    f = arm_spec(name)["flags"]
    return int(f.get("train_window", 0)), int(f.get("train_anchors", 0))


def _keep(T: int, window: int, anchors: int, device):
    return (window_anchor_keep_mask(T, window, anchors, device=device)
            if window > 0 else None)


def _agg_dist(agg: dict, main: dict) -> None:
    for d, v in main["by_distance"].items():
        slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
        slot["n"] += v["n"]
        slot["n_hits"] += round(v["accuracy"] * v["n"])


def _by_dist_out(agg: dict) -> dict:
    return {d: {"accuracy": round(s["n_hits"] / s["n"], 6), "n": s["n"]}
            for d, s in sorted(agg["by_dist"].items(), key=lambda kv: int(kv[0]))}


@torch.no_grad()
def accuracy_detailed(model, task, keep_mask=None, device=None, max_chunk: int = 64):
    """与 `longrange_tasks.accuracy` **同一件仪器**，但保留**逐查询命中**。

    用于按查询块（`block_index`）分解多查询任务——聚合读数与 `accuracy()` 逐位相同
    （`reconcile_numbers.py` 的「仪器一致性」检查钉住）。
    """
    dev = device if device is not None else next(model.parameters()).device
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        if keep_mask is None:
            km = None
        elif keep_mask.dim() == 3:
            km = keep_mask[lo:hi].to(dev)
        else:
            km = keep_mask.to(dev)
        logits = masked_logits(model, task.tokens[lo:hi].to(dev), km)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(dev))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(dev)).to(torch.long).cpu()
    return hits


def _block_profile(hits, task, seq_len: int, n_queries: int, gap: int) -> dict:
    """逐查询块命中率（j = 0..Q−1；末块 = 排除法伪影所在）。"""
    blk = block_index(task.rows, seq_len, n_queries, gap)
    out = {}
    for j in range(n_queries):
        sel = blk == j
        n = int(sel.sum().item())
        k = int(hits[sel].sum().item())
        out[str(j)] = {"n": n, "n_hits": k, "accuracy": round(k / max(n, 1), 6),
                       "accuracy_ci": wilson_ci(k, n)}
    return out


def recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K,
                device, corrupt_bounds=(0, 16), max_chunk=64, seed_base=1000):
    """判决腿（单查询 assoc_recall）：主读 + 毁源对照 + 局部先验 + 距离分布。

    种子/批数与 2623 的 `run.py::recall_eval` 逐字段一致（对照复用 §9：
    同仪器 ⇒ 四臂 6k 读数应与 2623 的 `results.json` 逐位相同）。
    """
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
            "frac_outside_window": round(
                sum(1 for d in dists if window > 0 and d >= window) / len(dists), 6),
            "n": len(dists)},
    }


def multi_recall_eval(model, T, window, anchors, bs, nb, sym, fill, cue, K, Q,
                      gap, device, corrupt_bounds=(0, 16), seed_base=2000):
    """训练型任务（多查询 assoc-mqar）：in-distribution 腿 + **逐查询块分解**。

    Q=K（8）= 训练时的布局（末块带排除法伪影）；Q=K−1（7）= 无泄漏版。
    3+gap 块宽、`base = T − Q·(3+gap)` 与 `longrange_tasks` 一致。
    """
    keep = _keep(T, window, anchors, device)
    agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
    cagg = {"by_dist": {}}
    blk_hits = {str(j): {"n": 0, "n_hits": 0} for j in range(Q)}
    blk_hits_bad = {str(j): {"n": 0, "n_hits": 0} for j in range(Q)}
    local_prior = chance = leaky = None
    for i in range(nb):
        tb = assoc_recall_multi_batch(bs, T, K, Q, sym_vocab=sym,
                                      filler_token=fill, cue_token=cue, gap=gap,
                                      generator=torch.Generator().manual_seed(
                                          seed_base + 13 * K + Q + i))
        hits = accuracy_detailed(model, tb, keep_mask=keep, device=device)
        bad = corrupt_span(tb, corrupt_bounds[0], corrupt_bounds[1],
                           generator=torch.Generator().manual_seed(seed_base + i))
        hits_bad = accuracy_detailed(model, bad, keep_mask=keep, device=device)
        agg["n"] += hits.numel(); agg["n_hits"] += int(hits.sum().item())
        agg["cn"] += hits_bad.numel(); agg["chits"] += int(hits_bad.sum().item())
        agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
        _agg_dist(agg, {"by_distance": _by_dist_from_hits(hits, tb.dists)})
        _agg_dist(cagg, {"by_distance": _by_dist_from_hits(hits_bad, tb.dists)})
        for j, v in _block_profile(hits, tb, T, Q, gap).items():
            blk_hits[j]["n"] += v["n"]; blk_hits[j]["n_hits"] += v["n_hits"]
        for j, v in _block_profile(hits_bad, tb, T, Q, gap).items():
            blk_hits_bad[j]["n"] += v["n"]; blk_hits_bad[j]["n_hits"] += v["n_hits"]
        if local_prior is None:
            local_prior = local_prior_accuracy(tb, k=8)["accuracy"]
            chance = round(tb.chance(), 8)
            leaky = unique_unqueried_last_block(tb.rows, tb.dists, T, Q, K, gap)
    dists = agg["dists"]
    by_block = {j: {"n": v["n"], "n_hits": v["n_hits"],
                    "accuracy": round(v["n_hits"] / max(v["n"], 1), 6),
                    "accuracy_ci": wilson_ci(v["n_hits"], max(v["n"], 1))}
                for j, v in blk_hits.items()}
    by_block_corrupt = {j: {"n": v["n"], "n_hits": v["n_hits"],
                            "accuracy": round(v["n_hits"] / max(v["n"], 1), 6),
                            "accuracy_ci": wilson_ci(v["n_hits"], max(v["n"], 1))}
                        for j, v in blk_hits_bad.items()}
    return {
        "K": K, "Q": Q, "gap": gap, "n_batches": nb, "n": agg["n"],
        "n_hits": agg["n_hits"],
        "accuracy": round(agg["n_hits"] / agg["n"], 6),
        "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
        "chance": chance, "local_prior": local_prior,
        "by_distance": _by_dist_out(agg),
        "by_query_block": by_block,
        "by_query_block_corrupt": by_block_corrupt,
        "corrupt_by_distance": _by_dist_out(cagg),
        "leaky_last_block_all_rows": (None if leaky is None else all(leaky)),
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


def _by_dist_from_hits(hits, dists) -> dict:
    out = {}
    for d in sorted(set(int(v) for v in dists.flatten().tolist())):
        sel = dists == d
        n = int(sel.sum().item())
        out[str(d)] = {"accuracy": round(int(hits[sel].sum().item()) / n, 6), "n": n}
    return out


def _acc_excl_last_block(row: dict, Q: int) -> dict:
    """只聚合前 Q−1 个查询块（去掉末块 = 伪影/位置效应所在的那一块）。"""
    n = sum(row["by_query_block"][str(j)]["n"] for j in range(Q - 1))
    h = sum(row["by_query_block"][str(j)]["n_hits"] for j in range(Q - 1))
    return {"n": n, "n_hits": h, "accuracy": round(h / max(n, 1), 6),
            "accuracy_ci": wilson_ci(h, max(n, 1)),
            "dropped": f"查询块 {Q - 1}（末块）"}


def read_train_log(name: str, steps: int) -> dict:
    """每 200 步的活性量 + 收敛门禁（§17.15：答案行 CE 末窗斜率）。"""
    path = os.path.join(ckpt_dir(name), name, "train_log.jsonl")
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
        "last": {"kv_heads": last.get("kv_heads"),
                 "attn_carry": last.get("attn_carry"),
                 "window_mask": last.get("window_mask")},
        "kv_heads_series_present": sum(1 for r in r200 if r.get("kv_heads")),
        "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
        "window_mask_series_present": sum(1 for r in r200 if r.get("window_mask")),
    }


# ------------------------------------------------------------------ 装载

def load_arm(name, mcfg, vocab, device, hashes):
    spec = arm_spec(name)
    path = os.path.join(ckpt_dir(name), name, "model.pt")
    want = hashes.get(name)
    if not want or want.startswith("TO_FILL"):
        raise SystemExit(f"{name}: config 里没有可用 sha256（先训练并回填）")
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, **model_config_kwargs(name)})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in model.parameters())
    if n != spec["params"]:
        raise SystemExit(f"{name}: 参数量 {n} != points.ARMS 登记的 {spec['params']}")
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": n, "flags": model_config_kwargs(name),
                   "steps": spec["steps"], "source": spec["source"]}


# ------------------------------------------------------------------ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default=None,
                    help="逗号分隔的臂子集（缺省 = 全部；用于冒烟 / 与 2623 的定向复现）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    sel = list(ARM_NAMES) if args.arms is None else args.arms.split(",")
    for n in sel:
        if n not in ARMS:
            raise SystemExit(f"--arms 里有未登记的臂 {n!r}；期望 {ARM_NAMES}")
    cfg = json.load(open(args.config))
    if args.smoke:
        for leg in ("eval", "cross_leg", "train_task_leg", "q7_leg", "bpc_leg"):
            cfg[leg]["n_batches"] = 1
            cfg[leg]["batch_size"] = 16
        cfg["distance_leg"]["n_batches"] = 1

    dev = cfg["device"]
    T, K = cfg["model"]["seq_len"], cfg["train"]["synth_pairs"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], \
        cfg["tokens"]["cue_token"]
    # 只把 DepthARConfig 认识的字段喂给配置（config.json 里的说明性键如 "note" 忽略）
    _known = {f.name for f in _dc_fields(DepthARConfig)}
    mcfg = {k: v for k, v in cfg["model"].items() if k in _known}
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    results: dict = {"config": cfg, "points": {"P1": 1, "P2": 2, "P3": 3}}
    hashes = cfg["checkpoint"]["sha256"]

    note(f"加载臂（sha256 断言 + 参数量断言）：{sel}")
    arms, ck = {}, {}
    for name in sel:
        arms[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"],
                                        dev, hashes)
        note(f"  {name:18s} {ck[name]['sha256'][:12]}… {ck[name]['block_class']:18s} "
             f"params={ck[name]['n_params']} steps={ck[name]['steps']}")
    results["checkpoints"] = ck

    # ---------------- 自检：参数量账 / 括号非退化 / 窗口非退化 ----------------
    note("自检：参数量账 + ① 的容量括号非退化（§17.16）+ 窗口掩码非退化")
    n_base = ck[BASELINE_ARM]["n_params"]
    param_decl = {n: {"n_params": ck[n]["n_params"],
                      "delta_vs_base": ck[n]["n_params"] - n_base,
                      "delta_pct_vs_base": round(100.0 * (ck[n]["n_params"] - n_base)
                                                 / n_base, 3),
                      "steps": ck[n]["steps"]}
                  for n in sel}
    results["param_declaration"] = param_decl
    p1_lo, p1_hi = bracket_points()
    results["p1_bracket_geometry"] = bracket_span(ARMS[p1_lo]["params"],
                                                  ARMS[p1_hi]["params"],
                                                  ARMS[P1_TARGET]["params"])
    wliv = window_mask_liveness(T, 16, 0)
    results["window_nondegenerate"] = wliv
    note(f"  ① 括号：{p1_lo}({ARMS[p1_lo]['params']}) < {P1_TARGET}"
         f"({ARMS[P1_TARGET]['params']}) < {p1_hi}({ARMS[p1_hi]['params']})；"
         f"窗口非退化={wliv['alive']}")

    g = torch.Generator().manual_seed(1234)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    results["kv_head_liveness"] = {
        n: {k: v for k, v in kv_head_liveness(arms[n], x_probe).items()}
        for n in sel}
    note(f"  KV 头形状：{ {n: results['kv_head_liveness'][n]['k_batch_shape'] for n in ('gqa', 'stock')} }")

    # ---------------- 判决腿（单查询，距离 241–255；native 可见性）+ 毁源 ----------------
    note("判决腿（单查询 assoc-mqar，native 可见性）+ 毁源对照")
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    main_leg = {}
    for name in sel:
        w, a = expected_window(name)
        row = recall_eval(arms[name], T, w, a, bs, nb, sym, fill, cue, 8, dev)
        main_leg[name] = {**row, "native_window": w, "native_anchors": a}
        note(f"  {name:18s} acc={row['accuracy']:.4f} CI={row['accuracy_ci']} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["main_leg"] = main_leg

    note("交叉口径：全因果（所有臂 keep_mask=None）")
    cbs, cnb = cfg["cross_leg"]["batch_size"], cfg["cross_leg"]["n_batches"]
    results["cross_leg_full_causal"] = {
        n: recall_eval(arms[n], T, 0, 0, cbs, cnb, sym, fill, cue, 8, dev)
        for n in sel}

    # ---------------- ② Q=8（训练布局，逐块）+ Q=7（无泄漏） ----------------
    gap = cfg["train"]["synth_gap"]
    mt = cfg["train_task_leg"]
    note("② 分布内腿 Q=8（训练布局，逐查询块分解 ⇒ 直接暴露末块伪影）")
    q8 = {}
    for name in sel:
        w, a = expected_window(name)
        row = multi_recall_eval(arms[name], T, w, a, mt["batch_size"],
                                mt["n_batches"], sym, fill, cue, K, Q_TRAIN, gap, dev)
        q8[name] = row
        note(f"  {name:18s} acc={row['accuracy']:.4f} "
             f"末块={row['by_query_block'][str(Q_TRAIN - 1)]['accuracy']:.4f} "
             f"leaky={row['leaky_last_block_all_rows']}")
    results["train_task_leg_q8"] = q8
    # 2623 口径的「去伪影块」（按距离 d ≤ 223 过滤），用于与 2623 的对账
    no_last = {}
    for name in sel:
        n = h = 0
        for d, v in q8[name]["by_distance"].items():
            if int(d) <= 223:
                n += v["n"]; h += round(v["accuracy"] * v["n"])
        no_last[name] = {"n": n, "n_hits": h, "accuracy": round(h / n, 6),
                         "accuracy_ci": wilson_ci(h, n),
                         "dropped": "距离 226–240（= Q=K 最后一块，排除法伪影）"}
    results["train_task_leg_q8_no_last_block"] = no_last

    Ql = leak_free_leg_gap(T, K, gap)
    note(f"② 无泄漏分布内腿 Q={Ql}（Q=K−1 ⇒ 末块不再是「唯一没被问过」的那一对）")
    q7 = {}
    for name in sel:
        w, a = expected_window(name)
        row = multi_recall_eval(arms[name], T, w, a, mt["batch_size"],
                                mt["n_batches"], sym, fill, cue, K, Ql, gap, dev)
        q7[name] = row
        note(f"  {name:18s} acc={row['accuracy']:.4f} "
             f"末块={row['by_query_block'][str(Ql - 1)]['accuracy']:.4f} "
             f"leaky={row['leaky_last_block_all_rows']} "
             f"corrupt={row['corrupt_source']['accuracy']:.4f}")
    results["train_task_leg_q7_leakfree"] = q7
    results["leakfree_construction"] = {
        "Q_train": Q_TRAIN, "Q_leakfree": Ql, "K": K, "gap": gap,
        "q8_leaky_last_block": q8[BASELINE_ARM]["leaky_last_block_all_rows"],
        "q7_leaky_last_block": q7[BASELINE_ARM]["leaky_last_block_all_rows"],
        "q8_last_block_acc": {n: q8[n]["by_query_block"][str(Q_TRAIN - 1)]["accuracy"]
                              for n in sel},
        "q7_last_block_acc": {n: q7[n]["by_query_block"][str(Ql - 1)]["accuracy"]
                              for n in sel}}

    # ---------------- 距离腿（辅助） ----------------
    note("辅助：变距离回忆曲线")
    distance_leg = {}
    for name in sel:
        w, a = expected_window(name)
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

    # ---------------- clean bpc（非退化锚；① 的小表也用它） ----------------
    note("clean bpc（enwik8 评测尾段；非退化锚，非质量判据）")
    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    bpc = {}
    for name in sel:
        b = masked_bpc(arms[name], eval_arr, keep_mask=None,
                       n_batches=cfg["bpc_leg"]["n_batches"],
                       batch_size=cfg["bpc_leg"]["batch_size"],
                       window_seed=cfg["bpc_leg"]["window_seed"], device=dev)
        bpc[name] = {"bpc": round(b, 6), "n_batches": cfg["bpc_leg"]["n_batches"],
                     "batch_size": cfg["bpc_leg"]["batch_size"]}
        note(f"  {name:18s} clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    results["bpc_damage_vs_stock"] = {n: round(bpc[n]["bpc"] - bpc[BASELINE_ARM]["bpc"], 6)
                                      for n in sel}
    note(f"  enwik8 vocab={bpc_vocab}（模型 vocab={cfg['data']['vocab_size']}）")

    # ---------------- 活性量 + 收敛门禁 ----------------
    activity = {n: read_train_log(n, ARMS[n]["steps"]) for n in sel}
    results["train_activity"] = activity
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）：" +
         ", ".join(f"{n}={activity[n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in sel))


    q7_acc = {n: q7[n]["accuracy"] for n in sel}
    main_acc = {n: main_leg[n]["accuracy"] for n in sel}
    bpc_v = {n: bpc[n]["bpc"] for n in sel}

    p1_need = {p1_lo, p1_hi, P1_TARGET}
    if p1_need <= set(sel):
        note("① GQA 的容量括号 + 插值（**不是逐位等参数臂**；README §4b 声明）")
        p1 = {"bracket_arms": [p1_lo, p1_hi], "target_arm": P1_TARGET,
              "declared_not_bitwise_equal_params": True,
              "table": {n: {"n_params": ck[n]["n_params"], "bpc": bpc_v[n],
                            "main_leg_acc": main_acc[n], "q7_leakfree_acc": q7_acc[n],
                            "steps": ck[n]["steps"]}
                        for n in (p1_lo, p1_hi, P1_TARGET)},
              "metrics": {}}
        # ④ 口径鲁棒性：把「无泄漏」再收紧一档——只取 Q=7 的前 Q−1 个查询块，
        # 甩掉仍带块序梯度的末块（见 README §8 失败决策树第 4 条），重算同一个括号。
        q7_mid = {n: _acc_excl_last_block(q7[n], Ql)["accuracy"] for n in sel}
        for metric, values in (("clean_bpc", bpc_v), ("main_leg_acc", main_acc),
                               ("q7_leakfree_acc", q7_acc),
                               ("q7_mid_blocks_acc", q7_mid)):
            x_lo, x_hi = ARMS[p1_lo]["params"], ARMS[p1_hi]["params"]
            y_lo, y_hi = values[p1_lo], values[p1_hi]
            x_t, y_t = ARMS[P1_TARGET]["params"], values[P1_TARGET]
            if metric == "clean_bpc":
                slope = round(capacity_slope(x_lo, x_hi, y_lo, y_hi) * 1e6, 9)
                unit = "bpc / 1M 参数"
            else:
                slope = round(capacity_slope(x_lo, x_hi, y_lo, y_hi) * 1e6 * 100, 3)
                unit = "pp / 1M 参数"
            y_i = interp_at_target(x_t, x_lo, x_hi, y_lo, y_hi)
            res = residual_vs_interp(y_t, y_i)
            if metric == "clean_bpc":
                res_out, orderable_flag = round(res, 6), abs(res) >= 0.014
            else:
                res_out, orderable_flag = round(res * 100, 3), orderable(res * 100)
            p1["metrics"][metric] = {
                "mha_slope_per_1M_params": slope, "unit": unit,
                "mha_bracket": {p1_lo: y_lo, p1_hi: y_hi},
                "mha_interp_at_gqa_params": round(y_i, 6),
                "gqa_observed": y_t, "gqa_residual_vs_interp": res_out,
                "residual_orderable": bool(orderable_flag)}
            note(f"  {metric:18s} MHA 斜率={slope} {unit}；插值={y_i:.6f} "
                 f"GQA={y_t:.6f} 残差={res_out}（可排序={orderable_flag}）")
        results["p1_gqa_capacity_bracket"] = p1
    else:
        results["p1_gqa_capacity_bracket"] = None
        note(f"① 跳过：臂子集缺 {sorted(p1_need - set(sel))}（--arms 限定）")

    # ---------------- ② 无泄漏腿的判定 ----------------
    note("② 无泄漏 Q=7 判定（逐块 + 与 Q=8 对照）")
    p2 = {"q8_vs_q7": {}, "leak_artifact": {}}
    for name in [n for n in P2_ARMS if n in sel]:
        p2["q8_vs_q7"][name] = {
            "q8_acc": q8[name]["accuracy"],
            "q8_acc_excl_last_block": no_last[name]["accuracy"],
            "q7_acc": q7[name]["accuracy"],
            "q8_minus_q7_pp": increment_pp(q7[name]["accuracy"], q8[name]["accuracy"]),
            "pairwise_q8_vs_q7": two_proportion(q8[name]["n_hits"], q8[name]["n"],
                                                q7[name]["n_hits"], q7[name]["n"])}
        p2["q8_vs_q7"][name]["q8_acc_excl_last_block_by_query"] = _acc_excl_last_block(
            q8[name], Q_TRAIN)["accuracy"]
        p2["q8_vs_q7"][name]["q7_acc_excl_last_block_by_query"] = _acc_excl_last_block(
            q7[name], Ql)["accuracy"]
        p2["q8_vs_q7"][name]["q7_minus_q8_excl_last_pp"] = increment_pp(
            p2["q8_vs_q7"][name]["q8_acc_excl_last_block_by_query"],
            p2["q8_vs_q7"][name]["q7_acc_excl_last_block_by_query"])
        p2["leak_artifact"][name] = {
            "q8_by_block": {j: v["accuracy"] for j, v in q8[name]["by_query_block"].items()},
            "q7_by_block": {j: v["accuracy"] for j, v in q7[name]["by_query_block"].items()},
            "q8_last_block_corrupt_acc": q8[name]["by_query_block_corrupt"][
                str(Q_TRAIN - 1)]["accuracy"],
            "q7_last_block_corrupt_acc": q7[name]["by_query_block_corrupt"][
                str(Ql - 1)]["accuracy"],
            "q8_mid_blocks_acc": _acc_excl_last_block(q8[name], Q_TRAIN),
            "q7_mid_blocks_acc": _acc_excl_last_block(q7[name], Ql),
            "q8_last_minus_prev_block_pp": increment_pp(
                q8[name]["by_query_block"][str(Q_TRAIN - 2)]["accuracy"],
                q8[name]["by_query_block"][str(Q_TRAIN - 1)]["accuracy"]),
            "q7_last_minus_prev_block_pp": increment_pp(
                q7[name]["by_query_block"][str(Ql - 2)]["accuracy"],
                q7[name]["by_query_block"][str(Ql - 1)]["accuracy"]),
            "q8_last_minus_mid_mean_pp": increment_pp(
                _acc_excl_last_block(q8[name], Q_TRAIN)["accuracy"],
                q8[name]["by_query_block"][str(Q_TRAIN - 1)]["accuracy"]),
            "q7_last_minus_mid_mean_pp": increment_pp(
                _acc_excl_last_block(q7[name], Ql)["accuracy"],
                q7[name]["by_query_block"][str(Ql - 1)]["accuracy"])}
    # Q=7 腿的四臂增量（无泄漏分布内腿，与 2623 的「去伪影块」口径并报）
    if set(P2_ARMS) <= set(sel):
      acc7 = {n: q7[n]["accuracy"] for n in P2_ARMS}
      p2["q7_increments"] = {
        "net_vs_baseline_pp": increment_pp(acc7[BASELINE_ARM], acc7["gqa_carry_window"]),
        "steps": [{"from": a, "to": b, "delta_pp": increment_pp(acc7[a], acc7[b])}
                  for a, b in (("stock", "gqa"), ("gqa", "gqa_carry"),
                               ("gqa_carry", "gqa_carry_window"))],
        "note": "四臂都是 2623 的 Q=8 训练检查点；Q=7 是**评测**口径（不重训）"}
    results["p2_leakfree_q7"] = p2


    p3_need = {a for pair in P3_PAIRS for a in pair}
    if p3_need <= set(sel):
        note("③ carry 的 6k → 10k（同相对刻度；§17.17 分量分解）")
        steps_pairs = scale_pair_check()
        p3 = {"pairs": steps_pairs, "scale_ratio": scale_ratio(), "carry_increment": {}}
        for steps in (6000, 10000):
            inc = carry_increment({**main_acc, "gqa": main_acc["gqa"],
                                   "gqa_carry": main_acc["gqa_carry"],
                                   "gqa_10k": main_acc["gqa_10k"],
                                   "gqa_carry_10k": main_acc["gqa_carry_10k"]}, steps)
            inc["q7_leakfree_pp"] = increment_pp(
                q7_acc["gqa_10k" if steps == 10000 else "gqa"],
                q7_acc["gqa_carry_10k" if steps == 10000 else "gqa_carry"])
            inc["bpc_delta"] = round(
                (bpc_v["gqa_carry_10k"] if steps == 10000 else bpc_v["gqa_carry"]) -
                (bpc_v["gqa_10k"] if steps == 10000 else bpc_v["gqa"]), 6)
            inc["orderable_target"] = orderable(inc["target_pp"])
            inc["pairwise"] = two_proportion(
                main_leg["gqa_10k" if steps == 10000 else "gqa"]["n_hits"],
                main_leg["gqa_10k" if steps == 10000 else "gqa"]["n"],
                main_leg["gqa_carry_10k" if steps == 10000 else "gqa_carry"]["n_hits"],
                main_leg["gqa_carry_10k" if steps == 10000 else "gqa_carry"]["n"])
            p3["carry_increment"][str(steps)] = inc
            note(f"  steps={steps}: carry Δ(判决腿)={inc['target_pp']}pp "
                 f"（可排序={inc['orderable_target']}）；Q=7 分布内 Δ={inc['q7_leakfree_pp']}pp；"
                 f"bpc Δ={inc['bpc_delta']}")
        p3["increment_retention"] = {
            "target_pp_6k": p3["carry_increment"]["6000"]["target_pp"],
            "target_pp_10k": p3["carry_increment"]["10000"]["target_pp"],
            "delta_pp": round(p3["carry_increment"]["10000"]["target_pp"] -
                              p3["carry_increment"]["6000"]["target_pp"], 3),
            "orderable_change": orderable(p3["carry_increment"]["10000"]["target_pp"] -
                                          p3["carry_increment"]["6000"]["target_pp"]),
            "note": ("两点的臂内配方逐字段相同（只差步数）；读的是 Δ 的变化量，"
                     "不是绝对水平（§17.16/§17.11）")}
        results["p3_carry_6k_vs_10k"] = p3
    else:
        results["p3_carry_6k_vs_10k"] = None
        note(f"③ 跳过：臂子集缺 {sorted(p3_need - set(sel))}（--arms 限定）")
    # ---------------- 毁源对照有效性（所有腿） ----------------
    destroy = {}
    for leg in ("main_leg", "train_task_leg_q8", "train_task_leg_q7_leakfree"):
        for name in sel:
            ci = results[leg][name]["corrupt_source"]["accuracy_ci"]
            chance = results[leg][name]["chance"]
            destroy[f"{leg}.{name}"] = {
                "corrupt_accuracy": results[leg][name]["corrupt_source"]["accuracy"],
                "corrupt_ci": ci, "chance": chance, "valid": ci[0] <= 4 * chance,
                "rule": "毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废"}
    results["destroy_source_validity"] = destroy
    bad = [k for k, v in destroy.items() if not v["valid"]]
    note(f"  毁源对照：{len(destroy) - len(bad)}/{len(destroy)} 通过"
         + (f"；作废：{bad}" if bad else ""))

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 3),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "pure_inference": True,
    }
    results["provenance"] = {"run_log": log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("MAINLEG:", main_acc)
    print("Q7:", q7_acc)
    print("P1:", json.dumps(
        (results["p1_gqa_capacity_bracket"] or {}).get("metrics", {}),
        ensure_ascii=False)[:1200])
    print("P3:", json.dumps(
        (results["p3_carry_6k_vs_10k"] or {}).get("increment_retention", {}),
        ensure_ascii=False))


if __name__ == "__main__":
    main()
