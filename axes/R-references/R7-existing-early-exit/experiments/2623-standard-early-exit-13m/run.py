"""R7（WI-16）推断电池：四腿对照（质量 / 等价算力 / 墙钟 / 内存）。

用法（GPU，锁内；阶段可分开跑）：
    python3 run.py --stage battery          # 深度曲线 / tf@k / CALM 标定与门控 / 真草稿链
    python3 run.py --stage wall             # 交错轮次墙钟（3 轮 × 4 prompt）+ 显存
    python3 run.py --stage all --out ...    # 全部（默认）

**口径（契约 §3/§4）**：
- 质量：teacher-forcing 的逐深度 bpc 与 `tf@k`（depth-k argmax == depth-L argmax），
  以及真解码路径上的 CALM 一致率 / 草稿接受率；
- 等价算力：**解析 layer-position/token**（CALM = 实测平均退出深度；spec =
  `L(γ+1)/(1+τ)`；固定退出 = k；MoD = ratio·L），墙钟只作相对排序（`methodology` §3）；
- 墙钟：**同一进程内交错轮次**（同一轮依次测所有 (臂, 策略)），报中位与极差；
- 内存：torch 峰值 + `nvidia-smi memory.free` 前后（写 `run_log.txt`）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import exit_policies as EP                                    # noqa: E402
from pathlm.data import load_enwik8_full, batch               # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,     # noqa: E402
                             depth_curve, tf_acceptance)
from pathlm.mod_router import MoDModel, mod_eval              # noqa: E402

OUT_ROOT = ".scratch/52-r7-early-exit/evidence"
LOG = os.path.join(HERE, "run_log.txt")
PROMPT_OFFSETS = (1000, 40000, 90000, 160000)
PROMPT_LEN = 16
N_PROMPTS = 4
GAMMAS = (1, 2, 4, 8)
DRAFT_DEPTH = 4


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def smi_free() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def load_arm(name: str, out_root: str):
    """从 run 目录重建模型（config 逐字段来自该 run 的 `results.json`）。"""
    run_dir = os.path.join(out_root, name)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    cfg = DepthARConfig(**res["config"])
    model = (MoDModel(cfg, res["vocab_size"],
                      router_weight=res.get("mod_router_weight") or 0.01)
             if cfg.mod_ratio > 0 else DepthARModel(cfg, res["vocab_size"]))
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location="cuda")
    model.load_state_dict(sd)
    model = model.cuda().eval()
    return model, res, run_dir


def run_battery(arms, out_root, n_batches_eval=20, n_new=150):
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                  "data/enwik8_full.npz")
    out = {}
    prompts = []
    gen = torch.Generator().manual_seed(11)
    for off in PROMPT_OFFSETS:
        prompts.append(eval_arr[off:off + PROMPT_LEN].astype("int64"))
    for name in arms:
        try:
            model, res, run_dir = load_arm(name, out_root)
        except FileNotFoundError as e:
            log(f"[battery] 跳过 {name}（{e}）")
            continue
        cfg = model.cfg
        log(f"[battery] {name}: params={sum(p.numel() for p in model.parameters())}"
            f" dense={cfg.dense} gate={cfg.carry_gate} mod={cfg.mod_ratio}")
        arm = {"run": name, "config": asdict(cfg),
               "params": sum(p.numel() for p in model.parameters()),
               "train_wall_minutes": res.get("wall_minutes"),
               "train_peak_gpu_mem_mb": res.get("peak_gpu_mem_mb"),
               "train_avg_depth_ce": res.get("avg_train_depth_ce"),
               "train_gate_mean": res.get("avg_train_gate_mean")}
        if cfg.mod_ratio > 0:
            arm["mod_eval"] = mod_eval(model, eval_arr, n_batches=n_batches_eval,
                                       batch_size=16, device="cuda")
            out[name] = arm
            del model
            torch.cuda.empty_cache()
            continue
        arm["depth_curve"] = depth_curve(model, eval_arr, device="cuda",
                                         n_batches=n_batches_eval)
        arm["tf_acceptance"] = tf_acceptance(model, eval_arr, device="cuda",
                                             n_batches=n_batches_eval)
        log(f"[battery] {name}: d8 bpc {arm['depth_curve'][-1]['bpc']} "
            f"tf4 {arm['tf_acceptance'][4]['accept'] if len(arm['tf_acceptance']) > 4 else None}")
        # ---- CALM 标定（teacher forcing，前后半分离）----
        arm["calm_calibration"] = EP.calm_calibrate(model, eval_arr,
                                                    eps_list=(0.02, 0.05, 0.10),
                                                    n_batches=8, batch_size=16)
        thetas = arm["calm_calibration"]["by_eps"]["0.05"]["theta"]
        # ---- 真解码路径：CALM 门控 + 全深度/固定退出参照 + spec ----
        arm["decode"] = {"prompts": PROMPT_OFFSETS, "prompt_len": PROMPT_LEN,
                         "n_new": n_new}
        arm["decode"]["calm_eps0.05"] = []
        arm["decode"]["full"] = []
        arm["decode"]["exit4"] = []
        arm["decode"]["spec"] = {str(g): [] for g in GAMMAS}
        for pi, pr in enumerate(prompts):
            g_full, s_full = EP.greedy_decode(model, pr, n_new)
            arm["decode"]["full"].append({"prompt": PROMPT_OFFSETS[pi],
                                          "layerpos_per_token":
                                              round(s_full["layerpos_per_emitted_token"], 4),
                                          "matched_greedy": True})
            g4, s4 = EP.greedy_decode(model, pr, n_new, forced_depth=4)
            arm["decode"]["exit4"].append({
                "prompt": PROMPT_OFFSETS[pi],
                "layerpos_per_token": round(s4["layerpos_per_emitted_token"], 4),
                "match_full_frac": round(sum(a == b for a, b in zip(g4, g_full))
                                         / max(len(g_full), 1), 4)})
            gc, sc = EP.calm_decode(model, pr, n_new, thetas)
            arm["decode"]["calm_eps0.05"].append({
                "prompt": PROMPT_OFFSETS[pi],
                "mean_exit_depth": round(sc["mean_exit_depth"], 4),
                "layerpos_per_token": round(sc["layerpos_per_emitted_token"], 4),
                "match_full_frac": round(sum(a == b for a, b in zip(gc, g_full))
                                         / max(len(g_full), 1), 4),
                "depth_hist": sc["depth_hist"]})
            if cfg.dense:
                continue        # dense 臂没有训练过的浅出口 ⇒ spec 腿无意义
            for g in GAMMAS:
                log_cyc = []
                gs, ss = EP.spec_decode(model, pr, n_new, gamma=g,
                                        draft_depth=DRAFT_DEPTH,
                                        cycle_log=log_cyc)
                arm["decode"]["spec"][str(g)].append({
                    "prompt": PROMPT_OFFSETS[pi],
                    "accept_rate": round(ss["accept_rate"], 4),
                    "accept_by_pos": ss["accept_by_pos"],
                    "tokens_per_cycle": round(ss["tokens_per_cycle"], 4),
                    "layerpos_per_token": round(ss["layerpos_per_emitted_token"], 4),
                    "layerpos_per_token_analytic":
                        ss["layerpos_per_emitted_token_analytic"],
                    "match_full_frac": round(sum(a == b for a, b in zip(gs, g_full))
                                             / max(len(g_full), 1), 4),
                    "n_drafts": ss["n_drafts"], "n_cycles": ss["n_cycles"]})
        # 汇总：γ 维接受率（池化，二项 SE）
        for g in (GAMMAS if not cfg.dense else ()):
            rows = arm["decode"]["spec"][str(g)]
            n = sum(r["n_drafts"] for r in rows)
            acc = sum(r["accept_rate"] * r["n_drafts"] for r in rows)
            arm["decode"][f"spec_g{g}_pooled"] = {
                "n_drafts": n, "accept_rate": round(acc / max(n, 1), 4),
                "binom_se": round((acc / max(n, 1) * (1 - acc / max(n, 1))
                                   / max(n, 1)) ** 0.5, 5)}
        out[name] = arm
        del model
        torch.cuda.empty_cache()
    return out


def run_wall(arms, out_root, rounds=3, n_new=150, policies_filter=()):
    """交错轮次墙钟：同一轮内依次测所有 (臂, 策略)，报中位与极差（比值 vs stock/full）。"""
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    prompts = [eval_arr[o:o + PROMPT_LEN].astype("int64") for o in PROMPT_OFFSETS]
    loaded = {}
    for name in arms:
        try:
            loaded[name], res, _ = load_arm(name, out_root)
        except FileNotFoundError:
            continue
    rows = []
    calm_thetas = {}
    calm_joint = {}
    for name, model in loaded.items():
        cfg = model.cfg
        if not cfg.dense and cfg.mod_ratio == 0:
            cal = EP.calm_calibrate(model, eval_arr, eps_list=(0.05,), n_batches=8,
                                    batch_size=16)
            calm_thetas[name] = cal["by_eps"]["0.05"]["theta"]
            # CALM 原式（联合不一致率 ≤ ε）的**单个全局阈值**：墙钟腿用它才是
            # 「CALM 真的在退」的那一档（逐深度条件口径在 13M 上几乎不退）。
            calm_joint[name] = cal["calm_joint"]["0.05"]["theta"]
            with open(os.path.join(out_root, name, "calm_thetas.json"), "w") as f:
                json.dump({"eps": 0.05, "theta": calm_thetas[name],
                           "theta_joint_eps0.05": calm_joint[name]}, f, indent=2)
    n_smi_before = smi_free()
    for r in range(rounds):
        for name, model in loaded.items():
            cfg = model.cfg
            if cfg.mod_ratio > 0:
                continue          # MoD 的单流解码语义不适用（前向吞吐另测）
            for pi, pr in enumerate(prompts):
                pols = ["full", "exit4", "calm0.05", "spec_g4"]
                if not cfg.dense and cfg.mod_ratio == 0 \
                        and calm_joint.get(name) is not None:
                    pols.append("calm_joint0.05")
                for label in (pols if not cfg.dense else ["full"]):
                    if policies_filter and label not in policies_filter:
                        continue
                    torch.cuda.synchronize()
                    t0 = time.time()
                    if label == "full":
                        body = lambda: EP.greedy_decode(model, pr, n_new,
                                                        timeit=True)
                    elif label == "exit4":
                        body = lambda: EP.greedy_decode(model, pr, n_new,
                                                        forced_depth=4, timeit=True)
                    elif label == "calm0.05":
                        body = lambda: EP.calm_decode(
                            model, pr, n_new, calm_thetas[name], timeit=True)
                    elif label == "calm_joint0.05":
                        th = calm_joint[name]
                        body = lambda: EP.calm_decode(
                            model, pr, n_new, [th] * (model.cfg.n_layers + 1),
                            timeit=True)
                    else:
                        body = lambda: EP.spec_decode(model, pr, n_new, gamma=4,
                                                      draft_depth=DRAFT_DEPTH,
                                                      timeit=True)
                    gen, st = body()
                    dt = time.time() - t0
                    rows.append({"round": r, "arm": name, "policy": label,
                                 "prompt": PROMPT_OFFSETS[pi],
                                 "n_tokens": len(gen), "wall_s": round(dt, 4),
                                 "tok_s": round(len(gen) / dt, 3),
                                 "layerpos_per_token": round(
                                     st["layerpos_per_emitted_token"], 4)})
            log(f"[wall] round {r} done ({len(rows)} rows)")
    n_smi_after = smi_free()
    # 汇总：同轮同策略的所有观测（3 轮 × 4 prompt）→ 中位 tok/s、极差、（对 stock/full 的比值中位）
    ref = {}
    for row in rows:
        if row["arm"] == "r7-stock-s0" and row["policy"] == "full":
            ref[(row["round"], row["prompt"])] = row["tok_s"]
    summary = {}
    for row in rows:
        key = f"{row['arm']}|{row['policy']}"
        summary.setdefault(key, {"tok_s": [], "ratio_vs_stock_full": [],
                                 "layerpos_per_token": []})
        summary[key]["tok_s"].append(row["tok_s"])
        summary[key]["layerpos_per_token"].append(row["layerpos_per_token"])
        base = ref.get((row["round"], row["prompt"]))
        if base:
            summary[key]["ratio_vs_stock_full"].append(row["tok_s"] / base)
    for k, v in summary.items():
        ts = sorted(v["tok_s"])
        med = statistics.median(ts)
        v["tok_s_median"] = round(med, 3)
        v["tok_s_min"] = round(ts[0], 3)
        v["tok_s_max"] = round(ts[-1], 3)
        v["tok_s_range_frac"] = round((ts[-1] - ts[0]) / med, 4) if med else None
        v["n"] = len(ts)
        if v["ratio_vs_stock_full"]:
            rs = sorted(v["ratio_vs_stock_full"])
            v["ratio_median"] = round(statistics.median(rs), 4)
            v["ratio_min"] = round(rs[0], 4)
            v["ratio_max"] = round(rs[-1], 4)
        v["layerpos_per_token_median"] = round(
            statistics.median(v["layerpos_per_token"]), 4)
    return {"rows": rows, "summary": summary, "rounds": rounds, "n_new": n_new,
            "protocol": ("同一进程内交错轮次：每轮依次测所有 (臂, 策略, prompt)；"
                         "比值 = 同轮同 prompt 对 r7-stock-s0/full"),
            "nvidia_smi_free_mb_before": n_smi_before,
            "nvidia_smi_free_mb_after": n_smi_after}


def run_mod_forward(arms, out_root, rounds=3, bs=8, seq=256):
    """MoD 的墙钟腿（前向吞吐：它优化的就是「每 FLOP 换更多深度」）：
    与 stock 在同一进程内交错若干轮，测稳态前向墙钟。"""
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    loaded = {}
    for name in arms:
        try:
            loaded[name], _, _ = load_arm(name, out_root)
        except FileNotFoundError:
            continue
    gen = torch.Generator().manual_seed(5)
    x, _ = batch(eval_arr, bs, seq, gen)
    x = x.cuda()
    rows = []
    for r in range(rounds):
        for name, model in loaded.items():
            model.eval()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    if model.cfg.mod_ratio > 0:
                        model(x)
                    else:
                        model(x, exit_depths=None)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    if model.cfg.mod_ratio > 0:
                        model(x)
                    else:
                        model(x, exit_depths=None)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 5
            rows.append({"round": r, "arm": name, "wall_s_per_forward": round(dt, 4),
                         "tok_per_s": round(bs * seq / dt, 1),
                         "peak_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)})
    summary = {}
    for row in rows:
        summary.setdefault(row["arm"], []).append(row)
    out = {"rows": rows, "bs": bs, "seq": seq, "rounds": rounds,
           "summary": {k: {"tok_per_s_median": round(statistics.median(
               [r["tok_per_s"] for r in v]), 1),
               "tok_per_s_min": min(r["tok_per_s"] for r in v),
               "tok_per_s_max": max(r["tok_per_s"] for r in v),
               "peak_mb": max(r["peak_mb"] for r in v)} for k, v in summary.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("battery", "wall", "all"), default="all")
    ap.add_argument("--out", default="results.json",
                    help="输出文件名（默认 results.json；分段跑时用不同名字再合并）")
    ap.add_argument("--wall-policies", default="",
                    help="只测这些墙钟策略（逗号分隔，空 = 全部）")
    ap.add_argument("--out-root", default=OUT_ROOT)
    ap.add_argument("--arms", default="")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n-new", type=int, default=150)
    ap.add_argument("--n-batches", type=int, default=20)
    args = ap.parse_args()
    default_arms = ["r7-stock-s0", "r7-stock-s1", "r7-c1-legacy-s0", "r7-c1-fair-s0",
                    "r7-c1-fair-s1", "r7-c1-fair-l1-s0", "r7-ls-fair-s0",
                    "r7-ls-legacy-s0", "r7-ls-fair-s1", "r7-mod-s0"]
    arms = [a for a in (args.arms.split(",") if args.arms else default_arms) if a]
    args.wall_policies = [p for p in args.wall_policies.split(",") if p]
    present = [a for a in arms
               if os.path.exists(os.path.join(args.out_root, a, "results.json"))]
    log(f"[run] stage={args.stage} arms={present} free_mb={smi_free()}")
    res = {"stage": args.stage, "arms_requested": arms, "arms_present": present,
           "wall_policies": args.wall_policies,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": vars(args),
           "free_mb_before": smi_free()}
    if args.stage in ("battery", "all"):
        res["battery"] = run_battery(present, args.out_root,
                                     n_batches_eval=args.n_batches,
                                     n_new=args.n_new)
    if args.stage in ("wall", "all"):
        res["wall_clock_interleaved"] = run_wall(present, args.out_root,
                                                 rounds=args.rounds,
                                                 n_new=args.n_new,
                                                 policies_filter=args.wall_policies)
        res["mod_forward_throughput"] = run_mod_forward(
            [a for a in present if "mod" in a or "stock" in a], args.out_root,
            rounds=args.rounds)
    res["free_mb_after"] = smi_free()
    res["peak_torch_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    out_path = os.path.join(HERE, args.out)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log(f"[run] wrote {out_path} free_after={res['free_mb_after']}")


if __name__ == "__main__":
    main()
