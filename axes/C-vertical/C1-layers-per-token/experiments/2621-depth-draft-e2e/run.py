"""ticket 38 W2 · 深度草稿端到端（2621-depth-draft-e2e）驱动。

纯推理，无训练。量四件事（契约判据 ①–④）：
  ① 实测接受率 vs TF 代理口径（同 checkpoint：enwik8 代理 / 自生成流代理 / 真解码路径）；
  ② 端到端 tok/s（同批大小同硬件，3 次重测取中位数）；
  ③ 分级验证（d4→d6→全栈）vs 单级（一次全栈）的 tok/s 与质量；
  ④ 等效 FLOPs 与墙钟两口径。

用法：
    python3 run.py --ckpt <path> --out <exp_dir> [--smoke] [--device cuda]
输出：<exp_dir>/results.json、<exp_dir>/run_log.txt、<exp_dir>/cycle_log.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import time
from statistics import median

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import spec_decode as S  # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             tf_acceptance, depth_curve)

CKPT_CFG = dict(d_model=256, n_layers=8, n_heads=8, seq_len=256, mlp_mult=4)

# 主臂配置（预登记）
PRIMARY = dict(scheme="single", gamma=4, draft_depth=4)
GAMMAS = (2, 4, 8)
DRAFT_DEPTHS = (4, 6)
GRADED = dict(scheme="graded", gamma=4, draft_depth=4, mid_depth=6)

PROMPT_LEN = 16
N_NEW = 200
PROMPT_OFFSETS = (1000, 40000, 90000, 160000, 260000, 380000)


def log(msg, fh=None):
    """只打 stdout——`gpu_run.sh` 把 stdout 重定向进 `run_log.txt`。
    （旧版同时内部 append 同一文件 ⇒ 每行重复一次；已修，见 analysis.md §0。）"""
    print(msg, flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(ckpt, device):
    cfg = DepthARConfig(**CKPT_CFG)
    model = DepthARModel(cfg, vocab_size=206)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.to(device).eval()


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device, repeats):
    """跑 `repeats` 次，返回 (结果列表, 每次墙钟秒数)。"""
    times, outs = [], []
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        times.append(time.perf_counter() - t0)
        outs.append(out)
    return outs, times


def load_prompts(offsets, prompt_len=PROMPT_LEN):
    val = np.load(os.path.join(ROOT, "data", "enwik8_full.npz"))["val"]
    out = []
    for off in offsets:
        seg = val[off:off + prompt_len]
        assert len(seg) == prompt_len, f"val 太短（offset {off}）"
        out.append([int(t) for t in seg])
    return out


@torch.no_grad()
def proxy_on_self_stream(model, prompt, gen, depth=4, device="cpu"):
    """在**模型自己生成的流**上做 teacher-forcing 代理：depth-k 与 depth-L
    的 argmax 一致率（新鲜 embedding，未走提交态）。"""
    L = model.cfg.n_layers
    seq = torch.tensor([prompt + gen], device=device)
    _, aux = model(seq, exit_depths=None)
    pl = len(prompt)
    lg = aux["depth_logits"]
    full = lg[L][0, pl - 1:pl - 1 + len(gen)].argmax(-1)
    dk = lg[depth][0, pl - 1:pl - 1 + len(gen)].argmax(-1)
    agree = int((dk == full).sum())
    return {"depth": depth, "agree": agree, "n": int(full.numel()),
            "accept": round(agree / max(int(full.numel()), 1), 4)}


def run_arm(model, prompts, device, repeats, n_new, prompt_len, **kw):
    """一个臂 × 所有 prompt：3 次重测取中位数，逐 cycle 账本汇总。"""
    per_prompt, wall, genref = [], [], None
    all_cycles = []
    for i, prompt in enumerate(prompts):
        outs, times = timed(lambda: S.spec_run(model, prompt, n_new, **kw),
                            device, repeats)
        gens = [o[0] for o in outs]
        assert all(g == gens[0] for g in gens), "同一 prompt 多次重测的贪心流不一致"
        gen, stats, _ = outs[0]
        if genref is None:
            genref = gen
        per_prompt.append({
            "prompt_idx": i, "n_gen": len(gen),
            "acceptance": stats["acceptance"], "n_cycles": stats["n_cycles"],
            "n_drafts": stats["n_drafts"], "n_accepted": stats["n_accepted"],
            "tokens_per_cycle": stats["tokens_per_cycle"],
            "eq_layerpos": stats["eq_layerpos"],
            "accept_by_pos": _accept_by_pos(stats),
            "wall_s": [round(t, 4) for t in times],
            "wall_median_s": round(median(times), 4),
            "tok_s": round(len(gen) / median(times), 2),
        })
        wall.extend(times)
        for c in stats["cycles"]:
            c2 = dict(c, prompt_idx=i)
            all_cycles.append(c2)
    n_drafts = sum(p["n_drafts"] for p in per_prompt)
    n_acc = sum(p["n_accepted"] for p in per_prompt)
    n_gen = sum(p["n_gen"] for p in per_prompt)
    total_wall = sum(wall)
    n_reps = len(wall) // max(len(per_prompt), 1)      # 每 prompt 的重测次数
    return {
        "config": kw, "per_prompt": per_prompt,
        "sum_n_gen": n_gen, "sum_n_drafts": n_drafts, "sum_n_accepted": n_acc,
        "acceptance_pooled": round(n_acc / n_drafts, 4) if n_drafts else None,
        "accept_by_pos_pooled": _accept_by_pos({"cycles": all_cycles}),
        "eq_layerpos": sum(p["eq_layerpos"] for p in per_prompt),
        "eq_layerpos_prefill": (model.cfg.n_layers * len(per_prompt) * prompt_len),
        "wall_s_all": round(total_wall, 4),
        "wall_s_per_round": round(total_wall / n_reps, 4),
        "n_repeats_per_prompt": n_reps,
        "tok_s_pooled": round(n_gen * n_reps / total_wall, 2),
        "tok_s_median_of_prompts": round(median([p["tok_s"] for p in per_prompt]), 2),
        "n_repeats_used": len(wall),
    }, all_cycles


def _accept_by_pos(stats):
    """逐个草稿位置（轮内第 1、2、… 个草稿）的接受率——位置 1 与 TF 代理最可比。"""
    L = 16
    ok = [0] * L
    tot = [0] * L
    for c in stats["cycles"]:
        for i in range(c["gamma_effective"]):
            if i < L:
                tot[i] += 1
                ok[i] += 1 if i < c["accepted"] else 0
    return [{"pos": i + 1, "accept": round(ok[i] / tot[i], 4), "n": tot[i]}
            for i in range(L) if tot[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--repeats", type=int, default=3,
                    help="交错墙钟口径的轮数")
    ap.add_argument("--ledger-repeats", type=int, default=1,
                    help="账本/代理口径每 prompt 重测次数")
    ap.add_argument("--prompts", type=int, default=len(PROMPT_OFFSETS))
    ap.add_argument("--tf-batches", type=int, default=20)
    ap.add_argument("--n-new", type=int, default=N_NEW)
    ap.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cycf = open(os.path.join(args.out, "cycle_log.jsonl"), "a")

    n_new = args.n_new
    ckpt_sha = sha256_file(args.ckpt)
    log(f"[ckpt] {args.ckpt} sha256={ckpt_sha}")
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        log(f"[gpu] {torch.cuda.get_device_name(0)} "
            f"free={torch.cuda.mem_get_info()[0]/2**20:.0f}MB")

    model = build_model(args.ckpt, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    prompts = load_prompts(PROMPT_OFFSETS[:args.prompts], args.prompt_len)

    # ---- 冒烟：先测步时再定预算（SCHEMA §17.8）----
    if args.smoke:
        t0 = time.perf_counter()
        gen, st = S.spec_generate(model, prompts[0], 32, gamma=4, draft_depth=4,
                                  scheme="single")
        sync(args.device)
        dt = time.perf_counter() - t0
        log(f"[smoke] 32 token 一次 {dt:.3f}s（{len(gen)/dt:.1f} tok/s），"
            f"cycles={st['n_cycles']} acc={st['acceptance']}")
        if args.device.startswith("cuda"):
            log(f"[smoke] peak_mem={torch.cuda.max_memory_allocated()/2**20:.1f}MB")
        cycf.close()
        return

    results = {
        "experiment": "2621-depth-draft-e2e", "ticket": "38-W2",
        "grid": "C1-layers-per-token",
        "ckpt": {"path": args.ckpt, "sha256": ckpt_sha, "params": n_params,
                 "config": CKPT_CFG},
        "protocol": {
            "device": args.device, "repeats": args.repeats,
            "ledger_repeats": args.ledger_repeats,
            "prompt_len": args.prompt_len, "n_new": n_new,
            "prompt_offsets": list(PROMPT_OFFSETS[:args.prompts]),
            "dataset": "data/enwik8_full.npz val split（字符级，V=206）",
            "commit_mode": "hard（embed(argmax)+pos，cap-norm）",
            "primary": PRIMARY, "primary_arm": "single_g4", "graded": GRADED,
            "gammas": list(GAMMAS),
            "draft_depths": list(DRAFT_DEPTHS),
        },
        "arms": {}, "proxies": {},
    }

    # ---- ① TF 代理口径（enwik8 真值流）----
    val = np.load(os.path.join(ROOT, "data", "enwik8_full.npz"))["val"]
    tab = tf_acceptance(model, val, n_batches=args.tf_batches, batch_size=16,
                        device=args.device)
    results["proxies"]["tf_enwik8"] = tab
    dc = depth_curve(model, val, n_batches=args.tf_batches, batch_size=16,
                     device=args.device)
    results["proxies"]["depth_curve"] = dc
    log(f"[proxy] tf(enwik8)={[(t['depth'], t['accept']) for t in tab]}")

    # ---- ②/④ 普通基线的墙钟（三种路径，口径分别写清）----
    # plain_full / plain_exit4 / plain_exit6：仓库原有 Decoder 路径（ticket 24b 同款）
    # plain_cache_full：与本实验 spec 共用同一套缓存机械（剥离实现层开销）
    plain_specs = (("plain_full", "decoder", None), ("plain_exit4", "decoder", 4),
                   ("plain_exit6", "decoder", 6), ("plain_cache_full", "cache", None),
                   ("plain_cache_exit4", "cache", 4))
    for name, path, ed in plain_specs:
        if path == "cache" and ed is not None:
            continue          # plain_greedy_cache 只支持全栈（见 spec_decode）
        per_prompt, walls = [], []
        for i, prompt in enumerate(prompts):
            fn = (S.plain_greedy_cache if path == "cache" else S.plain_greedy)
            outs, times = timed(lambda: fn(model, prompt, n_new, exit_depth=ed),
                                args.device, args.ledger_repeats)
            gens = [o for o in outs]
            assert all(g == gens[0] for g in gens)
            per_prompt.append({"prompt_idx": i, "n_gen": len(gens[0]),
                               "wall_median_s": round(median(times), 4),
                               "tok_s": round(len(gens[0]) / median(times), 2)})
            walls.extend(times)
        fd = ed if ed is not None else model.cfg.n_layers
        n_gen_total = sum(p["n_gen"] for p in per_prompt)
        results["arms"][name] = {
            "exit_depth": fd,
            "baseline_path": path,
            "eq_layerpos_prefill": len(prompts) * args.prompt_len * fd,
            "sum_n_gen": n_gen_total,
            "per_prompt": per_prompt,
            "tok_s_pooled": round(sum(p["n_gen"] for p in per_prompt) / sum(walls), 2),
            # 与 spec 臂同口径：含 prompt 前向（spec 的 eq_layerpos 也含 prefill）
            "eq_layerpos": (n_gen_total + len(prompts) * args.prompt_len) * fd,
        }
        log(f"[{name}] tok/s={results['arms'][name]['tok_s_pooled']}")

    # ---- ③ spec 臂 ----
    arm_specs = []
    for g in GAMMAS:
        arm_specs.append((f"single_g{g}",
                          dict(scheme="single", gamma=g, draft_depth=4)))
    for dd in DRAFT_DEPTHS:
        if dd != PRIMARY["draft_depth"]:
            arm_specs.append((f"single_g4_dd{dd}",
                              dict(scheme="single", gamma=4, draft_depth=dd)))
    arm_specs.append(("graded", dict(scheme="graded", gamma=4, draft_depth=4,
                                     mid_depth=6)))

    for name, kw in arm_specs:
        t0 = time.perf_counter()
        arm, cycles = run_arm(model, prompts, args.device, args.ledger_repeats,
                              n_new, args.prompt_len, **kw)
        for c in cycles:
            cycf.write(json.dumps({"arm": name, **c}) + "\n")
        results["arms"][name] = arm
        log(f"[{name}] acc={arm['acceptance_pooled']} "
            f"tok/s={arm['tok_s_pooled']} eq_layerpos={arm['eq_layerpos']} "
            f"({time.perf_counter()-t0:.1f}s)")

    # ---- ① 自生成流代理（用主臂发出的流）----
    for i, prompt in enumerate(prompts):
        gen, _ = S.spec_generate(model, prompt, n_new, **PRIMARY)
        p4 = proxy_on_self_stream(model, prompt, gen, depth=4, device=args.device)
        p6 = proxy_on_self_stream(model, prompt, gen, depth=6, device=args.device)
        results["proxies"].setdefault("tf_selfstream", []).append(
            {"prompt_idx": i, "depth4": p4, "depth6": p6})
    ts = results["proxies"]["tf_selfstream"]
    results["proxies"]["tf_selfstream_pooled"] = {
        "depth4": round(sum(x["depth4"]["agree"] for x in ts) /
                        sum(x["depth4"]["n"] for x in ts), 4),
        "depth6": round(sum(x["depth6"]["agree"] for x in ts) /
                        sum(x["depth6"]["n"] for x in ts), 4),
        "n": sum(x["depth4"]["n"] for x in ts),
    }
    log(f"[proxy] tf(self-stream)={results['proxies']['tf_selfstream_pooled']}")

    # ---- ② 墙钟主口径：**交错轮次**协议（抗跨轮漂移）----
    # 共享卡上绝对 tok/s 跨 run 漂移可达 2×（同代码 plain_full 两次实测 29.8 / 13.7），
    # 因此主口径改为「同一窗口内交错测量、报相对 plain_full 的比值」。
    inter = [("plain_full", "decoder-none"), ("plain_exit4", "decoder-4"),
             ("plain_cache_full", "cache-none"), ("single_g4", "spec"),
             ("graded", "spec")]
    rows = []
    for r in range(args.repeats):
        for name, _tag in inter:
            for i, prompt in enumerate(prompts):
                if name == "plain_full":
                    fn = lambda: S.plain_greedy(model, prompt, n_new)
                elif name == "plain_exit4":
                    fn = lambda: S.plain_greedy(model, prompt, n_new, exit_depth=4)
                elif name == "plain_cache_full":
                    fn = lambda: S.plain_greedy_cache(model, prompt, n_new)
                elif name == "single_g4":
                    fn = lambda: S.spec_generate(model, prompt, n_new, gamma=4,
                                                 draft_depth=4)
                else:
                    fn = lambda: S.spec_generate(model, prompt, n_new, gamma=4,
                                                 draft_depth=4, mid_depth=6,
                                                 scheme="graded")
                out = fn()
                sync(args.device)
                t0 = time.perf_counter()
                out = fn()
                sync(args.device)
                dt = time.perf_counter() - t0
                n = len(out[0]) if isinstance(out, tuple) else len(out)
                rows.append({"round": r, "arm": name, "prompt_idx": i,
                             "wall_s": dt, "n_gen": n})
    by_arm = {}
    for name, _ in inter:
        ws = [x["wall_s"] for x in rows if x["arm"] == name]
        ns = [x["n_gen"] for x in rows if x["arm"] == name]
        assert len(set(ns)) == 1, f"{name} 生成长度不一致"
        by_arm[name] = {"tok_s": round(ns[0] / median(ws), 2),
                        "wall_median_s": round(median(ws), 4),
                        "wall_min_s": round(min(ws), 4),
                        "wall_max_s": round(max(ws), 4),
                        "n_obs": len(ws), "n_gen": ns[0]}
    base = by_arm["plain_full"]["wall_median_s"]
    base_cache = by_arm["plain_cache_full"]["wall_median_s"]
    for name, a in by_arm.items():
        a["speedup_vs_plain_full"] = round(base / a["wall_median_s"], 3)
        a["speedup_vs_plain_cache_full"] = round(base_cache / a["wall_median_s"], 3)
    results["wall_clock_interleaved"] = {
        "protocol": ("交错轮次：每轮依次测 5 个臂 × 全部 prompt，轮内相邻 ⇒ 漂移对"
                     "各臂近似同权；报中位数与相对 plain_full 的比值"),
        "by_arm": by_arm, "raw_rows": rows,
        "rounds": args.repeats, "prompts": len(prompts),
    }
    for name, a in by_arm.items():
        log(f"[wall/inter] {name}: {a['tok_s']} tok/s "
            f"(×{a['speedup_vs_plain_full']} vs plain_full, "
            f"×{a['speedup_vs_plain_cache_full']} vs plain_cache_full)")

    # ---- 质量护栏：spec 流 == 仓库普通贪心流（真 checkpoint，逐 prompt）----
    fid_rows = []
    for i, prompt in enumerate(prompts):
        gen_spec, _ = S.spec_generate(model, prompt, n_new, **PRIMARY)
        gen_plain = S.plain_greedy(model, prompt, n_new)
        gen_cache = S.plain_greedy_cache(model, prompt, n_new)
        mism = [j for j in range(min(len(gen_spec), len(gen_plain)))
                if gen_spec[j] != gen_plain[j]]
        fid_rows.append({"prompt_idx": i, "n": len(gen_spec),
                         "spec_vs_decoder_mismatch": len(mism),
                         "first_mismatch": mism[0] if mism else None,
                         "plain_decoder_vs_cache_mismatch":
                             sum(1 for a2, b2 in zip(gen_plain, gen_cache) if a2 != b2)})
    results["quality_guard"] = {
        "note": "自 spec 不改质量：主臂发出的 token 流必须 == 同一 checkpoint 的普通全栈贪心流",
        "rows": fid_rows,
        "total_mismatch": sum(r["spec_vs_decoder_mismatch"] for r in fid_rows),
    }
    log(f"[fidelity] spec vs plain mismatch={results['quality_guard']['total_mismatch']}")

    # ---- ④ 等效 FLOPs（解析口径，由 eq_layerpos 与每 block 参数量推出）----
    d, m = CKPT_CFG["d_model"], CKPT_CFG["mlp_mult"]
    params_per_block = (4 + 2 * m) * d * d          # qkv 3d² + proj d² + mlp 2md²
    for name, a in results["arms"].items():
        if "eq_layerpos" in a:
            n_gen = a.get("sum_n_gen", sum(p["n_gen"] for p in a["per_prompt"]))
            a["params_per_block"] = params_per_block
            a["tokens"] = n_gen
            a["eq_flops"] = int(a["eq_layerpos"] * 2 * params_per_block)  # 含 prefill
            # 与生成 token 对齐的口径：扣掉 prompt prefill（按该臂自己的深度算）
            a["eq_layerpos_prefill"] = a.get("eq_layerpos_prefill", 0)
            a["eq_layerpos_gen_only"] = a["eq_layerpos"] - a["eq_layerpos_prefill"]
            a["eq_flops_per_token"] = round(
                a["eq_layerpos_gen_only"] * 2 * params_per_block / n_gen, 1)
            a["eq_layerpos_per_token"] = round(
                a["eq_layerpos_gen_only"] / n_gen, 3)
    results["flops_note"] = (
        "等效 FLOPs = eq_layerpos × 2 × 每 block 参数量(qkv/proj/mlp) —— 只计线性项，"
        "注意力项在 T≤256、d=256 下占比 <10% 且各臂同上下文可比；口径 = 解析，"
        "非实测。墙钟 = 实测（3 次重测中位数）。")

    if args.device.startswith("cuda"):
        results["gpu_peak_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1, sort_keys=False)
    log("[done] results.json 已落盘")
    cycf.close()


if __name__ == "__main__":
    main()
