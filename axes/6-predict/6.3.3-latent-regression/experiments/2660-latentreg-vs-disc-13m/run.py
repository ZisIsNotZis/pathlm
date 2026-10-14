#!/usr/bin/env python3
"""2660-latentreg-vs-disc-13m · 阶段 1 驱动（契约见本目录 README.md）。

    python3 run.py --stage train --out . --ckpt <2631 主干目录> [--seeds 0 1] [--arms ...]

产出（写进 --out）：
- results.json（原始读数；**禁墙钟**）
- analysis.md（人话摘要在前）
- MODEL.md（头权重哈希与复现命令）
- run_log.txt（含 GPU 实耗墙钟——只允许出现在这里）
- train_log.jsonl（L5：逐臂逐 step 的 CE/MSE 轨迹）
- heads/<arm>_s<seed>.pt（头权重，不入库）
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import latent_reg as L  # noqa: E402
from latent_reg import (BOOT_B, BOOT_SEED, DATA_SEED, LR, SEQ_LEN, TRAIN_STEPS,  # noqa: E402
                        VOCAB_SIZE)
from gpu_lock import gpu_lock  # noqa: E402

ARMS = {
    "disc_k2":      {"k": 2, "lam": 0.0},
    "lat_k2_lam0":  {"k": 2, "lam": 0.0},
    "lat_k2_lam01": {"k": 2, "lam": 0.1},
    "lat_k2_lam1":  {"k": 2, "lam": 1.0},
    "disc_k4":      {"k": 4, "lam": 0.0},
    "lat_k4_lam0":  {"k": 4, "lam": 0.0},
    "lat_k4_lam1":  {"k": 4, "lam": 1.0},
}
# 禁墙钟自检：只要 key 命中这些“墙钟语义”就报（长子串 + 短词精确匹配，避免误伤 mse/arms/params）
WALL_SUBSTR = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
               "latency", "throughput")
WALL_EXACT = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def has_wall_clock(obj, path="") -> list[str]:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in WALL_SUBSTR) or kl in WALL_EXACT:
                bad.append(f"{path}.{k}")
            bad += has_wall_clock(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += has_wall_clock(v, f"{path}[{i}]")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train", choices=["train", "features", "draftverify"])
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--dv-arms", nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, "run_log.txt"))

    ckpt = a.ckpt or os.path.join(
        L.REPO, "axes", "6-predict", "6.2-speculative-decoding",
        "experiments", "2631-spec-three-arms", "train", "13m_s0")
    ckpt = os.path.abspath(ckpt)
    n_tr = 8 if a.smoke else L.N_TRAIN_BLOCKS
    n_va = 4 if a.smoke else L.N_VAL_BLOCKS
    steps = 20 if a.smoke else TRAIN_STEPS
    arms = a.arms or ["ar_t1", "disc_k2", "lat_k2_lam0", "lat_k2_lam01", "lat_k2_lam1"]
    t_wall0 = time.time()

    if a.stage == "draftverify":
        import draftverify as D
        need_gpu = torch.cuda.is_available()
        ctx = (lambda: gpu_lock("wp6.3.3-p2"))() if (need_gpu and not a.no_lock) else _NullCtx()
        with ctx:
            log(f"stage=draftverify ckpt={ckpt} out={out} smoke={a.smoke} "
                f"device={'cuda' if need_gpu else 'cpu'}")
            dv = D.stage(out, ckpt, log, smoke=a.smoke,
                         n_prompts=2 if a.smoke else D.N_PROMPTS,
                         n_new=32 if a.smoke else D.N_NEW, arms=a.dv_arms)
            log(f"draftverify DONE wall_seconds={time.time() - t_wall0:.1f} "
                f"repro_gate={dv['repro_gate']['pass'] if dv['repro_gate'] else None}")
        return 0

    need_gpu = a.stage == "train"
    ctx = (lambda: gpu_lock("wp6.3.3-p1"))() if (need_gpu and not a.no_lock) else _NullCtx()
    with ctx:
        log(f"stage={a.stage} ckpt={ckpt} seeds={a.seeds} arms={arms} "
            f"n_train_blocks={n_tr} n_val_blocks={n_va} steps={steps} smoke={a.smoke}")
        dev = "cuda" if (torch.cuda.is_available() and need_gpu) else "cpu"
        log(f"device={dev}")

        model = L.load_ckpt(ckpt)
        model.to(dev)
        embed_sha = L.embed_sha256(model)
        ckpt_sha = _file_sha(os.path.join(ckpt, "model.pt"))
        log(f"ckpt_sha256={ckpt_sha} embed_sha256={embed_sha}")

        blob = np.load(L.DATA_NPZ)
        Htr, toktr, _ = L.extract_blocks(model, blob["train"], n_tr)
        Hva, tokva, n1va = L.extract_blocks(model, blob["val"], n_va)
        log(f"features: train H={tuple(Htr.shape)} val H={tuple(Hva.shape)}")

        E = model.embed.weight.detach()
        results = {
            "experiment": "2660-latentreg-vs-disc-13m",
            "contract": "README.md (pre-registered)",
            "ckpt": os.path.relpath(ckpt, os.path.dirname(os.path.dirname(HERE))),
            "ckpt_sha256": ckpt_sha,
            "embed_sha256": embed_sha,
            "config": {"k_arms": {k: v["k"] for k, v in ARMS.items() if k in arms},
                       "lambda_arms": {k: v["lam"] for k, v in ARMS.items() if k in arms},
                       "steps": steps, "lr": LR, "batch_positions": L.BATCH_POSITIONS,
                       "data_seed": DATA_SEED, "head_init_seeds": a.seeds,
                       "seq_len": SEQ_LEN, "vocab_size": VOCAB_SIZE,
                       "n_train_blocks": n_tr, "n_val_blocks": n_va,
                       "optimizer": "Adam(no weight decay)",
                       "eval_positions_per_block": int(len(L.valid_positions(SEQ_LEN, 2))),
                       "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "cluster": "eval_block",
                                     "ci": "percentile95"}},
            "verdicts": {}, "ar": {}, "arms": {}, "L0": {}, "L1": {}, "L3": {}, "L5": {},
        }

        # --- ar_t1（参照，非变量）：冻结主干 node 1 的 t+1 命中率 ---
        k_ref = 2
        ar_blk = L.ar_t1_block_hits(tokva, n1va, k_ref)
        npos = int(len(L.valid_positions(SEQ_LEN, k_ref)))
        ar_hits = int(ar_blk.sum())
        results["ar"]["ar_t1"] = {
            "desc": "冻结 13M 主干 node 1（原生 AR 下一 token 头，decode.py:399）对 t_{i+1} 的 teacher-forced 命中率",
            "hit_t1": round(ar_hits / (ar_blk.shape[0] * npos), 5),
            "wilson95_t1": L.wilson_ci(ar_hits, ar_blk.shape[0] * npos),
            "desc_note": "非本项变量；1 次前向/token 的质量参照",
        }
        log(f"ar_t1 hit_t1={results['ar']['ar_t1']['hit_t1']}")

        # --- 训头 + 评臂 ---
        arm_eval: dict[str, dict[int, dict]] = {}
        for arm in arms:
            if arm == "ar_t1":
                continue
            spec = ARMS[arm]
            k = spec["k"]
            arm_eval[arm] = {}
            for seed in a.seeds:
                heads, traces = L.train_arm(Htr, toktr, E, k, spec["lam"], seed,
                                            steps=steps, device=dev)
                ev = L.eval_hits(heads, Hva, tokva, E, k, device=dev)
                hpath = os.path.join(out, "heads", f"{arm}_s{seed}.pt")
                hsha = L.save_heads(heads, hpath)
                arm_eval[arm][seed] = ev
                entry = {
                    "k": k, "lambda": spec["lam"], "seed": seed,
                    "n_pos": ev["n_pos"], "n_blocks": ev["n_blocks"],
                    "head_sha256": hsha, "head_path": os.path.relpath(hpath, out),
                    "ce_trace": traces,
                }
                for j in range(k):
                    tag = f"t{j+2}"
                    entry[f"hit_{tag}"] = ev[f"hit_{tag}"]
                    entry[f"ce_{tag}"] = round(ev[f"ce_{tag}"], 5)
                    entry[f"mse_{tag}"] = round(ev[f"mse_{tag}"], 5)
                    entry[f"wilson95_{tag}"] = L.wilson_ci(int(ev["_hits"][j].sum()), ev["n_pos"])
                    entry[f"head{j+1}_ce_trace"] = [
                        {"step": r["step"], "ce": round(r["per_head_ce"][j], 5)} for r in traces]
                results["arms"].setdefault(arm, {})[str(seed)] = entry
                log(f"arm={arm} seed={seed} hit_t2={entry['hit_t2']:.5f} "
                    f"hit_t3={entry.get('hit_t3')} head_sha256={hsha[:12]}")

        # --- L0：lat_k{k}_lam0 == disc_k{k}（同种子逐位相同）---
        l0_pairs = [(f"disc_k{kk}", f"lat_k{kk}_lam0", kk)
                    for kk in sorted({ARMS[x]["k"] for x in arm_eval})]
        for ref, eq, kk in l0_pairs:
            if ref not in arm_eval or eq not in arm_eval:
                continue
            for seed in a.seeds:
                d = arm_eval[ref].get(seed)
                l0 = arm_eval[eq].get(seed)
                if d is None or l0 is None:
                    continue
                same_hits = all(np.array_equal(d["_hits"][j], l0["_hits"][j])
                                for j in range(d["_hits"].shape[0]))
                same_ce = np.array_equal(d["_ce_blk"], l0["_ce_blk"])
                # 权重同一性 = 整份 state_dict 逐键 torch.equal（覆盖两个头的全部张量）；
                # `.pt` 的 sha256 只标识文件字节，不作内容同一性证明。
                same_w = L.head_files_all_tensors_equal(
                    os.path.join(out, "heads", f"{ref}_s{seed}.pt"),
                    os.path.join(out, "heads", f"{eq}_s{seed}.pt"))
                results["L0"][str(seed)] = {
                    "pair": [eq, ref],
                    "hits_bit_identical": bool(same_hits),
                    "ce_bit_identical": bool(same_ce),
                    "head_weights_bit_identical": bool(same_w),
                    "state_dict_all_tensors_equal": bool(same_w),
                    "pass": bool(same_hits and same_ce and same_w),
                    "note": "λ=0 两臂走同一代码路径（完全跳过 MSE）；同种子头初始化相同",
                }
        results["L0"]["all_pass"] = bool(results["L0"]) and all(
            results["L0"][str(s)]["pass"] for s in a.seeds if str(s) in results["L0"])
        l0_ks = sorted({kk for _, _, kk in l0_pairs})
        l0_pair_txt = ", ".join(f"lat_k{kk}_lam0 == disc_k{kk}" for kk in l0_ks)

        # --- 冻结断言（L0 的一部分）：主干 requires_grad=False 且 backward 后 grad is None ---
        frozen = all(not p.requires_grad for p in model.parameters())
        probe_heads = L.make_heads(2, 0, model.mcfg.d_model, dev)
        _zs, _lg = L.head_forward(probe_heads, Hva[:4].to(dev, dtype=torch.float32),
                                  E.to(dev), 2)
        (_lg[0].sum() + _lg[1].sum()).backward()
        no_grad = all(p.grad is None for p in model.parameters())
        results["L0"]["frozen_backbone"] = {
            "requires_grad_all_false": bool(frozen),
            "grad_is_none_after_backward": bool(no_grad),
            "pass": bool(frozen and no_grad),
        }
        probe_heads = None

        # --- L1：lat_* 相对 disc_k2 的配对差（t+2 主判 / t+3 并报）---
        def paired(ref, cand, j):
            per_seed = {}
            ds = []
            for seed in a.seeds:
                if seed not in arm_eval.get(ref, {}) or seed not in arm_eval.get(cand, {}):
                    continue
                r, c = arm_eval[ref][seed], arm_eval[cand][seed]
                bs = L.cluster_bootstrap_paired(c["_hits"][j], r["_hits"][j], c["_n_pos_blk"])
                per_seed[str(seed)] = bs
                ds.append(c["_hits"][j] - r["_hits"][j])
            if ds:
                comb = np.mean(np.stack(ds, 0), 0)
                npb = arm_eval[cand][a.seeds[0]]["_n_pos_blk"]
                per_seed["combined_mean_over_seeds"] = L.cluster_bootstrap_paired(
                    comb, np.zeros_like(comb), npb)
            return per_seed

        for cand in sorted(x for x in arm_eval if ARMS[x]["lam"] > 0.0):
            ref = f"disc_k{ARMS[cand]['k']}"
            if ref in arm_eval:
                results["L1"][cand] = {
                    "t+2": paired(ref, cand, 0),
                    "ref": ref,
                    "note": "Δ = cand - ref；按评测块聚类的配对 bootstrap；combined = 两 seed 的逐块 Δ 先平均再聚类",
                }
                for j in range(1, ARMS[cand]["k"]):
                    results["L1"][cand][f"t+{j+2}"] = paired(ref, cand, j)
        # ar_t1 与头的差（参考，不作判据：位置/训练 regime 不同）
        results["L1"]["_ref_note"] = (
            "t+2/t+3 为头输出位；ar_t1 为 t+1（同一冻结主干原生头）——位置不同，不作配对判据，仅质量参照")

        # --- L3：代价轴 ---
        d_model = model.mcfg.d_model
        all_arms_k = {ARMS[x]["k"] for x in arms if x in ARMS}
        results["L3"] = {
            "d_model": d_model, "n_layers": model.mcfg.n_layers,
            "backbone_params_frozen": int(sum(p.numel() for p in model.parameters())),
            "per_arm_param_stats": {x: L.param_stats(ARMS[x]["k"], d_model, VOCAB_SIZE)
                                    for x in arms if x in ARMS},
            "path_i_trunk_layer_passes_per_token": {f"k={kk}": model.mcfg.n_layers / kk
                                                    for kk in sorted(all_arms_k)},
            "path_ii_trunk_layer_passes_per_token": {f"k={kk}": (1 + kk) / kk * model.mcfg.n_layers
                                                     for kk in sorted(all_arms_k)},
            "note": "契约 §4-L3 口径（等效理论量）；阶段 1 只实测路径 (i)（teacher-forced 直接发）；"
                    "路径 (ii) 的 τ/α₁ 与真实接受成本由阶段 2 落盘",
        }

        # --- L5：λ 确实在起作用（CE/MSE 轨迹末端）---
        l5 = {}
        for arm, spec in ARMS.items():
            if arm not in results["arms"]:
                continue
            for seed_s, e in results["arms"][arm].items():
                tr = e["ce_trace"]
                l5[f"{arm}_s{seed_s}"] = {
                    "lambda": spec["lam"],
                    "ce_first": round(tr[0]["ce"], 5), "ce_last": round(tr[-1]["ce"], 5),
                    "mse_first": round(tr[0]["mse"], 5), "mse_last": round(tr[-1]["mse"], 5),
                    "mse_fraction_of_loss_last": round(
                        spec["lam"] * tr[-1]["mse"] / max(spec["lam"] * tr[-1]["mse"]
                                                          + tr[-1]["ce"], 1e-9), 5),
                }
        results["L5"] = {"traces_tail": l5,
                         "note": "λ>0 臂的 MSE 分量须非零且随训练下降，证明 λ 在起作用（未被 CE 淹没）",
                         "full_traces_in": "results.json → arms.<arm>.<seed>.ce_trace"}

        # --- 判词 ---
        l0_pass = results["L0"].get("all_pass", False) and \
            results["L0"]["frozen_backbone"]["pass"]
        l1_verdict = "否证（CI 含 0）"
        adopted = []
        for cand in sorted(x for x in arm_eval if ARMS[x]["lam"] > 0.0):
            blk = results["L1"].get(cand, {}).get("t+2", {}).get("combined_mean_over_seeds")
            if blk and blk["excludes_zero"]:
                adopted.append(cand)
        if not l0_pass:
            l1_verdict = "blocker：L0 未过（实现不等价），L1 不可判"
        elif adopted:
            l1_verdict = f"采纳（t+2 配对 95% CI 不含 0：{','.join(adopted)}）"
        results["verdicts"] = {
            "L0": f"pass（{l0_pair_txt} 逐位相同；冻结断言 pass）" if l0_pass
                  else "fail（见 L0 细节）",
            "L1": l1_verdict,
            "L3": "reported（等效理论量，不作判据）",
            "L5": "reported（λ>0 的 CE/MSE 轨迹落盘）",
            "L2": "阶段 2（另一 worker）：当草稿再验证 α₁/τ",
            "L4": "若 L1 与阶段 2 的 L2 都无显著差 ⇒ 判『本设定下无独立增益』",
        }

        # 禁墙钟自检
        bad = has_wall_clock(results)
        results["_wall_clock_fields"] = bad
        if bad:
            raise AssertionError(f"results.json 含墙钟字段：{bad}")
        with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(os.path.join(out, "train_log.jsonl"), "a", encoding="utf-8") as f:
            for arm, spec in ARMS.items():
                if arm not in results["arms"]:
                    continue
                for seed_s, e in results["arms"][arm].items():
                    f.write(json.dumps({"arm": arm, "seed": int(seed_s),
                                        "lambda": spec["lam"], "k": spec["k"],
                                        "trace": e["ce_trace"]}, ensure_ascii=False) + "\n")

        _write_model_md(out, ckpt, ckpt_sha, embed_sha, results, log)
        _write_analysis(out, results, a)
        log(f"DONE wall_seconds={time.time() - t_wall0:.1f} 写盘完成")
    return 0


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_model_md(out, ckpt, ckpt_sha, embed_sha, results, log):
    lines = [
        "# MODEL — 2660-latentreg-vs-disc-13m",
        "",
        "权重不入库（`.gitignore` 覆盖 `*.pt`）。本实验**零新增主干参数**：头是新训的",
        "`pathlm.model.TransformHead(d)`，主干与 tied 读出完全冻结。",
        "",
        "## 1 冻结主干（只读）",
        "",
        f"- 目录：`{results['ckpt']}`",
        f"- `model.pt` sha256：`{ckpt_sha}`",
        f"- tied 读出（embed.weight）sha256：`{embed_sha}`（两臂共用同一份，L0 硬门）",
        "",
        "## 2 训练好的头（阶段 2 直接加载）",
        "",
        "| 臂 | 种子 | k | λ | 文件 | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for arm, seeds in results["arms"].items():
        for s, e in seeds.items():
            lines.append(f"| `{arm}` | {s} | {e['k']} | {e['lambda']} | "
                         f"`{e['head_path']}` | `{e['head_sha256']}` |")
    lines += [
        "",
        "每个 `.pt` 是 `{head1: state_dict, head2: state_dict, ...}`（`TransformHead` 权重）。",
        "阶段 2 加载：`sd = torch.load('heads/<arm>_s<seed>.pt'); heads[j].load_state_dict(sd[f'head{j}'])`；",
        "读出 = `heads[j](h_i) @ model.embed.weight.T`（同一份冻结 embed）。",
        "",
        "## 3 复现命令",
        "",
        "```bash",
        "cd /home/z/vibe/pathlm",
        "python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q",
        "python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/run.py \\",
        "  --stage train --out <本目录> --ckpt axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0",
        "```",
        "",
    ]
    with open(os.path.join(out, "MODEL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("MODEL.md 落盘")


def _write_analysis(out, results, a):
    L0 = results["L0"]
    l0_ok = L0.get("all_pass") and L0.get("frozen_backbone", {}).get("pass")
    l0_arm_txt = " / ".join(f"`lat_k{kk}_lam0` 与 `disc_k{kk}`"
                             for kk in sorted(set(results["config"]["k_arms"].values())))
    ar = results["ar"]["ar_t1"]
    def fmt_ci(blk, tag):
        if not blk:
            return "—"
        return (f"{tag} Δ={blk['delta']:+.5f} CI[{blk['ci95'][0]:+.5f},{blk['ci95'][1]:+.5f}]"
                f" {'不含0' if blk['excludes_zero'] else '含0'}")
    lines = ["# analysis — 2660-latentreg-vs-disc-13m（阶段 1：直接发 / teacher-forced）", ""]
    lines.append("## 人话摘要")
    lines.append("")
    lines.append(
        "我们在**同一份冻结的 13M 主干**上，只改「未来位置怎么产出」这一件事：一种是直接把主干隐状态"
        "经几个小变换头**分类到词表**（离散臂），另一种是让这些头**回归到主干在未来位置的真实隐状态**"
        "（回归臂，多一个 λ·MSE 项把输出拉向真值隐状态），两者共用同一份 tied 读出、同一参数量、同一训练"
        "预算（3000 步）。")
    hit = results["arms"].get("disc_k2", {}).get("0", {})
    if hit:
        lines.append(
            f"结果：同参数量下，离散臂在 t+2 命中率 {hit['hit_t2']:.3f}、t+3 命中率 "
            f"{hit.get('hit_t3', float('nan')):.3f}；冻结主干自己的 AR 下一 token（t+1）命中率是 "
            f"{ar['hit_t1']:.3f}。")
    lines.append("")
    lines.append(f"- **L0（实现保真硬门）**：{'**通过**' if l0_ok else '**未过（blocker）**'} —— "
                 f"{l0_arm_txt} 同种子逐位相同（命中率 / CE / 头权重");
    lines.append("  " + "）；冻结主干 `requires_grad=False` 且一次 backward 后 `.grad is None`。")
    lines.append(f"- **L1（主判，路径 (i) 直接发）**：{results['verdicts']['L1']}")
    for cand in sorted(x for x in results["L1"] if not x.startswith("_")):
        blk = results["L1"].get(cand, {})
        if blk:
            parts = [fmt_ci(v.get("combined_mean_over_seeds"), k)
                     for k, v in sorted(blk.items()) if k.startswith("t+")]
            lines.append(f"  - `{cand}` vs `{blk.get('ref','disc_k2')}`：" + "；".join(parts))
    lines.append(f"- **L3/L5**：{results['verdicts']['L3']}；{results['verdicts']['L5']}")
    lines.append("")
    lines.append("> 口径提醒：率类指标走**配对 bootstrap / Wilson 95% CI**（不跨量纲借用 0.014 bpc 带）；"
                 "`results.json` **无任何墙钟字段**。")
    lines.append("")
    lines.append("## 臂与口径")
    lines.append("")
    lines.append(f"- 主干：`{results['ckpt']}`（冻结），tied 读出 embed sha256 `{results['embed_sha256'][:16]}…`")
    lines.append(f"- 特征 `h_i` = 最终层残差流；全序列前向经非侵入式包装 `_mtp_nodes` 抓取（包装前后主干输出逐位不变，见单测）")
    lines.append(f"- `ar_t1`（参照，非变量）= 冻结主干 node 1 的 t+1 命中率 = {ar['hit_t1']:.5f} {ar['wilson95_t1']}")
    lines.append(f"- j=1 ⇒ 预测 `t_2+…` 见契约 §2；k=2 头预测 **t+2 / t+3**")
    lines.append("")
    lines.append("## 数字表（teacher-forced 命中率，越低越差的率类指标）")
    lines.append("")
    lines.append("| 臂 | 种子 | λ | hit@t+2 | Wilson95 | hit@t+3 | Wilson95 | CE@t+2 | MSE@t+2 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arm, seeds in sorted(results["arms"].items()):
        for s, e in sorted(seeds.items()):
            lines.append(
                f"| `{arm}` | {s} | {e['lambda']} | {e.get('hit_t2',float('nan')):.5f} | "
                f"{e.get('wilson95_t2')} | {e.get('hit_t3',float('nan')):.5f} | "
                f"{e.get('wilson95_t3')} | {e.get('ce_t2')} | {e.get('mse_t2')} |")
    lines.append("")
    lines.append("## L0 明细")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({k: v for k, v in results["L0"].items() if k != "all_pass"},
                            indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## L3 代价轴（等效理论量，禁墙钟）")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results["L3"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## L5 λ 是否在起作用（CE/MSE 轨迹端点）")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results["L5"]["traces_tail"], indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 边界与未测")
    lines.append("")
    lines.append("- 结论只适用于本设定（该冻结 ckpt / char enwik8 / k=2 / 该训练预算 / tied 共享读出），不外推。")
    lines.append("- **路径 (ii)**（回归头当草稿再验证，α₁/τ）由**阶段 2**的另一个 worker 做；本文件只覆盖路径 (i)。")
    if any(x in results["config"]["k_arms"].values() for x in (4,)):
        lines.append("- k=4 臂：已测，见上表。")
    else:
        lines.append("- **k=4 臂未测**（时间盒）：下一步 = 同 harness 加 `disc_k4`/`lat_k4_lam1`，"
                     "成本 ≈ 每臂同 3000 步头训练（特征缓存复用，GPU 约 +1 min/臂）。")
    lines.append("")
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
