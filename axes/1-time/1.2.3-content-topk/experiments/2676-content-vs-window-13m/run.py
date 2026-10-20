"""2676-content-vs-window-13m · 评测与对账（`1.2.3` 的 WP2 判决腿）。

预登记判据见 `README.md` §3；本脚本只做三件事：

1. **对齐核查**（判据④⑤⑥）：两条掩码路径（按内容 / 按位置）的逐行可见键数逐行相等、
   掩码恒为因果子集、五臂参数量逐位相等；
2. **配对质量读数**（判据①②）：20 批 × batch_size 16、`window_seed=2` 的逐批 clean bpc，
   与 `2614` 的 `paired_eval.py` 同一批位置、同一 batch 顺序 ⇒ 臂间差是配对差；
3. **申报量**（判据⑦）：逐臂一步耗时与峰值显存（**不进判据**，只申报）。

用法（GPU 串行，<2 min）：
    python3 run.py                 # 写 results.json
    python3 run.py --check-only    # 只跑对齐核查与等值门，不写盘

等值门：本脚本重算的 `full` / `topk32` / `topk64` 逐批 bpc 必须与
`2614-topk-sparse-attn-13m/paired_eval.json::per_batch_bpc` 逐位相同（口径未漂移的证据）。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, causal_topk_keep_mask,
                             window_anchor_keep_mask)
from pathlm.data import load_enwik8_full, batch
from tools.evalkit.retention import masked_logits

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP = 256, 8, 8, 256, 4
N_BATCHES, BATCH_SIZE, DATA_SEED = 20, 16, 2
LN2 = 0.6931471805599453

EXP_2614 = os.path.join(ROOT, "axes/1-time/1.2.3-content-topk/experiments/"
                        "2614-topk-sparse-attn-13m")

#: 五臂：前三条复用 2614 的 checkpoint（判据里的「少看键」参照与内容选择基线），
#: 后两条是本实验训练的按位置选择臂。`pick` 决定可见集由谁决定。
ARMS = [
    {"name": "full", "kind": "full", "k": 0, "window": 0,
     "ckpt": os.path.join(EXP_2614, "runs/full-12k-s0/model.pt"),
     "role": "基线：全因果（每行可见 i+1 个键）"},
    {"name": "topk32", "kind": "topk", "k": 32, "window": 0,
     "ckpt": os.path.join(EXP_2614, "runs/topk32-12k-s0/model.pt"),
     "role": "内容选择 k=32（因果内按 q·kᵀ 取前 32）"},
    {"name": "topk64", "kind": "topk", "k": 64, "window": 0,
     "ckpt": os.path.join(EXP_2614, "runs/topk64-12k-s0/model.pt"),
     "role": "内容选择 k=64"},
    {"name": "win32", "kind": "window", "k": 32, "window": 32,
     "ckpt": os.path.join(HERE, "runs/win32/model.pt"),
     "role": "位置选择 k=32（最近的 32 个位置）"},
    {"name": "win64", "kind": "window", "k": 64, "window": 64,
     "ckpt": os.path.join(HERE, "runs/win64/model.pt"),
     "role": "位置选择 k=64"},
]


def cfg_for(arm: dict) -> DepthARConfig:
    """内容 top-k 与位置窗口是**互斥的两个开关**（引擎的解析器拒两者同时开着），
    所以每臂只开自己那一个：`topk*` 开 `attn_sparse_k=k`，`win*` 开 `train_window=k`。"""
    window = arm["kind"] == "window"
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True,
                         attn_sparse_k=0 if window else arm["k"],
                         train_window=arm["window"] if window else 0)


def load_arm(arm: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)                      # 构造期的随机初始化不影响载入后的权重
    model = DepthARModel(cfg_for(arm), vocab).to(DEVICE)
    state = torch.load(arm["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def keep_mask_for(arm: dict, device=DEVICE) -> torch.Tensor | None:
    """该臂评测时的可见集：`full`/`topk` 由模型自己决定（`None`），`window` 显式给掩码。"""
    if arm["kind"] == "window":
        return window_anchor_keep_mask(SEQ, arm["window"], 0, device=device)
    return None


def per_batch_bpc(arm: dict, model: DepthARModel, eval_arr) -> list[float]:
    """逐批 clean bpc（每臂在自身可见性下评测，判据口径见 README §4）。"""
    keep = keep_mask_for(arm)
    gen = torch.Generator().manual_seed(DATA_SEED)
    out: list[float] = []
    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, _ = batch(eval_arr, BATCH_SIZE, SEQ, gen)
            x = x.to(DEVICE)
            logits = masked_logits(model, x, keep)[:, :-1].float()
            tgt = x[:, 1:]
            nats = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt.reshape(-1), reduction="sum").item()
            out.append(round(nats / tgt.numel() / LN2, 5))
    return out


def paired_stats(comparator: list[float], arm: list[float]) -> dict:
    """Δ = 对照 − 实验（正 = 实验更好）。"""
    d = [c - a for c, a in zip(comparator, arm)]
    n = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / n ** 0.5
    return {"n_batches": n, "delta_bpc": round(mean, 5),
            "sd_of_paired_delta": round(sd, 5),
            "se_of_mean": round(se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "min_delta": round(min(d), 5), "max_delta": round(max(d), 5)}


def alignment_checks() -> dict:
    """判据④⑤：两条掩码路径的因果性与逐行可见键数相等（掩码是静态的，逐行可核）。"""
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    rows = {"win_kept_per_row": {}, "topk_kept_per_row": {}}
    future_pairs, mismatch = 0, {}
    torch.manual_seed(0)
    # 内容 top-k 的逐行键数只取决于「因果允许键数」与 k，与分数取值无关（topk 取满 min(k,i+1)）；
    # 用随机分数把这条事实跑出来，而不是引用实现注释。
    scores = torch.randn(1, 1, SEQ, SEQ)
    for k in (32, 64):
        keep_topk = causal_topk_keep_mask(scores, k, causal)[0, 0]
        keep_win = window_anchor_keep_mask(SEQ, k, 0)
        c_topk = keep_topk.sum(-1).tolist()
        c_win = keep_win.sum(-1).tolist()
        rows["topk_kept_per_row"][f"k{k}"] = c_topk
        rows["win_kept_per_row"][f"k{k}"] = c_win
        mismatch[f"k{k}"] = int(sum(1 for a, b in zip(c_topk, c_win) if a != b))
        future_pairs += int((keep_topk & ~causal).sum()) + int((keep_win & ~causal).sum())
        assert c_topk == [min(k, i + 1) for i in range(SEQ)], "topk 行计数不是 min(k,i+1)"
        assert c_win == [min(k, i + 1) for i in range(SEQ)], "win 行计数不是 min(k,i+1)"
    return {"causal_pairs": SEQ * (SEQ + 1) // 2,
            "mask_offending_future_pairs": future_pairs,
            "mismatch_rows": mismatch,
            "kept_per_row_min": min(rows["win_kept_per_row"]["k32"]),
            "kept_per_row_max": max(rows["win_kept_per_row"]["k32"]),
            "win_kept_per_row": rows["win_kept_per_row"],
            "topk_kept_per_row": rows["topk_kept_per_row"],
            "note": ("逐行可见键数两臂相等（都 = min(k, i+1)，都含自身 j=i）；"
                     "本字段即判据⑤的落盘列")}


def forward_path_check(arm: dict, model: DepthARModel, eval_arr) -> dict:
    """判据④的同源证明：模型自己的前向（训练同路径）与显式掩码前向逐位相同。"""
    gen = torch.Generator().manual_seed(11)
    x, _ = batch(eval_arr, 4, SEQ, gen)
    x = x.to(DEVICE)
    with torch.no_grad():
        own = model(x, exit_depths=None)[1]["depth_logits"][-1].float()
        ref = masked_logits(model, x, keep_mask_for(arm)).float()
    return {"max_abs_logit_diff": float((own - ref).abs().max()), "batch": 4,
            "note": "模型自己的前向（含该臂可见集）vs masked_logits 的显式掩码前向"}


def step_time(model: DepthARModel, eval_arr, keep, iters: int = 8) -> dict:
    """一步（bs12 / T=256 / 前向+反向）的耗时与峰值显存：**只申报，不进判据**。"""
    gen = torch.Generator().manual_seed(7)
    x, _ = batch(eval_arr, 12, SEQ, gen)
    x = x.to(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        # 走模型自己的前向（= 训练同一条路径；`masked_logits` 自带 no_grad，量不了含反向的一步）
        logits = model(x, exit_depths=None)[1]["depth_logits"][-1][:, :-1].float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)
        times.append((time.perf_counter() - t0) * 1000)
    model.eval()
    torch.cuda.synchronize()
    return {"median_ms": round(statistics.median(times), 2),
            "min_ms": round(min(times), 2),
            "n_iters": iters,
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)

    align = alignment_checks()
    print(f"[align] mismatch_rows={align['mismatch_rows']} "
          f"future_pairs={align['mask_offending_future_pairs']}", flush=True)

    # 等值门：复用臂的读数必须复现 2614 的 paired_eval.json（口径未漂移）
    ref = json.load(open(os.path.join(EXP_2614, "paired_eval.json")))
    per_batch, params, cost, path_check = {}, {}, {}, {}
    for arm in ARMS:
        if not os.path.exists(arm["ckpt"]):
            raise SystemExit(f"缺 checkpoint：{arm['ckpt']}（先跑 train_arms.sh）")
        model = load_arm(arm, vocab)
        params[arm["name"]] = sum(p.numel() for p in model.parameters())
        per_batch[arm["name"]] = per_batch_bpc(arm, model, eval_arr)
        path_check[arm["name"]] = forward_path_check(arm, model, eval_arr)
        cost[arm["name"]] = step_time(model, eval_arr, keep_mask_for(arm))
        print(f"[eval] {arm['name']}: mean bpc {statistics.mean(per_batch[arm['name']]):.5f} "
              f"params {params[arm['name']]} step {cost[arm['name']]['median_ms']} ms",
              flush=True)

    reuse_ref = {}
    for name, ref_name in (("full", "full-12k-s0"), ("topk32", "topk32-12k-s0"),
                           ("topk64", "topk64-12k-s0")):
        same = per_batch[name] == [round(v, 5) for v in ref["per_batch_bpc"][ref_name]]
        reuse_ref[name] = {"reproduced": same,
                           "per_batch_equal": same,
                           "ref": "2614-topk-sparse-attn-13m/paired_eval.json::per_batch_bpc"}
        if not same:
            raise SystemExit(f"等值门失败：{name} 的逐批 bpc 未复现 2614 落盘值")

    mean_bpc = {k: round(statistics.mean(v), 5) for k, v in per_batch.items()}

    # Δ = 对照 − 实验（正 = 实验更好）；主判的对照 = 位置窗口、实验 = 内容 top-k
    vs_full = {name: paired_stats(per_batch["full"], per_batch[name])
               for name in ("topk32", "topk64", "win32", "win64")}
    content_vs_window = {f"k{k}": paired_stats(per_batch[f"win{k}"], per_batch[f"topk{k}"])
                         for k in (32, 64)}

    def verdict(block: dict) -> str:
        d = block["delta_bpc"]
        if d >= 0.014:
            return "内容选择占优（Δ ≥ +0.014，越带）"
        if d <= -0.014:
            return "内容选择更差（Δ ≤ -0.014，越带）"
        return "无质量增量（|Δ| < 0.014，落带内）"

    out = {
        "experiment": "2676-content-vs-window-13m",
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "sd_of_paired_delta": "comparator_minus_arm",
            "frac_batches_delta_gt_0": "comparator_minus_arm",
            "min_delta": "comparator_minus_arm",
            "max_delta": "comparator_minus_arm",
        },
        "delta_direction_note": (
            "所有 Δ 读数一律 Δ = bpc_对照 − bpc_实验，正 = 实验更好。"
            "`vs_full` 的对照 = full、实验 = 该臂；`content_vs_window` 的对照 = "
            "同 k 的位置窗口、实验 = 内容 top-k（README §3 的 Δ 方向）。"),
        "config": {
            "d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ, "mlp_mult": MLP,
            "steps": 12000, "batch_size": 32, "lr": 1e-3, "seed": 0,
            "data": DATA, "data_split": "前 90M 训练 / 末 10M 评测（enwik8，vocab 206）",
            "eval": {"n_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                     "window_seed": DATA_SEED, "paired": True},
            "variable": "可见键的选择规则（内容 vs 位置），每行可见键数逐行相等",
        },
        "arms": [{"name": a["name"], "kind": a["kind"], "k": a["k"],
                  "window": a["window"], "role": a["role"], "ckpt": a["ckpt"],
                  "params": params[a["name"]]} for a in ARMS],
        "visible_keys_per_row": align,
        "params_all_equal": len(set(params.values())) == 1,
        "params_value": params["full"],
        "mean_bpc_by_arm": mean_bpc,
        "per_batch_bpc": per_batch,
        "paired_vs_full": vs_full,
        "paired_content_vs_window": content_vs_window,
        "main_verdict": {
            "criterion": ("主判：Δ = bpc_win − bpc_topk（对照 = 位置窗口，实验 = 内容 top-k）；"
                          "Δ ≥ +0.014 ⇒ 内容选择占优；|Δ| < 0.014 ⇒ 无质量增量；"
                          "Δ ≤ -0.014 ⇒ 内容选择更差"),
            "band_bpc": 0.014,
            "k32": verdict(content_vs_window["k32"]),
            "k64": verdict(content_vs_window["k64"]),
            "n_seed": 1,
            "n_note": ("五臂都是 seed 0 单 run ⇒ 越带只能报「临时结果（n=1）」，"
                       "不越带则「同质量」这一读法对标量差本就成立"),
        },
        "cost_theory": {
            "kv_bytes_per_position_fp16": {
                "full": 12288 * SEQ, "topk32": 12288 * SEQ, "topk64": 12288 * SEQ,
                "win32": 12288 * 32, "win64": 12288 * 64},
            "kv_bytes_formula": "12288 B/键 × 每位置可见键数（fp16、MHA、d=256 / L=8）",
            "kv_cache_reduction_vs_full": {"topk32": 0.0, "topk64": 0.0,
                                          "win32": round(1 - 32 / SEQ, 4),
                                          "win64": round(1 - 64 / SEQ, 4)},
            "note": ("按内容选必须先算完整 T×T 分数矩阵 ⇒ 全部 K/V 必须在场，结构性 −0%；"
                     "按位置选的 −87.5%/−75% 是结构上界，本实装是稠密 bool 掩码，"
                     "未兑现为实测节省"),
        },
        "forward_path_check": path_check,
        "measured_step": cost,
        "reuse_equivalence": reuse_ref,
        "train_declaration": {},
    }

    for arm in ARMS:
        run_res = os.path.join(os.path.dirname(arm["ckpt"]), "results.json")
        if os.path.exists(run_res):
            r = json.load(open(run_res))
            out["train_declaration"][arm["name"]] = {
                "wall_minutes": r.get("wall_minutes"),
                "peak_mem_mib": (r.get("peak_gpu_mem_mb")
                                 or r.get("gnorm_monitor", {}).get("peak_gpu_mem_mb")),
                "source": os.path.relpath(run_res, ROOT),
            }

    if args.check_only:
        print("[check-only] 对齐核查与等值门全过", flush=True)
        return

    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps({"mean_bpc_by_arm": mean_bpc,
                      "content_vs_window": content_vs_window,
                      "verdict": out["main_verdict"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
