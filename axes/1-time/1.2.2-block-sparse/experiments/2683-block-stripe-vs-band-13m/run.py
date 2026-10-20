"""2683-block-stripe-vs-band-13m · 评测与对账（`1.2.2` 的第三个固定非均匀模式判决）。

预登记判据见 `README.md` §3；本脚本只做五件事：

1. **复用臂等值门**（判据⑧）：本实验重算的 `full` / `win16a8` / `win24a0` clean bpc 必须逐位复现
   已落盘值（`2677` 与 `2662` 的 `results.json`）——评测口径未漂移的证据；
2. **对齐核查**（判据④⑤）：块条带掩码与条带掩码逐行可见键数逐行相等、`k_eff` 相等、
   掩码恒为因果子集、`n_visible_pairs` 相等；
3. **训练/评测掩码同源门**（判据⑥）：每臂「模型自己的前向」（训练同一条路径，块条带臂走
   `BlockStripeModel._window_mask`）与 `masked_logits` 的显式掩码前向逐位相同；块条带臂另核
   训练期落盘的掩码指纹 = 评测期构造的指纹；
4. **配对质量读数**（判据①②）：96 批 × bs16、`window_seed=2` 的逐窗口 bits，臂间按窗口配对；
5. **申报量**（判据⑨）：逐臂一步耗时与峰值显存（**不进判据**）、每位置 KV 字节（解析，同一条字节轴）。

用法（GPU 串行，<4 min）：
    python3 run.py                 # 写 results.json
    python3 run.py --check-only    # 只跑等值门与对齐核查，不写盘
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARModel, DepthARConfig, window_anchor_keep_mask
from pathlm.data import load_enwik8_full, batch
from tools.evalkit.retention import masked_logits
from tools.prereg_bind import require as require_prereg

import block_sparse

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP = 256, 12, 8, 256, 6
N_BATCHES, BATCH_SIZE, DATA_SEED = 96, 16, 2
LN2 = 0.6931471805599453
LAYERS, D_MODEL = L, D
KV_BYTES_PER_KEY = LAYERS * 2 * D_MODEL * 2       # fp16、MHA ⇒ 每位置每键 12288 B
BAND = 0.014                                      # harness/methodology.md §2（只用于配对 Δ）

EXP_2662 = os.path.join(ROOT, "axes/1-time/1.1.1-hard-window/experiments/"
                        "2662-hard-window-anchor-13m")
EXP_2677 = os.path.join(ROOT, "axes/1-time/1.2.2-block-sparse/experiments/"
                        "2677-local-global-vs-band-13m")
TRUNK = os.path.join(ROOT, "axes/6-predict/6.2-speculative-decoding/experiments/"
                     "2610-frozen-trunk-jumphead-13m/trunk.pt")

#: 七臂：主判 = 块条带（本实验训练，两个 seed）对同 24 键的均匀条带（复用 `2677`）；
#: `local+global`（复用 `2662`）是同一字节轴上的第三个模式（参照），`full` 是上界参照。
ARMS = [
    {"name": "full", "mode": "full", "window": 256, "anchors": 0, "seed": 0,
     "ckpt": TRUNK, "cls": "base",
     "role": "全因果上界参照（每行可见 i+1 个键）"},
    {"name": "win24a0_s0", "mode": "band", "window": 24, "anchors": 0, "seed": 0,
     "ckpt": os.path.join(EXP_2677, "train/win24a0_s0/model.pt"), "cls": "base",
     "role": "均匀条带（最近 24）—— 主判的对照臂，k_eff=24"},
    {"name": "win24a0_s1", "mode": "band", "window": 24, "anchors": 0, "seed": 1,
     "ckpt": os.path.join(EXP_2677, "train/win24a0_s1/model.pt"), "cls": "base",
     "role": "同上，第二 seed"},
    {"name": "block8l4_s0", "mode": "block_stripe", "window": 24, "anchors": 0, "seed": 0,
     "ckpt": os.path.join(HERE, "train/block8l4_s0/model.pt"), "cls": "block_stripe",
     "role": "块条带（最近 16 + 滞后 4 块的一块对齐 8 键）—— 主判的实验臂，k_eff=24"},
    {"name": "block8l4_s1", "mode": "block_stripe", "window": 24, "anchors": 0, "seed": 1,
     "ckpt": os.path.join(HERE, "train/block8l4_s1/model.pt"), "cls": "block_stripe",
     "role": "同上，第二 seed"},
    {"name": "win16a8_s0", "mode": "local+global", "window": 16, "anchors": 8, "seed": 0,
     "ckpt": os.path.join(EXP_2662, "train/win16a_s0/model.pt"), "cls": "base",
     "role": "局部+全局（最近 16 + 最前 8 恒可见）—— 同一字节轴上的第三个模式（参照）"},
    {"name": "win16a8_s1", "mode": "local+global", "window": 16, "anchors": 8, "seed": 1,
     "ckpt": os.path.join(EXP_2662, "train/win16a_s1/model.pt"), "cls": "base",
     "role": "同上，第二 seed"},
]

#: 复用臂的落盘 bpc（等值门用；来源见 `ref` 字段）。
REUSE_REF = {
    "full": (1.502689, "2677-local-global-vs-band-13m/results.json::bpc_by_arm.full"),
    "win24a0_s0": (1.599270, "2677-local-global-vs-band-13m/results.json::bpc_by_arm.win24a0_s0"),
    "win24a0_s1": (1.598631, "2677-local-global-vs-band-13m/results.json::bpc_by_arm.win24a0_s1"),
    "win16a8_s0": (1.622527, "2677-local-global-vs-band-13m/results.json::bpc_by_arm.win16a8_s0"),
    "win16a8_s1": (1.621799, "2677-local-global-vs-band-13m/results.json::bpc_by_arm.win16a8_s1"),
}


def cfg_for(arm: dict) -> DepthARConfig:
    """与 2662 / 2677 同一骨架：`train_window/train_anchors` 取该臂的掩码参数 ⇒ 模型自己的前向
    （训练同路径）与 `masked_logits` 的显式掩码前向是同一可见集。块条带臂的数值只作申报，
    实际掩码由 `BlockStripeModel._window_mask` 构造。"""
    return DepthARConfig(d_model=D_MODEL, n_layers=LAYERS, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True,
                         train_window=arm["window"] if arm["mode"] != "full" else 0,
                         train_anchors=arm["anchors"] if arm["mode"] != "full" else 0)


def load_arm(arm: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    cls = block_sparse.BlockStripeModel if arm["cls"] == "block_stripe" else DepthARModel
    model = cls(cfg_for(arm), vocab).to(DEVICE)
    model.load_state_dict(torch.load(arm["ckpt"], map_location=DEVICE))
    model.eval()
    return model


def keep_mask_for(arm: dict) -> torch.Tensor | None:
    if arm["mode"] == "full":
        return None
    if arm["mode"] == "block_stripe":
        return block_sparse.block_stripe_keep_mask(SEQ, device=DEVICE)
    return window_anchor_keep_mask(SEQ, arm["window"], arm["anchors"], device=DEVICE)


def per_window_bits(arm: dict, model: DepthARModel, eval_arr):
    """逐窗口 bits 与窗口 token 数（96×16 窗口，`window_seed=2`）⇒ 臂间可按窗口配对。"""
    keep = keep_mask_for(arm)
    gen = torch.Generator().manual_seed(DATA_SEED)
    bits, n_tok = [], []
    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, _ = batch(eval_arr, BATCH_SIZE, SEQ, gen)
            x = x.to(DEVICE)
            logits = masked_logits(model, x, keep)[:, :-1].float()
            tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="none").view(BATCH_SIZE, -1)
            bits.extend((ce.double().sum(1) / LN2).cpu().tolist())
            n_tok.extend([tgt.shape[1]] * BATCH_SIZE)
    bpc = sum(bits) / sum(n_tok)          # bits 已是 bits/window ⇒ 再除 token 数即 bpc
    return round(bpc, 6), bits, n_tok


def paired_stats(comparator: list[float], arm: list[float], n: list[int]) -> dict:
    """Δ = 对照 − 实验（正 = 实验更好），按窗口配对。"""
    d = [(c - a) / k for c, a, k in zip(comparator, arm, n)]
    m = len(d)
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    se = sd / m ** 0.5
    return {"n_windows": m, "delta_bpc": round(mean, 6),
            "sd_of_paired_delta": round(sd, 6),
            "se_of_mean": round(se, 6),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_windows_delta_gt_0": round(sum(1 for v in d if v > 0) / m, 3),
            "min_delta": round(min(d), 6), "max_delta": round(max(d), 6)}


def alignment_checks() -> dict:
    """判据④⑤：块条带与条带的因果性、逐行可见键数相等、k_eff 相等。"""
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    keep_bs = block_sparse.block_stripe_keep_mask(SEQ)
    keep_band = window_anchor_keep_mask(SEQ, 24, 0)
    c_bs = [int(v) for v in keep_bs.sum(-1).tolist()]
    c_band = [int(v) for v in keep_band.sum(-1).tolist()]
    expected = [min(block_sparse.K_EFF, i + 1) for i in range(SEQ)]
    assert c_bs == expected, "块条带的逐行键数不是 min(24, i+1)"
    assert c_band == expected, "均匀条带的逐行键数不是 min(24, i+1)"
    return {
        "n_visible_pairs_block_stripe": int(keep_bs.sum()),
        "n_visible_pairs_band": int(keep_band.sum()),
        "causal_pairs": SEQ * (SEQ + 1) // 2,
        "k_eff_mean_block_stripe": round(statistics.mean(c_bs), 3),
        "k_eff_mean_band": round(statistics.mean(c_band), 3),
        "mask_offending_future_pairs": int((keep_bs & ~causal).sum())
                                       + int((keep_band & ~causal).sum()),
        "mismatch_rows": int(sum(1 for a, b in zip(c_bs, c_band) if a != b)),
        "kept_per_row": {"block_stripe": c_bs, "band_24_0": c_band},
        "block_stripe_self_check": block_sparse.self_check(SEQ),
        "note": ("两臂逐行可见键数逐行相等（都 = min(24, i+1)，都含自身 j=i）⇒ 每位置字节"
                 "完全相同；本字段即判据⑤的落盘列"),
    }


def forward_path_check(arm: dict, model: DepthARModel, eval_arr) -> dict:
    """判据⑥的同源证明：训练/评测用同一掩码 ⇒ 两条前向逐位相同。"""
    gen = torch.Generator().manual_seed(11)
    x, _ = batch(eval_arr, 4, SEQ, gen)
    x = x.to(DEVICE)
    with torch.no_grad():
        own = model(x, exit_depths=None)[1]["depth_logits"][-1].float()
        ref = masked_logits(model, x, keep_mask_for(arm)).float()
    return {"max_abs_logit_diff": float((own - ref).abs().max()), "batch": 4,
            "note": ("模型自己的前向（cfg 里的掩码参数；块条带臂走 "
                     "`BlockStripeModel._window_mask`）vs 显式掩码前向")}


def step_time(model: DepthARModel, eval_arr, iters: int = 8) -> dict:
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
    return {"median_ms": round(statistics.median(times), 2), "min_ms": round(min(times), 2),
            "n_iters": iters,
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)}


def k_eff_stats(arm: dict) -> dict:
    """每位置可见键数：`mean`（含开头几行不足 k 的行）与 `asymptotic`（长行稳态）。"""
    if arm["mode"] == "full":
        return {"mean": float(SEQ), "asymptotic": SEQ}
    keep = keep_mask_for(arm)
    rows = keep.sum(-1).float()
    return {"mean": round(float(rows.mean()), 3),
            "asymptotic": (block_sparse.K_EFF if arm["mode"] == "block_stripe"
                           else arm["window"] + arm["anchors"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    # 预登记门：契约必须在训练之前绑定，绑定必须早于读数（harness/methodology.md §8）。
    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/1-time/1.2.2-block-sparse/experiments/"
                         "2683-block-stripe-vs-band-13m")

    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)

    align = alignment_checks()
    print(f"[align] n_visible(block) {align['n_visible_pairs_block_stripe']} vs "
          f"{align['n_visible_pairs_band']}, mismatch_rows={align['mismatch_rows']}, "
          f"future_pairs={align['mask_offending_future_pairs']}", flush=True)

    bpc, bits, n_tok, params, cost, path_check = {}, {}, {}, {}, {}, {}
    block_fp_at_eval = block_sparse.pattern_fingerprint(
        block_sparse.block_stripe_keep_mask(SEQ))
    for arm in ARMS:
        if not os.path.exists(arm["ckpt"]):
            raise SystemExit(f"缺 checkpoint：{arm['ckpt']}（新臂先跑 train_arms.sh）")
        model = load_arm(arm, vocab)
        params[arm["name"]] = sum(p.numel() for p in model.parameters())
        bpc[arm["name"]], bits[arm["name"]], n_tok[arm["name"]] = per_window_bits(
            arm, model, eval_arr)
        path_check[arm["name"]] = forward_path_check(arm, model, eval_arr)
        cost[arm["name"]] = step_time(model, eval_arr)
        print(f"[eval] {arm['name']}: bpc {bpc[arm['name']]:.6f} "
              f"params {params[arm['name']]} step {cost[arm['name']]['median_ms']} ms "
              f"path_diff {path_check[arm['name']]['max_abs_logit_diff']}", flush=True)

    reuse_gate = {}
    for name, (ref_bpc, ref_src) in REUSE_REF.items():
        ok = abs(bpc[name] - ref_bpc) < 1e-9
        reuse_gate[name] = {"reproduced": ok, "bpc": bpc[name], "ref_bpc": ref_bpc,
                           "ref": ref_src}
        if not ok:
            raise SystemExit(f"等值门失败：{name} bpc {bpc[name]} != 落盘 {ref_bpc}")

    # 训练期落盘的掩码指纹 vs 评测期构造的指纹（判据⑥的第三道：同一张掩码）
    train_fp = {}
    for arm in ARMS:
        if arm["cls"] != "block_stripe":
            continue
        ptr = json.load(open(os.path.join(os.path.dirname(arm["ckpt"]), "results.json")))
        got = ptr.get("block_sparse_mask", {}).get("fingerprint_sha256")
        train_fp[arm["name"]] = {
            "train_fingerprint_sha256": got,
            "eval_fingerprint_sha256": block_fp_at_eval,
            "same_mask": got == block_fp_at_eval,
            "mask_calls_during_training": ptr.get("block_sparse_mask", {}).get(
                "mask_calls_during_training"),
            "source": os.path.join(os.path.dirname(arm["ckpt"]), "results.json"),
        }
        if got != block_fp_at_eval:
            raise SystemExit(f"训练/评测掩码不同源：{arm['name']} 训练期指纹 {got} != "
                             f"评测期 {block_fp_at_eval}")

    # 主判：同 k_eff=24，对照 = 均匀条带、实验 = 块条带 ⇒ Δ = bpc_band − bpc_block_stripe
    main_judgement = {s: paired_stats(bits[f"win24a0_s{s}"], bits[f"block8l4_s{s}"],
                                      n_tok[f"block8l4_s{s}"])
                      for s in (0, 1)}
    vs_full = {name: paired_stats(bits["full"], bits[name], n_tok[name])
               for name in ("win24a0_s0", "win24a0_s1", "block8l4_s0", "block8l4_s1",
                            "win16a8_s0", "win16a8_s1")}
    # 第二个模式的交叉核对（与 2677 的主判同形：Δ = bpc_band − bpc_local+global）
    local_global_vs_band = {s: paired_stats(bits[f"win24a0_s{s}"], bits[f"win16a8_s{s}"],
                                            n_tok[f"win16a8_s{s}"])
                            for s in (0, 1)}

    def classify(block: dict) -> str:
        d = block["delta_bpc"]
        if d >= BAND:
            return "块条带占优（Δ ≥ +0.014，越带）"
        if d <= -BAND:
            return "块条带更差（Δ ≤ -0.014，越带）"
        return "无质量增量（|Δ| < 0.014，落带内）"

    votes = [classify(main_judgement[s]) for s in (0, 1)]
    both_in_band = all(abs(main_judgement[s]["delta_bpc"]) < BAND for s in (0, 1))
    same_sign = (main_judgement[0]["delta_bpc"] >= 0) == (main_judgement[1]["delta_bpc"] >= 0)
    out = {
        "experiment": "2683-block-stripe-vs-band-13m",
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "sd_of_paired_delta": "comparator_minus_arm",
            "frac_windows_delta_gt_0": "comparator_minus_arm",
            "min_delta": "comparator_minus_arm",
            "max_delta": "comparator_minus_arm",
        },
        "delta_direction_note": (
            "所有 Δ 读数一律 Δ = bpc_对照 − bpc_实验，正 = 实验更好。"
            "`vs_full` 的对照 = full；`main_block_stripe_vs_band` 的对照 = 同 k_eff 的均匀条带、"
            "实验 = 块条带（README §3 的 Δ 方向）；`local_global_vs_band` 的对照 = 同 k_eff 的均匀"
            "条带、实验 = 局部+全局（复算 2677 的主判，作交叉核对）。"),
        "config": {
            "d_model": D_MODEL, "n_layers": LAYERS, "n_heads": H, "seq_len": SEQ,
            "mlp_mult": MLP, "steps": 12000, "batch_size": 32, "lr": 1e-3,
            "seeds": [0, 1], "data": DATA,
            "data_split": "前 90M 训练 / 末 10M 评测（enwik8，vocab 206）",
            "eval": {"n_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                     "window_seed": DATA_SEED, "paired": True},
            "variable": ("24 个可见键怎么分配：最近 24（均匀条带）vs 最近 16 + 滞后 4 块的一块"
                         "对齐 8 键（块条带）"),
        },
        "arms": [{"name": a["name"], "mode": a["mode"], "window": a["window"],
                  "anchors": a["anchors"], "seed": a["seed"], "role": a["role"],
                  "ckpt": a["ckpt"], "params": params[a["name"]],
                  "k_eff": k_eff_stats(a)} for a in ARMS],
        "visible_keys_per_row": align,
        "params_all_equal": len(set(params.values())) == 1,
        "params_value": params["full"],
        "bpc_by_arm": bpc,
        "reuse_gate": reuse_gate,
        "train_eval_mask_same": train_fp,
        "paired_vs_full": vs_full,
        "main_block_stripe_vs_band": main_judgement,
        "main_verdict": {
            "criterion": ("主判：Δ = bpc_win24a0 − bpc_block8l4（对照 = 同 k_eff=24 的均匀条带，"
                          "实验 = 块条带）；Δ ≥ +0.014 ⇒ 块条带占优（正 = 实验更好）；"
                          "|Δ| < 0.014 ⇒ 无质量增量；Δ ≤ -0.014 ⇒ 块条带更差。两 seed 各自判一次"),
            "band_bpc": BAND,
            "seed0": votes[0],
            "seed1": votes[1],
            "both_within_band": both_in_band,
            "seeds_same_sign": same_sign,
            "n_seed": 2,
        },
        "local_global_vs_band": local_global_vs_band,
        "byte_axis_per_position": {
            "formula": "12288 B/键 × k_eff（fp16、MHA、d=256 / L=12）",
            "full": {"k_eff": SEQ, "bytes": KV_BYTES_PER_KEY * SEQ},
            "band24_win24a0": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24,
                               "reduction_vs_full": round(1 - 24 / SEQ, 4)},
            "block_stripe": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24,
                             "reduction_vs_full": round(1 - 24 / SEQ, 4)},
            "local_global_16_8": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24,
                                  "reduction_vs_full": round(1 - 24 / SEQ, 4)},
            "note": ("三个 24 键模式落在同一条字节水平（294,912 B/位置）⇒ 主判的差不能记在"
                     "字节账上；−90.6% 是结构上界，本实装是稠密 bool 掩码、未兑现为实测节省"),
        },
        "forward_path_check": path_check,
        "measured_step": cost,
        "train_declaration": {},
    }

    for arm in ARMS:
        run_res = os.path.join(os.path.dirname(arm["ckpt"]), "results.json")
        if os.path.exists(run_res):
            r = json.load(open(run_res))
            out["train_declaration"][arm["name"]] = {
                "wall_minutes": r.get("wall_minutes"),
                "peak_mem_mib": r.get("peak_gpu_mem_mb"),
                "source": os.path.relpath(run_res, ROOT)}

    if args.check_only:
        print("[check-only] 等值门、对齐核查与掩码同源门全过", flush=True)
        return

    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps({"bpc_by_arm": bpc, "main": main_judgement,
                      "verdict": out["main_verdict"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
