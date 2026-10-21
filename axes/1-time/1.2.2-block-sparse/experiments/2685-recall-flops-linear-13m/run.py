"""2685 · `1.2.2` 里程碑的另一半：长程召回腿 + 每 token 等效 FLOPs + 线性核无伤质量点。

预登记判据见 `README.md` §3；本脚本只做五件事：

1. **复用臂等值门**：本实验重算的 `win24a0` / `block8l4` / `win16a8`（六个复用臂）的 clean bpc 必须
   逐位复现 `2683` 落盘值——评测口径未漂移的证据；新增的 `full_s{0,1}` 与 `linear_s{0,1}` 由本实验训练。
2. **健康门**（判据 H，WP2 的前提）：线性臂 `grad_skips == 0`、`gnorm_max ≤ 12`、评测前向
   `den < 1e-30` 的行占比 == 0、φ 零占比 ≤ 1e-3（`linear_phi_audit.py` 逐层量）。
3. **质量腿**（判据 Q，主判）：`Δ = bpc_full − bpc_linear`，逐 seed 对 0.014 配对带三态；
   健康门未过则记「未取得」并写明原因，不给 Δ 判词。
4. **召回腿**（判据 R）：同一台针仪器（`recall_leg.py`，配置与 `2662` 逐字段相同），逐臂
   「植入 vs 配对地板」的 Newcombe + 按针配对 bootstrap CI；含块条带远块专桶 `far_slot`。
5. **等效 FLOPs**（判据 F）：同一次前向里按实际掩码计数（`flops_count.py`），与稠密基线和窗口臂对照。

用法（GPU 串行，≈4 min）：
    python3 run.py                 # 写 results.json
    python3 run.py --check-only    # 只跑等值门与健康门，不写盘
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "axes/1-time/1.2.2-block-sparse/experiments/"
                                      "2683-block-stripe-vs-band-13m"))

import block_sparse                                                            # noqa: E402
import flops_count                                                             # noqa: E402
import linear_phi_audit                                                        # noqa: E402
import recall_leg                                                              # noqa: E402
import stable_linear                                                           # noqa: E402
from pathlm.data import load_enwik8_full, batch                                # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, window_anchor_keep_mask  # noqa: E402
from tools.evalkit.retention import masked_logits                              # noqa: E402
from tools.prereg_bind import require as require_prereg                        # noqa: E402

EXP_2662 = os.path.join(ROOT, "axes/1-time/1.1.1-hard-window/experiments/"
                        "2662-hard-window-anchor-13m")
EXP_2677 = os.path.join(ROOT, "axes/1-time/1.2.2-block-sparse/experiments/"
                        "2677-local-global-vs-band-13m")
EXP_2683 = os.path.join(ROOT, "axes/1-time/1.2.2-block-sparse/experiments/"
                        "2683-block-stripe-vs-band-13m")
DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP = 256, 12, 8, 256, 6
N_BATCHES, BATCH_SIZE, DATA_SEED = 96, 16, 2
LN2 = 0.6931471805599453
KV_BYTES_PER_KEY = L * 2 * D * 2                 # fp16、MHA ⇒ 每位置每键 12288 B
BAND = 0.014                                     # harness/methodology.md §2（只用于配对 Δ）
GNORM_MAX_HEALTHY = 12.0                         # 已训纯注意力臂实测 9.4–10.8 的 1.2× 上界
GRAD_SKIPS_HEALTHY = 0

#: 十个臂：主判 = `full`（本实验训练，两 seed）对 `linear`（本实验训练，两 seed）；
#: 其余六个复用已训 checkpoint，用同一把尺子重测（等值门 + 召回腿 + FLOPs 腿）。
ARMS = [
    {"name": "full_s0", "mode": "full", "seed": 0, "cls": "base",
     "ckpt": os.path.join(HERE, "train/full_s0/model.pt"),
     "role": "全因果（主判对照臂，本实验训练）"},
    {"name": "full_s1", "mode": "full", "seed": 1, "cls": "base",
     "ckpt": os.path.join(HERE, "train/full_s1/model.pt"),
     "role": "全因果（第二 seed）"},
    {"name": "linear_s0", "mode": "linear", "seed": 0, "cls": "base",
     "ckpt": os.path.join(HERE, "train/linear_s0/model.pt"),
     "role": "线性核 φ=elu+1（主判实验臂，无归一化、无 QK-Norm）"},
    {"name": "linear_s1", "mode": "linear", "seed": 1, "cls": "base",
     "ckpt": os.path.join(HERE, "train/linear_s1/model.pt"),
     "role": "线性核（第二 seed）"},
    {"name": "win24a0_s0", "mode": "band", "window": 24, "anchors": 0, "seed": 0,
     "cls": "base", "ckpt": os.path.join(EXP_2677, "train/win24a0_s0/model.pt"),
     "role": "均匀条带最近 24（复用 `2677`；同字节参照）"},
    {"name": "win24a0_s1", "mode": "band", "window": 24, "anchors": 0, "seed": 1,
     "cls": "base", "ckpt": os.path.join(EXP_2677, "train/win24a0_s1/model.pt"),
     "role": "同上，第二 seed"},
    {"name": "block8l4_s0", "mode": "block_stripe", "window": 24, "anchors": 0, "seed": 0,
     "cls": "block_stripe", "ckpt": os.path.join(EXP_2683, "train/block8l4_s0/model.pt"),
     "role": "块条带（复用 `2683`；远块专桶的臂）"},
    {"name": "block8l4_s1", "mode": "block_stripe", "window": 24, "anchors": 0, "seed": 1,
     "cls": "block_stripe", "ckpt": os.path.join(EXP_2683, "train/block8l4_s1/model.pt"),
     "role": "同上，第二 seed"},
    {"name": "win16a8_s0", "mode": "local+global", "window": 16, "anchors": 8, "seed": 0,
     "cls": "base", "ckpt": os.path.join(EXP_2662, "train/win16a_s0/model.pt"),
     "role": "局部+全局（复用 `2662`；同字节轴上的第三个模式）"},
    {"name": "win16a8_s1", "mode": "local+global", "window": 16, "anchors": 8, "seed": 1,
     "cls": "base", "ckpt": os.path.join(EXP_2662, "train/win16a_s1/model.pt"),
     "role": "同上，第二 seed"},
]

#: 复用臂的落盘 bpc（等值门用；来源 = `2683/results.json::bpc_by_arm`）。
REUSE_REF = {
    "win24a0_s0": 1.599270, "win24a0_s1": 1.598631,
    "block8l4_s0": 1.610822, "block8l4_s1": 1.609415,
    "win16a8_s0": 1.622527, "win16a8_s1": 1.621799,
}
TRUNK_FULL_BPC = 1.502689        # `2683` 的 full（复用 2610 trunk，seed 0 口径）——只作 sanity 参照


def cfg_for(arm: dict) -> DepthARConfig:
    if arm["mode"] == "full":
        return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                             mlp_mult=MLP, dense=True)
    if arm["mode"] == "linear":
        return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                             mlp_mult=MLP, dense=True, attn_linear=True, linear_chunk=32)
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                         dense=True, train_window=arm["window"],
                         train_anchors=arm["anchors"])


def load_arm(arm: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    cls = block_sparse.BlockStripeModel if arm["cls"] == "block_stripe" else DepthARModel
    model = cls(cfg_for(arm), vocab).to(DEVICE)
    model.load_state_dict(torch.load(arm["ckpt"], map_location=DEVICE))
    model.eval()
    return model


def keep_mask_for(arm: dict) -> torch.Tensor | None:
    if arm["mode"] in ("full", "linear"):
        return None
    if arm["mode"] == "block_stripe":
        return block_sparse.block_stripe_keep_mask(SEQ, device=DEVICE)
    return window_anchor_keep_mask(SEQ, arm["window"], arm["anchors"], device=DEVICE)


def per_window_bits(arm: dict, model: DepthARModel, eval_arr):
    """逐窗口 bits（96×16 窗口，`window_seed=2`）⇒ 臂间可按窗口配对（与 `2683` 同口径）。"""
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
    return round(sum(bits) / sum(n_tok), 6), bits, n_tok


def paired_stats(comparator: list[float], arm: list[float], n: list[int]) -> dict:
    d = [(c - a) / k for c, a, k in zip(comparator, arm, n)]
    m = len(d)
    mean, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / m ** 0.5
    return {"n_windows": m, "delta_bpc": round(mean, 6),
            "sd_of_paired_delta": round(sd, 6), "se_of_mean": round(se, 6),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_windows_delta_gt_0": round(sum(1 for v in d if v > 0) / m, 3),
            "min_delta": round(min(d), 6), "max_delta": round(max(d), 6)}


def classify(delta: float) -> str:
    if delta >= BAND:
        return "线性更好（Δ ≥ +0.014，越带）"
    if delta <= -BAND:
        return "线性更差（Δ ≤ −0.014，越带）"
    return "无质量增量（|Δ| < 0.014，落带内）"


def linear_health(arm_name: str, model: DepthARModel, eval_arr) -> dict:
    """判据 H：训练期 telemetry（skips / gnorm）+ 评测前向的 φ 与分母审计。"""
    res = json.load(open(os.path.join(HERE, "train", arm_name, "results.json"),
                         encoding="utf-8"))
    g = res.get("gnorm_monitor") or {}
    gen = torch.Generator().manual_seed(11)
    x, _ = batch(eval_arr, 8, SEQ, gen)
    audit = linear_phi_audit.audit(model, x.to(DEVICE))
    agg = audit["aggregate"]
    checks = {
        "grad_skips_zero": {"value": g.get("grad_skips"),
                            "pass": g.get("grad_skips") == GRAD_SKIPS_HEALTHY},
        "gnorm_in_healthy_band": {"value": g.get("gnorm_max"), "max_allowed": GNORM_MAX_HEALTHY,
                                  "pass": g.get("gnorm_max") is not None
                                          and g["gnorm_max"] <= GNORM_MAX_HEALTHY},
        "den_guard_never_binds_at_eval": {"value": agg["den_below_eps_frac"], "max_allowed": 0.0,
                                          "pass": agg["den_below_eps_frac"] == 0.0},
        "phi_zero_fraction_small": {"value": agg["phi_zero_frac_max_any_layer"],
                                    "max_allowed": stable_linear.PHI_ZERO_FRACTION_MAX,
                                    "pass": agg["phi_zero_frac_max_any_layer"]
                                            <= stable_linear.PHI_ZERO_FRACTION_MAX},
    }
    healthy = all(c["pass"] for c in checks.values())
    return {"healthy": healthy, "checks": checks, "audit": audit,
            "train_source": os.path.relpath(os.path.join(HERE, "train", arm_name,
                                                         "results.json"), ROOT),
            "phi_numeric": res.get("linear_phi_numeric"),
            "criterion": (f"glossary: grad_skips == {GRAD_SKIPS_HEALTHY} ∧ gnorm_max ≤ "
                          f"{GNORM_MAX_HEALTHY}（已训纯注意力臂 9.4–10.8 的 1.2× 上界）∧ 评测前向 "
                          f"den < 1e-30 的行占比 == 0 ∧ φ 零占比 ≤ "
                          f"{stable_linear.PHI_ZERO_FRACTION_MAX}")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/1-time/1.2.2-block-sparse/experiments/2685-recall-flops-linear-13m")

    stable_linear.install()                     # 线性臂的 φ 与训练同源（无消去求值 + eps 1e-30）
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)

    missing = [a["name"] for a in ARMS if not os.path.exists(a["ckpt"])]
    if missing:
        raise SystemExit(f"缺 checkpoint：{missing}（新臂先跑 train_arms.sh）")

    bpc, bits, n_tok, params, flops, health = {}, {}, {}, {}, {}, {}
    models = {}
    for arm in ARMS:
        model = load_arm(arm, vocab)
        models[arm["name"]] = model
        params[arm["name"]] = sum(p.numel() for p in model.parameters())
        bpc[arm["name"]], bits[arm["name"]], n_tok[arm["name"]] = per_window_bits(
            arm, model, eval_arr)
        gen = torch.Generator().manual_seed(5)
        xf, _ = batch(eval_arr, 4, SEQ, gen)
        flops[arm["name"]] = flops_count.count_attention(model, xf, keep_mask_for(arm))
        if arm["mode"] == "linear":
            health[arm["name"]] = linear_health(arm["name"], model, eval_arr)
        print(f"[eval] {arm['name']}: bpc {bpc[arm['name']]:.6f} "
              f"equiv {flops[arm['name']]['equiv_macs_per_token']} MACs/token", flush=True)

    reuse_gate = {}
    for name, ref in REUSE_REF.items():
        ok = abs(bpc[name] - ref) < 1e-9
        reuse_gate[name] = {"reproduced": ok, "bpc": bpc[name], "ref_bpc": ref,
                            "ref": "2683/results.json::bpc_by_arm"}
        if not ok:
            raise SystemExit(f"等值门失败：{name} bpc {bpc[name]} != 落盘 {ref}")

    # 判据 Q：主判 = 同 seed 的 full 对 linear（Δ = bpc_full − bpc_linear，正 = 线性更好）
    quality = {s: paired_stats(bits[f"full_s{s}"], bits[f"linear_s{s}"], n_tok[f"linear_s{s}"])
               for s in (0, 1)}
    healthy_all = all(health[f"linear_s{s}"]["healthy"] for s in (0, 1))
    votes = [classify(quality[s]["delta_bpc"]) for s in (0, 1)]
    both_in_band = all(abs(quality[s]["delta_bpc"]) < BAND for s in (0, 1))
    same_sign = (quality[0]["delta_bpc"] >= 0) == (quality[1]["delta_bpc"] >= 0)
    if not healthy_all:
        verdict = "未取得（线性臂数值腿未过健康门：见 linear_health）"
    elif both_in_band:
        verdict = "无质量增量（两 seed 都落 0.014 带内）"
    elif same_sign and quality[0]["delta_bpc"] >= BAND and quality[1]["delta_bpc"] >= BAND:
        verdict = "线性更好（两 seed 越带、同向）"
    elif same_sign and quality[0]["delta_bpc"] <= -BAND and quality[1]["delta_bpc"] <= -BAND:
        verdict = "线性更差（两 seed 越带、同向）"
    else:
        verdict = "不可判（两 seed 不同向或只有一个越带）"

    # 判据 R：召回腿（同一台针仪器）
    batches = recall_leg.build_batches(eval_arr)
    block_keep = block_sparse.block_stripe_keep_mask(SEQ)
    band_keep = window_anchor_keep_mask(SEQ, 24, 0)
    far_flags = None
    recall = {}
    for arm in ARMS:
        rec = recall_leg.run_arm(models[arm["name"]], batches, keep_mask_for(arm))
        if far_flags is None:
            far_flags = recall_leg.far_slot_flags(rec["planted"], block_keep, band_keep)
        recall[arm["name"]] = recall_leg.readings(rec, far_flags)
        print(f"[recall] {arm['name']}: overall {recall[arm['name']]['overall']['planted_recall']}"
              f" floor {recall[arm['name']]['overall']['control_recall']}"
              f" | far_slot n={recall[arm['name']]['far_slot']['n']}", flush=True)

    out = {
        "experiment": "2685-recall-flops-linear-13m",
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "sd_of_paired_delta": "comparator_minus_arm",
            "frac_windows_delta_gt_0": "comparator_minus_arm",
            "min_delta": "comparator_minus_arm",
            "max_delta": "comparator_minus_arm",
        },
        "delta_direction_note": (
            "所有 bpc 读数一律 Δ = bpc_对照 − bpc_实验，正 = 实验更好；本实验主判的对照 = 同 seed 的"
            "`full`、实验 = `linear`（README §3 判据 Q）。召回读数是率类：`pp_diff` = 植入 − 地板"
            "（正 = 植入更好），不借用 0.014 带，因此不作为 Δ 字段声明。"),
        "config": {
            "d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ, "mlp_mult": MLP,
            "steps": 12000, "batch_size": 32, "lr": 1e-3, "seeds": [0, 1], "data": DATA,
            "linear": {"phi": "elu(x)+1", "phi_evaluation": "stable (exp(min(x,0)) + relu(x))",
                       "linear_chunk": 64,
                       "linear_attn_eps_declared": stable_linear.LINEAR_ATTN_EPS_DECLARED,
                       "kernel_changed": False},
            "eval": {"n_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                     "window_seed": DATA_SEED, "paired": True},
        },
        "arms": [{"name": a["name"], "mode": a["mode"], "seed": a["seed"], "role": a["role"],
                  "ckpt": os.path.relpath(a["ckpt"], ROOT), "params": params[a["name"]],
                  "k_eff": ({"mean": 256.0, "asymptotic": SEQ} if a["mode"] == "full"
                            else ({"mean": None, "asymptotic": None} if a["mode"] == "linear"
                                  else {"mean": round(float(
                                      keep_mask_for(a).sum(-1).float().mean()), 3),
                                      "asymptotic": 24}))}
                 for a in ARMS],
        "params_all_equal": len(set(params.values())) == 1,
        "params_value": params["full_s0"],
        "bpc_by_arm": bpc,
        "reuse_gate": reuse_gate,
        "full_vs_trunk_sanity": {
            "full_s0_bpc": bpc["full_s0"], "trunk_full_bpc": TRUNK_FULL_BPC,
            "bpc_diff_arm_minus_comparator": round(bpc["full_s0"] - TRUNK_FULL_BPC, 6),
            "note": ("本实验的 `full` 是新训的两条臂（同配方、seed 0/1）；与 `2683` 复用 2610 trunk 的"
                     "`full`（1.502689）只作 sanity 参照，不作等值门（不同 seed、不同训练目录）。")},
        "linear_health": health,
        "quality_linear_vs_full": quality,
        "main_verdict": {
            "criterion": ("Δ = bpc_full − bpc_linear（同 seed 配对；正 = 线性更好），对 0.014 带三态；"
                          "两 seed 各自判一次，同向且都越带才算可排序；健康门未过 ⇒ 记「未取得」"),
            "band_bpc": BAND, "seed0": votes[0], "seed1": votes[1],
            "both_within_band": both_in_band, "seeds_same_sign": same_sign, "n_seed": 2,
            "healthy_both_seeds": healthy_all, "verdict": verdict,
        },
        "recall_leg": {
            "instrument": ("needle v3：`tools/evalkit/retention.py::phrase_needle_batch` + "
                           "`needle_recall`；统计层复用 `2662/needle_stats.py`"
                           "（`paired_bootstrap_diff_pp`）与 `tools/evalkit/carry_channel.py`"
                           "（`two_proportion`）"),
            "config": recall_leg.NCFG, "n_real_tokens": recall_leg.N_REAL_TOKENS,
            "mask_token": recall_leg.MASK_TOKEN,
            "criteria": ("逐臂：`beyond`（与全保留臂的 `overall`）与 `far_slot` 的「植入 vs 配对地板」"
                         "两比例差 —— Newcombe 95% CI 下界 > 0 **且** 按针配对 bootstrap CI 下界 > 0 "
                         "⇒ 该臂在该桶上「高于地板」；否则记「不高于地板」并按观测比例报所需针数。"
                         "率类不借用 0.014 bpc 带。"),
            "far_slot_definition": ("短语在块条带掩码下 7 个位置全可见、且在均匀条带（最近 24）掩码下"
                                    "至少一个位置不可见 ⇒ 只可能经远块被看见；该子集按掩码定义、"
                                    "与臂无关，各臂在同一索引集上可比"),
            "n_far_slot_needles": int(sum(far_flags)) if far_flags else 0,
            "per_arm": recall,
        },
        "flops_leg": {
            "criteria": ("逐臂在同一次前向里计数：`equiv_macs_per_token`（按实际可见键数）与 "
                         "`executed_macs_per_token`（本条实装真正算掉的）分开报；两条结构关系必须成立："
                         "(a) 两个 24 键掩码模式（`win24a0` / `block8l4`）的 equiv 计数逐位相等；"
                         "(b) 24 键臂 / `full` 的 equiv 比值 = 解析预期 22.922/128.5；线性臂按 "
                         "`linear_attn_chunked` 的算式计数，对 T 线性（T=256 的绝对值如实报，不外推）。"),
            "per_arm": flops,
        },
        "byte_axis_per_position": {
            "formula": "12288 B/键 × k_eff（fp16、MHA、d=256 / L=12）",
            "full": {"k_eff": SEQ, "bytes": KV_BYTES_PER_KEY * SEQ},
            "win24a0": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24},
            "block8l4": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24},
            "win16a8": {"k_eff": 24, "bytes": KV_BYTES_PER_KEY * 24},
            "linear": {"state_bytes_per_position": None,
                       "note": ("线性核不保留 KV：流式推理每层每头一个 dh×dh 状态 + dh 归一化向量 ⇒ "
                                "每位置字节是**常数**（与 T 无关）。本实验不实测这项，不填数。")},
        },
    }

    if args.check_only:
        print("[check-only] 等值门与健康门已跑完（未写盘）", flush=True)
        return
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps({"bpc": bpc, "main": quality, "verdict": out["main_verdict"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
