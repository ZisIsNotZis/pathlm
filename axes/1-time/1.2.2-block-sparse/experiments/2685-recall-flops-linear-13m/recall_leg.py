"""2685 · 长程召回腿：**同一台针仪器、同一判据**，在本实验自己的臂上给读数。

仪器与口径的原文在 `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/README.md`
§3 判据②③：needle v3（真实文本短语针，`tools/evalkit/retention.py::phrase_needle_batch` / `needle_recall`）
＋「植入 vs 配对无针地板」的两比例差（Newcombe 95% CI）与**按针配对** bootstrap CI，两者下界都 > 0
才算「高于地板」。率类读数**不**借用 0.014 bpc 带（`harness/methodology.md` §2 的跨量纲纪律）。

本模块只做三件事：

1. 用**与 `2662` 逐字段相同**的配置造针批次（`NCFG`：64 批 × 16 行 × 8 针/行、`phrase_len=6`、
   归属桶一律 `anchors=8`、`seed=3`），并对**同一批**窗口跑植入 / 配对控制两遍；
2. 逐臂用**它自己训练时那张掩码**评测（分布内，不换掩码）⇒ 召回读数回答「这个臂能不能召回
   远端内容」，而不是「换掩码会怎样」；
3. 按预登记的子集判：

   - `beyond`：非锚针且短语有任意位置对该答案行不可见（`2662` 的「窗外」桶）；
   - `far_slot`：**块条带臂的远块专桶** —— 短语在块条带掩码下 7 个位置全可见、而在均匀条带（最近 24）
     掩码下至少一个位置不可见 ⇒ 这些针只可能经远块被看见。该子集按掩码定义、与臂无关，因此各臂
     在同一索引集上可比；
   - 全保留掩码的臂（`full` / 线性核）**没有** `beyond` 桶（构造使然）：它们的参照读数是总体召回，
     落盘时标 `no_beyond_bucket_by_construction`。

样本量规则：CI 下界 ≤ 0 时按观测比例报所需针数
`n* = ceil((z₀.₉₇₅ + z₀.₈₀)² · (p₁(1−p₁) + p₀(1−p₀)) / (p₁−p₀)²)`（两比例差的常用规划式，
`z = 1.96 / 0.8416`，检验水准 0.05、把握度 0.8）；观测到 `p₁ ≤ p₀` 时报「不存在（效应 ≤ 0）」。
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from tools.evalkit.carry_channel import two_proportion                     # noqa: E402
from tools.evalkit.retention import needle_recall, needle_summary, phrase_needle_batch  # noqa: E402

EXP_2662 = os.path.join(ROOT, "axes/1-time/1.1.1-hard-window/experiments/"
                        "2662-hard-window-anchor-13m")


def _load_needle_stats():
    """复用 `2662` 的统计层（`paired_bootstrap_diff_pp` / `recall_on_indices`）：同一件仪器。"""
    spec = importlib.util.spec_from_file_location(
        "needle_stats_2662", os.path.join(EXP_2662, "needle_stats.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ns = _load_needle_stats()

#: 与 `2662/config.json::needle` 逐字段相同（同入口的证明就在这里：改这里就不再是同一条腿）。
NCFG = {"n_needles": 8, "phrase_len": 6, "anchors": 8, "anchor_frac": 0.25,
        "n_batches": 64, "batch_size": 16, "seed": 3, "bucket_anchors": 8}
#: `2662/config.json::data`：真实字节 0..204 可用，205 是掩码标记。
N_REAL_TOKENS = 205
MASK_TOKEN = 205

Z_975, Z_80 = 1.959964, 0.841621


def build_batches(eval_arr) -> list[tuple[torch.Tensor, torch.Tensor, list[list[dict]]]]:
    """按 `2662` 的抽样序列造 `n_batches` 对（植入, 配对控制, 同一 meta）。"""
    gen = torch.Generator().manual_seed(NCFG["seed"])
    batch = int(NCFG["batch_size"])
    T = 256
    out = []
    for _ in range(int(NCFG["n_batches"])):
        state = gen.get_state()
        x, meta = phrase_needle_batch(batch, T, eval_arr, N_REAL_TOKENS, MASK_TOKEN, gen,
                                      n_needles=NCFG["n_needles"],
                                      phrase_len=NCFG["phrase_len"],
                                      anchors=NCFG["anchors"],
                                      anchor_frac=NCFG["anchor_frac"], plant=True)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(batch, T, eval_arr, N_REAL_TOKENS, MASK_TOKEN, gen,
                                        n_needles=NCFG["n_needles"],
                                        phrase_len=NCFG["phrase_len"],
                                        anchors=NCFG["anchors"],
                                        anchor_frac=NCFG["anchor_frac"], plant=False)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        out.append((x, x0, meta))
    return out


def far_slot_flags(records: list[dict], block_keep: torch.Tensor, band_keep: torch.Tensor,
                   phrase_len: int = NCFG["phrase_len"]) -> list[bool]:
    """逐针（与 `records` 索引对齐）：短语在块条带掩码下全可见、且在均匀条带（最近 24）下有位置不可见。"""
    flags = []
    for r in records:
        q, p = r["q_row"], r["p"]
        vis_block = all(bool(block_keep[q, p + j]) for j in range(phrase_len + 1))
        vis_band = all(bool(band_keep[q, p + j]) for j in range(phrase_len + 1))
        flags.append(bool(vis_block and not vis_band))
    return flags


def run_arm(model, batches, keep_mask, anchors: int = NCFG["bucket_anchors"],
            phrase_len: int = NCFG["phrase_len"], device: str = "cuda") -> dict:
    """一个臂跑两遍（植入 / 配对控制），返回逐针记录（索引对齐）。"""
    planted, control = [], []
    for x, x0, meta in batches:
        planted.extend(needle_recall(model, x, meta, keep_mask, anchors=anchors,
                                     phrase_len=phrase_len, device=device)["per_needle"])
        control.extend(needle_recall(model, x0, meta, keep_mask, anchors=anchors,
                                     phrase_len=phrase_len, device=device)["per_needle"])
    if len(planted) != len(control):
        raise SystemExit("植入与控制的针数不一致（不配对）")
    return {"planted": planted, "control": control,
            "summary_planted": needle_summary(planted),
            "summary_control": needle_summary(control)}


def paired_reading(planted_hits: list[int], control_hits: list[int]) -> dict:
    """「植入 vs 地板」：Newcombe 两比例差 CI + 按针配对 bootstrap CI（`2662` 判据②的两件套）。"""
    n = len(planted_hits)
    if n == 0:
        return {"n": 0, "note": "空桶", "passes_both": False}
    kp, kc = int(sum(planted_hits)), int(sum(control_hits))
    tp = two_proportion(kc, n, kp, n)          # diff = 植入 − 地板
    pb = ns.paired_bootstrap_diff_pp(planted_hits, control_hits)
    lo = float(tp["diff_ci_95"][0])
    return {"n": n, "planted_hits": kp, "control_hits": kc,
            "planted_recall": round(kp / n, 6), "control_recall": round(kc / n, 6),
            "delta_pp": round(100 * (kp - kc) / n, 3),
            "newcombe_diff_ci95": tp["diff_ci_95"],
            "newcombe_diff_ci95_pp": [round(100 * v, 3) for v in tp["diff_ci_95"]],
            "paired_bootstrap_pp": pb["diff_ci_95_pp"],
            "passes_newcombe": bool(lo > 0),
            "passes_paired_bootstrap": bool(pb["passes_lower_gt_0"]),
            "passes_both": bool(lo > 0 and pb["passes_lower_gt_0"]),
            "direction_note": "正 = 植入高于地板",
            "required_n_for_observed_delta": required_n(kp / n, kc / n),
            "required_n_for_5pp_target": required_n(min(1.0, kc / n + 0.05), kc / n)}


def required_n(p1: float, p0: float, alpha_z: float = Z_975, power_z: float = Z_80) -> dict:
    """两比例差的规划式样本量（逐针 n）：报观测 Δ 与 +5pp 目标各一次。"""
    d = p1 - p0
    if d <= 0:
        return {"p1": round(p1, 6), "p0": round(p0, 6), "delta_pp": round(100 * d, 3),
                "n_per_arm": None, "note": "观测效应 ≤ 0 ⇒ 样本量不存在（先要一个正效应）"}
    n = (alpha_z + power_z) ** 2 * (p1 * (1 - p1) + p0 * (1 - p0)) / (d * d)
    return {"p1": round(p1, 6), "p0": round(p0, 6), "delta_pp": round(100 * d, 3),
            "n_per_arm": int(math.ceil(n)),
            "formula": ("n = ceil((z0.975 + z0.80)² · (p1(1−p1) + p0(1−p0)) / (p1−p0)²)，"
                        "z = 1.959964 / 0.841621，双侧 0.05、把握度 0.8（两比例差的常用规划式）")}


def bucket_hits(records: list[dict], bucket: str) -> list[int]:
    return [r["hit"] for r in records if r["bucket"] == bucket]


def readings(arm_records: dict, far_flags: list[bool]) -> dict:
    """把一个臂的逐针记录变成判据读数：总体 / `beyond` / `far_slot`。"""
    planted, control = arm_records["planted"], arm_records["control"]
    out = {"n_needles": len(planted),
           "overall": paired_reading([r["hit"] for r in planted], [r["hit"] for r in control])}
    beyond_p = bucket_hits(planted, "beyond")
    beyond_c = bucket_hits(control, "beyond")
    if beyond_p:
        out["beyond"] = paired_reading(beyond_p, beyond_c)
    else:
        out["beyond"] = {"n": 0, "note": "该臂的掩码不产生 beyond 桶（全保留 ⇒ 所有针都可见）",
                         "no_beyond_bucket_by_construction": True}
    p_far = [r["hit"] for r, f in zip(planted, far_flags) if f]
    c_far = [r["hit"] for r, f in zip(control, far_flags) if f]
    out["far_slot"] = paired_reading(p_far, c_far)
    out["per_bucket_planted"] = arm_records["summary_planted"]["buckets"]
    out["by_distance_band"] = _distance_bands(planted)
    return out


def _distance_bands(records: list[dict], bands=((0, 32), (32, 64), (64, 128), (128, 256))) -> dict:
    """只在 `beyond` 桶里按针到答案行的距离分带（人话：远端召回随距离怎么变）。"""
    out: dict[str, dict] = {}
    for lo, hi in bands:
        hits = [r["hit"] for r in records if r["bucket"] == "beyond" and lo <= r["dist"] < hi]
        if hits:
            out[f"[{lo},{hi})"] = {"n": len(hits), "recall": round(sum(hits) / len(hits), 6)}
    return out
