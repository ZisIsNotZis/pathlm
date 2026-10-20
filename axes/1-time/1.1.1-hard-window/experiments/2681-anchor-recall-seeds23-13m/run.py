#!/usr/bin/env python3
"""2681-anchor-recall-seeds23-13m · 评测入口（纯推理）：锚点召回腿补到 n=4。

契约：同目录 `README.md`（§4 判据线预登记，写下后不改）。仪器（`realtext_probe.py` /
`needle_stats.py` / 窗口序 / 针参数）逐字复制 `2662`，因此**复用臂（seed 0/1）的读数必须
逐位复现 `2662`**（判据①，本实验的仪器自检门）。本入口做：

  ① **复现门**：`2662` 的 6 个复用臂 clean bpc 与锚点臂 anchor/beyond 桶的
     `(n, planted_hits, control_hits)` 必须逐位相同；不等 ⇒ 本实验作废（不得改判据线去凑）；
  ② 自检：掩码非退化 + 参数量 + 因果性 + 掩码反事实（与 2662 同一批探针）；

  ② 自检：掩码非退化（visible_pairs == 期望）+ 参数量 + **因果性**（只改后半段 token
     ⇒ 前半段 logits 逐位不变）+ 锚点臂掩码反事实探针；
  ③ 质量腿（价格）：每臂在其训练可见性下的 clean bpc；`price_bpc = bpc(win16a) − bpc(win16)`
     （同 seed，n=4）；方差带用 `2662` 冻结的 σ_cell=0.000983（**不重算**），另报本 n 的
     种子极差作描述量；
  ④ **锚点召回腿（主判据）**：needle v3 仪器（真实文本短语针），归属桶一律 `anchors=8`
     ⇒ 逐臂报 `anchor` / `beyond` / `window` 桶「植入 vs 配对无针地板」+ Newcombe 两比例差 CI
     + 按针配对 bootstrap CI；**n=4 的判定单位 = 跨 seed**：逐 seed Δ_pp 的算术均值 +
     两级（重采 seed、seed 内重采针）bootstrap CI + t 区间（df=3）；
 ⑤ 毁源敏感度（负对照）：两变体——变体 A `far_all` 被毁 `[0,64)`；变体 B `beyond_anchor`
     被毁 `[8,64)`（保住锚点槽 `[0,8)`）。锚点臂对变体 A **必然**敏感（锚点槽在被毁区内，
     `premise_holds=False`），对变体 B 应逐位不敏感 ⇒「锚点不搬运锚点槽之外的远处内容」；
  ⑥ 代价：每位置等效 KV 字节（fp16）= 12288×k_eff；活性（掩码可见对数 + 反事实探针 + 训练日志）。

用法：
  python3 run.py            # 全电池
  python3 run.py --smoke    # 冒烟：少批
  python3 run.py --config /tmp/xxx.json --out-dir /tmp/yyy   # CPU dry-run 等
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             window_anchor_keep_mask, window_mask_probe)
from tools.evalkit.retention import (needle_recall, needle_summary,  # noqa: E402
                                     phrase_needle_batch)
from tools.evalkit.carry_channel import two_proportion              # noqa: E402
import realtext_probe as rp                                        # noqa: E402
import needle_stats as ns                                          # noqa: E402

# 跨格参考带（C1 裁决：本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用）。
CROSS_CELL_BAND = 0.014
#: **冻结**的本格自测带（= `2662` 的两 seed 极差，`quality_leg.bands.sigma_cell`）。
#: 本实验**不重算它**（判据线、方差带不得事后修改），新 n 的种子极差只作描述量另报。
FROZEN_SIGMA_CELL = 0.000983
#: 期望的掩码可见对数（window_anchor_keep_mask(256, w, a)）。
EXPECTED_VISIBLE = {(16, 0): 3976, (16, 8): 5868}
#: 2634 落盘的 win16 clean bpc（seed 0/1 的纯窗复用臂必须逐位复现）。
REF_2634_WIN16_BPC = {"win16_s0": 1.62811, "win16_s1": 1.629093}
#: **判据①复现门**：`2662` 落盘的读数（本实验的复用臂是同一条 checkpoint + 同一仪器，
#: 必须逐位相同）。键路径见 `2662/results.json`。
REF_2662 = {
    "bpc": {"win16a_s0": 1.622527, "win16a_s1": 1.621799,
            "win16_s0": 1.62811, "win16_s1": 1.629093, "full": 1.502689},
    "anchor_bucket": {"win16a_s0": {"n": 1003, "planted_hits": 483, "control_hits": 457},
                      "win16a_s1": {"n": 1003, "planted_hits": 484, "control_hits": 473}},
    "beyond_bucket": {"win16a_s0": {"n": 7175, "planted_hits": 3347, "control_hits": 3340},
                      "win16a_s1": {"n": 7175, "planted_hits": 3352, "control_hits": 3356}},
}
#: `config.json`（训练后由 `make_config.py` 生成）必须与契约一致的字段。
#: 判据线与口径的家是 `README.md` + 本文件；`config.json` 只是检查点 sha256 的落盘面，
#: 它不参与绑定（见 `prereg_binding.json::contract_files`），所以每项都在这里核对一遍。
CONTRACT_ARMS = ["win16a_s0", "win16a_s1", "win16a_s2", "win16a_s3",
                 "win16_s0", "win16_s1", "win16_s2", "win16_s3", "full"]
CONTRACT_SEEDS = [0, 1, 2, 3]
CONTRACT_EVAL = {"bpc": {"n_batches": 96, "batch_size": 16, "window_seed": 2},
                 "probe": {"n_batches": 96, "batch_size": 16, "window_seed": 2,
                           "corrupt_hi": 64, "eval_lo": 244, "eval_hi": 255,
                           "corrupt_seed": 2634, "rf": 180}}
CONTRACT_NEEDLE = {"n_needles": 8, "phrase_len": 6, "anchors": 8, "anchor_frac": 0.25,
                   "n_batches": 64, "batch_size": 16, "seed": 3, "bucket_anchors": 8}
#: 跨 seed 均值的 t 区间临界值（双侧 α=0.05，df = 种子数 − 1）。
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}


def _check_config_against_contract(cfg: dict) -> None:
    """`config.json` 是训练后生成的产物（含检查点 sha256）⇒ 在**已绑定的本文件**里核对它。

    不一致就拒绝启动：判据线与口径的唯一家是 `README.md` + 本文件，`config.json` 若被事后
    改动（换臂、换批数、换针参数）必须在这里红掉，而不是默默跑。
    """
    got = [a["name"] for a in cfg["arms"]]
    if got != CONTRACT_ARMS:
        raise SystemExit(f"config.json 的臂清单与契约不一致：\n want {CONTRACT_ARMS}\n got  {got}")
    if [int(s) for s in cfg["seeds"]] != CONTRACT_SEEDS:
        raise SystemExit(f"config.json 的 seeds 与契约不一致：{cfg['seeds']} != {CONTRACT_SEEDS}")
    for block, want in CONTRACT_EVAL.items():
        for k, v in want.items():
            if int(cfg["eval"][block][k]) != int(v):
                raise SystemExit(f"config.json eval.{block}.{k} 与契约不一致："
                                 f"{cfg['eval'][block][k]} != {v}")
    for k, v in CONTRACT_NEEDLE.items():
        if int(cfg["needle"][k]) != int(v):
            raise SystemExit(f"config.json needle.{k} 与契约不一致：{cfg['needle'][k]} != {v}")
    for spec in cfg["arms"]:
        want_anchors = 8 if spec["name"].startswith("win16a") else 0
        if int(spec.get("anchors", 0)) != want_anchors:
            raise SystemExit(f"config.json {spec['name']}: anchors={spec.get('anchors')} "
                             f"≠ 契约 {want_anchors}")
        if spec["kind"] == "window" and int(spec["window"]) != 16:
            raise SystemExit(f"config.json {spec['name']}: window={spec['window']} ≠ 契约 16")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _strict(obj):
    """非有限浮点（NaN/Inf）→ None，保证落盘严格 JSON（禁裸 NaN/Infinity）。"""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


def model_kwargs(spec: dict, mcfg: dict) -> dict:
    kw = dict(mcfg)
    for k in ("note",):
        kw.pop(k, None)
    kw["train_window"] = int(spec["window"]) if spec["kind"] != "full" else 0
    kw["train_anchors"] = int(spec.get("anchors", 0)) if spec["kind"] != "full" else 0
    kw["carry_rule"] = None if spec["rule"] in (None, "none") else spec["rule"]
    return kw


def keep_mask_for(spec: dict, T: int, dev: str):
    if spec["kind"] == "full":
        return None
    return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                   device=dev)


def load_arm(spec: dict, mcfg: dict, vocab: int, dev: str):
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{spec['name']}: checkpoint sha256 mismatch:\n "
                         f"want {want}\n got  {got}")
    cfg = DepthARConfig(**model_kwargs(spec, mcfg))
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "kind": spec["kind"], "window": spec["window"],
                   "anchors": spec.get("anchors", 0), "rule": spec["rule"],
                   "seed": spec["seed"], "reused_from": spec.get("reused_from"),
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """**严格因果性探针（必须不变项）**：只改后半段 token，前半段 logits 必须逐位不变。"""
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            a = rp.masked_logits(model, x, keep_mask).float()
            b = rp.masked_logits(model, x2, keep_mask).float()
    finally:
        model.train(was_training)
    d = (a[:, :half] - b[:, :half]).abs().max()
    return {"perturbed_from": half,
            "max_abs_logit_diff_prefix": round(float(d), 9),
            "causal_no_future_leak": bool(float(d) == 0.0),
            "changed_suffix_max_abs_logit_diff": round(
                float((a[:, half:] - b[:, half:]).abs().max()), 6),
            "note": ("只改后半段 token ⇒ 前半段 logits 逐位不变 ⇔ 严格因果（无未来泄漏）；"
                     "后半段必须真的变（否则探针没抓力）")}


def bucket_records(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["bucket"], []).append(r)
    return out


def paired_reading(planted: list[int], control: list[int]) -> dict:
    """「植入 vs 配对无针地板」：Newcombe 两比例差 CI + 按针配对 bootstrap CI。"""
    if not planted:
        return {"n": 0, "note": "空桶"}
    kp, kc, n = sum(planted), sum(control), len(planted)
    tp = two_proportion(kc, n, kp, n)             # diff = 植入 − 地板
    pb = ns.paired_bootstrap_diff_pp(planted, control)
    lo = float(tp["diff_ci_95"][0])
    return {"n": n, "planted_hits": int(kp), "control_hits": int(kc),
            "planted_recall": round(kp / n, 6), "control_recall": round(kc / n, 6),
            "delta_pp": round(100 * (kp - kc) / n, 3),
            "newcombe_diff_ci95": tp["diff_ci_95"],
            "newcombe_diff_ci95_pp": [round(100 * v, 3) for v in tp["diff_ci_95"]],
            "newcombe_ci_overlap": tp["ci_overlap"], "z": tp["z"], "p_value": tp["p_value"],
            "paired_bootstrap_pp": pb["diff_ci_95_pp"],
            "paired_bootstrap_mean_pp": pb["mean_diff_pp"],
            "passes_newcombe": bool(lo > 0),
            "passes_paired_bootstrap": bool(pb["passes_lower_gt_0"]),
            "passes_both": bool(lo > 0 and pb["passes_lower_gt_0"]),
            "direction_note": "正 = 植入高于地板（判据②要的方向）"}


def cross_seed_mean(values: list[float], *, unit_note: str = "") -> dict:
    """跨 seed 均值的 95% 区间（种子 = 重复单位）。

    `n=2` 时的「逐 seed 各自 CI 排除 0」是一个**合取式**门：每个 seed 单靠自己的
    1003 根针要把 1–3pp 的效应从 ~±4.4pp 的两比例噪声里分出来，功效近乎 0（README §4 的
    功效表）。加 seed 只会加合取项 ⇒ 判定单位必须是**种子层**的均值 + 跨 seed 区间。

    t 区间用 `T_CRIT_95[df]`（双侧 α=0.05）；`n < 2` 或 `df` 不在表里时区间置 None 并注明。
    """
    n = len(values)
    if n == 0:
        return {"n_seeds": 0, "note": "无值"}
    mean = sum(values) / n
    out = {"n_seeds": n, "mean": round(mean, 6), "values": [round(v, 6) for v in values],
           "n_positive": int(sum(1 for v in values if v > 0)),
           "n_negative": int(sum(1 for v in values if v < 0)),
           "same_sign": bool(all(v > 0 for v in values) or all(v < 0 for v in values)),
           "unit_note": unit_note}
    if n < 2:
        out.update({"sd": None, "t_interval_95": None,
                    "note": "n<2，无跨 seed 区间"})
        return out
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    crit = T_CRIT_95.get(n - 1)
    out["sd"] = round(sd, 6)
    out["t_crit"] = crit
    out["t_interval_95"] = (None if crit is None else
                            [round(mean - crit * sd / math.sqrt(n), 6),
                             round(mean + crit * sd / math.sqrt(n), 6)])
    return out


def _pct(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数（与 `needle_stats._percentile` 同口径）。"""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def two_level_bootstrap(per_seed: list[list[int]], *, n_boot: int = 2000,
                        seed: int = 2681, alpha: float = 0.05) -> dict:
    """两级 bootstrap（重采 **seed**、seed 内重采 **针**）：跨 seed 的 95% 区间。

    `per_seed[s]` = 第 s 个 seed 的逐针配对差 `hit_planted − hit_control` ∈ {−1,0,1}（锚点桶）。
    重采 seed 那一步才是「换成别的种子还会不会为正」的那部分不确定性；只在针上重采的版本
    （`fixed_seed_pooled_bootstrap`）把 seed 当成固定效应，**不能**用来回答本判据。
    """
    k = len(per_seed)
    if k == 0 or any(len(d) == 0 for d in per_seed):
        return {"n_seeds": k, "note": "空桶"}
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        num = den = 0
        for _ in range(k):
            d = per_seed[rng.randrange(k)]
            sample = rng.choices(d, k=len(d))
            num += sum(sample)
            den += len(d)
        draws.append(100.0 * num / den)
    draws.sort()
    lo = _pct(draws, alpha / 2)
    hi = _pct(draws, 1 - alpha / 2)
    return {"n_seeds": k, "n_boot": n_boot, "n_needles_per_seed": [len(d) for d in per_seed],
            "diff_ci_95_pp": [round(lo, 3), round(hi, 3)],
            "mean_pp": round(sum(draws) / len(draws), 3),
            "passes_lower_gt_0": bool(lo > 0),
            "note": "重采 seed + seed 内重采针（跨 seed 区间；回答「换成别的种子还行不行」）"}


def fixed_seed_pooled_bootstrap(per_seed: list[list[int]], *, n_boot: int = 2000,
                                seed: int = 2681, alpha: float = 0.05) -> dict:
    """只重采针、**不重采 seed** 的版本（把 seed 当固定效应）⇒ 只能作旁证，不当主判。"""
    k = len(per_seed)
    if k == 0 or any(len(d) == 0 for d in per_seed):
        return {"n_seeds": k, "note": "空桶"}
    rng = random.Random(seed + 1)
    draws: list[float] = []
    for _ in range(n_boot):
        num = den = 0
        for d in per_seed:
            sample = rng.choices(d, k=len(d))
            num += sum(sample)
            den += len(d)
        draws.append(100.0 * num / den)
    draws.sort()
    lo = _pct(draws, alpha / 2)
    hi = _pct(draws, 1 - alpha / 2)
    return {"n_seeds": k, "n_boot": n_boot,
            "diff_ci_95_pp": [round(lo, 3), round(hi, 3)],
            "passes_lower_gt_0": bool(lo > 0),
            "note": ("只重采针、不重采 seed ⇒ 把「这四个模型」当固定效应；"
                      "仅供旁证，不得单独用来判「有区分力」")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke", "eval") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    _check_config_against_contract(cfg)
    if args.smoke:
        for k in ("bpc", "probe"):
            cfg["eval"][k]["n_batches"] = 2
            cfg["eval"][k]["batch_size"] = 4
        cfg["needle"]["n_batches"] = 2
        cfg["needle"]["batch_size"] = 4
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    eb, ep2 = cfg["eval"]["bpc"], cfg["eval"]["probe"]
    vcfg = cfg["needle"]
    window, anchors = int(cfg["model"]["train_window"]), 8
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]
    unit = cfg["model"]["n_layers"] * 2 * cfg["model"]["d_model"] * 2   # fp16 每位置 KV 字节
    log: list[dict] = []
    t0 = time.time()
    on_cuda = str(dev).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段做评测（路径锚到仓库根）")
    _, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                          os.path.join(ROOT, "data/enwik8_full.npz"))
    assert vocab == cfg["data"]["vocab_size"], f"vocab {vocab} != {cfg['data']['vocab_size']}"

    note("加载各臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:12]}… kind={spec['kind']} "
             f"w={spec['window']} anchors={spec['anchors']} seed={spec['seed']} "
             f"params={ck[spec['name']]['n_params']}")

    # ---------------- ① 自检：掩码非退化 + 参数量 + 因果性 + 掩码反事实 ----------------
    note("自检①：掩码非退化 / 参数量 / 因果性 / 锚点掩码反事实")
    selfchecks = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0), "n_params": ck[name]["n_params"]}
        if spec["kind"] != "full":
            km = keep_mask_for(spec, T, dev)
            s = int(km.sum().item())
            exp = EXPECTED_VISIBLE[(int(spec["window"]), int(spec.get("anchors", 0)))]
            row["mask"] = {"visible_pairs": s, "expected": exp, "ok": s == exp,
                           "causal_pairs": T * (T + 1) // 2, "nondegenerate": s < T * (T + 1) // 2}
            assert s == exp, f"{name}: visible pairs {s} != {exp}"
            assert s < T * (T + 1) // 2, f"{name}: 掩码是恒等算子（退化）"
        selfchecks[name] = row
    if "win16a_s0" in ck and "win16_s0" in ck:
        assert ck["win16a_s0"]["n_params"] == ck["win16_s0"]["n_params"], \
            "锚点臂与纯窗臂参数量应逐位相同（掩码不引入参数）"

    g_probe = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, vocab, (2, T), generator=g_probe).to(dev)
    for spec in cfg["arms"]:
        name = spec["name"]
        selfchecks[name]["causality"] = causality_probe(arms[name], x_probe,
                                                        keep_mask_for(spec, T, dev))
        if spec["kind"] != "full":
            selfchecks[name]["mask_counterfactual"] = window_mask_probe(arms[name], x_probe)
        note(f"  {name}: causal_prefix_diff="
             f"{selfchecks[name]['causality']['max_abs_logit_diff_prefix']} "
             f"mask_counterfactual="
             f"{selfchecks[name].get('mask_counterfactual', {}).get('max_abs_logit_diff_masked_vs_open')}")

    # ---------------- ② 质量腿（价格）----------------
    note(f"质量腿②：clean bpc（每臂自己的训练可见性，{eb['n_batches']}×{eb['batch_size']}，"
         f"window_seed={eb['window_seed']}）")
    quality = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        bpc = rp.win16_bpc(arms[name], eval_arr, km, n_batches=eb["n_batches"],
                           batch_size=eb["batch_size"], window_seed=eb["window_seed"],
                           device=dev)
        quality[name] = round(bpc, 6)
        note(f"  {name}: bpc={bpc:.6f}")

    # 方差带：**冻结**的 `2662` 本格带（不重算）+ 本 n 的种子极差（描述量，不当判据）
    sigma_cell = FROZEN_SIGMA_CELL
    seed_range = {}
    for kind, pref, anc in (("win16a", "win16a", 8), ("win16", "win16", 0)):
        vals = [quality[f"{pref}_s{s}"] for s in cfg["seeds"]
                if f"{pref}_s{s}" in quality]
        if len(vals) >= 2:
            seed_range[kind] = {"n_seeds": len(vals), "max_minus_min": round(max(vals) - min(vals), 6)}
    price = {}
    for s in cfg["seeds"]:
        a, w = f"win16a_s{s}", f"win16_s{s}"
        if a not in quality or w not in quality:
            continue
        d = round(quality[a] - quality[w], 6)
        price[str(s)] = {"bpc_win16a": quality[a], "bpc_win16": quality[w],
                         "price_bpc": d,
                         "sortable_vs_sigma_cell": bool(abs(d) > sigma_cell),
                         "sortable_vs_0.014": bool(abs(d) > CROSS_CELL_BAND),
                         "sign": "锚点是质量税（更差）" if d > 0 else
                                 ("锚点是质量收益（更好）" if d < 0 else "相等")}
    price_cross = cross_seed_mean([price[k]["price_bpc"] for k in price],
                                  unit_note=("每 seed 一个 price_bpc（配对：同 seed 的锚点臂 − 纯窗臂）"))
    reuse_equiv = {"note": "复用臂（eval-only）clean bpc 必须与落盘值逐位相同"}
    for nm, ref in REF_2634_WIN16_BPC.items():
        if nm not in quality:
            continue
        got = quality[nm]
        reuse_equiv[nm] = {"bpc": got, "bpc_ref": ref, "equal": got == ref,
                           "reused_from": ck[nm]["reused_from"],
                           "ref_source": "2634-carry-realtext-13m/results.json"}
    # ------------------ ① 复现门：与 2662 逐位比对（bpc 部分） ------------------
    bpc_repro = {}
    for nm, ref in REF_2662["bpc"].items():
        got = quality.get(nm)
        bpc_repro[nm] = {"bpc": got, "bpc_2662": ref,
                         "equal": bool(got is not None and got == ref)}

    # ---------------- ③ 锚点召回腿（主判据，needle v3）----------------
    ncfg = dict(n_needles=vcfg["n_needles"], phrase_len=vcfg["phrase_len"],
                anchors=vcfg["anchors"], anchor_frac=vcfg["anchor_frac"])
    bucket_anchors = int(vcfg["bucket_anchors"])
    L = vcfg["phrase_len"]
    note(f"锚点召回腿③：needle v3（{vcfg['n_batches']}×{vcfg['batch_size']}，"
         f"{vcfg['n_needles']} 针/行，seed={vcfg['seed']}，归属桶 anchors={bucket_anchors}）")
    gen = torch.Generator().manual_seed(vcfg["seed"])
    arm_names = [s["name"] for s in cfg["arms"]]
    records = {f"{n}|{kind}": [] for n in arm_names for kind in ("planted", "control")}
    n_needles_total = 0
    for bi in range(vcfg["n_batches"]):
        state = gen.get_state()
        x, meta = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                      mask_tok, gen, **ncfg, plant=True)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                        mask_tok, gen, **ncfg, plant=False)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        n_needles_total += sum(len(m) for m in meta)
        for name in arm_names:
            spec = next(s for s in cfg["arms"] if s["name"] == name)
            km = keep_mask_for(spec, T, dev)
            for kind, xx in (("planted", x), ("control", x0)):
                rec = needle_recall(arms[name], xx, meta, km,
                                    anchors=bucket_anchors, phrase_len=L, device=dev)
                records[f"{name}|{kind}"].extend(rec["per_needle"])
        log.append({"t": round(time.time() - t0, 2), "msg": f"needle batch {bi}"})

    planted_b = {n: bucket_records(records[f"{n}|planted"]) for n in arm_names}
    control_b = {n: bucket_records(records[f"{n}|control"]) for n in arm_names}
    needle = {"conditions": {}, "n_needles_total": n_needles_total,
              "seed": vcfg["seed"], "n_batches": vcfg["n_batches"],
              "bucket_anchors": bucket_anchors,
              "bucket_rule": ("归属桶一律 anchors=%d：anchor = 针植入 p<%d（锚点槽恒可见区）；"
                              "window = 非锚针且短语全可见；beyond = 非锚针且有位置不可见。"
                              "五臂同归属规则 ⇒ 同索引可比" % (bucket_anchors, bucket_anchors))}
    bands = [(0, 32), (32, 64), (64, 128), (128, 256)]

    def band_of(d):
        for lo, hi in bands:
            if lo <= d < hi:
                return f"[{lo},{hi})"
        return f"[{bands[-1][1]},)"

    for key, recs in records.items():
        s = needle_summary(recs)
        by_band: dict[str, list[int]] = {}
        for r in recs:
            if r["bucket"] == "beyond":
                by_band.setdefault(band_of(r["dist"]), []).append(r["hit"])
        needle["conditions"][key] = {
            "recall": s["recall"], "n": s["n"], "buckets": s["buckets"],
            "beyond_by_distance_band": {b: {"recall": round(sum(v) / len(v), 6),
                                            "n": len(v)} for b, v in sorted(by_band.items())}}

    # 逐臂 × 逐桶：植入 vs 配对地板（Newcombe + 配对 bootstrap）
    per_arm_bucket = {}
    for n in arm_names:
        row = {}
        for b in ("anchor", "window", "beyond"):
            p = [r["hit"] for r in planted_b[n].get(b, [])]
            c = [r["hit"] for r in control_b[n].get(b, [])]
            assert len(p) == len(c), f"{n}/{b}: 植入与地板针数不一致"
            row[b] = paired_reading(p, c)
        per_arm_bucket[n] = row
    needle["per_arm_bucket_planted_vs_floor"] = per_arm_bucket

    # 锚点桶内：按「短语是否全可见」拆分（诊断，锚点臂内部说明）
    anchor_split = {}
    for n in arm_names:
        vis = [r["hit"] for r in planted_b[n].get("anchor", []) if r.get("visible")]
        nvis = [r["hit"] for r in planted_b[n].get("anchor", []) if not r.get("visible")]
        anchor_split[n] = {
            "fully_visible": {"n": len(vis),
                              "recall": (round(sum(vis) / len(vis), 6) if vis else None)},
            "partially_visible": {"n": len(nvis),
                                  "recall": (round(sum(nvis) / len(nvis), 6) if nvis else None)},
            "note": ("锚点槽 [0,8) 只有 8 格，而短语占 L+1=7 格 ⇒ 只有 p∈{0,1} 的针"
                     "「短语全可见」，p∈{2..7} 只部分可见；此拆分只作诊断")}
    needle["anchor_bucket_visibility_split"] = anchor_split

    # 同索引硬子集（跨臂可比）：对 anchor / beyond 两桶各算一次
    needle["hard_subset_same_index"] = {}
    for b in ("anchor", "beyond"):
        ctrl_by_arm = {n: control_b[n].get(b, []) for n in arm_names}
        if not all(ctrl_by_arm.values()):
            needle["hard_subset_same_index"][b] = {"note": "该桶在至少一臂为空，跳过同索引"}
            continue
        idx_sets = ns.hard_index_sets(ctrl_by_arm)
        planted_by_arm = {n: planted_b[n].get(b, []) for n in arm_names}
        same = {}
        for n in arm_names:
            same[n] = {
                "own_recall": round(sum(r["hit"] for r in planted_by_arm[n]) / len(planted_by_arm[n]), 6),
                "intersection": ns.recall_on_indices(planted_by_arm[n], idx_sets["intersection"]),
                "union": ns.recall_on_indices(planted_by_arm[n], idx_sets["union"])}
        cross = {}
        for s in cfg["seeds"]:
            a, w = f"win16a_s{s}", f"win16_s{s}"
            cross[f"{a}_minus_{w}"] = {}
            for set_name in ("intersection", "union"):
                idx = idx_sets[set_name]
                pa = ns.recall_on_indices(planted_by_arm[a], idx)
                pw = ns.recall_on_indices(planted_by_arm[w], idx)
                cross[f"{a}_minus_{w}"][set_name] = ns.beyond_above_floor(
                    pa["hits"], pa["n"], pw["hits"], pw["n"])
        needle["hard_subset_same_index"][b] = {
            "n_intersection": idx_sets["n_intersection"], "n_union": idx_sets["n_union"],
            "n_indexed": idx_sets["n_indexed"], "per_arm": same,
            "cross_arm_anchor_vs_window": cross}

    # 判据①④ 汇总：复现门（针）+ 逐 seed 读数 + n=4 跨 seed 主判
    def _passes(arm, b, key="passes_both"):
        return bool(per_arm_bucket[arm][b].get(key, False)) if per_arm_bucket[arm][b].get("n") else False

    anchor_arms = [f"win16a_s{s}" for s in cfg["seeds"]]
    floor_arms = [f"win16_s{s}" for s in cfg["seeds"]]
    # 判据①：复用臂（seed 0/1）的 anchor / beyond 桶逐位复现 2662
    needle_repro = {}
    for bucket, refs in (("anchor", REF_2662["anchor_bucket"]),
                         ("beyond", REF_2662["beyond_bucket"])):
        for arm, ref in refs.items():
            got = per_arm_bucket[arm][bucket]
            row = {k: {"got": got.get(k), "want": ref[k], "equal": got.get(k) == ref[k]}
                   for k in ("n", "planted_hits", "control_hits")}
            needle_repro[f"{arm}|{bucket}"] = {"checks": row,
                                               "equal": all(v["equal"] for v in row.values())}

    # per-seed 读数：口径不变（Newcombe + 按针配对 bootstrap），**但不再是 n=4 的门**
    per_seed_recall = {}
    for s in cfg["seeds"]:
        arm = f"win16a_s{s}"
        r = per_arm_bucket[arm]["anchor"]
        per_seed_recall[str(s)] = r
        note(f"  per-seed anchor 桶 {arm}: n={r.get('n')} Δ={r.get('delta_pp')}pp "
             f"Newcombe{r.get('newcombe_diff_ci95_pp')} 配对{r.get('paired_bootstrap_pp')} "
             f"passes_both={r.get('passes_both')}")

    # 主判（③ n=4）：跨 seed 均值 + 两个跨 seed 区间 + 方向一致性
    deltas_per_seed = []
    for s in cfg["seeds"]:
        p = [r["hit"] for r in planted_b[f"win16a_s{s}"].get("anchor", [])]
        c = [r["hit"] for r in control_b[f"win16a_s{s}"].get("anchor", [])]
        deltas_per_seed.append([a - b for a, b in zip(p, c)])
    ds = [per_seed_recall[str(s)]["delta_pp"] for s in cfg["seeds"]]
    csm = cross_seed_mean(ds, unit_note="每 seed 一个 anchor 桶 Δ_pp（植入 − 配对地板）")
    two_level = two_level_bootstrap(deltas_per_seed)
    fixed_pooled = fixed_seed_pooled_bootstrap(deltas_per_seed)
    t_lo = None if csm.get("t_interval_95") is None else csm["t_interval_95"][0]
    decidable = bool(csm["mean"] > 0 and csm["n_positive"] >= 3
                     and t_lo is not None and t_lo > 0
                     and two_level.get("passes_lower_gt_0") is True)
    crit = {
        "c1_reproducer_2662": {
            "bpc": bpc_repro,
            "bpc_all_equal": bool(all(v["equal"] for v in bpc_repro.values())),
            "needle_buckets": needle_repro,
            "needle_all_equal": bool(all(v["equal"] for v in needle_repro.values())),
            "passes": bool(all(v["equal"] for v in bpc_repro.values())
                           and all(v["equal"] for v in needle_repro.values())),
            "note": "复用臂 = 2662 的同一条 checkpoint + 同一仪器 ⇒ 逐位相同；不等即本实验作废",
        },
        "c2_per_seed_recall_unchanged_instrument": {
            "per_seed": per_seed_recall,
            "note": (
                "口径不变：Newcombe 两比例差 + 按针配对 bootstrap（每 seed n=1003 根锚点桶针）。"
                "n=2 时预登记的门是「两 seed 都过两套 CI」；单 seed 要把 1–3pp 从 ~±4.4pp 的"
                "两比例噪声里分出需要 2.8k–16k 根针（README §4a 功效表）⇒ 该合取式在本仪器下"
                "功效近乎 0，加 seed 只会加合取项，所以**它不再是 n=4 的门**，只如实报（不挑 seed）"),
        },
        "c3_cross_seed_n4": {
            "mean_delta_pp": csm,
            "two_level_bootstrap": two_level,
            "fixed_seed_pooled": fixed_pooled,
            "direction_rule": "4 seed 中 ≥3 个 Δ_pp > 0",
            "interval_rule": "跨 seed 均值 > 0 且 **两个**跨 seed 区间下界均 > 0（t 区间 df=3 与两级 bootstrap）",
            "decidable": decidable,
            "note": ("主判。两个区间都要排除 0 是保守取法（不得挑对自己有利的那个区间）；"
                     "`fixed_seed_pooled` 把 seed 当固定效应，只作旁证"),
        },
        "c4_beyond_bucket_stays_at_floor": {
            n: {"delta_pp": per_arm_bucket[n]["beyond"].get("delta_pp"),
                "newcombe_diff_ci95_pp": per_arm_bucket[n]["beyond"].get("newcombe_diff_ci95_pp"),
                "ci_contains_0": (None if not per_arm_bucket[n]["beyond"].get("newcombe_diff_ci95_pp")
                                  else bool(per_arm_bucket[n]["beyond"]["newcombe_diff_ci95_pp"][0] <= 0
                                            <= per_arm_bucket[n]["beyond"]["newcombe_diff_ci95_pp"][1]))}
            for n in anchor_arms + floor_arms},
    }
    note("判据③ n=4 主判："
         f"Δ_pp per-seed={[per_seed_recall[str(s)]['delta_pp'] for s in cfg['seeds']]} "
         f"mean={csm['mean']}pp n_positive={csm['n_positive']}/{csm['n_seeds']} "
         f"t_interval={csm['t_interval_95']} bootstrap={two_level.get('diff_ci_95_pp')} "
         f"⇒ 可判={decidable}")
    note(f"判据① 复现门：bpc 全等={crit['c1_reproducer_2662']['bpc_all_equal']} "
         f"针桶全等={crit['c1_reproducer_2662']['needle_all_equal']} "
         f"⇒ passes={crit['c1_reproducer_2662']['passes']}")
    for n in floor_arms:
        r = per_arm_bucket[n]["anchor"]
        note(f"  负对照 {n}: anchor 桶 {r.get('delta_pp')}pp CI{r.get('newcombe_diff_ci95_pp')}")

    # ---------------- ④ 毁源敏感度（两变体）----------------
    pkw = rp.probe_kwargs(cfg)
    assert pkw["rf"] == cfg["model"]["n_layers"] * (window - 1) == 180
    note(f"毁源敏感度④：变体 A far_all 被毁 [0,64)；变体 B beyond_anchor 被毁 [8,64)；"
         f"被评行 [{pkw['eval_lo']},{pkw['eval_hi']})，rf={pkw['rf']}")
    probe = {}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        arms_kw = {k: pkw[k] for k in ("n_batches", "batch_size", "window_seed",
                                      "corrupt_hi", "eval_lo", "eval_hi", "corrupt_seed")}
        arms_kw["rf"] = pkw["rf"]
        row = {}
        # 变体 A：被毁 [0,64)。窗口臂要求不重叠；锚点臂必然重叠（锚点槽 [0,8) 恒可见）
        is_anchor = int(spec.get("anchors", 0)) > 0
        row["far_all"] = rp.far_source_probe(
            arms[name], eval_arr, km, **arms_kw, corrupt_lo=0,
            require_no_overlap=not is_anchor, device=dev)
        # 变体 B：被毁 [8,64)（保住锚点槽完整）
        row["beyond_anchor"] = rp.far_source_probe(
            arms[name], eval_arr, km, **arms_kw, corrupt_lo=int(spec.get("anchors", 0)),
            require_no_overlap=True, device=dev)
        probe[name] = row
        note(f"  {name}: far_all Δbpc={row['far_all']['delta_bpc_far']:+.6f} "
             f"max|Δlogit|={row['far_all']['max_abs_logit_diff_on_eval_rows']:.3e} "
             f"premise={row['far_all']['premise_holds']} | "
             f"beyond_anchor Δbpc={row['beyond_anchor']['delta_bpc_far']:+.6f} "
             f"max|Δlogit|={row['beyond_anchor']['max_abs_logit_diff_on_eval_rows']:.3e} "
             f"premise={row['beyond_anchor']['premise_holds']}")

    crit["c5_far_source"] = {
        n: {"far_all": {"delta_bpc_far": probe[n]["far_all"]["delta_bpc_far"],
                        "max_abs_logit_diff_on_eval_rows":
                            probe[n]["far_all"]["max_abs_logit_diff_on_eval_rows"],
                        "premise_holds": probe[n]["far_all"]["premise_holds"]},
            "beyond_anchor": {"delta_bpc_far": probe[n]["beyond_anchor"]["delta_bpc_far"],
                              "max_abs_logit_diff_on_eval_rows":
                                  probe[n]["beyond_anchor"]["max_abs_logit_diff_on_eval_rows"],
                              "premise_holds": probe[n]["beyond_anchor"]["premise_holds"]}}
        for n in probe}
    crit["c5_beyond_anchor_bitwise_insensitive"] = bool(all(
        probe[n]["beyond_anchor"]["max_abs_logit_diff_on_eval_rows"] == 0.0 for n in probe))
    crit["c5_far_all_window_bitwise_insensitive"] = bool(all(
        probe[n]["far_all"]["max_abs_logit_diff_on_eval_rows"] == 0.0
        for n in probe if int(next(s for s in cfg["arms"] if s["name"] == n)["anchors"]) == 0))
    crit["c6_causality_no_future_leak"] = bool(all(
        selfchecks[n]["causality"]["causal_no_future_leak"] for n in arm_names))
    crit["c7_mask_nondegenerate_and_reuse_equal"] = bool(
        all(v.get("mask", {}).get("nondegenerate", True) for v in selfchecks.values())
        and all(reuse_equiv[nm]["equal"] for nm in REF_2634_WIN16_BPC if nm in reuse_equiv))

    # ---------------- ⑤ 代价 + 活性 ----------------
    rows = []
    for spec in cfg["arms"]:
        name = spec["name"]
        k_eff = T if spec["kind"] == "full" else int(spec["window"]) + int(spec.get("anchors", 0))
        kv = unit * k_eff
        rows.append({"name": name, "kind": spec["kind"], "seed": spec["seed"],
                     "window": spec["window"], "anchors": spec.get("anchors", 0),
                     "k_eff": k_eff, "kv_bytes_per_position": kv,
                     "bpc": quality[name], "n_params": ck[name]["n_params"]})
    liveness = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0),
               "mask": selfchecks[name].get("mask"),
               "mask_counterfactual": selfchecks[name].get("mask_counterfactual"),
               "causality": selfchecks[name]["causality"]}
        tdir = os.path.dirname(resolve(spec["path"]))
        tlog = os.path.join(tdir, "train_log.jsonl")
        tres = os.path.join(tdir, "results.json")
        if os.path.exists(tlog):
            lines = [json.loads(x) for x in open(tlog) if x.strip()]
            row["train_window_mask"] = lines[-1].get("window_mask")
            row["train_last_loss"] = lines[-1].get("loss")
        if os.path.exists(tres):
            tr = json.load(open(tres))
            row["train_wall_minutes"] = tr.get("wall_minutes")
            row["train_peak_gpu_mem_mb"] = tr.get("peak_gpu_mem_mb")
            row["train_window_liveness"] = tr.get("window_liveness")
        liveness[name] = row
        note(f"  {name}: visible_pairs={row['mask']['visible_pairs'] if row['mask'] else 'n/a'} "
             f"counterfactual={row['mask_counterfactual']['max_abs_logit_diff_masked_vs_open'] if row['mask_counterfactual'] else 'n/a'} "
             f"train_window_mask_alive="
             f"{(row.get('train_window_mask') or {}).get('alive')}")

    results = {
        "delta_directions": {
            "delta_pp": "arm_minus_comparator",        # 植入 − 配对地板（正 = 埋针后召回更高）
            "mean_delta_pp": "arm_minus_comparator",   # 跨 seed 均值（同 delta_pp 的方向）
            "delta_bpc_far": "arm_minus_comparator",   # 毁源 − 干净（正 = 对远处源敏感）
            "price_bpc": "arm_minus_comparator",       # 锚点臂 − 纯窗臂（正 = 质量更差）
        },
        "config": cfg, "checkpoints": ck, "selfchecks": selfchecks,
        "quality_leg": {"bpc": quality, "price": price, "price_cross_seed": price_cross,
                        "bands": {"sigma_cell_frozen": sigma_cell,
                                  "sigma_cell_frozen_source": (
                                      "2662/results.json::quality_leg.bands.sigma_cell（2 seed "
                                      "极差；本实验**不重算**，判据线不得事后修改）"),
                                  "seed_range_observed_n4": seed_range,
                                  "cross_cell_band": CROSS_CELL_BAND,
                                  "note": ("本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用"
                                           "（C1）。n=4 的 `seed_range_observed_n4` 只是描述量："
                                           "样本更多必然更宽，不得回填成新带")},
                        "reuse_equivalence": reuse_equiv,
                        "gap_win16_full": round(quality["win16_s0"] - quality["full"], 6),
                        "gap_win16a_full": round(quality["win16a_s0"] - quality["full"], 6),
                        "gap_win16a_full_by_seed": {
                            str(s): round(quality[f"win16a_s{s}"] - quality["full"], 6)
                            for s in cfg["seeds"]}},
        "needle_leg": needle,
        "far_source_probe": probe,
        "criteria": crit,
        "rows": rows,
        "liveness": liveness,
        "code_hashes": {
            rel: sha256_file(os.path.join(ROOT, rel)) for rel in
            ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/retention.py",
             "tools/evalkit/carry_channel.py", "pathlm/data.py")
            if os.path.exists(os.path.join(ROOT, rel))},
        "cost": {
            "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
            "eval_peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                     if on_cuda else None),
            "n_eval_windows_bpc": eb["n_batches"] * eb["batch_size"],
            "n_probe_windows": ep2["n_batches"] * ep2["batch_size"],
            "n_needle_batches": vcfg["n_batches"],
            "note": "墙钟只作过程指标，不进判据",
        },
    }
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "realtext_probe.py", "needle_stats.py", "make_config.py", "config.json",
         "train_arms.sh") if os.path.exists(os.path.join(HERE, rel))})

    results = _strict(results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(args.out_dir, "needle_raw.json"), "w") as f:
        json.dump(_strict(records), f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "run_log.txt"), "w") as f:
        for r in log:
            f.write(f"[{r['t']:7.1f}s] {r['msg']}\n")
    note(f"落盘：{args.out_dir}/{{results.json,needle_raw.json,run_log.jsonl,run_log.txt}}")


if __name__ == "__main__":
    main()
