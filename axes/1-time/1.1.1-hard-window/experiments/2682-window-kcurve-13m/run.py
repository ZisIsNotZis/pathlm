#!/usr/bin/env python3
"""2682-window-kcurve-13m · 评测入口（纯推理）：窗口臂 k 全曲线的两列口径。

契约：同目录 `README.md`（§5 判据线预登记，写下后不改）。本入口做：

  ① **复现门**：1.1.1 目录内已有的 9 条 checkpoint（`win16_s0..s3`、`win16a_s0..s3`、`full`）
     的 clean bpc 必须与 `2662` / `2681` 落盘值**逐位相同**；不等 ⇒ 本实验作废（读数不入账）；
  ② **训练期累计预算列（主判）**：每 k 的窗口税 `tax = bpc(win-k) − bpc(full)`，逐 seed 报，
     跨 seed 均值 + t 区间，按 `README.md` §5a 三分类（above_band / below_band / undecidable），
     `undecidable` 必须附所需样本量；
  ③ **评测期累计预算列**：`recent_keep_mask(256,D)`（只留最近 D）——0 训练 token、改掩码；
  ④ **单距离列**：`distance_keep_mask(256,d)`（删距离恰为 d 的键）——**只描述形状**，不得求和外推；
  ⑤ **两列不可互换的演示**：`cum(D=32)` 与 `Σ_{d≥32} single(d)` 的比值（预登记预测 > 1）；
  ⑥ **锚点杠杆在第二个 k**（可选臂 `win64a_s0/s1`）：`price = bpc(win64a) − bpc(win64)`；
  ⑦ 自检：因果性 + 掩码可见对数 + 参数量 + `recent_keep_mask(T)` 与释放版逐位相同。

用法：
  python3 run.py            # 全电池
  python3 run.py --smoke    # 冒烟：少批、少档位（只用于测步时 / 查错）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                            # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,            # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import (distance_keep_mask, mask_stats,  # noqa: E402
                                     recent_keep_mask)
import realtext_probe as rp                                          # noqa: E402

#: 跨格参考带（C1 裁决：本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用）。
CROSS_CELL_BAND = 0.014
#: **冻结**的本格自测带（= `2662` 的两 seed 极差）。本实验**不重算它**（判据线不得事后修改）。
FROZEN_SIGMA_CELL = 0.000983
#: 跨 seed 均值的 t 区间临界值（双侧 α=0.05，df = 种子数 − 1）。
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
#: 判据② 的功效近似式常数（双侧 1.96 + 单侧 0.84）。
POWER_Z = 2.9

#: 契约臂清单（`config.json` 必须逐项对上，顺序也一致）。可选臂只服务判据⑥。
CONTRACT_ARMS_REQUIRED = ["full",
                          "win16_s0", "win16_s1", "win16_s2", "win16_s3",
                          "win16a_s0", "win16a_s1", "win16a_s2", "win16a_s3",
                          "win32_s0", "win32_s1", "win64_s0", "win64_s1",
                          "win128_s0", "win128_s1", "win192_s0", "win192_s1"]
CONTRACT_ARMS_OPTIONAL = ["win64a_s0", "win64a_s1"]
CONTRACT_ARMS = CONTRACT_ARMS_REQUIRED + CONTRACT_ARMS_OPTIONAL
#: 判据② 的 k 档位（k=16 的两列复本已在 1.1.1 目录里，见 README §2）。顺序 = 报告顺序。
CONTRACT_KS = [16, 32, 64, 128, 192]
CONTRACT_SEEDS = [0, 1, 2, 3]
CONTRACT_EVAL = {"bpc": {"n_batches": 96, "batch_size": 16, "window_seed": 2}}
#: 判据③ 的 D 档位（累计预算列）与判据④ 的 d 档位（单距离列）。
CONTRACT_RECENT_DS = [16, 32, 64, 128, 192, 240]
CONTRACT_DISTANCE_DS = [1, 2, 4, 8, 16, 32, 64, 128, 255]
#: 两条曲线覆盖的臂（控成本：full + 三个代表性窗口臂，见 README §7 的砍法）。
CONTRACT_CURVE_ARMS = ["full", "win16_s0", "win64_s0", "win192_s0"]
#: 判据⑤ 用的对比点。
CONTRACT_DEMO_D = 32
CONTRACT_DEMO_SINGLE_DS = [32, 64, 128, 255]
#: 判据⑤ **只在释放版是全因果的臂上判**（只有那里 D 是真正的预算；窗口臂的 D=32 比它的训练
#: 窗口更**宽**，不是收紧 ⇒ 比率只作描述量）。
CONTRACT_DEMO_JUDGED_ARMS = ["full"]
#: 判据① 复现门：1.1.1 目录内已有 checkpoint 的落盘 clean bpc（`2662` / `2681` 的
#: `results.json::quality_leg.bpc`，本目录用同一仪器重评后必须逐位相同）。
REF_BPC = {"full": 1.502689,
           "win16_s0": 1.62811, "win16_s1": 1.629093,
           "win16_s2": 1.624611, "win16_s3": 1.625184,
           "win16a_s0": 1.622527, "win16a_s1": 1.621799,
           "win16a_s2": 1.618953, "win16a_s3": 1.615477}
#: 判据① 里被复算的那几条（= 1.1.1 目录内已有的全部 checkpoint）。
CONTRACT_REPRO_ARMS = ["full", "win16_s0", "win16_s1", "win16_s2", "win16_s3",
                       "win16a_s0", "win16a_s1", "win16a_s2", "win16a_s3"]
#: 期望的掩码可见对数（`window_anchor_keep_mask(256, w, a)`，判据⑦）。
EXPECTED_VISIBLE = {(16, 0): 3976, (16, 8): 5868}
N_PARAMS = 13536768


def expected_visible_pairs(T: int, w: int, a: int) -> int:
    """`window_anchor_keep_mask(T, w, a)` 的可见 (query, key) 对数（闭式）。

    每行 `i` 的可见数 = 最近窗口 `min(i+1, w)` + 锚点额外放行的 `min(a, max(0, i-w+1))`
    （锚点 `[0, a)` 里那些**不在**最近窗口内的 key）。与 `EXPECTED_VISIBLE` 的两个硬编码
    点互为独立校验。
    """
    return sum(min(i + 1, w) + min(a, max(0, i - w + 1)) for i in range(T))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _check_config_against_contract(cfg: dict) -> None:
    """`config.json` 是训练后生成的产物（含检查点 sha256）⇒ 在**已绑定的本文件**里核对它。

    不一致就拒绝启动：判据线与口径的唯一家是 `README.md` + 本文件，`config.json` 若被事后
    改动（换臂、换批数）必须在这里红掉，而不是默默跑。
    """
    got = [a["name"] for a in cfg["arms"]]
    missing_req = [a for a in CONTRACT_ARMS_REQUIRED if a not in got]
    extra = [a for a in got if a not in CONTRACT_ARMS]
    want_order = [a for a in CONTRACT_ARMS if a in got]
    if missing_req or extra or got != want_order:
        raise SystemExit(f"config.json 的臂清单与契约不一致：缺必到臂 {missing_req}、"
                         f"多出 {extra}；或缺/多之外顺序不对：\n want {want_order}\n got  {got}")
    if [int(s) for s in cfg["seeds"]] != CONTRACT_SEEDS:
        raise SystemExit(f"config.json 的 seeds 与契约不一致：{cfg['seeds']} != {CONTRACT_SEEDS}")
    for block, want in CONTRACT_EVAL.items():
        for k, v in want.items():
            if int(cfg["eval"][block][k]) != int(v):
                raise SystemExit(f"config.json eval.{block}.{k} 与契约不一致："
                                 f"{cfg['eval'][block][k]} != {v}")
    if [int(x) for x in cfg["curves"]["recent_ds"]] != CONTRACT_RECENT_DS:
        raise SystemExit("config.json curves.recent_ds 与契约不一致")
    if [int(x) for x in cfg["curves"]["distance_ds"]] != CONTRACT_DISTANCE_DS:
        raise SystemExit("config.json curves.distance_ds 与契约不一致")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    _check_config_against_contract(cfg)
    return cfg


def model_kwargs(spec: dict, mcfg: dict) -> dict:
    kw = dict(mcfg)
    kw.pop("note", None)
    is_full = spec["kind"] == "full"
    kw["train_window"] = 0 if is_full else int(spec["window"])
    kw["train_anchors"] = 0 if is_full else int(spec.get("anchors", 0))
    return kw


def keep_mask_for(spec: dict, T: int, dev: str):
    """臂**自己的训练可见性**（`full` ⇒ 全因果）。"""
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
    m = DepthARModel(DepthARConfig(**model_kwargs(spec, mcfg)), vocab)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"path": spec["path"], "sha256": got,
               "size_bytes": os.path.getsize(path), "sha256_verified": True,
               "kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0), "seed": spec["seed"],
               "n_params": sum(p.numel() for p in m.parameters())}


def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """**严格因果性探针（必须不变项）**：只改后半段 token，前半段 logits 必须逐位不变。"""
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    with torch.no_grad():
        a = rp.masked_logits(model, x, keep_mask)
        b = rp.masked_logits(model, x2, keep_mask)
    return {"max_abs_logit_diff_prefix":
            round(float((a[:, :half] - b[:, :half]).abs().max()), 8),
            "max_abs_logit_diff_suffix":
            round(float((a[:, half:] - b[:, half:]).abs().max()), 8),
            "note": "前半段必须逐位为 0；后半段必然 >0（否则探针无效）"}


def cross_seed_mean(values: list[float], *, unit_note: str = "") -> dict:
    """跨 seed 均值的 95% 区间（种子 = 重复单位）。`n<2` 时区间置 None 并注明。"""
    n = len(values)
    if n == 0:
        return {"n_seeds": 0, "note": "无值"}
    mean = sum(values) / n
    out = {"n_seeds": n, "mean": round(mean, 6), "values": [round(v, 6) for v in values],
           "same_sign": bool(all(v > 0 for v in values) or all(v < 0 for v in values)),
           "unit_note": unit_note}
    if n < 2:
        out.update({"sd": None, "t_interval_95": None, "note": "n<2，无跨 seed 区间"})
        return out
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    crit = T_CRIT_95.get(n - 1)
    out["sd"] = round(sd, 6)
    out["t_crit"] = crit
    out["t_interval_95"] = (None if crit is None else
                            [round(mean - crit * sd / math.sqrt(n), 6),
                             round(mean + crit * sd / math.sqrt(n), 6)])
    return out


def classify_tax(mean: float | None, sd: float | None, n: int,
                 band: float = FROZEN_SIGMA_CELL) -> dict:
    """判据② 的三分类（`README.md` §5a 预登记）：**不许压成一个布尔**。

    - `above_band`：区间**下界** > band ⇒ 税确定越本格带；
    - `below_band`：区间**上界** < band ⇒ 税确定不越本格带；
    - `undecidable`：区间跨 band（或 n<2 ⇒ 无区间）⇒ 只能写「本 n 下不可判」，并给所需样本量。

    `n_req = ceil((2.9·sd/|mean − band|)²)` 是双侧 α=0.05、功效 0.8 的近似式；`sd` 未知
    （n<2）时按「先补到 n=2」返回 `None` 并在 note 里说清。
    """
    if mean is None or n < 1:
        return {"verdict": "undecidable", "n": n, "n_req": 2,
                "note": "无读数；先补到 n≥2 才能判"}
    band_delta = mean - band
    if n < 2 or sd is None:
        return {"verdict": "undecidable", "n": n, "n_req": 2,
                "mean_minus_band": round(band_delta, 6),
                "note": "n<2，无跨 seed 区间 ⇒ 不可判；先补到 n=2"}
    half = T_CRIT_95.get(n - 1) * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    if lo > band:
        verdict = "above_band"
    elif hi < band:
        verdict = "below_band"
    else:
        verdict = "undecidable"
    if band_delta == 0:
        n_req = None
        n_req_note = "效应恰在带边界上，样本量不救（区间缩到 0 也跨带）"
    else:
        n_req = int(math.ceil((POWER_Z * sd / abs(band_delta)) ** 2))
        n_req_note = ""
    return {"verdict": verdict, "n": n, "mean_minus_band": round(band_delta, 6),
            "interval_95": [round(lo, 6), round(hi, 6)],
            "band": band, "n_req": n_req, "n_req_note": n_req_note,
            "n_req_rule": "n_req = ceil((2.9·sd/|mean − band|)²)（双侧 α=0.05、功效 0.8 的近似式）",
            "note": ("above_band = 区间下界越带（税确定存在）；below_band = 区间上界低于带"
                     "（税确定不越带）；undecidable = 区间跨带（不许写成任何一方）")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    t0 = time.time()
    log_lines: list[str] = []

    def note(msg: str) -> None:
        stamp = f"[{time.time() - t0:7.1f}s]"
        log_lines.append(f"{stamp} {msg}")
        print(stamp, msg, flush=True)

    cfg = load_config(args.config)
    T = int(cfg["model"]["seq_len"])
    eb = cfg["eval"]["bpc"]
    if args.smoke:
        eb = dict(eb, n_batches=4)

    _, eval_arr, vocab_size_file = load_enwik8_full(resolve(cfg["data"]["path"]),
                                                    resolve("data/enwik8_full.npz"))
    vocab = int(cfg["data"]["vocab_size"])
    assert vocab_size_file == vocab, f"vocab {vocab_size_file} != {vocab}"
    n_real = int(cfg["data"]["n_real_tokens"])
    mask_token = int(cfg["data"]["mask_token"])
    assert vocab == mask_token + 1, (vocab, mask_token)

    arms, ckpts = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ckpts[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
    note(f"装载 {len(arms)} 臂（sha256 全部核对）；数据 {cfg['data']['name']} vocab={vocab}")

    g = torch.Generator().manual_seed(eb["window_seed"])
    from pathlm.data import batch as _batch
    x_probe, _ = _batch(eval_arr, int(eb["batch_size"]), T, g)
    x_probe = x_probe.to(dev)

    # ---------------- ⑦ 自检（必须不变项）----------------
    selfchecks = {}
    for name in ("full", "win16_s0", "win64_s0", "win192_s0", "win16a_s0"):
        if name not in arms:
            continue
        km = keep_mask_for(next(s for s in cfg["arms"] if s["name"] == name), T, dev)
        selfchecks[name] = {"causality": causality_probe(arms[name], x_probe, km),
                            "n_params": ckpts[name]["n_params"]}
    visible = {}
    for k in EXPECTED_VISIBLE:
        w, a = k
        got = int(window_anchor_keep_mask(T, w, a).sum().item())
        visible[f"({w},{a})"] = {"visible_pairs": got, "expected": expected_visible_pairs(T, w, a),
                                 "expected_hardcoded": EXPECTED_VISIBLE[k],
                                 "equal": got == expected_visible_pairs(T, w, a) == EXPECTED_VISIBLE[k]}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        w, a = int(spec["window"]), int(spec.get("anchors", 0))
        if f"({w},{a})" in visible:
            continue
        got = int(window_anchor_keep_mask(T, w, a).sum().item())
        visible[f"({w},{a})"] = {"visible_pairs": got, "expected": expected_visible_pairs(T, w, a),
                                 "equal": got == expected_visible_pairs(T, w, a)}
    rel_ident = recent_keep_mask(T, T, dev)
    causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    selfchecks["mask_identity"] = {
        "recent_keep_mask_T_equals_causal": bool(torch.equal(rel_ident, causal)),
        "note": "recent_keep_mask(T) 必须与精确因果掩码逐位相同（否则累计预算列的 released 参照错）"}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        w, a = int(spec["window"]), int(spec.get("anchors", 0))
        if a == 0:
            own = window_anchor_keep_mask(T, w, 0, device=dev)
            selfchecks.setdefault("mask_matches_recent_clip", {})[f"k={w}"] = bool(
                torch.equal(own, recent_keep_mask(T, w, dev)))
    note(f"自检⑦：因果性 prefix diff = "
         f"{ {k: v['causality']['max_abs_logit_diff_prefix'] for k, v in selfchecks.items() if isinstance(v, dict) and 'causality' in v} }")

    # ---------------- ① 复现门（仪器自检）----------------
    note(f"复现门①：重评 {len(CONTRACT_REPRO_ARMS)} 条 1.1.1 已有 checkpoint，"
         f"要求与落盘值逐位相同（{eb['n_batches']}×{eb['batch_size']}，"
         f"window_seed={eb['window_seed']}）")
    bpc = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        bpc[name] = round(rp.win16_bpc(arms[name], eval_arr, km,
                                       n_batches=eb["n_batches"],
                                       batch_size=eb["batch_size"],
                                       window_seed=eb["window_seed"], device=dev), 6)
        note(f"  {name}: bpc={bpc[name]:.6f}")
    repro = {n: {"got": bpc[n], "want": REF_BPC[n], "equal": bpc[n] == REF_BPC[n]}
             for n in CONTRACT_REPRO_ARMS if n in bpc}
    repro_all_equal = bool(all(v["equal"] for v in repro.values()))
    note(f"复现门①：{sum(v['equal'] for v in repro.values())}/{len(repro)} 逐位相同 "
         f"⇒ passes={repro_all_equal}")

    # ---------------- ② 训练期累计预算列（主判）----------------
    note(f"训练期累计预算列②（主判）：k ∈ {CONTRACT_KS}，税 = bpc(win-k) − bpc(full)，"
         f"带 = 冻结 {FROZEN_SIGMA_CELL}")
    train_curve = {}
    full_bpc = bpc.get("full")
    for k in CONTRACT_KS:
        per_seed, taxes = {}, []
        for s in CONTRACT_SEEDS:
            name = f"win{k}_s{s}"
            if name not in bpc:
                continue
            tax = round(bpc[name] - full_bpc, 6)
            per_seed[str(s)] = {"arm": name, "bpc": bpc[name], "delta_bpc": tax}
            taxes.append(tax)
        csm = cross_seed_mean(taxes, unit_note="每 seed 一个窗口税（同 seed 配对：bpc(win-k) − bpc(full)）")
        cls = classify_tax(csm.get("mean"), csm.get("sd"), csm["n_seeds"])
        seed_range = (round(max(taxes) - min(taxes), 6) if len(taxes) >= 2 else None)
        train_curve[f"k{k}"] = {
            "k": k, "n_seeds": csm["n_seeds"], "per_seed": per_seed,
            "cross_seed": csm, "classification": cls,
            "seed_range_observed": seed_range,
            "seed_range_note": "本 n 的种子极差只是描述量，不得回填成新带（带冻结在 0.000983）",
        }
        note(f"  k={k:3d}: n={csm['n_seeds']} mean_tax={csm.get('mean')} "
             f"interval={cls.get('interval_95')} ⇒ {cls['verdict']}"
             + (f"（n_req={cls['n_req']}）" if cls["verdict"] == "undecidable" else ""))
    means = {k: train_curve[f"k{k}"]["cross_seed"].get("mean") for k in CONTRACT_KS
             if f"k{k}" in train_curve}
    k_list = [k for k in CONTRACT_KS if f"k{k}" in train_curve]
    monotone_pairs = [{"k_from": k_list[i], "k_to": k_list[i + 1],
                       "mean_from": means[k_list[i]], "mean_to": means[k_list[i + 1]],
                       "decreasing": bool(means[k_list[i + 1]] <= means[k_list[i]])}
                      for i in range(len(k_list) - 1)]
    # 落进本格带的第一个 k（税确定不越带的**最小** k），没有则 None
    below = [k for k in k_list
             if train_curve[f"k{k}"]["classification"]["verdict"] == "below_band"]
    band_edge = {
        "smallest_k_below_band": (min(below) if below else None),
        "below_band_k": below,
        "note": ("below_band = 该 k 的税**确定**不越本格带；本 n 下不可判的 k 不算进来，"
                 "也不许当成 below_band 读"),
    }

    # ---------------- ③ 评测期累计预算列 ---------------- 
    note(f"评测期累计预算列③：recent_keep_mask(256,D)，D ∈ {CONTRACT_RECENT_DS}")
    recent_col = {}
    for arm in CONTRACT_CURVE_ARMS:
        if arm not in arms:
            continue
        released = bpc[arm]
        spec = next(s for s in cfg["arms"] if s["name"] == arm)
        own = None if spec["kind"] == "full" else int(spec["window"])
        rows = {}
        for D in CONTRACT_RECENT_DS:
            km = recent_keep_mask(T, D, dev)
            v = round(rp.win16_bpc(arms[arm], eval_arr, km, n_batches=eb["n_batches"],
                                   batch_size=eb["batch_size"],
                                   window_seed=eb["window_seed"], device=dev), 6)
            relation = ("released_causal" if own is None
                        else ("identity" if D == own else
                              ("narrower" if D < own else "wider_than_training")))
            rows[str(D)] = {"D": D, "bpc": v, "delta_bpc": round(v - released, 6),
                            "relation_to_arm_budget": relation,
                            "mask": mask_stats(km.cpu())}
        recent_col[arm] = {
            "released_bpc": released, "released_budget": (T if own is None else own),
            "rows": rows,
            "x_axis_unit": "位置（最近 D 个 key 可见）",
            "drops_what": "所有距离 ≥ D 的 key（累计配置，直接测，不是单点求和）",
            "additivity": "direct_cumulative",
            "relation_note": ("relation_to_arm_budget：identity = D 等于该臂的训练窗口（掩码恒等，"
                              "Δ 必须逐位为 0）；narrower = 比训练更窄（对窗口臂是真正的「丢前缀」）；"
                              "wider_than_training = 比训练更宽（OOD 放行，不是预算收紧）"),
        }
        note(f"  {arm}: " + " ".join(f"D={D}:{rows[str(D)]['delta_bpc']:+.6f}"
                                     for D in CONTRACT_RECENT_DS))
    # 判据③ 自检：D == 该臂训练窗口且 anchors=0 ⇒ 掩码恒等 ⇒ Δ 必须逐位为 0
    ident_check = {}
    for arm in CONTRACT_CURVE_ARMS:
        if arm not in recent_col or arm == "full":
            continue
        spec = next(s for s in cfg["arms"] if s["name"] == arm)
        w, a = int(spec["window"]), int(spec.get("anchors", 0))
        if a != 0:
            continue
        got = recent_col[arm]["rows"][str(w)]["delta_bpc"]
        ident_check[f"{arm}|D={w}"] = {"delta_bpc": got, "equal_zero": got == 0.0,
                                       "why": f"D == 训练窗口 {w} 且无锚点 ⇒ 掩码恒等"}
    note(f"  自检③ 掩码恒等：{sum(v['equal_zero'] for v in ident_check.values())}"
         f"/{len(ident_check)} 处 Δ 逐位为 0")

    # ---------------- ④ 单距离列 ----------------
    note(f"单距离列④：distance_keep_mask(256,d)，d ∈ {CONTRACT_DISTANCE_DS}（只描述形状）")
    dist_col = {}
    for arm in CONTRACT_CURVE_ARMS:
        if arm not in arms:
            continue
        released = bpc[arm]
        rows = {}
        for d in CONTRACT_DISTANCE_DS:
            km = distance_keep_mask(T, d, dev)
            v = round(rp.win16_bpc(arms[arm], eval_arr, km, n_batches=eb["n_batches"],
                                   batch_size=eb["batch_size"],
                                   window_seed=eb["window_seed"], device=dev), 6)
            rows[str(d)] = {"d": d, "bpc": v, "delta_bpc": round(v - released, 6),
                            "mask": mask_stats(km.cpu())}
        dist_col[arm] = {
            "released_bpc": released, "rows": rows,
            "x_axis_unit": "位置（与查询距离恰为 d 的 1 个 key 被删）",
            "drops_what": "距离恰为 d 的那一个 key（每 query 删 1 对可见性）",
            "additivity": "shape_only",
            "note": "单点曲线只描述形状；不得累加、不得外推到别的预算（SCHEMA §17.11）",
        }
        note(f"  {arm}: " + " ".join(f"d={d}:{rows[str(d)]['delta_bpc']:+.6f}"
                                     for d in CONTRACT_DISTANCE_DS))

    # ---------------- ⑤ 两列不可互换的演示（只在 full 臂上判，README §5c）----------------
    demo = {}
    for arm in CONTRACT_CURVE_ARMS:
        if arm not in recent_col or arm not in dist_col:
            continue
        cum = recent_col[arm]["rows"][str(CONTRACT_DEMO_D)]["delta_bpc"]
        parts = {str(d): dist_col[arm]["rows"][str(d)]["delta_bpc"]
                 for d in CONTRACT_DEMO_SINGLE_DS}
        sum_single = round(sum(parts.values()), 6)
        ratio = (None if sum_single == 0 else round(cum / sum_single, 6))
        judged = arm in CONTRACT_DEMO_JUDGED_ARMS
        demo[arm] = {
            "cum_delta_bpc": cum, "sum_single_delta_bpc": sum_single,
            "single_parts": parts, "ratio_cum_over_sum_single": ratio,
            "cum_above_frozen_band": bool(abs(cum) > FROZEN_SIGMA_CELL),
            "participates_in_prediction": judged,
            "prediction_holds": (bool(ratio is not None and ratio > 1
                                       and abs(cum) > FROZEN_SIGMA_CELL) if judged else None),
            "prediction": "cum(32) / Σ_{d∈{32,64,128,255}} single(d) > 1 且 cum 越本格带",
            "note": (None if judged else
                     "窗口臂的释放版本来就是窄的（D=32 比训练窗口更宽）⇒ 这里不是预算收紧，"
                     "比率不作预测判定，只作描述量"),
        }
        note(f"  {arm}: cum(32)={cum:+.6f} Σsingle={sum_single:+.6f} ratio={ratio} "
             f"⇒ 参与预测={judged} 成立={demo[arm]['prediction_holds']}")

    # ---------------- ⑥ 锚点杠杆在第二个 k（可选臂）----------------
    anchor_k64 = None
    if all(a in bpc for a in ("win64a_s0", "win64a_s1", "win64_s0", "win64_s1")):
        per_seed, prices = {}, []
        for s in (0, 1):
            d = round(bpc[f"win64a_s{s}"] - bpc[f"win64_s{s}"], 6)
            per_seed[str(s)] = {"bpc_win64a": bpc[f"win64a_s{s}"],
                                "bpc_win64": bpc[f"win64_s{s}"], "delta_bpc": d}
            prices.append(d)
        csm = cross_seed_mean(prices, unit_note="每 seed 一个价格（同 seed 配对：bpc(win64a) − bpc(win64)）")
        ci = csm.get("t_interval_95")
        if ci is None:
            verdict, n_req = "undecidable", 2
        elif ci[0] > 0 or ci[1] < 0:
            verdict, n_req = "decidable", None
        else:
            verdict = "undecidable"
            n_req = (int(math.ceil((POWER_Z * csm["sd"] / abs(csm["mean"])) ** 2))
                     if csm["mean"] else None)
        anchor_k64 = {"k": 64, "anchors": 8, "n_seeds": csm["n_seeds"],
                      "per_seed": per_seed, "cross_seed": csm, "verdict": verdict,
                      "n_req": n_req,
                      "reference_k16": -0.00706,
                      "reference_note": "k=16 的 n=4 跨 seed 均值见 2681 判据⑦（−0.00706 bpc）",
                      "rule": "t 区间不含 0 ⇒ 可判（报方向）；含 0 ⇒ 不可判并给所需 n"}
        note(f"锚点杠杆⑥ k=64：价格 {prices} 均值 {csm['mean']} 区间 {ci} ⇒ {verdict}")
    else:
        anchor_k64 = {"verdict": "not_produced",
                      "note": "可选臂 win64a_s0/s1 未训完（README §4b 已预登记该回落）"}
        note("锚点杠杆⑥：可选臂未产出 ⇒ 如实回落到 not_produced")

    # ---------------- 成本 ----------------
    kv_bytes = int(cfg["model"]["n_layers"]) * 2 * int(cfg["model"]["d_model"]) * 2
    cost = {
        "gpu_seconds": round(time.time() - t0, 1),
        "wall_seconds": round(time.time() - t0, 1),
        "kv_bytes_per_position_unit": kv_bytes,
        "k_eff_by_arm": {k: (int(v["window"]) + int(v.get("anchors", 0)))
                         for k, v in ckpts.items()},
        "kv_bytes_by_arm": {k: kv_bytes * (int(v["window"]) + int(v.get("anchors", 0)))
                            for k, v in ckpts.items()},
        "note": "KV 字节按 fp16（每位置 = 层数 × 2(K,V) × d_model × 2 B × k_eff）",
    }

    code_hashes = {f: sha256_file(os.path.join(HERE, f))
                   for f in ("run.py", "make_config.py", "realtext_probe.py",
                             "train_arms.sh", "config.json")}
    for rel in ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/retention.py",
                "pathlm/data.py"):
        code_hashes[rel] = sha256_file(os.path.join(ROOT, rel))

    results = {
        "sign_convention": (
            "所有 `*delta_bpc*` 读数 = bpc(被测配置) − bpc(该臂的释放版)；数值越大说明掩码造成的"
            "损伤越大。训练期累计预算列的 delta_bpc 是 bpc(win-k) − bpc(full)，数值越大说明"
            "窗口税越大。"),
        "config": {"contract": "axes/1-time/1.1.1-hard-window/experiments/"
                               "2682-window-kcurve-13m/README.md",
                   "scope": ("只做单轴单格实验，基底恒为 stock dense 全因果；"
                             "本实验只改训练期窗口宽度 k，并报两条评测期掩码曲线。"),
                   "model": cfg["model"], "data": cfg["data"], "eval": cfg["eval"],
                   "arms": cfg["arms"], "seeds": cfg["seeds"], "curves": cfg["curves"]},
        "checkpoints": ckpts,
        "selfchecks": selfchecks,
        "visible_pairs": visible,
        "criteria": {
            "c1_reproduction_gate": {
                "repro": repro, "all_equal": repro_all_equal,
                "note": ("1.1.1 目录内已有 9 条 checkpoint 用本入口的仪器重评，必须与 2662 / 2681 "
                         "落盘值逐位相同；不等即本实验作废"),
            },
            "c2_train_budget_column": {
                "k_grid": CONTRACT_KS, "per_k": train_curve,
                "band": {"sigma_cell_frozen": FROZEN_SIGMA_CELL,
                         "source": "2662/results.json::quality_leg.bands.sigma_cell",
                         "cross_cell_band": CROSS_CELL_BAND,
                         "note": "两带并列报、不混用（C1 裁决）；本实验不重算 sigma_cell"},
                "monotone_pairs": monotone_pairs, "band_edge": band_edge,
                "classification_rule": "above_band / below_band / undecidable（README §5a）",
                "note": ("主判。三分类不许压成布尔：undecidable 的 k 只能写「本 n 下不可判」，"
                         "并附 n_req"),
            },
            "c3_eval_budget_column": {
                "D_grid": CONTRACT_RECENT_DS, "by_arm": recent_col,
                "identity_check": ident_check,
                "note": ("评测期丢前缀：同一条已训权重、只改掩码、0 训练 token。与判据②的"
                         "训练期税并列成两列表，口径不许互换"),
            },
            "c4_single_distance_column": {
                "d_grid": CONTRACT_DISTANCE_DS, "by_arm": dist_col,
                "note": "单距离列只描述形状，不得求和或外推到别的预算（SCHEMA §17.11）",
            },
            "c5_columns_not_interchangeable": {
                "demo_D": CONTRACT_DEMO_D, "demo_single_ds": CONTRACT_DEMO_SINGLE_DS,
                "by_arm": demo,
                "note": ("预登记预测：cum(32) / Σ_{d≥32} single(d) > 1 且 cum 越本格带。"
                         "不成立必须写清原因，不得默默只报一条曲线"),
            },
            "c6_anchor_lever_at_k64": anchor_k64,
        },
        "cost": cost,
        "code_hashes": code_hashes,
        "delta_directions": {
            "delta_bpc": "arm_minus_comparator",
            "cum_delta_bpc": "arm_minus_comparator",
            "sum_single_delta_bpc": "arm_minus_comparator",
        },
        "readings_units": {
            "bpc": "bits/byte（越低越好），96×16 窗口序、window_seed=2",
            "delta_bpc": "bpc 差，正 = 该配置比释放版更差",
        },
    }

    out = os.path.join(args.out_dir, "results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(args.out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    note(f"写出 {out}")

    if not repro_all_equal:
        note("判据① 不过 ⇒ 本实验作废（读数不入账，先修仪器）")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
