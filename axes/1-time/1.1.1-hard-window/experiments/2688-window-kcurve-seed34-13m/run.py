#!/usr/bin/env python3
"""2688-window-kcurve-seed34-13m · 评测入口（纯推理）：把 `2686` 的两条「不可判」再补一步。

契约：同目录 `README.md`（§6 判据线预登记，写下后不改）。**本实验只增大样本量、不改判据线**
（§6 判据⑤）：判据② 的 k=128 从 n=3 补到 n=5、判据⑥ 的 k=64 锚点杠杆从 n=3 补到 n=4，
冻结带 0.000983、`t` 表、功效式、三分类、配对规则全部照抄 `2682` §5a / `2686` §6。

本入口做：

  ⓪ **预登记门**：调用 `tools/prereg_bind.py` 的 `require()`——契约未绑定或被改过即拒绝启动；
  ① **复现门**：`2686` 目录内已有的 10 条 checkpoint（`full`、`win64_s0/s1/s2`、`win64a_s0/s1/s2`、
     `win128_s0/s1/s2`）的 clean bpc 必须与 `2686/results.json` 落盘值**逐位相同**；不等 ⇒ 本实验作废；
  ② **判据② 的 k=128 窗口税补到 n=5**：同 seed 配对税 `bpc(win128_s) − bpc(full)`，三分类；
  ③ **判据⑥ 的 k=64 锚点杠杆补到 n=4**：同 seed 配对价 `bpc(win64a_s) − bpc(win64_s)`，区间不含 0 即可判；
  ④ 自检：因果性 + 掩码可见对数 + 参数量 + 窗口掩码与 `recent_keep_mask` 逐位相同；
  ⑤ 契约里写下的两条**预测**（由 `2686` 的 n=3 读数外推）与实测判定比对，写进 `results.json`。

用法：
  python3 run.py            # 全电池
  python3 run.py --smoke    # 冒烟：少批（只用于测步时 / 查错）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.prereg_bind as prereg_bind                            # noqa: E402
from pathlm.data import batch as data_batch, load_enwik8_full       # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,           # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import recent_keep_mask                # noqa: E402
import realtext_probe as rp                                        # noqa: E402

#: 跨格参考带（C1 裁决：本格自测带为主判、跨格 0.014 为参考；两带并列报、不混用）。
CROSS_CELL_BAND = 0.014
#: **冻结**的本格自测带（= `2662` 的两 seed 极差，`2682` / `2686` 也未重算）。本实验照抄、不重算。
FROZEN_SIGMA_CELL = 0.000983
#: 跨 seed 均值的 t 区间临界值（双侧 α=0.05，df = 种子数 − 1）。
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}
#: 判据② 的功效近似式常数（双侧 1.96 + 单侧 0.84）。
POWER_Z = 2.9

#: 契约臂清单（`config.json` 必须逐项对上，顺序也一致）。顺序 = 报告顺序。
CONTRACT_ARMS = ["full",
                 "win64_s0", "win64_s1", "win64_s2", "win64_s3",
                 "win64a_s0", "win64a_s1", "win64a_s2", "win64a_s3",
                 "win128_s0", "win128_s1", "win128_s2", "win128_s3", "win128_s4"]
#: 本实验训练的臂（其余为复用；`config.json` 里这四条必须有 `train_rc`）。
CONTRACT_NEW_ARMS = ["win64_s3", "win64a_s3", "win128_s3", "win128_s4"]
#: 两条判据各自的种子集合（判据② 到 n=5、判据③ 到 n=4）。
CONTRACT_SEEDS_K128 = [0, 1, 2, 3, 4]
CONTRACT_SEEDS_LEVER = [0, 1, 2, 3]
CONTRACT_EVAL = {"bpc": {"n_batches": 96, "batch_size": 16, "window_seed": 2}}
#: 判据① 复现门：`2686/results.json` 里同一条 checkpoint 的落盘 clean bpc（本目录用同一仪器重评）。
REF_BPC = {"full": 1.502689,
           "win64_s0": 1.555204, "win64_s1": 1.555988, "win64_s2": 1.548166,
           "win64a_s0": 1.547552, "win64a_s1": 1.547006, "win64a_s2": 1.545000,
           "win128_s0": 1.515296, "win128_s1": 1.512497, "win128_s2": 1.505451}
CONTRACT_REPRO_ARMS = ["full",
                       "win64_s0", "win64_s1", "win64_s2",
                       "win64a_s0", "win64a_s1", "win64a_s2",
                       "win128_s0", "win128_s1", "win128_s2"]
#: 判据② 的 k 档位（本实验只补这一档；其余 k 见 `2682`）。
CONTRACT_K128 = 128
#: 判据③ 的 k 与锚点数（同 seed 配对）。
CONTRACT_LEVER_K = 64
CONTRACT_LEVER_ANCHORS = 8
#: k=16 的锚点杠杆参照（`2681` 判据⑦，n=4）——只作并列参照，不参与本实验的判定。
REFERENCE_K16_LEVER = -0.00706
#: `2686` 的 n=3 落盘读数（判据② 的 k=128 与判据③ 的 k=64 杠杆）：只作增量对照与预测输入。
K128_N3 = {"values": [0.012607, 0.009808, 0.002762], "mean": 0.008392, "sd": 0.005073,
           "interval_95": [-0.004211, 0.020995], "n_req": 4,
           "source": "2686-window-kcurve-seed2-13m/results.json::criteria.c2_k128_tax_n3"}
LEVER_N3 = {"values": [-0.007652, -0.008982, -0.003166], "mean": -0.0066, "sd": 0.003047,
            "interval_95": [-0.014171, 0.000971],
            "source": "2686-window-kcurve-seed2-13m/results.json::criteria.c3_anchor_lever_k64_n3"}
#: 期望的掩码可见对数（`window_anchor_keep_mask(256, w, a)` 的闭式，判据④）。
EXPECTED_VISIBLE = {(64, 0): 14368, (64, 8): 15876, (128, 0): 24640}
N_PARAMS = 13536768


def expected_visible_pairs(T: int, w: int, a: int) -> int:
    """`window_anchor_keep_mask(T, w, a)` 的可见 (query, key) 对数（闭式）。

    每行 `i` 的可见数 = 最近窗口 `min(i+1, w)` + 锚点额外放行的 `min(a, max(0, i-w+1))`
    （锚点 `[0, a)` 里那些**不在**最近窗口内的 key）。与 `EXPECTED_VISIBLE` 的硬编码点互为独立校验。
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
    """`config.json` 是训练后生成的产物（含检查点 sha256）⇒ 在**已绑定的本文件**里核对它。"""
    got = [a["name"] for a in cfg["arms"]]
    if got != CONTRACT_ARMS:
        raise SystemExit(f"config.json 的臂清单与契约不一致：\n want {CONTRACT_ARMS}\n got  {got}")
    for block, want in CONTRACT_EVAL.items():
        for k, v in want.items():
            if int(cfg["eval"][block][k]) != int(v):
                raise SystemExit(f"config.json eval.{block}.{k} 与契约不一致："
                                 f"{cfg['eval'][block][k]} != {v}")


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
        raise SystemExit(f"{spec['name']}: checkpoint sha256 mismatch:\n want {want}\n got  {got}")
    m = DepthARModel(DepthARConfig(**model_kwargs(spec, mcfg)), vocab)
    m.load_state_dict(torch.load(path, map_location=dev))
    m.to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"path": spec["path"], "sha256": got,
               "size_bytes": os.path.getsize(path), "sha256_verified": True,
               "kind": spec["kind"], "window": spec["window"],
               "anchors": spec.get("anchors", 0), "seed": spec["seed"],
               "reused_from": spec.get("reused_from"),
               "train_rc": spec.get("train_rc"),
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
    """判据② 的三分类（`2682/README.md` §5a 的冻结规则）：**不许压成一个布尔**。

    - `above_band`：区间**下界** > band ⇒ 税确定越本格带；
    - `below_band`：区间**上界** < band ⇒ 税确定不越本格带；
    - `undecidable`：区间跨 band（或 n<2）⇒ 只能写「本 n 下不可判」，并给所需样本量。
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
        n_req, n_req_note = None, "效应恰在带边界上，样本量不救（区间缩到 0 也跨带）"
    else:
        n_req, n_req_note = int(math.ceil((POWER_Z * sd / abs(band_delta)) ** 2)), ""
    return {"verdict": verdict, "n": n, "mean_minus_band": round(band_delta, 6),
            "interval_95": [round(lo, 6), round(hi, 6)],
            "band": band, "n_req": n_req, "n_req_note": n_req_note,
            "n_req_rule": "n_req = ceil((2.9·sd/|mean − band|)²)（双侧 α=0.05、功效 0.8 的近似式）",
            "note": ("above_band = 区间下界越带（税确定存在）；below_band = 区间上界低于带"
                     "（税确定不越带）；undecidable = 区间跨带（不许写成任何一方）")}


def classify_lever(csm: dict) -> dict:
    """判据③ 的判定（`2682` 判据⑥ 的同一条规则）：t 区间不含 0 ⇒ 可判并报方向。

    `sd` 未知（n<2）或区间含 0 ⇒ `undecidable`；后者给出按功效式估计的所需样本量。
    """
    mean, sd, n = csm.get("mean"), csm.get("sd"), csm.get("n_seeds", 0)
    iv = csm.get("t_interval_95")
    if mean is None or n < 2 or sd is None or iv is None:
        return {"verdict": "undecidable", "n": n, "n_req": 2,
                "note": "n<2，无跨 seed 区间 ⇒ 不可判；先补到 n=2"}
    lo, hi = iv
    out = {"n": n, "interval_95": iv,
           "rule": "t 区间不含 0 ⇒ 可判（报方向）；含 0 ⇒ 不可判并给所需 n"}
    if lo > 0:
        out.update({"verdict": "decidable", "direction": "price_positive"})
    elif hi < 0:
        out.update({"verdict": "decidable", "direction": "price_negative"})
    elif mean == 0:
        out.update({"verdict": "undecidable", "direction": None, "n_req": None,
                    "n_req_note": "均值恰为 0，样本量不救（区间中心就落在 0）",
                    "note": "区间含 0 ⇒ 只能写「本 n 下不可判」；不得读成「锚点无效」"})
    else:
        out.update({"verdict": "undecidable", "direction": None,
                    "n_req": int(math.ceil((POWER_Z * sd / abs(mean)) ** 2)),
                    "n_req_rule": "n_req = ceil((2.9·sd/|mean|)²)（双侧 α=0.05、功效 0.8 的近似式）",
                    "note": "区间含 0 ⇒ 只能写「本 n 下不可判」；不得读成「锚点无效」"})
    return out


def lever_from_mean_sd(mean: float, sd: float, n: int) -> dict:
    """把 (mean, sd, n) 拼成 `classify_lever` 要的形状（预测与实测走**同一条**判定规则）。"""
    crit = T_CRIT_95.get(n - 1)
    iv = (None if crit is None else
          [round(mean - crit * sd / math.sqrt(n), 6), round(mean + crit * sd / math.sqrt(n), 6)])
    return classify_lever({"mean": mean, "sd": sd, "n_seeds": n, "t_interval_95": iv})


def forecast_tax(prev: dict, n: int) -> dict:
    """契约 §1 的预测：把 n=3 的 mean/sd **原样**代入下一个 n 的三分类（`sd` 不涨是前提）。"""
    crit = T_CRIT_95.get(n - 1)
    half = crit * prev["sd"] / math.sqrt(n)
    mean = prev["mean"]
    sd_needed = (None if mean - FROZEN_SIGMA_CELL <= 0 else
                 (mean - FROZEN_SIGMA_CELL) * math.sqrt(n) / crit)
    return {"assume_mean": mean, "assume_sd": prev["sd"], "n": n, "t_crit": crit,
            "half_width_if_sd_holds": round(half, 6),
            "predicted_verdict": classify_tax(mean, prev["sd"], n)["verdict"],
            "sd_upper_bound_for_prediction": (None if sd_needed is None else round(sd_needed, 6)),
            "basis": prev.get("source")}


def forecast_lever(prev: dict, n: int) -> dict:
    """契约 §1 的预测：把 n=3 的 mean/sd 原样代入 n=4 的杠杆判定（`sd` 不涨是前提）。"""
    crit = T_CRIT_95.get(n - 1)
    mean, sd = prev["mean"], prev["sd"]
    half = crit * sd / math.sqrt(n)
    sd_needed = (None if mean >= 0 else abs(mean) * 2 / crit)
    return {"assume_mean": mean, "assume_sd": sd, "n": n, "t_crit": crit,
            "half_width_if_sd_holds": round(half, 6),
            "predicted_verdict": lever_from_mean_sd(mean, sd, n)["verdict"],
            "predicted_direction": lever_from_mean_sd(mean, sd, n).get("direction"),
            "sd_upper_bound_for_prediction": (None if sd_needed is None else round(sd_needed, 6)),
            "basis": prev.get("source")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # ⓪ 预登记门：契约必须先绑
    if prereg_bind.require(Path(HERE)) != 0:
        return 3

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

    arms, ckpts = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ckpts[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
    note(f"装载 {len(arms)} 臂（sha256 全部核对）；数据 {cfg['data']['name']} vocab={vocab}")

    g = torch.Generator().manual_seed(eb["window_seed"])
    x_probe, _ = data_batch(eval_arr, int(eb["batch_size"]), T, g)
    x_probe = x_probe.to(dev)

    # ---------------- ④ 自检（必须不变项）----------------
    selfchecks = {}
    specs = {s["name"]: s for s in cfg["arms"]}
    for name in CONTRACT_NEW_ARMS:
        km = keep_mask_for(specs[name], T, dev)
        selfchecks[name] = {"causality": causality_probe(arms[name], x_probe, km),
                            "n_params": ckpts[name]["n_params"]}
    visible = {}
    for (w, a), want in EXPECTED_VISIBLE.items():
        got = int(window_anchor_keep_mask(T, w, a).sum().item())
        visible[f"({w},{a})"] = {"visible_pairs": got, "expected": expected_visible_pairs(T, w, a),
                                 "expected_hardcoded": want,
                                 "equal": got == expected_visible_pairs(T, w, a) == want}
    selfchecks["mask_identity"] = {
        "win64_matches_recent_keep_mask": bool(torch.equal(
            window_anchor_keep_mask(T, 64, 0, device=dev), recent_keep_mask(T, 64, dev))),
        "win128_matches_recent_keep_mask": bool(torch.equal(
            window_anchor_keep_mask(T, 128, 0, device=dev), recent_keep_mask(T, 128, dev))),
        "recent_keep_mask_T_equals_causal": bool(torch.equal(
            recent_keep_mask(T, T, dev),
            torch.ones(T, T, dtype=torch.bool, device=dev).tril())),
        "note": "窗口掩码必须与共享底座的 recent_keep_mask 逐位相同（否则两列口径的参照错）"}
    selfchecks_all_ok = bool(
        all(v["causality"]["max_abs_logit_diff_prefix"] == 0.0
            for k, v in selfchecks.items() if isinstance(v, dict) and "causality" in v)
        and all(v["equal"] for v in visible.values())
        and all(v["n_params"] == N_PARAMS for v in selfchecks.values() if "n_params" in v)
        and selfchecks["mask_identity"]["win64_matches_recent_keep_mask"]
        and selfchecks["mask_identity"]["win128_matches_recent_keep_mask"]
        and selfchecks["mask_identity"]["recent_keep_mask_T_equals_causal"])
    note(f"自检④：prefix diff = "
         f"{ {k: v['causality']['max_abs_logit_diff_prefix'] for k, v in selfchecks.items() if isinstance(v, dict) and 'causality' in v} }"
         f"，掩码可见对数全等 = {all(v['equal'] for v in visible.values())}")

    # ---------------- bpc（仪器逐字不变）----------------
    bpc = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        bpc[name] = round(rp.win16_bpc(arms[name], eval_arr, km,
                                       n_batches=eb["n_batches"],
                                       batch_size=eb["batch_size"],
                                       window_seed=eb["window_seed"], device=dev), 6)
        note(f"  {name}: bpc={bpc[name]:.6f}")

    # ---------------- ① 复现门 ----------------
    repro = {n: {"got": bpc[n], "want": REF_BPC[n], "equal": bpc[n] == REF_BPC[n]}
             for n in CONTRACT_REPRO_ARMS}
    repro_all_equal = bool(all(v["equal"] for v in repro.values()))
    note(f"复现门①：{sum(v['equal'] for v in repro.values())}/{len(repro)} 逐位相同 "
         f"⇒ passes={repro_all_equal}")

    # ---------------- ② 判据② k=128 的税补到 n=5 ----------------
    full_bpc = bpc["full"]
    per_seed, taxes = {}, []
    for s in CONTRACT_SEEDS_K128:
        name = f"win{CONTRACT_K128}_s{s}"
        tax = round(bpc[name] - full_bpc, 6)
        per_seed[str(s)] = {"arm": name, "bpc": bpc[name], "delta_bpc": tax}
        taxes.append(tax)
    csm = cross_seed_mean(taxes, unit_note="每 seed 一个窗口税（同 seed 配对：bpc(win128) − bpc(full)）")
    cls = classify_tax(csm.get("mean"), csm.get("sd"), csm["n_seeds"])
    k128 = {"k": CONTRACT_K128, "n_seeds": csm["n_seeds"], "per_seed": per_seed,
            "cross_seed": csm, "classification": cls,
            "seed_range_observed": round(max(taxes) - min(taxes), 6),
            "seed_range_note": "本 n 的种子极差只是描述量，不得回填成新带（带冻结在 0.000983）",
            "n3_reference": K128_N3}
    note(f"判据② k=128：n={csm['n_seeds']} mean_tax={csm.get('mean')} "
         f"interval={cls.get('interval_95')} ⇒ {cls['verdict']}"
         + (f"（n_req={cls['n_req']}）" if cls["verdict"] == "undecidable" else ""))

    # ---------------- ③ 判据⑥ k=64 的锚点杠杆补到 n=4 ----------------
    per_seed, prices = {}, []
    for s in CONTRACT_SEEDS_LEVER:
        a_name, w_name = f"win{CONTRACT_LEVER_K}a_s{s}", f"win{CONTRACT_LEVER_K}_s{s}"
        price = round(bpc[a_name] - bpc[w_name], 6)
        per_seed[str(s)] = {"bpc_win64a": bpc[a_name], "bpc_win64": bpc[w_name],
                            "delta_bpc": price}
        prices.append(price)
    lcsm = cross_seed_mean(prices, unit_note="每 seed 一个价（同 seed 配对：bpc(win64a) − bpc(win64)）")
    lever = {"k": CONTRACT_LEVER_K, "anchors": CONTRACT_LEVER_ANCHORS,
             "n_seeds": lcsm["n_seeds"], "per_seed": per_seed, "cross_seed": lcsm,
             "verdict": classify_lever(lcsm),
             "reference_k16": REFERENCE_K16_LEVER,
             "reference_note": "k=16 的 n=4 跨 seed 均值见 2681 判据⑦（−0.00706 bpc），只作并列参照",
             "n3_reference": LEVER_N3}
    note(f"判据③ k=64 锚点杠杆：n={lcsm['n_seeds']} mean={lcsm.get('mean')} "
         f"interval={lcsm.get('t_interval_95')} ⇒ {lever['verdict']['verdict']}"
         f"（direction={lever['verdict'].get('direction')}）")

    # ---------------- ⑤ 契约 §1 的两条预测 vs 实测 ----------------
    fc_tax = forecast_tax(K128_N3, len(CONTRACT_SEEDS_K128))
    fc_lever = forecast_lever(LEVER_N3, len(CONTRACT_SEEDS_LEVER))
    predictions = {
        "c2": {**fc_tax, "observed_verdict": cls["verdict"],
               "observed_sd": csm.get("sd"),
               "held": cls["verdict"] == fc_tax["predicted_verdict"],
               "note": "契约 §1 写下时本轮读数尚未产生；预测不成立也照实报（README §7）"},
        "c3": {**fc_lever, "observed_verdict": lever["verdict"]["verdict"],
               "observed_direction": lever["verdict"].get("direction"),
               "observed_sd": lcsm.get("sd"),
               "held": (lever["verdict"]["verdict"] == fc_lever["predicted_verdict"]),
               "note": "同上"},
    }

    out = {
        "experiment": "axes/1-time/1.1.1-hard-window/experiments/2688-window-kcurve-seed34-13m",
        "contract": f"{cfg['contract']}（§6 判据线预登记）",
        "sign_convention": (
            "`delta_bpc` 一律 = 被测配置 − 对照，两者都写在同一行：判据② 的行是 "
            "`bpc(win128_s) − bpc(full)`（数值越大说明窗口税越大），判据③ 的行是 "
            "`bpc(win64a_s) − bpc(win64_s)`（数值越小说明锚点带来的质量越好）。"),
        "config": {k: cfg[k] for k in ("contract", "model", "data", "eval")},
        "seeds_k128": CONTRACT_SEEDS_K128,
        "seeds_lever": CONTRACT_SEEDS_LEVER,
        "checkpoints": ckpts,
        "selfchecks": selfchecks,
        "visible_pairs": visible,
        "criteria": {
            "c1_reproduction_gate": {
                "repro": repro, "all_equal": repro_all_equal,
                "note": f"`2686` 目录内已有的 {len(CONTRACT_REPRO_ARMS)} 条 checkpoint 用本入口的仪器"
                        "重评，必须与 2686 落盘值逐位相同；不等即本实验作废"},
            "c2_k128_tax_n5": k128,
            "c3_anchor_lever_k64_n4": lever,
            "c4_selfchecks": {"all_ok": selfchecks_all_ok,
                              "note": "因果性前缀逐位 0、掩码可见对数与闭式及硬编码点三处一致、"
                                      "参数量逐位相同、窗口掩码与 recent_keep_mask 逐位相同"},
            "c5_contract_inherited": {
                "new_criteria": False,
                "note": "本实验不新增判据线：② ③ 是 2682 §5a / 2686 §6 的同一条规则在更大的 n 上重算，"
                        "冻结带 0.000983、t 表、功效式、三分类、同 seed 配对规则一格不动（补 seed 不改判据）",
            },
            "c6_contract_predictions": predictions,
        },
        "cost": {"wall_seconds": None,
                 "gpu_seconds": None,
                 "n_new_arms": len(CONTRACT_NEW_ARMS),
                 "train_seconds": {a: ckpts[a]["train_rc"] is not None for a in CONTRACT_NEW_ARMS},
                 "note": "本轮只做评测；训练墙钟与 rc 在 config.json（*.log 是被忽略的派生物）"},
        "code_hashes": {
            "run.py": sha256_file(os.path.join(HERE, "run.py")),
            "make_config.py": sha256_file(os.path.join(HERE, "make_config.py")),
            "realtext_probe.py": sha256_file(os.path.join(HERE, "realtext_probe.py")),
            "train_arms.sh": sha256_file(os.path.join(HERE, "train_arms.sh")),
            "config.json": sha256_file(args.config),
            "pathlm/depth_ar.py": sha256_file(os.path.join(ROOT, "pathlm/depth_ar.py")),
            "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
            "tools/evalkit/retention.py": sha256_file(os.path.join(ROOT, "tools/evalkit/retention.py")),
            "pathlm/data.py": sha256_file(os.path.join(ROOT, "pathlm/data.py")),
        },
        "delta_directions": {"delta_bpc": "arm_minus_comparator"},
        "readings_units": {
            "bpc": "bits/byte（越低越好），96×16 窗口序、window_seed=2",
            "delta_bpc": "bpc 差（被测配置 − 对照）；判据② 越大越差，判据③ 越小越好",
        },
        "run_log": log_lines,
    }
    wall = round(time.time() - t0, 1)
    out["cost"]["wall_seconds"] = wall
    out["cost"]["gpu_seconds"] = wall

    dst = os.path.join(args.out_dir, "results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    with open(os.path.join(args.out_dir, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    note(f"写出 {dst}（{wall}s）")
    if not repro_all_equal:
        print("复现门① 未过 ⇒ 本实验作废：读数不入账，先修仪器", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
