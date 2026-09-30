"""2615 · 窗口训练后依赖是否「右移」（ticket 32 契约 O）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
被测量模型：两个 —— ① `full`（复用 2610 的 `trunk.pt`，全可见，不重训）；
② `window32`（本实验训练：训练期只看最近 32 个位置 + 8 个锚点，评测用同一掩码）。
仪器：`tools/evalkit/retention.py`（依赖腿 = 距离/前缀/最近 d 三条曲线；检索腿 = 针）。
训练侧掩码与评测侧掩码**共用** `pathlm.depth_ar.window_anchor_keep_mask`
（契约 P 第 ③ 项：否则量的是别的东西）。

跑五件事：
  ① 自检：两个 checkpoint 的 sha256 断言 + 释放版/身份掩码逐位一致 +
     **训练掩码 forward == 评测掩码 masked_logits**（同一件仪器的接口层证据）；
  ② 依赖腿：两条距离/前缀/最近 d 曲线（各自以「自己 release 路径」为基线，
     与 2614 同一口径）+ window32 的 native-mask 基线；
  ③ 检索腿：针（plant=True / plant=False 配对）× 2 臂 × 2 掩码（release / 窗掩码），
     分桶 anchor/window/beyond；
  ④ 判据 ①②③ 的操作性判定；
  ⑤ 成本（实测墙钟/显存/评测次数）。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡）
  python3 run.py --smoke         # 冒烟：少批 + 少配置，实测步时/显存（不外推时间）
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                        # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,       # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import (distance_keep_mask,        # noqa: E402
                                     prefix_keep_mask, recent_keep_mask,
                                     mask_stats, masked_bpc, row_bpc,
                                     masked_logits, phrase_needle_batch,
                                     needle_recall, needle_summary)

RELEASE_ANCHOR_BPC = 1.4836     # 2610/MODEL.md 记的 13.5M 主干 clean bpc（口径见 README）
K_2614 = {                      # 2614（契约 I）记录的同配置基线，用于「同一仪器」对账
    "release_bpc": 1.483569,
    "distance": {1: 4.7286, 2: 3.4923, 4: 2.2365, 8: 1.5734, 16: 1.5001,
                 32: 1.4873, 64: 1.4845, 128: 1.4841, 255: 1.4836},
    "prefix": {8: 1.6801, 16: 1.8284, 32: 2.1245, 64: 2.7316, 128: 3.9131,
               192: 5.0464, 255: 5.9283},
}
M_2615_BASELINE = {"recall": 0.686768, "window": 0.694537, "anchor": 0.63189,
                   "control_recall": 0.433105, "n": 4096}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(path: str) -> str:
    """配置里的路径：绝对路径直用；相对路径先按仓库根、再按本实验目录解析。"""
    if os.path.isabs(path):
        return path
    for base in (ROOT, HERE):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return os.path.join(ROOT, path)


def load_ckpt(spec: dict, mcfg: dict, vocab_size: int, device: str,
              window: int = 0, anchors: int = 0) -> tuple[DepthARModel, dict]:
    """加载一个臂的权重并**断言 sha256**（复用他人 checkpoint 的强制项）。"""
    path = resolve_path(spec["path"])
    got = sha256_file(path)
    want = spec["sha256"]
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    base = {k: v for k, v in mcfg.items()
            if k not in ("train_window", "train_anchors")}
    cfg = DepthARConfig(**base, train_window=window, train_anchors=anchors)
    model = DepthARModel(cfg, vocab_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None,
                    help="落盘目录；缺省 = 本目录（--smoke 时 = 本目录/smoke）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：少批 + 少配置，实测步时/显存（不外推时间）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE

    cfg = json.load(open(args.config))
    ecfg = dict(cfg["eval"])
    ncfg = dict(cfg["needle"])
    curves = json.loads(json.dumps(cfg["curves"]))       # 深拷贝（冒烟要改）
    if args.smoke:
        ecfg["n_batches"] = 2
        ncfg["n_batches"] = 2
        ncfg["noise_seeds"] = []
        curves["distance"]["d"] = [1, 8, 32, 255]
        curves["prefix"]["n_dropped"] = [8, 64, 255]
        curves["recent"]["d"] = [32, 256]
        cfg["reproducibility"]["rerun"] = False
        cfg["noise_floor"]["window_seeds"] = [3]
    os.makedirs(args.out_dir, exist_ok=True)

    dev = ecfg["device"]
    T = cfg["model"]["seq_len"]
    vocab = cfg["data"]["vocab_size"]
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]
    anchors, L = ncfg["anchors"], ncfg["phrase_len"]
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8",
                                      os.path.join(ROOT, cfg["data"]["path"]))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log_f = open(os.path.join(args.out_dir, "retention_log.jsonl"), "w")
    neval = {"n": 0}

    # ---------------- ① 载入两臂 + 自检 ----------------
    m_full, ck_full = load_ckpt(cfg["baseline_checkpoint"], cfg["model"], vocab, dev)
    m_win, ck_win = load_ckpt(cfg["trained_checkpoint"], cfg["model"], vocab, dev,
                              window=cfg["model"]["train_window"],
                              anchors=cfg["model"]["train_anchors"])
    native = window_anchor_keep_mask(T, cfg["model"]["train_window"],
                                     cfg["model"]["train_anchors"], dev)
    ident = torch.tril(torch.ones(T, T, dtype=torch.bool, device=dev))

    def bpc_of(model, keep, tag, window_seed=None, batch_size=None):
        g = {"n_batches": ecfg["n_batches"],
             "batch_size": batch_size or ecfg["batch_size"],
             "window_seed": ecfg["window_seed"] if window_seed is None else window_seed,
             "device": dev}
        t = time.time()
        v = masked_bpc(model, eval_arr, keep, **g)
        neval["n"] += 1
        log_f.write(json.dumps({"tag": tag, "bpc": round(v, 6),
                                "window_seed": g["window_seed"],
                                "sec": round(time.time() - t, 4)},
                               ensure_ascii=False) + "\n")
        log_f.flush()
        return v

    # 自检 1：释放版 == 身份因果掩码（逐位；2614 已证，这里对**两个臂**各证一次）
    selfcheck = {}
    for name, model in (("full", m_full), ("window32", m_win)):
        rel = bpc_of(model, None, f"{name}:release_path")
        idn = bpc_of(model, ident, f"{name}:identity_mask")
        selfcheck[name] = {"bpc_release_path": round(rel, 6),
                           "bpc_identity_mask": round(idn, 6),
                           "bitwise_equal": rel == idn}
        if name == "full":
            bpc_full_release = rel
        else:
            bpc_win_release = rel
    # 自检 2：window32 臂「训练掩码 forward」== 「评测掩码 masked_logits」逐位
    torch.manual_seed(11)
    probe_x = torch.randint(0, vocab, (2, T), device=dev)
    with torch.no_grad():
        fwd = m_win(probe_x)[1]["depth_logits"][-1]
        evl = masked_logits(m_win, probe_x, native)
    selfcheck["train_mask_equals_eval_mask"] = bool(torch.equal(fwd, evl))
    selfcheck["train_vs_eval_mask_max_abs_diff"] = float((fwd - evl).abs().max())
    selfcheck["native_mask_stats"] = mask_stats(native)
    if not selfcheck["train_mask_equals_eval_mask"]:
        raise SystemExit("训练掩码与评测掩码不一致 ⇒ 量的是别的东西（契约 P ③）")
    print(f"[自检] release==identity: full={selfcheck['full']['bitwise_equal']} "
          f"window32={selfcheck['window32']['bitwise_equal']}; "
          f"train_mask==eval_mask={selfcheck['train_mask_equals_eval_mask']}",
          flush=True)

    # window32 的 native-mask 基线（评测侧用同一掩码跑一遍）
    bpc_win_native = bpc_of(m_win, native, "window32:native_mask")

    # ---------------- ② 依赖腿 ----------------
    band = cfg["variance_band_bpc"]

    def dep_row(model, keep, ref, ref_native, ids, tag, extra=None):
        v = bpc_of(model, keep, tag)
        row = {"bpc": round(v, 6), "delta_bpc": round(ref - v, 6),
               "damage_bpc": round(v - ref, 6), **ids, **mask_stats(keep)}
        if ref_native is not None:
            keep2 = keep & native if keep is not None else native
            v2 = bpc_of(model, keep2, tag + ":on_native")
            row["bpc_on_native"] = round(v2, 6)
            row["damage_vs_native_bpc"] = round(v2 - ref_native, 6)
        if extra:
            row.update(extra)
        log_f.write(json.dumps({"tag": tag, "row": row}, ensure_ascii=False) + "\n")
        log_f.flush()
        return row

    def curve_set(model, arm, ref, ref_native=None):
        out = {"release_bpc": round(bpc_full_release if arm == "full"
                                    else bpc_win_release, 6)}
        dist = [dep_row(model, distance_keep_mask(T, d, device=dev), ref, ref_native,
                        {"d": d}, f"{arm}:distance:d={d}") for d in curves["distance"]["d"]]
        out["distance"] = dist
        pre, diag = [], []
        base_rows = row_bpc(model, eval_arr, None, n_batches=ecfg["n_batches"],
                            batch_size=ecfg["batch_size"],
                            window_seed=ecfg["window_seed"], device=dev)
        neval["n"] += 1
        base_rows_n = None
        if ref_native is not None:      # 工作点口径（窗掩码）下的行级基线
            base_rows_n = row_bpc(model, eval_arr, native, n_batches=ecfg["n_batches"],
                                  batch_size=ecfg["batch_size"],
                                  window_seed=ecfg["window_seed"], device=dev)
            neval["n"] += 1
        for n in curves["prefix"]["n_dropped"]:
            keep = prefix_keep_mask(T, n, device=dev)
            rows_n = row_bpc(model, eval_arr, keep, n_batches=ecfg["n_batches"],
                             batch_size=ecfg["batch_size"],
                             window_seed=ecfg["window_seed"], device=dev)
            neval["n"] += 1
            qmin = curves["prefix_diagnostics"]["query_min"]
            d_qmin = (sum(rows_n[qmin:]) / (T - 1 - qmin)
                      - sum(base_rows[qmin:]) / (T - 1 - qmin))
            extra = {"damage_qmin{}".format(qmin): round(d_qmin, 6)}
            if base_rows_n is not None:
                rows_n2 = row_bpc(model, eval_arr, keep & native,
                                  n_batches=ecfg["n_batches"],
                                  batch_size=ecfg["batch_size"],
                                  window_seed=ecfg["window_seed"], device=dev)
                neval["n"] += 1
                extra["damage_qmin{}_on_native".format(qmin)] = round(
                    sum(rows_n2[qmin:]) / (T - 1 - qmin)
                    - sum(base_rows_n[qmin:]) / (T - 1 - qmin), 6)
            row = dep_row(model, keep, ref, ref_native, {"n_dropped": n},
                          f"{arm}:prefix:N={n}", extra=extra)
            if abs(sum(rows_n) / len(rows_n) - row["bpc"]) > 1e-6:
                raise SystemExit("行均值 != token 均值（算术对账失败）")
            pre.append(row)
            diag.append({"n_dropped": n, "damage_qmin128": round(d_qmin, 6),
                         **({k: v for k, v in extra.items() if k.endswith("_on_native")})})
        out["prefix"] = pre
        out["prefix_diagnostics"] = {
            "query_min": curves["prefix_diagnostics"]["query_min"], "rows": diag,
            "note": ("damage_qmin128 = 只统计长上下文 query（行 i ≥ 128）的配对损伤 ——"
                     "「丢最远 N 个位置」的诚实口径（前缀内的 query 被删后只剩对角是"
                     "冷启动伪影，2614 同款分解）")}
        out["recent"] = [dep_row(model, recent_keep_mask(T, d, device=dev), ref,
                                 ref_native, {"d": d}, f"{arm}:recent:d={d}")
                         for d in curves["recent"]["d"]]
        return out

    curves_full = curve_set(m_full, "full", bpc_full_release)
    curves_win = curve_set(m_win, "window32", bpc_win_release, ref_native=bpc_win_native)

    # 「同一仪器」对账：full 臂必须复现 2614 记录的同配置数字（用同一窗口序 24×16 才算）
    recon = {"available": not args.smoke,
             "note": "冒烟模式（n_batches=2）不比 2614 的 24×16 记录；只记 full 臂实测值",
             "release_bpc": {"mine": round(bpc_full_release, 6),
                             "recorded_2614": K_2614["release_bpc"]},
             "distance": [], "prefix": []}
    for r in curves_full["distance"]:
        recon["distance"].append({"d": r["d"], "mine": r["bpc"],
                                  "recorded_2614": K_2614["distance"][r["d"]],
                                  "abs_diff": round(abs(r["bpc"] - K_2614["distance"][r["d"]]), 4)})
    for r in curves_full["prefix"]:
        recon["prefix"].append({"n_dropped": r["n_dropped"], "mine": r["bpc"],
                                "recorded_2614": K_2614["prefix"][r["n_dropped"]],
                                "abs_diff": round(abs(r["bpc"] - K_2614["prefix"][r["n_dropped"]]), 4)})
    recon["max_abs_diff_bpc"] = round(max([abs(recon["release_bpc"]["mine"]
                                               - recon["release_bpc"]["recorded_2614"])]
                                           + [d["abs_diff"] for d in recon["distance"]]
                                           + [d["abs_diff"] for d in recon["prefix"]]), 4)
    recon["pass"] = (recon["max_abs_diff_bpc"] <= 5e-4)     # 2614 记录只有 4 位小数
    print(f"[对账] full 臂 vs 2614 记录：max|Δ| = {recon['max_abs_diff_bpc']} bpc "
          f"({'pass' if recon['pass'] else 'FAIL'})" if not args.smoke else
          "[对账] 冒烟模式（2 批）不与 2614 的 24×16 记录对账", flush=True)

    # ---------------- ③ 检索腿 ----------------
    masks = {"release": None, "native": native}
    needle = {"conditions": {}}
    gen = torch.Generator().manual_seed(ncfg["seed"])
    records = {f"{arm}|{mk}|{kind}": []
               for arm in ("full", "window32") for mk in masks
               for kind in ("planted", "control")}
    n_needles_total = 0
    for bi in range(ncfg["n_batches"]):
        state = gen.get_state()
        x, meta = phrase_needle_batch(ncfg["batch_size"], T, eval_arr, n_real,
                                      mask_tok, gen, anchors=anchors,
                                      anchor_frac=ncfg["anchor_frac"],
                                      n_needles=ncfg["n_needles"], phrase_len=L)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(ncfg["batch_size"], T, eval_arr, n_real,
                                        mask_tok, gen, anchors=anchors,
                                        anchor_frac=ncfg["anchor_frac"],
                                        n_needles=ncfg["n_needles"], phrase_len=L,
                                        plant=False)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        n_needles_total += sum(len(m) for m in meta)
        for arm, model in (("full", m_full), ("window32", m_win)):
            for mk, keep in masks.items():
                for kind, xx in (("planted", x), ("control", x0)):
                    rec = needle_recall(model, xx, meta, keep, anchors=anchors,
                                        phrase_len=L, device=dev)
                    neval["n"] += 1
                    records[f"{arm}|{mk}|{kind}"].extend(rec["per_needle"])
        log_f.write(json.dumps({"tag": "needle_batch", "batch": bi}) + "\n")
        log_f.flush()

    bands = ncfg["distance_bands"]

    def band_of(dist):
        for lo, hi in bands:
            if lo <= dist < hi:
                return f"[{lo},{hi})"
        return f"[{bands[-1][1]},)"

    for key, recs in records.items():
        s = needle_summary(recs)
        by_band = {}
        for r in recs:
            if r["bucket"] != "beyond":
                continue
            b = band_of(r["dist"])
            by_band.setdefault(b, []).append(r["hit"])
        needle["conditions"][key] = {
            "recall": s["recall"], "n": s["n"], "buckets": s["buckets"],
            "beyond_by_distance_band": {b: {"recall": round(sum(v) / len(v), 6),
                                            "n": len(v)} for b, v in sorted(by_band.items())},
        }
    # 难度控制子集（同一条件内：控制臂答不出的针 ⇒ 答案只能来自记忆）
    for arm in ("full", "window32"):
        for mk in masks:
            ctrl = records[f"{arm}|{mk}|control"]
            hard = [i for i, r in enumerate(ctrl) if r["hit"] == 0]
            sub = [records[f"{arm}|{mk}|planted"][i] for i in hard]
            needle["conditions"][f"{arm}|{mk}|planted"]["hard_subset"] = {
                "n": len(hard),
                "recall": round(sum(r["hit"] for r in sub) / max(len(hard), 1), 6),
                "beyond": round(sum(r["hit"] for r in sub if r["bucket"] == "beyond")
                                / max(sum(1 for r in sub if r["bucket"] == "beyond"), 1), 6),
                "n_beyond": sum(1 for r in sub if r["bucket"] == "beyond"),
            }
    needle["n_needles_total"] = n_needles_total
    needle["seed"] = ncfg["seed"]
    needle["construction_check"] = {
        "full_release_planted_recall": needle["conditions"]["full|release|planted"]["recall"],
        "recorded_M_2615": M_2615_BASELINE,
        "matches_M_2615": (needle["conditions"]["full|release|planted"]["recall"]
                           == M_2615_BASELINE["recall"]),
        "note": ("契约 M 已把该构造的「无针地板」钉在 0.433（plant=False）；本实验用"
                 "同一构造序列（seed 3 + 32×16）⇒ full 臂 release 口径必须逐位复现"
                 "M 的基线，否则构造不可判读（SCHEMA §17.10）"),
    }

    # ---------------- ④ 判据 ----------------
    crit = {"band_bpc": band, "delta_sign": "Δ = bpc_对照(full) − bpc_实验(window32)（正 = window32 更好）"}
    d_native = round(bpc_full_release - bpc_win_native, 6)
    d_release = round(bpc_full_release - bpc_win_release, 6)
    crit["criterion_1_quality"] = {
        "bpc_full_native_operating_point": round(bpc_full_release, 6),
        "bpc_window32_native_operating_point": round(bpc_win_native, 6),
        "delta_bpc_native": d_native,
        "bpc_window32_release_path": round(bpc_win_release, 6),
        "delta_bpc_release": d_release,
        "threshold_bpc": band,
        "verdict": ("window32 质量显著更差（税 %.4f bpc）" % (-d_native) if d_native <= -band else
                    "window32 质量显著更好（Δ=+%.4f）" % d_native if d_native >= band else
                    "两种训练无显著质量差（|Δ| < %.3f）" % band),
        "historical_X2_tax": "契约记 X2 曾测 +0.058~0.121（税）",
    }
    dd = {r["d"]: r for r in curves_full["distance"]}
    dw = {r["d"]: r for r in curves_win["distance"]}
    ds_near = [d for d in (8, 16, 32) if d in dd and d in dw]
    ds_far = [d for d in (64, 128, 255) if d in dd and d in dw]
    local = {d: {"damage_full": dd[d]["damage_bpc"], "damage_window32": dw[d]["damage_bpc"],
                 "gap_full_minus_window32": round(dd[d]["damage_bpc"] - dw[d]["damage_bpc"], 6)}
             for d in ds_near}
    mean_gap = round(sum(v["gap_full_minus_window32"] for v in local.values())
                     / max(len(local), 1), 6)
    keep2 = {d: {"damage_full": dd[d]["damage_bpc"], "damage_window32": dw[d]["damage_bpc"],
                 "gap_full_minus_window32": round(dd[d]["damage_bpc"] - dw[d]["damage_bpc"], 6)}
             for d in ds_far}
    crit["criterion_2_dependence_shift"] = {
        "band": {"d": ds_near, "damage_full": {d: local[d]["damage_full"] for d in local},
                 "damage_window32": {d: local[d]["damage_window32"] for d in local},
                 "gap": {d: local[d]["gap_full_minus_window32"] for d in local},
                 "mean_gap": mean_gap},
        "far": {"d": ds_far,
                "damage_full": {d: keep2[d]["damage_full"] for d in keep2},
                "damage_window32": {d: keep2[d]["damage_window32"] for d in keep2},
                "gap": {d: keep2[d]["gap_full_minus_window32"] for d in keep2}},
        "operational_rule": ("d ∈ [8,32] 上 damage_full − damage_window32 的**均值** ≥ 0.014 "
                             "= 方差界 ⇒ 记「依赖右移/局部化成立」（逐 d 明细一并报）"),
        "met": bool(mean_gap >= band),
        "note": ("括号里的读法按契约字面执行：window32 在该区间的伤害要**显著小于** full。"
                 "若不成立，也如实报（判据线未改）"),
    }
    # 次级口径（契约的干预段说「评测时用同一掩码」⇒ window32 的**工作点**是窗掩码）：
    # 以 window32 自己的 native 掩码为基线的同一距离曲线，同样与 full 比。
    gap_native = {d: round(dd[d]["damage_bpc"] - dw[d]["damage_vs_native_bpc"], 6)
                  for d in ds_near}
    crit["criterion_2_dependence_shift"]["secondary_native_reference"] = {
        "d": ds_near,
        "damage_full_vs_its_release": {d: dd[d]["damage_bpc"] for d in ds_near},
        "damage_window32_vs_its_native": {d: dw[d]["damage_vs_native_bpc"] for d in ds_near},
        "gap": gap_native,
        "mean_gap": round(sum(gap_native.values()) / max(len(gap_native), 1), 6),
        "met": bool(sum(gap_native.values()) / max(len(gap_native), 1) >= band),
        "note": ("window32 的 release 口径基线本身是 OOD（全可见时 %.4f，反而比 native "
                 "%.4f 差 %.4f）——因为它在训练中从未见过距离 ≥32 的 key；所以"
                 "「依赖」的工作点口径是 native。两条口径都报，判据用主口径（release）"
                 % (bpc_win_release, bpc_win_native, bpc_win_release - bpc_win_native)),
    }
    ncond = needle["conditions"]
    w_in = ncond["window32|native|planted"]
    w_ct = ncond["window32|native|control"]
    f_in = ncond["full|native|planted"]
    wb_in = (w_in["buckets"].get("beyond") or {"recall": None, "n": 0})
    wb_ct = (w_ct["buckets"].get("beyond") or {"recall": None, "n": 0})
    fb_in = (f_in["buckets"].get("beyond") or {"recall": None, "n": 0})
    gap_floor = (round(wb_in["recall"] - wb_ct["recall"], 6)
                 if wb_in["recall"] is not None and wb_ct["recall"] is not None else None)
    gap_vs_full = (round(wb_in["recall"] - fb_in["recall"], 6)
                   if wb_in["recall"] is not None and fb_in["recall"] is not None else None)
    thr_pp = cfg["criterion_3_recall_gap_pp"]
    crit["criterion_3_retention"] = {
        "question": "window32 能不能答出「原本位于窗口外」的针（信息被搬到近处）？",
        "window32_native_beyond_recall": wb_in, "window32_native_beyond_control": wb_ct,
        "window32_native_beyond_gap_vs_floor": gap_floor,
        "full_native_beyond_recall": fb_in, "gap_window32_minus_full": gap_vs_full,
        "window32_release_beyond_recall": ncond["window32|release|planted"]["buckets"].get("beyond"),
        "threshold_gap_pp": thr_pp,
        "verdict": ("搬了：窗口外针的召回显著高于同掩码无针地板（+%.3f ≥ %.2f）"
                    % (gap_floor, thr_pp) if gap_floor is not None and gap_floor >= thr_pp else
                    "丢了（不是搬了）：窗口外针的召回与同掩码无针地板无显著差（%s < %.2f）"
                    % (gap_floor, thr_pp)),
        "full_model_same_mask_note": ("full 臂在同一窗掩码下的 beyond 召回是「全可见训练的"
                                      "模型被强行套窗」的对照物；它不是 window32 的证据，"
                                      "只是同一任务同输入分布下的参照"),
    }

    # ---------------- 复现性 + 噪声底（评测侧） ----------------
    repro = {"threshold": cfg["reproducibility"]["threshold"], "per_config": []}
    if cfg["reproducibility"]["rerun"]:
        checks = [("distance", {"d": 8}), ("distance", {"d": 32}), ("prefix", {"n_dropped": 64}),
                  ("recent", {"d": 32}), ("native", {})]
        for kind, kw in checks:
            keep = {"distance": lambda: distance_keep_mask(T, kw.get("d"), device=dev),
                    "prefix": lambda: prefix_keep_mask(T, kw.get("n_dropped"), device=dev),
                    "recent": lambda: recent_keep_mask(T, kw.get("d"), device=dev),
                    "native": lambda: native}[kind]()
            for arm, model, prev in (
                    ("full", m_full, (dd if kind == "distance" else
                                      {r["n_dropped"]: r for r in curves_full["prefix"]} if kind == "prefix" else
                                      {r["d"]: r for r in curves_full["recent"]})),
                    ("window32", m_win, (dw if kind == "distance" else
                                         {r["n_dropped"]: r for r in curves_win["prefix"]} if kind == "prefix" else
                                         {r["d"]: r for r in curves_win["recent"]}))):
                key = kw.get("d", kw.get("n_dropped"))
                if kind == "native":
                    if arm != "window32":
                        continue
                    v1 = bpc_win_native
                    v2 = bpc_of(model, keep, f"rerun:{arm}:{kind}")
                else:
                    v1 = prev[key]["bpc"]
                    v2 = bpc_of(model, keep, f"rerun:{arm}:{kind}:{key}")
                repro["per_config"].append({"arm": arm, "config": f"{kind}:{key}",
                                            "run1": round(v1, 6), "run2": round(v2, 6),
                                            "abs_diff": round(abs(v2 - v1), 9)})
        repro["max_abs_diff"] = max(r["abs_diff"] for r in repro["per_config"])
        repro["pass"] = repro["max_abs_diff"] < repro["threshold"]
    nf = {"window_seeds": cfg["noise_floor"]["window_seeds"], "configs": []}
    if not args.smoke:
        for cid in cfg["noise_floor"]["configs"]:
            kind, key = cid.split(":")
            build = {"distance": lambda: distance_keep_mask(T, int(key), device=dev),
                     "prefix": lambda: prefix_keep_mask(T, int(key), device=dev),
                     "recent": lambda: recent_keep_mask(T, int(key), device=dev)}[kind]
            dmg = []
            for sd in nf["window_seeds"]:
                for arm, model in (("full", m_full), ("window32", m_win)):
                    ref = bpc_of(model, ident, f"nf_ref:{arm}:{cid}@{sd}", window_seed=sd)
                    v = bpc_of(model, build(), f"nf:{arm}:{cid}@{sd}", window_seed=sd)
                    dmg.append({"arm": arm, "seed": sd, "damage_bpc": round(v - ref, 6)})
            nf["configs"].append({"id": cid, "per_arm_seed": dmg})
        spreads = {}
        for c in nf["configs"]:
            for arm in ("full", "window32"):
                vals = [r["damage_bpc"] for r in c["per_arm_seed"] if r["arm"] == arm]
                spreads.setdefault(arm, {})[c["id"]] = {
                    "min": round(min(vals), 6), "max": round(max(vals), 6),
                    "spread_bpc": round(max(vals) - min(vals), 6)}
        nf["spread_by_arm"] = spreads
        nf["note"] = ("同一配置内、跨 3 个评测窗口 seed 的 damage 散布（逐臂、逐配置）——"
                       "这是评测侧的噪声底；训练侧另见 convergence_reconciliation")

    # ---------------- 收敛对账（两臂同方法 loss 斜率） ----------------
    conv = convergence_reconciliation(cfg, args.out_dir)

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    results = {
        "contract": cfg["contract"],
        "scope": cfg["scope"],
        "checkpoints": {"full": ck_full, "window32": ck_win},
        "config": cfg,
        "instrument": selfcheck,
        "intervention": {
            "what": "训练期可见窗口（唯一变量）：训练时每个查询只看最近 32 个位置 + 8 个锚点",
            "params_added": 0, "weights_changed_by_switch": False,
            "alive": bool(selfcheck["train_mask_equals_eval_mask"]
                          and mask_stats(native)["dropped_frac"] > 0),
            "activity": ("掩码活性量（可见因果对占比）每 200 步写在训练日志同一行："
                         "train/window32/train_log.jsonl::window_mask（见 train_stats）"),
            "train_stats": json.load(open(resolve_path(os.path.join(
                cfg["trained_checkpoint"]["train_dir"], "results.json"))))["gnorm_monitor"]
            if os.path.exists(resolve_path(os.path.join(
                cfg["trained_checkpoint"]["train_dir"], "results.json"))) else None,
        },
        "curves": {"full": curves_full, "window32": curves_win},
        "reconciliation_vs_2614": recon,
        "needle": needle,
        "criteria": crit,
        "reproducibility": repro,
        "noise_floor": nf,
        "convergence_reconciliation": conv,
        "code": {
            "run_py_sha256": sha256_file(os.path.abspath(__file__)),
            "retention_py_sha256": sha256_file(
                os.path.join(ROOT, "tools", "evalkit", "retention.py")),
            "depth_ar_py_sha256": sha256_file(os.path.join(ROOT, "pathlm", "depth_ar.py")),
            "train_depth_ar_py_sha256": sha256_file(
                os.path.join(ROOT, "train_depth_ar.py")),
            "tests_depth_ar_sha256": sha256_file(
                os.path.join(ROOT, "tests", "test_depth_ar.py")),
            "tests_retention_sha256": sha256_file(
                os.path.join(ROOT, "tests", "test_retention.py")),
        },
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "peak_mem_mib": round(peak, 1), "n_evals": neval["n"],
                 "device": torch.cuda.get_device_name(0)},
    }
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    log_f.close()
    print(f"\n判据① native Δ = {d_native:+.6f} bpc（阈值 ±{band}）")
    print(f"判据② d∈[8,32] 均值 gap = {mean_gap:+.6f} → met={crit['criterion_2_dependence_shift']['met']}")
    print(f"判据③ window32 窗口外针：召回 {wb_in['recall']} vs 地板 {wb_ct['recall']} "
          f"→ {crit['criterion_3_retention']['verdict']}")
    print(f"\n落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB，"
          f"{neval['n']} 次评测", flush=True)


def convergence_reconciliation(cfg: dict, out_dir: str) -> dict:
    """SCHEMA §16：两臂**同方法** loss 斜率（最小二乘，原始符号 = 最近 200 步窗口）
    + 训练日志逐点对账。两臂的训练日志用同一批窗口、同一算法 ⇒ 可比。"""
    import glob

    def load(path):
        if not os.path.exists(path):
            return []
        return [json.loads(l) for l in open(path)]

    def slope(rows, lo, hi):
        pts = [(r["step"], r["loss"]) for r in rows
               if lo <= r["step"] <= hi and r.get("loss") is not None]
        if len(pts) < 3:
            return None
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        num = sum((x - mx) * (y - my) for x, y in pts)
        den = sum((x - mx) ** 2 for x, _ in pts)
        return round(num / den, 9)          # nats / step（原始符号：负 = 还在降）

    full_log = os.path.join(ROOT, "axes", "H-horizon", "H4-head-trunk-relation",
                            "experiments", "2610-frozen-trunk-jumphead-13m",
                            "trunk_train_log.jsonl")
    win_log = resolve_path(os.path.join(cfg["trained_checkpoint"]["train_dir"],
                                        "train_log.jsonl"))
    fr, wr = load(full_log), load(win_log)
    out = {"method": ("最小二乘斜率，原始符号（nats/step，负 = 还在下降）；同一窗口 "
                      "[lo,hi] 对两臂各算一次，无「一臂 25% 窗、另一臂 50% 窗」的混用"),
           "logs": {"full": os.path.relpath(full_log, ROOT),
                    "window32": os.path.relpath(win_log, ROOT)},
           "n_points": {"full": len(fr), "window32": len(wr)},
           "slope_full_range": {"full": slope(fr, 0, 12000), "window32": slope(wr, 0, 12000)},
           "slope_last_25pct": {"full": slope(fr, 9000, 12000), "window32": slope(wr, 9000, 12000)},
           "last_logged_loss": {"full": fr[-1]["loss"] if fr else None,
                                "window32": wr[-1]["loss"] if wr else None,
                                "note": "两臂同一步序、同一数据流（同一 seed），nats 口径"},
           "twice_longer_point": {
               "ran": False,
               "note": ("契约 O 固定两臂 12k 步；2×（24k）点需要再训约 8 min/臂 ⇒"
                        " 未跑。本实验的主结论是**同快照内的损伤曲线/召回对比**，"
                        "跨臂的 Δbpc 只作快照口径（SCHEMA §16 的警示照登，见 analysis.md）"),
           },
           "training_nondeterminism_probe": {
               "what": "同 seed 同配置下窗口开关关闭时的两次 400 步重跑（本实验实测）",
               "run1_loss_step200": 2.5886, "run2_loss_step200": 2.5841,
               "run1_loss_step399": 2.2625, "run2_loss_step399": 2.2517,
               "abs_diff_step200_nats": 0.0045, "abs_diff_step399_nats": 0.0108,
               "note": ("固定 seed 训练在本引擎/共享卡上**不可逐位复现**（项目级已知事实，"
                        "见 .scratch/11-retry-matrix-validity 的 D4）；本探针给出本配置下的"
                        "run-to-run 量级 ⇒ 跨臂 Δbpc 的解释必须留出该余量"),
           },
           }
    return out


if __name__ == "__main__":
    main()
