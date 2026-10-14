#!/usr/bin/env python3
"""2639-carry-perlayer-frontier-13m · 评测入口（纯推理）：完整机制（逐层状态 + 内容写门 ± γ）
的质量-代价前沿。

契约：同目录 `README.md`（§4 判据线预登记）。手段：

  ① 每个臂在其**训练期可见性 / 其训练时配置**下评测 clean bpc —— 窗口臂 / carry 臂用
     `window_anchor_keep_mask(T, window, 0)`；full 锚用 `keep_mask=None`（全因果）。
     全部臂走**同一份** 96×16 窗口序（`window_seed=2`）⇒ 臂间可比。
  ② 代价轴 = 每位置等效 KV 字节（fp16）= n_layers × 2 × d_model × 2 × k，
     carry 臂另加**其自己的**状态字节：单共享槽 8192 floats；逐层独立状态
     n_layers × 8192 floats（= 196,608 B）⇒ `k_eff`。
  ③ 前沿：按 (代价, bpc) 排序；主判据 = 完备机制点与其**等字节**的纯窗口点（`win32`）
     直接对撞（`pl_minus_win32 = bpc(win32) − bpc(carrypl)`）。
  ④ 活性（SCHEMA §10）：掩码可见对数断言 + 评测期掩码反事实 + **逐层独立状态探针**
     （monkeypatch `write_read` 抓 12 个层的 mem，必须两两不同）+ 训练日志活性量。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~6 min）
  python3 run.py --smoke    # 冒烟：少批
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full                       # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             window_anchor_keep_mask, window_mask_probe)
from tools.evalkit.retention import masked_bpc, masked_logits   # noqa: E402

# 本格自测带（C1 裁决：σ_cell 为主判带；跨格 0.014 为参考带）——常量，来自 2634 的
# 同配置换种子极差（win16/carry16），2638/2639 沿用。
SIGMA_CELL = 0.00166
CROSS_CELL_BAND = 0.014


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def model_kwargs(spec: dict, mcfg: dict) -> dict:
    kw = dict(mcfg)
    kw["train_window"] = int(spec["window"]) if spec["kind"] in ("window", "carry") else 0
    kw["train_anchors"] = int(spec.get("anchors", 0))
    kw["carry_rule"] = None if spec["rule"] in (None, "none") else spec["rule"]
    if kw["carry_rule"] is not None:
        # 每臂用自己的训练时实现细节（chunk）+ E1 完备度开关——**不改共享默认**。
        kw["carry_chunk"] = int(spec.get("carry_chunk", 64))
        kw["carry_per_layer"] = bool(spec.get("carry_per_layer", False))
        kw["carry_decay"] = float(spec.get("carry_decay", 1.0))
    return kw


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
    # 状态字节：单共享槽 = carry_heads × hd²；逐层 = n_layers × 上面那个。
    state_floats = 0
    if model.cfg.carry_rule is not None:
        per_layer_state = (model.carry.n_heads * model.carry.head_dim ** 2)
        state_floats = per_layer_state * (model.cfg.n_layers
                                          if model.cfg.carry_per_layer else 1)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "kind": spec["kind"], "window": spec["window"],
                   "anchors": spec.get("anchors", 0),
                   "rule": spec["rule"], "seed": spec["seed"],
                   "carry_chunk": spec.get("carry_chunk"),
                   "carry_per_layer": bool(spec.get("carry_per_layer", False)),
                   "carry_decay": float(spec.get("carry_decay", 1.0)),
                   "state_floats_actual": state_floats,
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


def keep_mask_for(spec: dict, T: int, dev: str):
    if spec["kind"] == "full":
        return None
    return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                   device=dev)


def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """**严格因果性探针（必须不变项）**：只改后半段 token，前半段 logits 必须逐位不变。

    形式：取 `x`（B,T）；构造 `x2 = x` 但把位置 ≥ T//2 换成另一组 token；两次前向的
    最后一层 logits 在位置 < T//2 上的最大绝对差必须 == 0（无未来泄露）。逐层状态是
    因果递推（位置 t 的读出只用 S_t，而 S_t 只含 ≤ t 的写入），故必须逐位不变。
    """
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            a = masked_logits(model, x, keep_mask).float()
            b = masked_logits(model, x2, keep_mask).float()
    finally:
        model.train(was_training)
    d = (a[:, :half] - b[:, :half]).abs().max()
    return {"perturbed_from": half,
            "max_abs_logit_diff_prefix": round(float(d), 9),
            "causal_no_future_leak": bool(float(d) == 0.0),
            "changed_suffix_max_abs_logit_diff": round(
                float((a[:, half:] - b[:, half:]).abs().max()), 6),
            "note": ("只改后半段 token ⇒ 前半段 logits 逐位不变 ⇔ 严格因果（无未来泄露）；"
                     "后半段必须真的变（否则探针没抓力）")}


def frontier(rows: list[dict]) -> dict:
    """在 (total_bytes, bpc) 上算支配关系与前沿包络（越小越好，两轴皆是）。"""
    pts = sorted(rows, key=lambda r: r["total_bytes"])
    dominated = {}
    for a in pts:
        dom_by = [b["name"] for b in pts
                  if b["total_bytes"] <= a["total_bytes"]
                  and b["bpc"] < a["bpc"] and b["name"] != a["name"]]
        dominated[a["name"]] = dom_by
    env, best = [], float("inf")
    for a in pts:
        if a["bpc"] < best:
            env.append(a["name"])
            best = a["bpc"]
    return {"sorted_by_bytes": [r["name"] for r in pts],
            "dominated_by": dominated, "lower_envelope": env}


def interp_window_at(rows: list[dict], k_eff: float) -> dict | None:
    """纯窗口臂 (k, bpc) 的线性插值（只用于定位，不外推；见 README §4 的凹性警告）。"""
    pts = sorted([r for r in rows if r["kind"] == "window" and r["seed"] == 0],
                 key=lambda r: r["k_eff"])
    for lo, hi in zip(pts, pts[1:]):
        if lo["k_eff"] <= k_eff <= hi["k_eff"]:
            w = (k_eff - lo["k_eff"]) / (hi["k_eff"] - lo["k_eff"])
            return {"between": [lo["name"], hi["name"]], "w": round(w, 4),
                    "bpc_interp": round(lo["bpc"] + w * (hi["bpc"] - lo["bpc"]), 6)}
    for p in pts:                      # 落在网格点上：插值 = 该点本身（非外推）
        if abs(p["k_eff"] - k_eff) < 1e-9:
            return {"between": [p["name"], p["name"]], "w": 0.0,
                    "bpc_interp": p["bpc"]}
    return None


def perlayer_state_probe(model, x: torch.Tensor) -> dict:
    """评测期**逐层独立状态**探针：monkeypatch `carry.write_read`，抓每层返回的 mem。

    证伪形式：若逐层开关没接线（或 12 层读同一个槽），则抓到的 mem 会逐位相同 /
    `layer_idx` 不会取满 0..L−1。返回：层号集合 + 两两不同的层对数 + 最大相对差。
    """
    chan = getattr(model, "carry", None)
    if chan is None or not getattr(model.cfg, "carry_per_layer", False):
        return {"applicable": False,
                "note": "该臂不是逐层模式（单共享槽 / 无通道）：逐层探针不适用"}
    seen: list[tuple[int, torch.Tensor]] = []
    orig = chan.write_read

    def spy(h0, rule=None, decay=None, layer_idx=None):
        mem, tele = orig(h0, rule=rule, decay=decay, layer_idx=layer_idx)
        seen.append((int(layer_idx), mem.detach().float().clone()))
        return mem, tele

    chan.write_read = spy              # type: ignore[assignment]
    try:
        with torch.no_grad():
            model(x)
    finally:
        chan.write_read = orig         # type: ignore[assignment]
    idx = sorted(i for i, _ in seen)
    diffs, n_pairs, n_diff = 0.0, 0, 0
    for a in range(len(seen)):
        for b in range(a + 1, len(seen)):
            d = float((seen[a][1] - seen[b][1]).abs().max())
            diffs = max(diffs, d)
            n_pairs += 1
            n_diff += int(d > 0)
    return {"applicable": True, "layer_idx_seen": idx,
            "n_layers_seen": len(seen),
            "layer_idx_covers_all": idx == list(range(len(model.blocks))),
            "n_mem_pairs": n_pairs, "n_pairs_bitwise_different": n_diff,
            "all_pairs_different": n_pairs > 0 and n_diff == n_pairs,
            "max_abs_mem_diff": round(diffs, 6),
            "note": "结论：层号取满全部层 且 抓到的 mem 两两不同 ⇒ 逐层状态真的独立"}


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
    if args.smoke:
        cfg["eval"]["n_batches"] = 2
        cfg["eval"]["batch_size"] = 8
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    eb = cfg["eval"]
    ca = cfg["cost_axis"]
    bpe = ca["bytes_per_elem"]
    unit = ca["kv_bytes_per_position_unit"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段做评测")
    _, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                          os.path.join(ROOT, "data/enwik8_full.npz"))
    assert vocab == cfg["data"]["vocab_size"]

    note("加载各臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:12]}… kind={spec['kind']} "
             f"window={spec['window']} rule={spec['rule']} "
             f"per_layer={ck[spec['name']]['carry_per_layer']} "
             f"chunk={ck[spec['name']]['carry_chunk']} "
             f"state_floats={ck[spec['name']]['state_floats_actual']} "
             f"params={ck[spec['name']]['n_params']}")

    selfchecks = {}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        km = keep_mask_for(spec, T, dev)
        s = int(km.sum().item())
        exp = sum(min(i + 1, int(spec["window"])) for i in range(T))
        causal = T * (T + 1) // 2
        selfchecks[spec["name"]] = {"visible_pairs": s, "expected": exp,
                                    "ok": s == exp, "nondegenerate": s < causal}
        assert s == exp, f"{spec['name']}: visible pairs {s} != {exp}"
        assert s < causal, f"{spec['name']}: mask is identity (visible==causal)"

    note(f"质量腿：bpc（{eb['n_batches']}×{eb['batch_size']}，window_seed={eb['window_seed']}）")
    rows = []
    for spec in cfg["arms"]:
        name = spec["name"]
        km = keep_mask_for(spec, T, dev)
        bpc = masked_bpc(arms[name], eval_arr, km, n_batches=eb["n_batches"],
                         batch_size=eb["batch_size"], window_seed=eb["window_seed"],
                         device=dev)
        k = int(spec["window"]) if spec["kind"] != "full" else T
        kv = cfg["model"]["n_layers"] * 2 * cfg["model"]["d_model"] * bpe * k
        state = ck[name]["state_floats_actual"] * bpe
        total = kv + state
        rows.append({"name": name, "kind": spec["kind"], "seed": spec["seed"],
                     "window": spec["window"], "rule": spec["rule"],
                     "carry_per_layer": ck[name]["carry_per_layer"],
                     "carry_decay": ck[name]["carry_decay"],
                     "carry_chunk": ck[name]["carry_chunk"],
                     "bpc": round(bpc, 6),
                     "kv_bytes": kv, "state_bytes": state, "total_bytes": total,
                     "k_eff": round(total / unit, 4),
                     "n_params": ck[name]["n_params"]})
        note(f"  {name}: bpc={bpc:.6f} k_eff={total/unit:.2f} "
             f"total={total} B (kv={kv} + state={state})")

    by = {r["name"]: r for r in rows}
    fr = frontier(rows)
    full = by["full"]
    bpc_win16_s0 = by["win16_s0"]["bpc"]
    gap_win16_full = bpc_win16_s0 - full["bpc"]

    # ---- 定位：每个 carry 臂相对同窗无通道 / 同 k_eff 纯窗口曲线 / 等字节窗口点 ----
    location = {}
    for r in rows:
        if r["kind"] != "carry":
            continue
        loc = interp_window_at(rows, r["k_eff"])
        rel = None if loc is None else round(loc["bpc_interp"] - r["bpc"], 6)
        # 等字节的纯窗口点（逐字节相等 ⇒ 直接点比，不用插值）
        same_bytes = [w["name"] for w in rows
                      if w["kind"] == "window" and w["total_bytes"] == r["total_bytes"]]
        entry = {
            "delta_bpc_vs_win16_s0": round(bpc_win16_s0 - r["bpc"], 6),
            "recovery_ratio_r": (round((bpc_win16_s0 - r["bpc"]) / gap_win16_full, 4)
                                 if gap_win16_full else None),
            "window_curve_at_k_eff": loc,
            "carry_minus_interp_window": rel,
            "same_bytes_window_arms": same_bytes,
        }
        for nm in same_bytes:
            entry[f"carry_minus_{nm}"] = round(by[nm]["bpc"] - r["bpc"], 6)
        location[r["name"]] = entry

    # ---- 主判据读数（等字节点比 + gap-closing + 机制税）----
    win32 = by.get("win32_s0")
    pl = by.get("carrypl16_s0")
    key_readings = {}
    if pl is not None and win32 is not None:
        same = pl["total_bytes"] == win32["total_bytes"]
        d = round(win32["bpc"] - pl["bpc"], 6)
        key_readings["pl_minus_win32"] = {
            "bpc_win32_s0": win32["bpc"], "bpc_carrypl16_s0": pl["bpc"],
            "same_bytes": same, "bytes": pl["total_bytes"],
            "pl_minus_win32": d,
            "verdict": ("改善前沿（机制翻转）" if d > SIGMA_CELL else
                        ("被等预算纯窗口支配" if d < -SIGMA_CELL else "带内不可排序")),
            "band_note": (f"本格自测带 σ_cell={SIGMA_CELL}；跨格参考带 "
                          f"{CROSS_CELL_BAND}；|Δ| 与两带比："
                          f"|{abs(d)}|"),
            "sigma_cell": SIGMA_CELL, "cross_cell_band": CROSS_CELL_BAND,
        }
    for nm in ("carry16_s0", "c16c256_s0"):
        if nm in by and pl is not None:
            key_readings[f"pl_vs_{nm.replace('c16c256_s0','shared_c256')}"] = {
                "bpc_other": by[nm]["bpc"], "bpc_carrypl16_s0": pl["bpc"],
                "delta": round(by[nm]["bpc"] - pl["bpc"], 6),
                "note": "完备度（消掉退化实现）本身值多少 bpc：正 = 完备机制更好"}
    if "carry16_s0" in by and "c16c256_s0" in by:
        key_readings["chunk_sensitivity_shared"] = {
            "bpc_c16_c64": by["carry16_s0"]["bpc"],
            "bpc_c16_c256": by["c16c256_s0"]["bpc"],
            "delta": round(by["carry16_s0"]["bpc"] - by["c16c256_s0"]["bpc"], 6),
            "note": ("同配置（单共享槽 delta 窗口16 seed0）chunk 64 vs 256 的 clean bpc 差 "
                     "⇒ chunk 数值路径的量级（chunk 只改浮点累加顺序，见 README §2）"),
        }

    # ---- gap-closing ----
    def _seed_spread(a: str, b: str):
        return (round(abs(by[a]["bpc"] - by[b]["bpc"]), 6)
                if a in by and b in by else None)

    sigma_win16 = _seed_spread("win16_s0", "win16_s1")
    sigma_carry16 = _seed_spread("carry16_s0", "carry16_s1")

    # ---- 敏感性读数（**不构成判据**，执行前登记）----
    # 冻结口径（62-F2 / 2638）：total = kv + state（整块计）⇒ k_eff = total / 12288。
    # 另一口径（只报数）：状态的固定成本与 T 无关，若按 T=256 摊到每位置 ⇒
    # k_eff_amort = (kv + state/T) / 12288。两个口径指向不同问题（「同 T 下同内存 footprint」
    # vs「长上下文下摊平」）；本块**只作敏感性**，判据一律用冻结口径。
    sensitivity = {}
    for r in rows:
        if r["kind"] != "carry" or r["state_bytes"] == 0:
            continue
        k_amort = (r["kv_bytes"] + r["state_bytes"] / T) / unit
        loc = interp_window_at(rows, k_amort)
        sensitivity[r["name"]] = {
            "k_eff_amortized": round(k_amort, 4),
            "window_curve_at_k_eff_amortized": loc,
            "carry_minus_interp_window_amortized": (
                None if loc is None else round(loc["bpc_interp"] - r["bpc"], 6)),
            "note": ("状态是固定成本（与 T 无关）；本块把 T=256 的固定成本摊到每位置，是"
                     "**另一口径**，只作敏感性报数，不当判据（判据用冻结口径 "
                     "k_eff = 总字节/12288，见 README §4a）"),
        }

    # ---- 活性（SCHEMA §10）----
    g_probe = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, vocab, (2, T), generator=g_probe).to(dev)
    liveness = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"], "rule": spec["rule"],
               "carry_per_layer": ck[name]["carry_per_layer"],
               "carry_chunk": ck[name]["carry_chunk"]}
        if spec["kind"] == "full":
            row["note"] = "全因果锚：无掩码开关，活性量不适用"
        else:
            km = keep_mask_for(spec, T, dev)
            s = int(km.sum().item())
            row["mask"] = {"n_visible_pairs": s, "causal_pairs": T * (T + 1) // 2,
                           "visible_frac": round(s / (T * (T + 1) // 2), 6),
                           "alive": s < T * (T + 1) // 2}
        if spec["kind"] == "carry":
            row["perlayer_state_probe"] = perlayer_state_probe(arms[name], x_probe)
        # 严格因果性（objective_criteria 的「必须不变」项）：有掩码的臂都探
        if spec["kind"] != "full":
            row["causality"] = causality_probe(arms[name], x_probe,
                                                keep_mask_for(spec, T, dev))
        tdir = os.path.dirname(resolve(spec["path"]))
        tlog = os.path.join(tdir, "train_log.jsonl")
        tres = os.path.join(tdir, "results.json")
        if os.path.exists(tlog):
            lines = [json.loads(x) for x in open(tlog) if x.strip()]
            row["train_window_mask"] = lines[-1].get("window_mask")
            row["train_read_share"] = lines[-1].get("attn_carry")
        if os.path.exists(tres):
            tr = json.load(open(tres))
            if tr.get("carry_liveness"):
                row["carry_liveness_training"] = tr["carry_liveness"]
            if tr.get("carry_read_mass"):
                # 逐层模式的**逐层读出占比**（12 个不同的值 = 12 层状态真的各自被用）
                row["train_carry_read_mass"] = tr["carry_read_mass"]
            if tr.get("attn_carry_layer_stats"):
                row["train_attn_carry_layer_stats"] = tr["attn_carry_layer_stats"]
            row["train_wall_minutes"] = tr.get("wall_minutes")
            row["train_peak_gpu_mem_mb"] = tr.get("peak_gpu_mem_mb")
        liveness[name] = row
    wprobe = None
    for spec in cfg["arms"]:
        if spec["kind"] == "window" and int(spec["window"]) == 17:
            wprobe = {"arm": spec["name"],
                      **window_mask_probe(arms[spec["name"]], x_probe)}
            break
    liveness["_mask_probe"] = {
        "note": ("评测期反事实：同一批输入在『掩码开 / 掩码关』两次前向的最后一层 logits "
                 "最大绝对差（>0 ⇔ 掩码真的改了前向）"),
        "window17_masked_vs_open": wprobe}

    # ---- 鲁棒性：换一个评测窗口序（window_seed=7）重算关键臂 ----
    # 主判据是等字节点比（|Δ|=0.068 ≫ 两带），换窗口序只用于排除「窗口序运气」解释。
    key_arms = [s for s in cfg["arms"]
                if s["name"] in ("full", "win16_s0", "win32_s0", "carrypl16_s0")]
    alt_seed = int(ca.get("alt_window_seed", 7))
    robust = {"window_seed": alt_seed,
              "note": "与主评价（window_seed=2）同一 96×16 规模、只换窗口采样序",
              "rows": {}}
    for spec in key_arms:
        km = keep_mask_for(spec, T, dev)
        bpc_alt = masked_bpc(arms[spec["name"]], eval_arr, km, n_batches=eb["n_batches"],
                             batch_size=eb["batch_size"], window_seed=alt_seed, device=dev)
        robust["rows"][spec["name"]] = round(bpc_alt, 6)
        note(f"  [alt seed {alt_seed}] {spec['name']}: bpc={bpc_alt:.6f}")
    if "win32_s0" in robust["rows"] and "carrypl16_s0" in robust["rows"]:
        robust["pl_minus_win32_alt_seed"] = round(
            robust["rows"]["win32_s0"] - robust["rows"]["carrypl16_s0"], 6)
        robust["same_sign_as_main"] = bool(
            robust["pl_minus_win32_alt_seed"] * (
                key_readings.get("pl_minus_win32", {}).get("pl_minus_win32", 1.0)) > 0)

    # ---- 复用等值性（复用臂 bpc 与 2638 落盘值逐位比）----
    reuse_equiv = {"note": "复用臂（eval-only）的 clean bpc 必须与来源实验落盘值逐位相同"}
    ref_path = os.path.join(ROOT, "axes/1-time/1.1.4.1-explicit-memory/experiments/"
                                  "2638-carry-frontier-13m/results.json")
    if os.path.exists(ref_path):
        ref = {r["name"]: r["bpc"] for r in json.load(open(ref_path))["rows"]}
        for spec in cfg["arms"]:
            nm = spec["name"]
            if nm in ref and spec.get("reused_from"):
                reuse_equiv[nm] = {"bpc": by[nm]["bpc"], "bpc_2638": ref[nm],
                                   "equal": by[nm]["bpc"] == ref[nm],
                                   "reused_from": spec["reused_from"]}

    results = {
        "config": cfg,
        "checkpoints": ck,
        "selfchecks": selfchecks,
        "liveness": liveness,
        "rows": rows,
        "frontier": fr,
        "location": location,
        "key_readings": key_readings,
        "sensitivity": sensitivity,
        "robustness_alt_window_seed": robust,
        "gap_closing": {
            "bpc_win16_s0": bpc_win16_s0,
            "bpc_full": full["bpc"],
            "gap_win16_full": round(gap_win16_full, 6),
            "note": "gap = 窗口 16 相对全因果在 clean bpc 上的损失；recovery_ratio_r 见 location",
        },
        "sigma_within_config_2634": {
            "win16": (None if sigma_win16 is None else round(sigma_win16, 6)),
            "carry16": (None if sigma_carry16 is None else round(sigma_carry16, 6)),
            "note": "同配置换种子的极差（2 样本 ⇒ 描述量，不是方差上界）；C1 裁决的主判带",
        },
        "bands": {"sigma_cell": SIGMA_CELL, "cross_cell_band": CROSS_CELL_BAND,
                  "note": "本格自测带为主判、跨格 0.014 带为参考，两带并列报出、不混用（C1）"},
        "reuse_equivalence": reuse_equiv,
        "cost": {
            "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
            "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "n_eval_windows": eb["n_batches"] * eb["batch_size"],
            "tokens_scored": eb["n_batches"] * eb["batch_size"] * (T - 1),
        },
        "code_hashes": {
            rel: sha256_file(os.path.join(ROOT, rel)) for rel in
            ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/retention.py",
             "pathlm/data.py") if os.path.exists(os.path.join(ROOT, rel))},
    }
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "config.json", "train_pl.py", "train_arms.sh")
        if os.path.exists(os.path.join(HERE, rel))})
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"落盘：{args.out_dir}/results.json")
    note(f"前沿包络 = {fr['lower_envelope']}；被支配 = "
         f"{ {k: v for k, v in fr['dominated_by'].items() if v} }")
    for c, v in location.items():
        rel = v["carry_minus_interp_window"]
        rel_s = "n/a" if rel is None else f"{rel:+.6f}"
        extra = {k: v2 for k, v2 in v.items() if k.startswith("carry_minus_")
                 and k != "carry_minus_interp_window"}
        note(f"  {c}: Δbpc={v['delta_bpc_vs_win16_s0']:+.6f} r={v['recovery_ratio_r']} "
             f"vs 窗口插值={rel_s} {extra}")
    for k, v in key_readings.items():
        note(f"  [{k}] {json.dumps(v, ensure_ascii=False)[:260]}")


if __name__ == "__main__":
    main()
