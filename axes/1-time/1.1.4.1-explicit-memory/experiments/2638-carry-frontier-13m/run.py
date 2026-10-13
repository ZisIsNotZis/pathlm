#!/usr/bin/env python3
"""2638-carry-frontier-13m · 评测入口（纯推理）：把显式记忆状态放进质量-代价前沿。

契约：同目录 `README.md`（§4 判据线预登记）。手段：

  ① 每个臂在其**训练期可见性**下评测 clean bpc —— 窗口臂 / carry 臂用
     `window_anchor_keep_mask(T, window, 0)`；full 锚用 `keep_mask=None`（全因果）。
     全部臂走**同一份** 96×16 窗口序（`window_seed=2`）⇒ 臂间可比。
  ② 代价轴 = 每位置等效 KV 字节（fp16）= n_layers × 2 × d_model × 2 × k，
     carry 臂另加 8192 floats × 2（单共享槽）。
  ③ 前沿：按 (代价, bpc) 排序，判定显式状态点是否被同代价的纯窗口点支配；
     报 gap-closing 比例 r = Δc / Δg（Δc = carry 相对同窗无通道的 bpc 增益，
     Δg = full 相对同窗无通道的 gap）。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~4 min）
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
from tools.evalkit.retention import masked_bpc                 # noqa: E402


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
    kw["train_window"] = int(spec["window"]) if spec["kind"] == "window" or \
        spec["kind"] == "carry" else 0
    kw["train_anchors"] = int(spec.get("anchors", 0))
    kw["carry_rule"] = None if spec["rule"] in (None, "none") else spec["rule"]
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
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "kind": spec["kind"], "window": spec["window"],
                   "anchors": spec.get("anchors", 0),
                   "rule": spec["rule"], "seed": spec["seed"],
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


def keep_mask_for(spec: dict, T: int, dev: str):
    if spec["kind"] == "full":
        return None
    return window_anchor_keep_mask(T, int(spec["window"]), int(spec.get("anchors", 0)),
                                   device=dev)


def frontier(rows: list[dict]) -> dict:
    """在 (total_bytes, bpc) 上算支配关系与前沿包络（越小越好，两轴皆是）。"""
    pts = sorted(rows, key=lambda r: r["total_bytes"])
    dominated = {}
    for a in pts:
        dom_by = [b["name"] for b in pts
                  if b["total_bytes"] <= a["total_bytes"]
                  and b["bpc"] < a["bpc"] and b["name"] != a["name"]]
        dominated[a["name"]] = dom_by
    # 单调前沿：只保留「随字节增加而 bpc 下降」的点
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
    return None


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
        km = keep_mask_for(spec, T, dev)
        n_vis = None if km is None else int(km.sum().item())
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:12]}… kind={spec['kind']} "
             f"window={spec['window']} rule={spec['rule']} "
             f"params={ck[spec['name']]['n_params']} visible_pairs={n_vis}")

    # 掩码自检：窗口臂的可见对数必须等于声明值（非退化、非恒等）
    selfchecks = {}
    for spec in cfg["arms"]:
        if spec["kind"] == "full":
            continue
        km = keep_mask_for(spec, T, dev)
        s = int(km.sum().item())
        exp = sum(min(i + 1, int(spec["window"]))
                  for i in range(T))  # anchors=0 时每行可见 min(i+1, window)
        causal = T * (T + 1) // 2
        selfchecks[spec["name"]] = {"visible_pairs": s, "expected": exp,
                                    "ok": s == exp,
                                    "nondegenerate": s < causal}
        assert s == exp, f"{spec['name']}: visible pairs {s} != {exp}"
        # 非退化是**硬断言**（复核 N2）：掩码在 T=256 上不得是恒等算子
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
        state = int(spec.get("state_floats", 0)) * bpe
        total = kv + state
        rows.append({"name": name, "kind": spec["kind"], "seed": spec["seed"],
                     "window": spec["window"], "rule": spec["rule"],
                     "bpc": round(bpc, 6),
                     "kv_bytes": kv, "state_bytes": state, "total_bytes": total,
                     "k_eff": round(total / unit, 4),
                     "n_params": ck[name]["n_params"]})
        note(f"  {name}: bpc={bpc:.6f} k_eff={total/unit:.2f} "
             f"total={total} B (kv={kv} + state={state})")

    # ---- 前沿与定位 ----
    fr = frontier(rows)
    by = {r["name"]: r for r in rows}
    win16 = [r for r in rows if r["name"].startswith("win16")]
    carry16 = [r for r in rows if r["name"].startswith("carry16")]
    full = by["full"]
    bpc_win16_s0 = by["win16_s0"]["bpc"]
    gap_win16_full = bpc_win16_s0 - full["bpc"]
    location = {}
    for c in carry16:
        dc = bpc_win16_s0 - c["bpc"]              # carry 相对同窗无通道的增益（正=更好）
        loc = interp_window_at(rows, c["k_eff"])
        rel = None if loc is None else round(loc["bpc_interp"] - c["bpc"], 6)
        location[c["name"]] = {
            "delta_bpc_vs_win16_s0": round(dc, 6),
            "recovery_ratio_r": round(dc / gap_win16_full, 4) if gap_win16_full else None,
            "window_curve_at_k_eff": loc,
            "carry_minus_interp_window": rel,
            "note": ("carry_minus_interp_window = bpc_interp − bpc_carry（正 = carry 更好）："
                     "< 0 ⇒ carry bpc 高于同 k_eff 的纯窗口插值点 ⇒ 被同预算纯窗口支配；"
                     "> 0 ⇒ 前沿改善；≈0 ⇒ 不可排序；"
                     "null ⇒ 无窗口点跨越该 k_eff（不外推）"),
        }

    # ---- gap-closing / 噪声 ----
    sigma_win16 = abs(by["win16_s0"]["bpc"] - by["win16_s1"]["bpc"])
    sigma_carry16 = abs(by["carry16_s0"]["bpc"] - by["carry16_s1"]["bpc"])
    effects = {}
    for s in (0, 1):
        n_ = by[f"win16_s{s}"]["bpc"]
        d_ = by[f"carry16_s{s}"]["bpc"]
        effects[str(s)] = {"win16": n_, "carry16": d_, "delta_bpc": round(n_ - d_, 6)}
    # ---- 干预活性（SCHEMA §10）：掩码活性 + 训练日志 window_mask + carry liveness ----
    g_probe = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, vocab, (2, T), generator=g_probe).to(dev)
    liveness = {}
    for spec in cfg["arms"]:
        name = spec["name"]
        row = {"kind": spec["kind"], "window": spec["window"], "rule": spec["rule"]}
        if spec["kind"] == "full":
            row["note"] = "全因果锚：无掩码开关，活性量不适用"
        else:
            km = keep_mask_for(spec, T, dev)
            s = int(km.sum().item())
            row["mask"] = {"n_visible_pairs": s, "causal_pairs": T * (T + 1) // 2,
                           "visible_frac": round(s / (T * (T + 1) // 2), 6),
                           "alive": s < T * (T + 1) // 2}
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
            row["train_wall_minutes"] = tr.get("wall_minutes")
            row["train_peak_gpu_mem_mb"] = tr.get("peak_gpu_mem_mb")
        liveness[name] = row
    # 评测期反事实：掩码开/关是否真的改变前向（SCHEMA §10；复核 N2）
    wprobe = None
    for spec in cfg["arms"]:
        if spec["kind"] == "window" and int(spec["window"]) == 17:
            wprobe = {"arm": spec["name"],
                      **window_mask_probe(arms[spec["name"]], x_probe)}
            break
    liveness["_mask_probe"] = {
        "note": "评测期反事实：同一批输入在『掩码开 / 掩码关』两次前向的最后一层 logits 最大绝对差（>0 ⇔ 掩码真的改了前向）",
        "window17_masked_vs_open": wprobe}

    results = {
        "config": cfg,
        "checkpoints": ck,
        "selfchecks": selfchecks,
        "liveness": liveness,
        "rows": rows,
        "frontier": fr,
        "location": location,
        "gap_closing": {
            "bpc_win16_s0": bpc_win16_s0,
            "bpc_full": full["bpc"],
            "gap_win16_full": round(gap_win16_full, 6),
            "note": "gap = 窗口 16 相对全因果在 clean bpc 上的损失；recovery_ratio_r 见 location",
        },
        "effects_by_seed": effects,
        "sigma_within_config": {
            "win16": round(sigma_win16, 6),
            "carry16": round(sigma_carry16, 6),
            "note": "同配置换种子的极差（2 样本 ⇒ 描述量，不是方差上界）",
        },
        "cost": {
            "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
            "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "n_eval_windows": eb["n_batches"] * eb["batch_size"],
            "tokens_scored": eb["n_batches"] * eb["batch_size"] * (T - 1),
        },
        "code_hashes": {
            rel: sha256_file(os.path.join(ROOT, rel)) for rel in
            ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/retention.py",
             "pathlm/data.py") if os.path.exists(os.path.join(ROOT, rel))
        },
    }
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in ("run.py", "config.json")})
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
        note(f"  {c}: Δbpc={v['delta_bpc_vs_win16_s0']:+.6f} "
             f"r={v['recovery_ratio_r']} vs 窗口插值={rel_s}")


if __name__ == "__main__":
    main()
