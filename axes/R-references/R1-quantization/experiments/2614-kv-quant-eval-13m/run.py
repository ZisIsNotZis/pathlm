"""2614 · KV 量化参照线（ticket 31 契约 K）—— 实验入口（纯推理，不训练）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
实现：`tools/evalkit/kv_quant.py`（评测期 KV cache 量化-反量化，不改权重）。
被测量模型：`2610-frozen-trunk-jumphead-13m/trunk.pt`（复用契约 I 的同一
checkpoint，运行时断言 sha256）。

跑四件事：① 仪器自检（fp32 档前向 == 释放版前向，逐位；且锚 bpc 1.4836）；
② 三档 KV 精度（fp16 基线 / int8 / int4）的 bpc + 活性量 + 内存口径；
③ 复现性（同 seed 重跑）；④ 跨评测窗口 seed 的噪声底。

用法：
  python3 run.py                 # 全电池（约 1 min，共享卡）
  python3 run.py --smoke         # 冒烟：2 个窗口批次，实测步时/显存（不外推时间）
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

from pathlm.data import load_enwik8_full                       # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel        # noqa: E402
from tools.evalkit.kv_quant import (kv_quant_bpc, release_bpc,  # noqa: E402
                                    kv_cache_footprint, cache_reduction)

RELEASE_ANCHOR_BPC = 1.4836   # 2610/MODEL.md 记的 13.5M 主干 clean bpc（口径见 README）


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trunk(cfg: dict, device: str):
    """加载被测量主干，并**断言 sha256**（复用他人 checkpoint 的强制项）。"""
    ckpt = os.path.join(ROOT, cfg["checkpoint"]["path"])
    got = sha256_file(ckpt)
    want = cfg["checkpoint"]["sha256"]
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    size = os.path.getsize(ckpt)
    model = DepthARModel(DepthARConfig(**cfg["model"]), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": cfg["checkpoint"]["path"], "sha256": got,
                   "size_bytes": size, "sha256_verified": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None,
                    help="落盘目录；缺省 = 本目录（--smoke 时 = 本目录/smoke，"
                         "避免冒烟覆盖正式 results.json）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：2 个窗口批次，实测步时与显存（不外推时间字段）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE

    cfg = json.load(open(args.config))
    ecfg = cfg["eval"]
    modes = cfg["kv_quant"]["modes"]
    baseline = cfg["kv_quant"]["baseline_mode"]
    if args.smoke:
        ecfg["n_batches"] = 2
        cfg["reproducibility"]["rerun"] = False
        cfg["noise_floor"]["window_seeds"] = [3, 4]
    os.makedirs(args.out_dir, exist_ok=True)

    dev = ecfg["device"]
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    model, ck = load_trunk(cfg, dev)
    T = cfg["model"]["seq_len"]
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated() / 2 ** 20
    t0 = time.time()
    log_f = open(os.path.join(args.out_dir, "kv_quant_log.jsonl"), "w")
    neval = {"n": 0}

    def bpc_of(mode, window_seed=None, tag=""):
        """一次评测：bpc + 活性量；逐 batch 把活性量落盘（周期性，SCHEMA §10）。"""
        sd = ecfg["window_seed"] if window_seed is None else window_seed
        t = time.time()

        def on_batch(i, stats, cur_bpc):
            log_f.write(json.dumps(
                {"tag": tag, "mode": mode, "window_seed": sd, "batch": i,
                 "bpc_running": round(cur_bpc, 6), "stats": stats,
                 "sec": round(time.time() - t, 4)}, ensure_ascii=False) + "\n")
            log_f.flush()

        v, stats = kv_quant_bpc(model, eval_arr, mode,
                                n_batches=ecfg["n_batches"],
                                batch_size=ecfg["batch_size"],
                                window_seed=sd, device=dev, on_batch=on_batch)
        neval["n"] += 1
        log_f.write(json.dumps({"tag": f"{tag}:done", "mode": mode,
                                "window_seed": sd, "bpc": round(v, 6),
                                "activity": stats,
                                "sec": round(time.time() - t, 4)},
                               ensure_ascii=False) + "\n")
        log_f.flush()
        return v, stats

    # ---------- 仪器自检：fp32 档前向 ≡ 释放版前向 ----------
    t = time.time()
    bpc_release = release_bpc(model, eval_arr, n_batches=ecfg["n_batches"],
                              batch_size=ecfg["batch_size"],
                              window_seed=ecfg["window_seed"], device=dev)
    neval["n"] += 1
    bpc_fp32, _ = bpc_of("fp32", tag="fp32")
    instrument = {
        "bpc_release_path": round(bpc_release, 6),
        "bpc_fp32_mode": round(bpc_fp32, 6),
        "abs_diff_release_vs_fp32": abs(bpc_release - bpc_fp32),
        "bitwise_equal": bpc_release == bpc_fp32,
        "release_anchor_bpc_2610": RELEASE_ANCHOR_BPC,
        "abs_diff_vs_anchor": round(abs(bpc_release - RELEASE_ANCHOR_BPC), 6),
    }
    print(f"[自检] 释放版 {bpc_release:.6f} / fp32 档 {bpc_fp32:.6f} "
          f"(逐位 {'=' if instrument['bitwise_equal'] else '≠'}); "
          f"锚 {RELEASE_ANCHOR_BPC}", flush=True)

    # ---------- 各档 KV 精度 ----------
    results_by_mode = {}
    for mode in modes:
        v, stats = bpc_of(mode, tag=f"mode:{mode}")
        results_by_mode[mode] = {"bpc": round(v, 6), "activity": stats}
    bpc_base = results_by_mode[baseline]["bpc"]

    memory = {}
    for mode in ("fp16", "int8", "int4"):
        memory[mode] = kv_cache_footprint(model.cfg, mode,
                                          seq_lens=tuple(cfg["memory"]["seq_lens"]))
    memory_reduction = {m: cache_reduction(memory["fp16"], memory[m])
                        for m in ("int8", "int4")}

    quant_rows = []
    for mode in ("fp16", "int8", "int4"):
        v = results_by_mode[mode]["bpc"]
        quant_rows.append({
            "mode": mode,
            "bpc": round(v, 6),
            # Δ = bpc_对照 − bpc_实验（正 = 改善，SCHEMA §17.3）；对照 = fp16
            "delta_bpc": round(bpc_base - v, 6),
            "damage_bpc": round(v - bpc_base, 6),
            "abs_delta_bpc": round(abs(v - bpc_base), 6),
            "activity": results_by_mode[mode]["activity"],
            "per_position_bytes": memory[mode]["per_position_bytes"],
            "reduction_vs_fp16_frac": memory_reduction.get(mode, {}).get(
                "reduction_frac", 0.0),
        })

    # ---------- 复现性（同 seed 重跑）----------
    repro = {"threshold": cfg["reproducibility"]["max_abs_delta_diff_threshold"],
             "rerun": bool(cfg["reproducibility"]["rerun"]), "per_config": []}
    if cfg["reproducibility"]["rerun"]:
        for mode in cfg["reproducibility"]["rerun_modes"]:
            v2, _ = bpc_of(mode, tag=f"rerun:{mode}")
            v1 = results_by_mode[mode]["bpc"]
            repro["per_config"].append({
                "config": f"mode={mode}",
                "damage_run1": round(v1 - bpc_base, 6),
                "damage_run2": round(v2 - bpc_base, 6),
                "abs_diff": round(abs(v2 - v1), 9)})
        repro["max_abs_damage_diff"] = max(r["abs_diff"] for r in repro["per_config"])
        repro["pass"] = repro["max_abs_damage_diff"] < repro["threshold"]

    # ---------- 噪声底（跨评测窗口 seed 的配对 damage 散布）----------
    nf = {"window_seeds": cfg["noise_floor"]["window_seeds"], "modes": {}}
    for mode in cfg["noise_floor"]["modes"]:
        dams = []
        for sd in nf["window_seeds"]:
            ref, _ = bpc_of("fp16", window_seed=sd, tag=f"nf_ref@{sd}")
            v, _ = bpc_of(mode, window_seed=sd, tag=f"nf:{mode}@{sd}")
            dams.append(v - ref)
        mean = sum(dams) / len(dams)
        var = sum((d - mean) ** 2 for d in dams) / max(len(dams) - 1, 1)
        nf["modes"][mode] = {"n": len(dams), "mean_damage_bpc": round(mean, 6),
                             "std_damage_bpc": round(var ** 0.5, 6),
                             "damage_bpc": [round(d, 6) for d in dams]}
    stds = sorted(v["std_damage_bpc"] for v in nf["modes"].values())
    nf["max_std_damage_bpc"] = stds[-1] if stds else None
    nf["median_std_damage_bpc"] = stds[len(stds) // 2] if stds else None

    # ---------- 判据 ----------
    band = cfg["variance_band_bpc"]
    int8 = next(r for r in quant_rows if r["mode"] == "int8")
    int4 = next(r for r in quant_rows if r["mode"] == "int4")
    criteria = {
        "band_bpc": band,
        "delta_sign": "Δ = bpc_对照(fp16) − bpc_实验(mode)（正=改善，SCHEMA §17.3）；damage = −Δ",
        "int8_delta_bpc": int8["delta_bpc"],
        "int8_abs_delta_bpc": int8["abs_delta_bpc"],
        "int8_memory_reduction_frac": int8["reduction_vs_fp16_frac"],
        "int8_near_lossless": int8["abs_delta_bpc"] < band,
        "int8_verdict": ("近无损（|Δbpc| < 0.014）⇒ 写入参照线，后续所有压缩主张必须与之比较"
                         if int8["abs_delta_bpc"] < band else
                         f"有损（|Δbpc| = {int8['abs_delta_bpc']:.4f} ≥ 0.014）"),
        "int4_delta_bpc": int4["delta_bpc"],
        "int4_damage_bpc": int4["damage_bpc"],
        "int4_memory_reduction_frac": int4["reduction_vs_fp16_frac"],
        "int4_damage_ge_band": int4["damage_bpc"] >= band,
        "int4_verdict": (f"代价 = +{int4['damage_bpc']:.4f} bpc（劣化 ≥ 0.014）"
                         if int4["damage_bpc"] >= band else
                         f"劣化 {int4['damage_bpc']:.4f} bpc（< 0.014，近无损）"),
        "fp16_vs_fp32_delta_bpc": round(bpc_fp32 - bpc_base, 6),
        "noise_tol_bpc_2sigma": round(2 * (nf["max_std_damage_bpc"] or 0.0), 6),
    }

    alive = all(results_by_mode[m]["activity"]["changed_frac"] > 0
                for m in ("fp16", "int8", "int4"))
    intervention = {
        "what": "评测期 KV cache 量化：K/V 存成 fp16/int8/int4 再读回（注意力前往返）",
        "params_added": 0, "weights_changed": False, "trained": False,
        "alive": alive,
        "activity_note": ("活性量 = 每档的 n_values / mean_abs_err / rel_err / "
                          "changed_frac（逐 batch 落盘 kv_quant_log.jsonl）——证明"
                          "量化真的改了数值。恒等档 fp32 的 changed_frac = 0"),
        "activity_by_mode": {m: results_by_mode[m]["activity"]
                             for m in ("fp32", "fp16", "int8", "int4")},
    }

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    results = {
        "contract": ("axes/R-references/R1-quantization/experiments/"
                     "2614-kv-quant-eval-13m/README.md"),
        "checkpoint": ck,
        "config": cfg,
        "instrument": instrument,
        "intervention": intervention,
        "modes": results_by_mode,
        "quant_rows": quant_rows,
        "memory": memory,
        "memory_reduction_vs_fp16": memory_reduction,
        "reproducibility": repro,
        "noise_floor": nf,
        "criteria": criteria,
        "code": {
            "run_py_sha256": sha256_file(os.path.abspath(__file__)),
            "kv_quant_py_sha256": sha256_file(
                os.path.join(ROOT, "tools", "evalkit", "kv_quant.py")),
            "tests_sha256": sha256_file(os.path.join(ROOT, "tests", "test_kv_quant.py")),
        },
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "peak_mem_mib": round(peak, 1), "base_mem_mib": round(base_mem, 1),
                 "net_peak_mem_mib": round(peak - base_mem, 1),
                 "n_evals": neval["n"], "device": torch.cuda.get_device_name(0)},
    }
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    log_f.close()
    print(json.dumps({k: v for k, v in results.items() if k not in ("config",)},
                     indent=2, ensure_ascii=False)[:4000])
    print(f"\n落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB "
          f"(net {peak-base_mem:.0f})，{neval['n']} 次评测", flush=True)


if __name__ == "__main__":
    main()
