#!/usr/bin/env python3
"""2624-carry-under-compression-quality · 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 Z3；§4b 记录 DESIGN 裁决）。
本入口只做**评测**（不训练、不改权重），把三行——`carry_mha@fp16` /
`carry_gqa@fp16` / `carry_gqa@int8`——在合成长程任务上量出来：

  ① 合成关联回忆（主腿；`assoc_recall_batch`，native 可见性 window16）；
  ② 交叉全因果口径（keep_mask=None）与训练型任务（多查询）口径并报；
  ③ 头共享增量（GQA − MHA）与评测期 int8 增量（int8 − fp16），§17.17 分量分解；
  ④ 毁源对照臂（源 KV 段毁掉）；
  ⑤ 参数量差声明（GQA 混杂，§15）；
  ⑥ 活性量（KV 头形状 / carry read_share / window mask 每 200 步；量化活性）；
  ⑦ 收敛门禁（§17.15：答案行 CE 末窗斜率）。

用法：
  python3 run.py                # 全电池（纯推理）
  python3 run.py --smoke        # 冒烟：少批（实测耗时/显存，不外推）
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
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, kv_head_liveness,  # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)
from tools.evalkit.kv_quant import _new_stats                             # noqa: E402
from tools.evalkit.longrange_tasks import (assoc_recall_batch,              # noqa: E402
                                           var_recall_batch)
from tools.evalkit.retention import masked_logits                           # noqa: E402
from tools.evalkit.carry_channel import two_proportion                      # noqa: E402
from compression_arms import (BASELINE_EVAL_ARM, COMPRESSED_EVAL_ARM, EVAL_ARMS,        # noqa: E402
                  EVAL_ARM_NAMES, GQA_EVAL_ARM, TRAINED_ARMS,
                  TRAINED_ARM_NAMES, compress_increment, eval_arm,
                  expected_block_class, model_config_kwargs, native_visibility,
                  trained_arm)
from quant_carry import (QUANT_MODES, quant_carry_logits,                   # noqa: E402
                         quant_liveness_of, quant_multi_recall_row,
                         quant_recall_row)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, mcfg: dict, vocab: int, device: str, ckpt: dict):
    """加载一条训练臂的权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    spec = trained_arm(name)
    path = resolve(os.path.join(ckpt["dir"], name, "model.pt"))
    got = sha256_file(path)
    want = ckpt.get("sha256", {}).get(name)
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{name}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, **model_config_kwargs(spec)})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "block_class": type(model.blocks[0]).__name__,
                   "block_class_expected": expected_block_class(spec),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "flags": model_config_kwargs(spec)}


def kvcache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per_value: float,
                  t: int) -> int:
    """每位置 KV 字节（解析；与 kv_quant.kv_cache_footprint 同式，但用 n_kv_heads）。"""
    return int(2 * n_layers * n_kv_heads * head_dim * bytes_per_value * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["eval"]["n_batches"] = 1
        cfg["eval"]["batch_size"] = 16
        cfg["cross_leg"]["n_batches"] = 1
        cfg["cross_leg"]["batch_size"] = 16
        cfg["train_task_leg"]["n_batches"] = 1
        cfg["train_task_leg"]["batch_size"] = 16
        cfg["distance_leg"]["n_batches"] = 1
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    mcfg = cfg["model"]
    K = cfg["eval"]["K_main"]
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    win, anc = cfg["eval"]["window"], cfg["eval"]["anchors"]
    sym = cfg["tokens"]["sym_vocab"]
    fill = cfg["tokens"]["filler_token"]
    cue = cfg["tokens"]["cue_token"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    results: dict = {"config": cfg}

    # ---------------- 加载两条训练臂 ----------------
    note("加载两条训练臂（sha256 断言）")
    models, ck = {}, {}
    for name in TRAINED_ARM_NAMES:
        models[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"],
                                          dev, cfg["checkpoint"])
        note(f"  {name:10s} {ck[name]['sha256'][:16]}… block={ck[name]['block_class']} "
             f"params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    # ---------------- ① 仪器自检：块类 / 参数 / 非退化 / fp32 逐位等值 ----------------
    note("自检①：块类 / 参数量 / 窗口非退化（§17.16）/ fp32 逐位等值（实现锚）")
    n_base = ck["carry_mha"]["n_params"]
    param_decl = {}
    for name in TRAINED_ARM_NAMES:
        n = ck[name]["n_params"]
        param_decl[name] = {"n_params": n,
                            "delta_vs_carry_mha": n - n_base,
                            "delta_pct_vs_carry_mha": round(100.0 * (n - n_base) / n_base, 3)}
    results["param_declaration"] = param_decl
    wliv = window_mask_liveness(T, win, anc)
    results["window_nondegenerate"] = wliv

    # fp32 档必须与释放版前向（masked_logits 经真实块类）逐位相等 —— 两种块类各一条
    g = torch.Generator().manual_seed(1234)
    probe = assoc_recall_batch(8, T, K, sym_vocab=sym, filler_token=fill,
                               cue_token=cue,
                               generator=torch.Generator().manual_seed(7))
    probe_tok = probe.tokens[:8].to(dev)
    keep = window_anchor_keep_mask(T, win, anc, device=dev)
    bitwise = {}
    with torch.no_grad():
        for name in TRAINED_ARM_NAMES:
            rel = masked_logits(models[name], probe_tok, keep)
            f32, _ = quant_carry_logits(models[name], probe_tok, "fp32", keep)
            bitwise[name] = {"equal_to_released": bool(torch.equal(rel, f32)),
                             "max_abs_diff": float((rel.float() - f32.float()).abs().max())}
        # 同输入重跑逐位一致（纯推理确定性）
        r1 = masked_logits(models["carry_gqa"], probe_tok, keep)
        r2 = masked_logits(models["carry_gqa"], probe_tok, keep)
        bitwise["rerun_identical_carry_gqa"] = bool(torch.equal(r1, r2))
    results["instrument_selfcheck"] = {"fp32_bitwise": bitwise,
                                       "block_classes": {n: ck[n]["block_class"]
                                                         for n in TRAINED_ARM_NAMES}}
    note(f"  fp32 逐位等值：{ {n: bitwise[n]['equal_to_released'] for n in TRAINED_ARM_NAMES} }")

    # KV 头活性（形状类，取自真实前向）
    kv_live = {}
    for name in TRAINED_ARM_NAMES:
        out = kv_head_liveness(models[name], probe_tok)
        kv_live[name] = {"n_heads": out["n_heads"], "n_kv_heads": out["n_kv_heads"],
                         "k_shape": out["k_batch_shape"], "qkv_out_width": out["qkv_out_width"],
                         "kv_reduction_frac": out["kv_reduction_frac"], "alive": out["alive"]}
    results["kv_head_liveness"] = kv_live
    note(f"  KV 头形状：{ {n: kv_live[n]['k_shape'] for n in TRAINED_ARM_NAMES} }")

    # ---------------- ② 主腿（合成长程任务，native 可见性）+ 毁源对照 ----------------
    note("判据①③④：合成关联回忆（native window16）+ 毁源对照臂")
    main_leg, cross_leg, train_task_leg = {}, {}, {}
    mt = cfg["train_task_leg"]
    for row in EVAL_ARMS:
        rname, mname, mode = row["name"], row["weights"], row["kv_mode"]
        r = quant_recall_row(models[mname], T, win, anc, bs, nb, sym, fill, cue, K,
                             mode, dev)
        main_leg[rname] = r
        note(f"  {rname:16s} mode={mode:4s} acc={r['accuracy']:.4f} CI={r['accuracy_ci']} "
             f"chance={r['chance']:.4f} lp={r['local_prior']:.4f} "
             f"corrupt={r['corrupt_source']['accuracy']:.4f}")
    results["main_leg"] = main_leg

    note("交叉口径：全因果（keep_mask=None）")
    for row in EVAL_ARMS:
        rname, mname, mode = row["name"], row["weights"], row["kv_mode"]
        r = quant_recall_row(models[mname], T, 0, 0, cfg["cross_leg"]["batch_size"],
                             cfg["cross_leg"]["n_batches"], sym, fill, cue, K, mode, dev)
        cross_leg[rname] = r
        note(f"  {rname:16s} full-causal acc={r['accuracy']:.4f} "
             f"corrupt={r['corrupt_source']['accuracy']:.4f}")
    results["cross_leg_full_causal"] = cross_leg

    note("训练型任务（多查询 assoc-mqar）召回 —— in-distribution 口径")
    for row in EVAL_ARMS:
        rname, mname, mode = row["name"], row["weights"], row["kv_mode"]
        r = quant_multi_recall_row(models[mname], T, win, anc, mt["batch_size"],
                                   mt["n_batches"], sym, fill, cue, K, mt["n_queries"],
                                   mt["gap"], mode, dev)
        train_task_leg[rname] = r
        note(f"  {rname:16s} acc={r['accuracy']:.4f} CI={r['accuracy_ci']} "
             f"corrupt={r['corrupt_source']['accuracy']:.4f} "
             f"dist={r['query_key_distance']['min']}-{r['query_key_distance']['max']}")
    results["train_task_leg"] = train_task_leg

    # ---------------- ③ 距离腿（辅助） ----------------
    note("辅助腿：变距离回忆距离曲线（native 可见性）")
    distance_leg = {}
    keep_native = window_anchor_keep_mask(T, win, anc, device=dev)
    for row in EVAL_ARMS:
        rname, mname, mode = row["name"], row["weights"], row["kv_mode"]
        curve = {}
        for d in cfg["distance_leg"]["distances"]:
            accs, ns = 0, 0
            for i in range(cfg["distance_leg"]["n_batches"]):
                tb = var_recall_batch(cfg["distance_leg"]["batch_size"], T, d,
                                      sym_vocab=sym, filler_token=fill, cue_token=cue,
                                      generator=torch.Generator().manual_seed(5000 + i))
                from quant_carry import quant_accuracy
                r = quant_accuracy(models[mname], tb, mode, keep_mask=keep_native,
                                   device=dev)
                accs += r["n_hits"]; ns += r["n"]
            curve[str(d)] = {"accuracy": round(accs / ns, 6), "n": ns}
        distance_leg[rname] = curve
    results["distance_leg"] = distance_leg

    # ---------------- ④ 增量与 CI（§17.17 分量分解） ----------------
    note("判据②③：头共享增量 / int8 增量 / 净和（§17.17）")
    increments = {}
    for leg_name, leg in (("native", main_leg), ("full_causal", cross_leg),
                          ("train_task", train_task_leg)):
        acc = {r: leg[r]["accuracy"] for r in EVAL_ARM_NAMES}
        increments[leg_name] = {**compress_increment(acc), "accuracy": acc}
        pairs = {}
        for a, b in ((BASELINE_EVAL_ARM, GQA_EVAL_ARM),
                     (GQA_EVAL_ARM, COMPRESSED_EVAL_ARM),
                     (BASELINE_EVAL_ARM, COMPRESSED_EVAL_ARM)):
            pairs[f"{b}_vs_{a}"] = two_proportion(
                leg[a]["n_hits"], leg[a]["n"], leg[b]["n_hits"], leg[b]["n"])
        increments[leg_name]["pairwise"] = pairs
    results["increments"] = increments
    for leg in ("native", "train_task"):
        note(f"  {leg}: {increments[leg]['gqa_vs_mha_pp']}pp / "
             f"{increments[leg]['int8_vs_fp16_pp']}pp / net {increments[leg]['net_pp']}pp")

    # ---------------- ⑤ 毁源对照有效性 ----------------
    destroy = {}
    for rname in EVAL_ARM_NAMES:
        ci = main_leg[rname]["corrupt_source"]["accuracy_ci"]
        chance = main_leg[rname]["chance"]
        destroy[rname] = {"corrupt_accuracy": main_leg[rname]["corrupt_source"]["accuracy"],
                          "corrupt_ci": ci, "chance": chance,
                          "valid": ci[0] <= 4 * chance,
                          "rule": "毁源后 CI 下界 > 4×chance ⇒ 该行读数作废"}
    results["destroy_source_validity"] = destroy
    note(f"  毁源对照有效：{ {r: destroy[r]['valid'] for r in EVAL_ARM_NAMES} }")

    # ---------------- ⑥ 量化活性（§10） ----------------
    note("判据⑥：量化活性（changed_frac / rel_err）")
    qlive = {}
    for long_name, mode in (("carry_mha_fp16", "fp16"), ("carry_gqa_fp16", "fp16"),
                            ("carry_gqa_int8", "int8")):
        mname = eval_arm(long_name)["weights"]
        acc = _new_stats()
        with torch.no_grad():
            quant_carry_logits(models[mname], probe_tok, mode, keep, acc)
        qlive[long_name] = quant_liveness_of(acc)
    results["quant_liveness"] = qlive
    note(f"  int8 活性：changed_frac={qlive['carry_gqa_int8']['changed_frac']} "
         f"rel_err={qlive['carry_gqa_int8']['rel_err']}")

    # ---------------- ⑦ 训练活性 + 收敛门禁（from 训练日志） ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    activity = {}
    for name in TRAINED_ARM_NAMES:
        rows = [json.loads(ln) for ln in open(os.path.join(train_dir, name,
                                                           "train_log.jsonl")) if ln.strip()]
        r200 = [r for r in rows if r["step"] % 200 == 0]
        last = rows[-1]
        lo = 0.75 * steps
        winr = [r for r in rows if r["step"] >= lo and r.get("synth_answer_ce") is not None]
        slope = None
        if len(winr) >= 2:
            (s0, c0), (s1, c1) = (winr[0]["step"], winr[0]["synth_answer_ce"]), \
                                 (winr[-1]["step"], winr[-1]["synth_answer_ce"])
            slope = round((c1 - c0) / max(s1 - s0, 1) * 1000, 5)
        activity[name] = {
            "n_log_lines": len(rows), "final_step": last["step"],
            "final_loss": last["loss"], "final_synth_answer_ce": last.get("synth_answer_ce"),
            "synth_chance_ce": last.get("synth_chance_ce"),
            "answer_ce_slope_nats_per_1k_lastquarter": slope,
            "kv_heads_series_present": sum(1 for r in r200 if r.get("kv_heads")),
            "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
            "window_mask_series_present": sum(1 for r in r200 if r.get("window_mask")),
            "last": {"attn_carry": last.get("attn_carry"),
                     "window_mask": last.get("window_mask"),
                     "kv_heads": last.get("kv_heads")},
        }
    results["train_activity"] = activity
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）：" +
         ", ".join(f"{n}={activity[n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in TRAINED_ARM_NAMES))

    # ---------------- ⑧ 解析字节口径（辅助，非判据） ----------------
    head_dim = mcfg["d_model"] // mcfg["n_heads"]
    bytes_analytic = {}
    for rname, nheads, bpv in (("carry_mha_fp16", mcfg["n_heads"], 2.0),
                               ("carry_gqa_fp16", 2, 2.0),
                               ("carry_gqa_int8", 2, 1.0)):
        bytes_analytic[rname] = {"n_kv_heads": nheads, "bytes_per_value": bpv,
                                 "per_position_bytes": kvcache_bytes(
                                     mcfg["n_layers"], nheads, head_dim, bpv, 1)}
    base_b = bytes_analytic["carry_mha_fp16"]["per_position_bytes"]
    for rname, r in bytes_analytic.items():
        r["reduction_vs_carry_mha_fp16"] = round(1 - r["per_position_bytes"] / base_b, 6)
    results["bytes_analytic"] = {
        "rows": bytes_analytic, "baseline": "carry_mha_fp16", "baseline_bytes": base_b,
        "note": ("解析每位置字节 = 2×layers×n_kv_heads×head_dim×value_bytes（辅助，"
                 "非判据；int8 的 scale 开销单列、与 T 无关）。")}

    # ---------------- 成本 / 口径 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 3),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "pure_inference": True,
    }
    results["provenance"] = {"run_log": log}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("MAINLEG:", {r: main_leg[r]["accuracy"] for r in EVAL_ARM_NAMES})
    print("INCREMENTS(native):", {k: v for k, v in increments["native"].items()
                                  if k != "pairwise"})


if __name__ == "__main__":
    main()
