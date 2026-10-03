#!/usr/bin/env python3
"""2625-ood-leg-variance · 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 W2）。
本入口对**四条同配置臂**（`gqa_carry_win_s0` 复用 / `s1`,`s2` 本实验训练 /
`y1_gqa_carry_win_s0` 复用参照）在**同一批位置**上量三套口径：

  ① **判决腿（近 OOD，距离 241–255）**：`assoc_recall_batch`（单查询，native window16）
     —— 本线要量方差的那条腿；
  ② **训练分布内腿（≤240）**：`assoc_recall_multi_batch`（训练型任务，多查询，gap14）
     —— 对照腿；附变距离曲线（辅助）；
  ③ **交叉全因果口径**（keep_mask=None）—— 去掉窗口需求后的参照读数。

每条腿报：逐臂准确率 + Wilson 95% CI、**n=3 主集**的极差 / 均值 ± sd、
以及预登记判据 ②（极差 ≥ 3pp ⇒ 该腿降级为辅助腿，§17.18）。

仪器：复用 2624（Z3）的 `quant_carry.py`（与 2618/W3/2623 同一件仪器；
`fp32` 档与释放版前向逐位相等，本入口自检钉住）。本实验**不引入新语义层**。

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
# 复用 2624 的评测仪器 `quant_carry.py`（同目录改名会破坏「同件仪器」的可追溯性）。
Z3_DIR = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type",
                      "experiments", "2624-carry-under-compression-quality")
for p in (ROOT, HERE, Z3_DIR):
    sys.path.insert(0, p)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,  # noqa: E402
                             kv_head_liveness, window_anchor_keep_mask,
                             window_mask_liveness)
from tools.evalkit.retention import masked_logits  # noqa: E402
from tools.evalkit.longrange_tasks import (assoc_recall_batch,  # noqa: E402
                                           var_recall_batch)

import quant_carry as qc  # noqa: E402  (2624 的评测仪器)
import ood_variance_arms as oa  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, mcfg: dict, vocab: int, device: str, cfg: dict):
    """加载一条臂的权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    spec = oa.arm(name)
    if name in oa.TRAINED_ARM_NAMES:
        path = resolve(os.path.join(cfg["checkpoint"]["dir"], name, "model.pt"))
        want = cfg["checkpoint"]["sha256"].get(name)
    else:
        path = resolve(spec["path"])
        want = spec["sha256"]
    got = sha256_file(path)
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{name}: 没有可用 sha256（先训练并回填 config）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    kwargs = {**mcfg, **oa.model_config_kwargs(spec)}
    model = DepthARModel(DepthARConfig(**kwargs), vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    block_class = type(model.blocks[0]).__name__
    block_expected = oa.expected_block_class(spec)
    if block_class != block_expected:
        raise SystemExit(f"{name}: 块类不符：实际 {block_class} ≠ 期望 {block_expected}")
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "seed": oa.train_seed(spec),
                   "block_class": block_class,
                   "block_class_expected": block_expected,
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "flags": oa.model_config_kwargs(spec),
                   "role": ("trained" if name in oa.TRAINED_ARM_NAMES else "reused")}


def _variance_block(accs: dict, names) -> dict:
    vals = [accs[n]["accuracy"] for n in names]
    return {"names": list(names),
            "accuracies": {n: accs[n]["accuracy"] for n in names},
            "range_pp": oa.range_pp(vals),
            "mean_pp": oa.mean_pp(vals),
            "sd_pp": oa.sd_pp(vals),
            "min_pp": round(100.0 * min(vals), 3),
            "max_pp": round(100.0 * max(vals), 3)}


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
        for leg in ("main_leg", "cross_leg", "train_task_leg"):
            cfg[leg]["n_batches"] = 1
            cfg[leg]["batch_size"] = 16
        cfg["distance_leg"]["n_batches"] = 1
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    mcfg = cfg["model"]
    K = cfg["main_leg"]["K_main"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], \
        cfg["tokens"]["cue_token"]
    log: list[dict] = []
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def note(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    results: dict = {"config": cfg}

    # ---------------- 加载四条臂 ----------------
    note("加载四条臂（sha256 断言）")
    models, ck = {}, {}
    for name in oa.ALL_ARMS:
        models[name], ck[name] = load_arm(name, mcfg, cfg["data"]["vocab_size"],
                                          dev, cfg)
        note(f"  {name:20s} seed={ck[name]['seed']} {ck[name]['sha256'][:16]}… "
             f"block={ck[name]['block_class']} params={ck[name]['n_params']}")
    results["checkpoints"] = ck

    # ---------------- ① 仪器自检 + 窗口非退化（§17.16） ----------------
    note("自检①：块类 / 窗口非退化 / fp32 逐位等值（实现锚）")
    wliv = window_mask_liveness(T, cfg["main_leg"]["window"], cfg["main_leg"]["anchors"])
    results["window_nondegenerate"] = wliv
    probe = assoc_recall_batch(8, T, K, sym_vocab=sym, filler_token=fill,
                               cue_token=cue,
                               generator=torch.Generator().manual_seed(7))
    probe_tok = probe.tokens[:8].to(dev)
    keep = window_anchor_keep_mask(T, cfg["main_leg"]["window"],
                                   cfg["main_leg"]["anchors"], device=dev)
    bitwise = {}
    with torch.no_grad():
        for name in oa.ALL_ARMS:
            rel = masked_logits(models[name], probe_tok, keep)
            f32, _ = qc.quant_carry_logits(models[name], probe_tok, "fp32", keep)
            bitwise[name] = {"equal_to_released": bool(torch.equal(rel, f32)),
                             "max_abs_diff": float((rel.float() - f32.float()).abs().max())}
        r1 = masked_logits(models[oa.ALL_ARMS[0]], probe_tok, keep)
        r2 = masked_logits(models[oa.ALL_ARMS[0]], probe_tok, keep)
        bitwise["rerun_identical"] = bool(torch.equal(r1, r2))
    results["instrument_selfcheck"] = {
        "fp32_bitwise": bitwise,
        "block_classes": {n: ck[n]["block_class"] for n in oa.ALL_ARMS},
        "same_block_class_all_arms": len({ck[n]["block_class"] for n in oa.ALL_ARMS}) == 1,
        "same_n_params_all_arms": len({ck[n]["n_params"] for n in oa.ALL_ARMS}) == 1}
    note(f"  fp32 逐位等值：{ {n: bitwise[n]['equal_to_released'] for n in oa.ALL_ARMS} }")

    kv_live = {}
    for name in oa.ALL_ARMS:
        out = kv_head_liveness(models[name], probe_tok)
        kv_live[name] = {"n_kv_heads": out["n_kv_heads"], "k_shape": out["k_batch_shape"],
                         "alive": out["alive"]}
    results["kv_head_liveness"] = kv_live

    # ---------------- ② 判决腿（单查询，native window16） + 毁源 ----------------
    mleg = cfg["main_leg"]
    note("判据①：判决腿（近 OOD，距离 241–255，native window16）")
    main_leg, main_leg_released = {}, {}
    for name in oa.ALL_ARMS:
        r = qc.quant_recall_row(models[name], T, mleg["window"], mleg["anchors"],
                                mleg["batch_size"], mleg["n_batches"], sym, fill,
                                cue, K, "fp16", dev)
        r_rel = qc.quant_recall_row(models[name], T, mleg["window"], mleg["anchors"],
                                    mleg["batch_size"], mleg["n_batches"], sym, fill,
                                    cue, K, "fp32", dev)
        main_leg[name] = r
        main_leg_released[name] = r_rel
        note(f"  {name:20s} fp16 acc={r['accuracy']:.4f} CI={r['accuracy_ci']} "
             f"hits={r['n_hits']}/{r['n']} dist={r['query_key_distance']['min']}-"
             f"{r['query_key_distance']['max']} | fp32(released)={r_rel['accuracy']:.4f} "
             f"corrupt={r['corrupt_source']['accuracy']:.4f}")
    results["main_leg"] = main_leg
    results["main_leg_released"] = main_leg_released
    results["main_leg_variance"] = _variance_block(main_leg, oa.PRIMARY_ARMS)
    results["main_leg_variance_reference_included"] = _variance_block(
        main_leg, oa.ALL_ARMS)

    # ---------------- ③ 交叉口径（全因果） ----------------
    cl = cfg["cross_leg"]
    note("交叉口径：全因果（keep_mask=None）")
    cross_leg = {}
    for name in oa.ALL_ARMS:
        r = qc.quant_recall_row(models[name], T, 0, 0, cl["batch_size"],
                                cl["n_batches"], sym, fill, cue, K, "fp16", dev)
        cross_leg[name] = r
        note(f"  {name:20s} full-causal acc={r['accuracy']:.4f}")
    results["cross_leg_full_causal"] = cross_leg
    results["cross_leg_variance"] = _variance_block(cross_leg, oa.PRIMARY_ARMS)

    # ---------------- ④ 训练分布内腿（多查询，≤240） ----------------
    mt = cfg["train_task_leg"]
    note("判据③：训练分布内腿（训练型任务，多查询，距离 ≤240）")
    train_task_leg = {}
    for name in oa.ALL_ARMS:
        r = qc.quant_multi_recall_row(models[name], T, mt["window"], mt["anchors"],
                                      mt["batch_size"], mt["n_batches"], sym, fill,
                                      cue, K, mt["n_queries"], mt["gap"], "fp16", dev)
        train_task_leg[name] = r
        note(f"  {name:20s} acc={r['accuracy']:.4f} CI={r['accuracy_ci']} "
             f"hits={r['n_hits']}/{r['n']} dist={r['query_key_distance']['min']}-"
             f"{r['query_key_distance']['max']} corrupt={r['corrupt_source']['accuracy']:.4f}")
    results["train_task_leg"] = train_task_leg
    results["train_task_leg_variance"] = _variance_block(train_task_leg, oa.PRIMARY_ARMS)

    # ---------------- ⑤ 距离曲线（辅助） ----------------
    note("辅助腿：变距离回忆距离曲线（native 可见性）")
    keep_native = window_anchor_keep_mask(T, mleg["window"], mleg["anchors"], device=dev)
    distance_leg = {}
    for name in oa.ALL_ARMS:
        curve = {}
        for d in cfg["distance_leg"]["distances"]:
            accs, ns = 0, 0
            for i in range(cfg["distance_leg"]["n_batches"]):
                tb = var_recall_batch(
                    cfg["distance_leg"]["batch_size"], T, d, sym_vocab=sym,
                    filler_token=fill, cue_token=cue,
                    generator=torch.Generator().manual_seed(oa.VAR_SEED_BASE + i))
                rr = qc.quant_accuracy(models[name], tb, "fp16", keep_mask=keep_native,
                                       device=dev)
                accs += rr["n_hits"]; ns += rr["n"]
            curve[str(d)] = {"accuracy": round(accs / ns, 6), "n": ns}
        distance_leg[name] = curve
    results["distance_leg"] = distance_leg

    # ---------------- ⑥ 预登记判据 ②（降级判定） ----------------
    note("判据②：极差 ≥ 3pp ⇒ 该腿永久降级为辅助腿（§17.18）")
    dv_main = results["main_leg_variance"]
    dv_in = results["train_task_leg_variance"]
    results["verdict"] = {
        "leg": "assoc_recall_batch single-query, key distance 241-255, native window16",
        "in_training_dist": False,
        "n_primary": len(oa.PRIMARY_ARMS),
        "primary_arms": list(oa.PRIMARY_ARMS),
        "accuracy_pp": {n: round(100.0 * main_leg[n]["accuracy"], 3)
                        for n in oa.PRIMARY_ARMS},
        "range_pp": dv_main["range_pp"],
        "mean_pp": dv_main["mean_pp"],
        "sd_pp": dv_main["sd_pp"],
        "effect_threshold_pp": oa.EFFECT_THRESHOLD_PP,
        "downgrade_to_auxiliary": oa.downgrade_to_auxiliary(dv_main["range_pp"]),
        "in_dist_leg": {"leg": "assoc_recall_multi_batch (train task), distance 107-240",
                        "n_primary": len(oa.PRIMARY_ARMS),
                        "accuracy_pp": {n: round(100.0 * train_task_leg[n]["accuracy"], 3)
                                        for n in oa.PRIMARY_ARMS},
                        "range_pp": dv_in["range_pp"], "mean_pp": dv_in["mean_pp"],
                        "sd_pp": dv_in["sd_pp"],
                        "high_variance": oa.downgrade_to_auxiliary(dv_in["range_pp"])},
        "reference_arm": {n: {"seed": ck[n]["seed"],
                              "main_leg_pp": round(100.0 * main_leg[n]["accuracy"], 3),
                              "in_dist_pp": round(100.0 * train_task_leg[n]["accuracy"], 3)}
                          for n in oa.REFERENCE_ARMS}}
    note(f"  判决腿 n=3：{results['verdict']['accuracy_pp']} pp，极差 "
         f"{dv_main['range_pp']} pp ⇒ 降级={results['verdict']['downgrade_to_auxiliary']}")
    note(f"  分布内腿 n=3：{results['verdict']['in_dist_leg']['accuracy_pp']} pp，极差 "
         f"{dv_in['range_pp']} pp ⇒ 高方差={results['verdict']['in_dist_leg']['high_variance']}")

    # ---------------- ⑦ 训练活性 + 收敛门禁（来自训练日志） ----------------
    steps = cfg["train"]["steps"]
    train_dir = resolve(cfg["checkpoint"]["dir"])
    activity = {}
    for name in oa.TRAINED_ARM_NAMES:
        rows = [json.loads(ln) for ln in
                open(os.path.join(train_dir, name, "train_log.jsonl")) if ln.strip()]
        r200 = [r for r in rows if r["step"] % 200 == 0]
        last = rows[-1]
        winr = [r for r in rows if r["step"] >= 0.75 * steps
                and r.get("synth_answer_ce") is not None]
        slope = None
        if len(winr) >= 2:
            (s0, c0), (s1, c1) = (winr[0]["step"], winr[0]["synth_answer_ce"]), \
                                 (winr[-1]["step"], winr[-1]["synth_answer_ce"])
            slope = round((c1 - c0) / max(s1 - s0, 1) * 1000, 5)
        activity[name] = {
            "n_log_lines": len(rows), "final_step": last["step"],
            "final_loss": last["loss"], "final_synth_answer_ce": last.get("synth_answer_ce"),
            "gnorm_max": round(max(r["gnorm"] for r in rows if r.get("gnorm") is not None), 3),
            "gnorm_last": round(last["gnorm"], 4),
            "answer_ce_slope_nats_per_1k_lastquarter": slope,
            "kv_heads_series_present": sum(1 for r in r200 if r.get("kv_heads")),
            "attn_carry_series_present": sum(1 for r in r200 if r.get("attn_carry")),
            "window_mask_series_present": sum(1 for r in r200 if r.get("window_mask")),
            "last_read_share": (last.get("attn_carry") or {}).get("read_share"),
            "last_window_mask": last.get("window_mask"),
        }
    results["train_activity"] = activity
    note("收敛门禁（§17.15，答案行 CE 末窗斜率 nats/1k）：" +
         ", ".join(f"{n}={activity[n]['answer_ce_slope_nats_per_1k_lastquarter']}"
                   for n in oa.TRAINED_ARM_NAMES))

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
    print("MAINLEG:", {r: main_leg[r]["accuracy"] for r in oa.ALL_ARMS})
    print("IN-DIST :", {r: train_task_leg[r]["accuracy"] for r in oa.ALL_ARMS})
    print("VERDICT :", json.dumps(results["verdict"]["accuracy_pp"],
                                  ensure_ascii=False),
          "range_pp=", dv_main["range_pp"])


if __name__ == "__main__":
    main()
