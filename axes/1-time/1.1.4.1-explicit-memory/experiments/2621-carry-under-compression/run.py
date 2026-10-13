"""2621 · 搬运通道 × KV 精度压缩 —— 实验入口（纯推理评测，**不训练**）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录 W3 契约；§4b 记录 GQA 因子未测的
偏离与裁决）。本入口只做**评测**，回答「对一个 delta 写通道已被训练用起来的模型，
评测期把注意力 KV 降成 int8，召回会不会崩、字节/位置省多少」。

被测量模型：ticket 35 的 **U2** 检查点（delta 写通道 + 窗口 16 + MHA，只读 + 运行时
sha256 断言；权重不入库）。新语义层只有 `carry_quant.py`（carry-aware 量化前向），
`pathlm/` 与 `tools/evalkit/` 只读。

用法：
  python3 run.py                # 全电池（纯推理，共享卡 ~1–3 min）
  python3 run.py --smoke        # 冒烟：少批（实测步时/显存，不外推）
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

from pathlm.depth_ar import (DepthARConfig, DepthARModel, carry_liveness,  # noqa: E402
                             carry_probe, window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import (assoc_recall_batch,             # noqa: E402
                                           var_recall_batch, accuracy)
from tools.evalkit.retention import masked_logits                         # noqa: E402
from tools.evalkit.carry_channel import (wilson_ci, two_proportion)       # noqa: E402
from tools.evalkit.kv_quant import summarize_stats                        # noqa: E402
import carry_quant as cq                                                  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_checkpoint(spec: dict, mcfg: dict, vocab: int, device: str):
    """加载 U2 权重并**断言 sha256**（权重不入库，哈希必须逐次核）。"""
    path = resolve(spec["path"])
    got = sha256_file(path)
    if got != spec["sha256"]:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {spec['sha256']}\n"
                         f"  got  {got}")
    cfg = DepthARConfig(**mcfg)
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "carry_rule": cfg.carry_rule,
                   "n_params": sum(p.numel() for p in model.parameters())}


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
        cfg["eval"]["batch_size"] = 16
        cfg["eval"]["n_batches"] = 1
        cfg["distance_leg"]["n_batches"] = 1
        cfg["distance_leg"]["batch_size"] = 8
        cfg["bpc_leg"]["n_batches"] = 1
        cfg["bpc_leg"]["batch_size"] = 8
        cfg["modes"] = cfg["modes"]
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window = cfg["visibility"]["train_window"]
    anchors = cfg["visibility"]["train_anchors"]
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    modes = cfg["modes"]
    base_mode = cfg["primary_baseline_mode"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 U2 检查点（sha256 断言；delta 写 + 窗口 16 + MHA）")
    mcfg = {**cfg["model"]}
    torch.manual_seed(cfg["seed"])
    model, ck = load_checkpoint(cfg["checkpoint"], mcfg, 206, dev)
    note(f"  U2 {ck['sha256'][:16]}… params={ck['n_params']} carry={ck['carry_rule']}")
    results: dict = {"config": cfg, "checkpoint": ck,
                     "visibility": {"train_window": window, "train_anchors": anchors}}

    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, 206, (2, T), generator=g).cuda()

    # ---------------- ① 仪器自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（逐位）")
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        a1 = masked_logits(model, x_probe, None)
        a2 = masked_logits(model, x_probe, exact)
    release_vs_exact = float((a1 - a2).abs().max())
    note(f"  max|Δlogit| = {release_vs_exact:.3e}")

    note("自检 ①：模型自带窗口掩码 == masked_logits + window_anchor_keep_mask（逐位）")
    with torch.no_grad():
        b1 = model(x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(model, x_probe, keep)
    win_selfcheck = float((b1 - b2).abs().max())
    note(f"  max|Δlogit| = {win_selfcheck:.3e}")

    note("自检 ①（本实验关键）：carry_quant fp32 == 释放版前向（逐位）")
    with torch.no_grad():
        c1, tele32 = cq.carry_quant_logits(model, x_probe, "fp32", keep)
    fp32_vs_release = float((b1 - c1).abs().max())
    fp32_vs_masked = float((b2 - c1).abs().max())
    note(f"  max|Δlogit| vs 模型前向 = {fp32_vs_release:.3e}；"
         f"vs masked_logits = {fp32_vs_masked:.3e}")

    # quantize_kv 活性自检：fp32 档必须逐位恒等（不改数值）
    with torch.no_grad():
        acc0 = cq._new_stats()
        cq.carry_quant_logits(model, x_probe, "fp32", keep, acc0)
    fp32_quant_stats = summarize_stats(acc0)
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": release_vs_exact,
        "model_window_forward_vs_mask_instrument_max_abs_logit": win_selfcheck,
        "carry_quant_fp32_vs_model_release_max_abs_logit": fp32_vs_release,
        "carry_quant_fp32_vs_masked_logits_max_abs_logit": fp32_vs_masked,
        "carry_quant_fp32_quant_stats": fp32_quant_stats,
        "fp32_is_bitwise_identity": (fp32_vs_release == 0.0
                                     and fp32_quant_stats["n_values"] == 0),
        "note": ("carry_quant fp32 档逐行复刻 CarryChannelBlock.forward（含窗口掩码与"
                 "通道读出），未量化 ⇒ 与释放版逐位一致。`n_values=0` 证明 fp32 档"
                 "确实没有触碰任何数值（量化算子恒等）。"),
    }
    note(f"  fp32 档 quantize_kv 活性：{fp32_quant_stats}")

    # ---------------- ②③ 主腿 + 毁源对照 ----------------
    Kmain = cfg["eval"]["K_main"]
    note(f"判据 ①②③：关联回忆（K={Kmain}，主腿）+ 毁源对照臂（{len(modes)} 个 mode）")
    main_leg, quant_acc = {}, {m: cq._new_stats() for m in modes}
    for mode in modes:
        batches = [assoc_recall_batch(bs, T, Kmain, sym_vocab=sym,
                                      filler_token=fill, cue_token=cue,
                                      generator=torch.Generator().manual_seed(
                                          1000 + 7 * Kmain + i))
                   for i in range(nb)]
        agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [], "by_dist": {}}
        for tb in batches:
            r = cq.quant_arm_battery(model, tb, window, mode, anchors, dev,
                                     corrupt_span_bounds=(0, 2 * Kmain),
                                     acc=quant_acc[mode])
            agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
            agg["cn"] += r["corrupt_source"]["n"]
            agg["chits"] += r["corrupt_source"]["n_hits"]
            agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
            for d, v in r["by_distance"].items():
                slot = agg["by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
                slot["n"] += v["n"]
                slot["n_hits"] += round(v["accuracy"] * v["n"])
            agg["chance"] = r["chance_uniform_answer_set"]
            agg["local_prior"] = r["local_prior"]
        row = {
            "mode": mode, "K": Kmain, "n_batches": nb,
            "accuracy": round(agg["n_hits"] / agg["n"], 6),
            "n": agg["n"], "n_hits": agg["n_hits"],
            "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
            "chance": agg["chance"], "local_prior": agg["local_prior"],
            "corrupt_source": {
                "span": [0, 2 * Kmain],
                "accuracy": round(agg["chits"] / agg["cn"], 6),
                "n": agg["cn"], "n_hits": agg["chits"],
                "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
            "query_key_distance": {
                "min": min(agg["dists"]), "max": max(agg["dists"]),
                "median": sorted(agg["dists"])[len(agg["dists"]) // 2],
                "frac_outside_window": round(
                    sum(1 for d in agg["dists"] if d >= window) / len(agg["dists"]), 6),
                "window": window, "n": len(agg["dists"])},
            "by_distance": agg["by_dist"],
            "quant_liveness": summary_of(quant_acc[mode], mode),
            "read_share": probe_read_share(model, mode, keep),
        }
        main_leg[mode] = row
        note(f"  {mode}: acc={row['accuracy']:.4f} CI={row['accuracy_ci']} n={row['n']} "
             f"| 毁源={row['corrupt_source']['accuracy']:.4f} "
             f"| read_share={row['read_share']}")

    premise = {
        "task": "assoc_recall", "K": Kmain, "window": window,
        "chance": main_leg[base_mode]["chance"],
        "local_prior": main_leg[base_mode]["local_prior"],
        "query_key_distance": main_leg[base_mode]["query_key_distance"],
        "note": ("前提：被问的 pair 落在窗口外（keys 100%% 窗外）⇒ 需要通道搬运；"
                 "MHA 全因果基线（U0）在 2618 已实测 ≈ chance。本实验的主对照是"
                 "fp16（全精度 KV），不是 U0。"),
    }
    comparisons = {}
    for mode in modes:
        if mode == base_mode:
            continue
        comparisons[f"{mode}_vs_{base_mode}"] = two_proportion(
            main_leg[base_mode]["n_hits"], main_leg[base_mode]["n"],
            main_leg[mode]["n_hits"], main_leg[mode]["n"])
    # 判据 ① 的操作化：崩 = delta_pp < −3 且 CI 不重叠
    c1_ok, c1_broken = {}, {}
    for mode in modes:
        if mode == base_mode:
            continue
        cmp_ = comparisons[f"{mode}_vs_{base_mode}"]
        c1_ok[mode] = (cmp_["delta_pp"] >= -3.0) or cmp_["ci_overlap"]
        c1_broken[mode] = (cmp_["delta_pp"] < -3.0) and (not cmp_["ci_overlap"])
    results["main_leg"] = main_leg
    results["premise"] = premise
    results["comparisons"] = comparisons

    # 判据 ③：毁源对照（任一 mode 若毁源后 CI 下界仍 > 4×chance ⇒ 作废）
    c3 = {mode: {"main": main_leg[mode]["accuracy"],
                 "corrupt": main_leg[mode]["corrupt_source"]["accuracy"],
                 "corrupt_ci": main_leg[mode]["corrupt_source"]["accuracy_ci"],
                 "chance": main_leg[mode]["chance"],
                 "invalidated": (main_leg[mode]["corrupt_source"]["accuracy_ci"][0]
                                 > 4 * main_leg[mode]["chance"])}
          for mode in modes}
    results["corrupt_source_control"] = c3

    # ---------------- ⑤ 辅助腿：变距离回忆 ----------------
    note("辅助腿：变距离回忆（fp16 vs int8 × 六档距离）")
    dleg, dbs, dnb = {}, cfg["distance_leg"]["batch_size"], cfg["distance_leg"]["n_batches"]
    for mode in cfg["distance_leg"]["modes"]:
        rows = {}
        for d in cfg["distance_leg"]["distances"]:
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0}
            for i in range(dnb):
                tb = var_recall_batch(dbs, T, d, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      generator=torch.Generator().manual_seed(
                                          2000 + 13 * d + i))
                r = cq.quant_arm_battery(model, tb, window, mode, anchors, dev,
                                         corrupt_span_bounds=(tb.meta["src_pos"],
                                                              tb.meta["src_pos"] + 2))
                agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
                agg["cn"] += r["corrupt_source"]["n"]
                agg["chits"] += r["corrupt_source"]["n_hits"]
                chance, lp = r["chance_uniform_answer_set"], r["local_prior"]
            rows[str(d)] = {"distance": d, "n": agg["n"],
                            "accuracy": round(agg["n_hits"] / agg["n"], 6),
                            "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
                            "corrupt_source": {
                                "accuracy": round(agg["chits"] / agg["cn"], 6),
                                "n": agg["cn"],
                                "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
                            "chance": chance, "local_prior": lp,
                            "source_position": tb.meta["src_pos"],
                            "source_inside_window": (T - 1) - tb.meta["src_pos"] < window}
        dleg[mode] = rows
        note("  " + mode + ": " + " ".join(
            f"d{d}:{rows[str(d)]['accuracy']:.3f}"
            for d in cfg["distance_leg"]["distances"]))
    results["distance_leg"] = dleg

    # ---------------- ② 字节/位置（解析）+ bpc 腿 ----------------
    note("判据 ②：解析每位置字节 + clean bpc（enwik8 评测尾段）")
    results["bytes_analytic"] = cq.bytes_table(model.cfg)
    for mode, r in results["bytes_analytic"]["per_mode"].items():
        rel = f"相对 fp16 下降 {100 * r['reduction_vs_fp16']:.1f}%"
        if mode == "fp16":
            rel = "基线"
        elif mode == "fp32":
            rel = "释放版锚（无压缩，两倍于 fp16）"
        note(f"  {mode}: {r['per_position_bytes']} B/位置（{rel}）")

    from pathlm.data import load_enwik8_full
    _, eval_arr, bpc_vocab = load_enwik8_full(
        ".tmp/enwik8", os.path.join(ROOT, cfg["data"]["bpc_path"]))
    note(f"  enwik8 评测尾段 vocab={bpc_vocab}（模型 vocab=206，一致）")
    bpc = {}
    for mode in modes:
        per_batch = []

        def on_batch(i, stats, bpc_so_far, _m=mode, _pb=per_batch):
            _pb.append({"batch": i, "quant": stats, "bpc_so_far": round(bpc_so_far, 6)})
            log.append({"t": round(time.time() - t0, 2),
                        "msg": f"bpc[{_m}] batch {i}: bpc_so_far={bpc_so_far:.6f}"})

        b, ql = cq.carry_quant_bpc(
            model, eval_arr, mode, keep_mask=keep,
            n_batches=cfg["bpc_leg"]["n_batches"],
            batch_size=cfg["bpc_leg"]["batch_size"],
            window_seed=cfg["bpc_leg"]["window_seed"], device=dev, on_batch=on_batch)
        bpc[mode] = {"bpc": round(b, 6), "quant_liveness": ql,
                     "per_batch": per_batch, "n_batches": cfg["bpc_leg"]["n_batches"]}
        note(f"  {mode}: clean bpc={b:.6f}")
    results["bpc_leg"] = bpc
    results["bpc_damage"] = {
        m: round(bpc[m]["bpc"] - bpc[base_mode]["bpc"], 6)
        for m in modes if m != base_mode}

    # ---------------- ④ 活性：通道 + 量化 ----------------
    note("判据 ④：通道活性（read_share / carry_liveness / carry_probe）")
    probe_x = torch.randint(0, 206, (2, T),
                            generator=torch.Generator().manual_seed(1234)).cuda()
    with torch.no_grad():
        model._input_states(probe_x)          # 写入并落 `_carry_tele`（状态扫描活性）
        scan_tele = model._carry_tele
        carry_layers = model(probe_x)[1]["attn_carry"]
    results["carry_activity"] = {
        "carry_liveness": carry_liveness(model, scan_tele, carry_layers,
                                         prob=carry_probe(model, probe_x)),
        "read_share_by_mode": {m: main_leg[m]["read_share"] for m in modes},
        "quant_liveness_by_mode": {m: main_leg[m]["quant_liveness"] for m in modes},
        "activity_cadence": (
            "eval-only 设计无训练步 ⇒ 「每 200 步」的训练期活性量由被复用检查点的"
            "原始训练日志提供（见 training_log_activity）；本实验的量化活性量按批"
            "落盘（bpc 腿每批，见 bpc_leg[*].per_batch），召回腿按 mode 汇总。"),
    }
    # 复用检查点的训练期「每 200 步」通道活性（判据 ④ 的原始要求，见 README §4）
    tl_path = resolve("axes/1-time/1.1.4.1-explicit-memory/experiments/"
                      "2618-carry-channel-13m/train/U2/train_log.jsonl")
    tl_rows = [json.loads(l) for l in open(tl_path) if l.strip()]
    cadence = [r for r in tl_rows if r["step"] % 200 == 0]
    results["training_log_activity"] = {
        "path": os.path.relpath(tl_path, ROOT),
        "n_rows": len(tl_rows), "n_rows_step_multiple_of_200": len(cadence),
        "last_carry_activity": cadence[-1].get("attn_carry") if cadence else None,
        "first_carry_activity": cadence[0].get("attn_carry") if cadence else None,
        "note": ("U2 训练日志每 200 步落盘通道活性（read_share/state_fro/beta_mean/"
                 "write_rel）——它是被复用检查点的「活性真的活着」证据。"),
    }

    # ---------------- 收尾：成本 / 复现性 / 指纹 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "n_recall_batches": nb * len(modes) * 2,
        "n_distance_batches": (dnb * len(cfg["distance_leg"]["distances"])
                               * len(cfg["distance_leg"]["modes"]) * 2),
        "n_bpc_batches": cfg["bpc_leg"]["n_batches"] * len(modes),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
    }
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": results["cost"]["peak_gpu_mem_mb"],
        "note": "实测 = 本进程 torch.cuda.max_memory_allocated（净分配）",
    }

    note("复现性：主腿 fp16 再跑一遍（同 seed）")
    repro = {"pass": True, "same_seed_rerun_max_abs_diff": 0.0}
    for mode in (base_mode, "int8"):
        accs = []
        for _ in range(2):
            a = {"n": 0, "n_hits": 0}
            for i in range(4):
                tb = assoc_recall_batch(bs, T, Kmain, sym_vocab=sym,
                                        filler_token=fill, cue_token=cue,
                                        generator=torch.Generator().manual_seed(
                                            1000 + 7 * Kmain + i))
                r = cq.quant_arm_battery(model, tb, window, mode, anchors, dev)
                a["n"] += r["n"]; a["n_hits"] += r["n_hits"]
            accs.append(a)
        d = abs(accs[0]["n_hits"] / accs[0]["n"] - accs[1]["n_hits"] / accs[1]["n"])
        repro[f"{mode}_n_hits_pass1"] = accs[0]["n_hits"]
        repro[f"{mode}_n_hits_pass2"] = accs[1]["n_hits"]
        repro[f"{mode}_max_abs_diff"] = d
        repro["pass"] = repro["pass"] and (accs[0] == accs[1])
    results["reproducibility"] = repro
    note(f"  {repro}")

    code_files = ("run.py", "carry_quant.py", "config.json", "README.md",
                  "reconcile_numbers.py", "mutate.py",
                  "tests/test_carry_quant.py",
                  "pathlm/depth_ar.py", "tools/evalkit/kv_quant.py",
                  "tools/evalkit/carry_channel.py",
                  "tools/evalkit/longrange_tasks.py")
    results["code"] = {}
    for p in code_files:
        # 实验目录内的文件优先（避免误取仓库根的 README.md），否则从仓库根解析
        cand = os.path.join(HERE, p)
        if not os.path.exists(cand):
            cand = resolve(p)
        results["code"][p] = sha256_file(cand) if os.path.exists(cand) else None

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "run_log.txt"), "w") as f:
        for r in log:
            f.write(f"[{r['t']:7.1f}s] {r['msg']}\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min，"
         f"峰值 {results['cost']['peak_gpu_mem_mb']} MB）")


def summary_of(acc: dict, mode: str):
    from tools.evalkit.kv_quant import summarize_stats
    return summarize_stats(acc) if mode != "fp32" else None


def probe_read_share(model, mode: str, keep: torch.Tensor) -> float:
    """在一小批上测该 mode 的通道读出占比（read_share，与训练日志同一量）。"""
    g = torch.Generator().manual_seed(77)
    x = torch.randint(0, 206, (2, model.cfg.seq_len), generator=g)
    x = x.to(next(model.parameters()).device)
    with torch.no_grad():
        _, tele = cq.carry_quant_logits(model, x, mode, keep)
    return tele["read_share"]


if __name__ == "__main__":
    main()
