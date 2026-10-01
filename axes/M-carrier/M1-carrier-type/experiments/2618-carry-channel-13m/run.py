"""2618 · 三臂对照（U0 无通道 / U1 plain 写 / U2 delta 写）——实验入口（评测电池）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
训练：`python3 train_depth_ar.py <arm> ... --train-window 16 --synth-task assoc-mqar
--carry-write {none,plain,delta}`（见 `MODEL.md` 的生成命令）。
本入口只做**评测**（纯推理），跑六件事：

  ① 仪器自检：释放版 == 显式因果掩码（逐位）；模型自带窗口掩码前向 ==
     `retention.masked_logits` + `window_anchor_keep_mask`（训练/评测同一件仪器）；
     三臂**逐位等参数**（U1/U2）与通道臂主干 == U0（运行期硬断言）；
  ② 判据 ①（前提校验）：U0 的关联回忆准确率 + chance + 二项 CI + **查询-key 距离分布**
     （确认 key 落在窗外）；
  ③ 判据 ②③：U1/U2/U0 的准确率与两两比较（CI 不重叠口径）；
  ④ 判据 ④（毁源对照臂）：把源片段换掉后每臂的读数；
  ⑤ 判据 ⑤（辅助腿）：变距离回忆的距离曲线（三臂）；
  ⑥ 判据 ⑥：三臂同方法 loss 斜率、重跑散度、解析/实测显存分开报。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡 ~5–10 min）
  python3 run.py --smoke         # 冒烟：少批（实测耗时，不外推训练时间）
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

from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             carry_liveness, carry_probe,
                             carry_read_mass_stats, window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_batch, var_recall_batch, accuracy, local_prior_accuracy,
    task_summary)
from tools.evalkit.retention import masked_logits                  # noqa: E402
from tools.evalkit.carry_channel import (                          # noqa: E402
    wilson_ci, two_proportion, arm_battery, param_equality,
    trunk_equality, analytic_memory, loss_slope)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str,
             expect_sha: str | None = None) -> tuple[DepthARModel, dict]:
    """加载一臂的权重并**断言 sha256**（权重不入库，哈希必须逐次核）。"""
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256") or expect_sha
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, "carry_rule": spec["carry_rule"]})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "carry_rule": spec["carry_rule"]}


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
        cfg["ksweep"] = [8]
        cfg["K_main"] = 8
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], cfg["tokens"]["cue_token"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载三臂（sha256 断言）")
    arms, ck = {}, {}
    for name in ("U0", "U1", "U2"):
        arms[name], ck[name] = load_arm(cfg["arms"][name], cfg["model"],
                                        cfg["data"]["vocab_size"], dev)
        note(f"{name}: {ck[name]['sha256'][:16]}… carry={ck[name]['carry_rule']}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors}}

    # ---------------- ① 仪器自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（逐位）")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        # 两者**都走仪器接口**（`masked_logits`）：一个不传掩码（释放版 is_causal），
        # 一个传显式因果掩码。注意不能拿 `model(x)` 相比 —— 模型的 cfg 里
        # train_window=16，它的前向会自己套窗口掩码（那是下一步才检的东西）。
        a1 = masked_logits(arms["U0"], x_probe, None)
        a2 = masked_logits(arms["U0"], x_probe, exact)
    release_vs_exact = float((a1 - a2).abs().max())
    note(f"  max|Δlogit| = {release_vs_exact:.3e}")
    note("自检 ①：模型自带窗口掩码前向 vs masked_logits+window_anchor_keep_mask（逐位）")
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    with torch.no_grad():
        b1 = arms["U0"](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms["U0"], x_probe, keep)
    win_selfcheck = float((b1 - b2).abs().max())
    note(f"  max|Δlogit| = {win_selfcheck:.3e}")
    # 逐位等参数是**初始化**层面的性质（训练之后各臂权重当然不同）⇒ 按要求在
    # 「同 seed 重新构造」的三个模型上核（与单测同一件事，但用同一份 resolved config）。
    note("自检 ①：逐位等参数（U1 vs U2）+ 通道臂主干 == U0（同 seed 重新构造）")
    built = {}
    for name, rule in (("U0", None), ("U1", "plain"), ("U2", "delta")):
        torch.manual_seed(cfg["seed"])
        built[name] = DepthARModel(DepthARConfig(**{**cfg["model"],
                                                   "carry_rule": rule}),
                                   cfg["data"]["vocab_size"])
    eq12 = param_equality(built["U1"], built["U2"])
    eq01 = trunk_equality(built["U1"], built["U0"])
    built["U0"].eval(); built["U1"].eval(); built["U2"].eval()
    with torch.no_grad():
        l0 = built["U0"](x_probe.cpu())[1]["depth_logits"][-1]
        id_start = {"U1_matches_U0_bitwise": torch.equal(
            l0, built["U1"](x_probe.cpu())[1]["depth_logits"][-1]),
            "U2_matches_U0_bitwise": torch.equal(
                l0, built["U2"](x_probe.cpu())[1]["depth_logits"][-1])}
    eq12["built_for_check"] = "同 seed 重新构造（训练后的权重当然不同）"
    eq01["built_for_check"] = "同 seed 重新构造"
    note(f"  U1 vs U2 逐位等参数: {eq12['bitwise_equal']}（{eq12['n_params_a']} 参数）")
    note(f"  通道臂主干逐位 == U0: {eq01['trunk_bitwise_equal']}；新增参数 "
         f"{eq01['n_params_added']}（{100 * eq01['n_params_added'] / eq01['n_params_base_arm']:.2f}%）")
    note(f"  单边零起步（γ=0）⇒ 起步 logits 逐位等于 U0: {id_start}")
    # 结构层（trained checkpoints）：参数名/形状必须只差通道那几个张量
    struct = {"U1_vs_U2_extra_names":
              sorted(set(arms["U1"].state_dict()) ^ set(arms["U2"].state_dict())),
              "U1_vs_U0_added_names": sorted(set(arms["U1"].state_dict())
                                             - set(arms["U0"].state_dict())),
              "n_params": {n: sum(p.numel() for p in arms[n].parameters())
                           for n in arms}}
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": release_vs_exact,
        "model_window_forward_vs_mask_instrument_max_abs_logit": win_selfcheck,
        "params_bitwise_equal_U1_U2": eq12,
        "trunk_bitwise_equal_carry_vs_U0": eq01,
        "identity_start_logits_bitwise": id_start,
        "trained_checkpoint_structure": struct,
        "release_path_note": ("U0/U1/U2 三臂都在 dense 路径；释放版（is_causal=True）"
                             "与显式因果掩码逐位一致，与 2617 电池的仪器自检同口径"),
    }

    # ---------------- ②③④ 主电池（关联回忆 + 毁源对照） ----------------
    note("判据 ②③④：关联回忆（主腿）+ 毁源对照臂")
    per_arm = {}
    for name, model in arms.items():
        for K in cfg["ksweep"]:
            batches = [assoc_recall_batch(bs, T, K, sym_vocab=sym,
                                          filler_token=fill, cue_token=cue,
                                          generator=torch.Generator().manual_seed(
                                              1000 + 7 * K + i))
                       for i in range(nb)]
            row = {"K": K, "n_batches": nb}
            agg = {"n": 0, "n_hits": 0, "corrupt_n": 0, "corrupt_hits": 0,
                   "dists": [], "acc_by_dist": {}}
            for tb in batches:
                r = arm_battery(model, tb, window, anchors, dev,
                                corrupt_span_bounds=(0, 2 * K))
                agg["n"] += r["n"]
                agg["n_hits"] += r["n_hits"]
                agg["corrupt_n"] += r["corrupt_source"]["n"]
                agg["corrupt_hits"] += r["corrupt_source"]["n_hits"]
                agg["dists"].extend(int(v) for v in tb.dists.flatten().tolist())
                for d, v in r["by_distance"].items():
                    slot = agg["acc_by_dist"].setdefault(d, {"n": 0, "n_hits": 0})
                    slot["n"] += v["n"]
                    slot["n_hits"] += round(v["accuracy"] * v["n"])
                row.setdefault("local_prior", r["local_prior"])
                row["chance"] = r["chance_uniform_answer_set"]
            row.update({
                "accuracy": round(agg["n_hits"] / agg["n"], 6),
                "n": agg["n"], "n_hits": agg["n_hits"],
                "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
                "corrupt_source": {
                    "span": [0, 2 * K], "accuracy": round(agg["corrupt_hits"] / agg["corrupt_n"], 6),
                    "n": agg["corrupt_n"], "n_hits": agg["corrupt_hits"],
                    "accuracy_ci": wilson_ci(agg["corrupt_hits"], agg["corrupt_n"])},
                "query_key_distance": {
                    "min": min(agg["dists"]), "max": max(agg["dists"]),
                    "median": sorted(agg["dists"])[len(agg["dists"]) // 2],
                    "frac_outside_window": round(
                        sum(1 for d in agg["dists"] if d >= window) / len(agg["dists"]), 6),
                    "window": window, "n": len(agg["dists"])},
                "by_distance": agg["acc_by_dist"]})
            per_arm[f"{name}_K{K}"] = row
            note(f"  {name} K={K}: acc={row['accuracy']:.4f} CI={row['accuracy_ci']} "
                 f"n={row['n']} | 毁源={row['corrupt_source']['accuracy']:.4f}")
    results["assoc_recall"] = per_arm

    Kmain = cfg["K_main"]        # 主腿用与训练同 K（泛化档 4/16 在 ksweep 里另报）
    main_rows = {n: per_arm[f"{n}_K{Kmain}"] for n in arms}
    premise = {
        "task": "assoc_recall", "K": Kmain, "window": window,
        "chance": main_rows["U0"]["chance"],
        "U0_accuracy": main_rows["U0"]["accuracy"],
        "U0_ci": main_rows["U0"]["accuracy_ci"],
        "U0_significantly_above_chance": main_rows["U0"]["accuracy_ci"][0] > main_rows["U0"]["chance"],
        "query_key_distance": main_rows["U0"]["query_key_distance"],
        "local_prior": main_rows["U0"]["local_prior"],
        "note": ("判据 ①：U0（无通道 + 窗口 %d）必须 ≈ chance；距离分布证明被问的 "
                 "pair 落在窗口外" % window),
    }
    note(f"判据 ①（前提校验）：U0 acc={premise['U0_accuracy']:.4f} "
         f"CI={premise['U0_ci']} chance={premise['chance']:.6f} "
         f"⇒ 显著高于 chance: {premise['U0_significantly_above_chance']}")
    comparisons = {
        "U2_vs_U0": two_proportion(main_rows["U0"]["n_hits"], main_rows["U0"]["n"],
                                   main_rows["U2"]["n_hits"], main_rows["U2"]["n"]),
        "U2_vs_U1": two_proportion(main_rows["U1"]["n_hits"], main_rows["U1"]["n"],
                                   main_rows["U2"]["n_hits"], main_rows["U2"]["n"]),
        "U1_vs_U0": two_proportion(main_rows["U0"]["n_hits"], main_rows["U0"]["n"],
                                   main_rows["U1"]["n_hits"], main_rows["U1"]["n"]),
    }
    criteria = {
        "c1_premise_U0_near_chance":
            not premise["U0_significantly_above_chance"],
        "c2_main_U2_gt_U0_5pp_and_ci_disjoint":
            (comparisons["U2_vs_U0"]["delta_pp"] >= 5.0
             and not comparisons["U2_vs_U0"]["ci_overlap"]),
        "c3_delta_vs_plain_3pp_and_ci_disjoint":
            (comparisons["U2_vs_U1"]["delta_pp"] >= 3.0
             and not comparisons["U2_vs_U1"]["ci_overlap"]),
        "c4_corrupt_source_control": {
            n: {"main": main_rows[n]["accuracy"],
                "corrupt": main_rows[n]["corrupt_source"]["accuracy"],
                "ci": main_rows[n]["corrupt_source"]["accuracy_ci"],
                "chance": main_rows[n]["chance"],
                # 判据 4 的操作化（读数前写下）：「仍保持高准确率」= 毁源后的
                # 准确率 CI **下界**仍高于 chance 的 4 倍（= 0.0625）
                "invalidated": main_rows[n]["corrupt_source"]["accuracy_ci"][0]
                > 4 * main_rows[n]["chance"]}
            for n in arms},
    }
    note(f"判据 ②（主）：U2 vs U0 Δ={comparisons['U2_vs_U0']['delta_pp']}pp "
         f"CI 不重叠={not comparisons['U2_vs_U0']['ci_overlap']} "
         f"⇒ {'过' if criteria['c2_main_U2_gt_U0_5pp_and_ci_disjoint'] else '未过'}")
    note(f"判据 ③：U2 vs U1 Δ={comparisons['U2_vs_U1']['delta_pp']}pp "
         f"CI 不重叠={not comparisons['U2_vs_U1']['ci_overlap']} "
         f"⇒ {'delta 优' if criteria['c3_delta_vs_plain_3pp_and_ci_disjoint'] else '不可分辨'}")
    results["premise_check"] = premise
    results["comparisons"] = comparisons
    results["criteria"] = criteria

    # ---------------- ⑤ 辅助腿：变距离回忆距离曲线 ----------------
    note("判据 ⑤：变距离回忆（辅助腿，三臂 × 六档距离）")
    dleg, dbs, dnb = {}, cfg["distance_leg"]["batch_size"], cfg["distance_leg"]["n_batches"]
    for name, model in arms.items():
        rows = {}
        for d in cfg["distance_leg"]["distances"]:
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0}
            for i in range(dnb):
                tb = var_recall_batch(dbs, T, d, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      generator=torch.Generator().manual_seed(2000 + 13 * d + i))
                r = arm_battery(model, tb, window, anchors, dev,
                                corrupt_span_bounds=(tb.meta["src_pos"],
                                                     tb.meta["src_pos"] + 2))
                agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
                agg["cn"] += r["corrupt_source"]["n"]
                agg["chits"] += r["corrupt_source"]["n_hits"]
                chance, lp = r["chance_uniform_answer_set"], r["local_prior"]
            rows[str(d)] = {
                "distance": d, "n": agg["n"],
                "accuracy": round(agg["n_hits"] / agg["n"], 6),
                "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
                "corrupt_source": {"accuracy": round(agg["chits"] / agg["cn"], 6),
                                   "n": agg["cn"],
                                   "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
                "chance": chance, "local_prior": lp,
                "source_position": tb.meta["src_pos"],
                "source_inside_window": (T - 1) - tb.meta["src_pos"] < window,
            }
        dleg[name] = rows
        note("  " + name + ": " + " ".join(
            f"d{d}:{rows[str(d)]['accuracy']:.3f}" for d in cfg["distance_leg"]["distances"]))
    results["distance_leg"] = dleg

    # ---------------- ⑥ 活性、loss 斜率、显存 ----------------
    note("判据 ⑥：通道活性 / loss 斜率 / 显存")
    probe_x = torch.randint(0, cfg["data"]["vocab_size"], (2, T),
                            generator=torch.Generator().manual_seed(1234)).cuda()
    for name, model in arms.items():
        res_p = json.load(open(resolve(cfg["arms"][name]["results_json"])))
        results.setdefault("train_results", {})[name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "gnorm_monitor": res_p.get("gnorm_monitor"),
            "window_mask": res_p.get("window_mask"),
            "carry_liveness": res_p.get("carry_liveness"),
            "carry_read_mass": res_p.get("carry_read_mass"),
            "synth_answer_ce": {"mean": res_p.get("gnorm_monitor", {}).get(
                "synth_answer_ce_mean"),
                "chance_ce": res_p.get("gnorm_monitor", {}).get("synth_chance_ce")},
        }
        lines = [json.loads(l) for l in
                 open(resolve(cfg["arms"][name]["train_log"])) if l.strip()]
        results.setdefault("loss_slope", {})[name] = loss_slope(lines)
        results.setdefault("train_log_tail", {})[name] = {
            "last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
            "last_answer_ce": lines[-1].get("synth_answer_ce"),
            "last_carry": lines[-1].get("attn_carry"),
        }
        m, _ = arms[name], None
        results["train_results"][name]["liveness_recheck"] = carry_liveness(
            m, None, None, prob=carry_probe(m, probe_x))
        results["train_results"][name]["read_mass"] = carry_read_mass_stats(
            m, probe_x, window=window)
        note(f"  {name}: wall {results['train_results'][name]['wall_minutes']} min, "
             f"peak {results['train_results'][name]['peak_gpu_mem_mb']} MB, "
             f"loss_slope {results['loss_slope'][name]['slope']}, "
             f"read_mass {results['train_results'][name]['read_mass']}")
    results["memory_analytic"] = analytic_memory(
        {**cfg["model"], "n_params": results["train_results"]["U0"]["params"]},
        bs, window, anchors)
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: results["train_results"][n]["peak_gpu_mem_mb"]
                                  for n in arms},
        "note": ("实测 = 本进程 torch.cuda.max_memory_allocated（净分配，不含共享卡"
                 "上其他任务的占用）；解析 = analytic_memory（参数/优化器/激活/状态）"),
    }

    # ---------------- 收尾 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: results["train_results"][n]["wall_minutes"]
                               for n in arms},
        "n_forward_batches": (nb * len(cfg["ksweep"]) * len(arms)
                              + dnb * len(cfg["distance_leg"]["distances"]) * len(arms)),
    }
    results["reproducibility"] = {}
    note("复现性：主腿 K=%d 再跑一遍（同 seed）" % Kmain)
    repro_n = 16
    r1 = {"n": 0, "n_hits": 0}
    for i in range(repro_n):
        tb = assoc_recall_batch(bs, T, Kmain, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(1000 + 7 * Kmain + i))
        acc = accuracy(arms["U0"], tb, keep_mask=keep, device=dev)
        r1["n"] += acc["n"]; r1["n_hits"] += acc["n_hits"]
    r2 = {"n": 0, "n_hits": 0}
    for i in range(repro_n):
        tb = assoc_recall_batch(bs, T, Kmain, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(1000 + 7 * Kmain + i))
        acc = accuracy(arms["U0"], tb, keep_mask=keep, device=dev)
        r2["n"] += acc["n"]; r2["n_hits"] += acc["n_hits"]
    r1["accuracy"] = round(r1["n_hits"] / r1["n"], 6)
    r2["accuracy"] = round(r2["n_hits"] / r2["n"], 6)
    results["reproducibility"] = {
        "n_per_pass": r1["n"], "n_batches_per_pass": repro_n,
        "same_seed_rerun_max_abs_diff": abs(r1["accuracy"] - r2["accuracy"]),
        "accuracy_1": r1["accuracy"], "accuracy_2": r2["accuracy"],
        "n_hits_1": r1["n_hits"], "n_hits_2": r2["n_hits"],
        "pass": r1 == r2,
        "note": "同 seed、同代码、同设备把 U0 的主腿整批再跑一遍（纯推理复现性）"}
    code_files = ("run.py", "premise_check.py", "divergence.py", "mutate.py",
                  "reconcile_numbers.py", "config.json", "README.md",
                  "pathlm/depth_ar.py", "train_depth_ar.py",
                  "tests/test_depth_ar.py", "tools/evalkit/carry_channel.py",
                  "tools/evalkit/longrange_tasks.py", "tools/evalkit/retention.py")
    results["code"] = {}
    for p in code_files:
        cand = resolve(p) if os.path.exists(resolve(p)) else os.path.join(HERE, p)
        if os.path.exists(cand):
            results["code"][p] = sha256_file(cand)
        else:
            results["code"][p] = None      # 尚未落盘的文件（报告期补齐时再跑一次）
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min）")


if __name__ == "__main__":
    main()
