"""2619 · 写入通道的难度—容量曲线——实验入口（评测电池，纯推理）。

契约：同目录 `README.md`（判据线 §4 预登记，不许改）。
训练：`python3 train_depth_ar.py <arm> ... --carry-write delta --carry-state-dim <dh>`
（见 `MODEL.md`）。本入口只做**评测**（G32 的显存与步时在训练侧实测）：

  ① 仪器/结构自检：释放版（is_causal）== 显式因果掩码（逐位）；模型自带窗口掩码前向 ==
     `masked_logits` + `window_anchor_keep_mask`（训练/评测同一件仪器）；四臂**注意力头逐位
     不变**、状态宽度与总状态浮点 = 256·dh、主干张量逐位相同的同 seed 构造对照；
  ② 4 臂 × 4 难度档（K=4/8/16/32）的**关联回忆准确率 + Wilson CI + chance + 毁源对照臂**；
  ③ 判据 ②：难度曲线（dh=32 固定，4 点）单调性/拐点 + 相邻点 CI；
  ④ 判据 ③：容量曲线（K=8 与 K=32 各 4 点）单调性/拐点；
  ⑤ 判据 ④：交换率实测点 K*(dh) 与 dh*(K)（阈值 = 4×chance，禁外推）；
  ⑥ loss 斜率、活性量、解析/实测内存分开报。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~3–5 min）
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
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             carry_liveness, carry_probe,
                             carry_read_mass_stats, window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_multi_batch)
from tools.evalkit.retention import masked_logits                  # noqa: E402
from tools.evalkit.carry_channel import (                          # noqa: E402
    wilson_ci, two_proportion, arm_battery, loss_slope, analytic_memory)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str) -> tuple[DepthARModel, dict]:
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, "carry_rule": "delta",
                           "carry_state_dim": spec["carry_state_dim"]})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "carry_state_dim": spec["carry_state_dim"]}


def monotone_report(rows: list[dict], axis: str) -> dict:
    """曲线判读（判据 ②/③）：沿 `axis` 排序，报单调性 / 拐点 / 不可分辨段。

    - `monotone_non_increasing` / `monotone_non_decreasing`：只看**Wilson CI 不重叠**
      的相邻对；CI 重叠的对记「不可分辨」，不参与方向判定。
    - `knee_detected`：相邻步长 `|Δ|` 的**内部**最大步 ≥ 2× 最小步（肘部在曲线中段）。
    - `monotone_or_knee`：判据 ②/③ 的「单调或明确拐点」总旗标。
    - `indistinguishable`：没有任何可分辨相邻对 ⇒ 整条曲线不可分辨。
    """
    pts = sorted(rows, key=lambda r: r[axis])
    vals = [r["accuracy"] for r in pts]
    ci = [r["accuracy_ci"] for r in pts]
    adj = []
    for i in range(len(pts) - 1):
        lo, hi = ci[i], ci[i + 1]
        adj.append({
            "from": pts[i][axis], "to": pts[i + 1][axis],
            "delta_pp": round(100 * (vals[i + 1] - vals[i]), 3),
            "ci_overlap": not (lo[0] > hi[1] or hi[0] > lo[1]),
        })
    resolvable = [a for a in adj if not a["ci_overlap"]]
    ups = [a["delta_pp"] for a in resolvable if a["delta_pp"] > 0]
    downs = [a["delta_pp"] for a in resolvable if a["delta_pp"] < 0]
    monotone_up = bool(ups) and not downs
    monotone_down = bool(downs) and not ups
    absd = [abs(a["delta_pp"]) for a in adj]
    interior = absd[1:-1] if len(absd) >= 3 else []
    knee = bool(absd) and max(absd) > 0 and (
        (interior and max(interior) == max(absd) and max(absd) >= 2 * min(absd))
    )
    return {
        "axis": axis, "points": pts, "adjacent": adj,
        "n_resolvable_pairs": len(resolvable),
        "monotone_non_increasing": monotone_down,
        "monotone_non_decreasing": monotone_up,
        "knee_detected": knee,
        "monotone_or_knee": bool(monotone_up or monotone_down or knee),
        "indistinguishable": len(resolvable) == 0,
        "note": ("相邻点 Wilson CI 重叠记「不可分辨」、不计入方向判定；"
                 "knee = 曲线中段的相邻步长 ≥2× 最小步"),
    }


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
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    sym, fill, cue = cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"], cfg["tokens"]["cue_token"]
    levels = cfg["difficulty_levels"]           # [(K, Q), ...]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载四臂（sha256 断言）")
    arms, ck = {}, {}
    for name in cfg["arm_order"]:
        arms[name], ck[name] = load_arm(cfg["arms"][name], cfg["model"],
                                        cfg["data"]["vocab_size"], dev)
        note(f"{name}: dh={ck[name]['carry_state_dim']} "
             f"{ck[name]['sha256'][:16]}…")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors}}

    # ---------------- ① 仪器 / 结构自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（逐位）")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        a1 = masked_logits(arms[cfg["arm_order"][0]], x_probe, None)
        a2 = masked_logits(arms[cfg["arm_order"][0]], x_probe, exact)
    note(f"  max|Δlogit| = {float((a1 - a2).abs().max()):.3e}")
    note("自检 ①：模型自带窗口掩码前向 vs masked_logits+window_anchor_keep_mask（逐位）")
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    with torch.no_grad():
        b1 = arms[cfg["arm_order"][0]](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms[cfg["arm_order"][0]], x_probe, keep)
    note(f"  max|Δlogit| = {float((b1 - b2).abs().max()):.3e}")
    note("自检 ①：同 seed 重新构造 —— 注意力头不变、主干逐位相同、只有记忆槽头结构变")
    built = {}
    for name in cfg["arm_order"]:
        torch.manual_seed(cfg["seed"])
        built[name] = DepthARModel(DepthARConfig(
            **{**cfg["model"], "carry_rule": "delta",
               "carry_state_dim": cfg["arms"][name]["carry_state_dim"]}),
            cfg["data"]["vocab_size"])
    ref = cfg["arm_order"][0]
    struct = {}
    for name in cfg["arm_order"]:
        m = built[name]
        sa, sb = built[ref].state_dict(), m.state_dict()
        carry_names = {k for k in sa if k.startswith("carry.") or k.endswith("carry_scale")}
        struct[name] = {
            "carry_state_dim": m.carry.head_dim,
            "carry_heads": m.carry.n_heads,
            "carry_state_floats": m.carry.n_heads * m.carry.head_dim ** 2,
            "attn_heads": m.blocks[0].n_heads,
            "n_params": sum(p.numel() for p in m.parameters()),
            "trunk_bitwise_equal_to_ref": all(
                torch.equal(sa[k], sb[k]) for k in sa if k not in carry_names),
        }
    attn_set = {struct[n]["attn_heads"] for n in cfg["arm_order"]}
    state_floats = {struct[n]["carry_state_dim"]: struct[n]["carry_state_floats"]
                    for n in cfg["arm_order"]}
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": float((a1 - a2).abs().max()),
        "model_window_forward_vs_mask_instrument_max_abs_logit": float((b1 - b2).abs().max()),
        "attention_heads_unchanged_across_arms": len(attn_set) == 1,
        "attention_heads": sorted(attn_set),
        "structure_by_arm": struct,
        "total_state_floats_equals_d_model_times_dh": all(
            struct[n]["carry_state_floats"] == cfg["model"]["d_model"]
            * struct[n]["carry_state_dim"] for n in cfg["arm_order"]),
        "state_floats_map": state_floats,
        "note": ("容量轴只动记忆槽：注意力头数/主干张量逐位不变（同 seed 构造对照）；"
                 "总状态浮点 = d_model·dh 与 T 无关"),
    }
    note(f"  注意力头 {sorted(attn_set)}（应恒为 8）；状态浮点 "
         f"{ {n: struct[n]['carry_state_floats'] for n in cfg['arm_order']} }")

    # ---------------- ② 主电池：4 臂 × 4 难度档 + 毁源对照 ----------------
    note("判据 ①–⑤：4 臂 × 4 难度档（含毁源对照臂）")
    grid = {}
    for name, model in arms.items():
        for (K, Q) in levels:
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0}
            meta_row = {}
            dists = []
            for i in range(nb):
                tb = assoc_recall_multi_batch(
                    bs, T, K, Q, sym_vocab=sym, filler_token=fill, cue_token=cue,
                    generator=torch.Generator().manual_seed(5000 + 101 * K + 17 * i))
                r = arm_battery(model, tb, window, anchors, dev,
                                corrupt_span_bounds=(0, 2 * K))
                agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
                agg["cn"] += r["corrupt_source"]["n"]
                agg["chits"] += r["corrupt_source"]["n_hits"]
                dists.extend(int(v) for v in tb.dists.flatten().tolist())
                meta_row = {"chance": r["chance_uniform_answer_set"],
                            "local_prior": r["local_prior"]}
            row = {
                "arm": name, "dh": ck[name]["carry_state_dim"], "K": K, "Q": Q,
                "n": agg["n"], "n_hits": agg["n_hits"],
                "accuracy": round(agg["n_hits"] / agg["n"], 6),
                "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
                "chance": meta_row["chance"], "local_prior": meta_row["local_prior"],
                "corrupt_source": {
                    "span": [0, 2 * K], "n": agg["cn"], "n_hits": agg["chits"],
                    "accuracy": round(agg["chits"] / agg["cn"], 6),
                    "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
                "query_key_distance": {
                    "min": min(dists), "max": max(dists),
                    "median": sorted(dists)[len(dists) // 2], "window": window},
            }
            grid[f"{name}_K{K}"] = row
            note(f"  {name} dh={row['dh']} K={K} Q={Q}: acc={row['accuracy']:.4f} "
                 f"CI={row['accuracy_ci']} n={row['n']} | 毁源={row['corrupt_source']['accuracy']:.4f}")
    results["grid"] = grid

    # ---------------- 判据 ①：前提校验 ----------------
    anchor = grid[cfg["premise_cell"]]
    premise = {
        "cell": cfg["premise_cell"], "accuracy": anchor["accuracy"],
        "accuracy_ci": anchor["accuracy_ci"], "chance": anchor["chance"],
        "significantly_above_chance": anchor["accuracy_ci"][0] > anchor["chance"],
        "multiple_of_chance": round(anchor["accuracy"] / anchor["chance"], 3),
        "query_key_distance": anchor["query_key_distance"], "local_prior": anchor["local_prior"],
        "note": "判据 ①：dh=32/K=8 臂（ticket 35 的 U2 口径）必须显著高于 chance",
    }
    note(f"判据 ①：{cfg['premise_cell']} acc={premise['accuracy']:.4f} "
         f"CI={premise['accuracy_ci']} chance={premise['chance']:.6f} ⇒ "
         f"显著高于 chance: {premise['significantly_above_chance']}")
    results["premise_check"] = premise

    # ---------------- 判据 ②③：两条曲线 ----------------
    dif_arm = cfg["difficulty_arm"]
    difficulty = monotone_report([grid[f"{dif_arm}_K{K}"] for K, _ in levels], "K")
    capacity = {f"K{K}": monotone_report(
        [grid[f"{n}_K{K}"] for n in cfg["arm_order"]], "dh") for K, _ in levels}
    results["curves"] = {
        "difficulty": {"arm": dif_arm, **difficulty},
        "capacity": capacity,
        "capacity_main_K": cfg["capacity_main_K"],
    }
    note(f"判据 ②（难度曲线，arm={dif_arm}）：单调升={difficulty['monotone_non_decreasing']} "
         f"单调降={difficulty['monotone_non_increasing']} 拐点={difficulty['knee_detected']} "
         f"⇒ 单调或拐点={difficulty['monotone_or_knee']}")
    for K, _ in levels:
        c = capacity[f"K{K}"]
        note(f"判据 ③（容量曲线 K={K}）：单调升={c['monotone_non_decreasing']} "
             f"单调降={c['monotone_non_increasing']} 拐点={c['knee_detected']}")

    # ---------------- 判据 ④：交换率实测点 ----------------
    thr = cfg["exchange_threshold_mult"] * anchor["chance"]
    kstar, dhstar = {}, {}
    for name in cfg["arm_order"]:
        ks = sorted(K for K, _ in levels)
        passed = [K for K in ks
                  if grid[f"{name}_K{K}"]["accuracy_ci"][0] > thr]
        kstar[name] = {"dh": ck[name]["carry_state_dim"], "K_star": max(passed) if passed else 0,
                       "K_passed": passed,
                       "state_floats": struct[name]["carry_state_floats"]}
    for K, _ in levels:
        passed = [n for n in cfg["arm_order"]
                  if grid[f"{n}_K{K}"]["accuracy_ci"][0] > thr]
        dhstar[f"K{K}"] = {
            "dh_min_passing": min((ck[n]["carry_state_dim"] for n in passed), default=0),
            "arms_passing": passed}
    n_pts = len({v["K_star"] for v in kstar.values()})
    exchange = {
        "threshold": thr, "threshold_mult_of_chance": cfg["exchange_threshold_mult"],
        "definition": "K*(dh) = 使该格 Wilson CI 下界 > 4×chance 的最大 K；无外推",
        "K_star_by_arm": kstar, "dh_min_by_K": dhstar,
        "n_distinct_measured_points": n_pts,
        "meets_ge3_points": n_pts >= 3,
    }
    results["exchange_rate"] = exchange
    note(f"判据 ④（交换率）：K*(dh) = "
         f"{ {n: kstar[n]['K_star'] for n in cfg['arm_order']} }；"
         f"distinct={n_pts} ⇒ ≥3 点: {exchange['meets_ge3_points']}")

    # ---------------- 判据 ⑤：毁源对照判读 ----------------
    criteria = {
        "c1_premise_anchor_above_chance": premise["significantly_above_chance"],
        "c2_difficulty_monotone_or_knee": difficulty["monotone_or_knee"],
        "c3_capacity_monotone_or_knee":
            capacity[f"K{cfg['capacity_main_K']}"]["monotone_or_knee"],
        "c4_exchange_ge3_measured_points": exchange["meets_ge3_points"],
        "c5_corrupt_source_control": {
            f"{n}_K{K}": {
                "main": grid[f"{n}_K{K}"]["accuracy"],
                "corrupt": grid[f"{n}_K{K}"]["corrupt_source"]["accuracy"],
                "corrupt_ci": grid[f"{n}_K{K}"]["corrupt_source"]["accuracy_ci"],
                "chance": grid[f"{n}_K{K}"]["chance"],
                "invalidated": (grid[f"{n}_K{K}"]["corrupt_source"]["accuracy_ci"][0]
                                > 4 * grid[f"{n}_K{K}"]["chance"]),
            } for n in cfg["arm_order"] for K, _ in levels},
    }
    n_invalid = sum(1 for v in criteria["c5_corrupt_source_control"].values()
                    if v["invalidated"])
    criteria["c5_n_invalidated_cells"] = n_invalid
    results["criteria"] = criteria
    note(f"判据 ⑤（毁源对照）：作废格数 = {n_invalid}（应为 0）")

    # ---------------- ⑥ 活性、loss 斜率、显存 ----------------
    note("判据 ⑥：活性 / loss 斜率 / 显存")
    probe_x = torch.randint(0, cfg["data"]["vocab_size"], (2, T),
                            generator=torch.Generator().manual_seed(1234)).cuda()
    results["train_results"], results["loss_slope"] = {}, {}
    for name, model in arms.items():
        res_p = json.load(open(resolve(cfg["arms"][name]["results_json"])))
        results["train_results"][name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "carry_liveness": res_p.get("carry_liveness"),
            "carry_read_mass": res_p.get("carry_read_mass"),
            "gnorm_monitor": {k: res_p.get("gnorm_monitor", {}).get(k)
                              for k in ("gnorm_max", "gnorm_mean", "grad_skips",
                                        "synth_answer_ce_mean", "synth_chance_ce",
                                        "window_mask_probe")},
        }
        lines = [json.loads(l) for l in
                 open(resolve(cfg["arms"][name]["train_log"])) if l.strip()]
        results["loss_slope"][name] = loss_slope(lines)
        results["train_results"][name]["liveness_recheck"] = carry_liveness(
            model, None, None, prob=carry_probe(model, probe_x))
        results["train_results"][name]["read_mass"] = carry_read_mass_stats(
            model, probe_x, window=window)
        note(f"  {name}: wall {results['train_results'][name]['wall_minutes']} min, "
             f"peak {results['train_results'][name]['peak_gpu_mem_mb']} MB, "
             f"slope {results['loss_slope'][name]['slope']}")
    results["memory_analytic"] = analytic_memory(
        {**cfg["model"], "n_params": results["train_results"][cfg["arm_order"][1]]["params"]},
        bs, window, anchors)
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: results["train_results"][n]["peak_gpu_mem_mb"]
                                  for n in cfg["arm_order"]},
        "note": ("实测 = torch.cuda.max_memory_allocated（净分配，不含共享卡其他任务）；"
                 "解析 = analytic_memory（参数/优化器/激活/状态）"),
    }

    # ---------------- 收尾 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: results["train_results"][n]["wall_minutes"]
                               for n in cfg["arm_order"]},
        "n_forward_batches": nb * len(levels) * len(cfg["arm_order"]) * 2,
        "n_grid_cells": len(grid),
    }
    # 复现性：anchor 格整批再跑一遍（同 seed）
    note("复现性：anchor 格同 seed 再跑一遍")
    K0, Q0 = next((K, Q) for K, Q in levels if f"{cfg['difficulty_arm']}_K{K}"
                  == cfg["premise_cell"])
    passes = []
    for _ in range(2):
        agg = {"n": 0, "n_hits": 0}
        for i in range(8):
            tb = assoc_recall_multi_batch(
                bs, T, K0, Q0, sym_vocab=sym, filler_token=fill, cue_token=cue,
                generator=torch.Generator().manual_seed(5000 + 101 * K0 + 17 * i))
            r = arm_battery(arms[cfg["difficulty_arm"]], tb, window, anchors, dev)
            agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
        passes.append(agg)
    results["reproducibility"] = {
        "n_per_pass": passes[0]["n"], "accuracy_1": round(passes[0]["n_hits"] / passes[0]["n"], 6),
        "accuracy_2": round(passes[1]["n_hits"] / passes[1]["n"], 6),
        "same_seed_rerun_max_abs_diff": abs(passes[0]["n_hits"] - passes[1]["n_hits"]) / passes[0]["n"],
        "pass": passes[0] == passes[1],
        "note": "同 seed/同代码/同设备把 anchor 格 8 个批再跑一遍（纯推理复现性）",
    }
    code_files = ("run.py", "mutate.py", "reconcile_numbers.py", "config.json", "README.md",
                  "pathlm/depth_ar.py", "train_depth_ar.py", "tests/test_depth_ar.py",
                  "tools/evalkit/carry_channel.py", "tools/evalkit/longrange_tasks.py",
                  "tools/evalkit/retention.py")
    results["code"] = {}
    for p in code_files:
        cand = resolve(p) if os.path.exists(resolve(p)) else os.path.join(HERE, p)
        results["code"][p] = sha256_file(cand) if os.path.exists(cand) else None
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min）")


if __name__ == "__main__":
    main()
