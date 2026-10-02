"""2620 · 容量轴「收敛后」是否有效——评测与分析入口（纯推理）。

契约：同目录 `README.md`（判据线 §4 预登记，读数前写下，不许改）。
训练：`python3 train_depth_ar.py <S16|S128|S16_rep> ... --carry-state-dim <16|128|16>`
（见 `MODEL.md` 与 `train_all.sh`）。本入口做：

  ① 仪器/结构自检：释放版（is_causal）== 显式因果掩码（逐位）；模型自带窗口掩码前向 ==
     `masked_logits` + `window_anchor_keep_mask`（训练/评测同一件仪器）；三臂**注意力头逐位
     不变**、状态宽度与总状态浮点 = 256·dh、主干张量逐位相同的同 seed 构造对照；
  ② 3 臂 × 4 难度档（K=4/8/16/32）的**关联回忆准确率 + Wilson CI + chance + 毁源对照臂**；
  ③ 判据 ①（门禁）：`synth_answer_ce` 末窗线性斜率（+ 总 CE 与 log-log 三套并报，
     并复算 U1 的 10k 日志作为「总 CE 口径恒真」的证据）；
  ④ 判据 ②：主腿 `K=8` 的 `capacity_verdict`（≥ +3pp 且 CI 不重叠），四个 K 档全报；
  ⑤ 判据 ③：毁源作废线、**训练侧重跑散度**（`S16` vs `S16_rep` 同步骤差）；
  ⑥ 活性量、解析/实测内存分开报、成本与复现性。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~2 min）
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
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             carry_liveness, carry_probe,
                             carry_read_mass_stats, window_anchor_keep_mask)
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_multi_batch)
from tools.evalkit.retention import masked_logits                  # noqa: E402
from tools.evalkit.carry_channel import (                          # noqa: E402
    wilson_ci, two_proportion, arm_battery, analytic_memory)
import capacity_converged_lib as L                                 # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str):
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
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    levels = [tuple(x) for x in cfg["difficulty_levels"]]
    arms_cfg, order = cfg["arms"], cfg["arm_order"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载三臂（sha256 断言）")
    arms, ck = {}, {}
    for name in order:
        arms[name], ck[name] = load_arm(arms_cfg[name], cfg["model"],
                                        cfg["data"]["vocab_size"], dev)
        note(f"{name}: dh={ck[name]['carry_state_dim']} {ck[name]['sha256'][:16]}…")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors}}

    # ---------------- ① 仪器 / 结构自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（逐位）")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        a1 = masked_logits(arms[order[0]], x_probe, None)
        a2 = masked_logits(arms[order[0]], x_probe, exact)
    note(f"  max|Δlogit| = {float((a1 - a2).abs().max()):.3e}")
    note("自检 ①：模型自带窗口掩码前向 vs masked_logits+window_anchor_keep_mask（逐位）")
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    with torch.no_grad():
        b1 = arms[order[0]](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms[order[0]], x_probe, keep)
    note(f"  max|Δlogit| = {float((b1 - b2).abs().max()):.3e}")
    note("自检 ①：同 seed 重新构造 —— 注意力头不变、主干逐位相同、只有记忆槽头结构变")
    built = {}
    for name in order:
        torch.manual_seed(cfg["seed"])
        built[name] = DepthARModel(DepthARConfig(
            **{**cfg["model"], "carry_rule": "delta",
               "carry_state_dim": arms_cfg[name]["carry_state_dim"]}),
            cfg["data"]["vocab_size"])
    ref = order[0]
    struct = {}
    for name in order:
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
    attn_set = {struct[n]["attn_heads"] for n in order}
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": float((a1 - a2).abs().max()),
        "model_window_forward_vs_mask_instrument_max_abs_logit": float((b1 - b2).abs().max()),
        "attention_heads_unchanged_across_arms": len(attn_set) == 1,
        "attention_heads": sorted(attn_set),
        "structure_by_arm": struct,
        "total_state_floats_equals_d_model_times_dh": all(
            struct[n]["carry_state_floats"] == cfg["model"]["d_model"]
            * struct[n]["carry_state_dim"] for n in order),
        "state_floats_map": {n: struct[n]["carry_state_floats"] for n in order},
        "note": ("容量轴只动记忆槽：注意力头数/主干张量逐位不变（同 seed 构造对照）；"
                 "总状态浮点 = d_model·dh 与 T 无关"),
    }
    note(f"  注意力头 {sorted(attn_set)}（应恒为 8）；状态浮点 "
         f"{ {n: struct[n]['carry_state_floats'] for n in order} }")

    # ---------------- ② 主电池：3 臂 × 4 难度档 + 毁源对照 ----------------
    note("判据 ②③：3 臂 × 4 难度档（含毁源对照臂）")
    grid = {}
    for name, model in arms.items():
        for (K, Q) in levels:
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0}
            meta_row, dists = {}, []
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
                 f"CI={row['accuracy_ci']} n={row['n']} | "
                 f"毁源={row['corrupt_source']['accuracy']:.4f}")
    results["grid"] = grid

    # ---------------- 前提校验：锚点格显著高于 chance（失败决策树 §10.1） ----------------
    anchor = grid[cfg["premise_cell"]]
    premise = {
        "cell": cfg["premise_cell"], "accuracy": anchor["accuracy"],
        "accuracy_ci": anchor["accuracy_ci"], "chance": anchor["chance"],
        "significantly_above_chance": anchor["accuracy_ci"][0] > anchor["chance"],
        "multiple_of_chance": round(anchor["accuracy"] / anchor["chance"], 3),
        "query_key_distance": anchor["query_key_distance"],
        "local_prior": anchor["local_prior"],
        "note": ("前提校验：小状态臂在 K=8（训练同档）必须显著高于 chance，否则先查"
                 "窗口/活性量，不进入容量判读（README §10.1）"),
    }
    results["premise_check"] = premise
    note(f"前提校验：{cfg['premise_cell']} acc={premise['accuracy']:.4f} "
         f"CI={premise['accuracy_ci']} chance={premise['chance']:.6f} ⇒ "
         f"显著高于 chance: {premise['significantly_above_chance']}")

    # ---------------- 判据 ① 收敛门禁（读训练日志） ----------------
    note("判据 ①：收敛门禁（synth_answer_ce 末窗线性斜率）+ 三套口径并报")
    win = tuple(cfg["convergence"]["window"])
    conv = {"window": list(win), "by_arm": {}, "u1_10k_reference": {},
            "gate_definitions": {
                "main_gate_metric": "synth_answer_ce（答案行 CE；唯一决定准确率的量）",
                "abs_slope_per_1k_max": L.CONVERGENCE_SLOPE_MAX,
                "rel_drop_max": L.CONVERGENCE_REL_DROP_MAX,
                "or_rule": "|slope| < 0.01/1k 或 相对下降 < 1%",
                "not_a_gate": "总 CE 末窗斜率与 log-log 斜率（照报，不作门禁）"},
            "rationale_total_ce_vacuous": (
                "总 CE 口径在本任务上恒真、无区分度：U1 四臂末窗 [6000,10000] 的总 CE 斜率"
                "只有 −0.00036~−0.00055 /1k（相对下降 0.85–1.00%），按合同字面在 10k 步就已"
                "「收敛」；同窗答案行 CE 斜率是 −0.027~−0.032 /1k（相对下降 6.8–8.0%），远未"
                "收敛。⇒ 采信答案行口径（编排方 2026-10-02 裁决）。")}
    for name in order:
        lines = read_jsonl(resolve(arms_cfg[name]["train_log"]))
        ansce_lin = L.linear_window(lines, "synth_answer_ce", *win)
        loss_lin = L.linear_window(lines, "loss", *win)
        conv["by_arm"][name] = {
            "n_log_lines": len(lines),
            "train_steps": lines[-1]["step"] if lines else None,
            "ansce_last_window_linear": ansce_lin,
            "total_loss_last_window_linear": loss_lin,
            "loglog": {
                "loss_[200,24000]": L.loglog_window(lines, "loss", 200, 24000),
                "loss_[16000,24000]": L.loglog_window(lines, "loss", 16000, 24000),
                "synth_answer_ce_[200,24000]": L.loglog_window(
                    lines, "synth_answer_ce", 200, 24000),
                "synth_answer_ce_[16000,24000]": L.loglog_window(
                    lines, "synth_answer_ce", 16000, 24000),
                "u1_method_window_[200,12000]": {
                    "loss": L.loglog_window(lines, "loss", 200, 12000),
                    "synth_answer_ce": L.loglog_window(
                        lines, "synth_answer_ce", 200, 12000)}},
            "gate": L.convergence_gate(ansce_lin),
        }
        note(f"  {name}: ansCE 末窗 slope={ansce_lin['slope_per_1k']}/1k "
             f"rel={ansce_lin['rel_drop']} ⇒ 收敛={conv['by_arm'][name]['gate']['converged']}"
             f" | 总 CE slope={loss_lin['slope_per_1k']}/1k")
    # U1 的 10k 日志：同口径复算（证明「总 CE 口径恒真」不是我们编的）
    for arm, key in (("D16", "S16"), ("D128", "S128")):
        p = resolve(cfg["u1_reference"]["train_log_template"].format(arm=arm))
        if not os.path.exists(p):
            conv["u1_10k_reference"][arm] = {"missing": p}
            continue
        lines = read_jsonl(p)
        conv["u1_10k_reference"][arm] = {
            "source": p, "n_log_lines": len(lines),
            "total_loss_[6000,10000]_linear": L.linear_window(lines, "loss", 6000, 10000),
            "ansce_[6000,10000]_linear": L.linear_window(
                lines, "synth_answer_ce", 6000, 10000),
            "loglog_loss_[200,10000]": L.loglog_window(lines, "loss", 200, 10000),
        }
        note(f"  U1/{arm} 10k: 总 CE slope="
             f"{conv['u1_10k_reference'][arm]['total_loss_[6000,10000]_linear']['slope_per_1k']}"
             f"/1k  vs 答案行 slope="
             f"{conv['u1_10k_reference'][arm]['ansce_[6000,10000]_linear']['slope_per_1k']}/1k")
    results["convergence"] = conv
    conv_both = all(conv["by_arm"][a]["gate"]["converged"]
                    for a in cfg["main_arms"])
    note(f"判据 ① 门禁：两主臂均收敛 = {conv_both}")

    # ---------------- 判据 ② 主判据（四个 K 档全报，主腿 K=8） ----------------
    small, big = cfg["main_arms"]
    main_by_K, main_leg = {}, None
    for (K, Q) in levels:
        a, b = grid[f"{small}_K{K}"], grid[f"{big}_K{K}"]
        v = L.capacity_verdict(a["accuracy"], a["n"], b["accuracy"], b["n"],
                               a["accuracy_ci"], b["accuracy_ci"])
        # 符号约定与判据线一致：Δ = 大状态 − 小状态（正 = 大状态更好）。
        # `two_proportion(k_a, n_a, k_b, n_b)` 返回 delta = p_b − p_a ⇒ 传（小, 大）。
        v["two_proportion"] = two_proportion(a["n_hits"], a["n"],
                                              b["n_hits"], b["n"])
        v["two_proportion"]["note"] = ("ci_a = 小状态臂，ci_b = 大状态臂；"
                                        "delta_pp = 大 − 小（与主判据同号）")
        v["is_main_leg"] = (K == L.MAIN_LEVEL_K)
        v["gated_by_convergence"] = conv_both
        main_by_K[f"K{K}"] = {"small_arm": small, "big_arm": big, "verdict": v}
        if K == L.MAIN_LEVEL_K:
            main_leg = v
        note(f"判据 ② K={K}: Δ={v['delta_pp']}pp CI重叠={v['ci_overlap']} "
             f"⇒ 容量轴有效={v['capacity_axis_effective']}"
             f"{'（主腿）' if v['is_main_leg'] else ''}")
    results["criteria"] = {
        "c1_convergence_gate": {
            "main_arms": cfg["main_arms"], "both_converged": conv_both,
            "per_arm": {a: conv["by_arm"][a]["gate"] for a in cfg["main_arms"]},
            "note": "门禁 = 答案行 CE 末窗线性斜率（README §4①，编排方裁决口径）"},
        "c2_main_by_K": main_by_K,
        "c2_main_leg_K": L.MAIN_LEVEL_K,
        "c2_capacity_axis_effective": bool(
            conv_both and main_leg and main_leg["capacity_axis_effective"]),
        "c3_1_corrupt": {
            f"{n}_K{K}": {"arm": n, "dh": grid[f"{n}_K{K}"]["dh"], "K": K,
                          "main": grid[f"{n}_K{K}"]["accuracy"],
                          "main_ci": grid[f"{n}_K{K}"]["accuracy_ci"],
                          "corrupt": grid[f"{n}_K{K}"]["corrupt_source"]["accuracy"],
                          "accuracy_ci": grid[f"{n}_K{K}"]["corrupt_source"]["accuracy_ci"],
                          "chance": grid[f"{n}_K{K}"]["chance"],
                          "invalidated": L.corrupt_invalidates(
                              grid[f"{n}_K{K}"]["corrupt_source"]["accuracy"],
                              grid[f"{n}_K{K}"]["corrupt_source"]["accuracy_ci"])}
            for n in order for (K, _) in levels},
    }
    n_invalid = sum(1 for v in results["criteria"]["c3_1_corrupt"].values()
                    if v["invalidated"])
    results["criteria"]["c3_1_n_invalidated_cells"] = n_invalid
    note(f"判据 ③.1（毁源对照）：作废格数 = {n_invalid}（应为 0）")

    # ---------------- 判据 ③.2 训练侧重跑散度 ----------------
    note("判据 ③.2：S16 vs S16_rep 训练侧散度（同步骤 loss / 答案行 CE）")
    rep_pairs = cfg.get("rerun_pairs", [])
    div = {}
    for orig, rep in rep_pairs:
        la = read_jsonl(resolve(arms_cfg[orig]["train_log"]))
        lb = read_jsonl(resolve(arms_cfg[rep]["train_log"]))
        d = L.training_divergence(la, lb)
        ko = grid[f"{orig}_K{L.MAIN_LEVEL_K}"]
        kr = grid[f"{rep}_K{L.MAIN_LEVEL_K}"]
        d["accuracy_at_main_leg"] = {
            "K": L.MAIN_LEVEL_K,
            "orig": {"arm": orig, "accuracy": ko["accuracy"], "ci": ko["accuracy_ci"],
                     "n": ko["n"]},
            "rerun": {"arm": rep, "accuracy": kr["accuracy"], "ci": kr["accuracy_ci"],
                      "n": kr["n"]},
            "delta_pp": round(100 * (kr["accuracy"] - ko["accuracy"]), 3),
            "ci_overlap": not (kr["accuracy_ci"][0] > ko["accuracy_ci"][1]
                               or ko["accuracy_ci"][0] > kr["accuracy_ci"][1]),
        }
        d["two_proportion"] = two_proportion(ko["n_hits"], ko["n"],
                                             kr["n_hits"], kr["n"])
        d["two_proportion"]["note"] = ("ci_a = 原跑，ci_b = 重跑；"
                                        "delta_pp = 重跑 − 原跑（与 accuracy_at_main_leg 同号）")
        d["verdict"] = ("训练侧散度 ≥ 判据线（同配置两跑 CI 不重叠）⇒ 本尺度 3pp 不可分辨"
                        if not d["accuracy_at_main_leg"]["ci_overlap"]
                        else "同配置两跑 CI 重叠 ⇒ 训练侧散度不足以解释主判据")
        div[f"{orig}_vs_{rep}"] = d
        note(f"  {orig} vs {rep}: K=8 Δ={d['accuracy_at_main_leg']['delta_pp']}pp "
             f"CI重叠={d['accuracy_at_main_leg']['ci_overlap']} | loss max|Δ|="
             f"{d['per_key']['loss']['max_abs_diff']}")
    results["criteria"]["c3_2_training_divergence"] = div

    # ---------------- 判据 ④ 活性量 ----------------
    note("活性量：读训练侧 results.json（每 200 步落盘）+ 评测侧 carry_probe")
    probe_x = torch.randint(0, cfg["data"]["vocab_size"], (2, T),
                            generator=torch.Generator().manual_seed(1234)).cuda()
    results["train_results"], results["liveness_series"] = {}, {}
    for name, model in arms.items():
        res_p = json.load(open(resolve(arms_cfg[name]["results_json"])))
        lines = read_jsonl(resolve(arms_cfg[name]["train_log"]))
        series = [{"step": r["step"],
                   **(r.get("attn_carry") or {})} for r in lines]
        results["liveness_series"][name] = series
        results["train_results"][name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "steps": res_p.get("steps"),
            "train_tokens": res_p.get("train_tokens"),
            "steps_derived": (res_p.get("train_tokens") // (cfg["train_recipe"]["batch_size"]
                                                           * cfg["model"]["seq_len"])
                              if res_p.get("train_tokens") else None),
            "carry_liveness": res_p.get("carry_liveness"),
            "carry_read_mass": res_p.get("carry_read_mass"),
            "gnorm_monitor": {k: res_p.get("gnorm_monitor", {}).get(k)
                              for k in ("gnorm_max", "gnorm_mean", "grad_skips",
                                        "synth_answer_ce_mean", "synth_chance_ce",
                                        "window_mask_probe")},
        }
        results["train_results"][name]["liveness_recheck"] = carry_liveness(
            model, None, None, prob=carry_probe(model, probe_x))
        results["train_results"][name]["read_mass"] = carry_read_mass_stats(
            model, probe_x, window=window)
        note(f"  {name}: wall {results['train_results'][name]['wall_minutes']} min, "
             f"peak {results['train_results'][name]['peak_gpu_mem_mb']} MB, "
             f"steps={results['train_results'][name]['steps_derived']}, "
             f"read_share_last={series[-1].get('read_share') if series else None}")
    results["memory_analytic"] = analytic_memory(
        {**cfg["model"], "n_params": results["train_results"][small]["params"]},
        bs, window, anchors)
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: results["train_results"][n]["peak_gpu_mem_mb"]
                                  for n in order},
        "note": ("实测 = torch.cuda.max_memory_allocated（净分配，不含共享卡其他任务）；"
                 "解析 = analytic_memory（参数/优化器/激活/状态）"),
    }

    # ---------------- U1 10k 横向参照（只读引用，不复现） ----------------
    note("U1 10k 臂横向参照（只读引用，不复现）")
    u1 = {"source_results": cfg["u1_reference"]["results_json"],
          "source_commit": cfg["u1_reference"]["commit"],
          "grid_by_arm_K": {}, "note": ("本实验**不复现** U1 的 10k 读数，直接引用其 "
                                        "results.json 作横向参照（同 seed 同配方，唯一差别 = 步数）")}
    u1p = resolve(cfg["u1_reference"]["results_json"])
    if os.path.exists(u1p):
        u1res = json.load(open(u1p))
        for arm_u1, arm_here in (("D16", "S16"), ("D128", "S128")):
            for (K, _) in levels:
                src = u1res.get("grid", {}).get(f"{arm_u1}_K{K}")
                if src:
                    u1["grid_by_arm_K"][f"{arm_here}_K{K}"] = {
                        "accuracy": src["accuracy"], "accuracy_ci": src["accuracy_ci"],
                        "n": src["n"], "n_hits": src["n_hits"],
                        "corrupt": src["corrupt_source"]["accuracy"]}
        u1["u1_loss_slope_loglog"] = u1res.get("loss_slope")
    results["u1_10k_reference"] = u1

    # 与 U1 的 10k 跑**同难度档**横向对照（本实验 24k vs U1 10k，同 seed 同配方）
    u1_cmp = {"note": ("每格：U1 的 10k 读数 / 本实验的 24k 读数 / Δpp（本实验 − U1）与 CI 是否重叠；"
                        "**直接测点，不外推**——两者差异同时包含「步数」与「训练侧散度」两个因素"),
              "by_arm_K": {}}
    for arm_u1, arm_here in (("D16", "S16"), ("D128", "S128")):
        for (K, _) in levels:
            src = u1["grid_by_arm_K"].get(f"{arm_here}_K{K}")
            if not src:
                continue
            now = grid[f"{arm_here}_K{K}"]
            tp = two_proportion(src["n_hits"], src["n"], now["n_hits"], now["n"])
            u1_cmp["by_arm_K"][f"{arm_here}_K{K}"] = {
                "u1_arm": arm_u1, "u1_accuracy": src["accuracy"],
                "u1_ci": src["accuracy_ci"], "u1_n": src["n"],
                "now_accuracy": now["accuracy"], "now_ci": now["accuracy_ci"],
                "now_n": now["n"],
                "delta_pp_now_minus_u1": round(100 * (now["accuracy"] - src["accuracy"]), 3),
                "ci_overlap": not (now["accuracy_ci"][0] > src["accuracy_ci"][1]
                                   or src["accuracy_ci"][0] > now["accuracy_ci"][1]),
                "two_proportion": tp,
            }
    results["u1_10k_vs_now"] = u1_cmp

    # 与 U1 的 10k 跑**同步骤** loss / 答案行 CE 差（第三次/第四次散度读数，只读引用）
    u1_step = {"note": ("本实验两臂前 10k 步与 U1 同配置臂在**同步骤**上的 loss / 答案行 CE 差；"
                        "U1 的日志只读引用（不复现）"), "by_pair": {}}
    for arm_u1, arm_here in (("D16", "S16"), ("D128", "S128")):
        p = resolve(cfg["u1_reference"]["train_log_template"].format(arm=arm_u1))
        if not os.path.exists(p):
            u1_step["by_pair"][f"{arm_here}_vs_{arm_u1}"] = {"missing": p}
            continue
        mine = read_jsonl(resolve(arms_cfg[arm_here]["train_log"]))
        theirs = read_jsonl(p)
        d = L.training_divergence(mine, theirs)
        src = u1["grid_by_arm_K"].get(f"{arm_here}_K{L.MAIN_LEVEL_K}")
        u1_step["by_pair"][f"{arm_here}_vs_{arm_u1}"] = {
            "source": p,
            "loss": d["per_key"]["loss"],
            "synth_answer_ce": d["per_key"]["synth_answer_ce"],
            "u1_10k_accuracy_at_K8": (src or {}).get("accuracy"),
        }
    results["u1_10k_matched_step_divergence"] = u1_step

    # ---------------- 收尾：复现性 / 成本 / 哈希 ----------------
    note("复现性：主腿格同 seed 再跑一遍（纯推理）")
    K0 = L.MAIN_LEVEL_K
    Q0 = next(Q for K, Q in levels if K == K0)
    passes = []
    for _ in range(2):
        agg = {"n": 0, "n_hits": 0}
        for i in range(min(nb, 8)):
            tb = assoc_recall_multi_batch(
                bs, T, K0, Q0, sym_vocab=sym, filler_token=fill, cue_token=cue,
                generator=torch.Generator().manual_seed(5000 + 101 * K0 + 17 * i))
            r = arm_battery(arms[big], tb, window, anchors, dev)
            agg["n"] += r["n"]; agg["n_hits"] += r["n_hits"]
        passes.append(agg)
    results["reproducibility"] = {
        "cell": f"{big}_K{K0}", "n_per_pass": passes[0]["n"],
        "accuracy_1": round(passes[0]["n_hits"] / passes[0]["n"], 6),
        "accuracy_2": round(passes[1]["n_hits"] / passes[1]["n"], 6),
        "same_seed_rerun_max_abs_diff": abs(passes[0]["n_hits"] - passes[1]["n_hits"]) / passes[0]["n"],
        "pass": passes[0] == passes[1],
        "note": "同 seed/同代码/同设备把主腿格的前 8 个批再跑一遍（纯推理复现性）",
    }
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: results["train_results"][n]["wall_minutes"]
                               for n in order},
        "train_wall_minutes_total": round(sum(
            v for v in (results["train_results"][n]["wall_minutes"] for n in order)
            if isinstance(v, (int, float))), 3),
        "n_forward_batches": nb * len(levels) * len(order) * 2,
        "n_grid_cells": len(grid),
        "gpu_wall_budget_min": cfg["cost"]["gpu_wall_budget_min"],
    }
    code_files = ("run.py", "mutate.py", "reconcile_numbers.py", "config.json", "README.md",
                  "capacity_converged_lib.py", "test_capacity_converged.py",
                  "pathlm/depth_ar.py", "train_depth_ar.py", "tests/test_depth_ar.py",
                  "tools/evalkit/carry_channel.py", "tools/evalkit/longrange_tasks.py",
                  "tools/evalkit/retention.py")
    results["code"] = {}
    for p in code_files:
        # **本目录文件优先**（否则 `README.md` 这类同名文件会被解析成仓库根的 README；
        # 2026-10-02 自查发现：上一轮 `results.json::code.README.md` 记的是仓库根 README 的哈希，
        # 与本实验契约不是同一份 ⇒ 已修并重跑）
        local = os.path.join(HERE, p)
        cand = local if os.path.exists(local) else resolve(p)
        results["code"][p] = sha256_file(cand) if os.path.exists(cand) else None
    results["reconciliation"] = L.reconciliation_checks(results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min；"
         f"对账 all_consistent={results['reconciliation']['all_consistent']}）")


if __name__ == "__main__":
    main()
