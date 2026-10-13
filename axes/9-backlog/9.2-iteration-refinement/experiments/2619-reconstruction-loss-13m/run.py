"""2619 · 训练信号（重建损失）三臂对照 —— 实验入口（评测电池）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
训练：`python3 train_depth_ar.py <arm> ... --recon-window 16 --recon-lambda {0.0,1.0}`
（见 `MODEL.md` 的生成命令）。本入口只做**评测**（纯推理）+ 汇总，跑八件事：

  ① 仪器自检：释放版 == 显式因果掩码（逐位）；模型自带窗口掩码前向 ==
     `retention.masked_logits` + `window_anchor_keep_mask`（训练/评测同一件仪器）；
     **λ=0 实现级等价**（同 seed 建「有头/无头」两模型 ⇒ 主干参数与 logits 逐位）；
  ② 目标前提（本实验最关键的一条）：在**训练同分布**的批上，行 i 的目标恰好是
     `tokens[i−16]`，且在真实掩码下「行 i 不可见、行 i−1 可见」**零违例**，
     并给出目标的符号/filler 分布；
  ③ 前提校验 c0：A（无损失臂）必须 ≈ chance（+ 查询-key 距离分布 + 局部先验）；
  ④ 主判据 c1 + 泛化档：A/B/B_rep 的关联回忆准确率（Wilson CI）+ 两两比较；
  ⑤ 毁源对照臂 c2（每臂强制，评测期）；
  ⑥ 重跑臂 c3：B vs B_rep 的读数与训练日志散度；
  ⑦ 辅助腿：变距离回忆距离曲线（三臂 × 六档）+ clean CE 税（trunk / 答案行）；
  ⑧ 活性量：**已训练头**在固定批上的重建读数（A 臂 = 未训练头参照）、
     loss 斜率（同方法）、解析/实测显存分开报。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡 ~3–5 min）
  python3 run.py --smoke         # 冒烟：少批
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
                             window_anchor_keep_mask)
from pathlm.recon_signal import recon_loss, recon_targets          # noqa: E402
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_batch, assoc_recall_multi_batch, var_recall_batch,
    check_local_prior_resistance, accuracy)
from tools.evalkit.retention import masked_logits                  # noqa: E402
from tools.evalkit.carry_channel import (                          # noqa: E402
    wilson_ci, two_proportion, arm_battery, analytic_memory, loss_slope)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str) -> tuple:
    """加载一臂的权重并**断言 sha256**（权重不入库 ⇒ 哈希必须逐次核）。"""
    path = resolve(spec["path"])
    if not os.path.exists(path):
        raise SystemExit(f"缺权重 {spec['path']}（先训练 + 跑 fill_config_hashes.py）")
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**mcfg)
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "recon_lambda": spec.get("recon_lambda")}


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
        cfg["eval"] = {"batch_size": 16, "n_batches": 2}
        cfg["distance_leg"] = {**cfg["distance_leg"], "batch_size": 16,
                               "n_batches": 1, "distances": [16, 64]}
        cfg["ksweep"] = [8]
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    W = cfg["model"]["train_window"]
    anchors = cfg["model"]["train_anchors"]
    bs, nb = cfg["eval"]["batch_size"], cfg["eval"]["n_batches"]
    sym = cfg["tokens"]["sym_vocab"]
    fill = cfg["tokens"]["filler_token"]
    cue = cfg["tokens"]["cue_token"]
    Kmain = cfg["K_main"]
    arms_order = ["A", "B", "B_rep"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载三臂（sha256 断言）")
    arms, ck = {}, {}
    for name in arms_order:
        arms[name], ck[name] = load_arm(cfg["arms"][name], cfg["model"],
                                        cfg["data"]["vocab_size"], dev)
        note(f"{name}: {ck[name]['sha256'][:16]}… λ={ck[name]['recon_lambda']} "
             f"params={sum(p.numel() for p in arms[name].parameters())}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": W, "train_anchors": anchors,
                              "recon_window": cfg["recon"]["window"]}}

    # ---------------- ① 仪器自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（逐位）")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        a1 = masked_logits(arms["A"], x_probe, None)
        a2 = masked_logits(arms["A"], x_probe, exact)
    release_vs_exact = float((a1 - a2).abs().max())
    note(f"  max|Δlogit| = {release_vs_exact:.3e}")

    note("自检 ①：模型自带窗口掩码前向 vs masked_logits+window_anchor_keep_mask")
    keep = window_anchor_keep_mask(T, W, anchors, device=dev)
    with torch.no_grad():
        b1 = arms["A"](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms["A"], x_probe, keep)
    win_selfcheck = float((b1 - b2).abs().max())
    note(f"  max|Δlogit| = {win_selfcheck:.3e}")

    note("自检 ①：λ=0 实现级等价（同 seed 建「有头 / 无头」两模型）")
    built = {}
    for tag, rw in (("head", cfg["model"]["recon_window"]), ("nohead", 0)):
        torch.manual_seed(cfg["seed"])
        built[tag] = DepthARModel(DepthARConfig(**{**cfg["model"],
                                                   "recon_window": rw}),
                                  cfg["data"]["vocab_size"])
    p_head, p_none = built["head"].state_dict(), built["nohead"].state_dict()
    added = sorted(set(p_head) - set(p_none))
    trunk_equal = all(torch.equal(p_head[k], p_none[k]) for k in p_none)
    with torch.no_grad():
        l_head = built["head"](x_probe.cpu())[1]["depth_logits"][-1]
        l_none = built["nohead"](x_probe.cpu())[1]["depth_logits"][-1]
    logits_equal = torch.equal(l_head, l_none)
    note(f"  新增张量 {added}；主干逐位相同={trunk_equal}；"
         f"logits 逐位相同={logits_equal}")

    note("自检 ②：重建目标在**训练同分布**批上的前提（零违例才算设置成立）")
    tb_probe = assoc_recall_multi_batch(16, T, cfg["train_recipe"]["synth_pairs"],
                                       cfg["train_recipe"]["synth_queries"],
                                       sym_vocab=sym, filler_token=fill,
                                       cue_token=cue,
                                       gap=cfg["train_recipe"]["synth_gap"],
                                       generator=torch.Generator().manual_seed(1234))
    xp = tb_probe.tokens
    rows, targets = recon_targets(xp, cfg["recon"]["window"])
    keep_p = window_anchor_keep_mask(T, cfg["recon"]["window"], anchors)
    # 对角取值：[行 i, 列 i−W] —— 直接双索引，不要 `keep_p[rows][:, rows-W]`（那是矩阵）
    masked_readings = int((~keep_p[rows, rows - cfg["recon"]["window"]]).sum())
    visible_prev = int(keep_p[rows - 1, rows - cfg["recon"]["window"]].sum())
    n_rows = int(rows.numel())
    with torch.no_grad():
        _, aux_p = arms["A"](xp.cuda())
        rp = recon_loss(arms["A"].recon_head, aux_p["states"][-1], xp.cuda(),
                        cfg["recon"]["window"], fill)
    target_premise = {
        "rows": [int(rows.min()), int(rows.max())], "n_rows": n_rows,
        "target_equals_tokens_i_minus_W": bool(torch.equal(targets, xp[:, :T - cfg["recon"]["window"]])),
        "target_hidden_at_row_i": {"violations": n_rows - masked_readings,
                                   "expect": 0},
        "target_visible_at_row_i_minus_1": {"ok": visible_prev, "expect": n_rows},
        "target_split": {"n_symbol": int(rp["n_symbol"]), "n_filler": int(rp["n_filler"]),
                         "n_targets": int(rp["n_targets"])},
        "local_prior": check_local_prior_resistance(tb_probe, k=8),
        "note": ("目标 = 刚掉出可见窗口的那一格；两臂的目标完全一致，"
                 "`target_split` 说明非平凡（符号）目标占比"),
    }
    note(f"  目标行 {target_premise['rows']}；被遮住违例="
         f"{target_premise['target_hidden_at_row_i']['violations']}；"
         f"前一行可见 {visible_prev}/{n_rows}；符号目标 {int(rp['n_symbol'])}/"
         f"{int(rp['n_targets'])}；局部先验 passes={target_premise['local_prior']['passed']}")

    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit": release_vs_exact,
        "model_window_forward_vs_mask_instrument_max_abs_logit": win_selfcheck,
        "lambda0_implementation_equivalence": {
            "added_tensors": added, "trunk_bitwise_equal": trunk_equal,
            "logits_bitwise_equal": logits_equal,
            "n_params_head": sum(p.numel() for p in built["head"].parameters()),
            "n_params_nohead": sum(p.numel() for p in built["nohead"].parameters()),
            "note": ("同 seed 重建两模型：唯一差异 = recon_head.weight（建在参数 init "
                     "循环之后 ⇒ 不消耗其余参数的随机数）")},
        "recon_target_premise": target_premise,
    }

    # ---------------- ③④⑤ 主电池（关联回忆 + 毁源对照） ----------------
    note("判据 c0/c1/c2：关联回忆（主腿）+ 毁源对照臂")
    per_arm = {}
    for name in arms_order:
        for K in cfg["ksweep"]:
            batches = [assoc_recall_batch(bs, T, K, sym_vocab=sym,
                                          filler_token=fill, cue_token=cue,
                                          generator=torch.Generator().manual_seed(
                                              1000 + 7 * K + i))
                       for i in range(nb)]
            row = {"K": K, "n_batches": nb}
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0, "dists": [],
                   "acc_by_dist": {}}
            for tb in batches:
                r = arm_battery(arms[name], tb, W, anchors, dev,
                                corrupt_span_bounds=(0, 2 * K))
                agg["n"] += r["n"]
                agg["n_hits"] += r["n_hits"]
                agg["cn"] += r["corrupt_source"]["n"]
                agg["chits"] += r["corrupt_source"]["n_hits"]
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
                    "span": [0, 2 * K],
                    "accuracy": round(agg["chits"] / agg["cn"], 6),
                    "n": agg["cn"], "n_hits": agg["chits"],
                    "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
                "query_key_distance": {
                    "min": min(agg["dists"]), "max": max(agg["dists"]),
                    "median": sorted(agg["dists"])[len(agg["dists"]) // 2],
                    "frac_outside_window": round(
                        sum(1 for d in agg["dists"] if d >= W) / len(agg["dists"]), 6),
                    "window": W, "n": len(agg["dists"])},
                "by_distance": agg["acc_by_dist"]})
            per_arm[f"{name}_K{K}"] = row
            note(f"  {name} K={K}: acc={row['accuracy']:.4f} CI={row['accuracy_ci']} "
                 f"n={row['n']} | 毁源={row['corrupt_source']['accuracy']:.4f}")
    results["assoc_recall"] = per_arm

    main_rows = {n: per_arm[f"{n}_K{Kmain}"] for n in arms_order}
    comparisons = {
        "B_vs_A": two_proportion(main_rows["A"]["n_hits"], main_rows["A"]["n"],
                                 main_rows["B"]["n_hits"], main_rows["B"]["n"]),
        "Brep_vs_A": two_proportion(main_rows["A"]["n_hits"], main_rows["A"]["n"],
                                    main_rows["B_rep"]["n_hits"], main_rows["B_rep"]["n"]),
        "B_vs_Brep": two_proportion(main_rows["B_rep"]["n_hits"], main_rows["B_rep"]["n"],
                                    main_rows["B"]["n_hits"], main_rows["B"]["n"]),
    }
    chance = main_rows["A"]["chance"]
    premise = {
        "task": "assoc_recall", "K": Kmain, "window": W, "chance": chance,
        "A_accuracy": main_rows["A"]["accuracy"], "A_ci": main_rows["A"]["accuracy_ci"],
        "A_ceiling": round(max(3 * chance, 0.02), 6),
        "A_near_chance": main_rows["A"]["accuracy_ci"][0] <= max(3 * chance, 0.02),
        "query_key_distance": main_rows["A"]["query_key_distance"],
        "local_prior": main_rows["A"]["local_prior"],
        "note": ("c0：A（无损失臂）必须 ≈ chance ⇒ 任务+窗口确实制造了「不搬就答不出」"
                 "的需求；距离分布证明被问的 pair 落在窗口外"),
    }
    note(f"c0（前提）：A acc={premise['A_accuracy']:.4f} CI={premise['A_ci']} "
         f"chance={chance:.6f} ⇒ 近 chance={premise['A_near_chance']}")

    def _valid(name):
        """c2 毁源对照：读数来自「搬过来的信息」还是位置/构造？"""
        cs = main_rows[name]["corrupt_source"]
        thr = max(4 * chance, 0.03)
        return {"main": main_rows[name]["accuracy"], "corrupt": cs["accuracy"],
                "corrupt_ci": cs["accuracy_ci"], "chance": chance,
                "invalid_threshold": round(thr, 6),
                "invalidated": cs["accuracy_ci"][0] > thr}

    criteria = {
        "c0_premise_A_near_chance": bool(premise["A_near_chance"]),
        "c1_main_B_ge_A_3pp_and_ci_disjoint":
            bool(comparisons["B_vs_A"]["delta_pp"] >= 3.0
                 and not comparisons["B_vs_A"]["ci_overlap"]),
        "c2_corrupt_source_control": {n: _valid(n) for n in arms_order},
        "c3_rerun_rep_ge_A_3pp_and_ci_disjoint":
            bool(comparisons["Brep_vs_A"]["delta_pp"] >= 3.0
                 and not comparisons["Brep_vs_A"]["ci_overlap"]),
        "c3_B_vs_Brep_ci_disjoint":
            bool(not comparisons["B_vs_Brep"]["ci_overlap"]),
        "c3_interpretation": None,          # 读数后按 README §4 写下
    }
    note(f"c1（主）：B−A = {comparisons['B_vs_A']['delta_pp']}pp "
         f"CI 不重叠={not comparisons['B_vs_A']['ci_overlap']} "
         f"⇒ {'过' if criteria['c1_main_B_ge_A_3pp_and_ci_disjoint'] else '未过'}")
    note(f"c3（重跑）：B_rep−A = {comparisons['Brep_vs_A']['delta_pp']}pp "
         f"CI 不重叠={not comparisons['Brep_vs_A']['ci_overlap']}；"
         f"B vs B_rep CI 不重叠={criteria['c3_B_vs_Brep_ci_disjoint']}")
    # c3 的**预登记解读规则**（README §4，读数后不再改）：同配置两次的散度优先于主判据
    if criteria["c3_B_vs_Brep_ci_disjoint"]:
        interp = ("B 与 B_rep（同配置同 seed）读数 CI 不重叠 ⇒ 训练侧双稳态/散度显著，"
                  "本尺度不可分辨；c1 即使过也不得记为「成立」")
    elif criteria["c1_main_B_ge_A_3pp_and_ci_disjoint"] \
            and criteria["c3_rerun_rep_ge_A_3pp_and_ci_disjoint"]:
        interp = "c1 在重跑臂上复现（两次都 ≥ A+3pp 且 CI 不重叠）⇒ 稳健"
    elif criteria["c1_main_B_ge_A_3pp_and_ci_disjoint"]:
        interp = "c1 未在重跑臂上复现 ⇒ 降级为「待定」（训练侧散度，不得记为成立）"
    else:
        interp = ("主判据未过 ⇒ 记「13M / 12k 步 / 本任务上不成立」"
                  "（SCHEMA §19：不得写成「机制不可能」）")
    criteria["c3_interpretation"] = interp
    note("c3 解读：" + interp)
    results["premise_check"] = premise
    results["comparisons"] = comparisons
    results["criteria"] = criteria

    # ---------------- ⑦ 辅助腿：变距离回忆距离曲线 ----------------
    note("辅助腿：变距离回忆（三臂 × 距离档）")
    dleg, dbs, dnb = {}, cfg["distance_leg"]["batch_size"], cfg["distance_leg"]["n_batches"]
    for name in arms_order:
        rows_d = {}
        for d in cfg["distance_leg"]["distances"]:
            agg = {"n": 0, "n_hits": 0, "cn": 0, "chits": 0}
            for i in range(dnb):
                tb = var_recall_batch(dbs, T, d, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      generator=torch.Generator().manual_seed(
                                          2000 + 13 * d + i))
                r = arm_battery(arms[name], tb, W, anchors, dev,
                                corrupt_span_bounds=(tb.meta["src_pos"],
                                                     tb.meta["src_pos"] + 2))
                agg["n"] += r["n"]
                agg["n_hits"] += r["n_hits"]
                agg["cn"] += r["corrupt_source"]["n"]
                agg["chits"] += r["corrupt_source"]["n_hits"]
                chance_d, lp = r["chance_uniform_answer_set"], r["local_prior"]
            rows_d[str(d)] = {
                "distance": d, "n": agg["n"],
                "accuracy": round(agg["n_hits"] / agg["n"], 6),
                "accuracy_ci": wilson_ci(agg["n_hits"], agg["n"]),
                "corrupt_source": {"accuracy": round(agg["chits"] / agg["cn"], 6),
                                   "n": agg["cn"],
                                   "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
                "chance": chance_d, "local_prior": lp,
                "source_position": tb.meta["src_pos"],
                "source_inside_window": (T - 1) - tb.meta["src_pos"] < W}
        dleg[name] = rows_d
        note("  " + name + ": " + " ".join(
            f"d{d}:{rows_d[str(d)]['accuracy']:.3f}"
            for d in cfg["distance_leg"]["distances"]))
    results["distance_leg"] = dleg

    # ---------------- ⑧ 活性量 / 税 / 斜率 / 显存 ----------------
    note("活性量：已训练头在固定批上的重建读数（A 臂 = 未训练头参照）")
    tb_act = assoc_recall_multi_batch(bs, T, cfg["train_recipe"]["synth_pairs"],
                                      cfg["train_recipe"]["synth_queries"],
                                      sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      gap=cfg["train_recipe"]["synth_gap"],
                                      generator=torch.Generator().manual_seed(1234))
    xa = tb_act.tokens.cuda()
    results["recon_activity_on_trained_heads"] = {}
    for name in arms_order:
        with torch.no_grad():
            _, aux = arms[name](xa)
            r = recon_loss(arms[name].recon_head, aux["states"][-1], xa,
                           cfg["recon"]["window"], fill)
        rec = {"fixed_batch": "assoc_recall_multi_batch(bs=%d, K=%d, Q=%d, gap=%d, seed=1234)"
               % (bs, cfg["train_recipe"]["synth_pairs"],
                  cfg["train_recipe"]["synth_queries"], cfg["train_recipe"]["synth_gap"]),
               "ce_all": round(float(r["ce_all"]), 4),
               "ce_symbol": round(float(r["ce_symbol"]), 4),
               "ce_filler": round(float(r["ce_filler"]), 4),
               "acc_all": round(float(r["acc_all"]), 6),
               "acc_symbol": round(float(r["acc_symbol"]), 6),
               "n_symbol": int(r["n_symbol"]), "n_filler": int(r["n_filler"]),
               "chance_uniform_vocab": round(float(r["chance_uniform_vocab"]), 6),
               "chance_symbol_set": round(1.0 / sym, 6),
               "head_w_fro": round(float(arms[name].recon_head.weight.detach()
                                         .float().norm()), 6)}
        results["recon_activity_on_trained_heads"][name] = rec
        note(f"  {name}: ce_all={rec['ce_all']} ce_symbol={rec['ce_symbol']} "
             f"acc_symbol={rec['acc_symbol']} (chance {rec['chance_symbol_set']}) "
             f"|w|={rec['head_w_fro']}")

    note("活性量 / 税 / 斜率 / 显存：读训练产物")
    results["train_results"] = {}
    results["loss_slope"] = {}
    results["train_log_tail"] = {}
    for name in arms_order:
        res_p = json.load(open(resolve(cfg["arms"][name]["results_json"])))
        results["train_results"][name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "gnorm_monitor": res_p.get("gnorm_monitor"),
            "window_mask": res_p.get("window_mask"),
            "recon_liveness": res_p.get("recon_liveness"),
            "intervention": res_p.get("intervention"),
            "avg_train_trunk_ce": res_p.get("avg_train_depth_ce"),
            "synth_answer_ce_mean": res_p.get("gnorm_monitor", {}).get(
                "synth_answer_ce_mean"),
        }
        lines = [json.loads(ln) for ln in
                 open(resolve(cfg["arms"][name]["train_log"])) if ln.strip()]
        results["loss_slope"][name] = loss_slope(lines)
        results["train_log_tail"][name] = {
            "last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
            "last_answer_ce": lines[-1].get("synth_answer_ce"),
            "last_recon": lines[-1].get("recon"),
            "first_recon": lines[0].get("recon")}
        note(f"  {name}: wall {results['train_results'][name]['wall_minutes']} min, "
             f"peak {results['train_results'][name]['peak_gpu_mem_mb']} MB, "
             f"slope {results['loss_slope'][name]['slope']}, "
             f"answer_ce {results['train_results'][name]['synth_answer_ce_mean']}")
    results["memory_analytic"] = analytic_memory(
        {**cfg["model"], "n_params": results["train_results"]["A"]["params"]},
        bs, W, anchors)
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: results["train_results"][n]["peak_gpu_mem_mb"]
                                  for n in arms_order},
        "note": ("实测 = 本进程 torch.cuda.max_memory_allocated（净分配，不含共享卡上"
                 "其他任务的占用）；解析 = analytic_memory（复用 evalkit）")}

    # ---------------- 复现性 ----------------
    note("复现性：主腿 K=%d 整批再跑一遍（同 seed）" % Kmain)
    repro_n = max(nb // 4, 2)
    passes = []
    for _ in range(2):
        agg = {"n": 0, "n_hits": 0}
        for i in range(repro_n):
            tb = assoc_recall_batch(bs, T, Kmain, sym_vocab=sym, filler_token=fill,
                                    cue_token=cue,
                                    generator=torch.Generator().manual_seed(
                                        1000 + 7 * Kmain + i))
            acc = accuracy(arms["A"], tb, keep_mask=keep, device=dev)
            agg["n"] += acc["n"]
            agg["n_hits"] += acc["n_hits"]
        agg["accuracy"] = round(agg["n_hits"] / agg["n"], 6)
        passes.append(agg)
    results["reproducibility"] = {
        "n_per_pass": passes[0]["n"], "n_batches_per_pass": repro_n,
        "accuracy_1": passes[0]["accuracy"], "accuracy_2": passes[1]["accuracy"],
        "n_hits_1": passes[0]["n_hits"], "n_hits_2": passes[1]["n_hits"],
        "same_seed_rerun_max_abs_diff": abs(passes[0]["accuracy"] - passes[1]["accuracy"]),
        "pass": passes[0] == passes[1],
        "note": "同 seed、同代码、同设备把 A 的主腿整批再跑一遍（纯推理复现性）"}

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: results["train_results"][n]["wall_minutes"]
                               for n in arms_order},
        "gpu_wall_budget_min": 45,
        "n_forward_batches": (nb * len(cfg["ksweep"]) * len(arms_order)
                              + dnb * len(cfg["distance_leg"]["distances"])
                              * len(arms_order)),
    }
    code_files = ("run.py", "reconcile_numbers.py", "mutate.py", "config.json",
                  "slope_diagnosis.py", "pathlm/depth_ar.py", "pathlm/recon_signal.py",
                  "train_depth_ar.py", "tests/test_recon_signal.py",
                  "tools/evalkit/carry_channel.py", "tools/evalkit/longrange_tasks.py",
                  "tools/evalkit/retention.py")
    results["code"] = {}
    for p in code_files:
        cand = resolve(p) if os.path.exists(resolve(p)) else os.path.join(HERE, p)
        results["code"][p] = sha256_file(cand) if os.path.exists(cand) else None
    # 契约 README 必须**显式**取本目录的（`resolve("README.md")` 会命中仓库根 README，
    # 那是另一份文档——踩过一次，写下来防复发）。
    results["code"]["2619-README.md (本契约)"] = sha256_file(
        os.path.join(HERE, "README.md"))
    results["code"]["repo-root README.md (参照，非本契约)"] = sha256_file(
        os.path.join(ROOT, "README.md"))
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min）")


if __name__ == "__main__":
    main()
