"""2623 · A9 合成长程判决（训练 window vs full）——评测入口（纯推理）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
训练：见 `train_all.sh` / `MODEL.md`（`train_depth_ar.py` + `--synth-task assoc-mqar`）。
本入口只做**评测**：

  ① 仪器自检：释放版 == 显式因果掩码（逐位，full 臂）；模型自带窗口掩码前向 ==
     `retention.masked_logits` + `window_anchor_keep_mask`（逐位，win16 臂）；两臂
     `params` 相等（同架构，唯一变量是掩码不是参数）。
  ② 判据 ①（主腿，**训练同族任务**）：`assoc_recall_multi_batch`（多查询关联回忆，
     与训练同一布局、held-out 新键值）K∈{4,8,16} 的准确率 + chance + Wilson CI +
     查询-key 距离分布；**判据 ②（毁源对照臂）**同批 `corrupt_span(0,2K)`。
  ②b 主腿的距离读数 = 同一批的 by-distance 曲线（源全部落在窗外）。
  ③ 判据 ③（距离腿）：`var_recall` 六档（每臂原生掩码 + 「源是否在窗内」标注）——
     **注意**：`var_recall` 与单查询 `assoc_recall_batch` 都把查询键放在**最后一个
     位置**，而训练布局里最后一个位置恒为 filler ⇒ 对 full 臂是**位置外推**（它会把
     末位预测成 filler）。故这两个单查询读数作**次级**如实报告，并附证据。
  ④ 判据 ④（辅助腿）：任务分布上的 clean bit/token（全位置 + 判决行）。
  ⑤ 活性/成本：训练 loss 斜率、峰值显存、墙钟、同 seed 复现性。

用法：
  python3 run.py                 # 全电池（纯推理，需先有 train/ 下权重；先取 GPU 锁）
  python3 run.py --smoke         # 冒烟（缩批，实测耗时/显存，不外推训练时间）
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
                             window_anchor_keep_mask)
from tools.evalkit.carry_channel import (two_proportion, wilson_ci,  # noqa: E402
                                         analytic_memory, loss_slope)
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_batch, assoc_recall_multi_batch, var_recall_batch,
    accuracy, local_prior_accuracy, corrupt_span)
from tools.evalkit.retention import masked_logits                  # noqa: E402
from lr_eval import (native_keep_mask, arm_visibility,             # noqa: E402
                     distance_source_inside_window, aggregate_recall,
                     clean_bits_per_token)

ARM_ORDER = ("full", "win16", "win16_rep")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str):
    """加载一臂权重并**断言 sha256**（权重不入库，哈希必须逐次核）。"""
    path = resolve(spec["path"])
    if not os.path.exists(path):
        raise SystemExit(f"缺权重：{path}（先训练 / 先跑 train_all.sh）")
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    vis = arm_visibility(spec)
    cfg = DepthARConfig(**{**mcfg, "train_window": vis["train_window"],
                           "train_anchors": vis["train_anchors"]})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   **vis}


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
        cfg["eval"]["n_batches_main"] = 2
        cfg["eval"]["n_batches_other"] = 1
        cfg["distance_leg"]["n_batches"] = 1
        cfg["eval"]["batch_size"] = 16
        cfg["distance_leg"]["batch_size"] = 8
        cfg["eval"]["ksweep"] = [8]
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    local_k = cfg["tokens"]["local_k"]
    bs = cfg["eval"]["batch_size"]
    nb_main = cfg["eval"]["n_batches_main"]
    nb_other = cfg["eval"]["n_batches_other"]
    K_main = cfg["eval"]["K_main"]
    Q_train = cfg["train_recipe"]["synth_queries"]
    gap_train = cfg["train_recipe"]["synth_gap"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载各臂（sha256 断言；缺臂跳过，允许只跑 full+win16）")
    arms, ck = {}, {}
    for name in ARM_ORDER:
        if name not in cfg["arms"]:
            continue
        if not os.path.exists(resolve(cfg["arms"][name]["path"])):
            note(f"{name}: 缺权重，跳过（预算砍臂，见契约 §成本）")
            continue
        arms[name], ck[name] = load_arm(cfg["arms"][name], cfg["model"],
                                        cfg["data"]["vocab_size"], dev)
        note(f"{name}: {ck[name]['sha256'][:16]}… window={ck[name]['train_window']}")
    if "full" not in arms or "win16" not in arms:
        raise SystemExit("full 与 win16 两臂都必须存在才能判读（判据 ①②③）")
    results = {"config": cfg, "checkpoints": ck, "visibility": cfg["visibility"]}

    # ---------------- ① 仪器自检 ----------------
    note("自检 ①：释放版（is_causal）vs 显式因果掩码（full 臂，逐位）")
    g = torch.Generator().manual_seed(11)
    x_probe = torch.randint(0, cfg["data"]["vocab_size"], (2, T), generator=g).cuda()
    exact = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    with torch.no_grad():
        a1 = masked_logits(arms["full"], x_probe, None)
        a2 = masked_logits(arms["full"], x_probe, exact)
    release_vs_exact = float((a1 - a2).abs().max())
    note(f"  full 臂 max|Δlogit| = {release_vs_exact:.3e}")
    note("自检 ①：模型自带窗口掩码前向 vs masked_logits+window_anchor_keep_mask（win16 臂）")
    keep16 = window_anchor_keep_mask(T, 16, 0, device=dev)
    with torch.no_grad():
        b1 = arms["win16"](x_probe)[1]["depth_logits"][-1].float()
        b2 = masked_logits(arms["win16"], x_probe, keep16)
    win_selfcheck = float((b1 - b2).abs().max())
    note(f"  win16 臂 max|Δlogit| = {win_selfcheck:.3e}")
    n_params = {n: sum(p.numel() for p in arms[n].parameters()) for n in arms}
    results["selfchecks"] = {
        "release_vs_explicit_causal_max_abs_logit_full": release_vs_exact,
        "model_window_forward_vs_mask_instrument_max_abs_logit_win16": win_selfcheck,
        "n_params": n_params,
        "params_equal": len(set(n_params.values())) == 1,
        "window16_nondegenerate": {
            "visible_frac": round(float(keep16.sum().item())
                                  / (T * (T + 1) // 2), 6),
            "alive": bool(keep16.sum().item() < T * (T + 1) // 2)},
        "visibility_consistency": (
            "训练侧 DepthARModel._window_mask 与评测侧 lr_eval.native_keep_mask "
            "调同一函数 window_anchor_keep_mask ⇒ 训练掩码 == 评测掩码"),
    }
    note(f"  参数量 {n_params}，两臂相等={results['selfchecks']['params_equal']}；"
         f"window16 非退化 visible_frac="
         f"{results['selfchecks']['window16_nondegenerate']['visible_frac']}")

    # ---------------- ② 主腿（训练同族任务 multi-query）+ 毁源对照 ----------------
    note("判据 ①②：多查询关联回忆（训练同族任务，held-out；每臂原生掩码）+ 毁源对照")
    main_leg = {}
    for name, model in arms.items():
        vis = arm_visibility(cfg["arms"][name])
        km = native_keep_mask(T, vis["train_window"], vis["train_anchors"], dev)
        for K in cfg["eval"]["ksweep"]:
            n_batches = nb_main if K == K_main else nb_other
            Q = min(Q_train, K)         # multi-query 要求 Q ≤ K（同一行内问不同 pair）
            row = {"K": K, "Q": Q, "gap": gap_train, "n_batches": n_batches,
                   "train_window": vis["train_window"]}
            agg_rows, agg_cor, dists, acc_bd = [], [], [], {}
            for i in range(n_batches):
                tb = assoc_recall_multi_batch(
                    bs, T, K, Q, sym_vocab=sym, filler_token=fill,
                    cue_token=cue, gap=gap_train,
                    generator=torch.Generator().manual_seed(3000 + 7 * K + i))
                r = accuracy(model, tb, keep_mask=km, device=dev)
                agg_rows.append(r)
                dists.extend(int(v) for v in tb.dists.flatten().tolist())
                for d, v in r["by_distance"].items():
                    slot = acc_bd.setdefault(d, {"n": 0, "n_hits": 0})
                    slot["n"] += v["n"]
                    slot["n_hits"] += round(v["accuracy"] * v["n"])
                tc = corrupt_span(tb, 0, 2 * K,
                                  generator=torch.Generator().manual_seed(
                                      9000 + 7 * K + i))
                agg_cor.append(accuracy(model, tc, keep_mask=km, device=dev))
                if K == K_main and i == 0:
                    row["local_prior"] = local_prior_accuracy(tb, k=local_k)
                    row["chance"] = round(tb.chance(), 8)
            main = aggregate_recall(agg_rows, dists, row.get("chance", 1.0 / sym))
            cor = aggregate_recall(agg_cor, [], row.get("chance", 1.0 / sym))
            row.update({
                "accuracy": main["accuracy"], "n": main["n"],
                "n_hits": main["n_hits"], "accuracy_ci": main["accuracy_ci"],
                "chance": row.get("chance", round(1.0 / sym, 8)),
                "query_key_distance": main.get("query_key_distance"),
                "corrupt_source": {
                    "span": [0, 2 * K], "accuracy": cor["accuracy"],
                    "n": cor["n"], "n_hits": cor["n_hits"],
                    "accuracy_ci": cor["accuracy_ci"]},
                "by_distance": acc_bd})
            main_leg[f"{name}_K{K}"] = row
            note(f"  {name} K={K} Q={row['Q']}: acc={row['accuracy']:.4f} "
                 f"CI={row['accuracy_ci']} n={row['n']} | "
                 f"毁源={row['corrupt_source']['accuracy']:.4f}")
    results["main_leg_multiquery"] = main_leg

    rows_main = {n: main_leg[f"{n}_K{K_main}"] for n in arms}
    premise = {}
    for n, r in rows_main.items():
        chance = r["chance"]
        premise[n] = {
            "accuracy": r["accuracy"], "accuracy_ci": r["accuracy_ci"],
            "chance": chance,
            "significantly_above_chance": r["accuracy_ci"][0] > chance,
            "at_chance": r["accuracy_ci"][0] <= chance,
            "multiple_of_chance": round(r["accuracy"] / chance, 3),
            "query_key_distance": r["query_key_distance"],
            "local_prior": r.get("local_prior", {}).get("accuracy"),
            "train_window": r["train_window"]}
    comparisons = {"full_vs_win16": two_proportion(
        rows_main["win16"]["n_hits"], rows_main["win16"]["n"],
        rows_main["full"]["n_hits"], rows_main["full"]["n"])}
    if "win16_rep" in arms:
        comparisons["full_vs_win16_rep"] = two_proportion(
            rows_main["win16_rep"]["n_hits"], rows_main["win16_rep"]["n"],
            rows_main["full"]["n_hits"], rows_main["full"]["n"])
    criteria = {
        "c1_full_learns_recall": {
            "full_sig_above_chance": premise["full"]["significantly_above_chance"],
            "full_multiple_of_chance": premise["full"]["multiple_of_chance"],
            "note": "判据①主腿：full 臂召回 CI 下界 > chance（训练后真的会回忆）"},
        "c2_win16_at_chance": {
            "win16_at_chance": premise["win16"]["at_chance"],
            "note": "判据①主腿：win16 臂召回与 chance 不可区分（窗口=16 ⇒ 源在窗外，"
                    "无通道就搬不动；训练不产生绕路）"},
        "c3_window_hurts_longrange": {
            "delta_pp_full_minus_win16": comparisons["full_vs_win16"].get("delta_pp"),
            "ci_disjoint": (not comparisons["full_vs_win16"].get("ci_overlap", True)),
            "pass": (comparisons["full_vs_win16"].get("delta_pp", 0) >= 10.0
                     and not comparisons["full_vs_win16"].get("ci_overlap", True)),
            "note": "判据①主腿：full 与 win16 的两比例差 ≥10pp 且 CI 不重叠 ⇒ 窗口伤长程"},
        "c4_corrupt_source_control": {
            n: {"main": rows_main[n]["accuracy"],
                "corrupt": rows_main[n]["corrupt_source"]["accuracy"],
                "corrupt_ci": rows_main[n]["corrupt_source"]["accuracy_ci"],
                "chance": rows_main[n]["chance"],
                "source_was_used": rows_main[n]["corrupt_source"]["accuracy_ci"][1]
                < rows_main[n]["accuracy"]}
            for n in arms},
    }
    results["premise_check"] = premise
    results["comparisons"] = comparisons
    results["criteria"] = criteria
    note(f"判据①：full acc={premise['full']['accuracy']:.4f} "
         f"({premise['full']['multiple_of_chance']}× chance, "
         f">chance={premise['full']['significantly_above_chance']})；"
         f"win16 acc={premise['win16']['accuracy']:.4f} "
         f"(at_chance={premise['win16']['at_chance']})；"
         f"Δ={comparisons['full_vs_win16'].get('delta_pp')}pp")

    # ---------------- ②b 次级：单查询布局（位置外推） ----------------
    note("次级：单查询布局 assoc_recall_batch（查询键在**最后一个位置**，位置外推）")
    single = {}
    for name, model in arms.items():
        vis = arm_visibility(cfg["arms"][name])
        km = native_keep_mask(T, vis["train_window"], vis["train_anchors"], dev)
        rows = []
        for i in range(nb_main):
            tb = assoc_recall_batch(bs, T, K_main, sym_vocab=sym, filler_token=fill,
                                    cue_token=cue,
                                    generator=torch.Generator().manual_seed(
                                        1000 + 7 * K_main + i))
            rows.append(accuracy(model, tb, keep_mask=km, device=dev))
        agg = aggregate_recall(rows, [], round(1.0 / sym, 8))
        single[name] = {"accuracy": agg["accuracy"], "n": agg["n"],
                        "n_hits": agg["n_hits"], "accuracy_ci": agg["accuracy_ci"],
                        "chance": agg["chance"], "train_window": vis["train_window"],
                        "note": "查询键在末位（位置外推）；full 的末位先验=预测 filler"}
        note(f"  {name}: 单查询 acc={agg['accuracy']:.4f} CI={agg['accuracy_ci']} "
             f"n={agg['n']}")
    results["secondary_single_query"] = single

    # ---------------- ③ 距离腿：var_recall（次级，位置外推） ----------------
    note("判据 ③：变距离回忆（var_recall；每臂原生掩码 × 六档距离）——次级，位置外推")
    dleg, dbs, dnb = {}, cfg["distance_leg"]["batch_size"], cfg["distance_leg"]["n_batches"]
    for name, model in arms.items():
        vis = arm_visibility(cfg["arms"][name])
        km = native_keep_mask(T, vis["train_window"], vis["train_anchors"], dev)
        rowout = {}
        for d in cfg["distance_leg"]["distances"]:
            rows, cn, chits, chance = [], 0, 0, 1.0 / sym
            for i in range(dnb):
                tb = var_recall_batch(dbs, T, d, sym_vocab=sym, filler_token=fill,
                                      cue_token=cue,
                                      generator=torch.Generator().manual_seed(
                                          2000 + 13 * d + i))
                rows.append(accuracy(model, tb, keep_mask=km, device=dev))
                tb_c = corrupt_span(tb, tb.meta["src_pos"], tb.meta["src_pos"] + 2,
                                    generator=torch.Generator().manual_seed(
                                        7000 + 13 * d + i))
                rc = accuracy(model, tb_c, keep_mask=km, device=dev)
                cn += rc["n"]; chits += rc["n_hits"]; chance = round(tb.chance(), 8)
            agg = aggregate_recall(rows, [], chance)
            rowout[str(d)] = {
                "distance": d, "window": vis["train_window"],
                "source_inside_window": distance_source_inside_window(
                    T, d, vis["train_window"]),
                "n": agg["n"], "accuracy": agg["accuracy"],
                "accuracy_ci": agg["accuracy_ci"], "chance": chance,
                "significantly_above_chance": agg["accuracy_ci"][0] > chance,
                "corrupt_source": {"accuracy": round(chits / max(cn, 1), 6), "n": cn,
                                   "accuracy_ci": wilson_ci(chits, cn)}}
        dleg[name] = rowout
        note("  " + name + ": " + " ".join(
            f"d{d}:{rowout[str(d)]['accuracy']:.3f}"
            f"{'*' if rowout[str(d)]['source_inside_window'] else ''}"
            for d in cfg["distance_leg"]["distances"])
            + "（*=源在窗内）")
    results["distance_leg"] = dleg

    # ---------------- ④ clean bpc 辅助腿 ----------------
    note("判据 ④（辅助腿）：任务分布上的 clean bit/token（全位置 + 判决行）")
    clean = {}
    for name, model in arms.items():
        vis = arm_visibility(cfg["arms"][name])
        km = native_keep_mask(T, vis["train_window"], vis["train_anchors"], dev)
        tb = assoc_recall_multi_batch(
            bs, T, K_main, Q_train, sym_vocab=sym, filler_token=fill,
            cue_token=cue, gap=gap_train,
            generator=torch.Generator().manual_seed(4242))
        clean[name] = clean_bits_per_token(model, tb, km, device=dev)
        note(f"  {name}: clean bit/token all={clean[name]['all_positions']} "
             f"answer_rows={clean[name]['answer_rows']} "
             f"(chance={clean[name]['chance_bits']})")
    results["clean_bpc_task_distribution"] = clean

    # ---------------- ⑤ 活性、loss 斜率、显存、复现 ----------------
    note("活性 / loss 斜率 / 显存")
    for name, model in arms.items():
        res_p = json.load(open(resolve(cfg["arms"][name]["results_json"])))
        lines = [json.loads(l) for l in
                 open(resolve(cfg["arms"][name]["train_log"])) if l.strip()]
        results.setdefault("train_results", {})[name] = {
            "wall_minutes": res_p.get("wall_minutes"),
            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
            "params": res_p.get("params"),
            "train_window": res_p.get("train_window"),
            "window_liveness": lines[-1].get("window_mask"),
            "gnorm_monitor": {k: res_p.get("gnorm_monitor", {}).get(k)
                              for k in ("synth_answer_ce_mean", "synth_chance_ce",
                                        "gnorm_max", "n_grad_skip")}}
        results.setdefault("loss_slope", {})[name] = loss_slope(lines)
        results.setdefault("answer_ce_slope", {})[name] = _answer_ce_slope(lines)
        results.setdefault("train_log_tail", {})[name] = {
            "last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
            "last_answer_ce": lines[-1].get("synth_answer_ce"),
            "last_window": lines[-1].get("window_mask")}
        note(f"  {name}: wall {results['train_results'][name]['wall_minutes']} min, "
             f"peak {results['train_results'][name]['peak_gpu_mem_mb']} MB, "
             f"answer_ce_slope {results['answer_ce_slope'][name]}")

    results["memory_analytic"] = analytic_memory(
        {**cfg["model"], "n_params": results["train_results"]["full"]["params"]},
        bs, 0, 0)
    results["memory_measured"] = {
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: results["train_results"][n]["peak_gpu_mem_mb"]
                                  for n in arms}}

    note("复现性：win16 主腿 K=K_main 同 seed 再跑一遍")
    vis = arm_visibility(cfg["arms"]["win16"])
    km = native_keep_mask(T, vis["train_window"], vis["train_anchors"], dev)
    r1 = {"n": 0, "n_hits": 0}
    for i in range(nb_main):
        tb = assoc_recall_multi_batch(
            bs, T, K_main, Q_train, sym_vocab=sym, filler_token=fill,
            cue_token=cue, gap=gap_train,
            generator=torch.Generator().manual_seed(3000 + 7 * K_main + i))
        a = accuracy(arms["win16"], tb, keep_mask=km, device=dev)
        r1["n"] += a["n"]; r1["n_hits"] += a["n_hits"]
    results["reproducibility"] = {
        "win16_Kmain_same_seed_rerun": {
            "n": r1["n"], "n_hits": r1["n_hits"],
            "accuracy": round(r1["n_hits"] / r1["n"], 6),
            "matches_first_pass": (r1["n_hits"] == rows_main["win16"]["n_hits"])}}

    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: results["train_results"][n]["wall_minutes"]
                               for n in arms}}
    code_files = ("run.py", "lr_eval.py", "config.json", "README.md",
                  "reconcile_numbers.py", "mutate.py", "train_all.sh",
                  "pathlm/depth_ar.py", "train_depth_ar.py",
                  "tools/evalkit/longrange_tasks.py",
                  "tools/evalkit/carry_channel.py", "tools/evalkit/retention.py")
    results["code"] = {}
    for p in code_files:
        local = os.path.join(HERE, p)
        cand = local if os.path.exists(local) else resolve(p)
        results["code"][p] = sha256_file(cand) if os.path.exists(cand) else None
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"完成：{args.out_dir}/results.json（评测墙钟 "
         f"{results['cost']['wall_minutes_total_eval']} min）")


def _answer_ce_slope(lines: list[dict], lo: int = 200, hi: int = 12000) -> dict:
    """判决行 CE（决定准确率的那个损失）的末段斜率（SCHEMA §17.15）。"""
    import math
    pts = [(r["step"], r["synth_answer_ce"]) for r in lines
           if lo <= r["step"] <= hi and r.get("synth_answer_ce") is not None]
    n = len(pts)
    if n < 3:
        return {"n_points": n, "slope": None}
    mx = sum(math.log(max(s, 1)) for s, _ in pts) / n
    my = sum(y for _, y in pts) / n
    cov = sum((math.log(max(s, 1)) - mx) * (y - my) for s, y in pts)
    var = sum((math.log(max(s, 1)) - mx) ** 2 for s, _ in pts)
    return {"n_points": n, "slope": round(cov / var, 6) if var else None,
            "first": round(pts[0][1], 4), "last": round(pts[-1][1], 4)}


if __name__ == "__main__":
    main()
