"""2631 · 因果口径 KV 预算图重测（R1 线，`docs/REMEDIATION.md` §4 R1）—— 实验入口。

契约：同目录 `README.md`（判据线**预登记**，§4）。
动机：复核 A2——旧的 27 点预算图是在「沿 batch+token 取 amax」的**非因果**量化下测得，
过去位置的 K/V 反量化值会因未来 token / 同批其他请求改变（因果掩码挡不住）。
本实验用 L1 修复后的**逐请求、因果/流式**量化实现（`tools/evalkit/kv_quant.quantize_kv`）
重测**原 27 点**（同 checkpoint、同评测流、同协议），与旧口径**并列**报差异方向。

复用：评估管线直接**复用 `2621/run.py`**（`importlib` 载入，不改它）⇒ 与旧图**同一条
路径、同一 seed、同一任务**、横轴口径逐字一致。本批新逻辑只在 `causal_check.py`。

用法：
  python3 run.py            # 全跑（纯推理，共享卡 ≈3–8 min）
  python3 run.py --smoke    # 冒烟：缩小批量/档数，实测步时与显存（不外推时间）
  python3 run.py --charts-only   # 只从已有 results.json 重画图表（纯 CPU，不取 GPU 锁）
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import causal_check as cc                                              # noqa: E402
from pathlm.data import load_enwik8_full                              # noqa: E402
from tools.evalkit.longrange_tasks import (                           # noqa: E402
    corrupt_span, local_prior_accuracy, check_local_prior_resistance)

P2621 = os.path.join(ROOT, "axes/R-references/R1-quantization/experiments/"
                           "2621-kv-budget-rest")
RES_2621_PATH = os.path.join(P2621, "results.json")
sys.path.insert(0, P2621)   # 让 2621/run.py 里的 `import rest_arms` 能解析

VARIANCE_BAND_BPC = 0.014


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2621 的入口（内含 2619 管线 `b19`、`rest_arms`/`gaps_arms` 的臂语义与分派）；只读复用。
b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")


# ----------------------------------------------------------------- 图 / 表

def render_table(cfg, points, controls, rec) -> str:
    L = []
    L.append("# 2631 · 因果口径 KV 预算图 —— 一张表（27 点）")
    L.append("**口径**：本表是 `2621` 的 27 点预算图在 L1 修复后的**逐请求、因果/流式** "
             "KV 量化下重测的结果。除量化器 scale 约定外，checkpoint / 数据流 / 窗口序 / "
             "任务 / 判据与 `2621` 逐字段相同；横轴公式（判据 C1）与 `2619/2620/2621` "
             "逐字同一式：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · "
             f"value_bytes · (positions_kept / T)`，T = {cfg['model']['seq_len']}。"
             "**本列是解析口径**（值字节）；因果化的 scale 存储开销见文末口径说明。")
    L.append(f"两腿全部由本实验**同进程、同一路径**重测（复用 2619 的 `bpc_of` / "
             f"`synth_accuracy`）。召回地板：chance = {controls['chance_uniform']:.4f}（1/64）、"
             f"局部先验 = {controls['local_prior_copy']['accuracy']:.4f}（copy）。"
             "『Δbpc(新−旧)』= 本因果口径 bpc − 2621 旧口径 bpc（负 = 因果口径质量代价更小）。")
    head = ("| 点 | 类别 | 每位置 KV 字节 | bpc（因果） | bpc（旧） | Δbpc(新−旧) | "
            "方向 | damage_bpc（因果） | copy 召回（因果） | 95% CI | 来源 |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    cmp_rows = {r["arm"]: r for r in rec["causal_vs_old"]["rows"]}
    for p in points:
        r = cmp_rows[p["arm"]]
        ci = (f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}]")
        rows.append(
            f"| `{p['arm']}` | {b21.CLASSES[p['class']]} | {p['bytes_per_position']:.0f} | "
            f"{p['bpc']:.6f} | {r.get('bpc_old', float('nan')):.6f} | "
            f"{r.get('delta_bpc', float('nan')):+.6f} | {r.get('direction', '—')} | "
            f"{p['damage_bpc']:+.6f}"
            f"{' ⚠不可排序' if abs(p['damage_bpc']) < VARIANCE_BAND_BPC and p['damage_bpc'] != 0 else ''} | "
            f"{p['recall_copy']['accuracy']:.4f}"
            f"{' (地板)' if p['recall_copy']['at_floor'] else ''} | {ci} | `{p['source']}` |")
    L.append(head + "\n".join(rows))

    cv = rec["causal_vs_old"]
    L.append("## 表 2 · 新旧口径并列（判据 ②/③）\n\n"
             f"- 非量化臂 max |Δbpc| = {cv['nonquant_max_abs_delta_bpc']:.2e}"
             f"（逐位保持 = {cv['nonquant_bitwise_preserved']}）——本轮**唯一**改动的因子是"
             "量化器 scale 约定。\n"
             f"- 量化臂中发生改变的：{', '.join(cv['quant_arms_changed']) or '（无）'}；\n"
             f"  逐位不变的：{', '.join(cv['quant_arms_unchanged']) or '（无）'}。\n"
             f"- int8/int4 同族纯量化臂的 Δbpc（新−旧）："
             f"{json.dumps(rec['int8_int4_quality_delta'], ensure_ascii=False)}")
    L.append("| 量化臂 | 每位置字节 | bpc（因果） | bpc（旧） | Δbpc(新−旧) | 方向 | "
             "copy 召回 Δ |\n|---|---|---|---|---|---|---|")
    for r in cv["rows"]:
        if not r["quantized"] or r.get("missing_in_old"):
            continue
        L.append(f"| `{r['arm']}` | {r['bytes_per_position']:.0f} | {r['bpc_new']:.6f} | "
                 f"{r['bpc_old']:.6f} | {r['delta_bpc']:+.6f} | {r['direction']} | "
                 f"{r['delta_recall_copy']:+.6f} |")

    L.append("## 表 3 · 前缀不变性（判据 ①，硬门禁）\n\n"
             "| 臂 | 未来 token 不改过去 logits | 同批邻居不改本请求 logits |\n"
             "|---|---|---|")
    for p in points:
        inv = p["prefix_invariance"]
        L.append(f"| `{p['arm']}` | {inv['future_token_past_logits_equal']} | "
                 f"{inv['peer_request_logits_equal']} |")
    L.append("> 因果掩码挡不住非因果量化的旁路（复核 A2）——这两条不成立，本实验作废。")

    L.append("## 表 4 · 与来源实验 / 2619 / 2620 的复现性（非量化臂）\n\n"
             "| 点 | 本实验 bpc | 来源 bpc | abs diff | 一致 |\n|---|---|---|---|---|")
    for r in rec["cross_check_vs_source"]["rows"]:
        L.append(f"| `{r['arm']}` | {r['this_bpc']:.6f} | {r.get('source_bpc', '—')} | "
                 f"{r.get('abs_diff', '—')} | "
                 f"{'✓' if r['match'] else ('—' if r['match'] is None else '✗')} |")

    L.append("## 实测显存（与解析列分列，判据 C2）\n\n"
             f"- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）："
             f"**{rec['memory_convention']['measured_gpu_peak_mib']} MiB**。\n"
             "- 评测实现把 K/V 立刻反量化、按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值"
             "不随档下降，故只报解析列。\n"
             "- **部署口径提醒**：严格逐 token 因果 ⇒ 每值一个 fp16 scale（int8 净 3 B/值、"
             "int4 净 2.5 B/值）；旧的「int8 省一半」在因果口径下不成立，部署价签须用 "
             "`kv_cache_footprint` 的 `streaming_bytes_by_seq_len`。")

    L.append("## ASCII 图（因果口径）\n\n```\n" + b21.ascii_chart(points) + "\n```")
    return "\n\n".join(L) + "\n"


def render_png(points, controls, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"baseline": ("k", "o"), "quant": ("tab:red", "s"),
             "window": ("tab:blue", "^"), "head": ("tab:green", "D"),
             "quant_head": ("tab:purple", "P"), "window_anchor": ("tab:orange", "*"),
             "combo_quant_window": ("tab:brown", "X"),
             "combo_head_window": ("tab:cyan", "v")}
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))
    for p in points:
        c, m = style[p["class"]]
        axes[0].scatter(p["bytes_per_position"], p["damage_bpc"], color=c, marker=m, s=60)
        axes[0].annotate(p["arm"], (p["bytes_per_position"], p["damage_bpc"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=6)
        axes[1].scatter(p["bytes_per_position"], p["recall_copy"]["accuracy"],
                        color=c, marker=m, s=60)
        axes[1].annotate(p["arm"], (p["bytes_per_position"], p["recall_copy"]["accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=6)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("bytes per position (analytic)")
    axes[0].axhline(VARIANCE_BAND_BPC, ls="--", c="gray", lw=1)
    axes[0].set_ylabel("damage_bpc (vs own family baseline; lower = better)")
    axes[0].set_title("leg 1: clean bpc (causal caliber)")
    axes[1].axhline(controls["chance_uniform"], ls=":", c="gray", lw=1)
    axes[1].set_ylabel("synthetic recall: copy (pooled; higher = better)")
    axes[1].set_title("leg 2: long-range retrieval (synthetic)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=b21.CLASSES_EN[k])
               for k, (c, m) in style.items()]
    axes[1].legend(handles=handles, fontsize=7, loc="upper left")
    fig.suptitle("2631 · KV budget chart, CAUSAL caliber (T=256; bytes/position ANALYTIC)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def write_charts(cfg, points, controls, rec, out_dir) -> None:
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(cfg, points, controls, rec))
    with open(os.path.join(out_dir, "chart.txt"), "w") as f:
        f.write(b21.ascii_chart(points) + "\n")
    try:
        render_png(points, controls, os.path.join(out_dir, "chart.png"))
        rec["chart_png"] = True
    except Exception as e:                                  # pragma: no cover
        rec["chart_png"] = f"unavailable: {e}"


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--charts-only", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    if args.charts_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        rec["code"] = {name: b21.b19.sha256_file(os.path.join(ROOT, name))[:16]
                       for name in rec["config"]["reuse"]["code_anchors"]}
        write_charts(rec["config"], rec["points"], rec["controls"], rec, args.out_dir)
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print(f"[charts-only] 重画完成（纯 CPU）→ {args.out_dir}", flush=True)
        return 0

    cfg = json.load(open(args.config))
    dev = cfg["eval"]["device"]
    T = cfg["model"]["seq_len"]
    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"],
           "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke),
           "causal_caliber": cfg["causal_caliber"]}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))

    # ---------- ① 仪器自检：三条腿先证明「本实验路径 == 既有仪器路径」 ----------
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    fp32_fn = b21.b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    anchor = cfg["controls"]["baseline_anchor_bpc"]
    lib_bpc = b21.b19.masked_bpc(trunk, eval_arr, None,
                                 n_batches=cfg["eval"]["n_batches"],
                                 batch_size=cfg["eval"]["batch_size"],
                                 window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = b21.b19.bpc_of(fp32_fn, eval_arr, T, cfg["eval"]["n_batches"],
                            cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    tasks = b21.b19.build_tasks(cfg, smoke=args.smoke)
    probe = tasks["var_recall"]
    lib_acc = b21.b19.lib_accuracy(trunk, probe, device=dev)
    my_acc = b21.b19.synth_accuracy(fp32_fn, probe, dev)
    sc = b21.ra.reduction_self_check(trunk, tasks["copy"].tokens, T, dev)
    # 前缀不变性（判据 ①）：在 trunk 上直接对 int8/int4 生产量化器检验
    inv_trunk = {m: cc.prefix_invariance_check(trunk, tasks["copy"].tokens[:4].to(dev), m)
                 for m in ("int8", "int4")}
    rec["instrument"] = {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "retention.masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc), "bitwise_equal": my_bpc == lib_bpc,
                     "anchor_2614_release_bpc": anchor,
                     "abs_diff_vs_anchor": abs(my_bpc - anchor),
                     "mode": "fp32（恒等：KV 数值未改）"},
        "synth_path": {"run::synth_accuracy": my_acc["accuracy"],
                       "longrange_tasks.accuracy": lib_acc["accuracy"],
                       "n_equal": my_acc["n"] == lib_acc["n"],
                       "n_hits_equal": my_acc["n_hits"] == lib_acc["n_hits"],
                       "by_distance_equal": all(
                           my_acc["by_distance"][k]["accuracy"]
                           == lib_acc["by_distance"][k]["accuracy"]
                           for k in lib_acc["by_distance"])},
        "combo_mask_reduction": sc,
        "prefix_invariance_trunk": inv_trunk,
        "note": "bpc/synth 两腿先证明与既有仪器逐位一致；combo_mask 的两条退化路径"
                "（mode=fp32 ⇒ masked_logits；掩码=全因果 ⇒ kv_quant）逐位还原；"
                "prefix_invariance_trunk 直接对生产量化器检验复核 A2 的两条旁路。",
    }
    print(f"[自检] 恒等臂 bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f} "
          f"(逐位={rec['instrument']['bpc_path']['bitwise_equal']})；"
          f"前缀不变性 trunk {inv_trunk}", flush=True)

    # ---------- ② 对照臂：chance / 局部先验 / 毁源 ----------
    copy_len = cfg["tasks"]["copy"]["copy_len"]
    tasks["copy_corrupt_by_pos"] = {
        p: corrupt_span(tasks["copy_by_pos"][p], p, p + copy_len)
        for p in tasks["copy_by_pos"]}
    controls = {
        "chance_uniform": round(1.0 / cfg["tokens"]["sym_vocab"], 8),
        "local_prior_copy": local_prior_accuracy(tasks["copy"], k=cfg["tokens"]["local_k"]),
        "local_prior_induction": local_prior_accuracy(tasks["induction"],
                                                      k=cfg["tokens"]["local_k"]),
        "local_prior_var_recall": local_prior_accuracy(tasks["var_recall"],
                                                       k=cfg["tokens"]["local_k"]),
        "local_prior_resistance": {
            "copy": check_local_prior_resistance(tasks["copy"], k=cfg["tokens"]["local_k"]),
            "induction": check_local_prior_resistance(tasks["induction"],
                                                      k=cfg["tokens"]["local_k"]),
            "var_recall": check_local_prior_resistance(tasks["var_recall"],
                                                       k=cfg["tokens"]["local_k"])},
        "corrupt_source_note": ("把 copy 主腿的源片段整段换成随机符号（其余逐位不变），"
                                "逐臂报该对照（源区间 = 该档自己的 [src_pos, src_pos+32)）。"),
    }
    rec["controls"] = controls
    print(f"[对照] copy 局部先验={controls['local_prior_copy']['accuracy']:.4f} / "
          f"chance={controls['chance_uniform']:.4f}", flush=True)

    # ---------- ③ 逐臂：bpc + 合成召回 + 前缀不变性 ----------
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)
    points = []
    n_arms = len(cfg["arms"])
    for i, (arm, spec) in enumerate(cfg["arms"].items(), 1):
        model, ck = b21.load_model_for_arm(cfg, arm, spec, dev)
        if spec["kind"] == "quant_head":
            ck = {**ck, "base_arm": spec["base_arm"], "quantized_mode": spec["mode"],
                  "note": "头共享 checkpoint（fp16 存储）× 评测期 KV 量化往返；权重未改"}
        elif spec["kind"] == "combo_mask":
            ck = {**ck, "base_model": spec.get("base_model", "trunk"),
                  "mask_window": spec["window"], "mask_anchors": spec.get("anchors", 0),
                  "quantized_mode": spec["mode"],
                  "note": "base 模型 checkpoint × 评测期可见性掩码 × KV 量化往返；权重未改"}
        fn = b21.make_logits_fn(model, spec, T, dev)
        bpc = b21.b19.bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"],
                             cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
        copy_pooled = b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
        copy_corrupt = b21.b19.pooled_accuracy(fn, tasks["copy_corrupt_by_pos"], dev)
        ind_pooled = b21.b19.pooled_accuracy(fn, tasks["induction_by_start"], dev)
        var_pooled = b21.b19.pooled_accuracy(fn, tasks["var_recall_by_d"], dev)
        geo = b21.bytes_per_position(spec, cfg)
        chance = controls["chance_uniform"]
        at_floor = copy_pooled["accuracy"] <= b21.b19.floor_threshold(copy_pooled["n"], chance)
        points.append({
            "arm": arm, "role": spec["role"], "family": spec["family"],
            "class": b21.class_of(arm), "checkpoint": ck,
            "bpc": round(bpc, 6),
            "activity": b21.activity_of(spec, model, T, dev, probe_tokens),
            "prefix_invariance": cc.prefix_invariance_of_logits(fn, probe_tokens),
            "recall_copy": {**copy_pooled, "at_floor": bool(at_floor)},
            "recall_copy_source_corrupt": copy_corrupt,
            "recall_induction": ind_pooled, "recall_var_recall": var_pooled,
            "chance_uniform": chance,
            "local_prior_copy": controls["local_prior_copy"]["accuracy"],
            "local_prior_induction": controls["local_prior_induction"]["accuracy"],
            "local_prior_var_recall": controls["local_prior_var_recall"]["accuracy"],
            "mark": b21.MARK[arm], "source": b21.SOURCE[arm],
            **geo,
        })
        print(f"[{i}/{n_arms}] {arm:<22} bpc={bpc:.6f} copy={copy_pooled['accuracy']:.4f} "
              f"inv={points[-1]['prefix_invariance']['future_token_past_logits_equal']}/"
              f"{points[-1]['prefix_invariance']['peer_request_logits_equal']} "
              f"bytes/pos={geo['bytes_per_position']:.0f}", flush=True)

    # ---------- ④ damage / 相对 stock / 量化增量 / 窗口增量 ----------
    by_arm = {p["arm"]: p for p in points}
    for p in points:
        base_arm = cfg["family_baseline"][p["family"]]
        base = by_arm[base_arm]
        p["family_baseline"] = base_arm
        p["damage_bpc"] = round(p["bpc"] - base["bpc"], 6)
        p["delta_schema"] = round(base["bpc"] - p["bpc"], 6)
        p["kv_reduction_vs_family_baseline"] = round(
            1 - p["bytes_per_position"] / base["bytes_per_position"], 6)
        p["total_damage_vs_stock"] = round(p["bpc"] - anchor, 6)
        ref = cfg.get("quant_increment_ref", {}).get(p["arm"])
        p["quant_increment_bpc"] = (round(p["bpc"] - by_arm[ref]["bpc"], 6)
                                    if ref else None)
        if ref:
            p["quant_increment_vs_arm"] = ref
            p["quant_increment_recall_delta"] = round(
                p["recall_copy"]["accuracy"] - by_arm[ref]["recall_copy"]["accuracy"], 6)
        wref = cfg.get("window_increment_ref", {}).get(p["arm"])
        p["window_increment_bpc"] = (round(p["bpc"] - by_arm[wref]["bpc"], 6)
                                     if wref else None)
        if wref:
            p["window_increment_vs_arm"] = wref
            p["window_increment_recall_delta"] = round(
                p["recall_copy"]["accuracy"] - by_arm[wref]["recall_copy"]["accuracy"], 6)

    # ---------- ⑤ 新旧口径并列（本实验核心判据 ②/③） ----------
    old_rec = json.load(open(os.path.join(ROOT, cfg["compare_against_2621"]["results"])))
    quant_arms = cc.quantized_arms_of(cfg)
    comparison = cc.compare_causal_vs_noncausal(old_rec["points"], points, quant_arms)
    rec["causal_vs_old"] = comparison
    rec["int8_int4_quality_delta"] = cc.int8_int4_quality_delta(comparison)
    rec["quantized_arms"] = quant_arms
    print(f"[新旧] 非量化臂逐位保持={comparison['nonquant_bitwise_preserved']} "
          f"(max|Δ|={comparison['nonquant_max_abs_delta_bpc']:.2e})；"
          f"量化臂改变={comparison['quant_arms_changed']}", flush=True)

    # ---------- ⑥ 与来源实验逐一对照（非量化臂的可控性） ----------
    src = b21.b19.source_values()
    rows = []
    for p in points:
        row = {"arm": p["arm"], "this_bpc": p["bpc"], "source": p["source"]}
        w3, r1, m2 = src.get("W3", {}), src.get("R1", {}), src.get("M2", {})
        if p["arm"] in r1 and p["arm"] not in ("int8", "int4"):
            row["source_bpc"] = r1[p["arm"]]["bpc"]     # fp16 基线（int8/int4 已改口径）
        elif p["arm"] in w3:
            row["source_bpc"] = w3[p["arm"]].get("bpc_second_path_masked_bpc")
        elif cfg["arms"][p["arm"]]["kind"] == "window":
            row["source_bpc"] = m2.get(f"recent:d={cfg['arms'][p['arm']]['d']}", {}).get("bpc")
        else:
            row["source_bpc"] = None
        if row.get("source_bpc") is not None:
            row["abs_diff"] = round(abs(p["bpc"] - row["source_bpc"]), 8)
            row["match"] = row["abs_diff"] <= 1e-5
        else:
            row["match"] = None
        rows.append(row)
    rec["cross_check_vs_source"] = {
        "rows": rows, "sources": src.get("sources", {}),
        "all_matched": all(r["match"] is not False for r in rows),
        "note": "非量化（窗口 / 头共享 / 纯掩码 / fp16）臂与其来源实验逐位一致；"
                "int8/int4 及其组合臂因口径更新**预期**改变（见 causal_vs_old）。"}

    # ---------- ⑦ 汇总落盘 ----------
    rec["points"] = points
    rec["memory_convention"] = {
        "analytic_bytes_per_position": "见 points[*].bytes_per_position（判据 C1 的唯一公式，值字节口径）",
        "measured_gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "streaming_scale_overhead": ("因果/流式量化每值 +1 个 fp16 scale ⇒ int8 净 3 B/值、"
                                     "int4 净 2.5 B/值；部署价签用 kv_cache_footprint 的 "
                                     "streaming_bytes_by_seq_len，本表横轴仍用旧值字节口径以便与 2621 并列。"),
        "isolation_note": ("判据 C2：解析字节是本图的横轴；实测显存峰值单列在此，不进横轴。"),
    }
    rec["code"] = {name: b21.b19.sha256_file(os.path.join(ROOT, name))[:16]
                   for name in cfg["reuse"]["code_anchors"]}
    rec["cost"] = {"device": dev}
    rec["cost"]["wall_seconds"] = round(time.time() - t0, 2)
    rec["cost"]["wall_minutes"] = round(rec["cost"]["wall_seconds"] / 60, 3)
    rec["cost"]["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)

    if not args.smoke:
        write_charts(cfg, points, controls, rec, args.out_dir)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"[完成] {rec['cost']['wall_minutes']} min，峰值 {rec['cost']['peak_mem_mib']} MiB，"
          f"{len(points)} 点 → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
