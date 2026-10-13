"""2620 · 补齐 U3「统一 KV 预算图」的三个最优先缺失组合（ticket 37 · V1 线）—— 入口。

契约：同目录 `README.md`（判据线**逐字**抄录 `.scratch/37-gaps/issues/01-gaps.md`
的「契约 V1」，执行期间未改）。
仪器：`tools/evalkit/longrange_tasks.py`（合成长程任务电池，**不用 enwik8 当长程主任务**）、
      `tools/evalkit/kv_quant.py`（KV 量化往返）、`tools/evalkit/retention.py`（可见性掩码）。
新组合臂的唯一语义层：同目录 `gaps_arms.py`（`quant_head` / `window_anchor`）。
被测量模型：trunk / head 两族 checkpoint 复用（只读，运行时断言 sha256）；见 `MODEL.md`。
参考臂的评估管线：直接**复用 2619 的 run.py**（`importlib` 载入，不改它）⇒ 与 U3 的图
**同一条路径、同一 seed、同一任务**，横轴口径逐字一致。

用法：
  python3 run.py            # 全跑（纯推理，共享卡 ≈1 min）
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

import gaps_arms as ga                                                    # noqa: E402
from pathlm.data import load_enwik8_full                                  # noqa: E402
from tools.evalkit.longrange_tasks import (                               # noqa: E402
    corrupt_span, local_prior_accuracy, check_local_prior_resistance)

CHART_2619_PATH = os.path.join(ROOT, "axes/R-references/R1-quantization/experiments/"
                                     "2619-kv-budget-chart/run.py")
RES_2619_PATH = os.path.join(ROOT, "axes/R-references/R1-quantization/experiments/"
                                   "2619-kv-budget-chart/results.json")

VARIANCE_BAND_BPC = 0.014


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2619 的评估管线（只读取用，不改）。
b19 = _load_module(CHART_2619_PATH, "kv_budget_chart_2619")

CLASSES = {"baseline": "不压缩基线", "quant": "量化（R1）", "window": "窗口/驱逐（M2）",
           "head": "头共享（W3）", "quant_head": "组合：头共享 × 量化（本批）",
           "window_anchor": "组合：窗口 × 锚点（本批）"}
CLASSES_EN = {"baseline": "no compression (trunk fp16)", "quant": "quantization (R1)",
              "window": "window / eviction (M2)", "head": "KV head sharing (W3)",
              "quant_head": "combo: head sharing x quant (this batch)",
              "window_anchor": "combo: window x anchor (this batch)"}

#: 各臂的类别（画图/表用）。新组合臂单列两类，其余沿用 2619 的三类。
MARK = {"fp16": "b", "int8": "8", "win16": "w", "win32": "w", "win64": "w",
        "mha": "M", "gqa": "G", "mqa": "Q",
        "gqa_int8": "g", "mqa_int8": "q", "win16_anchor16": "a",
        "win32_anchor32": "A", "win32_anchor8": "n"}


def class_of(spec: dict, arm: str) -> str:
    if arm in ("fp16", "mha"):
        return "baseline"
    return spec["kind"]


# ----------------------------------------------------------------- 臂的装载 / 分派


def load_model_for_arm(cfg, arm: str, spec: dict, dev: str):
    if spec["kind"] in ("head", "quant_head"):
        base = spec.get("base_arm", arm)
        return b19.load_head_arm(cfg, base, dev)
    return b19.load_trunk(cfg, dev)


def make_logits_fn(model, spec: dict, T: int, dev: str):
    """新组合臂走 `gaps_arms`，其余臂走 2619 的分派（同一 `logits_fn` 接口）。"""
    if spec["kind"] in ga.NEW_KINDS:
        return ga.make_logits_fn(model, spec, T, dev)
    return b19.make_logits_fn(model, spec, T, dev)


def activity_of(spec: dict, model, T: int, dev: str, probe: torch.Tensor):
    if spec["kind"] in ga.NEW_KINDS:
        return ga.activity_of(spec, model, T, dev, probe)
    return b19.activity_of(spec, model, T, dev, probe)


# ----------------------------------------------------------------- 图表渲染（含新类）


def ascii_chart(points, width=72, height=24) -> str:
    xs = [math.log10(p["bytes_per_position"]) for p in points]
    ys = [p["damage_bpc"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    if y1 - y0 < 1e-9:
        y1 = y0 + 0.01
    grid = [[" "] * width for _ in range(height)]
    order = [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (3, 0), (-3, 0),
             (0, -1), (1, -1), (-1, -1), (0, 1), (1, 1), (-1, 1)]
    for p, xv, yv in zip(points, xs, ys):
        cx = int(round((xv - x0) / (x1 - x0) * (width - 1)))
        cy = height - 1 - int(round((yv - y0) / (y1 - y0) * (height - 1)))
        for dx, dy in order:
            nx, ny = cx + dx, max(0, min(height - 1, cy + dy))
            if 0 <= nx < width and grid[ny][nx] == " ":
                grid[ny][nx] = p["mark"]
                break
    lines = ["x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；"
             "y 轴 = damage_bpc（上=更差）", ""]
    for r, row in enumerate(grid):
        yval = y1 - r * (y1 - y0) / (height - 1)
        lines.append(f"{yval:+7.4f} |" + "".join(row))
    lines.append(" " * 9 + "+" + "-" * width)
    lines.append(" " * 10 + " ".join(f"{math.pow(10, x0 + i * (x1 - x0) / 4):>8.0f}"
                                     for i in range(5)))
    lines.append(" " * 10 + "每位置 KV 字节（解析）")
    lines.append("")
    by_mark: dict = {}
    for p in points:
        by_mark.setdefault(p["mark"], []).append(p["arm"])
    lines.append("标记→点：" + "  ".join(f"{m}={','.join(a)}" for m, a in by_mark.items()))
    lines.append("标记→类别：" + "  ".join(f"{k}=({c})" for k, c in CLASSES.items()))
    return "\n".join(lines)


def markdown_table(points) -> str:
    head = ("| 点 | 类别 | 每位置 KV 字节（解析） | vs 本族基线 | bpc | Δbpc==damage | "
            "量化增量 | 相对 stock | 合成召回 copy（汇总） | 95% CI | copy 毁源 | "
            "vs chance | 来源路径 | 口径 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for p in points:
        ci = f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}]"
        inc = (f"{p['quant_increment_bpc']:+.6f}" if p.get("quant_increment_bpc")
               is not None else "—")
        rows.append(
            f"| `{p['arm']}` | {CLASSES[p['class']]} | {p['bytes_per_position']:.0f} "
            f"({p['kv_reduction_vs_family_baseline'] * 100:.1f}%↓) | "
            f"{p['family_baseline']} | {p['bpc']:.6f} | {p['damage_bpc']:+.6f}"
            f"{' ⚠不可排序' if abs(p['damage_bpc']) < VARIANCE_BAND_BPC and p['damage_bpc'] != 0 else ''} | "
            f"{inc} | {p['total_damage_vs_stock']:+.6f} | "
            f"{p['recall_copy']['accuracy']:.4f}"
            f"{' (地板)' if p['recall_copy']['at_floor'] else ''} | {ci} | "
            f"{p['recall_copy_source_corrupt']['accuracy']:.4f} | "
            f"{p['recall_copy']['accuracy'] / p['chance_uniform']:.2f}× | "
            f"`{p['source']}` | 解析字节（非实测显存） |")
    return head + "\n".join(rows) + "\n"


def render_table(cfg, points, controls, rec) -> str:
    L = []
    L.append("# 2620 · KV 预算图缺失组合 —— 一张表")
    L.append("横轴公式（判据 C1，与 2619 **逐字同一式**）："
             "`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · "
             "value_bytes · (positions_kept / T)`，T = "
             f"{cfg['model']['seq_len']}。**本列是解析口径**；实测显存单列在文末。"
             "`window_anchor` 档的 `positions_kept = window + anchors`（锚点占用位置预算）。")
    L.append("bpc 与召回全部由本实验**同进程、同一路径**重测（复用 2619 的 `bpc_of` / "
             "`synth_accuracy`），与来源实验逐一对照见 `results.json::cross_check_vs_source`，"
             "与 2619 的参考臂逐位对照见 `results.json::reproducibility_vs_2619`。"
             f"召回地板：chance = {controls['chance_uniform']:.4f}（1/64）、"
             f"局部先验 = {controls['local_prior_copy']['accuracy']:.4f}（copy）/ "
             f"{controls['local_prior_induction']['accuracy']:.4f}（induction）/ "
             f"{controls['local_prior_var_recall']['accuracy']:.4f}（var_recall）。"
             "『量化增量』= 组合臂 bpc − 同架构 fp16 臂 bpc（隔离 int8 叠加代价）；"
             "『相对 stock』= bpc − 1.483569（不压缩 trunk 的释放版 bpc）。")
    L.append(markdown_table(points).rstrip("\n"))

    L.append("## 表 2 · 逐任务召回 + copy 分 src_pos（同一仪器，同一 seed）\n\n"
             "| 点 | 每位置字节 | copy 汇总 | copy@src0（锚区） | copy@src96 | "
             "copy@src191 | induction | var_recall | 不可分辨的腿 |\n"
             "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for p in points:
        ps = p["recall_copy"]["per_setting"]
        floors = [n for n, k in (("copy", "recall_copy"), ("induction", "recall_induction"),
                                 ("var_recall", "recall_var_recall"))
                  if p[k]["accuracy"]
                  <= b19.floor_threshold(p[k]["n"], controls["chance_uniform"])]
        rows.append(f"| `{p['arm']}` | {p['bytes_per_position']:.0f} | "
                    f"{p['recall_copy']['accuracy']:.4f} | "
                    f"{ps['0']['accuracy']:.4f} | {ps['96']['accuracy']:.4f} | "
                    f"{ps['191']['accuracy']:.4f} | "
                    f"{p['recall_induction']['accuracy']:.4f} | "
                    f"{p['recall_var_recall']['accuracy']:.4f} | "
                    f"{', '.join(floors) if floors else '（无）'} |")
    L.append("\n".join(rows))
    L.append("> 判据 C4：某腿 ≤ `chance + 2·σ_wilson` ⇒ 在该任务上**不可分辨**"
             "（不是「机制无效」，SCHEMA §19）。copy@src0 是唯一落在锚区（绝对位置 "
             "[0,anchors)）的源档，锚点臂对它有效、对 src96/191 无效——这条分解是"
             "「锚点是不是拉动窗口召回最便宜的杠杆」的判读核心。")

    L.append("## 表 3 · 与来源实验逐一对照\n\n"
             "| 点 | 本实验 bpc | 来源实验路径 | 来源值 | abs diff | 一致 |\n"
             "|---|---|---|---|---|---|")
    for r in rec["cross_check_vs_source"]["rows"]:
        L.append(f"| `{r['arm']}` | {r['this_bpc']:.6f} | `{r['source']}` | "
                 f"{r.get('source_bpc', '—')} | {r.get('abs_diff', '—')} | "
                 f"{'✓' if r['match'] else ('—' if r['match'] is None else '✗')} |")

    L.append("## 表 4 · 与 2619 参考臂的复现性（跨 run 可控性）\n\n"
             "| 点 | 本实验 bpc | 2619 bpc | abs diff | copy abs diff | 一致 |\n"
             "|---|---|---|---|---|---|")
    for r in rec["reproducibility_vs_2619"]["rows"]:
        L.append(f"| `{r['arm']}` | {r['bpc']:.6f} | {r['bpc_2619']:.6f} | "
                 f"{r['bpc_abs_diff']:.2e} | {r['copy_abs_diff']:.2e} | "
                 f"{'✓' if r['match'] else '✗'} |")

    L.append("## 实测显存（**与解析列分列**，判据 C2）\n\n"
             f"- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）："
             f"**{rec['memory_convention']['measured_gpu_peak_mib']} MiB**（全臂共用一个进程，"
             f"是上界，不是逐臂读数）。\n"
             "- 为什么不逐臂报：评测实现把 K/V 立刻反量化、窗口/锚点按掩码取子集，SDPA 仍"
             "物化 `T×T` 分数矩阵 ⇒ **实测峰值不随档下降**，把它当横轴会得出「量化/窗口不省"
             "显存」的错结论。\n"
             "- 真 cache 实现的实测峰值仍是 U3 表 4 的缺失项（不在本批范围）。")

    L.append("## ASCII 图\n\n```\n" + ascii_chart(points) + "\n```")
    return "\n\n".join(L) + "\n"


def render_png(points, controls, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"baseline": ("k", "o"), "quant": ("tab:red", "s"),
             "window": ("tab:blue", "^"), "head": ("tab:green", "D"),
             "quant_head": ("tab:purple", "P"), "window_anchor": ("tab:orange", "*")}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))
    for p in points:
        c, m = style[p["class"]]
        axes[0].scatter(p["bytes_per_position"], p["damage_bpc"], color=c, marker=m, s=60)
        axes[0].annotate(p["arm"], (p["bytes_per_position"], p["damage_bpc"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        axes[1].scatter(p["bytes_per_position"], p["recall_copy"]["accuracy"],
                        color=c, marker=m, s=60)
        axes[1].annotate(p["arm"], (p["bytes_per_position"], p["recall_copy"]["accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("bytes per position (analytic)")
    axes[0].axhline(VARIANCE_BAND_BPC, ls="--", c="gray", lw=1)
    axes[0].text(points[0]["bytes_per_position"], VARIANCE_BAND_BPC * 1.06,
                 "0.014 noise band", fontsize=7, color="gray")
    axes[0].set_ylabel("damage_bpc (vs own family baseline; lower = better)")
    axes[0].set_title("leg 1: clean bpc")
    axes[1].axhline(controls["chance_uniform"], ls=":", c="gray", lw=1)
    axes[1].text(points[0]["bytes_per_position"], controls["chance_uniform"] * 1.2,
                 "chance 1/64", fontsize=7, color="gray")
    axes[1].axhline(controls["local_prior_copy"]["accuracy"], ls="--", c="gray", lw=1)
    axes[1].text(points[0]["bytes_per_position"], controls["local_prior_copy"]["accuracy"] * 1.15,
                 "local prior (copy)", fontsize=7, color="gray")
    axes[1].set_ylabel("synthetic recall: copy (pooled; higher = better)")
    axes[1].set_title("leg 2: long-range retrieval (synthetic, local-prior resistant)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=CLASSES_EN[k])
               for k, (c, m) in style.items()]
    axes[1].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("2620 · KV budget gaps (T=256; bytes/position ANALYTIC, not measured memory)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def write_charts(cfg, points, controls, rec, out_dir) -> None:
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(cfg, points, controls, rec))
    with open(os.path.join(out_dir, "chart.txt"), "w") as f:
        f.write(ascii_chart(points) + "\n")
    try:
        render_png(points, controls, os.path.join(out_dir, "chart.png"))
        rec["chart_png"] = True
    except Exception as e:                                  # pragma: no cover
        rec["chart_png"] = f"unavailable: {e}"


# ----------------------------------------------------------------- 主流程


def source_of(cfg, arm: str) -> str:
    spec = cfg["arms"][arm]
    if spec["kind"] == "quant_head":
        return ("axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/"
                "results.json × R1 int8（本批组合，无单一来源）")
    if spec["kind"] == "quant":
        return "axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json"
    if spec["kind"] in ("window", "window_anchor"):
        return ("axes/M-carrier/M2-carrier-compression/experiments/"
                "2615-retention-retrieval-leg-13m/results.json")
    return "axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json"


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
        rec["code"] = {name: b19.sha256_file(os.path.join(ROOT, name))[:16]
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
           "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke)}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))

    # ---------- ① 仪器自检（与 2619 同一路径） ----------
    trunk, trunk_ck = b19.load_trunk(cfg, dev)
    trunk.eval()
    fp32_fn = b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    anchor = cfg["controls"]["baseline_anchor_bpc"]
    lib_bpc = b19.masked_bpc(trunk, eval_arr, None, n_batches=cfg["eval"]["n_batches"],
                             batch_size=cfg["eval"]["batch_size"],
                             window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = b19.bpc_of(fp32_fn, eval_arr, T, cfg["eval"]["n_batches"],
                        cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    tasks = b19.build_tasks(cfg, smoke=args.smoke)
    probe = tasks["var_recall"]
    lib_acc = b19.lib_accuracy(trunk, probe, device=dev)
    my_acc = b19.synth_accuracy(fp32_fn, probe, dev)
    rec["instrument"] = {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "retention.masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc), "bitwise_equal": my_bpc == lib_bpc,
                     "anchor_2614_release_bpc": anchor,
                     "abs_diff_vs_anchor": abs(my_bpc - anchor),
                     "mode": "fp32（恒等：KV 数值未改）"},
        "synth_path": {"run_2620::synth_accuracy": my_acc["accuracy"],
                       "longrange_tasks.accuracy": lib_acc["accuracy"],
                       "n_equal": my_acc["n"] == lib_acc["n"],
                       "n_hits_equal": my_acc["n_hits"] == lib_acc["n_hits"],
                       "by_distance_equal": all(
                           my_acc["by_distance"][k]["accuracy"]
                           == lib_acc["by_distance"][k]["accuracy"]
                           for k in lib_acc["by_distance"])},
        "quant_head_self_check": None,
        "note": "两条腿都先证明「本实验路径 == 既有仪器路径」，再进入跨臂比较（SCHEMA §9）。",
    }
    print(f"[自检] 恒等臂 bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f} "
          f"(逐位={rec['instrument']['bpc_path']['bitwise_equal']})；"
          f"synth={my_acc['accuracy']:.4f} vs lib={lib_acc['accuracy']:.4f}", flush=True)

    # ---------- ② 对照臂：chance / 局部先验 / 毁源 ----------
    copy_positions = cfg["tasks"]["copy"]["src_positions"]
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
                                "**逐臂**报该对照（源区间 = 该档自己的 [src_pos, src_pos+32)）。"),
    }
    rec["controls"] = controls
    print(f"[对照] copy 局部先验={controls['local_prior_copy']['accuracy']:.4f} / "
          f"chance={controls['chance_uniform']:.4f}", flush=True)

    # ---------- ③ 逐臂：bpc + 合成召回 ----------
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)
    points = []
    n_arms = len(cfg["arms"])
    for i, (arm, spec) in enumerate(cfg["arms"].items(), 1):
        model, ck = load_model_for_arm(cfg, arm, spec, dev)
        if spec["kind"] == "quant_head":
            ck = {**ck, "base_arm": spec["base_arm"], "quantized_mode": spec["mode"],
                  "note": "头共享 checkpoint（fp16 存储）× 评测期 KV 量化往返；权重未改"}
        fn = make_logits_fn(model, spec, T, dev)
        bpc = b19.bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"],
                         cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
        copy_pooled = b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
        copy_corrupt = b19.pooled_accuracy(fn, tasks["copy_corrupt_by_pos"], dev)
        ind_pooled = b19.pooled_accuracy(fn, tasks["induction_by_start"], dev)
        var_pooled = b19.pooled_accuracy(fn, tasks["var_recall_by_d"], dev)
        geo = ga.bytes_per_position(spec, n_layers=cfg["model"]["n_layers"],
                                    n_heads=cfg["model"]["n_heads"],
                                    head_dim=cfg["model"]["d_model"] // cfg["model"]["n_heads"],
                                    T=T)
        chance = controls["chance_uniform"]
        at_floor = copy_pooled["accuracy"] <= b19.floor_threshold(copy_pooled["n"], chance)
        points.append({
            "arm": arm, "role": spec["role"], "family": spec["family"],
            "class": class_of(spec, arm), "checkpoint": ck,
            "bpc": round(bpc, 6),
            "activity": activity_of(spec, model, T, dev, probe_tokens),
            "recall_copy": {**copy_pooled, "at_floor": bool(at_floor)},
            "recall_copy_source_corrupt": copy_corrupt,
            "recall_induction": ind_pooled, "recall_var_recall": var_pooled,
            "chance_uniform": chance,
            "local_prior_copy": controls["local_prior_copy"]["accuracy"],
            "local_prior_induction": controls["local_prior_induction"]["accuracy"],
            "local_prior_var_recall": controls["local_prior_var_recall"]["accuracy"],
            "mark": MARK[arm], "source": source_of(cfg, arm),
            **geo,
        })
        print(f"[{i}/{n_arms}] {arm:<15} bpc={bpc:.6f} copy={copy_pooled['accuracy']:.4f} "
              f"(src0={copy_pooled['per_setting']['0']['accuracy']:.4f}) "
              f"corrupt={copy_corrupt['accuracy']:.4f} bytes/pos={geo['bytes_per_position']:.0f}",
              flush=True)

    # ---------- ④ damage / 相对 stock / 量化增量 ----------
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

    # ---------- ⑤ 与来源实验逐一对照 ----------
    src = b19.source_values()
    rows = []
    for p in points:
        row = {"arm": p["arm"], "this_bpc": p["bpc"], "source": p["source"]}
        w3, r1, m2 = src.get("W3", {}), src.get("R1", {}), src.get("M2", {})
        if p["arm"] in r1:
            row["source_bpc"] = r1[p["arm"]]["bpc"]
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
        "note": "参考臂应与其来源实验逐位一致；新组合臂没有来源（match=None，属预期）。"}

    # ---------- ⑥ 与 2619 的复现性 ----------
    rep = {"compared_to": os.path.relpath(RES_2619_PATH, ROOT), "rows": []}
    if os.path.exists(RES_2619_PATH):
        prev = {p["arm"]: p for p in json.load(open(RES_2619_PATH))["points"]}
        for arm in cfg["compare_against_2619"]["arms"]:
            a, b = by_arm[arm], prev[arm]
            rep["rows"].append({
                "arm": arm, "bpc": a["bpc"], "bpc_2619": b["bpc"],
                "bpc_abs_diff": abs(a["bpc"] - b["bpc"]),
                "copy_abs_diff": abs(a["recall_copy"]["accuracy"]
                                     - b["recall_copy"]["accuracy"]),
                "match": (abs(a["bpc"] - b["bpc"]) <= 1e-5
                          and abs(a["recall_copy"]["accuracy"]
                                  - b["recall_copy"]["accuracy"]) <= 1e-5)})
        rep["max_bpc_abs_diff"] = max(r["bpc_abs_diff"] for r in rep["rows"])
        rep["all_matched"] = all(r["match"] for r in rep["rows"])
    rec["reproducibility_vs_2619"] = rep

    # ---------- ⑦ 汇总落盘 ----------
    rec["points"] = points
    rec["memory_convention"] = {
        "analytic_bytes_per_position": "见 points[*].bytes_per_position（判据 C1 的唯一公式）",
        "measured_gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "isolation_note": ("判据 C2：解析字节是本图的**横轴**；实测显存峰值单列在此，"
                           "**不**进横轴。评测实现把 K/V 立刻反量化 / 按掩码取子集"
                           "（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值不随档下降。"),
    }
    rec["code"] = {name: b19.sha256_file(os.path.join(ROOT, name))[:16]
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
