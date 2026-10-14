"""2653 · fp8 vs int8+块标尺（同「每位置部署字节」轴）—— 实验入口。

契约：同目录 `README.md`（运行前预登记）+ `config.json`（判据线**运行前**写入）。
语义层：同目录 `fp8_arms.py`（fp8 往返 + 块标尺解析字节账 + head_dim/G 阶梯）。

设备口径（ticket 66 §13 F5）：**整程 CPU，不取 GPU 锁**。单卡 4090 被多轴共享、
`/tmp/pathlm_gpu.lock` 长期被其它轴占用 ⇒ 纯 CPU 推理按 SCHEMA §20 不适用该锁。
**全部臂同一设备**；跨设备（CPU vs GPU 历史锚 1.483569）不构成判据，判据用同设备、
同窗口序的**配对 Δ**（`pathlm.data.batch` 的窗口序与设备无关）。

逐臂持久化（子 agent 有硬超时，先保 results.json）：每跑完一臂立刻把已完成的臂写进
`results.json`（`partial`/`arms_completed`/`arms_pending`）并重画 `table.md`。

用法：
  python3 run.py                # 全跑（4 主臂 + 3 辅助臂 + 仪器自检）
  python3 run.py --smoke        # 冒烟：n_batches=1、只跑 fp16 + fp8_e4m3
  python3 run.py --charts-only  # 只从已有 results.json 重画 table.md（纯 CPU）
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import fp8_arms as fa                                                        # noqa: E402
from pathlm.data import load_enwik8_full                                     # noqa: E402

P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
P2631 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal")
P2632 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2621 的评估管线（内含 2619 的 b19）；只读复用。
b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")
#: 2631 的前缀不变性仪器（两条反例做成可失败的运行时检查）。
cc2631 = _load_module(os.path.join(P2631, "causal_check.py"), "causal_check_2631")
#: 2632 的 scheme-aware 前向（**只作 int8 块标尺的实现层 equality 对照**，不改）。
ba2632 = _load_module(os.path.join(P2632, "block_arms.py"), "block_arms_2632")

MAIN_ARM_ORDER = ("fp16", "int8_block_g32", "fp8_e4m3", "fp8_e5m2")
SIDE_ARM_ORDER = ("fp8_e4m3_tokscale", "int8_block_g8", "int8_block_g16")
CHANCE = 0.0156          # 1/64（synth 词表 64），与 2619/2621/2632 同口径


# ----------------------------------------------------------------- 小工具


def pooled_synth(fn, task_map, dev, chance=CHANCE) -> dict:
    """把多档任务的批拼起来求汇总准确率 + Wilson 95% CI（与 `b19.pooled_accuracy` 同式，
    只是逐档各前向一次、不重复跑汇总批 ⇒ 每档一次前向的成本）。"""
    per, hits, n = {}, 0, 0
    for k, t in task_map.items():
        a = b21.b19.synth_accuracy(fn, t, dev)
        per[str(k)] = {"accuracy": a["accuracy"], "n": a["n"], "n_hits": a["n_hits"],
                       "wilson95": a["wilson95"]}
        hits += a["n_hits"]
        n += a["n"]
    lo, hi = b21.b19.wilson_ci(hits, n) if n else (0.0, 0.0)
    at_floor = bool(hits / max(n, 1) <= b21.b19.floor_threshold(n, chance)) if n else None
    return {"accuracy": round(hits / max(n, 1), 6), "n": n, "n_hits": hits,
            "wilson95": [round(lo, 6), round(hi, 6)], "at_floor": at_floor,
            "chance": chance,
            "floor_threshold": round(b21.b19.floor_threshold(n, chance), 6) if n else None,
            "per_setting": per}


def build_tasks(cfg, smoke: bool) -> dict:
    old = cfg["tasks"]["copy"]["n_batches"]
    cfg["tasks"]["copy"]["n_batches"] = 1 if smoke else old
    cfg["tasks"]["induction"]["n_batches"] = 1 if smoke else cfg["tasks"]["induction"]["n_batches"]
    cfg["tasks"]["var_recall"]["n_batches"] = 1 if smoke else cfg["tasks"]["var_recall"]["n_batches"]
    return b21.b19.build_tasks(cfg, smoke=smoke)


# ----------------------------------------------------------------- 一臂的读数


def eval_arm(arm, spec, model, cfg, dev, tasks, eval_arr, ledger, main_n, pair_n,
             scan_n, seed_main, measure=None) -> dict:
    """一臂的读数。`measure` 选哪些腿要跑（`main`/`pairing`/`scan`/`recall`），
    默认全跑；扫描臂只跑 `scan`，以把每臂成本控在预算内。"""
    measure = measure or {"main": True, "pairing": True, "scan": True, "recall": True}
    T = cfg["model"]["seq_len"]
    bs = cfg["eval"]["batch_size"]
    probe = tasks["var_recall"].tokens[:4].to(dev)
    fn = fa.make_logits_fn(model, spec, T, dev)
    t0 = time.time()
    bpc_main = (b21.b19.bpc_of(fn, eval_arr, T, main_n, bs, seed_main, dev)
                if measure["main"] else None)
    paired = ({str(ws): round(b21.b19.bpc_of(fn, eval_arr, T, pair_n, bs, ws, dev), 6)
               for ws in cfg["eval"]["pairing_seeds"]}
              if measure["pairing"] else None)
    scan = (round(b21.b19.bpc_of(fn, eval_arr, T, scan_n, bs, seed_main, dev), 6)
            if scan_n and measure["scan"] else None)
    out = {
        "arm": arm, "mode": spec["mode"], "scheme": spec.get("scheme"),
        "group": spec.get("group"), "scale_mode": spec.get("scale_mode", "none"),
        "bpc": (round(bpc_main, 6) if bpc_main is not None else None),
        "bpc_caliber": ({"n_batches": main_n, "batch_size": bs,
                         "window_seed": seed_main, "device": dev} if bpc_main is not None
                        else None),
        "bpc_paired_windows": paired,
        "bpc_paired_caliber": ({"n_batches": pair_n, "batch_size": bs,
                                "window_seeds": cfg["eval"]["pairing_seeds"], "device": dev}
                               if paired else None),
        "bpc_scan": scan,
        "bpc_scan_caliber": ({"n_batches": scan_n, "window_seed": seed_main, "device": dev}
                             if scan is not None else None),
        "prefix_invariance": cc2631.prefix_invariance_of_logits(fn, probe),
        "activity": fa.activity_of(spec, model, probe),
        "deployment": ledger["per_arm"].get(arm) or ledger["per_side_arm"][arm],
        "measure": measure,
        "wall_seconds_this_arm": None,
    }
    if measure["recall"]:
        out["recall_copy"] = pooled_synth(fn, tasks["copy_by_pos"], dev)
        out["recall_induction"] = pooled_synth(fn, tasks["induction_by_start"], dev)
        out["recall_var_recall"] = pooled_synth(fn, tasks["var_recall_by_d"], dev)
    out["wall_seconds_this_arm"] = round(time.time() - t0, 2)
    return out


# ----------------------------------------------------------------- 表格渲染


def render_table(cfg, rec) -> str:
    led = rec["deployment_ledger"]
    L = []
    L.append("# 2653 · fp8 vs int8+块标尺（同一「每位置部署字节」轴）")
    L.append(
        f"**设备口径 = CPU**（`device: cpu`，全部臂同一设备；不取 GPU 锁）。"
        f"**跨设备不构成判据**：本表不宣称与 GPU 历史锚 1.483569 逐位相等，只并列报告并声明设备差异。"
        f"横轴 = 每位置**部署**字节 = `2·L·n_kv_heads·head_dim·(每值字节)·(kept/T)`，"
        f"每位置值数 = {led['values_per_position']:.0f}（K+V），fp16 基线 = "
        f"{led['fp16_bytes_per_position']:.0f} B/位置。"
        f"rank 线：主判带 = 跨格 **0.014 bpc**（限配对 Δ、≥5 窗口，本 run 有 {len(cfg['eval']['pairing_seeds'])} 个窗口）。")
    L.append("## 表 1 · WP1 主臂（KV 存放格式 = 单因素）\n\n"
             "| 臂 | 每值字节 | 每位置字节 | ×fp16 | 省幅 | clean bpc(CPU,24×16,s2) | 配对 Δbpc vs fp16(5 窗) max\\|Δ\\| | 越 0.014 带 | "
             "copy 召回 | 95% CI | at_floor | 前缀不变(未来/邻居) |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    pa = rec.get("paired_delta", {}).get("per_arm", {})
    rows = []
    for p in rec.get("points", []):
        d = pa.get(p["arm"], {})
        max_abs = f"{d['max_abs']:.6f}" if d else "—"
        verdict = ("越带" if d["exceeds_band_0_014"] else "不越带") if d else "—"
        rows.append(
            f"| `{p['arm']}` | {p['deployment']['bytes_per_value']:.4f} | "
            f"{p['deployment']['bytes_per_position']:.0f} | ×{p['deployment']['ratio_vs_fp16']:.3f} | "
            f"{p['deployment']['reduction_vs_fp16']:.3%} | {p['bpc']:.6f} | "
            f"{max_abs} | {verdict} | "
            f"{p['recall_copy']['accuracy']:.4f} | "
            f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}] | "
            f"{'是' if p['recall_copy']['at_floor'] else '否'} | "
            f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
            f"{p['prefix_invariance']['peer_request_logits_equal']} |")
    L[-1] += "\n" + "\n".join(rows) if rows else "\n（尚无完成臂）"
    L.append("## 表 2 · 辅助臂（fp8 是否真免标尺 / WP2 的 G 扫描）\n\n"
             "| 臂 | 每值字节 | 每位置字节 | clean bpc | 口径 | 配对 Δbpc vs fp16 max\\|Δ\\| | 实测 n_scales/n_values | 饱和占比 | 前缀不变 |\n"
             "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for p in rec.get("side_points", []):
        act = p["activity"]
        nv = max(act.get("n_values", 0), 1)
        if p["bpc"] is not None:
            val, cal = f"{p['bpc']:.6f}", (f"{p['bpc_caliber']['n_batches']}×"
                                           f"{p['bpc_caliber']['batch_size']} 主表口径")
        else:
            val = f"{p['bpc_scan']:.6f}"
            cal = (f"{p['bpc_scan_caliber']['n_batches']}×{cfg['eval']['batch_size']} scan 口径")
        d = pa.get(p["arm"], {})
        dmax = (f"{d['max_abs']:.6f}" if d else "—")
        rows.append(f"| `{p['arm']}` | {p['deployment']['bytes_per_value']:.4f} | "
                    f"{p['deployment']['bytes_per_position']:.0f} | {val} | {cal} | "
                    f"{dmax} | "
                    f"{act.get('n_scales', 0)}/{nv} = {act.get('n_scales', 0) / nv:.4f} | "
                    f"{act.get('saturated_frac', 0):.4f} | "
                    f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
                    f"{p['prefix_invariance']['peer_request_logits_equal']} |")
    L[-1] += "\n" + "\n".join(rows) if rows else "\n（尚无完成臂）"
    L.append("## 表 3 · WP2 解析阶梯（head_dim × G ⇒ 每值字节 / 每位置字节 / 省幅）\n\n"
             "| head_dim | G | 合法 | 每值字节 `1+2/G` | 每位置字节 | ×fp16 | 省幅 | 前缀不变 |\n"
             "|---|---|---|---|---|---|---|---|")
    rows = []
    for r in led["head_dim_ladder"]:
        rows.append(f"| {r['head_dim']} | {r['group']} | {'是' if r['legal'] else '否（' + r['reason'] + '）'} | "
                    f"{r['bytes_per_value']:.5f} | {r['bytes_per_position']:.0f} | "
                    f"×{r['ratio_vs_fp16']:.4f} | {r['reduction_vs_fp16']:.3%} | "
                    f"{'True' if r['prefix_invariant'] else 'False'} |")
    L[-1] += "\n" + "\n".join(rows)
    cap = led["head_dim_cap"]
    L.append(f"> **封顶**：head_dim=32 ⇒ G 上限 32 ⇒ 每值 {cap['32']['bytes_per_value']:.4f} B、"
             f"每位置 {cap['32']['bytes_per_position']:.0f} B、省幅 **{cap['32']['reduction_vs_fp16']:.3%}**（G=64 需跨 token，"
             f"违反前缀确定，故 head_dim=32 封顶于此）；head_dim=64 ⇒ G=64 合法 ⇒ 每值 "
             f"{cap['64']['bytes_per_value']:.5f} B、每位置 {cap['64']['bytes_per_position']:.0f} B、省幅 "
             f"**{cap['64']['reduction_vs_fp16']:.3%}**（更接近 50%）。**head_dim=64 的省幅是解析值**；"
             f"若 `wp2_head_dim_64.empirical` 为 null ⇒ 该点无实测支撑、不外推。")
    ins = rec.get("instrument")
    if ins:
        L.append("## 表 4 · 仪器自检\n\n"
                 f"- fp32 恒等臂：`bpc_of` = {ins['bpc_path']['bpc_of']:.6f}，"
                 f"`retention.masked_bpc`（独立第二路径）= {ins['bpc_path']['masked_bpc']:.6f}，"
                 f"逐位相等 = **{ins['bpc_path']['bitwise_equal']}**。\n"
                 f"- 本模块 fp32 前向与 `kv_quant` 生产前向逐位相等 = "
                 f"**{ins['fp32_forward_eq_2619']}**。\n"
                 f"- int8 块标尺 G=32 与 `2632/block_arms` 前向逐位相等 = "
                 f"**{ins['int8_block_eq_2632']}**。\n"
                 f"- 前缀不变性（fp32 恒等臂，探针 4 请求）："
                 f"{ins['prefix_invariance_fp32']}\n"
                 f"> 跨设备声明：GPU 历史锚 {ins['anchor_gpu']['clean_bpc']} bpc（device=cuda）"
                 f"与本 run 的 CPU 读数**不同设备**，不构成判据、也不声称逐位相等。")
    else:
        L.append("## 表 4 · 仪器自检\n\n（尚未运行）")
    return "\n\n".join(L) + "\n"


def dump(out_dir, rec, cfg) -> None:
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(cfg, rec))


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--charts-only", action="store_true")
    ap.add_argument("--arms", default=None, help="逗号分隔，只跑这些主臂（调试用）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.config))
    parent = json.load(open(os.path.join(ROOT, cfg["parent"]["config"])))
    for k in ("model", "data", "tokens", "checkpoints"):
        cfg[k] = parent[k]
    cfg["head_dim"] = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    dev = "cpu"
    T = cfg["model"]["seq_len"]

    if args.charts_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        with open(os.path.join(args.out_dir, "table.md"), "w") as f:
            f.write(render_table(rec["config"], rec))
        print(f"[charts-only] 重画 table.md → {args.out_dir}", flush=True)
        return 0

    arms = [a for a in (args.arms.split(",") if args.arms else MAIN_ARM_ORDER)]
    if args.smoke:
        arms = ["fp16", "fp8_e4m3"]
    side_arms = [] if args.smoke else list(SIDE_ARM_ORDER)
    main_n = 1 if args.smoke else cfg["eval"]["n_batches"]
    pair_n = 1 if args.smoke else cfg["eval"]["pairing_n_batches"]
    scan_n = 0 if args.smoke else cfg["eval"]["scan_n_batches"]
    seed_main = cfg["eval"]["window_seed"]
    ledger = fa.deployment_ledger(cfg)

    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"], "scope": cfg["scope"],
           "config": cfg, "criterion": cfg["criterion"], "device": dev,
           "caliber_note": cfg["device"]["cross_device_rule"],
           "gpu_lock": None, "gpu_lock_note": cfg["device"]["reason"],
           "deployment_ledger": ledger, "partial": True,
           "arms_completed": [], "arms_pending": list(arms),
           "side_arms_completed": [], "side_arms_pending": list(side_arms),
           "points": [], "side_points": [], "wp2_head_dim_64": {"empirical": None,
             "status": "未跑（见 run_log.txt / wp2_hd64_status.json）"}}
    dump(args.out_dir, rec, cfg)
    print(f"[落盘] 解析字节账已写入 results.json（fp16={ledger['fp16_bytes_per_position']:.0f}、"
          f"int8 G=32={ledger['per_arm']['int8_block_g32']['bytes_per_position']:.0f}、"
          f"fp8 免标尺={ledger['per_arm']['fp8_e4m3']['bytes_per_position']:.0f} B/位置）", flush=True)

    t0 = time.time()
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    print(f"[装载] trunk sha256 校验通过 = {trunk_ck['sha256_verified']}；评测尾段已就绪"
          f"（{time.time() - t0:.1f}s）", flush=True)
    tasks = build_tasks(cfg, args.smoke)

    # ---------- ① 仪器自检（便宜、先跑） ----------
    eq_tokens = tasks["var_recall"].tokens[:4].to(dev)
    fp32_fn = fa.make_logits_fn(trunk, {"mode": "fp32"}, T, dev)
    my_bpc = b21.b19.bpc_of(fp32_fn, eval_arr, T, 2, cfg["eval"]["batch_size"], seed_main, dev)
    lib_bpc = b21.b19.masked_bpc(trunk, eval_arr, None, n_batches=2,
                                 batch_size=cfg["eval"]["batch_size"],
                                 window_seed=seed_main, device=dev)
    f_b19 = b21.b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    f_2632 = ba2632.masked_kv_quant_logits(trunk, eq_tokens, None, "int8", "block", 32)
    f_mine = fa.kv_logits(trunk, eq_tokens, "int8", "block", 32)
    rec["instrument"] = {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc), "bitwise_equal": my_bpc == lib_bpc,
                     "n_batches": 2, "window_seed": seed_main,
                     "note": "fp32 恒等臂两条独立路径（bpc_of / retention.masked_bpc）"},
        "fp32_forward_eq_2619": bool(torch.equal(fp32_fn(eq_tokens), f_b19(eq_tokens))),
        "int8_block_eq_2632": bool(torch.equal(f_mine[0], f_2632[0])),
        "prefix_invariance_fp32": cc2631.prefix_invariance_of_logits(fp32_fn, eq_tokens),
        "note": "int8 块标尺 G=32 的实现层 equality 证明本实验与 2632/5.1 是同一条前向；"
                "fp8 臂只是把同一个往返算子换成 fp8。",
    }
    rec["instrument"]["anchor_gpu"] = {"clean_bpc": 1.483569, "device": "cuda",
                                       "source": "2614/2610 口径", "comparable": False,
                                       "reason": cfg["device"]["cross_device_rule"]}
    dump(args.out_dir, rec, cfg)
    print(f"[自检] bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f} 逐位={my_bpc == lib_bpc}；"
          f"fp32 前向 equality={rec['instrument']['fp32_forward_eq_2619']}；"
          f"int8 块标尺 eq 2632={rec['instrument']['int8_block_eq_2632']}", flush=True)
    assert rec["instrument"]["fp32_forward_eq_2619"], "本模块 fp32 前向与生产前向不逐位相等"
    assert rec["instrument"]["int8_block_eq_2632"], "int8 块标尺前向与 2632 不逐位相等"
    assert rec["instrument"]["bpc_path"]["bitwise_equal"], "两条 bpc 路径不逐位相等"

    # ---------- ② 主臂（逐臂持久化） ----------
    for arm in arms:
        spec = cfg["arms"][arm]
        p = eval_arm(arm, spec, trunk, cfg, dev, tasks, eval_arr, ledger, main_n, pair_n,
                     scan_n, seed_main)
        rec["points"].append(p)
        rec["arms_completed"].append(arm)
        rec["arms_pending"] = [a for a in arms if a not in rec["arms_completed"]]
        print(f"[主臂] {arm:<16} bpc={p['bpc']:.6f} 每位置={p['deployment']['bytes_per_position']:.0f} "
              f"copy={p['recall_copy']['accuracy']:.4f} 饱和={p['activity'].get('saturated_frac')} "
              f"inv={p['prefix_invariance']['future_token_past_logits_equal']}/"
              f"{p['prefix_invariance']['peer_request_logits_equal']} "
              f"({p['wall_seconds_this_arm']:.0f}s)", flush=True)
        rec["paired_delta"] = paired_deltas(rec, cfg, seed_main)
        rec["cost_so_far"] = {"wall_seconds": round(time.time() - t0, 2)}
        dump(args.out_dir, rec, cfg)

    # ---------- ③ 辅助臂 ----------
    side_measure = {
        "fp8_e4m3_tokscale": {"main": True, "pairing": True, "scan": False, "recall": True},
        "int8_block_g8": {"main": False, "pairing": False, "scan": True, "recall": False},
        "int8_block_g16": {"main": False, "pairing": False, "scan": True, "recall": False},
    }
    for arm in side_arms:
        spec = cfg["side_arms"][arm]
        p = eval_arm(arm, spec, trunk, cfg, dev, tasks, eval_arr, ledger, main_n, pair_n,
                     scan_n, seed_main, side_measure.get(arm))
        rec["side_points"].append(p)
        rec["side_arms_completed"].append(arm)
        rec["side_arms_pending"] = [a for a in side_arms
                                    if a not in rec["side_arms_completed"]]
        shown = p["bpc"] if p["bpc"] is not None else p["bpc_scan"]
        print(f"[辅助] {arm:<20} bpc={shown:.6f} 每位置={p['deployment']['bytes_per_position']:.0f} "
              f"饱和={p['activity'].get('saturated_frac')} ({p['wall_seconds_this_arm']:.0f}s)",
              flush=True)
        dump(args.out_dir, rec, cfg)

    # ---------- ④ 结论量 ----------
    rec["paired_delta"] = paired_deltas(rec, cfg, seed_main)
    rec["ranking_band"] = {
        "primary_band_bpc": cfg["criterion"]["band_primary_bpc"],
        "primary_caliber": cfg["criterion"]["band_primary_caliber"],
        "windows_used": cfg["eval"]["pairing_seeds"], "n_windows": len(cfg["eval"]["pairing_seeds"]),
        "standard_met": len(cfg["eval"]["pairing_seeds"]) >= 5,
        "secondary_in_force_band_bpc": round(
            max((max(p["bpc_paired_windows"].values()) - min(p["bpc_paired_windows"].values()))
                for p in rec["points"]), 6) if rec["points"] else None,
        "secondary_caliber": "单臂跨窗口 bpc 极差（同臂在 5 个窗口 seed 上重测；只作辅助描述）",
        "bands_not_mixed": cfg["criterion"]["bands_not_mixed"],
    }
    rec["partial"] = bool(rec["arms_pending"] or rec["side_arms_pending"])
    rec["cost"] = {"device": dev, "wall_seconds": round(time.time() - t0, 2),
                   "wall_minutes": round((time.time() - t0) / 60, 3),
                   "gpu_lock_taken": False, "peak_mem_mib_this_run": None,
                   "note": "纯 CPU；无 GPU 锁、无显存足迹。wall 含全部主臂 + 辅助臂 + 仪器自检。"}
    codes = [os.path.relpath(os.path.join(HERE, f), ROOT) for f in
             ("run.py", "fp8_arms.py", "config.json")]
    rec["code"] = {c: b21.b19.sha256_file(os.path.join(ROOT, c))[:16] for c in codes}
    drop_keys = [k for k in ("cost_so_far",) if k in rec]
    for k in drop_keys:
        rec.pop(k)
    dump(args.out_dir, rec, cfg)
    print(f"[完成] {rec['cost']['wall_minutes']} min；主臂 {len(rec['points'])} / 辅助 {len(rec['side_points'])}；"
          f"partial={rec['partial']} → {args.out_dir}", flush=True)
    return 0


def paired_deltas(rec, cfg, seed_main) -> dict:
    """配对 Δ（臂 − fp16），5 个窗口 seed，同一 n_batches 的均匀口径。"""
    pts = {p["arm"]: p for p in rec["points"] + rec["side_points"]}
    base = pts.get("fp16")
    if base is None:
        return {"per_arm": {}, "note": "fp16 基线臂尚未完成 ⇒ 无法算配对 Δ"}
    band = cfg["criterion"]["band_primary_bpc"]
    per = {}
    for arm, p in pts.items():
        if arm == "fp16":
            continue
        if not p["bpc_paired_windows"] or not base["bpc_paired_windows"]:
            continue
        if p["bpc_paired_caliber"]["n_batches"] != base["bpc_paired_caliber"]["n_batches"]:
            continue
        d = {ws: round(p["bpc_paired_windows"][ws] - base["bpc_paired_windows"][ws], 9)
             for ws in p["bpc_paired_windows"]}
        vals = list(d.values())
        per[arm] = {
            "delta_by_window": d,
            "mean": round(sum(vals) / len(vals), 9),
            "min": min(vals), "max": max(vals),
            "max_abs": round(max(abs(v) for v in vals), 9),
            "sign": "正 = 该臂比 fp16 更差",
            "exceeds_band_0_014": bool(max(abs(v) for v in vals) > band),
            "n_windows": len(vals),
        }
    return {"base_arm": "fp16", "comparison": "臂 − fp16", "band_bpc": band,
            "caliber": f"同窗口 seed、同 n_batches={base['bpc_paired_caliber']['n_batches']}、"
                       f"同 batch_size={base['bpc_paired_caliber']['batch_size']}、同设备（CPU）",
            "per_arm": per,
            "note": "5 个窗口满足『配对 Δ、≥5 窗口』；主判带 = 0.014。"}


if __name__ == "__main__":
    sys.exit(main())
