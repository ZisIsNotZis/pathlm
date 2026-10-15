"""2654 · WP1：按保留预算的窗口曲线 + 窗口 × int8/int4 块标尺组合 —— 实验入口（CPU 口径）。

契约：同目录 `README.md` + `config.json::criterion`（**运行前**写入）；
补充契约：`.scratch/66-axis5-quant/DISPATCH-5.6-PHASEC.md`（CPU 口径、逐臂持久化、同字节水平、块标尺）。

仪器（全部**只读复用**，本目录不改它们）：
  - `2619/run.py`（`b21`→其 `b19`）：`load_trunk` / `bpc_of` / `masked_bpc` / `build_tasks` /
    `pooled_accuracy` / `sync_accuracy` / `bytes_per_position` / `floor_threshold`；
  - `2621/run.py`（`b21`）：臂的分派（`make_logits_fn` / `activity_of` / `bytes_per_position`）；
  - `2621/rest_arms.py`（`ra`）：掩码 × 量化前向（`combo_mask`，per_value 标尺）+ 两条退化路径自检；
  - `2632/block_arms.py`（`ba`）：**块标尺**（`scheme="block"`, `G=32`）的量化前向（本实验量化腿的唯一口径）；
  - `2632/scale_caliber.py`（`sc`）：部署字节账（值 + 标尺）；
  - `2631/causal_check.py`（`cc`）：前缀不变性检验。

用法：
  python3 run.py --smoke                       # 冒烟：2 批 bpc、1 批召回、2 臂，验代码路径
  python3 run.py --stage main                  # 主表臂（见 win_arms.ARM_ORDER）+ 配对带
  python3 run.py --stage extra                 # 预算内补跑剩余臂
  python3 run.py --stage main --deadline-seconds 780
每跑完一臂即写 `results.json` + `table.md`（被切断也留证据）。
"""

import argparse
import hashlib
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

import win_arms as wa                                                        # noqa: E402
from pathlm.data import load_enwik8_full                                     # noqa: E402
from tools.evalkit.longrange_tasks import accuracy as lib_accuracy           # noqa: E402
from tools.evalkit.longrange_tasks import (corrupt_span,                     # noqa: E402
                                           local_prior_accuracy,
                                           check_local_prior_resistance)

P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
P2632 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale")
P2631 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal")
for _p in (P2621, P2632, P2631):
    sys.path.insert(0, _p)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


b21 = _load_module(os.path.join(P2621, "run.py"), "kv_budget_rest_2621")
ra = _load_module(os.path.join(P2621, "rest_arms.py"), "rest_arms_2621")
ba = _load_module(os.path.join(P2632, "block_arms.py"), "block_arms_2632")
sc = _load_module(os.path.join(P2632, "scale_caliber.py"), "scale_caliber_2632")
cc = _load_module(os.path.join(P2631, "causal_check.py"), "causal_check_2631")

#: 历史落盘点的只读摘录（**他设备**：GPU；只作对照/prior art，不作本实验判据）。
#: 来源：2615/2619/2621（逐值标尺或非量化）、2632（块标尺 G=32）。
PRIOR_XCHECK = [
    ("full", "2619/2621 fp16 点（GPU, per_value 无标尺）", 1.483571, 12288.0),
    ("win16", "2615/2619 recent:d=16（GPU）", 2.085526, 768.0),
    ("win32", "2615/2619 recent:d=32（GPU）", 1.695384, 1536.0),
    ("win64", "2615/2619 recent:d=64（GPU）", 1.564077, 3072.0),
    ("win128", "2619 recent:d=128（GPU）", 1.503634, 6144.0),
    ("win160", "2619 recent:d=160（GPU）", 1.492435, 7680.0),
    ("win192", "2619 recent:d=192（GPU）", 1.487744, 9216.0),
    ("win32_int8", "2632 win32_int8（GPU, block G=32）", 1.695617, 816.0),
    ("win64_int8", "2632 win64_int8（GPU, block G=32）", 1.564194, 1632.0),
    ("mqa_int8", "2632 mqa_int8（GPU, block G=32）", 1.493933, 816.0),
    ("gqa_int8", "2632 gqa_int8（GPU, block G=32）", 1.496441, 1632.0),
    ("gqa_win64", "2632 gqa_win64（GPU, fp16）", 1.559100, 768.0),
    ("mqa", "2619/2632 mqa（GPU, fp16）", 1.493763, 1536.0),
    ("gqa", "2619/2632 gqa（GPU, fp16）", 1.496325, 3072.0),
    ("win32_anchor32", "2620/2632 win32+anchor32（GPU, fp16）", 1.590539, 3072.0),
    ("mqa_int4", "2632 mqa_int4（GPU, block G=32）", 1.642750, 432.0),
    ("gqa_int4", "2632 gqa_int4（GPU, block G=32）", 1.553126, 864.0),
]
M2_ERRATA = {"keep32": 0.211815, "keep64": 0.080508,
             "source": "axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m"
                       "（M2 ERRATA：keep32=+0.212、keep64=+0.081，由 2615 的 recent:d 点推得）"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- 臂的前向


def make_fn(arm: str, model, spec: dict, T: int, dev: str):
    """量化腿（int8/int4）走块标尺；其余走 2621/2619 的既有分派。"""
    if wa.is_block_quant(arm):
        return ba.make_logits_fn(model, spec, T, dev, "block", 32, 32)
    return b21.make_logits_fn(model, spec, T, dev)


def activity(arm: str, model, spec: dict, T: int, dev: str, probe: torch.Tensor) -> dict:
    if wa.is_block_quant(arm):
        return ba.activity_of(spec, model, T, dev, probe, "block", 32, 32)
    return b21.activity_of(spec, model, T, dev, probe)


def geo_of(arm: str, spec: dict, cfg: dict, point_geo: dict, scheme_geo: dict | None) -> dict:
    """横轴几何：值字节（判据 C1 旧口径）+ 部署字节（值 + 标尺）+ 2642 同字节水平落点。"""
    v = point_geo["bytes_per_position"]
    dep_calc = wa.deployment_bytes_of(spec, point_geo, cfg)
    out = {"value_bytes_per_position": v,
           "deployment_bytes_per_position": round(dep_calc, 6),
           "ratio_vs_fp16": round(dep_calc / wa.FP16_FULL_BYTES_PER_POSITION, 4),
           "scheme": "block(G=32)" if wa.is_block_quant(arm) else "none",
           **{k: point_geo[k] for k in ("factor_value_bytes", "factor_n_kv_heads_eff",
                                        "factor_positions_kept", "factor_positions_kept_over_T",
                                        "T")}}
    if scheme_geo is not None:
        out["scale_caliber_bytes_per_position"] = scheme_geo["bytes_per_position"]
        out["scale_caliber_match"] = abs(scheme_geo["bytes_per_position"] - dep_calc) < 1e-6
    out["level_2642"] = wa.level_of(dep_calc)
    return out


# ----------------------------------------------------------------- 表格


def render_table(res: dict) -> str:
    pts = res.get("points", [])
    cfg = res["config"]
    L = []
    L.append("# 2654 · WP1 · 保留预算窗口曲线 × 窗口×int8/int4 块标尺组合（CPU 口径）")
    L.append(
        f"**口径**：device = `{res['device']['device']}`（不取 GPU 锁：`gpu_lock_held=false`）；"
        f"主表 `window_seed={cfg['eval']['window_seed']}`、`{cfg['eval']['n_batches']}×"
        f"{cfg['eval']['batch_size']}`；横轴 = **部署字节/位置**（值 + 标尺，块标尺 G=32 ⇒ "
        f"+2/32 B/值），T={cfg['model']['seq_len']}，fp16 全因果 = 12288 B/位置。"
        f"召回地板：chance = {res.get('controls', {}).get('chance_uniform', 1/64):.4f}（1/64）。"
        f"`level_2642` = 命中的 `2642` 同字节水平（精确命中，禁插值）。")
    L.append("## 表 1 · 逐臂读数（两腿并报）\n\n"
             "| 臂 | 类别 | 值字节 | 部署字节 | ×fp16 | level_2642 | bpc | damage vs full | "
             "copy 召回 | 95% CI | at_floor | 毁源 copy | 活性（挡掉因果对/量化 rel_err） |\n"
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        ci = (f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}]"
              if p.get("recall_copy") else "—")
        cr = (f"{p['recall_copy_source_corrupt']['accuracy']:.4f}"
              if p.get("recall_copy_source_corrupt") else "未跑")
        g = p["geo"]
        act = p.get("activity", {})
        parts = []
        if "dropped_frac" in act:
            parts.append(f"挡掉因果对 {act['dropped_frac']:.4f}")
        if isinstance(act.get("quant"), dict) and "rel_err" in act["quant"]:
            parts.append(f"量化 rel_err={act['quant']['rel_err']:.5f}"
                         f"/改值 {act['quant'].get('changed_frac', float('nan')):.3f}")
        else:
            parts.append("无量化" if "dropped_frac" in act else "无量化/无掩码")
        a_txt = "；".join(parts)
        L.append(f"| `{p['arm']}` | {'组合' if p['arm'].count('_') else '单因子'} | "
                 f"{g['value_bytes_per_position']:.0f} | "
                 f"{g['deployment_bytes_per_position']:.0f} | ×{g['ratio_vs_fp16']:.2f} | "
                 f"{g['level_2642'] if g['level_2642'] else '—'} | {p['bpc']:.6f} | "
                 f"{p['damage_bpc_vs_full']:+.6f} | "
                 f"{p['recall_copy']['accuracy']:.4f}"
                 f"{' (地板)' if p['recall_copy']['at_floor'] else ''} | {ci} | "
                 f"{'是' if p['recall_copy']['at_floor'] else '否'} | {cr} | {a_txt} |")
    L.append("> 判据：`damage_bpc_vs_full` = bpc − full 臂（本机同口径）；两腿并报是为了暴露"
             "「低损伤 ≠ 保住了」。`at_floor` 只读作「该任务上不可分辨」，不写成「机制无效」。")

    L.append("## 表 2 · 保留预算曲线（damage vs full）\n\n"
             "| kept | 部署字节 | damage_bpc | 相对 M2/2615 历史值（他设备） |\n|---|---|---|---|")
    prev = {r[0]: r for r in PRIOR_XCHECK}
    for d in res.get("retention_curve", {}).get("points", []):
        hist = prev.get(f"win{d['kept']}")
        L.append(f"| {d['kept']} | {d.get('deployment_bytes_per_position', float('nan')):.0f} | "
                 f"{d['damage_bpc']:+.6f} | "
                 f"{('—' if not hist else f'{hist[2]:.6f}（{hist[1]}）')} |")
    mc = res.get("retention_curve", {})
    L.append(f"> 单调（保留越多损伤越小）= **{mc.get('monotone_nonincreasing')}**；"
             f"违例 {mc.get('violations')}；端到端落差 = {mc.get('range_bpc')} bpc。")

    L.append("## 表 3 · 配对 Δ 的 5 窗口带（标准口径：配对 Δ、≥5 窗口）\n\n"
             "| 配对 | 逐窗口 Δ（seeds） | 区间 | 均值 | 极差 | 用哪个带 |\n|---|---|---|---|---|---|")
    for b in res.get("ranking_band", {}).get("paired", []):
        L.append(f"| {b['pair']} | {b['deltas_by_seed']} | {b['interval']} | {b['mean']} | "
                 f"{b.get('spread')} | "
                 f"{'配对带（≥5 窗口达标）' if b.get('standard_met') else '单臂带（退路）'} |")
    rb = res.get("ranking_band", {})
    if rb.get("single_arm"):
        L.append(f"> 单臂跨窗口 bpc 极差（退路，仅在两带不混用时参考）= "
                 f"{ {k: v['range'] for k, v in rb['single_arm']['per_arm'].items()} }；"
                 f"在力带 = {rb['single_arm']['in_force_band_bpc']}")

    L.append("## 表 4 · 同字节水平（部署口径）对照 —— 组合是否优于单用其一\n\n"
             "| 部署字节 | 本实验同字节的臂 | 他设备同字节的臂（`2632`，只作对照） |\n|---|---|---|")
    bylevel = {}
    for p in pts:
        lv = p["geo"]["level_2642"]
        bylevel.setdefault(lv, []).append(p["arm"])
    for lv in sorted(k for k in bylevel if k is not None):
        others = [f"{a}（{s}）" for a, s, _, b in PRIOR_XCHECK if abs(b - lv) < 1e-6]
        L.append(f"| {lv:.0f} | {', '.join('`%s`' % a for a in bylevel[lv])} | "
                 f"{', '.join(others) if others else '—'} |")
    L.append("## 表 5 · 与本机复现的历史锚/兄弟线点对照（他设备 ⇒ 只作等价证据）\n\n"
             "| 臂 | 本机 CPU bpc | 历史 bpc | 来源 | |Δ| |\n|---|---|---|---|---|")
    for p in pts:
        for arm, src, bpc, _b in PRIOR_XCHECK:
            if arm == p["arm"]:
                L.append(f"| `{p['arm']}` | {p['bpc']:.6f} | {bpc:.6f} | {src} | "
                         f"{abs(p['bpc'] - bpc):.2e} |")
                break
    return "\n\n".join(L) + "\n"


def retention_curve(res: dict) -> dict:
    """保留预算曲线：逐点（保留位置数、部署字节、damage vs full）+ 单调性判读。"""
    by = {p["arm"]: p for p in res["points"]}
    pts = []
    for arm in wa.ARM_ORDER:
        spec = wa.SPECS[arm]
        if spec["kind"] != "window" or arm not in by:
            continue
        pts.append({"kept": spec["d"], "damage_bpc": by[arm]["damage_bpc_vs_full"],
                    "deployment_bytes_per_position":
                        by[arm]["geo"]["deployment_bytes_per_position"]})
    return {**wa.monotone_report(pts),
            "points": pts}   # 逐点带部署字节；monotone_report 只回 kept/damage


def save(res: dict, out_dir: str) -> None:
    res["arms_completed"] = [a for a in wa.ARM_ORDER if any(p["arm"] == a for p in res["points"])]
    res["arms_pending"] = [a for a in wa.ARM_ORDER if a not in res["arms_completed"]]
    res["partial"] = bool(res["arms_pending"])
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(render_table(res))


# ----------------------------------------------------------------- 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stage", default="main", choices=["main", "extra", "all"])
    ap.add_argument("--arms", default=None, help="逗号分隔，覆盖 --stage 的臂集合")
    ap.add_argument("--legs", default="bpc,copy,corrupt",
                    help="逗号分隔：bpc / copy / corrupt（corrupt 只对 wa.CORRUPT_ARMS 生效）")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--deadline-seconds", type=float, default=780.0)
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.config))
    dev = "cpu"
    T = cfg["model"]["seq_len"]
    if args.smoke:
        cfg["eval"]["n_batches"] = 2
        cfg["tasks"]["copy"]["n_batches"] = 1
        cfg["tasks"]["induction"]["n_batches"] = 1
        cfg["tasks"]["var_recall"]["n_batches"] = 1

    legs = set(args.legs.split(","))
    if args.arms:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    elif args.smoke:
        arms = ["full", "win32_int8"]
    elif args.stage == "main":
        arms = wa.ARM_ORDER[:wa.ARM_ORDER.index(wa.BAND_AFTER) + 1]
    else:
        arms = wa.ARM_ORDER[wa.ARM_ORDER.index(wa.BAND_AFTER) + 1:]
    do_band = args.band or (args.stage == "main" and not args.smoke)

    res_path = os.path.join(args.out_dir, "results.json")
    if os.path.exists(res_path) and not args.smoke:
        res = json.load(open(res_path))
        res["config"] = cfg
    else:
        res = {"experiment": cfg["experiment"], "contract": cfg["contract"],
               "scope": cfg["scope"], "config": cfg, "criterion": cfg["criterion"],
               "proposition": cfg["proposition"],
               "device": {**cfg["device"], "torch_threads": torch.get_num_threads(),
                          "torch_version": torch.__version__},
               "same_byte_levels_2642": cfg["same_byte_levels_2642"],
               "quant_caliber": cfg["quant_caliber"],
               "legs_run": sorted(legs), "corrupt_arms": list(wa.CORRUPT_ARMS),
               "points": [], "smoke": bool(args.smoke)}
    res["legs_run"] = sorted(legs)
    t0 = time.time()

    res["cost"] = {"device": dev, "wall_seconds": 0.0, "per_arm_seconds": {}}
    _, eval_arr, _ = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                      os.path.join(ROOT, cfg["data"]["path"]))
    trunk, trunk_ck = b21.b19.load_trunk(cfg, dev)
    trunk.eval()
    res["checkpoint"] = trunk_ck
    tasks = b21.b19.build_tasks(cfg, smoke=args.smoke)
    print(f"[模型] trunk sha256 verified={trunk_ck['sha256_verified']}，data ok", flush=True)

    # ---------- ① 仪器自检（各一次） ----------
    if "instrument" not in res:
        fp32_fn = b21.b19.make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
        nb_chk = min(2, cfg["eval"]["n_batches"])
        my = b21.b19.bpc_of(fp32_fn, eval_arr, T, nb_chk, cfg["eval"]["batch_size"],
                            cfg["eval"]["window_seed"], dev)
        lib = b21.b19.masked_bpc(trunk, eval_arr, None, n_batches=nb_chk,
                                 batch_size=cfg["eval"]["batch_size"],
                                 window_seed=cfg["eval"]["window_seed"], device=dev)
        small = b21.b19.concat([tasks["copy_by_pos"][cfg["tasks"]["copy"]["src_positions"][0]]
                                ])
        acc_mine = b21.b19.synth_accuracy(fp32_fn, small, dev)
        acc_lib = lib_accuracy(trunk, small, device=dev)
        sc_red = ra.reduction_self_check(trunk, tasks["var_recall"].tokens, T, dev)
        eq_tokens = tasks["copy"].tokens[:4].to(dev)
        spec_i8 = wa.SPECS["win32_int8"]
        f_per = ba.make_logits_fn(trunk, spec_i8, T, dev, "per_value", 32, 32)
        f_prod = b21.make_logits_fn(trunk, spec_i8, T, dev)
        res["instrument"] = {
            "bpc_path": {"bpc_of": round(my, 6), "masked_bpc": round(lib, 6),
                         "abs_diff": abs(my - lib), "bitwise_equal": my == lib,
                         "n_batches_chk": nb_chk,
                         "note": "同一算法两条独立路径在同窗口序上逐位一致（算法恒等检查）"},
            "synth_path": {"synth_accuracy": acc_mine["accuracy"],
                           "longrange_tasks.accuracy": acc_lib["accuracy"],
                           "n_hits_equal": acc_mine["n_hits"] == acc_lib["n_hits"],
                           "by_distance_equal": all(
                               acc_mine["by_distance"][k]["accuracy"]
                               == acc_lib["by_distance"][k]["accuracy"]
                               for k in acc_lib["by_distance"])},
            "mask_quant_reduction_self_check": sc_red,
            "block_arms_per_value_vs_rest_arms_bitwise": bool(
                torch.equal(f_per(eq_tokens), f_prod(eq_tokens))),
            "note": "rest_arms 的两条退化路径 + block_arms 的 per_value 与生产前向逐位一致 ⇒ "
                    "本实验量化腿的唯一改动就是标尺方案（block G=32）。",
        }
        print(f"[自检] bpc_of={my:.6f} vs masked_bpc={lib:.6f} 逐位="
              f"{res['instrument']['bpc_path']['bitwise_equal']}；"
              f"block_arms(per_value)==rest_arms 逐位="
              f"{res['instrument']['block_arms_per_value_vs_rest_arms_bitwise']}", flush=True)
    copy_positions = cfg["tasks"]["copy"]["src_positions"]
    copy_len = cfg["tasks"]["copy"]["copy_len"]
    tasks["copy_corrupt_by_pos"] = {
        p: corrupt_span(tasks["copy_by_pos"][p], p, p + copy_len)
        for p in tasks["copy_by_pos"]}
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)

    # ---------- ② 对照 ----------
    if "controls" not in res:
        res["controls"] = {
            "chance_uniform": round(1.0 / cfg["tokens"]["sym_vocab"], 8),
            "local_prior_copy": local_prior_accuracy(tasks["copy"], k=cfg["tokens"]["local_k"]),
            "local_prior_induction": local_prior_accuracy(
                tasks["induction"], k=cfg["tokens"]["local_k"]),
            "local_prior_var_recall": local_prior_accuracy(
                tasks["var_recall"], k=cfg["tokens"]["local_k"]),
            "local_prior_resistance": {
                "copy": check_local_prior_resistance(tasks["copy"], k=cfg["tokens"]["local_k"]),
                "induction": check_local_prior_resistance(
                    tasks["induction"], k=cfg["tokens"]["local_k"]),
                "var_recall": check_local_prior_resistance(
                    tasks["var_recall"], k=cfg["tokens"]["local_k"])},
            "anchor_bpc": 1.483569,
            "anchor_bpc_source": "2614/results.json::instrument.bpc_full_release_path（GPU）",
            "corrupt_source_note": "copy 主腿源片段整段换随机符号（其余逐位不变），逐臂报该对照。",
            "task_pool_note": "induction / var_recall 只用于局部先验对照（本实验不跑这两条召回腿）。",
        }
        print(f"[对照] copy 局部先验={res['controls']['local_prior_copy']['accuracy']:.4f} / "
              f"chance={res['controls']['chance_uniform']:.4f}", flush=True)

    # ---------- ③ 逐臂（每臂落盘） ----------
    done = {p["arm"] for p in res["points"]}
    for arm in arms:
        if arm in done:
            print(f"[跳过] {arm} 已在 results.json", flush=True)
            continue
        if time.time() - t0 > args.deadline_seconds:
            print(f"[截止] 已用 {time.time() - t0:.0f}s > {args.deadline_seconds}s，"
                  f"停在新臂 {arm} 之前（如实记 pending）", flush=True)
            break
        spec = wa.SPECS[arm]
        t_arm = time.time()
        fn = make_fn(arm, trunk, spec, T, dev)
        point_geo = b21.bytes_per_position(spec, cfg)
        scheme_geo = (sc.deployment_bytes({**point_geo}, spec, cfg, "block", 32, 32)
                      if wa.is_block_quant(arm) else None)
        p = {"arm": arm, "spec": spec, "family": "trunk",
             "class": "combo_mask" if spec["kind"] == "combo_mask" else spec["kind"],
             "source": f"本实验自跑（CPU, seed={cfg['eval']['window_seed']}, "
                       f"{cfg['eval']['n_batches']}×{cfg['eval']['batch_size']}, "
                       f"scheme={'block G=32' if wa.is_block_quant(arm) else 'none'}）",
             "reused": False, "bpc": None}
        if "bpc" in legs:
            p["bpc"] = round(b21.b19.bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"],
                                            cfg["eval"]["batch_size"],
                                            cfg["eval"]["window_seed"], dev), 6)
        if "copy" in legs:
            cp = b21.b19.pooled_accuracy(fn, tasks["copy_by_pos"], dev)
            p["recall_copy"] = {**cp,
                                "at_floor": bool(cp["accuracy"] <= b21.b19.floor_threshold(
                                    cp["n"], res["controls"]["chance_uniform"]))}
        if "corrupt" in legs and arm in wa.CORRUPT_ARMS:
            p["recall_copy_source_corrupt"] = b21.b19.pooled_accuracy(
                fn, tasks["copy_corrupt_by_pos"], dev)
        p["prefix_invariance"] = cc.prefix_invariance_of_logits(fn, tasks["copy"].tokens[:4].to(dev))
        p["activity"] = activity(arm, trunk, spec, T, dev, probe_tokens)
        p["geo"] = geo_of(arm, spec, cfg, point_geo, scheme_geo)
        p["seconds"] = round(time.time() - t_arm, 2)
        res["points"] = [q for q in res["points"] if q["arm"] != arm] + [p]
        res["points"].sort(key=lambda q: wa.ARM_ORDER.index(q["arm"]))
        full = next((q for q in res["points"] if q["arm"] == "full"), None)
        if full and full.get("bpc") and p.get("bpc") is not None:
            p["damage_bpc_vs_full"] = round(p["bpc"] - full["bpc"], 9)
            p["damage_frac_vs_full"] = round(p["bpc"] / full["bpc"] - 1, 6)
        for q in res["points"]:
            if full and full.get("bpc") and q.get("bpc") is not None:
                q["damage_bpc_vs_full"] = round(q["bpc"] - full["bpc"], 9)
                q["damage_frac_vs_full"] = round(q["bpc"] / full["bpc"] - 1, 6)
        res["retention_curve"] = retention_curve(res)
        res["cost"] = {"device": dev, "wall_seconds": round(time.time() - t0, 2),
                       "per_arm_seconds": {q["arm"]: q["seconds"] for q in res["points"]}}
        save(res, args.out_dir)
        print(f"[{len(res['points'])}] {arm:<12} bpc={p.get('bpc')} "
              f"copy={p.get('recall_copy', {}).get('accuracy')} 部署字节="
              f"{p['geo']['deployment_bytes_per_position']:.0f} "
              f"level={p['geo']['level_2642']} {p['seconds']:.0f}s", flush=True)

    # ---------- ④ 5 窗口配对带 ----------
    if do_band and not args.smoke:
        seeds = [cfg["eval"]["window_seed"]] + list(cfg["eval"]["band_extra_seeds"])
        band_arms = [a for a in cfg["eval"]["band_arms"]]
        bb = res.setdefault("band", {"n_batches": cfg["eval"]["band_n_batches"],
                                     "bpc_by_seed": {a: {} for a in band_arms}})
        for a in band_arms:
            bb["bpc_by_seed"].setdefault(a, {})
        for seed in seeds:
            for arm in band_arms:
                if str(seed) in bb["bpc_by_seed"][arm]:
                    continue
                if time.time() - t0 > args.deadline_seconds:
                    print(f"[截止] 配对带到 seed={seed} arm={arm} 停（如实记 pending）", flush=True)
                    break
                fn = make_fn(arm, trunk, wa.SPECS[arm], T, dev)
                v = round(b21.b19.bpc_of(fn, eval_arr, T, cfg["eval"]["band_n_batches"],
                                         cfg["eval"]["batch_size"], seed, dev), 9)
                bb["bpc_by_seed"][arm][str(seed)] = v
                res["cost"]["wall_seconds"] = round(time.time() - t0, 2)
                save(res, args.out_dir)
                print(f"[带] seed={seed} {arm:<12} bpc={v:.6f}", flush=True)
        pairs = [("full", "win32"), ("win32", "win32_int8"), ("full", "win32_int8")]
        paired = [wa.paired_band(bb["bpc_by_seed"], a, b) for a, b in pairs
                  if bb["bpc_by_seed"].get(a) and bb["bpc_by_seed"].get(b)]
        single = wa.single_arm_band(bb["bpc_by_seed"])
        std_met = all(b["standard_met"] for b in paired) if paired else False
        res["ranking_band"] = {
            "paired": paired, "single_arm": single,
            "standard_caliber": "配对 Δ（同一臂对在 ≥5 个评测窗口重测）的区间",
            "standard_met": std_met,
            "in_force": ("配对带（区间半宽即带；≥5 窗口达标）" if std_met else
                         f"单臂跨窗口极差 {single['in_force_band_bpc']}（配对口径未达标，退路）"),
            "seeds": seeds, "declaration":
                f"配对 Δ 在 {len(seeds)} 个评测窗口上（seeds={seeds}，n_batches="
                f"{cfg['eval']['band_n_batches']}）⇒ "
                f"{'达标（≥5）' if std_met else '**未达标（<5）**'}；两带不混用。"}
        save(res, args.out_dir)
        print(f"[带] 配对 Δ = {[(b['pair'], b['interval'], b['mean']) for b in paired]}；"
              f"达标={std_met}", flush=True)

    # ---------- ⑤ 收尾落盘 ----------
    res["cost"] = {"device": dev, "wall_seconds": round(time.time() - t0, 2),
                   "wall_minutes": round((time.time() - t0) / 60, 3),
                   "per_arm_seconds": {q["arm"]: q["seconds"] for q in res["points"]},
                   "torch_threads": torch.get_num_threads(), "smoke": bool(args.smoke),
                   "gpu_lock_held": False, "gpu_lock_wait_seconds": None}
    res["code"] = {name: sha256_file(os.path.join(ROOT, name))[:16] for name in (
        "tools/evalkit/kv_quant.py", "tools/evalkit/retention.py",
        "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/rest_arms.py",
        "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/block_arms.py",
        "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/scale_caliber.py",
        "axes/5-quant/5.6-window-quant-retrieval/experiments/"
        "2654-window-retention-frontier-13m/win_arms.py",
        "axes/5-quant/5.6-window-quant-retrieval/experiments/"
        "2654-window-retention-frontier-13m/run.py")}
    res["prior_xcheck"] = [{"arm": a, "source": s, "bpc": b, "deployment_bytes": d}
                           for a, s, b, d in PRIOR_XCHECK]
    res["m2_errata"] = M2_ERRATA
    save(res, args.out_dir)
    print(f"[完成] {res['cost']['wall_minutes']} min，{len(res['points'])} 臂 → {args.out_dir}；"
          f"pending={res['arms_pending']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
