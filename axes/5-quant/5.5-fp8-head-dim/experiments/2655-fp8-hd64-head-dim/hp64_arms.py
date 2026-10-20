#!/usr/bin/env python3
"""2655 · `head_dim=64` 的 KV 标尺摊薄 —— 评测入口（WP2 的实测腿）。

契约：同目录 `README.md`（**运行前**预登记）+ `config.json::criterion`（判据线**运行前**写入）。

本文件**不实现任何量化算子**，只做三件事（`harness/SCHEMA.md` §22：新逻辑尽量进新模块、不改共享底座）：

1. 组装 2655 的配置与臂表（`config.json` + 只读复用 `2621` 的 `data`），装载 `head_dim=64` 的
   checkpoint（sha256 在评测时现算并落盘，因为预登记时该权重还不存在）；
2. 跑仪器自检（两条 bpc 路径逐位相等 / fp32 前向与生产前向逐位相等 / int8 块标尺 G=64 与
   `2632` 前向逐位相等 / 前缀不变性），再把 `2653/run.py::eval_arm` 的评测电池逐臂跑一遍；
3. **WP2 的对账量**：把每臂**实测**的标尺密度（`n_scales/n_values` ⇒ 每值摊到的标尺字节）与
   **解析**价签（`1 + 2/G` B/值 ⇒ 每位置字节）逐项对齐，并对 `head_dim × G` 阶梯给解析表。

只读复用（不复制改写）：`2653/fp8_arms.py`（fp8 往返 / int8 块标尺走 `tools/evalkit/kv_quant.py::quantize_kv` /
解析字节账）、`2653/run.py`（评测电池 `eval_arm` / `paired_deltas` / `build_tasks`）、
`2621/run.py`（内含 `2619`：`bpc_of` / `masked_bpc` / `pooled_accuracy`）、`2631/causal_check.py`、
`2632/block_arms.py`（int8 块标尺的实现层 equality 对照）。

用途（Phase 2 = finisher 一条命令接手）：

    cd /home/z/vibe/pathlm && python3 <expdir>/hp64_arms.py                # 正式评测（主表，hd64-s0）
    cd /home/z/vibe/pathlm && python3 <expdir>/hp64_arms.py --seed-leg second_seed --out-dir <expdir>/hd64-s1  # n≥2 的第二 seed 腿
    cd /home/z/vibe/pathlm && python3 <expdir>/hp64_arms.py --ledger-only   # 只打印解析字节账（秒级、不装载模型）
    cd /home/z/vibe/pathlm && python3 <expdir>/hp64_arms.py --smoke --checkpoint-path <pt> --checkpoint-sha256 <hex>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
P2653 = os.path.join(ROOT, "axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m")
sys.path.insert(0, ROOT)

from tools.evalkit.kv_quant import SCALE_BYTES                                   # noqa: E402
from pathlm.data import load_enwik8_full                                         # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel                          # noqa: E402


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 2653 的入口（内含 `fp8_arms` / `2621` 的 `b21` / `2631` 的因果仪器 / `2632` 的 equality 对照）；只读。
r2653 = _load_module(os.path.join(P2653, "run.py"), "fp8_vs_int8_2653")
fa = r2653.fa                  # 语义层：fp8 往返 + 解析字节账
b21 = r2653.b21                # 2621 的评测管线（内含 2619 的 b19）
cc2631 = r2653.cc2631          # 前缀不变性两条
ba2632 = r2653.ba2632          # int8 块标尺的实现层对照

MAIN_ARM_ORDER = ("fp16", "int8_block_g64", "fp8_e4m3")
SIDE_ARM_ORDER = ("int8_block_g16", "int8_block_g32")
#: 辅助臂只跑 `scan` 口径（6×16, seed=2），只用于 G 阶梯内部的解析对账，不进配对表。
SIDE_MEASURE = {"int8_block_g16": {"main": False, "pairing": False, "scan": True, "recall": False},
                "int8_block_g32": {"main": False, "pairing": False, "scan": True, "recall": False}}
#: GPU 历史锚：**不同 checkpoint、不同设备** ⇒ 只并列报告，不作判据（见 config.json::eval）。
GPU_ANCHOR = {"clean_bpc": 1.483569, "device": "cuda", "source": "2614/2610 口径",
              "comparable": False,
              "reason": "本实验的被测模型是 head_dim=64 的另训 checkpoint；跨设备、跨 checkpoint 都不构成判据"}
#: 2653 的 CPU fp16 读数（同设备但**不同 checkpoint**）：并列报告用。
CPU_REF_2653 = {"arm": "2653::fp16", "clean_bpc": 1.4835687, "device": "cpu",
                "checkpoint": "2610 trunk (head_dim=32)", "comparable": False,
                "reason": "同设备、同窗口序，但被测 checkpoint 不同（head_dim=32 的 trunk）"}


# ----------------------------------------------------------------- 配置与权重


def build_cfg(config_path: str, ckpt_path: str | None, ckpt_sha: str | None,
              seed_leg: str | None = None) -> dict:
    """`config.json` + 只读复用 `2621` 的 `data`；`head_dim` 由 `d_model/n_heads` **求出**并核对声明值。

    `seed_leg` 选 `config.json::checkpoints` 里预登记的另一条 seed 腿（如 `second_seed` = `runs/hd64-s1`）：
    臂表 / 口径 / 判据线逐字不变，只换被测权重 ⇒ 该腿是**读数腿**，不是 plumbing。
    """
    with open(config_path) as f:
        cfg = json.load(f)
    with open(os.path.join(ROOT, cfg["harness"]["parent_config"])) as f:
        cfg["data"] = json.load(f)["data"]
    derived = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    if cfg.get("head_dim") != derived:
        raise SystemExit(f"config.head_dim={cfg.get('head_dim')} 与 d_model/n_heads={derived} 不符")
    if seed_leg is not None:
        if ckpt_path is not None or ckpt_sha is not None:
            raise SystemExit("--seed-leg 与 --checkpoint-path 互斥：前者走预登记的 seed 腿，后者是冒烟覆盖")
        legs = cfg.get("checkpoints") or {}
        if seed_leg not in legs or seed_leg == "trunk":
            raise SystemExit(f"config.json::checkpoints 无 {seed_leg!r}；已知腿 = "
                             f"{[k for k in legs if k != 'trunk']}")
        cfg["checkpoints"]["trunk"] = dict(legs[seed_leg])
        cfg["seed_leg"] = seed_leg
    elif ckpt_path is not None or ckpt_sha is not None:
        if not (ckpt_path and ckpt_sha):
            raise SystemExit("--checkpoint-path 与 --checkpoint-sha256 必须同时给出")
        cfg["checkpoints"]["trunk"] = {"path": os.path.relpath(os.path.abspath(ckpt_path), ROOT),
                                       "sha256": ckpt_sha,
                                       "role": "命令行覆盖（冒烟/仪器自检用，非本实验的被测权重）"}
    return cfg


def load_ckpt(cfg: dict, dev: str):
    """按 `checkpoints.trunk` 装载；`sha256 = null`（预登记时期望值缺失）时现算并落盘。"""
    spec = cfg["checkpoints"]["trunk"]
    ckpt = os.path.join(ROOT, spec["path"])
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint 不存在：{ckpt}\n⇒ 训练腿未完成；先跑 hp64_train_driver.py（见 README.md §9）")
    got = b21.b19.sha256_file(ckpt)
    expect = spec.get("sha256")
    if expect is not None and got != expect:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {expect}\n  got  {got}")
    model = DepthARModel(DepthARConfig(**cfg["model"]), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(ckpt, ROOT), "sha256": got, "sha256_expected": expect,
                   "size_bytes": os.path.getsize(ckpt),
                   "sha256_source": "pinned_in_config" if expect else "computed_at_eval",
                   "sha256_verified": (got == expect) if expect else None,
                   "model_cfg": dict(cfg["model"]), "head_dim": cfg["head_dim"]}


# ----------------------------------------------------------------- WP2 对账量


def wp2_reconciliation(cfg: dict, ledger: dict, points: list[dict]) -> dict:
    """把每臂**实测**的标尺密度与**解析**价签对齐 —— 这就是 WP2 的验收量。

    解析：每值字节 = value_bytes + SCALE_BYTES/G（块标尺）；实测：`n_scales/n_values` 是探针批上
    每值摊到的标尺**把数**（块标尺 ⇒ 1/G），乘 `SCALE_BYTES` 得每值摊到的标尺字节。两者一致 ⇒
    「每位置字节随 head_dim/G 的实测与解析一致」成立。
    """
    vpp = ledger["values_per_position"]
    per = {}
    for p in points:
        act = p["activity"]
        value_bytes = {"fp16": 2.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0}.get(p["mode"], 1.0)
        nv, ns = int(act.get("n_values", 0)), int(act.get("n_scales", 0))
        density = (ns / nv) if nv else None
        meas_bpe = (value_bytes + SCALE_BYTES * density) if density is not None else None
        meas_bpp = (vpp * meas_bpe) if meas_bpe is not None else None
        analytic_bpp = p["deployment"]["bytes_per_position"]
        per[p["arm"]] = {
            "mode": p["mode"], "group": p["group"], "scale_mode": p["scale_mode"],
            "value_bytes_per_value": value_bytes,
            "measured_scales_per_value": (round(density, 9) if density is not None else None),
            "measured_scale_bytes_per_value": (round(SCALE_BYTES * density, 9) if density is not None else None),
            "measured_bytes_per_value": (round(meas_bpe, 9) if meas_bpe is not None else None),
            "measured_bytes_per_position": (round(meas_bpp, 3) if meas_bpp is not None else None),
            "analytic_bytes_per_value": p["deployment"]["bytes_per_value"],
            "analytic_bytes_per_position": analytic_bpp,
            "delta_bytes_per_position": (round(meas_bpp - analytic_bpp, 9) if meas_bpp is not None else None),
            "consistent": (abs(meas_bpp - analytic_bpp) < 1e-9) if meas_bpp is not None else None,
            "prefix_invariant": p["prefix_invariance"]["future_token_past_logits_equal"]
                                and p["prefix_invariance"]["peer_request_logits_equal"],
        }
    return {"values_per_position": vpp, "scale_bytes_per_scale": SCALE_BYTES,
            "caliber": ("解析 = 1 + SCALE_BYTES/G B/值（fp8 免标尺 ⇒ 1.0）；实测 = 探针批上 n_scales/n_values "
                        "× SCALE_BYTES + 值字节；每位置字节 = 每值字节 × 6144"),
            "per_arm": per,
            "ladder_analytic": ledger["head_dim_ladder"],
            "caps": ledger["head_dim_cap"],
            "all_consistent": (all(v["consistent"] for v in per.values()
                                   if v["consistent"] is not None) if per else None)}


# ----------------------------------------------------------------- 仪器自检


def instrument(cfg: dict, ledger: dict, model, tasks: dict, eval_arr, dev: str) -> dict:
    """跑 `2653` 的三道 equality + 两条前缀不变性（instrument 红 ⇒ 不得出数）。"""
    T = cfg["model"]["seq_len"]
    bs = cfg["eval"]["batch_size"]
    seed = cfg["eval"]["window_seed"]
    eq_tokens = tasks["var_recall"].tokens[:4].to(dev)
    fp32_fn = fa.make_logits_fn(model, {"mode": "fp32"}, T, dev)
    my_bpc = b21.b19.bpc_of(fp32_fn, eval_arr, T, 2, bs, seed, dev)
    lib_bpc = b21.b19.masked_bpc(model, eval_arr, None, n_batches=2, batch_size=bs,
                                 window_seed=seed, device=dev)
    f_b19 = b21.b19.make_logits_fn(model, {"kind": "quant", "mode": "fp32"}, T, dev)
    f_2632 = ba2632.masked_kv_quant_logits(model, eq_tokens, None, "int8", "block", 64)
    f_mine = fa.kv_logits(model, eq_tokens, "int8", "block", 64)
    return {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc), "bitwise_equal": my_bpc == lib_bpc,
                     "n_batches": 2, "window_seed": seed,
                     "note": "fp32 恒等臂两条独立路径（bpc_of / retention.masked_bpc）"},
        "fp32_forward_eq_2619": bool(torch.equal(fp32_fn(eq_tokens), f_b19(eq_tokens))),
        "int8_block_eq_2632": bool(torch.equal(f_mine[0], f_2632[0])),
        "int8_block_group_checked": 64,
        "prefix_invariance_fp32": cc2631.prefix_invariance_of_logits(fp32_fn, eq_tokens),
        "anchor_gpu": GPU_ANCHOR, "reference_cpu_2653": CPU_REF_2653,
        "note": ("int8 块标尺 G=64 与 `2632/block_arms` 前向逐位相等 ⇒ 本实验的标尺算子与 `5.1`/`2632` 同源；"
                 "fp8 臂只是把同一个往返算子换成 fp8（`2653/fp8_arms.py`）。"),
        "deployment_anchors": {k: ledger["anchors_5_1"][k] for k in ledger["anchors_5_1"]},
    }


# ----------------------------------------------------------------- 表格与落盘


def _pct(x) -> str:
    return f"{x:.3%}" if isinstance(x, (int, float)) else "—"


def _num(x, fmt: str = ".6f") -> str:
    """**缺失值声明为缺失**：None / 未跑一概打「—」，不填零、不外推（`AGENTS.md` absent）。"""
    return format(x, fmt) if isinstance(x, (int, float)) else "—"


def render_table(cfg: dict, rec: dict) -> str:
    led = rec["deployment_ledger"]
    pa = rec.get("paired_delta", {}).get("per_arm", {})
    L = ["# 2655 · `head_dim=64` 的 KV 标尺摊薄（WP2 实测腿）",
         f"**设备口径 = CPU**（`device: cpu`；全部臂同一设备；不取 GPU 锁）。"
         f"横轴 = 每位置**部署**字节 = `2·L·n_kv_heads·head_dim·(每值字节)`，"
         f"每位置值数 = {led['values_per_position']:.0f}（K+V），fp16 基线 = "
         f"{led['fp16_bytes_per_position']:.0f} B/位置。"
         f"主判带 = 跨格 **0.014 bpc**（限配对 Δ、≥5 窗口，本 run 有 "
         f"{len(cfg['eval']['pairing_seeds'])} 个窗口）。"]
    L.append("## 表 1 · 主臂（`head_dim=64`）\n\n"
             "| 臂 | 每值字节 | 每位置字节 | ×fp16 | 省幅 | clean bpc(CPU,24×16,s2) | 配对 Δbpc vs fp16(5 窗) max\\|Δ\\| | 越 0.014 带 | "
             "copy 召回 | 95% CI | at_floor | 前缀不变(未来/邻居) |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for p in rec.get("points", []):
        d = pa.get(p["arm"], {})
        rc = p.get("recall_copy")
        max_abs = _num(d.get("max_abs"))
        verdict = ("越带" if d["exceeds_band_0_014"] else "不越带") if d else "—"
        ci = f"[{rc['wilson95'][0]:.4f}, {rc['wilson95'][1]:.4f}]" if rc else "—"
        dep = p["deployment"]
        rows.append(f"| `{p['arm']}` | {dep['bytes_per_value']:.5f} | "
                    f"{dep['bytes_per_position']:.0f} | ×{dep['ratio_vs_fp16']:.4f} | "
                    f"{_pct(dep['reduction_vs_fp16'])} | {_num(p['bpc'])} | "
                    f"{max_abs} | {verdict} | {_num(rc and rc['accuracy'], '.4f')} | {ci} | "
                    f"{('是' if rc['at_floor'] else '否') if rc else '—'} | "
                    f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
                    f"{p['prefix_invariance']['peer_request_logits_equal']} |")
    L[-1] += "\n" + "\n".join(rows) if rows else "\n（尚无完成臂）"
    L.append("## 表 2 · 辅助 G 扫描臂（scan 口径 6×16，seed 2；只作解析对账）\n\n"
             "| 臂 | G | 每值字节 | 每位置字节 | clean bpc(scan) | 实测 n_scales/n_values | 实测每值字节 | 与解析一致 | 前缀不变 |\n"
             "|---|---|---|---|---|---|---|---|---|")
    recon = rec.get("wp2_reconciliation", {}).get("per_arm", {})
    rows = []
    for p in rec.get("side_points", []):
        r = recon.get(p["arm"], {})
        dep = p["deployment"]
        rows.append(f"| `{p['arm']}` | {p['group']} | {dep['bytes_per_value']:.5f} | "
                    f"{dep['bytes_per_position']:.0f} | {_num(p['bpc_scan'])} | "
                    f"{_num(r.get('measured_scales_per_value'))} | "
                    f"{_num(r.get('measured_bytes_per_value'))} | "
                    f"{('一致' if r.get('consistent') else '不一致')} | "
                    f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
                    f"{p['prefix_invariance']['peer_request_logits_equal']} |")
    L[-1] += "\n" + "\n".join(rows) if rows else "\n（尚无完成辅助臂）"
    L.append("## 表 3 · WP2 解析阶梯（`head_dim` × G ⇒ 每值字节 / 每位置字节 / 省幅）\n\n"
             "| head_dim | G | 合法（G 整除 head_dim） | 每值字节 `1+2/G` | 每位置字节 | ×fp16 | 省幅 | 前缀不变 |\n"
             "|---|---|---|---|---|---|---|---|")
    for r in led["head_dim_ladder"]:
        L[-1] += (f"\n| {r['head_dim']} | {r['group']} | {'是' if r['legal'] else '否（' + r['reason'] + '）'} | "
                  f"{r['bytes_per_value']:.5f} | {r['bytes_per_position']:.0f} | "
                  f"×{r['ratio_vs_fp16']:.4f} | {_pct(r['reduction_vs_fp16'])} | "
                  f"{'True' if r['prefix_invariant'] else 'False'} |")
    cap = led["head_dim_cap"]
    L.append(f"> **封顶**：`head_dim=32` ⇒ G 上限 32 ⇒ 每值 {cap['32']['bytes_per_value']:.4f} B、"
             f"每位置 {cap['32']['bytes_per_position']:.0f} B、省幅 **{_pct(cap['32']['reduction_vs_fp16'])}**；"
             f"`head_dim=64` ⇒ G=64 合法 ⇒ 每值 {cap['64']['bytes_per_value']:.5f} B、"
             f"每位置 {cap['64']['bytes_per_position']:.0f} B、省幅 **{_pct(cap['64']['reduction_vs_fp16'])}**"
             f"（更接近 50%）。")
    ins = rec.get("instrument")
    if ins:
        L.append("## 表 4 · 仪器自检与设备口径\n\n"
                 f"- fp32 恒等臂：`bpc_of` = {ins['bpc_path']['bpc_of']:.6f}，"
                 f"`retention.masked_bpc` = {ins['bpc_path']['masked_bpc']:.6f}，"
                 f"逐位相等 = **{ins['bpc_path']['bitwise_equal']}**。\n"
                 f"- fp32 前向与 `2619` 生产前向逐位相等 = **{ins['fp32_forward_eq_2619']}**。\n"
                 f"- int8 块标尺 G=64 与 `2632/block_arms` 前向逐位相等 = **{ins['int8_block_eq_2632']}**。\n"
                 f"- 前缀不变性（fp32 恒等臂，探针 4 请求）：{ins['prefix_invariance_fp32']}\n"
                 f"- checkpoint：`{rec['checkpoint']['path']}` sha256 = "
                 f"`{rec['checkpoint']['sha256'][:16]}…`（{rec['checkpoint']['sha256_source']}）\n"
                 f"> 跨设备/跨 checkpoint 声明：GPU 历史锚 {ins['anchor_gpu']['clean_bpc']} bpc"
                 f"（device=cuda、`head_dim=32` 的 trunk）与 2653 的 CPU fp16 读数 "
                 f"{ins['reference_cpu_2653']['clean_bpc']:.7f} 都**不是**本实验的被测权重 ⇒ "
                 f"只并列报告，不构成判据、也不声称逐位相等。")
    else:
        L.append("## 表 4 · 仪器自检与设备口径\n\n（尚未运行）")
    return "\n\n".join(L) + "\n"


def dump(out_dir: str, rec: dict, cfg: dict) -> None:
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
    ap.add_argument("--ledger-only", action="store_true")
    ap.add_argument("--arms", default=None, help="逗号分隔，只跑这些主臂（调试用）")
    ap.add_argument("--checkpoint-path", default=None,
                    help="覆盖被测权重（**只用于冒烟/仪器自检**；与 --checkpoint-sha256 同时给出）")
    ap.add_argument("--checkpoint-sha256", default=None)
    ap.add_argument("--seed-leg", default=None,
                    help="跑 `config.json::checkpoints` 里预登记的另一条 seed 腿（如 second_seed）；"
                         "臂表/口径/判据线不变，**构成读数**（不是 plumbing）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = build_cfg(args.config, args.checkpoint_path, args.checkpoint_sha256, args.seed_leg)
    dev = cfg["eval"]["device"]
    T = cfg["model"]["seq_len"]
    ledger = fa.deployment_ledger(cfg)          # 逐臂解析字节账（含 head_dim × G 阶梯与 hd32/hd64 封顶）
    if args.ledger_only:
        print(json.dumps({"values_per_position": ledger["values_per_position"],
                          "fp16_bytes_per_position": ledger["fp16_bytes_per_position"],
                          "per_arm": {k: {kk: v[kk] for kk in
                                          ("bytes_per_value", "bytes_per_position",
                                           "reduction_vs_fp16")}
                                      for k, v in ledger["per_arm"].items()},
                          "per_side_arm": {k: {kk: v[kk] for kk in
                                               ("bytes_per_value", "bytes_per_position",
                                                "reduction_vs_fp16")}
                                           for k, v in ledger["per_side_arm"].items()},
                          "head_dim_cap": ledger["head_dim_cap"]},
                         ensure_ascii=False, indent=1), flush=True)
        return 0

    if args.charts_only:
        with open(os.path.join(args.out_dir, "results.json")) as f:
            rec = json.load(f)
        with open(os.path.join(args.out_dir, "table.md"), "w") as f:
            f.write(render_table(rec["config"], rec))
        print(f"[charts-only] 重画 table.md → {args.out_dir}", flush=True)
        return 0

    arms = list(MAIN_ARM_ORDER if not args.arms else args.arms.split(","))
    if args.smoke:
        arms = ["fp16", "int8_block_g64"]
    side_arms = [] if args.smoke else list(SIDE_ARM_ORDER)
    main_n = 1 if args.smoke else cfg["eval"]["n_batches"]
    pair_n = 1 if args.smoke else cfg["eval"]["pairing_n_batches"]
    scan_n = 0 if args.smoke else cfg["eval"]["scan_n_batches"]
    seed_main = cfg["eval"]["window_seed"]

    rec = {"experiment": cfg["experiment"], "contract": cfg["contract"], "scope": cfg["scope"],
           "config": cfg, "criterion": cfg["criterion"], "device": dev,
           "delta_directions": {"delta_bytes_per_position": "arm_minus_comparator",
                                "paired_delta": "arm_minus_comparator",
                                "delta_by_window": "arm_minus_comparator"},
           "caliber_note": cfg["eval"]["cross_device_rule"], "gpu_lock": None,
           "gpu_lock_note": "纯 CPU 推理，按 SCHEMA §20 不取 GPU 锁",
           "deployment_ledger": ledger, "partial": True,
           "arms_completed": [], "arms_pending": list(arms),
           "side_arms_completed": [], "side_arms_pending": list(side_arms),
           "points": [], "side_points": [],
           "smoke": bool(args.smoke),
           "plumbing_only": bool(args.smoke or (args.checkpoint_path is not None
                                                and args.seed_leg is None)),
           "checkpoint_override": bool(args.checkpoint_path is not None),
           "checkpoint_role": ("smoke" if args.smoke else
                               args.seed_leg if args.seed_leg else
                               "plumbing_override" if args.checkpoint_path is not None else
                               "trunk")}
    dump(args.out_dir, rec, cfg)

    t0 = time.time()
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    model, ck = load_ckpt(cfg, dev)
    rec["checkpoint"] = ck
    tasks = r2653.build_tasks(cfg, args.smoke)
    print(f"[装载] checkpoint sha256={ck['sha256'][:16]}…（{ck['sha256_source']}）；"
          f"head_dim={cfg['head_dim']}；评测尾段就绪（{time.time() - t0:.1f}s）", flush=True)

    # ---------- ① 仪器自检（便宜、先跑；红则不得出数） ----------
    rec["instrument"] = instrument(cfg, ledger, model, tasks, eval_arr, dev)
    dump(args.out_dir, rec, cfg)
    ins = rec["instrument"]
    print(f"[自检] bpc_of={ins['bpc_path']['bpc_of']:.6f} vs masked_bpc="
          f"{ins['bpc_path']['masked_bpc']:.6f} 逐位={ins['bpc_path']['bitwise_equal']}；"
          f"fp32 前向 equality={ins['fp32_forward_eq_2619']}；"
          f"int8 块标尺 G=64 eq 2632={ins['int8_block_eq_2632']}", flush=True)
    assert ins["fp32_forward_eq_2619"], "本模块 fp32 前向与生产前向不逐位相等"
    assert ins["int8_block_eq_2632"], "int8 块标尺 G=64 前向与 2632 不逐位相等"
    assert ins["bpc_path"]["bitwise_equal"], "两条 bpc 路径不逐位相等"

    # ---------- ② 主臂（逐臂持久化：子 agent 有硬超时，先保 results.json） ----------
    for arm in arms:
        spec = cfg["arms"][arm]
        p = r2653.eval_arm(arm, spec, model, cfg, dev, tasks, eval_arr, ledger, main_n, pair_n,
                           scan_n, seed_main)
        p["plumbing_only"] = rec["plumbing_only"]
        rec["points"].append(p)
        rec["arms_completed"].append(arm)
        rec["arms_pending"] = [a for a in arms if a not in rec["arms_completed"]]
        rec["wp2_reconciliation"] = wp2_reconciliation(cfg, ledger, rec["points"])
        rec["paired_delta"] = r2653.paired_deltas(rec, cfg, seed_main)
        dump(args.out_dir, rec, cfg)
        r = rec["wp2_reconciliation"]["per_arm"][arm]
        print(f"[主臂] {arm:<16} bpc={p['bpc']:.6f} 每位置={p['deployment']['bytes_per_position']:.0f} "
              f"实测每位置={r['measured_bytes_per_position']} 一致={r['consistent']} "
              f"inv={r['prefix_invariant']} ({p['wall_seconds_this_arm']:.0f}s)", flush=True)

    # ---------- ③ 辅助臂（G 扫描：只跑 scan 口径） ----------
    for arm in side_arms:
        spec = cfg["side_arms"][arm]
        p = r2653.eval_arm(arm, spec, model, cfg, dev, tasks, eval_arr, ledger, main_n, pair_n,
                           scan_n, seed_main, SIDE_MEASURE.get(arm))
        p["plumbing_only"] = rec["plumbing_only"]
        rec["side_points"].append(p)
        rec["side_arms_completed"].append(arm)
        rec["side_arms_pending"] = [a for a in side_arms if a not in rec["side_arms_completed"]]
        rec["wp2_reconciliation"] = wp2_reconciliation(cfg, ledger, rec["points"] + rec["side_points"])
        dump(args.out_dir, rec, cfg)
        print(f"[辅助] {arm:<16} bpc_scan={p['bpc_scan']:.6f} "
              f"每位置={p['deployment']['bytes_per_position']:.0f} "
              f"实测={rec['wp2_reconciliation']['per_arm'][arm]['measured_bytes_per_position']} "
              f"一致={rec['wp2_reconciliation']['per_arm'][arm]['consistent']} "
              f"({p['wall_seconds_this_arm']:.0f}s)", flush=True)

    # ---------- ④ 结论量与记账 ----------
    rec["paired_delta"] = r2653.paired_deltas(rec, cfg, seed_main)
    rec["ranking_band"] = {
        "primary_band_bpc": cfg["criterion"]["band_primary_bpc"],
        "primary_caliber": cfg["criterion"]["band_primary_caliber"],
        "windows_used": cfg["eval"]["pairing_seeds"],
        "n_windows": len(cfg["eval"]["pairing_seeds"]),
        "standard_met": len(cfg["eval"]["pairing_seeds"]) >= 5,
        "secondary_in_force_band_bpc": round(
            max(max(p["bpc_paired_windows"].values()) - min(p["bpc_paired_windows"].values())
                for p in rec["points"] if p["bpc_paired_windows"]), 6) if rec["points"] else None,
        "secondary_caliber": "单臂跨窗口 bpc 极差（只作辅助描述）",
        "bands_not_mixed": cfg["criterion"]["bands_not_mixed"],
    }
    rec["partial"] = bool(rec["arms_pending"] or rec["side_arms_pending"])
    rec["cost"] = {"device": dev, "wall_seconds": round(time.time() - t0, 2),
                   "gpu_lock_taken": False,
                   "note": "纯 CPU；无 GPU 锁、无显存足迹。科学价签用**等效理论量**（每位置部署字节），"
                           "墙钟只作预算记账、不进判据。"}
    code_files = [(os.path.join(HERE, "hp64_arms.py")), os.path.join(HERE, "config.json"),
                  os.path.join(HERE, "hp64_train_driver.py"),
                  os.path.join(P2653, "fp8_arms.py"), os.path.join(P2653, "run.py"),
                  os.path.join(ROOT, "tools/evalkit/kv_quant.py")]
    rec["code"] = {os.path.relpath(p, ROOT): b21.b19.sha256_file(p) for p in code_files if os.path.exists(p)}
    dump(args.out_dir, rec, cfg)
    print(f"[完成] wall={rec['cost']['wall_seconds']:.0f}s；主臂 {len(rec['points'])} / 辅助 "
          f"{len(rec['side_points'])}；WP2 全臂一致={rec['wp2_reconciliation']['all_consistent']}；"
          f"partial={rec['partial']} → {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
