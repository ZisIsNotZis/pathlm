#!/usr/bin/env python3
"""2664 · 低秩（MLA 式）vs 直接块标尺量化 —— 同一「每位置部署字节」下的判决（唯一评测入口）。

契约：同目录 `README.md`（判据线**运行前**写入）与 `config.json`（臂表 / 字节轴 / 判据落地口径）。
本脚本只做一件事：把每条臂变成 `tokens -> logits`，走**同一条**评测路径（同一 `bpc_of`、
同一窗口序、同一合成任务批），算出

- clean bpc（主判：同字节配对 Δ vs 0.014 带，24 个评测窗口 ≥ 5）；
- 合成长程召回（第二腿：copy / induction / var_recall，Wilson 95% CI + 配对 bootstrap）；
- 每位置部署字节（解析式 + 实测张量形状反算，逐位对账）；
- 前缀不变性（量化臂必须 True）、参数量、上投影每 token 等效解码 MACs/FLOPs、训练健康量。

两族共用一把尺：**缓存都按 fp16 存**。量化臂的 K/V 走 `quantize_kv` 往返（fp16 档也走这条路径
⇒ 三个量化档之间唯一的差别是 `mode`）；低秩臂的潜向量 `c` 走 fp16 往返后再上投影
（部署侧缓存的就是 fp16 的 `c`）。`round_fp16=False` 时必须与 `retention.masked_logits`
**逐位相等**（仪器自检 ①）。

用法：
    python3 run.py                 # 全臂（取 GPU 锁 → 仪器自检 → 逐臂读数 → 汇总）
    python3 run.py --smoke         # 缩小批量/臂数，只验证管线
    python3 run.py --only mla_c272 # 只跑某条臂（调试；汇总阶段缺臂会如实报）
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from gpu_lock import GpuLock                                       # noqa: E402
from pathlm.data import load_enwik8_full                           # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, kv_head_liveness  # noqa: E402
from tools.evalkit.retention import masked_bpc, masked_logits      # noqa: E402
from tools.prereg_bind import require as require_prereg            # noqa: E402
from tools.run_provenance import sha256_file, write_results        # noqa: E402

LN2 = 0.6931471805599453
#: 量化臂的标尺方案：沿 `head_dim` 的块标尺，G = 32（= head_dim ⇒ per-token 标尺，仍严格前缀确定）。
SCHEME, GROUP = "block", 32
#: 复用来源（只读；不复制第二份实现）
P2656 = os.path.join(ROOT, "axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m")
P2632 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale")
P2619 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart")
P2631 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, P2656)
import mla_arms as MLA                                             # noqa: E402
sys.path.insert(0, P2632)
import block_arms as BA                                            # noqa: E402
#: 2619 的评测管线（`bpc_of` / 任务批 / Wilson / 地板阈值）；2621 起各实验都只读复用它。
B19 = _load_module(os.path.join(P2619, "run.py"), "kv_budget_chart_2619")
#: 2631 的前缀不变性仪器（两条检查）。
CC = _load_module(os.path.join(P2631, "causal_check.py"), "causal_check_2631")

CFG = json.loads(Path(HERE, "config.json").read_text(encoding="utf-8"))
T = CFG["model"]["seq_len"]
VOCAB = CFG["model"]["vocab_size"]
BAND = float(CFG["criterion"]["planar_rules"]["band"])


def log(msg: str) -> None:
    print(f"[2664] {msg}", flush=True)


def _rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


# --------------------------------------------------------------- 前向（三种结构）
def mla_cache_logits(model, tokens: torch.Tensor, round_fp16: bool = True) -> torch.Tensor:
    """低秩臂的评测前向：逐块复刻 `MLABlock.forward`，唯一加的一步是潜向量的 fp16 存储往返。

    `round_fp16=True` 是**部署口径**（缓存里的 `c` 就是 fp16）；`round_fp16=False` 时本函数
    必须与 `retention.masked_logits` 逐位相等（仪器自检 ①，运行时不通过即作废）。
    """
    x = model._input_states(tokens)
    B, T_, d = x.shape
    for blk in model.blocks:
        x_ln = blk.ln1(x)
        c = blk.wkv_down(x_ln)
        if round_fp16:
            c = c.half().float()
        k = blk.wk_up(c)
        v = blk.wv_up(c)
        q = blk.q(x_ln)
        nh, dh = blk.n_heads, blk.head_dim

        def shape(t, _B=B, _T=T_, _nh=nh, _dh=dh):
            return t.view(_B, _T, _nh, _dh).transpose(1, 2)

        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           attn_mask=None, is_causal=True,
                                           enable_gqa=False)
        a = a.transpose(1, 2).reshape(B, T_, d)
        x = x + blk.proj(a)
        x = x + blk.mlp(blk.ln2(x))
    return model.readout(x)


def make_logits_fn(model, structure: str, mode: str | None):
    """把一条臂变成 `tokens -> logits`（全实验唯一的接口）。

    - `mha` / `gqa`：走 `2632` 的块标尺感知前向（`mode` 取 fp16/int8/int4；fp16 也走这条路
      ⇒ 三个量化档之间唯一的差别是 `mode`）；
    - `mla`：走本文件的 `mla_cache_logits`（潜向量 fp16 往返）。
    """
    if structure in ("mha", "gqa"):
        if mode is None:
            raise ValueError(f"{structure} 臂必须给定量化档")
        # fp16 档没有标尺（`kv_quant` 里 scheme 只对 int8/int4 有意义）⇒ 退回 `per_value`
        scheme = "per_value" if mode == "fp16" else SCHEME
        return BA.make_logits_fn(model, {"kind": "quant", "mode": mode}, T, "cuda",
                                 scheme, GROUP)
    if structure == "mla":
        return lambda x: mla_cache_logits(model, x, round_fp16=True)
    raise ValueError(f"未知结构 {structure!r}")


# --------------------------------------------------------------- 模型装载
def config_for(arm: dict) -> DepthARConfig:
    c = {"d_model": CFG["model"]["d_model"], "n_layers": CFG["model"]["n_layers"],
         "n_heads": CFG["model"]["n_heads"], "seq_len": T,
         "mlp_mult": CFG["model"]["mlp_mult"], "dense": True,
         "dropout": CFG["model"]["dropout"], "norm_cap": CFG["model"]["norm_cap"],
         "n_kv_heads": arm["n_kv_heads"] if arm["structure"] == "gqa" else None,
         "mlp_hidden_sched": arm.get("mlp_hidden_sched")}
    return DepthARConfig(**c)


def load_arm_model(arm: dict, run: dict, dev: str):
    ckpt = os.path.join(ROOT, run["ckpt"])
    if not os.path.exists(ckpt):
        raise SystemExit(f"checkpoint 缺失：{run['ckpt']}（该臂未产出 ⇒ 先训练）")
    got = sha256_file(ckpt)
    if run.get("sha256") and got != run["sha256"]:
        raise SystemExit(f"{arm['label']}/{run['run']} sha256 不符：\n"
                         f"  契约 {run['sha256']}\n  实测 {got}")
    cfg = config_for(arm)
    if arm["structure"] == "mla":
        model = MLA.load_mla_model(ckpt, cfg, VOCAB, arm["latent"], device=dev)
    else:
        model = DepthARModel(cfg, VOCAB)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.to(dev).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    params = int(sum(p.numel() for p in model.parameters()))
    if params != arm["params_analytic"]:
        raise SystemExit(f"{arm['label']} 参数量 {params} != 解析 {arm['params_analytic']}")
    info = {"path": run["ckpt"], "sha256": got, "size_bytes": os.path.getsize(ckpt),
            "sha256_verified": True, "params": params}
    return model, info


def train_health(ckpt_rel: str) -> dict:
    """训练健康量：从该 run 自己的 `results.json` + `train_log.jsonl` 读（不重算训练）。"""
    run_dir = os.path.dirname(os.path.join(ROOT, ckpt_rel))
    out: dict = {"run_dir": _rel(run_dir)}
    res = os.path.join(run_dir, "results.json")
    if os.path.exists(res):
        d = json.load(open(res))
        out["grad_skips"] = d.get("gnorm_monitor", {}).get("grad_skips")
        out["gnorm_max"] = d.get("gnorm_monitor", {}).get("gnorm_max")
        out["train_steps_config"] = d.get("train", {}).get("steps")
        out["params_run"] = d.get("params")
    log_p = os.path.join(run_dir, "train_log.jsonl")
    if os.path.exists(log_p):
        losses = []
        steps = []
        for line in open(log_p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "loss" in r:
                losses.append(float(r["loss"]))
                steps.append(int(r.get("step", len(steps))))
        out["steps_done"] = steps[-1] if steps else 0
        out["n_log_lines"] = len(losses)
        if len(losses) >= 400:
            head = float(np.mean(losses[-400:-200]))
            tail = float(np.mean(losses[-200:]))
            out["ce_slope_last400"] = round(tail - head, 6)
            out["final_loss_mean_last200"] = round(tail, 6)
    # 引擎逐 step 记 `step` 从 0 起 ⇒ 末行 step = steps − 1 也算跑满
    out["healthy"] = bool(out.get("steps_done", 0) + 1 >= CFG["train"]["steps"]
                          and out.get("grad_skips", 1) == 0)
    return out


# --------------------------------------------------------------- 字节账 / 解码代价
def bytes_analytic(arm: dict, mode: dict | None) -> dict:
    L, H, dh = CFG["model"]["n_layers"], CFG["model"]["n_heads"], CFG["model"]["head_dim"]
    if arm["structure"] == "mla":
        values = L * arm["latent"]
        vb, sb = 2.0, 0.0
    elif arm["structure"] == "gqa":
        values = L * 2 * arm["n_kv_heads"] * dh
        vb, sb = 2.0, 0.0
    else:
        values = L * 2 * H * dh
        vb, sb = mode["value_bytes"], mode["scale_bytes_per_value"]
    per_pos = values * (vb + sb)
    return {"values_per_position": values, "value_bytes": vb,
            "scale_bytes_per_value": sb,
            "bytes_per_position": round(per_pos, 6),
            "to_fp16_mha_multiple": round(per_pos / 12288.0, 8)}


def bytes_measured(arm: dict, mode: dict | None, model, probe: torch.Tensor) -> dict:
    """实测张量形状反算每位置字节（活性证据，不是标签）。"""
    if arm["structure"] == "mla":
        live = MLA.mla_liveness(model, probe)
        return {"source": "mla_arms.mla_liveness", "latent_dim": live["latent_dim"],
                "kv_values_per_position": live["kv_values_per_position"],
                "n_scales": 0,
                "bytes_per_position": live["kv_bytes_per_position_fp16"],
                "alive": live["alive"]}
    if arm["structure"] == "gqa":
        live = kv_head_liveness(model, probe)
        return {"source": "depth_ar.kv_head_liveness",
                "n_kv_heads": live.get("n_kv_heads"),
                "kv_values_per_position": live.get("kv_elems_per_position"),
                "n_scales": 0,
                "bytes_per_position": live.get("kv_bytes_per_position_fp16"),
                "alive": bool(live.get("enabled"))}
    scheme = "per_value" if mode["mode"] == "fp16" else SCHEME
    act = BA.activity_of({"kind": "quant", "mode": mode["mode"]}, model, T, "cuda",
                         probe, scheme, GROUP)
    q = act["quant"]
    n_values, n_scales = q["n_values"], q["n_scales"]
    per_request_positions = probe.shape[0] * T
    deployed = n_values * mode["value_bytes"] + n_scales * 2.0
    return {"source": "block_arms.activity_of", "n_values": n_values,
            "n_scales": n_scales, "rel_err": q["rel_err"],
            "changed_frac": q["changed_frac"],
            "bytes_per_position": round(deployed / per_request_positions, 6),
            "alive": True if mode["mode"] == "fp16" else n_scales == n_values // GROUP}


def decode_cost(arm: dict) -> dict:
    L = CFG["model"]["n_layers"]
    h = CFG["model"]["mlp_mult"] * CFG["model"]["d_model"]
    if arm["structure"] == "mla":
        d = MLA.decode_macs(CFG["model"]["d_model"], h, "mla", arm["latent"], n_layers=L)
    elif arm["structure"] == "gqa":
        d = MLA.decode_macs(CFG["model"]["d_model"], h, "gqa", 0,
                            n_kv_heads=arm["n_kv_heads"], n_layers=L)
    else:
        d = MLA.decode_macs(CFG["model"]["d_model"], h, "mha", 0, n_layers=L)
    return d


# --------------------------------------------------------------- 召回 / 配对
def pooled_task(task_or_dict):
    """把「按距离/起点分组」的 `{key: TaskBatch}` 拼成一个批（与 `2619.pooled_accuracy` 同序）。

    `2619.build_tasks` 里 `*_by_*` 是 dict；本实验的逐项命中与地板参照都在**拼接后**的批上算，
    这样两侧臂的每一项逐位对齐（同一次任务构造 ⇒ 逐项同答案）。
    """
    if isinstance(task_or_dict, dict):
        return B19.concat([task_or_dict[k] for k in sorted(task_or_dict)])
    return task_or_dict


def item_hits(fn, task, dev: str, max_chunk: int = 64) -> torch.Tensor:
    """逐任务项的命中矩阵 `[B, M]`（与 `b19.synth_accuracy` 同算法）；同批任务 ⇒ 跨臂逐项对齐。"""
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        logits = fn(task.tokens[lo:hi].to(dev))
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(dev))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(dev)).to(torch.long).cpu()
    return hits


def recall_block(hits: torch.Tensor, task) -> dict:
    out = B19._summarize_hits(hits, task.dists)
    out["chance"] = round(task.chance(), 6)
    out["at_floor"] = bool(out["accuracy"] <= B19.floor_threshold(out["n"], task.chance()))
    return out


def paired_bootstrap(a: torch.Tensor, b: torch.Tensor, n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """两种臂在同一批任务项上的配对差（a − b）与 bootstrap 95% CI。率类指标不借用 0.014。"""
    da = a.reshape(-1).to(torch.float64).numpy()
    db = b.reshape(-1).to(torch.float64).numpy()
    if da.shape != db.shape:
        raise ValueError(f"配对的两臂任务项数不同：{da.shape} vs {db.shape}")
    d = da - db
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    means = d[idx].mean(axis=1)
    return {"delta": round(float(d.mean()), 6), "n": int(d.size),
            "ci95": [round(float(np.percentile(means, 2.5)), 6),
                     round(float(np.percentile(means, 97.5)), 6)],
            "separates_zero": bool(np.percentile(means, 2.5) > 0
                                   or np.percentile(means, 97.5) < 0)}


# --------------------------------------------------------------- 主流程
def arena_for(cfg: dict, dev: str):
    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))
    return eval_arr


def eval_arm(arm: dict, run: dict, mode: dict | None, tasks: dict, eval_arr, dev: str,
             smoke: bool) -> tuple[dict, dict]:
    model, info = load_arm_model(arm, run, dev)
    structure = arm["structure"]
    fn = make_logits_fn(model, structure, None if mode is None else mode["mode"])
    probe = tasks["var_recall"].tokens[:4].to(dev)

    rec: dict = {"arm": arm["label"], "run": run["run"], "seed": run["seed"],
                 "structure": structure,
                 "mode": mode["label"] if mode else "latent_fp16",
                 "checkpoint": info, "params": info["params"],
                 "train_health": train_health(run["ckpt"])}
    rec.update(bytes_analytic(arm, mode))
    bm = bytes_measured(arm, mode, model, probe)
    rec["bytes_measured"] = bm
    if abs(bm["bytes_per_position"] - rec["bytes_per_position"]) > 1e-6:
        raise SystemExit(f"{arm['label']}/{rec['mode']}: 解析字节 {rec['bytes_per_position']} "
                         f"!= 实测 {bm['bytes_per_position']}（字节账不一致）")
    bpc = B19.bpc_of(fn, eval_arr, T, cfg_n_batches(smoke),
                     CFG["eval"]["batch_size"], CFG["eval"]["window_seed"], dev)
    rec["bpc"] = round(bpc, 6)

    # 第二路径：纯 fp16 前向（无量化往返 / 无潜向量往返）——只对同一条模型可比
    if structure == "mla":
        pure = lambda x: mla_cache_logits(model, x, round_fp16=False)
        rec["bpc_roundtrip_effect"] = round(
            bpc - B19.bpc_of(pure, eval_arr, T, cfg_n_batches(smoke),
                             CFG["eval"]["batch_size"], CFG["eval"]["window_seed"], dev), 6)

    rec["prefix_invariance"] = CC.prefix_invariance_of_logits(fn, probe)
    rec["decode_cost"] = decode_cost(arm)

    hits = {}
    for key in ("copy_by_pos", "induction_by_start", "var_recall_by_d"):
        task = pooled_task(tasks[key])
        h = item_hits(fn, task, dev)
        hits[key] = h
        rec["recall_" + key] = recall_block(h, task)
    return rec, hits


def cfg_n_batches(smoke: bool) -> int:
    return 2 if smoke else CFG["eval"]["n_batches"]


def instrument_checks(arms_by_label: dict, tasks: dict, eval_arr, dev: str,
                      smoke: bool) -> dict:
    """仪器自检：路径逐位、跨实验同口径、fp16 存储效应。任一条不过就不接读数（门）。"""
    out: dict = {}

    # ② 路径一致：bpc_of 与 retention.masked_bpc（纯 fp16 前向，MHA）
    mha = arms_by_label["mha"]
    model, _ = load_arm_model(mha, mha["runs"][0], dev)
    pure = lambda x: masked_logits(model, x, None)
    a = B19.bpc_of(pure, eval_arr, T, cfg_n_batches(smoke), CFG["eval"]["batch_size"],
                   CFG["eval"]["window_seed"], dev)
    b = masked_bpc(model, eval_arr, None, n_batches=cfg_n_batches(smoke),
                   batch_size=CFG["eval"]["batch_size"],
                   window_seed=CFG["eval"]["window_seed"], device=dev)
    out["bpc_paths_equal"] = {"bpc_of": round(a, 6), "masked_bpc": round(b, 6),
                              "bitwise_equal": bool(abs(a - b) < 1e-12)}

    # ⑤ fp16 存储效应对 MHA：量化路径的 fp16 往返 vs 纯 fp16 前向（同一条 checkpoint）
    a_q = B19.bpc_of(make_logits_fn(model, "mha", "fp16"), eval_arr, T,
                     cfg_n_batches(smoke), CFG["eval"]["batch_size"],
                     CFG["eval"]["window_seed"], dev)
    out["kv_fp16_roundtrip_effect"] = {
        "bpc_kv_path_fp16": round(a_q, 6),
        "bpc_pure_fp16_mha": round(a, 6),
        "delta": round(a_q - a, 6),
        "note": "量化路径对 K/V 做 fp16 存储往返（部署口径），与纯 fp32 前向的差是**存储精度**效应；"
                "低秩臂的潜向量往返同源（见各臂 bpc_roundtrip_effect）"}
    del model
    torch.cuda.empty_cache()
    # ③ 跨实验同口径：5.1 的 trunk（2610）三点必须复现 2632 的 1.483571 / 1.483672 / 1.506093
    trunk = os.path.join(ROOT, "axes/6-predict/6.2-speculative-decoding/experiments/"
                               "2610-frozen-trunk-jumphead-13m/trunk.pt")
    if os.path.exists(trunk) and not smoke:
        mcfg = DepthARConfig(**{"d_model": 256, "n_layers": 12, "n_heads": 8,
                                "seq_len": T, "mlp_mult": 6, "dense": True,
                                "dropout": 0.0, "norm_cap": 1.0})
        tm = DepthARModel(mcfg, VOCAB)
        tm.load_state_dict(torch.load(trunk, map_location=dev))
        tm.to(dev).eval()
        got = {}
        for m in CFG["kv_modes"]:
            fn = make_logits_fn(tm, "mha", m["mode"])
            got[m["label"]] = round(B19.bpc_of(fn, eval_arr, T, CFG["eval"]["n_batches"],
                                               CFG["eval"]["batch_size"],
                                               CFG["eval"]["window_seed"], dev), 6)
        want = {"fp16": 1.483571, "int8": 1.483672, "int4": 1.506093}
        out["cross_experiment_5_1"] = {
            "checkpoint": _rel(trunk), "sha256": sha256_file(trunk),
            "expected": want, "got": got,
            "reproduces_2632": bool(all(abs(got[k] - want[k]) < 1e-9 for k in want))}
        del tm
        torch.cuda.empty_cache()

    # ① 低秩前向 = MLABlock 前向（round_fp16=False 时逐位相等）
    mla = arms_by_label.get("mla_c128") or arms_by_label.get("mla_c272")
    model, _ = load_arm_model(mla, mla["runs"][0], dev)
    probe = tasks["var_recall"].tokens[:4].to(dev)
    a_l = mla_cache_logits(model, probe, round_fp16=False)
    b_l = masked_logits(model, probe, None)
    out["mla_forward_equals_block"] = {
        "arm": mla["label"], "max_abs_delta": float((a_l - b_l).abs().max().item()),
        "bitwise_equal": bool(torch.equal(a_l, b_l))}

    out["gate_passed"] = bool(out["bpc_paths_equal"]["bitwise_equal"]
                              and out["mla_forward_equals_block"]["bitwise_equal"]
                              and out.get("cross_experiment_5_1", {}).get(
                                  "reproduces_2632", True))
    return out


def compare_same_byte(groups: list, per_arm: dict) -> dict:
    """同字节格：配对 Δ bpc（低秩 − 对照）+ 0.014 带裁决 + 召回配对 bootstrap。"""
    out = []
    for g in groups:
        lo_arm, comp = g["low_rank"], g["comparator"]
        comp_label, comp_mode = (comp.split("@") + [None])[:2] if "@" in comp else (comp, None)
        lo = per_arm.get(lo_arm) or {}
        co = per_arm.get(comp_label) or {}
        # `checkpoint_missing` 的 run 条目只有 `status`、没有 `mode`/`bpc` ⇒ 用 `.get` 滤掉
        lo_runs = [r for r in lo.get("runs", []) if r.get("mode") == "latent_fp16"]
        co_runs = [r for r in co.get("runs", [])
                   if r.get("mode") is not None
                   and (r["mode"] == comp_mode if comp_mode else True)]
        # 配对：按 seed 对齐；对不上的（格 C 的头共享一侧是两次 seed 0）如实标注
        pairs = []
        for i, r in enumerate(lo_runs):
            if not co_runs:
                break
            cand = [c for c in co_runs if c["seed"] == r["seed"]]
            c = cand[0] if cand else co_runs[min(i, len(co_runs) - 1)]
            pairs.append({"low_rank": r["run"], "seed_low_rank": r["seed"],
                          "comparator": c["run"], "seed_comparator": c["seed"],
                          "seed_aligned": bool(c["seed"] == r["seed"]),
                          "delta_bpc": round(r["bpc"] - c["bpc"], 6)})
        deltas = [p["delta_bpc"] for p in pairs]
        entry = {"bytes_per_position": g["bytes_per_position"], "low_rank": lo_arm,
                 "comparator": comp, "role": g["role"], "param_note": g["param_note"],
                 "delta_bpc_pairs": pairs,
                 "delta_bpc_mean": round(float(np.mean(deltas)), 6) if deltas else None,
                 "n_pairs": len(deltas)}
        if deltas:
            above = [d for d in deltas if d > BAND]
            below = [d for d in deltas if d < -BAND]
            entry["band"] = BAND
            entry["n_above_band"] = len(above)
            entry["n_below_band"] = len(below)
            entry["n_in_band"] = len(deltas) - len(above) - len(below)
            if len(above) == len(deltas):
                entry["verdict"] = "comparator_better_sortable"
            elif len(below) == len(deltas):
                entry["verdict"] = "low_rank_better_sortable"
            elif not above and not below:
                entry["verdict"] = "indistinguishable_in_band"
            else:
                entry["verdict"] = "unjudgeable_this_n"
                sd = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
                if sd:
                    entry["sigma_paired_delta"] = round(sd, 6)
                    entry["n_required"] = int(np.ceil((2 * sd / BAND) ** 2))
        out.append(entry)
    return {"groups": out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-lock", action="store_true", help="调试用；正式读数必须取锁")
    args = ap.parse_args()

    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         f"{_rel(HERE)}（再提交绑定，最后才跑读数）")

    want = "cuda" if torch.cuda.is_available() else "cpu"
    dev = want
    lock = None
    if not args.no_lock and want == "cuda":
        lock = GpuLock(timeout_s=600, poll_s=15)
        lock.acquire()
        log(f"GPU lock acquired wait={lock.ev['wait_seconds']}s")
    try:
        t0 = time.time()
        arms = CFG["model_arms"]
        by_label = {a["label"]: a for a in arms}
        if args.only:
            arms = [a for a in arms if a["label"] == args.only]
        if args.smoke:
            keep = {"mha", "mla_c272"}
            arms = [a for a in arms if a["label"] in keep] or arms[:1]
        tasks = B19.build_tasks(CFG, smoke=args.smoke)
        eval_arr = arena_for(CFG, dev)
        log(f"tasks built; eval windows = {cfg_n_batches(args.smoke)}; device={dev}")

        rec: dict = {"experiment": CFG["experiment"], "item": CFG["item"],
                     "work_package": CFG["work_package"], "scope": CFG["scope"],
                     "criterion": CFG["criterion"]["planar_rules"],
                     "band": BAND, "device": dev, "smoke": args.smoke,
                     "eval": {k: CFG["eval"][k] for k in ("n_batches", "batch_size",
                                                          "window_seed")}}
        rec["instrument"] = instrument_checks(by_label, tasks, eval_arr, dev, args.smoke)
        log(f"instrument gate: {rec['instrument']['gate_passed']}")
        if not rec["instrument"]["gate_passed"]:
            write_results(os.path.join(HERE, "results.json"), rec, source=__file__, out_dir=HERE)
            raise SystemExit("仪器自检未过（门禁）：不接读数，先修实现")

        per_arm: dict = {}
        hits_store: dict = {}
        for arm in arms:
            entry = {"label": arm["label"], "family": arm["family"],
                     "mechanism": arm["mechanism"], "structure": arm["structure"],
                     "latent": arm["latent"], "n_kv_heads": arm["n_kv_heads"],
                     "mlp_hidden_sched": arm.get("mlp_hidden_sched"),
                     "params_analytic": arm["params_analytic"], "runs": []}
            if arm["structure"] == "mla":
                modes = [None]
            elif args.smoke:
                modes = [m for m in CFG["kv_modes"] if m["label"] == "fp16"]
            else:
                modes = CFG["kv_modes"]
            for run in arm["runs"]:
                if not os.path.exists(os.path.join(ROOT, run["ckpt"])):
                    entry["runs"].append({"run": run["run"], "seed": run["seed"],
                                          "status": "checkpoint_missing"})
                    log(f"SKIP {arm['label']}/{run['run']}（checkpoint 未产出）")
                    continue
                for mode in modes:
                    r, hits = eval_arm(arm, run, mode, tasks, eval_arr, dev, args.smoke)
                    entry["runs"].append(r)
                    hits_store[(arm["label"], run["run"],
                                r["mode"])] = hits
                    tag = f"{arm['label']}/{run['run']}/{r['mode']}"
                    log(f"{tag}: bpc={r['bpc']} bytes={r['bytes_per_position']} "
                        f"prefix={r['prefix_invariance']['future_token_past_logits_equal']}"
                        f"/{r['prefix_invariance']['peer_request_logits_equal']}")
                    write_results(os.path.join(
                        HERE, f"results_{arm['label']}_{r['mode']}_{run['run']}.json"),
                        r, source=__file__, out_dir=HERE)
            per_arm[arm["label"]] = entry
        rec["points"] = per_arm

        # 同字节判决
        rec["same_byte"] = compare_same_byte(CFG["same_byte_groups"], per_arm)
        # 族内失效模式（量化：同 checkpoint 的 int8/int4 − fp16；低秩：− MHA fp16 同 seed）
        rec["damage"] = damage_table(per_arm)
        # 召回配对 bootstrap（同字节格两侧）
        rec["recall_paired"] = recall_pairs(CFG["same_byte_groups"], per_arm, hits_store)
        rec["recall_floor_reference"] = recall_floors(tasks)
        rec["answer"] = answer_brief(rec)
        rec["delta_directions"] = {
            "delta_bpc": "arm_minus_comparator",
            "delta_bpc_mean": "arm_minus_comparator",
            "delta_bpc_pairs": "arm_minus_comparator",
            "delta": "arm_minus_comparator"}
        rec["gpu_lock"] = lock.ev if lock else {"lock_path": None, "note": "--no-lock（调试）"}
        rec["cost"] = {"gpu_wall_min_budget": CFG["cost"]["gpu_wall_budget_min"],
                       "wall_minutes_this_run": round((time.time() - t0) / 60.0, 2)}
        write_results(os.path.join(HERE, "results.json"), rec, source=__file__, out_dir=HERE)
        Path(os.path.join(HERE, "table.md")).write_text(render_table(rec), encoding="utf-8")
        log("results.json / table.md 已落盘")
    finally:
        if lock is not None:
            lock.release()
            log("GPU lock released")
    return 0


def damage_table(per_arm: dict) -> dict:
    """族内失效模式：每条臂相对「同族 fp16 载体」的损伤（正 = 更差）。"""
    out = {}
    mha_runs = {(r["run"], r["seed"]): r for r in per_arm.get("mha", {}).get("runs", [])
                if r.get("mode") == "fp16"}
    quant = {}
    for mode in ("int8", "int4"):
        rows = []
        for r in per_arm.get("mha", {}).get("runs", []):
            if r.get("mode") != mode:
                continue
            base = mha_runs.get((r["run"], r["seed"]))
            if base:
                rows.append({"run": r["run"], "seed": r["seed"],
                             "delta_bpc": round(r["bpc"] - base["bpc"], 6)})
        if rows:
            quant[mode] = {"pairs": rows,
                           "delta_bpc_mean": round(float(np.mean([x["delta_bpc"] for x in rows])), 6)}
    out["quant_vs_fp16"] = quant
    lowrank = {}
    for label in ("mla_c272", "mla_c272_eq", "mla_c144", "mla_c128"):
        rows = []
        for r in per_arm.get(label, {}).get("runs", []):
            if r.get("mode") != "latent_fp16":
                continue
            base = mha_runs.get((r["run"], r["seed"])) or next(
                (v for (run, seed), v in mha_runs.items() if seed == r["seed"]), None)
            if base:
                rows.append({"run": r["run"], "seed": r["seed"],
                             "delta_bpc": round(r["bpc"] - base["bpc"], 6)})
        if rows:
            lowrank[label] = {"pairs": rows,
                              "delta_bpc_mean": round(float(np.mean([x["delta_bpc"] for x in rows])), 6)}
    out["lowrank_vs_mha_fp16"] = lowrank
    out["sign_convention"] = "delta = 臂 − 对照（正 = 该臂更差）"
    return out


def recall_pairs(groups: list, per_arm: dict, hits_store: dict) -> list:
    out = []
    for g in groups:
        comp_label, _, comp_mode = g["comparator"].partition("@")
        lo_arm = g["low_rank"]
        lo_run = next((r["run"] for r in per_arm.get(lo_arm, {}).get("runs", [])
                       if r.get("mode") == "latent_fp16"), None)
        co_run = next((r["run"] for r in per_arm.get(comp_label, {}).get("runs", [])
                       if r.get("mode") == (comp_mode or "fp16")), None)
        if not lo_run or not co_run:
            continue
        entry = {"bytes_per_position": g["bytes_per_position"], "low_rank": lo_arm,
                 "comparator": g["comparator"], "pairs": {}}
        for key in ("copy_by_pos", "induction_by_start", "var_recall_by_d"):
            a = hits_store.get((lo_arm, lo_run, "latent_fp16"))
            b = hits_store.get((comp_label, co_run, comp_mode or "fp16"))
            if a is None or b is None:
                continue
            entry["pairs"][key] = paired_bootstrap(a, b)
        out.append(entry)
    return out


def recall_floors(tasks: dict) -> dict:
    """chance 与局部先验地板（与臂无关的参照系）。"""
    from tools.evalkit.longrange_tasks import local_prior_accuracy
    out = {}
    for key in ("copy_by_pos", "induction_by_start", "var_recall_by_d"):
        task = pooled_task(tasks[key])
        lp = local_prior_accuracy(task, CFG["tokens"]["local_k"])
        out[key] = {"n": int(task.B * task.M), "chance": round(task.chance(), 6),
                    "local_prior_accuracy": round(float(lp["accuracy"]), 6),
                    "floor_threshold": round(B19.floor_threshold(task.B * task.M,
                                                                 task.chance()), 6)}
    return out


def answer_brief(rec: dict) -> dict:
    """把同字节格的裁决压成一句话（判词的人话骨架，最终判读写在 analysis.md）。"""
    lines = []
    for g in rec.get("same_byte", {}).get("groups", []):
        lines.append({"bytes_per_position": g["bytes_per_position"],
                      "low_rank": g["low_rank"], "comparator": g["comparator"],
                      "delta_bpc_mean": g.get("delta_bpc_mean"),
                      "verdict": g.get("verdict")})
    return {"same_byte_verdicts": lines,
            "note": "Δ = 低秩 − 对照（正 = 低秩更差）；带 = 0.014（限配对 Δ、24 个评测窗口）"}


def render_table(rec: dict) -> str:
    """把读数渲染成 markdown 表（人话在前；数字都带键路径）。"""
    L = []
    L.append("# 2664 · 同字节判决表（`results.json` 的渲染）\n")
    L.append("> 本文件由 `run.py::render_table` 从 `results.json` 生成；判读见 `analysis.md`。\n")
    L.append("## 1. 逐臂读数\n")
    L.append("| 臂 | 模式 | 运行 | bpc | 每位置字节 | ×fp16 | 参数 | 前缀不变 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, arm in rec.get("points", {}).items():
        for r in arm.get("runs", []):
            if r.get("status") == "checkpoint_missing":
                L.append(f"| `{label}` | — | `{r['run']}` | 未跑 | 未跑 | 未跑 | 未跑 | 未跑 |")
                continue
            pi = r["prefix_invariance"]
            ok = "True" if (pi["future_token_past_logits_equal"]
                            and pi["peer_request_logits_equal"]) else "**False**"
            L.append(f"| `{label}` | {r['mode']} | `{r['run']}` | {r['bpc']} | "
                     f"{r['bytes_per_position']} | {r['to_fp16_mha_multiple']} | "
                     f"{r['params']} | {ok} |")
    L.append("\n## 2. 同字节格（主判）\n")
    L.append("| 字节/位置 | 低秩 | 对照 | 配对 Δ（低秩−对照） | 带 | 裁决 |")
    L.append("|---|---|---|---|---|---|")
    for g in rec.get("same_byte", {}).get("groups", []):
        L.append(f"| {g['bytes_per_position']} | `{g['low_rank']}` | `{g['comparator']}` | "
                 f"{g.get('delta_bpc_mean')} | ±{g.get('band')} | {g.get('verdict')} |")
    L.append("\n逐对明细：")
    for g in rec.get("same_byte", {}).get("groups", []):
        for p in g.get("delta_bpc_pairs", []):
            L.append(f"- {g['bytes_per_position']} B：`{p['low_rank']}` − `{p['comparator']}` "
                     f"= {p['delta_bpc']}（seed {p['seed_low_rank']} / {p['seed_comparator']}，"
                     f"对齐={p['seed_aligned']}）")
    L.append("\n## 3. 族内失效模式\n")
    for k, v in rec.get("damage", {}).items():
        if k == "sign_convention":
            continue
        for name, row in v.items():
            L.append(f"- `{k}` / `{name}`：Δbpc 均值 {row['delta_bpc_mean']}（"
                     + ", ".join(f"{p['run']}={p['delta_bpc']}" for p in row["pairs"]) + "）")
    L.append("\n## 4. 召回（配对 bootstrap，a − b）\n")
    for e in rec.get("recall_paired", []):
        for key, v in e["pairs"].items():
            L.append(f"- {e['bytes_per_position']} B `{e['low_rank']}` − `{e['comparator']}` "
                     f"@ {key}: Δ {v['delta']}，95% CI {v['ci95']}，n={v['n']}")
    L.append("\n## 5. 仪器自检与字节账\n")
    ins = rec.get("instrument", {})
    L.append(f"- 路径逐位：{ins.get('bpc_paths_equal')}")
    L.append(f"- 低秩前向 == MLABlock：{ins.get('mla_forward_equals_block')}")
    L.append(f"- 跨实验 5.1 口径复现：{ins.get('cross_experiment_5_1')}")
    L.append(f"- fp16 存储效应（MHA）：{ins.get('kv_fp16_roundtrip_effect')}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
