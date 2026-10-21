#!/usr/bin/env python3
"""2685 · 把 int4 档与 carry 族的 int8 档叠到「每位置部署字节 vs clean bpc」同一张图上。

契约：同目录 `README.md`（判据线）与 `config.json`（臂 / 格 / 判据常数 / 字节口径在 `analytic`）。
只做四件事：过预登记门 → 取 §20 GPU 锁 → 对 13 条已落盘 checkpoint 跑各自的精度格
（纯窗臂走 `2632` 的仪器：fp16 / int8 / int4；carry 臂走本目录 `carry_quant.py` 的新量化前向：
fp16 / int8只KV / int8全档 / int4只KV / int4全档）→ 在合并网格上重算支配关系并逐条落判据，
写 `results.json`。**0 训练 token，不改任何共享文件**（`pathlm/**`、`train_depth_ar.py`、`tools/**`）。

判据① 的两道复现门是load-bearing的：纯窗臂 fp16/int8 必须 1e-6 内等于 `2681` 落盘值，
carry 臂在 (KV 不量化, 状态不量化) 下必须 1e-6 内等于同一份落盘值——后者证明新前向是
已入库仪器的忠实复刻，而不是另一个模型。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
EXP = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                       # noqa: E402
import torch.nn.functional as F                                    # noqa: E402

import carry_quant as cq                                           # noqa: E402

from pathlm.data import batch, load_enwik8_full                    # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, window_anchor_keep_mask  # noqa: E402
from tools.evalkit.retention import masked_logits                  # noqa: E402

P2632 = ROOT / "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale"
P2662 = ROOT / "axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOCK = Path("/tmp/pathlm_gpu.lock")
LOCK_WAIT_S = 20 * 60
LN2 = 0.6931471805599453


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ba = _load_module(P2632 / "block_arms.py", "block_arms_2632")
rp = _load_module(P2662 / "realtext_probe.py", "rp2662")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class GpuLock:
    def __init__(self, path: Path = LOCK, wait_s: int = LOCK_WAIT_S):
        self.path, self.wait_s, self.held = path, wait_s, False

    def __enter__(self):
        waited = 0
        while True:
            try:
                self.path.mkdir()
            except FileExistsError:
                hp = self.path / "holder_pid"
                pid = hp.read_text().strip() if hp.exists() else ""
                if pid and not _alive(int(pid)):
                    shutil.rmtree(self.path, ignore_errors=True)
                    continue
                if waited >= self.wait_s:
                    raise SystemExit(f"等 GPU 锁超时（{self.wait_s}s，holder={pid or '未知'}）")
                time.sleep(15)
                waited += 15
                continue
            (self.path / "holder_pid").write_text(str(os.getpid()))
            self.held = True
            print(f"lock acquired at {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
            return self

    def __exit__(self, *exc):
        if self.held:
            shutil.rmtree(self.path, ignore_errors=True)
        return False


def gate_binding() -> dict:
    res = subprocess.run([sys.executable, "tools/prereg_bind.py", str(EXP), "--require"],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise SystemExit("预登记绑定门未过：先跑 python3 tools/prereg_bind.py <本目录>")
    return {"require_rc": 0, "stdout": res.stdout.strip()}


def model_kwargs(cfg: dict, arm: dict) -> dict:
    kw = {k: v for k, v in cfg["model"].items() if k != "note"}
    kw["train_window"] = int(arm["window"])
    kw["train_anchors"] = 0
    if arm["kind"] == "carry":
        spec = cfg["carry_specs"][arm["name"]]
        kw.update({"carry_rule": spec["carry_rule"],
                   "carry_per_layer": spec["carry_per_layer"],
                   "carry_decay": spec["carry_decay"],
                   "carry_chunk": spec["carry_chunk"]})
    return kw


def load_arm(cfg: dict, arm: dict, vocab: int, dev: str):
    path = ROOT / arm["path"]
    if not path.exists():
        raise SystemExit(f"{arm['name']}: 权重不存在 {arm['path']}")
    got = sha256_file(path)
    if got != arm["sha256"]:
        raise SystemExit(f"{arm['name']}: sha256 mismatch\n want {arm['sha256']}\n got  {got}")
    model = DepthARModel(DepthARConfig(**model_kwargs(cfg, arm)), vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": arm["path"], "sha256": got, "sha256_verified": True,
                   "size_bytes": os.path.getsize(path), "window": arm["window"],
                   "seed": arm["seed"], "kind": arm["kind"],
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "n_state_values": cq.num_state_values(model)}


@torch.no_grad()
def bpc_of(fn, eval_arr, T: int, n_batches: int, batch_size: int, window_seed: int,
           device: str, per_window: bool = False):
    """clean bpc（bits/byte）。与 `retention.masked_bpc` 同一算法、同一窗口序。"""
    gen = torch.Generator().manual_seed(window_seed)
    nats, n_tok = 0.0, 0
    bits_per_window, n_per_window = [], []
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits = fn(x)[:, :-1]
        tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none").view(batch_size, -1)
        nats += ce.double().sum().item()
        n_tok += tgt.numel()
        if per_window:
            bits_per_window.extend((ce.double().sum(1) / LN2).cpu().tolist())
            n_per_window.extend([tgt.shape[1]] * batch_size)
    bpc = nats / n_tok / LN2
    return (bpc, bits_per_window, n_per_window) if per_window else bpc


def bytes_of(analytic: dict, window: int, n_state: int, kv_mode: str,
             state_mode: str) -> dict:
    """`config.json::analytic` 的解析字节账（整数，非整数即解析口径写错）。

    `mode="none"` 的部署字节按 fp16 计（不量化 ⇒ 存 fp16 原值），因此字节轴上
    「carry fp16 格」与已入库 `2639` 的 393,216 B 同一口径。
    """
    bpv = analytic["bytes_per_value"]
    kv = window * analytic["kv_values_per_position_per_key"] * bpv[
        "fp16" if kv_mode == "none" else kv_mode]
    st = n_state * bpv["fp16" if state_mode == "none" else state_mode]
    for v, what in ((kv, "kv"), (st, "state")):
        if abs(v - round(v)) > 1e-9:
            raise SystemExit(f"字节账非整数（{what}）：{v}——analytic 的 G / 位宽口径写错")
    return {"kv_bytes": int(round(kv)), "state_bytes": int(round(st)),
            "total_bytes": int(round(kv + st))}


def domination(points: list[dict], band: float) -> dict:
    """被支配 ⇔ 存在不更多字节且 bpc 低出一个带的点。"""
    out = {}
    for p in points:
        dom = [q["name"] for q in points
               if q["name"] != p["name"] and q["total_bytes"] <= p["total_bytes"]
               and q["bpc"] <= p["bpc"] - band]
        out[p["name"]] = {"dominated_by": dom}
    return out


def cell_fn(model, arm: dict, cell: dict, km, dev: str):
    """把一格 (KV 精度, 状态精度) 变成 `tokens -> logits`。纯窗臂走 `2632` 的仪器，
    carry 臂走本目录的新量化前向（两档都不量化时它必须复现已入库仪器）。"""
    if arm["kind"] == "window":
        mode = cell["kv"]["mode"]
        if mode == "none":
            return lambda x, m=model, k=km: masked_logits(m, x, k)
        scheme = cell["kv"].get("scheme", "block")
        group = int(cell["kv"].get("group", 32))
        return lambda x, m=model, k=km, md=mode, s=scheme, g=group: \
            ba.masked_kv_quant_logits(m, x, k, md, s, g)[0]
    return cq.make_logits_fn(model, km, cell["kv"], cell["state"])


def main() -> int:
    cfg = json.loads((EXP / "config.json").read_text())
    crit, ev, analytic = cfg["criterion"], cfg["eval"], cfg["analytic"]
    quant = cfg["quant"]
    dev, T = cfg["device"], int(cfg["model"]["seq_len"])
    cells = {c["id"]: c for c in quant["cells"]}
    t0 = time.time()
    out = {"experiment": cfg["experiment"], "owner_cell": cfg["owner_cell"],
           "contract": "README.md", "budget": cfg["budget"], "prereg_gate": gate_binding()}

    _train, val, vocab = load_enwik8_full(str(ROOT / cfg["data"]["raw_path"]),
                                          str(ROOT / cfg["data"]["cache"]))
    rows, bits, checkpoints = {}, {}, {}
    with GpuLock():
        torch.cuda.reset_peak_memory_stats()
        for arm in cfg["arms"]:
            model, checkpoints[arm["name"]] = load_arm(cfg, arm, vocab, dev)
            n_state = checkpoints[arm["name"]]["n_state_values"]
            km = window_anchor_keep_mask(T, int(arm["window"]), 0, device=dev)
            name = arm["name"]
            wanted = (quant["window_cells"] if arm["kind"] == "window"
                      else quant["carry_cells"])
            rows[name] = {"window": arm["window"], "seed": arm["seed"], "kind": arm["kind"],
                          "n_params": checkpoints[name]["n_params"],
                          "n_state_values": n_state,
                          "key_visible_pairs": int(km.sum()), "cells": {}}
            for cid in wanted:
                cell = cells[cid]
                ts = time.time()
                fn = cell_fn(model, arm, cell, km, dev)
                bpc, bpw, npw = bpc_of(fn, val, T, ev["n_batches"], ev["batch_size"],
                                       ev["window_seed"], dev, per_window=True)
                ledger = bytes_of(analytic, int(arm["window"]), n_state,
                                  cell["kv"]["mode"], cell["state"]["mode"])
                rows[name]["cells"][cid] = {
                    "kv_mode": cell["kv"]["mode"], "state_mode": cell["state"]["mode"],
                    "bpc": round(float(bpc), 6), **ledger,
                    "wall_s": round(time.time() - ts, 1)}
                bits[(name, cid)] = (bpw, npw)
        out["cost"] = {"eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                       "wall_minutes_total": round((time.time() - t0) / 60, 2),
                       "n_training_tokens": 0,
                       "n_cells": sum(len(r["cells"]) for r in rows.values()),
                       "note": "墙钟与显存峰值只作过程指标，不进判据"}

    out["checkpoints"], out["rows"] = checkpoints, rows

    # 判据①：两道复现门
    tol = crit["equality_gate_tolerance_bpc"]
    gate = {"fp16": {}, "int8": {}}
    for name, want in crit["equality_gate_fp16"].items():
        got = rows[name]["cells"]["fp16"]["bpc"]
        gate["fp16"][name] = {"bpc": got, "bpc_2681": want, "equal": abs(got - want) <= tol}
    for name, want in crit["equality_gate_int8"].items():
        got = rows[name]["cells"]["kv8"]["bpc"]
        gate["int8"][name] = {"bpc": got, "bpc_2681": want, "equal": abs(got - want) <= tol}
    out["equality_gate"] = {
        "fp16": {"arms": gate["fp16"], "all_equal": all(v["equal"] for v in gate["fp16"].values())},
        "int8": {"arms": gate["int8"], "all_equal": all(v["equal"] for v in gate["int8"].values())},
        "tolerance_bpc": tol, "reference": crit["equality_gate_source"],
        "carry_fp16_source": "carry 臂的 fp16 格走的是本目录 `carry_quant.py`（新代码）；"
                             "它与 `2681` 落盘值的相等性即「新前向是已入库仪器的忠实复刻」",
        "consequence": "任一不等 ⇒ 本实验作废（仪器 / 权重 / 窗口序变了），判据线不改"}
    if not (out["equality_gate"]["fp16"]["all_equal"]
            and out["equality_gate"]["int8"]["all_equal"]):
        (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        raise SystemExit("判据① 复现门未过：见 results.json::equality_gate")

    # 判据② 字节账 + 判据③④ 质量腿（逐臂逐对配对 bootstrap）
    def pair_delta(name: str, cell_a: str, cell_b: str, key: str) -> dict:
        """Δ = bpc(cell_a) − bpc(cell_b)，逐窗配对 bootstrap 95% CI。"""
        a_bits, b_bits = bits[(name, cell_a)][0], bits[(name, cell_b)][0]
        n_tok = sum(bits[(name, cell_a)][1])
        d_bits = [x - y for x, y in zip(a_bits, b_bits)]
        delta = sum(d_bits) / n_tok
        expect = rows[name]["cells"][cell_a]["bpc"] - rows[name]["cells"][cell_b]["bpc"]
        if abs(delta - expect) >= 2e-6:
            raise SystemExit(f"{name}/{key}: Δ 与两格 bpc 差对不上：{delta:.9f} vs {expect:.9f}")
        ci = [round(v, 6) for v in rp.paired_cluster_bootstrap_ci(
            d_bits, bits[(name, cell_a)][1], n_boot=crit["n_boot"],
            seed=crit["boot_seed"], alpha=crit["alpha"])]
        return {key: round(delta, 6), "ci95_paired_bootstrap": ci,
                "n_windows": len(bits[(name, cell_a)][1]), "n_tokens": n_tok,
                "above_cross_cell_band": bool(abs(delta) >= crit["cross_cell_band"]),
                "in_sigma_cell_units": round(abs(delta) / crit["sigma_cell"], 1),
                "verdict": ("本尺度无质量增量" if abs(delta) < crit["cross_cell_band"]
                            else "有可排序质量代价")}

    int4_window, carry_tiers = {}, {}
    for name, r in rows.items():
        c = r["cells"]
        if r["kind"] == "window":
            int4_window[name] = pair_delta(name, "kv4", "fp16", "delta_int4_minus_fp16_bpc")
        else:
            carry_tiers[name] = {
                "kv8_vs_fp16": pair_delta(name, "kv8", "fp16",
                                          "delta_kv8_minus_fp16_bpc"),
                "state8_vs_kv8": pair_delta(name, "kv8_state8", "kv8",
                                            "delta_state8_minus_kv8_bpc"),
                "tier4_vs_fp16": pair_delta(name, "kv4_state4", "fp16",
                                            "delta_tier4_minus_fp16_bpc"),
                "kv4_vs_fp16": pair_delta(name, "kv4", "fp16",
                                          "delta_kv4_minus_fp16_bpc"),
            }
    out["quant_leg"] = {
        "int4_window": int4_window,
        "carry_tiers": carry_tiers,
        "band": crit["cross_cell_band"],
        "definition": "Δ 一律是「实验格 bpc − 对照格 bpc」；bpc 越低越好 ⇒ 正 = 量化更差。"
                      "kv8 对 fp16 = 只换 KV 精度；state8 对 kv8 = 只换状态精度（单因素对）；"
                      "tier4 对 fp16 = KV 与状态都到 int4 档；kv4 对 fp16 = 只换 KV 到 int4"}

    # 判据② 解析对账（G 与位宽来自 config.json::analytic）
    vpp = analytic["kv_values_per_position_per_key"]
    out["byte_ledger"] = {
        "group": analytic["group"], "bits": analytic["bits"],
        "bytes_per_value": analytic["bytes_per_value"],
        "kv_values_per_position_per_key": vpp,
        "kv_bytes_per_key_analytic": {m: round(vpp * b, 4)
                                      for m, b in analytic["bytes_per_value"].items()},
        "state_values": analytic["state_values"],
        "state_values_read_from_model": {n: ck["n_state_values"]
                                         for n, ck in checkpoints.items() if ck["n_state_values"]},
        "matches_cost_axis_fp16": bool(vpp * 2 == 12288),
        "matches_cost_axis_int8": bool(vpp * (1 + 2 / analytic["group"]) == 6528),
        "matches_cost_axis_int4": bool(vpp * (0.5 + 2 / analytic["group"]) == 3456),
        "carry_tier_bytes": {
            n: {"kv8_state8": rows[n]["cells"]["kv8_state8"]["total_bytes"],
                "kv8": rows[n]["cells"]["kv8"]["total_bytes"],
                "kv4_state4": rows[n]["cells"]["kv4_state4"]["total_bytes"],
                "fp16": rows[n]["cells"]["fp16"]["total_bytes"]}
            for n in carry_tiers},
        "ratio_int8_vs_fp16_kv": round(analytic["bytes_per_value"]["int8"]
                                       / analytic["bytes_per_value"]["fp16"], 6),
        "ratio_int4_vs_fp16_kv": round(analytic["bytes_per_value"]["int4"]
                                       / analytic["bytes_per_value"]["fp16"], 6)}

    # 判据⑤⑥：合并网格上的前沿与支配
    points = []
    for name, r in rows.items():
        for cid, c in r["cells"].items():
            points.append({"name": f"{name}@{cid}", "arm": name, "cell": cid,
                           "kv_mode": c["kv_mode"], "state_mode": c["state_mode"],
                           "total_bytes": c["total_bytes"], "kv_bytes": c["kv_bytes"],
                           "state_bytes": c["state_bytes"], "bpc": c["bpc"]})
    points.sort(key=lambda p: (p["total_bytes"], p["bpc"]))
    band = crit["cross_cell_band"]
    dom = domination(points, band)
    envelope = [p["name"] for p in points if not dom[p["name"]]["dominated_by"]]

    def tier_report(point_name: str) -> dict:
        p = next(q for q in points if q["name"] == point_name)
        cheaper = [q for q in points if q["total_bytes"] <= p["total_bytes"]
                   and q["name"] != p["name"]]
        best = min(cheaper, key=lambda q: q["bpc"]) if cheaper else None
        same = [q for q in points if q["total_bytes"] == p["total_bytes"]
                and q["name"] != p["name"]]
        return {"point": point_name, "total_bytes": p["total_bytes"], "bpc": p["bpc"],
                "dominated_by": dom[p["name"]]["dominated_by"],
                "dominated": bool(dom[p["name"]]["dominated_by"]),
                "domination_set": f"total_bytes ≤ {p['total_bytes']} 且 bpc ≤ "
                                  f"{round(p['bpc'] - band, 6)}",
                "best_cheaper_or_equal": best["name"] if best else None,
                "margin_vs_best_cheaper": (round(best["bpc"] - p["bpc"], 6) if best else None),
                "same_bytes_points": [{"name": q["name"], "bpc": q["bpc"],
                                       "bpc_gap_vs_this": round(q["bpc"] - p["bpc"], 6)}
                                      for q in same],
                "vs_band": ("不被支配，且最优不更多字节的点差一个可排序带"
                            if best and best["bpc"] - p["bpc"] > band else
                            "不被支配，但同字节/更便宜的最强点与之不可排序"
                            if best and abs(best["bpc"] - p["bpc"]) <= band else
                            "该格被支配")}

    out["frontier"] = {
        "points": points, "dominated_by": dom, "lower_envelope": envelope,
        "band": band,
        "carry_fp16_point": tier_report("carrypl16_s0@fp16"),
        "carry_int8_tier": tier_report("carrypl16_s0@kv8_state8"),
        "carry_int4_tier": tier_report("carrypl16_s0@kv4_state4"),
        "carry_kv8_only": tier_report("carrypl16_s0@kv8"),
        "grid": {"n_points": len(points), "n_arms": len(rows),
                 "window_cells": quant["window_cells"], "carry_cells": quant["carry_cells"],
                 "natural_same_bytes_pairs": [
                     {"carry": "carrypl16_s0@kv8_state8", "window": "win32_s0@kv8",
                      "total_bytes": 208896},
                     {"carry": "carrypl16_s0@kv4_state4", "window": "win32_s0@kv4",
                      "total_bytes": 110592}]},
        "grid_qualifier": "结论的适用范围 = 这张网格（9 条纯窗臂 × {fp16,int8,int4} + 4 条 carry 臂 × "
                          "{fp16, 只KV(int8), int8 全档, 只KV(int4), int4 全档}）；不得外推到 T ≫ 256、"
                          "多 seed（除 win16 / carry16 各两 seed 外均为 n=1）、以及除块标尺 G=32 外的标尺方案",
        "no_interpolation": "所有点都是在盘 checkpoint 的实测评测；两点之间未插值"}
    out["delta_directions"] = {
        "delta_int4_minus_fp16_bpc": "arm_minus_comparator",
        "delta_kv8_minus_fp16_bpc": "arm_minus_comparator",
        "delta_state8_minus_kv8_bpc": "arm_minus_comparator",
        "delta_tier4_minus_fp16_bpc": "arm_minus_comparator",
        "delta_kv4_minus_fp16_bpc": "arm_minus_comparator"}
    out["delta_direction_note"] = (
        "全部 Δ 读数都是「实验格 − 对照格」的 bpc 差；bpc 越低越好 ⇒ 正 = 量化更差。"
        "`margin_vs_best_cheaper` / `bpc_gap_vs_this` 不是 Δ 读数，是前沿差。")
    out["bands"] = {"sigma_cell": crit["sigma_cell"], "sigma_cell_source": crit["sigma_cell_source"],
                    "cross_cell_band": crit["cross_cell_band"],
                    "cross_cell_band_role": crit["cross_cell_band_role"]}
    out["code_hashes"] = {
        "run.py": sha256_file(Path(__file__)),
        "carry_quant.py": sha256_file(EXP / "carry_quant.py"),
        "config.json": sha256_file(EXP / "config.json"),
        "block_arms.py(2632)": sha256_file(P2632 / "block_arms.py"),
        "realtext_probe.py(2662)": sha256_file(P2662 / "realtext_probe.py"),
        "pathlm/depth_ar.py": sha256_file(ROOT / "pathlm/depth_ar.py")}
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({
        "equality_gate": {k: out["equality_gate"][k]["all_equal"] for k in ("fp16", "int8")},
        "int4_window": {k: {"delta": v["delta_int4_minus_fp16_bpc"],
                            "ci": v["ci95_paired_bootstrap"],
                            "verdict": v["verdict"]}
                        for k, v in int4_window.items()},
        "carry_tiers": {k: {pair: v[pair][f"delta_{pair.replace('_vs_', '_minus_')}_bpc"]
                            for pair in ("kv8_vs_fp16", "state8_vs_kv8",
                                         "tier4_vs_fp16", "kv4_vs_fp16")}
                        for k, v in carry_tiers.items()},
        "carry_int8_tier": {k: out["frontier"]["carry_int8_tier"][k] for k in
                            ("total_bytes", "bpc", "dominated_by", "best_cheaper_or_equal",
                             "margin_vs_best_cheaper", "same_bytes_points")},
        "carry_int4_tier": {k: out["frontier"]["carry_int4_tier"][k] for k in
                            ("total_bytes", "bpc", "dominated_by", "best_cheaper_or_equal",
                             "margin_vs_best_cheaper")},
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
