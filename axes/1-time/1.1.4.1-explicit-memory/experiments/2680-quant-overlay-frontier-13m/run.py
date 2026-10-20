#!/usr/bin/env python3
"""2680 · 把 KV 的 int8 块标尺档叠加到「每位置部署字节 vs clean bpc」图上。

契约：同目录 `README.md`（判据线）与 `config.json`（判据常数 + 臂 + 字节口径）。
只做三件事：过预登记门 → 取 §20 GPU 锁 → 对 11 条已训臂跑纯 fp16 与（纯窗臂另加）int8 块标尺两档，
重算支配关系，写 `results.json`。**不改任何共享文件**，量化前向按路径只读导入 `2632/block_arms.py`。
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

from pathlm.data import batch, load_enwik8_full                    # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, window_anchor_keep_mask  # noqa: E402
from tools.evalkit.retention import masked_logits                  # noqa: E402

P2632 = ROOT / "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale"
P2662 = ROOT / "axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOCK = Path("/tmp/pathlm_gpu.lock")
LOCK_WAIT_S = 20 * 60
LN2 = 0.6931471805599453
#: 复现门的舍入门限：bpc 两侧都已 round 到 1e-6 ⇒ 「相等」只在此精度下成立。
GATE_TOL_BPC = 1e-6


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
    got = sha256_file(path)
    if got != arm["sha256"]:
        raise SystemExit(f"{arm['name']}: sha256 mismatch\n want {arm['sha256']}\n got  {got}")
    model = DepthARModel(DepthARConfig(**model_kwargs(cfg, arm)), vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": arm["path"], "sha256": got, "size_bytes": os.path.getsize(path),
                   "window": arm["window"], "seed": arm["seed"], "kind": arm["kind"],
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


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


def domination(points: list[dict], band: float) -> dict:
    """`points[*] = {name, total_bytes, bpc}`；被支配 ⇔ 存在更少字节且 bpc 低出一个带。"""
    out = {}
    for p in points:
        dom = [q["name"] for q in points
               if q["name"] != p["name"] and q["total_bytes"] <= p["total_bytes"]
               and q["bpc"] <= p["bpc"] - band]
        worse_same = [q["name"] for q in points
                      if q["name"] != p["name"] and q["total_bytes"] <= p["total_bytes"]
                      and q["bpc"] < p["bpc"]]
        out[p["name"]] = {"dominated_by": dom, "cheaper_and_better_within_band": worse_same,
                          "nearest_cheaper": min(
                              (q for q in points if q["name"] != p["name"]
                               and q["total_bytes"] <= p["total_bytes"]),
                              key=lambda q: q["bpc"], default=None)}
    return out


def main() -> int:
    cfg = json.loads((EXP / "config.json").read_text())
    crit, ev, axis = cfg["criterion"], cfg["eval"], cfg["cost_axis"]
    dev, T = cfg["device"], int(cfg["model"]["seq_len"])
    t0 = time.time()
    out = {"experiment": cfg["experiment"], "owner_cell": cfg["owner_cell"],
           "contract": "README.md", "budget": cfg["budget"], "prereg_gate": gate_binding()}

    _train, val, vocab = load_enwik8_full(str(ROOT / cfg["data"]["raw_path"]),
                                          str(ROOT / cfg["data"]["cache"]))
    rows, bits, checkpoints = {}, {}, {}
    peaks = {}
    with GpuLock():
        torch.cuda.reset_peak_memory_stats()
        for arm in cfg["arms"]:
            model, checkpoints[arm["name"]] = load_arm(cfg, arm, vocab, dev)
            km = window_anchor_keep_mask(T, int(arm["window"]), 0, device=dev)
            name = arm["name"]
            rows[name] = {"window": arm["window"], "seed": arm["seed"], "kind": arm["kind"],
                          "n_params": checkpoints[name]["n_params"]}
            ts = time.time()
            fp16, bpw, npw = bpc_of(lambda x, m=model: masked_logits(m, x, km), val, T,
                                    ev["n_batches"], ev["batch_size"], ev["window_seed"], dev,
                                    per_window=True)
            rows[name]["bpc_fp16"] = round(float(fp16), 6)
            rows[name]["key_visible_pairs"] = int(km.sum())
            bits[(name, "fp16")] = (bpw, npw)
            state = axis["state_bytes"].get(name, 0)
            rows[name]["state_bytes"] = state
            rows[name]["bytes_fp16"] = axis["fp16_bytes_per_key"] * int(arm["window"]) + state
            rows[name]["wall_s_fp16"] = round(time.time() - ts, 1)
            if arm["quant"]:
                ts = time.time()
                q = lambda x, m=model, k=km: ba.masked_kv_quant_logits(  # noqa: E731
                    m, x, k, cfg["quant"]["mode"], cfg["quant"]["scheme"],
                    int(cfg["quant"]["group"]))[0]
                bq, bqw, nqw = bpc_of(q, val, T, ev["n_batches"], ev["batch_size"],
                                      ev["window_seed"], dev, per_window=True)
                rows[name]["bpc_int8"] = round(float(bq), 6)
                rows[name]["bytes_int8"] = axis["int8_block32_bytes_per_key"] * int(arm["window"]) + state
                rows[name]["wall_s_int8"] = round(time.time() - ts, 1)
                bits[(name, "int8")] = (bqw, nqw)
        peaks["eval_peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    out["checkpoints"], out["rows"] = checkpoints, rows
    out["cost"] = {**peaks, "wall_minutes_total": round((time.time() - t0) / 60, 2),
                   "n_training_tokens": 0, "note": "墙钟与显存峰值只作过程指标，不进判据"}

    # 判据①：复现门（1e-6 下相等：两侧 bpc 都已 round 到 1e-6，不是全精度逐位）
    gate = {}
    for name, want in crit["equality_gate"].items():
        got = rows[name]["bpc_fp16"]
        gate[name] = {"bpc_fp16": got, "bpc_2639": want, "equal": abs(got - want) <= GATE_TOL_BPC}
    out["equality_gate"] = {"arms": gate, "all_equal": all(v["equal"] for v in gate.values()),
                            "tolerance_bpc": GATE_TOL_BPC,
                            "consequence": "不等 ⇒ 本实验作废（仪器/权重不一致），判据线不改"}
    if not out["equality_gate"]["all_equal"]:
        (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        raise SystemExit("判据① 复现门未过：见 results.json::equality_gate")

    # 判据② 字节账 + 判据③ 质量腿
    quant, bytes_ok = {}, True
    for name, r in rows.items():
        if "bpc_int8" not in r:
            continue
        ratio = (r["bytes_int8"] - r["state_bytes"]) / (r["bytes_fp16"] - r["state_bytes"])
        bytes_ok &= abs(ratio - crit["bytes_ratio_int8_vs_fp16"]) == 0.0
        d_bits = [a - b for a, b in zip(bits[(name, "int8")][0], bits[(name, "fp16")][0])]
        n_tok = sum(bits[(name, "int8")][1])
        # `bpc_of(per_window=True)` 的逐窗值是**比特**、`paired_cluster_bootstrap_ci` 也收/返回
        # 比特/token ⇒ `sum(d_bits)/n_tok` 已是两档 bpc 之差，**不得再除 LN2**（会放大 1.4427 倍）。
        # 与两档聚合 bpc 的差做 fail-loud 对账：对不上即单位或窗口配对出错。
        delta = sum(d_bits) / n_tok
        expect = r["bpc_int8"] - r["bpc_fp16"]
        if abs(delta - expect) >= 2e-6:
            raise SystemExit(
                f"{name}: Δ_int8 与两档 bpc 差对不上：delta={delta:.9f} "
                f"bpc_int8−bpc_fp16={expect:.9f} 差={delta - expect:.3e}"
                "（逐窗比特/token 已是 bpc；再除一次 LN2 会把 Δ 放大 1.4427 倍）")
        ci = [round(v, 6) for v in rp.paired_cluster_bootstrap_ci(
            d_bits, bits[(name, "int8")][1], n_boot=crit["n_boot"],
            seed=crit["boot_seed"], alpha=crit["alpha"])]
        quant[name] = {"delta_int8_minus_fp16_bpc": round(delta, 6),
                       "ci95_paired_bootstrap": ci,
                       "n_windows": len(bits[(name, "int8")][1]), "n_tokens": n_tok,
                       "bytes_ratio": round(ratio, 6), "bytes_ratio_expected":
                       crit["bytes_ratio_int8_vs_fp16"],
                       "above_cross_cell_band": bool(abs(delta) >= crit["cross_cell_band"]),
                       "verdict": ("本尺度无质量增量（int8 块标尺近无损）"
                                   if abs(delta) < crit["cross_cell_band"]
                                   else "量化在纯窗族有可排序质量代价")}
    out["quant_leg"] = {"per_arm": quant, "bytes_ratio_exact": bool(bytes_ok),
                        "band": crit["cross_cell_band"],
                        "n_with_int8": len(quant),
                        "n_seeds_checked": 2 if ("win16_s0" in quant and "win16_s1" in quant) else 1}

    # 判据④⑤：前沿与支配
    points = []
    for name, r in rows.items():
        points.append({"name": f"{name}@fp16", "arm": name, "bytes": r["bytes_fp16"],
                       "total_bytes": r["bytes_fp16"], "bpc": r["bpc_fp16"], "prec": "fp16"})
        if "bpc_int8" in r:
            points.append({"name": f"{name}@int8", "arm": name, "bytes": r["bytes_int8"],
                           "total_bytes": r["bytes_int8"], "bpc": r["bpc_int8"], "prec": "int8"})
    points.sort(key=lambda p: (p["total_bytes"], p["bpc"]))
    dom = domination(points, crit["cross_cell_band"])
    envelope = [p["name"] for p in points if not dom[p["name"]]["dominated_by"]]
    carry = [p for p in points if p["arm"] == "carrypl16_s0"][0]
    cheaper = [p for p in points if p["total_bytes"] <= carry["total_bytes"]
               and p["name"] != carry["name"]]
    best_cheaper = min(cheaper, key=lambda p: p["bpc"]) if cheaper else None
    same_bytes = [p for p in points if p["total_bytes"] == carry["total_bytes"]
                  and p["name"] != carry["name"]]
    out["frontier"] = {
        "points": points, "dominated_by": dom, "lower_envelope": envelope,
        "carry_point": carry,
        "carry_dominated": bool(dom[carry["name"]]["dominated_by"]),
        "best_cheaper_or_equal": best_cheaper,
        "carry_margin_vs_best_cheaper": (round(best_cheaper["bpc"] - carry["bpc"], 6)
                                         if best_cheaper else None),
        "same_total_bytes_points": [p["name"] for p in same_bytes],
        "grid_gap": {
            "decisive_missing_point": "int8 下 k≈60（6528×60 = 391,680 B，与 carry 的 393,216 B 几乎同字节）",
            "on_disk": any(p["prec"] == "int8" and 300_000 <= p["total_bytes"] <= 400_000
                           for p in points),
            "note": "禁止在两点之间插值造点；缺这一格就只能报「本网格上不被支配」"}}
    out["delta_directions"] = {"delta_int8_minus_fp16_bpc": "arm_minus_comparator"}
    out["delta_direction_note"] = (
        "本文件唯一的 Δ 读数 `delta_int8_minus_fp16_bpc` = 实验格（评测期 int8 块标尺）− 对照格"
        "（同一 checkpoint 的 fp16 评测）；bpc 越低越好 ⇒ Δ > 0 = 量化更差。"
        "`carry_margin_vs_best_cheaper` 不是 Δ 读数而是前沿差（同一条字节轴上的 bpc 差）。")
    out["bands"] = {"sigma_cell": crit["sigma_cell"], "sigma_cell_source": crit["sigma_cell_source"],
                    "cross_cell_band": crit["cross_cell_band"],
                    "cross_cell_band_role": crit["cross_cell_band_role"]}
    out["code_hashes"] = {
        "run.py": sha256_file(Path(__file__)),
        "config.json": sha256_file(EXP / "config.json"),
        "block_arms.py(2632)": sha256_file(P2632 / "block_arms.py"),
        "realtext_probe.py(2662)": sha256_file(P2662 / "realtext_probe.py"),
        "pathlm/depth_ar.py": sha256_file(ROOT / "pathlm/depth_ar.py"),
        "tools/evalkit/kv_quant.py": sha256_file(ROOT / "tools/evalkit/kv_quant.py"),
    }
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({"equality_gate_all_equal": out["equality_gate"]["all_equal"],
                      "quant_leg": {k: {"delta": v["delta_int8_minus_fp16_bpc"],
                                        "ci": v["ci95_paired_bootstrap"],
                                        "verdict": v["verdict"]} for k, v in quant.items()},
                      "frontier": {k: out["frontier"][k] for k in
                                   ("carry_dominated", "carry_margin_vs_best_cheaper",
                                    "same_total_bytes_points", "lower_envelope", "grid_gap")}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
