#!/usr/bin/env python3
"""2681 · 补齐 int8 档与显式状态「同字节」的那一格（纯窗 k=60 / k=64）。

契约：同目录 `README.md`（判据线）与 `config.json`（判据常数 + 臂 + 字节口径）。
只做四件事：过预登记门 → 取 §20 GPU 锁 → 对 13 条臂跑 fp16（纯窗臂另加）int8 块标尺两档 →
在合并网格上重算支配关系并逐条落判据，写 `results.json`。

**不改任何共享文件**：量化前向按路径只读导入 `2632/block_arms.py`，配对 bootstrap 只读导入
`2662/realtext_probe.py`。本文件是判据③④⑤的实现面，故随 `config.json` 一起进预登记绑定。
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
from datetime import datetime
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


def bound_at() -> datetime:
    rec = json.loads((EXP / "prereg_binding.json").read_text())
    return datetime.fromisoformat(rec["bound_at"])


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
        raise SystemExit(f"{arm['name']}: 权重不存在 {arm['path']}（先跑 train_arms.sh）")
    got = sha256_file(path)
    if arm.get("sha256") is not None and got != arm["sha256"]:
        raise SystemExit(f"{arm['name']}: sha256 mismatch\n want {arm['sha256']}\n got  {got}")
    if arm.get("trained_here"):
        # 本实验新训的臂：权重必须晚于预登记绑定（先绑契约、后产权重，定序可判）
        born = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        if born <= bound_at():
            raise SystemExit(f"{arm['name']}: 权重 mtime {born.isoformat()} 早于绑定 "
                             f"{bound_at().isoformat()} ⇒ 不是本实验训练产物")
    model = DepthARModel(DepthARConfig(**model_kwargs(cfg, arm)), vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": arm["path"], "sha256": got, "sha256_verified": True,
                   "size_bytes": os.path.getsize(path), "window": arm["window"],
                   "seed": arm["seed"], "kind": arm["kind"],
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
    """`points[*] = {name, total_bytes, bpc}`；被支配 ⇔ 存在不更多字节且 bpc 低出一个带的点。"""
    out = {}
    for p in points:
        dom = [q["name"] for q in points
               if q["name"] != p["name"] and q["total_bytes"] <= p["total_bytes"]
               and q["bpc"] <= p["bpc"] - band]
        out[p["name"]] = {"dominated_by": dom}
    return out


def main() -> int:
    cfg = json.loads((EXP / "config.json").read_text())
    crit, ev, axis = cfg["criterion"], cfg["eval"], cfg["cost_axis"]
    dev, T = cfg["device"], int(cfg["model"]["seq_len"])
    t0 = time.time()
    out = {"experiment": cfg["experiment"], "owner_cell": cfg["owner_cell"],
           "contract": "README.md", "budget": "39 min GPU（2 臂 × 12k 步 ≈18 min + 13 臂 × 2 档 ≈1 min）",
           "prereg_gate": gate_binding()}

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
                          "n_params": checkpoints[name]["n_params"],
                          "trained_here": bool(arm.get("trained_here")),
                          "key_visible_pairs": int(km.sum())}
            ts = time.time()
            fp16, bpw, npw = bpc_of(lambda x, m=model: masked_logits(m, x, km), val, T,
                                    ev["n_batches"], ev["batch_size"], ev["window_seed"], dev,
                                    per_window=True)
            rows[name]["bpc_fp16"] = round(float(fp16), 6)
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
                rows[name]["bytes_int8"] = (axis["int8_block32_bytes_per_key"] * int(arm["window"])
                                            + state)
                rows[name]["wall_s_int8"] = round(time.time() - ts, 1)
                bits[(name, "int8")] = (bqw, nqw)
        peaks["eval_peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)

    out["checkpoints"], out["rows"] = checkpoints, rows
    out["cost"] = {**peaks, "wall_minutes_total": round((time.time() - t0) / 60, 2),
                   "n_training_tokens": 2 * cfg["train"]["steps"] * cfg["train"]["batch_size"] * T,
                   "note": "墙钟与显存峰值只作过程指标，不进判据"}

    # 判据①：仪器复现门（复用臂的 fp16 bpc 必须逐位等于 2680 的落盘值）
    tol = crit["equality_gate_tolerance_bpc"]
    gate = {}
    for name, want in crit["equality_gate"].items():
        got = rows[name]["bpc_fp16"]
        gate[name] = {"bpc_fp16": got, "bpc_2680": want, "equal": abs(got - want) <= tol}
    out["equality_gate"] = {"arms": gate, "all_equal": all(v["equal"] for v in gate.values()),
                            "tolerance_bpc": tol,
                            "reference": crit["equality_gate_source"],
                            "consequence": "不等 ⇒ 本实验作废（仪器 / 权重 / 窗口序变了），判据线不改"}
    if not out["equality_gate"]["all_equal"]:
        (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
        raise SystemExit("判据① 复现门未过：见 results.json::equality_gate")

    # 判据② 字节账 + 判据③ 质量腿（新臂）+ 判据⑥ 两档并存
    new_arms = [a["name"] for a in cfg["arms"] if a.get("trained_here")]
    quant, bytes_ok = {}, True
    for name in new_arms:
        r = rows[name]
        ratio = (r["bytes_int8"] - r["state_bytes"]) / (r["bytes_fp16"] - r["state_bytes"])
        bytes_ok &= abs(ratio - crit["bytes_ratio_int8_vs_fp16"]) == 0.0
        d_bits = [a - b for a, b in zip(bits[(name, "int8")][0], bits[(name, "fp16")][0])]
        n_tok = sum(bits[(name, "int8")][1])
        # `bpc_of(per_window=True)` 的逐窗值是比特、bootstrap 也收/返回比特/token
        # ⇒ `sum(d_bits)/n_tok` 已是两档 bpc 之差，不得再除 LN2。
        delta = sum(d_bits) / n_tok
        expect = r["bpc_int8"] - r["bpc_fp16"]
        if abs(delta - expect) >= 2e-6:
            raise SystemExit(
                f"{name}: Δ_int8 与两档 bpc 差对不上：delta={delta:.9f} "
                f"bpc_int8−bpc_fp16={expect:.9f} 差={delta - expect:.3e}（单位或窗口配对出错）")
        ci = [round(v, 6) for v in rp.paired_cluster_bootstrap_ci(
            d_bits, bits[(name, "int8")][1], n_boot=crit["n_boot"],
            seed=crit["boot_seed"], alpha=crit["alpha"])]
        quant[name] = {"delta_int8_minus_fp16_bpc": round(delta, 6),
                       "ci95_paired_bootstrap": ci,
                       "n_windows": len(bits[(name, "int8")][1]), "n_tokens": n_tok,
                       "bytes_ratio": round(ratio, 6),
                       "bytes_ratio_expected": crit["bytes_ratio_int8_vs_fp16"],
                       "bytes_int8": r["bytes_int8"], "bytes_fp16": r["bytes_fp16"],
                       "above_cross_cell_band": bool(abs(delta) >= crit["cross_cell_band"]),
                       "verdict": ("本尺度无质量增量（int8 块标尺近无损）"
                                   if abs(delta) < crit["cross_cell_band"]
                                   else "量化在这条臂上有可排序质量代价")}
    out["quant_leg"] = {"per_arm": quant, "bytes_ratio_exact": bool(bytes_ok),
                        "band": crit["cross_cell_band"], "n_new_arms": len(quant),
                        "n_existing_arms_reproduced": len(crit["equality_gate"])}

    # 判据② 解析对账：每位置每键的值数 × 每值字节
    values_per_position_key = (int(cfg["model"]["n_layers"]) * 2 * int(cfg["model"]["d_model"]))
    out["byte_ledger"] = {
        "values_per_position_per_key": values_per_position_key,
        "fp16_bytes_per_key_analytic": values_per_position_key * 2,
        "int8_block32_bytes_per_key_analytic": round(
            values_per_position_key * (1 + 2 / int(cfg["quant"]["group"])), 4),
        "matches_cost_axis": bool(values_per_position_key * 2 == axis["fp16_bytes_per_key"]
                                  and values_per_position_key * (1 + 2 / int(cfg["quant"]["group"]))
                                  == axis["int8_block32_bytes_per_key"]),
        "win60_int8_bytes": axis["int8_block32_bytes_per_key"] * 60,
        "win64_int8_bytes": axis["int8_block32_bytes_per_key"] * 64,
        "carry_total_bytes": crit["carry_point"]["total_bytes"],
        "win60_int8_leq_carry": bool(axis["int8_block32_bytes_per_key"] * 60
                                     <= crit["carry_point"]["total_bytes"]),
        "win64_int8_leq_carry": bool(axis["int8_block32_bytes_per_key"] * 64
                                     <= crit["carry_point"]["total_bytes"])}

    # 判据④⑤⑥：合并网格上的前沿与支配
    points = []
    for name, r in rows.items():
        points.append({"name": f"{name}@fp16", "arm": name, "total_bytes": r["bytes_fp16"],
                       "bpc": r["bpc_fp16"], "prec": "fp16", "trained_here": r["trained_here"]})
        if "bpc_int8" in r:
            points.append({"name": f"{name}@int8", "arm": name, "total_bytes": r["bytes_int8"],
                           "bpc": r["bpc_int8"], "prec": "int8", "trained_here": r["trained_here"]})
    points.sort(key=lambda p: (p["total_bytes"], p["bpc"]))
    band = crit["cross_cell_band"]
    dom = domination(points, band)
    envelope = [p["name"] for p in points if not dom[p["name"]]["dominated_by"]]
    carry_b = crit["carry_point"]["total_bytes"]
    carry_bpc = crit["carry_point"]["bpc"]
    cheaper = [p for p in points if p["total_bytes"] <= carry_b
               and not (p["arm"] == "carrypl16_s0" and p["prec"] == "fp16")]
    best_cheaper = min(cheaper, key=lambda p: p["bpc"])
    dominating = [p for p in cheaper if p["bpc"] <= carry_bpc - band]
    k60 = [p for p in points if p["arm"] == "win60_s0" and p["prec"] == "int8"][0]
    k64 = [p for p in points if p["arm"] == "win64_s0" and p["prec"] == "int8"][0]
    out["frontier"] = {
        "points": points, "dominated_by": dom, "lower_envelope": envelope,
        "carry_point": {**crit["carry_point"], "prec": "fp16"},
        "domination_set": "total_bytes ≤ 393,216（含两条新臂；win64@int8 = 417,792 不在集合内）",
        "domination_threshold_bpc": crit["domination_threshold_bpc"],
        "carry_dominated": bool(dominating),
        "dominating_points": [p["name"] for p in dominating],
        "best_cheaper_or_equal": best_cheaper["name"],
        "carry_margin_vs_best_cheaper": round(best_cheaper["bpc"] - carry_bpc, 6),
        "k60_cell": {"name": k60["name"], "total_bytes": k60["total_bytes"], "bpc": k60["bpc"],
                     "gap_vs_carry": round(k60["bpc"] - carry_bpc, 6),
                     "vs_band": ("高于 carry 一个带以上（carry 更好）" if k60["bpc"] > carry_bpc + band
                                 else "与 carry 不可排序" if abs(k60["bpc"] - carry_bpc) < band
                                 else "低于 carry 一个带以上（carry 被支配）")},
        "k64_cell": {"name": k64["name"], "total_bytes": k64["total_bytes"], "bpc": k64["bpc"],
                     "gap_vs_carry": round(k64["bpc"] - carry_bpc, 6),
                     "note": "417,792 > 393,216 ⇒ 不进支配集，只作方向读数"},
        "grid": {"n_points": len(points),
                 "windows_int8_on_disk": sorted({p["arm"] for p in points if p["prec"] == "int8"}),
                 "carry_int8_on_disk": [p["name"] for p in points
                                        if p["prec"] == "int8" and "carry" in p["arm"]]},
        "grid_qualifier": "结论的适用范围 = 这张 13 臂网格（纯窗 k∈{16,16,17,18,24,32,60,64,128} × {fp16,int8} + "
                          "carry 4 臂 fp16）；不得外推到 carry 族 int8 档（未测）",
        "no_interpolation": "所有点都是在盘 checkpoint 的实测评测；两点之间未插值",
    }
    out["delta_directions"] = {"delta_int8_minus_fp16_bpc": "arm_minus_comparator"}
    out["delta_direction_note"] = (
        "本文件唯一的 Δ 读数 `delta_int8_minus_fp16_bpc` = 实验格（评测期 int8 块标尺）− 对照格"
        "（同一 checkpoint 的 fp16 评测）；bpc 越低越好 ⇒ Δ > 0 = 量化更差。"
        "`carry_margin_vs_best_cheaper` 与 `*_gap_vs_carry` 不是 Δ 读数而是前沿差（同一条字节轴上的 bpc 差）。")
    out["bands"] = {"sigma_cell": crit["sigma_cell"], "sigma_cell_source": crit["sigma_cell_source"],
                    "cross_cell_band": crit["cross_cell_band"],
                    "cross_cell_band_role": crit["cross_cell_band_role"]}
    out["code_hashes"] = {
        "run.py": sha256_file(Path(__file__)),
        "config.json": sha256_file(EXP / "config.json"),
        "train_arms.sh": sha256_file(EXP / "train_arms.sh"),
        "block_arms.py(2632)": sha256_file(P2632 / "block_arms.py"),
        "realtext_probe.py(2662)": sha256_file(P2662 / "realtext_probe.py"),
        "pathlm/depth_ar.py": sha256_file(ROOT / "pathlm/depth_ar.py"),
    }
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"equality_gate_all_equal": out["equality_gate"]["all_equal"],
                      "quant_leg": {k: {"delta": v["delta_int8_minus_fp16_bpc"],
                                        "ci": v["ci95_paired_bootstrap"],
                                        "bytes": v["bytes_int8"],
                                        "verdict": v["verdict"]} for k, v in quant.items()},
                      "frontier": {k: out["frontier"][k] for k in
                                   ("carry_dominated", "dominating_points",
                                    "carry_margin_vs_best_cheaper", "k60_cell", "k64_cell",
                                    "lower_envelope")}},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
