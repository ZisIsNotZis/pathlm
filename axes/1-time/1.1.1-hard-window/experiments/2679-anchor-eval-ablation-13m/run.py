#!/usr/bin/env python3
"""2679 · 评测期锚点消融：同一个 checkpoint，只改评测期掩码，0 训练 token。

契约：同目录 `README.md`（判据线）与 `config.json`（判据常数 + 臂 + 格子）。
本脚本只做三件事：过预登记门 → 取 §20 GPU 锁 → 对 4 个已训 checkpoint × 2 个评测掩码
共 8 格算 clean bpc，给配对 bootstrap CI，写 `results.json`。

**不改任何共享文件**：只读 `pathlm/**` 与 `tools/evalkit/**`，并以只读方式按路径导入
`../2662-hard-window-anchor-13m/realtext_probe.py`（`win16_bpc` / `paired_cluster_bootstrap_ci`
一字不改）。判据的**应用**逻辑（`decide_attribution`）就在本文件里 ⇒ 它必须一起被
`prereg_bind.py --contract README.md config.json run.py` 绑住。
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

from pathlm.data import batch, load_enwik8_full                    # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,           # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import masked_logits                   # noqa: E402

P2662 = ROOT / "axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m"
LOCK = Path("/tmp/pathlm_gpu.lock")
LOCK_WAIT_S = 20 * 60
#: 复现门的舍入门限：bpc 两侧都已 round 到 1e-6 ⇒ 「相等」只在此精度下成立。
GATE_TOL_BPC = 1e-6


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rp = _load_module(P2662 / "realtext_probe.py", "rp2662")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GpuLock:
    """`SCHEMA.md` §20 互斥锁：`mkdir` 抢锁 + `holder_pid` + 死进程回收 + 等锁上限。"""

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
                    print(f"reclaim: holder_pid {pid} 已死", flush=True)
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
            print("lock released", flush=True)
        return False


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gate_binding() -> dict:
    """预登记门：契约未绑定 / 绑定后被改过 ⇒ 拒绝启动（非零退出）。"""
    res = subprocess.run([sys.executable, "tools/prereg_bind.py", str(EXP), "--require"],
                         cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + res.stderr)
        raise SystemExit("预登记绑定门未过：先跑 python3 tools/prereg_bind.py <本目录>")
    return {"require_rc": 0, "stdout": res.stdout.strip()}


def model_kwargs(mcfg: dict, arm: dict) -> dict:
    kw = dict(mcfg)
    kw.pop("note", None)
    kw["train_window"] = int(arm["trained_window"])
    kw["train_anchors"] = int(arm["trained_anchors"])
    return kw


def load_arm(arm: dict, mcfg: dict, vocab: int, dev: str):
    path = ROOT / arm["path"]
    got = sha256_file(path)
    if got != arm["sha256"]:
        raise SystemExit(f"{arm['name']}: checkpoint sha256 mismatch\n want {arm['sha256']}\n got  {got}")
    model = DepthARModel(DepthARConfig(**model_kwargs(mcfg, arm)), vocab)
    model.load_state_dict(torch.load(path, map_location=dev))
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": arm["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "provenance": arm["provenance"],
                   "trained_window": arm["trained_window"],
                   "trained_anchors": arm["trained_anchors"], "seed": arm["seed"],
                   "n_params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """必须不变项：只改后半段 token ⇒ 前半段 logits 逐位不变（判据④）。"""
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    a = masked_logits(model, x, keep_mask).float()
    b = masked_logits(model, x2, keep_mask).float()
    d = float((a[:, :half] - b[:, :half]).abs().max())
    return {"perturbed_from": half, "max_abs_logit_diff_prefix": round(d, 9),
            "causal_no_future_leak": d == 0.0,
            "changed_suffix_max_abs_logit_diff": round(
                float((a[:, half:] - b[:, half:]).abs().max()), 6)}


def paired_delta(exp_bits, exp_n, cmp_bits, cmp_n, crit: dict,
                 bpc_exp: float, bpc_cmp: float) -> dict:
    """`Δ = 实验格 − 对照格`（arm_minus_comparator），bpc 越低越好 ⇒ Δ>0 = 实验格更差。

    单位：`2662/realtext_probe.py::win16_bpc` 的逐窗值是**比特**（`ce.sum(1)/LN2`），
    `paired_cluster_bootstrap_ci` 的入参与返回值也是**比特/token** ⇒ `sum(diff)/n_tok`
    已等于两格 bpc 之差，**不得再除 LN2**（除一次就把 Δ 与 CI 放大 1.4427 倍）。
    `bpc_exp/bpc_cmp` 是同一批窗口的独立聚合读数（`win16_bpc` 的 nats 累加器路径）：
    Δ 必须与这两级 bpc 之差在 1e-6 的舍入误差内对得上，对不上即单位或配对出错 ⇒ fail loud。
    """
    if exp_n != cmp_n:
        raise SystemExit("配对前提不成立：两格的逐窗 token 数不同")
    diff = [a - b for a, b in zip(exp_bits, cmp_bits)]
    n_tok = sum(exp_n)
    delta = sum(diff) / n_tok
    expect = bpc_exp - bpc_cmp
    if abs(delta - expect) >= 2e-6:
        raise SystemExit(
            f"Δ 与两级 bpc 差对不上：delta={delta:.9f} bpc_exp−bpc_cmp={expect:.9f} "
            f"差={delta - expect:.3e}（逐窗比特/token 已是 bpc；再除一次 LN2 会把 Δ 放大 1.4427 倍）")
    ci = [round(v, 6) for v in rp.paired_cluster_bootstrap_ci(
        diff, exp_n, n_boot=crit["n_boot"], seed=crit["boot_seed"], alpha=crit["alpha"])]
    return {"delta_bpc": round(delta, 6), "ci95_paired_bootstrap": ci,
            "n_windows": len(exp_n), "n_tokens": n_tok,
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "above_sigma_cell": bool(abs(delta) >= crit["sigma_cell"]),
            "above_cross_cell_band": bool(abs(delta) >= crit["cross_cell_band"])}


def decide_attribution(per_seed: dict, crit: dict) -> dict:
    """判据②的**应用**：两 seed 同向 > 0、CI 下界 > 0、|Δ| ≥ 本格带 ⇒ H_A 分支；两 seed 都落带内 ⇒ H_B。

    H_A 分支被机械触发时，机器判词仍取 `undecidable_at_this_n`（`fired_branch` 记触发的是哪一支）：
    反向对照证明评测期换掩码是分布外操作，该 Δ 不具区分力（见 `results.json::attribution.verdict_caveat`）。
    """
    sigma = crit["sigma_cell"]
    h_a = all(v["delta_bpc"] > 0 and v["ci95_paired_bootstrap"][0] > 0
              and abs(v["delta_bpc"]) >= sigma for v in per_seed.values())
    h_b = all(abs(v["delta_bpc"]) < sigma for v in per_seed.values())
    if len(per_seed) < 2:
        verdict, label = "insufficient_seeds", "seed 数 < 2 ⇒ 不能下归因判词"
        fired, caveat = None, None
    elif h_a:
        # 机械分支被触发，但反向对照（纯窗臂 + 评测期锚点 = +1.30 / +1.06 bpc）证明**评测期换掩码
        # 本身就是分布外操作** ⇒ 同一个 Δ 同时兼容「推理期可见性承载」与「分布外失配」两种解释，
        # 不具区分力。机器判词因此不升到 H_A，如实报「本 n 下不可判」。
        verdict = "undecidable_at_this_n"
        label = "本 n 下不可判：判据②的机械分支（H_A）被触发，但该 Δ 不具区分力"
        fired = "H_A_inference_visibility"
        caveat = ("反向对照（纯窗臂评测期白加 8 个锚点）掉 +1.30 / +1.06 bpc ⇒ 评测期换掩码是"
                  "分布外操作；Δ_abl 同时兼容「锚点可见性承载质量收益」与「分布外失配」两种解释"
                  "⇒ 不能把它读成 H_A 成立，两半拆不开")
    elif h_b:
        verdict, label = "H_B_training_side", "质量收益属训练侧（评测期去掉锚点不改质量）"
        fired, caveat = "H_B_training_side", None
    else:
        verdict, label = "undecidable_at_this_n", "本 n 下不可判（异向 / CI 含 0 / 骑带）"
        fired, caveat = None, None
    out = {"verdict": verdict, "label": label, "fired_branch": fired,
           "rule": crit["decision"]["ablation_at_or_above_band"] + " / "
                   + crit["decision"]["ablation_inside_band"]}
    if caveat:
        out["verdict_caveat"] = caveat
    return out


def main() -> int:
    cfg = json.loads((EXP / "config.json").read_text())
    crit, ev = cfg["criterion"], cfg["eval"]
    dev = cfg["device"]
    t0 = time.time()
    out = {"experiment": cfg["experiment"], "owner_cell": cfg["owner_cell"],
           "contract": "README.md", "budget": cfg["budget"],
           "prereg_gate": gate_binding()}

    _train, val, vocab = load_enwik8_full(str(ROOT / cfg["data"]["raw_path"]),
                                          str(ROOT / cfg["data"]["cache"]))
    mcfg = {k: v for k, v in cfg["model"].items() if k != "note"}

    with GpuLock():
        checkpoints, models = {}, {}
        for arm in cfg["arms"]:
            models[arm["name"]], checkpoints[arm["name"]] = load_arm(arm, mcfg, vocab, dev)

        cells, bits = {}, {}
        peaks = {}
        torch.cuda.reset_peak_memory_stats()
        for cell in cfg["cells"]:
            name = cell["name"]
            arm = next(a for a in cfg["arms"] if a["name"] == cell["arm"])
            model = models[arm["name"]]
            km = window_anchor_keep_mask(int(cfg["model"]["seq_len"]), int(cell["eval_window"]),
                                         int(cell["eval_anchors"]), device="cpu")
            ts = time.time()
            bpc, per_win, per_win_n = rp.win16_bpc(
                model, val, km, n_batches=ev["n_batches"], batch_size=ev["batch_size"],
                window_seed=ev["window_seed"], device=dev, per_window=True)
            bpc6 = round(float(bpc), 6)
            cells[name] = {"arm": cell["arm"], "eval_window": cell["eval_window"],
                           "eval_anchors": cell["eval_anchors"], "bpc": bpc6,
                           "visible_pairs": int(km.sum()),
                           "visible_pairs_expected": crit["visible_pairs_expected"][
                               "anchors8" if cell["eval_anchors"] else "anchors0"],
                           "n_windows": len(per_win_n), "n_tokens": int(sum(per_win_n)),
                           "wall_s": round(time.time() - ts, 1)}
            bits[name] = (per_win, per_win_n, arm["name"])
            peaks["eval_peak_gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        out["checkpoints"], out["cells"] = checkpoints, cells
        out["cost"] = {**peaks, "wall_minutes_total": round((time.time() - t0) / 60, 2),
                       "n_training_tokens": 0,
                       "note": "墙钟与显存峰值只作过程指标，不进判据"}

        # 判据①：复现门。两侧都已 round 到 1e-6 ⇒ 本门判的是「1e-6 下相等」，不是全精度逐位相等。
        gate = {}
        for name, want in crit["equality_gate"].items():
            got = cells[name]["bpc"]
            gate[name] = {"bpc": got, "bpc_2662": want,
                          "equal": abs(got - want) <= GATE_TOL_BPC}
        out["equality_gate"] = {"cells": gate, "all_equal": all(v["equal"] for v in gate.values()),
                                "tolerance_bpc": GATE_TOL_BPC,
                                "consequence": "不等 ⇒ 本实验作废（仪器/权重不一致），判据线不改"}
        if not out["equality_gate"]["all_equal"]:
            (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
            raise SystemExit("判据① 复现门未过：见 results.json::equality_gate")

        # 判据②：主判（消融），判据③：反向对照
        ablation, reverse = {}, {}
        for seed in ("0", "1"):
            e0 = f"a_s{seed}_eval_anchors0"
            e8 = f"a_s{seed}_eval_anchors8"
            ablation[seed] = paired_delta(bits[e0][0], bits[e0][1], bits[e8][0], bits[e8][1], crit,
                                          cells[e0]["bpc"], cells[e8]["bpc"])
            ablation[seed].update({"bpc_ablated": cells[e0]["bpc"],
                                   "bpc_with_anchors": cells[e8]["bpc"]})
            w0 = f"w_s{seed}_eval_anchors0"
            w8 = f"w_s{seed}_eval_anchors8"
            reverse[seed] = paired_delta(bits[w8][0], bits[w8][1], bits[w0][0], bits[w0][1], crit,
                                         cells[w8]["bpc"], cells[w0]["bpc"])
            reverse[seed].update({"bpc_plain_plus_anchors": cells[w8]["bpc"],
                                  "bpc_plain": cells[w0]["bpc"]})
        out["attribution"] = {"ablation": ablation, "reverse_control": reverse,
                              **decide_attribution(ablation, crit)}
        out["attribution"]["gain_to_explain_bpc"] = {
            "0": round(cells["a_s0_eval_anchors8"]["bpc"] - cells["w_s0_eval_anchors0"]["bpc"], 6),
            "1": round(cells["a_s1_eval_anchors8"]["bpc"] - cells["w_s1_eval_anchors0"]["bpc"], 6),
            "source": "2662::quality_leg.price[*].price_bpc（本实验复现的同一对数）"}
        out["attribution"]["gap_window16_to_full_bpc"] = crit["cell_band_must_close_the_gap"]

        # 判据④：必须不变项
        gen = torch.Generator().manual_seed(11)
        x, _ = batch(val, 2, int(cfg["model"]["seq_len"]), gen)
        x = x.to(dev)
        self_checks = {}
        for arm_name in ("win16a_s0", "win16_s0"):
            ks = {}
            for tag, anch in (("anchors8", 8), ("anchors0", 0)):
                km = window_anchor_keep_mask(int(cfg["model"]["seq_len"]), 16, anch, device=dev)
                ks[tag] = {"visible_pairs": int(km.sum()),
                           "visible_pairs_expected": crit["visible_pairs_expected"][tag],
                           "nondegenerate": bool(int(km.sum()) < cfg["model"]["seq_len"] ** 2),
                           "causality": causality_probe(models[arm_name], x, km)}
            self_checks[arm_name] = ks
        out["selfchecks"] = {
            "arms": self_checks,
            "all_causal": all(c["causality"]["causal_no_future_leak"]
                              for a in self_checks.values() for c in a.values()),
            "all_visible_pairs_ok": all(c["visible_pairs"] == c["visible_pairs_expected"]
                                        for a in self_checks.values() for c in a.values()),
        }

    out["bands"] = {"sigma_cell": crit["sigma_cell"],
                    "sigma_cell_source": crit["sigma_cell_source"],
                    "cross_cell_band": crit["cross_cell_band"],
                    "cross_cell_band_role": crit["cross_cell_band_role"],
                    "n_boot": crit["n_boot"], "boot_seed": crit["boot_seed"]}
    out["delta_directions"] = {"delta_bpc": "arm_minus_comparator"}
    out["delta_direction_note"] = (
        "本文件所有 delta_bpc 一律 = 实验格 − 对照格（arm_minus_comparator）。"
        "ablation 里实验格 = 同一 checkpoint 的**评测期去锚点**格、对照格 = 带锚点格；"
        "reverse_control 里实验格 = 纯窗臂的**评测期加锚点**格、对照格 = 纯窗臂自有掩码格。"
        "bpc 越低越好 ⇒ Δ > 0 = 实验格更差。")
    out["code_hashes"] = {
        "run.py": sha256_file(Path(__file__)),
        "config.json": sha256_file(EXP / "config.json"),
        "realtext_probe.py(2662)": sha256_file(P2662 / "realtext_probe.py"),
        "pathlm/depth_ar.py": sha256_file(ROOT / "pathlm/depth_ar.py"),
        "tools/evalkit/retention.py": sha256_file(ROOT / "tools/evalkit/retention.py"),
        "pathlm/data.py": sha256_file(ROOT / "pathlm/data.py"),
    }
    (EXP / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("equality_gate", "attribution", "selfchecks")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
