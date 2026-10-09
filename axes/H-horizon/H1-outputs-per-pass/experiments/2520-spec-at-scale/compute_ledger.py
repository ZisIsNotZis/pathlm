"""2520-spec-at-scale 算力口径纠正（纯 CPU，不取 GPU 锁）。

从**现有原始产物**独立重算每 token 解析算力比：

    γ = 每次前向投喂的草稿位数（= max_drafts / k）
    τ = 每次前向平均接受位数 = tokens_per_forward − 1
    R = (1+γ)/(1+τ) = (1+γ) / tokens_per_forward        （native = 1.0）

旧判读把**实测 width-(1+γ)/width-1 墙钟比** `c_wall` 当成了算力比，写出
`c_wall/(tokens_per_forward)`。`c_wall` 必须与宽度配对（γ 草稿 ⇒ 宽度 1+γ：
k=1 用 `w2_vs_w1`、k=2 用 `w3_vs_w1`），且它含核回退/启动开销/外部占用
——都不是算力。可追溯实测：13M B3 `1.283`（`.scratch/13-four-experiments/evidence/mtp3/spec_speed.json`）、
100M INT3 `5.276`（`.scratch/20-spec-scale/evidence/spec_speed.json`）、
100M 重跑 `2.439`（本目录 `results.json:micro.w3_vs_w1`）。本脚本并报
`R`（解析算力）、`c_wall/(tokens_per_forward)`（旧口径）、与墙钟 speedup，以量化背离。

输入（只读）：见 SOURCES。输出：`ledger.json`。
用法：python3 compute_ledger.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

def compute_ledger(gamma: int, tok_per_fwd: float, speedup: float | None = None,
                   c_wall: float | None = None) -> dict:
    """每 token 解析算力比 R=(1+γ)/(1+τ)（τ=tok/fwd−1）；c_wall 只作对照，不进 R。

    这是 H1 算力账的**唯一定义**（run.py 与 rows_from 都调用它）。
    """
    tau = tok_per_fwd - 1.0
    R = (1 + gamma) / (1 + tau)
    out = {
        "gamma_drafts": gamma,
        "tau_accepted": round(tau, 4),
        "tokens_per_forward": round(tok_per_fwd, 4),
        "per_token_compute_ratio__analytic": round(R, 4),
        "native_reference": 1.0,
        "verdict": "loss" if R > 1.05 else ("gain" if R < 0.95 else "flat"),
        "wall_speedup": None if speedup is None else round(speedup, 4),
    }
    if c_wall is not None:
        # 旧口径：把墙钟 c_wall 当 width-(1+γ)/width-1 算力比 ⇒ c_wall/(tok/fwd)
        out["old_wallclock_ledger__c_wall_over_tokfwd"] = round(
            c_wall / (1 + tau), 4)
    return out


SOURCES = [
    ("13M B3", ".scratch/13-four-experiments/evidence/mtp3/spec_speed.json"),
    ("13M B3 (indep repro)", ".scratch/13-four-experiments/evidence/mtp3/spec_speed_independent_repro.json"),
    ("100M INT3", ".scratch/20-spec-scale/evidence/spec_speed.json"),
    ("100M INT3 (rerun)", os.path.join(HERE, "results.json")),
]


def rows_from(path: str) -> list[dict]:
    if not os.path.exists(path if os.path.isabs(path) else os.path.join(ROOT, path)):
        return []
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    with open(full) as f:
        blob = json.load(f)
    # rerun 格式（run.py 产物）：顶层直接是 checkpoint 元数据 + 各臂；包一层以统一处理
    if "checkpoint" in blob and any(k.endswith("_k1") or k.endswith("_k2")
                                    for k in blob):
        blob = {blob["checkpoint"]: blob}
    out: list[dict] = []
    for ck, entry in blob.items():
        micro = entry.get("micro") or {}
        for key, r in entry.items():
            if "_k" not in key or not isinstance(r, dict) or "tokens_per_forward" not in r:
                continue
            src, k = key.rsplit("_k", 1)
            gamma = int(k)
            tk = float(r["tokens_per_forward"])
            if "speedup_best" in r:
                speed = r["speedup_best"]
            elif "speedup" in r:
                speed = r["speedup"]
            elif "speedup_ratio_of_best" in r:
                speed = r["speedup_ratio_of_best"]
            else:
                speed = None
            # 墙钟前向比必须与宽度配对：γ 草稿 ⇒ 宽度 1+γ（k=1→w2/w1、k=2→w3/w1）
            c_wall = micro.get(f"w{1 + gamma}_vs_w1")
            led = compute_ledger(gamma, tk, speed, c_wall)
            row = {
                "checkpoint": os.path.basename(ck.rstrip("/")),
                "prompt": src,
                "k_drafts": gamma,
                "tokens_per_forward": led["tokens_per_forward"],
                "tau_accepted": led["tau_accepted"],
                "per_token_compute_ratio_R": led["per_token_compute_ratio__analytic"],
                "R_verdict": led["verdict"],
                "wall_speedup": None if speed is None else round(speed, 3),
                "c_wall_forward_ratio": c_wall,
                "old_ledger_cwall_over_tokfwd": led.get(
                    "old_wallclock_ledger__c_wall_over_tokfwd"),
            }
            if r.get("speedup_reps"):
                row["wall_speedup_reps"] = r["speedup_reps"]
            elif r.get("speedup_reps_paired"):
                row["wall_speedup_reps"] = r["speedup_reps_paired"]
            out.append(row)
    return out


def main() -> None:
    all_rows: list[dict] = []
    per_source: dict[str, list[dict]] = {}
    for label, path in SOURCES:
        rs = rows_from(path)
        if rs:
            per_source[label] = rs
            for r in rs:
                r["source_file"] = path
                r["source_label"] = label
                all_rows.append(r)

    hdr = f"{'source_label':22} {'prompt':8} {'k':>2} {'tok/fwd':>8} {'tau':>7} " \
          f"{'R=(1+k)/fwd':>12} {'verdict':>7} {'wall':>7} {'c_wall':>7} {'old=c_wall/fwd':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['source_label']:22} {r['prompt']:8} "
              f"{r['k_drafts']:>2} {r['tokens_per_forward']:>8.4f} "
              f"{r['tau_accepted']:>7.4f} {r['per_token_compute_ratio_R']:>12.4f} "
              f"{r['R_verdict']:>7} "
              f"{('%.3f' % r['wall_speedup']) if r['wall_speedup'] is not None else '—':>7} "
              f"{('%.3f' % r['c_wall_forward_ratio']) if r['c_wall_forward_ratio'] is not None else '—':>7} "
              f"{('%.3f' % r['old_ledger_cwall_over_tokfwd']) if r['old_ledger_cwall_over_tokfwd'] is not None else '—':>14}")

    # 每 token 算力比 = 每次前向算力(1+γ) / 每次前向产出 token(1+τ)
    # 解析上界：τ<γ ⇒ R>1；τ=0 ⇒ R=1+γ。断言所有行都在 [1, 1+γ]。
    for r in all_rows:
        lo, hi = 1.0, 1.0 + r["k_drafts"]
        assert lo - 1e-9 <= r["per_token_compute_ratio_R"] <= hi + 1e-9, r

    summary = {
        "formula": "R = (1+gamma)/(tokens_per_forward); native=1.0",
        "note": "R>1 = 每 token 比 native 贵；旧口径 c_wall/tokfwd 把墙钟当算力"
                 "（c_wall = w(1+γ)/w1，与宽度配对）",
        "R_by_scale": {},
    }
    for lab, rs in per_source.items():
        k2 = [r for r in rs if r["k_drafts"] == 2]
        if k2:
            summary["R_by_scale"][lab] = {
                "R_k2": sorted(set(r["per_token_compute_ratio_R"] for r in k2)),
                "wall_speedup_k2": sorted(set(r["wall_speedup"] for r in k2 if r["wall_speedup"])),
                "all_loss": all(r["R_verdict"] == "loss" for r in k2),
            }

    with open(os.path.join(HERE, "ledger.json"), "w") as f:
        json.dump({"rows": all_rows, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    print("\nsummary:", json.dumps(summary["R_by_scale"], ensure_ascii=False, indent=2))
    print("saved ledger.json")


if __name__ == "__main__":
    main()
