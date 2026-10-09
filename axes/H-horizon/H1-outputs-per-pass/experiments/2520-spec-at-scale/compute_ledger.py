"""2520-spec-at-scale 算力口径纠正（纯 CPU，不取 GPU 锁）。

从**现有原始产物**独立重算每 token 解析算力比：

    γ = 每次前向投喂的草稿位数（= max_drafts / k）
    τ = 每次前向平均接受位数 = tokens_per_forward − 1
    R = (1+γ)/(1+τ) = (1+γ) / tokens_per_forward        （native = 1.0）

旧判读把**实测 width-(1+γ)/width-1 墙钟比** `c₃` 当成了算力比，写出
`c₃/(tokens_per_forward)`；`c₃` 在 13M 是 1.28（launch-bound），在 100M 是
5.28（微基准，疑被核回退/启动开销污染）——两者都不是算力。本脚本并报
`R`（解析算力）、`c₃/(tokens_per_forward)`（旧口径）、与墙钟 speedup，以量化背离。

输入（只读）：见 SOURCES。输出：`ledger.json`。
用法：python3 compute_ledger.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

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
        c3 = micro.get("w3_vs_w1")
        for key, r in entry.items():
            if "_k" not in key or not isinstance(r, dict) or "tokens_per_forward" not in r:
                continue
            src, k = key.rsplit("_k", 1)
            gamma = int(k)
            tk = float(r["tokens_per_forward"])
            R = (1 + gamma) / tk
            if "speedup_best" in r:
                speed = r["speedup_best"]
            elif "speedup" in r:
                speed = r["speedup"]
            elif "speedup_ratio_of_best" in r:
                speed = r["speedup_ratio_of_best"]
            else:
                speed = None
            row = {
                "checkpoint": os.path.basename(ck.rstrip("/")),
                "prompt": src,
                "k_drafts": gamma,
                "tokens_per_forward": round(tk, 4),
                "tau_accepted": round(tk - 1, 4),
                "per_token_compute_ratio_R": round(R, 4),
                "R_verdict": "loss" if R > 1.05 else ("gain" if R < 0.95 else "flat"),
                "wall_speedup": None if speed is None else round(speed, 3),
                "c3_wall_forward_ratio": c3,
                "old_ledger_c3_over_tokfwd": None if c3 is None else round(c3 / tk, 4),
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
          f"{'R=(1+k)/fwd':>12} {'verdict':>7} {'wall':>7} {'c3':>6} {'old=c3/fwd':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in all_rows:
        print(f"{r['source_label']:22} {r['prompt']:8} "
              f"{r['k_drafts']:>2} {r['tokens_per_forward']:>8.4f} "
              f"{r['tau_accepted']:>7.4f} {r['per_token_compute_ratio_R']:>12.4f} "
              f"{r['R_verdict']:>7} "
              f"{('%.3f' % r['wall_speedup']) if r['wall_speedup'] else '—':>7} "
              f"{('%.3f' % r['c3_wall_forward_ratio']) if r['c3_wall_forward_ratio'] else '—':>6} "
              f"{('%.3f' % r['old_ledger_c3_over_tokfwd']) if r['old_ledger_c3_over_tokfwd'] else '—':>11}")

    # 每 token 算力比 = 每次前向算力(1+γ) / 每次前向产出 token(1+τ)
    # 解析上界：τ<γ ⇒ R>1；τ=0 ⇒ R=1+γ。断言所有行都在 [1, 1+γ]。
    for r in all_rows:
        lo, hi = 1.0, 1.0 + r["k_drafts"]
        assert lo - 1e-9 <= r["per_token_compute_ratio_R"] <= hi + 1e-9, r

    summary = {
        "formula": "R = (1+gamma)/(tokens_per_forward); native=1.0",
        "note": "R>1 = 每 token 比 native 贵；旧口径 c3/tokfwd 把墙钟当算力",
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
