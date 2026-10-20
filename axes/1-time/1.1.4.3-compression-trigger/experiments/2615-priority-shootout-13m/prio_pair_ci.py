#!/usr/bin/env python3
"""2615-priority-shootout · 保留腿（召回）**配对 bootstrap** 分析（0 训练 token，纯再分析）。

背景（`harness/research-schema.md` §12.1 的 C1 细分 + ticket 62 §10.1 第 11 行）：
本项 `falsification` 声明「长程召回等率类指标走**配对 bootstrap / Wilson 95% CI 不含 0**」。
原 `results.json` 只存聚合计数、`retention_log.jsonl` 的 `needle_batch` 只含 full/lru 两列
与 `n_so_far` 累计 ⇒ **逐针配对不可复原**（见 `PAIR-CI-NOTE.md` / `recompute_pair_ci.py`）。

本脚本吃**重跑落盘的逐针记录** `prio_needle_raw.json`（由隔离副本的 `run.py` 加法落盘，
判据线/阈值/掩码/臂集/批数/窗口序/统计口径一字未改），做三件事：

  1. **保真校验（先做）**：逐规则 × N 的聚合召回与仓库 `results.json::criteria.per_N`
     逐位比对；任一不一致 ⇒ 判定「重跑不保真 ⇒ 判据仍不可判」，停止算 CI。
  2. **配对 bootstrap（预登记口径）**：对每个 N、每条规则，按**同索引配对**算
     `recall(rule) − recall(distance)` 及其 95% CI（重抽单位 = 针，percentile）。
     分别报**全集（4096 针）**与**硬子集**（基线 `control_recall=0` 的那批）。
  3. 写出 `PAIR-CI-RESULT.md`（结果 + 口径）。

用法：python3 prio_pair_ci.py            # 打印表并写 PAIR-CI-RESULT.md
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 20000
SEED = 20261014
N_BOOT_CHUNK = 2000
PRIMARY_K = 64
NS = (8, 32, 64)


def paired_bootstrap(delta: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = SEED) -> dict:
    """配对 bootstrap：重抽单位 = 针；返回差值点估计与 95% percentile CI。"""
    n = int(delta.size)
    d = float(delta.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(N_BOOT_CHUNK, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        boots[done:done + m] = delta[idx].mean(axis=1)
        done += m
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "diff": d, "lo": float(lo), "hi": float(hi),
            "ci_excludes_0": bool(lo > 0 or hi < 0)}


def hits(records: list[dict]) -> np.ndarray:
    return np.array([r["hit"] for r in records], dtype=np.int64)


def main() -> None:
    raw = json.load(open(os.path.join(HERE, "prio_needle_raw.json"), encoding="utf-8"))
    res = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))

    # ---- 逐针向量（同索引配对：所有臂与 control 在同一批针上按同一顺序 extend）----
    dist = {N: hits(raw[f"distance:N={N}|planted"]) for N in NS}
    rule = {(r, N): hits(raw[f"{r}:N={N}|planted" if r == "content"
                             else f"lru:k={PRIMARY_K}:N={N}|planted"])
            for r in ("lru", "content") for N in NS}
    full = hits(raw["full|planted"])
    control = hits(raw["control|control"])

    # ---- 保真校验：逐规则 × N 的聚合召回 vs results.json::criteria.per_N ----
    by_n = {int(e["N"]): e for e in res["criteria"]["per_N"]}
    fid_rows, all_match = [], True
    for N in NS:
        for name, vec in (("distance", dist[N]), ("lru", rule[("lru", N)]),
                          ("content", rule[("content", N)])):
            got = round(float(vec.mean()), 6)
            want = round(float(by_n[N]["recall"][name]), 6)
            ok = abs(got - want) < 1e-9
            all_match &= ok
            fid_rows.append({"N": N, "rule": name, "rerun": got, "recorded": want,
                             "match": ok})
    got_full = round(float(full.mean()), 6)
    want_full = round(float(res["criteria"]["baseline_recall_full"]), 6)
    full_ok = abs(got_full - want_full) < 1e-9
    all_match &= full_ok
    # 硬子集定义一致性（control hit == 0）
    hard_idx = np.array([i for i, h in enumerate(control) if h == 0], dtype=np.int64)
    n_hard_recorded = int(res["hard_needle_subset"]["n_hard"])
    hard_ok = int(hard_idx.size) == n_hard_recorded
    all_match &= hard_ok

    # ---- 配对 bootstrap：rule − distance，全集 / 硬子集 ----
    rows_full, rows_hard = [], []
    for N in NS:
        for name in ("lru", "content"):
            r, d = rule[(name, N)], dist[N]
            b = paired_bootstrap(r.astype(np.float64) - d.astype(np.float64))
            rows_full.append({"N": N, "rule": name, **b})
            bh = paired_bootstrap(r[hard_idx].astype(np.float64)
                                  - d[hard_idx].astype(np.float64))
            rows_hard.append({"N": N, "rule": name, **bh})

    # ---- 写出报告 ----
    def pp(x: float) -> float:
        return round(100.0 * x, 4)

    def table(rows: list[dict], title: str) -> list[str]:
        out = [f"### {title}", "",
               "| N | 规则 | 差（pp，正=规则更好） | 配对 bootstrap 95% CI（pp） | 含 0 |",
               "|---|---|---|---|---|"]
        for r in rows:
            out.append(f"| {r['N']} | `{r['rule']}` | {pp(r['diff']):+.4f} | "
                       f"[{pp(r['lo']):+.4f}, {pp(r['hi']):+.4f}] | "
                       f"{'是' if not r['ci_excludes_0'] else '否'} |")
        out.append("")
        return out

    verdict_judgeable = bool(all_match)
    md = []
    md += ["# 2615 · 保留腿配对 bootstrap（预登记口径）结果", "",
           "> 本文件由 `prio_pair_ci.py` 从 `prio_needle_raw.json` + `results.json` 生成；"
           "逐针记录来自隔离副本的 2615 评测腿重跑（0 训练 token）。", "",
           "## 0. 人话摘要", ""]
    if not all_match:
        md += ["**重跑不保真**：逐规则 × N 的聚合召回与仓库 `results.json` 不一致 ⇒ "
               "重跑没有复现原评测腿 ⇒ **配对判据仍不可判**（停止算 CI）。见 §1 的对照表。", ""]
    else:
        exc = [r for r in rows_full + rows_hard if r["ci_excludes_0"]]
        md += [f"重跑**保真**（逐规则 × N 聚合召回与仓库 `results.json` 逐位一致，"
               f"硬子集 n={n_hard_recorded} 一致）⇒ 预登记的**配对 bootstrap 判据可判**。"
               f"在全集（4096 针）与硬子集（{n_hard_recorded} 针）上，`rule − distance` "
               f"的召回差及其 95% 配对 CI 如下："
               f"共 {len(rows_full) + len(rows_hard)} 个（N×规则×子集）区间里 "
               f"**{len(exc)} 个不含 0**。", ""]
    md += ["## 1. 保真校验（逐规则 × N 聚合召回：重跑 vs 仓库）", "",
           "| N | 规则 | 重跑召回 | 仓库记录 | 一致 |", "|---|---|---|---|---|"]
    for r in fid_rows:
        md.append(f"| {r['N']} | `{r['rule']}` | {r['rerun']:.6f} | {r['recorded']:.6f} | "
                  f"{'✅' if r['match'] else '❌'} |")
    md += ["", f"全保留基线召回：重跑 {got_full:.6f} vs 仓库 {want_full:.6f} "
               f"（{'✅' if full_ok else '❌'}）；硬子集 n：重跑 {hard_idx.size} vs "
               f"仓库 {n_hard_recorded}（{'✅' if hard_ok else '❌'}）。", "",
           f"**总体保真：{'通过' if all_match else '不通过'}。**", "",
           "## 2. 配对 bootstrap（`rule − distance`，重抽单位 = 针，20000 次，seed=20261014）", ""]
    md += table(rows_full, f"2.1 全集（{full.size} 针）")
    md += table(rows_hard, f"2.2 硬子集（`control_recall=0`，n={n_hard_recorded}）")
    md += ["## 3. 口径与读法", "",
           "- **配对**：同一批 4096 根针、同 `q_row`、同答案；差 = `recall(rule) − "
           "recall(distance)`，逐针配对，重抽单位 = 针。",
           "- **95% CI** = bootstrap percentile；**含 0** = 与「无差异」一致（不越噪声带）。",
           "- 契约门槛是 **+10pp**（`results.json::criteria.recall_threshold_pp`）；"
           "配对 CI 回答的是「差是否可分辨于 0」，不是「是否过 10pp」。",
           "- **硬子集** = 无针控制臂答不出的针（仅诊断口径，非主判据，SCHEMA §17.10）。", ""]
    if all_match:
        md += ["## 4. 裁决", "",
               "- **判据可判**：保真校验通过 + 配对 CI 可算。",
               "- 「越噪声带」（CI 不含 0）的条目见 §2 表的「含 0 = 否」行；其余与无差异一致。", ""]
    md += ["## 5. 复现", "",
           "```", "python3 prio_pair_ci.py", "```", ""]
    open(os.path.join(HERE, "PAIR-CI-RESULT.md"), "w", encoding="utf-8").write(
        "\n".join(md) + "\n")

    # ---- stdout 摘要 ----
    print("=== 保真校验（重跑 vs 仓库）===")
    for r in fid_rows:
        flag = "OK" if r["match"] else "MISMATCH"
        print(f"  N={r['N']:>2} {r['rule']:<8} rerun={r['rerun']:.6f} "
              f"recorded={r['recorded']:.6f} {flag}")
    print(f"  full   rerun={got_full:.6f} recorded={want_full:.6f} "
          f"{'OK' if full_ok else 'MISMATCH'}")
    print(f"  hard n rerun={hard_idx.size} recorded={n_hard_recorded} "
          f"{'OK' if hard_ok else 'MISMATCH'}")
    print(f"  >>> 总体保真: {'PASS' if all_match else 'FAIL'}")
    print()
    for title, rows in ((f"全集 {full.size}", rows_full), (f"硬子集 {n_hard_recorded}", rows_hard)):
        print(f"=== 配对 bootstrap rule-distance ({title}) ===")
        for r in rows:
            print(f"  N={r['N']:>2} {r['rule']:<8} diff={pp(r['diff']):+.4f}pp "
                  f"CI=[{pp(r['lo']):+.4f},{pp(r['hi']):+.4f}]pp "
                  f"{'EXCL0' if r['ci_excludes_0'] else 'contains0'}")
    print(f"\n判据可判: {'YES' if verdict_judgeable else 'NO (重跑不保真)'}")
    print(f"写出：{os.path.join(HERE, 'PAIR-CI-RESULT.md')}")


if __name__ == "__main__":
    main()
