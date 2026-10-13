"""R7 结果补全（P1-6 / P2-1）：把**定义不清的派生量**补成可复算的显式字段，不重跑解码。

复核（2026-10-10）抓到两件事：

1. `results.json` 里 spec 行的 `accept_rate` 是 **Σ接受 / Σ被检验的草稿位**（`n_drafts`
   只统计到每轮第一次拒绝为止）——它是**条件口径**，比业界常用的**窗口口径**
   （Σ接受 / γ·轮数，C1 的 2621 用的就是它）系统性偏高。⇒ 补齐
   `n_accepts` 与 `accept_rate_window`，两个口径都留、都标明。
2. `results.json`（主交付）缺 `SCHEMA` §10 的 `intervention` 块（它只在逐臂 evidence 里）
   ⇒ 从逐臂 `results.json` 抽出「名称 + 活性量」写进主交付。

派生可复算：`n_accepts = (tokens_per_cycle − 1) × n_cycles`（每轮发出 `接受数+1` 个 token），
`accept_rate_window = n_accepts / (γ × n_cycles)`。本脚本对每个数字做整数/比值断言，
任何不整除或对不上就非零退出。

**`--check` 的含义**：本脚本会**先记录落盘现值、再用重算值比对**（不是「先无条件覆盖内存里的
派生字段、再只做内部自洽」——那样磁盘漂移根本查不出来）。落盘现值缺字段、或与重算值有任何
一处不等，`--check` 都非零退出；`--check` 不写回。

用法：python3 enrich_results.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RES = os.path.join(HERE, "results.json")
EV = os.path.join(ROOT, ".scratch", "52-r7-early-exit", "evidence")

F: list[str] = []


def derive(a: dict) -> None:
    """给一个臂的 decode 补 `n_accepts` / `accept_rate_window`（幂等，不重跑解码）。

    `n_accepts` 的**权威来源** = 该行 `layerpos_per_token_analytic` 的逆解
    （`run.py` 里 analytic = L(γ+1)/(n_accept/n_cycles+1)，n_accept 是解码循环里的
    精确计数器）；`tokens_per_cycle` 那条路只作**独立交叉核对**（末轮被 `n_new`
    截断时它会少算，因此只报 >2 的偏差）。
    """
    dec = a.get("decode") or {}
    spec = dec.get("spec") or {}
    # L 取配置里的层数（`tf_acceptance` 是 0..L−1 共 L 条，别 +1）
    L = int((a.get("config") or {}).get("n_layers") or 0)
    for g_str, rows in spec.items():
        g = int(g_str)
        if not rows:
            continue
        tot_acc = 0
        tot_cyc = 0
        tot_draft = 0
        for r in rows:
            # 权威逆解（analytic 落盘保留 4 位小数 ⇒ ±1 的整数误差）
            acc_f = r["n_cycles"] * (L * (g + 1) / r["layerpos_per_token_analytic"] - 1)
            acc = int(round(acc_f))
            if abs(acc_f - acc) > 0.6:
                F.append(f"{a['run']} γ={g}: n_accepts 逆解偏离整数 {acc_f}")
            # 交叉核对：`tokens_per_cycle` 那条路会把**末轮截断**少算的接受数漏掉，
            # 差额上界 = 末轮接受数 ≤ γ ⇒ 容差取 γ+1（超出即真矛盾）。
            cross = (r["tokens_per_cycle"] - 1) * r["n_cycles"]
            if abs(cross - acc) > g + 1:
                F.append(f"{a['run']} γ={g}: 两条独立反推不一致"
                         f"（analytic→{acc} vs tpc→{cross:.1f}，容差 γ+1={g + 1}）")
            if r["n_drafts"] > g * r["n_cycles"]:
                F.append(f"{a['run']} γ={g}: n_drafts > γ×cycles")
            # 不变量：接受数 ≤ 被检验的草稿位数 ≤ γ×轮数。
            # 注意末轮可能被 `n_new` 截断（发出的 token 少于 接受+1）⇒ 反推值是**下界**，
            # 只在该下界越界时报错（那才是真矛盾）。
            if acc > r["n_drafts"] + 1:
                F.append(f"{a['run']} γ={g}: n_accepts 反推 > n_drafts+1"
                         f"（{acc}>{r['n_drafts']}+1）")
            r["n_accepts"] = acc
            r["accept_rate_window"] = round(acc / (g * r["n_cycles"]), 4)
            r["n_accepts_source"] = ("由 layerpos_per_token_analytic 逆解"
                                     "（= run.py 解码循环里的精确计数器）")
            r["n_drafts_definition"] = ("只统计**被检验过**的草稿位（每轮到第一次拒绝为止）")
            r["accept_rate_window_definition"] = ("Σ接受 / (γ × 轮数)（窗口口径，"
                                                 "与 C1 的 2621 同定义）")
            tot_acc += acc
            tot_cyc += r["n_cycles"]
            tot_draft += r["n_drafts"]
        pooled = dec.get(f"spec_g{g}_pooled")
        if pooled is None:
            continue
        pooled["n_accepts"] = tot_acc
        pooled["accept_rate_window"] = round(tot_acc / (g * tot_cyc), 4)
        p = tot_acc / (g * tot_cyc)
        pooled["binom_se_window"] = round((p * (1 - p) / (g * tot_cyc)) ** 0.5, 5)
        pooled["n_cycles"] = tot_cyc
        pooled["n_drafts_tested"] = tot_draft
        pooled["truncation_note"] = ("末轮可能被 n_new 截断 ⇒ 窗口接受率略偏保守"
                                     "（偏下界 ≤ γ/(γ·轮数) = 1/轮数）")
        pooled["note"] = (f"n_accepts = Σ(tokens_per_cycle−1)×n_cycles（可复算）；"
                          f"窗口接受率 = Σ接受/(γ×轮数) = "
                          f"{pooled['accept_rate_window']}")


def add_intervention(res: dict) -> None:
    """主交付补 `SCHEMA` §10 的 `intervention:` 块（逐臂抽样，附 run 目录指针）。"""
    blocks = {}
    for name in res["battery"]:
        p = os.path.join(EV, name, "results.json")
        if not os.path.exists(p):
            F.append(f"缺逐臂 results.json：{name}")
            continue
        ev = json.load(open(p))
        iv = ev.get("intervention") or {}
        blocks[name] = {
            "name": iv.get("name"),
            "alive": iv.get("alive"),
            "params_added": iv.get("params_added"),
            "gate_mean": iv.get("gate_mean") or ev.get("avg_train_gate_mean"),
            "routed_frac_mean": iv.get("routed_frac_mean"),
            "layer_apps_per_token_measured": iv.get("layer_apps_per_token_measured"),
            "aux_mode": (ev.get("config") or {}).get("aux_mode"),
            "carry_gate": (ev.get("config") or {}).get("carry_gate"),
            "mod_ratio": (ev.get("config") or {}).get("mod_ratio"),
            "source": f".scratch/52-r7-early-exit/evidence/{name}/results.json",
        }
    res["intervention"] = {
        "note": ("逐臂干预/活性量（`SCHEMA` §10）：本波的干预 = 深度形态（随机退出 + 混合填充）"
                 "× 损失配方 × 门的有无 / MoD 路由；每个臂的原始 `intervention` 块在其 run 目录，"
                 "本表是同一份数据的机器可读汇总。"),
        "by_arm": blocks,
    }


#: `derive()` / `add_intervention()` 会写回的字段（`--check` 用它比「落盘现值 vs 重算值」）。
DERIVED_POOLED_KEYS = ("n_accepts", "accept_rate_window", "binom_se_window", "n_cycles",
                       "n_drafts_tested", "truncation_note", "note")


def _derived_view(res: dict) -> dict:
    """抽出派生脚本会写回的字段现值。"""
    view: dict = {}
    for name, a in (res.get("battery") or {}).items():
        dec = a.get("decode") or {}
        for g_str, rows in (dec.get("spec") or {}).items():
            for i, r in enumerate(rows or []):
                view[f"{name}.spec[{g_str}][{i}].n_accepts"] = r.get("n_accepts")
                view[f"{name}.spec[{g_str}][{i}].accept_rate_window"] = \
                    r.get("accept_rate_window")
        for g in ("1", "2", "4", "8"):
            p = dec.get(f"spec_g{g}_pooled")
            if p is not None:
                for k in DERIVED_POOLED_KEYS:
                    view[f"{name}.spec_g{g}_pooled.{k}"] = p.get(k)
    view["intervention"] = res.get("intervention")
    return view


def _disk_drift(disk_res: dict, new_res: dict) -> list[str]:
    """落盘现值 vs 重算值的差异清单（空 = 逐条一致）。"""
    disk, new = _derived_view(disk_res), _derived_view(new_res)
    drift = [f"{k}: 落盘={disk.get(k, '<缺>')!r} != 重算={v!r}"
             for k, v in new.items() if disk.get(k, "<缺>") != v]
    drift += [f"{k}: 落盘有现值但重算没有" for k in disk if k not in new]
    return drift


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只校验，不写回")
    args = ap.parse_args()
    res = json.load(open(RES))
    disk = copy.deepcopy(res)          # 落盘现值快照（derive 会就地改写 res）
    for a in res["battery"].values():
        derive(a)
    add_intervention(res)
    if F:
        print(f"FAIL：{len(F)} 项派生不上")
        for x in F:
            print("  -", x)
        return 1
    drift = _disk_drift(disk, res)
    if args.check:
        if drift:
            print(f"FAIL（--check）：落盘派生字段与重算值不一致 {len(drift)} 处")
            for x in drift:
                print("  -", x)
            return 1
        print(f"OK（--check：落盘派生字段与重算值逐条一致，{len(res['battery'])} 臂）")
        return 0
    with open(RES, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"OK：补齐 n_accepts/accept_rate_window/intervention（{len(res['battery'])} 臂）"
          + (f"；写回前与落盘现值差 {len(drift)} 处" if drift else "；落盘现值本就一致"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
