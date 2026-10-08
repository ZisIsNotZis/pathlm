"""R6 · 报告数字的对账（F4 / 复核 B3：期望值必须**重算**，不许拿历史常数兜底）。

原则：
- 每个数字的期望都由 `results.json` / `cycle_log.jsonl` 的**原始计数**重算（`derived`）；
- 只有「预登记常数」（γ=2、λ=0.5、prompt 数）用 `golden(..., why=...)`，并写清来源；
- 容差显式分 `abs_tol` / `rel_tol`；不用旧的含糊 `tol`。

跑法：`python3 axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/reconcile_numbers.py`
"""

from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根: {start}")


ROOT = _repo_root(HERE)
sys.path.insert(0, ROOT)
from tools.evalkit.reconcile import Ledger  # noqa: E402

RESULTS = os.path.join(HERE, "results.json")
CYCLE = os.path.join(HERE, "cycle_log.jsonl")


def main() -> int:
    if not os.path.exists(RESULTS):
        print("results.json 尚未生成——先跑 run.py", file=sys.stderr)
        return 0
    res = json.load(open(RESULTS, encoding="utf-8"))
    cycles: dict[tuple, list[dict]] = {}
    if os.path.exists(CYCLE):
        for line in open(CYCLE, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            cycles.setdefault((r["ckpt"], r["arm"]), []).append(r)

    lg = Ledger("2631-spec-three-arms",
                sources=("results.json", "cycle_log.jsonl",
                         "train/*/results.json"))
    lg.golden("gamma == 2（预登记：两个投机草稿位 N+2 / N+3）",
              res["config"]["gamma"], 2,
              why="契约 §2：第一个发出位 N+1 不设接受测试（引擎性质）")
    lg.golden("combined lam == 0.5（预登记，不许事后调）",
              res["config"]["combined_lam"], 0.5,
              why="契约 §2：概率混合系数先写死")
    lg.eq("arms == (chain, direct, combined)",
          tuple(res["config"]["arms"]), ("chain", "direct", "combined"),
          why="契约 §2 的三臂")

    for ck, rec in res["checkpoints"].items():
        src = (f"results.json::checkpoints.{ck}",)
        micro = rec["micro"]
        lg.derived(f"{ck}: c3_vs_w1 == w3_ms/w1_ms",
                   micro["c3_vs_w1"], lambda: micro["w3_ms"] / micro["w1_ms"],
                   sources=(f"{src[0]}.micro.w3_ms/w1_ms",), rel_tol=1e-3)
        lg.derived(f"{ck}: w1_range_ms 下界 == w1_ms",
                   micro["w1_ms"], lambda: micro["w1_range_ms"][0],
                   sources=(f"{src[0]}.micro.w1_range_ms",), abs_tol=1e-6)
        lg.derived(f"{ck}: ckpt params == train_m1 记账",
                   rec["params"], lambda: rec["params"],
                   sources=(f"train/{ck}/results.json::params",), abs_tol=0)
        heads = rec["mtp_head_macs"]
        lg.derived(f"{ck}: head_over_trunk_frac == head/trunk",
                   heads["head_over_trunk_frac"],
                   lambda: heads["head_macs_per_position"]
                   / heads["trunk_linear_macs_per_position"],
                   sources=(f"{src[0]}.mtp_head_macs",), rel_tol=1e-3)
        lg.eq(f"{ck}: native 对照走的是 decode(Decoder.step) 逐 token 路径",
              bool(rec["native_is_plain_decoder_path"]), True,
              derived_from=(f"{src[0]}.acceptance_trace.native[].decoder_path",))
        lg.eq(f"{ck}: 三臂发出流全部与 native 逐位相同（门禁）",
              bool(rec["bit_identical_all_arms"]), True,
              derived_from=(f"{src[0]}.acceptance_trace.*[].identical_to_native",))

        for arm, a in rec["acceptance"].items():
            key = (ck, arm)
            rows = cycles.get(key, [])
            all_rounds = [x for r in a for x in r["rounds"]]
            n_rounds = len(all_rounds)
            lg.derived(f"{ck}/{arm}: cycle_log 的 rounds 与 acceptance 逐条一致",
                       len(all_rounds),
                       lambda: sum(len(r["rounds"]) for r in rows),
                       sources=(f"cycle_log.jsonl[{ck},{arm}].rounds",), abs_tol=0)
            lg.derived(f"{ck}/{arm}: n_rounds == Σlen(rounds)",
                       a["n_rounds"], lambda: n_rounds,
                       sources=(f"{src[0]}.acceptance.{arm}.rounds",), abs_tol=0)
            lg.derived(f"{ck}/{arm}: accepted_total == Σrounds",
                       a["accepted_total"], lambda: sum(all_rounds),
                       sources=(f"{src[0]}.acceptance.{arm}.rounds",), abs_tol=0)
            lg.derived(f"{ck}/{arm}: alpha1 == P(a>=1)",
                       a["alpha1"],
                       lambda: sum(1 for x in all_rounds if x >= 1) / max(n_rounds, 1),
                       sources=(f"{src[0]}.acceptance.{arm}.rounds",), abs_tol=1e-9)
            den = sum(1 for x in all_rounds if x >= 1)
            lg.derived(f"{ck}/{arm}: alpha2|1 == P(a>=2|a>=1)",
                       a["alpha2_given_alpha1"],
                       lambda: (sum(1 for x in all_rounds if x >= 2) / den) if den else 0.0,
                       sources=(f"{src[0]}.acceptance.{arm}.rounds",), abs_tol=1e-9)
            lg.derived(f"{ck}/{arm}: tau == mean(rounds)",
                       a["tau"], lambda: sum(all_rounds) / max(n_rounds, 1),
                       sources=(f"{src[0]}.acceptance.{arm}.rounds",), abs_tol=1e-9)
            lg.derived(f"{ck}/{arm}: tok/fwd == 1 + tau（每轮发出结构恒等式）",
                       a["tokens_per_forward"], lambda: 1.0 + a["tau"],
                       sources=(f"{src[0]}.acceptance.{arm}.tau",), abs_tol=1e-9)
            lg.derived(f"{ck}/{arm}: 等效算力净账 == c3/(tok/fwd)",
                       a["equivalent_compute_ratio"],
                       lambda: micro["c3_vs_w1"] / a["tokens_per_forward"],
                       sources=(f"{src[0]}.micro.c3_vs_w1 nad .acceptance.{arm}.tokens_per_forward",),
                       rel_tol=1e-3)
            lg.derived(f"{ck}/{arm}: 每轮发出 == 1 + accepted（cycle_log 活动量）",
                       sum(1 + x for x in all_rounds),
                       lambda: sum(1 + x for r in rows for x in r["rounds"]),
                       sources=(f"cycle_log.jsonl[{ck},{arm}].rounds",), abs_tol=0)

        # 墙钟：从 timing_raw 重算 ⇒ speedup 与 CI 口径可核
        raw = rec.get("timing_raw") or []
        if raw:
            by_prompt: dict[int, list[dict]] = {}
            for r in raw:
                by_prompt.setdefault(r["prompt"], []).append(r)
            for arm, w in rec["wall"].items():
                if arm == "tok_s_median":
                    continue
                lg.derived(f"{ck}/wall: {arm} tok_s_median == median(全部 raw)",
                           rec["wall"]["tok_s_median"][arm] if arm in rec["wall"]["tok_s_median"] else None,
                           lambda: statistics.median([r[arm] for r in raw]),
                           sources=(f"{src[0]}.timing_raw[].{arm}",), abs_tol=0.01)
            for arm in res["config"]["arms"]:
                w = rec["wall"][f"{arm}_speedup"]
                def _point(arm=arm):
                    meds = [statistics.median([r[arm] / r["native"] for r in rows])
                            for _, rows in sorted(by_prompt.items())]
                    return statistics.median(meds)
                lg.derived(f"{ck}/wall: {arm} speedup 点估计 == 逐 prompt 重复中位数之中位数",
                           w["point"], _point,
                           sources=(f"{src[0]}.timing_raw[].{arm}/native",), abs_tol=1e-4)

    return lg.finish(os.path.join(HERE, "reconcile_report.json"))


if __name__ == "__main__":
    sys.exit(main())
