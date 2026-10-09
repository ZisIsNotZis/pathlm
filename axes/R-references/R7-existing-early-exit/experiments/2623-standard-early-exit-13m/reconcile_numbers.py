"""R7（WI-16）对账脚本：`analysis.md` 的每个数字都必须能在 `results.json` 里找到出处。

三道对账（`SCHEMA` §12）里的**算术对账**与**证据可回溯**由本脚本断言：
1. 结构不变量：每臂的（臂 × 策略）观测数 = 轮数 × prompt 数；spec 的
   `layerpos/token == L(γ+1)/(1+τ)`；接受率 = Σ接受/Σ草稿；CALM 的
   `layerpos/token` == 平均退出深度（门控语义的解析恒等式）；
2. 数字表：`CLAIMS` 里逐条 (来源路径, 期望值) —— 期望值取自 `analysis.md`；
3. 活性量非空：门均值 / 放行比例 / 接受率 / 路由器参数都真的存在且非退化。

用法：python3 reconcile_numbers.py   （退出码 0 = 全部对上）
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..",
                                        ".scratch", "52-r7-early-exit", "evidence"))
RES = os.path.join(HERE, "results.json")

F = []          # 失败清单


def _split_path(path: str) -> list[str]:
    """按 `.` 切字段，但**方括号内不切**（键本身含点，如 `calm_joint[0.05]`）。"""
    parts, cur, depth = [], "", 0
    for ch in path:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "." and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def get(obj, path: str):
    """路径：`.` 分隔字段，`[k]` 取列表元素或**字符串键**（如 `calm_joint[0.05]`）。"""
    cur = obj
    for part in _split_path(path):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif "[" in part:
            name, idx = part[:-1].split("[")
            cur = cur[name]
            cur = cur[int(idx)] if isinstance(cur, list) else cur[idx]
        else:
            cur = cur[part]
    return cur


def eq(label, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) \
        else (got == want)
    if not ok:
        F.append(f"{label}: got={got!r} want={want!r} (tol={tol})")
    return ok


def load_results():
    if not os.path.exists(RES):
        print(f"FAIL：缺 {RES}（先跑 run.py）")
        sys.exit(1)
    with open(RES) as f:
        return json.load(f)


def check_structure(res):
    wall = res["wall_clock_interleaved"]
    rounds, n_new = wall["rounds"], wall["n_new"]
    rows = wall["rows"]
    n_prompts = 4
    per = {}
    for row in rows:
        per.setdefault((row["arm"], row["policy"]), []).append(row)
    for key, v in per.items():
        if len(v) != rounds * n_prompts:
            F.append(f"wall[{key}] 观测数 {len(v)} != rounds×prompts "
                     f"{rounds * n_prompts}")
    for key, s in wall["summary"].items():
        if s["n"] != rounds * n_prompts:
            F.append(f"wall.summary[{key}].n={s['n']} != {rounds * n_prompts}")
        if not (s["tok_s_min"] <= s["tok_s_median"] <= s["tok_s_max"]):
            F.append(f"wall.summary[{key}] 中位不在极差内")
    for name, a in res["battery"].items():
        if "mod_eval" in a:
            eq(f"{name}.mod routed_frac==capacity", a["mod_eval"]["routed_frac_mean"],
               a["mod_eval"]["capacity_frac"], 1e-6)
            eq(f"{name}.mod 解析账", a["mod_eval"]["layer_apps_per_token_analytic"],
               round(a["mod_eval"]["capacity_frac"]
                     * len(a["mod_eval"]["routed_frac_per_layer"]), 4), 1e-6)
            continue
        L = len(a["tf_acceptance"]) + 1
        for g_key, rows_g in a["decode"].items():
            if not g_key.startswith("spec_g") or g_key.endswith("_pooled"):
                continue
            g = int(g_key[6:])
            for r in rows_g:
                tau = r["n_drafts"] / max(r["n_cycles"], 1)
                eq(f"{name}.{g_key} 解析算力", r["layerpos_per_token_analytic"],
                   round(L * (g + 1) / (tau + 1), 3), 0.02)
                if r["n_drafts"] != g * r["n_cycles"]:
                    F.append(f"{name}.{g_key} n_drafts != γ×cycles")
            pooled = a["decode"][f"spec_g{g}_pooled"]
            tot_d = sum(r["n_drafts"] for r in rows_g)
            tot_a = sum(r["accept_rate"] * r["n_drafts"] for r in rows_g)
            eq(f"{name}.{g_key}_pooled n", pooled["n_drafts"], tot_d)
            eq(f"{name}.{g_key}_pooled acc", pooled["accept_rate"],
               round(tot_a / tot_d, 4), 5e-5)
            eq(f"{name}.{g_key}_pooled se", pooled["binom_se"],
               round(((tot_a / tot_d) * (1 - tot_a / tot_d) / tot_d) ** 0.5, 5),
               5e-5)
            for r in rows_g:
                eq(f"{name}.{g_key} 流一致率（spec 必须 == 全深度贪心）",
                   r["match_full_frac"], 1.0)
        # 层×位置按**发出** token 归一；末位 token 由上一次前向的 argmax 免费给出 ⇒
        # layerpos/token = 深度 × (n_new − 1)/n_new（分母口径已在分析里写明）
        n_new = a["decode"]["n_new"]
        frac = (n_new - 1) / n_new
        for r in a["decode"]["calm_eps0.05"]:
            eq(f"{name}.calm 解析算力==平均退出深度×(n−1)/n", r["layerpos_per_token"],
               r["mean_exit_depth"] * frac, 1e-3)
        for r in a["decode"]["exit4"]:
            eq(f"{name}.exit4 算力==4×(n−1)/n", r["layerpos_per_token"],
               4.0 * frac, 1e-6)
        # CALM 标定的算术界：0 ≤ 平均退出深度 ≤ L；一致率在 [0,1]
        for eps_key, cal in a["calm_calibration"]["by_eps"].items():
            assert 0 <= cal["mean_exit_depth_holdout"] <= L, eps_key
            assert 0 <= cal["agree_with_full_holdout"] <= 1, eps_key
            assert cal["agree_with_full_holdout"] >= 1 - cal["epsilon"] - 0.02, \
                f"{name} {eps_key} 样本外一致率未达 1−ε（标定失败）"


# (标签, results.json 路径, analysis.md 里的期望值, 容差) —— 期望值逐条抄自
# `analysis.md`（发现不符 ⇒ 先改分析、再改本表；不许反过来）。
CLAIMS: list[tuple[str, str, object, float]] = [
    ("stock d8 bpc", "battery.r7-stock-s0.depth_curve.8.bpc", 1.5284, 1e-4),
    ("stock tf@4", "battery.r7-stock-s0.tf_acceptance.4.accept", 0.4441, 1e-4),
    ("c1-legacy d8 bpc", "battery.r7-c1-legacy-s0.depth_curve.8.bpc", 1.8575, 1e-4),
    ("c1-legacy tf@4", "battery.r7-c1-legacy-s0.tf_acceptance.4.accept", 0.9719, 1e-4),
    ("c1-fair d8 bpc", "battery.r7-c1-fair-s0.depth_curve.8.bpc", 2.1185, 1e-4),
    ("c1-fair tf@4", "battery.r7-c1-fair-s0.tf_acceptance.4.accept", 0.9102, 1e-4),
    ("c1-fair-l1 d8 bpc", "battery.r7-c1-fair-l1-s0.depth_curve.8.bpc", 1.9842, 1e-4),
    ("c1-fair-l1 tf@4", "battery.r7-c1-fair-l1-s0.tf_acceptance.4.accept", 0.9614, 1e-4),
    ("ls-fair d8 bpc", "battery.r7-ls-fair-s0.depth_curve.8.bpc", 1.9075, 1e-4),
    ("ls-fair tf@4", "battery.r7-ls-fair-s0.tf_acceptance.4.accept", 0.8402, 1e-4),
    ("ls-legacy d8 bpc", "battery.r7-ls-legacy-s0.depth_curve.8.bpc", 1.7857, 1e-4),
    ("ls-legacy tf@4", "battery.r7-ls-legacy-s0.tf_acceptance.4.accept", 0.9645, 1e-4),
    ("mod bpc", "battery.r7-mod-s0.mod_eval.bpc", 1.9264, 1e-4),
    ("mod 实测算力", "battery.r7-mod-s0.mod_eval.layer_apps_per_token_measured", 8.0, 1e-6),
    ("calm 原式 legacy ε=0.05 深度", "battery.r7-c1-legacy-s0.calm_calibration.calm_joint[0.05].mean_exit_depth_holdout", 3.6434, 1e-4),
    ("calm 原式 ls-legacy ε=0.05 深度", "battery.r7-ls-legacy-s0.calm_calibration.calm_joint[0.05].mean_exit_depth_holdout", 3.432, 1e-4),
    ("calm 原式 stock ε=0.05 深度", "battery.r7-stock-s0.calm_calibration.calm_joint[0.05].mean_exit_depth_holdout", 6.5895, 1e-4),
    ("spec legacy γ=4 接受率", "battery.r7-c1-legacy-s0.decode.spec_g4_pooled.accept_rate", 0.952, 1e-4),
    ("spec ls-legacy γ=4 接受率", "battery.r7-ls-legacy-s0.decode.spec_g4_pooled.accept_rate", 0.9784, 1e-4),
    ("wall stock|full 中位 tok/s", "wall_clock_interleaved.summary.r7-stock-s0|full.tok_s_median", 34.473, 1e-3),
    ("wall legacy|spec_g4 比值中位", "wall_clock_interleaved.summary.r7-c1-legacy-s0|spec_g4.ratio_median", 2.432, 1e-3),
    ("wall ls-fair|full 比值中位（门墙钟）", "wall_clock_interleaved.summary.r7-ls-fair-s0|full.ratio_median", 1.1732, 1e-4),
    ("moD 前向吞吐比值", "mod_forward_throughput.summary.r7-mod-s0.tok_per_s_median", 101685.4, 1e-1),
    ("moD 前向吞吐 stock", "mod_forward_throughput.summary.r7-stock-s0.tok_per_s_median", 337776.8, 1e-1),
]


def check_claims(res):
    for label, path, want, tol in CLAIMS:
        eq(label, get(res, path), want, tol)


def check_liveness(res):
    for name, a in res["battery"].items():
        if "mod_eval" in a:
            eq(f"{name} router_params>0", a["mod_eval"]["router_params"] > 0, True)
            continue
        gm = a.get("train_gate_mean") or []
        if a["config"]["carry_gate"] and not a["config"]["dense"]:
            if not gm or max(gm) <= 0:
                F.append(f"{name}: 门活性量全零（干预可能是死的）")
        if (not a["config"]["carry_gate"]) and not a["config"]["dense"]:
            if gm and max(gm) != 0:
                F.append(f"{name}: 无门臂的门活性量非零（槽位语义被改）")
    return


def main() -> int:
    res = load_results()
    check_structure(res)
    check_claims(res)
    check_liveness(res)
    if F:
        print(f"FAIL：{len(F)} 项对不上")
        for x in F:
            print("  -", x)
        return 1
    print(f"OK：结构不变量 + {len(CLAIMS)} 条数字表全部对上；"
          f"臂数 {len(res['battery'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
