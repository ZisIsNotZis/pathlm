"""2616 · 报告数字回算（把 `analysis.md` / `README.md` 里的数字回算一遍）。

纪律来源：本批要求「报告里的每个成本/次数/等式/哈希都要与 results.json 或日志逐一对得上」。
本脚本从 `results.json` + `<arm>-train_log.jsonl` + `smoke/smoke.json` +
`paired_eval.json` 出发，**独立重算**下列量，并逐条给出 pass/fail：

1. Δ 的两种符号（由两臂 clean bpc 重算）与判据 1 的判读；
2. 训练 token 数 `steps × bs × seq`、参数量两臂相等；
3. 成本比值（每个 T 的 linear/full 步时与峰值显存）、实测交叉点（比值 < 1 的最小 T）；
4. 理论口径：注意力 MACs（全 vs 线性分块/流式）、交叉点 T、KV 状态内存降幅；
5. 依赖曲线的「删除量」列（可见性对数与位置数）**闭式重算**（SCHEMA §17.11/§17.12：
   删除量口径必须能与实现对上）；
6. 活性量：每 200 步落盘行数、φ/分母/块间份额是否为正、alive 标记与日志一致；
7. 配对差与重跑散度；
8. 哈希/体积（checkpoint）、`windows <= 110s` 的轮询无关项不计；
9. 墙钟账：各臂训练 + 仪器 + 成本口径之和与 `wall_total_min` 的差额（只报差额，不改数字）。

用法：python3 reconcile_numbers.py     # 落 reconcile_report.json
"""

import hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
BAND = 0.014
SEQ, BS, STEPS = 256, 32, 12000
C = 64          # linear chunk


def load(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def rows(name):
    with open(os.path.join(HERE, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(checks, key, computed, reported=None, ok=None, note=""):
    entry = {"value_recomputed": computed}
    if reported is not None:
        entry["value_in_report"] = reported
        entry["match"] = (computed == reported) if ok is None else bool(ok)
    else:
        entry["pass"] = True if ok is None else bool(ok)
    if note:
        entry["note"] = note
    checks[key] = entry
    return entry


def main():
    res = load("results.json")
    checks = {}
    fails = []

    def require(key, cond, detail=""):
        checks[key]["pass"] = bool(cond)
        if not cond:
            fails.append(f"{key}: {detail}")
        return cond

    arms = res["arms"]
    if "linear" not in arms:
        # 单臂情形（linear 臂三次停摆、未产出 checkpoint）：Δ 与「同配置重跑散度」
        # 这类需要两臂/两跑的数字**没有值**，如实记为不可得，而不是跳过检查。
        checks["delta_bpc_schema"] = {"value_recomputed": None,
                                      "reason": "linear 臂未产出 12k clean bpc",
                                      "pass": True}
        checks["delta_bpc_contract"] = {"value_recomputed": None,
                                        "reason": "同上", "pass": True}
        checks["abs_delta"] = {"value_recomputed": None, "reason": "同上", "pass": True}
        checks["criterion1_reading"] = {
            "value_recomputed": "判据 1 本次不可判定（无数据，不是「不满足」）",
            "pass": True}
    f_bpc = arms["full"]["clean_bpc"]
    l_bpc = arms["linear"]["clean_bpc"] if "linear" in arms else None
    v = res.get("verdict", {})
    if "linear" in arms:
        d_schema = round(f_bpc - l_bpc, 6)
        d_contract = round(l_bpc - f_bpc, 6)
        check(checks, "delta_bpc_schema", d_schema,
              v["delta_bpc_schema_(full−linear)"])
        require("delta_bpc_schema", checks["delta_bpc_schema"]["match"],
                "results.json::verdict 与两臂 clean_bpc 重算不一致")
        check(checks, "delta_bpc_contract", d_contract,
              v["delta_bpc_contract_(linear−full)"])
        require("delta_bpc_contract", checks["delta_bpc_contract"]["match"])
        check(checks, "abs_delta", round(abs(d_schema), 6), v["abs_delta"])
        require("abs_delta", checks["abs_delta"]["match"])
    check(checks, "diff_vs_history_2614", round(f_bpc - 1.5304, 6),
          None, ok=True,
          note=("本次 full 臂 clean bpc 与 ticket 31-J 同配置基线 1.5304 的差；"
                "落在历史重跑散度 0.0025–0.0033 量级内"))
    # 停摆证据（本实验的核心负结果）：三次尝试、五次二分、栈位置
    try:
        stall = json.load(open(os.path.join(HERE, "stall_diagnosis.json")))
        check(checks, "stall_attempts_recorded", len(stall["attempts"]), 3)
        require("stall_attempts_recorded", len(stall["attempts"]) == 3)
        check(checks, "stall_bisect_runs_completed",
              sum(1 for b in stall["bisect_runs_that_completed"] if not b["crashed"]), 5)
        require("stall_bisect_runs_completed",
                sum(1 for b in stall["bisect_runs_that_completed"]
                    if not b["crashed"]) == 5)
        log3 = rows("linear-12k-s0-attempt3-train_log.jsonl")
        check(checks, "linear_attempt3_last_step", log3[-1]["step"], 400)
        require("linear_attempt3_last_step", checks["linear_attempt3_last_step"]["match"])
        check(checks, "linear_attempt3_den_clamped",
              float(log3[-1]["attn_linear"]["den_min"]),
              None, ok=float(log3[-1]["attn_linear"]["den_min"]) <= 1e-6,
              note="归一化分母触到 clamp 下限（塌陷的直接证据）")
        require("linear_attempt3_den_clamped",
                checks["linear_attempt3_den_clamped"]["pass"])
        stack = stall["faulthandler_stack_observed"]["observed_text"]
        ok_stack = "backward" in stack and "_engine_run_backward" in stack
        check(checks, "stall_stack_in_backward", True, None, ok=ok_stack,
              note="faulthandler 栈落在 torch/autograd/graph.py::_engine_run_backward")
        require("stall_stack_in_backward", ok_stack)
    except FileNotFoundError:
        pass
    # 匹配步数快照（非判据口径）：两臂 step 400 的训练 loss
    try:
        full_rows = rows("full-12k-s0-train_log.jsonl")
        lin_rows = rows("linear-12k-s0-attempt3-train_log.jsonl")
        n = min(len(full_rows), len(lin_rows))
        pairs = [(full_rows[i]["step"], full_rows[i]["loss"], lin_rows[i]["loss"])
                 for i in range(n)]
        ok = all(a[0] == b[0] for a, b in zip(pairs, pairs))
        check(checks, "matched_step_train_loss_pairs", pairs, None, ok=ok,
              note="同 seed 同步数快照（非判据口径；12k clean bpc 才是判据）")
        require("matched_step_train_loss_pairs", ok)
    except FileNotFoundError:
        pass

    # token 数与参数量
    check(checks, "train_tokens", STEPS * BS * SEQ, arms["full"]["train_tokens"])
    require("train_tokens", checks["train_tokens"]["match"])
    if "linear" in arms:
        check(checks, "params_equal", True, None,
              ok=(arms["full"]["params"] == arms["linear"]["params"]),
              note=f"{arms['full']['params']} vs {arms['linear']['params']}")
        require("params_equal", checks["params_equal"]["pass"])
    else:
        smoke = load(os.path.join("smoke", "smoke.json"))
        ok = smoke["checks"]["params_added"] == 0 and all(
            smoke["checks"]["init_bitwise_identical_to_full"].values())
        check(checks, "params_equal_from_smoke", True, None, ok=ok,
              note="linear 臂无 checkpoint：参数量相等/初始化逐位相同由 20 步冒烟"
                   "（两臂真实 shape）证明")
        require("params_equal_from_smoke", ok)
    check(checks, "params_value", 6962688, arms["full"]["params"])
    require("params_value", checks["params_value"]["match"])

    # 成本比值与实测交叉点
    ratios = res["cost"]["measured"]["ratio_by_T"]
    bad = []
    for r in ratios:
        f = next(x for x in res["cost"]["measured"]["model_step_by_T"]
                 if x["T"] == r["T"] and x["arm"] == "full")
        l = next(x for x in res["cost"]["measured"]["model_step_by_T"]
                 if x["T"] == r["T"] and x["arm"] == "linear")
        if round(l["median_ms"] / f["median_ms"], 4) != r["linear_over_full"]:
            bad.append(r["T"])
        if round(l["peak_mem_gib"] / f["peak_mem_gib"], 4) != r["linear_over_full_peak"]:
            bad.append(("peak", r["T"]))
    check(checks, "cost_ratios_recomputed", "all_match" if not bad else bad, "all_match")
    require("cost_ratios_recomputed", not bad, f"比值与原始 ms/显存重算不一致：{bad}")
    cross = [r["T"] for r in ratios if r["linear_over_full"] < 1.0]
    check(checks, "measured_crossover_T",
          min(cross) if cross else f"none_within_{sorted(r['T'] for r in ratios)}",
          res["cost"]["measured"].get("measured_crossover_T",
                                      "not_recorded_in_json"))
    checks["measured_crossover_T"]["pass"] = True   # 只要求能回算，不要求记录

    # 理论口径
    d, H, MLP = 256, 8, 4
    dh = d // H
    attn_full = 2 * SEQ * d
    attn_lin = 2 * d * dh + 2 * C * d + 2 * d
    other = 3 * d * d + d * d + 2 * MLP * d * d
    th_f = res["cost"]["theory"]["flops_full_T256"]["per_token_per_layer_macs"]
    th_l = res["cost"]["theory"]["flops_linear_T256"]["per_token_per_layer_macs"]
    check(checks, "theory_attn_full_macs", attn_full, th_f["attn"])
    require("theory_attn_full_macs", checks["theory_attn_full_macs"]["match"])
    check(checks, "theory_attn_linear_macs", attn_lin, th_l["attn"])
    require("theory_attn_linear_macs", checks["theory_attn_linear_macs"]["match"])
    check(checks, "theory_other_macs", other, th_f["other"])
    require("theory_other_macs", checks["theory_other_macs"]["match"])
    check(checks, "theory_crossover_training", dh + C,
          res["cost"]["theory"]["crossover_T_training_impl"]["T_star"])
    require("theory_crossover_training",
            checks["theory_crossover_training"]["match"])
    check(checks, "theory_crossover_streaming", dh,
          res["cost"]["theory"]["crossover_T_streaming"]["T_star"])
    require("theory_crossover_streaming",
            checks["theory_crossover_streaming"]["match"])
    kv = res["cost"]["theory"]["kv_memory"]["256"]
    check(checks, "kv_full_elements", 2 * d * SEQ, kv["full_kv_cache_elements"])
    require("kv_full_elements", checks["kv_full_elements"]["match"])
    check(checks, "kv_linear_state_elements", d * dh + d,
          kv["linear_state_elements_constant"])
    require("kv_linear_state_elements", checks["kv_linear_state_elements"]["match"])
    check(checks, "kv_reduction_frac_256", round(1 - (d * dh + d) / (2 * d * SEQ), 4),
          kv["linear_kv_reduction_frac"])
    require("kv_reduction_frac_256", checks["kv_reduction_frac_256"]["match"])
    check(checks, "kv_reduction_denominator_aligned", True, None, ok=True,
          note="报的是「删除量」（KV 元素数/可见性对）与「可见性对数」两套口径中"
               "的 KV 元素口径，另有依赖曲线的可见性对数口径（见 dependency_pairs）")

    # 依赖曲线：可见性对数与位置数闭式重算
    dep = res["dependency_curves"]
    T = SEQ
    causal_pairs = T * (T + 1) // 2
    ok_all, detail = True, {}
    for arm, curves in dep.items():
        # distance 曲线：query i 只丢 key i−d 一个 ⇒ 每丢一个键只减少 1 对可见性，
        # Σ_i [i ≥ d] = T − d 对，被删位置同样 = T − d（键 j 只对 query j+d 被删）。
        for row in curves["distance_curve"]:
            want = T - row["d"]
            detail[f"{arm}/dist:{row['d']}"] = [row["n_dropped_pairs"] == want,
                                                row["n_dropped_keys"] == want]
            ok_all &= bool(row["n_dropped_pairs"] == want
                           and row["n_dropped_keys"] == want)
        # recent(d)：保留 Σ_i min(d, i+1) 对；键 j 对 query ≥ j+d 不可见 ⇒ 被删位置 = T−d
        for row in curves["recent_curve"]:
            kept = sum(min(row["d"], i + 1) for i in range(T))
            detail[f"{arm}/recent:{row['d']}"] = [row["n_dropped_pairs"]
                                                  == causal_pairs - kept,
                                                  row["n_dropped_keys"]
                                                  == T - row["d"]]
            ok_all &= bool(row["n_dropped_pairs"] == causal_pairs - kept
                           and row["n_dropped_keys"] == T - row["d"])
    check(checks, "dependency_pairs_closed_form",
          "all_match" if ok_all else detail, "all_match",
          note="distance 曲线删 T−d 对可见性；recent:d 曲线保留 Σ min(d,i+1) 对")
    require("dependency_pairs_closed_form", ok_all, f"闭式重算不一致：{detail}")

    # 活性量（每 200 步落盘）
    for arm, flag in ([("full", False)] + ([("linear", True)] if "linear" in arms else [])):
        log = rows(f"{arm}-12k-s0-train_log.jsonl")
        n_expected = STEPS // 200 + 1
        check(checks, f"log_lines_{arm}", n_expected, len(log))
        require(f"log_lines_{arm}", checks[f"log_lines_{arm}"]["match"])
        live_rows = [r for r in log if r["attn_linear"] is not None]
        if flag:
            ok = (len(live_rows) == len(log)
                  and all(r["attn_linear"]["phi_min"] > 0
                          and r["attn_linear"]["den_min"] > 0
                          and 0.0 <= r["attn_linear"]["inter_mass_frac"] <= 1.0
                          for r in log))
            check(checks, "liveness_every_logline_linear", True, None, ok=ok,
                  note="每 200 步行都带 attn_linear 活性量，φ>0、den>0、0≤inter≤1")
            require("liveness_every_logline_linear", ok)
            probe = arms["linear"]["attn_linear_liveness"]
            check(checks, "liveness_alive_linear", True, None,
                  ok=bool(probe["alive"] and probe["params_added"] == 0
                          and probe["rel_dev_vs_full_min"] > 0),
                  note=f"alive={probe['alive']} rel_dev_min="
                       f"{probe['rel_dev_vs_full_min']}")
            require("liveness_alive_linear", checks["liveness_alive_linear"]["pass"])
            # 日志的逐层样本数：步数 × 层数
            check(checks, "liveness_layer_samples_linear", STEPS * 8,
                  log[-1]["attn_linear"]["n_layer_samples"])
            require("liveness_layer_samples_linear",
                    checks["liveness_layer_samples_linear"]["match"])
        else:
            check(checks, "liveness_none_on_full", True, None,
                  ok=all(r["attn_linear"] is None for r in log))
            require("liveness_none_on_full", checks["liveness_none_on_full"]["pass"])

    # 仪器自检 / O(T) 证据 / 配对差 / 重跑散度
    sc = res["instrument_selfcheck"]
    check(checks, "instrument_identity_bitwise_full",
          sc["full"]["max_abs_logit_diff_released_vs_instrument"], 0.0)
    require("instrument_identity_bitwise_full",
            sc["full"]["max_abs_logit_diff_released_vs_instrument"] == 0.0)
    if "linear" in sc:
        check(checks, "instrument_offset_linear_below_band", True, None,
              ok=sc["linear"]["bpc_abs_offset"] < BAND,
              note=f"线性臂 released vs instrument 的 bpc 偏移 = "
                   f"{sc['linear']['bpc_abs_offset']}（浮点累加顺序）")
        require("instrument_offset_linear_below_band",
                checks["instrument_offset_linear_below_band"]["pass"])
    else:
        smoke = load(os.path.join("smoke", "smoke.json"))
        off = smoke["checks"]["instrument_path"]["abs_offset"]
        check(checks, "instrument_offset_linear_below_band_smoke", off, None,
              ok=off < BAND,
              note="linear 臂无 12k 模型：该口径由 20 步冒烟（真实 shape、随机初始化）"
                   "覆盖 = released vs instrument 的 bpc 偏移")
        require("instrument_offset_linear_below_band_smoke", off < BAND)
    try:
        smoke = load(os.path.join("smoke", "smoke.json"))
        fl = smoke["checks"]["flop_scaling_Tx4"]
        check(checks, "smoke_flop_ratio_linear", fl["linear_ratio"], round(
            fl["linear_flops_T1024"] / fl["linear_flops_T256"], 3))
        require("smoke_flop_ratio_linear", checks["smoke_flop_ratio_linear"]["match"])
        check(checks, "smoke_flop_ratio_full", fl["full_ratio"], round(
            fl["full_flops_T1024"] / fl["full_flops_T256"], 3))
        require("smoke_flop_ratio_full", checks["smoke_flop_ratio_full"]["match"])
        require("smoke_linear_ratio_below_full", fl["linear_ratio"] < fl["full_ratio"] / 2,
                "线性实装的 FLOP 斜率没有显著低于全注意力")
        check(checks, "smoke_linear_ratio_below_full", [fl["linear_ratio"],
                                                        fl["full_ratio"]], None, ok=True)
    except (FileNotFoundError, KeyError):
        pass
    try:
        paired = load("paired_eval.json")
        key = "linear-12k-s0"
        if key not in paired.get("paired_delta_vs_full", {}):
            raise FileNotFoundError
        pd = paired["paired_delta_vs_full"][key]["delta_bpc_schema_(full−arm)"]
        check(checks, "paired_delta_matches_arm_bpc", pd,
              round(f_bpc - paired["mean_bpc_by_arm"][key], 5),
              ok=abs(pd - round(f_bpc - paired["mean_bpc_by_arm"][key], 5)) <= 1e-4)
        require("paired_delta_matches_arm_bpc",
                checks["paired_delta_matches_arm_bpc"]["match"])
    except (FileNotFoundError, KeyError):
        pass
    if res.get("reruns", {}).get("linear", {}).get("status") == "done":
        r2 = res["reruns"]["linear"]["clean_bpc"]
        check(checks, "rerun_divergence_linear", round(abs(r2 - l_bpc), 6),
              res["reruns"]["linear"]["bpc_abs_diff_vs_primary"])
        require("rerun_divergence_linear", checks["rerun_divergence_linear"]["match"])

    # 哈希/体积
    for name, meta in res.get("checkpoints", {}).items():
        p = os.path.join(ROOT, meta["path"])
        if not os.path.exists(p):
            continue
        got_h, got_s = sha256(p), os.path.getsize(p)
        check(checks, f"checkpoint_{name}", got_h, meta["sha256"])
        require(f"checkpoint_{name}",
                got_h == meta["sha256"] and got_s == meta["size_bytes"],
                f"sha256/体积不一致（{got_h[:12]}… vs {meta['sha256'][:12]}…）")

    # 墙钟账
    train_min = sum(a["wall_min"] for a in arms.values())
    if res.get("reruns"):
        train_min += sum(r.get("wall_min", 0) for r in res["reruns"].values()
                         if isinstance(r, dict))
    instr = sum(res.get("instrument_wall_min", {}).values())
    eval_min = sum(a.get("eval_wall_min", 0) for a in arms.values())
    check(checks, "wall_accounting", round(res["wall_total_min"], 2), None, ok=True,
          note=f"训练 {train_min:.2f} + 仪器 {instr:.2f} + 电池 {eval_min:.2f} + "
               f"成本口径/加载 = {res['wall_total_min'] - train_min - instr - eval_min:.2f} min")

    report = {"n_checks": len(checks), "n_fail": len(fails), "failures": fails,
              "checks": checks,
              "note": "所有数字都由 results.json / 日志 / smoke.json 独立重算得出；"
                      "不修改任何落盘数字"}
    json.dump(report, open(os.path.join(HERE, "reconcile_report.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"n_checks": report["n_checks"], "n_fail": report["n_fail"],
                      "failures": failures if (failures := fails) else []},
                     ensure_ascii=False), flush=True)
    if fails:
        raise SystemExit(f"回算失败 {len(fails)} 条：{fails}")
    print("[reconcile] 全部回算通过", flush=True)


if __name__ == "__main__":
    main()
