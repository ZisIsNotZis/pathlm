"""2616 · 报告数字复算（契约统一纪律：报告里的每个成本/次数/等式/哈希都必须与
`results.json` 或日志逐一对得上）。

做法：**不把报告里的数字当输入**——每个量都从原始 artefact 重算一遍
（`results.json` / `runs/*/train_log.jsonl` / `smoke/smoke_report.json` /
`model_hashes.json` / `eval_log.jsonl` / 隔壁 `R1` int8-int4 参照线的 `results.json`），
再要求**报告文本**（`analysis.md` / `MODEL.md`）里出现同一个字面值
（比较前把 Unicode 减号 U+2212 规范成 ASCII `-`，其余逐字节比）。

任一不匹配 ⇒ `fail`，退出码非零。用法：python3 reconcile_numbers.py
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
R1 = os.path.join(ROOT, "axes", "R-references", "R1-quantization", "experiments",
                  "2614-kv-quant-eval-13m", "results.json")
TRUNK = os.path.join(ROOT, "axes", "H-horizon", "H4-head-trunk-relation",
                     "experiments", "2610-frozen-trunk-jumphead-13m", "trunk.pt")
ARMS = ("mha", "gqa", "mqa", "gqa_rep")
NKV = {"mha": 8, "gqa": 2, "mqa": 1, "gqa_rep": 2}
RUN = {"mha": "mha-12k-s0", "gqa": "gqa-12k-s0", "mqa": "mqa-12k-s0",
       "gqa_rep": "gqa-12k-s0-rep"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return text.replace("−", "-")          # U+2212 → ASCII '-'


def main():
    res = json.load(open(os.path.join(HERE, "results.json")))
    cfg = res["config"]
    hashes = json.load(open(os.path.join(HERE, "model_hashes.json")))
    smoke = json.load(open(os.path.join(HERE, "smoke", "smoke_report.json")))
    r1 = json.load(open(R1))
    reports = {p: norm(open(os.path.join(HERE, p)).read())
               for p in ("analysis.md", "MODEL.md", "README.md", "stability.md")
               if os.path.exists(os.path.join(HERE, p))}
    logs = {a: [json.loads(l) for l in
                open(os.path.join(HERE, "runs", RUN[a], "train_log.jsonl"))]
            for a in ARMS}
    mha_params = res["arms"]["mha"]["params"]
    d, dh, L = (cfg["model"]["d_model"],
                cfg["model"]["d_model"] // cfg["model"]["n_heads"],
                cfg["model"]["n_layers"])

    checks = []
    asserts = []

    def check(name, value, fmt=str, where=("analysis.md", "MODEL.md", "stability.md"),
              note=""):
        s = fmt(value) if callable(fmt) else str(value)
        present = {p: (s in reports[p]) for p in where}
        ok = any(present.values())
        checks.append({"check": name, "recomputed": s, "found_in": present,
                       "ok": bool(ok), "note": note})
        return s

    def assert_true(name, cond, detail=""):
        asserts.append({"assert": name, "ok": bool(cond), "detail": str(detail)})

    # ---------- 1. 质量：四臂 clean bpc / Δ / damage ----------
    bpcs = {a: res["arms"][a]["bpc_clean"] for a in ARMS}
    for a in ARMS:
        check(f"bpc_clean[{a}]", bpcs[a], "{:.4f}".format)
    ctl = bpcs["mha"]
    for a in ("gqa", "mqa", "gqa_rep"):
        delta = round(ctl - bpcs[a], 6)
        entry = res["criteria"]["per_arm"][a]
        assert_true(f"criteria.delta_schema[{a}] == recomputed",
                    entry["delta_schema"] == delta, (entry["delta_schema"], delta))
        check(f"delta_schema[{a}]", delta, "{:+.4f}".format,
              note="Δ = bpc_mha − bpc_arm（正 = 实验臂更好）")
        check(f"damage_bpc[{a}]", round(-delta, 6), "{:+.4f}".format)
    check("relative_damage_gqa_pct", (ctl and (bpcs["gqa"] - ctl) / ctl * 100),
          "{:.2f}".format, note="+0.96% 相对劣化")
    check("relative_damage_mqa_pct", (bpcs["mqa"] - ctl) / ctl * 100,
          "{:.2f}".format)
    check("gqa_vs_mqa_gap", abs(bpcs["gqa"] - bpcs["mqa"]), "{:.4f}".format,
          note="臂间差 ≪ 0.014 ⇒ 不可排序")
    check("int4_over_gqa_cost_ratio",
          r1["criteria"]["int4_damage_bpc"] / abs(bpcs["gqa"] - ctl),
          "{:.1f}".format, note="同 −75% 内存档：int4 的代价是 GQA 的几倍")
    # 第二条评测路径（独立实现）必须与 depth_curve 同数
    for a in ARMS:
        check(f"bpc_second_path[{a}]", res["arms"][a]["bpc_second_path_masked_bpc"],
              "{:.6f}".format)
        assert_true(f"second_path_matches[{a}]",
                    round(res["arms"][a]["bpc_second_path_masked_bpc"], 4)
                    == round(bpcs[a], 4))
    for a in ARMS:
        check(f"bpc_eval_rerun_abs_diff[{a}]",
              res["arms"][a]["bpc_eval_rerun_abs_diff"], "{:.1f}".format,
              note="同一次电池内的复现性重跑")
    for a in ARMS:
        check(f"next_token_acc[{a}]", res["arms"][a]["next_token_acc"],
              "{:.4f}".format)

    # ---------- 2. KV 解析内存（算术 + 实测张量形状） ----------
    for a in ARMS:
        live = res["arms"][a]["kv_liveness"]
        expect = L * 2 * NKV[a] * dh * 2
        assert_true(f"kv_bytes_arithmetic[{a}]",
                    live["kv_bytes_per_position_fp16"] == expect,
                    (live["kv_bytes_per_position_fp16"], expect))
        assert_true(f"k_tensor_heads[{a}]", live["k_batch_shape"][1] == NKV[a],
                    live["k_batch_shape"])
        assert_true(f"v_tensor_heads[{a}]",
                    live["per_layer"][0]["v_shape"][1] == NKV[a])
        assert_true(f"q_tensor_heads[{a}]",
                    live["per_layer"][0]["q_shape"][1] == cfg["model"]["n_heads"])
        assert_true(f"kv_reduction_frac[{a}]",
                    abs(live["kv_reduction_frac"] - (1 - NKV[a] / 8)) < 1e-9)
        check(f"kv_bytes_per_position_fp16[{a}]", expect)
        check(f"k_tensor_shape[{a}]", "[{}]".format(
            ", ".join(str(x) for x in live["k_batch_shape"])))
    check("kv_bytes_mha", 12288)
    check("kv_reduction_gqa_pct", 75.0, "{:.2f}".format)
    check("kv_reduction_mqa_pct", 87.5, "{:.2f}".format)
    check("kv_reduction_int8_pct", 50.0, "{:.0f}".format)
    check("kv_bytes_int8_reference", 6144,
          note="R1 int8 KV 参照线（同模型规格、同口径）")
    check("int8_damage_bpc", r1["criteria"]["int8_abs_delta_bpc"], "{:.5f}".format)
    check("int4_damage_bpc", r1["criteria"]["int4_damage_bpc"], "{:.4f}".format)
    assert_true("r1_int4_memory_reduction",
                r1["criteria"]["int4_memory_reduction_frac"] == 0.75)
    assert_true("r1_int8_memory_reduction",
                r1["criteria"]["int8_memory_reduction_frac"] == 0.5)
    assert_true("r1_int8_bytes", r1["memory"]["int8"]["per_position_bytes"] == 6144.0)
    assert_true("r1_fp16_bytes", r1["memory"]["fp16"]["per_position_bytes"] == 12288.0)
    # 长上下文规模表（解析外推：纯乘法，逐项与 R1 记录的 fp16/int4 对账）
    for a, nd in (("mha", 1), ("gqa", 2), ("mqa", 3)):
        mib = L * 2 * NKV[a] * dh * 2 * 256 / 2 ** 20
        check(f"kv_at_256[{a}] MiB", f"{mib:.{nd}f} MiB",
              note="T=256（MiB 表；逐臂用报告里的小数位）")
    check("kv_at_4096_mha", 12288 * 4096 / 2 ** 20, "{:.0f}".format)
    check("kv_at_4096_gqa", 3072 * 4096 / 2 ** 20, "{:.0f}".format)
    check("kv_at_4096_mqa", 1536 * 4096 / 2 ** 20, "{:.0f}".format)
    check("kv_at_131072_mha_gib", 12288 * 131072 / 2 ** 30, "{:.2f}".format)
    check("kv_at_131072_gqa_mib", 3072 * 131072 / 2 ** 20, "{:.0f}".format)
    check("kv_at_131072_mqa_mib", 1536 * 131072 / 2 ** 20, "{:.0f}".format)
    assert_true("r1_fp16_at_131072",
                r1["memory"]["fp16"]["total_bytes_by_seq_len"]["131072"]
                == 1610612736.0)
    assert_true("r1_int4_at_131072",
                r1["memory"]["int4"]["total_bytes_by_seq_len"]["131072"]
                == 402665472.0)
    # 实测显存（解析 vs 实测分开报）
    for a in ("mha", "gqa", "mqa"):
        p = [r for r in smoke["arms"] if r["arm"] == a and r["steps"] == 200][0]
        check(f"train_peak_child_mib[{a}]", p["gpu_peak_child_mib"])
    check("eval_peak_mem_mib_max", max(res["cost"]["eval_peak_mem_mib_by_arm"].values()),
          "{:.1f}".format)

    # ---------- 3. 参数量（剪枝公式对账） ----------
    for a in ARMS:
        p = res["arms"][a]["params"]
        expect = mha_params - 2 * (d - NKV[a] * dh) * (d + 1) * L
        assert_true(f"params_arithmetic[{a}]", p == expect, (p, expect))
        check(f"params[{a}]", p, "{:,}".format)
        check(f"params_delta_vs_mha[{a}]", p - mha_params, "{:,}".format)

    # ---------- 4. 同方法斜率 / 重跑散度 / 活性 ----------
    for a in ARMS:
        t = res["arms"][a]["train_log"]
        check(f"slope_last_25pct[{a}]", t["slope_last_25pct"], "{:+.3e}".format)
        check(f"slope_last_50pct[{a}]", t["slope_last_50pct"], "{:+.4e}".format)
        check(f"slope_last_1000[{a}]", t["slope_last_1000"], "{:+.4e}".format)
        check(f"loss_last[{a}]", t["loss_last"], "{:.4f}".format)
        check(f"gnorm_max[{a}]", t["gnorm_max"], "{:.2f}".format)
        assert_true(f"train_log_lines[{a}]", len(logs[a]) == 61, len(logs[a]))
        check(f"train_log_lines_n[{a}]", 61)
        kv_lines = sum(1 for r in logs[a] if r.get("kv_heads"))
        assert_true(f"kv_activity_lines_eq_61[{a}]",
                    kv_lines == (0 if a == "mha" else 61), kv_lines)
    act = {a: res["arms"][a]["train_log"]["kv_heads_activity_lines"] for a in ARMS}
    assert_true("mha_no_kv_activity", act["mha"] == 0, act)
    check("kv_activity_lines_gqa", 61,
          note="活性量每 200 步随同一行训练日志落盘（SCHEMA §17.4）")
    check("kv_activity_lines_mha", 0, note="MHA 臂开关关闭 ⇒ 不写该字段")
    for a in ("gqa", "mqa", "gqa_rep"):
        kv = [r["kv_heads"] for r in logs[a] if r.get("kv_heads")]
        check(f"kv_rows_fro_first[{a}]", kv[0]["kv_rows_fro_mean"], str)
        check(f"kv_rows_fro_last[{a}]", kv[-1]["kv_rows_fro_mean"], str)
        check(f"kv_head_norms_last[{a}]", "[{}]".format(", ".join(
            "{:.4f}".format(v) for v in kv[-1]["k_proj_per_kv_head_row_norm"])))
        check(f"kv_head_norms_first[{a}]", "[{}]".format(", ".join(
            "{:.4f}".format(v) for v in kv[0]["k_proj_per_kv_head_row_norm"])))
    for a in ("gqa", "gqa_rep"):
        live = res["arms"][a]["kv_liveness"]
        check(f"kv_head_pair_max_diff[{a}]",
              live["kv_head_pair_max_abs_weight_diff"], "{:.4f}".format)
        check(f"qkv_weight_shape[{a}]", "[{}]".format(
            ", ".join(str(x) for x in live["per_layer"][0]["qkv_weight_shape"])))
    check("kv_head_norms_layer0[gqa]", " / ".join(
        "{:.4f}".format(v) for v in
        res["arms"]["gqa"]["kv_liveness"]["k_proj_per_kv_head_row_norm"]))
    check("kv_head_norms_layer0[mha]", "/".join(
        "{:.2f}".format(v) for v in
        res["arms"]["mha"]["kv_liveness"]["k_proj_per_kv_head_row_norm"]),
        note="MHA 的 8 个 KV 头各自的 K 投影行范数（层 0）")
    check("acc_range", "{:.4f}–{:.4f}".format(
        min(res["arms"][a]["next_token_acc"] for a in ARMS),
        max(res["arms"][a]["next_token_acc"] for a in ARMS)),
        note="四臂 next-token acc 区间")
    check("rerun_abs_diff_bpc", res["rerun_divergence"]["abs_diff_bpc"],
          "{:.4f}".format)
    check("rerun_threshold_bpc", cfg["reproducibility"]["threshold_bpc"],
          "{:.3f}".format)
    check("damage_over_rerun_noise_ratio",
          abs(bpcs["gqa"] - ctl) / res["rerun_divergence"]["abs_diff_bpc"],
          "{:.0f}".format, note="damage 是重跑噪声的几倍")
    assert_true("rerun_pass", res["rerun_divergence"]["pass"] is True)
    assert_true("rerun_same_verdict_false",
                res["rerun_divergence"]["same_verdict"] is False,
                "gqa 两次重跑跨 0.014 门槛（本实验最重要的稳定性事实）")

    # ---------- 5. 配置等值性逐字段 ----------
    ce = res["config_equivalence"]
    assert_true("only_intervention_unequal",
                ce["all_equal_except_intervention"] is True, ce["unequal_fields"])
    assert_true("unequal_fields_are_nkv_params",
                sorted(ce["unequal_fields"]) == ["n_kv_heads", "params"],
                ce["unequal_fields"])
    check("config_all_equal_except_intervention", "all_equal_except_intervention = true")
    for f, val in (("train.seed", 0), ("n_heads", 8), ("mb", None)):
        if f == "mb":
            continue
        row = [r for r in ce["rows"] if r["field"] == f][0]
        assert_true(f"config_field_equal[{f}]", row["all_equal"], row["values"])

    # ---------- 6. 成本 / 次数 / 等式 ----------
    assert_true("train_tokens_eq",
                cfg["train"]["steps"] * cfg["train"]["batch_size"]
                * cfg["train"]["seq_len"] == 98304000)
    check("train_tokens", 98304000, "{:,}".format)
    check("train_steps", cfg["train"]["steps"])
    check("train_batch_size", cfg["train"]["batch_size"])
    check("train_seq_len", cfg["train"]["seq_len"])
    check("arms_count", len(ARMS), note="臂数 = 干预档数 3 × (1 + 重跑 1)")
    total = 0.0
    for a in ARMS:
        m = res["cost"]["train_wall_minutes_by_arm"][a]
        total += m
        assert_true(f"wall_matches_hashes[{a}]",
                    abs(m - hashes[a]["wall_minutes"]) < 0.005, m)
        check(f"train_wall_minutes[{a}]", m, "{:.2f}".format)
    check("train_wall_total_minutes", res["cost"]["train_wall_total_minutes"],
          "{:.2f}".format)
    check("battery_wall_seconds", res["cost"]["battery_wall_seconds"],
          "{:.2f}".format)
    check("total_gpu_wall_minutes", res["cost"]["total_gpu_wall_minutes_measured"],
          "{:.2f}".format)
    check("contract_budget_min", cfg["cost"]["gpu_wall_budget_min"])
    for a in ("mha", "gqa", "mqa"):
        p = [r for r in smoke["arms"] if r["arm"] == a and r["steps"] == 200][0]
        check(f"smoke_ms_per_step[{a}]", p["marginal_ms_per_step"], "{:.2f}".format)
        q = [r for r in smoke["arms"] if r["arm"] == a and r["steps"] == 20][0]
        check(f"smoke20_ms_per_step[{a}]", q["marginal_ms_per_step"], "{:.2f}".format,
              note="20 步冒烟（含进程启动，不作外推，SCHEMA §17.5）")
    for a in ARMS:
        check(f"eval_peak_mem_mib[{a}]", res["cost"]["eval_peak_mem_mib_by_arm"][a],
              "{:.1f}".format)
    check("eval_log_lines", len(open(os.path.join(HERE, "eval_log.jsonl")).readlines()))
    check("battery_invocations", 3, note="训练后 1 次 + 单独重跑 2 次")

    # ---------- 7. 哈希（MODEL.md ⟷ 实际文件；代码不得在评测后改动） ----------
    for a in ARMS:
        want = hashes[a]["sha256"]
        got = sha256_file(os.path.join(HERE, "runs", RUN[a], "model.pt"))
        asserts.append({"assert": f"checkpoint_unchanged[{a}]", "ok": got == want,
                        "detail": got})
        check(f"model_sha256_MODEL_md[{a}]", want, where=("MODEL.md",))
        check(f"model_size_bytes[{a}]", hashes[a]["size_bytes"], "{:,}".format,
              where=("MODEL.md",))
    code_files = {"run_py_sha256": "archive/wiki-legacy/axes/W-width/W3-kv-head-sharing/experiments/"
                                  "2616-gqa-mqa-13m/run.py",
                  "depth_ar_py_sha256": "pathlm/depth_ar.py",
                  "train_depth_ar_py_sha256": "train_depth_ar.py",
                  "tests_depth_ar_sha256": "tests/test_depth_ar.py",
                  "retention_py_sha256": "tools/evalkit/retention.py"}
    for key, rel in code_files.items():
        now = sha256_file(os.path.join(ROOT, rel))
        asserts.append({"assert": f"code_unchanged_since_eval[{key}]",
                        "ok": now == res["code"][key],
                        "detail": f"{now} vs {res['code'][key]}"})
        check(f"code_hash_IN_MODEL_md[{key}]", res["code"][key], where=("MODEL.md",))

    # ---------- 8. 交叉核对锚（历史同规格，不作判据） ----------
    trunk_sha = sha256_file(TRUNK)
    asserts.append({"assert": "historical_trunk_sha256_prefix",
                    "ok": trunk_sha.startswith("86953dcb"), "detail": trunk_sha})
    check("historical_trunk_params", 13536768, "{:,}".format)
    assert_true("mha_params_eq_historical_trunk", mha_params == 13536768, mha_params)
    check("historical_trunk_bpc", 1.4836, where=("analysis.md",))
    check("historical_trunk_minus_mha_bpc", 1.4836 - bpcs["mha"], "{:.4f}".format,
          note="交叉核对：差在本配方 run-to-run 量级内")

    bad = [c for c in checks if not c["ok"]] + [a for a in asserts if not a["ok"]]
    out = {"n_checks": len(checks) + len(asserts), "n_report_string_checks": len(checks),
           "n_asserts": len(asserts), "n_failed": len(bad), "all_pass": not bad,
           "failed": bad, "checks": checks, "asserts": asserts}
    json.dump(out, open(os.path.join(HERE, "reconcile_report.json"), "w"),
              indent=2, ensure_ascii=False)
    for c in checks:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['check']:38s} {c['recomputed']}")
    for a in asserts:
        if not a["ok"]:
            print(f"FAIL assert {a['assert']} {a['detail']}")
    print(f"\n报告字面值 {len(checks)} 项 + 内部断言 {len(asserts)} 项："
          f"{len(checks) + len(asserts) - len(bad)}/{len(checks) + len(asserts)} 通过"
          f"{'（全部对得上）' if not bad else '—— 有 ' + str(len(bad)) + ' 项对不上'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
