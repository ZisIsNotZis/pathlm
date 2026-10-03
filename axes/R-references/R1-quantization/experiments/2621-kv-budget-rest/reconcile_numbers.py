"""2621 · 报告数字对账（SCHEMA §12 三道）：`analysis.md` / `table.md` / `chart.txt`
里报告的每个数字都必须能回到 `results.json`（以及来源实验 / 2619 / 2620 的 json）。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出
输出：reconcile_report.json（逐条 [label, ok, got, want]）
"""

import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import rest_arms as ra                                                    # noqa: E402

GA_PATH = os.path.join(ROOT, "axes", "R-references", "R1-quantization",
                       "experiments", "2620-kv-budget-gaps", "gaps_arms.py")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ga = _load(GA_PATH, "gaps_arms_2620")

R = json.load(open(os.path.join(HERE, "results.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
TB = open(os.path.join(HERE, "table.md")).read()
CT = open(os.path.join(HERE, "chart.txt")).read()
CFG = R["config"]
PTS = {p["arm"]: p for p in R["points"]}

checks: list[list] = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


def num_in(text, value, label, places=6):
    """报告里是否出现该数字（按几个常见小数位的四舍五入形式找）。"""
    cands = {f"{value:.{places}f}", f"{value:.4f}", f"{value:.3f}", f"{value:.2f}",
             f"{value:.1f}", f"{value:.0f}"}
    hit = sorted(c for c in cands if c in text)
    ck(label, bool(hit), hit or None, f"one of {sorted(cands)}")


def _wilson_lo(n_hits: int, n: int, z: float = 1.959963984540054) -> float:
    p = n_hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half)


# ------------------------------------------------------------------ 0 自检 / 等值
sc = R["instrument"]
ck("自检 bpc_of 与 retention.masked_bpc 逐位相等",
   sc["bpc_path"]["bitwise_equal"] is True, sc["bpc_path"]["abs_diff"], 0.0)
ck("自检恒等臂 bpc == 2614 记录锚 1.483569",
   abs(sc["bpc_path"]["bpc_of"] - 1.483569) < 1e-9, sc["bpc_path"]["bpc_of"], 1.483569)
ck("自检 synth_accuracy 与 longrange_tasks.accuracy 一致（n/n_hits/by_distance）",
   sc["synth_path"]["n_equal"] and sc["synth_path"]["n_hits_equal"]
   and sc["synth_path"]["by_distance_equal"], sc["synth_path"], "all True")
ck("combo_mask 两条退化路径逐位还原既有仪器（masked_logits / kv_quant_logits）",
   all(sc["combo_mask_reduction"].values()), sc["combo_mask_reduction"], "all True")

# ------------------------------------------------------------------ 1 横轴公式（判据 C1）
n_layers = CFG["model"]["n_layers"]
n_heads = CFG["model"]["n_heads"]
hd = CFG["model"]["d_model"] // n_heads
T = CFG["axis_definition"]["seq_len_T"]


def want_bytes(arm):
    spec = CFG["arms"][arm]
    if spec["kind"] in ra.COMBO_KINDS:
        return ra.bytes_per_position(spec, n_layers, n_heads, hd, T)
    return ga.bytes_per_position(spec, n_layers, n_heads, hd, T)


for arm, p in PTS.items():
    want = want_bytes(arm)
    ck(f"{arm}: bytes_per_position == 唯一公式",
       abs(p["bytes_per_position"] - want["bytes_per_position"]) < 1e-9,
       p["bytes_per_position"], want["bytes_per_position"])
    ck(f"{arm}: positions_kept 解析值一致",
       p["factor_positions_kept"] == want["factor_positions_kept"])
    ck(f"{arm}: n_kv_heads_eff / value_bytes 一致",
       p["factor_n_kv_heads_eff"] == want["factor_n_kv_heads_eff"]
       and abs(p["factor_value_bytes"] - want["factor_value_bytes"]) < 1e-9)
ck("fp16 基线 = 12288 B/位置", PTS["fp16"]["bytes_per_position"] == 12288.0)
# 本批 8 点的横轴预算
ck("gqa_int4 = 768", PTS["gqa_int4"]["bytes_per_position"] == 768.0)
ck("mqa_int4 = 384", PTS["mqa_int4"]["bytes_per_position"] == 384.0)
ck("win32_int8 = 768 / win64_int8 = 1536",
   PTS["win32_int8"]["bytes_per_position"] == 768.0
   and PTS["win64_int8"]["bytes_per_position"] == 1536.0)
ck("win16_anchor16_int4 = 384 / win32_anchor32_int4 = 768",
   PTS["win16_anchor16_int4"]["bytes_per_position"] == 384.0
   and PTS["win32_anchor32_int4"]["bytes_per_position"] == 768.0)
ck("gqa_win64 = 768 / gqa_win128 = 1536",
   PTS["gqa_win64"]["bytes_per_position"] == 768.0
   and PTS["gqa_win128"]["bytes_per_position"] == 1536.0)
ck("等字节对：gqa_int4 == gqa_win64 == win32_int8 == win32_anchor32_int4 == 768",
   len({PTS[a]["bytes_per_position"] for a in
        ("gqa_int4", "gqa_win64", "win32_int8", "win32_anchor32_int4")}) == 1)
ck("等字节对：gqa_int8 == win64_int8 == gqa_win128 == 1536",
   len({PTS[a]["bytes_per_position"] for a in
        ("gqa_int8", "win64_int8", "gqa_win128")}) == 1)

# ------------------------------------------------------------------ 2 damage / 增量
for arm, p in PTS.items():
    base = PTS[p["family_baseline"]]
    ck(f"{arm}: damage_bpc == bpc - 本族基线 bpc",
       abs(p["damage_bpc"] - (p["bpc"] - base["bpc"])) < 1e-9,
       p["damage_bpc"], p["bpc"] - base["bpc"])
    ck(f"{arm}: kv_reduction == 1 - bytes/base",
       abs(p["kv_reduction_vs_family_baseline"]
           - (1 - p["bytes_per_position"] / base["bytes_per_position"])) < 1e-6,
       p["kv_reduction_vs_family_baseline"],
       1 - p["bytes_per_position"] / base["bytes_per_position"])
    ck(f"{arm}: total_damage_vs_stock == bpc - 1.483569",
       abs(p["total_damage_vs_stock"]
           - (p["bpc"] - CFG["controls"]["baseline_anchor_bpc"])) < 1e-6,
       p["total_damage_vs_stock"], p["bpc"] - CFG["controls"]["baseline_anchor_bpc"])
for key in ("quant_increment_ref", "window_increment_ref"):
    for arm, ref in CFG.get(key, {}).items():
        if arm == "note":
            continue
        p, r = PTS[arm], PTS[ref]
        field = "quant_increment_bpc" if key == "quant_increment_ref" else "window_increment_bpc"
        ck(f"{arm}: {field} == bpc - bpc({ref})",
           abs(p[field] - (p["bpc"] - r["bpc"])) < 1e-6, p[field], p["bpc"] - r["bpc"])

# ------------------------------------------------------------------ 3 召回腿口径
chance = R["controls"]["chance_uniform"]
ck("chance == 1/64", abs(chance - 1 / 64) < 1e-9, chance, 1 / 64)


def expected_n(key: str) -> int:
    t = CFG["tasks"][key]
    m = t["copy_len"] if key == "copy" else 1
    sets = len(t.get("src_positions") or t.get("src_starts") or t.get("distances"))
    return t["n_batches"] * t["batch_size"] * sets * m


copy_n = expected_n("copy")
ck("copy 汇总 n == 配置推出的 n", copy_n == 2 * 16 * 32 * 3, copy_n, 2 * 16 * 32 * 3)
for arm, p in PTS.items():
    ck(f"{arm}: copy n == {copy_n}", p["recall_copy"]["n"] == copy_n,
       p["recall_copy"]["n"], copy_n)
    ck(f"{arm}: copy accuracy == n_hits/n",
       abs(p["recall_copy"]["accuracy"] - p["recall_copy"]["n_hits"] / copy_n) < 1e-6,
       p["recall_copy"]["accuracy"], p["recall_copy"]["n_hits"] / copy_n)
    ps = p["recall_copy"]["per_setting"]
    ck(f"{arm}: 分 src_pos 三档都在", set(ps.keys()) == {"0", "96", "191"}, sorted(ps))
    for k, v in ps.items():
        ck(f"{arm}: copy@{k} accuracy == n_hits/n",
           abs(v["accuracy"] - v["n_hits"] / v["n"]) < 1e-6, v["accuracy"],
           v["n_hits"] / v["n"])
    ck(f"{arm}: induction n == {expected_n('induction')}",
       p["recall_induction"]["n"] == expected_n("induction"))
    ck(f"{arm}: var_recall n == {expected_n('var_recall')}",
       p["recall_var_recall"]["n"] == expected_n("var_recall"))
    ck(f"{arm}: copy 毁源对照落回地板（<= chance + 2σ）",
       p["recall_copy_source_corrupt"]["accuracy"]
       <= chance + 2 * (chance - _wilson_lo(int(round(chance * copy_n)), copy_n)),
       p["recall_copy_source_corrupt"]["accuracy"],
       chance + 2 * (chance - _wilson_lo(int(round(chance * copy_n)), copy_n)))

# ------------------------------------------------------------------ 4 来源 / 跨 run 对照
cc = R["cross_check_vs_source"]
ck("来源对照行数 == 臂数", len(cc["rows"]) == len(PTS), len(cc["rows"]), len(PTS))
# cross_check_vs_source 只对 R1/W3/M2（2619 的参考臂）有单一来源；2620 的组合臂由
# reproducibility_vs_2620 负责逐位对照（两者口径不同，不得混）。
refs = set(CFG["compare_against_2619"]["arms"])
bad = [r["arm"] for r in cc["rows"] if r["arm"] in refs and r["match"] is not True]
ck("全部 2619 参考臂与来源实验一致（abs diff <= 1e-5）", not bad, bad, "[]")
for key in ("reproducibility_vs_2619", "reproducibility_vs_2620"):
    rep = R[key]
    ck(f"{key}: 参考臂逐位一致", rep["all_matched"] is True,
       [r["arm"] for r in rep["rows"] if not r["match"]] or "none", "all")
    ck(f"{key}: max_bpc_abs_diff == 0",
       rep.get("max_bpc_abs_diff") == 0.0, rep.get("max_bpc_abs_diff"), 0.0)

# ------------------------------------------------------------------ 5 报告里出现
for arm, p in PTS.items():
    ck(f"table.md 含臂 {arm}", f"`{arm}`" in TB)
    num_in(TB, p["bpc"], f"table.md 含 {arm} bpc", places=6)
    ck(f"table.md 含 {arm} bytes/位置 {p['bytes_per_position']:.0f}",
       f"{p['bytes_per_position']:.0f}" in TB)
num_in(TB, chance, "table.md 含 chance")
num_in(TB, R["memory_convention"]["measured_gpu_peak_mib"], "table.md 含实测显存峰值",
       places=1)
for key in ("gqa_int4", "mqa_int4", "win32_int8", "win64_int8", "win16_anchor16_int4",
            "win32_anchor32_int4", "gqa_win64", "gqa_win128"):
    ck(f"analysis.md 提到本批组合臂 {key}", f"`{key}`" in AN or key in AN)
ck("analysis.md 提到两腿一致性判读（C7 口径）", "两腿" in AN or "C7" in AN)
ck("analysis.md 提到禁止单点外推（§17.11 / C6）", "外推" in AN or "C6" in AN)
ck("analysis.md 单列实测显存（判据 C2）",
   "实测显存" in AN or "measured_gpu_peak" in AN or "GPU 峰值" in AN)
ck("analysis.md 显式列出仍缺的组合（W1 交付项）", "仍缺" in AN)
ck("chart.txt 非空且含 x 轴说明", "每位置 KV 字节" in CT and len(CT) > 200)
for arm in PTS:
    ck(f"analysis.md 的总表行含 {arm}", f"`{arm}`" in AN)

# ------------------------------------------------------------------ 6 成本 / 活性
ck("cost.wall_seconds > 0", R["cost"]["wall_seconds"] > 0, R["cost"]["wall_seconds"])
ck("墙钟在预算内（<= 20 min）", R["cost"]["wall_minutes"] <= 20,
   R["cost"]["wall_minutes"], 20)
ck("峰值显存 <= 5GB 预算",
   R["memory_convention"]["measured_gpu_peak_mib"] <= 5 * 1024,
   R["memory_convention"]["measured_gpu_peak_mib"], 5120)
for arm in ("gqa_int8", "mqa_int8", "gqa_int4", "mqa_int4"):
    act = PTS[arm]["activity"]
    ck(f"{arm}: 活性报头数（qkv_out_width != 3d）",
       act["qkv_out_width"] != 3 * CFG["model"]["d_model"], act["qkv_out_width"])
    ck(f"{arm}: 活性量化 rel_err > 0", act["quant"]["rel_err"] > 0, act["quant"]["rel_err"])
for arm in ("win16_anchor16", "win32_anchor32", "win32_anchor8",
            "win16_anchor16_int4", "win32_anchor32_int4"):
    act = PTS[arm]["activity"]
    ck(f"{arm}: 活性 dropped_frac > 0", act["dropped_frac"] > 0, act["dropped_frac"])
    ck(f"{arm}: 活性锚点放行对数 > 0", act["anchor_contribution_pairs"] > 0,
       act["anchor_contribution_pairs"])
for arm in ("win32_int8", "win64_int8", "gqa_win64", "gqa_win128"):
    act = PTS[arm]["activity"]
    ck(f"{arm}: combo_mask 活性 dropped_frac > 0", act["dropped_frac"] > 0,
       act["dropped_frac"])
    ck(f"{arm}: combo_mask 量化活性 rel_err 与 mode 一致（fp32=0 恒等）",
       (act["quant"]["rel_err"] > 0) == (act["mode"] != "fp32"),
       (act["mode"], act["quant"]["rel_err"]), "int8/int4 > 0; fp32 == 0")

# ------------------------------------------------------------------ 汇总

n_ok = sum(1 for c in checks if c[1])
report = {"n_checks": len(checks), "n_ok": n_ok, "n_fail": len(checks) - n_ok,
          "checks": checks}
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
for c in checks:
    if not c[1]:
        print(f"FAIL  {c[0]}  got={c[2]} want={c[3]}")
print(f"[reconcile] {n_ok}/{len(checks)} 通过")
sys.exit(0 if n_ok == len(checks) else 1)
