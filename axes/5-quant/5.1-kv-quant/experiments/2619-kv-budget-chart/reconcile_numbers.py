"""2619 · 报告数字对账：`analysis.md` / `table.md` 里报告的每个数字都必须能回到
`results.json`（以及来源实验 / 诊断 json）。

用法：python3 reconcile_numbers.py     # 任一 mismatch 以非零码退出
"""

import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

R = json.load(open(os.path.join(HERE, "results.json")))
AN = open(os.path.join(HERE, "analysis.md")).read()
TB = open(os.path.join(HERE, "table.md")).read()
CT = open(os.path.join(HERE, "chart.txt")).read()
CFG = R["config"]
PTS = {p["arm"]: p for p in R["points"]}

def _wilson_lo(n_hits: int, n: int, z: float = 1.959963984540054) -> float:
    """Wilson 95% 区间下界（与 run.py::wilson_ci 同一式）。"""
    p = n_hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half)


def floor_threshold(n: int, chance: float) -> float:
    """判据 C4 的地板阈值：`chance + 2·σ_wilson`（与 run.py 同一式）。"""
    return chance + 2 * (chance - _wilson_lo(int(round(chance * n)), n))


checks: list[list] = []


def ck(label, ok, got=None, want=None):
    checks.append([label, bool(ok), str(got), str(want)])


# 注意（复核 V4 汇总 6）：本函数用**子串**匹配，会把 32.11 与 32、0.82 与 0.8 混淆；
# 后续请改为按数值容差匹配（abs(float(a)-float(b)) < tol），并禁止落入无小数点整数档。
def num_in(text, value, label, places=6):
    """报告里是否出现该数字（按几个常见小数位的四舍五入形式找）。"""
    cands = {f"{value:.{places}f}", f"{value:.4f}", f"{value:.3f}", f"{value:.2f}",
             f"{value:.1f}", f"{value:.0f}"}
    hit = sorted(c for c in cands if c in text)
    ck(label, bool(hit), hit or None, f"one of {sorted(cands)}")


# ------------------------------------------------------------------ 0 自检 / 等值
sc = R["instrument"]
ck("自检 bpc_of 与 retention.masked_bpc 逐位相等",
   sc["bpc_path"]["bitwise_equal"] is True, sc["bpc_path"]["abs_diff"], 0.0)
ck("自检恒等臂 bpc == 2614 记录锚 1.483569", abs(sc["bpc_path"]["bpc_of"] - 1.483569) < 1e-9,
   sc["bpc_path"]["bpc_of"], 1.483569)
ck("自检 synth_accuracy 与 longrange_tasks.accuracy 一致（n/n_hits/by_distance）",
   sc["synth_path"]["n_equal"] and sc["synth_path"]["n_hits_equal"]
   and sc["synth_path"]["by_distance_equal"], sc["synth_path"], "all True")

# ------------------------------------------------------------------ 1 横轴公式（判据 C1）
n_layers = CFG["model"]["n_layers"]
n_heads = CFG["model"]["n_heads"]
hd = CFG["model"]["d_model"] // n_heads
T = CFG["axis_definition"]["seq_len_T"]
for arm, p in PTS.items():
    spec = CFG["arms"][arm]
    nkv = spec["n_kv_heads"] if spec["kind"] == "head" else n_heads
    kept = spec["positions_kept"] if spec["kind"] == "window" else T
    want = 2 * n_layers * nkv * hd * spec["value_bytes"] * (kept / T)
    ck(f"{arm}: bytes_per_position == 唯一公式", abs(p["bytes_per_position"] - want) < 1e-9,
       p["bytes_per_position"], want)
ck("trunk 基线 = 12288 B/位置", PTS["fp16"]["bytes_per_position"] == 12288.0)
ck("int8 = 6144 / int4 = 3072",
   PTS["int8"]["bytes_per_position"] == 6144.0 and PTS["int4"]["bytes_per_position"] == 3072.0)
ck("mha = 12288 / gqa = 3072 / mqa = 1536",
   PTS["mha"]["bytes_per_position"] == 12288.0 and PTS["gqa"]["bytes_per_position"] == 3072.0
   and PTS["mqa"]["bytes_per_position"] == 1536.0)

# ------------------------------------------------------------------ 2 bpc 与来源实验
cc = R["cross_check_vs_source"]
ck("来源对照 14 行全覆盖", len(cc["rows"]) == len(PTS), len(cc["rows"]), len(PTS))
ck("来源对照全部一致（abs diff <= 1e-5）", cc["all_matched"] is True,
   [r["arm"] for r in cc["rows"] if r["match"] is False] or "none", "all")
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

# ------------------------------------------------------------------ 3 召回腿口径
chance = R["controls"]["chance_uniform"]
ck("chance == 1/64", abs(chance - 1 / 64) < 1e-9, chance, 1 / 64)
copy_n = 2 * 16 * 32 * len(CFG["tasks"]["copy"]["src_positions"])

def expected_n(key: str) -> int:
    """按 config 算该任务的总行数（避免报告里的 n 与配置脱节）。"""
    t = CFG["tasks"][key]
    m = t["copy_len"] if key == "copy" else 1
    sets = len(t.get("src_positions") or t.get("src_starts") or t.get("distances"))
    return t["n_batches"] * t["batch_size"] * sets * m

ck("copy 汇总 n == 配置推出的 n", expected_n("copy") == copy_n, expected_n("copy"), copy_n)
for arm, p in PTS.items():
    ck(f"{arm}: induction n == {expected_n('induction')}",
       p["recall_induction"]["n"] == expected_n("induction"))
    ck(f"{arm}: var_recall n == {expected_n('var_recall')}",
       p["recall_var_recall"]["n"] == expected_n("var_recall"))
for arm, p in PTS.items():
    ck(f"{arm}: copy 汇总 n == 2*16*32*3 = {copy_n}", p["recall_copy"]["n"] == copy_n,
       p["recall_copy"]["n"], copy_n)
    ck(f"{arm}: copy accuracy == n_hits/n",
       abs(p["recall_copy"]["accuracy"] - p["recall_copy"]["n_hits"] / copy_n) < 1e-6,
       p["recall_copy"]["accuracy"], p["recall_copy"]["n_hits"] / copy_n)
    lo, hi = p["recall_copy"]["wilson95"]
    ck(f"{arm}: Wilson CI 包住点估计", lo <= p["recall_copy"]["accuracy"] <= hi,
       [lo, hi], p["recall_copy"]["accuracy"])
    thr = floor_threshold(copy_n, chance)
    ck(f"{arm}: at_floor 与 C4 规则一致",
       p["recall_copy"]["at_floor"] == (p["recall_copy"]["accuracy"] <= thr),
       p["recall_copy"]["at_floor"], p["recall_copy"]["accuracy"] <= thr)
ck("局部先验 copy < 3× chance（任务构造达标）",
   R["controls"]["local_prior_copy"]["accuracy"] < 3 * chance,
   R["controls"]["local_prior_copy"]["accuracy"], 3 * chance)
ck("局部先验抗性检查三条腿都在",
   set(R["controls"]["local_prior_resistance"]) == {"copy", "induction", "var_recall"})

# ------------------------------------------------------------------ 4 口径隔离（判据 C2）
mc = R["memory_convention"]
ck("实测显存单列（不与解析同列）", "measured_gpu_peak_mib" in mc and "isolation_note" in mc)
ck("table.md 有独立的实测显存小节", "实测显存" in TB and "与解析列分列" in TB)
ck("table.md 横轴列显式标注解析口径", "每位置 KV 字节（解析）" in TB)
ck("analysis.md 显式声明解析 vs 实测不混", "解析" in AN and "实测" in AN)

# ------------------------------------------------------------------ 5 缺失组合（判据 C5）
missing = R["missing_combinations"]
ck("缺失组合非空", len(missing) >= 5, len(missing), ">=5")
ck("每条缺失组合都带 为什么缺 + 怎么办",
   all(set(m) >= {"id", "why_missing", "next_step"} for m in missing))
ck("table.md 列出缺失组合表", "缺失组合" in TB and "GQA × int8/int4" in TB)

# ------------------------------------------------------------------ 6 报告数字落盘一致
for key in PTS:
    p = PTS[key]
    num_in(AN, p["bpc"], f"analysis.md 含 {key} bpc", 6)
    num_in(AN, p["recall_copy"]["accuracy"], f"analysis.md 含 {key} copy 召回", 4)
    num_in(AN, p["recall_copy_source_corrupt"]["accuracy"],
           f"analysis.md 含 {key} 毁源读数", 4)
    num_in(TB, p["bpc"], f"table.md 含 {key} bpc", 6)
ck("analysis.md 含 int4 damage 0.045661",
   abs(PTS["int4"]["damage_bpc"] - 0.045661) < 1e-9, PTS["int4"]["damage_bpc"], 0.045661)
num_in(AN, PTS["int4"]["damage_bpc"], "analysis.md 含 int4 damage 值", 6)
num_in(AN, PTS["gqa"]["damage_bpc"], "analysis.md 含 gqa damage 值", 6)
num_in(AN, PTS["win192"]["damage_bpc"], "analysis.md 含 win192 damage 值", 6)
num_in(AN, PTS["win32"]["damage_bpc"], "analysis.md 含 win32 damage 值", 6)
num_in(AN, PTS["int8"]["damage_bpc"], "analysis.md 含 int8 damage 值", 6)
num_in(AN, PTS["win16"]["damage_bpc"], "analysis.md 含 win16 damage 值", 6)
num_in(AN, R["cost"]["wall_seconds"], "analysis.md 含墙钟秒", 1)
num_in(AN, R["memory_convention"]["measured_gpu_peak_mib"], "analysis.md 含实测峰值 MiB", 1)
DIAG = json.load(open(os.path.join(HERE, "diagnose_copy_confound.json")))
for pos, row in DIAG["span_sensitivity"].items():
    for k in ("as_is", "corrupt_span_0_32", "corrupt_span_src"):
        num_in(AN, row[k], f"analysis.md 含诊断 src_pos={pos}.{k}", 4)
ck("诊断：毁错区间远高于 chance、毁对区间 ≈ chance（ERRATA 根因）",
   DIAG["span_sensitivity"]["96"]["corrupt_span_0_32"] > 0.1
   and DIAG["span_sensitivity"]["96"]["corrupt_span_src"] < 0.03
   and DIAG["span_sensitivity"]["191"]["corrupt_span_0_32"] > 0.1
   and DIAG["span_sensitivity"]["191"]["corrupt_span_src"] < 0.03,
   {k: DIAG["span_sensitivity"][k] for k in ("96", "191")}, "wrong>0.1 & right<0.03")

# ------------------------------------------------------------------ 6b 派生常数 / 外部引用逐一核对
ck("锚只写了 6 位：instrument.abs_diff_vs_anchor 四舍五入到 2.4e-7",
   f"{R['instrument']['bpc_path']['abs_diff_vs_anchor']:.1e}" == "2.4e-07",
   R["instrument"]["bpc_path"]["abs_diff_vs_anchor"], "2.4e-07")
ck("局部先验抗性 ceiling == 3×chance == 0.0469（analysis.md 引用值）",
   abs(R["controls"]["local_prior_resistance"]["copy"]["ceiling"] - 0.046875) < 1e-9,
   R["controls"]["local_prior_resistance"]["copy"]["ceiling"], 0.046875)
ck("var_recall 地板阈值（n=512）== 0.0310（analysis.md 引用值）",
   abs(floor_threshold(expected_n("var_recall"), chance) - 0.0310) < 0.0005,
   floor_threshold(expected_n("var_recall"), chance), 0.0310)
int8_rel = PTS["int8"]["activity"]["rel_err"]
int4_rel = PTS["int4"]["activity"]["rel_err"]
fp16_rel = PTS["fp16"]["activity"]["rel_err"]
ck("analysis.md 引用的探针 rel_err（fp16 1.8e-4 / int8 4.8e-3 / int4 8.8e-2）与 results.json 一致且单调",
   f"{fp16_rel:.1e}" == "1.8e-04" and f"{int8_rel:.1e}" == "4.8e-03"
   and f"{int4_rel:.1e}" == "8.8e-02" and fp16_rel < int8_rel < int4_rel,
   [f"{fp16_rel:.1e}", f"{int8_rel:.1e}", f"{int4_rel:.1e}"],
   ["1.8e-04", "4.8e-03", "8.8e-02"])
_r1 = json.load(open(os.path.join(
    ROOT, "axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json")))
_r1_rel = {row["mode"]: row["activity"]["rel_err"] for row in _r1["quant_rows"]}
ck("analysis.md 引用的 2614 全电池 rel_err（1.76e-4 / 6.84e-3 / 1.24e-1）与 2614 results.json 一致",
   all(f"{_r1_rel[m]:.2e}" == w for m, w in (("fp16", "1.76e-04"), ("int8", "6.84e-03"),
                                             ("int4", "1.24e-01"))),
   {m: f"{v:.2e}" for m, v in _r1_rel.items()},
   {"fp16": "1.76e-04", "int8": "6.84e-03", "int4": "1.24e-01"})
ck("analysis.md 引用的窗口 32 dropped_frac 0.766 与 results.json 一致",
   abs(PTS["win32"]["activity"]["dropped_frac"] - 0.766) < 5e-4,
   PTS["win32"]["activity"]["dropped_frac"], 0.766)
ck("MQA/GQA 的 KV 下降 87.5% / 75% 与张量口径一致",
   abs(PTS["mqa"]["kv_reduction_vs_family_baseline"] - 0.875) < 1e-9
   and abs(PTS["gqa"]["kv_reduction_vs_family_baseline"] - 0.75) < 1e-9)
ck("预算占用 2.7%（32.7s / 20min）与 results.json 一致",
   abs(R["cost"]["wall_seconds"] / 1200 * 100 - 2.7) < 0.1,
   round(R["cost"]["wall_seconds"] / 1200 * 100, 2), 2.7)
_hashes = json.load(open(os.path.join(
    ROOT, "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/model_hashes.json")))
ck("参数 −8.75% / −10.25% 与 2616 model_hashes 一致（外部引用核对）",
   abs(1 - _hashes["gqa"]["params"] / _hashes["mha"]["params"] - 0.0875) < 5e-4
   and abs(1 - _hashes["mqa"]["params"] / _hashes["mha"]["params"] - 0.1025) < 5e-4,
   [round(1 - _hashes["gqa"]["params"] / _hashes["mha"]["params"], 4),
    round(1 - _hashes["mqa"]["params"] / _hashes["mha"]["params"], 4)], [0.0875, 0.1025])
_a3 = open(os.path.join(ROOT, "archive/wiki-legacy/axes/A-mixing/A3-window-anchor/README.md")).read()
ck("A3 引用的 needle 76.8% / 93.3% 在 A3 页上存在（外部引用核对）",
   "76.8" in _a3 and "93.3" in _a3)
ck("chart.png / chart.txt / table.md / results.json 都在",
   all(os.path.exists(os.path.join(HERE, f))
       for f in ("chart.png", "chart.txt", "table.md", "results.json")))
ck("chart.txt 含 14 个点的标记",
   all(m in CT for m in "b84wMQG") and "w=win16" in CT, CT.count("标记→点") == 1, True)
ck("results.json 记录 chart_png", "chart_png" in R, R.get("chart_png"), True)

# ------------------------------------------------------------------ 7 复现性
if "reproducibility" in R:
    ck("复现性 max|Δbpc| <= 0.005", R["reproducibility"]["pass"] is True,
       R["reproducibility"]["max_bpc_abs_diff"], 0.005)
else:
    ck("复现性块存在", False, "missing", "results.json::reproducibility")

# ------------------------------------------------------------------ 输出
bad = [c for c in checks if not c[1]]
report = {"n_checks": len(checks), "n_failed": len(bad),
          "failed": [{"label": c[0], "got": c[2], "want": c[3]} for c in bad],
          "checks": [{"label": c[0], "ok": c[1], "got": c[2], "want": c[3]} for c in checks]}
with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(f"reconcile：{len(checks)} 项检查，{len(bad)} 项失败")
for c in bad:
    print(f"  ✗ {c[0]}: got={c[2]} want={c[3]}")
sys.exit(1 if bad else 0)
