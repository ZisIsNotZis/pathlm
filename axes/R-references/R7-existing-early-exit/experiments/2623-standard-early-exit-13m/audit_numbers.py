"""`analysis.md` 里**每个数字**都要能追到落盘证据或已声明的算式（可失败审计）。

**覆盖与局限（如实声明）**：本审计只判**幅值**——「正文这个数是否等于某条证据或某条已声明
算式的结果」。它**不覆盖**下面三件事，读到「审计绿」时**不得外推**：

1. **归属**：这个数是否真属于正文所声称的那个量（值对、贴错标签它抓不到）；
2. **符号**：除非某个差值在正文里**只声明过一个方向**，否则把差写成反向它抓不到
   ——因此派生只写正文声明的**有向**差（不写反向差、不写绝对值），且秒↔分不靠 `×60` 撞；
3. **来源**：是否来自允许的 12k 步实测而非同名探针（60 步探针产物已从允许集剔除，
   但别的产物里同名的值仍可能撞上）。

覆盖来源：

1. 证据：主交付 `results.json`、8 条臂的逐臂 `results.json`。**60 步探针产物
   `step_time.json` 整份与 `config.json::step_time_probe` 子树已剔除**——探针的峰值显存
   `1205.0` 曾掩蔽 12k 步实测的 `1203.5`（探针的**计时** 3.0 min 改由派生列入，见下）；
2. **派生**：按正文声明的算式重算（有向差值 / 交互项 / 比值 / 均值 / SE 公式 / 参数量差 /
   GPU 分钟求和 / 丢弃 run 的秒数换算）；测量值一律**从 evidence 派生**，不写进 `ANCHORS`；
3. `ANCHORS`：外部可核的历史入账值（23d/23c/2621 等）与仓内口径常数（0.0138 / 0.014 等）。

数值比较按 **token 自身的小数位数**四舍五入；千分位逗号先合并。用法：
`python3 audit_numbers.py`（有待核对即非零退出）/ `python3 audit_numbers.py --selftest`。
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
ANA = os.path.join(HERE, "analysis.md")
EV = os.path.join(ROOT, ".scratch", "52-r7-early-exit", "evidence")

#: 外部可核的历史入账值 / 仓内口径常数 / 已声明算式之外的常数。
#: **测量值不在这里**——它们从 `results.json` / 逐臂证据派生（`derived_values()`）。
ANCHORS = [
    1.5287, 1.8582, 0.9694, 0.3295,            # 23d-dense-B-12k / 23c-H2-w0-12k 对照
    0.9423, 8.45, 1.86, 169, 200,              # C1 的 2621（历史测量）
    0.0138, 0.014, 0.0003, 0.0007, 0.057,      # 方差带与"无出处的旧 σ_tf"（用于说明，不参与判据）
    0.5, 0.1, 1.0, 8.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.3,  # 剂量/阈值等超参
    0.4362,                                     # C1 在 100M 的历史形态税（未重拆，原样引用）
    83, 87, 100, 128, 256, 512, 12000, 98304000,   # 步数/层数/top-k/位置数
    60,                                          # SCHEMA §20.1 的臂间最小间隔（口径常数，非测量值）
]

#: 单位换算比例：k/M（10³/10⁶）与 MB/GB（10⁻³）。**不含 min↔s（×60 / ÷60）**——
#: 那会让 `wall_s × 60` 偶然放行一个根本不是派生量的分钟数（复核实测：正文 `≈104 min`
#: 曾被 `wall_s: 1.7359 × 60 = 104.154` 放行）。秒↔分换算改由 `derived_values()` 显式列出。
SCALES = (1e-6, 1e-3, 1e3, 1e6)


def numbers_in(obj, out: list[float]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            numbers_in(v, out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out.append(float(obj))


def evidence_values() -> list[float]:
    """允许的**证据**值（12k 步实测；60 步探针不进允许集）。"""
    vals: list[float] = []
    numbers_in(json.load(open(os.path.join(HERE, "results.json"))), vals)
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    cfg.pop("step_time_probe", None)          # 探针子树整块剔除（含 peak_gpu_mem_mb 1205.0）
    numbers_in(cfg, vals)
    for d in sorted(os.listdir(EV)):
        p = os.path.join(EV, d, "results.json")
        if os.path.exists(p):
            numbers_in(json.load(open(p)), vals)
        q = os.path.join(EV, d, "cycle_log.jsonl")
        if os.path.exists(q):
            with open(q) as f:
                for line in f:
                    try:
                        numbers_in(json.loads(line), vals)
                    except Exception:
                        pass
    return vals


def derived_values() -> list[float]:
    res = json.load(open(os.path.join(HERE, "results.json")))
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    b = res["battery"]
    arms = {"stock": "r7-stock-s0", "legacy": "r7-c1-legacy-s0", "fair": "r7-c1-fair-s0",
            "l1": "r7-c1-fair-l1-s0", "share": "r7-c1-fair-share-s0",
            "lsfair": "r7-ls-fair-s0", "lslegacy": "r7-ls-legacy-s0", "mod": "r7-mod-s0"}

    def d8(k):
        a = b[arms[k]]
        if "mod_eval" in a:
            return a["mod_eval"]["bpc"]
        return [x["bpc"] for x in a["depth_curve"] if x["depth"] == 8][0]

    def tf4(k):
        return [x["accept"] for x in b[arms[k]]["tf_acceptance"] if x["depth"] == 4][0]

    v = {k: d8(k) for k in arms}
    t = {k: tf4(k) for k in arms if k != "mod"}
    out: list[float] = []

    # —— 正文声明的**有向** d8 bpc 差：写成 `bpc(被减臂) − bpc(减臂)`（只写正文声明的方向，
    #    不写反向差或绝对值 ⇒ 方向写反会被抓到）——
    for a_, b_ in [("legacy", "stock"), ("fair", "stock"), ("lslegacy", "stock"),
                   ("legacy", "fair"), ("lslegacy", "lsfair"), ("fair", "lsfair"),
                   ("legacy", "lslegacy"), ("legacy", "share"), ("legacy", "l1"),
                   ("mod", "stock")]:
        out.append(round(v[a_] - v[b_], 4))
    # 交互项（difference-in-differences）；正文同时声明了它的「绝对量」
    inter = (v["legacy"] - v["lslegacy"]) - (v["fair"] - v["lsfair"])
    out += [round(inter, 4), round(-inter, 4)]
    # —— 正文声明的 tf@4 差（有向）——
    for a_, b_ in [("fair", "stock"), ("legacy", "lslegacy")]:
        out.append(round(t[a_] - t[b_], 4))
    # 参数量差 / 门参数量 / 路由参数量（结构性，但由证据重算；无门臂的「−526k」按正文的负号）
    out += [6962688 - 6436352, -(6962688 - 6436352), 8 * (256 * 256 + 256),
            13811216 - 6962688, 13811216 / 6962688, 6962688 / 6436352]
    # 接受率：窗口口径的差与 SE（差比例 SE、单比例 SE）
    aw = {k: b[arms[k]]["decode"]["spec_g4_pooled"]["accept_rate_window"]
          for k in ("legacy", "fair", "l1", "share", "lsfair", "lslegacy")}
    n = 81600
    for k, p in aw.items():
        out += [p, 1 - p]
    import math
    for p_ in aw.values():
        out += [2 * math.sqrt(p_ * (1 - p_) / n), math.sqrt(p_ * (1 - p_) / n),
                2 * math.sqrt(2 * p_ * (1 - p_) / n)]
    out += [2 * math.sqrt((aw["fair"] * (1 - aw["fair"]) + aw["lsfair"] * (1 - aw["lsfair"])) / n),
            aw["legacy"] - aw["lslegacy"]]
    # C1 的 2621 窗口率与本波之差（正文 L229-230）
    out += [0.9423 - aw["legacy"], (0.9423 - aw["legacy"]) / 0.0153]
    # 逐 prompt 层×位置的池化均值（正文按均值报）
    for k in ("legacy", "fair", "lsfair", "lslegacy"):
        for g in ("1", "2", "4", "8"):
            rows = b[arms[k]]["decode"]["spec"][g]
            if rows:
                out.append(sum(r["layerpos_per_token"] for r in rows) / len(rows))
                out.append(sum(r["match_full_frac"] for r in rows) / len(rows))
    # 墙钟：比值与范围分数、tokens/s
    w = res["wall_clock_interleaved"]["summary"]
    for k, s in w.items():
        out += [s["tok_s_median"], s.get("ratio_median") or 0, s["tok_s_range_frac"],
                s["layerpos_per_token_median"], s.get("ratio_min") or 0,
                s.get("ratio_max") or 0]
    mf = res["mod_forward_throughput"]["summary"]
    out += [mf["r7-stock-s0"]["tok_per_s_median"], mf["r7-mod-s0"]["tok_per_s_median"],
            mf["r7-mod-s0"]["tok_per_s_median"] / mf["r7-stock-s0"]["tok_per_s_median"]]
    # GPU 分钟：训练墙钟列**只取 `results.json::battery.*.train_wall_minutes` 一个来源**
    # （84.02）；探针计时 / 电池+墙钟 / 丢弃 run 由 `config.json::gpu_wall_used_min` 台账换算。
    used = cfg["gpu_wall_used_min"]
    train = sum(a["train_wall_minutes"] for a in b.values())
    probe = used["step_time_probe"]
    extra = used["battery+wall(interleaved)+insertions"]
    disc = (used["discarded_spec_accident_L4_run_min"]
            + used["discarded_spec_accident_second_run_min"])
    out += [train, probe, extra, train + probe + extra, disc,
            train + probe + extra + disc,
            round(used["discarded_spec_accident_L4_run_min"], 1),          # 4.5 min
            round(used["discarded_spec_accident_L4_run_min"] * 60),        # 269 s
            round(used["discarded_spec_accident_second_run_min"] * 60)]     # 280 s
    # exit4 / calm 的层×位置与一致率（正文表格）
    for k in arms:
        dec = b[arms[k]].get("decode") or {}
        for r in dec.get("exit4", []):
            out += [r["layerpos_per_token"], r["match_full_frac"]]
        for r in dec.get("calm_eps0.05", []):
            out += [r["mean_exit_depth"], r["layerpos_per_token"], r["match_full_frac"]]
    return [x for x in out if x == x]


def _allowed_table(allowed: list[float]) -> dict[int, set]:
    table: dict[int, set] = {}
    for v in allowed:
        for d in range(0, 6):
            table.setdefault(d, set()).add(round(v, d))
        for scale in SCALES:
            vv = v * scale
            for d in range(0, 6):
                table.setdefault(d, set()).add(round(vv, d))
    return table


def _scan(text: str, allowed: list[float]) -> list[tuple]:
    table = _allowed_table(allowed)
    flags = []
    for lineno, line in enumerate(text.splitlines(), 1):
        # 负号先行归一（正文用 `−` U+2212；不归一则在扫描时看不到符号 ⇒ 方向写反抓不到）。
        # 破折号（`–` U+2013，如“1.76–2.01”）**不**归一（它不是符号）。
        line_n = line.replace("\u2212", "-")
        line_n = re.sub(r"(?<=\d),(?=\d{3}\b)", "", line_n)      # 千分位合并
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", line_n):
            tok = m.group(0)
            val = float(tok)
            d = len(tok.split(".")[1]) if "." in tok else 0
            if round(val, d) in table.get(d, set()):
                continue
            start = m.start()
            prev = line_n[start - 1] if start > 0 else ""
            if prev and prev.isascii() and prev.isalnum():
                continue        # 标识符内的连字符：seed-1 / top-128 / P1-4
            if prev == "_":
                continue
            ctx = line_n[max(0, start - 20):start]
            if re.search(r"(WI|ticket|SCHEMA`? §|RTX |top|P[0-9]|C[0-9]+|A[0-9]+)\s*-?\s*$",
                         ctx):
                continue
            if tok.lstrip("+-") in {"2621", "2623", "2630", "2631", "2632", "2633",
                                     "2634", "2635", "2636", "2637", "16"}:
                continue        # 实验编号 / 工单号（非测量值）
            flags.append((lineno, tok, line.strip()[:100]))
    return flags


def selftest() -> int:
    """**证明本审计会红**：往正文里注入三个值，审计必须分别报出——
    `1.9641`（当年那个编造的 bpc）、`104`（当年被 `wall_s × 60` 假匹配放行的墙钟分钟，
    验证量纲碰撞已堵）、`1205`（60 步探针的峰值显存，验证探针值不再掩蔽 12k 步实测）。"""
    allowed = evidence_values() + derived_values() + ANCHORS
    base = open(ANA).read()
    bad = 0
    for val in ("1.9641", "104", "1205"):
        text = base + f"\n\n<!-- selftest 注入未入账值 --> A={val} B={val}\n"
        flags = _scan(text, allowed)
        hit = [f for f in flags if f[1] == val]
        if not hit:
            print(f"FAIL(selftest)：注入值 {val} 没有被审计抓到 ⇒ 审计无效")
            bad += 1
        else:
            print(f"OK(selftest)：{val} 被抓到 ⇒ L{hit[0][0]}: {hit[0][2][:70]}")
    return 1 if bad else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    text = open(ANA).read()
    allowed = evidence_values() + derived_values() + ANCHORS
    flags = _scan(text, allowed)
    if flags:
        print(f"待人工核对：{len(flags)} 处")
        for lineno, tok, line in flags:
            print(f"  L{lineno}  {tok:>12}  | {line}")
        return 1
    print(f"OK：analysis.md 全部数字 token 均可追到证据/派生/口径常数（允许值 {len(allowed)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
