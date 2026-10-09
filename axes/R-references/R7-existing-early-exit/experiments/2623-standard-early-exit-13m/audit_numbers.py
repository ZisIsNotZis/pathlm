"""`analysis.md` 里**每个数字**都要能追到落盘证据或已声明的算式（可失败审计）。

动机（复核 P1-A 的「顺手」要求）：`fill_share.py` 只覆盖**标记位**，正文里手打的数字
不在其覆盖内——`1.9641` 这种错值就是从这个缝里漏出去的。本脚本取代「靠标记覆盖」的做法：
把正文的每个数字 token 拿去和「证据 + 派生值 + 口径常数白名单」比对，**未覆盖者逐条列出**。

覆盖来源：
1. 证据：主交付 `results.json`、8 条臂的逐臂 `results.json`、`config.json`、`step_time.json`；
2. **派生**：按正文声明的算式重算（bpc 差 / tf 差 / 交互项 / 比值 / 均值 / SE 公式 / 参数量差 /
   GPU 分钟求和 / 事件计数差等）；
3. `ANCHORS`：外部可核的历史入账值（23d/23c/2621 等）与仓内口径常数（0.0138 / 0.014）。

数值比较按 **token 自身的小数位数**四舍五入；带 `k`/`M` 后缀按 10³/10⁶ 折算；
千分位逗号先合并。用法：python3 audit_numbers.py
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

# 外部可核的历史入账值 / 仓内口径常数 / 已声明算式之外的常数
ANCHORS = [
    1.5287, 1.8582, 0.9694, 0.3295,            # 23d-dense-B-12k / 23c-H2-w0-12k 对照
    0.9423, 8.45, 1.86, 169, 200,              # C1 的 2621（历史测量）
    0.0138, 0.014, 0.0003, 0.0007, 0.057,      # 方差带与"无出处的旧 σ_tf"（用于说明，不参与判据）
    0.5, 0.1, 1.0, 8.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.3,  # 剂量/阈值等超参
    0.4362,                                     # C1 在 100M 的历史形态税（未重拆，原样引用）
    3.0, 6.8, 83, 87, 100, 128, 256, 512, 12000, 98304000,   # 步数/层数/top-k/位置数
]


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
    vals: list[float] = []
    for p in [os.path.join(HERE, "results.json"), os.path.join(HERE, "config.json"),
              os.path.join(ROOT, ".scratch", "52-r7-early-exit", "step_time.json")]:
        if os.path.exists(p):
            numbers_in(json.load(open(p)), vals)
    for d in sorted(os.listdir(EV)):
        p = os.path.join(EV, d, "results.json")
        if os.path.exists(p):
            numbers_in(json.load(open(p)), vals)
        for name in ("cycle_log.jsonl",):
            q = os.path.join(EV, d, name)
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
    pairs = [("legacy", "stock"), ("fair", "stock"), ("lsfair", "stock"),
             ("lslegacy", "stock"), ("legacy", "fair"), ("lslegacy", "lsfair"),
             ("legacy", "l1"), ("fair", "lsfair"), ("legacy", "lslegacy"),
             ("legacy", "share"), ("share", "stock"), ("mod", "stock"),
             ("share", "fair"), ("l1", "stock"), ("mod", "legacy")]
    for a_, b_ in pairs:
        out += [v[a_] - v[b_], v[b_] - v[a_], abs(v[a_] - v[b_])]
    for a_, b_ in [("fair", "stock"), ("legacy", "stock"), ("lsfair", "stock"),
                   ("legacy", "lsfair"), ("legacy", "lslegacy"), ("fair", "lsfair"),
                   ("lslegacy", "lsfair"), ("share", "legacy"), ("share", "fair"),
                   ("l1", "legacy"), ("l1", "fair")]:
        out += [t[a_] - t[b_], t[b_] - t[a_], abs(t[a_] - t[b_])]
    inter = (v["legacy"] - v["lslegacy"]) - (v["fair"] - v["lsfair"])
    out += [inter, abs(inter)]
    # 参数量差 / 门参数量 / 路由参数量（结构性，但由证据重算）
    out += [6962688 - 6436352, 8 * (256 * 256 + 256), 13811216 - 6962688,
            13811216 / 6962688, 6962688 / 6436352]
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
    # GPU 分钟求和（正文 L96 写 85.2 / 104 / 113）
    used = cfg["gpu_wall_used_min"]
    training = sum(v_ for k, v_ in used.items() if k.startswith("r7-"))
    out += [training, used["step_time_probe"] + training,
            used["discarded_spec_accident_L4_run_min"]
            + used["discarded_spec_accident_second_run_min"],
            used["total_incl_discarded"], used["battery+wall(interleaved)+insertions"]]
    # exit4 / calm 的层×位置与一致率（正文表格）
    for k in arms:
        dec = b[arms[k]].get("decode") or {}
        for r in dec.get("exit4", []):
            out += [r["layerpos_per_token"], r["match_full_frac"]]
        for r in dec.get("calm_eps0.05", []):
            out += [r["mean_exit_depth"], r["layerpos_per_token"], r["match_full_frac"]]
    return [x for x in out if x == x]


def main() -> int:
    text = open(ANA).read()
    allowed = evidence_values() + derived_values() + ANCHORS
    table: dict[int, set] = {}
    for v in allowed:
        for d in range(0, 6):
            table.setdefault(d, set()).add(round(v, d))
        for scale in (1e-6, 1e-3, 1e3, 1e6, 60.0, 1 / 60.0):  # k/M 后缀、千分位、分↔秒
            vv = v * scale
            for d in range(0, 6):
                table.setdefault(d, set()).add(round(vv, d))
    flags = []
    for lineno, line in enumerate(text.splitlines(), 1):
        line_n = re.sub(r"(?<=\d),(?=\d{3}\b)", "", line)      # 千分位合并
        for m in re.finditer(r"[-+]?\d+(?:\.\d+)?", line_n):
            tok = m.group(0)
            val = float(tok)
            d = len(tok.split(".")[1]) if "." in tok else 0
            if round(val, d) in table.get(d, set()):
                continue
            # 标识符（实验编号 / 工单号 / 型号 / SCHEMA 节号）不是测量值：按上下文跳过
            start = m.start()
            ctx = line_n[max(0, start - 20):start]
            # 注：`-128` / `-6` 这类 token 会把前缀的连字符一起吞掉，故也允许以
            # "top" / "P1" / "WI" 等**标识符词干**结尾即跳过
            if re.search(r"(WI|ticket|SCHEMA`? §|RTX |top|P[0-9]|C[0-9]+|A[0-9]+)\s*-?\s*$",
                         ctx):
                continue
            if tok.lstrip("+-") in {"2621", "2623", "2630", "2631", "2632", "2633",
                                     "2634", "2635", "2636", "2637", "16"}:
                continue        # 实验编号 / 工单号（非测量值）
            flags.append((lineno, tok, line.strip()[:100]))
    if flags:
        print(f"待人工核对：{len(flags)} 处")
        for lineno, tok, line in flags:
            print(f"  L{lineno}  {tok:>12}  | {line}")
        return 1
    print(f"OK：analysis.md 全部数字 token 均可追到证据/派生/口径常数（允许值 {len(allowed)} 个）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
