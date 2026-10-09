"""把 analysis.md 里 **同份额对照臂（`r7-c1-fair-share-s0`）** 的数字从落盘证据回填。

为什么要脚本（2026-10-10 复核教训）：我在该臂跑完前先在正文里写了它的数字——那是**编造**。
现在正文只放标记（`SHARE_TABLE` / `SHARE_ROW` / `SHARE_Q_ROW` / `SHARE_DELTA` / `SHARE_BCP`），
由本脚本从 `results.json` + 逐臂 `results.json` 生成；**数据缺失即非零退出**，
不可能留下编造值。用法：python3 fill_share.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
RES = os.path.join(HERE, "results.json")
ARM = "r7-c1-fair-share-s0"
EV = os.path.join(ROOT, ".scratch", "52-r7-early-exit", "evidence", ARM, "results.json")
ANA = os.path.join(HERE, "analysis.md")


def main() -> int:
    for p in (RES, EV):
        if not os.path.exists(p):
            print(f"FAIL：缺 {p}（同份额臂未落盘 ⇒ 不得填数）")
            return 1
    res = json.load(open(RES))
    ev = json.load(open(EV))
    b = res["battery"].get(ARM)
    if b is None:
        print(f"FAIL：results.json 的 battery 里没有 {ARM}（先跑该臂的电池腿）")
        return 1
    d8 = [x for x in b["depth_curve"] if x["depth"] == 8][0]
    dc = {x["depth"]: x["bpc"] for x in b["depth_curve"]}
    tf = {x["depth"]: x["accept"] for x in b["tf_acceptance"]}
    cj = b["calm_calibration"]["calm_joint"]
    legacy = [x for x in res["battery"]["r7-c1-legacy-s0"]["depth_curve"]
              if x["depth"] == 8][0]["bpc"]
    fair = [x for x in res["battery"]["r7-c1-fair-s0"]["depth_curve"]
            if x["depth"] == 8][0]["bpc"]
    vals = {
        "SHARE_TABLE": f"{d8['bpc']} / tf@4 {tf[4]}",
        "SHARE_Q_ROW": " | ".join(str(dc[k]) for k in (1, 2, 4, 6, 8)) +
                       f" | {tf[4]} | {tf[6]}",
        "SHARE_ROW": (f"d8 bpc **{d8['bpc']}** / tf@4 **{tf[4]}**"
                      f"（同份额下 additive 比 renorm 差 "
                      f"**{round(d8['bpc'] - legacy, 4)}**；训练墙钟 "
                      f"{b['train_wall_minutes']} min、峰值 {b['train_peak_gpu_mem_mb']} MB"),
        "SHARE_DELTA": (f"差 {round(legacy - d8['bpc'], 4):+}（同份额：renorm {legacy} vs "
                        f"additive λ=8 {d8['bpc']}）"),
        "SHARE_BCP": f"{d8['bpc']}",
        "SHARE_ARM_ROW": (f"{d8['bpc']} | {tf[4]} | {b['train_wall_minutes']} | "
                          f"{b['train_peak_gpu_mem_mb']} |"),
        "SHARE_CALM": (f"**{cj['0.05']['mean_exit_depth_holdout']}**"
                       f"（一致率 {cj['0.05']['agree_with_full_holdout']}；"
                       f"保守口径 {b['calm_calibration']['by_eps']['0.05']['mean_exit_depth_holdout']}）"),
    }
    s = open(ANA).read()
    present = {k: v for k, v in vals.items() if k in s}
    if not present:
        print(f"OK（幂等）：analysis.md 里已无标记（{ARM}：d8 {d8['bpc']} / tf@4 {tf[4]}）")
        return 0
    n = 0
    for key, val in present.items():
        n += s.count(key)
        s = s.replace(key, val)
    open(ANA, "w").write(s)
    left = [k for k in vals if k in s]
    print(f"OK：回填 {n} 处（{ARM}：d8 {d8['bpc']} / tf@4 {tf[4]}；参考："
          f"c1-legacy {legacy}、c1-fair {fair}）；剩余标记 {left}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
