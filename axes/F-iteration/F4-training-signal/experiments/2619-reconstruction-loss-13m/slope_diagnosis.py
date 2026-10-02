"""2619 · 辅助对账：**同量**的训练 loss 斜率（SCHEMA §16）。

`results.json::loss_slope` 用的是日志里的 `loss` 字段：
- `A`（λ=0）的 `loss` **就是** 主干 CE（单深度 dense ⇒ `depth_ce[-1]` 逐位相同）；
- `B`/`B_rep` 的 `loss` = 主干 CE + λ·重建 CE（重建项趋 0）⇒ **与 A 不同量**。

把「B 的 loss 下降更快」直接读成「辅助任务加速了主任务」是错的。本脚本按
**逐行 `depth_ce[-1]`**（主干 CE，三臂同一量、同一窗口、同一拟合方法）重算斜率，
与总 loss 斜率并列落盘 `slope_report.json`。

用法：python3 slope_diagnosis.py      # 纯 CPU，不取 GPU 锁
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("A", "B", "B_rep")


def track(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()]


def loglog_slope(rows, key, lo=200, hi=12000):
    xs, ys = [], []
    for r in rows:
        if not lo <= r["step"] <= hi:
            continue
        v = r[key]
        if key == "depth_ce":
            v = v[-1]
        if v is None or v <= 0:
            continue
        xs.append(math.log(max(r["step"], 1)))
        ys.append(math.log(v))
    n = len(xs)
    if n < 3:
        return {"n_points": n, "slope": None}
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = sum((x - mx) ** 2 for x in xs)
    return {"n_points": n, "slope": round(cov / var if var else 0.0, 6),
            "first": round(ys[0], 5), "last": round(ys[-1], 5),
            "window_steps": [lo, hi]}


out = {"note": ("同方法（末段 log-log 线性拟合，窗口 200–12000 步）在**同一物理量**"
                "（主干 CE = 每行的 depth_ce[-1]）上的斜率；总 loss 斜率见 "
                "results.json::loss_slope（B/B_rep 含 λ·重建项，量不同）"),
       "trunk_ce_slope": {}, "total_loss_slope": {}}
for a in ARMS:
    rows = track(os.path.join(HERE, "train", a, "train_log.jsonl"))
    out["trunk_ce_slope"][a] = loglog_slope(rows, "depth_ce")
    out["total_loss_slope"][a] = loglog_slope(rows, "loss")
    out["trunk_ce_ratio_B_over_A"] = None
ratios = {f"B/A": out["trunk_ce_slope"]["B"]["slope"] / out["trunk_ce_slope"]["A"]["slope"],
          f"B_rep/A": out["trunk_ce_slope"]["B_rep"]["slope"] / out["trunk_ce_slope"]["A"]["slope"]}
out["trunk_ce_ratio_B_over_A"] = {k: round(v, 4) for k, v in ratios.items()}
# 末段（最后 2000 步）的主干 CE 均值：快照是否「还在收敛」的直观量
for a in ARMS:
    rows = track(os.path.join(HERE, "train", a, "train_log.jsonl"))
    tail = [r["depth_ce"][-1] for r in rows if r["step"] >= 10000]
    early = [r["depth_ce"][-1] for r in rows if 200 <= r["step"] <= 2200]
    out.setdefault("trunk_ce_levels", {})[a] = {
        "first_logged": rows[0]["depth_ce"][-1], "last_logged": rows[-1]["depth_ce"][-1],
        "mean_200_2200": round(sum(early) / len(early), 5),
        "mean_10000_11999": round(sum(tail) / len(tail), 5)}
with open(os.path.join(HERE, "slope_report.json"), "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(json.dumps(out, ensure_ascii=False, indent=2))
