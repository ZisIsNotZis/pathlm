"""2613 · 稳定性专报生成器（stability.md 的数据源）。

读 `{arm}-train_log.jsonl`（每 200 步采样点）与 `results.json`（全程 gnorm_monitor、
逐深度 ECE），产出 `stability.md`。两臂同一口径。

用法：python3 stability.py
"""

import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["baseline", "linres"]


def arm_stats(arm: str, results: dict) -> dict:
    rows = [json.loads(l) for l in open(os.path.join(HERE, f"{arm}-s0-train_log.jsonl"))
            if l.strip()]
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    finite = all(x == x and abs(x) != float("inf")
                 for x in g + loss + [c for r in rows for c in r["depth_ce"]])
    monitor = results.get(f"{arm}_arm", {}).get("gnorm_monitor") or {}
    return {
        "n_logged_samples": len(rows),
        "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
        "gnorm_spikes_gt3x_median": sum(1 for x in g if x > 3 * med),
        "gnorm_spikes_gt5x_median": sum(1 for x in g if x > 5 * med),
        "all_finite": finite,
        "loss_first": loss[0], "loss_last": loss[-1],
        "loss_max_step_increase": round(max(inc), 4),
        "loss_jumps_gt_0.25": sum(1 for x in inc if x > 0.25),
        "gnorm_monitor_all_steps": monitor,
    }


def main():
    results = json.load(open(os.path.join(HERE, "results.json")))
    stats = {a: arm_stats(a, results) for a in ARMS}
    ece = {a: results.get(f"{a}_arm", {}).get("calibration", {}).get("ece_by_depth")
           for a in ARMS}
    json.dump({"by_arm": stats, "ece_by_depth": ece},
              open(os.path.join(HERE, "stability.json"), "w"),
              indent=2, ensure_ascii=False)

    def row(label, key):
        return (f"| {label} | {stats['baseline'][key]} | {stats['linres'][key]} |")

    lines = [
        "# 2613 · 读出重参数化在 dense 上的可迁移性 —— 稳定性专报",
        "",
        "> 数据源：`{arm}-s0-train_log.jsonl`（seed 0，每 200 步采样点）与 `results.json`",
        "> （`gnorm_monitor` = 全程 12000 步真值；逐深度 ECE 为训练结束后的 8-batch 评测）。",
        "> 口径分清：`gnorm_logged_*` 只是采样点，`gnorm_monitor_all_steps` 是全程真值。",
        "",
        "## 1. 梯度 / 数值稳定性（两臂同口径）",
        "",
        "| 量 | baseline | linres |",
        "|---|---|---|",
        row("采样点数", "n_logged_samples"),
        f"| 采样点 step 范围 | {stats['baseline']['step_first']}–{stats['baseline']['step_last']} | {stats['linres']['step_first']}–{stats['linres']['step_last']} |",
        row("gnorm 采样点 max", "gnorm_logged_max"),
        row("gnorm 采样点中位数", "gnorm_logged_median"),
        row("gnorm 尖峰（>3×中位数）次数", "gnorm_spikes_gt3x_median"),
        row("gnorm 尖峰（>5×中位数）次数", "gnorm_spikes_gt5x_median"),
        row("全部有限（无 NaN/Inf）", "all_finite"),
        row("loss 首采样点", "loss_first"),
        row("loss 末采样点", "loss_last"),
        row("相邻采样点 loss 最大上升", "loss_max_step_increase"),
        row("loss 突跳（上升 >0.25）次数", "loss_jumps_gt_0.25"),
        "",
        "全程 gnorm 监控（12000 步真值，含梯度跳步数）：",
        "",
        "| 臂 | gnorm_max | gnorm_mean | grad_skips |",
        "|---|---|---|---|",
    ]
    for a in ARMS:
        m = stats[a]["gnorm_monitor_all_steps"]
        lines.append(f"| {a} | {m.get('gnorm_max')} | {m.get('gnorm_mean')} | "
                     f"{m.get('grad_skips')} |")
    lines += [
        "",
        "## 2. 逐深度 ECE（prob0 校准，越低越好）",
        "",
        "| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in ARMS:
        e = ece[a]
        lines.append(f"| {a} | " + " | ".join(str(x) for x in e) + " |")
    lines += [
        "",
        "> dense 形态只有最终深度（depth 8）被直接监督，浅层读出的 ECE 属探索性量。",
        "",
    ]
    open(os.path.join(HERE, "stability.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
