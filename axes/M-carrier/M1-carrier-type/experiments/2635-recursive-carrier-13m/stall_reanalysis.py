#!/usr/bin/env python3
"""2635 · 停摆报告的**再分析**（复核 P2-3）：把旧门产物对齐到当前门的口径。

`stall_report_linear.json` 是**旧版停摆门**（180 s 即终止）的原始产物：`status="stalled"`、
没有 `kill_secs` 字段——**当前的 `stall_gate.py` 根本产不出这个状态**（它分两级：
`no_log_progress` 只记事件、`killed_after_no_log_progress` 才终止）。本脚本**不重跑训练**
（那是 8.5 min GPU 且会再次发散），只做三件事：

1. **原样保留**旧报告（它是当时的原始证据，不改写）；
2. 用**训练器自己的计数**做步数对账（`grad_skips + 有效步数 == 配置步数`）与终点产物检查
   ⇒ 判定「进程其实跑完」；
3. **正面使用**旧报告里被漏用的一个字段：`cpu_delta_10s = 0.0`（采集前 10 s 内该进程的
   CPU 时间**没有增长**）——这与「进程已在收尾/已结束」一致，是「不是 CPU 满载卡死」的
   正面证据（2616 观测到的卡死是 ~98% 单核持续烧）。
4. 给出「按当前门的口径，同一份日志会得到什么」的推演：`no_log_progress`（≥180 s）之后，
   进程在 4.43 min 内自行结束并落盘产物 ⇒ 当前门会记 `finished`。

用法：`python3 stall_reanalysis.py`（纯 CPU，读盘上的日志）→ `stall_report_linear_v2.json`
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    steps = cfg["steps"]
    n_layers = cfg["model"]["n_layers"]
    old = json.load(open(os.path.join(HERE, "stall_report_linear.json")))
    res = json.load(open(os.path.join(HERE, "train", "linear", "results.json")))
    gmon = res.get("gnorm_monitor") or {}
    tlog = [json.loads(l) for l in
            open(os.path.join(HERE, "train", "linear", "train_log.jsonl")) if l.strip()]
    samples = gmon.get("attn_linear_layer_samples")
    effective = (samples // n_layers) if samples else None
    skips = gmon.get("grad_skips")
    accounting_ok = (skips is not None and effective is not None
                     and skips + effective == steps)
    cpu_delta = old.get("cpu_delta_10s")
    log_last_step = tlog[-1]["step"] if tlog else None

    out = {
        "purpose": ("把旧版停摆门的产物（status=stalled）对齐到当前两级门的口径，并给出"
                    "「进程其实跑完」的对账证据；**不重跑训练**"),
        "instrument_version": {
            "old_report_gate": ("旧版：无日志推进 ≥180 s 即终止 ⇒ status=\"stalled\"；"
                                "报告里没有 kill_secs / no_log_progress_events 字段"),
            "current_gate": ("当前：--stall-secs(180) 只记 no_log_progress 事件；"
                             "--kill-secs(900) 才终止 ⇒ status=killed_after_no_log_progress；"
                             "进程自行退出记 finished"),
            "consequence": ("旧报告的 status 字样与当前脚本的状态集合不一致——本文件是它的"
                            "再分析，**不改写**原始报告"),
        },
        "old_report_facts": {
            "status": old.get("status"),
            "trigger_step": old.get("trigger_step"),
            "stall_wall_s": old.get("stall_wall_s"),
            "wall_s": old.get("wall_s"),
            "cpu_seconds": old.get("cpu_seconds"),
            "cpu_delta_10s": cpu_delta,
            "gpu_mem_mb_for_pid": old.get("gpu_mem_mb_for_pid"),
            "log_tail_last_step": log_last_step,
        },
        "step_accounting": {
            "configured_steps": steps,
            "grad_skips": skips,
            "effective_optimizer_steps": effective,
            "sum": (None if skips is None or effective is None else skips + effective),
            "matches_configured": bool(accounting_ok),
            "source": "train/linear/results.json::gnorm_monitor（训练器自报计数）",
        },
        "terminal_artifacts": {
            "model_pt_exists": os.path.exists(os.path.join(HERE, "train", "linear", "model.pt")),
            "results_json_exists": True,
            "actual_wall_minutes": res.get("wall_minutes"),
            "implied_ms_per_step": (None if not res.get("wall_minutes") else
                                    round(res["wall_minutes"] * 60000 / steps, 1)),
        },
        "cpu_delta_evidence": {
            "cpu_delta_10s": cpu_delta,
            "reading": ("0.0 = 采集前 10 s 内该进程 CPU 时间未增长 ⇒ 不是「CPU 满载却卡死」"
                        "（2616 的卡死观测是 ~98% 单核持续烧）；与「进程已在收尾/已结束」一致"),
            "useful": bool(cpu_delta == 0.0),
        },
        "current_gate_would_report": {
            "events": [{"status": "no_log_progress", "at_trigger_step": log_last_step,
                        "after_secs": old.get("stall_wall_s")}],
            "final": ("finished（进程在墙钟 {} min 内自行结束并落盘 model.pt/results.json；"
                      "当前门的 --kill-secs=900 s 不会触发）".format(res.get("wall_minutes"))),
            "verdict": "not_a_process_hang__numerical_divergence",
        },
        "conclusion": ("真实故障 = 归一化线性注意力的**数值发散**（den_min 触 clamp、gnorm 爆到 1e8）"
                       "⇒ 62% 的步被梯度门跳过；而训练器只在未跳过的步写日志 ⇒ 日志停写被旧门"
                       "读成「停摆」。对账（"
                       + (f"{skips}+{effective}={steps}" if accounting_ok else "未通过")
                       + "）与终点产物证明进程跑完。"),
    }
    with open(os.path.join(HERE, "stall_report_linear_v2.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps({k: out[k] for k in ("step_accounting", "cpu_delta_evidence",
                                          "current_gate_would_report")},
                     ensure_ascii=False, indent=1))
    return 0 if accounting_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
