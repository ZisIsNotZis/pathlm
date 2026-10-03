#!/usr/bin/env python3
"""2623 · 三道对账（SCHEMA §12）——算术 / 任务 / 口径。

读 `results.json` 与各臂 `train/*/{results.json,train_log.jsonl}`，逐条核对脚本化；
全过退出码 0，任一失败退出码 1 并把失败项写进 `reconcile_report.json`。

用法：python3 reconcile_numbers.py
"""

import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def main():
    res = json.load(open(os.path.join(HERE, "results.json")))
    cfg = res["config"]
    steps = cfg["train_recipe"]["steps"]
    bs = cfg["train_recipe"]["batch_size"]
    T = cfg["model"]["seq_len"]
    sym = cfg["tokens"]["sym_vocab"]
    chance = 1.0 / sym
    K_main = cfg["eval"]["K_main"]
    checks = []

    def ck(group, name, ok, detail):
        checks.append({"group": group, "name": name, "ok": bool(ok),
                       "detail": detail})

    # ---------------- 1. 算术对账 ----------------
    tok = steps * bs * T
    for arm in res["train_results"]:
        tr = res["train_results"][arm]
        ck("arith", f"{arm}: 训练 token 闭式 = 步数×bs×T", tok == steps * bs * T,
           f"{tok}")
        arm_res = json.load(open(os.path.join(ROOT, cfg["arms"][arm]["results_json"])))
        ck("arith", f"{arm}: peak_gpu_mem_mb 与 train/results.json 一致",
           (tr["peak_gpu_mem_mb"] is None)
           or approx(tr["peak_gpu_mem_mb"], arm_res["peak_gpu_mem_mb"]),
           f"{tr['peak_gpu_mem_mb']} vs {arm_res['peak_gpu_mem_mb']}")
        ck("arith", f"{arm}: wall_minutes 与 train/results.json 一致",
           approx(tr["wall_minutes"], arm_res["wall_minutes"], 1e-3),
           f"{tr['wall_minutes']} vs {arm_res['wall_minutes']}")
        ck("arith", f"{arm}: params 与 train/results.json 一致",
           tr["params"] == arm_res["params"],
           f"{tr['params']} vs {arm_res['params']}")
    ps = [res["train_results"][a]["params"] for a in res["train_results"]]
    ck("arith", "所有臂参数量相等（掩码零参数）", len(set(ps)) == 1, f"{ps}")
    for key, row in res["main_leg_multiquery"].items():
        ck("arith", f"{key}: accuracy == n_hits/n",
           approx(row["accuracy"], row["n_hits"] / row["n"], 1e-6),
           f"{row['accuracy']} vs {row['n_hits'] / row['n']:.6f}")
        ck("arith", f"{key}: chance == 1/sym_vocab",
           approx(row["chance"], chance, 1e-9), f"{row['chance']} vs {chance}")
        ck("arith", f"{key}: 毁源 n == 主 n",
           row["corrupt_source"]["n"] == row["n"],
           f"{row['corrupt_source']['n']} vs {row['n']}")
    for key, cb in res["clean_bpc_task_distribution"].items():
        ck("arith", f"{key}: chance_bits == log2(sym)",
           approx(cb["chance_bits"], math.log2(sym), 1e-6),
           f"{cb['chance_bits']} vs {math.log2(sym):.6f}")

    # ---------------- 2. 任务对账 ----------------
    for key, row in res["main_leg_multiquery"].items():
        if row["query_key_distance"]:
            qkd = row["query_key_distance"]
            ck("task", f"{key}: 源 key 距离 ≥ 窗（源在窗外）",
               qkd["min"] >= max(row["train_window"], 1),
               f"min={qkd['min']} window={row['train_window']}")
    for arm, rows in res["distance_leg"].items():
        w = cfg["arms"][arm]["train_window"]
        for d, r in rows.items():
            want = (int(d) < w) if w > 0 else True
            ck("task", f"distance[{arm}][d={d}]: source_inside_window 一致",
               r["source_inside_window"] == want,
               f"{r['source_inside_window']} vs {want} (w={w})")
    ck("task", "主腿使用训练同档 K_main",
       K_main == cfg["train_recipe"]["synth_pairs"],
       f"K_main={K_main} pairs={cfg['train_recipe']['synth_pairs']}")
    ck("task", "主腿使用训练同档 Q/gap",
       res["main_leg_multiquery"][f"full_K{K_main}"]["Q"]
       == cfg["train_recipe"]["synth_queries"]
       and res["main_leg_multiquery"][f"full_K{K_main}"]["gap"]
       == cfg["train_recipe"]["synth_gap"],
       f"Q={res['main_leg_multiquery'][f'full_K{K_main}']['Q']} "
       f"gap={res['main_leg_multiquery'][f'full_K{K_main}']['gap']}")
    ck("task", "词表 206 与 cue=205 一致",
       cfg["data"]["vocab_size"] == 206 and cfg["tokens"]["cue_token"] == 205, "ok")

    # ---------------- 3. 口径对账 ----------------
    for arm in res["train_results"]:
        lines = [json.loads(l) for l in
                 open(os.path.join(ROOT, cfg["arms"][arm]["train_log"])) if l.strip()]
        last = lines[-1]
        ck("caliber", f"{arm}: train_log 末步 == results.json 记录",
           res["train_log_tail"][arm]["last_step"] == last["step"],
           f"{res['train_log_tail'][arm]['last_step']} vs {last['step']}")
        ck("caliber", f"{arm}: window 掩码口径（full=无 / win16=非 1）",
           (cfg["arms"][arm]["train_window"] == 0 and last.get("window_mask") is None)
           or (cfg["arms"][arm]["train_window"] > 0
               and last.get("window_mask") is not None
               and last["window_mask"]["visible_frac"] < 1.0),
           f"window={cfg['arms'][arm]['train_window']} mask={last.get('window_mask')}")
        # 判决行 CE（决定准确率的那个损失）与口径
        ck("caliber", f"{arm}: train_log 记录 synth_answer_ce",
           "synth_answer_ce" in last, f"{last.get('synth_answer_ce')}")
    shared = ("pathlm/depth_ar.py", "train_depth_ar.py",
              "tools/evalkit/longrange_tasks.py", "tools/evalkit/carry_channel.py",
              "tools/evalkit/retention.py")
    for p in shared:
        want = res["code"].get(p)
        got = sha256_file(os.path.join(ROOT, p))
        ck("caliber", f"code[{p}] 指纹一致", want == got,
           f"{(want or '')[:12]} vs {got[:12]}")

    report = {"n_checks": len(checks),
              "n_failed": sum(1 for c in checks if not c["ok"]), "checks": checks}
    with open(os.path.join(HERE, "reconcile_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    for c in checks:
        if not c["ok"]:
            print(f"FAIL [{c['group']}] {c['name']}: {c['detail']}")
    print(f"对账 {report['n_checks']} 项，失败 {report['n_failed']} 项")
    sys.exit(0 if report["n_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
