"""2618 · 重跑散度（判据 ⑥ 要求）：同配置、同 seed 重跑一臂，量训练侧散度。

背景（**本实验实测到的、必须如实报告的一条**）：同 seed 同配置的两次训练**不可逐位
复现**（项目级已知事实，见 `harness/methodology.md` §2 与 2615 的 `stability.md`），
本实验还观测到它的**后果被放大**：U1（plain）第一次跑在 step 3600 的答案行 CE 已降到
3.5015（通道真的在学），重跑同配置却停在 4.19（≈ 边际 4.1589，没学到）。
因此「重跑散度」不是形式动作，而是**这条结论的主要不确定性来源**。

本脚本把两臂用**同一套评测协议**（窗口掩码一致、同 seed 任务批、含毁源对照）各评一遍，
并给出差值的二项 CI；同时从两份训练日志里抽出答案行 CE 的轨迹与末值。

用法：python3 divergence.py [--arm U1 --rep U1rep] [--config config.json]
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import window_anchor_keep_mask                  # noqa: E402
from tools.evalkit.longrange_tasks import assoc_recall_batch         # noqa: E402
from tools.evalkit.carry_channel import (wilson_ci, two_proportion,  # noqa: E402
                                         arm_battery)
from run import load_arm                                             # noqa: E402


def answer_ce_track(log_path: str) -> dict:
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    return {"steps": [r["step"] for r in rows],
            "answer_ce": [r.get("synth_answer_ce") for r in rows],
            "loss": [r["loss"] for r in rows],
            "last_answer_ce": rows[-1].get("synth_answer_ce"),
            "min_answer_ce": min([r["synth_answer_ce"] for r in rows
                                  if r.get("synth_answer_ce") is not None]),
            "chance_ce": rows[-1].get("synth_chance_ce")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="U1")
    ap.add_argument("--rep", default="U1rep")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "divergence.json"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=64)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    keep = window_anchor_keep_mask(T, window, anchors)
    K = cfg["K_main"]
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    out = {"protocol": {"task": "assoc_recall", "K": K, "window": window,
                        "batch_size": args.batch_size, "n_batches": args.n_batches,
                        "note": "两臂同一套任务批（同 seed）、同一窗口掩码、含毁源对照"},
           "arms": {}}
    stats = {}
    for name, spec in ((args.arm, cfg["arms"][args.arm]),
                       (args.rep, cfg["arms"].get(args.rep, {}))):
        if not spec:
            print(f"{name}: config 里没有该臂（跳过）")
            continue
        model, ck = load_arm(spec, cfg["model"], cfg["data"]["vocab_size"],
                             cfg["device"])
        agg = {"n": 0, "hits": 0, "cn": 0, "chits": 0}
        for i in range(args.n_batches):
            tb = assoc_recall_batch(args.batch_size, T, K, sym_vocab=sym,
                                    filler_token=fill, cue_token=cue,
                                    generator=torch.Generator().manual_seed(1000 + 7 * K + i))
            r = arm_battery(model, tb, window, anchors, cfg["device"],
                            corrupt_span_bounds=(0, 2 * K))
            agg["n"] += r["n"]; agg["hits"] += r["n_hits"]
            agg["cn"] += r["corrupt_source"]["n"]
            agg["chits"] += r["corrupt_source"]["n_hits"]
        stats[name] = agg
        out["arms"][name] = {
            "checkpoint": ck, "n": agg["n"], "n_hits": agg["hits"],
            "accuracy": round(agg["hits"] / agg["n"], 6),
            "accuracy_ci": wilson_ci(agg["hits"], agg["n"]),
            "corrupt_source": {"n": agg["cn"], "n_hits": agg["chits"],
                               "accuracy": round(agg["chits"] / agg["cn"], 6),
                               "accuracy_ci": wilson_ci(agg["chits"], agg["cn"])},
            "answer_ce_track": answer_ce_track(os.path.join(
                HERE, "train", name, "train_log.jsonl")),
        }
        print(f"{name}: acc={out['arms'][name]['accuracy']:.4f} "
              f"last_answer_ce={out['arms'][name]['answer_ce_track']['last_answer_ce']} "
              f"min_answer_ce={out['arms'][name]['answer_ce_track']['min_answer_ce']}")
    if len(stats) == 2:
        a, b = stats[args.arm], stats[args.rep]
        out["rerun_comparison"] = two_proportion(a["hits"], a["n"], b["hits"], b["n"])
        out["rerun_comparison"]["answer_ce_gap"] = round(
            (out["arms"][args.arm]["answer_ce_track"]["min_answer_ce"]
             - out["arms"][args.rep]["answer_ce_track"]["min_answer_ce"]), 4)
        print(json.dumps(out["rerun_comparison"], indent=2, ensure_ascii=False))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
