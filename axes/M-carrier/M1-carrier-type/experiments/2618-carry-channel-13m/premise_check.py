"""2618 · 判据 ① 的前提校验（单独可跑）：U0 在关联回忆上是否真的 ≈ chance。

为什么单拎出来：判据 ① 是**前提校验**，不是结论——若 U0（无通道 + 窗口 16）显著高于
chance，说明「不搬就答不出」这个需求没被制造出来（窗口没生效 / 任务有漏洞 / 距离不够），
那么 U1/U2 的读数不可判。所以它必须在看 U1/U2 之前先报出来。

跑四件事：
  ① U0 在判决任务（单查询关联回忆，K=8）上的准确率 + 二项 95% CI + chance；
  ② **查询-key 距离分布**（确认被问的 pair 落在窗口外）；
  ③ 局部先验基线（方法学 §5 的最强局部预测器）；
  ④ 打乱答案的变异验证（读数确实挂在上下文上）。

用法：python3 premise_check.py [--arm U0] [--config config.json]
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import window_anchor_keep_mask                # noqa: E402
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    assoc_recall_batch, accuracy, local_prior_accuracy, shuffle_answers,
    corrupt_span)
from tools.evalkit.carry_channel import wilson_ci                  # noqa: E402
from run import load_arm                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="U0")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "premise_check.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    bs, nb = (16, 1) if args.smoke else (64, 64)
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    sym, fill, cue = (cfg["tokens"]["sym_vocab"], cfg["tokens"]["filler_token"],
                      cfg["tokens"]["cue_token"])
    model, ck = load_arm(cfg["arms"][args.arm], cfg["model"],
                         cfg["data"]["vocab_size"], cfg["device"])
    keep = window_anchor_keep_mask(T, window, anchors)
    K = 8
    agg = {"n": 0, "hits": 0, "cn": 0, "chits": 0, "sn": 0, "shits": 0}
    dists, lp = [], None
    for i in range(nb):
        tb = assoc_recall_batch(bs, T, K, sym_vocab=sym, filler_token=fill,
                                cue_token=cue,
                                generator=torch.Generator().manual_seed(1000 + 7 * K + i))
        r = accuracy(model, tb, keep_mask=keep, device=cfg["device"])
        agg["n"] += r["n"]
        agg["hits"] += r["n_hits"]
        c = accuracy(model, corrupt_span(tb, 0, 2 * K,
                                         generator=torch.Generator().manual_seed(i)),
                     keep_mask=keep, device=cfg["device"])
        agg["cn"] += c["n"]
        agg["chits"] += c["n_hits"]
        s = accuracy(model, shuffle_answers(tb, generator=torch.Generator().manual_seed(i)),
                     keep_mask=keep, device=cfg["device"])
        agg["sn"] += s["n"]
        agg["shits"] += s["n_hits"]
        dists.extend(int(v) for v in tb.dists.flatten().tolist())
        lp = local_prior_accuracy(tb, k=8)["accuracy"]
    dists.sort()
    out = {
        "arm": args.arm, "checkpoint": ck, "task": "assoc_recall", "K": K,
        "n_pairs": K, "n_queries_per_row": 1, "window": window,
        "chance_uniform_answer_set": round(1.0 / sym, 8),
        "model_accuracy": round(agg["hits"] / agg["n"], 6),
        "model_accuracy_ci_95": wilson_ci(agg["hits"], agg["n"]),
        "n": agg["n"],
        "query_key_distance": {
            "min": dists[0], "median": dists[len(dists) // 2], "max": dists[-1],
            "frac_outside_window": round(sum(1 for d in dists if d >= window)
                                         / len(dists), 6),
            "n": len(dists)},
        "local_prior_accuracy_k8": lp,
        "corrupt_source_accuracy": round(agg["chits"] / agg["cn"], 6),
        "corrupt_source_ci_95": wilson_ci(agg["chits"], agg["cn"]),
        "shuffled_answers_accuracy": round(agg["shits"] / agg["sn"], 6),
        "shuffled_answers_ci_95": wilson_ci(agg["shits"], agg["sn"]),
    }
    ci = out["model_accuracy_ci_95"]
    out["premise_pass_c1"] = ci[0] <= out["chance_uniform_answer_set"]
    out["note"] = ("判据 ①：U0（无通道 + 窗口 %d）的 CI 下界必须 ≤ chance=%.6f"
                   % (window, out["chance_uniform_answer_set"]))
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
