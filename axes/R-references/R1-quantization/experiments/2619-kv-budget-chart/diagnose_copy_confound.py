"""2619 诊断脚本 · copy 任务的「毁源后仍高于 chance」混杂排查（只读，不训练）。

`run.py` 的主腿读数出现了一个必须解释的现象：
**把源片段整段换成随机符号后，copy 准确率没有落回 chance**
（fp16 臂：真源 0.1413 / 毁源 0.0990；win128 臂：真源 0.1100 / 毁源 0.1159）。
本脚本用三个决定性对照把「读数来自源信息」分离出来：

1. **逐 k 曲线**：按输出块内位置 k（0..31）分别报准确率。
   `copy_batch` 的 teacher-forced 输出块把 S 放进了输入（位置 `T-32..T-1`），
   所以 k 越大、可见的 S 前缀越长 —— 若读数随 k 升高，说明模型在用**输出块自身**
   （不是远处源片段）。
2. **毁两端**：把源区间 `[0,32)` **与**输出块 `[224,256)` 同时换掉（答案不变）。
   此时副本的**两份**出现都没了 —— 这才是「无任何源信息」的真正地板。
3. **打乱答案**（`shuffle_answers`）：答案与上下文解耦；若仍高于 chance，说明
   读数与答案无关（仪器/构造混杂）。

用法（需 GPU 锁）：`python3 diagnose_copy_confound.py`
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from tools.evalkit.longrange_tasks import (copy_batch, corrupt_span,  # noqa: E402
                                           shuffle_answers, _summarize)
from tools.evalkit.retention import masked_logits                    # noqa: E402

sys.path.insert(0, HERE)
from run import load_trunk, floor_threshold                           # noqa: E402


@torch.no_grad()
def per_k_accuracy(model, task, device, max_chunk=64):
    """按输出块内位置 k 分解的准确率（`rows` 按 k 递增 ⇒ 列序即 k）。"""
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        logits = masked_logits(model, task.tokens[lo:hi].to(device), None)
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(device))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(device)).to(torch.long).cpu()
    return {"accuracy": round(float(hits.to(torch.float64).mean().item()), 6),
            "n": int(hits.numel()), "n_hits": int(hits.sum().item()),
            "by_k": {str(k): round(float(hits[:, k].to(torch.float64).mean().item()), 6)
                     for k in range(task.M)},
            "by_distance": _summarize(hits, task.dists)["by_distance"]}


def main() -> int:
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    dev = cfg["eval"]["device"]
    T = cfg["model"]["seq_len"]
    model, ck = load_trunk(cfg, dev)
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = dict(sym_vocab=64, filler_token=64, cue_token=205, generator=g)
    base = copy_batch(batch_size=32, seq_len=T, copy_len=32, src_pos=0, **kw)
    variants = {
        "as_is": base,
        "corrupt_source_only": corrupt_span(base, 0, 32),
        "corrupt_source_and_tail": corrupt_span(corrupt_span(base, 0, 32), 224, 256),
        "shuffle_answers": shuffle_answers(base),
    }
    chance = 1.0 / 64
    out = {"checkpoint": ck, "chance": chance,
           "floor_threshold_2sigma": floor_threshold(base.B * base.M, chance),
           "variants": {}}
    for name, task in variants.items():
        out["variants"][name] = per_k_accuracy(model, task, dev)
        r = out["variants"][name]
        print(f"{name:<24} acc={r['accuracy']:.4f} n={r['n']} "
              f"by_k[0,4,16,31]={[r['by_k'][str(k)] for k in (0, 4, 16, 31)]}", flush=True)

    # 4）毁源必须毁**该档自己的源区间**：逐 src_pos 对比「毁错的区间 [0,32)」与
    #    「毁对的区间 [src_pos, src_pos+32)」。这是 run.py 首轮虚高读数的根因定位。
    out["span_sensitivity"] = {}
    for p in (0, 96, 191):
        t = copy_batch(batch_size=16, seq_len=T, copy_len=32, src_pos=p, **kw)
        row = {"as_is": per_k_accuracy(model, t, dev)["accuracy"],
               "corrupt_span_0_32": per_k_accuracy(model, corrupt_span(t, 0, 32), dev)["accuracy"],
               "corrupt_span_src": per_k_accuracy(
                   model, corrupt_span(t, p, p + 32), dev)["accuracy"]}
        out["span_sensitivity"][str(p)] = row
        print(f"src_pos={p:<4} as_is={row['as_is']:.4f} "
              f"corrupt[0,32)={row['corrupt_span_0_32']:.4f} "
              f"corrupt[src,src+32)={row['corrupt_span_src']:.4f}", flush=True)
    with open(os.path.join(HERE, "diagnose_copy_confound.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
