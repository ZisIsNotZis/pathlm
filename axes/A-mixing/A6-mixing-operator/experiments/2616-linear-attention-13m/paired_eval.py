"""2616 · 配对评测（阈值附近的噪声量化，SCHEMA §12/§16 的补充证据）。

两臂的 clean bpc 用**同一批 eval 位置**（`depth_curve` 的同一 generator seed=2、
同一 batch 顺序、同 batch_size=16 × 20 批）⇒ 臂间差是**配对差**。本脚本逐批复现
这条配对关系，报：

- 每臂每批 bpc（含 linear 重跑臂，若存在）；
- 逐批配对差 `Δ_schema = bpc_full − bpc_linear`（SCHEMA §17.3：正 = 线性更好）与
  `Δ_contract = bpc_linear − bpc_full`（契约句面读法：正 = 线性更差）的均值 / SE /
  t 值 / 符号计数；
- 重跑臂的逐批配对差 ⇒ **同配置重跑散度**的配对口径值。

用途：判据阈值 0.014 是本项目的方差界，必须把「这条差是否超过噪声」量化出来。

用法（<1 min，GPU 单卡串行）：
    python3 paired_eval.py
"""

import json, os, statistics, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full, batch
import run as exp_run

LN2 = 0.6931471805599453
N_BATCHES = 20
BATCH_SIZE = 16
DATA_SEED = 2          # 与 depth_curve 内部一致


def arm_bpc_by_batch(model, eval_arr):
    """逐批 depth-L CE（与 depth_curve 同一批位置：seed 2 + 同一 batch 顺序）。"""
    gen = torch.Generator().manual_seed(DATA_SEED)
    out = []
    model.eval()
    with torch.no_grad():
        for _ in range(N_BATCHES):
            x, _ = batch(eval_arr, BATCH_SIZE, exp_run.SEQ, gen)
            x = x.to("cuda")
            _, aux = model(x, exit_depths=None)
            lg = aux["depth_logits"][-1][:, :-1].float()
            tgt = x[:, 1:]
            nats = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                   tgt.reshape(-1), reduction="sum").item()
            out.append(nats / tgt.numel() / LN2)
    return out


def load(name: str, attn_linear: bool, vocab: int):
    path = os.path.join(HERE, "runs", name, "model.pt")
    if not os.path.exists(path):
        return None
    torch.manual_seed(0)
    m = exp_run.DepthARModel(exp_run.cfg_base(attn_linear), vocab).cuda()
    m.load_state_dict(torch.load(path, map_location="cuda"))
    return m


def paired_stats(a: list[float], b: list[float]) -> dict:
    """配对差 a − b 的汇总（不改变任何判据线，只量化噪声）。"""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = statistics.mean(d)
    se = statistics.stdev(d) / n ** 0.5
    return {"n_batches": n, "mean_delta_bpc": round(mean, 5),
            "sd_of_paired_delta": round(statistics.stdev(d), 5),
            "se_of_mean": round(se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "min": round(min(d), 5), "max": round(max(d), 5)}


def main():
    os.chdir(ROOT)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", exp_run.DATA)
    targets = [("full-12k-s0", False), ("linear-12k-s0", True),
               ("linear-12k-s0-r2", True), ("full-12k-s0-r2", False)]
    per_batch = {}
    for name, flag in targets:
        m = load(name, flag, vocab)
        if m is None:
            print(f"[paired] {name}: checkpoint 缺失，跳过", flush=True)
            continue
        per_batch[name] = arm_bpc_by_batch(m, eval_arr)
        print(f"[paired] {name}: mean bpc over batches "
              f"{statistics.mean(per_batch[name]):.4f}", flush=True)
        del m
        torch.cuda.empty_cache()
    out = {"data_seed": DATA_SEED, "batch_size": BATCH_SIZE,
           "n_batches": N_BATCHES, "paired": True,
           "per_batch_bpc": {k: [round(v, 5) for v in vals]
                             for k, vals in per_batch.items()},
           "mean_bpc_by_arm": {k: round(statistics.mean(v), 5)
                               for k, v in per_batch.items()},
           "paired_delta_vs_full": {}, "note": (
               "配对差 = bpc_full − bpc_实验（SCHEMA §17.3：正 = 实验更好）；同一批"
               "位置 ⇒ 配对差的标准误远小于单臂 bpc 的标准误；本脚本不改变任何判据线")}
    base = per_batch.get("full-12k-s0")
    if base is not None:
        for name, vals in per_batch.items():
            if name == "full-12k-s0":
                continue
            schema = paired_stats(base, vals)
            out["paired_delta_vs_full"][name] = {
                "delta_bpc_schema_(full−arm)": schema["mean_delta_bpc"],
                "delta_bpc_contract_(arm−full)": round(-schema["mean_delta_bpc"], 5),
                **{k: v for k, v in schema.items() if k != "mean_delta_bpc"}}
    json.dump(out, open(os.path.join(HERE, "paired_eval.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["paired_delta_vs_full"], indent=2, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
