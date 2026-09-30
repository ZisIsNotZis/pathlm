"""2616 · 训练步显存探针（「先实测步时/显存再定预算」的显存那一半）。

作用：契约纪律要求「先实测步时/显存再定预算」，而 `train_depth_ar.py` 只落步时
不落显存。本探针**复现同一步**（同模型配置 / 同批大小 / 同优化器 / 同 autocast），
跑 30 步并报 **net 峰值显存**（PyTorch allocator 口径：`max_memory_allocated` /
`max_memory_reserved`），与评测侧的净显存（`results.json::cost.peak_mem_mib`）分开报。

用法：python3 smoke/mem_probe.py
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel   # noqa: E402
from pathlm.data import batch, load_enwik8_full           # noqa: E402

CFG = dict(d_model=256, n_layers=12, n_heads=8, seq_len=256, mlp_mult=6,
           dense=True, train_window=192, train_anchors=8)
STEPS, BS, SEED = 30, 32, 0


def main():
    torch.manual_seed(SEED)
    train_arr, _, vocab = load_enwik8_full(".tmp/enwik8",
                                           os.path.join(ROOT, "data/enwik8_full.npz"))
    model = DepthARModel(DepthARConfig(**CFG), vocab).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(STEPS):
        x, _ = batch(train_arr, BS, CFG["seq_len"], gen)
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    out = {
        "config": CFG, "steps": STEPS, "batch_size": BS, "seed": SEED,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
        "device": torch.cuda.get_device_name(0),
        "note": ("net 峰值显存（allocator 口径）= 本进程自己的用量，不含共享卡上其他"
                 "任务的占用；训练预算 ≤5GB net 的对照值。"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(os.path.join(HERE, "mem_probe.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
