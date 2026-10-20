"""2683 · 契约外归因读数（评测期掩码反事实，0 训练 token）：区分「机制不成立」与「本实现不成立」。

**本脚本的读数不参与判词。** 判词只由 `README.md` §3 的判据①（同 k_eff 的配对 Δ 对 0.014 带）
决定，落盘在 `results.json::main_verdict`。这里问的是判词之后的第二个问题：
「块条带不如均匀条带」坏在**机制**（远块那 8 个键在真实文本上不承载可用信息），还是坏在
**本实现 / 本训练配置**（掩码是活的，但把 8 个键挪到距离 25–39、中间留 9–16 个键的洞，
并不比留在最近处更好）。

做法沿用 `axes/1-time/1.1.1-hard-window/experiments/2679-anchor-eval-ablation-13m` 的评测期掩码消融：
**同一个 checkpoint、只换评测期可见集、0 训练 token**。

三个读数（都作用于块条带臂的 checkpoint，`seed 0/1` 各一遍）：

| 读数 | 评测期可见集 | 它回答什么 |
|---|---|---|
| `own` | 块条带（最近 16 + 滞后 4 块的一块对齐 8 键）＝该臂训练时那张掩码 | 自洽门：必须逐位复现 `results.json::bpc_by_arm.block8l4_s{0,1}` |
| `far_block_off` | 只留最近 16（远块关掉） | 远处那 8 个键**对这个模型值多少 bpc** |
| `band24_on_block_ckpt` | 均匀条带最近 24（＝把远块换成中间那 8 个洞） | 只换推理掩码、不重训，能补回多少 |

Δ 方向按 `harness/SCHEMA.md` §17.3：**Δ = bpc_对照 − bpc_实验**，两个反事实的对照都是 `own`：

- `far_block_off`：Δ = bpc_own − bpc_far_block_off。**负且越带 ⇒ 远块被这个模型用上了**（关掉更差）；
  落带内 ⇒ 远块在这个尺度上没有可用信息（机制疑）；正且越带 ⇒ 远块是有害的。
- `band24_on_block_ckpt`：Δ = bpc_own − bpc_band24。**负且越带 ⇒ 只换推理掩码就能补回**，
  差的不是训练收敛形态；落带内 ⇒ 换推理掩码补不回，差的正是「按块条带收敛出来的解」。

用法（GPU，≈3 min；§20 互斥锁由调用方持有）：
    python3 attribution.py            # 写 attribution.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import block_sparse  # noqa: E402
import run as R  # noqa: E402  —— 复用同一套评测口径（臂表、加载器、配对统计、批生成）
from pathlm.data import load_enwik8_full  # noqa: E402
from pathlm.depth_ar import window_anchor_keep_mask  # noqa: E402
from tools.prereg_bind import require as require_prereg  # noqa: E402

OUT = os.path.join(HERE, "attribution.json")
BAND = R.BAND  # 0.014，只用于配对 Δ（harness/methodology.md §2）


def keep_mask(kind: str) -> torch.Tensor:
    """反事实可见集；`own` 走与训练/判据评测同一个函数。"""
    if kind == "own":
        return block_sparse.block_stripe_keep_mask(R.SEQ, device=R.DEVICE)
    if kind == "far_block_off":
        return window_anchor_keep_mask(R.SEQ, block_sparse.LOCAL, 0, device=R.DEVICE)
    if kind == "band24_on_block_ckpt":
        return window_anchor_keep_mask(R.SEQ, 24, 0, device=R.DEVICE)
    raise ValueError(f"未知可见集 {kind}")


def bits_under(model, eval_arr, kind: str) -> tuple[float, list[float], list[int]]:
    """同 `run.per_window_bits` 的口径，只把显式掩码换成 `kind` 指定的那一张。"""
    keep = keep_mask(kind)
    gen = torch.Generator().manual_seed(R.DATA_SEED)
    bits, n_tok = [], []
    with torch.no_grad():
        for _ in range(R.N_BATCHES):
            x, _ = R.batch(eval_arr, R.BATCH_SIZE, R.SEQ, gen)
            x = x.to(R.DEVICE)
            logits = R.masked_logits(model, x, keep)[:, :-1].float()
            tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="none").view(R.BATCH_SIZE, -1)
            bits.extend((ce.double().sum(1) / R.LN2).cpu().tolist())
            n_tok.extend([tgt.shape[1]] * R.BATCH_SIZE)
    return round(sum(bits) / sum(n_tok), 6), bits, n_tok


def in_band(delta: float) -> str:
    if delta >= BAND:
        return "越带（正：反事实臂更好）"
    if delta <= -BAND:
        return "越带（负：反事实臂更差）"
    return "落带内"


def main() -> None:
    os.chdir(ROOT)
    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：本脚本必须在已绑定契约的实验里跑")

    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", R.DATA)
    judged = json.load(open(os.path.join(HERE, "results.json")))
    arms = {a["name"]: a for a in R.ARMS}

    out = {
        "experiment": "2683-block-stripe-vs-band-13m",
        "what_this_is": ("契约外·评测期掩码反事实归因（0 训练 token）。不参与判词：判词由 "
                         "README.md §3 判据① 与 results.json::main_verdict 决定；本文件只把"
                         "「块条带不如均匀条带」拆成「远块有没有用」与「换推理掩码能不能补回」。"),
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "comparator": "own（该 checkpoint 训练时的那张块条带掩码）",
        },
        "delta_direction_note": (
            "两个反事实都记 Δ = bpc_own − bpc_反事实（正 = 反事实更好）。"
            "far_block_off 负且越带 ⇒ 关掉远块更差 ⇒ 远块被模型用上；"
            "band24_on_block_ckpt 负且越带 ⇒ 只换推理掩码就能补回。"),
        "band_bpc": BAND,
        "eval": {"n_batches": R.N_BATCHES, "batch_size": R.BATCH_SIZE,
                 "window_seed": R.DATA_SEED},
        "per_seed": {},
    }

    for s in (0, 1):
        name = f"block8l4_s{s}"
        model = R.load_arm(arms[name], vocab)
        own_bpc, own_bits, n_tok = bits_under(model, eval_arr, "own")
        gate = abs(own_bpc - judged["bpc_by_arm"][name]) < 1e-9
        if not gate:
            raise SystemExit(f"自洽门失败：{name} 本轮 own 读数 {own_bpc} != "
                             f"判据落盘 {judged['bpc_by_arm'][name]}")
        far_bpc, far_bits, _ = bits_under(model, eval_arr, "far_block_off")
        band_bpc, band_bits, _ = bits_under(model, eval_arr, "band24_on_block_ckpt")
        far = R.paired_stats(own_bits, far_bits, n_tok)
        band = R.paired_stats(own_bits, band_bits, n_tok)
        far["in_band"] = in_band(far["delta_bpc"])
        band["in_band"] = in_band(band["delta_bpc"])
        out["per_seed"][name] = {
            "bpc_own": own_bpc,
            "bpc_far_block_off": far_bpc,
            "bpc_band24_on_block_ckpt": band_bpc,
            "own_matches_judged_reading": gate,
            "far_block_off": far,
            "band24_on_block_ckpt": band,
            "judged_comparator_bpc": judged["bpc_by_arm"][f"win24a0_s{s}"],
        }
        print(f"[attrib] {name}: own {own_bpc:.6f} / far_off {far_bpc:.6f} (Δ {far['delta_bpc']:+.6f}, "
              f"{far['in_band']}) / band24@ckpt {band_bpc:.6f} (Δ {band['delta_bpc']:+.6f}, "
              f"{band['in_band']})", flush=True)

    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[attrib] 写出 {os.path.relpath(OUT, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
