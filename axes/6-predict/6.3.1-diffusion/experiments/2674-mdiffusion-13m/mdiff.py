#!/usr/bin/env python3
"""2674-mdiffusion-13m · 契约实现库（契约见本目录 `../../6.3.1-diffusion/experiments/2674-mdiffusion-13m/README.md`）。

**唯一变量 = 掩码扩散目标 + K 步置信度解掩码采样器**：复用 2671-padfill-13m 的 harness
`padfill.py`（只读 import，不修改；sha256 记入 `MODEL.md`）拿到模型 `PadFillLM(attn_mask="full")`
（双向）、mask 调度 `sample_fill_mask` / `make_fill_input`、数据 `load_data`、固定 offset
`eval_offsets`、bpc / CI / 禁墙钟工具。

本文件新增：

- `train_mdiff`：**吸收态掩码扩散**训练（契约 §2）——每 step 抽 `r ~ U(0,1)`（固定调度种子
  `MASK_SEED`），对**每个位置**独立以概率 `r` 替换为 `[mask]`，损失 = **只在被 mask 位置**上的 CE。
  matched-step = 2671 的 `S`（从 `2671/results.json::config.steps` 读，本文件不写死）。
- `sample_denoise_k`：**K 步置信度解掩码采样**（契约 §2）——第 0 步把目标位置集 `M` 全部置 `[mask]`；
  每步一次全段前向，按「该位置预测分布的最大概率」排序，定格置信度最高的 `ceil(|M|/K)` 个
  （argmax、温度 0、确定性），其余保持 `[mask]`；末步定格全部剩余。⇒ 每 block 顺序前向 = K，
  `tokens-per-forward = |M|/K`。采样的模型输入**从不含 M 位置真值**（只含未定格位置的 `[mask]`
  与已定格位置的 argmax 预测），这是 L0 泄漏门。
- `mdiff_metrics`：逐步 NELBO（每步在**仍被 mask 位置**上 CE 的均值）与**末步 bpc**、**逐位置首定格
  bpc**（每位置只计一次，供与 ar_at_M 配对对照）。

口径（禁墙钟）：一次去噪前向 = 全段双向前向（layer-passes = L·T），K 步 ⇒ 每 token 等效 FLOPs = K·L
（以 AR 每 token 的 L 为单位，契约 §5）。
"""

from __future__ import annotations

import math
import os
import sys

import importlib.util

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
PADFILL_DIR = os.path.join(REPO, "axes", "6-predict", "6.3.2-pad-fill",
                           "experiments", "2671-padfill-13m")
# 只读 import 2671 的 harness（不改）：按显式路径加载为 `padfill`，不把 2671 目录插到 sys.path 前部
# （否则 `import run` 会命中 2671 的 run.py，与本目录的 run.py 撞名）。padfill.py 自己会插 REPO。
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if PADFILL_DIR not in sys.path:
    sys.path.append(PADFILL_DIR)          # 兜底，仅追加（前端留给本源目录，避免撞名）
if "padfill" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("padfill", os.path.join(PADFILL_DIR, "padfill.py"))
    P = importlib.util.module_from_spec(_spec)
    sys.modules["padfill"] = P
    _spec.loader.exec_module(P)

import padfill as P  # noqa: E402  只读 harness（2671）

# ---- 本项常量（契约 §2/§3）----------------------------------------------------
R_EVAL = 0.25                 # 评测位置集 M 的 mask rate（与 2671 fill_bidir_r25 同 M，逐字对齐）
K_LIST = (1, 4, 8, 16)        # K 步解掩码扫描（契约 §2）
REFERENCE_BAND_BPC = 0.014    # 主判参考带（bpc 口径，限配对 Δ、≥5 窗口；契约 §3）
RESULTS_2671 = os.path.join(PADFILL_DIR, "results.json")

LN2 = math.log(2)


# ---- 训练：吸收态掩码扩散（契约 §2）-------------------------------------------
def train_mdiff(seed: int, steps: int, train_arr, device: str = "cuda",
                log_fn=None) -> P.PadFillLM:
    """按契约 §2 训一个 `mdiff` 臂（双向 `attn_mask="full"`）。

    - 模型 init seed = `seed`（与 2671 同为 {0,1}）；
    - 数据顺序 seed = `P.DATA_SEED`（与 2671 同 token 预算）；
    - mask 调度 seed = `P.MASK_SEED`（固定、可复现）：每 step 抽 `r ~ U(0,1)`，逐位置独立 `< r` ⇒ `[mask]`；
    - 损失 = **只在被 mask 位置**上的 CE（`logits[M]` vs `inp[M]`）。
    """
    torch.manual_seed(seed)
    model = P.PadFillLM(attn_mask="full").to(device)
    opt = P.build_optimizer(model)
    dg = torch.Generator().manual_seed(P.DATA_SEED)
    mg = torch.Generator().manual_seed(P.MASK_SEED)
    model.train()
    for i in range(steps):
        win = P.sample_window(train_arr, P.BATCH_SIZE, P.SEQ_LEN + 1, dg).to(device)
        inp = win[:, :P.SEQ_LEN]
        r = float(torch.rand((), generator=mg).item())          # per-step 吸收率
        M = P.sample_fill_mask(inp.shape, r, mg).to(device)     # 逐位置独立，全部位置可 mask
        if not bool(M.any()):                                   # 极端 r≈0 时的退化保护（确定性）
            k = int(inp.shape[1] // 2)
            M = torch.zeros_like(M)
            M[:, k] = True
        logits = model(P.make_fill_input(inp, M, "mask"))
        loss = F.cross_entropy(logits[M], inp[M])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_fn is not None and (i + 1) % 100 == 0:
            log_fn(i + 1, float(loss.detach()))
    model.eval()
    return model


# ---- K 步置信度解掩码采样器（契约 §2）-----------------------------------------
@torch.no_grad()
def sample_denoise_k(model: P.PadFillLM, true_tokens: torch.Tensor, M: torch.Tensor,
                     K: int, mask_token: int | None = None) -> dict:
    """K 步置信度解掩码；返回逐步统计与逐位置首定格 CE。

    参数：`true_tokens` (B,T) 真 token；`M` (B,T) bool 目标位置集；`K >= 1`。
    返回 dict（全部确定性，无随机）：
    - `x_final` (B,T)：K 步后定格的整段（M 位置 = argmax 预测，非 M 位置 = 真值）。
    - `fix_step` (B,T)：每个 M 位置在第几步被定格（非 M = 0）。
    - `fix_ce` (B,T)：每个 M 位置在其**首定格步**的 CE（非 M = nan）。
    - `step_ce_mean_bpc` (list[K])：每步在**仍被 mask 位置**上 CE 均值（bpc，契约 §3）。
    - `step_n` (list[K])：每步仍被 mask 的位置数。
    - `step_fix` (list[K])：每步定格的位置数。
    - `n_forwards` = K；`s_mdiff`/`n_m`/`mdiff_M_bpc` (B,)：逐 block 首定格 CE 和 / 位置数 / bpc。
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if mask_token is None:
        mask_token = model.mask_token
    B, T = true_tokens.shape
    dev = true_tokens.device
    x = true_tokens.clone()
    x[M] = mask_token                                   # 第 0 步：M 全置 [mask]
    fixed = torch.zeros(B, T, dtype=torch.bool, device=dev)
    fix_ce = torch.full((B, T), float("nan"), device=dev)
    fix_step = torch.zeros(B, T, dtype=torch.long, device=dev)
    n_m = M.sum(dim=1)                                  # (B,)
    per_step_fix = torch.ceil(n_m.float() / K).long()   # 每步定格上限（契约 §2）
    step_ce_mean_bpc, step_n, step_fix = [], [], []
    for k in range(1, K + 1):
        logits = model(x)                               # 一次全段前向 / 步
        logp = F.log_softmax(logits.float(), dim=-1)
        ce = -logp.gather(-1, true_tokens.unsqueeze(-1)).squeeze(-1)   # (B,T)
        conf = logp.exp().max(dim=-1).values            # 该位置预测分布的最大概率
        remaining = M & (~fixed)
        conf = conf.masked_fill(~remaining, -1.0)
        # 记录「仍被 mask 位置」的 CE 均值（契约 §3 的逐步 NELBO）
        rem_count = int(remaining.sum())
        step_n.append(rem_count)
        step_ce_mean_bpc.append((float(ce[remaining].sum()) / rem_count / LN2) if rem_count else float("nan"))
        if k == K:
            n_fix = remaining.sum(dim=1)                # 末步定格全部剩余
        else:
            n_fix = torch.clamp(per_step_fix, max=remaining.sum(dim=1))
        order = torch.argsort(conf, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        ar = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        rank.scatter_(1, order, ar)
        sel = remaining & (rank < n_fix.unsqueeze(1))
        pred = logits.argmax(dim=-1)                    # 温度 0、确定性
        x = torch.where(sel, pred, x)                   # 定格为预测 token（不是真值）
        fix_ce = torch.where(sel, ce, fix_ce)
        fix_step = torch.where(sel, torch.full_like(fix_step, k), fix_step)
        fixed = fixed | sel
        step_fix.append(int(sel.sum()))
    if not bool(torch.equal(fixed, M)):                 # 守恒：K 步后 M 全被定格且只定格 M
        raise AssertionError("denoise did not fix exactly M after K steps")
    s_mdiff = torch.nan_to_num(fix_ce, nan=0.0).sum(dim=1)
    return {
        "x_final": x,
        "fix_step": fix_step,
        "fix_ce": fix_ce,
        "step_ce_mean_bpc": step_ce_mean_bpc,
        "step_n": step_n,
        "step_fix": step_fix,
        "n_forwards": K,
        "s_mdiff": s_mdiff,
        "n_m": n_m,
        "mdiff_M_bpc": s_mdiff / (n_m.clamp(min=1).to(s_mdiff.dtype) * LN2),
    }


def mdiff_metrics(res: dict) -> dict:
    """从一次 `sample_denoise_k` 结果聚合出 NELBO / 末步 / 首定格 bpc（标量）。"""
    step_means = [x for x in res["step_ce_mean_bpc"] if not math.isnan(x)]
    nelbo_step_mean = float(np.mean(step_means)) if step_means else float("nan")
    total_s = float(res["s_mdiff"].sum())
    total_n = int(res["n_m"].sum())
    return {
        "nelbo_step_mean_bpc": nelbo_step_mean,             # 逐步均值（契约 §3 L1）
        "last_step_bpc": res["step_ce_mean_bpc"][-1],       # 末步 bpc（契约 §3 L1）
        "mdiff_M_bpc": total_s / (total_n * LN2) if total_n else float("nan"),  # 逐位置首定格（配对用）
        "n_m_total": total_n,
        "step_ce_mean_bpc": res["step_ce_mean_bpc"],
        "step_n": res["step_n"],
        "step_fix": res["step_fix"],
    }


# ---- 单次前向填全部 M（L0 的 K=1 退化参照 / 与 2671 fill 口径同构）-------------
@torch.no_grad()
def single_forward_fill_ce(model: P.PadFillLM, true_tokens: torch.Tensor,
                           M: torch.Tensor, mask_token: int | None = None) -> torch.Tensor:
    """把 M 全置 `[mask]` 后一次前向，返回每位置的 CE (B,T)（= K=1 采样器的退化目标）。"""
    if mask_token is None:
        mask_token = model.mask_token
    x = P.make_fill_input(true_tokens, M, "mask", mask_token=mask_token)
    logits = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp.gather(-1, true_tokens.unsqueeze(-1)).squeeze(-1)


# ---- 2671 只读参照（AR 读数 / 固定 offset / S）-------------------------------
def load_2671() -> dict:
    import json
    with open(RESULTS_2671, encoding="utf-8") as f:
        return json.load(f)


def ar_reference_blocks(ref: dict, r: float = R_EVAL) -> dict[int, dict]:
    """从 2671 `results.json` 取 `ar` 臂在 r 上逐 block 的 `s_ar_M` / `n_m`（r=0.25 与落盘 blocks 对齐）。"""
    out: dict[int, dict] = {}
    if abs(r - 0.25) > 1e-9:
        raise ValueError("2671 只落盘了 r=0.25 的逐 block s_ar_M；其它 r 无 per-block 配对数据")
    for seed_s, e in ref.get("ar", {}).items():
        out[int(seed_s)] = {
            "ar_full_bpc": e["ar_full_bpc"],
            "blocks": {b["offset"]: {"n_m": b["n_m"], "s_ar_M": b["s_ar_M"]} for b in e["blocks"]},
        }
    return out


def main() -> int:
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
