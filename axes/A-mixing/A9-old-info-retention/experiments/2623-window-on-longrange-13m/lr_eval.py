"""2623 · A9 合成长程判决的评测语义层（训练/评测「同掩码」口径 + 聚合 + clean bpc）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。

本模块只放**本实验新增的语义**（不碰 `pathlm/`、`tools/evalkit/`，SCHEMA §22），
供 `run.py` 与 `tests/test_a9_window_longrange.py` 共用：

- `native_keep_mask`：把「一臂的可见性口径」翻译成 SDPA 的 `keep_mask`——
  `full` 臂 = `None`（释放版 `is_causal=True`），`win16` 臂 =
  `window_anchor_keep_mask(T, 16, 0)`。契约要求**训练掩码 == 评测掩码**，这是唯一
  实现点（训练侧由 `DepthARModel._window_mask` 调同一函数）。
- `distance_source_inside_window`：距离腿的判读辅助（key 是否落在窗外）。
- `aggregate_recall`：把一个任务在多个批上的逐查询命中聚合成
  准确率 + Wilson CI + 距离分桶（口径与 `longrange_tasks._summarize` 一致）。
- `clean_bits_per_token`：辅助腿「clean bpc」——在**任务分布**上、各臂**原生掩码**
  下的干净下一 token CE（bit/token）。定义见 README §判据④。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from pathlm.depth_ar import window_anchor_keep_mask
from tools.evalkit.carry_channel import wilson_ci
from tools.evalkit.longrange_tasks import TaskBatch
from tools.evalkit.retention import masked_logits

LN2 = math.log(2.0)


def native_keep_mask(T: int, train_window: int,
                     train_anchors: int = 0,
                     device=None) -> torch.Tensor | None:
    """一臂的可见性口径 → SDPA `keep_mask`。

    `train_window <= 0`（full 臂）⇒ `None`（释放版，`is_causal=True`）；
    `train_window > 0`（窗口臂）⇒ `window_anchor_keep_mask(T, w, anchors)`。
    训练侧 `DepthARModel._window_mask` 调**同一函数**（契约：训练掩码必须
    与评测掩码一致）。
    """
    if train_window <= 0:
        if train_anchors > 0:
            raise ValueError("anchors 只在 train_window > 0 时有意义")
        return None
    return window_anchor_keep_mask(T, train_window, train_anchors, device)


def arm_visibility(arm: dict) -> dict:
    """从 config 的臂条目取可见性（唯一来源，避免两处口径漂移）。"""
    return {"train_window": int(arm.get("train_window", 0)),
            "train_anchors": int(arm.get("train_anchors", 0))}


def distance_source_inside_window(T: int, distance: int, window: int) -> bool:
    """变距离回忆里，源 key 是否落在可见窗口内：可见 ⇔ 距离 `d < window`。

    `var_recall` 的源 key 在 `src_pos = (T-1) - d`，查询行 = `T-1`，
    `query - key = d`；`window_anchor_keep_mask` 放行 `(i-j) < window` ⇒ `d < window`。
    `window <= 0`（full 臂）= 全可见 ⇒ 恒 True。
    """
    if window <= 0:
        return True
    return distance < window


def aggregate_recall(rows: list[dict], dists: list[int],
                     chance: float) -> dict:
    """把多批的 `longrange_tasks.accuracy`/`arm_battery` 读数聚合成总读数 + CI。

    `rows`：每批一条，须含 `n` 与 `n_hits`；`dists`：该批逐查询距离（用于窗外占比）。
    """
    n = sum(int(r["n"]) for r in rows)
    hits = sum(int(r["n_hits"]) for r in rows)
    if n <= 0:
        raise ValueError("aggregate_recall 需要至少一个非空批")
    out = {"n": n, "n_hits": hits, "accuracy": round(hits / n, 6),
           "accuracy_ci": wilson_ci(hits, n), "chance": chance}
    if dists:
        k = sum(1 for d in dists if d >= 0)
        out["query_key_distance"] = {
            "min": min(dists), "max": max(dists), "n": k,
            "median": sorted(dists)[k // 2]}
    return out


@torch.no_grad()
def clean_bits_per_token(model, task: TaskBatch, keep_mask: torch.Tensor | None,
                         device: str = "cuda", max_chunk: int = 64) -> dict:
    """干净（未毁源）下一 token CE——在任务分布上、各臂原生掩码下。

    返回 `{all_positions, answer_rows, chance_bits}`（单位 bit/token，越低越好）：

    - `all_positions` = 全位置因果移位 CE / ln2（「clean bpc」辅助腿，README 判据④）；
    - `answer_rows` = 只统计判决行（该行输入 = 被问的 key）的 CE / ln2——它才是
      决定准确率的那个量（SCHEMA §17.15：门禁/判读要落在决定判据的损失上）；
    - `chance_bits` = `log(n_answers)/ln2`（均匀猜测的 CE，作参照）。
    """
    dev = device if device is not None else next(model.parameters()).device
    ce_sum, ce_n = 0.0, 0
    ans_sum, ans_n = 0.0, 0
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        x = task.tokens[lo:hi].to(dev)
        km = None
        if keep_mask is not None:
            if keep_mask.dim() == 3:
                km = keep_mask[lo:hi].to(dev)
            else:
                km = keep_mask.to(dev)
        logits = masked_logits(model, x, km).float()          # [b, T, V]
        ce = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                             x[:, 1:].reshape(-1), reduction="sum")
        ce_sum += float(ce.detach())
        ce_n += (hi - lo) * (task.T - 1)
        rows = task.rows[lo:hi].to(dev)
        sel = torch.gather(logits, 1, rows.unsqueeze(-1).expand(-1, -1,
                                                                logits.shape[-1]))
        tgt = task.answers[lo:hi].to(dev)                     # [b, M]
        a = F.cross_entropy(sel.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                            reduction="sum")
        ans_sum += float(a.detach())
        ans_n += (hi - lo) * task.M
    return {"all_positions": round(ce_sum / ce_n / LN2, 6),
            "answer_rows": round(ans_sum / max(ans_n, 1) / LN2, 6),
            "chance_bits": round(math.log(task.n_answers) / LN2, 6),
            "n_all": ce_n, "n_answer_rows": ans_n}
