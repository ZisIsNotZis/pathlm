"""2634-carry-realtext-13m · 真实文本口径的三件仪器（契约 README §4a）。

1. `far_source_probe`：**毁源式窗外源敏感度探针**（主判据的仪器）。同一批真实文本窗口
   造两份输入——原件 `x` 与「窗口外前缀被换成随机字节」的 `x_corrupt`——在同一
   窗口掩码下各前向一次，只在**可见窗口完全落在未毁区**的行上算 CE：
   `Δbpc_far = CE(x_corrupt) − CE(x)`。无通道臂必须**逐位不变**（前提①）。
2. `channel_off`：评测期**通道消融**（把 `gate_w` 与每层 `carry_scale` 置零）——
   前向退化为「同架构无通道」的纯主干路径（诊断用，不改权重文件）。
3. `paired_cluster_bootstrap_ci`：按**窗口**聚类的配对 bootstrap（每个窗口 16 行共享
   同一段真实文本与同一份记忆状态 ⇒ 不是独立样本；与 2631 复核 B2 同一纪律）。

本模块**只读导入**仓库既有仪器（`retention` / `depth_ar`），不改任何共享文件。
"""

from __future__ import annotations

import contextlib
import random

import torch
import torch.nn.functional as F

from pathlm.data import batch
from tools.evalkit.retention import masked_logits

LN2 = 0.6931471805599453

#: 本实验采用的布局（T=256 / L=12 层 / 窗口 16 ⇒ `rf = 12·15 = 180`）：
#: 被毁前缀 = 位置 `[0, 64)`；被评行 = `[244, 255)`（这些行预测的目标在 245..255）。
#: `DEFAULT_EVAL_LO − DEFAULT_RF ≥ DEFAULT_CORRUPT_HI`（244−180 = 64）**恰好**满足不变式；
#: 换成别的 `rf` 时由 `far_source_probe` 的硬断言接住。
#: （历史的「[0,128) / [144,255)」布局在本配置下**泄漏**：144−180 < 128 ⇒ 已废弃；
#: 见 `rf_scan.py` 的实测与 `analysis.md` §5。）
DEFAULT_CORRUPT_HI = 64
DEFAULT_EVAL_LO = 244
DEFAULT_EVAL_HI = 255

#: 本实验模型的跨层多跳感受野（位置数）：`n_layers·(window−1)`
RF_NOTE = "rf = n_layers·(train_window−1)（L 层窗口 w 的因果注意力可跨层多跳看到 [i−rf, i]）"


def probe_kwargs(cfg: dict) -> dict:
    """把 `config.json` 的 `eval.probe` 变成 `far_source_probe` 的**完整** kwargs。

    存在的理由（本实验的一次真实事故）：`rf` 曾以默认值 0 被漏传，`eval_lo − rf ≥
    corrupt_hi` 于是退化成 `244 ≥ 64`（**恒真**），硬断言形同虚设。现在：

    - `rf` **由模型规格派生**（`rf = n_layers·(train_window−1)`），不接受手工填空；
    - 与 config 里记录的 `eval.probe.rf` **交叉校验**（不一致即报错）；
    - `train_window ≤ 0` 直接报错（无窗时被毁前缀可经全注意力直达，本仪器无定义）。
    """
    model, probe = cfg["model"], cfg["eval"]["probe"]
    window = int(model["train_window"])
    if window <= 0:
        raise ValueError(
            f"far_source_probe 只对开窗模型有定义，收到 train_window={window}（无窗时"
            "被毁前缀可经全注意力直达被评行，前提①不成立）")
    rf = int(model["n_layers"]) * (window - 1)
    if "rf" in probe and int(probe["rf"]) != rf:
        raise ValueError(
            f"config 里 eval.probe.rf={probe['rf']} 与由模型规格派生的 rf={rf}"
            f"（n_layers={model['n_layers']}·(train_window−1)）不一致")
    kw = {k: probe[k] for k in ("n_batches", "batch_size", "window_seed",
                               "corrupt_hi", "eval_lo", "eval_hi", "corrupt_seed")}
    kw["rf"] = rf
    return kw


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("percentile 收到空列表")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def paired_cluster_bootstrap_ci(cluster_diff_bits: list[float], cluster_n: list[int],
                                n_boot: int = 2000, seed: int = 2634,
                                alpha: float = 0.05) -> list[float]:
    """按**窗口**聚类的配对 bootstrap：逐簇「总比特差 / 总 token 数」重采样后的 95% CI。

    配对成立的前提：两份输入是**同一批窗口**（原件 vs 毁源），且逐窗独立。
    """
    if len(cluster_diff_bits) != len(cluster_n):
        raise ValueError("cluster_diff_bits 与 cluster_n 长度不一致")
    m = len(cluster_n)
    if m < 2:
        raise ValueError("配对 bootstrap 至少需要 2 个簇")
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n_boot):
        idx = rng.choices(range(m), k=m)
        num = sum(cluster_diff_bits[i] for i in idx)
        den = sum(cluster_n[i] for i in idx)
        out.append(num / den)
    out.sort()
    return [round(_percentile(out, alpha / 2), 6),
            round(_percentile(out, 1 - alpha / 2), 6)]


@torch.no_grad()
def far_source_probe(model, eval_arr, keep_mask, *, rf: int,
                     n_batches: int = 96,
                     batch_size: int = 16, window_seed: int = 2,
                     corrupt_hi: int = DEFAULT_CORRUPT_HI,
                     eval_lo: int = DEFAULT_EVAL_LO, eval_hi: int = DEFAULT_EVAL_HI,
                     corrupt_seed: int = 2634,
                     device: str = "cuda") -> dict:
    """毁源式窗外源敏感度探针（见模块文档）。

    `rf` = 主干注意力在本配置下的**有效感受野**（位置数）= `n_layers·(window−1)`：
    L 层窗口 w 的因果注意力，第 i 行的表征可以**跨层多跳**地看到 `[i−L(w−1), i]`
    （`rf_scan.py` 对 canonical 配置实测：第 244 行起对前缀修改**逐位**不敏感）。
    ⇒ 被评行必须选在 `eval_lo − rf ≥ corrupt_hi`，否则被毁前缀会**经注意力**（而非
    通道）传到被评行，前提①就不成立。本函数对该条件做硬断言。

    **`rf` 是必填关键字参数**（无默认值）：默认 0 会让断言退化成恒真式——这正是本实验
    一次真实事故（见 `probe_kwargs` 的文档）。它只是**保守**条件：只要被毁前缀对多层
    多跳注意力不可达，无论实际可达半径是多少，前提①都成立。

    返回 `ce_clean` / `ce_corrupt`（bits/byte，只算被评行）/ `delta_bpc_far` /
    逐簇配对 bootstrap CI / `max_abs_logit_diff_on_eval_rows`（无通道臂必须为 0）/
    以及 `ce_clean` 的按行剖面（形状描述，不外推）。
    """
    T = model.cfg.seq_len
    if rf <= 0:
        raise ValueError(
            f"rf 必须 > 0（收到 {rf}）：rf=0 会让下面的泄漏断言退化成恒真式"
            "——用 `probe_kwargs(cfg)` 派生 rf，不要手填")
    assert 0 < corrupt_hi <= eval_lo <= eval_hi <= T, \
        f"布局非法：corrupt_hi={corrupt_hi} eval=[{eval_lo},{eval_hi}) T={T}"
    assert eval_lo - rf >= corrupt_hi, (
        f"布局泄漏：被评行起点 {eval_lo} 减去感受野 {rf} 仍小于被毁前缀末端 {corrupt_hi}"
        f"⇒ 被毁前缀经 L 层多跳注意力可达（不是通道搬来的）；当前模型 L={model.cfg.n_layers}"
        f"、T={T} ⇒ 需要 eval_lo ≥ {corrupt_hi + rf}")
    rows = list(range(eval_lo, eval_hi))          # 行 i 预测 token i+1
    # 第一层的直接可见集也必须完全落在未毁区（冗余校验，防掩码语义被改）
    if keep_mask is not None:
        km_cpu = keep_mask if keep_mask.dim() == 2 else keep_mask[0]
        assert not bool(km_cpu[rows, :corrupt_hi].any().item()), \
            "单层可见集与被毁前缀重叠（掩码语义异常）"
    gen = torch.Generator().manual_seed(window_seed)
    cgen = torch.Generator().manual_seed(corrupt_seed)
    model.eval()
    n_rows = len(rows)
    ce_clean = torch.zeros(n_rows, dtype=torch.float64)
    ce_corrupt = torch.zeros(n_rows, dtype=torch.float64)
    cdiff: list[float] = []
    csize: list[int] = []
    max_logit_diff = 0.0
    n_tok = 0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        xc = x.clone()
        fill = torch.randint(0, model.vocab_size, (batch_size, corrupt_hi),
                             generator=cgen, dtype=torch.long).to(device)
        xc[:, :corrupt_hi] = fill
        km = keep_mask.to(device) if keep_mask is not None else None
        lg = masked_logits(model, x, km)
        lgc = masked_logits(model, xc, km)
        # 前提①的仪器读数：被评行上两份输入的 logits 最大绝对差
        sel = torch.tensor(rows, device=device)
        max_logit_diff = max(max_logit_diff,
                             float((lg[:, sel] - lgc[:, sel]).abs().max().item()))
        tgt = x[:, 1:][:, rows]                    # 被评目标 = token 145..255
        ce_a = F.cross_entropy(lg[:, rows].reshape(-1, lg.shape[-1]).float(),
                               tgt.reshape(-1), reduction="none").view(batch_size, -1)
        ce_b = F.cross_entropy(lgc[:, rows].reshape(-1, lg.shape[-1]).float(),
                               tgt.reshape(-1), reduction="none").view(batch_size, -1)
        ce_clean += ce_a.double().sum(0).cpu()
        ce_corrupt += ce_b.double().sum(0).cpu()
        # 逐窗（簇）配对差
        a_bits = (ce_a.double().sum(1) / LN2).cpu().tolist()
        b_bits = (ce_b.double().sum(1) / LN2).cpu().tolist()
        cdiff.extend([b - a for a, b in zip(a_bits, b_bits)])
        csize.extend([n_rows] * batch_size)
        n_tok += batch_size * n_rows
    ce_c = (ce_clean / n_batches / batch_size / LN2).tolist()
    ce_k = (ce_corrupt / n_batches / batch_size / LN2).tolist()
    mean_clean = sum(ce_c) / n_rows
    mean_corrupt = sum(ce_k) / n_rows
    return {
        "delta_bpc_far": round(mean_corrupt - mean_clean, 6),
        "ce_far_clean": round(mean_clean, 6),
        "ce_far_corrupt": round(mean_corrupt, 6),
        "delta_ci95_clustered": paired_cluster_bootstrap_ci(cdiff, csize),
        "max_abs_logit_diff_on_eval_rows": max_logit_diff,
        "n_tokens": n_tok,
        "n_batches": n_batches, "batch_size": batch_size,
        "window_seed": window_seed, "corrupt_seed": corrupt_seed,
        "layout": {"corrupt_prefix": [0, corrupt_hi],
                   "eval_rows": [eval_lo, eval_hi],
                   "receptive_field_positions": rf,
                   "required_min_eval_lo": corrupt_hi + rf,
                   "min_distance_of_corrupted_prefix_from_eval_rows": eval_lo - corrupt_hi,
                   "note": ("被评行的 L 层多跳感受野 [i−rf, i] 完全落在未毁区 "
                            "(eval_lo−rf ≥ corrupt_hi) ⇒ 对**无通道**臂逐位不敏感（前提①）；"
                            "对通道臂，被毁前缀只能经记忆槽到达被评行")},
        "ce_by_row_clean": [round(v, 6) for v in ce_c],
        "ce_by_row_corrupt": [round(v, 6) for v in ce_k],
        "note": ("Δbpc_far > 0 = 窗外前缀内容对该模型有用（通道把它搬来了）；"
                 "它测的是**敏感度**（必要条件），不是「帮上了忙」（看 bpc_win16）"),
    }


@contextlib.contextmanager
def channel_off(model):
    """评测期**通道消融**：`gate_w` 与每层 `carry_scale` 置零（临时，退出即还原）。

    置零后前向退化为「同架构无通道」的纯主干路径（读出不贡献、输入流不被预热）
    ⇒ 用于把「通道读出的贡献」与「参数量/容量」分开（README §4a 诊断）。
    未开通道的模型上为空操作。
    """
    if getattr(model.cfg, "carry_rule", None) is None:
        yield model
        return
    saved = []
    with torch.no_grad():
        for blk in model.blocks:
            if hasattr(blk, "carry_scale"):
                saved.append((blk.carry_scale, blk.carry_scale.detach().clone()))
                blk.carry_scale.zero_()
        for name, p in model.carry.named_parameters():
            if name == "gate_w":
                saved.append((p, p.detach().clone()))
                p.zero_()
    try:
        yield model
    finally:
        with torch.no_grad():
            for p, v in saved:
                p.copy_(v)


@torch.no_grad()
def win16_bpc(model, eval_arr, keep_mask, *, n_batches: int = 96,
              batch_size: int = 16, window_seed: int = 2,
              device: str = "cuda", per_window: bool = False):
    """窗口掩码下的 clean bpc（bits/byte，越低越好）。

    与 `retention.masked_bpc` 同一算法/同一窗口序，但批数可调（本实验用更大样本
    压低测量噪声）；**不与 2616 的 24×16 绝对水平逐位可比**（只做臂间比较）。

    `per_window=True` 时返回 `(bpc, per_window_bits)`：逐窗口的 CE 比特总数与 token 数，
    供**跨臂配对**的聚类 bootstrap 用（四臂的窗口序相同 ⇒ 同一索引 = 同一段真实文本、
    同一批位置）。**注意口径**：配对 CI 只含「同一批窗口上的测量噪声」，**不含**训练
    种子方差（后者只能由换种子重训估计）。
    """
    T = model.cfg.seq_len
    gen = torch.Generator().manual_seed(window_seed)
    nats, n_tok = 0.0, 0
    bits_per_window: list[float] = []
    n_per_window: list[int] = []
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        km = keep_mask.to(device) if keep_mask is not None else None
        logits = masked_logits(model, x, km)[:, :-1]
        tgt = x[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             tgt.reshape(-1), reduction="none").view(batch_size, -1)
        nats += ce.double().sum().item()
        n_tok += tgt.numel()
        if per_window:
            bits_per_window.extend((ce.double().sum(1) / LN2).cpu().tolist())
            n_per_window.extend([tgt.shape[1]] * batch_size)
    bpc = nats / n_tok / LN2
    return (bpc, bits_per_window, n_per_window) if per_window else bpc
