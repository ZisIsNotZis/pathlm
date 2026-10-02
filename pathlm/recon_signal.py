"""重建训练信号（ticket 36 波二 U2 线，F4 训练信号）。

契约：`axes/F-iteration/F4-training-signal/experiments/2619-reconstruction-loss-13m/README.md`
（判据线逐字抄录，不许改）。本模块 = 该实验的**干预实现**，与 `pathlm/depth_ar.py`
的接缝只有两处（`DepthARConfig.recon_window` 字段 + `__init__` 里建头的那一行）。

## 干预是什么（一句话）

在**无显式记忆槽**的 stock dense 模型上，加一条辅助损失：位置 `i` 的隐状态必须
**重建刚刚掉出可见窗口的那一格** `tokens[i − W]`（`W = recon_window`），
外加一个单层 d→V 线性解码头。目标不是 bpc，而是主任务的**关联回忆准确率**。

## 为什么「被窗口遮住」正好是 `tokens[i − W]`

训练期可见掩码（`window_anchor_keep_mask`，本仓既有仪器）下，位置 `i` 的可见集是
`[i − W + 1, i]` ⇒ `i − W` **不可见**；而位置 `i − 1` 的可见集是 `[i − W, i − 1]`
⇒ 同一格**仍可见**。于是这条损失在物理上只要求一件事：**每一格在离开窗口前，
把它交班给下一格的隐状态（经注意力的 K/V 或残差流）**。「一路交班」累积起来
= 残差流必须当作递归记忆来用（这正是 `2617-design-rightcarry` §1.1 的梯度缺口：
只看最终 CE 时，最短路径是「当场读源」，没有任何梯度奖励搬运）。

两条性质由单测用**真实掩码**钉住（不是文字承诺）：
`test_recon_targets_are_exactly_window_masked`。

## 实现选择与理由（写清，避免后人猜）

1. **独立解码头**（`nn.Linear(d, V, bias=False)`，**不 tied**）：如果复用 tied 读出，
   同一个 softmax 分布要同时拟合「下一个 token」与「W 步前的 token」两个**不同**
   目标 ⇒ 辅助损失会与主任务直接抢同一个输出，变成纯反噬。独立头让两者只共享
   隐状态容量（这是我们要测的东西）。
2. **fp32**（`torch.autocast(enabled=False)` 内做头 + CE）：F4 的既有教训——
   这类损失在 bf16 autocast 下两次发散（`本地实测`，见 F4 课题页）。主循环另有
   **pre-step grad-norm gate**（`train_depth_ar.train`，`gnorm > 1e4` 或不有限则跳过该步）。
3. **λ = 0 逐位等价基线**：λ=0（或缺省 `recon_window=0`）时本模块的函数**根本不被调用**
   ⇒ 不建图、不产生浮点运算、不消耗随机数；头参数 `p.grad is None` ⇒ AdamW 整条跳过
   （连 weight decay 都不作用）⇒ 主干轨迹与基线逐位相同。单测：
   `test_recon_lambda_zero_returns_same_loss_object`、`test_recon_head_dead_at_lambda_zero`。
4. **`filler_token` 只是诊断参数**：损失对所有目标行**等权平均**，与 `filler_token`
   无关（单测 `test_recon_loss_split_is_diagnostic_only` 钉住）；它只用来把读数拆成
   「符号行 / filler 行」两组，作为 SCHEMA §10 的活性量。

## 活性量（SCHEMA §10 / §17.4）

- 参数侧：`params_added`、`w_fro`、`w_absmax`、`nonzero_frac`；
- 运行侧（每 200 步随训练日志落盘）：`recon_ce_mean`、`recon_ce_symbol_mean`
  （重建**内容**的难度，非平凡项）、`recon_acc_symbol_mean`（重建命中率）、
  `head_gnorm_mean`（该头的梯度量级——λ=0 臂必须恒为 0）。

## 口径（不要与「搬运成功」混用）

`recon_acc_symbol` 量的是**「刚掉出窗口的那一格重建得对不对」**，是**训练信号的活性**，
不是判据（判据 = 关联回忆准确率）。两者必须分开报。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: 重建头的初始化标准差（与 trunk 的 nn.Linear 缺省同款量级；显式写死便于核对）。
RECON_HEAD_INIT_STD = 0.02


def recon_targets(tokens: torch.Tensor,
                  window: int) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 `(rows, targets)`：行 `i ∈ [window, T−1]` 负责重建 `tokens[i − window]`。

    行 `i` 是**隐状态的行号**（与主损失同一约定：行 i 的读出预测 token i+1）。
    目标 = **刚掉出可见窗口的那一格**（见模块 docstring 的可见性论证）。
    """
    T = int(tokens.shape[1])
    if window <= 0:
        raise ValueError(f"window must be >= 1, got {window}")
    if window >= T:
        raise ValueError(f"window={window} 必须小于 seq_len={T}（否则没有目标）")
    rows = torch.arange(window, T, device=tokens.device)
    return rows, tokens[:, :T - window]


def build_recon_head(cfg, vocab_size: int) -> nn.Linear:
    """建重建头（单层线性 d→V，无 bias）+ 单因素校验。

    必须在 trunk 的参数 init 循环**之后**调用（`DepthARModel.__init__` 的既有惯例，
    同 `readout_gated` / `carry`）：这样它不消耗其余参数的初始化随机数 ⇒ 同 seed 下
    「有头」与「无头」两个模型的主干参数**逐位相同**（单测钉住）。

    单因素约束：`dense` + `topology="sequential"` + `train_window > 0`
    （目标定义在窗口掩码上），且与其它干预（`attn_sparse_k` / `attn_linear` /
    `n_kv_heads` / `carry_rule`）互斥——一次只改一个因素。
    """
    if cfg.recon_window <= 0:
        raise ValueError(f"recon_window must be >= 1, got {cfg.recon_window}")
    if not cfg.dense:
        raise ValueError(
            "重建训练信号只在 dense=True（stock decoder-only 形态）下验证（单因素）"
            f"——收到 recon_window={cfg.recon_window} 且 dense=False")
    if cfg.topology != "sequential":
        raise ValueError(
            "recon_window 与层间拓扑互斥（单因素：一次只改一个因素）——"
            f"收到 topology={cfg.topology!r}")
    if cfg.train_window <= 0:
        raise ValueError(
            "重建目标 = 刚掉出**可见窗口**的那一格 ⇒ 必须 train_window > 0；"
            f"收到 train_window={cfg.train_window}")
    if cfg.recon_window >= cfg.seq_len:
        raise ValueError(f"recon_window={cfg.recon_window} 必须小于 seq_len="
                         f"{cfg.seq_len}")
    for other, on in (("attn_sparse_k", cfg.attn_sparse_k > 0),
                      ("attn_linear", cfg.attn_linear),
                      ("n_kv_heads", cfg.n_kv_heads is not None
                       and cfg.n_kv_heads < cfg.n_heads),
                      ("carry_rule", cfg.carry_rule is not None)):
        if on:
            raise ValueError(
                f"recon_window 与 {other} 互斥（单因素：一次只改一个因素）")
    head = nn.Linear(cfg.d_model, vocab_size, bias=False)
    nn.init.normal_(head.weight, std=RECON_HEAD_INIT_STD)
    return head


def recon_logits(head: nn.Module, hidden_rows: torch.Tensor) -> torch.Tensor:
    """`hidden_rows [.., d]` → 词表 logits `[.., V]`，**强制 fp32**（autocast 之外）。

    `enabled=False` 的 device_type 必须与**外层 autocast 的 device_type 一致**
    才能真的把它关掉（实测教训：外层是 `torch.autocast("cpu")` 时，内层用
    `"cuda"` 的 off 上下文拦不住它，输出仍是 bf16）⇒ 取输入张量的 device.type。
    单测 `test_recon_logits_is_fp32_under_bf16_autocast` 在 CPU autocast 下钉住。
    """
    with torch.autocast(hidden_rows.device.type, enabled=False):
        return head(hidden_rows.float())


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def recon_loss(head: nn.Module, hidden: torch.Tensor, tokens: torch.Tensor,
               window: int, filler_token: int | None = None) -> dict:
    """重建项：`hidden [B,T,d]`（最终深度隐状态）+ 目标 `tokens[i−window]`。

    返回**张量/计数**（不在每步 `.item()`，避免共享卡上的同步风暴）：

    - `loss` = 全部目标行的 mean CE（= `ce_all`，**就是被 λ 乘的那一项**）；
    - `ce_symbol` / `ce_filler`、`n_symbol` / `n_filler`：按 `filler_token` 拆开的
      诊断量（`filler_token=None` 时不拆，全部计为 symbol）；
    - `acc_all` / `acc_symbol`：top-1 命中率（诊断量，**不是判据**）。
    """
    rows, targets = recon_targets(tokens, window)
    logits = recon_logits(head, hidden[:, rows])                 # [B, T−W, V] fp32
    V = logits.shape[-1]
    flat = logits.reshape(-1, V)
    tt = targets.reshape(-1)
    ce = F.cross_entropy(flat, tt, reduction="none")
    hit = (flat.argmax(-1) == tt).to(ce.dtype)
    if filler_token is None:
        sym = torch.ones_like(tt, dtype=torch.bool)
    else:
        sym = tt != int(filler_token)
    return {
        "loss": ce.mean(),
        "ce_all": ce.mean(),
        "ce_symbol": _masked_mean(ce, sym),
        "ce_filler": _masked_mean(ce, ~sym),
        "n_symbol": sym.sum(),
        "n_filler": (~sym).sum(),
        "n_targets": tt.numel(),
        "acc_all": hit.mean(),
        "acc_symbol": _masked_mean(hit, sym),
        "chance_uniform_vocab": 1.0 / V,
    }


def recon_liveness(head: nn.Module, recon_window: int, recon_lambda: float,
                   run_stats: dict | None = None) -> dict:
    """SCHEMA §10 的干预活性块（`params_added / w_fro / w_absmax / nonzero_frac`
    + 运行侧活性）。`head=None`（关闭）时给出显式的 `alive=False` 与全 None。

    `run_stats`：`train_depth_ar.train` 返回的 `recon_*` 均值（每 200 步落盘那份的
    同口径汇总），用于交叉核对「参数活着」与「梯度真的流过」。
    """
    if head is None:
        return {"recon_window": 0, "recon_lambda": 0.0, "enabled": False,
                "params_added": 0, "w_fro": None, "w_absmax": None,
                "nonzero_frac": None, "head_gnorm_mean": None,
                "recon_ce_mean": None, "recon_ce_symbol_mean": None,
                "recon_acc_symbol_mean": None, "alive": False,
                "alive_note": "缺省关闭（recon_window=0）：无头、无损失项、逐位还原 v1"}
    w = head.weight.detach().float()
    out = {
        "recon_window": int(recon_window), "recon_lambda": float(recon_lambda),
        "enabled": True,
        "params_added": int(head.weight.numel()),
        "w_fro": round(float(w.norm()), 6),
        "w_absmax": round(float(w.abs().max()), 6),
        "nonzero_frac": round(float((w != 0).float().mean()), 6),
    }
    rs = run_stats or {}
    out["head_gnorm_mean"] = rs.get("head_gnorm")
    out["recon_ce_mean"] = rs.get("ce_all")
    out["recon_ce_symbol_mean"] = rs.get("ce_symbol")
    out["recon_ce_filler_mean"] = rs.get("ce_filler")
    out["recon_acc_symbol_mean"] = rs.get("acc_symbol")
    out["recon_acc_all_mean"] = rs.get("acc_all")
    out["recon_n_steps"] = rs.get("n_steps")
    out["recon_n_symbol_frac"] = rs.get("n_symbol_frac")
    # 「干预活着」= 参数非零 + （λ>0 时）该头确实拿到过梯度；λ=0 时梯度恒为 0，
    # 此时它**不是**干预（是基线的一部分），活性判读必须显式区分这两件事。
    g = out.get("head_gnorm_mean")
    out["grad_flowed"] = (g is not None and float(g) > 0.0) if recon_lambda > 0 else None
    out["alive"] = bool(recon_lambda > 0 and out["grad_flowed"]
                        and out["w_fro"] > 0)
    if recon_lambda <= 0:
        out["alive_note"] = ("λ=0 臂：头存在但**未被训练**（p.grad is None ⇒ AdamW "
                            "整条跳过）⇒ 这不是干预，也不会扰动主干（逐位等价基线）")
    return out
