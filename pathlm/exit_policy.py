"""逐 token 退出决策头（轴 2 · `2.1.1 early exit` · WP1）。

本模块是**退出决策**的唯一落点：策略、序列级口径、校准表与算力账。模型与前向语义
一律复用 `pathlm.depth_ar`（不在这里重造）。

为什么单独开模块（`harness/SCHEMA.md` §22.7）：退出决策是**新逻辑**，塞进 2819 行的
`pathlm/depth_ar.py` 会与该文件已有的十几条研究线的未提交改动抢同一片编辑空间。
共享文件只做最小接线：`DepthARDecoder.step_auto`（生产解码器的逐 token 退出）。

三类策略（同一接口 `decide(depth, logits) -> bool[B,T]`）：

- `Top1ProbPolicy(threshold, min_depth, max_depth)`：读出 top-1 概率 ≥ 阈值即停
  （CALM 式置信门控；参照实现 `2623/exit_policies.py` 同款判据，但本实现进的是
  **生产解码器** `DepthARDecoder.step_auto`，不是参照实验的私有缓存）；
- `PrescribedDepths(depths)`：预设逐位置退出深度（对照臂 + 一致性单测）；
- `NeverExit()`：永不过早退出（退化臂 / 口径自洽锚点）。

两个端到端口径：

- `score_stream(model, tokens, policy)`：teacher-forcing、**逐位置在线**决策下的整段
  bpc。位置 i 的退出深度 d_i 只依赖 (k, 该位置第 k 层 logits) —— 严格因果；d_i > k 的
  位置在层 k 的 K/V 由该位置的**冻结表征**经本层投影填充（fill / proj-fill），与
  `DepthARModel.forward` 的混合深度语义逐位一致（单测钉住）。**这是本格的序列级主判据
  口径**：它同时含「退出读出的预测质量」与「异构深度 KV 语义对后续位置的代价」。
  注意它与 `depth_curve`（全深度上下文下的逐深度读出）**不是**同一个量：后者所有位置
  都跑到 L 层，填充语义不生效，故不含异构深度的代价（55-review C-P06）。二者只在
  「所有位置都退到同一深度 k」时相等——本实验用它做口径自洽锚点。
- `generate_with_exit(model, prompt, n_new, policy, commit)`：自由生成，逐 token 走
  `DepthARDecoder.step_auto`；用于「发出流与全深度贪心流的逐 token 一致率」。

口径纪律：只产出等效理论量（bpc / layer-passes-per-token / 跳过层占比 / 深度直方图），
**不产出也不接收任何墙钟或吞吐量**（`harness/research-schema.md` §3、`harness/SCHEMA.md` §7/§8）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .depth_ar import LN2, DepthARDecoder, commit_input, token_input

__all__ = [
    "Top1ProbPolicy", "PrescribedDepths", "NeverExit",
    "score_stream", "stream_bpc", "generate_with_exit",
    "exit_calibration", "risk_coverage", "layerpass_ledger", "depth_histogram",
    "ece_from_bins", "merge_bin_tables",
]


# ---------- 策略 ----------

class _Policy:
    """策略接口：`decide(depth, logits) -> bool[B,T]`（逐位置）。

    `depth` = 该位置**已跑完的层数**（0 = 输入态，未过任何层）；
    `logits` 形状 `[B,T,V]`（该深度的 tied 读出）。返回 `logits.shape[:-1]` 的 bool。
    实现必须是 (depth, 该位置 logits) 的纯函数 —— 只有这样，增量解码与批量逐层
    两种实现才会给出同一套退出深度（单测钉住）。
    """

    def decide(self, depth: int, logits: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def slice_for(self, i: int, j: int) -> "_Policy":
        """分块评测时的按行切片（无状态策略返回自身，预设策略切自己的深度表）。"""
        return self


def _falses(logits: torch.Tensor) -> torch.Tensor:
    return torch.zeros(logits.shape[:-1], dtype=torch.bool, device=logits.device)


@dataclass(frozen=True)
class Top1ProbPolicy(_Policy):
    """置信门控（CALM 式）：读出 top-1 softmax 概率 ≥ `threshold` 即停。

    - `min_depth`：低于该深度一律不停（默认 1 = 至少跑一层，depth 0 是 embedding
      直连读出，不构成「跑了一层」的语义）；
    - `max_depth`：到达该深度必须停（缺省 = 模型层数 L；由调用方传入）。
    """
    threshold: float
    min_depth: int = 1
    max_depth: int | None = None

    def decide(self, depth: int, logits: torch.Tensor) -> torch.Tensor:
        hi = self.max_depth
        if depth < self.min_depth:
            return _falses(logits)
        if hi is not None and depth >= hi:
            return ~_falses(logits)          # 到顶必停
        p = logits.float().softmax(-1).max(-1).values
        return p >= self.threshold


@dataclass(frozen=True)
class PrescribedDepths(_Policy):
    """预设逐位置退出深度（对照臂 / 一致性单测；不做任何在线判断）。"""
    depths: torch.Tensor                  # [B,T] long，值域 [0, L]

    def decide(self, depth: int, logits: torch.Tensor) -> torch.Tensor:
        d = self.depths.to(logits.device)
        if d.shape != logits.shape[:-1]:
            raise ValueError(f"PrescribedDepths 形状 {tuple(d.shape)} 与 logits "
                             f"{tuple(logits.shape[:-1])} 不符；增量解码链上逐位置"
                             "回放请用 `at(row, pos)` 切片，不要靠广播")
        return d == depth

    def slice_for(self, i: int, j: int) -> "PrescribedDepths":
        return PrescribedDepths(self.depths[i:j])

    def at(self, row: int, pos: int) -> "PrescribedDepths":
        """增量解码链上的单位置切片（`DepthARDecoder.step_auto` 每次只看一个位置）。"""
        return PrescribedDepths(self.depths[row:row + 1, pos:pos + 1])


class NeverExit(_Policy):
    """永不过早退出：所有位置跑满 L 层（退化臂 = 全深度语义的自洽锚点）。"""

    def decide(self, depth: int, logits: torch.Tensor) -> torch.Tensor:
        return _falses(logits)


# ---------- 序列级口径 ----------

def _check_streamable(model) -> int:
    cfg = model.cfg
    if getattr(cfg, "dense", False):
        raise ValueError(
            "dense（stock）形态没有退出/填充语义：逐 token 退出只在非 dense 的"
            "变深度路径上有定义（单因素 scope）")
    if getattr(cfg, "topology", "sequential") != "sequential":
        raise ValueError(
            f"逐 token 退出只定义在 sequential 拓扑上（收到 {cfg.topology!r}）："
            "层间拓扑干预只在 dense 形态下开放")
    return int(cfg.n_layers)


@torch.no_grad()
def score_stream(model, tokens: torch.Tensor, policy: _Policy) -> dict:
    """teacher-forcing、逐位置**在线**退出决策的整段前向。

    返回 `{"depths", "logits", "layer_passes_per_token", "skip_frac", "layers"}`：
    `depths[B,T]` = 每个位置的退出深度；`logits[B,T,V]` = 该位置在**其退出深度**的
    fp32 读出。逐层循环与 `DepthARModel.forward` 的非 dense 分支逐句对应，唯一差别是
    `exit_depths` 在本循环里**在线**产生（其余填充/冻结/门语义完全一致）。
    """
    L = _check_streamable(model)
    cfg = model.cfg
    if tokens.dim() != 2:
        raise ValueError(f"tokens 需要 [B,T]，收到 {tuple(tokens.shape)}")
    B, T = tokens.shape
    dev = tokens.device
    h = model._input_states(tokens)
    frozen = h
    depths = torch.full((B, T), L, dtype=torch.long, device=dev)
    decided = torch.zeros((B, T), dtype=torch.bool, device=dev)
    out_logits = torch.zeros(B, T, model.vocab_size, dtype=torch.float32, device=dev)

    def _apply(depth: int, state: torch.Tensor, want: torch.Tensor) -> None:
        nonlocal depths, frozen, out_logits, decided
        want = want & ~decided
        lg = model.readout(state, depth=depth)
        out_logits = torch.where(want.unsqueeze(-1), lg.float(), out_logits)
        depths = torch.where(want, torch.full_like(depths, depth), depths)
        frozen = torch.where(want.unsqueeze(-1), state, frozen)
        decided = decided | want

    # depth 0：输入态直连读出（缺省策略 min_depth ≥ 1 ⇒ 此调用恒为 no-op，
    # 但语义上必须问一次，否则「预设深度含 0」的臂不成立）
    _apply(0, h, policy.decide(0, model.readout(h, depth=0)))

    for k in range(1, L + 1):
        active = ~decided                       # ≡ depths ≥ k
        g = (torch.sigmoid(model.gates[k - 1](h)) if cfg.carry_gate else None)
        if cfg.fill_kv:
            kv_src = frozen
            if cfg.proj_fill:
                kv_src = model.kv_adapters[k - 1](frozen)
            kv_in = torch.where(active.unsqueeze(-1), h, kv_src)
            out, _ = model.blocks[k - 1](h, kv_from=kv_in, gate_vec=g)
        else:
            mask = model._ragged_mask(depths, k)
            mask = mask.expand(-1, cfg.n_heads, -1, -1)
            out, _ = model.blocks[k - 1](h, attn_mask=mask, gate_vec=g)
        h_next = torch.where(active.unsqueeze(-1), out, frozen)
        _apply(k, h_next, policy.decide(k, model.readout(h_next, depth=k)))
        h = h_next

    # 兜底：策略若永不退出（自定义策略），按深度 L 收口，不留半截状态
    if not bool(decided.all()):
        _apply(L, h, ~decided)

    dp = depths.float()
    return {
        "depths": depths,
        "logits": out_logits,
        "layer_passes_per_token": dp.mean().item(),
        "skip_frac": (1.0 - dp.mean().item() / L),
        "layers": L,
    }


@torch.no_grad()
def stream_bpc(model, tokens: torch.Tensor, policy: _Policy,
               full_argmax: torch.Tensor | None = None) -> dict:
    """`score_stream` + 序列级 bpc（bits/char）+ 按退出深度的损伤分解。

    bpc = `mean CE(第 i 位置退出深度读出, t_{i+1}) / ln2`。另报：
    - `bpc_per_row`（每行的 bpc）：同 checkpoint 比较的**窗口间 SE** 由此算；
    - `by_exit_depth`：按退出深度分桶的 CE 与计数（损伤来自哪些深度，可归因）；
    - `exit_acc_tf` / `exit_acc_true`：退出位置的 top-1 与「全深度 argmax」/「真实
      下一字节」的一致率（前者是本仓历史 `tf@k` 口径，后者是正确性口径；并排报，
      见 55-review C-P02）。

    `full_argmax`（可选，[B,T-1]）：全深度 argmax 的**预计算**值（分块评测时跨块复用，
    省掉每块一次全深度前向）。缺省时本函数自己跑一次全深度前向。
    """
    out = score_stream(model, tokens, policy)
    lg = out["logits"][:, :-1].float()
    tgt = tokens[:, 1:]
    V = lg.shape[-1]
    ce = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="none")
    out["bpc"] = ce.mean().item() / LN2
    out["bpc_per_row"] = (ce.view(lg.shape[0], lg.shape[1]).mean(-1) / LN2)

    d = out["depths"][:, :-1]
    am = lg.argmax(-1)
    if full_argmax is None:
        with torch.no_grad():
            _, aux = model(tokens, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
    else:
        full = full_argmax
    out["exit_acc_tf"] = (am == full).float().mean().item()
    out["exit_acc_true"] = (am == tgt).float().mean().item()

    by_depth = []
    for k in range(out["layers"] + 1):
        m = d == k
        n = int(m.sum())
        by_depth.append({
            "depth": k, "n": n,
            "frac": (n / d.numel()) if d.numel() else 0.0,
            "bpc": (ce.view_as(d)[m].mean().item() / LN2) if n else None,
            "acc_true": ((am[m] == tgt[m]).float().mean().item() if n else None),
            "acc_tf": ((am[m] == full[m]).float().mean().item() if n else None),
        })
    # by_exit_depth 覆盖**预测行**（位置 i 的退出深度读出预测 t_{i+1}）；
    # depths_hist / layer-passes 覆盖**全部位置**（最后一个位置也要前向算力）
    out["by_exit_depth"] = by_depth
    # 逐位置 CE（诊断用：可对任意位置子集做**同位置**的配对比较）
    out["ce_per_pos"] = ce.view(lg.shape[0], lg.shape[1])
    out.pop("logits", None)          # [B,T,V] 太大，不入 results.json
    out["depths_hist"] = depth_histogram(out["depths"], n_layers=out["layers"])
    return out


def depth_histogram(depths: torch.Tensor,
                    n_layers: int | None = None) -> list[dict]:
    """退出深度直方图（算力账的分布面：均值可能藏双峰）。`n_layers` 给定则固定
    桶数 0..n_layers（缺省按实测最大值，会漏掉空桶）。"""
    L = int(depths.max().item()) if n_layers is None else int(n_layers)
    out = []
    n = depths.numel()
    for k in range(L + 1):
        c = int((depths == k).sum())
        out.append({"depth": k, "n": c, "frac": c / n if n else 0.0})
    return out


def layerpass_ledger(out: dict) -> dict:
    """等效理论算力账（**只有理论量**）：layer-passes/token 与跳过层占比。"""
    return {
        "layers": out["layers"],
        "layer_passes_per_token": round(out["layer_passes_per_token"], 6),
        "skip_frac": round(out["skip_frac"], 6),
    }


@torch.no_grad()
def generate_with_exit(model, prompt, n_new: int, policy: _Policy,
                       commit: str = "soft") -> tuple[list[int], dict]:
    """自由生成（逐 token 走生产解码器 `DepthARDecoder.step_auto`）。

    返回 `(gen_tokens, stats)`：`stats["depths"][j]` = **产出第 j 个生成 token 的那个
    位置**的退出深度（位置索引与生成 token 对齐，注意是「产出者位置」不是「消费位置」）；
    `stats["prompt_depths"]` = prompt 段各位置的退出深度。`stats["layer_passes_per_token"]`
    覆盖 prompt + 前 n_new−1 个生成位置（最后一个生成 token 没有后续前向）——即端到端
    算力账，不是只算生成段。
    """
    model.eval()
    dec = DepthARDecoder(model)
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    if len(prompt) + n_new + 1 > model.cfg.seq_len:
        raise ValueError("prompt+n_new exceeds seq_len")
    h = logits = None
    for t in prompt:
        h, logits, _ = dec.step_auto(token_input(model, t, dec.n), policy)
    pre_lp = sum(dec.depths)
    gen: list[int] = []
    gen_depths: list[int] = []
    for _ in range(n_new):
        gen.append(int(logits.argmax()))
        gen_depths.append(int(dec.depths[-1]))
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, commit, dec.n)
        h, logits, _ = dec.step_auto(inp, policy)
    total = sum(dec.depths)
    return gen, {
        "commit": commit,
        "prompt_depths": list(dec.depths[:len(prompt)]),
        "depths": gen_depths,
        "layer_passes_per_token": total / max(len(dec.depths), 1),
        "prompt_layer_passes": pre_lp,
    }


# ---------- 校准（回答 55-review C-P05） ----------

def ece_from_bins(bins: list[dict]) -> float:
    """由分箱汇总（`{n, mean_prob, acc}`）算 ECE —— 分块评测时把各块的箱表合并后
    再算，与一次性算逐位等价（ECE 只依赖每箱的样本数与两个均值）。"""
    n = sum(b["n"] for b in bins)
    if n == 0:
        return 0.0
    return sum(b["n"] / n * abs(b["mean_prob"] - b["acc"])
               for b in bins if b["n"])


def merge_bin_tables(tables: list[list[dict]], n_bins: int) -> list[dict]:
    """合并多块 `ece` 的分箱表（按箱加权合并 mean_prob / acc）。"""
    out = []
    for i in range(n_bins):
        n = 0
        p_sum = acc_sum = 0.0
        for t in tables:
            b = t[i]
            n += b["n"]
            if b["n"]:
                p_sum += b["n"] * b["mean_prob"]
                acc_sum += b["n"] * b["acc"]
        out.append({"bin": i, "range": tables[0][i]["range"], "n": n,
                    "mean_prob": (p_sum / n if n else None),
                    "acc": (acc_sum / n if n else None)})
    return out


def _ece(probs: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> tuple[float, list[dict]]:
    """等宽分箱 ECE（|置信 − 实际正确率| 的样本加权平均）。"""
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    n = max(probs.numel(), 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        m = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        c = int(m.sum())
        if c == 0:
            rows.append({"bin": i, "range": [round(lo, 4), round(hi, 4)], "n": 0,
                         "mean_prob": None, "acc": None})
            continue
        mp = probs[m].mean().item()
        ac = correct[m].float().mean().item()
        ece += c / n * abs(mp - ac)
        rows.append({"bin": i, "range": [round(lo, 4), round(hi, 4)], "n": c,
                     "mean_prob": round(mp, 6), "acc": round(ac, 6)})
    return ece, rows


@torch.no_grad()
def exit_calibration(model, tokens: torch.Tensor, n_bins: int = 10) -> dict:
    """退出决策头的校准表：每深度 top-1 概率 vs 两种「退对了没有」标签。

    标签 ①`tf`：`argmax_k == argmax_L`（「退出会不会改变答案」——本仓历史 tf@k 口径）；
    标签 ②`true`：`argmax_k == 真实下一字节`（正确性口径）。两种并排报是刻意的：
    `55-review` C-P02 指出 `tf@k` 高是「浅层模仿深层」的近同义反复，不能当质量；
    并排才能看出「自洽高、正确性低」的落差。
    """
    L = _check_streamable(model)
    _, aux = model(tokens, exit_depths=None)      # 全深度前向（校准的参照面）
    tgt = tokens[:, 1:]
    full = aux["depth_logits"][-1][:, :-1].argmax(-1)
    rows = []
    for k in range(0, L):
        lg = aux["depth_logits"][k][:, :-1].float()
        if lg.numel() == 0:
            continue
        p, am = lg.softmax(-1).max(-1)
        tf_ok, true_ok = (am == full), (am == tgt)
        ece_tf, bins_tf = _ece(p, tf_ok, n_bins)
        ece_true, bins_true = _ece(p, true_ok, n_bins)
        rows.append({
            "depth": k, "n": int(p.numel()),
            "mean_top1_prob": round(p.mean().item(), 6),
            "acc_tf": round(tf_ok.float().mean().item(), 6),
            "acc_true": round(true_ok.float().mean().item(), 6),
            "overconf_tf": round(p.mean().item() - tf_ok.float().mean().item(), 6),
            "overconf_true": round(p.mean().item() - true_ok.float().mean().item(), 6),
            "ece_tf": round(ece_tf, 6), "ece_true": round(ece_true, 6),
            "bins_tf": bins_tf, "bins_true": bins_true,
        })
    return {"n_bins": n_bins, "rows": rows}


@torch.no_grad()
def risk_coverage(model, tokens: torch.Tensor,
                  thresholds: tuple[float, ...] = (0.3, 0.6, 0.9, 0.95, 0.99),
                  depths: tuple[int, ...] | None = None) -> list[dict]:
    """逐深度 risk–coverage：在深度 k 用阈值 τ 当退出判据时，覆盖（保留比例）
    与保留部分的两种准确率。用来回答「top-1 概率当门控靠不靠谱、在哪一段可用」。
    """
    L = _check_streamable(model)
    depths = tuple(range(1, L)) if depths is None else tuple(depths)
    _, aux = model(tokens, exit_depths=None)
    tgt = tokens[:, 1:]
    full = aux["depth_logits"][-1][:, :-1].argmax(-1)
    out = []
    for k in depths:
        lg = aux["depth_logits"][k][:, :-1].float()
        p, am = lg.softmax(-1).max(-1)
        tf_ok, true_ok = (am == full), (am == tgt)
        for tau in thresholds:
            keep = p >= tau
            c = int(keep.sum())
            out.append({
                "depth": k, "tau": tau, "coverage": round(c / max(p.numel(), 1), 6),
                "n_kept": c, "n_total": int(p.numel()),
                "acc_tf_kept": (round(tf_ok[keep].float().mean().item(), 6) if c else None),
                "acc_true_kept": (round(true_ok[keep].float().mean().item(), 6) if c else None),
                "acc_true_dropped": (round(true_ok[~keep].float().mean().item(), 6)
                                     if c < p.numel() else None),
            })
    return out
