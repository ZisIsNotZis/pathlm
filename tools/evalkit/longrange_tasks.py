"""合成长程任务电池（ticket 34 契约 A）——「局部先验无法补全」的长程天平原语。

契约：`axes/M-carrier/M1-carrier-type/experiments/2617-longrange-task-battery/README.md`

## 为什么需要它（SCHEMA §19）

enwik8 的局部统计极强：13M 主干「不需要」真正的长程记忆也能靠语言先验补强。
在 enwik8 上测「信息能否往右搬」，会把**模型不需要**误读成**机制搬不动**。
本模块提供四个合成任务，答案**不可由局部先验猜出**——只有把远处的信息记住并
在提问处取出来才答得对。凡涉及长程记忆 / 压缩 / 信息搬运的结论，都要在这件
天平上复核一次。

## 四个任务（每个都有：可调距离/长度、答案不可由局部先验猜出、种子可控）

1. **关联回忆** `assoc_recall_batch`：K 对 (key, value) 连续排在最前面，末尾
   用 cue 提问某个 key，问它的 value；
2. **Induction** `induction_batch`：随机前缀，把其中一段原样重复到末尾，问
   「重复段之后的那一项」（要求用「之前出现过同样的片段」做续写）；
3. **复制** `copy_batch`：随机片段出现在远处，用 cue 要求把整段复制出来；
4. **变距离回忆** `var_recall_batch`：单个 (key, value) 对，距离 d，末尾提问。

## 三条基线（契约判据 (a)）

- **chance**：答案集（`symbols = {0..sym_vocab-1}`）上的均匀猜测 `1/sym_vocab`。
  另报**模型原始词表**的均匀基线 `1/vocab_size`——两者不同，见 `chance_dict`。
- **局部先验** `local_prior_accuracy`：**只看最近 k 个 token** 的最强预测器
  （最长后缀最近邻 + 窗口内词频回退）。答案若在窗口外，它只能退化为 chance
  ——这正是「局部先验抗性」的检验。
- **真值上界** = 1.0：四个任务在完整上下文下由构造决定可解（答案由写入的
  (key,value) / 前缀 / 源片段唯一确定）。这是「信息搬运成功」的天花板。

## 与既有仪器的关系

`retention.py` 量的是「删掉某些 KV 后掉多少（依赖）」与「针还在不在（保留）」。
本模块量的是**任务侧**：不靠长程就答不出的任务上，模型到底答对多少。
`recall_under_visibility` 直接复用 `retention.recent_keep_mask`（只保留最近 d 个
位置的可见性），不重写删除逻辑。

## token 约定

- 符号（key/value/片段内容）：`0 .. sym_vocab-1`（缺省 64），取自真实 byte id，
  但只当**随机符号**用；
- 间隙 `filler_token`（缺省 = `sym_vocab`）：把序列填满，与符号/提示词不重叠，
  避免「答案在 filler 里又出现一次」的混杂；
- 提问标记 cue：`cue_token`（缺省 = 主干的 mask 槽 = vocab_size-1），真实文本里
  不出现，是本仓库既有的 needle 构造同款约定。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import torch

from tools.evalkit.retention import masked_logits  # 复用既有前向（掩码语义同一件仪器）

#: 契约指定的变距离回忆距离档。
DISTANCES = (16, 32, 64, 128, 192, 255)
#: 局部先验基线缺省窗口（只看最近 k 个 token）。必须**严格小于**各任务的距离
#: （`k < min(DISTANCES)`），否则源信息落进局部窗口、局部先验就能看到答案
#: （任务构造失败）。实测：k=16 时变距离回忆 d=16 的 value 恰落在窗口边缘，
#: 局部先验被抬到 0.5；改 k=8 后回落到 chance。由 `check_local_prior_resistance` 把关。
DEFAULT_LOCAL_K = 8


@dataclass
class TaskBatch:
    """一个任务批：`[B, T]` token + 每个查询位置 `[B, M]` 的行/答案/距离。

    - `rows[b, m]`：查询行（`logits[rows[b,m]]` 预测 `rows[b,m]+1` 处的 token）；
    - `answers[b, m]`：该查询的正确答案（符号 id）；
    - `dists[b, m]`：该查询的「距离」（源信息到查询行的位置差；各任务的物理
      含义见生成器 docstring）——距离-准确率曲线按它分桶；
    - `n_answers`：答案集大小（均匀 chance = `1/n_answers`）；
    - `meta`：任务名、参数、符号集大小等，供落盘与复核。
    """

    name: str
    tokens: torch.Tensor                     # [B, T] int64
    rows: torch.Tensor                       # [B, M] int64
    answers: torch.Tensor                    # [B, M] int64
    dists: torch.Tensor                      # [B, M] int64
    n_answers: int
    meta: dict = field(default_factory=dict)

    @property
    def B(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def T(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def M(self) -> int:
        return int(self.rows.shape[1])

    def chance(self) -> float:
        """均匀猜测答案集的命中率。"""
        return 1.0 / self.n_answers

    def chance_dict(self, vocab_size: int | None = None) -> dict:
        """chance 的两个口径（必须分开报，契约 §12 口径对账）。

        - `uniform_answer_set` = `1/n_answers`：在**答案集**上均匀猜（本电池的
          headline chance，也是「打乱答案」变异验证比较的对象）；
        - `uniform_model_vocab` = `1/vocab_size`：模型输出层在**原始词表**上均匀
          分布时命中正确答案的概率（更低的朴素基线，需要时给出）。
        """
        out = {"uniform_answer_set": round(self.chance(), 8), "n_answers": self.n_answers}
        if vocab_size is not None:
            out["uniform_model_vocab"] = round(1.0 / vocab_size, 8)
            out["vocab_size"] = vocab_size
        return out


def _as_int_list(t: torch.Tensor) -> list[int]:
    return [int(v) for v in t.tolist()]


# ----------------------------------------------------------------- 生成器


def assoc_recall_batch(batch_size: int, seq_len: int, n_pairs: int,
                       sym_vocab: int = 64, filler_token: int = 64,
                       cue_token: int = 205, generator: torch.Generator | None = None,
                       query_index: int | None = None) -> TaskBatch:
    """关联回忆：K 对 (key, value) 连续排在 `[0, 2K)`，末尾提问。

    布局：`k_0 v_0 k_1 v_1 … k_{K-1} v_{K-1}` `[filler…]` `cue` `query_key`。
    查询行 = `T-1`（输入最后一个 token = 被问的 key），答案 = 对应的 value。
    距离 = `(T-1) - 2*query_index`（被问的那一对的 key 离提问处多远）。

    - keys：每行内**互不相同**的随机符号（`randperm`），关联无歧义；
    - values：随机符号（可重复）；
    - filler：`[2K, T-2)` 填 `filler_token`，与符号/提示词不重叠。

    `query_index=None` 时每行随机选一对提问；给整数则固定问第 `query_index` 对
    （距离曲线的控制旋钮）。
    """
    T = int(seq_len)
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if sym_vocab < n_pairs:
        raise ValueError(f"sym_vocab={sym_vocab} < n_pairs={n_pairs}：key 要互不相同")
    if 2 * n_pairs + 2 > T:
        raise ValueError(f"2*n_pairs+2={2 * n_pairs + 2} > seq_len={T}")
    if query_index is not None and not 0 <= query_index < n_pairs:
        raise ValueError(f"query_index must be in [0, {n_pairs}), got {query_index}")

    tok = torch.full((batch_size, T), filler_token, dtype=torch.long)
    keys = torch.stack([torch.randperm(sym_vocab, generator=generator)[:n_pairs]
                        for _ in range(batch_size)])                 # [B, K] distinct
    values = torch.randint(0, sym_vocab, (batch_size, n_pairs), generator=generator)
    for b in range(batch_size):
        for j in range(n_pairs):
            tok[b, 2 * j] = keys[b, j]
            tok[b, 2 * j + 1] = values[b, j]
    tok[:, T - 2] = cue_token
    if query_index is None:
        qi = torch.randint(0, n_pairs, (batch_size,), generator=generator)
    else:
        qi = torch.full((batch_size,), int(query_index), dtype=torch.long)
    ar = torch.arange(batch_size)
    tok[ar, T - 1] = keys[ar, qi]
    rows = torch.full((batch_size, 1), T - 1, dtype=torch.long)
    answers = values[ar, qi].view(-1, 1)
    dists = ((T - 1) - 2 * qi).view(-1, 1)
    return TaskBatch(
        name="assoc_recall", tokens=tok, rows=rows, answers=answers, dists=dists,
        n_answers=sym_vocab,
        meta={"task": "assoc_recall", "seq_len": T, "n_pairs": n_pairs,
              "sym_vocab": sym_vocab, "filler_token": filler_token,
              "cue_token": cue_token, "query_index": query_index,
              "layout": "[k0 v0 ... k{K-1} v{K-1}] filler cue query_key @ T-1"})


def induction_batch(batch_size: int, seq_len: int, prefix_len: int, copy_len: int,
                    sym_vocab: int = 64, filler_token: int = 64, cue_token: int = 205,
                    generator: torch.Generator | None = None,
                    src_start: int | None = None) -> TaskBatch:
    """Induction：随机前缀，把其中一段原样重复到末尾，问「重复段之后那一项」。

    布局：`prefix[0, prefix_len)` 随机符号；`[prefix_len, T-copy_len)` filler；
    `[T-copy_len, T)` = `prefix[a : a+copy_len]`（原样复制）。查询行 = `T-1`
    （输入最后一项 = 复制段的最后一项），答案 = `prefix[a+copy_len]`（原前缀里
    紧跟其后的一项）。距离 = `(T-1) - (a+copy_len-1)`（源片段末端到提问处）。

    要答对必须**认出末尾这段在前面出现过**并续写；`copy_len > k` 时局部窗口
    装不下源片段 ⇒ 局部先验只能退化为 chance。

    `src_start=None` 时每行随机 `a ∈ [0, prefix_len-copy_len-1]`；给整数则固定
    （距离曲线的控制旋钮）。
    """
    T = int(seq_len)
    if copy_len < 1:
        raise ValueError(f"copy_len must be >= 1, got {copy_len}")
    if prefix_len + copy_len > T:
        raise ValueError(f"prefix_len+copy_len={prefix_len + copy_len} > seq_len={T}")
    if prefix_len - copy_len - 1 < 0:
        raise ValueError(f"prefix_len={prefix_len} 太小，放不下 copy_len={copy_len} 的源+答案")
    if sym_vocab < 1:
        raise ValueError("sym_vocab must be >= 1")
    if src_start is not None and not 0 <= src_start <= prefix_len - copy_len - 1:
        raise ValueError(f"src_start must be in [0, {prefix_len - copy_len - 1}], "
                         f"got {src_start}")

    tok = torch.full((batch_size, T), filler_token, dtype=torch.long)
    prefix = torch.randint(0, sym_vocab, (batch_size, prefix_len), generator=generator)
    tok[:, :prefix_len] = prefix
    if src_start is None:
        a = torch.randint(0, prefix_len - copy_len, (batch_size,), generator=generator)
    else:
        a = torch.full((batch_size,), int(src_start), dtype=torch.long)
    for b in range(batch_size):
        tok[b, T - copy_len:] = prefix[b, a[b]:a[b] + copy_len]
    ar = torch.arange(batch_size)
    rows = torch.full((batch_size, 1), T - 1, dtype=torch.long)
    answers = prefix[ar, a + copy_len].view(-1, 1)
    dists = ((T - 1) - (a + copy_len - 1)).view(-1, 1)
    return TaskBatch(
        name="induction", tokens=tok, rows=rows, answers=answers, dists=dists,
        n_answers=sym_vocab,
        meta={"task": "induction", "seq_len": T, "prefix_len": prefix_len,
              "copy_len": copy_len, "sym_vocab": sym_vocab,
              "filler_token": filler_token, "cue_token": cue_token,
              "src_start": src_start,
              "layout": "prefix + filler + [copied prefix segment] @ tail; src=a"})


def copy_batch(batch_size: int, seq_len: int, copy_len: int, src_pos: int,
               sym_vocab: int = 64, filler_token: int = 64, cue_token: int = 205,
               generator: torch.Generator | None = None) -> TaskBatch:
    """复制：随机片段出现在远处，用 cue 要求把整段复制出来（teacher-forced）。

    布局：源片段 `S` 在 `[src_pos, src_pos+copy_len)`；`[.., T-copy_len-1)` filler；
    cue 在 `T-copy_len-1`；输出块 `[T-copy_len, T)` 在输入里**已填入** `S`
    （teacher forcing，只读模型在各行的预测）。查询行 = `T-copy_len-1+k`
    （`k=0..copy_len-1`），答案 = `S[k]`。距离 = `(T-copy_len-1) - src_pos`
    （源片段到 cue 的间隔；对该片段每个位置相同）。

    局部窗口装不下源片段（`copy_len > k` 且 `src_pos` 远离末尾）⇒ 局部先验退化
    为 chance。
    """
    T = int(seq_len)
    if copy_len < 1 or copy_len > T:
        raise ValueError(f"copy_len must be in [1, {T}], got {copy_len}")
    if src_pos < 0 or src_pos + copy_len > T - copy_len - 1:
        raise ValueError(f"src_pos={src_pos} 越界：需 0 <= src_pos 且 "
                         f"src_pos+{copy_len} <= {T - copy_len - 1}")

    tok = torch.full((batch_size, T), filler_token, dtype=torch.long)
    source = torch.randint(0, sym_vocab, (batch_size, copy_len), generator=generator)
    tok[:, src_pos:src_pos + copy_len] = source
    tok[:, T - copy_len - 1] = cue_token
    tok[:, T - copy_len:T] = source                    # 输出块（teacher-forced 输入）
    qi = torch.arange(copy_len).view(1, -1).expand(batch_size, -1)
    rows = qi + (T - copy_len - 1)                     # [B, m]
    dists = torch.full((batch_size, copy_len), (T - copy_len - 1) - src_pos,
                       dtype=torch.long)
    return TaskBatch(
        name="copy", tokens=tok, rows=rows, answers=source.clone(), dists=dists,
        n_answers=sym_vocab,
        meta={"task": "copy", "seq_len": T, "copy_len": copy_len, "src_pos": src_pos,
              "sym_vocab": sym_vocab, "filler_token": filler_token,
              "cue_token": cue_token,
              "layout": "S @ src_pos; filler; cue @ T-m-1; output block S @ tail"})


def var_recall_batch(batch_size: int, seq_len: int, distance: int,
                     sym_vocab: int = 64, filler_token: int = 64, cue_token: int = 205,
                     generator: torch.Generator | None = None) -> TaskBatch:
    """变距离回忆：单个 (key, value) 对，距离 d，末尾用 cue 提问。

    布局：key 在 `src_pos = (T-1) - d`，value 在 `src_pos+1`；filler 填补；
    cue 在 `T-2`；query key（与 key 相同）在 `T-1`。查询行 = `T-1`，答案 = value。
    `d = (T-1) - src_pos`（源 key 到提问行的位置差，即契约的距离档）。

    与 `retention.py` 兼容：用 `recall_under_visibility` 复用
    `recent_keep_mask(T, d_keep)` 量「只保留最近 d_keep 个位置」时的召回曲线。
    """
    T = int(seq_len)
    if not 1 <= distance <= T - 1:
        raise ValueError(f"distance must be in [1, {T - 1}], got {distance}")
    src_pos = (T - 1) - distance
    if src_pos + 1 >= T - 2:
        raise ValueError(f"distance={distance} 与 cue 位置冲突（value 撞 cue）")
    tok = torch.full((batch_size, T), filler_token, dtype=torch.long)
    key = torch.randint(0, sym_vocab, (batch_size,), generator=generator)
    value = torch.randint(0, sym_vocab, (batch_size,), generator=generator)
    tok[:, src_pos] = key
    tok[:, src_pos + 1] = value
    tok[:, T - 2] = cue_token
    tok[:, T - 1] = key
    rows = torch.full((batch_size, 1), T - 1, dtype=torch.long)
    return TaskBatch(
        name="var_recall", tokens=tok, rows=rows, answers=value.view(-1, 1),
        dists=torch.full((batch_size, 1), distance, dtype=torch.long),
        n_answers=sym_vocab,
        meta={"task": "var_recall", "seq_len": T, "distance": distance,
              "src_pos": src_pos, "sym_vocab": sym_vocab,
              "filler_token": filler_token, "cue_token": cue_token,
              "layout": "key @ (T-1-d), value @ (T-d); filler; cue @ T-2; query key @ T-1"})


# ----------------------------------------------------------------- 基线


def _answer_set(task: TaskBatch) -> set[int]:
    """可答答案集（本电池里 = 符号集 `{0..n_answers-1}`）。"""
    return set(range(task.n_answers))


def local_prior_predict(seq: list[int], row: int, k: int,
                        answer_set: set[int]) -> int:
    """**只看最近 k 个 token** 的局部先验预测器（可失败单测钉住）。

    规则（最强局部预测器）：
    1. 取窗口 `seq[max(0,row-k+1) : row+1]`（含当前 token）；
    2. 从**最长后缀**开始找「同一后缀在窗口内更早处出现过」的最近一次；
       命中的话预测「那次之后紧跟的 token」——但只接受落在 `answer_set` 里的
       候选（预测 filler/提示词永远不可能对，理性的局部预测器会把质量集中在
       可答集合上）；
    3. 无有效命中 → 回退到窗口内在 `answer_set` 里出现最多的符号（并列取最小
       id）；窗口里没有任何符号 → 返回 `min(answer_set)`（均匀退化的产物）。

    它故意保守：**只看 k 个 token**，看不到更远的源信息。答案是「不使用长程
    记忆时最强的猜法」，即局部先验基线。
    """
    if not answer_set:
        raise ValueError("answer_set 不能为空")
    start = max(0, row - k + 1)
    window = seq[start:row + 1]
    n = len(window)
    for s in range(n - 1, 0, -1):              # 最长后缀优先
        suffix = window[n - s:]
        for i in range(n - s - 1, -1, -1):     # 更早处：最近的优先
            if window[i:i + s] == suffix:
                cand = window[i + s]
                if cand in answer_set:
                    return cand
    counts = Counter(t for t in window if t in answer_set)
    if counts:
        return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return min(answer_set)


def _summarize(hits: torch.Tensor, dists: torch.Tensor) -> dict:
    """把逐查询命中 `[B,M]` 与距离 `[B,M]` 聚合成总准确率 + 距离分桶曲线。"""
    hits = hits.to(torch.float64)
    dists = dists.to(torch.long)
    by_dist: dict[str, dict] = {}
    for d in sorted(set(int(v) for v in dists.flatten().tolist())):
        sel = dists == d
        by_dist[str(d)] = {"accuracy": round(float(hits[sel].mean().item()), 6),
                           "n": int(sel.sum().item())}
    return {"accuracy": round(float(hits.mean().item()), 6),
            "n": int(hits.numel()),
            "n_hits": int(hits.sum().item()),
            "by_distance": by_dist}


def local_prior_accuracy(task: TaskBatch, k: int = DEFAULT_LOCAL_K) -> dict:
    """局部先验基线的总准确率 + 距离-准确率曲线（与 `accuracy` 同口径）。"""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    answer_set = _answer_set(task)
    tok = task.tokens.tolist()
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for b in range(task.B):
        row_b = task.rows[b].tolist()
        ans_b = task.answers[b].tolist()
        for m in range(task.M):
            pred = local_prior_predict(tok[b], int(row_b[m]), k, answer_set)
            hits[b, m] = int(pred == ans_b[m])
    return {**_summarize(hits, task.dists), "k": k}


@torch.no_grad()
def accuracy(model, task: TaskBatch, keep_mask: torch.Tensor | None = None,
             device=None, max_chunk: int = 64) -> dict:
    """模型在任务上的 top-1 准确率 + 距离-准确率曲线。

    `keep_mask`：`[T,T]` 共享或 `[B,T,T]` 逐样本（与 `retention.masked_logits`
    同一语义）；`None` = 全因果（释放版）。传 `retention.recent_keep_mask` 的
    输出即可量「只保留最近 d 个位置」时的召回——删除逻辑复用，不重写。

    `max_chunk`：把批切成若干块逐块前向（判据与读数不受影响，只是不一次物化
    整批的 `[B,H,T,T]` 注意力矩阵——拼接出来的大电池会 OOM）。
    """
    if max_chunk < 1:
        raise ValueError(f"max_chunk must be >= 1, got {max_chunk}")
    dev = device if device is not None else next(model.parameters()).device
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        if keep_mask is None:
            km = None
        elif keep_mask.dim() == 3:
            km = keep_mask[lo:hi].to(dev)
        else:
            km = keep_mask.to(dev)
        logits = masked_logits(model, task.tokens[lo:hi].to(dev), km)
        pred = logits.argmax(-1)                               # [b, T]
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(dev))  # [b, M]
        hits[lo:hi] = (sel == task.answers[lo:hi].to(dev)).to(torch.long).cpu()
    return _summarize(hits, task.dists)


def task_summary(model, task: TaskBatch, local_k: int = DEFAULT_LOCAL_K,
                 vocab_size: int | None = None, keep_mask: torch.Tensor | None = None,
                 device=None) -> dict:
    """一件任务的完整读数：chance（两口径）+ 局部先验 + 模型（含距离曲线）。"""
    return {"name": task.name,
            "params": dict(task.meta),
            "chance": task.chance_dict(vocab_size),
            "local_prior": local_prior_accuracy(task, k=local_k),
            "model": accuracy(model, task, keep_mask=keep_mask, device=device)}


@torch.no_grad()
def recall_under_visibility(model, task: TaskBatch, keep_ds, device=None,
                            keep_fn=None) -> dict:
    """变距离回忆 × 可见性：复用 `retention.recent_keep_mask` 量召回曲线。

    `keep_fn(T, d) -> [T,T]` 缺省 = `retention.recent_keep_mask`（只保留最近 d 个
    位置）。返回 `{str(d): accuracy_dict}`——「源信息落在可见窗口外时还答不答得
    出来」= 长程记忆的直接读数。
    """
    from tools.evalkit.retention import recent_keep_mask
    keep_fn = keep_fn or (lambda T, d: recent_keep_mask(T, d))
    out = {}
    for d in keep_ds:
        if not 0 < d <= task.T:
            raise ValueError(f"keep d must be in (0, {task.T}], got {d}")
        km = keep_fn(task.T, d)
        out[str(d)] = accuracy(model, task, keep_mask=km, device=device)
    return out


# ----------------------------------------------------------------- 变异验证


def corrupt_span(task: TaskBatch, lo: int, hi: int,
                 generator: torch.Generator | None = None) -> TaskBatch:
    """**源信息对照臂**：把 `[lo, hi)` 的 token 换成全新随机符号（同分布），
    其余（查询行 / 答案 / 距离）逐位不变。

    用法：复制 / induction 任务里把「源片段」毁掉。若主臂读数来自那段源信息，
    毁掉后准确率必落回 chance；若毁掉后仍然高，说明读数来自位置/局部先验
    （任务或仪器有混杂）。这是对「弱但真实的长程能力」这类结论的必要对照。
    """
    if not 0 <= lo < hi <= task.T:
        raise ValueError(f"span [{lo}, {hi}) 越界（T={task.T}）")
    t = task.tokens.clone()
    t[:, lo:hi] = torch.randint(0, task.n_answers, (task.B, hi - lo),
                                generator=generator)
    return TaskBatch(name=task.name + "/corrupt_source", tokens=t, rows=task.rows,
                     answers=task.answers, dists=task.dists,
                     n_answers=task.n_answers,
                     meta={**task.meta, "mutated": "corrupt_span",
                           "corrupt_span": [lo, hi]})


def shuffle_answers(task: TaskBatch, generator: torch.Generator | None = None) -> TaskBatch:
    """**变异验证**：把答案标签整体随机置换（答案与上下文解耦）。

    一个只靠猜测的读法下，打乱答案后的准确率必须落回 chance——若没有，说明
    准确率不是从上下文来的（任务/仪器有混杂）。单测用 `accuracy` 比较两版。
    """
    flat = task.answers.flatten()
    perm = torch.randperm(flat.numel(), generator=generator)
    shuffled = flat[perm].view(task.answers.shape)
    return TaskBatch(name=task.name + "/shuffled", tokens=task.tokens, rows=task.rows,
                     answers=shuffled, dists=task.dists, n_answers=task.n_answers,
                     meta={**task.meta, "mutated": "shuffle_answers"})


# ----------------------------------------------------------------- 护栏


def check_local_prior_resistance(task: TaskBatch, k: int = DEFAULT_LOCAL_K,
                                 factor: float = 3.0, floor: float = 0.02) -> dict:
    """契约判据 (a)：局部先验必须**显著低于**真值（真值上界 = 1.0）。

    判据线：`local_prior.accuracy < ceiling`，其中
    `ceiling = max(factor * chance, floor)`。`factor=3` 是「≈ chance 的 3 倍」
    的经验容差，`floor=0.02` 防 `chance` 过小导致容差退化。
    """
    lp = local_prior_accuracy(task, k=k)
    ceiling = max(factor * task.chance(), floor)
    return {"task": task.name, "chance": round(task.chance(), 6),
            "ceiling": round(ceiling, 6), "local_prior_accuracy": lp["accuracy"],
            "passed": lp["accuracy"] < ceiling, "n": lp["n"], "k": k,
            "by_distance": lp["by_distance"]}
