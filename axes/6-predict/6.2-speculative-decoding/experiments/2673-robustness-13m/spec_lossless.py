"""WP 6.2.4 · 2673-robustness-13m：**温度 T=1 的无损投机采样**（库）+ 统计量。

本模块**只读** import `2640-tree-conf-13m/spec_tree.py`（`Decoder` / `tree_forward` /
`write_back_accepted_row` / `_slice_nodes` / `_take_row`：K 行平铺验证前向，给出每个被验证
位置的 **node-1 logits** = 目标分布 p_target）与 `pathlm/**`（不改）。**不修改** 2640/2650/2660。

## 为什么这样实现（一页速读）

生产解码器 `pathlm.decode.decode_spec` 是**贪心**验证（候选必须 == 目标 argmax）。温度>0 下
「无损」的正确形态是 Leviathan/Chen 的**接受–残差重采**：对每个被验证位置，草稿分布 q 提议
一个 token x，以 `min(1, p_target(x)/q_draft(x))` 接受；拒绝时从残差分布
`norm(max(0, p_target − q_draft))` 重采。这样该位置**发出 token 的边缘分布恰为目标分布**。

引擎的轮次布局（与 2640/`decode_spec` 逐位一致）：

```
dec.n = 待填位置；dec.last_nodes = 位置 dec.n−1 的 node 输出
v0 = 位置 dec.n 的目标采样（node-1 分布）        # 「验证器自己的输出」⇒ 不打接受测试
d2 = 位置 dec.n+1 的草稿（node-2 直连头分布 q2）
d3 = 位置 dec.n+2 的草稿（node-3 直连头分布 q3）
一次前向 rows=[[v0, d2, d3]]（K=3）→ logs[0,0]=p2（给 dec.n+1）、logs[0,1]=p3（给 dec.n+2）
```

**拒绝时残差重采出的 token 属于「已被前向过的位置」，本引擎里没有它的 K/V。**若直接把它喂进
下一次前向，就多花一次主干前向（与契约 §5「残差重采不增加主干前向」相悖）。本实现的解法是
**carry（结转）**：拒绝时发出的重采 token **不立刻写回缓存**，而是作为**下一轮的 v0** 结转过去
——下一轮的前向 `[carry, d2', d3']` 会把 carry 放在位置 `dec.n` 上并顺带算出它的 K/V。
于是**每轮恒 1 次主干前向**，tokens-per-forward = 1+τ 成立（τ = 每轮平均接受草稿数）。

无损性（边缘分布恰为目标分布）对**任意归一化草稿 q** 都成立；q 只影响接受率，不影响正确性。
本实现取**直连头** node-2 / node-3 的分布当 q（= `decode_spec` 的默认草稿来源）。

## 可证伪

`accept_always=True`（接受概率恒为 1）或 `resample_from_draft=True`（残差重采写成直接用草稿
分布）都是**故意改错的接受规则**：此时位置 dec.n+1 的边缘分布 = 草稿 q ≠ 目标 p，无损性检验
**必须变红**。见 `test_spec_lossless.py` 的 L0 gate probe。
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DIR2640 = os.path.abspath(os.path.join(HERE, "..", "2640-tree-conf-13m"))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for _p in (DIR2640, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_tree as T  # noqa: E402  2640 的只读实现

#: 固定温度（契约 §2：T 固定为 1.0，不为好看的数字扫 T）
TEMPERATURE = 1.0
#: 投机位数（位置 dec.n+1, dec.n+2）；K = 1 + GAMMA = 3 是每次前向的位置数
GAMMA = T.GAMMA
#: 残差/概率的对数保护
_EPS = 1e-30


def _softmax_last(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float() / max(TEMPERATURE, 1e-6), -1)


def draft_dists(nodes: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """位置 dec.n+1 / dec.n+2 的**草稿分布**（node-2 / node-3 直连头，softmax 概率）。"""
    q2 = _softmax_last(nodes[2]["logits"][0, 0])
    q3 = _softmax_last(nodes[3]["logits"][0, 0])
    return q2, q3


def target_dist_at_here(nodes: dict) -> torch.Tensor:
    """位置 dec.n 的**目标分布**（node-1 分布，softmax 概率）——v0 从它采样。"""
    return _softmax_last(nodes[1]["logits"][0, 0])


def accept_prob(p_x: float, q_x: float) -> float:
    """`min(1, p_target(x)/q_draft(x))`；`q_x == 0` 时按 1（概率 0 的样本不会出现）。"""
    if q_x <= 0.0:
        return 1.0
    r = p_x / q_x
    return 1.0 if r >= 1.0 else r


def accept_test(p: torch.Tensor, q: torch.Tensor, x: int, u: float) -> bool:
    """接受判据：`u <= min(1, p[x]/q[x])`。u 为 U(0,1) 抽样值。"""
    return u <= accept_prob(float(p[x]), float(q[x]))


def residual_dist(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """残差分布 `norm(max(0, p − q))`；退化（全 0）时退回 p（拒绝事件下不应发生）。"""
    r = torch.clamp(p - q, min=0.0)
    s = float(r.sum())
    if s <= 0.0:
        return p.clone()
    return r / s


def _emit_logp(dist: torch.Tensor, tok: int) -> float:
    return float(torch.log(dist[tok] + _EPS))


def _sample(dist: torch.Tensor, gen: torch.Generator) -> int:
    return int(torch.multinomial(dist, 1, generator=gen))


@torch.no_grad()
def spec_sample_T1(model, prompt: torch.Tensor, n_new: int, pcap, seed: int, *,
                   accept_always: bool = False, resample_from_draft: bool = False):
    """温度 T=1 的**无损投机采样**（接受 + 残差重采 + carry 结转）。

    参数：
    - `accept_always`（**故意改错**）：跳过接受判据，草稿恒被接受 ⇒ 发出位置 dec.n+1 的
      边缘分布 = 草稿 q ≠ 目标 p。用于 L0 gate probe（必须让无损性检验变红）。
    - `resample_from_draft`（**故意改错**）：拒绝时直接从草稿分布 q 重采（而非残差 max(0,p−q)）
      ⇒ 同样破坏边缘分布。

    返回 `(gen, stats)`：`gen` 是长度 n_new 的发出 token 列表；`stats` 含
    `logp`（每个发出 token 在其**目标条件分布**下的 log 概率）、`accepts`（逐轮接受草稿数）、
    `n_forward`（轮数 = 主干前向次数）、`carry_rounds`、`n_rounds`、`cache_len`。
    """
    gen_dev = str(model.embed.weight.device)
    gen = torch.Generator(device=gen_dev)
    gen.manual_seed(int(seed))
    dec = T.Decoder(model, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    K = 1 + GAMMA
    out: list[int] = []
    logp: list[float] = []
    accepts: list[int] = []
    n_forward = 0
    carry: int | None = None
    carry_logp = 0.0
    carry_rounds = 0
    while len(out) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None, "step must populate last_nodes"
        p1 = target_dist_at_here(nodes)
        q2, q3 = draft_dists(nodes)
        if carry is None:
            v0 = _sample(p1, gen)
            v0_logp = _emit_logp(p1, v0)
        else:
            v0, v0_logp = carry, carry_logp
            carry_rounds += 1
        d2 = _sample(q2, gen)
        d3 = _sample(q3, gen)
        if dec.n + K > model.mcfg.seq_len:
            break
        rows = [[v0, d2, d3]]
        _h, fnodes, new_kv, p0 = T.tree_forward(model, dec, rows)
        n_forward += 1
        lp = fnodes[1]["logits"][0]
        p2 = torch.softmax(lp[0].float() / max(TEMPERATURE, 1e-6), -1)   # 给 dec.n+1（条件 = v0）
        p3 = torch.softmax(lp[1].float() / max(TEMPERATURE, 1e-6), -1)   # 给 dec.n+2（条件 = v0,d2）
        if carry is None:
            out.append(v0)
            logp.append(v0_logp)
        n_acc = 0
        carry = None
        if accept_always:
            ok2 = True
        else:
            ok2 = accept_test(p2, q2, d2, float(torch.rand((), generator=gen, device=gen.device)))
        if ok2:
            out.append(d2)
            logp.append(_emit_logp(p2, d2))
            n_acc = 1
            if accept_always:
                ok3 = True
            else:
                ok3 = accept_test(p3, q3, d3, float(torch.rand((), generator=gen, device=gen.device)))
            if ok3:
                out.append(d3)
                logp.append(_emit_logp(p3, d3))
                n_acc = 2
            else:
                y3 = _resample(p3, q3, gen, resample_from_draft)
                out.append(y3)
                logp.append(_emit_logp(p3, y3))
                carry, carry_logp = y3, _emit_logp(p3, y3)
        else:
            y2 = _resample(p2, q2, gen, resample_from_draft)
            out.append(y2)
            logp.append(_emit_logp(p2, y2))
            carry, carry_logp = y2, _emit_logp(p2, y2)
        n_pos = 1 + n_acc
        T.write_back_accepted_row(dec, new_kv, p0, 0, n_pos)
        dec.last_nodes = T._take_row(T._slice_nodes(fnodes, n_pos - 1), 0)
        accepts.append(n_acc)
    n = min(len(out), n_new)
    n_rounds = n_forward
    return out[:n], {
        "logp": logp[:n],
        "accepts": accepts,
        "n_forward": n_forward,
        "n_rounds": n_rounds,
        "carry_rounds": carry_rounds,
        "tokens_per_forward": (n / n_rounds) if n_rounds else None,
        "tau": (n / n_rounds - 1.0) if n_rounds else None,
        "cache_len": dec.cache_len(),
    }


def _resample(p: torch.Tensor, q: torch.Tensor, gen: torch.Generator,
              from_draft: bool) -> int:
    """拒绝时的重采：正确 = 残差 `norm(max(0,p−q))`；改错 = 直接用草稿分布 q。"""
    return _sample(q if from_draft else residual_dist(p, q), gen)


@torch.no_grad()
def target_sample_T1(model, prompt: torch.Tensor, n_new: int, pcap, seed: int):
    """参照臂：冻结主干 node-1 **直接在 T=1 采样**（无草稿、无验证），每 token 1 次前向。

    返回 `(gen, stats)`，与 `spec_sample_T1` 同形（`logp` = 每个发出 token 在其目标条件
    分布下的 log 概率）。`tokens_per_forward ≡ 1`、`tau ≡ 0`（定义）。
    """
    gen_dev = str(model.embed.weight.device)
    g = torch.Generator(device=gen_dev)
    g.manual_seed(int(seed))
    dec = T.Decoder(model, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    out: list[int] = []
    logp: list[float] = []
    for _ in range(n_new):
        p = target_dist_at_here(dec.last_nodes)
        x = _sample(p, g)
        out.append(x)
        logp.append(_emit_logp(p, x))
        if len(out) < n_new:
            dec.step(x)
    return out, {"logp": logp, "accepts": [], "n_forward": n_new, "n_rounds": n_new,
                 "carry_rounds": 0, "tokens_per_forward": 1.0, "tau": 0.0,
                 "cache_len": dec.cache_len()}


# ------------------------------------------------------------------ 统计量（率类 / 分布类）

def tv_distance(hist_a: list[int], hist_b: list[int], vocab: int) -> float:
    """两个 token 计数向量的**全变差距离**（V 维频率向量的 ½·L1）。"""
    na, nb = sum(hist_a), sum(hist_b)
    if na == 0 or nb == 0:
        return float("nan")
    s = 0.0
    for i in range(vocab):
        s += abs(hist_a[i] / na - hist_b[i] / nb)
    return 0.5 * s


def wilson_ci(p: float, n: int, z: float = 1.96) -> list:
    """二项比例 Wilson 95% CI（率类指标；与 2640/2650 同口径）。"""
    if not n:
        return [None, None]
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(centre - half, 6), round(centre + half, 6)]
