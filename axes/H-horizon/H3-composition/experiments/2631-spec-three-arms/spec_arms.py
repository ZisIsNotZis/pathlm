"""R6 · Spec 三臂（chain / direct / combined）的臂定义与指标口径。

本模块**只读**模型：不改 `pathlm/**`。三臂复用已入账的生产解码器
`pathlm/decode.py::decode_spec`（Medusa 式批量验证），只把它的 `draft_fn` 钩子
接到本实验的臂定义上 ⇒ 「唯一变量 = 第 N+2 草稿位的分布来源」这一条在**实现层**
成立（同一 forward、同一验证器、同一验证宽度 K=3）。

## 位置与轮次的口径（写死，判读复用）

设已提交前缀到位置 N（含）。一次投机轮：

```
v1 = argmax node1(h_N)                    # 位置 N+1；**就是验证器自己的输出** ⇒ 不设接受测试
d2 = argmax <arm 分布>(nodes@N)           # 位置 N+2；**三臂唯一变量**
d3 = argmax node3(h_N)                    # 位置 N+3；三臂固定（共享背景）
inputs = [v1, d2, d3]   →  一次 width-3 前向（K = 1 + γ，γ = 2）
接受：a = 最长的 i 使  argmax node1(输入位 i-1) == d_{i+1}   （i = 1, 2）
发出：v1 + d2[:a]  （每轮 1 + a 个 token）
```

- `node1` = `T1(h)`、`node2` = `T2(h)`、`node3` = `T3(h)`（`pathlm/model.py::_mtp_nodes`）；
  `chain2` = `T1(chain_cat([h, embed(d1)]))`，`d1 = argmax node1(h)`（推理时自动走
  deploy 分支：`condition=None` ⇒ 条件 = node-1 的 argmax）。
- 指标：`alpha1 = P(a ≥ 1)`、`alpha2_given_alpha1 = P(a ≥ 2 | a ≥ 1)`、
  `tau = mean(a)`、`tokens_per_forward = n_new / n_rounds = 1 + tau`。

## 分布保真

全实验为 `temperature=0` 的 **argmax 路线** ⇒ 明确声明**非分布保持**；本实验给出的
保真保证是**更强的逐位等价**：发出流与同 checkpoint 的原生贪心流逐位相同（门禁）。
采样（温度 > 0）路线需要拒绝采样验证器，**未测**（记入下一步）。
"""

from __future__ import annotations

import statistics
import time

import torch

from pathlm.decode import Decoder, decode, decode_spec

#: 投机草稿位数（N+2, N+3）。第一发出位 N+1 不设接受测试（见模块文档）。
GAMMA = 2

#: 三臂 —— 唯一变量 = 第 N+2 草稿位的分布来源。
ARMS = ("chain", "direct", "combined")

#: `combined` 臂的预登记混合系数（判据线先写死，不许事后调）。0.5 = 等权概率混合。
COMBINED_LAM = 0.5


def second_slot_logits(nodes: dict, arm: str, lam: float = COMBINED_LAM):
    """位置 N+2 的草稿分布（logits 或已归一化的概率）——**三臂的唯一变量**。

    `nodes` = `pathlm/model.py::PathLM._mtp_nodes` 在位置 N 的返回
    （最后位置切片，形状 `[1, 1, V]`）。

    - `chain`：`node1.chain2` —— `T1(chain_cat([h_N, embed(d1)]))`，**以已出的 N+1 为条件**；
    - `direct`：`node2` —— `T2(h_N)`，从同一状态**一次直接**出 N+2；
    - `combined`：`λ·softmax(chain2) + (1−λ)·softmax(node2)`，**两条分布按概率混合**。

    注意 `T∘T`（把 `T1` 再套一次在同一 latent 上）**不是**这三者中的任何一个：
    它既无概率、也不以 token d1 为条件（`docs/REMEDIATION.md` §1.2 的反例警示）。
    本引擎里 `T∘T` 对应 `chain_mode="latent"`，本实验**不训**该模式。
    """
    if arm == "chain":
        return nodes[1]["chain2"]["logits"][0, -1].float()
    if arm == "direct":
        return nodes[2]["logits"][0, -1].float()
    if arm == "combined":
        pc = nodes[1]["chain2"]["logits"][0, -1].float().softmax(-1)
        pd = nodes[2]["logits"][0, -1].float().softmax(-1)
        return lam * pc + (1.0 - lam) * pd
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def make_draft_fn(arm: str, lam: float = COMBINED_LAM):
    """把 `decode_spec` 的 `draft_fn` 钩子接到三臂上。

    `draft_fn(nodes, max_drafts)` 必须返回 `max_drafts` 个 token：第 1 个（= N+2）
    是本臂的变量，其余（N+3..）= 各 direct 头（三臂固定）。
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    def fn(nodes: dict, max_drafts: int) -> list[int]:
        drafts = [int(second_slot_logits(nodes, arm, lam).argmax())]
        for k in range(3, 2 + max_drafts):
            drafts.append(int(nodes[k]["logits"][0, -1].argmax()))
        return drafts

    return fn


def arm_decode(model, prompt, n_new: int, pcap, arm: str,
               lam: float = COMBINED_LAM):
    """跑一个臂；返回 `(gen, stats)`。`native` = 原生逐 token 自回归对照。"""
    if arm == "native":
        gen, st = decode(model, prompt, n_new, pcap)
        return gen, {"n_forward": n_new, "accepts": [], "native": True}
    if pcap.n_mtp <= GAMMA:
        raise ValueError(f"本实验要求 n_mtp >= {GAMMA + 1}（γ={GAMMA} 需要 node3）："
                         f"收到 n_mtp={pcap.n_mtp}")
    gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=GAMMA,
                          draft_fn=make_draft_fn(arm, lam))
    return gen, st


def round_metrics(accepts: list[int], n_new: int) -> dict:
    """由每轮接受数算 α₁ / α₂|α₁ / τ / tok-fwd（口径见模块文档）。"""
    n_rounds = len(accepts)
    if n_rounds == 0:
        return {"n_rounds": 0, "alpha1": None, "alpha2_given_alpha1": None,
                "tau": None, "tokens_per_forward": None, "accepted_total": 0}
    n_ge1 = sum(1 for a in accepts if a >= 1)
    n_ge2 = sum(1 for a in accepts if a >= 2)
    tau = sum(accepts) / n_rounds
    return {
        "n_rounds": n_rounds,
        "accepted_total": int(sum(accepts)),
        "alpha1": n_ge1 / n_rounds,
        "alpha2_given_alpha1": (n_ge2 / n_ge1) if n_ge1 else None,
        "tau": tau,
        "tokens_per_forward": n_new / n_rounds,
    }


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def micro_forward_cost(model, pcap, reps: int = 120,
                       prompt_len: int = 32, device: str = "cuda") -> dict:
    """实测 width-1 / width-3 单次前向成本（本实验的算力账单位，SCHEMA §17.8）。

    **逐次交替**：同一循环里先测一次 width-1 `step()` 再一次 width-3 `step_multi()`，
    两个采样尽量贴在一起以对齐外部占用状况；共 `reps` 对。

    取各自的最小值作为「无干扰成本」的估计。共享卡上有**外部**长驻负载（如
    llama-server，本仓锁管不到），其占用率在毫秒尺度抖动；若按长块测量，min 会取到
    「被外部占用」的那一档 ⇒ w1 反而 > w3（物理不可能，本波实测到过 c3=0.46）。
    改单次前向后每对采样只占 ~10–30 ms，窗内出现干净调度的概率大得多。

    `w1_all_ms` / `w3_all_ms` 保留**全部样本**，供复核判断外部占用是否平稳
    （§20.2/§20.3 的外因记账）；`c3_vs_w1`（min 比）是算力账用的数。
    """
    g = torch.Generator().manual_seed(7)
    prompt = torch.randint(0, model.n_real_tokens, (prompt_len,), generator=g)
    if device == "cuda":
        prompt = prompt.cuda()

    def prefilled():
        d = Decoder(model, window=pcap.window, anchors=pcap.anchors)
        for t in prompt.tolist():
            d.step(t)
        return d

    toks = [int(t) for t in torch.randint(0, model.n_real_tokens, (3,),
                                         generator=torch.Generator().manual_seed(9))]
    # 两个独立解码器各自推进（w1 每对 +1 位、w3 每对 +3 位）；min 通常落在最短
    # 上下文的那次采样上 ⇒ min-vs-min 天然上下文对齐，后期上下文漂移不污染结论。
    d1, d3 = prefilled(), prefilled()
    times: dict[str, list[float]] = {"w1": [], "w3": []}
    with torch.no_grad():
        for _ in range(reps):
            if d3.n + 3 > model.mcfg.seq_len:          # 位置预算耗尽 ⇒ 重建短上下文
                d1, d3 = prefilled(), prefilled()
            sync()
            t0 = time.time()
            d1.step(toks[0])
            sync()
            times["w1"].append(time.time() - t0)
            sync()
            t0 = time.time()
            d3.step_multi(toks)
            sync()
            times["w3"].append(time.time() - t0)
    ms = lambda xs: [round(1000 * t, 4) for t in xs]
    w1_ms, w3_ms = min(times["w1"]), min(times["w3"])
    return {"w1_ms": round(1000 * w1_ms, 4), "w3_ms": round(1000 * w3_ms, 4),
            "c3_vs_w1": round(w3_ms / w1_ms, 4),
            "reps": reps,
            "w1_median_ms": round(1000 * statistics.median(times["w1"]), 4),
            "w3_median_ms": round(1000 * statistics.median(times["w3"]), 4),
            "w1_range_ms": [round(1000 * min(times["w1"]), 4),
                            round(1000 * max(times["w1"]), 4)],
            "w3_range_ms": [round(1000 * min(times["w3"]), 4),
                            round(1000 * max(times["w3"]), 4)],
            "w1_all_ms": ms(times["w1"]), "w3_all_ms": ms(times["w3"])}


def mtp_head_macs(cfg, vocab_size: int, n_conditioning_heads: int = 1) -> dict:
    """解析的 MTP 头成本（MAC / 位置），与主干逐层成本并报（SCHEMA §17.9）。

    每个前向（无论哪个臂）都会算：`T1..T3` 三个 d×d、一个 `chain_cat` 2d×d、
    4 个 node 读出 d×V + `chain2` 读出 d×V、5 个 conf d→1。
    ⇒ **头成本与臂无关**（四臂完全相同），只用来回答「草稿自身成本有多大」。
    """
    d, m = cfg.d_model, cfg.mlp_mult
    heads = 3 * d * d                       # T1 T2 T3
    chain = 2 * d * d * n_conditioning_heads  # chain_cat
    readouts = 5 * d * vocab_size           # node0..node3 + chain2
    confs = 5 * d
    head_total = heads + chain + readouts + confs
    per_layer = 3 * d * d + d * d + 2 * d * (m * d)   # qkv + proj + mlp
    trunk = cfg.n_layers * per_layer
    return {"head_macs_per_position": int(head_total),
            "trunk_linear_macs_per_position": int(trunk),
            "head_over_trunk_frac": round(head_total / trunk, 5),
            "note": "attention 项（∝ 上下文长度）未计入两侧；头成本与臂无关，臂间相消"}
