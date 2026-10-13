"""WP 6.2.2 · 概率多候选 / 树 + 路径概率加权组合：臂定义与**树验证**实现。

本模块**只读**模型与生产解码器：不改 `pathlm/**`。

## 位置与轮次口径（与 2631 一致）

设已提交前缀到位置 N（含）。一次投机轮：

```
v1 = argmax node1(h_N)              位置 N+1；**验证器自己的输出** ⇒ 不设接受测试
d2 = 草稿                           位置 N+2；**本实验唯一变量 = 它的来源（分布 / 候选集）**
d3 = argmax node3(h_N)              位置 N+3；**全部臂固定**（与 2631 共享背景）
inputs = [v1, d2, d3]  →  一次前向；接受最长前缀；发出 1 + a 个 token
```

`node1/2/3 = T1/T2/T3(h)`、`chain2 = T1(chain_cat([h, embed(v1)]))`
（`pathlm/model.py::_mtp_nodes`，`chain_mode="token"`，推理走 deploy 分支）。

## 单候选臂 vs 树臂

- 单候选臂（`direct1` / `chain1` / `combined_fixed` / `combo_pw`）走**生产解码器**
  `pathlm.decode.decode_spec`（B=1、K=3 的批量验证），只经它的 `draft_fn` 钩子注入分布 ⇒
  与 2631 是**同一条代码路径**。
- 树臂（`tree_*` / `oracle_accept`）一次前向验证 **K 个候选**：`tree_forward` 用 **B=K 行**
  的批量前向（共享前缀 KV 广播到 K 行，每行 = `[v1, c_i, d3]`，行内因果掩码），
  **顺序前向次数恒为 1/轮**；接受判据 = 存在候选 == 目标模型在该位置（同一真实前缀下）的
  argmax；随后**只把接受行的 K/V 与位置写回缓存**（`write_back_accepted_row`）。

## 分布保真

全实验 `temperature=0` 的 argmax 路线 ⇒ 明确声明**非分布保持**；给出更强的**逐位等价**门禁：
发出流与同 checkpoint 的原生贪心流逐位相同。采样路线需要拒绝采样验证器，**未测**。
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根（含 pathlm/ 的祖先目录）: {start}")


ROOT = _repo_root(HERE)

#: 2631（旧口径，本实验复用它而不是重造：checkpoint / prompt / 臂定义口径的一致性锚）
LEGACY_DIR = os.path.join(ROOT, "axes", "6-predict", "6.2-speculative-decoding",
                          "experiments", "2631-spec-three-arms")
for _p in (ROOT, HERE, LEGACY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.decode import Decoder, _slice_nodes, decode, decode_spec  # noqa: E402
from pathlm.model import Block, cap_norm  # noqa: E402

#: 投机草稿位数（N+2, N+3）；K = 1 + GAMMA = 3 是每次前向的位置数
GAMMA = 2

#: `combined_fixed` 的预登记混合系数（= 2631 的 λ，写死不许事后调）
COMBINED_LAM = 0.5

NATIVE = "native"
SINGLE_ARMS = ("direct1", "chain1", "combined_fixed", "combo_pw")
TREE_ARMS = ("tree_union2", "tree_direct_k2", "tree_qpw_k2", "tree_topk4")
ORACLE_ARM = "oracle_accept"
#: 优先级顺序：native → 4 个单候选臂 → 3 个 K=2 树臂 → oracle 上界 → K=4 树臂（最贵，最后）
ALL_ARMS = ((NATIVE,) + SINGLE_ARMS + ("tree_union2", "tree_direct_k2", "tree_qpw_k2")
            + (ORACLE_ARM, "tree_topk4"))

#: top-k 树臂：(候选排序分布, k)。`pw` = `combo_pw` 的路径概率加权分布。
TOPK_ARMS = {"tree_direct_k2": ("direct", 2), "tree_qpw_k2": ("pw", 2),
             "tree_topk4": ("pw", 4)}

#: 接受率的可排序带（SCHEMA 方差界）
DELTA_SORTABLE = 0.014


def _last_row(t: torch.Tensor) -> torch.Tensor:
    """取单位置节点分布（`[1, 1, V]` → `[V]`，float32）。"""
    if t.dim() != 3 or t.shape[0] != 1:
        raise ValueError(f"需要一个单位置批次 [1, 1, V]，收到 {tuple(t.shape)}")
    return t[0, -1].float()


def slot_probs(nodes: dict) -> dict:
    """位置 N 的四个候选分布（softmax 概率）：`p1`(node1)/`pc`(chain2)/`p2`(node2)/`p3`(node3)。

    `nodes` = `PathLM._mtp_nodes` 的单位置返回（`_slice_nodes` 之后的 `dec.last_nodes`）。
    """
    return {
        "p1": _last_row(nodes[1]["logits"]).softmax(-1),
        "pc": _last_row(nodes[1]["chain2"]["logits"]).softmax(-1),
        "p2": _last_row(nodes[2]["logits"]).softmax(-1),
        "p3": _last_row(nodes[3]["logits"]).softmax(-1),
    }


def pw_mix(nodes: dict) -> dict:
    """出资人的**路径概率加权**（原式，含 `p1[v1]` 因子）。

    ```
    v1 = argmax p1;  a = argmax pc;  b = argmax p2
    w_A = p1[v1] · pc[a]        # 路径 (v1, a)：先 N+1 再条件链出 N+2
    w_B = p2[b]                 # 路径 (v1, b)：直连头一次出 N+2
    q   = (w_A·pc + w_B·p2) / (w_A + w_B)
    ```

    **`p1[v1]` 不是公共因子**：它只乘 chain 分支（`w_B` 里没有对应的 `p1` 因子）——
    所以「约掉它」会改变两条分支的相对权重。本函数同时给出**原式** `q` 与
    **去掉该因子的对照口径** `q_nocancel`，两套权重逐轮落盘，供复核判断该因子是否真起作用。
    softmax 下溢到 `w_A+w_B == 0` 时按 `combined_fixed` 兜底并置 `fallback=True`（实测应为 0 次）。
    """
    probs = slot_probs(nodes)
    p1, pc, pd = probs["p1"], probs["pc"], probs["p2"]
    v1, a, b = int(p1.argmax()), int(pc.argmax()), int(pd.argmax())
    w_a = float(p1[v1]) * float(pc[a])
    w_b = float(pd[b])
    denom = w_a + w_b
    fallback = not (denom > 0.0)
    if fallback:
        q = COMBINED_LAM * pc + (1.0 - COMBINED_LAM) * pd
    else:
        q = (w_a * pc + w_b * pd) / denom
    w_a_nc = float(pc[a])
    denom_nc = w_a_nc + w_b
    q_nc = ((w_a_nc * pc + w_b * pd) / denom_nc) if denom_nc > 0.0 else q
    return {"q": q, "q_nocancel": q_nc, "w_A": w_a, "w_B": w_b,
            "w_A_nocancel": w_a_nc, "p1_v1": float(p1[v1]), "pc_a": float(pc[a]),
            "pd_b": float(pd[b]), "v1": v1, "a": a, "b": b, "fallback": fallback}


def first_slot_distribution(nodes: dict, arm: str, lam: float = COMBINED_LAM):
    """位置 N+2 的**草稿分布**（概率向量）——单候选臂的定义。"""
    probs = slot_probs(nodes)
    if arm == "direct1":
        return probs["p2"]
    if arm == "chain1":
        return probs["pc"]
    if arm == "combined_fixed":
        return lam * probs["pc"] + (1.0 - lam) * probs["p2"]
    if arm == "combo_pw":
        return pw_mix(nodes)["q"]
    raise ValueError(f"unknown single arm {arm!r}; expected one of {SINGLE_ARMS}")


def candidate_tokens(nodes: dict, arm: str, oracle_token: int | None = None) -> list[int]:
    """位置 N+2 的**候选集**（有序、去重）。

    单候选臂返回 1 个 token；树臂返回 1–K 个（同一次前向全部验证）。
    `oracle_accept` 需要外部给入的真值 token（**只作 τ 上界，不作判据**）。
    """
    if arm == ORACLE_ARM:
        if oracle_token is None:
            raise ValueError("oracle_accept 需要 oracle_token（真值 argmax）")
        return [int(oracle_token)]
    if arm in SINGLE_ARMS:
        return [int(first_slot_distribution(nodes, arm).argmax())]
    if arm == "tree_union2":
        probs = slot_probs(nodes)
        cands = [int(probs["pc"].argmax()), int(probs["p2"].argmax())]
        return list(dict.fromkeys(cands))          # 去重：两臂一致时 K=1
    if arm in TOPK_ARMS:
        which, k = TOPK_ARMS[arm]
        dist = slot_probs(nodes)["p2"] if which == "direct" else pw_mix(nodes)["q"]
        k = min(k, int(dist.numel()))
        return [int(i) for i in torch.topk(dist, k).indices.tolist()]
    raise ValueError(f"unknown draft arm {arm!r}; expected single/tree/oracle arms")


def first_draft(nodes: dict, arm: str, oracle_token: int | None = None) -> int:
    """该臂在位置 N+2 的**第一个**候选（树臂 = 排序后的首项）。"""
    return candidate_tokens(nodes, arm, oracle_token)[0]


def activity_record(nodes: dict, drafts: list[int] | None = None) -> dict:
    """一个位置上的逐轮活性量（SCHEMA §10）：权重、草稿、分歧标记。

    `div_pw_vs_fixed` 是 `combo_pw` 与 `combined_fixed` 的**逐轮草稿分歧**
    （≈0 表示「路径概率加权 vs 固定等权」在该尺度不可区分）。
    """
    pw = pw_mix(nodes)
    probs = slot_probs(nodes)
    d_fixed = int((COMBINED_LAM * probs["pc"] + (1.0 - COMBINED_LAM) * probs["p2"]).argmax())
    d_pw = int(pw["q"].argmax())
    rec = {"v1": pw["v1"], "draft_chain": pw["a"], "draft_direct": pw["b"],
           "draft_fixed": d_fixed, "draft_pw": d_pw,
           "draft_pw_nocancel": int(pw["q_nocancel"].argmax()),
           "w_A": pw["w_A"], "w_B": pw["w_B"], "w_A_nocancel": pw["w_A_nocancel"],
           "p1_v1": pw["p1_v1"], "pc_a": pw["pc_a"], "pd_b": pw["pd_b"],
           "pw_fallback": bool(pw["fallback"]),
           "div_pw_vs_fixed": bool(d_pw != d_fixed),
           "div_pw_vs_chain": bool(d_pw != pw["a"]),
           "div_pw_vs_direct": bool(d_pw != pw["b"])}
    if drafts is not None:
        rec["drafts"] = [int(d) for d in drafts]
    return rec


def make_draft_fn(arm: str, log: list | None = None, lam: float = COMBINED_LAM):
    """把生产解码器 `decode_spec` 的 `draft_fn` 钩子接到单候选臂上。

    `draft_fn(nodes, max_drafts)` 必须返回 `max_drafts` 个 token：第 1 个（= N+2）
    是本臂的变量，其余（N+3..）= 各 direct 头（**全部臂固定**，共享背景）。
    """
    if arm not in SINGLE_ARMS:
        raise ValueError(f"unknown single arm {arm!r}; expected one of {SINGLE_ARMS}")

    def fn(nodes: dict, max_drafts: int) -> list[int]:
        drafts = [int(first_slot_distribution(nodes, arm, lam).argmax())]
        for k in range(3, 2 + max_drafts):
            drafts.append(int(nodes[k]["logits"][0, -1].argmax()))
        if log is not None:
            log.append(activity_record(nodes, drafts))
        return drafts

    return fn


# ------------------------------------------------------------------ 树验证前向

def _take_row(node, i: int):
    """从 `_slice_nodes` 的结果里取第 i 行（`[B, 1, ...]` → `[1, 1, ...]`）。"""
    if isinstance(node, dict):
        return {k: _take_row(v, i) for k, v in node.items()}
    if torch.is_tensor(node) and node.dim() >= 2:
        return node[i:i + 1].contiguous()
    return node


@torch.no_grad()
def tree_forward(model, dec: Decoder, rows: list[list[int]]):
    """一次 **B=K 行** 的批量验证前向（共享前缀 KV 广播到 K 行）。

    `rows[i] = [v1, c_i, d3]`：第 i 行验证第 i 个候选（`c_i` 落在块内索引 1）。
    返回 `(h_out, nodes, new_kv, p0)`：

    - `nodes` = `model._mtp_nodes(h_out)`，形状 `[B, K, ...]`（每行每个位置的 node-1 logits）；
    - `new_kv[i] = (k_new, v_new)`，形状 `[B, n_heads, K, d_head]`（第 i 层新位置的 K/V，
      **尚未写回** `dec`）；前缀 KV 来自 `dec.kv`（只读）。
    - 接受判据由调用方做：候选 `c_i` 落在块内索引 **1** ⇒ 用第 i 行索引 0 的
      `nodes[1]["logits"]` 验它；`d3` 用**接受行**索引 1 的 logits 验。

    仅支持 `window<=0 且 anchors==0`（本实验三个 checkpoint 的配置）：此时「保留哪些前缀
    key」与 decode.py 的语义相同（全部保留），行内因果由 `tril` 掩码给出。
    """
    if not (dec.window <= 0 and dec.anchors == 0):
        raise NotImplementedError(
            f"tree_forward 仅实现在 window<=0/anchors==0；收到 window={dec.window}, "
            f"anchors={dec.anchors}（带驱逐的树验证需另行实现，本实验未测）")
    B = len(rows)
    if B < 1:
        raise ValueError("tree_forward 需要至少 1 行")
    K = len(rows[0])
    if any(len(r) != K for r in rows):
        raise ValueError("tree_forward 的每行必须等长")
    p0 = dec.n
    if p0 + K > model.mcfg.seq_len:
        raise ValueError(f"decode 长度 {p0 + K} 超过 seq_len {model.mcfg.seq_len}")
    dev = model.embed.weight.device
    idx = torch.tensor(rows, dtype=torch.long, device=dev)
    h = model.embed(idx) + model.pos_embed.weight[p0:p0 + K]
    h = cap_norm(h, model.mcfg.norm_cap)
    new_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    blocks = [b for b in model.blocks if isinstance(b, Block)]
    for i, block in enumerate(blocks):
        n_heads = block.n_heads
        d_head = h.shape[-1] // n_heads
        q, k, v = block.qkv(block.ln1(h)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, K, n_heads, d_head).transpose(1, 2)  # noqa: E731
        q, k, v = shape(q), shape(k), shape(v)
        if i in dec.kv:
            ck = dec.kv[i][0].expand(B, -1, -1, -1)   # 共享前缀：广播到 K 行（只读）
            cv = dec.kv[i][1].expand(B, -1, -1, -1)
            k_all = torch.cat([ck, k], dim=2)
            v_all = torch.cat([cv, v], dim=2)
        else:
            k_all, v_all = k, v
        L = k_all.shape[2]
        mask = torch.zeros(K, L, dtype=torch.bool, device=dev)
        # new keys are causal (position p0 + j attends new keys 0..j)
        mask[:, L - K:] = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev))
        # 前缀对**每个** query 都可见（window=0/anchors=0 ⇒ 不驱逐）；
        # 漏掉这一行 = query 只能看见新位置、看不到上下文（生产 `_run_stack_multi`
        # 的 `else: mask[:, :L - K] = True` 就是这条语义）
        mask[:, :L - K] = True
        a = F.scaled_dot_product_attention(q, k_all, v_all,
                                           attn_mask=mask.view(1, 1, K, L))
        a = a.transpose(1, 2).reshape(B, K, h.shape[-1])
        h = h + block.proj(a)
        h = h + block.mlp(block.ln2(h))
        new_kv[i] = (k, v)
    nodes = model._mtp_nodes(h)
    return h, nodes, new_kv, p0


def write_back_accepted_row(dec: Decoder, new_kv: dict, p0: int, row: int, n_pos: int) -> None:
    """**只把接受行**的 K/V 与位置写回缓存，并对齐 `dec.n`。

    写回的位置是 `p0 .. p0+n_pos-1`（`n_pos = 1 + 接受草稿数`）；被拒绝的后缀不进缓存，
    与 `decode_spec` 的 `truncate` 语义等价（此处直接不追加，无中间态）。
    """
    if not 1 <= n_pos <= next(iter(new_kv.values()))[0].shape[2]:
        raise ValueError(f"n_pos={n_pos} 越界")
    for i, (k_new, v_new) in new_kv.items():
        kk = k_new[row:row + 1, :, :n_pos, :].contiguous()
        vv = v_new[row:row + 1, :, :n_pos, :].contiguous()
        if i in dec.kv:
            dec.kv[i] = (torch.cat([dec.kv[i][0], kk], dim=2),
                         torch.cat([dec.kv[i][1], vv], dim=2))
        else:
            dec.kv[i] = (kk, vv)
        pos = dec.layer_pos.setdefault(i, [])
        dec.layer_pos[i] = list(pos) + [p0 + j for j in range(n_pos)]
    dec.n = p0 + n_pos


@torch.no_grad()
def decode_tree(model, prompt: torch.Tensor, n_new: int, pcap, arm: str,
                oracle_stream: list[int] | None = None,
                activity: list | None = None,
                force_max_candidates: int | None = None):
    """树臂 / oracle 臂的解码主循环：每轮 **一次** B=K 行验证前向。

    返回 `(gen, stats)`；`stats` 含逐轮接受数 `accepts`、前向次数 `n_forward`
    （必须 == 轮数）、每轮候选数 `candidate_sizes`、缓存长度、oracle 的 v1 失配计数。
    """
    if arm not in TREE_ARMS and arm != ORACLE_ARM:
        raise ValueError(f"decode_tree 只跑树臂/oracle 臂，收到 {arm!r}")
    if model.pcap.n_mtp <= GAMMA:
        raise ValueError(f"γ={GAMMA} 需要 node3：收到 n_mtp={model.pcap.n_mtp}")
    dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    gen: list[int] = []
    accepts: list[int] = []
    cand_sizes: list[int] = []
    n_forward = 0
    n_oracle_mismatch = 0
    n_fallback = 0
    while len(gen) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None, "step must populate last_nodes"
        v1 = int(nodes[1]["logits"][0, 0].argmax())
        d3 = int(nodes[3]["logits"][0, 0].argmax())
        L = len(gen)
        oracle_tok = None
        if arm == ORACLE_ARM:
            if oracle_stream is None:
                raise ValueError("oracle_accept 需要 oracle_stream（真值贪心流）")
            if L + 1 >= len(oracle_stream):
                break
            oracle_tok = int(oracle_stream[L + 1])
            if v1 != int(oracle_stream[L]):
                n_oracle_mismatch += 1
        rec = activity_record(nodes)
        cands = candidate_tokens(nodes, arm, oracle_tok)
        if force_max_candidates is not None:
            cands = cands[:force_max_candidates]
        K = 1 + GAMMA
        if dec.n + K > model.mcfg.seq_len:
            break
        rows = [[v1, c, d3] for c in cands]
        _h, fnodes, new_kv, p0 = tree_forward(model, dec, rows)
        n_forward += 1
        logs = fnodes[1]["logits"]
        # 接受判据：存在候选 == 该位置目标模型的 argmax（贪心 ⇒ 至多 1 行能匹配）
        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[i, 0].argmax()) == c), None)
        n_acc = 0
        if acc_i is not None:
            n_acc = 1
            if int(logs[acc_i, 1].argmax()) == d3:     # 只认**接受行**（真前缀）上的 d3
                n_acc = 2
        row_used = 0 if acc_i is None else acc_i
        n_pos = 1 + n_acc
        write_back_accepted_row(dec, new_kv, p0, row_used, n_pos)
        dec.last_nodes = _take_row(_slice_nodes(fnodes, n_pos - 1), row_used)
        gen.append(v1)
        if n_acc >= 1:
            gen.append(cands[acc_i])
        if n_acc >= 2:
            gen.append(d3)
        accepts.append(n_acc)
        cand_sizes.append(len(cands))
        n_fallback += int(rec["pw_fallback"])
        if activity is not None:
            rec.update({"candidates": [int(c) for c in cands],
                        "n_candidates": len(cands),
                        "n_acc": int(n_acc), "n_pos": int(n_pos),
                        "accepted_candidate_index": acc_i,
                        "v1": v1, "d3": d3, "n_forward_this_round": 1})
            activity.append(rec)
    n_rounds = len(accepts)
    return gen[:n_new], {
        "accepts": accepts,
        "n_forward": n_forward,
        "n_rounds": n_rounds,
        "candidate_sizes": cand_sizes,
        "mean_candidates_per_round": (sum(cand_sizes) / n_rounds) if n_rounds else None,
        "n_oracle_v1_mismatch": n_oracle_mismatch,
        "n_pw_fallback": n_fallback,
        "cache_len": dec.cache_len(),
    }
