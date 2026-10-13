"""WP 6.2.2 · 树/多候选 + 路径概率加权臂的本地可失败单测（全部 CPU，不取 GPU 锁）。

跑法（仓库根）：

    python3 -m pytest \\
      axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/test_spec_tree.py -q

覆盖：

① **臂定义**：`direct1`/`chain1`/`combined_fixed` 与 2631 的 `spec_arms.second_slot_logits`
   逐位一致（复用口径而非重造）；`combo_pw` 是**原式**路径概率加权；
   且 `p1[v1]` **不是**公共因子（去掉它会改变草稿）——这是「两臂是否真不同」的数学钉子；
② **候选集**：K 与预登记一致（`tree_direct_k2` K=2 / `tree_qpw_k2` K=2 / `tree_topk4` K=4 /
   `tree_union2` 去重后 ≤2），首候选 = 对应分布的 argmax；
③ **树验证语义**：每轮**一次**前向（`n_forward == n_rounds`）；
   **单候选退化**= 与对应单候选臂逐位同值（gen 与 accepts 全等）；
   **缓存正确性**：接受行写回的缓存的下一步 logits 与「从头重算同一前缀」一致；
④ **无损性门禁**：树臂发出流 == 原生贪心流逐位相同；
⑤ **产出口径**：`results.json` 不含任何墙钟字段（禁墙钟判据的机器可核标记）。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import spec_tree as T  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.decode import Decoder, _slice_nodes, decode, decode_spec  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

# 2631 的臂定义（同源复用：只读导入，用于口径交叉核对）
import spec_arms as LEGACY  # noqa: E402


# ---------------------------------------------------------------- fixtures

def tiny_model(seed: int = 0) -> PathLM:
    torch.manual_seed(seed)
    mcfg = ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2)
    pcap = PathConfig(n_mtp=3, chain_mode="token", transport="none")
    return PathLM(mcfg, pcap, vocab_size=41).eval()


PROMPT = [3, 11, 7, 2, 19, 5, 23, 1]

#: 手工构造的可分辨 fixture（V=8）：softmax(log p) = p。
#: p1[0]=0.30（v1=0）、pc=[0.36,0.28,...]（chain 选 0）、pd=[0.30,0.34,...]（direct 选 1）。
#: ⇒ `combo_pw`（原式，c=0.30）选 **1**；`combined_fixed`（等权）选 **0**；
#:    去掉 `p1[v1]` 因子的对照口径选 **0**。三个判定因此可分辨。
FIX_P1 = [0.30] + [0.10] * 7
FIX_PC = [0.36, 0.28] + [0.06] * 6
FIX_PD = [0.30, 0.34] + [0.06] * 6
FIX_P3 = [0.125] * 8


def _lg(probs) -> torch.Tensor:
    return torch.tensor(probs, dtype=torch.float32).log().view(1, 1, -1)


def fake_nodes() -> dict:
    return {
        0: {"logits": _lg(FIX_P3)},
        1: {"logits": _lg(FIX_P1), "chain2": {"logits": _lg(FIX_PC)}},
        2: {"logits": _lg(FIX_PD)},
        3: {"logits": _lg(FIX_P3)},
    }


# ---------------------------------------------------------------- ① 臂定义

def test_single_arm_first_drafts_match_legacy_2631():
    """`direct1`/`chain1`/`combined_fixed` 的首草稿必须与 2631 的臂定义**逐位相同**。"""
    nodes = fake_nodes()
    pairs = {"chain1": "chain", "direct1": "direct", "combined_fixed": "combined"}
    for mine, theirs in pairs.items():
        want = int(LEGACY.second_slot_logits(nodes, theirs).argmax())
        assert T.first_draft(nodes, mine) == want, f"{mine} 与 2631 的 {theirs} 不一致"
    assert LEGACY.COMBINED_LAM == T.COMBINED_LAM == 0.5, "预登记 λ=0.5 不许事后调"
    assert T.GAMMA == LEGACY.GAMMA == 2 and T.DELTA_SORTABLE == 0.014


def test_pw_mix_is_the_pre_registered_formula():
    """`combo_pw` 必须是 `(w_A·pc + w_B·pd)/(w_A+w_B)`，且 `w_A` 含 `p1[v1]` 因子。"""
    nodes = fake_nodes()
    probs = T.slot_probs(nodes)
    pc, pd, p1 = probs["pc"], probs["p2"], probs["p1"]
    v1, a, b = int(p1.argmax()), int(pc.argmax()), int(pd.argmax())
    w_a = float(p1[v1]) * float(pc[a])
    w_b = float(pd[b])
    mix = T.pw_mix(nodes)
    assert mix["v1"] == v1 == 0 and mix["a"] == a == 0 and mix["b"] == b == 1
    assert mix["w_A"] == pytest.approx(w_a) and mix["w_B"] == pytest.approx(w_b)
    assert torch.allclose(mix["q"], (w_a * pc + w_b * pd) / (w_a + w_b), atol=1e-6)
    assert abs(float(mix["q"].sum()) - 1.0) < 1e-5, "混合结果必须仍是概率分布"
    assert mix["fallback"] is False


def test_pw_factor_is_not_a_common_factor():
    """`p1[v1]` 只乘 chain 分支 ⇒ **不能**约掉；去掉它时草稿会变（0 vs 1）。"""
    nodes = fake_nodes()
    mix = T.pw_mix(nodes)
    rec = T.activity_record(nodes)
    assert rec["draft_pw"] == 1, "原式（含 p1[v1]）选直连候选"
    assert rec["draft_pw_nocancel"] == 0, "去掉 p1[v1] 后选条件链候选"
    assert mix["w_A_nocancel"] == pytest.approx(mix["pc_a"])
    ratio_with = mix["w_A"] / (mix["w_A"] + mix["w_B"])
    ratio_without = mix["w_A_nocancel"] / (mix["w_A_nocancel"] + mix["w_B"])
    assert abs(ratio_with - ratio_without) > 1e-3, "权重比必须随该因子改变"


def test_combo_pw_differs_from_combined_fixed_on_the_fixture():
    """`combo_pw`（路径概率加权）与 `combined_fixed`（等权）在该 fixture 上**不同**。"""
    nodes = fake_nodes()
    rec = T.activity_record(nodes)
    assert rec["draft_pw"] == 1 and rec["draft_fixed"] == 0
    assert rec["div_pw_vs_fixed"] is True
    assert rec["div_pw_vs_chain"] is True and rec["div_pw_vs_direct"] is False
    assert T.first_draft(nodes, "combo_pw") != T.first_draft(nodes, "combined_fixed")


def test_unknown_arm_raises():
    with pytest.raises(ValueError):
        T.first_slot_distribution(fake_nodes(), "latent_chain")
    with pytest.raises(ValueError):
        T.candidate_tokens(fake_nodes(), "beam3")
    with pytest.raises(ValueError):
        T.candidate_tokens(fake_nodes(), T.ORACLE_ARM)      # 缺真值 token


# ---------------------------------------------------------------- ② 候选集

def test_candidate_counts_match_preregistration():
    """K 必须与预登记一致：direct_k2 → 2、qpw_k2 → 2、topk4 → 4、union2 → ≤2（去重）。"""
    nodes = fake_nodes()
    assert T.TOPK_ARMS == {"tree_direct_k2": ("direct", 2), "tree_qpw_k2": ("pw", 2),
                           "tree_topk4": ("pw", 4)}
    assert len(T.candidate_tokens(nodes, "tree_direct_k2")) == 2
    assert len(T.candidate_tokens(nodes, "tree_qpw_k2")) == 2
    assert len(T.candidate_tokens(nodes, "tree_topk4")) == 4
    assert len(T.candidate_tokens(nodes, "tree_union2")) == 2
    assert T.candidate_tokens(nodes, "tree_union2") == [0, 1], "union2 = [argmax chain2, argmax node2]"


def test_topk_order_is_by_probability_and_first_equals_argmax():
    """top-k 按概率降序；首候选 = 该分布的 argmax（与单候选臂一致）。"""
    nodes = fake_nodes()
    assert T.candidate_tokens(nodes, "tree_direct_k2")[0] == T.first_draft(nodes, "direct1")
    assert T.candidate_tokens(nodes, "tree_qpw_k2")[0] == T.first_draft(nodes, "combo_pw")
    assert T.candidate_tokens(nodes, "tree_topk4")[:2] == T.candidate_tokens(nodes, "tree_qpw_k2")
    q = T.pw_mix(nodes)["q"]
    assert T.candidate_tokens(nodes, "tree_topk4") == [int(i) for i in q.topk(4).indices]


def test_tree_union2_dedups_when_heads_agree():
    """两臂一致时 union2 必须退化成 K=1（不重复送同一个候选）。"""
    nodes = fake_nodes()
    nodes[1]["chain2"]["logits"] = _lg(FIX_PD)          # chain2 与 node2 同分布
    assert T.candidate_tokens(nodes, "tree_union2") == [1]
    assert T.candidate_tokens(nodes, "oracle_accept", oracle_token=5) == [5]


def test_oracle_candidate_is_the_supplied_truth():
    assert T.first_draft(fake_nodes(), "oracle_accept", oracle_token=7) == 7


# ---------------------------------------------------------------- ③ 树验证语义

def test_tree_forward_rejects_eviction_configs():
    """带驱逐（window>0）的树验证**未实现** ⇒ 必须显式报错，不许静默近似。"""
    m = tiny_model(seed=0)
    dec = Decoder(m, window=4, anchors=1)
    dec.step(PROMPT[0])
    with pytest.raises(NotImplementedError):
        T.tree_forward(m, dec, [[1, 2, 3]])


def test_tree_arms_use_one_forward_per_round():
    """核心实现要求：每轮顺序前向次数 == 1（`n_forward == n_rounds`）。"""
    m = tiny_model(seed=1)
    prompt = torch.tensor(PROMPT)
    pcap = m.pcap
    for arm in T.TREE_ARMS:
        gen, st = T.decode_tree(m, prompt, 24, pcap, arm)
        assert st["n_rounds"] > 1
        assert st["n_forward"] == st["n_rounds"], f"{arm} 每轮前向次数必须为 1"
        assert all(1 <= k <= 4 for k in st["candidate_sizes"])
        assert len(gen) == 24


@pytest.mark.parametrize("tree_arm,single_arm", [
    ("tree_union2", "chain1"),        # union2 的首候选 = argmax chain2
    ("tree_direct_k2", "direct1"),    # top-2(node2) 的首候选 = argmax node2
    ("tree_qpw_k2", "combo_pw"),      # top-2(q_pw) 的首候选 = argmax q_pw
    ("tree_topk4", "combo_pw"),
])
def test_single_candidate_tree_degenerates_to_the_single_arm(tree_arm, single_arm):
    """候选集只剩 1 个 ⇒ 必须与对应单候选臂**逐位同值**（gen 与 accepts 全等）。

    这条是「树实现没有偷偷改验证语义」的等价性钉子：K=1 时树前向与生产
    `decode_spec`（B=1、K=3）走的是同一数学路径，逐位相同。
    """
    m = tiny_model(seed=2)
    prompt = torch.tensor(PROMPT)
    pcap = m.pcap
    gen_s, st_s = decode_spec(m, prompt, 24, pcap, max_drafts=T.GAMMA,
                              draft_fn=T.make_draft_fn(single_arm))
    gen_t, st_t = T.decode_tree(m, prompt, 24, pcap, tree_arm, force_max_candidates=1)
    assert st_t["candidate_sizes"] == [1] * st_t["n_rounds"]
    assert gen_t == gen_s, f"{tree_arm}（K=1）与 {single_arm} 的 token 流不一致"
    assert st_t["accepts"] == st_s["accepts"]


def test_cache_writeback_matches_from_scratch_recompute():
    """**缓存正确性**：接受行写回的缓存 ⇒ 下一步 logits == 从头重算同一前缀。

    容差按 float32 报实际最大差（本测试在 docstring 里记录实测值）；
    另要求 argmax（= 下一步发出的 token）相同。
    """
    m = tiny_model(seed=3)
    prompt = torch.tensor(PROMPT)
    pcap = m.pcap
    dec = Decoder(m, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    n_prefix = dec.n

    nodes = dec.last_nodes
    v1 = int(nodes[1]["logits"][0, 0].argmax())
    d3 = int(nodes[3]["logits"][0, 0].argmax())
    cands = T.candidate_tokens(nodes, "tree_union2")
    rows = [[v1, c, d3] for c in cands]
    _h, fnodes, new_kv, p0 = T.tree_forward(m, dec, rows)
    logs = fnodes[1]["logits"]
    acc_i = next((i for i, c in enumerate(cands) if int(logs[i, 0].argmax()) == c), None)
    n_acc = 0
    if acc_i is not None:
        n_acc = 1
        if int(logs[acc_i, 1].argmax()) == d3:
            n_acc = 2
    row_used = 0 if acc_i is None else acc_i
    emitted = [v1] + ([cands[acc_i]] if n_acc >= 1 else []) + ([d3] if n_acc == 2 else [])
    T.write_back_accepted_row(dec, new_kv, p0, row_used, 1 + n_acc)
    dec.last_nodes = T._take_row(_slice_nodes(fnodes, n_acc), row_used)

    # 从头重算同一前缀（逐 token，B=1）
    dec2 = Decoder(m, window=pcap.window, anchors=pcap.anchors)
    for tok in list(prompt.tolist()) + emitted:
        dec2.step(int(tok))

    assert dec.n == n_prefix + 1 + n_acc == dec2.n
    assert dec.cache_len() == dec.n, "window=0 时缓存长度必须等于已提交位置数"
    a = dec.last_nodes[1]["logits"][0, 0].detach().float()
    b = dec2.last_nodes[1]["logits"][0, 0].detach().float()
    max_diff = float((a - b).abs().max())
    assert int(a.argmax()) == int(b.argmax()), "写回缓存的下一步 argmax 必须与重算一致"
    assert max_diff < 2e-4, f"写回缓存与重算的 logits 最大差 {max_diff} 过大"
    # 同一断言对每一层缓存的形状有效性做一次弱检查：位置数一致
    for i in dec.kv:
        assert dec.kv[i][0].shape[2] == dec.n == len(dec.layer_pos[i])
    print(f"[cache check] max|Δlogits| = {max_diff:.3e} at n_acc={n_acc}, K={len(cands)}")


def test_tree_arms_need_node3():
    """γ=2 需要 node3；`n_mtp<=2` 必须报错而不是静默退化。"""
    torch.manual_seed(0)
    m = PathLM(ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2),
               PathConfig(n_mtp=2, chain_mode="token", transport="none"),
               vocab_size=41).eval()
    with pytest.raises(ValueError):
        T.decode_tree(m, torch.tensor(PROMPT), 8, m.pcap, "tree_union2")


def test_decode_tree_rejects_single_arms():
    m = tiny_model(seed=0)
    with pytest.raises(ValueError):
        T.decode_tree(m, torch.tensor(PROMPT), 8, m.pcap, "direct1")


def test_oracle_arm_needs_the_truth_stream():
    m = tiny_model(seed=0)
    with pytest.raises(ValueError):
        T.decode_tree(m, torch.tensor(PROMPT), 8, m.pcap, T.ORACLE_ARM)


# ---------------------------------------------------------------- ④ 无损性门禁

def test_tree_streams_are_bit_identical_to_native_greedy():
    """**门禁**：argmax 路线下所有树臂的发出流必须与原生贪心流逐位相同。"""
    m = tiny_model(seed=4)
    prompt = torch.tensor(PROMPT)
    pcap = m.pcap
    native, _ = decode(m, prompt, 24, pcap)
    for arm in T.TREE_ARMS:
        gen, st = T.decode_tree(m, prompt, 24, pcap, arm)
        assert gen == native, f"{arm} 的发出流与原生贪心不一致"
        for a in st["accepts"]:
            assert 0 <= a <= T.GAMMA


def test_oracle_stream_is_bit_identical_and_always_accepts():
    """`oracle_accept` 的候选是真值 ⇒ α₁ ≡ 1（τ 上界），且发出流仍 == native。"""
    m = tiny_model(seed=5)
    prompt = torch.tensor(PROMPT)
    pcap = m.pcap
    native, _ = decode(m, prompt, 26, pcap)
    gen, st = T.decode_tree(m, prompt, 24, pcap, T.ORACLE_ARM, oracle_stream=native)
    assert gen == native[:24]
    assert st["n_oracle_v1_mismatch"] == 0
    assert all(a >= 1 for a in st["accepts"]), "oracle 候选命中 ⇒ 每轮至少接受 1"


# ---------------------------------------------------------------- ⑤ 产出口径

def test_results_json_has_no_wall_clock_fields():
    """禁墙钟判据的机器可核标记：`results.json` 里不得出现墙钟字段。"""
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        pytest.skip("results.json 尚未生成（跑 run.py 之后才检查）")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    bad = []

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                import re as _re
                toks = _re.split(r"[_]+", k.lower())
                if ("wall" in k.lower() or "speedup" in k.lower()
                        or any(t in ("ms", "seconds", "toks", "tok_s", "wallclock") for t in toks)):
                    bad.append(f"{prefix}.{k}")
                walk(v, f"{prefix}.{k}")
        elif isinstance(node, list):
            for v in node:
                walk(v, prefix)

    walk(blob)
    assert not bad, f"results.json 出现墙钟字段: {bad}"
