"""WP 6.2.3 · 2650-verify-gate-13m 的本地可失败单测（全部 CPU，不取 GPU 锁）。

跑法（仓库根）：

    python3 -m pytest \
      axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/test_verify_gate.py -q

覆盖 README §6 的必要自检：

1. **V0 复现门**（离线）：`repro_topk4`/`repro_combo_pw` 的 accepts 与 2640 落盘逐位相同；
2. **对抗候选构造**：`adv_random_k4` 固定 seed 可复现且与 `q` top-4 不同；`adv_bottomk4` ∈ 最低 4 个；
   `aggr_union_heads` = 跨头去重联合；
3. **去重语义**：`dup_k4` 候选 = 同一 token × 4，接受语义与单候选同值；
4. **退化等价 / 无损性门禁**：K 行树机制下 `single_k1_ctl` == `combo_pw`；全部臂发出流 == native；
   每轮恰 1 次顺序前向（`n_forward == n_rounds`）；**接受行缓存回写用对行**；
5. **prod 变异（须抓红且逐字节还原）**见 `mutate.py`（对象 = `verify_gate.py`）；
6. **禁墙钟**：`results.json` 无墙钟字段的机器可核断言。

> **fixture 说明**：随机初始化的微型 `PathLM` 在贪心解码下会塌缩成常量流（任何 context 都预测
> 同一 token），因此对「缓存回写错行」这类只经缓存间接显现的 bug 没有判别力。这里把 embedding
> 行向量归一化到同范数（`tiny_model`）以打破塌缩，得到非退化、可判别的微型模型。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DIR2640 = os.path.abspath(os.path.join(HERE, "..", "2640-tree-conf-13m"))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for p in (HERE, DIR2640, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import spec_tree as T  # noqa: E402
import verify_gate as V  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.decode import decode  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

N_NEW = 24
VOCAB = 64
TREE_ARMS = ("repro_topk4", "single_k1_ctl", "dup_k4", "aggr_topk8",
             "aggr_union_heads", "adv_random_k4", "adv_bottomk4")


def tiny_model(seed: int = 0) -> PathLM:
    """非退化微型模型（embedding 行归一化 ⇒ 贪心流随 context 变化，可判别缓存 bug）。"""
    torch.manual_seed(seed)
    m = PathLM(ModelConfig(d_model=64, n_layers=4, n_heads=4, seq_len=128, mlp_mult=2),
               PathConfig(n_mtp=3, chain_mode="token", transport="none"),
               vocab_size=VOCAB).eval()
    with torch.no_grad():
        w = m.embed.weight.data
        m.embed.weight.data = w / w.norm(dim=1, keepdim=True) * 0.05
    return m


def make_prompt(seed: int = 0, length: int = 24) -> torch.Tensor:
    torch.manual_seed(999 + seed)
    return torch.randint(0, VOCAB, (length,))


def _lg(probs) -> torch.Tensor:
    return torch.tensor(probs, dtype=torch.float32).log().view(1, 1, -1)


def fake_nodes(vocab: int = 206, seed: int = 0) -> dict:
    """构造 `slot_probs` 可吃的 nodes（V=vocab 的确定性随机 logits）。"""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(vocab, generator=g)
    return {
        1: {"logits": logits.view(1, 1, -1), "chain2": {"logits": (logits + 0.1).view(1, 1, -1)}},
        2: {"logits": (logits - 0.2).view(1, 1, -1)},
        3: {"logits": logits.view(1, 1, -1)},
    }


# ---------------------------------------------------------------- V0 复现门（离线）

def _load_2640() -> dict | None:
    path = os.path.join(DIR2640, "results.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_v0_repro_gate_matches_2640_bitwise():
    """本实验 `repro_topk4`/`repro_combo_pw` 的逐轮 accepts 必须与 2640 落盘逐位相同。"""
    mine_path = os.path.join(HERE, "results.json")
    ref = _load_2640()
    if ref is None or not os.path.exists(mine_path):
        pytest.skip("results.json 尚未生成（跑 run.py 之后才检查）")
    with open(mine_path, encoding="utf-8") as f:
        mine = json.load(f)
    checked = 0
    for ckpt, ck in mine.get("checkpoints", {}).items():
        for my_arm, ref_arm in V.V0_MAP.items():
            if my_arm not in ck.get("arms", {}) or ref_arm not in ref["checkpoints"][ckpt]["arms"]:
                continue
            got = [pp["accepts"] for pp in ck["arms"][my_arm]["per_prompt"]]
            want = [pp["accepts"] for pp in ref["checkpoints"][ckpt]["arms"][ref_arm]["per_prompt"]]
            assert got == want, f"{ckpt}/{my_arm} 与 2640 {ref_arm} 的 accepts 不逐位相同"
            checked += 1
    assert checked > 0, "没有任何 (ckpt, 臂) 被 V0 检查到"


# ---------------------------------------------------------------- 对抗 / 联合候选构造

def test_adv_random_candidates_reproducible_and_off_distribution():
    """固定 seed ⇒ 候选可复现；且与 `pw_mix.q` 的 top-4 不同（与任何分布无关）。"""
    nodes = fake_nodes(vocab=206, seed=1)
    a = V.make_cands_adv_random(206, V.ADV_RANDOM_SEED)(nodes)
    b = V.make_cands_adv_random(206, V.ADV_RANDOM_SEED)(nodes)
    assert a == b, "同一 seed 的候选序列必须逐位相同"
    assert len(a) == 4 and all(0 <= t < 206 for t in a)
    assert a != V.cands_q_topk(nodes, 4), "固定 seed 的随机候选不得恰好等于 q 的 top-4"


def test_adv_bottomk4_are_the_lowest_four():
    """`adv_bottomk4` 的候选必须恰是 `pw_mix.q` 概率最低的 4 个（机器可核）。"""
    nodes = fake_nodes(vocab=206, seed=2)
    q = T.pw_mix(nodes)["q"]
    want = [int(i) for i in torch.topk(q, 4, largest=False).indices.tolist()]
    got = V.cands_adv_bottomk4(nodes)
    assert got == want
    assert float(q[got].sum()) < float(q[V.cands_q_topk(nodes, 4)].sum())


def test_aggr_union_heads_dedups_across_heads():
    """跨头联合 = dedup{argmax pc, argmax p2, top-2 pc, top-2 p2}，K ≤ 4。"""
    nodes = fake_nodes(vocab=206, seed=3)
    probs = T.slot_probs(nodes)
    pc, pd = probs["pc"], probs["p2"]
    want = list(dict.fromkeys([int(pc.argmax()), int(pd.argmax())]
                              + V._topk(pc, 2) + V._topk(pd, 2)))
    assert V.cands_aggr_union_heads(nodes) == want
    assert len(V.cands_aggr_union_heads(nodes)) <= 4


# ---------------------------------------------------------------- 去重 / 退化等价语义

def test_dup_k4_is_four_copies_and_same_accept_semantics():
    """`dup_k4` 候选 = 同一 token × 4；接受语义与单候选完全一致。"""
    nodes = fake_nodes(vocab=206, seed=4)
    d = int(T.first_slot_distribution(nodes, "combo_pw").argmax())
    assert V.cands_dup_k4(nodes) == [d, d, d, d]
    assert V.cands_combo_k1(nodes) == [d]
    m = tiny_model(seed=2)
    prompt = make_prompt(2)
    _g1, s1 = V.decode_gate(m, prompt, N_NEW, m.pcap, V.cands_combo_k1)
    _g4, s4 = V.decode_gate(m, prompt, N_NEW, m.pcap, V.cands_dup_k4)
    assert s1["accepts"] == s4["accepts"], "重复候选不得改变接受语义"


def test_single_k1_equals_combo_pw_on_timeline():
    """K=1 树机制（`single_k1_ctl`）必须与生产 `decode_spec` 的 `combo_pw` 逐位同值。"""
    m = tiny_model(seed=5)
    prompt = make_prompt(5)
    gen_s, st_s = T.decode_spec(m, prompt, N_NEW, m.pcap, max_drafts=T.GAMMA,
                                draft_fn=T.make_draft_fn("combo_pw"))
    gen_t, st_t = V.decode_gate(m, prompt, N_NEW, m.pcap, V.cands_combo_k1)
    assert gen_t == gen_s and st_t["accepts"] == st_s["accepts"]


# ---------------------------------------------------------------- 无损性 / 前向次数门禁

def test_gate_streams_are_bit_identical_to_native_greedy():
    """**门禁**：全部臂的发出流必须与原生贪心流逐位相同（含对抗臂）。"""
    m = tiny_model(seed=6)
    prompt = make_prompt(6)
    native, _ = decode(m, prompt, N_NEW, m.pcap)
    assert len(set(native)) > 1, "fixture 必须非退化（否则该门禁无判别力）"
    vocab = int(m.embed.weight.shape[0])
    for arm in TREE_ARMS:
        fn = V.build_cands_fn(arm, vocab, 0)
        gen, st = V.decode_gate(m, prompt, N_NEW, m.pcap, fn)
        assert gen == native, f"{arm} 的发出流与原生贪心不一致"
        assert st["n_forward"] == st["n_rounds"], f"{arm} 每轮前向次数必须为 1"


def _cands_true_last(nodes: dict) -> list[int]:
    """测试专用候选集：把**真值**（= `argmax pc`，即 `T1` 取完 `v1` 后的下一步）
    放到**最后一行**（索引 2）。

    正常臂（top-k）把真值排在第 0 行 ⇒ `acc_i=0`，无法区分「回写接受行」与「回写第 0 行」。
    这里强制 `acc_i=2`，让 `write_back_accepted_row` 的**行索引**成为可判别变量。
    """
    probs = T.slot_probs(nodes)
    t = int(probs["pc"].argmax())
    decoys = [int(i) for i in torch.topk(probs["p2"], 3).indices.tolist() if int(i) != t]
    return decoys[:2] + [t]


def test_gate_writeback_uses_the_accepted_row_not_row_zero(monkeypatch):
    """**接受行缓存回写**：`write_back_accepted_row` 收到的行索引必须 == 该轮接受行索引
    （真值在第 2 行时不得回写第 0 行）。期望索引由与 `decode_gate` 同一接受判据再生。"""
    m = tiny_model(seed=7)
    prompt = make_prompt(7)
    native, _ = decode(m, prompt, N_NEW, m.pcap)
    expected: list[int] = []
    written: list[int] = []
    real_wb = T.write_back_accepted_row
    real_tf = T.tree_forward

    def tf(model, dec, rows):
        out = real_tf(model, dec, rows)
        logs = out[1][1]["logits"]
        cands = [r[1] for r in rows]
        acc_i = next((i for i, c in enumerate(cands) if int(logs[i, 0].argmax()) == c), None)
        expected.append(0 if acc_i is None else acc_i)
        return out

    def spy(dec, new_kv, p0, row, n_pos):
        written.append(row)
        return real_wb(dec, new_kv, p0, row, n_pos)

    monkeypatch.setattr(T, "tree_forward", tf)
    monkeypatch.setattr(T, "write_back_accepted_row", spy)
    gen, st = V.decode_gate(m, prompt, N_NEW, m.pcap, _cands_true_last)
    assert gen == native, "真值在末行时发出流仍须 == native"
    assert st["candidate_sizes"] and all(k == 3 for k in st["candidate_sizes"])
    assert len(expected) == len(written) == st["n_rounds"] and written
    assert any(i != 0 for i in expected), "fixture 未产生非 0 接受行（该测试无判别力）"
    assert written == expected, f"回写行 {written} != 接受行 {expected}"


def test_gate_arms_need_node3():
    """γ=2 需要 node3：`n_mtp<=2` 必须报错而不是静默退化。"""
    torch.manual_seed(0)
    m = PathLM(ModelConfig(d_model=32, n_layers=3, n_heads=2, seq_len=64, mlp_mult=2),
               PathConfig(n_mtp=2, chain_mode="token", transport="none"),
               vocab_size=41).eval()
    with pytest.raises((KeyError, IndexError)):
        V.decode_gate(m, torch.tensor([3, 11, 7, 2, 19, 5, 23, 1]), 8, m.pcap, V.cands_combo_k1)


def test_win64_tree_not_implemented():
    """带驱逐（window=64）的树验证未实装 ⇒ `tree_forward` 必须显式报错（失败模式 V6-g）。"""
    m = tiny_model(seed=0)
    with pytest.raises(NotImplementedError):
        V.decode_gate(m, make_prompt(0), 8, m.pcap, V.cands_combo_k1, window=64)


# ---------------------------------------------------------------- 产出口径

def test_results_json_has_no_wall_clock_fields():
    """禁墙钟判据的机器可核标记：`results.json` 里不得出现墙钟字段。"""
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        pytest.skip("results.json 尚未生成（跑 run.py 之后才检查）")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    bad = []

    def walk(node, prefix=""):
        import re as _re
        if isinstance(node, dict):
            for k, v in node.items():
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
