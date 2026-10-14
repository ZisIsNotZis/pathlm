"""单测（CPU，不取 GPU 锁）。契约 §7 的必要自检，全部须能失败（见 mutate.py）。

    python3 -m pytest test_freerun.py -q
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import freerun as FR  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import TransformHead, PathLM  # noqa: E402


def tiny_model(seed: int = 0) -> PathLM:
    torch.manual_seed(seed)
    m = PathLM(ModelConfig(d_model=16, n_layers=2, n_heads=4, seq_len=64),
               PathConfig(n_mtp=2, chain_mode="token"), vocab_size=FR.VOCAB_SIZE)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def tiny_heads(d: int, seed: int = 0) -> nn.ModuleList:
    torch.manual_seed(seed)
    heads = nn.ModuleList(TransformHead(d) for _ in range(FR.K))
    return heads.eval()


# 1 --------------------------------------------------------------------------
def test_readout_is_shared_embed():
    """两臂的 logits 必须与主干 `embed.weight` 机器可核地同源（共享 tied 读出）。"""
    m = tiny_model(2)
    torch.manual_seed(0)
    z = torch.randn(3, m.mcfg.d_model)
    E = m.embed.weight
    assert torch.equal(FR.logits_of(z, E), z @ E.T), "读出不是同一份 embed"
    assert E.requires_grad is False


# 2 --------------------------------------------------------------------------
def test_head_order_and_no_node1():
    """路径 (i) 发出 K 个 token = [argmax(head0(h_i)@E^T), argmax(head1(h_i)@E^T)]，
    顺序钉死（head0 → 第 1 个、head1 → 第 2 个），且不经过主干 node-1。

    用受控头（mlp 归零 + 大偏置）把两个头的 argmax 钉到**不同** token，使顺序可判。"""
    m = tiny_model(1)
    d = m.mcfg.d_model
    with torch.no_grad():                       # E = 前 d 行为单位阵（读出 = 取前 d 维）
        m.embed.weight.zero_()
        m.embed.weight[:d, :d] = torch.eye(d)
    heads = tiny_heads(d, seed=0)
    for hi, coord in enumerate((1, 2)):         # head0 → token 1，head1 → token 2
        with torch.no_grad():
            nn.init.zeros_(heads[hi].mlp[0].weight)
            nn.init.zeros_(heads[hi].mlp[0].bias)
            nn.init.zeros_(heads[hi].mlp[2].weight)
            b = torch.zeros(d)
            b[coord] = 100.0
            heads[hi].mlp[2].bias.copy_(b)
    torch.manual_seed(3)
    prompts = torch.randint(0, FR.VOCAB_SIZE, (2, 5))
    _, stream = FR.freerun_path_i(m, heads, prompts, n_new=FR.K, k=FR.K)
    E = m.embed.weight
    with torch.no_grad():
        h_i = FR.L.trunk_h(m, prompts)[:, -1, :]
        e0 = int((heads[0](h_i) @ E.T).argmax(1)[0])
        e1 = int((heads[1](h_i) @ E.T).argmax(1)[0])
    assert e0 != e1, f"受控头未区分：e0={e0} e1={e1}"
    assert stream[0].tolist() == [e0, e1], f"头顺序/位置错：{stream[0].tolist()} != {[e0, e1]}"


# 3 --------------------------------------------------------------------------
def test_ar_uses_node1_greedy():
    """参照臂逐 token 必须 = 主干 node-1（`transforms[0]`）的贪心下一 token。"""
    m = tiny_model(4)
    torch.manual_seed(5)
    prompts = torch.randint(0, FR.VOCAB_SIZE, (1, 4))
    _, stream = FR.freerun_ar(m, prompts, n_new=3)
    E = m.embed.weight
    toks = prompts.clone()
    with torch.no_grad():
        for t in range(3):
            h = FR.L.trunk_h(m, toks)
            z1 = m.transforms[0](h[:, -1:, :])
            nxt = (z1 @ E.T)[:, 0, :].argmax(1)
            assert int(stream[0, t]) == int(nxt[0]), f"step {t}: ar 不是 node-1 贪心"
            toks = torch.cat([toks, nxt[:, None]], 1)


# 4 --------------------------------------------------------------------------
def test_bpc_formula():
    """bpc = Σ−ln p /(n·ln2)：每 token 的 −ln p = ln2 时 bpc 恰为 1.0。"""
    nll = np.array([FR.LOG2 * 10.0])
    assert abs(FR.bpc_from_nll(nll, 10) - 1.0) < 1e-12
    nll2 = np.array([FR.LOG2 * 10.0, FR.LOG2 * 30.0])
    assert abs(FR.bpc_from_nll(nll2, 40) - 1.0) < 1e-12


# 4b -------------------------------------------------------------------------
def test_results_bpc_matches_nll_aggregation():
    """落盘一致性：results.json 里每个臂的 bpc 必须等于用**总计分 token 数**（n_prompts·n_new）
    从 per-prompt NLL 重算的值（分母写成 n_new 而不乘 prompt 数时必须抓红）。"""
    p = os.path.join(HERE, "results.json")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as f:
        res = json.load(f)
    n_tok = res["config"]["n_prompts"] * res["config"]["n_new"]
    for b, d in res["arms"]["ar"].items():
        assert abs(FR.bpc_from_nll(d["nll_nats_per_prompt"], n_tok) - d["bpc"]) < 1e-6, \
            f"ar/{b} bpc 与 NLL 聚合不一致"
    for arm in res["config"]["head_arms"]:
        for b, bys in res["arms"][arm].items():
            for s, d in bys.items():
                assert abs(FR.bpc_from_nll(d["nll_nats_per_prompt"], n_tok) - d["bpc"]) < 1e-6, \
                    f"{arm}/{b}/{s} bpc 与 NLL 聚合不一致"


# 5 --------------------------------------------------------------------------
def test_paired_ci_sign():
    """配对差方向钉死：cand 的 NLL 更大 ⇒ Δbpc > 0，且 bootstrap CI 与 Δ 同号。"""
    ar = np.array([10.0, 12.0, 11.0, 13.0])
    cand = ar + 2.0
    n = np.full(4, 20.0)
    d = FR.paired_bpc_ci(cand, ar, n, B=500, seed=1)
    assert d["delta_bpc"] > 0, "Δ 方向错（应为候选 − 参照）"
    assert d["ci95"][0] > 0 and d["ci95"][1] > 0, f"CI 符号与 Δ 不一致：{d['ci95']}"
    assert d["excludes_zero"] is True


# 6 --------------------------------------------------------------------------
def test_paired_ci_percentiles_ordered():
    """percentile 顺序钉死：ci95[0] < ci95[1]（2.5 < 97.5）。"""
    rng = np.random.default_rng(0)
    ar = rng.normal(18.0, 1.0, 12)
    cand = rng.normal(20.0, 1.0, 12)
    d = FR.paired_bpc_ci(cand, ar, np.full(12, 200.0), B=800, seed=2)
    lo, hi = d["ci95"]
    assert lo < hi, f"percentile 顺序反了：{d['ci95']}"
    assert lo <= d["delta_bpc"] <= hi, "点值不在 CI 内"


# 7 --------------------------------------------------------------------------
def test_prompt_protocol_matches_2640():
    """prompt 取法与 2640/2650 同口径：`val[i*stride : i*stride+prompt_len]`，offset 逐条列出。"""
    val = np.arange(12 * 5000 + 64, dtype=np.int64)
    P, offs = FR.make_prompts(val, 3, 64, 5000)
    assert offs == [0, 5000, 10000]
    assert P.shape == (3, 64) and P.dtype == torch.int64
    assert int(P[1, 0]) == 5000 and int(P[2, 63]) == 10000 + 63


# 8 --------------------------------------------------------------------------
def test_frozen_check_passes():
    """冻结断言：主干 `requires_grad=False` 且 backward 后 `.grad is None`。"""
    m = tiny_model(3)
    r = FR.frozen_check(m)
    assert r["requires_grad_all_false"] is True
    assert r["grad_is_none_after_backward"] is True
    assert r["pass"] is True


# 9 --------------------------------------------------------------------------
def test_wall_clock_detection_and_results_clean():
    """禁墙钟自检：检测器能抓红；落盘的 results.json 不得有墙钟字段。"""
    assert FR.has_wall_clock({"hit": 1, "ce": 2.0}) == []
    assert FR.has_wall_clock({"wall_seconds": 1}) != []
    assert FR.has_wall_clock({"nested": {"elapsed_ms": 1}}) != []
    assert FR.has_wall_clock({"arms": [{"duration": 1}]}) != []
    p = os.path.join(HERE, "results.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        bad = [b for b in FR.has_wall_clock(res) if not b.endswith("_wall_clock_fields")]
        assert not bad, f"results.json 含墙钟字段：{bad}"


# 10 -------------------------------------------------------------------------
def test_heads_equal_vs_file_and_load():
    """头加载：`load_heads` 读 2660 落盘文件的 head{j}；逐张量等于源文件。"""
    import tempfile
    d = 8
    src = tiny_heads(d, seed=7)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "h.pt")
        sd = {f"head{j}": src[j].state_dict() for j in range(FR.K)}
        torch.save(sd, p)
        loaded = FR.load_heads(p, d, "cpu")
        assert FR.heads_equal_vs_file(loaded, p) is True
        # 改一个头的最后一个张量 ⇒ 必须报不同
        sd2 = torch.load(p, map_location="cpu", weights_only=True)
        last = sorted(sd2["head1"])[-1]
        sd2["head1"][last] = sd2["head1"][last] + 1.0
        torch.save(sd2, p)
        assert FR.heads_equal_vs_file(loaded, p) is False, "未覆盖全部张量"


# 11 -------------------------------------------------------------------------
def test_teacher_forced_stream_nll_predicts_next_token():
    """真值锚（复核 N1）：teacher-forced 流的 node-1 NLL 必须 = 逐位置「前缀→下一 token」的
    log_softmax gather（位置 pl-1+i 预测 stream[i]；口径同 `pathlm/eval.py::bpc`）。
    位置整体错一格（off-by-one）时必须变红。"""
    m = tiny_model(6)
    pl, n = 5, 4
    torch.manual_seed(11)
    prompt = torch.randint(0, FR.VOCAB_SIZE, (2, pl))
    stream = torch.randint(0, FR.VOCAB_SIZE, (2, n))
    nll = FR.teacher_forced_stream_nll(m, prompt, stream)
    E = m.embed.weight
    ref = torch.zeros(2, dtype=torch.float64)
    with torch.no_grad():
        for i in range(n):                      # 逐位置：[prefix, stream[:i]] 的末位预测 stream[i]
            seq_i = torch.cat([prompt, stream[:, :i]], dim=1)
            logits = (m.transforms[0](FR.L.trunk_h(m, seq_i)) @ E.T)[:, -1, :]
            ref += -torch.log_softmax(logits, -1).gather(
                -1, stream[:, i][:, None])[:, 0].double()
    assert torch.allclose(nll, ref, atol=1e-9), f"teacher-forced NLL 错位：{nll} != {ref}"
    assert abs(FR.bpc_from_nll(nll[0].item(), n) - float(nll[0]) / (n * FR.LOG2)) < 1e-12


# 12 -------------------------------------------------------------------------
def test_degeneracy_stats():
    """客观退化统计：最长连续相同 token、distinct-2、重复 4-gram 率钉死（含最长 run 不在开头）。"""
    s = [5, 5, 5, 5, 7, 8, 7, 8, 7, 8]
    d = FR.degeneracy_stats(s)
    assert d["max_identical_run"] == 4
    assert abs(d["distinct_2"] - 4 / 9) < 2e-6, d["distinct_2"]
    assert abs(d["repeated_4gram_rate"] - (1 - 6 / 7)) < 2e-6, d["repeated_4gram_rate"]
    # 最长 run 不在开头（run 计数器重播时必须正确）
    assert FR.degeneracy_stats([1, 2, 3, 3, 3, 3, 4])["max_identical_run"] == 4
    # 完全重复流
    d3 = FR.degeneracy_stats([9] * 20)
    assert d3["max_identical_run"] == 20
    assert d3["distinct_2"] == round(1 / 19, 6)
