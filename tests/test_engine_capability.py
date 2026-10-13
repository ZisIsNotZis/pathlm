"""F_CLOSURE：引擎能力面事实表（`docs/engine_capability.md`）的机器核对。

复核 B9 指出：收口文称「ENG2 已解锁 `carry × GQA × 非稠密`」，而 `pathlm/depth_ar.py`
的校验仍拒绝该组合。本文件把「哪些组合构造得起来 / 哪些明确拒绝 / 拒绝在哪一行」
钉在代码上：

1. 解析事实表的受控块（`<!-- CAPABILITY:begin/end -->`），逐行核对
   **结果**（构造成功/`ValueError`）、**运行时消息片段**、**拒绝点行号**（该行起 4 行内
   必须同时含 `raise ValueError(` 与源码字面片段）；
2. 表里出现的 id 与测试里的用例必须**一一对应**（防「表里悄悄加一行没人验」）。
"""

import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARDecoder,  # noqa: E402
                             DepthARModel)

DOC = os.path.join(ROOT, "docs", "engine_capability.md")
SRC_REL = "pathlm/depth_ar.py"
SRC = os.path.join(ROOT, SRC_REL)

_BASE = dict(d_model=64, n_layers=2, n_heads=4, seq_len=256, mlp_mult=4)


def _parse_rows() -> dict:
    """解析受控块里的 markdown 表（列序固定：id/组合/结果/拒绝点/源码片段/消息片段）。"""
    text = open(DOC, encoding="utf-8").read()
    block = text.split("<!-- CAPABILITY:begin -->")[1].split("<!-- CAPABILITY:end -->")[0]
    rows = {}
    for raw in block.strip().splitlines():
        if not raw.startswith("|") or raw.startswith("| id ") or set(raw) <= set("|- "):
            continue
        cells = [c.strip() for c in raw.strip().strip("|").split("|")]
        assert len(cells) == 6, f"事实表列数不是 6：{raw}"
        rid, combo, outcome, point, src_frag, msg_frag = cells
        rows[rid] = {"combo": combo, "outcome": outcome, "point": point,
                     "src_frag": src_frag, "msg_frag": msg_frag}
    assert rows, "受控块里没有解析到任何行"
    return rows


#: 每个 id 的**可执行**用例：kind = 构造 DepthARModel 还是再建 DepthARDecoder。
CASES = {
    "C1": {"kind": "model", "cfg": {"carry_rule": "plain"}, "block": "CarryChannelBlock"},
    "C2": {"kind": "model", "cfg": {"carry_rule": "plain", "train_window": 8, "dense": True},
           "block": "CarryChannelBlock"},
    "C3": {"kind": "model", "cfg": {"carry_rule": "delta", "n_kv_heads": 2, "dense": True},
           "block": "ComposableBlock"},
    "C4": {"kind": "model", "cfg": {"n_kv_heads": 2, "train_window": 8, "dense": True},
           "block": "GQABlock"},
    "C5": {"kind": "model", "cfg": {"n_kv_heads": 2, "dense": True}, "block": "GQABlock"},
    "C6": {"kind": "model", "cfg": {"carry_rule": "plain", "n_kv_heads": 2}},
    "C7": {"kind": "model", "cfg": {"carry_rule": "plain", "attn_sparse_k": 4, "dense": True}},
    "C8": {"kind": "model", "cfg": {"carry_rule": "plain", "topology": "parallel", "dense": True}},
    "C9": {"kind": "model", "cfg": {"n_kv_heads": 2, "attn_linear": True, "dense": True}},
    "C10": {"kind": "decoder", "cfg": {"carry_rule": "plain"}},
    "C11": {"kind": "decoder", "cfg": {"n_kv_heads": 2, "dense": True}},
    "C12": {"kind": "model", "cfg": {"attn_sparse_k": 4}},
    # R7（2623）新增开关的能力面：
    "C13": {"kind": "model", "cfg": {"carry_gate": False, "dense": True}},
    "C14": {"kind": "model", "cfg": {"mod_ratio": 0.5}},
    "C15": {"kind": "model", "cfg": {"mod_ratio": 0.5, "dense": True}},
    "C16": {"kind": "model", "cfg": {"aux_mode": "mean"}},
    # E1（ticket 67 §2）新增开关的能力面：逐层状态 / 可选衰减 γ
    "C17": {"kind": "model", "cfg": {"carry_rule": "delta", "carry_per_layer": True},
            "block": "CarryChannelBlock"},
    "C18": {"kind": "model", "cfg": {"carry_rule": "delta", "carry_per_layer": True,
            "n_kv_heads": 2, "dense": True}, "block": "ComposableBlock"},
    "C19": {"kind": "model", "cfg": {"carry_rule": "delta", "carry_decay": 0.5},
            "block": "CarryChannelBlock"},
    "C20": {"kind": "model", "cfg": {"carry_per_layer": True}},
    "C21": {"kind": "model", "cfg": {"carry_decay": 0.5}},
    "C22": {"kind": "model", "cfg": {"carry_rule": "delta", "carry_decay": 0.0}},
}


def _build(case: dict):
    cfg = DepthARConfig(**_BASE, **case["cfg"])
    model = DepthARModel(cfg, vocab_size=32)
    if case["kind"] == "decoder":
        DepthARDecoder(model)
    return model


def test_fact_table_ids_and_cases_are_in_sync():
    assert set(_parse_rows()) == set(CASES), "事实表的 id 与测试用例不一致"


@pytest.mark.parametrize("rid", sorted(CASES))
def test_fact_table_row_matches_engine(rid):
    row, case = _parse_rows()[rid], CASES[rid]
    if row["outcome"] == "接受":
        model = _build(case)
        assert type(model.blocks[0]).__name__ == row["point"], (
            f"{rid} 期望产物 {row['point']}，实际 {type(model.blocks[0]).__name__}")
        assert row["src_frag"] == "—" and row["msg_frag"] == "—"
        return
    assert row["outcome"] == "拒绝", f"{rid}: 结果列只能是 接受/拒绝，收到 {row['outcome']}"
    with pytest.raises(ValueError) as ei:
        _build(case)
    assert row["msg_frag"] in str(ei.value), (
        f"{rid}: 运行时消息不含表里的片段\n  got : {ei.value}\n  want: {row['msg_frag']}")
    rel, lineno = row["point"].split(":")
    assert rel == SRC_REL, f"{rid}: 拒绝点必须写 {SRC_REL}，收到 {rel}"
    src_lines = open(SRC, encoding="utf-8").read().splitlines()
    line = int(lineno)
    assert "raise ValueError(" in src_lines[line - 1], (
        f"{rid}: {row['point']} 不是 raise 行（实际内容：{src_lines[line - 1].strip()!r}）")
    window = "\n".join(src_lines[line - 1:line + 3])
    assert row["src_frag"] in window, (
        f"{rid}: {row['point']} 起 4 行内找不到源码字面片段 {row['src_frag']!r}\n{window}")


def test_fact_table_covers_the_b9_claim():
    """B9 的核心事实：`carry × GQA × 非稠密` 必须仍在拒绝侧（引擎未解锁）。"""
    rows = _parse_rows()
    assert rows["C6"]["outcome"] == "拒绝"
    with pytest.raises(ValueError, match="KV 头共享（GQA/MQA）只在 dense=True"):
        _build(CASES["C6"])
    text = open(DOC, encoding="utf-8").read()
    assert re.search(r"ENG2 解锁的是「`carry_rule` × 非 dense 前向」", text), \
        "事实表必须写明 ENG2 的真实范围（修复 B9 的表述）"
