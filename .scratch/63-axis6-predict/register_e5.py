#!/usr/bin/env python3
"""登记 E5（batch>1 批量验证）到受控引擎通道 `.scratch/67-engine-changes/TICKET.md`。
   scripted + 命中计数断言；只插入一节 + 一条变更记录，不改既有内容。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, ".scratch/67-engine-changes/TICKET.md")
TS = "2026-10-14T23:30:00+08:00"

E5 = """### E5 · 轴 6 · 批量验证（B>1）：`Decoder` / `decode_spec` / 树路径的 batch 维

| 项 | 内容 |
|---|---|
| 来源 | `axes/6-predict/6.2-speculative-decoding/project.yaml` 的 `milestone.output`（点名「**batch>1 的稳健性点**」）与 `gaps`（「温度>0 与 batch>1 的稳健性点未跑」）；WP 6.2.4（`experiments/2673-robustness-13m/README.md` §1）已显式把 batch>1 列为**本 WP 范围之外**。 |
| 需要什么 | 引擎现有的投机解码路径**全部是 B=1**：`Decoder` 的 `h`/KV 缓存是单序列、`decode_spec` 逐 token 写回、E3 新增的 `Decoder.step_tree_multi` 也是「一条序列 + B=K 行」。要回答「batch>1 时 τ / 接受判据是否仍成立」，需要：① `Decoder` 的隐状态与 KV 缓存加 **batch 维**；② `_run_stack_multi` 支持 `B>1` 且每行带 `(seq_idx, row)` 映射；③ `decode_spec` 增「多 prompt 并行」入口（返回 per-sequence 的 gen / accepts / 接受行号）；④ 树路径的 `tree_parents` 按序列独立构造。 |
| 为什么 | `milestone.output` 把 batch>1 写成**必须报的稳健性点**：B=1 的 τ 不能外推到部署批量；而**接受判据与 KV 写回的 bookkeeping 在批内是 per-sequence 的**，若引擎不支持就只能报 B=1，缺口径。不修则无法区分「机制不成立」与「本引擎只支持 B=1」。 |
| 最小 patch 描述 | 只动 `pathlm/decode.py`（`_mtp_nodes` 无需改）：① `Decoder` 的 `h`、每层 KV 由 `[1, ...]` 扩到 `[B, ...]`（缺省 B=1 逐位还原现状）；② `_run_stack_multi(rows, tree_parents, seq_idx)` 支持同一前向内多序列的行，写回按 `(seq_idx, row)`；③ `decode_spec(..., prompts: list[Tensor])` 入口 + per-sequence 返回；④ `step_tree_multi` 的 `tree_parents` 接受「每序列一棵树」。**约 80–150 行**。单测 3 条：**(a) 等价性**——B=2 的两条序列取同一批 prompt 时，逐序列读数与各自 B=1 运行**逐位相同**；**(b) 独立性（gate probe）**——把其中一条序列的草稿改错，另一条的接受行 / 发出流不得变化；**(c) 缺省 B=1 逐位还原**。风险：显存随 B 线性增长（须逐臂报 `peak_gpu_mem_mb`）；KV 写回索引易错 ⇒ 以 (a) 的等价性单测为准。 |
| 授权状态 | **未授权**（C3 已建通道；待 Main 派唯一 engine-writer）。轴 6 本轮**未改** `pathlm/**`：只交付 B=1 的温度>0 无损投机采样与 53.9M 对抗臂，batch>1 作为受阻项登记。 |
"""


def main():
    t = open(P, encoding="utf-8").read()
    anchor = "## 3. 与既有文档的关系"
    assert t.count(anchor) == 1, "anchor"
    t = t.replace(anchor, E5 + "\n" + anchor)
    t = t.rstrip("\n") + (
        '\n- `%s` · axis6-lead（轴 6 预测轴 lead，ticket 63 Phase D）· 登记 **E5**（批量验证 B>1）：'
        "来源 = `6.2` 的 `milestone.output`「batch>1 稳健性点」；本轴**未改引擎代码**，只登记「需要什么 + 为什么 + "
        "最小 patch 描述 + 预计单测（等价性 / 独立性 gate probe / B=1 逐位还原）」；**未授权**，待 Main 派唯一 engine-writer。\n"
        % TS
    )
    open(P, "w", encoding="utf-8").write(t)
    print("OK: E5 已登记进 %s" % os.path.relpath(P, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
