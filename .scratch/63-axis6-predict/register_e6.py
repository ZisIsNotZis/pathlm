#!/usr/bin/env python3
"""登记 E6（位置忠实的多兄弟树验证原语）到受控引擎通道。
   scripted + 命中计数断言；只插入一节 + 一条变更记录。"""
import os
import sys

ROOT = "/home/z/vibe/pathlm"
P = os.path.join(ROOT, ".scratch/67-engine-changes/TICKET.md")
TS = "2026-10-14T23:45:00+08:00"

E6 = """### E6 · 轴 6 · 位置忠实的「多兄弟」树验证原语（`step_multi` / `step_tree_multi` 的绝对位置表）

| 项 | 内容 |
|---|---|
| 来源 | `axes/6-predict/6.2-speculative-decoding/experiments/2675-second-tree-13m/GAPS.md` §G1（WP 6.2.5 在实现前实测发现；该 WP 的 `QUESTIONS.md` 记同一事实）。 |
| 需要什么 | `pathlm/decode.py` 的 `step_multi`（= `step_tree_multi` 的唯一实现）把**扁平序列第 j 个节点放在绝对位置 `p0+j`**（`model.pos_embed.weight[self.n:self.n+K]`，`decode.py:224`）。这在**链**（`parents=[-1,0,1,…]`）上与因果逐位一致，但**不能表达同深度多兄弟**：K 个 N+2 候选的正确位置**都是 N+2**，扁平布局只能给 p0+1、p0+2… ⇒ 候选 i≥2 的 N+3 验证用的是**错误位置**上的模型输出，且接受节点的 KV 写回位置也错。 |
| 为什么 | 这是「完整二级树（N+3 以被接受的 N+2 候选为条件）」的**最后一个机制缺口**：引擎能给**一条**链做精确树验证（E3 的 `decode_spec` 二级链正是如此），但**不能**做分支树。不修 ⇒ 轴 6 只能用实验侧的「位置忠实 K 行平铺」替代（2640 `spec_tree.tree_forward`，本轴已这么做），任何走引擎原语的 K≥2 分支树读数都**不是精确验证**（L3 无损性硬门会红）。 |
| 实测证据 | WP 6.2.5 的 CPU tiny-model probe：扁平布局里第 2 个候选节点的 logits 与「位置正确的条件分布」（同前缀 2 节点链）的**总变差距离 TV mean = 0.0116 / max = 0.0133**，而 fp 重排噪声仅 **3e-8**（差 10^6 量级）；根节点与第 1 个候选只剩 3e-8 ⇒ 误差恰好来自**兄弟/草稿节点的位置错位**。见 `2675-second-tree-13m/test_second_tree.py::test_flat_tree_siblings_are_not_position_faithful` 与 `QUESTIONS.md` §3。 |
| 最小 patch 描述 | 给 `Decoder.step_multi` / `_run_stack_multi` / `step_tree_multi` 增一个**可选** `positions: list[int] | None = None`（缺省 `None` ⇒ 保持现状 `p0+j`，**逐位不变**），把它用于 `model.pos_embed.weight[positions]` 与 `self.layer_pos[i] = list(positions)`；树的每个节点用**真实生成位置**（根 N+1、候选 N+2、草稿 N+3）。配套 3 条单测：① 缺省 `None` 与现状**逐位相同**；② `parents=[-1,0,0]` + `positions=[N+1,N+2,N+2]` 下每个节点的 logits **等于**对应根到叶路径（`step_multi`）的读数；③ 兄弟不可见（改兄弟行内容不改本行 logits）。无新增参数；位置表长度 = 节点数。**只动 `pathlm/decode.py`，预计 20–40 行**。 |
| 授权状态 | **未授权**（C3 已建通道；待 Main 派唯一 engine-writer）。轴 6 本 WP **未改** `pathlm/**`：改用实验侧「位置忠实的 K 行平铺」（2640 `spec_tree.tree_forward`，只读复用）实现两臂主判，并把字面 `step_tree_multi` 分支路径降为**诊断臂**（`lvl2_flat`）如实报 L3。 |
"""


def main():
    t = open(P, encoding="utf-8").read()
    anchor = "## 3. 与既有文档的关系"
    assert t.count(anchor) == 1, "anchor"
    t = t.replace(anchor, E6 + "\n" + anchor)
    t = t.rstrip("\n") + (
        '\n- `%s` · axis6-lead（轴 6 预测轴 lead，ticket 63 Phase D）· 登记 **E6**（位置忠实的多兄弟树验证原语）：'
        "来源 = WP 6.2.5 在实现前实测发现的引擎能力缺口（`step_tree_multi` 的扁平节点位置 = `p0+j`，对同深度兄弟不正确；"
        "TV 0.0116 vs fp 噪声 3e-8）；本轴**未改引擎代码**，只登记「需要什么 + 为什么 + 最小 patch 描述 + 预计 3 条单测」；"
        "**未授权**，待 Main 派唯一 engine-writer。\n" % TS
    )
    open(P, "w", encoding="utf-8").write(t)
    print("OK: E6 已登记")
    return 0


if __name__ == "__main__":
    sys.exit(main())
