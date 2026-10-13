#!/usr/bin/env python3
"""ticket 64 · Phase A 步骤 1（补丁 2）：role_map 实名 + 剩余事实修正。

第一遍（`edits_phase_a.py`）命中 16/26；未命中 10 处的原因：
  * 8 处 role_map —— ROLE_OLD 的 `{writer}` 占位被喂了「新」字符串（脚本 bug）；
  * 2 处 3.2.3 —— 目标文本实际在 **3.2 分组 README** 与 objective_criteria 的
    `必须报：参数增量 = vocab×d（单列）`（原文无粗体标记）。
本脚本只做这 10 处，命中纪律同上（必须恰好 1 次，否则非零退出）。
"""
import sys

EDITS = []
BASE = "axes/3-width"

ROLE_OLD = """role_map:
  owner: "轴 3 宽度轴 lead（待指派）"
  writer: "{old}"
  reviewer: "fresh 复核者（reviewer 池，不得为本项 writer）"
  integrator: "研究树 integrator"
"""
ROLE_NEW = """role_map:
  owner: "axis3-lead（轴 3 宽度轴 lead）"
  writer: "{new}"
  reviewer: "axis3-reviewer（fresh 独立上下文 subagent，**不得**为本项 writer；复核排在 writer 停止之后）"
  integrator: "tree-integrator（研究树 integrator；本轴合并与跨轴一致性核对由 axis3-lead 代行）"
"""
ROLES = {
    "3.1.1-uniform-narrowing": ("3.1.1 worker（单一写入者）",
                                "axis3-w3.1.1（3.1.1 worker，单一写入者）"),
    "3.1.2-layer-width": ("3.1.2 worker（单一写入者）",
                          "axis3-w3.1.2（3.1.2 worker，单一写入者）"),
    "3.1.3-position-width": ("3.1.3 worker（单一写入者）",
                             "axis3-w3.1.3（3.1.3 worker，单一写入者）"),
    "3.1.4-ffn-ratio": ("3.1.4 worker（单一写入者）",
                        "axis3-w3.1.4（3.1.4 worker，单一写入者；本轮 WP1 的 writer = axis3-lead）"),
    "3.1.5-channel-sparsity": ("3.1.5 worker（单一写入者）",
                               "axis3-w3.1.5（3.1.5 worker，单一写入者）"),
    "3.2.1-readout-reparam": ("3.2.1 worker（单一写入者）",
                              "axis3-w3.2.1（3.2.1 worker，单一写入者）"),
    "3.2.2-per-exit-readout": ("3.2.2 worker（单一写入者）",
                               "axis3-w3.2.2（3.2.2 worker，单一写入者）"),
    "3.2.3-tied-untied": ("3.2.3 worker（单一写入者）",
                          "axis3-w3.2.3（3.2.3 worker，单一写入者）"),
}
for slug, (old, new) in ROLES.items():
    EDITS.append((f"{BASE}/{slug}/project.yaml",
                  ROLE_OLD.format(old=old), ROLE_NEW.format(new=new),
                  f"role_map {slug}"))

EDITS.append((f"{BASE}/3.2-readout/README.md",
              "解绑 = vocab×d（本仓小词表 +52480，大词表不可 scale）",
              "解绑 = vocab×d（本仓 vocab=206、小词表 +52736，大词表不可 scale）",
              "3.2 README vocab"))
EDITS.append((f"{BASE}/3.2.3-tied-untied/project.yaml",
              "必须报：参数增量 = vocab×d（单列）",
              "必须报：参数增量 = vocab×d = 206×256 = 52736（单列）",
              "3.2.3 判据参数量"))


def main():
    n_hits, misses = 0, []
    for path, old, new, tag in EDITS:
        src = open(path, encoding="utf-8").read()
        c = src.count(old)
        if c != 1:
            misses.append((tag, path, c))
            continue
        open(path, "w", encoding="utf-8").write(src.replace(old, new))
        n_hits += 1
    print(f"命中 {n_hits}/{len(EDITS)} 处替换")
    if misses:
        print("未命中清单（必须为 0）：")
        for tag, path, c in misses:
            print(f"  [{tag}] {path} 命中 {c} 次")
        sys.exit(1)


if __name__ == "__main__":
    main()
