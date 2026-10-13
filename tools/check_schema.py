#!/usr/bin/env python3
"""check_schema.py — 兼容入口：转调 `tools/check_tree.py`。

**迁移通告**：新结构下研究项的 SSOT 是 `axes/**/project.yaml`（契约
`docs/research-schema.md`）。旧的生成式 wiki（`catalog.yaml` / `tools/build_wiki.py` /
`INDEX.md`）已**退役**，本文件原先把 `catalog.yaml` + 课题页九节 + 生成一致性当校验对象，
该职责一并取消。现在它只做一件事：把调用转给新校验器 `tools/check_tree.py`，
并继续导出证据锚点工具（`check_anchors` / `check_anchor_syntax` / `GitUnavailable`
/ `reachable_commits`）供测试与其它脚本复用。

用法：python3 tools/check_schema.py   （退出码 0 = 全过；1 = 有错）
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 转调 + 复用锚点工具（保持旧 import 路径可用）。
from check_tree import (  # noqa: E402,F401
    ANCHOR_RE,
    AT_TOKEN_RE,
    GitUnavailable,
    check_anchor_syntax,
    check_anchors,
    main,
    reachable_commits,
)

if __name__ == "__main__":
    sys.exit(main())
