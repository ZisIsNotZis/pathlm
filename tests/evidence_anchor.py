"""F_KEY（复核 B12）：必要证据不得**静默跳过**。

`runs/` 被 `.gitignore` 忽略 ⇒ 干净检出下「驱动保真度」这类断言原本 `pytest.skip`，
让「证据足以复现」变成空头支票。统一规则：

1. **必要证据入 git**：`tests/evidence/longT_step0_anchors.json` 保存比对所需的最小证据
   （两侧 `train_log.jsonl` 的 step0 行 + 该行 sha256）。锚点缺失 ⇒ `pytest.fail`。
2. **本地原始证据**（`runs/…`）在时逐位复算；不在时用 `pytest.xfail` **显式**标记
   （出现在测试汇总里），不再静默 skip。
3. **入 git 的产物**（`config.json` / `results.json` / `eval.json`）缺失 ⇒ `pytest.fail`
   ——那是检出损坏，不是「训练还没跑完」。
"""

import hashlib
import json
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ANCHOR_FILE = os.path.join(ROOT, "tests", "evidence", "longT_step0_anchors.json")


def load_anchors() -> dict:
    if not os.path.exists(ANCHOR_FILE):
        pytest.fail(f"证据锚点文件缺失（必要证据必须入 git）：{ANCHOR_FILE}")
    doc = json.load(open(ANCHOR_FILE, encoding="utf-8"))
    assert doc["anchors"], "锚点文件里没有条目"
    return doc["anchors"]


def anchor(anchors: dict, key: str) -> dict:
    if key not in anchors:
        pytest.fail(f"锚点文件缺少条目 {key!r}（必需证据入 git 的清单：{sorted(anchors)}）")
    return anchors[key]


def local_step0(rel_path: str, anchor_key: str, anchors: dict) -> dict | None:
    """读本地 `runs/…/train_log.jsonl` 首行并与入 git 的锚点逐位对齐。

    返回 None 表示本地原始证据不在（已 xfail，非静默跳过）。"""
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        pytest.xfail(f"本地原始证据缺失（{rel_path}；`runs/` 不入 git）"
                     f"——已用 tests/evidence 的入 git 锚点完成可比对部分")
    raw = open(path, encoding="utf-8").readline()
    anc = anchor(anchors, anchor_key)
    got = hashlib.sha256(raw.encode()).hexdigest()
    if got != anc["first_line_sha256"]:
        pytest.fail(f"{rel_path} 首行与入 git 锚点不一致：sha256 {got} != "
                    f"{anc['first_line_sha256']}（证据被换过/重跑过？）")
    return json.loads(raw)


def require_tracked(rel_path: str, why: str) -> str:
    """入 git 的产物缺失 ⇒ 显式失败（不是 skip）。"""
    path = rel_path if os.path.isabs(rel_path) else os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        pytest.fail(f"入 git 的产物缺失：{rel_path}（{why}）——检出损坏，不是「还没跑完」")
    return path
