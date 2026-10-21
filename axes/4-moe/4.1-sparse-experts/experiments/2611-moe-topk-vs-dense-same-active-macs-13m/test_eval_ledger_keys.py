"""2611 的**键面对账测试**：`eval` 段从 `ledger()` 取的每一个键，`ledger()` 都必须真的产出。

存在的理由是一条真实缺陷：专家数第二点从 E=16 换成 E=8 之后，`run.py` 的 `eval` 段仍写着
`led["checks"]["router_tax_e16_pct"]`，而 `ledger()` 只产出 `router_tax_e4_pct` / `router_tax_e8_pct`
⇒ 8 臂全部训完、`--stage eval` 才在构造 `c4_cost` 时 KeyError，实验根 `results.json` 写不出来。
更早一次同类漏网是写死的臂名 `moe-top1-E16-s0`（`ledger` 当场 KeyError）。两处都是「字面量键 vs
生成物键」的对账缺失，所以这里对 `eval` 段里所有**字面量**取值做静态提取，再与 `ledger()` 的实际
输出求差：

    led["checks"]["<键>"]          必须出现在 ledger()["checks"]
    led["arms"]["<臂名>"]          必须出现在 ledger()["arms"]

静态提取只认字面量（`led["arms"][n]` 这种变量取值不在本测试的判定范围，由 `ledger()` 自身的
KeyError 兜住）。`ledger()` 是纯 CPU 的解析记账（不训练、不读数据），可以直接调用。

跑法：
    python3 -m pytest axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/test_eval_ledger_keys.py -q
"""

from __future__ import annotations

import importlib.util
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_2611_keys", os.path.join(HERE, "run.py"))
run2611 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run2611)


def _literal_keys(field: str) -> set[str]:
    """`run.py` 里 `led["<field>"]["<字面量>"]` 的全部字面量键。"""
    src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    return set(re.findall(r'led\["%s"\]\["([A-Za-z0-9_.\-]+)"\]' % field, src))


def test_eval_checks_keys_exist_in_ledger():
    """`eval` 取用的每个 `led["checks"]` 键都由 `ledger()` 产出（E16→E8 那次漏网的守门）。"""
    used = _literal_keys("checks")
    assert used, "静态提取不到任何 led[\"checks\"] 字面量键 ⇒ 提取规则与源码脱节，测试失效"
    produced = set(run2611.ledger()["checks"])
    assert used <= produced, f"eval 取了 ledger() 不产出的键：{sorted(used - produced)}"


def test_literal_arm_names_exist_in_ledger():
    """`run.py` 里写死的臂名字面量都真实存在于 `ledger()["arms"]`（写死旧臂名的守门）。

    抓的是「臂名形状的字面量」（`moe-top1-E4-s0` / `dense-B0-s0` / `dense-eqparam-E4-s0`），
    不要求它外面套着 `led["arms"]`：E=16 换成 E=8 那次漏网写的就是 `for n in ("moe-top1-E16-s0", …)`
    这种元组字面量。
    """
    src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read()
    used = set(re.findall(r'"((?:moe-top\d+-E\d+|dense-(?:B\d+|eqparam-E\d+))-s\d)"', src))
    assert used, "静态提取不到任何臂名字面量 ⇒ 提取规则与源码脱节，测试失效"
    produced = set(run2611.ledger()["arms"])
    assert used <= produced, f"run.py 写死的臂名不在 ledger() 里：{sorted(used - produced)}"
