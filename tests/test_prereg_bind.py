"""预登记绑定（`tools/prereg_bind.py`）的四条红线，每条都构造出会失败的反例。

1. **缺绑定必须抓红**：有 `results.json` 的实验没有 `prereg_binding.json` ⇒ `check()` 报错。
2. **契约在绑定后被改过必须抓红**：改契约一个字符 ⇒ 报出差异文件（不是沉默通过）。
3. **定序必须真的早**：`bound_at` 晚于读数产物时间 ⇒ 抓红；早于 ⇒ 通过（反向对照，
   否则上一条的红可能只是「任何绑定都红」）。
4. **绑定前已有的读数不算定序证据**：`readings_existing_at_bind` 记着的那份读数若逐字节未变
   ⇒ 只计数（`stats["predate"]`），不冒充「契约先于读数」。
5. **unbound 是显式缺失**：`bound_at="unbound"` 只计数、不判失败。

临时目录里造一棵假 `axes/` 树并 monkeypatch 模块的 `ROOT` / `AXES`，真仓库一个文件都不动。
仓内定序（提交先后）在这些用例里恒为「未提交」⇒ 不参与判定，被单测钉住的只有
「存在 / 契约未改 / 时间先后 / 显式缺失」四条。
"""

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prereg_bind  # noqa: E402


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """一棵假的研究树：一个实验目录（README.md 契约）+ 可选的结果与绑定。"""
    root = tmp_path / "repo"
    exp = root / "axes" / "1-time" / "1.1-x" / "experiments" / "9001-arm"
    exp.mkdir(parents=True)
    (exp / "README.md").write_text("# 契约\n判据线写死在这里。\n", encoding="utf-8")
    monkeypatch.setattr(prereg_bind, "ROOT", root)
    monkeypatch.setattr(prereg_bind, "AXES", root / "axes")
    return exp


def _bind(exp, bound_at, contract=None, prior=None):
    record = {
        "experiment": "axes/1-time/1.1-x/experiments/9001-arm",
        "bound_at": bound_at,
        "contract_files": contract if contract is not None else
        {"README.md": prereg_bind.sha256(exp / "README.md")},
        "tool": "tools/prereg_bind.py",
    }
    if prior is not None:
        record["readings_existing_at_bind"] = prior
    (exp / "prereg_binding.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def _write_results(exp, body='{"bpc": 1.5}'):
    (exp / "results.json").write_text(body, encoding="utf-8")
    return exp / "results.json"


def test_missing_binding_fails(tree):
    """有读数没绑定 ⇒ 报「预登记未绑定」，且不是 unbound 那种「只计数」。"""
    _write_results(tree)
    errors, stats = prereg_bind.check()
    assert stats["missing"] == 1 and stats["experiments"] == 1
    assert any("预登记未绑定" in e for e in errors), errors


def test_contract_edited_after_binding_fails(tree):
    """契约改一个字符 ⇒ 报「契约在绑定后被改过」并列出该文件。"""
    _write_results(tree)
    _bind(tree, "2020-01-01T00:00:00+00:00")
    assert prereg_bind.check()[0] == [], "先确认这份绑定本身是绿的（否则下面的红是假的）"
    (tree / "README.md").write_text("# 契约\n判据线写死在这里。 \n", encoding="utf-8")
    errors, _ = prereg_bind.check()
    assert any("契约在绑定后被改过：README.md" in e for e in errors), errors


def test_binding_must_actually_predate_the_reading(tree):
    """bound_at 晚于读数产物时间 ⇒ 抓红；早于 ⇒ 通过。"""
    _write_results(tree)
    os.utime(tree / "results.json", (1_600_000_000, 1_600_000_000))     # 2020-09-13
    _bind(tree, "2021-06-01T00:00:00+00:00")                              # 晚于读数
    errors, _ = prereg_bind.check()
    assert any("不早于判序锚" in e for e in errors), errors

    _bind(tree, "2020-01-01T00:00:00+00:00")                              # 早于读数
    assert prereg_bind.check()[0] == []


def test_reading_that_existed_at_bind_is_counted_not_ordered(tree):
    """绑定时已有的读数逐字节未变 ⇒ 定序不可判（predate 计数），不是「契约先于读数」。"""
    results = _write_results(tree)
    os.utime(results, (1_700_000_000, 1_700_000_000))
    _bind(tree, "2020-01-01T00:00:00+00:00",
          prior={"axes/1-time/1.1-x/experiments/9001-arm/results.json":
                 prereg_bind.sha256(results)})
    errors, stats = prereg_bind.check()
    assert errors == [] and stats["predate"] == 1 and stats["ordered"] == 0

    _write_results(tree, '{"bpc": 1.4}')          # 重跑过：这份读数不再是绑定时那份
    os.utime(results, (1_700_000_100, 1_700_000_100))
    errors, stats = prereg_bind.check()
    assert errors == [] and stats["ordered"] == 1 and stats["predate"] == 0


def test_unbound_is_counted_not_failed(tree):
    """历史产物（bound_at="unbound"）只计数，不判失败。"""
    _write_results(tree)
    _bind(tree, prereg_bind.UNBOUND)
    errors, stats = prereg_bind.check()
    assert errors == [] and stats["unbound"] == 1


def test_backfill_writes_explicit_absence_and_never_a_time(tree):
    """回填只写 unbound（显式缺失），不填看起来像真的时间；已有绑定不覆盖。"""
    _write_results(tree)
    assert prereg_bind.backfill_unbound() == 0
    record = json.loads((tree / "prereg_binding.json").read_text(encoding="utf-8"))
    assert record["bound_at"] == prereg_bind.UNBOUND
    assert "unbound_reason" in record and "contract_files" not in record

    bound = _bind(tree, "2020-01-01T00:00:00+00:00")
    prereg_bind.backfill_unbound()
    kept = json.loads((tree / "prereg_binding.json").read_text(encoding="utf-8"))
    assert kept["bound_at"] == bound["bound_at"], "回填不许覆盖已有绑定"
