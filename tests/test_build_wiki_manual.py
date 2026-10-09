"""`build_wiki.py` 的 MANUAL 区块契约（独立复核查出缺陷的回归测试）。

缺陷：`--check` 直接拿「生成内容」与「既有页面」比对，不嫁接 MANUAL 块 ⇒ 页面里
`<!-- MANUAL:begin --> … <!-- MANUAL:end -->` 一加手写内容就永远报「不一致」，
手写块**实际失效**（与此同时不带 `--check` 的重跑是保留手写块的）。

钉住四件事（都可失败）：
1. 只手写块有差异 ⇒ `--check` **一致**（改前红）；重跑保留手写内容、AUTO 照常更新；
2. **幂等**：连跑两次不产生差异；
3. MANUAL 缺失 / 畸形 / 内含 AUTO 标记 ⇒ **显式报错**，且原文件不被覆盖；
4. MANUAL 只能改自己的区间：嫁接后 AUTO 区与重新生成的 AUTO 区逐字相同。

本文件按**磁盘格式字面量**校验标记（而不是复用模块常量）：标记本身就是对外的格式契约。
"""

import os
import sys

import pytest
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_wiki as bw                                                # noqa: E402

AUTO_BEGIN = "<!-- AUTO:begin -->"
AUTO_END = "<!-- AUTO:end -->"
MANUAL_BEGIN = "<!-- MANUAL:begin -->"
MANUAL_END = "<!-- MANUAL:end -->"
EMPTY_MANUAL = MANUAL_BEGIN + "\n" + MANUAL_END

GRID = {
    "id": "X1-demo",
    "axis": "X",
    "axis_dir": "X-demo",
    "title": "示例格",
    "claim": "示例命题。",
    "options": ["甲", "乙"],
    "judge": "bpc 是否下降（等 FLOPs）。",
    "cost": "13M 单臂 ~25 min",
    "status": "todo",
    "local": {},
}

AXIS = {"id": "X", "dir": "X-demo", "title": "示例轴"}

HAND = "手写：这一格为什么值得单独立项。"


def with_manual(text: str, body: str) -> str:
    """把生成页面的空 MANUAL 块替换成带手写内容的块。"""
    assert EMPTY_MANUAL in text
    return text.replace(EMPTY_MANUAL, MANUAL_BEGIN + "\n" + body + "\n" + MANUAL_END)


def auto_region(text: str) -> str:
    s = text.index(AUTO_BEGIN) + len(AUTO_BEGIN)
    e = text.index(AUTO_END)
    return text[s:e]


def regen(path, content, check=False):
    """调 `write`，返回 `(changed, 报错信息)`；契约是「必须拒绝」，不规定异常类名。"""
    changed: list = []
    try:
        bw.write(path, content, check=check, changed=changed)
    except Exception as e:                                             # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return changed, ""


def page_file(tmp_path, body: str = HAND):
    p = tmp_path / "README.md"
    p.write_text(with_manual(bw.page(GRID, AXIS), body), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 1. 缺陷回归
def test_check_consistent_when_only_manual_block_differs(tmp_path):
    """只手写块有差异时 `--check` 必须一致（改前：报「与 catalog 不一致」）。"""
    p = page_file(tmp_path)
    changed, err = regen(p, bw.page(GRID, AXIS), check=True)
    assert err == "", err
    assert changed == [], "只手写块差异被当成不一致 ⇒ MANUAL 块实际失效"


def test_check_still_reports_stale_auto(tmp_path):
    """反向对照：AUTO 内容真的过期时 `--check` 仍必须报出来。"""
    p = page_file(tmp_path)
    changed, err = regen(p, bw.page(dict(GRID, title="改过的标题"), AXIS), check=True)
    assert err == "", err
    assert changed == [p]


def test_regeneration_keeps_manual_and_updates_auto(tmp_path):
    p = page_file(tmp_path)
    fresh = bw.page(dict(GRID, title="改过的标题", status="judged", verdict="结论已定。"), AXIS)
    changed, err = regen(p, fresh)
    assert err == "", err
    text = p.read_text(encoding="utf-8")
    assert HAND in text, "重跑把手写块丢了"
    assert "# X1-demo · 改过的标题" in text, "AUTO 没有更新"
    assert "示例格" not in text, "旧 AUTO 没被换掉"
    assert changed == [p]


def test_regeneration_is_idempotent(tmp_path):
    p = page_file(tmp_path)
    fresh = bw.page(dict(GRID, title="改过的标题", status="judged"), AXIS)
    bw.write(p, fresh, check=False, changed=[])
    once = p.read_text(encoding="utf-8")
    for _ in range(2):
        changed: list = []
        bw.write(p, fresh, check=False, changed=changed)
        assert changed == [], "第二次重跑仍报改动 ⇒ 不幂等"
        assert p.read_text(encoding="utf-8") == once


# ---------------------------------------------------------------- 4. 越界防护
def test_manual_block_cannot_touch_auto_region(tmp_path):
    p = page_file(tmp_path, "## 1. 命题\n\n手写块里塞一个假标题，看它会不会顶掉 AUTO。")
    fresh = bw.page(GRID, AXIS)
    bw.write(p, fresh, check=False, changed=[])
    text = p.read_text(encoding="utf-8")
    assert auto_region(text) == auto_region(fresh), "手写内容渗进了 AUTO 区"
    assert text.rstrip().endswith("看它会不会顶掉 AUTO。\n" + MANUAL_END), text[-200:]


# ---------------------------------------------------------------- 3. 缺失 / 畸形
def test_missing_manual_block_raises_instead_of_overwriting(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(auto_region(bw.page(GRID, AXIS)), encoding="utf-8")   # 整页没有 MANUAL 块
    before = p.read_text(encoding="utf-8")
    for check in (False, True):
        changed, err = regen(p, bw.page(GRID, AXIS), check=check)
        assert changed is None, "MANUAL 块缺失时被静默整页覆盖"
        assert "没有 MANUAL 区块" in err, err
        assert p.read_text(encoding="utf-8") == before, "报错路径仍改了文件"


@pytest.mark.parametrize("bad", [
    MANUAL_BEGIN + "\n" + HAND + "\n",                       # 只有 begin
    HAND + "\n" + MANUAL_END + "\n",                         # 只有 end
    MANUAL_BEGIN + "\n" + HAND + "\n" + MANUAL_BEGIN + "\n" + MANUAL_END + "\n",  # 重复 begin
    MANUAL_END + "\n" + HAND + "\n" + MANUAL_BEGIN + "\n",   # 顺序颠倒
])
def test_malformed_manual_block_raises(tmp_path, bad):
    p = tmp_path / "README.md"
    p.write_text(bw.page(GRID, AXIS) + "\n" + bad, encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    changed, err = regen(p, bw.page(GRID, AXIS))
    assert changed is None, f"畸形 MANUAL 块被静默接受（changed={changed}）"
    assert "MANUAL" in err, err
    assert p.read_text(encoding="utf-8") == before


def test_manual_block_smuggling_auto_marker_raises(tmp_path):
    """手写块里出现 AUTO 标记（下游按首个标记解析会把内容并进 AUTO 区）⇒ 报错。"""
    p = page_file(tmp_path, "手写块里塞 " + AUTO_BEGIN + " 想伪装成自动块")
    before = p.read_text(encoding="utf-8")
    changed, err = regen(p, bw.page(GRID, AXIS))
    assert changed is None, f"手写块里的 AUTO 标记被静默接受（changed={changed}）"
    assert "AUTO 标记" in err, err
    assert p.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------- 端到端（真 CLI）
def build_min_repo(tmp_path):
    """一个最小 catalog + 空轴目录，供 main() 端到端跑。"""
    catalog = {
        "version": 1,
        "project": {"name": "demo"},
        "axes": [{"id": "X", "dir": "X-demo", "title": "示例轴", "primitive": "I",
                  "claim": "示例轴命题。", "definition": "示例定义。",
                  "subdims": "X1 · X2", "local_summary": "无本地实测。"}],
        "grids": [GRID],
    }
    (tmp_path / "axes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "catalog.yaml").write_text(
        yaml.safe_dump(catalog, allow_unicode=True, sort_keys=False), encoding="utf-8")


def test_cli_check_ok_after_manual_edit(tmp_path, monkeypatch, capsys):
    build_min_repo(tmp_path)
    monkeypatch.setattr(bw, "ROOT", tmp_path)
    monkeypatch.setattr(bw, "CATALOG", tmp_path / "catalog.yaml")

    monkeypatch.setattr(sys, "argv", ["build_wiki.py"])
    assert bw.main() == 0, capsys.readouterr().out

    p = tmp_path / "axes" / "X-demo" / "X1-demo" / "README.md"
    p.write_text(with_manual(p.read_text(encoding="utf-8"), HAND), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["build_wiki.py", "--check"])
    rc = bw.main()
    out = capsys.readouterr().out
    assert rc == 0 and "一致" in out, out

    # 手写块被现场改坏 ⇒ CLI 必须报错退出，不静默
    p.write_text(p.read_text(encoding="utf-8").replace(MANUAL_END, ""), encoding="utf-8")
    rc = bw.main()
    captured = capsys.readouterr()
    assert rc == 1 and "MANUAL" in (captured.err + captured.out), captured
