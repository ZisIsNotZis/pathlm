"""研究目录 `docs/catalog.md` 与其事实源（`axes/**/project.yaml`）。

钉住五件事，每件都可失败：
1. **生成物不过期**：`docs/catalog.md` 与当前 `project.yaml` 逐字节一致（改完事实要重跑生成器）；
2. **七列契约**：表头恰是 id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针，多一列少一列都不算；
3. **一行一个研究项**：数据行数 = `axes/**/project.yaml` 的个数，id 无重复，状态逐行等于
   `project.yaml::status`，指针逐行指向真实存在的文件；
4. **手改必红**（反向对照两遍）：复制到临时目录改一个字符 ⇒ 红；未改的同一份 ⇒ 绿；
5. **增删必红**：少一行、多一行都让 `--check` 变红（否则「行数 = 研究项数」是句空话）。
"""

import os
import shutil
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_catalog  # noqa: E402


def _project_docs():
    out = []
    for path in sorted(build_catalog.AXES.rglob("project.yaml")):
        out.append((path, yaml.safe_load(path.read_text(encoding="utf-8"))))
    return out


def test_catalog_is_up_to_date_with_project_yaml():
    rows, errors = build_catalog.collect()
    assert not errors, errors
    text = build_catalog.render(rows)
    assert text == build_catalog.OUT.read_text(encoding="utf-8"), (
        "docs/catalog.md 与 axes/**/project.yaml 不一致；改完事实请运行 "
        "python3 tools/build_catalog.py")


def test_header_is_exactly_seven_columns():
    rows, errors = build_catalog.collect()
    assert not errors, errors
    lines = build_catalog.render(rows).splitlines()
    header = next(line for line in lines if line.startswith("| id |"))
    cells = [c.strip() for c in header.strip("|").split("|")]
    assert cells == ["id", "标题", "目的", "验收", "期望产出", "状态", "指针"]


def test_one_row_per_research_item():
    rows, errors = build_catalog.collect()
    assert not errors, errors
    sources = list(build_catalog.AXES.rglob("project.yaml"))
    assert len(rows) == len(sources), f"目录 {len(rows)} 行，研究项 {len(sources)} 个"
    assert len({r["id"] for r in rows}) == len(rows), "研究项 id 在目录里重复"
    body = [ln for ln in build_catalog.OUT.read_text(encoding="utf-8").splitlines()
            if ln.startswith("| ") and not ln.startswith("| id ") and not ln.startswith("|---")]
    assert len(body) == len(rows), f"落盘目录 {len(body)} 行，研究项 {len(rows)} 个"


def test_status_and_pointer_come_from_the_fact_home():
    rows, errors = build_catalog.collect()
    assert not errors, errors
    by_source = {p.relative_to(build_catalog.ROOT).as_posix(): doc for p, doc in _project_docs()}
    for row in rows:
        doc = by_source[row["source"]]
        assert row["status"] == doc["status"], f"{row['id']} 状态与 project.yaml 不一致"
        assert row["acceptance"] == doc["milestone"]["acceptance"], row["id"]
        assert row["output"] == doc["milestone"]["output"], row["id"]
        assert (build_catalog.ROOT / row["pointer"]).is_file(), f"{row['id']} 的指针悬空：{row['pointer']}"


def test_ids_are_sorted_naturally():
    rows, errors = build_catalog.collect()
    assert not errors, errors
    assert [r["id"] for r in rows] == sorted((r["id"] for r in rows),
                                             key=build_catalog.id_key)
    # 自然序不等于字典序：`1.1.4.10` 必须排在 `1.1.4.9` 之后
    assert build_catalog.id_key("1.1.4.10") > build_catalog.id_key("1.1.4.9")


def test_check_fails_when_catalog_is_hand_edited(tmp_path, monkeypatch):
    """改一个字符 ⇒ `--check` 失败并打印差异摘要。"""
    hand = tmp_path / "catalog.md"
    shutil.copyfile(build_catalog.OUT, hand)
    text = hand.read_text(encoding="utf-8")
    assert text
    hand.write_text(text.replace("进行中", "已完成", 1), encoding="utf-8")
    monkeypatch.setattr(build_catalog, "OUT", hand)
    assert build_catalog.main(["--check"]) == 1


def test_check_passes_on_freshly_generated_copy(tmp_path, monkeypatch):
    """反向对照：未手改的同一份必须通过（否则上面的红是假的）。"""
    clean = tmp_path / "catalog.md"
    shutil.copyfile(build_catalog.OUT, clean)
    monkeypatch.setattr(build_catalog, "OUT", clean)
    assert build_catalog.main(["--check"]) == 0


def test_check_fails_when_a_row_is_dropped(tmp_path, monkeypatch):
    """少一行（漏了一个研究项）⇒ 红。"""
    short = tmp_path / "catalog.md"
    lines = build_catalog.OUT.read_text(encoding="utf-8").splitlines()
    data = [i for i, ln in enumerate(lines) if ln.startswith("| ") and not ln.startswith("| id ")]
    dropped = lines[: data[0]] + lines[data[0] + 1:]
    short.write_text("\n".join(dropped) + "\n", encoding="utf-8")
    monkeypatch.setattr(build_catalog, "OUT", short)
    assert build_catalog.main(["--check"]) == 1


def test_check_fails_when_a_row_is_added(tmp_path, monkeypatch):
    """多一行（目录比研究项多）⇒ 红。"""
    extra = tmp_path / "catalog.md"
    lines = build_catalog.OUT.read_text(encoding="utf-8").splitlines()
    last = max(i for i, ln in enumerate(lines) if ln.startswith("| ") and not ln.startswith("| id "))
    lines.insert(last + 1, "| 9.9 | 虚构研究项 | — | — | — | 未开始 | `axes/9-backlog/project.yaml` |")
    extra.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(build_catalog, "OUT", extra)
    assert build_catalog.main(["--check"]) == 1


def test_missing_field_fails_loudly_instead_of_inventing_one(tmp_path):
    """缺 `milestone.output` 时不补空值：报错，而不是生成一行「—」。"""
    axis = tmp_path / "axes" / "x"
    axis.mkdir(parents=True)
    (axis / "project.yaml").write_text(
        "id: '9.9'\ntitle: t\npurpose: p\nstatus: 未开始\ndir: axes/x\nmilestone:\n  acceptance: a\n",
        encoding="utf-8")
    errors: list[str] = []
    original = build_catalog.AXES
    try:
        build_catalog.AXES = tmp_path / "axes"
        rows, errors = build_catalog.collect()
    finally:
        build_catalog.AXES = original
    assert rows == []
    assert any("milestone.output" in e for e in errors), errors
