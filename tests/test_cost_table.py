"""代价表（`docs/cost-table.md`）与其事实源（`axes/**/project.yaml` 的 `laws` / `cost_rows`）。

钉住四件事，每件都可失败：
1. **生成物被手改必须被抓红**：把 `docs/cost-table.md` 复制到临时目录、改一个字符，
   `build_cost_table.py --check` 必须非零退出并打印差异摘要（真仓库文件不动）。
2. **字段校验**：未知定律编号、`source` 指向不存在的路径必须报错（`check_tree` 侧）。
3. **宪法级一致性**：`docs/cost-table.md` 与当前 `project.yaml` 逐字节一致（生成物不过期），
   且定律 1..11 每条都有 `cost_rows` 覆盖（没有覆盖就没法引用）。
4. **数字与落盘逐位一致**：每条 `cost_rows` 的 `delta` 必须真的出现在它 `source` 指的文件里
   ——这是「不许新算、不许换算口径」的机械检查（Unicode 减号归一为 ASCII `-`）。
"""

import os
import re
import shutil
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import build_cost_table  # noqa: E402
import check_tree  # noqa: E402

NUMBERISH = re.compile(r"[0-9][0-9,]*(?:\.[0-9]+)?")


def test_laws_unknown_number_fails():
    errs = check_tree.check_laws_and_costs({"laws": [1, 12], "cost_rows": []}, "axes/x/project.yaml")
    assert len(errs) == 1, errs
    assert "12" in errs[0] and "未知定律编号" in errs[0]


def test_laws_duplicate_and_non_int_fail():
    errs = check_tree.check_laws_and_costs({"laws": [2, 2, "3"], "cost_rows": []},
                                           "axes/x/project.yaml")
    assert any("重复" in e for e in errs), errs
    assert any("必须是整数定律编号" in e for e in errs), errs


def test_cost_row_source_must_exist():
    row = {"law": 1, "change": "x", "touches_trunk": False, "metric": "clean bpc Δ",
           "delta": "+0.1", "comparator": "y", "scale": "13M", "boundary": "z",
           "source": "axes/no-such-dir/no-such-file.md",
           "delta_direction": "comparator_minus_arm"}
    errs = check_tree.check_laws_and_costs({"laws": [1], "cost_rows": [row]}, "axes/x/project.yaml")
    assert len(errs) == 1, errs
    assert "不存在的路径" in errs[0] and "no-such-file.md" in errs[0]


def test_cost_row_shape_checks():
    base = {"law": 4, "change": "x", "touches_trunk": "yes", "metric": "m", "delta": "abc",
            "comparator": "c", "scale": "s", "boundary": "b", "source": "axes",
            "delta_direction": "comparator_minus_arm"}
    errs = check_tree.check_laws_and_costs({"laws": [1], "cost_rows": [dict(base, law=7),
                                                                      dict(base, law=99)]},
                                           "axes/x/project.yaml")
    text = "\n".join(errs)
    assert "touches_trunk" in text and "true/false" in text      # 必须 bool
    assert "delta" in text and "带符号数字" in text               # 必须数字
    assert "不在本项 laws" in text        # law=7 合法但本项没声明它
    assert "未知定律编号" in text          # law=99 不在 1..11


def test_check_fails_when_generated_file_is_hand_edited(tmp_path, monkeypatch):
    """把表复制到临时目录、改一个字符 ⇒ --check 必须失败并给出差异摘要。"""
    hand = tmp_path / "cost-table.md"
    shutil.copyfile(build_cost_table.OUT, hand)
    text = hand.read_text(encoding="utf-8")
    assert text, "生成物不得为空"
    hand.write_text(text.replace("定律", "定率", 1), encoding="utf-8")
    monkeypatch.setattr(build_cost_table, "OUT", hand)
    assert build_cost_table.main(["--check"]) == 1


def test_check_passes_on_freshly_generated_file(tmp_path, monkeypatch):
    """反向对照：未手改时同一条路径必须通过（否则上面的红是假的）。"""
    clean = tmp_path / "cost-table.md"
    shutil.copyfile(build_cost_table.OUT, clean)
    monkeypatch.setattr(build_cost_table, "OUT", clean)
    assert build_cost_table.main(["--check"]) == 0


def test_generated_table_is_up_to_date_with_project_yaml():
    """生成物过期（project.yaml 改了但没重新生成表）⇒ 失败。"""
    rows, errors = build_cost_table.collect()
    assert not errors, errors
    text = build_cost_table.render(rows, build_cost_table.law_titles())
    assert text == build_cost_table.OUT.read_text(encoding="utf-8"), (
        "docs/cost-table.md 与 axes/**/project.yaml 不一致；改完事实请运行 "
        "python3 tools/build_cost_table.py")


def test_every_law_has_at_least_one_cost_row():
    rows, errors = build_cost_table.collect()
    assert not errors, errors
    assert build_cost_table._uncovered(rows) == [], (
        f"这些定律没有任何 cost_rows 支撑：{build_cost_table._uncovered(rows)}")


def test_every_delta_literally_appears_in_its_source():
    """`delta` 必须逐位抄自 `source` 指向的文件（不许新算 / 换算口径）。"""
    rows, errors = build_cost_table.collect()
    assert not errors, errors
    bad = []
    for row in rows:
        src = build_cost_table.ROOT / check_tree.source_path(row["source"])
        text = src.read_text(encoding="utf-8", errors="replace").replace("\u2212", "-")
        delta = str(row["delta"]).strip()
        if delta not in text and delta.lstrip("+") not in text:
            bad.append(f"{row['_id']}: {delta} 不在 {row['source']}")
    assert bad == [], "这些 cost_rows 的 delta 在落盘文件里找不到：\n" + "\n".join(bad)


@pytest.mark.parametrize("law", range(1, check_tree.LAWS_TOTAL + 1))
def test_playbook_defines_every_law(law):
    """定律正文（标题）必须能在 `docs/playbook.md` 找到——表只指回它，不复制它。"""
    assert law in build_cost_table.law_titles(), f"docs/playbook.md 缺「定律 {law}」"
