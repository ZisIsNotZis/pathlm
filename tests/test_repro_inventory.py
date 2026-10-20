"""复现清单（`docs/repro.md` + `tools/inventory_runs.py`）的三条红线。

每件都可失败：
1. **清单与磁盘不一致必须抓红**：在一份只含入库文件的临时副本里多造一个 run 目录，
   `--check` 必须非零退出；同一份副本**不动**时必须通过（反向对照，否则上面的红是假的）。
2. **生成物被手改必须抓红**：改 `docs/repro.md` 里的一个字，`--check` 必须失败。
3. **`not_reproducible` 必须写明缺什么**：缺 `missing` 即判失败（构造行 + 现行清单各一遍）。

临时副本只放 `git ls-files` 列出的文件，并在副本里 `git init && git add`，
这样「入库集合」由副本自己的索引决定——造出来的假 run 目录对扫描可见，
其余条件与真仓库逐字相同（真仓库的文件一个都不动）。
"""

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import inventory_runs as inv  # noqa: E402

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="需要 git 才能造出确切的入库集合")


def _clone_of_tracked_files(dest):
    """把真仓库**已入库的文件**复制到 `dest`，并在那里建一个只含这些文件的 git 索引。"""
    tracked = subprocess.run(["git", "-C", ROOT, "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    rels = [r for r in tracked.split("\0") if r]
    # `docs/repro.md` 可能尚未提交（本工作包第一次生成它时），但它是被检查的对象，必须一起带过去。
    rels.append("docs/repro.md")
    for rel in rels:
        src = os.path.join(ROOT, rel)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    # `-f`：真仓库里有少数文件是 `git add -f` 加进去的（例如被 `.gitignore` 的 `*.log`
    # 里的训练日志）；不强制索引它们，副本的「入库集合」就会与真仓库不同。
    subprocess.run(["git", "-C", str(dest), "add", "-A", "-f"], check=True)


@requires_git
def test_extra_run_dir_fails_check(tmp_path):
    """磁盘上多出一个 run 目录（这里造在临时副本里）⇒ 清单过期 ⇒ `--check` 必须失败。"""
    tree = tmp_path / "tree"
    _clone_of_tracked_files(tree)

    # 反向对照：什么都不动时必须通过。
    assert inv.main(["--check", "--root", str(tree)]) == 0

    fake = tree / "axes" / "1-time" / "1.1.1-hard-window" / "experiments" / "2626-longT-clean" / "runs" / "fake-arm"
    fake.mkdir(parents=True)
    (fake / "results.json").write_text('{"bpc": 1.0}', encoding="utf-8")
    subprocess.run(["git", "-C", str(tree), "add", "-A", "-f"], check=True)

    assert inv.main(["--check", "--root", str(tree)]) == 1


@requires_git
def test_hand_edited_render_fails_check(tmp_path):
    """生成物被手改 ⇒ `--check` 必须失败（逐字节比对）。"""
    tree = tmp_path / "tree"
    _clone_of_tracked_files(tree)
    render = tree / "docs" / "repro.md"
    text = render.read_text(encoding="utf-8")
    assert text, "生成物不得为空"
    render.write_text(text.replace("rebuildable_only", "rebuildable-only", 1), encoding="utf-8")
    assert inv.main(["--check", "--root", str(tree)]) == 1


def _experiment(**kwargs):
    base = dict(id="x", path="axes/1-time/1.1.1-hard-window/experiments/x",
                item="axes/1-time/1.1.1-hard-window", item_title="", files=())
    return inv.Experiment(**{**base, **kwargs})


def test_not_reproducible_without_missing_fails_validation():
    """`not_reproducible` 却不写缺什么 ⇒ 校验失败（三态里唯一允许「留空」的只有别的两态）。"""
    exp = _experiment(verdict="not_reproducible", basis=["缺脚本"], missing=[])
    problems = inv.validate([exp], inv.ROOT)
    assert any("没写明缺什么" in p for p in problems), problems

    ok = _experiment(verdict="not_reproducible", basis=["缺脚本"], missing=["入口脚本：仓内没有"])
    problems = inv.validate([ok], inv.ROOT)
    assert not any("没写明缺什么" in p for p in problems), problems


def test_live_inventory_states_what_each_unrebuildable_row_lacks():
    """现行清单的每条 `not_reproducible` 都必须给出非空的「缺什么」。"""
    exps = inv.collect()
    assert exps, "扫不到任何实验"
    bad = [e.path for e in exps
           if e.verdict == "not_reproducible" and not [m for m in e.missing if m.strip()]]
    assert bad == [], f"这些 not_reproducible 行没写缺什么：{bad}"


def test_render_is_up_to_date_with_disk():
    """生成物过期（实验目录变了但没重新生成）⇒ 失败。"""
    text = inv.render(inv.collect())
    assert text == inv.OUT.read_text(encoding="utf-8"), (
        "docs/repro.md 与磁盘上的实验目录不一致；改完请运行 python3 tools/inventory_runs.py")


def test_every_listed_path_exists():
    """清单里每条路径都必须真实存在（`validate` 的路径检查）。"""
    exps = inv.collect()
    assert inv.validate(exps) == []
