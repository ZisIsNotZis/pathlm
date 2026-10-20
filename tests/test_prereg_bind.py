"""预登记绑定（`tools/prereg_bind.py`）的红线，每条都构造出会失败的反例。

1. **缺绑定必须抓红**：有 `results.json` 的实验没有 `prereg_binding.json` ⇒ `check()` 报错。
2. **契约在绑定后被改过必须抓红**：改契约一个字符 ⇒ 报出差异文件（不是沉默通过）。
3. **定序必须真的早**：`bound_at` 晚于读数产物时间 ⇒ 抓红；早于 ⇒ 通过（反向对照，
   否则上一条的红可能只是「任何绑定都红」）。
4. **绑定前已有的读数不算定序证据**：`readings_existing_at_bind` 记着的那份读数若逐字节未变
   ⇒ 只计数（`stats["predate"]`），不冒充「契约先于读数」。
5. **unbound 是显式缺失**：`bound_at="unbound"` 只计数、不判失败，且实验名要进汇总行。
6. **仓内定序必须由两个提交给出**：绑定与读数落在同一个提交里 ⇒ 抓红（「契约先于读数」仍不可判）；
   分成两个提交（绑定在前）⇒ 通过。这条是这一类缺口的正面判据，所以它在真 git 仓库里跑。
7. **`bound_at` 必须落在绑定文件那次提交的窗内**：事后补一条带旧 `bound_at` 的绑定（文件 mtime
   与提交先后都对得上）⇒ 抓红；落在窗内 ⇒ 通过。
8. **回填只收历史产物**：读数入库提交早于工具引入提交 ⇒ 写 `unbound`；晚于它、或根本没入库
   因而无从定序 ⇒ 拒绝回填并非零退出（否则本轮新实验能被洗成「历史产物」躲过定序）。
9. **训练器的门**：`owner_for_output()` 把 `axes/**` 下的 out-root 归到含 `README.md` 的那一层实验，
   `.scratch/` / `/tmp/` 下不归属任何实验（共享训练器 `train_depth_ar.py` 靠它定门）。

临时目录里造一棵假 `axes/` 树并 monkeypatch 模块的 `ROOT` / `AXES`，真仓库一个文件都不动。
前五条用例里那两个文件未入库 ⇒ 仓内定序记为「未提交」，不参与判定。
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import prereg_bind  # noqa: E402

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="仓内定序需要 git")


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


def test_unbound_is_counted_not_failed_and_names_are_listed(tree):
    """历史产物（bound_at="unbound"）只计数、不判失败，但实验名要出现在汇总行里。"""
    _write_results(tree)
    _bind(tree, prereg_bind.UNBOUND)
    other = tree.parent / "9002-second"
    other.mkdir()
    (other / "README.md").write_text("# 契约\n", encoding="utf-8")
    _write_results(other)
    prereg_bind.write_record(other, {"experiment": prereg_bind.rel(other),
                                     "bound_at": prereg_bind.UNBOUND,
                                     "unbound_reason": prereg_bind.UNBOUND_REASON,
                                     "tool": prereg_bind.TOOL})
    errors, stats = prereg_bind.check()
    assert errors == [] and stats["unbound"] == 2, (errors, stats)
    line = prereg_bind.summary_line(stats)
    assert prereg_bind.rel(tree) in line and prereg_bind.rel(other) in line, line


def _git(repo, *args, date=None):
    """在假仓库里跑一条 git 命令（提交时间可控，才能让「先后」是构造出来的而不是碰巧）。"""
    env = dict(os.environ)
    if date:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = date
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@example.com",
                    "-c", "user.name=t", *args],
                   check=True, env=env, capture_output=True)


def _commit(repo, message, date):
    _git(repo, "add", "-A", "-f")
    _git(repo, "commit", "-q", "-m", message, date=date)


def _commit_mechanism(repo, date):
    """在假仓库里入库「引入本工具的提交」（`--backfill-unbound` 的时间原点）。"""
    tools = repo / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "prereg_bind.py").write_text("# 假工具（只用来占定时间原点）\n", encoding="utf-8")
    _commit(repo, "引入预登记工具", date)


@requires_git
def test_same_commit_gives_no_in_repo_ordering(tree):
    """绑定与读数落在同一个提交 ⇒ 仓内仍无独立定序证据 ⇒ 抓红；分成两个提交 ⇒ 通过。"""
    repo = prereg_bind.ROOT                        # 仓库根（= fixture monkeypatch 的那一层）
    _write_results(tree)
    _bind(tree, "2021-01-01T00:00:00+00:00")        # 与下面那次提交同一时刻（落在容差窗内）
    _git(repo, "init", "-q")
    _commit(repo, "契约与读数同一个提交", "2021-01-01T00:00:00+00:00")

    errors, stats = prereg_bind.check()
    assert stats["git_same_commit"] == 1, stats
    assert any("没有先后" in e for e in errors), errors

    _write_results(tree, '{"bpc": 1.4}')            # 读数重跑，另起一个提交
    _commit(repo, "读数后到", "2021-02-01T00:00:00+00:00")

    errors, stats = prereg_bind.check()
    assert errors == [] and stats["git_ordered"] == 1, (errors, stats)


@requires_git
def test_bound_at_must_fall_in_the_window_of_its_own_commit(tree):
    """事后补一条带旧 `bound_at` 的绑定（mtime 与提交先后都对得上）⇒ 抓红；落在窗内 ⇒ 通过。

    这就是剩下的那条绕过路：先跑读数 → 再补一个 bound_at 写得更早的绑定并提交 → 再重跑读数。
    文件 mtime 与两个提交的先后都合法，只有 bound_at 与它自己那次提交相距很远。
    """
    repo = prereg_bind.ROOT
    results = _write_results(tree)
    _bind(tree, "2021-01-01T00:00:00+00:00")                  # 回填的旧时间
    _git(repo, "init", "-q")
    _commit(repo, "补一条带旧 bound_at 的绑定", "2021-06-01T00:00:00+00:00")
    _write_results(tree, '{"bpc": 1.4}')                  # 读数重跑，另起一个更晚的提交
    os.utime(results, (1_638_316_800, 1_638_316_800))         # 2021-12-01：读数晚于 bound_at
    _commit(repo, "读数后到", "2021-12-01T00:00:00+00:00")

    errors, stats = prereg_bind.check()
    assert stats["git_ordered"] == 1, stats          # 仓内先后本身是「合法」的
    assert any("窗内" in e for e in errors), errors

    _bind(tree, "2021-06-02T00:00:00+00:00")                  # 落在自己那次提交的窗内
    _commit(repo, "重绑到当下的时间", "2021-06-02T00:00:00+00:00")
    assert prereg_bind.check()[0] == []


@requires_git
def test_backfill_refuses_readings_not_older_than_the_mechanism(tree):
    """回填只收「读数入库早于工具引入」的历史产物；晚于它的拒绝并非零退出。"""
    repo = prereg_bind.ROOT
    old = tree
    new = tree.parent / "9002-new"
    new.mkdir()
    (new / "README.md").write_text("# 契约\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _write_results(old)
    _commit(repo, "旧读数", "2021-01-01T00:00:00+00:00")
    _commit_mechanism(repo, "2021-06-01T00:00:00+00:00")
    _write_results(new)
    _commit(repo, "本轮新实验的读数", "2021-12-01T00:00:00+00:00")

    assert prereg_bind.backfill_unbound() == 1
    assert (old / "prereg_binding.json").exists(), "早于机制引入的历史产物应该被回填"
    assert not (new / "prereg_binding.json").exists(), "晚于机制引入的读数不得被洗成历史产物"
    assert json.loads((old / "prereg_binding.json").read_text(encoding="utf-8"))["bound_at"] \
        == prereg_bind.UNBOUND, "回填只写显式缺失，不填一个看起来真的时间"

    bound = _bind(old, "2021-01-01T00:00:00+00:00")        # 已有绑定不被回填覆盖
    prereg_bind.backfill_unbound()
    kept = json.loads((old / "prereg_binding.json").read_text(encoding="utf-8"))
    assert kept["bound_at"] == bound["bound_at"], "回填不许覆盖已有绑定"


def test_backfill_refuses_readings_that_cannot_be_ordered(tree):
    """读数没入库 / 取不到工具引入时间 ⇒ 无从定序 ⇒ 拒绝回填（先回填再补读数就是绕过的路）。"""
    _write_results(tree)
    assert prereg_bind.backfill_unbound() == 1        # 假树不是 git 仓库：引入时间未知
    assert not (tree / "prereg_binding.json").exists(), "无从定序时不许写 unbound"


def test_owner_for_output_only_claims_axes_experiments(tree):
    """`axes/**` 的 out-root 归到含 README.md 的那一层；`.scratch/` 与 `/tmp/` 不归任何实验。"""
    assert prereg_bind.owner_for_output(tree / "runs" / "arm") == tree
    assert prereg_bind.owner_for_output(tree.parents[3] / ".scratch" / "x") is None
    assert prereg_bind.owner_for_output("/tmp/pathlm-not-an-experiment") is None
