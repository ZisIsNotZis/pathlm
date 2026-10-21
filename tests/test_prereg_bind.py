"""预登记绑定（`tools/prereg_bind.py`）的红线，每条都构造出会失败的反例。

1. **缺绑定必须抓红**：有读数（`results.json` 或臂级 `results_*.json`）的实验没有
   `prereg_binding.json` ⇒ `check()` 报错。
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
10. **臂级读数也在检查范围内（`results*.json`，不只 `results.json`）**：①一份臂级读数的代码哈希
    与绑定值不同 ⇒ 红并点名该文件；②一份臂级读数的产物时间早于 `bound_at` ⇒ 红并点名该文件；
    ③一份读数没记任何代码哈希 ⇒ 红（它无法核对产出它的代码）；④全部一致 ⇒ 绿（反向对照）。

临时目录里造一棵假 `axes/` 树并 monkeypatch 模块的 `ROOT` / `AXES`，真仓库一个文件都不动。
前五条用例里那两个文件未入库 ⇒ 仓内定序记为「未提交」，不参与判定。
"""

import hashlib
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
#: 假实验入口：读数里记的就是它的 sha256（绑定值与之相同才算「同一份代码」）。
RUN_PY = "print('arm')\n"


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """一棵假的研究树：一个实验目录（README.md + run.py 契约）+ 可选的结果与绑定。"""
    root = tmp_path / "repo"
    exp = root / "axes" / "1-time" / "1.1-x" / "experiments" / "9001-arm"
    exp.mkdir(parents=True)
    (exp / "README.md").write_text("# 契约\n判据线写死在这里。\n", encoding="utf-8")
    (exp / "run.py").write_text(RUN_PY, encoding="utf-8")
    monkeypatch.setattr(prereg_bind, "ROOT", root)
    monkeypatch.setattr(prereg_bind, "AXES", root / "axes")
    return exp


def _bind(exp, bound_at, contract=None, prior=None):
    record = {
        "experiment": "axes/1-time/1.1-x/experiments/9001-arm",
        "bound_at": bound_at,
        "contract_files": contract if contract is not None else
        {"README.md": prereg_bind.sha256(exp / "README.md"),
         "run.py": prereg_bind.sha256(exp / "run.py")},
        "tool": "tools/prereg_bind.py",
    }
    if prior is not None:
        record["readings_existing_at_bind"] = prior
    (exp / "prereg_binding.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return record


def _reading(exp, bpc=1.5, code_hash=None):
    """一份形态正常的读数：记着产出它的 `run.py` 哈希（= 契约绑的那份）。"""
    if code_hash is None:
        run_py = exp / "run.py"
        code_hash = (prereg_bind.sha256(run_py) if run_py.exists()
                     else hashlib.sha256(RUN_PY.encode()).hexdigest())
    return json.dumps({"bpc": bpc, "code_hash_run_py": code_hash})


def _write_results(exp, name="results.json", body=None):
    path = exp / name
    path.write_text(body if body is not None else _reading(exp), encoding="utf-8")
    return path


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
    assert any("不早于该读数的产物时间" in e for e in errors), errors

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

    _write_results(tree, body=_reading(tree, bpc=1.4))          # 重跑过：这份读数不再是绑定时那份
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

    _write_results(tree, body=_reading(tree, bpc=1.4))            # 读数重跑，另起一个提交
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
    _write_results(tree, body=_reading(tree, bpc=1.4))                  # 读数重跑，另起一个更晚的提交
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


def test_arm_reading_with_foreign_code_hash_fails(tree):
    """臂级读数（`results_<臂>_<种子>.json`）的代码哈希与绑定值不同 ⇒ 红，并点名该文件。

    这就是「只看 `results.json`」漏掉的那一类：聚合文件由绑定的 `run.py` 产出，而臂级读数由
    另一个版本的 `run.py` 产出；不检查臂级文件时 `--check` 全绿。
    """
    aggregate = _write_results(tree)
    arm = _write_results(tree, name="results_X_s0.json",
                         body=_reading(tree, code_hash="f" * 64))
    _bind(tree, "2020-01-01T00:00:00+00:00")
    for path in (aggregate, arm):
        os.utime(path, (1_700_000_000, 1_700_000_000))

    errors, stats = prereg_bind.check()
    assert stats["mismatch"] == 1, (errors, stats)
    hits = [e for e in errors if "不符" in e]
    assert len(hits) == 1 and prereg_bind.rel(arm) in hits[0], errors
    assert prereg_bind.rel(aggregate) not in hits[0], errors


def test_arm_reading_predating_the_binding_fails(tree):
    """臂级读数的产物时间早于 `bound_at` ⇒ 红，并点名该文件（聚合文件本身是合规的对照）。"""
    aggregate = _write_results(tree)
    arm = _write_results(tree, name="results_L50_s1.json")
    _bind(tree, "2021-01-01T00:00:00+00:00")
    os.utime(aggregate, (1_700_000_000, 1_700_000_000))       # 2023-11-14：晚于绑定
    os.utime(arm, (1_600_000_000, 1_600_000_000))             # 2020-09-13：早于绑定

    errors, stats = prereg_bind.check()
    hits = [e for e in errors if "不早于该读数的产物时间" in e]
    assert len(hits) == 1 and prereg_bind.rel(arm) in hits[0], errors
    assert stats["ordered"] == 1, stats


def test_reading_without_code_hash_fails(tree):
    """读数一份代码哈希都没记 ⇒ 红（产出它的代码无从核对，这本身就是要修的问题）。"""
    bare = _write_results(tree, name="results_K25_s0.json", body='{"bpc": 1.5}')
    _bind(tree, "2020-01-01T00:00:00+00:00")
    os.utime(bare, (1_700_000_000, 1_700_000_000))

    errors, stats = prereg_bind.check()
    assert stats["unhashed"] == 1, (errors, stats)
    assert any("未记录代码哈希" in e and prereg_bind.rel(bare) in e for e in errors), errors


def test_all_readings_agreeing_is_green(tree):
    """正向对照：聚合文件与臂级读数的哈希都等于绑定值、产物时间都晚于 `bound_at` ⇒ 绿。"""
    aggregate = _write_results(tree)
    arms = [_write_results(tree, name=f"results_{arm}_s0.json") for arm in ("K50", "L50")]
    _bind(tree, "2020-01-01T00:00:00+00:00")
    for path in (aggregate, *arms):
        os.utime(path, (1_700_000_000, 1_700_000_000))

    errors, stats = prereg_bind.check()
    assert errors == [], errors
    assert stats["readings"] == 3 and stats["ordered"] == 3, stats


def test_code_hash_field_names_are_adapted(tree):
    """`code_hashes` / `code.<文件>` / `code_hashes_by_arm` 记的哈希都读得出来（字段名各不相同）。"""
    _write_results(tree, name="results_A_s0.json", body=json.dumps(
        {"code_hashes": {"run.py": "a" * 64}}))
    _write_results(tree, name="results_B_s0.json", body=json.dumps(
        {"code": {"run_py_sha256": "b" * 64}}))
    _write_results(tree, name="results_C_s0.json", body=json.dumps(
        {"code_hashes_by_arm": {"K50": "c" * 64}}))
    _bind(tree, "2020-01-01T00:00:00+00:00")
    for path in tree.glob("results_*.json"):
        os.utime(path, (1_700_000_000, 1_700_000_000))

    errors, stats = prereg_bind.check()
    # 三份都读出了哈希（各自与绑定值不符），没有一份落进「未记录代码哈希」。
    assert stats["unhashed"] == 0, (errors, stats)
    assert stats["mismatch"] == 3, (errors, stats)

    # 记了哈希但指不到本仓的文件：那是「无从核对」，不该报成「没记代码哈希」。
    _write_results(tree, name="results_D_s0.json",
                   body=json.dumps({"code_hashes": {"no-such-file.py": "d" * 64}}))
    os.utime(tree / "results_D_s0.json", (1_700_000_000, 1_700_000_000))
    errors, stats = prereg_bind.check()
    hits = [e for e in errors if prereg_bind.rel(tree / "results_D_s0.json") in e]
    assert hits and "指不到本仓的任何一个文件" in hits[0], errors
    assert stats["unhashed"] == 0, (errors, stats)


def _declare_debt(rows):
    """在假仓库里写已声明存量基线（`ROOT` 已被 fixture 指到临时树）。"""
    path = prereg_bind.ROOT / prereg_bind.BASELINE
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ["# 已声明存量（测试用）"]
    body += ["\t".join(row) for row in rows]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def test_declared_debt_forgives_only_the_signature_it_declared(tree):
    """基线内且签名未变 ⇒ 只计「已声明存量」；文件变了而问题仍在 ⇒ 红（不许签一次长期免检）。"""
    arm = _write_results(tree, name="results_X_s0.json", body='{"bpc": 1.5}')
    _bind(tree, "2020-01-01T00:00:00+00:00")
    os.utime(arm, (1_700_000_000, 1_700_000_000))
    _declare_debt([(prereg_bind.rel(arm), "no-hash", prereg_bind.file_signature(arm),
                    "1.1.1", "冒烟读数不记代码哈希")])

    errors, stats = prereg_bind.check()
    assert errors == [] and stats["declared"] == 1, (errors, stats)
    assert stats["declared_stale"] == [], stats["declared_stale"]

    (arm).write_text(json.dumps({"bpc": 1.6}), encoding="utf-8")     # 读数变了，问题（无哈希）仍在
    os.utime(arm, (1_700_000_100, 1_700_000_100))
    errors, stats = prereg_bind.check()
    assert any("基线条目" in e and "文件已变而问题仍在" in e for e in errors), errors

    (arm).write_text(_reading(tree, bpc=1.6), encoding="utf-8")      # 修好：记上绑定的哈希
    errors, stats = prereg_bind.check()
    assert errors == [] and stats["declared"] == 0, (errors, stats)
    assert (prereg_bind.rel(arm), "no-hash") in stats["declared_stale"], stats["declared_stale"]


def test_violation_outside_the_declared_debt_is_red(tree):
    """基线只盖住它点名的那一条：同实验里新冒出来的违规照样红。"""
    declared = _write_results(tree, name="results_X_s0.json", body='{"bpc": 1.5}')
    fresh = _write_results(tree, name="results_Y_s0.json", body='{"bpc": 1.5}')
    _bind(tree, "2020-01-01T00:00:00+00:00")
    for path in (declared, fresh):
        os.utime(path, (1_700_000_000, 1_700_000_000))
    _declare_debt([(prereg_bind.rel(declared), "no-hash",
                    prereg_bind.file_signature(declared), "1.1.1", "冒烟读数不记代码哈希")])

    errors, stats = prereg_bind.check()
    assert stats["declared"] == 1, (errors, stats)
    hits = [e for e in errors if "未记录代码哈希" in e]
    assert len(hits) == 1 and prereg_bind.rel(fresh) in hits[0], errors


#: 已声明存量的条数上限。基线**只减不增**：新增一条必须由用户或元级批准，
#: 所以这个数字只能跟着「删掉了几条」往下走，不跟着新违规往上走。
PREREG_DEBT_LIMIT = 81


def test_declared_debt_only_shrinks():
    """真仓库的基线：条数不超上限，且每一条都有 owner 与理由（没有理由的免检就是漏洞）。"""
    rows = prereg_bind.load_baseline()
    assert len(rows) <= PREREG_DEBT_LIMIT, (
        f"{prereg_bind.BASELINE} 有 {len(rows)} 条，超过上限 {PREREG_DEBT_LIMIT}："
        "基线只减不增，新增条目要先请用户或元级批准")
    for (path, kind), row in rows.items():
        assert row["owner"] and row["reason"], (path, kind, row)


def test_debt_row_without_reason_is_refused(tree):
    """基线条目缺 owner / 原因 ⇒ 直接报错（一条没有理由的免检就是没人认领的漏洞）。"""
    arm = _write_results(tree, name="results_X_s0.json", body='{"bpc": 1.5}')
    _declare_debt([(prereg_bind.rel(arm), "no-hash", prereg_bind.file_signature(arm), "", "")])
    with pytest.raises(SystemExit):
        prereg_bind.load_baseline()
