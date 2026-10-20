"""`tools/supervise.sh`：有界重试、连续失败熔断、熔断后不再取锁（形态契约）。

钉住五件事，每件都可失败（用临时目录 + 假命令驱动，不写仓库、不占 GPU、不取真锁）：
1. 连续失败触达 `--fail-max` 即熔断：日志出现 `[sup] FUSED reason=consecutive`、退出码非零；
2. 熔断后**不再取锁、不再重试**：日志里的 `[sup] attempt N start` 行数不再增长；
3. `--attempts-max` 是另一条独立上限：连续失败远未触达时也按它熔断（`reason=attempts`）；
4. 锁被活人占着只算一次尝试、**不计入**连续失败（`lock-busy`，命令不启动），
   且命令成功一次就收口：`[sup] DONE rc=0` 且退出 0。
5. `--command-owns-lock`（锁归命令自己管）：supervisor 不探测、不建、不删锁目录，
   锁被占也照旧启动命令；上限与熔断照旧生效。
"""

import os
import re
import subprocess
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SUP = os.path.join(ROOT, "tools", "supervise.sh")
ATTEMPT = re.compile(r"^\[sup\] attempt \d+ start", re.M)


def run_sup(tmp_path, args, cmd, lock_dir=None):
    """跑一次 supervise.sh，返回 (CompletedProcess, 日志文本)。锁目录一律指向临时目录。"""
    log = os.path.join(str(tmp_path), "sup.log")
    env = dict(os.environ)
    env["PATHLM_GPU_LOCK"] = str(lock_dir or os.path.join(str(tmp_path), "no-such-lock"))
    p = subprocess.run(["bash", SUP, *args, "--log", log, "--", *cmd],
                       env=env, capture_output=True, text=True, timeout=120)
    text = open(log).read() if os.path.exists(log) else ""
    return p, log, text


def test_consecutive_failure_fuse_stops_and_records_banner(tmp_path):
    p, log, text = run_sup(tmp_path,
                           ["--attempts-max", "10", "--fail-max", "2",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", "exit 7"])
    assert p.returncode != 0, text
    assert "[sup] FUSED reason=consecutive" in text, text
    assert len(ATTEMPT.findall(text)) == 2, text

    # 熔断之后不得再取锁、不得再重试：等几个 gap 再看，attempt 行数必须纹丝不动
    time.sleep(3)
    assert len(ATTEMPT.findall(open(log).read())) == 2, "熔断后仍在重试"


def test_attempts_max_is_its_own_bound(tmp_path):
    p, log, text = run_sup(tmp_path,
                           ["--attempts-max", "3", "--fail-max", "99",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", "exit 9"])
    assert p.returncode != 0, text
    assert "[sup] FUSED reason=attempts" in text, text
    assert len(ATTEMPT.findall(text)) == 3, text


def test_lock_busy_counts_attempt_but_not_consecutive_failure(tmp_path):
    lock_dir = os.path.join(str(tmp_path), "lock")
    os.makedirs(lock_dir)
    with open(os.path.join(lock_dir, "holder_pid"), "w") as f:
        f.write(str(os.getpid()))            # 活着的持有者 ⇒ 锁不空闲
    marker = os.path.join(str(tmp_path), "ran.txt")
    p, log, text = run_sup(tmp_path,
                           ["--attempts-max", "3", "--fail-max", "1",
                            "--lock-wait-s", "0", "--gap-s", "0"],
                           ["bash", "-c", f"touch {marker}"],
                           lock_dir=lock_dir)
    assert p.returncode != 0, text
    # 连续失败上限只有 1，但锁被占用不计入连续失败 ⇒ 熔断只能由尝试上限触发
    assert "[sup] FUSED reason=attempts" in text, text
    assert text.count("lock-busy") == 3, text
    assert "consecutive=1" not in text, text
    assert not os.path.exists(marker), "锁被占用时命令不得启动"


def test_success_exits_zero_with_done_banner(tmp_path):
    p, log, text = run_sup(tmp_path,
                           ["--attempts-max", "5", "--fail-max", "2",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", "echo ok"])
    assert p.returncode == 0, text
    assert "[sup] DONE rc=0" in text, text
    assert len(ATTEMPT.findall(text)) == 1, text


def lock_state(lock_dir):
    """锁目录的可观察状态：不存在返回 None，存在则返回 {条目名: 内容}（不读 mtime，避开时钟噪声）。"""
    if not os.path.exists(lock_dir):
        return None
    state = {}
    for name in sorted(os.listdir(lock_dir)):
        path = os.path.join(lock_dir, name)
        state[name] = open(path).read() if os.path.isfile(path) else "<dir>"
    return state


def test_command_owns_lock_never_creates_or_touches_lock(tmp_path):
    """`--command-owns-lock`：锁归命令，supervisor 不探测、不建、不删锁目录（即便命令成功）。"""
    lock_dir = os.path.join(str(tmp_path), "lock")        # 故意不存在：若 supervisor 探测式取锁会建它
    marker = os.path.join(str(tmp_path), "ran.txt")
    p, log, text = run_sup(tmp_path,
                           ["--command-owns-lock", "--attempts-max", "5", "--fail-max", "2",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", f"touch {marker}"],
                           lock_dir=lock_dir)
    assert p.returncode == 0, text
    assert "[sup] DONE rc=0" in text, text
    assert os.path.exists(marker), "命令必须启动"
    assert lock_state(lock_dir) is None, "--command-owns-lock 下不得创建锁目录"
    assert "lock-busy" not in text, text
    assert "command-owns-lock=1" in text, text
    assert "holder=cmd-owned" in text, text


def test_command_owns_lock_does_not_probe_a_held_lock(tmp_path):
    """锁被活人占着也照旧启动命令，且锁目录内容一字不动（不探测 = 不写 lock-busy、不改锁）。"""
    lock_dir = os.path.join(str(tmp_path), "lock")
    os.makedirs(lock_dir)
    with open(os.path.join(lock_dir, "holder_pid"), "w") as f:
        f.write(str(os.getpid()))                          # 活着的持有者
    with open(os.path.join(lock_dir, "name"), "w") as f:
        f.write("other-experiment")
    before = lock_state(lock_dir)
    marker = os.path.join(str(tmp_path), "ran.txt")
    p, log, text = run_sup(tmp_path,
                           ["--command-owns-lock", "--attempts-max", "5", "--fail-max", "2",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", f"touch {marker}"],
                           lock_dir=lock_dir)
    assert p.returncode == 0, text
    assert os.path.exists(marker), "不探测锁 ⇒ 锁被占着命令也要启动"
    assert lock_state(lock_dir) == before, "锁目录被改动了"
    assert "lock-busy" not in text, text


def test_command_owns_lock_still_fuses_at_limits(tmp_path):
    """该模式保留上限与熔断：连续失败触达 `--fail-max` 即 FUSED 退 3，锁目录仍不存在。"""
    lock_dir = os.path.join(str(tmp_path), "lock")
    p, log, text = run_sup(tmp_path,
                           ["--command-owns-lock", "--attempts-max", "10", "--fail-max", "2",
                            "--lock-wait-s", "0", "--gap-s", "1"],
                           ["bash", "-c", "exit 7"],
                           lock_dir=lock_dir)
    assert p.returncode == 3, text
    assert "[sup] FUSED reason=consecutive" in text, text
    assert len(ATTEMPT.findall(text)) == 2, text
    assert lock_state(lock_dir) is None, "--command-owns-lock 下不得创建锁目录"

