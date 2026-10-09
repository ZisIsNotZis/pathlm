"""2635-recursive-carrier-13m · 停摆门与配置不变量的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的工程仪器 `stall_gate.py`（契约 §4a 的判据①）与 `config.json`：

① 停摆门**真的会判停摆**：子进程写一步日志后不再推进 ⇒ `status=stalled`、返回码 3、
   诊断字段（触发步 / CPU 时间 / 显存查询 / 日志尾）齐备；
② 停摆门**不会把正常结束判成停摆**：子进程写完就退出 ⇒ `status=finished`、返回码 0；
③ 助手函数：`_last_step` / `_tail` / `_cpu_seconds` 的边界行为；
④ 配置不变量 + **能力面限制**：引擎硬互斥 `attn_linear ⊕ train_window`
   （「窗口 + 线性载体」不可测，README §2），`attn_linear` 不新增参数；
⑤ `results.json` 存在时（跑完之后）：判据块齐备、停摆时不给质量结论的口径。
"""

import json
import os
import subprocess
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2635-recursive-carrier-13m")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)

from pathlm.depth_ar import DepthARConfig, DepthARModel            # noqa: E402
import stall_gate as sg                                            # noqa: E402

CFG = json.load(open(os.path.join(EXP, "config.json")))
GATE = os.path.join(EXP, "stall_gate.py")


def _run_gate(tmp_path, body: str, stall_secs: float = 2.0, kill_secs: float = 3.0):
    log = tmp_path / "train" / "arm" / "train_log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    report = tmp_path / "stall_report.json"
    script = (f"import json,os,time\n"
              f"p = r'{log}'\n"
              f"os.makedirs(os.path.dirname(p), exist_ok=True)\n"
              f"{body}")
    rc = subprocess.run([sys.executable, GATE, "--train-log", str(log),
                         "--report", str(report), "--stall-secs", str(stall_secs),
                         "--kill-secs", str(kill_secs),
                         "--poll-secs", "0.5", "--dump-stack-secs", "1",
                         "--", sys.executable, "-c", script],
                        capture_output=True, text=True, timeout=120).returncode
    return rc, json.load(open(report)) if report.exists() else None


# ---------------- ① 停摆必须被抓到 ----------------

def test_stall_gate_detects_stall(tmp_path):
    """真停摆：日志不推进 + 进程不退 ⇒ 超过 kill-secs 后被终止、返回码 3。"""
    body = ("open(p,'a').write(json.dumps({'step': 1}) + chr(10))\n"
            "time.sleep(60)\n")
    rc, rep = _run_gate(tmp_path, body, stall_secs=1.0, kill_secs=3.0)
    assert rc == 3
    assert rep["status"] == "killed_after_no_log_progress"
    assert rep["trigger_step"] == 1
    assert rep["stall_wall_s"] >= 3.0
    assert rep["cpu_seconds"] is not None
    assert isinstance(rep["log_tail"], list) and rep["log_tail"]
    # 两级信号：先记 no_log_progress 事件（非致命），再终止
    assert rep["no_log_progress_events"][0]["status"] == "no_log_progress"
    assert rep["pathlm_dump_stacks"]


def test_stall_gate_does_not_kill_log_frozen_but_finishing_run(tmp_path):
    """**实测教训钉住**：日志停写 ≠ 进程停摆。

    子进程写一步后长时间不写日志（模拟「数值发散后梯度门把后续步都跳过，日志不再写」），
    但在 kill-secs 之前自己退出 ⇒ 必须记 `finished`，**不得**误杀。
    """
    body = ("open(p,'a').write(json.dumps({'step': 1}) + chr(10))\n"
            "time.sleep(6)\n"
            "open(p,'a').write(json.dumps({'step': 2}) + chr(10))\n")
    rc, rep = _run_gate(tmp_path, body, stall_secs=1.0, kill_secs=30.0)
    assert rc == 0
    assert rep["status"] == "finished"
    assert rep["last_step"] == 2
    assert rep["n_no_log_progress_events"] >= 1        # 记过事件，但没有杀


def test_stall_gate_passes_normal_exit(tmp_path):
    body = ("open(p,'a').write(json.dumps({'step': 1}) + chr(10))\n"
            "open(p,'a').write(json.dumps({'step': 2}) + chr(10))\n")
    rc, rep = _run_gate(tmp_path, body, stall_secs=30.0, kill_secs=60.0)
    assert rc == 0
    assert rep["status"] == "finished"
    assert rep["last_step"] == 2
    assert rep["no_log_progress_events"] == []


# ---------------- ③ 助手函数 ----------------

def test_last_step_reads_tail_step(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"step": 1}\n\n{"step": 7}\n')
    step, mtime = sg._last_step(str(p))
    assert step == 7 and mtime is not None
    assert sg._last_step(str(tmp_path / "missing.jsonl")) == (None, None)


def test_tail_returns_last_lines(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("a\nb\nc\n")
    assert sg._tail(str(p), 2) == ["b", "c"]
    assert sg._tail(str(tmp_path / "missing")) == []


def test_cpu_seconds_of_current_process_is_positive():
    v = sg._cpu_seconds(os.getpid())
    assert v is not None and v >= 0.0
    assert sg._cpu_seconds(10 ** 9) is None


# ---------------- ④ 配置不变量与能力面限制 ----------------

def test_config_two_arms_same_steps_no_window():
    assert [a["name"] for a in CFG["arms"]] == ["stock", "linear"]
    assert [a["attn_linear"] for a in CFG["arms"]] == [False, True]
    assert CFG["model"]["train_window"] == 0 and CFG["model"]["train_anchors"] == 0
    assert CFG["visibility"]["train_window"] == 0
    assert CFG["steps"] == 3000
    assert CFG["train_recipe"]["tokens_per_arm"] == 3000 * 32 * 256
    assert CFG["stall_gate"]["stall_secs"] == 180


def test_engine_rejects_linear_plus_window():
    """能力面限制（SCHEMA §22.8）：引擎硬互斥 `attn_linear ⊕ train_window`

    校验在 `DepthARModel.__init__`（不是 config 的 `__post_init__`）⇒ 必须建模型才报错。
    """
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=2, seq_len=64, mlp_mult=2,
                        dense=True, topology="sequential", attn_linear=True,
                        train_window=16)
    with pytest.raises(ValueError, match="train_window"):
        DepthARModel(cfg, 206)


def test_attn_linear_adds_no_parameters():
    def build(on):
        torch.manual_seed(0)
        return DepthARModel(DepthARConfig(d_model=32, n_layers=2, n_heads=2, seq_len=64,
                                          mlp_mult=2, dense=True,
                                          topology="sequential", attn_linear=on), 206)
    n0 = sum(p.numel() for p in build(False).parameters())
    n1 = sum(p.numel() for p in build(True).parameters())
    assert n0 == n1


# ---------------- ⑤ results.json（跑完之后） ----------------

RESULTS = os.path.join(EXP, "results.json")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="训练/评测尚未完成")
def test_results_blocks_and_stall_caliber():
    r = json.load(open(RESULTS))
    c = r["criteria"]
    for k in ("c1_runnable", "c2_quality_reported", "c3_liveness_alive",
              "c4_battery_reported", "c5_no_extrapolation"):
        assert k in c, k
    assert r["selfchecks"]["n_params_equal"] is True
    assert "bpc" in r["quality_leg"]
    if not c["c1_runnable"]:
        # 停摆 ⇒ 不给质量结论：必须留下停摆报告
        assert r["stall_report"] and r["stall_report"]["status"] == "stalled"
    else:
        assert c["c2_quality_reported"] is True
