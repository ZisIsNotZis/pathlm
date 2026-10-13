#!/usr/bin/env python3
"""2628-eng2-carry-ragged · ENG2「carry 支持非 dense ragged 路径」验证入口。

本实验**不追求性能**，只追求**可测性**（SCHEMA §23 第 6 问的工程面）：证明
「写入通道 × 非 dense（随机退出 + ragged / 混合填充 / proj-fill）」这些组合从
「引擎里不存在（硬校验 + 块拒绝 kv_from/gate_vec）」变成「可组合、各自活着、
且写入时机有明确定义」。

CPU 侧（缺省）：对 4 个组合各建一个非 dense 模型（d=32/3L/4H，seq 12），量
每个组合的活性量（块类 / 读出占比 / 遥测是否进 aux / 退出模式是否生效 /
proj-fill adapter 是否被消费），并核对缺省路径不含任何通道结构。

用法：
  python3 run.py                 # CPU 自检 + 摄取已有 smoke / pytest / mutation 产物
  python3 run.py --smoke         # 额外跑 20 步 GPU 冒烟（先取 /tmp/pathlm_gpu.lock）

产出：results.json（原始结果）；冒烟直出 smoke/master/。
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel, sample_exit_depths  # noqa: E402

DEV = "cpu"
SMOKE_NAME = "master"
D_MODEL = 32
N_LAYERS = 3
N_HEADS = 4
SEQ = 12
#: 冒烟：ENG2 前**不存在**的组合（非 dense 随机退出 + fill + proj-fill + 写入通道），
#: M1 引擎（d=256/12L/8H）。数据用合成长程任务（M1 线主任务，SCHEMA §19）；
#: 非 dense + 合成任务的 enwik8 电池由 `train_depth_ar.main` 的 synth 分支跳过。
SMOKE_CMD = [
    "python3", "train_depth_ar.py", SMOKE_NAME,
    "--d", "256", "--layers", "12", "--heads", "8", "--mlp-mult", "6",
    "--seq-len", "256", "--steps", "20", "--batch-size", "8",
    "--seed", "0", "--exit-mode", "uniform",
    "--synth-task", "assoc-mqar", "--synth-pairs", "8", "--synth-queries", "8",
    "--synth-gap", "14",
    "--carry-write", "delta", "--carry-chunk", "64", "--proj-fill",
    "--out-root", os.path.join(HERE, "smoke"),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(**kw):
    """ENG2 的 CPU 测试位点：d=32 / 3 层 / 4 头 / dh=8，**非 dense**（RAP 形态）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ, dense=False, carry_rule="delta", **kw)
    return DepthARModel(cfg, vocab_size=10)


def _activate_carry(model, value=0.5):
    for blk in model.blocks:
        blk.carry_scale.data.fill_(value)


def _tokens():
    g = torch.Generator().manual_seed(1)          # 同一批对比 token：重跑逐位可复现
    return torch.randint(0, 10, (2, SEQ), generator=g)


def _fixed_depths():
    """固定混合退出模式（逐 batch 内容不同，防掩码广播侥幸）。"""
    return torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 0, 1, 3, 2],
                         [3, 3, 1, 2, 3, 0, 1, 3, 2, 1, 2, 3]])


def _logits(model, x, depths):
    with torch.no_grad():
        return model(x, exit_depths=depths)[1]["depth_logits"][-1].float()


def _frozen_ok(states, depths):
    """退出即冻结：d_i < k 的位置在 depth k 的状态必须等于其退出深度状态。"""
    L = states.__len__() - 1
    ok = True
    for b in range(depths.shape[0]):
        for i in range(depths.shape[1]):
            d = int(depths[b, i])
            for k in range(d + 1, L + 1):
                ok = ok and bool(torch.allclose(states[k][b, i],
                                                states[d][b, i], atol=1e-6))
    return ok


def combo_report(name, depths, **kw):
    """一个非 dense 组合的活性报告：通道活着 + 退出模式生效 + 遥测进 aux。"""
    exit_mode = kw.pop("_exit_mode", "fixed")
    x = _tokens()
    m = build(**kw)
    _activate_carry(m)
    rep = {"name": name, "block_class": type(m.blocks[0]).__name__,
           "carry_on": True, "fill_kv": bool(m.cfg.fill_kv),
           "proj_fill": bool(m.cfg.proj_fill),
           "exit_mode": exit_mode,
           "exit_min": int(depths.min()), "exit_max": int(depths.max()),
           "n_params": sum(p.numel() for p in m.parameters())}
    with torch.no_grad():
        _, aux = m(x, exit_depths=depths)
    rep["n_carry_layers_in_aux"] = len(aux.get("attn_carry", []))
    rep["carry_read_share"] = (round(float(aux["attn_carry"][0]["read_share"]), 6)
                               if aux.get("attn_carry") else None)
    rep["carry_scan_tele_present"] = hasattr(m, "_carry_tele")
    # 通道真的改前向：γ 置零 vs 激活，logits 必须不同
    with torch.no_grad():
        with_carry = _logits(m, x, depths)
        saved = [b.carry_scale.detach().clone() for b in m.blocks]
        for b in m.blocks:
            b.carry_scale.zero_()
        without = _logits(m, x, depths)
        for b, s in zip(m.blocks, saved):
            b.carry_scale.copy_(s)
    rep["carry_changes_logits"] = bool(not torch.equal(with_carry, without))
    rep["frozen_invariant_ok"] = _frozen_ok(aux["states"], depths)
    # 退出模式真的生效：与全深度前向不同（非退化点）
    full = torch.full_like(depths, N_LAYERS)
    rep["exit_changes_logits_vs_full"] = bool(
        not torch.equal(_logits(m, x, depths), _logits(m, x, full)))
    if m.cfg.proj_fill:
        # proj-fill：adapter 换方向 ⇒ logits 必变（否则 kv_from 未被消费）
        d = D_MODEL
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            base = _logits(m, x, depths).clone()
            for a in m.kv_adapters:
                a.weight.data.copy_(
                    torch.eye(d) + 0.2 * torch.randn(d, d, generator=gen))
            after = _logits(m, x, depths)
        rep["proj_fill_adapter_max_abs_logit_delta"] = round(
            float((base - after).abs().max()), 8)
    else:
        rep["proj_fill_adapter_max_abs_logit_delta"] = None
    return rep


def write_timing():
    """写入时机：记忆槽与退出模式**无关**（所有位置都写各自的 h₀）。"""
    m = build(fill_kv=False)
    _activate_carry(m)
    x = _tokens()
    mems = []
    for depths in (torch.full((2, SEQ), N_LAYERS, dtype=torch.long),
                   _fixed_depths(), torch.zeros(2, SEQ, dtype=torch.long)):
        with torch.no_grad():
            m(x, exit_depths=depths)
        mems.append(m._carry_box["mem"].clone())
    return {"slot_identical_full_vs_fixed": bool(torch.equal(mems[0], mems[1])),
            "slot_identical_full_vs_zero": bool(torch.equal(mems[0], mems[2])),
            "note": ("写入在 _input_states 对输入流 h₀ 扫描一次、发生在任何退出之前 "
                     "⇒ 同一 token 在任意退出模式下得到同一段记忆")}


def default_untouched():
    """缺省路径（carry_rule=None）不含任何通道结构：无 carry 模块/槽、无遥测。"""
    torch.manual_seed(0)
    m = DepthARModel(DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS,
                                   n_heads=N_HEADS, seq_len=SEQ), vocab_size=10)
    x = _tokens()
    depths = _fixed_depths()
    with torch.no_grad():
        _, aux = m(x, exit_depths=depths)
    return {"block_class": type(m.blocks[0]).__name__,
            "has_carry_module": hasattr(m, "carry"),
            "has_carry_box": hasattr(m, "_carry_box"),
            "attn_carry_in_aux": "attn_carry" in aux,
            "n_params": sum(p.numel() for p in m.parameters())}


def ingest_smoke():
    run_dir = os.path.join(HERE, "smoke", SMOKE_NAME)
    log = os.path.join(run_dir, "train_log.jsonl")
    res = os.path.join(run_dir, "results.json")
    if not os.path.exists(log):
        return None
    rows = [json.loads(ln) for ln in open(log) if ln.strip()]
    out = {"run_dir": os.path.relpath(run_dir, ROOT), "n_log_rows": len(rows),
           "final_step": rows[-1]["step"], "final_loss": rows[-1]["loss"],
           "wall_minutes": rows[-1]["min"],
           "attn_carry_step0": rows[0]["attn_carry"],
           "attn_carry_final": rows[-1]["attn_carry"]}
    if os.path.exists(res):
        r = json.load(open(res))
        out["params"] = r["params"]
        out["peak_gpu_mem_mb"] = r["peak_gpu_mem_mb"]
        out["wall_minutes"] = r["wall_minutes"]
        out["carry_liveness_alive"] = r["carry_liveness"]["alive"]
        out["carry_liveness"] = {k: v for k, v in r["carry_liveness"].items()
                                 if k in ("read_scale_absmean", "read_share",
                                          "state_fro", "beta_mean", "write_rel",
                                          "rel_dev_vs_channel_off",
                                          "block_python_class", "params_added")}
        out["rebind_carry_rule"] = r["config"]["carry_rule"]
        out["fill_kv"] = r["fill_kv"]
        out["proj_fill"] = r["proj_fill"]
        out["exit_mode"] = r["exit_mode"]
        out["train_tokens"] = r["train_tokens"]
        out["block_carry_liveness_class"] = r["carry_liveness"]["block_python_class"]
    return out


def ingest_pytest_file(name):
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        return None
    lines = [ln for ln in open(path).read().strip().splitlines() if ln.strip()]
    tail = lines[-1] if lines else ""
    out = {"source": name, "last_line": tail}
    for tok in tail.split(","):
        tok = tok.strip()
        for kind in ("passed", "failed", "error", "skipped"):
            if tok.endswith(kind):
                out[kind] = int(tok.split()[0])
    return out


def ingest_mutation():
    path = os.path.join(HERE, "mutation_report.json")
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    return {"n_mutations": len(r["mutations"]),
            "all_caught_red": r["all_caught_red"],
            "all_restored_bitwise": r["all_restored_bitwise"],
            "sha256_original": r["sha256_original"],
            "sha256_restored": r["sha256_restored"],
            "ids": [m["id"] for m in r["mutations"]],
            "caught": {m["id"]: m["caught_red"] for m in r["mutations"]}}


def run_smoke():
    """20 步 GPU 冒烟（取 /tmp/pathlm_gpu.lock；峰值 ≪5GB）。"""
    lock = "/tmp/pathlm_gpu.lock"
    n = 0
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            hp = None
            try:
                hp = int(open(os.path.join(lock, "holder_pid")).read().strip())
            except Exception:
                hp = None
            stale = (hp is not None and not _pid_alive(hp))
            if stale:
                for f in ("holder_pid", "heartbeat"):
                    try:
                        os.remove(os.path.join(lock, f))
                    except OSError:
                        pass
                try:
                    os.rmdir(lock)
                except OSError:
                    pass
                continue
            time.sleep(15)
            n += 1
            if n > 40:
                raise SystemExit("等 GPU 锁超时（>10 min）")
    try:
        with open(os.path.join(lock, "holder_pid"), "w") as f:
            f.write(str(os.getpid()))
        open(os.path.join(lock, "heartbeat"), "w").close()
        t0 = time.time()
        r = subprocess.run(SMOKE_CMD, cwd=ROOT, capture_output=True, text=True,
                           timeout=3600)
        with open(os.path.join(HERE, "smoke", "run_stdout.txt"), "w") as f:
            f.write(r.stdout + "\n--- stderr ---\n" + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"冒烟失败 rc={r.returncode}\n{r.stdout[-2000:]}")
        return round((time.time() - t0) / 60, 2)
    finally:
        for f in ("holder_pid", "heartbeat"):
            try:
                os.remove(os.path.join(lock, f))
            except OSError:
                pass
        try:
            os.rmdir(lock)
        except OSError:
            pass


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="额外跑 20 步 GPU 冒烟（取锁）")
    args = ap.parse_args()

    t0 = time.time()
    fixed = _fixed_depths()
    rand = sample_exit_depths(2, SEQ, N_LAYERS, torch.Generator().manual_seed(7))
    combos = [
        combo_report("carry_x_ragged", fixed, fill_kv=False, _exit_mode="fixed"),
        combo_report("carry_x_random_exit", rand, fill_kv=False,
                     _exit_mode="uniform(sampled seed7)"),
        combo_report("carry_x_random_exit_x_fill", rand, fill_kv=True,
                     _exit_mode="uniform(sampled seed7)"),
        combo_report("carry_x_proj_fill", rand, fill_kv=True, proj_fill=True,
                     _exit_mode="uniform(sampled seed7)"),
    ]
    results = {
        "config": {
            "device_combos": DEV,
            "model": {"d_model": D_MODEL, "n_layers": N_LAYERS,
                      "n_heads": N_HEADS, "head_dim": D_MODEL // N_HEADS,
                      "seq_len": SEQ, "dense": False, "topology": "sequential",
                      "carry_rule": "delta"},
            "seed": 0,
            "carry_activation_value": 0.5,
            "note": ("组合自检 = 非 dense CPU 前向；块类 / read_share / 退出生效 / "
                     "frozen 不变量 / adapter 消费 都是活性量，不是标签"),
        },
        "combos": combos,
        "write_timing": write_timing(),
        "default_untouched": default_untouched(),
        "smoke": ingest_smoke(),
        "smoke_this_run_minutes": (run_smoke() if args.smoke else None),
        "regression": ingest_pytest_file("pytest_full.txt"),
        "eng2_tests": ingest_pytest_file("pytest_eng2.txt"),
        "mutation": ingest_mutation(),
        # 键直接用**仓库相对路径字符串**（不要对相对路径再 relpath——那会按 CWD
        # 解析，从实验目录直跑时会把 cwd 前缀带进键里，见本线首个 results.json）。
        "code": {p: sha256(os.path.join(ROOT, p))
                 for p in ("pathlm/depth_ar.py", "train_depth_ar.py",
                           "tests/test_depth_ar.py",
                           "axes/1-time/1.1.4.1-explicit-memory/experiments/"
                           "2628-eng2-carry-ragged/run.py",
                           "axes/1-time/1.1.4.1-explicit-memory/experiments/"
                           "2628-eng2-carry-ragged/mutate.py",
                           "axes/1-time/1.1.4.1-explicit-memory/experiments/"
                           "2628-eng2-carry-ragged/reconcile_numbers.py")},
        "cost": {"cpu_selfcheck_minutes": round((time.time() - t0) / 60, 3),
                 "gpu_smoke_minutes": None},
    }
    if results["smoke"] is not None:
        results["cost"]["gpu_smoke_minutes"] = results["smoke"]["wall_minutes"]
        results["cost"]["peak_gpu_mem_mb"] = results["smoke"]["peak_gpu_mem_mb"]
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("combos", "smoke")}, indent=2,
                     ensure_ascii=False))
    print("combos:", [(c["name"], c["block_class"], c["carry_read_share"],
                       c["frozen_invariant_ok"], c["carry_changes_logits"])
                      for c in combos])


if __name__ == "__main__":
    main()
