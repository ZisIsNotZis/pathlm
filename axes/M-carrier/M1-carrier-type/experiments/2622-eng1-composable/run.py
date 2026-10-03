#!/usr/bin/env python3
"""2622-eng1-composable · ENG1 可组合性验证入口。

本实验**不追求性能**，只追求**可测性**（SCHEMA §23 第 6 问的工程面）：证明
`GQA × carry × window` 三个特性在引擎里从「硬互斥 / 块类二选一 ⇒ 组合不存在」
变成「可组合且各自活着」。

CPU 侧（缺省）：对 4 个组合各建一个模型，量每个特性的活性量（块类 / qkv 宽度 /
真进 SDPA 的 K/V 头数 / 窗口掩码是否进 SDPA / 写入通道读出量），并核对缺省路径与
stock `Block` 逐位一致、单特性快路径的块类不变；再摄取 20 步 GPU 冒烟日志与全量
pytest / 变异报告摘要，落 `results.json`。

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

from pathlm.config import ModelConfig          # noqa: E402
from pathlm.depth_ar import ComposableBlock, DepthARConfig, DepthARModel  # noqa: E402
from pathlm.model import Block                 # noqa: E402

DEV = "cpu"
SMOKE_NAME = "master"
#: 冒烟：契约 W3 要求过、ENG1 前不存在的组合（三特性齐开），M1 引擎（13.08M）
SMOKE_CMD = [
    "python3", "train_depth_ar.py", SMOKE_NAME,
    "--d", "256", "--layers", "12", "--heads", "8", "--mlp-mult", "6",
    "--seq-len", "256", "--dense", "--steps", "20", "--batch-size", "8",
    "--seed", "0", "--synth-task", "assoc-mqar", "--synth-pairs", "8",
    "--synth-queries", "8", "--synth-gap", "14",
    "--train-window", "16", "--carry-write", "delta", "--n-kv-heads", "2",
    "--out-root", os.path.join(HERE, "smoke"),
]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build(**kw):
    """ENG1 的 CPU 测试位点：d=32 / 8 头 / dh=4，dense + sequential。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=8, seq_len=12,
                        dense=True, **kw)
    return DepthARModel(cfg, vocab_size=10)


def _logits(model, x):
    with torch.no_grad():
        return model(x)[1]["depth_logits"][-1].float()


def _activate_carry(model, value=0.5):
    for blk in model.blocks:
        blk.carry_scale.data.fill_(value)


def combo_report(name, **kw):
    """一个组合的活性报告：三个特性各给一个「真的活着」的量。"""
    x = torch.randint(0, 10, (2, 12))
    m = build(**kw)
    rep = {"name": name, "block_class": type(m.blocks[0]).__name__,
           "gqa_on": kw.get("n_kv_heads") is not None,
           "carry_on": kw.get("carry_rule") is not None,
           "window_on": kw.get("train_window", 0) > 0,
           "qkv_out_width": int(m.blocks[0].qkv.weight.shape[0]),
           "qkv_out_width_mha_equivalent": 3 * 32,
           "n_params": sum(p.numel() for p in m.parameters())}
    if rep["carry_on"]:
        _activate_carry(m)
    with torch.no_grad():
        _, aux = m(x)
    rep["carry_read_share"] = (round(float(aux["attn_carry"][0]["read_share"]), 6)
                               if rep["carry_on"] else None)
    tele = getattr(m.blocks[0], "kv_telemetry", None)
    rep["kv_k_shape"] = list(tele["k_shape"]) if tele else None
    rep["kv_attn_mask_applied"] = bool(tele["attn_mask"]) if tele else None
    rep["kv_qkv_out_width_seen"] = int(tele["qkv_out_width"]) if tele else None
    if rep["window_on"]:
        m_full = build(**{**kw, "train_window": 12})       # 12 = seq_len ⇒ 恒等因果
        if rep["carry_on"]:
            _activate_carry(m_full)
        rep["window_changes_logits_vs_full"] = bool(
            not torch.equal(_logits(m, x), _logits(m_full, x)))
    else:
        rep["window_changes_logits_vs_full"] = None
    return rep


def default_bitwise_v1():
    """缺省路径（MHA、无 carry）与 stock `Block` 同 seed 下逐位一致。"""
    mcfg = ModelConfig(d_model=32, n_layers=1, n_heads=8, seq_len=12)
    torch.manual_seed(0)
    ref = Block(mcfg)
    torch.manual_seed(0)
    comp = ComposableBlock(mcfg)
    sd_ref, sd_comp = ref.state_dict(), comp.state_dict()
    weights_equal = (list(sd_ref) == list(sd_comp)
                     and all(torch.equal(a, b)
                             for a, b in zip(sd_ref.values(), sd_comp.values())))
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        fwd_equal = torch.equal(ref(x)[0], comp(x)[0])
    return {"weights_bitwise_equal": bool(weights_equal),
            "forward_bitwise_equal": bool(fwd_equal),
            "n_params_block": sum(p.numel() for p in ref.parameters()),
            "n_params_composable": sum(p.numel() for p in comp.parameters()),
            "state_dict_keys": list(sd_ref)}


def fast_paths():
    """单特性路径的块类不变（ENG1 只把「组合」接到新块）。"""
    return {
        "gqa_only": type(build(n_kv_heads=2).blocks[0]).__name__,
        "carry_only": type(build(carry_rule="plain", carry_chunk=64).blocks[0]).__name__,
        "default": type(build().blocks[0]).__name__,
        "gqa_x_carry": type(build(carry_rule="plain", carry_chunk=64,
                                  n_kv_heads=2).blocks[0]).__name__,
    }


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
           "final_log_row": rows[-1]}
    if os.path.exists(res):
        r = json.load(open(res))
        out["params"] = r["params"]
        out["peak_gpu_mem_mb"] = r["peak_gpu_mem_mb"]
        out["wall_minutes"] = r["wall_minutes"]
        out["kv_head_liveness_alive"] = r["kv_head_liveness"]["alive"]
        out["carry_liveness_alive"] = r["carry_liveness"]["alive"]
        out["block_class"] = r["kv_head_liveness"]["block_python_class"]
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
    """20 步 GPU 冒烟（取 /tmp/pathlm_gpu.lock；峰值 ≪5GB，20 步）。"""
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
                           timeout=1800)
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
    combos = [
        combo_report("gqa_x_carry", carry_rule="delta", carry_chunk=64,
                     n_kv_heads=2),
        combo_report("gqa_x_window", n_kv_heads=2, train_window=4),
        combo_report("carry_x_window", carry_rule="plain", carry_chunk=64,
                     train_window=4),
        combo_report("gqa_x_carry_x_window", carry_rule="delta", carry_chunk=64,
                     n_kv_heads=2, train_window=4),
    ]
    results = {
        "config": {
            "device_combos": DEV,
            "model": {"d_model": 32, "n_layers": 3, "n_heads": 8, "seq_len": 12,
                      "dense": True, "topology": "sequential"},
            "seed": 0,
            "carry_activation_value": 0.5,
            "note": ("组合自检 = CPU 前向；qkv 宽度 / K/V 张量形状 / read_share "
                     "都是活性量，不是标签"),
        },
        "combos": combos,
        "default_bitwise_v1": default_bitwise_v1(),
        "fast_paths": fast_paths(),
        "smoke": ingest_smoke(),
        "smoke_this_run_minutes": (run_smoke() if args.smoke else None),
        "regression": ingest_pytest_file("pytest_full.txt"),
        "eng1_tests": ingest_pytest_file("pytest_eng1.txt"),
        "mutation": ingest_mutation(),
        "code": {os.path.relpath(p, ROOT): sha256(os.path.join(ROOT, p))
                 for p in ("pathlm/depth_ar.py", "train_depth_ar.py",
                           "tests/test_eng1_composable.py",
                           "axes/M-carrier/M1-carrier-type/experiments/"
                           "2622-eng1-composable/run.py",
                           "axes/M-carrier/M1-carrier-type/experiments/"
                           "2622-eng1-composable/mutate.py",
                           "axes/M-carrier/M1-carrier-type/experiments/"
                           "2622-eng1-composable/reconcile_numbers.py")},
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
    print("combos:", [(c["name"], c["block_class"], c["qkv_out_width"],
                       c["carry_read_share"], c["window_changes_logits_vs_full"])
                      for c in combos])


if __name__ == "__main__":
    main()
