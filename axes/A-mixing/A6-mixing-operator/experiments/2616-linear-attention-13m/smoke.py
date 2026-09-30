"""2616 · 线性注意力 vs 全注意力（契约 R）· 20 步冒烟（先冒烟再全跑）。

落 `smoke/smoke.json`（**不含任何按 20 步外推的时间字段**，SCHEMA §17.5）。验证：

1. **关 → v1 逐位还原**：`attn_linear=False`（缺省）⇒ 块类 = stock `Block`、
   参数逐位相同、前向状态逐位相同、aux 里没有线性活性块；
2. **单因素参数账**（SCHEMA §15）：两臂参数量逐位相等、同 seed 下全部参数逐位
   相同、`state_dict` 键集相同（线性注意力不新增参数）；
3. **算子活性**（真实 shape T=256）：φ = elu+1 > 0、归一化分母 > 0、同权重把算子
   换回全注意力时输出有偏差（rel_dev > 0）、aux 里带 `attn_linear` 遥测；
4. **实现真的是 O(T)**（契约 R 实现要点）：FLOP 计数随 T 线性（T×4 ⇒ FLOPs×~4），
   对照的显式全注意力是平方（×~16）；
5. **严格因果**：改未来位置的 token，前面位置的状态逐位不变；
6. **仪器路径可用**：`retention.masked_bpc`（显式打分参考实现）在精确因果掩码下
   与释放版（O(T) 分块）的 bpc 差 = 仪器自检偏移（应 ≪ 0.014）；
7. **CLI 接线 + 活性落盘**：20 步 `train_depth_ar.py --dense [--attn-linear]`
   真的把开关打开、每 200 步行里带 `attn_linear` 活性量、记步时与峰值显存。

用法（<5 min）：
    python3 smoke.py
"""

import json, os, subprocess, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import (DepthARModel, Block, TopKBlock, LinearAttnBlock,
                             linear_attn_phi, linear_attn_chunked,
                             linear_attn_reference, attention_mass_stats,
                             attn_linear_probe)
from pathlm.data import load_enwik8_full, batch
from tools.evalkit.retention import masked_bpc
import run as exp_run          # 复用同一份配置/成本定义（单一来源）

SMOKE_DIR = os.path.join(HERE, "smoke")
STEPS = 20
T = exp_run.SEQ


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def op_flops(kind: str, T: int) -> int:
    """注意力子层的解析 FLOP 数（FlopCounterMode），用同一权重形状。"""
    from torch.utils.flop_counter import FlopCounterMode
    dh, B, H = 32, 2, 8
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, T, dh) for _ in range(3))
    qf, kf = linear_attn_phi(q), linear_attn_phi(k)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        if kind == "linear":
            linear_attn_chunked(qf, kf, v, exp_run.CHUNK)
        else:
            s = (q @ k.transpose(-2, -1) / dh ** 0.5).tril()
            _ = s.softmax(-1) @ v
    return counter.get_total_flops()


def main():
    os.chdir(ROOT)          # 数据/入口脚本都是仓库根相对路径（单一来源）
    os.makedirs(SMOKE_DIR, exist_ok=True)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", exp_run.DATA)
    out = {"steps": STEPS, "config": {"d": exp_run.D, "layers": exp_run.L,
                                      "heads": exp_run.H, "seq": exp_run.SEQ,
                                      "mlp_mult": exp_run.MLP,
                                      "batch_size": exp_run.BS,
                                      "linear_chunk": exp_run.CHUNK},
           "checks": {}, "cli": {}}

    # ---- 1. 关 → v1 逐位还原（CPU，真实 shape） ----
    torch.manual_seed(0)
    m_default = DepthARModel(exp_run.cfg_base(False), vocab)
    torch.manual_seed(0)
    m_off = DepthARModel(exp_run.cfg_base(False), vocab)
    x = torch.randint(0, vocab, (2, T))
    with torch.no_grad():
        s_default = m_default(x, exit_depths=None)[1]
        s_off = m_off(x, exit_depths=None)[1]
    out["checks"]["off_restores_v1_bitwise"] = bool(
        torch.equal(s_default["states"][-1], s_off["states"][-1]))
    out["checks"]["off_block_class"] = type(m_default.blocks[0]).__name__
    out["checks"]["off_has_attn_linear_block"] = "attn_linear" in s_default
    assert out["checks"]["off_restores_v1_bitwise"]
    assert out["checks"]["off_block_class"] == "Block"
    assert not out["checks"]["off_has_attn_linear_block"]

    # ---- 2. 单因素参数账（两臂逐位相同） ----
    models = {}
    for arm in exp_run.ARMS:
        torch.manual_seed(0)
        models[arm["arm"]] = DepthARModel(exp_run.cfg_base(arm["attn_linear"]),
                                          vocab)
    params = {k: n_params(m) for k, m in models.items()}
    out["checks"]["params"] = params
    assert len(set(params.values())) == 1, "线性臂与全注意力臂参数量不等（单因素被破坏）"
    sd0 = models["full"].state_dict()
    ident = {}
    for name, m in models.items():
        sd = m.state_dict()
        ident[name] = (set(sd) == set(sd0)
                       and all(torch.equal(v, sd[k]) for k, v in sd0.items()))
    out["checks"]["init_bitwise_identical_to_full"] = ident
    assert all(ident.values()), "同 seed 下两臂初始化不同"
    out["checks"]["block_class_by_arm"] = {
        name: type(m.blocks[0]).__name__ for name, m in models.items()}
    assert out["checks"]["block_class_by_arm"] == {"full": "Block",
                                                  "linear": "LinearAttnBlock"}
    out["checks"]["params_added"] = params["linear"] - params["full"]
    assert out["checks"]["params_added"] == 0

    # ---- 3. 算子活性（真实 shape） ----
    lin = models["linear"]
    lin.eval()
    with torch.no_grad():
        _, aux = lin(x, exit_depths=None)
    assert "attn_linear" in aux
    tele = aux["attn_linear"]
    devs = []
    blk = lin.blocks[0]
    h_in = aux["states"][0]
    with torch.no_grad():
        o_lin, t_lin = blk(h_in)
        o_full, _ = Block.forward(blk, h_in)
    dev = float((o_lin - o_full).norm() / o_full.norm())
    devs.append(dev)
    out["checks"]["operator_liveness"] = {
        "n_layers_with_telemetry": len(tele),
        # 活性量以 0 维张量返回（见 depth_ar.LinearAttnBlock）：这里显式 float()
        "phi_min": round(float(min(s["phi_min"] for s in tele)), 8),
        "phi_max": round(float(max(s["phi_max"] for s in tele)), 6),
        "den_min": round(float(min(s["den_min"] for s in tele)), 8),
        "inter_mass_frac_last_layer": round(float(tele[-1]["inter_mass_frac"]), 6),
        "rel_dev_vs_full_layer1": round(dev, 6),
        "params_added": 0,
    }
    assert out["checks"]["operator_liveness"]["phi_min"] > 0
    assert out["checks"]["operator_liveness"]["den_min"] > 0
    assert dev > 0, "算子没有改变前向（死开关）"
    assert tele[-1]["inter_mass_frac"] > 0, "状态通道没被读到（块间贡献恒为 0）"
    probe = attn_linear_probe(lin, eval_arr, n_batches=1, batch_size=2)
    out["checks"]["probe"] = {k: v for k, v in probe.items()
                             if k not in ("per_layer", "kernel_mass_note")}
    assert probe["alive"] and probe["rel_dev_vs_full_min"] > 0

    # ---- 4. O(T) 证据（FLOP 计数） ----
    lin_fl = {t: op_flops("linear", t) for t in (256, 1024)}
    full_fl = {t: op_flops("full", t) for t in (256, 1024)}
    out["checks"]["flop_scaling_Tx4"] = {
        "linear_flops_T256": lin_fl[256], "linear_flops_T1024": lin_fl[1024],
        "linear_ratio": round(lin_fl[1024] / lin_fl[256], 3),
        "full_flops_T256": full_fl[256], "full_flops_T1024": full_fl[1024],
        "full_ratio": round(full_fl[1024] / full_fl[256], 3)}
    assert out["checks"]["flop_scaling_Tx4"]["linear_ratio"] < 6.0
    assert out["checks"]["flop_scaling_Tx4"]["full_ratio"] > 12.0

    # ---- 5. 严格因果（真实 shape） ----
    x2 = x.clone()
    x2[0, 200] = (x2[0, 200] + 1) % vocab
    with torch.no_grad():
        s2 = lin(x2, exit_depths=None)[1]["states"][-1]
    with torch.no_grad():
        s1 = lin(x, exit_depths=None)[1]["states"][-1]
    out["checks"]["causal_before_t200_bitwise"] = bool(
        torch.equal(s1[0, :200], s2[0, :200]))
    out["checks"]["after_t200_changed"] = not bool(
        torch.equal(s1[0, 200:], s2[0, 200:]))
    assert out["checks"]["causal_before_t200_bitwise"]
    assert out["checks"]["after_t200_changed"]

    # ---- 6. 仪器路径（masked_bpc；仪器要求模型与输入同设备） ----
    lin = lin.cuda()
    ident_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    bpc_inst = masked_bpc(lin, eval_arr, ident_mask, n_batches=2,
                          batch_size=2, device="cuda")
    from pathlm.depth_ar import depth_curve
    bpc_rel = depth_curve(lin, eval_arr, n_batches=2, batch_size=2,
                          device="cuda")[-1]["bpc"]
    out["checks"]["instrument_path"] = {
        "bpc_released": bpc_rel, "bpc_instrument_identity_mask": round(bpc_inst, 6),
        "abs_offset": round(abs(bpc_rel - bpc_inst), 6)}
    assert out["checks"]["instrument_path"]["abs_offset"] < 0.014

    # ---- 7. CLI 20 步（接线 + 活性落盘 + 步时/显存） ----
    for arm in exp_run.ARMS:
        name = f"smoke-{arm['arm']}"
        run_dir = os.path.join(SMOKE_DIR, name)
        os.makedirs(run_dir, exist_ok=True)
        for f in ("train_log.jsonl", "gate_log.jsonl"):
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, "train_depth_ar.py", name,
               "--d", str(exp_run.D), "--layers", str(exp_run.L),
               "--heads", str(exp_run.H), "--seq-len", str(exp_run.SEQ),
               "--mlp-mult", str(exp_run.MLP), "--dense",
               "--steps", str(STEPS), "--batch-size", str(exp_run.BS),
               "--seed", "0", "--exit-mode", "uniform", "--eval-batches", "2",
               "--out-root", SMOKE_DIR]
        if arm["attn_linear"]:
            cmd += ["--attn-linear", "--linear-chunk", str(exp_run.CHUNK)]
        out["cli"][arm["arm"]] = " ".join(cmd)
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        t0, peak = time.time(), 0.0
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        while proc.poll() is None:
            try:
                q = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True)
                for line in q.stdout.strip().splitlines():
                    pid, mem = [s.strip() for s in line.split(",")]
                    if int(pid) == proc.pid:
                        peak = max(peak, float(mem) / 1024.0)
            except Exception:
                pass
            time.sleep(2)
        stdout = proc.stdout.read()
        wall = time.time() - t0
        if proc.returncode != 0:
            out["checks"]["failure"] = {"arm": arm["arm"],
                                        "stdout_tail": stdout[-2000:]}
            json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
                      indent=2, ensure_ascii=False)
            raise SystemExit(f"smoke training failed rc={proc.returncode}\n"
                             f"{stdout[-2000:]}")
        res = json.load(open(os.path.join(run_dir, "results.json")))
        log = [json.loads(l) for l in
               open(os.path.join(run_dir, "train_log.jsonl")) if l.strip()]
        out["checks"].setdefault("cli_effective", {})[arm["arm"]] = {
            "attn_linear_in_results": res["attn_linear"],
            "linear_chunk_in_results": res["linear_chunk"],
            "params": res["params"],
            "block_class": res["attn_linear_liveness"]["block_python_class"],
            "liveness": res["attn_linear_liveness"],
            "log_first_line_attn_linear": log[0]["attn_linear"],
            "log_last_line_attn_linear": log[-1]["attn_linear"],
            "log_lines": len(log),
            "train_tokens": res["train_tokens"],
            "cli_wall_s": round(wall, 2),
            "cli_wall_s_per_step_incl_startup": round(wall / STEPS, 3),
            "peak_child_mem_gb": round(peak, 2),
            "train_log_last_loss": log[-1]["loss"],
        }
        assert res["attn_linear"] is arm["attn_linear"], "CLI --attn-linear 未接线"
        assert res["params"] == params["full"]
        if arm["attn_linear"]:
            assert res["attn_linear_liveness"]["alive"]
            assert log[0]["attn_linear"] is not None, "训练日志缺活性量（SCHEMA §10）"
            assert log[0]["attn_linear"]["phi_min"] > 0
            assert log[0]["attn_linear"]["den_min"] > 0
            assert log[0]["attn_linear"]["n_layer_samples"] == exp_run.L
        else:
            assert res["attn_linear_liveness"]["enabled"] is False
            assert log[0]["attn_linear"] is None

    # ---- 8. 步时/显存实测（T=256，供预算判断；不做外推） ----
    out["checks"]["step_cost_measured_T256"] = {
        arm["arm"]: exp_run.model_step_cost(arm["attn_linear"], T=T, steps=6)
        for arm in exp_run.ARMS}
    out["checks"]["mass_stats_sanity"] = {
        "linear": {k: v for k, v in attention_mass_stats(
            lin, x[:2].cuda(), kind="linear").items()
            if k in ("mass_mean_dist", "mass_frac_within_64")},
        "full_on_linear_weights": {k: v for k, v in attention_mass_stats(
            lin, x[:2].cuda(), kind="full").items()
            if k in ("mass_mean_dist", "mass_frac_within_64")}}
    json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"params": params,
                      "init_bitwise_identical": ident,
                      "operator_liveness": out["checks"]["operator_liveness"],
                      "flop_scaling_Tx4": out["checks"]["flop_scaling_Tx4"],
                      "causal": {k: v for k, v in out["checks"].items()
                                 if k.startswith("causal") or k.startswith("after_")},
                      "instrument_path": out["checks"]["instrument_path"],
                      "cli_effective": {k: {kk: vv for kk, vv in v.items()
                                            if kk != "liveness"}
                                        for k, v in out["checks"]["cli_effective"].items()},
                      "step_cost_measured_T256": out["checks"]["step_cost_measured_T256"]},
                     indent=2, ensure_ascii=False)[:6000])
    print("[smoke] all checks passed", flush=True)


if __name__ == "__main__":
    main()
