"""2651 · 层间拓扑 seed=1 · 13M —— 20 步冒烟（先冒烟再全跑，SCHEMA §3 / harness/methodology §10）。

验证四件事，落 `smoke/smoke.json`：
1. **参数量逐位账**（SCHEMA §15 记账）：sequential = parallel-dual-ln，
   parallel = sequential − L·2d（少一个共享 pre-LN），dense = sequential + L(L+1)/2；
   且同 seed 下三个臂的**全部公共参数逐位相同**（子模块创建顺序一致 ⇒ 随机数流对齐）
   ⇒ 臂间唯一差异 = 接线；
2. **恒等起步**：dense 臂的层权 identity 起步（只取当前层）⇒ 全深度 states 与
   sequential 逐位一致；parallel 臂在初始化下与 sequential 的输出差异已非零
   （接线真的在改变计算，不是等价物）；
3. **配置接线**：CLI `--topology` 真的把 cfg.topology 打开，且 `--dense` 形态下
   参数量与上式一致；
4. **干预活性**：CLI 训练 20 步后 dense 臂的层权 α 已离开 identity；parallel 臂
   两分支增量范数非零。同时实测**步时与峰值显存**（全跑预算的输入）。

用法（<1 min）：
    python3 smoke.py
"""

import json, os, shutil, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, topology_liveness,
                             TOPOLOGIES)
from pathlm.data import load_enwik8_full

SMOKE_DIR = os.path.join(HERE, "smoke")
DATA = "data/enwik8_full.npz"
STEPS = 20
D, L, H, SEQ, MLP = 256, 8, 8, 256, 4


def cfg_base(topo: str, **kw) -> DepthARConfig:
    """契约 E 骨架：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True, topology=topo, **kw)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    os.makedirs(SMOKE_DIR, exist_ok=True)
    out = {"steps": STEPS, "config": {"d": D, "layers": L, "heads": H,
                                      "seq": SEQ, "mlp_mult": MLP},
           "checks": {}, "cli": {}}
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    out["checks"]["vocab_size"] = vocab

    # ---- 1. 参数量账 + 初始化逐位一致（CPU 侧，不训练） ----
    models = {}
    for topo in TOPOLOGIES:
        torch.manual_seed(1)
        models[topo] = DepthARModel(cfg_base(topo), vocab)
    params = {t: n_params(m) for t, m in models.items()}
    out["checks"]["params"] = params
    out["checks"]["params_delta_vs_sequential"] = {
        t: params[t] - params["sequential"] for t in TOPOLOGIES}
    out["checks"]["params_expected_delta"] = {
        "sequential": 0, "parallel": -L * 2 * D, "parallel-dual-ln": 0,
        "dense": L * (L + 1) // 2}
    assert out["checks"]["params_delta_vs_sequential"] == \
        out["checks"]["params_expected_delta"], "参数量账不符"
    assert params["sequential"] == params["parallel-dual-ln"], "等参数臂不等参数"

    def logical(sd):
        """包装层命名差（blocks.k.block.* vs blocks.k.*）按同一逻辑名映射。"""
        o = {}
        for k, v in sd.items():
            parts = k.split(".")
            if parts[0] == "blocks" and parts[2] == "block":
                parts = parts[:2] + parts[3:]
            o[".".join(parts)] = v
        return o

    sd = {t: logical(m.state_dict()) for t, m in models.items()}
    shared_keys = {t: set(sd[t]) for t in TOPOLOGIES}
    out["checks"]["nonzero_element_diff_vs_sequential"] = {
        t: sum(int(not torch.equal(sd[t][k], sd["sequential"][k]))
               for k in shared_keys[t] & shared_keys["sequential"])
        for t in TOPOLOGIES}
    assert out["checks"]["nonzero_element_diff_vs_sequential"]["parallel-dual-ln"] == 0
    assert out["checks"]["nonzero_element_diff_vs_sequential"]["dense"] == 0
    assert out["checks"]["params"]["parallel"] - params["sequential"] == -L * 2 * D
    out["checks"]["extra_keys_dense"] = sorted(set(sd["dense"]) - set(sd["sequential"]))
    out["checks"]["missing_keys_parallel"] = sorted(
        set(sd["sequential"]) - set(sd["parallel"]))

    # ---- 2. 恒等起步 / 接线活性（CPU 侧，随机 token） ----
    x = torch.randint(0, vocab, (2, SEQ))
    with torch.no_grad():
        states = {t: models[t](x)[1]["states"][-1] for t in TOPOLOGIES}
    seq_states = states["sequential"]
    rel = lambda u, v: round(float((u - v).norm() / v.norm()), 6)
    out["checks"]["identity_start_bitwise_vs_sequential"] = {
        t: bool(torch.equal(states[t], seq_states)) for t in TOPOLOGIES}
    out["checks"]["wiring_rel_dev_vs_sequential_at_init"] = {
        t: rel(states[t], seq_states) for t in TOPOLOGIES}
    assert out["checks"]["identity_start_bitwise_vs_sequential"]["dense"]
    assert not out["checks"]["identity_start_bitwise_vs_sequential"]["parallel"]
    assert not out["checks"]["identity_start_bitwise_vs_sequential"]["parallel-dual-ln"]
    out["checks"]["alpha_at_init_dense"] = topology_liveness(models["dense"])["alpha_by_layer"]

    # ---- 3. + 4. CLI 20 步：接线 / 活性 / 步时 / 峰值显存 ----
    for topo in ("parallel", "parallel-dual-ln", "dense"):
        run_name = f"smoke-{topo}"
        run_dir = os.path.join(SMOKE_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)
        for f in ("train_log.jsonl", "gate_log.jsonl"):
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, "train_depth_ar.py", run_name,
               "--d", str(D), "--layers", str(L), "--heads", str(H),
               "--seq-len", str(SEQ), "--mlp-mult", str(MLP), "--dense",
               "--topology", topo, "--steps", str(STEPS), "--batch-size", "32",
               "--seed", "1", "--exit-mode", "uniform", "--eval-batches", "2",
               "--out-root", SMOKE_DIR]
        out["cli"][topo] = " ".join(cmd)
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
            out["checks"]["failure"] = {"topology": topo,
                                        "stdout_tail": stdout[-2000:]}
            json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
                      indent=2, ensure_ascii=False)
            raise SystemExit(f"smoke training failed rc={proc.returncode}\n{stdout[-2000:]}")
        res = json.load(open(os.path.join(run_dir, "results.json")))
        out["checks"].setdefault("cli_effective", {})[topo] = {
            "cfg_topology": res["topology"],
            "params": res["params"],
            "params_delta_vs_expected": res["params"] - params["sequential"],
            "train_tokens": res["train_tokens"],
            "wall_min": round(wall / 60, 3),
            "min_per_12k_extrapolated": round(wall / 60 * (12000 / STEPS), 2),
            "peak_child_mem_gb": round(peak, 2),
            "loss_first": json.loads(open(os.path.join(run_dir, "train_log.jsonl"))
                                     .readline())["loss"],
            "loss_last": json.loads(open(os.path.join(run_dir, "train_log.jsonl"))
                                    .readlines()[-1])["loss"],
            "topology_liveness": res["topology_liveness"],
        }
        assert res["topology"] == topo, f"CLI --topology 未接线（{topo}）"
        assert res["params"] - params["sequential"] == \
            out["checks"]["params_expected_delta"][topo], "CLI 参数量账不符"

    # ---- 5. 20 步后活性：dense α 已离开 identity；parallel 分支增量非零 ----
    torch.manual_seed(1)
    m = DepthARModel(cfg_base("dense"), vocab).cuda()
    m.load_state_dict(torch.load(os.path.join(SMOKE_DIR, "smoke-dense", "model.pt"),
                                 map_location="cuda"))
    act = topology_liveness(m)
    out["checks"]["dense_alpha_activity_after_20steps"] = act
    assert act["alive"], "20 步后 dense 层权仍停在 identity（死的干预）"
    xt = torch.randint(0, vocab, (2, SEQ)).cuda()
    m.eval()
    with torch.no_grad():
        st = m(xt)[1]["states"]
    out["checks"]["dense_states_norm_rel_spread"] = round(
        float((st[-1].norm() - st[0].norm()).abs() / st[0].norm()), 6)

    torch.manual_seed(1)
    p = DepthARModel(cfg_base("parallel-dual-ln"), vocab).cuda()
    p.load_state_dict({("blocks." + k.split(".")[1] + "." + ".".join(k.split(".")[3:]))
                       if k.startswith("blocks.") and ".block." in k else k: v
                       for k, v in torch.load(
                           os.path.join(SMOKE_DIR, "smoke-parallel-dual-ln",
                                        "model.pt"), map_location="cuda").items()},
                      strict=True)
    p.eval()
    with torch.no_grad():
        _, aux_p = p(xt)
        branch = []
        for i, blk in enumerate(p.blocks):
            h = aux_p["states"][i]
            h1 = blk.ln1(h)
            q, k, v = blk.qkv(h1).chunk(3, dim=-1)
            B, T, d = h.shape
            shape = lambda t: t.view(B, T, blk.n_heads, d // blk.n_heads).transpose(1, 2)
            a = torch.nn.functional.scaled_dot_product_attention(
                shape(q), shape(k), shape(v), is_causal=True)
            a = blk.proj(a.transpose(1, 2).reshape(B, T, d))
            mlp_d = blk.mlp(blk.ln2(h))
            branch.append({"layer": i + 1,
                           "attn_delta_rel": round(float(a.norm() / h.norm()), 6),
                           "mlp_delta_rel": round(float(mlp_d.norm() / h.norm()), 6)})
    out["checks"]["parallel_branch_activity_after_20steps"] = branch
    assert max(b["attn_delta_rel"] for b in branch) > 0 and \
        max(b["mlp_delta_rel"] for b in branch) > 0, "并行分支为死（增量为 0）"

    json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["checks"]["params"], indent=2))
    print(json.dumps(out["checks"]["params_delta_vs_sequential"], indent=2))
    print(json.dumps(out["checks"]["identity_start_bitwise_vs_sequential"], indent=2))
    print(json.dumps(out["checks"]["wiring_rel_dev_vs_sequential_at_init"], indent=2))
    print(json.dumps(out["checks"]["cli_effective"], indent=2, ensure_ascii=False)[:2000])
    print("[smoke] all checks passed", flush=True)


if __name__ == "__main__":
    main()
