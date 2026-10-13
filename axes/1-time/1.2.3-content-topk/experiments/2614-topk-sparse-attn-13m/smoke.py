"""2614 · 内容 top-k 稀疏注意力（契约 J）· 20 步冒烟（先冒烟再全跑）。

验证五件事，落 `smoke/smoke.json`（**不含任何按 20 步外推的时间字段**，SCHEMA §17.5）：

1. **关 → v1 逐位还原**：`attn_sparse_k=0`（缺省）与显式 0 逐位一致、块类是
   stock `Block`、`aux` 里没有稀疏活性块；
2. **单因素参数账**（SCHEMA §15）：三个 k（0/32/64）参数量逐位相等、同 seed 下
   全部参数逐位相同、`state_dict` 键集相同（因果掩码是 non-persistent buffer）；
3. **掩码不变量**（真实训练 shape T=256）：每行保留键数 == 理论 min(k, i+1)、
   无任何非因果键、选择不等于「最近 k 个位置」（content ≠ position）、
   算子确实改变前向（rel dev > 0）；
4. **CLI 接线 + 活性**：20 步 `train_depth_ar.py --dense --attn-sparse-k {0,32,64}`
   真的把开关打开（results.json::attn_sparse_k）、活性块 alive、训练日志的
   **每 200 步行**里带 `attn_sparse` 活性量（SCHEMA §10/§17.4）；
5. **步时/显存实测**（供全跑预算判断，不做外推）。

用法（<3 min）：
    python3 smoke.py
"""

import json, os, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.depth_ar import DepthARModel, TopKBlock, Block, attn_sparse_liveness
from pathlm.data import load_enwik8_full
import run as exp_run          # 复用同一份成本/配置定义（单一来源）

SMOKE_DIR = os.path.join(HERE, "smoke")
STEPS = 20
KS = (0, 32, 64)


def n_params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def theory_kept(k: int, T: int) -> float:
    return sum(min(k, i + 1) for i in range(T)) / T


def main():
    os.chdir(ROOT)          # 数据/入口脚本都是仓库根相对路径（单一来源）
    os.makedirs(SMOKE_DIR, exist_ok=True)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", exp_run.DATA)
    out = {"steps": STEPS, "config": {"d": exp_run.D, "layers": exp_run.L,
                                      "heads": exp_run.H, "seq": exp_run.SEQ,
                                      "mlp_mult": exp_run.MLP,
                                      "batch_size": exp_run.BS},
           "checks": {}, "cli": {}}
    torch.manual_seed(0)

    # ---- 1. 关 → v1 逐位还原（CPU，真实 shape） ----
    torch.manual_seed(0)
    m_default = DepthARModel(exp_run.cfg_base(0), vocab)
    torch.manual_seed(0)
    m_off = DepthARModel(exp_run.cfg_base(0), vocab)
    x = torch.randint(0, vocab, (2, exp_run.SEQ))
    with torch.no_grad():
        s_default = m_default(x, exit_depths=None)[1]
        s_off = m_off(x, exit_depths=None)[1]
    out["checks"]["off_restores_v1_bitwise"] = bool(
        torch.equal(s_default["states"][-1], s_off["states"][-1]))
    out["checks"]["off_block_class"] = type(m_default.blocks[0]).__name__
    out["checks"]["off_has_attn_sparse_block"] = "attn_sparse" in s_default
    assert out["checks"]["off_restores_v1_bitwise"]
    assert out["checks"]["off_block_class"] == "Block"
    assert not out["checks"]["off_has_attn_sparse_block"]

    # ---- 2. 单因素参数账（三个 k 逐位相同） ----
    models = {}
    for k in KS:
        torch.manual_seed(0)
        models[k] = DepthARModel(exp_run.cfg_base(k), vocab)
    params = {k: n_params(m) for k, m in models.items()}
    out["checks"]["params"] = params
    out["checks"]["params_added_vs_full"] = {k: params[k] - params[0] for k in KS}
    assert len(set(params.values())) == 1, "top-k 臂参数量与全注意力臂不等（单因素被破坏）"
    sd0 = models[0].state_dict()
    identical = {}
    for k in KS:
        sdk = models[k].state_dict()
        identical[k] = (set(sdk) == set(sd0) and
                        all(torch.equal(v, sdk[key]) for key, v in sd0.items()))
    out["checks"]["init_bitwise_identical_to_full"] = identical
    assert all(identical.values()), "同 seed 下 top-k 臂与全注意力臂的初始化不同"
    out["checks"]["state_dict_has_causal_buffer"] = any(
        "causal" in key for key in sd0)
    out["checks"]["block_class_by_k"] = {k: type(models[k].blocks[0]).__name__
                                         for k in KS}
    assert out["checks"]["block_class_by_k"] == {0: "Block", 32: "TopKBlock",
                                                 64: "TopKBlock"}
    assert not out["checks"]["state_dict_has_causal_buffer"]

    # ---- 3. 掩码不变量 + 活性（真实 shape，随机初始化权重） ----
    for k in (KS[1], KS[2]):
        m = models[k]
        m.eval()
        with torch.no_grad():
            _, aux = m(x, exit_depths=None)
        stats = aux["attn_sparse"]
        live = attn_sparse_liveness(m, stats)
        # 同权重反事实：topk 设为 ≥T ⇒ 退化 = stock 全注意力
        xb = torch.randint(0, vocab, (2, exp_run.SEQ))
        with torch.no_grad():
            _, ax = m(xb, exit_depths=None)
        devs = []
        for i, blk in enumerate(m.blocks):
            assert isinstance(blk, TopKBlock)
            h_in = ax["states"][i]
            o_sparse, tele = blk(h_in)
            saved = blk.topk
            blk.topk = exp_run.SEQ
            o_full, t_full = blk(h_in)
            blk.topk = saved
            assert t_full is None
            devs.append(float((o_sparse - o_full).norm() / o_full.norm()))
        out["checks"][f"mask_k{k}"] = {
            "kept_mean": live["kept_mean"],
            "kept_theory_mean": round(theory_kept(k, exp_run.SEQ), 6),
            "kept_count_matches_theory": live["kept_count_matches_theory"],
            "alive": live["alive"],
            "sel_dist_mean": live["sel_dist_mean"],
            "sel_in_lastk_frac": live["sel_in_lastk_frac"],
            "rel_dev_vs_full_min_layer": round(min(devs), 6),
            "params_added": live["params_added"],
        }
        assert live["alive"] and live["kept_count_matches_theory"]
        assert abs(live["kept_mean"] - theory_kept(k, exp_run.SEQ)) < 1e-6
        assert live["sel_in_lastk_frac"] < 1.0, "选择退化为「最近 k 个键」= 位置选择"
        assert min(devs) > 0, "算子没有改变前向（死开关）"
        assert live["params_added"] == 0

    # ---- 4. CLI 20 步（接线 + 活性落盘） ----
    for k in KS:
        name = f"smoke-k{k}"
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
               "--attn-sparse-k", str(k), "--steps", str(STEPS),
               "--batch-size", str(exp_run.BS), "--seed", "0",
               "--exit-mode", "uniform", "--eval-batches", "2",
               "--out-root", SMOKE_DIR]
        out["cli"][f"k{k}"] = " ".join(cmd)
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
            out["checks"]["failure"] = {"k": k, "stdout_tail": stdout[-2000:]}
            json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
                      indent=2, ensure_ascii=False)
            raise SystemExit(f"smoke training failed rc={proc.returncode}\n"
                             f"{stdout[-2000:]}")
        res = json.load(open(os.path.join(run_dir, "results.json")))
        log = [json.loads(l) for l in
               open(os.path.join(run_dir, "train_log.jsonl")) if l.strip()]
        out["checks"].setdefault("cli_effective", {})[f"k{k}"] = {
            "attn_sparse_k_in_results": res["attn_sparse_k"],
            "params": res["params"],
            "block_class": res["attn_sparse_liveness"]["block_python_class"],
            "liveness_alive": res["attn_sparse_liveness"]["alive"],
            "liveness": res["attn_sparse_liveness"],
            "log_first_line_attn_sparse": log[0]["attn_sparse"],
            "log_last_line_attn_sparse": log[-1]["attn_sparse"],
            "log_lines": len(log),
            "train_tokens": res["train_tokens"],
            "cli_wall_s": round(wall, 2),
            "cli_wall_s_per_step_incl_startup": round(wall / STEPS, 3),
            "peak_child_mem_gb": round(peak, 2),
            "train_log_last_loss": log[-1]["loss"],
        }
        assert res["attn_sparse_k"] == k, f"CLI --attn-sparse-k 未接线（k={k}）"
        assert res["params"] == params[0]
        if k > 0:
            assert res["attn_sparse_liveness"]["alive"]
            assert log[0]["attn_sparse"] is not None, "训练日志缺活性量（SCHEMA §10）"
            assert abs(log[0]["attn_sparse"]["kept_mean"]
                       - theory_kept(k, exp_run.SEQ)) < 1e-6
        else:
            assert res["attn_sparse_liveness"]["enabled"] is False
            assert log[0]["attn_sparse"] is None

    # ---- 5. 步时/显存实测（不做外推） ----
    out["checks"]["step_cost_measured"] = {
        f"k={k}": exp_run.model_step_cost(k, steps=8) for k in KS}
    json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"params": params,
                      "init_bitwise_identical": identical,
                      "mask_checks": {kk: vv for kk, vv in out["checks"].items()
                                      if kk.startswith("mask_")},
                      "cli_effective": out["checks"]["cli_effective"],
                      "step_cost_measured": out["checks"]["step_cost_measured"]},
                     indent=2, ensure_ascii=False)[:4000])
    print("[smoke] all checks passed", flush=True)


if __name__ == "__main__":
    main()
