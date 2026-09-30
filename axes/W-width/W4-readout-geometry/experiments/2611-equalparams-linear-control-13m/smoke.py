"""2611 · 线性等参数对照读出 —— 20 步冒烟（先冒烟再全跑，SCHEMA §3）。

验证三件事，落 `smoke/smoke-linear/smoke.json`：
1. **严格恒等起步**：全新模型（readout_linear_resid=True，P=0/Q~N(0,0.02)）在
   训练前 logits 与 tied 线性逐位一致（`torch.equal`）——不引入初始税；
2. **干预活性**：CLI 训练 20 步后 P、Q 均非零（不是死的干预 / 不是鞍点），
   且读出 transform 相对幅度 > 0；
3. **配置接线**：`--readout-linear-resid` 真的把 cfg.readout_linear_resid 打开，
   参数量 = A 臂 + 2·d²。

用法（CPU 侧恒等检查 + GPU 侧 20 步；总计 <1 min）：
    python3 smoke.py
"""

import json, os, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel
from pathlm.data import load_enwik8_full, batch

SMOKE_DIR = os.path.join(HERE, "smoke", "smoke-linear")
DATA = "data/enwik8_full.npz"
STEPS = 20


def cfg_base(**kw) -> DepthARConfig:
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


def main():
    os.makedirs(SMOKE_DIR, exist_ok=True)
    out = {"steps": STEPS, "cli": None, "checks": {}}
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)

    # ---- 1. 训练前严格恒等起步（P=0） ----
    torch.manual_seed(0)
    m = DepthARModel(cfg_base(readout_linear_resid=True), vocab)
    x = torch.randint(0, vocab, (2, 256))
    _, aux = m(x, exit_depths=None)
    tied = all(torch.equal(lg, s.float() @ m.embed.weight.float().T)
               for s, lg in zip(aux["states"], aux["depth_logits"]))
    out["checks"]["identity_start_bitwise"] = tied
    out["checks"]["p_fro_at_init"] = round(float(m.readout_p.weight.detach().norm()), 6)
    out["checks"]["q_fro_at_init"] = round(float(m.readout_q.weight.detach().norm()), 4)
    out["checks"]["params_linear"] = sum(p.numel() for p in m.parameters())
    torch.manual_seed(0)
    a = DepthARModel(cfg_base(), vocab)
    out["checks"]["params_A"] = sum(p.numel() for p in a.parameters())
    assert tied, "起步必须严格恒等（P=0 ⇒ transform = 0）"
    assert out["checks"]["p_fro_at_init"] == 0.0
    assert out["checks"]["q_fro_at_init"] > 0
    assert (out["checks"]["params_linear"] - out["checks"]["params_A"]
            == 2 * 256 * 256), "参数量必须 = A 臂 + 2·d²"

    # ---- 2. 20 步 CLI 训练（真实入口 + 真实开关路径） ----
    cmd = [sys.executable, "train_depth_ar.py", "smoke-linear",
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(STEPS), "--batch-size", "32",
           "--seed", "0", "--exit-mode", "uniform", "--readout-linear-resid",
           "--eval-batches", "2", "--out-root", os.path.join(HERE, "smoke")]
    out["cli"] = " ".join(cmd)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(SMOKE_DIR, f)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ,
                                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    out["smoke_wall_min"] = round((time.time() - t0) / 60, 3)
    if r.returncode != 0:
        out["stdout_tail"] = r.stdout[-2000:]
        out["stderr_tail"] = r.stderr[-2000:]
        json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"), indent=2)
        raise SystemExit(f"smoke training failed rc={r.returncode}")

    res = json.load(open(os.path.join(SMOKE_DIR, "results.json")))
    out["checks"]["cfg_readout_linear_resid"] = res["readout_linear_resid"]
    out["checks"]["cfg_readout_gated"] = res["readout_gated"]
    assert res["readout_linear_resid"] is True and res["readout_gated"] is False

    # ---- 3. 干预活性：20 步后 P、Q 非零，transform 非零 ----
    torch.manual_seed(0)
    m2 = DepthARModel(cfg_base(readout_linear_resid=True), vocab).cuda()
    m2.load_state_dict(torch.load(os.path.join(SMOKE_DIR, "model.pt"),
                                  map_location="cuda"))
    P = m2.readout_p.weight.detach().float()
    Q = m2.readout_q.weight.detach().float()
    m2.eval()
    gen = torch.Generator().manual_seed(3)
    xb, _ = batch(eval_arr, 8, 256, gen)
    with torch.no_grad():
        _, aux2 = m2(xb.cuda(), exit_depths=None)
        rel = []
        for s in aux2["states"]:
            h = s.float()
            t = m2.readout_p(m2.readout_q(h))
            rel.append(round(float(t.norm() / h.norm()), 4))
    out["checks"]["p_fro_after_20steps"] = round(float(P.norm()), 4)
    out["checks"]["q_fro_after_20steps"] = round(float(Q.norm()), 4)
    out["checks"]["p_nonzero_frac"] = round(float(torch.count_nonzero(P) / P.numel()), 6)
    out["checks"]["q_nonzero_frac"] = round(float(torch.count_nonzero(Q) / Q.numel()), 6)
    out["checks"]["transform_rel_mag_by_depth"] = rel
    alive = (out["checks"]["p_fro_after_20steps"] > 0
             and out["checks"]["q_fro_after_20steps"] > 0 and max(rel) > 0)
    out["checks"]["intervention_alive_after_20steps"] = alive
    assert alive, "20 步后干预仍为死（P/Q 或 transform 为 0）"

    json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
