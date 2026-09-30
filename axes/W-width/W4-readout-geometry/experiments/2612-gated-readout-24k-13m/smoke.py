"""2612 · gated 读出 24k 收敛复测 —— 20 步冒烟（先冒烟再全跑，SCHEMA §3/§10）。

契约 C 复用既有开关（`--readout-gated`），不新增语义开关；本冒烟只验证**接线与
活性**，不重复单测已覆盖的公式语义（见 tests/test_depth_ar.py 的 readout_gated 组）：

1. **恒等起步**：全新 gated 模型（W1~N(0,0.02)/W2=0）训练前 logits 与 tied 线性
   逐位一致（`torch.equal`）——不引入初始税；
2. **参数量**：gated 臂 = A 臂 + 2·d²（= 131072）；
3. **接线**：两臂各跑 20 步 CLI，`results.json::readout_gated` 分别为 False / True，
   其余字段一致；
4. **干预活性**：gated 臂 20 步后 W1、W2 均非零（不是死的干预 / 不是鞍点），
   读出 transform 相对幅度 > 0。

用法（CPU 恒等检查 + GPU 侧两臂各 20 步；总计 < 2 min）：

    python3 smoke.py
"""

import json, os, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel
from pathlm.data import load_enwik8_full, batch

SMOKE_DIR = os.path.join(HERE, "smoke")
DATA = "data/enwik8_full.npz"
STEPS = 20


def cfg_base(**kw) -> DepthARConfig:
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


def run_cli(run_name: str, gated: bool) -> dict:
    run_dir = os.path.join(SMOKE_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    cmd = [sys.executable, "train_depth_ar.py", run_name,
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(STEPS), "--batch-size", "32",
           "--seed", "0", "--exit-mode", "uniform",
           "--eval-batches", "2", "--out-root", SMOKE_DIR]
    if gated:
        cmd.append("--readout-gated")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env=dict(os.environ,
                                PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"))
    if r.returncode != 0:
        raise SystemExit(f"{run_name} smoke rc={r.returncode}\n{r.stdout[-1500:]}"
                         f"\n{r.stderr[-1500:]}")
    return json.load(open(os.path.join(run_dir, "results.json")))


def main():
    os.makedirs(SMOKE_DIR, exist_ok=True)
    out = {"steps": STEPS, "checks": {}}
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)

    # ---- 1/2. 训练前的恒等起步与参数量（CPU） ----
    torch.manual_seed(0)
    g = DepthARModel(cfg_base(readout_gated=True), vocab)
    x = torch.randint(0, vocab, (2, 256))
    _, aux = g(x, exit_depths=None)
    identity = all(torch.equal(lg, s.float() @ g.embed.weight.float().T)
                   for s, lg in zip(aux["states"], aux["depth_logits"]))
    torch.manual_seed(0)
    a = DepthARModel(cfg_base(), vocab)
    out["checks"]["gated_identity_start_bitwise"] = identity
    out["checks"]["w2_fro_at_init"] = round(float(g.readout_w2.weight.detach().norm()), 6)
    out["checks"]["w1_fro_at_init"] = round(float(g.readout_w1.weight.detach().norm()), 4)
    out["checks"]["params_A"] = sum(p.numel() for p in a.parameters())
    out["checks"]["params_gated"] = sum(p.numel() for p in g.parameters())
    assert identity, "gated 起步必须严格恒等（W2=0 ⇒ transform = 0）"
    assert out["checks"]["w2_fro_at_init"] == 0.0
    assert out["checks"]["w1_fro_at_init"] > 0
    assert (out["checks"]["params_gated"] - out["checks"]["params_A"]
            == 2 * 256 * 256), "gated 参数量必须 = A 臂 + 2·d²"

    # ---- 3. 两臂 20 步 CLI 接线 ----
    t0 = time.time()
    res_a = run_cli("smoke-A", gated=False)
    res_g = run_cli("smoke-gated", gated=True)
    out["smoke_wall_min"] = round((time.time() - t0) / 60, 3)
    out["checks"]["A_cfg_readout_gated"] = res_a["readout_gated"]
    out["checks"]["gated_cfg_readout_gated"] = res_g["readout_gated"]
    assert res_a["readout_gated"] is False and res_g["readout_gated"] is True
    assert res_a["config"] == {**res_g["config"], "readout_gated": False}, \
        "两臂 resolved config 除 readout_gated 外必须一致"

    # ---- 4. 干预活性：20 步后 W1/W2 非零，transform 非零 ----
    torch.manual_seed(0)
    m = DepthARModel(cfg_base(readout_gated=True), vocab).cuda()
    m.load_state_dict(torch.load(os.path.join(SMOKE_DIR, "smoke-gated",
                                              "model.pt"), map_location="cuda"))
    W1 = m.readout_w1.weight.detach().float()
    W2 = m.readout_w2.weight.detach().float()
    m.eval()
    gen = torch.Generator().manual_seed(3)
    xb, _ = batch(eval_arr, 8, 256, gen)
    with torch.no_grad():
        _, aux2 = m(xb.cuda(), exit_depths=None)
        rel = []
        for s in aux2["states"]:
            h = s.float()
            t = torch.nn.functional.silu(m.readout_w1(h)) * m.readout_w2(h)
            rel.append(round(float(t.norm() / h.norm()), 4))
    out["checks"]["w1_fro_after_20steps"] = round(float(W1.norm()), 4)
    out["checks"]["w2_fro_after_20steps"] = round(float(W2.norm()), 4)
    out["checks"]["nonzero_frac_after_20steps"] = round(
        float((torch.count_nonzero(W1) + torch.count_nonzero(W2))
              / (W1.numel() + W2.numel())), 6)
    out["checks"]["transform_rel_mag_by_depth"] = rel
    alive = (out["checks"]["w1_fro_after_20steps"] > 0
             and out["checks"]["w2_fro_after_20steps"] > 0 and max(rel) > 0)
    out["checks"]["intervention_alive_after_20steps"] = alive
    assert alive, "20 步后干预仍为死（W1/W2 或 transform 为 0）"

    json.dump(out, open(os.path.join(SMOKE_DIR, "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
