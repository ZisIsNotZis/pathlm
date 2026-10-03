"""2621 · 冒烟 + 变异验证（SCHEMA §3/§10/§22）。

三部分（均可失败）：

A. **实现层冒烟**（CPU，tiny 模型）：开关关闭逐位还原 v1；恒等起步（P_k=0/A=0）
   且同 seed 与 A 臂其余参数逐位一致；参数量 = A + (L+1)·2d²（PD 与 EP 逐位相等）；
   逐深度独立性（只扰动 P_0 ⇒ 只有 depth 0 变）；EP 忽略 depth（共享语义）。

B. **训练冒烟**（GPU，20 步 × 两臂）：起步恒等 + 20 步后读出活性量非零
   （P_k/Q_k 或 A/B），证明「开关不是死的」。用 `--steps 20` 真跑一次
   train_depth_ar.py，读 `train_log.jsonl` 的 `readout` 字段。

C. **变异验证**（SCHEMA §10，**抓红 + 逐字节还原**）：三处故意破坏
   pathlm/depth_ar.py，跑对应单测必须失败；随后从备份**逐字节**还原并核对
   sha256。任何一步不符即非零退出。

用法：
    python3 smoke.py            # A + C（CPU）；B 需要 GPU，见 --train
    python3 smoke.py --train    # 追加 B（GPU，需先取锁）
"""

import argparse, hashlib, os, subprocess, sys, tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel  # noqa: E402

D, L, TSEQ = 16, 3, 12
DEPTH_AR = os.path.join(ROOT, "pathlm", "depth_ar.py")
TEST = "tests/test_depth_ar.py"


def tiny(**kw):
    torch.manual_seed(0)
    return DepthARModel(DepthARConfig(d_model=D, n_layers=L, n_heads=2,
                                      seq_len=TSEQ, **kw), vocab_size=10)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def part_a():
    ok = True
    a, pd, ep = tiny(), tiny(readout_perdepth=True), tiny(readout_wide_shared=(L + 1) * D)
    extra = (L + 1) * 2 * D * D
    assert nparams(pd) - nparams(a) == extra, "PD 新增参数不等于 (L+1)·2d²"
    assert nparams(ep) - nparams(a) == extra, "EP 新增参数不等于 (L+1)·2d²"
    assert nparams(pd) - nparams(a) == nparams(ep) - nparams(a), "PD/EP 名义参数不等"
    # 同 seed：除新增读出参数外逐位一致
    sa, sp = a.state_dict(), pd.state_dict()
    assert set(sp) - set(sa) == {f"readout_pd_{t}.{k}.weight"
                                 for t in ("p", "q") for k in range(L + 1)}
    for k in sa:
        assert torch.equal(sa[k], sp[k]), f"主干参数 {k} 不逐位一致"
    torch.manual_seed(1)
    x = torch.randint(0, 10, (2, TSEQ))
    la, lp = a(x, exit_depths=None)[1]["depth_logits"], \
        pd(x, exit_depths=None)[1]["depth_logits"]
    assert all(torch.equal(u, v) for u, v in zip(la, lp)), "起步不是逐位恒等（PD）"
    le = ep(x, exit_depths=None)[1]["depth_logits"]
    assert all(torch.equal(u, v) for u, v in zip(la, le)), "起步不是逐位恒等（EP）"
    # 逐深度独立性：只扰动 P_0 ⇒ 只有 depth 0 变
    import torch.nn as nn
    with torch.no_grad():
        nn.init.normal_(pd.readout_pd_p[0].weight, std=0.3)
    l2 = pd(x, exit_depths=None)[1]["depth_logits"]
    assert not torch.allclose(l2[0], la[0]), "扰动 P_0 后 depth 0 未变"
    assert torch.equal(l2[1], la[1]), "扰动 P_0 影响了 depth 1（非逐深度独立）"
    # EP 共享语义：忽略 depth
    h = torch.randn(2, TSEQ, D)
    assert torch.equal(ep.readout(h, depth=0), ep.readout(h, depth=2))
    print("[A] 守恒/恒等/独立性/等参数 —— OK（新增参数 %d，PD=EP）" % extra)
    return ok


# ------------------------------ 变异验证 ------------------------------

MUTATIONS = [
    ("M1_忽略_perdepth_开关（P_k 非零时仍返回 tied）",
     "            elif self.cfg.readout_perdepth:",
     "            elif False:  # MUT1",
     f"{TEST}::test_readout_perdepth_formula_and_independence"),
    ("M2_逐深度共用一套_P/Q（P_k 取同一张量）",
     "hf = hf + self.readout_pd_p[depth](self.readout_pd_q[depth](hf))",
     "hf = hf + self.readout_pd_p[0](self.readout_pd_q[0](hf))",
     f"{TEST}::test_readout_perdepth_formula_and_independence"),
    ("M3_忽略_wide_shared_开关（EP 静默退回 tied）",
     "            elif self.cfg.readout_wide_shared > 0:",
     "            elif False:  # MUT3",
     f"{TEST}::test_readout_wide_shared_equal_params_and_identity"),
]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def pytest_node(node):
    r = subprocess.run([sys.executable, "-m", "pytest", node, "-q", "-x"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1200)
    return r.returncode


def part_c():
    original = open(DEPTH_AR, "rb").read()
    orig_sha = hashlib.sha256(original).hexdigest()
    text = original.decode()
    print(f"[C] 基线 sha256={orig_sha[:16]}…")
    for label, old, new, node in MUTATIONS:
        assert text.count(old) == 1, f"{label}: 锚点不唯一（{text.count(old)}）"
        try:
            with open(DEPTH_AR, "w") as f:
                f.write(text.replace(old, new))
            rc = pytest_node(node)
            if rc == 0:
                raise AssertionError(f"{label}: 变异未被抓红（{node} 仍通过）")
            print(f"[C] {label} ⇒ 抓红（rc={rc}）OK")
        finally:
            with open(DEPTH_AR, "wb") as f:      # **逐字节**还原
                f.write(original)
        assert sha256(DEPTH_AR) == orig_sha, f"{label}: 还原后 sha256 不符（未逐字节还原）"
        assert pytest_node(node) == 0, f"{label}: 还原后 {node} 仍失败"
    print("[C] 三条变异全部抓红且逐字节还原 —— OK")


# ------------------------------ 训练冒烟 ------------------------------

def part_b():
    from pathlm.depth_ar import DepthARModel as M
    out_root = os.path.join(HERE, "smoke")
    for name, flag in (("perdepth", ["--readout-perdepth"]),
                       ("wide_shared", ["--readout-wide-shared", "2304"])):
        run_name = f"smoke-{name}"
        d = os.path.join(out_root, run_name)
        os.makedirs(d, exist_ok=True)
        for f in ("train_log.jsonl", "gate_log.jsonl"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                os.remove(p)
        cmd = [sys.executable, "train_depth_ar.py", run_name,
               "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
               "--mlp-mult", "4", "--steps", "20", "--batch-size", "32", "--seed", "0",
               "--exit-mode", "uniform", *flag, "--eval-batches", "2",
               "--out-root", out_root]
        env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=900)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
        import json
        rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))]
        log = rows[-1].get("readout")
        assert log is not None, f"{name}: 训练日志没有 readout 活性量"
        active = ([min(log["p_fro_by_depth"]) > 0 and min(log["q_fro_by_depth"]) > 0]
                  if log["mode"] == "perdepth" else
                  [log["a_fro"] > 0 and log["b_fro"] > 0])[0]
        assert active, f"{name}: 20 步后干预仍为 0（开关可能是死的）：{log}"
        print(f"[B] {name}: 20 步活性量 {log['mode']} 非零 —— OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="追加 GPU 训练冒烟（B）")
    args = ap.parse_args()
    part_a()
    if args.train:
        part_b()
    part_c()
    print("[smoke] 全部通过")


if __name__ == "__main__":
    main()
