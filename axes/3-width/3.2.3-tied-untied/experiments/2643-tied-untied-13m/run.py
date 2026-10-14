"""2643 · 输入/输出嵌入绑定 vs 解绑（tied / untied / 部分绑定）· 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P4 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成一个工厂（产出 `ReadoutModel` 子类：只覆写 `readout`），然后调用
  `train_depth_ar.main()` —— 参数解析、配置装配、`train()` 循环、checkpoint、
  `results.json` 全部是**共享实现**，本文件不改共享代码。
* **臂间唯一的差别 = 输出读出头**。`torch.manual_seed(seed)` 由共享 `main()` 在构造
  之前设好，三个臂先走**同一** `DepthARModel.__init__`（同一份主干抽样），`U`/`PR`
  臂在**之后**才初始化各自多出的读出头参数 ⇒ 不是「两个不同初始化的模型」在比。
* `T` 臂的 `readout` 表达式与 stock 逐字相同 ⇒ 逐位等价（单测 `T1`/`T2` 钉住）。
* `T` 臂是**复用** 2641 `U4-s0` 的 checkpoint（不重训），`evaluate()` 直接读它并按同一
  `depth_curve` 口径重评，用于 §9 三道门的口径核对。

用法：

    python3 run.py --arm U  --seed 0 --steps 12000            # 训练 + 评测
    python3 run.py --arm U  --seed 0 --steps 12000 --no-train # 仅评测（复用 model.pt）
    python3 run.py --arm T  --seed 0 --steps 12000 --no-train # 复用锚（2641 checkpoint）
    python3 run.py --aggregate                                # 汇总 P1–P4 → results.json
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full, batch                # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 臂定义（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VOCAB = 206
PR_RANK = 32
ARMS = ("T", "U", "PR")
VARIANCE_BOUND = 0.014

#: 复用锚（契约 §5-①③）：2641 的 U4-s0 = stock tied dense 13M，clean bpc 1.5314
ANCHOR_2641_RUN = os.path.join(
    ROOT, "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m",
    "runs/U4-s0-12k")
ANCHOR_2641_RESULTS = os.path.join(
    ROOT, "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m",
    "results_U4_s0.json")
ANCHOR_2641_CONFIG = os.path.join(
    ROOT, "axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m",
    "config.json")
T_ANCHOR_BPC = 1.5314


# ---------------------------------------------------------------- 干预：覆写读出

class ReadoutModel(DepthARModel):
    """只覆写 `readout`；`forward` / 主干 / 损失全部继承引擎（单因素）。

    `arm="T"` 时表达式的每一项与 stock `readout` 相同（`readout_dims=None`、无
    gated/linear_resid/perdepth/wide_shared 分支）⇒ 逐位等价。
    """

    def __init__(self, cfg: DepthARConfig, vocab_size: int, arm: str = "T"):
        super().__init__(cfg, vocab_size)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        d = cfg.d_model
        if arm == "U":
            # 独立 unembed：与 embedding 同尺度独立抽样（主干已在 super() 里抽完，
            # 此处不消耗任何被后续构造依赖的随机数）
            self.unembed = nn.Parameter(torch.empty(vocab_size, d))
            nn.init.normal_(self.unembed, std=d ** -0.5)
        elif arm == "PR":
            # 部分绑定：E + A·B；单边零（A=0, B~N(0,0.02)）⇒ 起步逐位 = tied 且 A 非鞍点
            self.pr_a = nn.Parameter(torch.zeros(d, PR_RANK))
            self.pr_b = nn.Parameter(torch.empty(PR_RANK, vocab_size))
            nn.init.normal_(self.pr_b, std=0.02)

    def effective_unembed(self) -> torch.Tensor:
        """有效读出矩阵 `U_eff`（[vocab, d]，fp32）：`logits = h @ U_eff.T`。

        `E` 是 [vocab, d]；低秩项 `A [d, r] @ B [r, vocab]` 是 [d, vocab]，故
        `U_eff = E + (A@B).T`，等价于 `logits = h·Eᵀ + (h·A)·B`（在线低秩实现）。
        `T` 臂返回 `E` 本身。
        """
        E = self.embed.weight.float()
        if self.arm == "T":
            return E
        if self.arm == "U":
            return self.unembed.float()
        return E + (self.pr_a.float() @ self.pr_b.float()).T

    def readout(self, h: torch.Tensor, depth: int | None = None) -> torch.Tensor:
        """fp32 头（autocast 之外强制 fp32），按臂选表。"""
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            if self.arm == "T":
                return hf @ self.embed.weight.float().T
            if self.arm == "U":
                return hf @ self.unembed.float().T
            # 部分绑定：h·Eᵀ + (h·A)·B（在线低秩；合并后 = h·(E + (A@B).T)ᵀ）
            return hf @ self.embed.weight.float().T + (hf @ self.pr_a.float()) @ self.pr_b.float()


def build_model(arm: str, vocab_size: int = VOCAB, seed: int = 0) -> ReadoutModel:
    """所有臂共用同一构造路径：`torch.manual_seed(seed)` → 同一主干抽样。"""
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)
    return ReadoutModel(cfg, vocab_size, arm)


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(arm: str, d: int = D_MODEL, T: int = SEQ_LEN,
                  vocab: int = VOCAB) -> dict:
    """每 token 等价 MACs（解析，**不含任何时间量**）+ 读出参数（解析）。

    读出头（在线实现）：`T`/`U` = d·V；`PR` = d·V + d·r + r·V（低秩项在线算）。
    离线合并 `E + A·B` 后 `PR` 的读出也是 d·V（单独列出 `readout_macs_merged`）。
    """
    L = N_LAYERS
    attn_per_layer = 4 * d * d + 2 * T * d
    ffn_per_layer = 2 * MLP_MULT * d * d
    attn = L * attn_per_layer
    ffn = L * ffn_per_layer
    readout_online = d * vocab
    if arm == "PR":
        readout_online += d * PR_RANK + PR_RANK * vocab
    added = {"T": 0, "U": vocab * d, "PR": d * PR_RANK + PR_RANK * vocab}[arm]
    total = attn + ffn + readout_online
    return {
        "d_model": d, "seq_len": T, "n_layers": L, "vocab": vocab,
        "macs_per_token": {"attention": attn, "ffn": ffn,
                           "readout_online": readout_online, "total": total},
        "readout_macs_merged": d * vocab,
        "readout_params_added": added,
        "params_total_analytic": 6962688 + added,
        "note": ("等效理论量：MACs/token（乘加），不含时间量。attention/FFN 三臂逐位相同，"
                 "唯一变量是读出头的结构与参数。在线读出：T/U = d·V；PR = d·V + d·r + r·V；"
                 "离线合并 E+A·B 后 PR 也是 d·V。"),
    }


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory_for(arm: str):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 `main()` 调它建模型。"""
    def factory(cfg, vocab_size):
        return ReadoutModel(cfg, vocab_size, arm)
    return factory


def shared_argv(run_name: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = 读出头）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"), run_name,
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    if arm == "T":
        return ANCHOR_2641_RUN
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory_for(arm)
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(f"{arm}-s{seed}-{steps // 1000}k", seed, steps,
                               batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()                       # ← 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_audit(model: ReadoutModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embedded = (0 if model.arm == "T" else
                (model.unembed.numel() if model.arm == "U"
                 else model.pr_a.numel() + model.pr_b.numel()))
    return {"params_total": total,
            "params_added_readout": embedded,
            "params_added_expected": analytic_cost(model.arm)["readout_params_added"],
            "readout_params_matches_analytic":
                embedded == analytic_cost(model.arm)["readout_params_added"]}


def intervention_liveness(model: ReadoutModel) -> dict:
    """`SCHEMA.md` §10：干预必须真的被激活（参数幅度非 0）且读出真的在用。"""
    out = {"arm": model.arm, "params_added": param_audit(model)["params_added_readout"]}
    def stat(name, t):
        tf = t.detach().float()
        out[name] = {"w_fro": round(float(tf.norm()), 6),
                     "w_absmax": round(float(tf.abs().max()), 6),
                     "nonzero_frac": round(float((tf != 0).float().mean()), 6)}
    if model.arm == "U":
        stat("unembed", model.unembed)
    elif model.arm == "PR":
        stat("pr_a", model.pr_a)
        stat("pr_b", model.pr_b)
        # A 必须已离开 0（否则 PR 退化成 tied，机制是死的）
        out["pr_correction_fro"] = round(
            float((model.pr_a.detach().float()
                   @ model.pr_b.detach().float()).norm()), 6)
        out["pr_active_nonzero"] = out["pr_a"]["w_fro"] > 0.0
    return out


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 6), round(centre + half, 6)]


def cluster_bootstrap_ci(rates: list[float], n_iter: int = 2000,
                         seed: int = 1234) -> list[float]:
    """按**评测批**聚类的配对 bootstrap 95% CI（重采样 batch，不重采样位置）。"""
    if not rates:
        return [None, None]
    g = torch.Generator().manual_seed(seed)
    n = len(rates)
    means = []
    for _ in range(n_iter):
        idx = torch.randint(0, n, (n,), generator=g)
        means.append(sum(rates[int(i)] for i in idx) / n)
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[min(int(0.975 * n_iter), n_iter - 1)]
    return [round(lo, 6), round(hi, 6)]


@torch.no_grad()
def prob0_consistency(model: ReadoutModel, eval_arr, n_batches: int = 20,
                      batch_size: int = 16, gen_seed: int = 2) -> dict:
    """解绑的语义代价：同一 h 下 `argmax(Eh) == argmax(U_eff h)` 的位置占比。

    在**最终深度状态**上算（`states[-1]`，序列位移位后逐位置）；`U_eff` = 该臂的有效
    读出头（`U` / `E+A·B`；`T` 臂 = `E` ⇒ 构造性 1.0）。率类指标走 Wilson +
    按 batch 聚类的 bootstrap 95% CI。
    """
    gen = torch.Generator().manual_seed(gen_seed)
    E = model.embed.weight.float()
    U = model.effective_unembed()
    agree = tot = 0
    per_batch = []
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        _, aux = model(x.to(DEVICE), exit_depths=None)
        h = aux["states"][-1][:, :-1].float().reshape(-1, model.cfg.d_model)
        a = (h @ E.T).argmax(-1)
        b = (h @ U.T).argmax(-1)
        k = int((a == b).sum())
        per_batch.append(k / h.shape[0])
        agree += k
        tot += h.shape[0]
    return {"metric": "prob0 一致性 = argmax(Eh)==argmax(U_eff h) 的位置占比",
            "n_positions": tot, "agree": agree, "rate": round(agree / tot, 6),
            "wilson_95ci": wilson_ci(agree, tot),
            "cluster_bootstrap_95ci": cluster_bootstrap_ci(per_batch),
            "per_batch_rates": [round(r, 6) for r in per_batch],
            "semantic_cost_1_minus_rate": round(1 - agree / tot, 6),
            "note": ("T 臂该值构造性 = 1.0（U_eff = E）；越小 = 解绑后「概率 = 状态在 "
                     "embedding 基的坐标」这一语义丢得越多。")}


@torch.no_grad()
def causal_check(model: ReadoutModel, gen_seed: int = 7) -> dict:
    """因果性（P3 必须不变）：只改位置 ≥ j 的 token，位置 < j 的 logits 必须逐位不变。"""
    g = torch.Generator().manual_seed(gen_seed)
    T = model.cfg.seq_len
    x = torch.randint(0, VOCAB, (2, T), generator=g).to(DEVICE)
    x2 = x.clone()
    j = T // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, T - j), generator=g).to(DEVICE)
    model.eval()
    _, a1 = model(x, exit_depths=None)
    _, a2 = model(x2, exit_depths=None)
    lg1 = a1["depth_logits"][-1][:, :j].float()
    lg2 = a2["depth_logits"][-1][:, :j].float()
    return {"max_abs_diff_before_j": float((lg1 - lg2).abs().max()),
            "bitwise_equal_before_j": bool(torch.equal(lg1, lg2))}


@torch.no_grad()
def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20) -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    ckpt = os.path.join(rd, "model.pt")
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                  "data/enwik8_full.npz")
    model = build_model(arm, vocab, seed).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    curve2 = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                         device=DEVICE)          # P3：重评确定性
    reeval_identical = bool(curve == curve2)
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": 2,
                 "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"],
                 "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]}
                                  for r in curve]},
        "params": param_audit(model),
        "cost": analytic_cost(arm, vocab=vocab),
        "liveness": intervention_liveness(model),
        "prob0_consistency": prob0_consistency(model, eval_arr,
                                               n_batches=n_batches),
        "causal": causal_check(model),
        "reeval_bitwise_identical": reeval_identical,
        "checkpoint": {"path": os.path.relpath(ckpt, ROOT),
                       "sha256": sha256_file(ckpt),
                       "missing_keys": list(missing),
                       "unexpected_keys": list(unexpected)},
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    if arm == "T":
        # 复用门 ③：口径核对 —— 同一 depth_curve 口径重评 2641 的锚
        out["reuse"] = {
            "anchor": "2641-ffn-ratio-frontier-13m / U4-s0-12k",
            "reused_clean_bpc_recorded": T_ANCHOR_BPC,
            "reevaluated_clean_bpc": curve[-1]["bpc"],
            "reproduces_recorded": (abs(curve[-1]["bpc"] - T_ANCHOR_BPC) < 1e-9),
            "anchor_results_json": os.path.relpath(ANCHOR_2641_RESULTS, ROOT),
            "anchor_config_json": os.path.relpath(ANCHOR_2641_CONFIG, ROOT),
            "gates": {"implementation_bitwise": "单测 T1/T2（pytest）",
                      "config_field_compare": "analysis.md §2 逐字段表",
                      "ruler": "n_batches=20, bs=16, gen seed 2, steps 12000, seed 0"},
        }
    path = os.path.join(HERE, f"results_{arm}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2643] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params_total={out['params']['params_total']} "
          f"prob0_rate={out['prob0_consistency']['rate']} "
          f"reeval_identical={reeval_identical}", flush=True)
    return out


# ---------------------------------------------------------------- 汇总（P1–P4）

def aggregate(steps: int, out_root: str, arms: list[str], seeds: list[int]) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(HERE, f"results_{arm}_s{s}.json")
            if os.path.exists(p):
                recs[(arm, s)] = json.load(open(p))

    def get(arm, s, *keys, default=None):
        if (arm, s) not in recs:
            return default
        v = recs[(arm, s)]
        for k in keys:
            if v is None:
                return default
            v = v.get(k) if isinstance(v, dict) else None
        return default if v is None else v

    deltas = {}
    p1_verdict = {}
    for s in seeds:
        t = get("T", s, "eval", "clean_bpc")
        for arm in ("U", "PR"):
            a = get(arm, s, "eval", "clean_bpc")
            if t is None or a is None:
                deltas[f"{arm}-s{s}"] = None
                p1_verdict[f"{arm}-s{s}"] = ("不可判：缺臂或锚"
                                             if t is None else "未跑")
                continue
            d = round(a - t, 4)
            deltas[f"{arm}-s{s}"] = {"delta_bpc": d, "arm_bpc": a, "anchor_bpc": t}
            if d <= -VARIANCE_BOUND:
                v = f"可排序更优：{arm} 解绑提升质量"
            elif d >= VARIANCE_BOUND:
                v = f"可排序更差：{arm} 解绑有质量代价"
            else:
                v = "不可排序（H0）：|Δ| < 0.014"
            p1_verdict[f"{arm}-s{s}"] = v

    big_vocab_undecidable = True
    out = {
        "experiment": "2643-tied-untied-13m",
        "criterion": ("P1 Δbpc(U−T)/Δbpc(PR−T) 按 0.014 带判可排序；"
                      "P2 参数账 + prob0 一致性（率类走配对 bootstrap/Wilson CI）；"
                      "P3 因果 + 重评确定性 + 活性；P4 否证线。"),
        "variance_bound": VARIANCE_BOUND,
        "band_declaration": ("0.014 仅限 bpc 口径、配对 Δ、≥5 窗口；prob0 一致性等率类指标"
                             "单独定门（batch 聚类 bootstrap / Wilson 95% CI），不跨量纲借用。"),
        "n_seeds": len([s for s in seeds if ("T", s) in recs]),
        "seed_note": "n=1（seed 0）：越带结论为单点快照，只报增量方向，不称最终质量。",
        "big_vocab_undecidable": big_vocab_undecidable,
        "big_vocab_note": ("本仓 vocab=206，读出参数 V·d=52,736 = 总参数 0.76% ⇒ 参数账在本尺度"
                           "几乎不可测；解绑的真实参数账只在 V≈1e5 及以上的大词表档显著，"
                           "本轮的结论**不得外推**到该档。"),
        "steps": steps,
        "arms": {f"{a}-s{s}": {"clean_bpc": get(a, s, "eval", "clean_bpc"),
                               "params_total": get(a, s, "params", "params_total"),
                               "params_added_readout": get(a, s, "params",
                                                           "params_added_readout"),
                               "macs_per_token": get(a, s, "cost", "macs_per_token",
                                                     "total"),
                               "readout_macs_online": get(a, s, "cost",
                                                          "macs_per_token",
                                                          "readout_online"),
                               "prob0_rate": get(a, s, "prob0_consistency", "rate"),
                               "reeval_bitwise_identical": get(
                                   a, s, "reeval_bitwise_identical"),
                               "causal_bitwise_equal": get(a, s, "causal",
                                                           "bitwise_equal_before_j")}
                          for (a, s) in sorted(recs)},
        "P1_deltas": deltas,
        "P1_verdict": p1_verdict,
        "P2_prob0": {f"{a}-s{s}": get(a, s, "prob0_consistency")
                     for (a, s) in sorted(recs)},
        "P3": {f"{a}-s{s}": {"reeval_bitwise_identical": get(
                    a, s, "reeval_bitwise_identical"),
                    "causal_bitwise_equal_before_j": get(a, s, "causal",
                                                         "bitwise_equal_before_j"),
                    "checkpoint_missing_keys": get(a, s, "checkpoint",
                                                   "missing_keys"),
                    "checkpoint_unexpected_keys": get(a, s, "checkpoint",
                                                      "unexpected_keys"),
                    "liveness": get(a, s, "liveness")}
               for (a, s) in sorted(recs)},
        "reuse_anchor": get("T", 0, "reuse"),
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("P1_deltas", "P1_verdict", "arms")},
                     indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/…/model.pt（或 T 臂复用 2641 锚）评测")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="T,U,PR")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.steps, args.out_root, args.arms.split(","),
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""])
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
    if not args.no_train:
        if args.arm == "T":
            raise SystemExit("T 是复用臂（2641 U4-s0 checkpoint），不允许重训："
                             "用 --no-train 走复评")
        os.makedirs(args.out_root, exist_ok=True)
        info = train_arm(args.arm, args.seed, args.steps, args.batch_size,
                         args.lr, args.out_root)
        rd = run_dir(args.arm, args.seed, args.steps, args.out_root)
        with open(os.path.join(rd, "train_wall.json"), "w") as f:
            json.dump(info, f)
    evaluate(args.arm, args.seed, args.steps, args.out_root)


if __name__ == "__main__":
    main()
