"""2654 · 跳层（skipping）的序列级重测 · 13M · depth-ar/dense · 工厂替换。

契约 = 同目录 `README.md`（判据线 P0–P5 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成一个**工厂**（构造完 stock 模型后，把每层 Block 包一层 `SkipBlock`，按本臂的
  跳层策略对**逐 token 掩码**做恒等旁路），然后调用 `train_depth_ar.main()` ——
  参数解析、配置装配、`train()` 循环、checkpoint、`results.json` 全部是**共享实现**，
  本文件不改共享代码（`pathlm/**` / `train_depth_ar.py` 零改动）。
* **跳层语义**（契约 §2.1）：被跳层对**该 token** 的输出 = 输入（恒等旁路），
  用逐 token 布尔掩码 `torch.where(skip, x, block_out)` 实现 —— 因果性不破
  （token i 的输出只依赖 ≤ i 的输入），且不产生该层的 KV。
* **训练/评测同一分布**：`skip-mid` 恒跳、`layerdrop-pt` 逐 token 独立 Bernoulli(0.1)，
  训练与评测走**同一份** `SkipBlock` 采样代码 ⇒ 分布一致；两边的逐层跳过占比都落盘。
* 四臂的唯一差别 = 跳层策略（层数/数据/步数/seed/lr/batch 逐字段相同）。

用法：

    python3 run.py --arms stock,skip-mid,layers6,layerdrop-pt    # 训练+评测，逐臂落盘
    python3 run.py --arms skip-mid --no-train                    # 仅评测（复用 model.pt）
    python3 run.py --smoke                                       # 20 步冒烟（全臂）
    python3 run.py --aggregate                                   # 汇总 P0–P5 → results.json

本文件不取 GPU 锁；训练由驱动脚本 `run_remaining.sh` 按臂取/放锁（`SCHEMA.md` §20.1）。
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402
from pathlm.data import load_enwik8_full, batch               # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 契约常量（README §4）

D_MODEL, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 256, 4
BATCH_SIZE, LR, STEPS, SEED = 32, 1e-3, 12000, 0
VARIANCE_BOUND = 0.014
EVAL_BATCHES, EVAL_BATCH = 20, 16          # 20 窗口（每个 = batch16 × seq256）
EVAL_SEED = 2                              # 与 2613 / 2641 同款评测抽样口径

SKIP_LAYERS = (3, 4)                       # 0-indexed block 下标 = 人类第 4、5 层（最中两层）
LAYERDROP_P = 0.1                          # 逐 token 逐层独立跳过概率

# 引擎等值锚（契约 P0）：2613 s0 sequential 臂（同引擎/同配方/同 seed）
ANCHOR_2613_S0 = 1.5254
ANCHOR_23D_DENSE_B = 1.5287

# ---------------------------------------------------------------- 臂定义（契约 §3）

ARMS = {
    "stock":       {"layers": 8, "mode": "none",  "lp_design": 8.0,
                    "role": "对照：8 层全活（layer-passes 8）"},
    "skip-mid":    {"layers": 8, "mode": "fixed", "lp_design": 6.0,
                    "role": f"固定跳层：block {SKIP_LAYERS} 对全部 token 恒等旁路（layer-passes 6）"},
    "layers6":     {"layers": 6, "mode": "none",  "lp_design": 6.0,
                    "role": "同预算固定深度对手：直接训 6 层 dense（layer-passes 6，参数更少）"},
    "layerdrop-pt": {"layers": 8, "mode": "drop", "lp_design": 7.2,
                     "role": f"逐 token 独立跳层 p={LAYERDROP_P}（layer-passes ≈ 8·0.9 = 7.2）"},
}
ARM_ORDER = ["stock", "skip-mid", "layers6", "layerdrop-pt"]

# 工厂读取的"当前臂"（train/eval 各设一次；与 2641 的 `_ACTIVE_SCHEDULE` 同款）
_ACTIVE_ARM: dict | None = None
_ACTIVE_GEN_SEED = 0
_BUILT: list = []


# ---------------------------------------------------------------- 干预：逐 token 跳层旁路

class SkipBlock(nn.Module):
    """把一层 stock `Block` 包成"可按逐 token 掩码恒等旁路"的层。

    mode="none"  ⇒ 逐位透传（stock / layers6；**不消耗随机数、不改变任何张量**）；
    mode="fixed" ⇒ `skip_layers` 里的层对**全部** token 恒等旁路；
    mode="drop"  ⇒ 每 token 每层独立 Bernoulli(p) 旁路（layerdrop-pt）。

    旁路语义 = `out = where(skip, x, block_out)`（被跳 token 的输出就是该层输入，
    即恒等残差旁路）。`x` 是层输入张量本身 ⇒ 被跳 token 的梯度沿恒等路径回传。
    活性量：记录**训练期**与**最近一次**前向的逐层跳过占比（`SCHEMA.md` §10）。
    """

    def __init__(self, block: nn.Module, layer_idx: int, mode: str,
                 skip_layers: tuple, p: float, gen_seed: int):
        super().__init__()
        self.block = block
        self.layer_idx = int(layer_idx)
        self.mode = mode
        self._is_skip_layer = layer_idx in skip_layers
        self.p = float(p)
        self.telemetry_kind = getattr(block, "telemetry_kind", None)
        self._gen = torch.Generator().manual_seed(int(gen_seed))
        # 活性账：训练期累计 + 最近一次前向（逐层跳过占比）
        self.train_skip_sum = 0.0
        self.train_skip_n = 0
        self.last_skip_mean = 0.0

    def __getattr__(self, name):
        # 代理到被包的 Block（`kv_head_liveness` 会取 `blk.qkv` 等）；
        # 先走 nn.Module 的正常查找，失败再透传到内层块。
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "block":
                raise
            return getattr(super().__getattr__("block"), name)

    def _sample_skip(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.mode == "none":
            return None
        B, T = x.shape[0], x.shape[1]
        if self.mode == "fixed":
            if not self._is_skip_layer:
                return None
            return torch.ones(B, T, dtype=torch.bool, device=x.device)
        # mode == "drop"：逐 token 逐层独立 Bernoulli(p)，CPU 采样后搬到设备
        r = torch.rand(B, T, generator=self._gen)
        return (r < self.p).to(x.device)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        out, tele = self.block(x, *args, **kwargs)
        skip = self._sample_skip(x)
        if skip is None:
            self.last_skip_mean = 0.0
            return out, tele
        s = float(skip.float().mean())
        self.last_skip_mean = s
        if self.training:
            self.train_skip_sum += s
            self.train_skip_n += 1
        out = torch.where(skip.unsqueeze(-1), x, out)
        return out, tele


def brand_layers(model: DepthARModel, spec: dict, gen_seed: int) -> DepthARModel:
    """按本臂 spec 给每层 Block 套 `SkipBlock`（mode="none" 也套 ⇒ 四臂 state_dict
    键名一致、评测重建路径一致）。`gen_seed` 每个臂-层不同，采样互相独立。"""
    wrapped = nn.ModuleList([
        SkipBlock(blk, i, spec["mode"], SKIP_LAYERS, spec.get("p", 0.0),
                  gen_seed + i)
        for i, blk in enumerate(model.blocks)])
    model.blocks = wrapped
    return model


def build_model(spec: dict, vocab_size: int, gen_seed: int,
                cfg: DepthARConfig | None = None) -> DepthARModel:
    """按 spec 建一个带 `SkipBlock` 的模型。`cfg` 给定则用之（评测侧 = 训练 run 的
    resolved config）；否则按本实验契约常量构造（`--smoke` 侧）。"""
    if cfg is None:
        cfg = DepthARConfig(d_model=D_MODEL, n_layers=spec["layers"],
                            n_heads=N_HEADS, seq_len=SEQ_LEN, mlp_mult=MLP_MULT,
                            dense=True, topology="sequential")
    model = DepthARModel(cfg, vocab_size)
    return brand_layers(model, spec, gen_seed)


def _factory(cfg, vocab_size):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    model = DepthARModel(cfg, vocab_size)
    brand_layers(model, _ACTIVE_ARM, _ACTIVE_GEN_SEED)
    _BUILT.append(model)
    return model


# ---------------------------------------------------------------- 训练（走共享 main）

def run_dir(arm: str, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-{steps // 1000}k-s{SEED}")


def shared_argv(arm: str, layers: int, steps: int, out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = 跳层策略）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-{steps // 1000}k-s{SEED}",
            "--d", str(D_MODEL), "--layers", str(layers),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(BATCH_SIZE), "--lr", str(LR), "--seed", str(SEED),
            "--eval-batches", str(EVAL_BATCHES), "--out-root", out_root]


def train_arm(arm: str, steps: int, out_root: str) -> dict:
    global _ACTIVE_ARM, _ACTIVE_GEN_SEED
    spec = ARMS[arm]
    _ACTIVE_ARM = spec
    _ACTIVE_GEN_SEED = SEED * 1000 + 7
    _BUILT.clear()
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(arm, spec["layers"], steps, out_root)
        t0 = time.time()
        TDA.main()                       # ← 共享实现：解析/配置/train()/落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    # 训练期活性账：从刚训完的模型对象上读 SkipBlock 累计（共享 train() 不改）
    train_skip = None
    if _BUILT and arm != "stock" and arm != "layers6":
        m = _BUILT[-1]
        train_skip = {"by_layer": [
            {"layer": i + 1,
             "skip_frac": (round(b.train_skip_sum / b.train_skip_n, 6)
                           if b.train_skip_n else None),
             "n_forwards": b.train_skip_n}
            for i, b in enumerate(m.blocks)]}
        vals = [x["skip_frac"] for x in train_skip["by_layer"]
                if x["skip_frac"] is not None]
        train_skip["mean_skip_frac"] = (round(statistics.mean(vals), 6)
                                        if vals else None)
        train_skip["layer_passes_per_token"] = (
            round(len(m.blocks) - sum(vals), 6) if vals else None)
    _ACTIVE_ARM = None
    return {"wall_s": round(wall, 1), "train_skip_stats": train_skip}


# ---------------------------------------------------------------- 评测（本实验自持）

@torch.no_grad()
def evaluate_arm(arm: str, steps: int, out_root: str,
                 n_batches: int = EVAL_BATCHES,
                 batch_size: int = EVAL_BATCH) -> dict:
    """序列级评测：逐窗口 final-depth bpc + 逐 token layer-passes 账 + 逐层跳过占比。

    窗口定义 = 一个评测 batch（16 序列 × 255 预测位），固定 `EVAL_SEED` ⇒ 四臂
    逐窗口**同分布同位置**（配对 Δ 的前提）。跳层掩码按本臂分布采样（layerdrop
    逐窗口重采 ⇒ 落盘窗口间 SE）。口径与 2613/2641 的 `depth_curve` 最终深度一致。
    """
    global _ACTIVE_ARM, _ACTIVE_GEN_SEED
    spec = ARMS[arm]
    _ACTIVE_ARM = spec
    _ACTIVE_GEN_SEED = 0xE7 * 1000 + 7          # 评测采样种子（确定性）
    rd = run_dir(arm, steps, out_root)
    shared = json.load(open(os.path.join(rd, "results.json")))
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                  "data/enwik8_full.npz")
    # 评测模型必须用**训练 run 的 resolved config** 重建（而不是重推一份），
    # 否则任何字段漂移都会让 state_dict 键/形状不匹配（SCHEMA §9 配置层 equality）。
    cfg = DepthARConfig(**shared["config"])
    model = build_model(spec, vocab, _ACTIVE_GEN_SEED, cfg=cfg).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=DEVICE))
    model.eval()

    gen = torch.Generator().manual_seed(EVAL_SEED)
    per_window = []
    layer_skip_acc = [0.0] * spec["layers"]
    lp_acc = 0.0
    n_pos = 0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, SEQ_LEN, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        lg = aux["depth_logits"][-1][:, :-1].float()
        tgt = x[:, 1:]
        nats = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                               reduction="sum").item()
        n_win = tgt.numel()
        # layer-passes 账：本窗口 = Σ_层 (1 − 该层逐 token 跳过占比)
        wins_skip = [b.last_skip_mean for b in model.blocks]
        for i, s in enumerate(wins_skip):
            layer_skip_acc[i] += s * n_win
        lp_win = spec["layers"] - sum(wins_skip)
        lp_acc += lp_win * n_win
        n_pos += n_win
        per_window.append({"window": len(per_window), "bpc": round(nats / n_win / LN2, 6),
                           "layer_passes": round(lp_win, 6), "n_pos": n_win})

    clean_bpc = round(sum(w["bpc"] * w["n_pos"] for w in per_window) / n_pos, 6)
    layer_skip = [round(v / n_pos, 6) for v in layer_skip_acc]
    lp = round(lp_acc / n_pos, 6)
    params = sum(p.numel() for p in model.parameters())
    out = {
        "arm": arm, "seed": SEED, "steps": steps, "layers": spec["layers"],
        "mode": spec["mode"], "skip_layers_0idx": list(SKIP_LAYERS),
        "layerdrop_p": (LAYERDROP_P if spec["mode"] == "drop" else None),
        "eval": {"metric": "clean bpc（最终深度读出、序列级、越低越好）",
                 "n_windows": n_batches, "batch_size": batch_size,
                 "eval_generator_seed": EVAL_SEED,
                 "clean_bpc": clean_bpc,
                 "per_window": per_window},
        "ledger": {"layer_passes_per_token_measured": lp,
                   "layer_passes_per_token_design": spec["lp_design"],
                   "layer_passes_match_design": abs(lp - spec["lp_design"]) < 0.2,
                   "skip_frac_by_layer": layer_skip,
                   "mean_skip_frac": round(sum(layer_skip) / spec["layers"], 6)},
        "params": params,
        "train_skip_stats": shared.get("_train_skip_stats"),
        "shared_train": {"wall_minutes": shared.get("wall_minutes"),
                         "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
                         "train_tokens": shared.get("train_tokens"),
                         "gnorm_monitor_max": (shared.get("gnorm_monitor") or {})
                         .get("gnorm_max")},
        "run_dir": os.path.relpath(rd, ROOT),
        "model_pt_sha256": sha256_file(os.path.join(rd, "model.pt")),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    _ACTIVE_ARM = None
    return out


# ---------------------------------------------------------------- 汇总（P0–P5）

def load_results(out_root: str) -> dict:
    p = os.path.join(HERE, "results.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"experiment": "2654-skipping-seqlevel-13m",
            "variance_bound": VARIANCE_BOUND, "arms": {}}


def save_results(res: dict, out_root: str, steps: int) -> None:
    res["completed_arms"] = [a for a in ARM_ORDER if a in res["arms"]]
    res["pending_arms"] = [a for a in ARM_ORDER if a not in res["arms"]]
    res["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    res["code_hash_run_py"] = sha256_file(os.path.join(HERE, "run.py"))
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def paired_delta(res: dict, a: str, b: str) -> dict | None:
    """配对 Δbpc = bpc_b − bpc_a（**正 = a 更好**，SCHEMA §17.3 的符号统一）。
    逐窗口配对：两臂窗口一一对应（同一 EVAL_SEED）⇒ 报均值 + 窗口间 SE + 两半。"""
    if a not in res["arms"] or b not in res["arms"]:
        return None
    wa = res["arms"][a]["eval"]["per_window"]
    wb = res["arms"][b]["eval"]["per_window"]
    n = min(len(wa), len(wb))
    d = [wb[i]["bpc"] - wa[i]["bpc"] for i in range(n)]
    mean = round(statistics.mean(d), 6)
    se = round(statistics.stdev(d) / math.sqrt(n), 6) if n > 1 else None
    h = n // 2
    halves = ([round(statistics.mean(d[:h]), 6), round(statistics.mean(d[h:]), 6)]
              if h >= 2 else None)
    return {"delta_mean": mean, "delta_window_se": se, "n_windows": n,
            "delta_halves": halves,
            "crosses_band": bool(abs(mean) >= VARIANCE_BOUND)}


def aggregate(out_root: str, steps: int) -> dict:
    res = load_results(out_root)
    arms = res["arms"]
    verdicts = {}
    for key, (a, b) in {"skip-mid_vs_stock": ("skip-mid", "stock"),
                        "skip-mid_vs_layers6": ("skip-mid", "layers6"),
                        "layerdrop-pt_vs_skip-mid": ("layerdrop-pt", "skip-mid")}.items():
        d = paired_delta(res, a, b)
        if d is None:
            verdicts[key] = {"status": "pending", "pair": [b, a]}
            continue
        m = d["delta_mean"]
        if abs(m) < VARIANCE_BOUND:
            v = "不可排序（H0）：|Δ| < 0.014"
        elif key == "skip-mid_vs_layers6":
            v = ("可排序：skip-mid 在 6 layer-passes 预算下优于 6 层 dense"
                 if m > 0 else
                 "可排序（反向）：6 层 dense 优于 skip-mid ⇒ 跳中间层无独立增量")
        elif m > 0:
            v = f"可排序：{b} 优于 {a}"
        else:
            v = f"可排序（反向）：{a} 优于 {b}"
        verdicts[key] = {"status": "judged", "pair": [b, a], **d, "verdict": v}
    # P0 引擎等值门
    gate = None
    if "stock" in arms:
        sb = arms["stock"]["eval"]["clean_bpc"]
        gate = {"stock_clean_bpc": sb,
                "vs_2613_s0": round(sb - ANCHOR_2613_S0, 4),
                "vs_23d_dense_B": round(sb - ANCHOR_23D_DENSE_B, 4),
                "pass": bool(min(abs(sb - ANCHOR_2613_S0),
                                 abs(sb - ANCHOR_23D_DENSE_B)) < 0.005),
                "rule": "与任一 13M/12k 同配方锚之差 < 0.005 ⇒ 同引擎同配方"}
    # P1/P1b 活性：设计 vs 实测 layer-passes
    liveness = {a: {"layer_passes_design": r["ledger"]["layer_passes_per_token_design"],
                    "layer_passes_measured": r["ledger"]["layer_passes_per_token_measured"],
                    "match": r["ledger"]["layer_passes_match_design"],
                    "mean_skip_frac": r["ledger"]["mean_skip_frac"],
                    "skip_frac_by_layer": r["ledger"]["skip_frac_by_layer"],
                    "train_mean_skip_frac": ((r.get("train_skip_stats") or {})
                                             .get("mean_skip_frac"))}
                for a, r in arms.items()}
    # P3 训练/评测分布一致（layerdrop-pt）
    dist = None
    if "layerdrop-pt" in arms:
        tr = arms["layerdrop-pt"].get("train_skip_stats") or {}
        ev = arms["layerdrop-pt"]["ledger"]["skip_frac_by_layer"]
        trl = [x["skip_frac"] for x in (tr.get("by_layer") or [])]
        dist = {"train_skip_frac_by_layer": trl, "eval_skip_frac_by_layer": ev,
                "max_abs_diff": (round(max(abs(a - b) for a, b in zip(trl, ev)), 6)
                                 if trl and len(trl) == len(ev) else None),
                "p_design": LAYERDROP_P,
                "consistent": (bool(trl and len(trl) == len(ev)
                                    and max(abs(a - b) for a, b in zip(trl, ev)) < 0.05))}
    res.update({"aggregate": {"steps": steps, "verdicts": verdicts,
                              "engine_equivalence_gate": gate,
                              "liveness": liveness,
                              "layerdrop_distribution_consistency": dist,
                              "arm_summary": {
                                  a: {"clean_bpc": r["eval"]["clean_bpc"],
                                      "layer_passes": r["ledger"]["layer_passes_per_token_measured"],
                                      "params": r["params"]}
                                  for a, r in arms.items()}}})
    save_results(res, out_root, steps)
    return res


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 20 步冒烟

def smoke(steps: int = 20) -> dict:
    """全臂 20 步冒烟：训练走共享 main()，落 `smoke/`；断言活性与掩码语义。"""
    out_root = os.path.join(HERE, "smoke", "runs")
    os.makedirs(out_root, exist_ok=True)
    report = {}
    for arm in ARM_ORDER:
        info = train_arm(arm, steps, out_root)
        rd = run_dir(arm, steps, out_root)
        res = evaluate_arm(arm, steps, out_root, n_batches=2, batch_size=4)
        lp = res["ledger"]["layer_passes_per_token_measured"]
        ok = res["ledger"]["layer_passes_match_design"]
        assert ok, f"{arm}: layer-passes 实测 {lp} 与设计 {ARMS[arm]['lp_design']} 不符"
        report[arm] = {"train_wall_s": info["wall_s"], "layer_passes": lp,
                       "clean_bpc_smoke": res["eval"]["clean_bpc"],
                       "skip_frac_by_layer": res["ledger"]["skip_frac_by_layer"]}
        print(f"[smoke] {arm}: lp={lp} ok, train {info['wall_s']}s, "
              f"bpc={res['eval']['clean_bpc']}", flush=True)
    with open(os.path.join(HERE, "smoke", "smoke_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARM_ORDER),
                    help="逗号分隔的臂（按 ARM_ORDER 顺序执行）")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true", help="跳过训练，复用 model.pt")
    ap.add_argument("--force", action="store_true", help="已有臂也重跑")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--budget-min", type=float, default=1e9,
                    help="超过该 GPU 墙钟（分钟）后不再起新臂（预算保护）")
    args = ap.parse_args()
    # 共享 `train_depth_ar.main()` 与数据加载用**仓库根相对路径**（`data/…`、`.tmp/…`）
    # ⇒ 无论从哪个目录调用本入口，先 chdir 到仓库根（输出路径全为 HERE 绝对路径）。
    os.chdir(ROOT)

    if args.smoke:
        smoke(args.steps)
        return
    if args.aggregate:
        aggregate(args.out_root, args.steps)
        return

    want = [a for a in ARM_ORDER if a in args.arms.split(",")]
    if not want:
        raise SystemExit(f"--arms 必须给出 ARM_ORDER 的子集，收到 {args.arms!r}")
    os.makedirs(args.out_root, exist_ok=True)
    res = load_results(args.out_root)
    res.setdefault("arms", {})
    t0 = time.time()
    for arm in want:
        xsim = os.path.exists(os.path.join(run_dir(arm, args.steps, args.out_root),
                                           "model.pt"))
        if arm in res["arms"] and not args.force and not args.no_train:
            print(f"[2654] {arm}: 已有结果，跳过（--force 可重跑）", flush=True)
            continue
        if not args.no_train and not xsim:
            if (time.time() - t0) / 60 > args.budget_min:
                print(f"[2654] 预算 {args.budget_min} min 用尽，停在 {arm} 前", flush=True)
                break
            info = train_arm(arm, args.steps, args.out_root)
            rd = run_dir(arm, args.steps, args.out_root)
            with open(os.path.join(rd, "train_wall.json"), "w") as f:
                json.dump(info, f)
            rec = evaluate_arm(arm, args.steps, args.out_root)
            rec["train_skip_stats"] = info["train_skip_stats"]
            res["arms"][arm] = rec
            save_results(res, args.out_root, args.steps)   # ← 逐臂增量落盘
            print(f"[2654] {arm}: clean_bpc={rec['eval']['clean_bpc']} "
                  f"lp={rec['ledger']['layer_passes_per_token_measured']} "
                  f"(train {info['wall_s']}s)", flush=True)
        else:
            rec = evaluate_arm(arm, args.steps, args.out_root)
            res["arms"][arm] = rec
            save_results(res, args.out_root, args.steps)
            print(f"[2654] {arm}: 仅评测 clean_bpc={rec['eval']['clean_bpc']} "
                  f"lp={rec['ledger']['layer_passes_per_token_measured']}", flush=True)
    aggregate(args.out_root, args.steps)


if __name__ == "__main__":
    main()
