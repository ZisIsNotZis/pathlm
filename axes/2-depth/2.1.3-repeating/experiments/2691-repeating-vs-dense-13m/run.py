"""2691 · 循环深度（同权重重复前向）对「同算力稠密」与「同参数稠密」· 13M · 工厂替换。

契约 = 同目录 `README.md`（判据线 P0–P5 **预登记、不可改**）。本文件是唯一入口。

设计要点：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成一个**工厂**（返回 `RepeatModel`：同一个 stock 模型，但前向按本臂的
  **层调度表** `ARMS[臂]["schedule"]` 依次调用各块），然后调用
  `train_depth_ar.main()` —— 参数解析、配置装配、`train()` 循环、checkpoint、
  `results.json` 全部是**共享实现**，本文件不改共享代码（`pathlm/**` /
  `train_depth_ar.py` 零改动）。
* **单因素**：`dense6` / `dense12` 的调度是恒等（0,1,…,n−1），与 stock 逐位等价；
  `repeat2x` 把 6 组层的下标各用两次；`repeat2x-tail` 前 3 层用一次、后 3 层各用三次。
  唯一变量 = 「每层被调用几次 / 哪几层被重复」。
* **等参数与等算力都是真训练出来的臂**，不是解析换算：`dense6` ↔ `repeat2x`
  参数量逐位相同（组数相同、同 seed 初始化也逐位相同），`dense12` ↔ `repeat2x`
  每 token 解析 MACs 逐位相同（传递次数与读出个数都相同）。
* **不加步嵌入**（README §2 设计条件 4）：只有残差流携带状态，所以结论的边界是
  「本设计下」。

用法：

    python3 run.py --arms dense6,dense12,repeat2x,repeat2x-tail   # 训练+评测，逐臂落盘
    python3 run.py --arms repeat2x --no-train                    # 仅评测（复用 model.pt）
    python3 run.py --p0-only                                     # 只跑 P0 历史锚重评
    python3 run.py --smoke --out-root .tmp/2691-smoke            # 20 步冒烟（四臂）
    python3 run.py --aggregate                                   # 汇总 P0/P1/P2 → results.json

本文件不取 GPU 锁；训练由驱动脚本 `run_arms.sh` 按臂取/放锁（`harness/SCHEMA.md` §20.1）。
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402
from pathlm.data import load_enwik8_full, batch               # noqa: E402
from tools.run_provenance import write_results                # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 契约常量（README §2 / §4）

D_MODEL, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 256, 4
BATCH_SIZE, LR, STEPS, SEED = 32, 1e-3, 12000, 0
VARIANCE_BOUND = 0.014
EVAL_BATCHES, EVAL_BATCH = 20, 16          # 20 窗口（每个 = batch16 × seq256 = 4080 预测位）
EVAL_SEED = 2                              # 与 2613 / 2641 / 2654 同款评测抽样口径

# P0 引擎等值锚：2654 的 stock 臂（8 层 / 12k 步 / seed 0；读数它的 results.json 里）
ANCHOR_EXP = "axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m"
ANCHOR_RUN = "runs/stock-12k-s0"
ANCHOR_TOL = 0.002

# ---------------------------------------------------------------- 臂定义（契约 §2）

ARMS = {
    "dense6": {
        "blocks": 6, "passes": 6,
        "schedule": (0, 1, 2, 3, 4, 5),
        "role": "问二对照：6 组参数跑 6 次（同参数稠密）"},
    "dense12": {
        "blocks": 12, "passes": 12,
        "schedule": tuple(range(12)),
        "role": "问一对照：12 组参数各跑一次（同算力稠密）"},
    "repeat2x": {
        "blocks": 6, "passes": 12,
        "schedule": (0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5),
        "role": "主臂：6 组参数各跑两次（同参数、同算力）"},
    "repeat2x-tail": {
        "blocks": 6, "passes": 12,
        "schedule": (0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5),
        "role": "诊断臂：只重复后半段（前 3 层各一次、后 3 层各三次；同参数、同算力）"},
}
ARM_ORDER = ["dense6", "dense12", "repeat2x", "repeat2x-tail"]
VOCAB_FALLBACK = 206

# ---------------------------------------------------------------- 解析算力（契约 §2）

def macs_per_token(passes: int, vocab_size: int = VOCAB_FALLBACK) -> int:
    """每 token 的解析 MACs：每次层传递 `12·d² + T·d`，每个深度读出 `d·V`。

    传递次数 = passes，读出个数 = passes + 1（0..passes 每个深度一次）⇒ 同 passes
    的两臂读出算力也相同（`repeat2x` 与 `dense12` 的每 token MACs 逐位相等）。
    """
    per_pass = 12 * D_MODEL * D_MODEL + SEQ_LEN * D_MODEL
    per_readout = D_MODEL * vocab_size
    return passes * per_pass + (passes + 1) * per_readout


# ---------------------------------------------------------------- 干预：层调度表

class RepeatModel(DepthARModel):
    """按**层调度表**执行 stock 模型的层：`schedule[i]` = 第 i 次传递用的块下标。

    与父类 `DepthARModel` 在 dense（`topology="sequential"`）路径上的唯一差别 =
    块的调用顺序/次数；`_input_states`、逐深度读出、损失公式与父类逐字一致。
    不新增任何参数 ⇒ `state_dict` 与同 `n_layers` 的 stock 模型逐位相同。

    `schedule` 为恒等时与父类**逐位等价**（单测 `test_repeat_schedule.py` 与
    `results.json::gates.p1_liveness.identity_equals_stock` 都钉住）。
    """

    def __init__(self, cfg: DepthARConfig, vocab_size: int, schedule):
        super().__init__(cfg, vocab_size)
        self.schedule = tuple(int(i) for i in schedule)
        n = cfg.n_layers
        for idx in self.schedule:
            if not 0 <= idx < n:
                raise ValueError(
                    f"schedule 里的块下标 {idx} 越界（本模型 {n} 组参数）")
        self.passes = len(self.schedule)
        #: 前向里真实调用块的总次数（活性量：每前向应 == passes）
        self.block_calls = 0

    def forward(self, tokens: torch.Tensor, exit_depths: torch.Tensor | None = None,
                targets: torch.Tensor | None = None,
                shallow_weight: float = 1.0, final_weight: float = 1.0,
                shallow_stopgrad: bool = False, dense_aux: bool = False):
        if not self.cfg.dense:
            raise NotImplementedError(
                "RepeatModel 只在 dense 形态（stock decoder-only）下验证：变深度/"
                "填充路径的「层传递次数」语义未定义（单因素）")
        B, T = tokens.shape
        h = self._input_states(tokens)
        states = [h]
        for idx in self.schedule:
            h, _tele = self.blocks[idx](h)
            self.block_calls += 1
            states.append(h)
        targets = tokens if targets is None else targets
        n_states = len(states)
        sg = bool(shallow_stopgrad)
        depth_logits = [self.readout(
            s.detach() if sg and k < n_states - 1 else s, depth=k)
            for k, s in enumerate(states)]
        readouts = depth_logits if dense_aux else depth_logits[-1:]
        w = float(shallow_weight)
        loss = tokens.new_zeros(()).float()
        depth_ce: list[torch.Tensor] = []
        for k, logits in enumerate(readouts):
            mask = torch.ones(B, T - 1, dtype=torch.bool, device=tokens.device)
            ce = self._masked_ce(logits[:, :-1], targets[:, 1:], mask)
            depth_ce.append(ce)
            if k == len(readouts) - 1:
                wk = final_weight
            elif self.cfg.aux_mode == "additive":
                wk = w / max(len(readouts) - 1, 1)
            else:
                wk = w
            loss = loss + wk * ce
        if self.cfg.aux_mode == "renorm":
            loss = loss / (w * (len(readouts) - 1) + final_weight)
        aux = {"states": states, "depth_logits": depth_logits,
               "depth_ce": depth_ce, "gate_mean": [], "exit_depths": exit_depths}
        return loss, aux


def spec_cfg(spec: dict) -> DepthARConfig:
    """按本实验契约常量构造配置（`--smoke` 与 CPU 自检用；训练侧用共享 CLI）。"""
    return DepthARConfig(d_model=D_MODEL, n_layers=spec["blocks"],
                         n_heads=N_HEADS, seq_len=SEQ_LEN, mlp_mult=MLP_MULT,
                         dense=True, topology="sequential")


def build_model(spec: dict, vocab_size: int, cfg: DepthARConfig | None = None):
    """按 spec 建一个 `RepeatModel`。`cfg` 给定则用之（评测侧 = 训练 run 的 resolved config）。"""
    return RepeatModel(spec_cfg(spec) if cfg is None else cfg, vocab_size,
                       spec["schedule"])


# 工厂读取的「当前臂」（train / eval 各设一次；与 2641 / 2654 同款）
_ACTIVE_SPEC: dict | None = None
_BUILT: list = []


def _factory(cfg, vocab_size):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    model = RepeatModel(cfg, vocab_size, _ACTIVE_SPEC["schedule"])
    _BUILT.append(model)
    return model


# ---------------------------------------------------------------- 数据（进程内只读一次）

_DATA: tuple | None = None


def load_data():
    global _DATA
    if _DATA is None:
        _DATA = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    return _DATA


# ---------------------------------------------------------------- 训练（走共享 main）

def run_dir(arm: str, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-{steps // 1000}k-s{SEED}")


def shared_argv(arm: str, spec: dict, steps: int, out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = 层调度表）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-{steps // 1000}k-s{SEED}",
            "--d", str(D_MODEL), "--layers", str(spec["blocks"]),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(BATCH_SIZE), "--lr", str(LR), "--seed", str(SEED),
            "--eval-batches", str(EVAL_BATCHES), "--out-root", out_root]


def train_arm(arm: str, steps: int, out_root: str) -> dict:
    global _ACTIVE_SPEC
    spec = ARMS[arm]
    _ACTIVE_SPEC = spec
    _BUILT.clear()
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(arm, spec, steps, out_root)
        t0 = time.time()
        TDA.main()                       # ← 共享实现：解析/配置/train()/落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    calls = (_BUILT[-1].block_calls if _BUILT else None)
    _ACTIVE_SPEC = None
    return {"wall_s": round(wall, 1),
            "train_block_calls_total": calls,
            "train_block_calls_per_forward": (round(calls / steps, 6)
                                              if calls is not None else None)}


# ---------------------------------------------------------------- 评测（本实验自持）

@torch.no_grad()
def run_eval(model, n_batches: int = EVAL_BATCHES,
             batch_size: int = EVAL_BATCH) -> dict:
    """序列级评测：逐窗口最终深度 bpc + 逐深度 bpc + 层调用计数。

    窗口定义 = 一个评测 batch（16 序列 × 255 预测位），固定 `EVAL_SEED` ⇒ 各臂
    逐窗口**同分布同位置**（配对 Δ 的前提）。口径 = 最终深度读出（与 2613 / 2641 /
    2654 的 `depth_curve` 最终深度同口径），fp32、无 autocast（与 P0 锚同一台仪器）。
    """
    _, eval_arr, vocab = load_data()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    model.eval()
    model.block_calls = 0
    per_window = []
    depth_nats: list[float] | None = None
    n_pos = 0
    for i in range(n_batches):
        x, _ = batch(eval_arr, batch_size, SEQ_LEN, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        logs = aux["depth_logits"]
        if depth_nats is None:
            depth_nats = [0.0] * len(logs)
        tgt = x[:, 1:]
        n_win = tgt.numel()
        nats = F.cross_entropy(logs[-1][:, :-1].float().reshape(-1, vocab),
                               tgt.reshape(-1), reduction="sum").item()
        for k, lg in enumerate(logs):
            depth_nats[k] += F.cross_entropy(
                lg[:, :-1].float().reshape(-1, vocab), tgt.reshape(-1),
                reduction="sum").item()
        per_window.append({"window": i, "bpc": round(nats / n_win / LN2, 6),
                           "n_pos": n_win})
        n_pos += n_win
    calls = model.block_calls
    clean = round(sum(w["bpc"] * w["n_pos"] for w in per_window) / n_pos, 6)
    return {"clean_bpc": clean, "per_window": per_window,
            "depth_bpc": [round(v / n_pos / LN2, 6) for v in depth_nats],
            "n_positions": n_pos, "n_windows": n_batches,
            "block_calls_total": calls,
            "block_calls_per_forward": round(calls / n_batches, 6)}


@torch.no_grad()
def liveness_probe(model, spec: dict, batch_size: int = 4) -> dict:
    """重复是否活着：同一 checkpoint、同一输入，「完整调度」vs「只跑前 `blocks` 次」。

    全等的两臂（`dense6` / `dense12`）调度本来就等于自己的前缀 ⇒ 偏差恒 0，
    该字段对它们记 `None`（不是活性量为 0，而是「无重复可测」）。
    """
    _, eval_arr, _vocab = load_data()
    gen = torch.Generator().manual_seed(EVAL_SEED)
    x, _ = batch(eval_arr, batch_size, SEQ_LEN, gen)
    x = x.to(DEVICE)
    model.eval()
    full = model(x, exit_depths=None)[1]["depth_logits"][-1].float().clone()
    saved = model.schedule
    prefix = tuple(range(min(spec["blocks"], len(saved))))
    differs = list(saved) != list(prefix)
    if not differs:
        return {"schedule_differs_from_prefix": False,
                "final_logits_rel_dev_vs_prefix": 0.0, "alive": None}
    model.schedule = prefix
    model.block_calls = 0
    pref = model(x, exit_depths=None)[1]["depth_logits"][-1].float().clone()
    model.schedule = saved
    rel = float((full - pref).norm() / full.norm().clamp_min(1e-6))
    return {"schedule_differs_from_prefix": True,
            "final_logits_rel_dev_vs_prefix": round(rel, 8),
            "alive": bool(rel > 0.0)}


def evaluate_arm(arm: str, steps: int, out_root: str) -> dict:
    spec = ARMS[arm]
    _, _eval_arr, vocab = load_data()
    rd = run_dir(arm, steps, out_root)
    shared = json.load(open(os.path.join(rd, "results.json")))
    # 用**训练 run 的 resolved config** 重建（任何字段漂移都会让 state_dict 不匹配）
    cfg = DepthARConfig(**shared["config"])
    model = build_model(spec, vocab, cfg=cfg).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=DEVICE))
    ev = run_eval(model)
    live = liveness_probe(model, spec)
    params = sum(p.numel() for p in model.parameters())
    return {
        "arm": arm, "seed": SEED, "steps": steps,
        "blocks": spec["blocks"], "passes": spec["passes"],
        "schedule": list(spec["schedule"]), "role": spec["role"],
        "eval": {"metric": "clean bpc（最终深度读出、序列级、越低越好）",
                 "clean_bpc": ev["clean_bpc"],
                 "n_windows": ev["n_windows"], "batch_size": EVAL_BATCH,
                 "eval_generator_seed": EVAL_SEED, "n_positions": ev["n_positions"],
                 "per_window": ev["per_window"], "depth_bpc": ev["depth_bpc"]},
        "ledger": {"passes_design": spec["passes"],
                   "block_calls_per_forward_measured":
                       ev["block_calls_per_forward"],
                   "passes_match_design":
                       abs(ev["block_calls_per_forward"] - spec["passes"]) < 1e-9},
        "liveness": live, "params": params,
        "macs_per_token": macs_per_token(spec["passes"], vocab),
        "shared_train": {"wall_minutes": shared.get("wall_minutes"),
                         "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
                         "train_tokens": shared.get("train_tokens"),
                         "gnorm_monitor_max": (shared.get("gnorm_monitor") or {})
                         .get("gnorm_max"),
                         "gnorm_skipped_steps": (shared.get("gnorm_monitor") or {})
                         .get("n_grad_skip")},
        "run_dir": os.path.relpath(rd, ROOT),
        "model_pt_sha256": sha256_file(os.path.join(rd, "model.pt")),
    }


def evaluate_anchor() -> dict:
    """P0：用**本实验的评测代码**重评 2654 的 stock checkpoint，与它落盘的值比。

    该 checkpoint 由 2654 的 `SkipBlock`（mode="none"，逐位透传）包过一层，
    所以重建必须用它自己的 `build_model`（只读 import，不改它）；评测循环用本文件的
    `run_eval`。差 < `ANCHOR_TOL` ⇒ 本实验的评测仪器与历史读数同一台。
    """
    exp = os.path.join(ROOT, ANCHOR_EXP)
    run = os.path.join(exp, ANCHOR_RUN)
    pt = os.path.join(run, "model.pt")
    out = {"anchor_experiment": ANCHOR_EXP, "anchor_run": ANCHOR_RUN,
           "tolerance": ANCHOR_TOL}
    if not os.path.exists(pt):
        out.update({"status": "not_run",
                    "reason": f"锚 checkpoint 不在本机：{os.path.relpath(pt, ROOT)}"
                              "（权重不入库）⇒ 仪器未复核，显式缺失"})
        return out
    shared = json.load(open(os.path.join(run, "results.json")))
    recorded = (json.load(open(os.path.join(exp, "results.json")))
                ["arms"]["stock"]["eval"]["clean_bpc"])
    cfg = DepthARConfig(**shared["config"])
    spec = importlib.util.spec_from_file_location(
        "anchor_2654_run", os.path.join(exp, "run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, _eval_arr, vocab = load_data()
    model = mod.build_model(mod.ARMS["stock"], vocab, 0, cfg=cfg).to(DEVICE)
    model.load_state_dict(torch.load(pt, map_location=DEVICE))
    ev = run_eval(model)
    diff = round(ev["clean_bpc"] - recorded, 6)
    out.update({"status": "judged", "recorded_clean_bpc": recorded,
                "reevaluated_clean_bpc": ev["clean_bpc"], "diff": diff,
                "pass": bool(abs(diff) < ANCHOR_TOL),
                "recorded_at": os.path.relpath(
                    os.path.join(exp, "results.json"), ROOT)})
    return out


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 结构自检（CPU）

def structure_checks() -> dict:
    """参数与起始条件的结构事实：等参数、同初始化、恒等调度等值、调度覆盖 config。

    只构造模型、不读数据（vocab 取契约常量，与 `config.json::data.vocab_size` 一致）。
    """
    vocab = VOCAB_FALLBACK
    out: dict = {}

    def sd_hash(model) -> str:
        h = hashlib.sha256()
        for k, v in sorted(model.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def build_seeded(spec):
        torch.manual_seed(SEED)
        return build_model(spec, vocab)

    m6 = build_seeded(ARMS["dense6"])
    mrep = build_seeded(ARMS["repeat2x"])
    mtail = build_seeded(ARMS["repeat2x-tail"])
    m12 = build_seeded(ARMS["dense12"])
    out["params"] = {a: sum(p.numel() for p in m.parameters())
                     for a, m in (("dense6", m6), ("repeat2x", mrep),
                                  ("repeat2x-tail", mtail), ("dense12", m12))}
    out["params_equal_dense6_vs_repeat2x"] = (out["params"]["dense6"]
                                              == out["params"]["repeat2x"])
    out["params_equal_dense6_vs_tail"] = (out["params"]["dense6"]
                                          == out["params"]["repeat2x-tail"])
    out["init_hash_equal_dense6_vs_repeat2x"] = sd_hash(m6) == sd_hash(mrep)
    out["init_hash_equal_dense6_vs_tail"] = sd_hash(m6) == sd_hash(mtail)

    # 恒等调度 == stock（逐位）
    cfg = spec_cfg(ARMS["dense6"])
    torch.manual_seed(1234)
    stock = DepthARModel(cfg, vocab)
    # 同一套权重（把 stock 的 state_dict 装进 RepeatModel）⇒ 唯一差别只剩前向接线
    m6.load_state_dict(stock.state_dict())
    x = torch.randint(0, vocab, (2, 32),
                      generator=torch.Generator().manual_seed(7))
    stock.eval()
    m6.eval()
    with torch.no_grad():
        ls = stock(x)[1]["depth_logits"][-1]
        lr = m6(x)[1]["depth_logits"][-1]
    out["identity_equals_stock_max_abs_diff"] = round(
        float((ls - lr).abs().max()), 10)
    out["identity_equals_stock"] = bool(
        out["identity_equals_stock_max_abs_diff"] == 0.0)

    # 调度覆盖：本文件的 ARMS 与绑定的 config.json 一致（防止两处漂移）
    cfg_doc = json.load(open(os.path.join(HERE, "config.json")))
    cfg_arms = {a["arm"]: a for a in cfg_doc["arms"]}
    out["schedules_match_config"] = all(
        cfg_arms[a]["blocks"] == ARMS[a]["blocks"]
        and list(cfg_arms[a]["schedule"]) == list(ARMS[a]["schedule"])
        for a in ARM_ORDER)
    out["arm_roles"] = {a: ARMS[a]["role"] for a in ARM_ORDER}
    return out


# ---------------------------------------------------------------- 汇总（P0 / P1 / P2）

def load_results() -> dict:
    p = os.path.join(HERE, "results.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {"experiment": "2691-repeating-vs-dense-13m",
            "variance_bound": VARIANCE_BOUND, "arms": {}}


def save_results(res: dict) -> None:
    res["completed_arms"] = [a for a in ARM_ORDER if a in res["arms"]]
    res["pending_arms"] = [a for a in ARM_ORDER if a not in res["arms"]]
    res["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # 读数落盘必须自带代码哈希（`harness/methodology.md` §8）：走同一入口。
    write_results(os.path.join(HERE, "results.json"), res, source=__file__)


def paired_delta(res: dict, candidate: str, control: str) -> dict:
    """配对 Δbpc = bpc_对照 − bpc_候选（**正 = 候选更好**，SCHEMA §17.3 的符号统一）。

    逐窗口配对：两臂窗口一一对应（同一 `EVAL_SEED`）⇒ 报均值 + 窗口间 SE + 两半样本。
    """
    if candidate not in res["arms"] or control not in res["arms"]:
        return {"status": "pending", "pair": [candidate, control]}
    wc = res["arms"][candidate]["eval"]["per_window"]
    wk = res["arms"][control]["eval"]["per_window"]
    n = min(len(wc), len(wk))
    d = [wk[i]["bpc"] - wc[i]["bpc"] for i in range(n)]
    mean = round(statistics.mean(d), 6)
    se = round(statistics.stdev(d) / math.sqrt(n), 6) if n > 1 else None
    h = n // 2
    halves = ([round(statistics.mean(d[:h]), 6),
               round(statistics.mean(d[h:]), 6)] if h >= 2 else None)
    if abs(mean) < VARIANCE_BOUND:
        verdict = f"不可排序（|Δ| {abs(mean):.6f} < {VARIANCE_BOUND}）"
    elif mean > 0:
        verdict = (f"可排序：{candidate} 更好（越带 {abs(mean) / VARIANCE_BOUND:.2f}×）")
    else:
        verdict = (f"可排序（反向）：{candidate} 更差（越负带 "
                   f"{abs(mean) / VARIANCE_BOUND:.2f}×）")
    return {"status": "judged", "pair": [candidate, control],
            "delta_mean": mean, "delta_window_se": se, "n_windows": n,
            "delta_halves": halves, "band": VARIANCE_BOUND,
            "crosses_band": bool(abs(mean) >= VARIANCE_BOUND),
            "verdict": verdict}


def aggregate() -> dict:
    res = load_results()
    res.setdefault("arms", {})
    structure = structure_checks()
    res["gates"] = {
        "p0_engine": evaluate_anchor(),
        "p1_liveness": {
            a: {"passes_design": r["ledger"]["passes_design"],
                "block_calls_per_forward_measured":
                    r["ledger"]["block_calls_per_forward_measured"],
                "passes_match_design": r["ledger"]["passes_match_design"],
                "repeat_alive": r["liveness"]["alive"],
                "final_logits_rel_dev_vs_prefix":
                    r["liveness"]["final_logits_rel_dev_vs_prefix"]}
            for a, r in res["arms"].items()},
        "structure": structure,
    }
    res["comparisons"] = {
        "q1_same_compute": paired_delta(res, "repeat2x", "dense12"),
        "q2_same_params": paired_delta(res, "repeat2x", "dense6"),
        "d1_tail_vs_repeat2x": paired_delta(res, "repeat2x-tail", "repeat2x"),
        "d2_tail_vs_dense12": paired_delta(res, "repeat2x-tail", "dense12"),
    }
    res["arm_summary"] = {
        a: {"clean_bpc": r["eval"]["clean_bpc"],
            "passes": r["ledger"]["passes_design"],
            "params": r["params"],
            "macs_per_token": r["macs_per_token"],
            "peak_gpu_mem_mb": r["shared_train"]["peak_gpu_mem_mb"],
            "wall_minutes": r["shared_train"]["wall_minutes"],
            "depth_bpc": r["eval"]["depth_bpc"]}
        for a, r in res["arms"].items()}
    # Δ 方向声明（harness/research-schema.md §2.3）：全部 = 对照 − 候选
    res["delta_directions"] = {"delta_mean": "comparator_minus_arm",
                               "delta_window_se": "comparator_minus_arm",
                               "delta_halves": "comparator_minus_arm"}
    save_results(res)
    return res


# ---------------------------------------------------------------- 20 步冒烟

def smoke(steps: int, out_root: str) -> dict:
    """四臂 20 步冒烟：训练走共享 main()，断言活性（层调用次数）与结构等值。"""
    os.makedirs(out_root, exist_ok=True)
    structure = structure_checks()
    assert structure["identity_equals_stock"], "恒等调度与 stock 不等值"
    assert structure["params_equal_dense6_vs_repeat2x"], "等参数不成立"
    assert structure["schedules_match_config"], "ARMS 与 config.json 漂移"
    report = {"structure": structure, "arms": {}}
    for arm in ARM_ORDER:
        info = train_arm(arm, steps, out_root)
        rec = evaluate_arm(arm, steps, out_root)
        assert rec["ledger"]["passes_match_design"], (
            f"{arm}: 层调用次数 {rec['ledger']['block_calls_per_forward_measured']}"
            f" != 设计 {rec['ledger']['passes_design']}")
        if ARMS[arm]["schedule"] != tuple(range(ARMS[arm]["blocks"])):
            assert rec["liveness"]["alive"], f"{arm}: 重复是死的（输出与只跑前缀相同）"
        report["arms"][arm] = {"train_wall_s": info["wall_s"],
                               "clean_bpc_smoke": rec["eval"]["clean_bpc"],
                               "block_calls": rec["ledger"][
                                   "block_calls_per_forward_measured"],
                               "liveness": rec["liveness"]}
        print(f"[smoke] {arm}: passes={rec['ledger']['block_calls_per_forward_measured']}"
              f" ok, bpc={rec['eval']['clean_bpc']}, train {info['wall_s']}s",
              flush=True)
    write_results(os.path.join(HERE, "smoke_report.json"), report, source=__file__)
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
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--budget-min", type=float, default=1e9,
                    help="超过该 GPU 墙钟（分钟）后不再起新臂（预算保护）")
    args = ap.parse_args()
    # 共享 `train_depth_ar.main()` 与数据加载用**仓库根相对路径**（`data/…`、`.tmp/…`）
    os.chdir(ROOT)

    if args.smoke:
        smoke(args.steps, args.out_root)
        return
    if args.p0_only:
        res = load_results()
        res.setdefault("arms", {})
        res.setdefault("gates", {})["p0_engine"] = evaluate_anchor()
        save_results(res)
        print(json.dumps(res["gates"]["p0_engine"], indent=2, ensure_ascii=False))
        return
    if args.aggregate:
        aggregate()
        return

    want = [a for a in ARM_ORDER if a in args.arms.split(",")]
    if not want:
        raise SystemExit(f"--arms 必须给出 ARM_ORDER 的子集，收到 {args.arms!r}")
    os.makedirs(args.out_root, exist_ok=True)
    res = load_results()
    res.setdefault("arms", {})
    t0 = time.time()
    for arm in want:
        xsim = os.path.exists(os.path.join(
            run_dir(arm, args.steps, args.out_root), "model.pt"))
        if arm in res["arms"] and not args.force and not args.no_train:
            print(f"[2691] {arm}: 已有结果，跳过（--force 可重跑）", flush=True)
            continue
        if not args.no_train and not xsim:
            if (time.time() - t0) / 60 > args.budget_min:
                print(f"[2691] 预算 {args.budget_min} min 用尽，停在 {arm} 前", flush=True)
                break
            info = train_arm(arm, args.steps, args.out_root)
            rd = run_dir(arm, args.steps, args.out_root)
            with open(os.path.join(rd, "train_wall.json"), "w") as f:
                json.dump(info, f)
            rec = evaluate_arm(arm, args.steps, args.out_root)
        else:
            info = {"wall_s": None}
            rec = evaluate_arm(arm, args.steps, args.out_root)
        rec["shared_train"]["train_block_calls_per_forward"] = info.get(
            "train_block_calls_per_forward")
        res["arms"][arm] = rec
        save_results(res)                     # ← 逐臂增量落盘
        print(f"[2691] {arm}: clean_bpc={rec['eval']['clean_bpc']} "
              f"passes={rec['ledger']['block_calls_per_forward_measured']} "
              f"params={rec['params']}", flush=True)
    aggregate()


if __name__ == "__main__":
    main()
