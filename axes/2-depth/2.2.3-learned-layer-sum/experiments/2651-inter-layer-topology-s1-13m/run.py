"""2651 · 层间拓扑 seed=1 复核 + 并行子层代价面 · 13M · depth-ar/dense —— 复现入口。

契约 = 同目录 `README.md`（预登记；判据线**逐字**来自 ticket 61 §9.2 WP-2651，
不得事后调整）。本文件是唯一入口：一条命令 `python3 run.py` 复现全臂。

单因素：**层与层之间的接法**（+ 本次唯一的第二因素 = 训练 seed，用于
2.2.3 的 n≥2）。四臂与 2613「契约 E」**逐字段相同，只把 seed 0 → 1**：

- `sequential`        h = h + attn(ln1 h)；h = h + mlp(ln2 h)（现有 Block，基线）；
- `parallel`          h = h + attn(ln h) + mlp(ln h)（单个共享 pre-LN，PaLM/GPT-J
                      原式；参数 −L·2d = −4096）；
- `parallel-dual-ln`  两个子层各带自己的 pre-LN、都读同一块输入 h（**与基线参数量
                      逐位相同**的等参数位点，SCHEMA §15）；
- `dense`             层 k 输入 = Σ_{j<k} α_kj·h_j（DenseFormer 式），α 初始化为
                      「只取当前层」（起步与基线逐位等价）；参数 +L(L+1)/2 = +36。

判读（ticket 61 §9.2 WP-2651，逐字未改）：

- 2.2.3：s0/s1 两 seed 的 **dense − sequential** 配对 Δ 是否越 0.014
  （不越 ⇒「无差异」可下；两 seed 同号且越带 ⇒「可排序」）。
- 2.2.4：`parallel` / `parallel-dual-ln` 相对 `sequential` 的质量代价
  （>0.014 即代价可排序）**与**代价面同框（层内串行依赖 2 段 → 1 段；
  参数量 −2d 与逐位等参对照两臂并列）；报「质量代价是否换到可兑现的等效收益」。

用法（后台运行 + ≤120s 轮询日志）：

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py > run_log.txt 2>&1 &
    tail -f run_log.txt

只重算电池（复用 runs/ 下 checkpoint，需从仓库根目录运行）：

    python3 run.py --skip-train
"""

import argparse, atexit, json, math, os, shutil, statistics, subprocess, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, ParallelBlock,
                             depth_curve, tf_acceptance, spec_cost_model,
                             topology_liveness)
from pathlm.data import load_enwik8_full, batch
from pathlm.metrics import ece

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP, BS, LR = 256, 8, 8, 256, 4, 32, 1e-3
LN2 = math.log(2.0)
SEED = 1
STEPS = 12000
# 同引擎 dense 13M matched 税分母（历史 23d-dense-B-12k，口径见 2613 analysis §7）
DENSE_BPC_13M = 1.5287
# s0（2613）对照目录：本实验的 2.2.3/2.2.4 首 seed 读数与 s0 checkpoint 重测都取此
S0_EXP = os.path.abspath(os.path.join(
    HERE, "..", "2613-inter-layer-topology-13m"))

ARMS = [
    {"topology": "sequential", "role": "基线（现有 Block：链式残差）"},
    {"topology": "parallel",
     "role": "并行子层（单个共享 pre-LN；参数 −L·2d）"},
    {"topology": "parallel-dual-ln",
     "role": "并行子层（两子层各带 pre-LN；与基线逐位等参数）"},
    {"topology": "dense", "role": "层间学习加权和（DenseFormer 式；参数 +36）"},
]
# 全臂名录（`--arms` 只改 ARMS，本常量不变 ⇒ 增量落盘能如实报 pending）
ALL_ARMS = list(ARMS)


def cfg_base(topo: str) -> DepthARConfig:
    """骨架：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True, topology=topo)


def run_name(topo: str) -> str:
    return f"{topo}-12k-s1"


RUN_LOG = os.path.join(HERE, "run_log.txt")


def append_run_log(line: str) -> None:
    """结构化逐臂流水：每跑完一臂 append 一行（`--arms` 增量跑时即使本次只
    完成部分臂也留下可读记录）。"""
    with open(RUN_LOG, "a") as f:
        f.write(line.rstrip("\n") + "\n")


def dump_results(out: dict, note: str = "") -> None:
    """增量落盘：每臂跑完立刻写 `results.json`，崩溃/预算中止也留下已完成的臂。"""
    out["completed_arms"] = [a["topology"] for a in ALL_ARMS
                             if a["topology"] in out.get("arms", {})]
    out["pending_arms"] = [a["topology"] for a in ALL_ARMS
                           if a["topology"] not in out.get("arms", {})]
    if note:
        out["partial_note"] = note
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)


# --------------------------- 2.2.4 代价面（等效理论量，禁墙钟） ---------------------------

def topology_cost_face(d: int = D, n_layers: int = L, mlp_mult: int = MLP,
                       seq_len: int = SEQ, vocab: int = 206) -> dict:
    """每层每 token 的等效 MACs 与层内串行依赖长度（解析，不含任何时间量）。

    - 注意力：qkv 3d² + proj d² + QKᵀ T·d + AV T·d = 4d² + 2Td；
    - MLP：2·mlp_mult·d²（d→4d 与 4d→d）；
    - pre-LN：≈2d MACs/token（量级 d，相对上面可忽略，逐臂如实列出）；
    - 层内串行依赖：`sequential` MLP 读注意力写回后的状态 ⇒ attn→MLP **2 段**；
      `parallel` / `parallel-dual-ln` 两分支同读块输入 ⇒ **1 段**；
    - `dense` 另加层间加权和 Σ_k (k−1)·d MACs/token（总 L(L−1)/2·d）。
    参数量 Δ 相对 sequential（−L·2d / 0 / +L(L+1)/2）由 `topology_liveness` 报。
    """
    d2 = d * d
    attn = 4 * d2 + 2 * seq_len * d
    mlp = 2 * mlp_mult * d2
    ln = 2 * d                                  # 一个 pre-LN 的近似 MACs/token
    readout = d * vocab
    mix_total = d * n_layers * (n_layers - 1) // 2   # dense 独有的层间加权和（全栈）
    out = {}
    for topo, segs, n_ln, mix in (("sequential", 2, 2, 0),
                                  ("parallel", 1, 1, 0),
                                  ("parallel-dual-ln", 1, 2, 0),
                                  ("dense", 2, 2, mix_total)):
        per_layer = attn + mlp + n_ln * ln
        out[topo] = {
            "layer_internal_segments": segs,
            "layer_internal_critical_path_steps": segs,
            "serial_steps_per_token": n_layers * segs,
            "attn_macs_per_token_per_layer": attn,
            "mlp_macs_per_token_per_layer": mlp,
            "ln_macs_per_token_per_layer": n_ln * ln,
            "mix_macs_per_token_total": mix,
            "macs_per_token_per_layer": per_layer + (mix // n_layers if mix else 0),
            "macs_per_token_total": n_layers * per_layer + mix + readout,
            "layer_passes_per_token": n_layers,
            "tokens_per_forward": 1,
        }
    seq = out["sequential"]["macs_per_token_total"]
    for topo in out:
        out[topo]["macs_per_token_total_vs_sequential_frac"] = round(
            out[topo]["macs_per_token_total"] / seq, 6)
    return out


# ---------------------- GPU 互斥锁（SCHEMA §20 / §20.1 按臂释放） ----------------------

LOCK = "/tmp/pathlm_gpu.lock"
_LOCK_HELD = False


def _pid_alive(pid_str: str) -> bool:
    try:
        os.kill(int(pid_str), 0)
        return True
    except (OSError, ValueError):
        return False


def _atexit_release():
    """崩溃/异常退出时不留锁（避免泄漏锁阻塞其它 worker）。"""
    if _LOCK_HELD:
        _lock_clean()


def _lock_clean():
    """先清空内容再 rmdir（锁目录含 holder_pid 时 rmdir 删不掉非空目录）。"""
    for f in ("holder_pid", "heartbeat"):
        try:
            os.remove(os.path.join(LOCK, f))
        except OSError:
            pass
    try:
        os.rmdir(LOCK)
    except OSError:
        pass


atexit.register(_atexit_release)


def gpu_lock_acquire(max_wait_s: int = 600):
    """§20 取锁：最多等 10 min；自愈回收陈旧锁（holder 死 / 无 holder 且锁龄 >15min）。
    不杀他人进程、不动他人锁目录（回收前先清内容再 rmdir）。"""
    t0, n = time.time(), 0
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            hp = None
            try:
                with open(os.path.join(LOCK, "holder_pid")) as f:
                    hp = f.read().strip()
            except OSError:
                hp = None
            stale, reason = False, ""
            if hp and not _pid_alive(hp):
                stale, reason = True, f"holder_pid {hp} 已死"
            elif not hp and os.path.isdir(LOCK) and \
                    (time.time() - os.stat(LOCK).st_mtime) > 900:
                stale, reason = True, "无 holder_pid 且锁龄 >15min"
            if stale:
                print(f"[lock] reclaim: {reason}", flush=True)
                _lock_clean()
                continue
            if time.time() - t0 > max_wait_s:
                raise TimeoutError("等锁超过 10 min，未取到 GPU 锁")
            time.sleep(15)
            n += 1
    with open(os.path.join(LOCK, "holder_pid"), "w") as f:
        f.write(str(os.getpid()))
    open(os.path.join(LOCK, "heartbeat"), "w").close()
    global _LOCK_HELD
    _LOCK_HELD = True
    print(f"[lock] acquired pid={os.getpid()}", flush=True)


def gpu_lock_heartbeat():
    try:
        os.utime(os.path.join(LOCK, "heartbeat"), None)
    except OSError:
        pass


def gpu_lock_release(hold_s: float = 60.0):
    """§20.1 按臂释放：清内容 + rmdir，留 ≥60s 窗口（给等待者公平插队）。"""
    global _LOCK_HELD
    _lock_clean()
    _LOCK_HELD = False
    print(f"[lock] released; {hold_s:.0f}s window", flush=True)
    if hold_s > 0:
        time.sleep(hold_s)


# ------------------------------ 评测电池 ------------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """prob0 校准：每深度 softmax 后 top-1 概率对 argmax 正确率的等频分箱 ECE
    （`pathlm.metrics.ece`）。prob0_ece = 最终深度；exit4_ece = 深度 4。"""
    gen = torch.Generator().manual_seed(seed)
    L_ = model.cfg.n_layers
    conf = [[] for _ in range(L_ + 1)]
    hit = [[] for _ in range(L_ + 1)]
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        tgt = x[:, 1:]
        for k, lg in enumerate(aux["depth_logits"]):
            lg = lg[:, :-1].float()
            p = lg.softmax(-1)
            conf[k].append(p.max(-1).values.reshape(-1).cpu())
            hit[k].append((lg.argmax(-1) == tgt).reshape(-1))
    per_depth = [round(ece(torch.cat(conf[k]), torch.cat(hit[k]).float(),
                           n_bins=n_bins), 4) for k in range(L_ + 1)]
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[4],
            "ece_by_depth": per_depth, "n_bins": n_bins}


def _layer_input(model: DepthARModel, aux: dict, i: int) -> torch.Tensor:
    """第 i 层（0-index）的输入：dense 拓扑 = 层权加权和，其余 = 上一层输出。"""
    if model.cfg.topology == "dense":
        return model._layer_mix_input(aux["states"], i + 1)
    return aux["states"][i]


def _block_branches(blk, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按该臂接线返回子层增量 (Δ_attn, Δ_mlp)：Block = 链式（mlp 读
    x + Δ_attn），ParallelBlock = 两分支同读块输入 h_in。"""
    b = getattr(blk, "block", blk)
    B, T, d = h_in.shape
    h1 = b.ln1(h_in)
    q, k, v = b.qkv(h1).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, b.n_heads, d // b.n_heads).transpose(1, 2)
    a = torch.nn.functional.scaled_dot_product_attention(
        shape(q), shape(k), shape(v), is_causal=True)
    da = b.proj(a.transpose(1, 2).reshape(B, T, d))
    if isinstance(blk, ParallelBlock):
        h2 = h1 if blk.share_ln else b.ln2(h_in)
    else:
        h2 = b.ln2(h_in + da)
    return da, b.mlp(h2)


def counterfactual_sequential_block(blk):
    """把并行臂的块权重复制进一个 sequential `Block`：同权重、只改接线的反事实
    （共享 LN 臂无 ln2 ⇒ 用其 ln1 顶替，语义 = 「同一套权重走链式接线」）。"""
    from pathlm.model import Block, ModelConfig
    b = getattr(blk, "block", blk)
    mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                       mlp_mult=MLP)
    sb = Block(mcfg)
    sd = {k: v for k, v in b.state_dict().items()}
    if "ln2.weight" not in sd:
        sd["ln2.weight"] = b.ln1.weight.detach().clone()
        sd["ln2.bias"] = b.ln1.bias.detach().clone()
    sb.load_state_dict(sd)
    return sb.to(next(b.parameters()).device)


@torch.no_grad()
def branch_activity(model: DepthARModel, eval_arr, n_batches: int = 4,
                    batch_size: int = 8, seed: int = 11) -> list[dict]:
    """逐层「子层增量相对幅度」||Δ_attn||/||h||、||Δ_mlp||/||h||（h = 该层输入，
    每臂按自己的接线取子层输入）：接线/分支是否真在干活的量化证据（SCHEMA §10）。
    parallel 臂另给 `wiring_rel_dev_vs_seq`：把本臂块权重拷进一个 sequential
    Block（共享 LN 臂的 ln2 用 ln1 顶替）后，同一输入下两种接线的输出相对差。"""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    acc = [{"attn": [], "mlp": [], "dev": []} for _ in range(model.cfg.n_layers)]
    use_dev = model.cfg.topology.startswith("parallel")
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        for i, blk in enumerate(model.blocks):
            h_in = _layer_input(model, aux, i)
            da, dm = _block_branches(blk, h_in)
            acc[i]["attn"].append(float(da.norm() / h_in.norm()))
            acc[i]["mlp"].append(float(dm.norm() / h_in.norm()))
            if use_dev:
                sb = counterfactual_sequential_block(blk)
                out_par = blk(h_in)[0]
                out_seq = sb(h_in)[0]
                acc[i]["dev"].append(float((out_par - out_seq).norm()
                                           / out_seq.norm()))
    return [{"layer": i + 1,
             "attn_delta_rel": round(statistics.mean(a["attn"]), 4),
             "mlp_delta_rel": round(statistics.mean(a["mlp"]), 4),
             "wiring_rel_dev_vs_seq": (round(statistics.mean(a["dev"]), 4)
                                       if a["dev"] else None)}
            for i, a in enumerate(acc)]


@torch.no_grad()
def window_bpc(model: DepthARModel, eval_arr, n_batches: int = 20,
               batch_size: int = 16) -> list[float]:
    """逐 window（同 `depth_curve` 的 generator seed=2、同序同窗口）的最终深度
    bpc —— 配对 Δ 的窗口级 SE 用。窗口顺序四臂一致 ⇒ Δ 天然配对。"""
    gen = torch.Generator().manual_seed(2)
    T = model.cfg.seq_len
    model.eval()
    out = []
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        lg = aux["depth_logits"][-1][:, :-1]
        tgt = x[:, 1:]
        nats = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                               tgt.reshape(-1), reduction="sum").item()
        out.append(round(nats / tgt.numel() / LN2, 6))
    return out


def run_battery(model: DepthARModel, eval_arr, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    last = out["depth_curve"][-1]
    if "bpc" not in last:
        # 显式失败而非静默填 0：clean_bpc 的唯一来源是 depth_curve 末项
        raise KeyError(
            f"depth_curve 末项缺 'bpc' 键（实有 {sorted(last)}）；"
            "clean_bpc ＝ depth_curve[-1]['bpc']，不得回退到 0 或其他来源")
    out["clean_bpc"] = last["bpc"]
    out["next_token_acc"] = out["depth_curve"][-1]["acc"]
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["tf@4"] = next(r["accept"] for r in out["tf_acceptance"] if r["depth"] == 4)
    out["spec_cost_model"] = spec_cost_model(out["tf_acceptance"], L,
                                             draft_depths=(4, 6))
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["branch_activity"] = branch_activity(model, eval_arr)
    out["window_bpc"] = window_bpc(model, eval_arr, n_batches=eval_batches)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_arm(topo: str, steps: int, seed: int, caps_min: float,
              poll_s: int = 30) -> dict:
    """后台启动 train_depth_ar.py（--dense --topology <topo>），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义）。"""
    name = run_name(topo)
    run_dir = os.path.join(HERE, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", name,
           "--d", str(D), "--layers", str(L), "--heads", str(H),
           "--seq-len", str(SEQ), "--mlp-mult", str(MLP), "--dense",
           "--topology", topo, "--steps", str(steps), "--batch-size", str(BS),
           "--seed", str(seed), "--exit-mode", "uniform", "--eval-batches", "20",
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {name}: training started pid={proc.pid} steps={steps}",
          flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{name} training exceeded {caps_min} min cap")
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for line in q.stdout.strip().splitlines():
                pid, mem = [s.strip() for s in line.split(",")]
                if int(pid) == proc.pid:
                    peak = max(peak, float(mem) / 1024.0)
        except Exception:
            pass
        gpu_lock_heartbeat()
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            if lines and int(json.loads(lines[-1])["step"]) % 2000 == 0:
                print(f"[run] {name} {lines[-1].strip()[:120]}", flush=True)
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} training failed rc={proc.returncode}")
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, f"{name}-{f}"))
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    print(f"[run] {name}: done wall={wall:.2f} min peak_child_mem={peak:.2f}GB "
          f"(train results.json 无 bpc；clean_bpc 唯一来源 = "
          f"run_battery → depth_curve[-1]['bpc'])", flush=True)
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2),
            "seconds_per_step": round(wall * 60 / steps, 4)}


def arm_trained(topo: str, steps: int) -> bool:
    """该臂是否已有完整训练产物（train_log 末步到位 + results.json 存在）
    ——用于崩溃后 resume，不重训已完成的臂（外部占用降速时省 GPU）。"""
    run_dir = os.path.join(HERE, "runs", run_name(topo))
    log_p = os.path.join(run_dir, "train_log.jsonl")
    res_p = os.path.join(run_dir, "results.json")
    if not (os.path.exists(log_p) and os.path.exists(res_p)):
        return False
    rows = [json.loads(l) for l in open(log_p) if l.strip()]
    return bool(rows) and int(rows[-1]["step"]) >= steps - 1


# ---------------------- 收敛对账（SCHEMA §16） ----------------------

def _slope(win: list[dict]) -> float:
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / \
        sum((x - xm) ** 2 for x in xs)


def tail_window_slope(rows: list[dict], frac: float = 0.25) -> dict:
    """末 frac 窗内 loss 对 step 的最小二乘斜率（**四臂同一方法、同一窗口比例**）
    ——收敛对账用：斜率相同 ⇒ 差异是水平差（质量），斜率悬殊 ⇒ 收敛速度差。"""
    i0 = int(len(rows) * (1 - frac))
    win = rows[i0:]
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    return {"window_frac": frac, "n_points": len(win),
            "step_from": xs[0], "step_to": xs[-1],
            "loss_from": ys[0], "loss_to": ys[-1],
            "slope_nats_per_1k_steps": round(_slope(win) * 1000, 5),
            "slope_nats_per_1k_steps_late_half": round(
                _slope(win[len(win) // 2:]) * 1000, 5)}


def load_arm_rows(topo: str) -> list[dict]:
    path = os.path.join(HERE, f"{run_name(topo)}-train_log.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------- 稳定性统计 ----------------------

def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    finite = all(torch.isfinite(torch.tensor(v)).item() for v in g + loss)
    return {
        "n_logged_samples": len(rows),
        "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
        "gnorm_spikes_gt3x_median": sum(1 for v in g if v > 3 * med),
        "gnorm_spikes_gt5x_median": sum(1 for v in g if v > 5 * med),
        "all_finite": finite,
        "loss_first": loss[0], "loss_last": loss[-1],
        "loss_max_step_increase": round(max(inc), 4),
        "loss_jumps_gt_0.25": sum(1 for v in inc if v > 0.25),
        "gnorm_monitor_all_steps": gnorm_monitor,
    }


# ---------------------- s0（2613）对照：口径锚 + 逐字段对照 ----------------------

def remeasure_s0(eval_arr, vocab_size: int, eval_batches: int) -> dict:
    """用**当前电池**重测 2613 的四条 s0 checkpoint（口径锚，不参与判据）：
    逐臂 remeasured clean_bpc 必须与 2613 `results.json` 记的值逐位一致 ⇒ 证明
    s0 与 s1 用同一评测口径、跨 seed 可比（SCHEMA §9 第 2/3 条）。"""
    rec = json.load(open(os.path.join(S0_EXP, "results.json")))
    out = {"s0_exp": os.path.relpath(S0_EXP, ROOT), "arms": {}}
    for arm in ARMS:
        topo = arm["topology"]
        ckpt = os.path.join(S0_EXP, "runs", f"{topo}-12k-s0", "model.pt")
        if not os.path.exists(ckpt):
            out["arms"][topo] = {"available": False}
            continue
        torch.manual_seed(SEED)
        model = DepthARModel(cfg_base(topo), vocab_size).cuda()
        model.load_state_dict(torch.load(ckpt, map_location="cuda"))
        cur = depth_curve(model, eval_arr, device=DEVICE, n_batches=eval_batches)
        recorded = rec["arms"][topo]["clean_bpc"]
        out["arms"][topo] = {
            "available": True,
            "recorded_bpc_2613": recorded,
            "remeasured_bpc_now": cur[-1]["bpc"],
            "bitwise_equal": cur[-1]["bpc"] == recorded,
        }
        del model
        torch.cuda.empty_cache()
    out["all_bitwise_equal"] = all(
        a.get("bitwise_equal", False) for a in out["arms"].values())
    out["note"] = ("口径锚：当前电池重测 s0 checkpoint 与 2613 记录逐位一致 ⇒ "
                   "s0/s1 同一评测口径")
    return out


def config_vs_2613(cfg_dump: dict) -> dict:
    """resolved config 与 2613 `config.json` **逐字段**对照（SCHEMA §9 第 2 条）。
    唯一允许不等的字段 = `seed`（0→1）与臂的 `run` 路径（s0→s1）。"""
    old = json.load(open(os.path.join(S0_EXP, "config.json")))
    rows = []
    for f in ("steps", "batch_size", "lr", "optimizer", "data", "data_split",
              "vocab_size", "train_tokens"):
        rows.append({"field": f, "s0": old[f], "s1": cfg_dump[f],
                     "equal": old[f] == cfg_dump[f]})
    for f in ("d_model", "n_layers", "n_heads", "seq_len", "mlp_mult",
              "dropout", "norm_cap", "dense", "exit_mode", "loss", "readout"):
        rows.append({"field": f"base.{f}", "s0": old["base"][f],
                     "s1": cfg_dump["base"][f],
                     "equal": old["base"][f] == cfg_dump["base"][f]})
    for i, a in enumerate(ARMS):
        topo = a["topology"]
        o = next(x for x in old["arms"] if x["topology"] == topo)
        n = cfg_dump["arms"][i]
        rows.append({"field": f"arms[{topo}].role", "s0": o["role"],
                     "s1": n["role"], "equal": o["role"] == n["role"]})
        rows.append({"field": f"arms[{topo}].params_expected_delta",
                     "s0": o["params_expected_delta"],
                     "s1": n["params_expected_delta"],
                     "equal": o["params_expected_delta"]
                     == n["params_expected_delta"]})
    rows.append({"field": "seed", "s0": old["seed"], "s1": cfg_dump["seed"],
                 "equal": old["seed"] == cfg_dump["seed"],
                 "note": "唯一有意不等的科学字段（n≥2 要求）"})
    rows.append({"field": "experiment", "s0": old["experiment"],
                 "s1": cfg_dump["experiment"],
                 "equal": old["experiment"] == cfg_dump["experiment"],
                 "note": "实验目录名（非科学字段）"})
    unequal = [r["field"] for r in rows if not r["equal"]]
    return {"rows": rows, "unequal_fields": unequal,
            "only_seed_and_name_differ": set(unequal) <= {"seed", "experiment"}}


# ------------------------------ 主流程 ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--caps-min", type=float, default=14.0,
                    help="单臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--skip-train", action="store_true",
                    help="复用 runs/ 下已有 checkpoint，只重算电池与判读")
    ap.add_argument("--budget-min", type=float, default=42.0,
                    help="GPU 墙钟预算（契约 45 min）；超出则在下一臂开跑前中止"
                         "（不静默跑残臂）")
    ap.add_argument("--arms", default=None,
                    help="只跑指定拓扑（逗号分隔，必须含 sequential）；缺省全跑")
    ap.add_argument("--skip-s0-anchor", action="store_true",
                    help="跳过 s0（2613）checkpoint 重测锚")
    args = ap.parse_args()
    if args.arms:
        want = [t.strip() for t in args.arms.split(",") if t.strip()]
        unknown = [t for t in want if t not in [a["topology"] for a in ARMS]]
        if unknown or "sequential" not in want:
            raise SystemExit(f"--arms 无效（{unknown}）：必须含 sequential，"
                             f"可选 {[a['topology'] for a in ARMS]}")
        ARMS[:] = [a for a in ARMS if a["topology"] in want]

    t_start = time.time()
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    cost_face = topology_cost_face(vocab=vocab_size)
    cfg_dump = {
        "experiment": "2651-inter-layer-topology-s1-13m",
        "ticket": "ticket 61 §9.2 WP-2651（2.2.3 + 2.2.4）",
        "seed": args.seed, "steps": args.steps, "batch_size": BS, "lr": LR,
        "optimizer": "AdamW(wd=0.01) + warmup 200 步 + cosine 到 0.1×",
        "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size, "train_tokens": args.steps * BS * SEQ,
        "base": {"d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ,
                 "mlp_mult": MLP, "dropout": 0.0, "norm_cap": 1.0,
                 "dense": True, "exit_mode": "uniform",
                 "loss": "仅最终深度 CE（stock AR）",
                 "readout": "tied h·Eᵀ，无 final LayerNorm（共享底座偏离 1）"},
        "arms": [{"topology": a["topology"], "role": a["role"],
                  "params_expected_delta": {"sequential": 0,
                                            "parallel": -L * 2 * D,
                                            "parallel-dual-ln": 0,
                                            "dense": L * (L + 1) // 2}[a["topology"]],
                  "run": f"runs/{run_name(a['topology'])}"} for a in ARMS],
        "single_factor_note": ("与 2613 s0（契约 E）逐字段相同，唯一科学字段差异 = "
                               "seed 0→1（2.2.3 n≥2 要求）"),
    }
    cfg_dump["vs_2613"] = config_vs_2613(cfg_dump)
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[cfg] vs 2613 unequal fields = {cfg_dump['vs_2613']['unequal_fields']}",
          flush=True)
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump,
           "cost_face_2_2_4": cost_face, "arms": {}}

    # ---- s0（2613）checkpoint 重测锚（口径一致，不参与判据） ----
    lock_held = False
    if not args.skip_s0_anchor:
        gpu_lock_acquire()
        lock_held = True
        out["s0_anchor_remeasured"] = remeasure_s0(eval_arr, vocab_size,
                                                   args.eval_batches)
        print(f"[s0] remeasured all_bitwise_equal="
              f"{out['s0_anchor_remeasured']['all_bitwise_equal']}", flush=True)

    # ---- 四臂训练 + 电池（§20.1 按臂释放锁） ----
    for i, arm in enumerate(ARMS):
        topo = arm["topology"]
        if not args.skip_train and (time.time() - t_start) / 60 > args.budget_min:
            if lock_held:
                gpu_lock_release(0)
            raise SystemExit(
                f"[budget] {args.budget_min} min 墙钟预算已到；已完成 "
                f"{list(out['arms'])}，未跑 {[a['topology'] for a in ARMS[i:]]}")
        if not lock_held:
            gpu_lock_acquire()
            lock_held = True
        if args.skip_train or arm_trained(topo, args.steps):
            res = json.load(open(os.path.join(
                HERE, "runs", run_name(topo), "results.json")))
            tr = {"results": res, "wall_min": res.get("wall_minutes"),
                  "peak_child_mem_gb": None, "seconds_per_step": None}
            if arm_trained(topo, args.steps) and not args.skip_train:
                print(f"[run] {run_name(topo)}: 已训练完成，resume 跳过重训",
                      flush=True)
        else:
            tr = train_arm(topo, args.steps, args.seed, args.caps_min,
                           args.poll_s)
        res = tr["results"]
        torch.manual_seed(args.seed)
        model = DepthARModel(cfg_base(topo), vocab_size).cuda()
        model.load_state_dict(torch.load(
            os.path.join(HERE, "runs", run_name(topo), "model.pt"),
            map_location="cuda"))
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        out["arms"][topo] = {
            "role": arm["role"], "topology": topo,
            "source": f"runs/{run_name(topo)}",
            "params": res["params"],
            "train_tokens": res["train_tokens"],
            "wall_min": tr["wall_min"], "peak_child_mem_gb": tr["peak_child_mem_gb"],
            "seconds_per_step": tr["seconds_per_step"],
            "gnorm_monitor": res.get("gnorm_monitor"),
            "intervention": {**topology_liveness(model),
                             "branch_activity": batt["branch_activity"]},
            "cost_face": cost_face[topo],
            **batt, "eval_wall_min": round((time.time() - t0) / 60, 2)}
        print(f"[arm] {topo}: clean_bpc={batt['clean_bpc']} tf@4={batt['tf@4']} "
              f"params={res['params']} wall={tr['wall_min']} min "
              f"eval={out['arms'][topo]['eval_wall_min']} min", flush=True)
        # 增量落盘：每臂跑完立刻写 results.json + append run_log.txt
        dump_results(out, note=f"增量：完成 {topo} 后落盘")
        append_run_log(
            f"[arm] {topo} clean_bpc={batt['clean_bpc']} tf@4={batt['tf@4']} "
            f"params={res['params']} wall_min={tr['wall_min']} "
            f"eval_wall_min={out['arms'][topo]['eval_wall_min']}")
        del model
        torch.cuda.empty_cache()
        gpu_lock_release(0 if i == len(ARMS) - 1 else 60.0)
        lock_held = False

    # ---- 逐字段对照表（单因素） ----
    fields = [("d_model", D), ("n_layers", L), ("n_heads", H), ("seq_len", SEQ),
              ("mlp_mult", MLP), ("dropout", 0.0), ("norm_cap", 1.0),
              ("dense", True), ("fill_kv", True), ("proj_fill", False),
              ("readout_dims", None), ("readout_gated", False),
              ("readout_linear_resid", False), ("exit_mode", "uniform"),
              ("aux_weight", 1.0), ("final_weight", 1.0),
              ("shallow_stopgrad", False), ("dense_aux", False),
              ("steps", args.steps), ("batch_size", BS), ("lr", LR),
              ("seed", args.seed), ("train_tokens", args.steps * BS * SEQ),
              ("vocab_size", vocab_size),
              ("data_split", "first 90M/ last 10M, vocab over full file")]
    rows = []
    for f, want in fields:
        row = {"field": f, "expected": want, "arms": {}}
        for arm in ARMS:
            topo = arm["topology"]
            res = json.load(open(os.path.join(HERE, "runs", run_name(topo),
                                              "results.json")))
            got = {
                "d_model": res["config"]["d_model"],
                "n_layers": res["config"]["n_layers"],
                "n_heads": res["config"]["n_heads"],
                "seq_len": res["config"]["seq_len"],
                "mlp_mult": res["config"]["mlp_mult"],
                "dropout": res["config"]["dropout"],
                "norm_cap": res["config"]["norm_cap"],
                "dense": res["config"]["dense"],
                "fill_kv": res["fill_kv"], "proj_fill": res["proj_fill"],
                "readout_dims": res["readout_dims"],
                "readout_gated": res["readout_gated"],
                "readout_linear_resid": res["readout_linear_resid"],
                "exit_mode": res["exit_mode"],
                "aux_weight": res["aux_weight"], "final_weight": res["final_weight"],
                "shallow_stopgrad": res["shallow_stopgrad"],
                "dense_aux": res["dense_aux"],
                "steps": res["train"]["steps"], "batch_size": res["train"]["batch_size"],
                "lr": res["train"]["lr"], "seed": res["train"]["seed"],
                "train_tokens": res["train_tokens"],
                "vocab_size": res["vocab_size"],
                "data_split": "first 90M/ last 10M, vocab over full file",
            }[f]
            row["arms"][topo] = got
            row["equal" + f"_{topo}"] = (got == want)
        row["all_arms_equal"] = all(row["equal" + f"_{a['topology']}"]
                                    for a in ARMS)
        rows.append(row)
    out["field_by_field"] = rows
    out["field_by_field_all_equal_except_topology"] = all(
        r["all_arms_equal"] for r in rows)
    out["field_by_field"].append(
        {"field": "topology（唯一变量，臂间）",
         "arms": {a["topology"]: a["topology"] for a in ARMS},
         "note": "臂间唯一不等字段；参数量随之变化（见 params_* 与对账块）"})

    # ---- 对账（SCHEMA §12 三道）+ 收敛对账（§16） ----
    seq_bpc = out["arms"]["sequential"]["clean_bpc"]
    params_seq = out["arms"]["sequential"]["params"]
    deltas = {a["topology"]: round(out["arms"][a["topology"]]["clean_bpc"]
                                   - seq_bpc, 4) for a in ARMS}
    slopes = {a["topology"]: tail_window_slope(load_arm_rows(a["topology"]))
              for a in ARMS}
    out["reconciliation"] = {
        "arithmetic": {
            "bpc_definition": "bpc_arm − bpc_sequential（同一条 depth_curve 的 depth-8 "
                              "读数、同 20 个 eval batch、同 eval 切分）",
            "delta_bpc": deltas,
            "variance_bound": 0.014,
            "judge_threshold_eq_variance_bound": True,
            "params": {a["topology"]: out["arms"][a["topology"]]["params"]
                       for a in ARMS},
            "params_delta": {a["topology"]: out["arms"][a["topology"]]["params"]
                             - params_seq for a in ARMS},
            "params_delta_expected": {"sequential": 0, "parallel": -L * 2 * D,
                                      "parallel-dual-ln": 0,
                                      "dense": L * (L + 1) // 2},
            "params_delta_exact": all(
                out["arms"][a["topology"]]["params"] - params_seq ==
                {"sequential": 0, "parallel": -L * 2 * D, "parallel-dual-ln": 0,
                 "dense": L * (L + 1) // 2}[a["topology"]] for a in ARMS),
            "dense_params_added_frac": round(
                (L * (L + 1) // 2) / params_seq, 8),
            "bpc_bounds": "clean bpc 理论下界 0；均匀预测参考值 log₂(206)=7.69；"
                          "四臂实测均在 (1.4, 1.7)",
        },
        "task": {
            "same_task": "四臂同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）、"
                         "同一位置集（全序列逐位置 × depth-8 读出）、同一目标定义"
                         "（tokens 自移位）",
            "same_battery": "四臂同一 run.py::run_battery（同 n_batches=20 / 同 eval "
                            "seed=2（depth_curve）/ 同 batch_size=16）",
            "deploy_correspondence": "四臂都是 stock decoder-only 前向；对照只改层与"
                                     "层之间的接法，无额外输入、无真值泄漏",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": args.steps, "seed": args.seed,
            "n_seed": 2, "train_tokens": args.steps * BS * SEQ,
            "note": f"四臂同为 {args.steps} 步 / {args.steps * BS * SEQ / 1e6:.1f}M "
                    "token / seed 1 的 matched-step 快照口径（非渐近、非墙钟、"
                    "非 FLOPs）；与 s0（seed 0）合成 n=2",
        },
        "convergence": {
            "method": "末 25% 窗（同一窗口比例、同一最小二乘算法）loss 斜率，"
                      "四臂同方法",
            "tail_slope_by_arm": slopes,
            "slope_spread_nats_per_1k": round(
                max(s["slope_nats_per_1k_steps"] for s in slopes.values()) -
                min(s["slope_nats_per_1k_steps"] for s in slopes.values()), 5),
            "note": "斜率相近 ⇒ 差异是水平差（质量）而非收敛速度差；另附末段"
                    "（窗内后半）斜率。更长训练点（2×）未做：本实验无增益可报，"
                    "§16 的 2× 要求针对「报增益」情形。",
        },
    }

    # ---- 配对 Δ（窗口级 SE + 两半） ----
    win = {a["topology"]: out["arms"][a["topology"]]["window_bpc"] for a in ARMS}
    paired = {}
    for a in ARMS:
        topo = a["topology"]
        if topo == "sequential":
            continue
        d = [round(v - s, 6) for v, s in zip(win[topo], win["sequential"])]
        half = len(d) // 2
        paired[topo] = {
            "delta_mean": round(sum(d) / len(d), 6),
            "delta_window_se": round(statistics.pstdev(d) / math.sqrt(len(d)), 6),
            "delta_halves": [round(sum(d[:half]) / half, 6),
                             round(sum(d[half:]) / (len(d) - half), 6)],
            "n_windows": len(d),
        }
    out["paired_delta_s1"] = paired

    # ---- 判读（ticket 61 §9.2 WP-2651，逐字未改） ----
    variants = [a["topology"] for a in ARMS if a["topology"] != "sequential"]
    success = [t for t in variants if deltas[t] <= -0.014]
    fail = [t for t in variants if deltas[t] >= 0.014]
    if success:
        ruling = (f"成功：{success} 相对基线改善 ≥0.014（Δ = "
                  f"{ {t: deltas[t] for t in success} }）")
    elif fail:
        ruling = f"失败（任一臂更差 ≥+0.014）：{ {t: deltas[t] for t in fail} }"
    else:
        ruling = (f"中性：全部变体 |Δ| < 0.014（Δ = {deltas}）")
    tf4 = {a["topology"]: out["arms"][a["topology"]]["tf@4"] for a in ARMS}
    out["verdict_s1"] = {
        "criterion": "任一变体 Δ clean bpc vs sequential ≤ −0.014 ⇒ 成功；"
                     "|Δ| < 0.014 ⇒ 中性；≥ +0.014 ⇒ 失败；并列 tf@4 ≥ 0.95",
        "delta_bpc_vs_sequential": deltas,
        "clean_bpc": {a["topology"]: out["arms"][a["topology"]]["clean_bpc"]
                      for a in ARMS},
        "success_arms": success, "fail_arms": fail, "ruling": ruling,
        "tf@4_by_arm": tf4,
        "next_token_acc_by_arm": {a["topology"]: out["arms"][a["topology"]]
                                  ["next_token_acc"] for a in ARMS},
    }

    # ---- 跨 seed（s0 vs s1）配对 Δ：2.2.3 / 2.2.4 判词 ----
    s0 = json.load(open(os.path.join(S0_EXP, "results.json")))
    s0_bpc = {t: s0["arms"][t]["clean_bpc"] for t in s0["arms"]}
    s1_bpc = out["verdict_s1"]["clean_bpc"]
    cross = {}
    for topo in variants:
        d0 = round(s0_bpc[topo] - s0_bpc["sequential"], 4)
        d1 = round(s1_bpc[topo] - s1_bpc["sequential"], 4)
        cross[topo] = {
            "bpc_s0": s0_bpc[topo], "bpc_s1": s1_bpc[topo],
            "bpc_seq_s0": s0_bpc["sequential"], "bpc_seq_s1": s1_bpc["sequential"],
            "delta_s0": d0, "delta_s1": d1,
            "same_sign": (d0 > 0) == (d1 > 0) and d0 != 0 and d1 != 0,
            "both_exceed_band": abs(d0) >= 0.014 and abs(d1) >= 0.014,
            "seed_spread_abs": round(abs(d1 - d0), 4),
            "sortable_two_seed": (abs(d0) >= 0.014 and abs(d1) >= 0.014
                                  and (d0 > 0) == (d1 > 0)),
        }
    out["cross_seed_s0_s1"] = cross
    seq_seed_spread = round(abs(s1_bpc["sequential"] - s0_bpc["sequential"]), 4)
    out["cross_seed_summary"] = {
        "band": 0.014,
        "sequential_bpc_s0_s1": [s0_bpc["sequential"], s1_bpc["sequential"]],
        "sequential_seed_spread": seq_seed_spread,
        "dense_delta_s0_s1": [cross["dense"]["delta_s0"],
                              cross["dense"]["delta_s1"]],
        "dense_sortable_two_seed": cross["dense"]["sortable_two_seed"],
        "dense_two_seed_conclusion": (
            "可排序（两 seed 同号且都越 0.014）"
            if cross["dense"]["sortable_two_seed"] else
            "无差异（至少一 seed 落在 0.014 带内）"),
        "parallel_delta_s0_s1": [cross["parallel"]["delta_s0"],
                                 cross["parallel"]["delta_s1"]],
        "parallel_dual_ln_delta_s0_s1": [cross["parallel-dual-ln"]["delta_s0"],
                                         cross["parallel-dual-ln"]["delta_s1"]],
        "parallel_cost_sortable_two_seed":
            cross["parallel"]["sortable_two_seed"]
            and cross["parallel-dual-ln"]["sortable_two_seed"],
        "n_seed": 2,
    }
    print(json.dumps(out["cross_seed_summary"], indent=2, ensure_ascii=False),
          flush=True)

    # ---- 稳定性块 + 逐深度 ECE ----
    out["stability"] = {
        a["topology"]: stability_stats(
            load_arm_rows(a["topology"]),
            out["arms"][a["topology"]]["gnorm_monitor"]) for a in ARMS}
    out["stability"]["ece_by_depth"] = {
        a["topology"]: out["arms"][a["topology"]]["calibration"]["ece_by_depth"]
        for a in ARMS}
    out["stability"]["gnorm_note"] = (
        "gnorm_monitor_all_steps 是 train_depth_ar 全程真值（每个优化器 step）；"
        "gnorm_logged_* 只是每 200 步采样点上可见的值。")

    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    out["gpu_peak_by_arm_gb"] = {a["topology"]: out["arms"][a["topology"]]
                                 ["peak_child_mem_gb"] for a in ARMS}
    dump_results(out, note="最终：四臂齐（含判读/对账/跨 seed）")
    append_run_log(f"[done] all_arms={out['completed_arms']} "
                   f"wall_total_min={out['wall_total_min']} "
                   f"delta_bpc_s1={out['verdict_s1']['delta_bpc_vs_sequential']}")
    print(json.dumps(out["verdict_s1"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
