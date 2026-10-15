"""WP 6.2.5 · 完整二级树（N+3 以被接受的 N+2 候选为条件）的草稿与**树验证**接线。

契约：本目录 `README.md`（预登记，不得事后调参）；机制缺口与主 Agent 裁决见 `QUESTIONS.md`、
未测因子见 `GAPS.md`。本模块**只读** `pathlm/**` 与 2640 的 `spec_tree.py`（复用其
`tree_forward` / `write_back_accepted_row`），不改任何共享文件。

## 位置口径（本 WP 的核心裁决点）

- 引擎原语 `Decoder.step_tree_multi`：行 j 只注意它的**祖先链**（父指针 `<0` = 根），但**位置仍按
  扁平顺序 `p0+j` 记**（`pathlm/decode.py:224`）。链结构（`parents[j]=j-1`）下与因果路径逐位一致；
  **同深度多兄弟**下不忠实（K 个 N+2 候选的正确位置都是 N+2）⇒ 见 `QUESTIONS.md`。
- 本 WP 的树实现 = **K 行平铺**（契约 §2 已为 `lvl1_tree` 预登记此机制）：B=K 行，每行是一条
  根到叶路径 `[v1, c_i, d_i]`，行内因果 ⇒ 三个节点的位置恰是 **N+1 / N+2 / N+3**，
  每行的前缀 KV 广播共享（只读）。一次前向验证 K 条路径。
  `d_i` 就是这个候选自己的孩子 ⇒ 「父指针 = 该候选」在行式布局下**由构造满足**
  （`test_second_tree.py` 的两条机械检查把它钉住）。

## 臂（契约 §2）

| 臂 | N+2 候选 | N+3 草稿 |
|---|---|---|
| `native` | — | 全深度逐 token 贪心（无损性参照） |
| `ar_1step` | `chain2` 头 argmax（条件 = 模型自己生成的 t_{N+1}） | 无（只替换紧邻位置；§11 诚实对手） |
| `direct1` | node-2 头 argmax | node-3 头 argmax（2640 `direct1` 口径） |
| `lvl1_tree(K)` | node-2 头 **top-K** | **全部行共用** node-3 头 argmax（无条件） |
| `lvl2_tree(K)` | 同上（同一候选集） | **每个候选**一行：`chain_second` 头以该候选为条件给出 |
| `lvl2_flat(K)`（诊断） | 同上 | 同上，但验证走**字面** `step_tree_multi` 扁平布局（预期 L3 红） |

`lvl1_tree` 与 `lvl2_tree` 的行数、位置数、前向次数、候选集**完全相同** ⇒ 唯一变量 = 「N+3 草稿
是否以 N+2 候选为条件」（契约 §2 的单一变量声明）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys

import numpy as np
import torch
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.decode import Decoder, decode, decode_spec, _slice_nodes
from pathlm.model import PathLM

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_repo(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根（含 pathlm/ 的祖先目录）: {start}")


ROOT = _find_repo(HERE)
EXP = os.path.join(ROOT, "axes", "6-predict", "6.2-speculative-decoding", "experiments")
DIR2631 = os.path.join(EXP, "2631-spec-three-arms")
DIR2640 = os.path.join(EXP, "2640-tree-conf-13m")
for _p in (ROOT, DIR2640):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spec_tree as T2640  # noqa: E402  （2640 的只读复用：tree_forward / write_back_accepted_row）

# ---------------------------------------------------------------- 常量（契约 §3）

VOCAB_SIZE = 206
PROMPT_LEN = 64           # 契约 §3：12 个固定 prompt、长度 64、enwik8 val 固定 offset
STRIDE = 5000             # 与 2640/2650/2670 同 offset 取法（2640/2650 用 len 32，本 WP 按契约用 64）
N_PROMPTS = 12
N_NEW = 200
GAMMA = 2                 # 投机草稿位数（N+2, N+3）
BOOT_B = 2000
BOOT_SEED = 2675
DATA_NPZ = os.path.join(ROOT, "data", "enwik8_full.npz")

#: 头训练配方（沿用 2660：Adam、无 weight decay、lr 1e-3、data_seed 2660、seq_len 256）
TRAIN_SEQ_LEN = 256
TRAIN_STEPS = 1200
TRAIN_LR = 1e-3
TRAIN_BATCH_POSITIONS = 4096
DATA_SEED = 2660

TRUNKS = {"13m_s0": os.path.join(DIR2631, "train", "13m_s0"),
          "13m_s1": os.path.join(DIR2631, "train", "13m_s1")}
CHAIN3_KEYS = ("transforms_chain2", "chain_cat2", "conf_chain3")

# 禁墙钟自检（沿用 2660 的 has_wall_clock 口径）
WALL_SUBSTR = ("wall", "elapsed", "clock", "duration", "seconds", "milliseconds",
               "latency", "throughput")
WALL_EXACT = {"time", "ms", "sec", "took", "runtime", "speed", "tps", "timing"}


def has_wall_clock(obj, path="") -> list[str]:
    bad: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = k.lower()
            if any(w in kl for w in WALL_SUBSTR) or kl in WALL_EXACT:
                bad.append(f"{path}.{k}")
            bad += has_wall_clock(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad += has_wall_clock(v, f"{path}[{i}]")
    return bad


# ---------------------------------------------------------------- 载入（只读）

def trunk_config(ckpt: str) -> tuple[dict, dict]:
    with open(os.path.join(ckpt, "results.json"), encoding="utf-8") as f:
        cfg = json.load(f)["config"]
    return cfg["model"], cfg["path"]


def load_trunk(ckpt: str, chain_second: bool = False, head: str | None = None) -> PathLM:
    """2631 主干 + （可选）本 WP 训出的二级头。

    `chain_second=False` 走**严格** `load_state_dict`（与 2640/2650 的 `load_ckpt` 同一条路径，
    L0 的保真基准）；`True` 时只允许缺 `transforms_chain2/chain_cat2/conf_chain3` 三个键。
    """
    mcfg, pcfg = trunk_config(ckpt)
    model = PathLM(ModelConfig(**mcfg), PathConfig(**{**pcfg, "chain_second": chain_second}),
                   vocab_size=VOCAB_SIZE)
    sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu", weights_only=True)
    if chain_second:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected or set(missing) - {
                f"{k}.{w}" for k in CHAIN3_KEYS for w in ("weight", "bias")}:
            raise RuntimeError(f"chain_second 开时 state_dict 不符：missing={missing} "
                               f"unexpected={unexpected}")
        if head is not None:
            hs = torch.load(head, map_location="cpu", weights_only=True)
            for k in CHAIN3_KEYS:
                model.load_state_dict({f"{k}.weight": hs[f"{k}.weight"],
                                       f"{k}.bias": hs[f"{k}.bias"]}, strict=False)
    else:
        model.load_state_dict(sd)
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def head_param_stats(model: PathLM) -> dict:
    return {k: sum(p.numel() for n, p in model.named_parameters() if n.startswith(k))
            for k in CHAIN3_KEYS}


def state_dict_sha(model: PathLM, prefix: str = "") -> str:
    import hashlib
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if n.startswith(prefix):
            h.update(n.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def prompts(val: np.ndarray, n: int = N_PROMPTS, length: int = PROMPT_LEN,
            stride: int = STRIDE) -> list[torch.Tensor]:
    return [torch.tensor(val[i * stride:i * stride + length].astype(np.int64))
            for i in range(n)]


def enwik8() -> dict:
    z = np.load(DATA_NPZ)
    return {"train": z["train"], "val": z["val"]}


# ---------------------------------------------------------------- 草稿与候选

@torch.no_grad()
def topk_candidates(nodes: dict, K: int) -> list[int]:
    """位置 N+2 的 K 个候选 = node-2 头（直连头）分布的 top-K（与 2640 `tree_direct_k2` 同源）。"""
    logits = nodes[2]["logits"][0, 0]
    return [int(t) for t in torch.topk(logits, min(K, logits.numel())).indices.tolist()]


@torch.no_grad()
def chain3_draft(model: PathLM, h_ctx: torch.Tensor, cand: int) -> int:
    """二级条件头的 N+3 草稿：`transforms_chain2(chain_cat2([h_ctx, embed(cand)]))` 的 argmax。

    与 `PathLM._mtp_nodes` 里 chain3 分支**同一算式**（同一对模块、同一 tied 读出），只是把
    条件换成「候选 c」，上下文取本轮前向之前的 h_N（严格对齐到 h_{N+1} 的版本见 `GAPS.md` G2）。
    """
    if model.chain_cat2 is None or model.transforms_chain2 is None:
        raise RuntimeError("模型没有 chain_second 头（需 chain_second=True）")
    e = model.embed(torch.tensor([int(cand)], device=h_ctx.device)).view(1, 1, -1)
    x = torch.cat([h_ctx.view(1, 1, -1), e], dim=-1)
    latent = model.transforms_chain2(model.chain_cat2(x))
    return int((latent @ model.embed.weight.T)[0, 0].argmax())


@torch.no_grad()
def node1_token(nodes: dict) -> int:
    return int(nodes[1]["logits"][0, 0].argmax())


# ---------------------------------------------------------------- 树验证（K 行平铺）

@torch.no_grad()
def decode_tree_rows(model: PathLM, prompt: torch.Tensor, n_new: int, pcap: PathConfig,
                     level: str, K: int, activity: list | None = None):
    """主判两臂：`level="lvl1"`（N+3 草稿 = node-3 头，全行共用）/ `"lvl2"`（每个候选一行）。

    每轮：一次前向、K 行、每行 `[v1, c_i, d_i]`（位置 N+1/N+2/N+3）⇒ 行数与前向次数两臂相同。
    接受判据（与 2640 逐字同口径）：候选 `c_i` 落在行内索引 1 ⇒ 用第 i 行索引 0 的 node-1 logits
    验它；`d_i` 用**接受行**索引 1 的 logits 验 ⇒ 至多接受 2 个草稿。
    """
    if level not in ("lvl1", "lvl2"):
        raise ValueError(f"level 只能是 lvl1/lvl2，收到 {level!r}")
    if model.pcap.n_mtp <= GAMMA:
        raise ValueError(f"γ={GAMMA} 需要 node3：收到 n_mtp={model.pcap.n_mtp}")
    dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    gen: list[int] = []
    accepts: list[int] = []
    n_forward = 0
    while len(gen) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None, "step must populate last_nodes"
        v1 = node1_token(nodes)
        cands = topk_candidates(nodes, K)
        if level == "lvl2":
            h_ctx = nodes[0]["latent"]                     # 位置 N 的主干隐状态
            drafts = [chain3_draft(model, h_ctx, c) for c in cands]
        else:
            d = int(nodes[3]["logits"][0, 0].argmax())     # node-3 直连头，无条件，全行共用
            drafts = [d] * len(cands)
        if dec.n + 1 + GAMMA > model.mcfg.seq_len:
            break
        rows = [[v1, c, d] for c, d in zip(cands, drafts)]
        _h, fnodes, new_kv, p0 = T2640.tree_forward(model, dec, rows)
        n_forward += 1
        logs = fnodes[1]["logits"]
        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[i, 0].argmax()) == c), None)
        n_acc = 0
        if acc_i is not None:
            n_acc = 1
            if int(logs[acc_i, 1].argmax()) == drafts[acc_i]:
                n_acc = 2
        row_used = 0 if acc_i is None else acc_i
        n_pos = 1 + n_acc
        T2640.write_back_accepted_row(dec, new_kv, p0, row_used, n_pos)
        dec.last_nodes = T2640._take_row(_slice_nodes(fnodes, n_pos - 1), row_used)
        gen.append(v1)
        if n_acc >= 1:
            gen.append(cands[acc_i])
        if n_acc >= 2:
            gen.append(drafts[acc_i])
        accepts.append(n_acc)
        if activity is not None:
            activity.append({"v1": v1, "candidates": [int(c) for c in cands],
                             "drafts": [int(d) for d in drafts],
                             "draft_source": "chain3" if level == "lvl2" else "node3_direct",
                             "n_acc": int(n_acc), "n_pos": int(n_pos),
                             "accepted_candidate_index": acc_i,
                             "n_forward_this_round": 1})
    return gen[:n_new], {"accepts": accepts, "n_forward": n_forward,
                         "n_rounds": len(accepts),
                         "rows_per_round": float(K),
                         "positions_per_round": float(3 * K),
                         "cache_len": dec.cache_len()}


# ---------------------------------------------------------------- 字面 `step_tree_multi` 诊断臂

def _cache_snapshot(dec: Decoder) -> dict:
    return {i: (kv[0].shape[2], list(dec.layer_pos.get(i, []))) for i, kv in dec.kv.items()}


def _prune_to_path(dec: Decoder, p0: int, snaps: dict, keep_flat: list[int]) -> None:
    """把 `step_tree_multi` 写进缓存的整棵树**截到接受路径**（只留 keep_flat 那些扁平索引）。"""
    for i, (k, v) in dec.kv.items():
        l_old, old_pos = snaps[i]
        keep = list(range(l_old)) + [l_old + j for j in keep_flat]
        dec.kv[i] = (k[:, :, keep].contiguous(), v[:, :, keep].contiguous())
        dec.layer_pos[i] = old_pos + [p0 + t for t in range(len(keep_flat))]
    dec.n = p0 + len(keep_flat)


@torch.no_grad()
def decode_tree_flat(model: PathLM, prompt: torch.Tensor, n_new: int, pcap: PathConfig,
                     K: int, activity: list | None = None):
    """**诊断臂**：按契约 §2 字面把 1+K+K 个节点塞进一次 `step_tree_multi`。

    扁平布局 `[v1, c_1..c_K, d_1..d_K]`，父指针 `[-1, 0×K, 1..K]`。它**不是位置忠实的**
    （`decode.py:224`：扁平第 j 节点位置 = p0+j；K 个候选的正确位置都是 N+2）⇒ 预期发出流
    与 native 不一致（L3 红）。此臂用来给 `GAPS.md` G1 提供可失败的机械证据，**不作主判**。
    """
    dec = Decoder(model, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    gen: list[int] = []
    accepts: list[int] = []
    n_forward = 0
    while len(gen) < n_new:
        nodes = dec.last_nodes
        assert nodes is not None
        v1 = node1_token(nodes)
        cands = topk_candidates(nodes, K)
        h_ctx = nodes[0]["latent"]
        drafts = [chain3_draft(model, h_ctx, c) for c in cands]
        toks = [v1] + list(cands) + list(drafts)
        parents = [-1] + [0] * K + [1 + i for i in range(K)]
        if dec.n + len(toks) > model.mcfg.seq_len:
            break
        p0 = dec.n
        snaps = _cache_snapshot(dec)
        _h, fnodes = dec.step_tree_multi(toks, parents)
        n_forward += 1
        logs = fnodes[1]["logits"]
        # 候选 c_i 在扁平索引 1+i，它的 logits 预测 N+3（但位置 = p0+1+i）
        acc_i = next((i for i, c in enumerate(cands)
                      if int(logs[0, 0].argmax()) == c), None)
        n_acc = 0
        keep = [0]
        if acc_i is not None:
            n_acc = 1
            keep.append(1 + acc_i)
            if int(logs[0, 1 + acc_i].argmax()) == drafts[acc_i]:
                n_acc = 2
                keep.append(1 + K + acc_i)
        _prune_to_path(dec, p0, snaps, keep)
        dec.last_nodes = _slice_nodes(fnodes, keep[-1])
        gen.append(v1)
        if n_acc >= 1:
            gen.append(cands[acc_i])
        if n_acc >= 2:
            gen.append(drafts[acc_i])
        accepts.append(n_acc)
        if activity is not None:
            activity.append({"v1": v1, "candidates": [int(c) for c in cands],
                             "drafts": [int(d) for d in drafts],
                             "draft_source": "chain3_flat", "n_acc": int(n_acc),
                             "flat_indices_kept": [int(j) for j in keep]})
    return gen[:n_new], {"accepts": accepts, "n_forward": n_forward,
                         "n_rounds": len(accepts), "rows_per_round": float(1 + 2 * K),
                         "positions_per_round": float(1 + 2 * K)}


# ---------------------------------------------------------------- 臂驱动

SINGLE_ARMS = ("ar_1step", "direct1")


@torch.no_grad()
def run_arm(model: PathLM, prompt: torch.Tensor, n_new: int, pcap: PathConfig, arm: str,
            K: int = 2, activity: list | None = None):
    """一个臂在一个 prompt 上的一段解码；返回 (gen, stats)。"""
    if arm == "native":
        gen, st = decode(model, prompt, n_new, pcap)
        return gen, {"accepts": None, "n_forward": n_new, "n_rounds": n_new,
                     "rows_per_round": 1.0, "positions_per_round": 1.0}
    if arm == "ar_1step":
        # §11 诚实对手：一次 AR 步 + 一次验证（本引擎里 v1 来自上一次前向，故实测 1 前向/轮）
        gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=1,
                              draft_fn=T2640.make_draft_fn("chain1"))
        st["n_rounds"] = len(st["accepts"])
        st.update({"rows_per_round": 2.0, "positions_per_round": 2.0,
                   "note": "§11「两次前向」= 1 AR 步 + 1 验证；本引擎两件事合并为一次批量前向"})
        return gen, st
    if arm == "direct1":
        gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=GAMMA,
                              draft_fn=T2640.make_draft_fn("direct1"))
        st["n_rounds"] = len(st["accepts"])
        st.update({"rows_per_round": 1.0, "positions_per_round": 1.0 + GAMMA,
                   "note": "B=1 链验证（引擎 step_tree_multi，父指针=[-1,0,1]）"})
        return gen, st
    if arm in ("lvl1_tree", "lvl2_tree"):
        return decode_tree_rows(model, prompt, n_new, pcap,
                                "lvl1" if arm == "lvl1_tree" else "lvl2", K, activity)
    if arm == "lvl2_flat":
        return decode_tree_flat(model, prompt, n_new, pcap, K, activity)
    raise ValueError(f"未知臂 {arm!r}")


# ---------------------------------------------------------------- 统计（口径抄自 2640，逐行同源）

def _load_2640_run():
    path = os.path.join(DIR2640, "run.py")
    spec = importlib.util.spec_from_file_location("run_2640", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R2640 = _load_2640_run()


def arm_stats(groups: list[list[int]], K: int) -> dict:
    """率类指标：α₁（N+2 被接受）、α₂|₁（N+3 在 N+2 接受条件下被接受）、τ、tokens/forward。"""
    tot = sum(len(g) for g in groups)
    return {
        "n_rounds": tot,
        "n_prompts": len(groups),
        "alpha1": round(R2640.stat_alpha1(groups), 6),
        "alpha1_ci": R2640.cluster_ci(groups, R2640.stat_alpha1),
        "alpha2_given_alpha1": round(R2640.stat_alpha2(groups), 6),
        "alpha2_given_alpha1_ci": R2640.cluster_ci(groups, R2640.stat_alpha2),
        "tau": round(R2640.stat_tau(groups), 6),
        "tau_ci": R2640.cluster_ci(groups, R2640.stat_tau),
        "tokens_per_forward": round(R2640.stat_tpf(groups), 6),
        "tokens_per_forward_ci": R2640.cluster_ci(groups, R2640.stat_tpf),
        "alpha1_wilson": R2640.wilson_ci(int(sum(1 for g in groups for a in g if a >= 1)), tot),
        "rows_per_round": float(K),
        "positions_per_round": float(3 * K),
        # R = 行数·(1+γ)/(1+τ)（契约 §6 的口径）；γ=1 ⇒ 每轮送进主干的「行车」为 K
        "R_equivalent_flops": round(K * (1 + (GAMMA - 1)) / max(1 + R2640.stat_tau(groups), 1e-9), 4),
        "R_note": "R>1 是范式定义性质（用 FLOPs 换顺序前向次数），不作判据；等 FLOPs 对照臂在本引擎 "
                  "n_mtp=3 上限下不可得（63-F2）",
    }


def paired_ci(groups_a: list[list[int]], groups_b: list[list[int]], stat) -> dict:
    return R2640.paired_diff(groups_a, groups_b, stat)


# ---------------------------------------------------------------- 头训练（只训二级头，主干冻结）

def unfreeze_chain3(model: PathLM) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    ps = [p for n, p in model.named_parameters() if n.split(".")[0] in CHAIN3_KEYS]
    if len(ps) != 2 * len(CHAIN3_KEYS):
        raise RuntimeError(f"二级头参数数不符：{len(ps)}")
    for p in ps:
        p.requires_grad = True
    return ps


def train_chain_head(model: PathLM, train_arr: np.ndarray, seed: int, log,
                     steps: int = TRAIN_STEPS, lr: float = TRAIN_LR,
                     batch_positions: int = TRAIN_BATCH_POSITIONS,
                     seq_len: int = TRAIN_SEQ_LEN, data_seed: int = DATA_SEED) -> dict:
    """在冻结主干的 13M 模型上只训 `transforms_chain2`/`chain_cat2`/`conf_chain3`。

    损失 = `PathLM.forward` 的总损失（其余节点无梯度路径：参数 `requires_grad=False`），
    逐 step 记链-3 节点的 CE。数据序固定 `data_seed`，头初始化由构造期的 `torch.manual_seed(seed)`
    决定（`PathLM.__init__` 的 std=0.02）。返回 trace。
    """
    ps = unfreeze_chain3(model)
    model.train()
    trunk_frozen = all(not p.requires_grad for n, p in model.named_parameters()
                       if n.split(".")[0] not in CHAIN3_KEYS)
    opt = torch.optim.Adam(ps, lr=lr)
    batch = max(1, batch_positions // seq_len)
    tgen = torch.Generator().manual_seed(data_seed)
    rng = random.Random(data_seed)
    dev = next(model.parameters()).device
    trace = []
    for step in range(steps):
        idx = torch.randint(0, len(train_arr) - seq_len - 1, (batch,), generator=tgen).numpy()
        x = torch.from_numpy(np.stack([train_arr[i:i + seq_len] for i in idx]).astype(np.int64))
        x = x.to(dev)
        paths = [sample_path(model.pcap, rng, model.mcfg.n_layers)]
        loss, aux = model(x, paths, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            node = aux["rounds"][0][1]["chain3"]
            with torch.no_grad():
                logits = node["logits"][:, :x.shape[1] - 3]
                ce3 = float(torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE), x[:, 3:].reshape(-1)))
            trace.append({"step": step, "loss": round(float(loss.detach()), 5),
                          "chain3_ce": round(ce3, 5)})
            log(f"  [train seed={seed}] step={step} loss={float(loss.detach()):.5f} "
                f"chain3_ce={ce3:.5f}")
    return {"seed": seed, "steps": steps, "lr": lr, "seq_len": seq_len,
            "batch_positions": batch * seq_len, "batch_size": batch,
            "data_seed": data_seed, "optimizer": "Adam(no weight decay)",
            "trace": trace,
            "frozen_assert": {
                "trunk_requires_grad_all_false": bool(trunk_frozen),
                "non_head_grad_is_none": all(
                    p.grad is None for n, p in model.named_parameters()
                    if n.split(".")[0] not in CHAIN3_KEYS),
                "head_grad_present": all(p.grad is not None for p in ps),
                "pass": bool(trunk_frozen and all(
                    p.grad is None for n, p in model.named_parameters()
                    if n.split(".")[0] not in CHAIN3_KEYS) and all(
                    p.grad is not None for p in ps))},
            "head_sha256": {k: state_dict_sha(model, k) for k in CHAIN3_KEYS}}


def save_head(model: PathLM, path: str) -> str:
    import hashlib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {f"{k}.{w}": getattr(model, k).state_dict()[w]
          for k in CHAIN3_KEYS for w in ("weight", "bias")}
    torch.save(sd, path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
