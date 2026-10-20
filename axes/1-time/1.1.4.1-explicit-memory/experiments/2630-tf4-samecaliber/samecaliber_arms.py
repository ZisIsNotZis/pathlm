"""2630-tf4-samecaliber · 两臂定义 / 单因素增量 / 能力面（可失败单测的靶子）。

本模块只放**纯逻辑**（不含 GPU、不读写权重），好让
`tests/test_tf4_samecaliber_13m.py` 把「两臂真的只差 carry 一个因子」
「两臂都在 enwik8 上训练（原生预训练口径、非合成）」「随机退出 + proj-fill 是两臂
同款的非 dense 配方」「tf@4 保持性判定是唯一基准」四件事钉成可失败单测，并被
`mutate.py` 逐处抓红。

**契约来源**（ticket 46 · W1）：任务下发文本逐字转录在同目录 `README.md` §4。

本线是 2629（ticket 46-W1 的产品级组合）的**同口径续判**：2629 的两臂训在**合成**
assoc-mqar 上，出口资产在两个口径下结论相反——分布内 synth tf@4 = 99.66% → 98.68%
（−0.98pp，带内 ⇒ 保持），而 enwik8 跨分布代理 tf@4 = 93.69% → 88.98%（−4.71pp，
超带 ⇒ 恶化）。本线把两臂**直接训在 enwik8 上**（原生预训练口径），让「训练分布 ==
评测分布 == enwik8」，从而把那个「跨分布」混杂拆掉，得到**同口径**判决。

两臂（**唯一变量** = 是否开写入通道 `carry_rule`；其余逐字段一致）：

| 臂 | carry（delta 写） | 随机退出 | proj-fill | 训练数据 | 块类 | `dense` |
|---|---|---|---|---|---|---|
| `base`（基准） | 关 | **开**（`--exit-mode uniform`） | **开** | **enwik8** | `Block` | False |
| `carry`（目标） | **开**（`--carry-write delta`） | **开** | **开** | **enwik8** | `CarryChannelBlock` | False |

- **口径不混用声明（SCHEMA §17.14 / 任务 §5）**：本线是 **enwik8 训练 + enwik8 评测**
  的**同口径**读数，与 2629 的「合成训练 + enwik8 跨分布代理」**是两个不同分布的口径**，
  **不得混用**、不得拼成一对「臂」。2629 的 synth 口径数字只作**动机**，不作本线的基准。
- **能力面前提（SCHEMA §22.8）**：ENG2（ticket 45-E1，`2628-eng2-carry-ragged`）解除了
  `carry_rule` 的 `dense=True` 硬校验 ⇒「carry × 随机退出 × proj-fill」在引擎里可建可跑；
  `train_depth_ar.py` 在 `synth is None and not dense` 分支里**本来就**算
  `depth_curve` / `tf_acceptance`（enwik8 电池）⇒ 本线训练入口即可直接产出**同口径**
  tf@4，不需要另接引擎。`capability_probe()` 在 CPU 上逐条核对。
- **仍存的硬互斥（如实记账）**：`train_window > 0` 与 `dense=False` 互斥 ⇒ 本线没有
  训练期可见窗口，长程压力只来自「随机退出 + 填充」的逐位置可见性差异。
"""

from __future__ import annotations

#: 两臂：唯一因子 = `carry`。基准 = 无 carry 的 proj-fill + 随机退出臂。
ARMS = (
    {"name": "base", "carry": False},
    {"name": "carry", "carry": True},
)

ARM_NAMES = tuple(a["name"] for a in ARMS)
#: 基准臂 = 无 carry（本实验「基准 = 同配方但关掉被检验的那一个因子」的口径）。
BASELINE_ARM = "base"
#: 目标臂 = 加上 delta 写入通道（判据①/②在这里读）。
TARGET_ARM = "carry"

#: 两臂同款的非 dense 形态件（**不是变量**，是 100M 配方的形态）。
CARRY_RULE = "delta"
PROJ_FILL = True
FILL_KV = True           # proj-fill 要求 fill_kv=True；缺省即 True
EXIT_MODE = "uniform"    # 逐位置随机退出深度 U{0..L}（100M 配方）
DENSE = False            # 非 dense = RAP 形态（随机退出 / 逐通道门 / 混合填充）
TOPOLOGY = "sequential"
#: 训练数据 = enwik8（`--synth-task none`，即 `train_depth_ar.py` 的缺省）。**两臂同款**。
DATA_SOURCE = "enwik8"

#: 方差带（pp）：`harness/methodology.md` 的 Δ < 0.014 不可排序 ⇒ 1.4pp。
VARIANCE_BAND_PP = 1.4


def arm_spec(name: str) -> dict:
    for a in ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知臂 {name!r}；期望 {ARM_NAMES}")


def model_config_kwargs(spec: dict) -> dict:
    """臂 → `DepthARConfig` 关键字：**唯一变量是 carry_rule**（其余形态件两臂相同）。"""
    return {
        "carry_rule": (CARRY_RULE if spec["carry"] else None),
        "dense": DENSE,
        "topology": TOPOLOGY,
        "fill_kv": FILL_KV,
        "proj_fill": PROJ_FILL,
    }


def train_flags(spec: dict) -> list[str]:
    """臂 → 训练入口 `train_depth_ar.py` 的额外 CLI 开关。

    两臂都带 `--proj-fill --exit-mode uniform`（形态件，非变量）；`carry` 臂再加
    `--carry-write delta`。**不传** `--dense`（保持非 dense RAP 形态）；**不传**
    `--synth-task`（缺省 = enwik8 原生预训练口径）。
    """
    flags = ["--proj-fill", "--exit-mode", EXIT_MODE]
    if spec["carry"]:
        flags += ["--carry-write", CARRY_RULE]
    return flags


def data_source(spec: dict) -> str:
    """两臂的训练数据源：**恒为 enwik8**（原生预训练口径，非合成）。"""
    return DATA_SOURCE


def expected_block_class(spec: dict, n_heads: int = 8) -> str:
    """该臂在引擎里应落的块类（`DepthARModel.__init__` 的选择规则镜像）。"""
    return "CarryChannelBlock" if spec["carry"] else "Block"


def native_visibility(spec: dict) -> tuple[int, int]:
    """评测可见性 = 该臂训练时的可见性。

    两臂都没有训练期可见窗口（`train_window > 0` 与 `dense=False` 硬互斥）
    ⇒ `(0, 0)` = 全因果；两臂同款、不是变量。"""
    return 0, 0


def single_factor_increment(metric: dict) -> dict:
    """单因素增量（SCHEMA §17.17：净和之外必须给分量 = 这里是唯一分量）。

    `metric = {arm_name: value}`（tf 接受率或 bpc）。返回：

    - `carry_effect` = `carry − base`（加上写入通道的效应）；
    - `net_vs_base`   = 同上（两臂设计下净和 == 单因素增量，仍显式给出以免误读）。

    对 bpc（越低越好）符号方向相反——调用方负责把 `metric` 换算成「越高越好」再
    解释符号；本函数只做算术。
    """
    b, c = metric[BASELINE_ARM], metric[TARGET_ARM]
    return {"carry_effect": round(c - b, 6), "net_vs_base": round(c - b, 6)}


def single_factor_increment_pp(metric: dict) -> dict:
    """同 `single_factor_increment`，但把比例换算成百分点（×100）。"""
    return {k: round(100.0 * v, 6) for k, v in single_factor_increment(metric).items()}


def tf4_holds(delta_pp: float, band_pp: float = VARIANCE_BAND_PP) -> bool:
    """判据①（预登记）的判定函数：|Δ| < 1.4pp ⇒ 记「carry 不弄坏出口资产」。

    Δ 的符号 = `carry − base`（越大越好，tf@4 是越高越好的量）；本判据对符号不敏感
    （只看绝对量级是否落在方差带内），故用 `abs`。
    """
    return abs(delta_pp) < band_pp


def capability_probe(d_model: int = 32, n_layers: int = 3, n_heads: int = 4,
                     seq_len: int = 12, vocab: int = 10, seed: int = 0) -> dict:
    """CPU 能力面自检（SCHEMA §22.8）：两臂可建 + 块类正确 + 唯一变量是 carry。

    - 两臂都能建（ENG2 前 `dense=False` + `carry_rule` 会 `ValueError`）；
    - 块类与 `expected_block_class` 一致；
    - **主干参数逐位相同**（同 seed；carry 参数只在目标臂 ⇒ 单因素对照成立，§9/§15）；
    - 目标臂**多出的参数**全部以 `carry.` 开头（没有偷改主干）。
    """
    import torch

    from pathlm.depth_ar import DepthARConfig, DepthARModel

    out = {"probe": "capability", "dense": DENSE, "topology": TOPOLOGY,
           "exit_mode": EXIT_MODE, "proj_fill": PROJ_FILL, "fill_kv": FILL_KV,
           "data_source": DATA_SOURCE, "arms": {}}
    models = {}
    for name in ARM_NAMES:
        spec = arm_spec(name)
        torch.manual_seed(seed)
        cfg = DepthARConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                            seq_len=seq_len, **model_config_kwargs(spec))
        m = DepthARModel(cfg, vocab_size=vocab)
        models[name] = m
        out["arms"][name] = {
            "block_class": type(m.blocks[0]).__name__,
            "block_class_expected": expected_block_class(spec, n_heads=n_heads),
            "n_params": sum(p.numel() for p in m.parameters()),
            "carry_rule": m.cfg.carry_rule,
            "proj_fill": bool(m.cfg.proj_fill),
            "fill_kv": bool(m.cfg.fill_kv),
            "dense": bool(m.cfg.dense),
        }
    base, target = models[BASELINE_ARM], models[TARGET_ARM]
    # 主干 = 不含任何 carry 相关键（含块上的单个读出增益 `blocks.k.carry_scale`）。
    tb = {k: v for k, v in base.state_dict().items() if "carry" not in k}
    tt = {k: v for k, v in target.state_dict().items() if "carry" not in k}
    trunk_diffs = [k for k in tb if k not in tt or not torch.equal(tb[k], tt[k])]
    out["trunk_bit_identical"] = (set(tb) == set(tt)) and not trunk_diffs
    out["trunk_diff_keys"] = trunk_diffs
    out["param_delta"] = out["arms"][TARGET_ARM]["n_params"] - \
        out["arms"][BASELINE_ARM]["n_params"]
    extra_keys = sorted(set(target.state_dict()) - set(base.state_dict()))
    out["carry_only_keys"] = extra_keys
    out["all_extra_params_are_carry"] = all("carry" in k for k in extra_keys)
    out["combo_buildable"] = out["arms"][TARGET_ARM]["block_class"] == \
        expected_block_class(arm_spec(TARGET_ARM), n_heads=n_heads)
    return out
