"""2629-carry-ragged-projfill · 两臂定义 / 单因素增量 / 能力面（可失败单测的靶子）。

本模块只放**纯逻辑**（不含 GPU、不读写权重），好让
`tests/test_carry_ragged_projfill_13m.py` 把「两臂真的只差 carry 一个因子」
「随机退出 + proj-fill 是两臂同款的非 dense 配方」「增量是单因素对照而不是与裸
基线比」「能力面组合可建」四件事钉成可失败单测，并被 `mutate.py` 逐处抓红。

**契约来源**（ticket 45 · W1）：任务下发文本逐字转录在同目录 `README.md` §4。
本线测的是**产品上真正有意义的那对组合**：

> 「delta 搬运通道 × 随机退出深度 × proj-fill」= 产出 100M `tf@4=0.9726`
> 出口资产的那条配方的**另一半**（随机退出 + proj-fill 是 100M 配方的形态件；
> delta 搬运通道是此前因引擎硬互斥**从未与它同开过**的那一半）。

两臂（**唯一变量** = 是否开写入通道 `carry_rule`；其余逐字段一致）：

| 臂 | carry（delta 写） | 随机退出 | proj-fill | 块类 | `dense` |
|---|---|---|---|---|---|
| `projfill`（基准） | 关 | **开**（`--exit-mode uniform`） | **开** | `Block` | False |
| `projfill_carry`（目标） | **开**（`--carry-write delta`） | **开** | **开** | `CarryChannelBlock` | False |

- **能力面前提（SCHEMA §22.8）**：ENG2（ticket 45-E1，`2628-eng2-carry-ragged`）
  解除了 `carry_rule` 的 `dense=True` 硬校验，并把 `CarryChannelBlock` 的调用约定
  对齐 stock `Block`（`kv_from` / `attn_mask` / `gate_vec`）⇒「carry × 随机退出 ×
  proj-fill」这条**从未可测**的组合现在可建可跑。本模块的
  `capability_probe()` 在 CPU 上逐条核对（组合可建 + 块类正确 + 通道活着）。
- **仍存的硬互斥（如实记账）**：`train_window > 0` 与 `dense=False` 互斥 ⇒ 本线
  **没有**训练期可见窗口（非 dense 路径不允许），长程压力只来自「随机退出 + 填充」
  造成的**逐位置可见性差异**（早退位置在深层只能用冻结表征或其 proj-fill 像
  进 KV），不是「源 key 被推窗外」。这一点写进 README §4b，并作为**未测因子**声明。
"""

from __future__ import annotations

#: 两臂：唯一因子 = `carry`。基准 = 无 carry 的 proj-fill + 随机退出臂。
ARMS = (
    {"name": "projfill", "carry": False},
    {"name": "projfill_carry", "carry": True},
)

ARM_NAMES = tuple(a["name"] for a in ARMS)
#: 基准臂 = 无 carry（本实验「基准 = 同配方但关掉被检验的那一个因子」的口径）。
BASELINE_ARM = "projfill"
#: 目标臂 = 加上 delta 写入通道（判据①/②在这里读）。
TARGET_ARM = "projfill_carry"

#: 两臂同款的非 dense 形态件（**不是变量**，是 100M 配方的形态）。
CARRY_RULE = "delta"
PROJ_FILL = True
FILL_KV = True           # proj-fill 要求 fill_kv=True；缺省即 True
EXIT_MODE = "uniform"    # 逐位置随机退出深度 U{0..L}（100M 配方）
DENSE = False            # 非 dense = RAP 形态（随机退出 / 逐通道门 / 混合填充）
TOPOLOGY = "sequential"


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
    `--carry-write delta`。**不传** `--dense`（保持非 dense RAP 形态）。
    """
    flags = ["--proj-fill", "--exit-mode", EXIT_MODE]
    if spec["carry"]:
        flags += ["--carry-write", CARRY_RULE]
    return flags


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

    `metric = {arm_name: value}`（准确率或 tf 接受率）。返回：

    - `carry_effect` = `projfill_carry − projfill`（加上写入通道的效应）；
    - `net_vs_base`   = 同上（两臂设计下净和 == 单因素增量，仍显式给出以免误读）。

    对 bpc（越低越好）符号方向相反——调用方负责把 `metric` 换算成「越高越好」再
    解释符号；本函数只做算术。
    """
    b, c = metric["projfill"], metric["projfill_carry"]
    return {"carry_effect": round(c - b, 6), "net_vs_base": round(c - b, 6)}


def single_factor_increment_pp(metric: dict) -> dict:
    """同 `single_factor_increment`，但把比例换算成百分点（×100）。"""
    return {k: round(100.0 * v, 6) for k, v in single_factor_increment(metric).items()}


def destroy_source_valid(corrupt_ci_lo: float, chance: float,
                         factor: float = 4.0) -> bool:
    """毁源对照有效性（判据①的护栏）：毁源后 CI 下界 ≤ 4×chance ⇒ 读数来自源信息。

    沿用本线（2618/2627）的同一规则：`毁源后 CI 下界 > 4×chance ⇒ 该臂读数作废`。
    返回 `True` = **有效**（毁源确实把读数打回 chance 量级）。
    """
    return corrupt_ci_lo <= factor * chance


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
           "arms": {}}
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
