"""2627-carry-x-depth · 四臂定义 / 因子分解 / 评测可见性（可失败单测的靶子）。

本模块只放**纯逻辑**（不含 GPU、不读写权重），好让 `tests/test_carry_x_depth_13m.py`
把「四臂真的是 2×2 因子设计」「深度因子只改训练信号、不改架构」「叠加增量是因子分解
而不是只看净和」这三件事钉成可失败单测，并被 `mutate.py` 逐处抓红。

**能力面前提（SCHEMA §22.8，主 Agent 裁决 2026-10-06）**：delta 搬运通道硬要求
`dense=True`（`pathlm/depth_ar.py:1845`），而「per-token 随机退出」只在 `dense=False`
的 ragged 路径成立、且该路径拒绝 `carry_rule`——两者**在引擎里不可组合**。因此本实验的
「深度」因子落地为 **dense 全深度并行监督**（`--dense-aux`：L+1 个出口 CE 穿 trunk），
它是 dense 兼容、且能产出 `tf@4` 出口资产的那一档；**未测**因子 = ragged 随机退出 +
proj-fill 的冻结/混合 KV 语义（即产出 100M `tf@4=0.9726` 的配方）。见 README §4b/§4c。

四臂（2×2 因子设计；**唯一变量** = `carry` × `depth` 两个因子的开关）：

| 臂 | carry(delta 写) | depth(dense-aux 全深度监督) | 块类 | 参数量 |
|---|---|---|---|---|
| `base`  | 关 | 关 | `Block` | 13,536,768 |
| `carry` | 开 | 关 | `CarryChannelBlock` | 13,801,064 |
| `depth` | 关 | 开 | `Block` | 13,536,768（**与 base 逐位同架构**） |
| `both`  | 开 | 开 | `CarryChannelBlock` | 13,801,064（**与 carry 逐位同架构**） |

`depth` 是**训练信号因子**（forward 的 `dense_aux=True`），不是模型配置字段 ⇒
`depth`/`both` 的 `DepthARConfig` 与各自不含深度因子的臂**逐字段相同**（`dense_aux`
只在 `forward()` 里，`model_config_kwargs()` 不含它）。
"""

from __future__ import annotations

#: 四臂，顺序 = (carry, depth) 的 (F,F) (T,F) (F,T) (T,T) 全因子格子。
ARMS = (
    {"name": "base", "carry": False, "depth": False},
    {"name": "carry", "carry": True, "depth": False},
    {"name": "depth", "carry": False, "depth": True},
    {"name": "both", "carry": True, "depth": True},
)

ARM_NAMES = tuple(a["name"] for a in ARMS)
#: 基准臂 = 两个因子都关的密集基线。
BASELINE_ARM = "base"
#: 目标臂 = 两个因子都开的组合臂（判据⑤「两资产是否可叠加」在这里读）。
TARGET_ARM = "both"

#: 训练侧（forward）因子：`--dense-aux` 只在此出现（不是模型配置字段）。
DEPTH_TRAIN_FLAG = "--dense-aux"
#: 训练侧 carry 因子（模型配置字段同时也要设 `carry_rule`）。
CARRY_RULE = "delta"


def arm_spec(name: str) -> dict:
    for a in ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"未知臂 {name!r}；期望 {ARM_NAMES}")


def model_config_kwargs(spec: dict) -> dict:
    """臂 → `DepthARConfig` 关键字：**只有 carry 因子进配置**（深度因子不进）。"""
    return {"carry_rule": (CARRY_RULE if spec["carry"] else None)}


def train_flags(spec: dict) -> list[str]:
    """臂 → 训练入口的额外 CLI 开关（深度因子 = `--dense-aux`；carry 交给
    `model_config_kwargs` 的 `carry_rule` 字段）。"""
    return [DEPTH_TRAIN_FLAG] if spec["depth"] else []


def expected_block_class(spec: dict, n_heads: int = 8) -> str:
    """该臂在引擎里应落的块类（`DepthARModel.__init__` 的选择规则镜像）。"""
    return "CarryChannelBlock" if spec["carry"] else "Block"


def native_visibility(spec: dict) -> tuple[int, int]:
    """评测口径的可见性 = 该臂训练时的可见性。

    四臂**全部**用 `window=16 / anchors=0`（它是制造长程需求的那一步，三/四臂同款，
    不是变量——SCHEMA §19：窗口把源 key 推到窗外，不搬就答不出）。"""
    return 16, 0


def factorial_increments(metric: dict) -> dict:
    """2×2 因子分解（SCHEMA §17.17：净和不可排序时必须给分量）。

    `metric = {arm_name: value}`（准确率或 bpc）。返回四个单向增量 + 交互项 + 净和：

    - `carry_at_depth_off` = carry − base（depth 关时 carry 的效应）；
    - `depth_at_carry_off` = depth − base（carry 关时 depth 的效应）；
    - `carry_at_depth_on`  = both  − depth（depth 开时 carry 的效应）；
    - `depth_at_carry_on`  = both  − carry（carry 开时 depth 的效应）；
    - `interaction`        = (both − carry) − (depth − base)
                           = 两因子是否**可叠加**（≈0 = 可加；<0 = 互相干扰）；
    - `net_vs_base`        = both − base（净和，**不得单独读**）。

    对 bpc（越低越好）四项的符号方向相反——调用方负责把 `metric` 换算成「越高越好」
    再解释符号；本函数只做算术。
    """
    b, c, d, bd = (metric[n] for n in ("base", "carry", "depth", "both"))
    return {
        "carry_at_depth_off": round(c - b, 6),
        "depth_at_carry_off": round(d - b, 6),
        "carry_at_depth_on": round(bd - d, 6),
        "depth_at_carry_on": round(bd - c, 6),
        "interaction": round((bd - c) - (d - b), 6),
        "net_vs_base": round(bd - b, 6),
    }


def factorial_increments_pp(metric: dict) -> dict:
    """同 `factorial_increments`，但把比例换算成百分点（×100）。"""
    return {k: round(100.0 * v, 6) for k, v in factorial_increments(metric).items()}
