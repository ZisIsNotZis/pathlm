# analysis — 2660-latentreg-vs-disc-13m（阶段 1：直接发 / teacher-forced）

## 人话摘要

我们在**同一份冻结的 13M 主干**上，只改「未来位置怎么产出」这一件事：一种是直接把主干隐状态经几个小变换头**分类到词表**（离散臂），另一种是让这些头**回归到主干在未来位置的真实隐状态**（回归臂，多一个 λ·MSE 项把输出拉向真值隐状态），两者共用同一份 tied 读出、同一参数量、同一训练预算（3000 步）。

- **L0（实现保真硬门）**：**通过** —— `lat_k4_lam0` 与 `disc_k4` 同种子逐位相同（命中率 / CE / 头权重
  ）；冻结主干 `requires_grad=False` 且一次 backward 后 `.grad is None`。
- **L1（主判，路径 (i) 直接发）**：采纳（t+2 配对 95% CI 不含 0：lat_k4_lam1）
  - `lat_k4_lam1` vs `disc_k4`：t+2 Δ=+0.02844 CI[+0.02679,+0.03016] 不含0；t+3 Δ=+0.02525 CI[+0.02357,+0.02686] 不含0；t+4 Δ=+0.01051 CI[+0.00897,+0.01208] 不含0；t+5 Δ=+0.01342 CI[+0.01182,+0.01497] 不含0
- **L3/L5**：reported（等效理论量，不作判据）；reported（λ>0 的 CE/MSE 轨迹落盘）

> 口径提醒：率类指标走**配对 bootstrap / Wilson 95% CI**（不跨量纲借用 0.014 bpc 带）；`results.json` **无任何墙钟字段**。

## 臂与口径

- 主干：`../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`（冻结），tied 读出 embed sha256 `566ec50c9d4aecac…`
- 特征 `h_i` = 最终层残差流；全序列前向经非侵入式包装 `_mtp_nodes` 抓取（包装前后主干输出逐位不变，见单测）
- `ar_t1`（参照，非变量）= 冻结主干 node 1 的 t+1 命中率 = 0.67140 [0.66884, 0.67396]
- j=1 ⇒ 预测 `t_2+…` 见契约 §2；k=2 头预测 **t+2 / t+3**

## 数字表（teacher-forced 命中率，越低越差的率类指标）

| 臂 | 种子 | λ | hit@t+2 | Wilson95 | hit@t+3 | Wilson95 | CE@t+2 | MSE@t+2 |
|---|---|---|---|---|---|---|---|---|
| `disc_k4` | 0 | 0.0 | 0.45790 | [0.45518, 0.46063] | 0.33776 | [0.33518, 0.34035] | 2.29818 | 6.20295 |
| `disc_k4` | 1 | 0.0 | 0.45476 | [0.45204, 0.45748] | 0.33800 | [0.33542, 0.34059] | 2.3004 | 6.23785 |
| `lat_k4_lam0` | 0 | 0.0 | 0.45790 | [0.45518, 0.46063] | 0.33776 | [0.33518, 0.34035] | 2.29818 | 6.20295 |
| `lat_k4_lam0` | 1 | 0.0 | 0.45476 | [0.45204, 0.45748] | 0.33800 | [0.33542, 0.34059] | 2.3004 | 6.23785 |
| `lat_k4_lam1` | 0 | 1.0 | 0.48457 | [0.48184, 0.4873] | 0.36483 | [0.3622, 0.36747] | 1.90445 | 2.69509 |
| `lat_k4_lam1` | 1 | 1.0 | 0.48498 | [0.48225, 0.48771] | 0.36142 | [0.3588, 0.36405] | 1.90516 | 2.7047 |

## L0 明细

```json
{
  "0": {
    "pair": [
      "lat_k4_lam0",
      "disc_k4"
    ],
    "hits_bit_identical": true,
    "ce_bit_identical": true,
    "head_weights_bit_identical": true,
    "pass": true,
    "note": "λ=0 两臂走同一代码路径（完全跳过 MSE）；同种子头初始化相同"
  },
  "1": {
    "pair": [
      "lat_k4_lam0",
      "disc_k4"
    ],
    "hits_bit_identical": true,
    "ce_bit_identical": true,
    "head_weights_bit_identical": true,
    "pass": true,
    "note": "λ=0 两臂走同一代码路径（完全跳过 MSE）；同种子头初始化相同"
  },
  "frozen_backbone": {
    "requires_grad_all_false": true,
    "grad_is_none_after_backward": true,
    "pass": true
  }
}
```

## L3 代价轴（等效理论量，禁墙钟）

```json
{
  "d_model": 256,
  "n_layers": 12,
  "backbone_params_frozen": 13734149,
  "per_arm_param_stats": {
    "disc_k4": {
      "k": 4,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 1051648,
      "head_macs_per_position": 1048576,
      "tied_readout_macs_per_position": 210944,
      "backbone_params_frozen": null
    },
    "lat_k4_lam0": {
      "k": 4,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 1051648,
      "head_macs_per_position": 1048576,
      "tied_readout_macs_per_position": 210944,
      "backbone_params_frozen": null
    },
    "lat_k4_lam1": {
      "k": 4,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 1051648,
      "head_macs_per_position": 1048576,
      "tied_readout_macs_per_position": 210944,
      "backbone_params_frozen": null
    }
  },
  "path_i_trunk_layer_passes_per_token": {
    "k=4": 3.0
  },
  "path_ii_trunk_layer_passes_per_token": {
    "k=4": 15.0
  },
  "note": "契约 §4-L3 口径（等效理论量）；阶段 1 只实测路径 (i)（teacher-forced 直接发）；路径 (ii) 的 τ/α₁ 与真实接受成本由阶段 2 落盘"
}
```

## L5 λ 是否在起作用（CE/MSE 轨迹端点）

```json
{
  "disc_k4_s0": {
    "lambda": 0.0,
    "ce_first": 47.33517,
    "ce_last": 6.2468,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "disc_k4_s1": {
    "lambda": 0.0,
    "ce_first": 47.30914,
    "ce_last": 6.24007,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k4_lam0_s0": {
    "lambda": 0.0,
    "ce_first": 47.33517,
    "ce_last": 6.2468,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k4_lam0_s1": {
    "lambda": 0.0,
    "ce_first": 47.30914,
    "ce_last": 6.24007,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k4_lam1_s0": {
    "lambda": 1.0,
    "ce_first": 47.33517,
    "ce_last": 7.51854,
    "mse_first": 25.06113,
    "mse_last": 9.51016,
    "mse_fraction_of_loss_last": 0.55848
  },
  "lat_k4_lam1_s1": {
    "lambda": 1.0,
    "ce_first": 47.30914,
    "ce_last": 7.48525,
    "mse_first": 25.03751,
    "mse_last": 9.50848,
    "mse_fraction_of_loss_last": 0.55953
  }
}
```

## 边界与未测

- 结论只适用于本设定（该冻结 ckpt / char enwik8 / k=2 / 该训练预算 / tied 共享读出），不外推。
- **路径 (ii)**（回归头当草稿再验证，α₁/τ）由**阶段 2**的另一个 worker 做；本文件只覆盖路径 (i)。
- k=4 臂：已测，见上表。
