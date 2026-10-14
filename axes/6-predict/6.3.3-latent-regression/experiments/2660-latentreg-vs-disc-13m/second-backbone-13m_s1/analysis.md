# analysis — 2660-latentreg-vs-disc-13m（阶段 1：直接发 / teacher-forced）

## 人话摘要

我们在**同一份冻结的 13M 主干**上，只改「未来位置怎么产出」这一件事：一种是直接把主干隐状态经几个小变换头**分类到词表**（离散臂），另一种是让这些头**回归到主干在未来位置的真实隐状态**（回归臂，多一个 λ·MSE 项把输出拉向真值隐状态），两者共用同一份 tied 读出、同一参数量、同一训练预算（3000 步）。
结果：同参数量下，离散臂在 t+2 命中率 0.460、t+3 命中率 0.339；冻结主干自己的 AR 下一 token（t+1）命中率是 0.671。

- **L0（实现保真硬门）**：**通过** —— `lat_k2_lam0` 与 `disc_k2` 同种子逐位相同（命中率 / CE / 头权重
  ）；冻结主干 `requires_grad=False` 且一次 backward 后 `.grad is None`。
- **L1（主判，路径 (i) 直接发）**：采纳（t+2 配对 95% CI 不含 0：lat_k2_lam1）
  - `lat_k2_lam1` vs `disc_k2`：t+2 Δ=+0.03286 CI[+0.03122,+0.03450] 不含0；t+3 Δ=+0.02795 CI[+0.02648,+0.02947] 不含0
- **L3/L5**：reported（等效理论量，不作判据）；reported（λ>0 的 CE/MSE 轨迹落盘）

> 口径提醒：率类指标走**配对 bootstrap / Wilson 95% CI**（不跨量纲借用 0.014 bpc 带）；`results.json` **无任何墙钟字段**。

## 臂与口径

- 主干：`../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1`（冻结），tied 读出 embed sha256 `da5b5d3343655963…`
- 特征 `h_i` = 最终层残差流；全序列前向经非侵入式包装 `_mtp_nodes` 抓取（包装前后主干输出逐位不变，见单测）
- `ar_t1`（参照，非变量）= 冻结主干 node 1 的 t+1 命中率 = 0.67147 [0.6689, 0.67402]
- j=1 ⇒ 预测 `t_2+…` 见契约 §2；k=2 头预测 **t+2 / t+3**

## 数字表（teacher-forced 命中率，越低越差的率类指标）

| 臂 | 种子 | λ | hit@t+2 | Wilson95 | hit@t+3 | Wilson95 | CE@t+2 | MSE@t+2 |
|---|---|---|---|---|---|---|---|---|
| `disc_k2` | 0 | 0.0 | 0.45983 | [0.45711, 0.46254] | 0.33908 | [0.33651, 0.34166] | 2.26285 | 6.031 |
| `disc_k2` | 1 | 0.0 | 0.46139 | [0.45868, 0.46411] | 0.33812 | [0.33555, 0.3407] | 2.25778 | 6.01929 |
| `lat_k2_lam0` | 0 | 0.0 | 0.45983 | [0.45711, 0.46254] | 0.33908 | [0.33651, 0.34166] | 2.26285 | 6.031 |
| `lat_k2_lam0` | 1 | 0.0 | 0.46139 | [0.45868, 0.46411] | 0.33812 | [0.33555, 0.3407] | 2.25778 | 6.01929 |
| `lat_k2_lam1` | 0 | 1.0 | 0.49351 | [0.49079, 0.49623] | 0.36607 | [0.36345, 0.3687] | 1.87231 | 2.60563 |
| `lat_k2_lam1` | 1 | 1.0 | 0.49342 | [0.4907, 0.49615] | 0.36703 | [0.36441, 0.36966] | 1.87644 | 2.6062 |

## L0 明细

```json
{
  "0": {
    "hits_bit_identical": true,
    "ce_bit_identical": true,
    "head_weights_bit_identical": true,
    "pass": true,
    "note": "λ=0 两臂走同一代码路径（完全跳过 MSE）；同种子头初始化相同"
  },
  "1": {
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
    "disc_k2": {
      "k": 2,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 525824,
      "head_macs_per_position": 524288,
      "tied_readout_macs_per_position": 105472,
      "backbone_params_frozen": null
    },
    "lat_k2_lam0": {
      "k": 2,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 525824,
      "head_macs_per_position": 524288,
      "tied_readout_macs_per_position": 105472,
      "backbone_params_frozen": null
    },
    "lat_k2_lam1": {
      "k": 2,
      "d": 256,
      "vocab": 206,
      "params_per_head": 262912,
      "head_params_total": 525824,
      "head_macs_per_position": 524288,
      "tied_readout_macs_per_position": 105472,
      "backbone_params_frozen": null
    }
  },
  "path_i_trunk_layer_passes_per_token": {
    "k=2": 6.0
  },
  "path_ii_trunk_layer_passes_per_token": {
    "k=2": 18.0
  },
  "note": "契约 §4-L3 口径（等效理论量）；阶段 1 只实测路径 (i)（teacher-forced 直接发）；路径 (ii) 的 τ/α₁ 与真实接受成本由阶段 2 落盘"
}
```

## L5 λ 是否在起作用（CE/MSE 轨迹端点）

```json
{
  "disc_k2_s0": {
    "lambda": 0.0,
    "ce_first": 22.85831,
    "ce_last": 2.44112,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "disc_k2_s1": {
    "lambda": 0.0,
    "ce_first": 22.70677,
    "ce_last": 2.44325,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k2_lam0_s0": {
    "lambda": 0.0,
    "ce_first": 22.85831,
    "ce_last": 2.44112,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k2_lam0_s1": {
    "lambda": 0.0,
    "ce_first": 22.70677,
    "ce_last": 2.44325,
    "mse_first": 0.0,
    "mse_last": 0.0,
    "mse_fraction_of_loss_last": 0.0
  },
  "lat_k2_lam1_s0": {
    "lambda": 1.0,
    "ce_first": 22.85831,
    "ce_last": 3.08493,
    "mse_first": 11.98111,
    "mse_last": 4.10266,
    "mse_fraction_of_loss_last": 0.5708
  },
  "lat_k2_lam1_s1": {
    "lambda": 1.0,
    "ce_first": 22.70677,
    "ce_last": 3.11304,
    "mse_first": 11.95057,
    "mse_last": 4.10095,
    "mse_fraction_of_loss_last": 0.56847
  }
}
```

## 边界与未测

- 结论只适用于本设定（该冻结 ckpt / char enwik8 / k=2 / 该训练预算 / tied 共享读出），不外推。
- **路径 (ii)**（回归头当草稿再验证，α₁/τ）由**阶段 2**的另一个 worker 做；本文件只覆盖路径 (i)。
- **k=4 臂未测**（时间盒）：下一步 = 同 harness 加 `disc_k4`/`lat_k4_lam1`，成本 ≈ 每臂同 3000 步头训练（特征缓存复用，GPU 约 +1 min/臂）。
