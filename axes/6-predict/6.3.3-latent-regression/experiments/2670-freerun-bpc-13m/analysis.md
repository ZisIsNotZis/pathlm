# analysis — 2670-freerun-bpc-13m（路径 (i)「一次前向直接发 k 个位置」的 free-run 序列级 bpc）

## 人话摘要

我们在**同一份冻结的 13M 主干**上，只改「未来位置怎么产出」这一件事：`ar` 是普通逐 token 自回归贪心（每步 1 次前向发 1 个 token）；`path_i` 是「估计得贪心」——一次前向拿最后一个位置的隐状态，经 2660 训好的 2 个头各自 argmax，**一次发 2 个 token、不做任何验证**。两个臂都按「自己发出的 token 在自家预测分布下的 NLL」算自由生成的**序列级 bpc**（越低越好），prompt 不计分。

结果（13m_s0）：`ar` 的 bpc 是 **0.5044**；一次发 2 个位置的两个头种子，`disc_k2` 是 **1.1359–1.3507**、`lat_k2_lam1` 是 **0.7005–1.1660**。相对 `ar` 的质量代价（两种子合并）：`disc_k2` Δbpc = **+0.7389** （95% CI [+0.5745, +0.8918]，不含 0，越 0.014 bpc 带；人话量级：每字符多花 0.74 bit，相当于困惑度约 ×1.67）；`lat_k2_lam1` Δbpc = **+0.4289** （95% CI [+0.2817, +0.5867]，不含 0，越 0.014 bpc 带）——即 λ=1.0 的监督把代价从 +0.74 拉到 +0.43 bpc，但**未拉回 0.014 带内**。

- **真值锚（复核 N1）**：外部真值锚口径下的 Δ 与自评口径**同向**且**仍越 0.014 带** ⇒ 原判读（该尺度质量面不成立）**成立**，不是 `ar` 流退化自抬 Δ 的伪影。

- **L0（保真硬门）**：**通过** —— 头权重逐张量等于 2660 落盘文件；主干 `requires_grad=False` 且 backward 后 `.grad is None`；`ar` 贪心流与 2640 `native`（生产 `decode` 贪心，prompt_len=32）逐位相同。
- **L1（主判）**：逐主干 × 头种子给出 Δbpc 点值 + prompt 聚类配对 bootstrap 95% CI 与是否越 0.014 带。
- **L2（n≥2 一致）**：见数字表（两主干点 × 两头种子的 Δ 符号）。
- **L3（代价轴）**：路径 (i) tokens-per-forward = 2、每 token 主干等效 layer-passes ≈ 6.0；`ar` 为 1 / 12。
- **L4（禁墙钟自检）**：`results.json` 墙钟语义字段命中 **0**（通过）。
- **L5（确定性）**：True

## 契约 §5 否证条件触发情况

- §5-1 触发：Δbpc(disc_k2 − ar) 的 CI 不含 0 且 |Δ| 越 0.014 带（全部窗口：13m_s0/0, 13m_s0/1, 13m_s0/combined_mean_over_seeds, 13m_s1/0, 13m_s1/1, 13m_s1/combined_mean_over_seeds），而 lat_k2_lam1 只把 |Δ| 缩小、同向、仍越带（13m_s0/0, 13m_s0/1, 13m_s0/combined_mean_over_seeds, 13m_s1/0, 13m_s1/1, 13m_s1/combined_mean_over_seeds）⇒ 判「路径 (i)『一次前向直接发 k 个位置』在本尺度（13M 冻结主干、char enwik8、温度 0、贪心）质量面不成立」
- §5-2 未触发（部分）：Δbpc(lat_k2_lam1 − disc_k2) 仅在 13m_s0/0 窗口（1/6）CI 含 0；其余窗口 CI 不含 0 且 Δ<0（lat 的 bpc 更低）⇒ λ=1.0 的 latent 匹配监督在多数窗口降低 free-run bpc
- §5-3 未触发：至少一条 Δbpc 的 CI 不含 0

## 臂与口径

- 主干 `13m_s0`：`axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`（冻结），embed sha256 `566ec50c9d4aecac…`
- 主干 `13m_s1`：`axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1`（冻结），embed sha256 `da5b5d3343655963…`
- prompt = 12 个固定串（长度 64，stride 5000），offset = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
- 每 prompt 生成 200 token；bpc = Σ−ln p/(n·ln2)，prompt 不计分。
- 主判 = Δbpc（候选 − 同主干 `ar`）的 prompt 聚类配对 bootstrap (B=2000, seed=2670)。

## 数字表：臂级 bpc

| 臂 | 主干 | 种子 | bpc | tokens-per-forward |
|---|---|---|---|---|
| `ar` | `13m_s0` | — | 0.504356 | 1 |
| `ar` | `13m_s1` | — | 0.489330 | 1 |
| `disc_k2` | `13m_s0` | 0 | 1.350679 | 2 |
| `disc_k2` | `13m_s0` | 1 | 1.135920 | 2 |
| `disc_k2` | `13m_s1` | 0 | 1.487249 | 2 |
| `disc_k2` | `13m_s1` | 1 | 1.330504 | 2 |
| `lat_k2_lam1` | `13m_s0` | 0 | 1.166020 | 2 |
| `lat_k2_lam1` | `13m_s0` | 1 | 0.700527 | 2 |
| `lat_k2_lam1` | `13m_s1` | 0 | 0.576513 | 2 |
| `lat_k2_lam1` | `13m_s1` | 1 | 0.728699 | 2 |

## 数字表：Δbpc（候选 − 同主干 ar），prompt 聚类配对 bootstrap 95% CI

| 候选臂 | 主干 | 种子 | Δbpc | 95% CI | 不含0 | |Δ|越0.014带 |
|---|---|---|---|---|---|---|
| `disc_k2` | `13m_s0` | 0 | +0.846322 | [+0.623191, +1.034831] | True | True |
| `disc_k2` | `13m_s0` | 1 | +0.631564 | [+0.429153, +0.800963] | True | True |
| `disc_k2` | `13m_s0` | combined_mean_over_seeds | +0.738943 | [+0.574457, +0.891811] | True | True |
| `disc_k2` | `13m_s1` | 0 | +0.997918 | [+0.890104, +1.109797] | True | True |
| `disc_k2` | `13m_s1` | 1 | +0.841174 | [+0.638508, +1.031297] | True | True |
| `disc_k2` | `13m_s1` | combined_mean_over_seeds | +0.919546 | [+0.792719, +1.055460] | True | True |
| `lat_k2_lam1` | `13m_s0` | 0 | +0.661664 | [+0.350147, +0.949180] | True | True |
| `lat_k2_lam1` | `13m_s0` | 1 | +0.196171 | [+0.081036, +0.313283] | True | True |
| `lat_k2_lam1` | `13m_s0` | combined_mean_over_seeds | +0.428918 | [+0.281697, +0.586722] | True | True |
| `lat_k2_lam1` | `13m_s1` | 0 | +0.087183 | [-0.087787, +0.279353] | False | True |
| `lat_k2_lam1` | `13m_s1` | 1 | +0.239368 | [-0.037759, +0.562803] | False | True |
| `lat_k2_lam1` | `13m_s1` | combined_mean_over_seeds | +0.163276 | [-0.044309, +0.405192] | False | True |

## 数字表：Δbpc(lat_k2_lam1 − disc_k2)（同主干同种子配对）

| 主干 | 种子 | Δbpc | 95% CI | 不含0 |
|---|---|---|---|---|
| `13m_s0` | 0 | -0.184658 | [-0.474051, +0.095643] | False |
| `13m_s0` | 1 | -0.435393 | [-0.654676, -0.179417] | True |
| `13m_s0` | combined_mean_over_seeds | -0.310025 | [-0.490155, -0.124275] | True |
| `13m_s1` | 0 | -0.910736 | [-1.079023, -0.729535] | True |
| `13m_s1` | 1 | -0.601805 | [-0.884806, -0.272332] | True |
| `13m_s1` | combined_mean_over_seeds | -0.756270 | [-0.968634, -0.541095] | True |

## L0 明细

```json
{
  "heads_byte_equal_2660": true,
  "frozen_backbone": {
    "13m_s0": {
      "requires_grad_all_false": true,
      "grad_is_none_after_backward": true,
      "pass": true
    },
    "13m_s1": {
      "requires_grad_all_false": true,
      "grad_is_none_after_backward": true,
      "pass": true
    }
  },
  "ar_vs_2640_native": {
    "ref": "axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/results.json",
    "ref_ckpt": "13m_s0",
    "ref_arm": "native",
    "prompt_len": 32,
    "n_head_tokens": 8,
    "n_prompts_matched": 12,
    "n_prompts": 12,
    "all_match": true,
    "per_prompt": [
      {
        "prompt": 0,
        "offset": 0,
        "got_head8": [
          78,
          78,
          91,
          2,
          85,
          87,
          82,
          82
        ],
        "ref_head8": [
          78,
          78,
          91,
          2,
          85,
          87,
          82,
          82
        ],
        "identical": true
      },
      {
        "prompt": 1,
        "offset": 5000,
        "got_head8": [
          31,
          31,
          31,
          2,
          54,
          74,
          71,
          2
        ],
        "ref_head8": [
          31,
          31,
          31,
          2,
          54,
          74,
          71,
          2
        ],
        "identical": true
      },
      {
        "prompt": 2,
        "offset": 10000,
        "got_head8": [
          86,
          74,
          2,
          81,
          72,
          2,
          61,
          61
        ],
        "ref_head8": [
          86,
          74,
          2,
          81,
          72,
          2,
          61,
          61
        ],
        "identical": true
      },
      {
        "prompt": 3,
        "offset": 15000,
        "got_head8": [
          2,
          86,
          74,
          71,
          2,
          85,
          86,
          67
        ],
        "ref_head8": [
          2,
          86,
          74,
          71,
          2,
          85,
          86,
          67
        ],
        "identical": true
      },
      {
        "prompt": 4,
        "offset": 20000,
        "got_head8": [
          86,
          74,
          71,
          2,
          61,
          61,
          35,
          79
        ],
        "ref_head8": [
          86,
          74,
          71,
          2,
          61,
          61,
          35,
          79
        ],
        "identical": true
      },
      {
        "prompt": 5,
        "offset": 25000,
        "got_head8": [
          67,
          63,
          63,
          2,
          67,
          80,
          70,
          2
        ],
        "ref_head8": [
          67,
          63,
          63,
          2,
          67,
          80,
          70,
          2
        ],
        "identical": true
      },
      {
        "prompt": 6,
        "offset": 30000,
        "got_head8": [
          72,
          2,
          81,
          72,
          2,
          86,
          74,
          71
        ],
        "ref_head8": [
          72,
          2,
          81,
          72,
          2,
          86,
          74,
          71
        ],
        "identical": true
      },
      {
        "prompt": 7,
        "offset": 35000,
        "got_head8": [
          81,
          87,
          86,
          2,
          86,
          74,
          71,
          2
        ],
        "ref_head8": [
          81,
          87,
          86,
          2,
          86,
          74,
          71,
          2
        ],
        "identical": true
      },
      {
        "prompt": 8,
        "offset": 40000,
        "got_head8": [
          86,
          74,
          71,
          2,
          61,
          61,
          55,
          80
        ],
        "ref_head8": [
          86,
          74,
          71,
          2,
          61,
          61,
          55,
          80
        ],
        "identical": true
      },
      {
        "prompt": 9,
        "offset": 45000,
        "got_head8": [
          67,
          75,
          85,
          81,
          80,
          80,
          67,
          68
        ],
        "ref_head8": [
          67,
          75,
          85,
          81,
          80,
          80,
          67,
          68
        ],
        "identical": true
      },
      {
        "prompt": 10,
        "offset": 50000,
        "got_head8": [
          75,
          85,
          75,
          81,
          80,
          85,
          2,
          67
        ],
        "ref_head8": [
          75,
          85,
          75,
          81,
          80,
          85,
          2,
          67
        ],
        "identical": true
      },
      {
        "prompt": 11,
        "offset": 55000,
        "got_head8": [
          81,
          80,
          2,
          85,
          86,
          67,
          86,
          71
        ],
        "ref_head8": [
          81,
          80,
          2,
          85,
          86,
          67,
          86,
          71
        ],
        "identical": true
      }
    ],
    "note": "2640 用 prompt_len=32；本实验主口径 prompt_len=64（契约 §3），故仅用 32 长度做逐位保真比对，64 长度主口径不做跨长度比对"
  },
  "pass": true
}
```

## L3 代价轴（等效理论量，禁墙钟）

```json
{
  "n_layers": 12,
  "d_model": 256,
  "ar": {
    "tokens_per_forward": 1,
    "trunk_layer_passes_per_token": 12,
    "sequential_forwards_per_token": 1.0
  },
  "path_i": {
    "tokens_per_forward": 2,
    "trunk_layer_passes_per_token": 6.0,
    "sequential_forwards_per_token": 0.5
  },
  "note": "契约 §6 等效理论量；路径 (i) 同一轮内 k 个 token 无顺序依赖（可并行），轮间顺序依赖。禁墙钟。"
}
```

## 生成流 sha256（可复现凭证，逐臂）

```json
{
  "ar": {
    "13m_s0": "582f0cca822b633f94e20fcc8a3951cc4280f51610f115bdfda82d30db320240",
    "13m_s1": "cae8aaeb568d62d863e2a5074c208e7e27d03e7a5ab2933f8cc47ea60e7dfc56"
  },
  "disc_k2": {
    "13m_s0_s0": "5e63843b10e0e5f078692709ddf3131654c2f2a7880256feb8bc63730ef5bfed",
    "13m_s0_s1": "1b2ef97ac2eb8d8c1ad2d7b2406fb7dd5a139fa5dec5d611dacedd0eb444979f",
    "13m_s1_s0": "073280bf1364129f715edd2a82d4cda55fd2f301eec5c7721d8afda0eb9f6d33",
    "13m_s1_s1": "f0981507752d5b9b57e84da52bc6ed3d50c7be711c09b8a83b83b14f2e87250e"
  },
  "lat_k2_lam1": {
    "13m_s0_s0": "886933b837bcd2b44842fb8f8dc5137efdc6bf66d4a2ac3dba8f50e68e1bf6c0",
    "13m_s0_s1": "68014678e00a3283f54e47098bd2cf8613e838608de35203c99eda7852fcd2f6",
    "13m_s1_s0": "808aa231e010c853adc5f8b3a8c01fd17b49992924c87a2217a9e856b7a82464",
    "13m_s1_s1": "1c04270b60215aa60e9addde12f133f35e36409f88c984cd02f753916c7325b3"
  }
}
```

## 真值锚（复核 N1 的补充读数）

独立复核（`REVIEW-C1-2670.md` 的 **N1**）指出：主判 Δbpc 是**两臂各自对自家生成流的自评 NLL 之差**，而不是「对真值的质量代价」；若 `ar` 流退化（重复/自洽），其自评 bpc 被压低，会系统性抬高 Δ。该隐忧未被落盘数据排除（原产物只有 prompt 级 NLL 与 sha256，无 token 流）。本节**只加读数**（不改预登记的臂与判据）：把每个（臂 × 主干 × 头种子 × prompt）的 200-token 生成流落盘（`streams/`），对其算**客观退化统计**（不需模型），并用 `pathlm/eval.py::bpc` 同口径的 **teacher-forced node-1 bpc** 分别在**自评主干**与**另一主干（外部锚）**下打分；再给**真实 enwik8 续写**的 teacher-forced bpc 作为水平参照。

- **真实 enwik8 续写（水平参照）**：`13m_s0` 1.6040 bpc、`13m_s1` 1.6027 bpc（每 prompt 200 个真实 val 字符、teacher-forced，同口径）。这是「自然文本」同一主干的水平线，供判断生成流的绝对值落在何处。

### Δ 三读法对照（自评 free-run vs 自评 teacher-forced vs 外部锚 teacher-forced）

| 候选臂 | 主干 | 自评 free-run（原判） | 自评 teacher-forced | 外部锚 teacher-forced | 外部锚同向 |
|---|---|---|---|---|---|
| `disc_k2` | `13m_s0` | +0.7389 [+0.5745,+0.8918] | +4.1031 [+3.5680,+4.6081] | +4.1541 [+3.5819,+4.6720] | 是 |
| `disc_k2` | `13m_s1` | +0.9195 [+0.7927,+1.0555] | +5.0469 [+4.6525,+5.3490] | +5.0708 [+4.6734,+5.3843] | 是 |
| `lat_k2_lam1` | `13m_s0` | +0.4289 [+0.2817,+0.5867] | +1.4521 [+1.0848,+1.8845] | +1.2611 [+0.9483,+1.6367] | 是 |
| `lat_k2_lam1` | `13m_s1` | +0.1633 [-0.0443,+0.4052] | +1.1296 [+0.5755,+1.8335] | +1.1701 [+0.6315,+1.8453] | 是 |

### 客观退化统计（每臂 × 主干，12 prompt 平均）

| 臂 | 主干 | 种子 | 最长连续相同 token | distinct-2 | 重复 4-gram 率 |
|---|---|---|---|---|---|
| `ar` | `13m_s0` | none | 2.08 | 0.2454 | 0.6612 |
| `ar` | `13m_s1` | none | 2.00 | 0.2203 | 0.7069 |
| `disc_k2` | `13m_s0` | s0 | 21.67 | 0.4531 | 0.2297 |
| `disc_k2` | `13m_s0` | s1 | 59.33 | 0.3740 | 0.3579 |
| `disc_k2` | `13m_s1` | s0 | 3.83 | 0.4853 | 0.1142 |
| `disc_k2` | `13m_s1` | s1 | 25.75 | 0.4883 | 0.1764 |
| `lat_k2_lam1` | `13m_s0` | s0 | 99.83 | 0.1357 | 0.7851 |
| `lat_k2_lam1` | `13m_s0` | s1 | 153.83 | 0.1374 | 0.7889 |
| `lat_k2_lam1` | `13m_s1` | s0 | 151.58 | 0.1319 | 0.7872 |
| `lat_k2_lam1` | `13m_s1` | s1 | 143.50 | 0.1461 | 0.7623 |

### 要点（人话）

- **同一 scorer 下的真值代价远大于自评代价**：原 Δ 是两臂各自用**自家预测头**自评之差（`ar` 用 node-1、候选臂用自家 2 个头）；而 `ar` 的自由生成**就是** node-1 贪心，所以它的自评 bpc 与 teacher-forced node-1 bpc **恒等**（13m_s0 0.5044、13m_s1 0.4893）。换成**同一个 scorer**（自评主干 node-1，或另一主干的 node-1）对**两条流**打分后，候选流的 teacher-forced bpc 明显更高（disc 4.3–5.8、lat 1.5–2.3 bpc，而 `ar` 流只有 0.49–0.57），故外部锚 Δ 在全部 4 个窗口都**比自评 free-run Δ 更大**（是；注意不是 teacher-forced 自评列：`lat_k2_lam1 / 13m_s0` 外部锚 +1.2611 略小于 teacher-forced 自评 +1.4521）——复核担心的「自评高估 Δ」在数据上反过来了：**自评低估**了真实代价。
- **`ar` 流不是长程退化**：落盘的 `ar` 流最长连续相同 token 平均 2.04、distinct-2 0.233（长程看是正常文本，只是短模式重复偏多）；复核担心的「`ar` 流退化 ⇒ 压低自评 bpc⇒ 抬高 Δ」这一具体机制在落盘数据上**不成立**。反面：候选臂（尤其 `lat_k2_lam1`）才明显退化（最长连续相同 token 平均 ~137，即大半条流是同一个字符）。
- **水平参照**：真实 enwik8 续写的 teacher-forced bpc 为 1.6040（s0）/ 1.6027（s1）——候选流的 teacher-forced bpc（1.5–5.8）远高于自然文本水平线，说明它们确实偏离自然文本，而非只是 scorer 偏差。

### 判读

- 外部真值锚口径下的 Δ 与自评口径**同向**且**仍越 0.014 带** ⇒ 原判读（该尺度质量面不成立）**成立**，不是 `ar` 流退化自抬 Δ 的伪影。
- 每个生成流的三读数（自评 teacher-forced bpc、外部锚 teacher-forced bpc、真实续写 teacher-forced bpc）与逐 prompt 值见 `results.json::truth_anchor.per_stream`；生成流原始 token 见 `streams/*.json`，其 sha256 与 `results.json::arms.*.stream_sha256_*` 逐一相等（`streams.all_sha_reproduced=True`）。

## 边界与未测

- **口径**：这里的 bpc 是 **free-run 自评**（对**模型自己贪心发出的 token** 按其**自家**预测分布算 NLL），不是对真实文本的 teacher-forced CE。故其**绝对值**不可与 2631 的 teacher-forced `bpc` = 1.5468（`pathlm/eval.py::bpc`，node-1 CE on clean val）直接比：自评奖励「对自己的输出有信心」（`ar` 自评 bpc 只有 0.50，低于同一主干的 teacher-forced 1.55），本实验的判据是**同一 prompt 上两臂的配对 Δ**，不是绝对值。复核 N1 之后，上述「自评 vs 真值」的缺口已由**上节「真值锚」**补上（生成流落盘 + 客观退化统计 + teacher-forced 自评/外部锚/真实续写三读数），故本节不再只是自评口径的声明。
- 结论只适用于本设定（该冻结 13M 主干 / char enwik8 / k=2 / 温度 0 贪心 / 共享 tied 读出），不外推；只有 13M 一种规模。
- 路径 (i) 每轮只发 k=2 个 token（未测 k=4）；token 级贪心无温度采样对照。
- 路径 (ii)（当草稿再验证）不在本实验重复；见 2660 阶段 2。
- 墙钟只出现在 `run_log.txt`，不作判据。
