# MODEL · 2631-carry-reseed 权重清单

> 权重（`train/<arm>/model.pt`）**不入库**（`.gitignore` 的 `*.pt`）；本文件记录 sha256 / 体积，
> `run.py` 加载时**逐次断言** sha256（`results.json::checkpoints`）。六个 checkpoint 由
> `train_arms.sh`（12k 步 × bs32 × T256 = 98.3M token / 臂）产出，唯一变量 = `--carry-write`
> （none/plain/delta）与 `--seed`（0/1）。

## 1. 六臂 checkpoint

| 臂 | 写入规则 | seed | 参数量 | 体积 (B) | sha256 |
|---|---|---|---:|---:|---|
| `none_s0` | 无通道 | 0 | 13,536,768 | 54,193,675 | `5e04df917aa0336f84ad4849b917fe8bb2eda2e374e509f408ec2fdcef81444b` |
| `plain_s0` | `S←S+βvkᵀ` | 0 | 13,801,064 | 55,255,619 | `91f8197027b2b1a370d7b5fb897ab28b6f6b568aa8fd99165ab02357282f4c33` |
| `delta_s0` | `S←S(I−βkkᵀ)+βvkᵀ` | 0 | 13,801,064 | 55,255,619 | `d6fff38ca2be1df4239730f89dd816dd242f48d3cb340c3f836e0d63fbe37c9b` |
| `none_s1` | 无通道 | 1 | 13,536,768 | 54,193,675 | `8fd9dff445966025292375eb58633cb4556db6be8605f4d4d23c1568eb46d096` |
| `plain_s1` | `S←S+βvkᵀ` | 1 | 13,801,064 | 55,255,619 | `09ff2fda8792e6029027ffb00b5d8271794a39f69b075e4cb7bbc7bf97652583` |
| `delta_s1` | `S←S(I−βkkᵀ)+βvkᵀ` | 1 | 13,801,064 | 55,255,619 | `7487540111e92cc8d065aa3ee590d7c967da3e4d68406355906b4c109049424e` |

闸门/训练日志：`train/<arm>/{gate_log.jsonl,train_log.jsonl,results.json}`（入 git）。
`config.json::arms[*].sha256` 是同一份哈希（由 `fill_config_hashes.py` 回填，`eval.sh` 先跑）。

## 2. 结构（与 2618 U0/U1/U2 同一定义）

- 主干：stock dense decoder-only（`DepthARModel(dense=True, topology="sequential")`：tied 读出 +
  无 final LayerNorm + 顺序残差），d256 × 12 层 × 8 头 × mlp6 × T256。
- 通道臂：每层加一个读出增益 `γ_ℓ`（零初始化 ⇒ 起步逐位 = 无通道）＋一个**共享记忆槽**
  （`H·dh·dh = 8192` floats，**与 T 无关**），共 **+264,296** 参数（`results.json::selfchecks` 断言
  plain/delta 逐位等参数、通道臂主干逐位 == 无通道臂）。
- 训练/评测可见性：`window_anchor_keep_mask(T=256, window=16, anchors=0)`（可见对数 3976 <
  因果对数 32896，非退化）。

## 3. 复现路径

```bash
# 训练（按臂释放锁，臂间 ≥60s；重活已完成，通常无需重跑）
bash axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed/train_arms.sh
# 评测（纯推理，整程 <10 min，可整程持锁）
bash axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed/eval.sh
```
