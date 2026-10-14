# MODEL — 2670-freerun-bpc-13m

本实验**零新增/零训练参数**：主干与 2660 的头权重全部**冻结、只读复用**。

## 1 冻结主干（只读，来自 2631）

| 主干 | 目录 | `model.pt` sha256 | tied 读出 embed sha256 | 参数 |
|---|---|---|---|---|
| `13m_s0` | `axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0` | `1df7840bc4813f89709805222cedf34b0384f8bae0a04e23f9ffd2818b2d4e4d` | `566ec50c9d4aecac6ab420dae3578236514eab426c66d548a35b04fca8ef8b3b` | 13734149 |
| `13m_s1` | `axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1` | `047893bda15104cc25794dc598d96bca96bbef7449bd5d45518e78e205a31b59` | `da5b5d3343655963d2ec56c408a9cae084e18b7f2bb382c6872c310da79fbb29` | 13734149 |

## 2 复用的头（只读，来自 2660）

| 臂 | 主干 | 种子 | 文件 | 文件 sha256 | 逐张量等于源文件 |
|---|---|---|---|---|---|
| `disc_k2` | `13m_s0` | 0 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/heads/disc_k2_s0.pt` | `0254ea87cdb335ba48b39ce4c48b5a08f648515b06823dd235b31b61c307a9b6` | True |
| `disc_k2` | `13m_s0` | 1 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/heads/disc_k2_s1.pt` | `68fc3d575fec731a5f7aabc54909e296addec8584c81ae1efe8abfdc4f55ed0d` | True |
| `disc_k2` | `13m_s1` | 0 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/second-backbone-13m_s1/heads/disc_k2_s0.pt` | `6e5c2d5c5f440f392b42ed29e779279b5e1b7f043a700a20c37ddf4648552fba` | True |
| `disc_k2` | `13m_s1` | 1 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/second-backbone-13m_s1/heads/disc_k2_s1.pt` | `4d2f8efb888a3aab1c729a9654ea2ad57244b4821b9ac9760e746606570c6c7e` | True |
| `lat_k2_lam1` | `13m_s0` | 0 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/heads/lat_k2_lam1_s0.pt` | `b17d15980d76463acfc447575774310c78f332e1f61cc0d3b02a5e89d2a31756` | True |
| `lat_k2_lam1` | `13m_s0` | 1 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/heads/lat_k2_lam1_s1.pt` | `c42a98dd17799892d873c3db3ea326279f9efb754dbaea9e3b0ae2237ff86696` | True |
| `lat_k2_lam1` | `13m_s1` | 0 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/second-backbone-13m_s1/heads/lat_k2_lam1_s0.pt` | `7a67a7c64a8f692b6cbcd7c03efc9bba88c2404896bed598cd7a6a8cd289bbaf` | True |
| `lat_k2_lam1` | `13m_s1` | 1 | `axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/second-backbone-13m_s1/heads/lat_k2_lam1_s1.pt` | `e0a508a3bfe1bc53ade21513fb45a3ccc27c75c4452da13010f81f5c5b56fb70` | True |

> `.pt` 的 sha256 只标识文件字节（zip entry 含保存时刻）；内容同一性以 `heads_equal_vs_file`
> 的逐张量 `torch.equal` 为准（见 `results.json::head_files.*.bytes_equal_source_file`）。

## 3 生成口径

- prompt = 12 个固定串 `val[i*5000 : i*5000+64]`（offset 见 `results.json::config.prompt_offsets`），每 prompt 生成 200 个 token，prompt 不计分。
- `ar`：`node-1 = model.transforms[0]`，`logits = z @ embed.weight^T`，逐 token 贪心（1 前向/token）。
- `path_i`：一次冻结主干前向 → 取最后位置最终层隐状态 `h_i` → 对 j=1..k 用头 `head_{j-1}` → `logits_j = head_{j-1}(h_i) @ embed.weight^T` → 各自 argmax、全部追加（不验证）。
- bpc = `Σ −ln p(emitted) / (n_tokens · ln2)`（自家预测分布下的 NLL）。

## 4 复现命令

```bash
cd /home/z/vibe/pathlm
python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/test_freerun.py -q
python3 axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/run.py --out <本目录>
```
