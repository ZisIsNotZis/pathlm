# MODEL — 2673-robustness-13m

**零新增参数**：本实验只在**冻结主干 + 既有 MTP 头**上做**推理**（温度>0 的无损投机采样 +
53.9M 对抗候选验证），不训练任何东西。权重不入库（`.gitignore` 覆盖 `*.pt`）。

## 1 冻结主干（只读，来自 2631-spec-three-arms/train）

| 名称 | 参数量 | enwik8 val bpc | `model.pt` sha256（前 40 位） | 字节 |
|---|---|---|---|---|
| `13m_s0` | 13,734,149 | 1.5468 | `1df7840bc4813f89709805222cedf34b0384f8ba` | 54,984,075 |
| `13m_s1` | 13,734,149 | 1.5469 | `047893bda15104cc25794dc598d96bca96bbef74` | 54,984,075 |
| `53m_s0` | 54,469,125 | 1.4211 | `0627862468867f1dcdf939adeaa7b65bc78bf92d` | 217,923,979 |

三个 checkpoint 的 path 配置**逐字段相同**：`n_mtp=3`、`chain_mode="token"`、`transport="none"`、
`window=0`、`anchors=0`、`exit_threshold=0.9`。`n_mtp=3` ⇒ 存在 node-1/2/3；`window=0/anchors=0`
⇒ `spec_tree.tree_forward` 的 K 行平铺验证前向可用（带驱逐的树验证未实装，不在本 WP）。

## 2 用到的头（既有，不新训）

- **node-1**（`transforms[0]`）= 目标模型：位置 `N+1` 的下一 token 分布 `p_target`。
- **node-2 / node-3**（`transforms[1]` / `transforms[2]`）= **草稿头**：位置 `dec.n−1` 的
  node-2 / node-3 直连头分别预测 `dec.n+1` / `dec.n+2`，给出草稿分布 `q2` / `q3`
  （= `pathlm.decode.decode_spec` 的默认草稿来源，与 2640/2650 口径一致）。

## 3 复现命令

```bash
cd /home/z/vibe/pathlm
# 单测（CPU，不取锁）
python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/test_spec_lossless.py -q
# 变异（CPU，不取锁）
python3 axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/mutate.py
# 主跑（GPU；per-job §20.1 释放 + ≥60s 让出窗口；detached）
cd axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m
nohup python3 run.py --out . --jobs p1_13m_s0 p1_13m_s1 p2_53m_s0 --resume > /tmp/wp624.nohup 2>&1 &
```

## 4 权重与证据的可回溯

- 主干清单（路径 + sha256 + 参数量 + bpc）在本文件；原始读数在 `results.json`；
  逐 prompt 的 token 头与 NLL 在 `results.json → checkpoints.<job>.per_prompt`。
- 变异报告在 `mutation_report.json`（不含逐次运行耗时）。
- GPU 实耗墙钟只出现在 `run_log.txt`（禁墙钟判据：`results.json` 无墙钟字段）。
