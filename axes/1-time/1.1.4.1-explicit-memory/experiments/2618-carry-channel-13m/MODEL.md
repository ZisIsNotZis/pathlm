# MODEL · 2618 三臂对照（13M dense；U0 无通道 / U1 plain 写 / U2 delta 写）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
sha256 由 `fill_config_hashes.py` 回填进 `config.json`，`run.py` 每次加载**断言**。

## 三臂

| 臂 | 文件 | 架构 | 参数 | 体积 | SHA-256 |
|---|---|---|---|---|---|
| `U0` 无通道 | `train/U0/model.pt` | stock dense（`carry_rule=None`，逐位还原 v1） | 13,536,768 | 54,193,675 B | `8e8ec726a8220b78a790be48d43e2f6e2847a0963dfb729b860aeb5484278dd3` |
| `U1` plain 写 | `train/U1/model.pt` | + 单共享槽（plain 累加写） | 13,801,064 | 55,255,619 B | `9b74ddd28087541653df6212602386c7d523433e7460f23a8c65a32121c8d258` |
| `U2` delta 写 | `train/U2/model.pt` | + 单共享槽（delta 残差写） | 13,801,064 | 55,255,619 B | `40ec6472fee5be3b8d3650ee74f541eb388f6d86e525942d44479e7279923420` |
| `U1rep` 重跑（判据⑥） | `train/U1rep/model.pt` | **与 U1 逐字段相同**（同配置同 seed 重跑） | 13,801,064 | 55,255,619 B | `7cdbb6f56266fd5cd2ac9633dad08c89010cc999508cd3597253bb131e09d8f7` |

- **U1 与 U2 逐位等参数**（同一组张量、同一初始化，唯一变量 = 写入规则）：
  `run.py` 每次运行都做 `param_equality(U1, U2)` 断言并落盘 `results.json::selfchecks`。
- **通道臂的主干参数与 U0 逐位相同**（同 seed；新增参数在其余参数 init 之后才抽随机数）：
  `trunk_equality(U1, U0)` 断言并落盘。
- 新增参数 = 264,296（+1.95%）：`w_k/w_v/w_q`（各 d²）+ `beta_w`（H·d）+ `beta_b`（H）
  + `gate_w`（d²）+ 每层读出增益 `carry_scale`（H）。

## 生成命令（三臂逐字段一致，唯一变量 = `--carry-write`）

```bash
cd axes/M-carrier/M1-carrier-type/experiments/2618-carry-channel-13m
for ARM in U0:none U1:plain U2:delta; do
  NAME=${ARM%%:*}; RULE=${ARM##*:}
  python3 ../../../../../train_depth_ar.py $NAME \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --carry-write $RULE --carry-chunk 64 \
    --eval-batches 1 --out-root train
done
```

评测：`python3 run.py`（同目录，配置 `config.json`）→ `results.json`。

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 三臂全电池：前提校验 + 主腿（含毁源对照）+ 距离腿 + 判据 + 活性 + 显存 + 成本 |
| `premise_check.json` | 判据 ① 单独可跑的前提校验（U0 的准确率/CI/距离分布/局部先验/变异验证） |
| `divergence.json` | 判据 ⑥ 的重跑散度（同配置重跑一臂 + 两份训练日志的答案行 CE 轨迹） |
| `train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}` | 训练日志（每 200 步含通道活性量）与训练器电池 |
| `mutation_report.json` | 7 处语义开关的变异验证（全部抓红 + 逐字节还原） |
| `reconcile_report.json` | `reconcile_numbers.py` 的报告/结果逐条对账 |
| `run_log.jsonl` / `run_log.txt` | 评测电池的逐步日志（`run_log.txt` 为同一次运行的 stdout） |
| `controls/README.md` | 对照臂索引（无通道基线 / 毁源对照 / 重跑臂）与方差界（SCHEMA §3 的 `controls/` 要求） |
| `analysis.md` / `stability.md` | 判读 / 稳定性与可复现性专报 |

## 代码指纹（`results.json::code`，与本文档同一修订（**注**：`mutation_report.json` 的 `pathlm/depth_ar.py` 指纹为 `72499b5b…`，与当前 `a141ef42…` 不同——变异验证跑在该文件最后一次编辑之前；被变异的两段代码在当前文件中仍逐字存在，故覆盖可人工复核，但**不声称在同一修订上跑过**））

| `run.py` | `8a73cd554aee27655c08467929fc035c83ece470d38a866f9fb4dfc98571fe8a` |
| `premise_check.py` | `d0c7dad925283f6bd3713c6519f673b54147f9b6a9b080fa708e68b64f1be98b` |
| `divergence.py` | `8b07ef2f62e5803a387aa276b71148cdc99cc536e652bab969c0a35940996520` |
| `mutate.py` | `7bbfa32c46eb69f6409c13d322c37d052dde6ab13fd75c8b0bbcade6e4ebb955` |
| `reconcile_numbers.py` | `3c037ee3c790e5108983e5f6b8cbef5961df275707b53b561c95d1275df29932` |
| `config.json` | `d61ac2612b13d9e40a9d6cd0c28e11bcce7ab413cb8445eab43d204cb51faa7e` |
| `README.md` | `4c82f2f4e5991dc8f30d8580b3b8d4e65ba147a836e45f4cc2c6bfe339e1a799` |
| `pathlm/depth_ar.py` | `a141ef4295666ef6f25b54006cae7c1836ff9bba736c30c330c41cba6ad7673f` |
| `train_depth_ar.py` | `093c628c837bf38f7c66cb91bbd6dab8fa8a0558a8bbb0054e437a3e80a409aa` |
| `tests/test_depth_ar.py` | `800940f7696b9273c4db7a1cba6cb51fc6ffd27550adc6c63702d97a5374b4ed` |
| `tools/evalkit/carry_channel.py` | `4dcd91facd60a512dcc983836fb29a7c91aad988bada1c95ada666824d226753` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` |

（以上为 `results.json::code` 原样抄录；`MODEL.md`/`analysis.md`/`stability.md` 自身是报告文本，不计入代码指纹。）

## 重跑臂（`U1rep`，判据 ⑥ 的重跑散度）

同配置、同 seed、同代码把 **plain** 臂再训一次（`--out-root train` 下 run 名换成 `U1rep`），
用于回答「训练侧散度」与「plain 臂是否可复现」。读数见 `divergence.json` 与
`stability.md` §3：答案行 CE 末值 **2.2524**（正式 U1 为 4.184）、评测准确率 **3.3447%**
（正式 U1 为 0.0732%），同配置两次相差 **+3.271pp**（CI 不重叠）。
