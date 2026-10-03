# MODEL.md · 2622 去相关 v2（权重指纹 / 复现命令 / 共享面）

> 权重（`*.pt`）**不入库**。本文件记录 sha256/体积以支持复现核对；`results_{13m,50m}.json::checkpoints`
> 与 `config.json::arms` 由 `run.py` / `fill_config_hashes.py` 落盘并互相断言。

## 0. 一句话

`l1` 与 `repr`/`repr01` 是**同一骨干**（参数量逐位相同）；`repr*` 只是每步多采一条并行路径并对两条
路径的 trunk 隐状态加一个跨路相关罚则（`w_repr_decorr`，0 参数增量）。

## 1. 权重指纹

| 臂 | 相对路径 | 体积 (B) | sha256 |
|---|---|---|---|
| 13m/stock | `train13m/stock/model.pt` | 52349379 | `f591c6bf63627c0e1ea8b7b32aacf84601a5f507272ecf844deb4cec3716788f` |
| 13m/l1 | `train13m/l1/model.pt` | 52349379 | `456d0300280be674667177a810eb2491d97759ae428954d6d93736664092c64c` |
| 13m/repr (λ=1.0) | `train13m/repr/model.pt` | 52349379 | `7d25c035a013c7ebe87141546586289ee9509feb4501de7bf80177be5cbd18ef` |
| 13m/repr01 (λ=0.1) | `train13m/repr01/model.pt` | 52349379 | `e5aea26ae97d6b87d1efe765fd83c91391524e4268b13d00e36b75cc23a7b851` |
| 50m/stock（复用 2620） | `../2620-decorrelation-at-scale/train/stock/model.pt` | 207415747 | `3649f076f0c4a4f30809550196553ad13dea50e485f3ba496017d65e77d7c6dd` |
| 50m/l1（复用 2620） | `../2620-decorrelation-at-scale/train/l1/model.pt` | 207415747 | `e9304612b1c5614c58b4140f0a4030aa4053f7f06d2b76f93d557ad5076b9a59` |
| 50m/repr01 (λ=0.1) | `train50m/repr01/model.pt` | 207415747 | `7b892d8fe3b0e4ac01ab7b57502bd87d622159062ba4e2c7b20e2debd31eaa0b` |

- 参数量：13M = **13,076,483**；50M = **51,843,075**（各臂逐位相同，去相关压力是**纯损失项**）。
- **复用的 50M `stock`/`l1` 哈希与 2620 `config.json` 逐位一致**（该臂 `gate_skips=0` ⇒ 新 gate
  代码下训练等价），`run.py` 每次加载都重算 sha 并断言。
- 中间检查点：`.tmp/v2-ckpts/{13m,50m}/<arm>/model_step{003000,006000}.pt`（不入库）。

## 2. 复现命令

```bash
# 0) 测试（CPU）
python3 -m pytest tests/test_decorrelation_v2.py tests/test_decorrelation_at_scale.py -q

# 1) 冒烟（400 步 × 5 臂；整程 <10min ⇒ 允许整程持锁）
nohup bash train_all.sh smoke > smoke_run.log 2>&1 &

# 2) 两规模训练（按臂取/放锁，臂间 ≥60s）
nohup bash train_all.sh train 13m > train13m.log 2>&1 &   # stock/l1/repr/repr01
nohup bash train_all.sh train 50m > train50m.log 2>&1 &   # repr01（stock/l1 复用 2620）

# 3) 评测 + 对账（纯推理 >2min ⇒ 取锁）
nohup bash train_all.sh eval 13m > eval13m.log 2>&1 &
nohup bash train_all.sh eval 50m > eval50m.log 2>&1 &

# 4) 变异验证（运行期猴子补丁，不改文件）
python3 mutate.py
```

## 3. 代码指纹（`results_{13m,50m}.json::code`）

两次评测记录到的 `run.py` sha256 **相同** = `2bb596cd876a9e93308a9fe0738963f2ac9ba05e90f0221a67f5203ac03e2762`。
`train_m1.py` = `a253fcaf61fb39e0042bf1086174c32f47b7c78580c3a48fc1c42a9bb3ae5459`，
`pathlm/model.py` = `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f`，
`pathlm/config.py` = `7c6c0fbf16fb115d5580ad2ade42917b32ad9ba2ec810d74e097ad4f934cdb4d`，
`tests/test_decorrelation_v2.py` = `6f03841df6fcd037f07e05175deee519a54ae106018f998d0a2277a5de43d90a`，
本契约 `README.md` = `98b47770fffc703ca3ab684fecae09ee24bd9997bceff97eddeeb142080986d7`。
> **说明（口径）**：`results_50m.json::code` 里的 `reconcile_numbers.py` 哈希是 50M 评测当时的值；
> 该脚本在 13M 评测前又改过一次（`PRIMARY_REPR` 泛化），故两文件里该哈希不同。
> `reconcile_numbers.py` **不参与**评测，重跑 `python3 reconcile_numbers.py --scale {13m,50m}`
> 才是其当前值的口径（均 pass）。

## 4. 共享文件与并发（`SCHEMA §22`）

| 文件 | 本线改动 | 邻线标记回读 |
|---|---|---|
| `pathlm/config.py` | 新增 `w_repr_decorr` 字段（+注释） | `grep -c "carry_state_dim\|recon_window\|recon_head"` = 0（不在本文件） |
| `pathlm/model.py` | 新增 `repr_decorr_penalty` + 一处 `if λ>0` 接线 | 同上 = 0；`pathlm/depth_ar.py`/`train_depth_ar.py` 的 `w_repr_decorr` = 0 |
| `train_m1.py` | `sample_rounds` 并行路径、`gate_skip`、`validate_pcap`、活性量 | 邻线标记 `carry_state_dim`/`recon_*` = 0 |
| `tests/test_decorrelation_at_scale.py` | 仅把新增字段列为该守卫单测的唯一例外 | 其余断言保持 |

**未**使用 `git checkout/stash/reset`；变异脚本 `mutate.py` 用运行期猴子补丁，**未改任何文件**。
