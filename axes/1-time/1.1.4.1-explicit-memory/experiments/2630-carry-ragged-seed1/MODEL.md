# 2630-carry-ragged-seed1 · 权重清单（权重本身不入库；`.gitignore` 覆盖 `*.pt`）

## 一、训练臂权重（本实验自己训练，seed=1 两臂）

| 臂 | 路径 | 体积（字节） | sha256 | 参数量 | 块类 |
|---|---|---|---|---|---|
| `projfill_s1` | `train/projfill_s1/model.pt` | 57,358,059 | `d2539fb43a550bc0e9af0d061f22361b62ab9cca10031aef8066a7c7c3f51ffe` | 14,326,272 | `Block` |
| `projfill_carry_s1` | `train/projfill_carry_s1/model.pt` | 58,420,003 | `e52d2569b801345b69afcc25a0e27e02901ff328cd77189a7099e280d9e26055` | 14,590,568 | `CarryChannelBlock` |

- **哈希口径**：`run.py::load_arm` 逐臂断言 `config.json::checkpoint.sha256`；不一致直接
  `SystemExit`（防「用错权重出数字」）。
- **生成命令**（两臂逐字段一致，唯一变量 = `--carry-write` 与 seed=1）：

  ```bash
  # 基准臂（随机退出 + proj-fill，无 carry）
  python3 train_depth_ar.py projfill_s1 \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --topology sequential \
    --steps 6000 --batch-size 32 --lr 1e-3 --seed 1 --exit-mode uniform \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --proj-fill --final-weight 2.0 \
    --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2630-carry-ragged-seed1/train
  # 目标臂：同上但 run 名 projfill_carry_s1 + --carry-write delta
  ```

  驱动脚本：`train_arms.sh`（按臂取/放 GPU 锁、臂间 65s，§20.1）。
- **权重不入库**：体积与哈希在此登记；复现时按上面命令重训（同 seed 逐次哈希应一致——若
  不一致，说明实现或库版本变了，必须查）。

## 二、复用的 seed=0 臂（2629 的权重 / 结果）

- **权重**（2629 `train/`）：`projfill` = `64cc88ec…`、`projfill_carry` = `9860eb95…`
  （见 `2629-carry-ragged-projfill/MODEL.md`）。
- **本线合并 n=2 时读的是 2629 的 `results.json`**（不重训 seed=0）：
  `sha256 = 7f4681bcdde14f7d67ea9d5f5d9300f6acc2ca70e529813da5cabd9178a4c3a5`
  （`results.json::cross_seed.seed0_source_sha256`，reconcile 逐条核对）。

## 三、代码指纹（本文件登记；跑离线校验/复核时逐条核对）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py`（共享引擎，本线**未改**） | `45f946793835f2afadf3a10f727b02ee879682ef6024b23fa179d3351c28c585` |
| `train_depth_ar.py`（共享训练入口，本线**未改**） | `2d1863bbc4403e026d12b597d98a1da2b85f3606110073f2117a7dd1db2a1f35` |
| 2629 `ragged_projfill_arms.py`（被**复用**的单因素语义层） | `7603549158b34e2199050d4efc94ca0c40e958e82c3e084c3b282cf05e01b308` |
| 2629 `run.py`（被**复用**的评测仪器） | `49ae02eacd6719dfe9f13e45d09549f22d2267947f9fc43ff995efbfda931a48` |
| `ragged_seed1_arms.py`（本线语义层：seed 命名 + 跨 seed 统计 + 判据②③） | `be7faf443e250ab47cc77157b920d37198969d3197dbc9920d71897ba1b20012` |
| `run.py`（评测入口 + 跨 seed 合并） | `c33e4743c5bd0400c0a078bd5c07584949bf8779960c7a05bcf0d23016e898a2` |
| `train_arms.sh`（训练驱动） | `fe7581f7b979a39fbc3e2409f38008b2e165ac7a602f02976c31867f6cc09461` |
| `smoke_steptime.sh`（先测步时） | `0c3d7cfd5d04582c09dcc0bb6a63ab2210baf485653f7e9d6aed1a1868638a09` |
| `eval.sh`（评测驱动，含锁协议） | `a9aef9123c59db692f6b0e6362fa534a51d7b258ca1d5d68ea3f1f1741694e98` |
| `reconcile_numbers.py`（对账） | `516184c8fa2f94f7bfe9fbbcff682b8a10f38dd317af0fa667dd28049e578a86` |
| `mutate.py`（变异验证） | `758008f974ccc8c0713017679e9153d4a9f83ee2c47cd96c4772e3f63b35b4bd` |
| `tests/test_carry_ragged_seed1_13m.py` | `23d52186b0da5fef875c5c3f640d2a7214a13303fc5aa6f870278da70b6ce069` |

## 四、共享文件并发写记账（SCHEMA §22）

- 本线**不改**任何共享引擎 / 训练入口文件，也**不改** 2629 的语义层 —— 上表前四个文件的
  sha256 是「本线跑评测时**读到**的版本」，本线未写它们（2629 的 `ragged_projfill_arms.py`
  与 `run.py` 仅被 `import` 复用）。新增逻辑全部放在本实验目录的新模块 + 新单测文件里。
- 若同波其它线改过 `pathlm/depth_ar.py` / `train_depth_ar.py`，哈希会随时间变化——本线只
  声明「我自己没写它们」。
