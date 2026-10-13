# MODEL — 2631-spec-three-arms · Spec 三臂（chain / direct / combined）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、
哈希、体积与生成命令；三臂**共享同一 checkpoint**（零新增参数，臂只是同一模型上
第 N+2 草稿位分布来源的不同接法），故本实验的「臂」不是三份权重。

## 1. 权重清单

| checkpoint | 规模 | seed | 参数量 | 体积 (bytes) | sha256 |
|---|---|---|---|---|---|
| `train/13m_s0/model.pt` | 13.4M（d256/L12） | 0 | 13,734,149 | 54,984,075 | `1df7840bc4813f89709805222cedf34b0384f8bae0a04e23f9ffd2818b2d4e4d` |
| `train/13m_s1/model.pt` | 13.4M（d256/L12） | 1 | 13,734,149 | 54,984,075 | `047893bda15104cc25794dc598d96bca96bbef7449bd5d45518e78e205a31b59` |
| `train/53m_s0/model.pt` | 53.9M（d512/L12） | 0 | 54,469,125 | 217,923,979 | `0627862468867f1dcdf939adeaa7b65bc78bf92d7d9efec5d106bebfb37ccf35` |

- 三个 checkpoint 的 `config.path` **逐字段一致**（`n_mtp=3` / `chain_mode="token"` /
  `transport="none"`，其余开关全 0）；`unspecified_path_keys = []`（无默认值泄漏）。
- 13.4M 档两次**独立训练**（seed {0,1}）；53.9M 档**只 1 个 seed（如实声明）**。
- 三臂读的是同一 checkpoint 的**既有头**（`T1`/`T2`/`T3` + `chain_cat`）⇒ `params_added = 0`，
  臂间无容量差。

## 2. 代码 revision（本实验绑定的实现版本）

| 文件 | sha256（判读绑定） |
|---|---|
| `pathlm/model.py` | `75bd84a013c71f2e6eef479517cc99b96d8cebff672383f2728f2752812655d3` |
| `pathlm/decode.py` | `84095bb03853e1c57894f5c6b116f7df8b187184fb16a8139ae2c147751ad81c` |
| `train_m1.py` | `a253fcaf61fb39e0042bf1086174c32f47b7c78580c3a48fc1c42a9bb3ae5459` |

- `pathlm/model.py` 的 F_ENG 修复（`_mtp_nodes` 对 T==1 单位置解码的右垫条件）已入 git
  （提交 `f957f19`；证据 `evidence/fix_mtp_nodes_T1/`：T≥2 逐位不变 `max|Δ|=0` + sha256 一致、
  可失败单测先红后绿、变异 7 条全抓红）。
- 本实验目录内代码（`spec_arms.py` / `run.py` / `reconcile_numbers.py` / `mutate.py` /
  `test_spec_arms.py`）随本提交入库，其 sha256 由复核者直接对工作树核对。
- `git rev-parse HEAD`（提交前）= `a25261b1b0bb4c26d671b99bf19da588e711761f`；
  本实验的改动在**后续提交**中落地，上表 sha256 是判读数字所绑定的确切实现版本。

## 3. 生成命令（逐条可复现）

```bash
cd /home/z/vibe/pathlm
EXP=axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms

# 步时冒烟（先测后定预算）：bash $EXP/smoke_steptime.sh
# 训练（按臂取/放 GPU 锁，臂间 65s）：
bash $EXP/train_arms.sh 13m_s0 13m_s1 53m_s0
#   等价单臂：python3 train_m1.py <scale>_s<seed> --config $EXP/config_<scale>.json \
#             --steps 6000 --batch-size 32 --seed <seed> --out-root $EXP/train
#   （batch-size/seed 由 config 与 --seed 决定；config 已 pin 全部 path 键）
# 评测（三臂 × native 对照；同进程交替计时）：bash $EXP/eval.sh --repeats 6
```

- 训练配方：6000 步 × bs32 × T512 = **98.3M token**（两档同 token 刻度，§20.3）。
- 实测训练墙钟：13.4M **4.9 min**/臂、53.9M **10.0 min**；`peak_gpu_mem_mb` 见
  `train/*/results.json`。
- 评测种子无关（贪心解码，prompt 取 enwik8 val 固定 offset）。

## 4. 证据文件绑定

| 文件 | 说明 |
|---|---|
| `results.json` | 六项指标原始读数 + `timing_raw` + `acceptance_trace` |
| `cycle_log.jsonl` | 逐 (ckpt, arm, prompt) 的每轮接受数（活性量，SCHEMA §10） |
| `reconcile_report.json` | 三道对账（期望值全部 `derived` 重算） |
| `mutation_report.json` | 变异 7 条全抓红 + 逐字节还原 |
| `smoke/steptime.md` | 步时实测与预算（§17.8/§20.3） |
