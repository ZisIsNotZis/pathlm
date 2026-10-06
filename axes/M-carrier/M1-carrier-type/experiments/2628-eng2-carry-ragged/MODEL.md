# 2628-eng2-carry-ragged · 权重清单与代码指纹（MODEL.md）

## 1. 权重（**不入库**，`.gitignore:11: *.pt` 已覆盖）

| 文件 | 体积 | sha256 | 生成命令 |
|---|---|---|---|
| `smoke/master/model.pt` | 58,420,003 B（≈56 MiB） | `81fb4518267daf480d4e561fa45b4f8813b29097a7cd4b46cfda22ea118d94c5` | 见下 |

生成命令（与 `run.py::SMOKE_CMD` 逐字相同；冒烟 20 步）：

```bash
python3 train_depth_ar.py master \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 \
  --steps 20 --batch-size 8 --seed 0 --exit-mode uniform \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --carry-write delta --carry-chunk 64 --proj-fill \
  --out-root axes/M-carrier/M1-carrier-type/experiments/2628-eng2-carry-ragged/smoke
```

- **权重只作 20 步冒烟的活性量证据**（`results.json::carry_liveness` / `train_log.jsonl`
  的 `attn_carry` 行），不构成任何质量结论；不入库。
- 参数量 14,590,568（非 dense 形态件 carry 写门 264,296 + proj-fill adapter 计入）。

## 2. 代码指纹（`results.json::code` 同源）

| 路径 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `45f946793835f2afadf3a10f727b02ee879682ef6024b23fa179d3351c28c585` |
| `train_depth_ar.py` | `2d1863bbc4403e026d12b597d98a1da2b85f3606110073f2117a7dd1db2a1f35` |
| `tests/test_depth_ar.py` | `d14e874f531ddfdc8b47157c90feb4f95ca544b5fdb89fd4835f98031b907192` |
| `axes/…/2628-eng2-carry-ragged/run.py` | `341154d788f6e164e3b6ca357ea7fedc918235432ca8476cb07526667f6eed96` |
| `axes/…/2628-eng2-carry-ragged/mutate.py` | `8534f7e596d1f2d4526cd9bae410f25658dfb58d0d7ca954df27fe05de605b2e` |
| `axes/…/2628-eng2-carry-ragged/reconcile_numbers.py` | `08fe4bc2e1966818281e7767cbc8a3c90dd653d153f8e5e478b091d19d1e9bcd` |

（完整 64 位值以 `results.json::code` 为准；本表为同源摘录。）

## 3. 共享文件与重叠归属（SCHEMA §22.6）

- **本波唯一写 `pathlm/depth_ar.py` / `train_depth_ar.py` / `tests/test_depth_ar.py`
  的线**（任务下发文本明确授权）。
- 变更前先声明锚点、只做局部追加；改完逐条 grep 断言邻线标记仍在：
  `carry_state_dim`(20) / `n_kv_heads`(66) / `attn_sparse_k`(25) / `topology`(36) /
  `recon_window`(4) / `readout_perdepth`(6) 全部命中。
- **重叠归属核对**：改动窗口内 `git status --short` 显示这三个文件**只有本线的改动**
  （其它条目均为别的线的未跟踪目录/文件），因此上述 sha256 不含他线未提交内容。
- **变异验证的隔离**：共享卡上另有一条线在各臂之间 `import` 同一份
  `pathlm/depth_ar.py`，为避免争写（§22），变异在**仓库副本**上执行；副本的
  `pathlm/depth_ar.py` 与仓库逐字节相同（sha256 均为 `45f94679…8c585`），报告中的
  `sha256_original` / `sha256_restored` 因此对仓库同样成立。

## 4. 复现命令

```bash
# CPU 组合自检（4 组合 × 活性量）+ 摄取冒烟/测试/变异产物
python3 run.py

# 20 步 GPU 冒烟（自行取 /tmp/pathlm_gpu.lock，峰值 ≪5GB）
python3 run.py --smoke

# 变异验证（5 处破坏 ⇒ 抓红 ⇒ 逐字节还原）
python3 mutate.py

# 三道对账
python3 reconcile_numbers.py

# 回归
python3 -m pytest tests/ -q --ignore=tests/test_jitter_l1.py   # 539 passed, 1 skipped
python3 -m pytest tests/test_depth_ar.py -q                    # 134 passed
```
