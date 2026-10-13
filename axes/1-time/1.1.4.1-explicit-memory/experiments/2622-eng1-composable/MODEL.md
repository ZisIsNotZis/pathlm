# MODEL · 2622 ENG1 引擎可组合性

权重（`*.pt`）**不入库**（`.gitignore` 屏蔽）。本实验唯一的权重是 20 步冒烟产物，
只作「三特性共存可训练」的存在性证据，**不用于任何性能/质量结论**。

## 冒烟检查点（不入库）

| 臂 | 文件 | 架构 | 参数 | 体积 | SHA-256 |
|---|---|---|---|---|---|
| `master`（GQA×carry×window） | `smoke/master/model.pt` | `ComposableBlock` × 12（GQA n_kv_heads=2 + 写入通道 delta + 训练窗口 16） | 12,616,808 | 50,518,595 B | `17861aba0d1623dddc63cff3b83c1d45c999243b2a83fa21d68d700aa0563120` |

- 生成命令（先取 `/tmp/pathlm_gpu.lock`，SCHEMA §20）：

```bash
python3 train_depth_ar.py master \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense \
  --steps 20 --batch-size 8 --seed 0 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --train-window 16 --carry-write delta --n-kv-heads 2 \
  --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2622-eng1-composable/smoke
```

- 该命令在 **ENG1 前会直接报错**（`carry_rule` 与 `n_kv_heads<n_heads` 硬互斥）；现在
  能建出 `ComposableBlock` 并训练（`results.json::kv_head_liveness.block_python_class`
  = `ComposableBlock`，`alive=True`）。
- 实测：20 步 / 40,960 token / 墙钟 0.04 min / 峰值显存 **735.6 MB**（≪5GB net）。
- 复现命令：`python3 run.py --smoke`（取锁后跑同一条命令并摄取产物）。

## 代码指纹（`results.json::code`，与本文档同一修订）

| 文件 | SHA-256 |
|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` |
| `train_depth_ar.py` | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e`（**未改动**） |
| `tests/test_eng1_composable.py` | `997542cdc9d028383dae4b9ea716c34903793d7a415e1e84dff9c9a56886a906` |
| `run.py` | `b0c7622323b1cce9ec624548b50ae49218fe6c6092045fd15b89d7370a3be960` |
| `mutate.py` | `88f83baaeefd0959590be46b8b2e7e98f13c2c67820dd29c4d5401d5bb198f6b` |
| `reconcile_numbers.py` | `201bdb6389f208f48bf0d6575767108347ea299390c486e7501758e451ca4168` |
| `README.md`（本实验目录内） | `217bdd1eaf9ebb7cce0e424a9c23bd027609f3e771056d3f7504cf8b7c3e0eb2` |
| `config.json` | `0f5a89558e188d7f1f21d3dca5e21fd7d7ad28ed291e44c9d5a44da3ac776807` |
| `controls/README.md` | `10af27d5f9ede412e32160fcb97c58a02bbb5021993e565aa57600d42d719a63` |

## 共享文件与并发（SCHEMA §22 / §22.8）

- 本线是**本轮唯一**写 `pathlm/depth_ar.py` 的线（持有代码锁）。
- **锚点声明 + 回读校验**：改动只在局部追加 / 局部改护栏，改完逐条 `grep` 断言邻线标记
  仍在：`carry_state_dim` / `n_kv_heads` / `attn_sparse_k` / `topology` / `recon_window`
  / `readout_perdepth`（命中数见 `analysis.md`）。
- **禁止回滚他人改动**：未做任何 `git checkout`/`stash`；变异脚本只对本线目标文件
  「改→跑→逐字节还原」，还原后 sha256 与原文一致（`mutation_report.json`）。
- `train_depth_ar.py` 未改动（本线只读 + grep 断言）。

### 能力面分析（§22.8，派发前已做）

`grep -n "ValueError\|互斥" pathlm/depth_ar.py` 列出的组合校验里，本线要用的
`--carry-write delta --n-kv-heads 2 --train-window 16` 原被两道硬互斥挡死
（carry⊕GQA、GQA⊕train_window）；本线**任务本身就是解除这两道互斥并使三者可组合**，
因此不是「擅自绕开」，而是契约明确授权的引擎改动（本线是本轮唯一写该文件的线）。
仍保持互斥的：`attn_sparse_k` / `attn_linear` 与上述任一（本线未开放）。
