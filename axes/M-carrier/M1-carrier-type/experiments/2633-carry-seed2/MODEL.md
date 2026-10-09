# MODEL · 2633-carry-seed2 权重清单

> 三臂（none/plain/delta）× seed {0,1,2}：seed0/1 六臂**复用 2631**（SCHEMA §9：`HEAD` 实现逐文件同哈希 + 工作树 vs `HEAD` 在本实验三档配置上逐位等价，见 `equality_report.json` / `engine_equivalence.json`），
> seed2 三臂本实验新训（12k 步 / bs32 / T256 = 98.3M token/臂）。权重不入库；`run.py` 加载时逐次断言 sha256。

| 臂 | 来源 | 参数量 | 体积 (B) | 训练墙钟 (min) | 训练峰值显存 (MB) | sha256 |
|---|---|---:|---:|---:|---:|---|
| `none_s0` | 2631（复用） | 13536768 | 54193675 | 9.07 | 2015.3 | `5e04df917aa0336f84ad4849b917fe8bb2eda2e374e509f408ec2fdcef81444b` |
| `plain_s0` | 2631（复用） | 13801064 | 55255619 | 13.08 | 2170.2 | `91f8197027b2b1a370d7b5fb897ab28b6f6b568aa8fd99165ab02357282f4c33` |
| `delta_s0` | 2631（复用） | 13801064 | 55255619 | 15.32 | 2283.4 | `d6fff38ca2be1df4239730f89dd816dd242f48d3cb340c3f836e0d63fbe37c9b` |
| `none_s1` | 2631（复用） | 13536768 | 54193675 | 8.57 | 2015.3 | `8fd9dff445966025292375eb58633cb4556db6be8605f4d4d23c1568eb46d096` |
| `plain_s1` | 2631（复用） | 13801064 | 55255619 | 12.83 | 2170.2 | `09ff2fda8792e6029027ffb00b5d8271794a39f69b075e4cb7bbc7bf97652583` |
| `delta_s1` | 2631（复用） | 13801064 | 55255619 | 15.79 | 2283.4 | `7487540111e92cc8d065aa3ee590d7c967da3e4d68406355906b4c109049424e` |
| `none_s2` | 2633（本实验新训） | 13536768 | 54193675 | 13.89 | 2015.3 | `f379ac79ccf104e22ae4d4fbf4f253623b50d465c4950a6592ad050033b29883` |
| `plain_s2` | 2633（本实验新训） | 13801064 | 55255619 | 16.46 | 2170.2 | `1d010ccacdb74fddaaeb4387710198825383d9e4724abfbb033aefda32a53891` |
| `delta_s2` | 2633（本实验新训） | 13801064 | 55255619 | 19.41 | 2283.4 | `0ad748e0f7cfef8b69561a70737b31b188090d5c357dd86431e249e81ed7f05c` |

> `train/<arm>/model.pt` 不入库（`.gitignore` 的 `*.pt`）；复用臂的权重路径在 `2631-carry-reseed/train/<arm>/`（`config.json::arms[*].path`）。

## 结构

- 主干：stock dense decoder-only（tied 读出 + 无 final LayerNorm + 顺序残差），d256 × 12 层 × 8 头 × mlp6 × T256。
- 通道臂：每层读出增益 γ_ℓ（零初始化）+ 单共享记忆槽（8192 floats，与 T 无关），+264,296 参数。
- 训练/评测可见性：`window_anchor_keep_mask(256, 16, 0)`（可见对 3976 < 因果对 32896）。

## 复现路径

```bash
bash axes/M-carrier/M1-carrier-type/experiments/2633-carry-seed2/train_seed2.sh     # seed=2 三臂训练（按臂释放 GPU 锁，臂间 ≥60s）
bash axes/M-carrier/M1-carrier-type/experiments/2633-carry-seed2/eval.sh           # 九臂一次评完（纯推理，整程 <10 min 持锁）
```

## 说明

- `--seed 2` 是 2631 的 `SEEDS=(0,1)` 之外的新档：`seed3_stats.arm_name` 在种子 0/1 上**委派** 2631 的命名，在 2 上按同一规则构造（单测钉住重叠档逐字相同）。
- 复用前提由两级证据保住：① `HEAD` 提交版的五个引擎/仪器文件 sha256 与 2631 运行时记录逐位相同；② `engine_equivalence.check()` 把 `HEAD` 版 `pathlm/` 整包解到临时目录，在本实验三档配置（none/plain/delta）上同权重比 logits，**全部逐位相同**。
