# 2646 · 逐层宽度 schedule（3.1.2）· process 记录（非 SSOT）

契约 = `axes/3-width/3.1.2-layer-width/experiments/2646-layer-width-13m/README.md`
（预登记、不可改）；事实读数 = 同目录 `results.json` / `results_<arm>_s<seed>.json` /
`schedule_search.json`。本文件只记过程与就地工程决定，**不得**与上述事实文档冲突；
冲突以事实文档为准。

## 授权边界（本 WP）

- 只写 `axes/3-width/3.1.2-layer-width/experiments/2646-layer-width-13m/**` 与
  `.scratch/64-axis3-width/**`；`pathlm/**`、`train_*.py`、`docs/**`、`tools/**`、
  其它轴与其它格（`2641`/`2642`/`2643`/`2644`/`2645`）**零改动**。
- 干预实现在实验目录内：`PrefixSubspaceBlock`（前缀子空间块，复用共享 `Block.forward`）+
  `prefix_block_from_stock`（前缀子矩阵切片）+ `apply_width_schedule`（逐层换块）。

## 决定记录（就地工程决定，不改契约）

- **实现路线**：`PrefixSubspaceBlock(Block)` **继承共享 `Block`**，把前缀 `x[..., :w_l]`
  交给 `super().forward()`（块内部 `d = w_l`、头数 `w_l/32`），只在本层外把未触及的通道
  `torch.cat` 拼回。理由：前缀路径与 stock 路径是**同一段数值代码** ⇒ `w_l = d` 时逐位
  相同是**构造上**的，而不是靠另写一份前向去凑（2642 的 `prefix_forward_full` 是复制一份
  前向；本项不需要）。门 T1 实测 state_dict 与 logits/loss 在 fp32 与 bf16 autocast ×
  train/eval 下全部逐位相同。
- **`HOLD_ALL`**：冒烟（60 步 × 4 臂）值得一次取锁跑完，不必臂间等 61 s；`train_arms.sh`
  支持 `HOLD_ALL=1`（缺省 0 = 正式模式，逐臂取/放 + 臂间 61 s）。
- **冒烟用 seed 99**：`evaluate()` 的落盘名是 `results_<arm>_s<seed>.json`（与步数无关）——
  若冒烟跑 seed 0，会把 60 步的读数写成 `results_EQ_s0.json`，既污染判据、又会让 supervisor
  误判「该臂已完成」。故冒烟固定 seed 99，run 目录 `runs/<arm>-s99-0k` 与结果文件都分开。
- **`HG` 的定义**：非均匀的严格左右对称沙漏在「32 的整数倍、Σw² = 294,912」的格点上
  **无解**（唯一的对称单峰解就是等宽本身，见 `schedule_search.json` 的
  `symmetric_unimodal_candidates`）。故取 **PYR 的宽度多重集按升序交替发到两端**的单峰重排：
  多重集不变 ⇒ PYR/HG/APYR 的 Σw²、Σw、参数、MACs **完全相等**，唯一差别是「哪个宽度放在
  哪一层」——这是比另找一个对称 schedule 更干净的同预算对照。HG 仍满足契约的「中段宽、两端窄」。
- **`PYR` 的选择规则**：全部 13 个非增可行解中，与契约示例
  `[256,224,224,192,192,160,160,128]`（Σw² = 307,200，**不可行**）L1 距离最小者 =
  `[256,224,224,192,192,160,160,64]`（与示例仅第 8 层不同）。规则写进 `search_schedule.py`
  与 `schedule_search.json`（含全部候选与排序），可复算。
- **共享探针替换**：`train_depth_ar._kv_head_liveness` 断言「每层 qkv 输出宽 == 3·256」并按
  单一全局 d 推 K/V 头数 —— 逐层宽度下必然失败。训练期把该全局名换成
  `_kv_head_liveness_prefix_safe`（真实前向抓每层 qkv 实际输出宽，与 `3·w_l` 逐位对照，
  不符抛错）。探针只读形状、不入判据、`train_*.py` 零改动。
- **读出恒在 256 宽**：契约 P1 的读出项 = `d·V` with d = 256（残差流恒 256 宽）。⇒ 前缀外
  通道是**只经 embedding 的直连旁路**（恒不被任何层处理，但进读出）。这是「残差流不变、
  只切逐层算子宽度」这一干预形态的固有性质，写进 `analysis.md` 的归因段（也是本项与 2642
  的原生 d=192 不是同一模型的原因，不跨干预比较结论）。
- **参数逐位对齐的口径**：对齐的是 `Σ_l w_l²`（契约要求）。总参数里另有 `13·Σw_l` 的偏置项
  ⇒ EQ（Σw = 1536）与三臂（Σw = 1472）的总参数差 832（0.020%）、MACs 差 32,768（0.75%）：
  **逐位报告**，不回避。

## 过程流水

- 2026-10-14T23:00+08:00 起（时间均为本机时钟 ISO8601）。读 `DISPATCH-B.md` §W-D、
  `2641/run.py`、`2642/run.py`+`README.md`+单测+`train_arms.sh`、`pathlm/model.py::Block`、
  `pathlm/depth_ar.py`（`DepthARConfig`/`DepthARModel.__init__`/`forward`/`readout`/
  `_kv_head_liveness`）、`train_depth_ar.py`（`main`/`train`/探针）各一次。
- 先落 `search_schedule.py` → `schedule_search.json`（13 个非增候选、对称单峰只有等宽一个解、
  PYR/HG/APYR 三臂 Σw² = 294,912 逐位相等）。
- 再落 `README.md`（预登记契约）→ `run.py` → `test_2646_layer_width.py`：
  `pytest -q` = **24 passed**（含 CUDA 上的 bf16 autocast × train/eval 逐位门）。
  单测期间修掉 2 个测试自身的错：① 列切片变异取 `proj.weight[:192, 192:384]` 越界
  （`256` 宽的 proj 只有 64 列可切）改成 `[32:224]`；② `shared_argv` 对比须忽略
  `run_name`（argv[1]），只比其余 CLI。
- `train_arms.sh`（照抄 2642 的 §20 取放协议 + 新增 `HOLD_ALL`）、`smoke.sh`（seed 99）。
- 非 GPU 自检：factory 路径下 PYR 模型实测参数 4,202,688 == 解析值。
- GPU：冒烟（`STEPS=60 HOLD_ALL=1 ARMS="EQ:99 PYR:99 HG:99 APYR:99"`）与 12k 四臂均走
  detached supervisor-retry，逐臂取/放 `/tmp/pathlm_gpu.lock`；**不杀他人进程、不回收他人锁**。
- 具体读数、锁等待与未完成项见 `analysis.md` §「未做项」与 `supervise.log` / `run_log.txt`。
