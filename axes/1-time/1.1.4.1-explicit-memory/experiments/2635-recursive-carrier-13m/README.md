# 2635 · 递推/线性状态载体（注意力换成线性递推状态）的质量腿 · 13M · dense

> 契约来源：主线一工作项的**载体类型空白**——M1 的载体枚举里「递归状态（SSM/RNN）」
> 至今**只有成本数据、没有任何质量读数**：既有的唯一线性注意力实装在训练期**概率性停摆**
> （见 `../../../A-mixing/A6-mixing-operator/experiments/2616-linear-attention-13m/stall_diagnosis.json`），
> 因此判了「当前实装不可行」。
> 入口：`smoke_steptime.sh`（先测步时）→ `train_state.sh`（带停摆门的两臂，按臂释放锁）→
> `run.py`（质量腿 + 长程电池 + 活性）；停摆门 `stall_gate.py`；原始结果 `results.json`；
> 判读 `analysis.md`；权重清单 `MODEL.md`。

## 1. 命题

**把 13M 主干的 softmax 注意力换成「固定大小的递推状态读/写」（归一化线性注意力
`out_i = Σ_{j≤i} φ(q_i)·φ(k_j)·v_j / Σ φ(q_i)·φ(k_j)`，φ = elu+1），在真实文本上
能不能**跑完**，质量代价是多少？**

一句话可证伪的形式：*两臂（stock 全注意力 / 线性递推状态载体）同配方、同数据、同 seed、
**同 step 预算**；唯一变量 = 注意力算子（参数量逐位相同）。若线性臂在预算内**跑完**
（日志每 200 步推进、无停摆记录），则给出 matched-step clean bpc 差与长程任务的
附带读数；若**再次停摆**，则记录触发步 / 全线程 Python 栈 / CPU-GPU 状态 / 诊断二分结果，
并把这格的结论记为「实现层停摆（未取得质量读数）」。*

## 2. 单因素声明

- **基底**：stock dense decoder-only trunk，13.54M（d256 × 12 层 × 8 头 × mlp6 × T256），
  与 2618/2631 **同一规格**（便于与本格其它腿并列）。
- **唯一变量**：`attn_linear` ∈ {False（stock softmax 注意力，SDPA 融合核）, True
  （归一化线性注意力，累积状态实装）}。**参数量逐位相同**（引擎自带单测
  `tests/test_depth_ar.py` 的相关条目：块与 `Block` 逐位同构、不新增参数）。
- 其余一致：数据（enwik8）、步数、bs32、lr 1e-3、seed 0、`--exit-mode uniform`、
  `--dense --topology sequential`（引擎要求）。
- **能力面限制（SCHEMA §22.8，显式记账）**：引擎**硬互斥** `attn_linear ⊕ train_window`
  （`pathlm/depth_ar.py` 的构造校验）⇒ **「窗口 + 线性载体」这个最贴近本格问题的组合
  现在不可测**。本实验只测「全可见（无窗口）下把注意力换成递推状态」这一忠实子问题；
  「窗口 16 + 线性载体」的搬运问题**未测**，不得外推（见 §9 待测项与成本）。

## 3. 干预定义

不新增任何代码：用既有 `--attn-linear` 开关（`pathlm/depth_ar.py::LinearAttentionBlock`
+ `linear_attn_chunked`）。状态 = 每层每头 `dh × dh` 的累积矩阵（T=256、d256/8 头时
8,448 个元素，**与 T 无关**）。**本实验不改任何 `pathlm/**`、`train_depth_ar.py`、
`tools/**` 文件**（能力面限制见 §2，不擅自绕开）。

## 4. 判据线（预登记；写下后不得改）

| # | 判据 | 判定线 |
|---|---|---|
| ① | **可跑性（工程门）** | 线性臂在预算步数内日志每 200 步推进、无停摆；停摆 = 日志 mtime 冻结 ≥180 s 且进程 CPU 仍在烧 ⇒ **判「实现层再次停摆」**，不进 ② |
| ② | **质量腿** | 若跑完：报 `Δbpc = bpc_stock − bpc_linear`（正 = 线性更好），口径 = matched-step 快照，阈值用本格自测噪声界（同配置两跑极差，若只跑一次则报单点并标注「单点、无方差界」） |
| ③ | **活性** | `attn_linear_liveness` 的 `alive` 为真且 `den_min` 不贴 clamp 下限；`gnorm` 无数量级尖峰 |
| ④ | **侧腿（不作判据）** | 长程电池四任务零样本准确率（与 2617 的 stock 基线并列）；距离曲线只报数、不外推 |
| ⑤ | **禁外推** | 结论只适用于该步数快照 / 13M / enwik8 / 全可见设定 |

### 4a. 停摆门（本实验新增的工程仪器，不改判据）

`stall_gate.py`：以 `PATHLM_DUMP_STACKS=150` 启动训练子进程（faulthandler 每 150 s 打全
线程 Python 栈，**与 2616 的停摆诊断同一手段**），并每 20 s 检查训练日志的 `step`/mtime，
**两级信号**：

- **无日志推进 ≥ `--stall-secs`（缺省 180 s）⇒ 只记事件**（`no_log_progress_events`，
  含触发步、墙钟、CPU 时间与 10 s 增量、该 pid 显存、最后 3 行日志；**不终止**）；
- **无日志推进 ≥ `--kill-secs`（缺省 900 s）且进程仍活着 ⇒ 终止**该臂（`status =
  killed_after_no_log_progress`，返回码 3）；进程在此期间自己正常退出则记 `finished`。

为什么要两级（**实测教训**）：`train_depth_ar.train` 只在**未被梯度门跳过的步**写
`train_log.jsonl`（`grad_skips` 那一路是 `continue`），所以「数值发散 ⇒ 绝大多数步被跳过 ⇒
日志停写」会**伪装**成「进程停摆」。单测
`test_stall_gate_does_not_kill_log_frozen_but_finishing_run` 钉住「日志冻住但进程能自己
结束 ⇒ 必须记 finished、不得误杀」。诊断落 `stall_report_<arm>.json`。

## 5. 两臂规格

| 臂 | `attn_linear` | 步数 | seed | 参数量 |
|---|---|---|---|---|
| `stock` | False | 3,000（+ 6,000 若预算允许） | 0 | 13,536,768 |
| `linear` | True | 3,000（+ 6,000 若预算允许） | 0 | 13,536,768 |

**步数为什么是 3,000 而不是 12,000**：2616 的停摆二分显示 `chunk ∈ {64,256}`、
`eps ∈ {1e-6,1e-12}` 的独立进程在 **1,200–3,000 步**上全部跑通（5/5），而 ≥6,000 步的
12k 运行有 ~1/2 概率在 step 400–1000 区间停摆 ⇒ 3,000 步是**能拿到质量读数**的最大
已知安全预算；两臂**同步数** ⇒ 只读「臂间增量」，绝对值是快照（SCHEMA §20.3 口径）。

## 6. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 30 min`（含步时冒烟 + 两臂 × 3,000 步 + 评测 + 停摆门开销）。

**步时实测（本实验 300 步冒烟，稳态窗 step 200→299；原始日志
`smoke/steps300/<arm>/train_log.jsonl`，数字回填在 `config.json::steptime_measured`）**：

| 臂 | 稳态 ms/step | 3,000 步估算 | 实测训练墙钟（`train/<arm>/results.json`） |
|---|---:|---:|---:|
| `stock` | 42.4 | 2.12 min | 2.56 min |
| `linear` | 127.3 | 6.36 min | 4.43 min |

两套数口径不同（外推 vs 含启动开销的实测墙钟），且 `linear` 的实测**低于**外推：发散步上
62% 的步被梯度门 `continue` 掉（同时跳过 telemetry/日志 CPU 开销）+ 两次测量时共享卡占用不同
⇒ 两套都不作跨臂速度结论。完整口径见 `analysis.md` §7。

**预算不够时砍什么**：砍第二档步数（6,000）与长程电池批数；**不砍判据线、不砍臂**。

## 7. 数据与脚本清单

- 数据：`data/enwik8_full.npz`（与 `train_depth_ar.py` 内部同一路径）；评测只用尾段。
- 训练：`python3 -m stall_gate` / `bash train_state.sh`：
  `python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256
  --dense --topology sequential --steps 3000 --batch-size 32 --lr 1e-3 --seed 0
  --exit-mode uniform --eval-batches 4 [--attn-linear] --out-root train`。
- 评测：`python3 run.py`；配置 `config.json`；结果 `results.json`。
- 落盘：`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`、
  `stall_report_<arm>.json`（停摆门原始诊断）、`train_<arm>.log`（含 faulthandler 栈）、
  `results.json`、`analysis.md`、`MODEL.md`、`smoke/`。
- 权重 `*.pt` 不入库；`MODEL.md` 记 sha256 / 体积。

## 8. 既有做法（prior art，SCHEMA §18）

1. **谁做过**：线性注意力（Katharopoulos 2020）、S4/Mamba（状态空间，`记忆·未核`）、
   RWKV、GLA、DeltaNet 一系——把「看所有位置」换成「维护固定大小状态」，动机是
   长度上线性/常数的时间与缓存。本仓已有一条实装（2616，A6 格）与一整套实测成本数据
   （T=256 整模型 2.58× 慢、训练峰值显存 1.41–1.47×、注意力子层在 T=1024 首次反超）。
2. **为什么这么设计**：把长上下文的算力/缓存从平方降到线性，代价是状态容量受限、
   需要「选择性」（输入依赖的门/写规则）。
3. **它的实测结果（记忆·未核）**：同尺寸下纯线性注意力在记忆受限的召回任务上明显弱于
   注意力；混合少量注意力层可补上。**本仓实测**：2616 的线性臂**概率性停摆**（卡在
   `loss.backward()` 内、归一化分母塌陷到 clamp 下限后），因此只有成本、没有质量数字。
4. **我们的增量**：不是重做 2616（成本数据已足够），而是**用停摆门把「能拿到质量读数」
   变成一次受控尝试**：同等步数、同规格、与 stock 臂并列出一条 matched-step 质量读数
   （或一份带触发步/栈/环境状态的停摆记录）。若停摆，本实验的增量是**「停摆 → 质量不可测」
   这个事实本身**以及「要让它可跑需要什么」的清单。

## 8b. 与 A6/2637 的分工（避免两处事实）

本实验**只**回答 M1 视角的三件事：① 「窗口 + 线性载体」这一组合因引擎硬互斥**不可测**；
② 该快照下**拿不到**线性载体的质量读数（发散步）；③ 停摆/发散的**诊断与可复用教训**
（「日志停写 ≠ 进程停摆」）。**「线性注意力这一族到底可不可行」的家在 A 轴**
（`archive/wiki-legacy/axes/A-mixing/A6-mixing-operator/`，其续作为 `2637-linear-attn-quality`）——本实验不重复
它的质量/成本判据，也不改写它的结论。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **停摆**（判据①）⇒ 走 §4a 的停摆门取证；若栈再次落在反向或归一化分母塌陷：
   记录触发步/栈/GPU 状态，**不调参救**（改 φ/加特征的归一化是另一干预、需新契约），
   并把「要让它可跑需要什么」写成待测项（候选：把归一化分母改成 `Σφ(k)` 的非塌陷形式 /
   用无归一化的递推（RWKV/衰减型）/ 更小步长大维度 warmup），标成本。
2. **跑完但质量差** ⇒ 报 matched-step Δbpc；按 SCHEMA §19 只读作「该尺度/该快照不划算」，
   不得写成「线性载体不可能」。
3. **跑完且质量持平** ⇒ 只作**单点**陈述（无方差界 ⇒ 不称「等价」，只称「不可分辨」）。
4. **活性为 0** ⇒ 先查接线（`attn_linear_liveness`），再判机制。

## 10. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；**组合默认留白**。「窗口 + 线性载体」
因引擎互斥**未测**（§2 已记账），不得外推。`deps` 只记录耦合方向。

## 11. 产出与复核（SCHEMA §14/§14b）

见 §7；完成后另派 fresh-context 只读复核者，**复核排在全部产物落盘之后**。
