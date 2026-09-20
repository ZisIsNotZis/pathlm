# 22 — 自由生成段的质量信号

- **Status:** done（判据 A/B/C 全部成立——A 经互补路线、B/C 直接成立；
  附带回修：ticket-15“13M prob0 反向”未复现，系 n=1 假象，方向与 100M 一致）
- **Type:** mechanism（在线监测）
- **Related:** 15-slider-rung3（prob0 代理的边界：13M 生成段反向、100M 恢复正向）、
  08/09（路径多样性与 TTS）、21（exit 双 profile）
- **执行方式：fork=false 子代理（fresh context），串行单卡**

## 问题

prob0 是"当前输入 token 是否正确"的检测器：TF 窗口上校准极好（ECE ≤0.01），
但**自由生成段失效**——13M 上损坏把生成推进高置信重复吸引子，mean prob0 反向
（clean 0.954 vs 流式损坏 0.976）。部署需要一个**无 ground truth、生成时可得**
的输出质量信号。

## 假设（可证伪）

**路径分歧度（path disagreement）是生成段质量信号**：同一生成窗口用 2–3 条
独立采样路径（不同 layer path / transport，重评分而非生成路径）teacher-forced
重前向，节点 1 分布的成对 JS 散度在错误位置显著更高，且该关系在自由生成文本
上存活（ unlike prob0）。

## 实验协议（纯推理，无需训练；INT2_r3 权重）

ckpt：`.scratch/15-slider-rung3/evidence/INT2_r3/model.pt`（config
`.tmp/INT2.json`，若缺失用 configs/ 重建或从 results.json 恢复）。

1. **TF 流校准**：eval 流上单 forward 取 2–3 条采样路径（`sample_path` 各自
   独立采样；rounds=paths 长度，`p.n_retries=0`），收集各路径 node-1 分布。
   按 per-position 计算 mean pairwise JS（或 1−mean argmax agreement），
   分桶 vs 实际 hit/miss → 分歧度-错误率曲线（对照：prob0-hist 错误率曲线）。
   判据 A：分歧度分桶的错误率单调性 ≥ prob0 分桶（或互补：联合分桶更优）。
2. **生成段迁移**：free-running 贪心生成 400 token（3 种 prompt：enwik8 真实
   前缀 / 随机字节 / 流式损坏 15%），每段生成完后 **重评分**（teacher-forced
   2 路径，不生成只读出）→ 窗口级分歧度。判据 B：分歧度在三种 prompt 间的
   方向与幅度 vs 已知劣化排序（真实 < 损坏 < 随机），并与 mean prob0 的
   同窗表现对照（prob0 已知反向/混淆）。
3. **控制实验**：对同一生成窗口人工注入 0/7.5/15% wrong 损坏再重评分——
   分歧度应随注入率单调上升（信号对真实劣化敏感，而非对 prompt 来源的伪相关）。

## 判据与产出

- 判据 A/B/C 任一失败 → 如实证伪入账本（同样有价值：确认生成段监测需要
  全新信号类型，而非重评分）。
- **结果对账**：A 成立（互补路线；主路线 corrupt 桶单调性 0.78<0.89 失败，
  如实记录）；B 成立（real 0.0114 < corrupt 0.0250 < random 0.0390）；C 成立
  （合并 0.0251/0.0314/0.0375，15/18 窗单调）。盲区：吸引子塌缩窗 js→0。
- 证据：`.scratch/22-gen-quality/evidence/`（探针 jsonl/json + SUMMARY.md
  含全部数字表）。
- 代码：`probe_gen_quality.py`（根目录，风格对齐 probe_exit.py）；
  纯函数进 `pathlm/slider.py` 或独立小模块 + 可失败单测。
- docs：findings.md（≤200 行）与 report 补节；WORKSPACE 状态行。

## 全局约束

- 仓库 `/home/z/vibe/pathlm`；策略 `/home/z/vibe/AGENTS.md`。
- 语言：docs/ticket/evidence/commit 中文；代码标识符英文。
- GPU 单卡独占（当前空闲）；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- 测试：`python3 -m pytest tests/ -q`（当前 74 绿）必须保持全绿。
- 只 add 自己的文件；commit 中文；**不要 push**（主会话审查后推送）。
- 长任务用后台 + 轮询或分步执行；所有 GPU 步骤设超时。
- 预算：纯推理研究，总 GPU 时间 ≤1.5h；超出则缩小规模（n_batches/prompt 数）。
