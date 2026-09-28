# 26 — 项目收口：回顾整理 + README + Artifact 归档

- **Status:** in_progress（最终任务，本票完成后项目收口）
- **授权：本票 commit 后 push（用户已明确授权）**

## 目标

项目正式收口。结论已定：「等智商免费加速」证伪——所有影响正常推理路径的
设计（回环、重试、跳层、全深度监督）都带来结构性性能税；transformer 主干
保持纯粹，速度只能走成熟旁路（MTP/Eagle3）。把整个轨迹整理成可读资产。

## 任务清单

### 1. 新建 `docs/retrospective.md`（核心交付：完整叙事回顾）

从 `.scratch/*/SUMMARY.md + issues/*.md + docs/mental_model.md §0 +
docs/findings.md` 重建完整轨迹，结构：

1. **起点**：原始 idea（解码串行低效 → 回环/权重复用换 token 吞吐），
   design.md 回环时代冻结架构
2. **引擎迭代期**（ticket 01-14）：m0→m1→m2→m3，corruption/retry 站、
   TTS probe、retry matrix、allocator、gap sweep——回环族被系统性探索
3. **第一次转向（规模判读）**（15-19）：Rung 3 Slider、100M 快照税不随
   规模增长、能力溢价 ≥1.54bpc、渐近税随规模+训练收缩 35%、spec 解码
   加速在 compute-bound 下消失（20）
4. **第二次转向（目标重聚焦）**：等参数/等智商/更快解码 → 退役
   retry/corruption 全家 → 深度自适应 AR（mental_model §0）
5. **机制验证期**（23 系）：四性质成立、workspace 实测、fill 胜出、
   proj-fill 最优、soft 提交 OOD
6. **规模判决**（24/24b）：税 +0.44 干净归属（形态 96%）
7. **归因收口**（25）：H-短视否证（切分恶化，涌现 workspace > 结构切分）、
   H-时长否证（48k 双锚零收缩）→ 税结构性存在 → 正式证伪
8. **最终结论与一般性教训**：推理路径侵入必付税（四个独立设计一致验证）；
   等智商免费加速不成立；可行域 = 成熟旁路；负结果 + 过程资产的价值

要求：每阶段给出关键数字与 commit 引用；写明 mindset 变化的**原因**
（哪个实验/哪次判读触发的转向）；诚实记录失败与混杂的排除过程。
篇幅 300-500 行，中文。

### 2. 重写 `README.md`（高层，给人看）

- 项目是什么（一句话）、探索过什么、发现了什么（含失败结论及其论证）、
  最终交付了什么资产（机制库、测量方法学、25 个 ticket 的证据链）、
  文档导航（retrospective → 全轨迹；findings → 结论账本；report → 实验细节；
  mental_model §0 → 设计哲学演化）
- 不超过 120 行

### 3. 压缩 `docs/findings.md`（335 → ≤200 行）

- 保留高价值结论行，把细节移入 `docs/report_details.md`（追加相应章节）
- 压缩原则：结论句保留、数字保留、过程叙述移走

### 4. `WORKSPACE.md` 收口状态

状态改为 archived/closed，写明最终结论一句话 + 指向 retrospective.md

### 5. Artifact 整理

- 检查 `.gitignore` 覆盖 `*.pt`（76 个权重文件共 20GB 不入库，仅证据
  json/log/md 入库——确认当前已如此）
- 新建 `docs/artifacts.md`：25 个 ticket 的 evidence 目录索引表
  （run 名 → 内容 → 关键数字 → 对应 SUMMARY 章节）
- 核对各 SUMMARY.md 完整性（发现缺失即从 run 目录的 results.json 补关键数字）

### 6. 提交与推送

- 中文 commit（可分多个逻辑提交：docs 收口 / README / findings 压缩）
- **push 到 origin/main（已授权）**
- `.pi-glla/` 的未暂存删除是遗留状态，不要触碰、不要 add

## 验证

- retrospective 的关键数字抽查 5 处对照 findings/SUMMARY 原文
- findings.md ≤200 行（wc -l）
- 所有文档内部链接有效；无中英混杂段落（代码标识符除外）
- push 后 `git log origin/main -1` 确认
