# 轴 1 时间轴 Phase A · 收口清理（REVIEW-4 终审后的非关键记账）

## 人话摘要

round-4 终审判的是「窄范围 BLOCK」，卡的全是**纯文字/指针**，不涉及任何数字或结论。Main 已裁定这些属「收口后可随手补的非关键记账」，所以这一轮就是把复核者点名的 3 处一次性清掉、再补两笔登记，然后提交。**没有动任何实现代码，也没有动任何测量数字**：`gap=0.125421`、`win17` 严格支配 `carry16_s0`（208,896 B < 212,992 B 且 1.624104 < 1.624979）等结论一个没碰。

具体清了 3 处：

1. `1.1.4.1` 的 `project.yaml` 里那段**和 940cb23 逐字相同、重复且过时**的「独立复核（REVIEW-1.md）……若有缺陷按 SCHEMA §14/§14b 返工」——删掉。同一事实在同一份事实文档里出现两次（违反 `AGENTS.md` §2.1「一条事实一处」），而且它只认 REVIEW-1、口气还像「复核未结」。上面那段已写全 REVIEW-1/2/3 判词的保留（并补回被删行带走的闭合引号）。
2. `WP1.review` 原来只列 `REVIEW-1.md` → 改为列全本 WP 三轮复核件 `REVIEW-1.md`/`REVIEW-2.md`/`REVIEW-3.md`。**WP2/WP3 尚未开始、零轮复核件，按规则「只有一轮就列一轮」保持 `fresh 复核` 原样不动。**
3. 实验 2638 的自包含单测 `test_2638_frontier.py` 的 module docstring 还写着「钉住三件事」，实际有 6 个 `test_*` → 改成**不写死数量**的「钉住以下事项」（免去下一轮再改）。

另两笔登记：

4. `REVIEW-4.md` 原先只存在于 runtime artifact，现**原样复制入仓**（`cmp` 逐字节一致，与 REVIEW-1/2/3 落盘模式一致）。
5. `TICKET.md` 追加 §8.8：记 Main 裁定原文、本轮处置、真实时间戳 `2026-10-13T16:17:46+08:00`；并**登记一条流程规则**——`conclusion_review` 只写当前（最后一轮）判词，历轮判词写 ticket，不得在同一字段累积多轮（本轮反复「账实不符」的根因）。

## 3 处落点

| # | 文件 | 原状 | 现状 |
|---|---|---|---|
| 1 | `axes/1-time/1.1.4.1-explicit-memory/project.yaml`（原 `:20`） | 多一行重复的「独立复核（REVIEW-1.md）：见该文件判词。复核前未改本 `project.yaml` 的判据或数字；若有缺陷按 SCHEMA §14/§14b 返工。」 | 已删；`conclusion_review` 现止于 `:19`（含 R1/R2/R3 判词，闭合引号补回）。删除后 `:20` 即 `experiments:`。 |
| 2 | 同文件 `WP1.review`（原 `:81`，现 `:80`） | `…复核件见 .scratch/62-axis1-time/REVIEW-1.md` | `…复核件见 .scratch/62-axis1-time/REVIEW-1.md、REVIEW-2.md、REVIEW-3.md` |
| 3 | `axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py`（`:6`） | `钉住三件事（复核 N8 / B1）：` | `钉住以下事项（复核 N8 / B1）：` |
| 4 | `.scratch/62-axis1-time/REVIEW-4.md` | 仅存在于 runtime artifact | 复制入仓，`cmp` 一致 |
| 5 | `.scratch/62-axis1-time/TICKET.md` | 止于 §8.7 | 追加 §8.8（裁定原文 + 处置 + 时间戳 + 流程规则） |

## 命令结果

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | **OK：46 个 project.yaml 全部符合 docs/research-schema.md** |
| `python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py -q` | **6 passed** |
| `grep -c '^def test_' …/test_2638_frontier.py` | **6**（与 docstring 不再冲突） |
| `grep -n "REVIEW-1.md\|REVIEW-2\|REVIEW-3" …/project.yaml` | `:19`（conclusion_review 判词）、`:80`（WP1.review 三件全列）、`:119`/`:122`（change_log）——一致 |
| `python3 -m pytest tests/ -q` | **834 passed in 312.96s**（共享底座未改） |
| `git diff --check` | 无输出（清掉 TICKET.md 行尾空行后） |
| `git status --porcelain` / `git diff --cached --name-only \| wc -l`（提交后） | 空 / **0** |
| `git show --stat HEAD` | 4 文件：`project.yaml`、`test_2638_frontier.py`、`TICKET.md`、`REVIEW-4.md`（**不含任何共享文件**） |
| `cmp <source>/REVIEW-4.md .scratch/62-axis1-time/REVIEW-4.md` | **IDENTICAL** |

提交：`01d25ff ticket 62 · 轴 1：按 REVIEW-4 收口清理（删重复 stale 复核句 + 补全 WP1.review + 单测 docstring 计数；不动实现/数字）`；`git add` 用 4 条显式路径，commit 用 `git commit -F - <<'EOF'`，提交前核对 staged 列表 = 4 条且与 message 声称逐条对应。

## 未决

- **无阻塞。** 5 项任务全部落地，自检全绿，工作树干净、无暂存。
- **范围边界（如实登记）**：本轮只改文字/指针；`conclusion_review` 里那段「已写全 R1/R2/R3」的多轮判词按 Main 裁定**保留未改**——它与新登记的「只写最后一轮」规则在**本项余量**上形式上不一致，但规则已明写「自下一次修订起生效、历轮归 ticket」。若后续有新一轮修订触发该字段，应据规则只留最后一轮，历轮判词留 §8。
- **未做**（不在本任务范围，未重开）：不重跑 GPU/训练/评测；未重审 REVIEW-1/2 的历史 PASS 项。科学数字本轮零改动。
