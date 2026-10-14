# 派单契约模板 · 轴 5 Phase B 的实验独立复核（reviewer，fresh 上下文）

> 由 axis5-lead 落盘；派发时把 `<占位>` 填好交付给 `reviewer`（builtin；**fresh 上下文，不得为本项 writer**）。
> 纪律：`SCHEMA.md` §14 / §14b（复核必须排在 writer **停止之后**）+ `docs/research-schema.md` §11.3（累计 ≤2 轮）
> + §11.4（按被改对象定门）。

## 你的角色与边界
- 你是 `<研究项>` 的独立复核者，上下文与 writer 不同（fresh），**没有**为本实验写过任何代码或文档。
- **只读 + 只写你的复核报告** `<报告路径>`；**禁止**改实验产物、`project.yaml`、任何其它文件；**禁止** `git add/commit`。
- 允许运行**纯 CPU** 命令（`python3`、`sha256sum`、`grep`、`git diff`/`git log`）；**禁止 GPU**（不许取 `/tmp/pathlm_gpu.lock`）。
- 你的会话若缺某类工具（curl/exe c）⇒ 把该项列为「无法执行」，**不要**猜结论。

## 复核对象（逐条自解：若你无法从文件本身推出复核任务，那是文件的问题，如实写）
- 被复核修订：`git rev-parse HEAD` = `<sha>`（= 你开始时的 HEAD；若随后前移，用 `git diff --name-only <sha>..HEAD -- <你的范围>` 判定被复核内容是否被改动）
- 实验目录：`<exp dir>`（`README.md` / `config.json` / `run.py` / `<语义层>.py` / `results.json` / `table.md` /
  `analysis.md` / `MODEL.md` / `run_log.txt` / `selftest_*.py` / `mutation_report.json`）
- 研究项契约：`axes/5-quant/<item>/project.yaml`（`method` / `validation` / `objective_criteria` / `falsification` / `milestone`）
- 权威纪律：`SCHEMA.md` §3/§6/§7/§8/§10/§14/§14b/§17.12/§20/§22/§23；`docs/research-schema.md` §12.1（判据带）/§11.2

## 只有「机器判不了」的才交给你（已跑过的机检结果当作输入，用 `results.json::instrument` 与 `run_log.txt` 的原文核对，不要重做算术）
请逐条判定，并**主动证伪**（missing scope / safety / regressions / stale evidence）：

1. **预登记与事后调整**：`README.md` 与 `config.json::criterion` 是否**运行前**写死（mtime 早于 `results.json`；
   `config.json::criterion` 与 `README.md` 的判据线是否一致）？判据是否在跑后被改写以迎合结果？
2. **单因素与实现等值**：是否只有一个因素被改？非量化臂是否**逐位**复原（用 `results.json::instrument` 与
   `masked_bpc`/历史锚的对账）；有没有「开关开着、部分臂没接」的死干预？语义层是否真的作用到目标张量（活性列）？
3. **字节账与实现一一对应**：横轴的每一个因子能否由实现反推（标尺个数/位置数/每值字节/kept_T）？有没有把
   标尺当免费、把两个口径（仅值 vs 部署）混用、或跨口径借用？fp8/量化实现是否真按声明的格式（含舍入方式）？
4. **判据带的适用性**：band 声明是否成立（0.014 只在「配对 Δ、≥5 窗口」下可用；否则是否如实退用本格在力带并
  声明两带不混用）？「越带/不越带/不可分」的判定与 `results.json` 的数字是否一致？有没有单点外推？
5. **否证路径**：`falsification` 每条是否被真的走查（有没有只报正面）？失败/不利结果是否如实报？
6. **成本口径**：`cost_model` 与判据里有没有墙钟/吞吐/日期（**必须为 0 条**）？等效理论量是否算对？
7. **结论强度**：`analysis.md` 的人话摘要/表/对账/局限与 `results.json` 是否逐条一致（数字、区间、指针不悬空）？
   结论是否停在「这些点的描述」、有没有外推或把「参数混杂」当机制发现？
8. **范围纪律**：改动是否只落在允许面（`axes/5-quant/<item>/experiments/**`）？有没有碰共享文件（`tools/**`、
   `pathlm/**`、`train_*.py`、`tests/**`、`docs/**`、其它轴、`project.yaml`）？`git status --short` 与
   `git diff --stat` 是否与该声明一致？
9. **§20 记账**：`results.json::gpu_lock`（`holder_pid`/`wait_seconds`/`free_mib_before`/`free_mib_after`/
   `release_ok`/`held_whole_run`）与 `peak_mem_mib_this_run` 是否齐备、是否与 §20.1（按臂释放 / 短任务例外）一致？
   是否存在「整程持锁而整程 >10 min」或「等锁超时/泄漏未登记」？

## 判词格式（写进 `<报告路径>`；报告第一段必须是「人话结论」）
- **VERDICT**：`PASS` / `OK with notes` / `FAIL`
- **blocker 条数 / normal 条数 / info 条数**，每条给：**位置**（file:line 或字段名）、**违反的判据**、
  **复现/证据**（命令 + 真实输出）、**建议修法**。
- 明确写出**你无法执行**的检查项（工具缺失 / 视野之外）与**你认为这份实验的结论强度上限**。
- 结尾给「对 lead 的处置建议」（哪些必须修、哪些可以如实登记为局限）。

预算：≤20 分钟；只读你要复核的文件一次，**不要遍历仓库**。
