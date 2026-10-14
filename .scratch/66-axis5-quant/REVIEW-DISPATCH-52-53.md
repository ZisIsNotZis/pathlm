# 派单契约 · 5.2 + 5.3 WP1 独立复核（reviewer，fresh 上下文，一个复核者覆盖两包）

> 由 axis5-lead 落盘（ticket 66 Phase B）；派发时把本文件路径交给 `reviewer`（builtin，fresh 上下文，
> **不得为本项 writer**）。纪律：`SCHEMA.md` §14/§14b（复核排在 writer 停止之后）、§20（你**不得取 GPU 锁**）、
> `docs/research-schema.md` §11.3（累计 ≤2 轮）/§11.4/§11.2。

## 你的角色与边界
- 你是 **`5.2 权重量化` 与 `5.3 激活量化` 两个 WP 的独立复核者**；上下文与两位 writer 不同（fresh），
  你没写过这两个包的任何代码或文档。**一个复核者覆盖两包**是允许的，但**必须对每一包各自给出独立判词**。
- **只读 + 只写你的报告**（两处：`.scratch/66-axis5-quant/REVIEW-5.2.md` 与 `.scratch/66-axis5-quant/REVIEW-5.3.md`，
  或一份 `.scratch/66-axis5-quant/REVIEW-52-53.md` 内含两节）；**禁止**改实验产物 / `project.yaml` / 任何其它文件；
  **禁止** `git add`/`commit`。
- 允许**纯 CPU** 命令（`python3`、`sha256sum`、`grep`、`git diff`/`git log`、`python3 tools/check_tree.py`、
  `python3 tools/summary.py --check`）；**禁止 GPU**、**禁止** `/tmp/pathlm_gpu.lock`。
- 会话缺某类工具（curl/exec）⇒ 把该项列为「无法执行」，**不要猜结论**。

## 复核对象
- 修订基线：`git rev-parse HEAD`（记下它；若随后前移，用 `git diff --name-only <sha>..HEAD -- axes/5-quant/` 判定被复核内容是否被动过）。
- **包 A（5.2）**：`axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/`
  （`README.md` / `config.json` / `run.py` / `wq_arms.py` / `results.json` / `table.md` / `analysis.md` /
  `MODEL.md` / `run_log.txt` / `selftest_wq.py` / `mutation_report.json` / `JOURNAL.md`）。
  契约：`axes/5-quant/5.2-weight-quant/project.yaml`。
- **包 B（5.3）**：`axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/`（同构产物）。
  契约：`axes/5-quant/5.3-activation-quant/project.yaml`。
- **口径背景（必须纳入判断，不是替你开脱）**：本批因单卡被多轴共享、`/tmp/pathlm_gpu.lock` 长期被其它轴占用，
  两个包**改用 CPU 口径**（全臂同设备；不取 GPU 锁）。这意味着：**不得**要求「`fp16` 臂逐位等于 GPU 历史锚
  1.483569」；应要求「同设备内部逐位一致（`fp16` 臂 vs 未量化路径 vs `masked_bpc` 独立路径）」+
  「`fp16` 的 CPU bpc 与 GPU 历史锚并列且口径差异显式声明」+「跨设备不作为判据」。若产物**没有**这样声明，那就是缺陷。

## 只有「机器判不了」的才交给你（机检结果当输入用，别重做算术）
逐条判定并**主动证伪**（missing scope / safety / regressions / stale evidence）：

1. **预登记 vs 事后调整**：`README.md` 与 `config.json::criterion` 是否**运行前**写死（mtime 早于 `results.json`；
   两者判据线一致）？跑完后判据是否被改写以迎合结果？`JOURNAL.md` 是否如实记了「转 CPU 口径」这件事？
2. **单因素与实现等值**：每包是否只有一个因素被改（5.2 = 权重量化方案；5.3 = 激活量化，且必须有**只改激活**的主因子臂，
   否则结论被权重因子混杂）？`fp16` 臂是否逐位复原（同设备）？有没有「开关开着、部分臂没接」的死干预
   （用活性列与 `logits_changed_when_patched` 判）？
3. **字节账与实现一一对应**：横轴每个因子能否由实现反推（标尺个数 / 值字节 / 元素数 / `kept_T`）？
   有没有把标尺当免费、把「仅值口径」与「部署口径」混用、跨口径借用？5.3 的**权重字节与激活字节是否分列**
   （不得混进同一条横轴）？
4. **判据带的适用性**：0.014 的可用条件（配对 Δ、≥5 窗口）是否真的满足（逐 seed 配对 Δ 与 `max|Δ|` 与 `results.json` 对得上）？
   「越带/不越带/不可分」的判定与数字一致吗？有没有单点外推？
5. **否证路径**：`project.yaml::falsification` 每条是否被真的走查（尤其 5.3 的「离群通道是否迫使部分层回退」）？
   不利结果是否如实报（含 `at_floor` 的正确读法）？
6. **成本口径**：`cost_model` 与判据里**墙钟/吞吐/日期必须 0 条**（CPU 口径下尤其要查有没有把墙钟混进判据）。等效理论量算对了吗？
7. **结论强度**：`analysis.md` 的人话摘要/表/三道对账/局限与 `results.json` 是否逐条一致（数字、区间、指针不悬空）？
   结论是否停在「这些点的描述」、有没有把「参数量/设备差异」当机制发现、有没有外推？
8. **范围纪律**：改动是否只落在各自 `experiments/**`？`git status --short` / `git diff --stat` 与声明一致吗？
   有没有碰 `tools/**`、`pathlm/**`、`train_*.py`、`tests/**`、`docs/**`、其它轴、`project.yaml`？
9. **变异/单测是否真能失败**：`mutation_report.json` 里每条变异是否真的被抓红？有没有「恒真断言」充当门禁
   （参照 2642 的 N2 教训：把「读数透传」写成独立检验就是循环论证）？

## 判词格式（每包一节；每节第一段必须是「人话结论」）
- **VERDICT**：`PASS` / `OK with notes` / `FAIL`；**blocker / normal / info 条数**。
- 每条给：**位置**（file:line 或字段名）、**违反的判据**、**复现/证据**（命令 + 真实输出）、**建议修法**。
- 明确写出**你无法执行**的检查项与**每包结论的强度上限**；结尾给「对 lead 的处置建议」（哪些必须修、哪些可登记为局限）。

预算：**≤20 分钟**；只读被复核的文件各一次，**不要遍历仓库**。
