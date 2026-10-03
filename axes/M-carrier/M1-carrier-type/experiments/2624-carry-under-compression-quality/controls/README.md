# 2624 · 对照臂与变异验证说明

本目录实验（Z3）的对照结构、破坏性验证与「对照可回溯」清单。原始结果见同目录
`results.json`；判读见 `analysis.md`；契约见 `README.md`。

## 1. 三条评测行（唯一变量 = 头投影 + 评测期 KV 精度）

| 行 | 权重来源 | 训练 flags | 评测 KV 精度 | 角色 |
|---|---|---|---|---|
| `carry_mha_fp16` | `carry_mha` | carry(delta) + window(16,0) | fp16 | **基准**（W3 的载体形态） |
| `carry_gqa_fp16` | `carry_gqa` | + `n_kv_heads=2` | fp16 | 分离「头共享」 |
| `carry_gqa_int8` | `carry_gqa`（**同权重**） | （同 `carry_gqa`） | int8 | 分离「评测期 int8」 |

- 三条行都在**同一任务、同一位置集、同一批**（`assoc_recall_batch`，K=8，seed 同）
  下打分 ⇒ 差值只含变量本身。
- `carry_gqa_int8` **不训练、不加参数**，只换 KV 精度；因此 `int8_vs_fp16_pp` 无参数量
  混杂。`gqa_vs_mha_pp` 含 **−8.75% 参数**混杂（§15 声明见 `analysis.md`）。
- **carry + window16 是常量**：让「被问 pair 的 key 落在窗外」⇒ 不搬就答不出
  （SCHEMA §19；2618/W3/Y1 同一件仪器）。

## 2. 毁源对照臂（强制，判据 C3）

每个评测行都并跑一条**毁源对照**：把源 KV 段 `[0, 2K)` 换成同分布新随机符号
（`corrupt_span`），查询行/答案/距离/掩码逐位不变。

- 规则：毁源后若准确率 **Wilson 95% CI 下界 > 4×chance** ⇒ 该行读数不是来自被搬运的
  源信息，**该行作废**（与 2618/2621 同款操作化）。
- 结果见 `results.json::destroy_source_validity`。

## 3. 实现等值锚（2 个块类各一条）

`quant_carry.py::quant_carry_logits(mode="fp32")` 必须与释放版前向
（`tools/evalkit/retention.masked_logits` 经真实块类）**逐位相等**：

- `carry_mha`（`CarryChannelBlock`）：与 2621 `carry_quant.py` 的 fp16/int8 接线**逐位一致**
  （单测 `test_fp32_bitwise_matches_released_carrychannel`；`run.py` 自检落盘）；
- `carry_gqa`（`ComposableBlock`）：2621 的模块**显式拒绝**该块，故本线新接线
  （单测 `test_fp32_bitwise_matches_released_composable`；`run.py` 自检落盘）。

## 4. 变异验证（每处「故意破坏 ⇒ 抓红 ⇒ 逐字节还原」）

`mutate.py` 对准本实验自己的语义层（`quant_carry.py` / `compression_arms.py`），
不碰任何共享引擎文件（SCHEMA §22）。五处破坏点：

| # | 文件 | 破坏点 | 期望抓红的单测 |
|---|---|---|---|
| M1 | `quant_carry.py` | K/V 量化钉死成 fp32（假压缩） | `test_int8_changes_values_and_logits` |
| M2 | `quant_carry.py` | 窗口掩码被吞掉（前向不再逐位等值） | `test_fp32_bitwise_matches_released_composable` |
| M3 | `compression_arms.py` | int8 增量符号反掉 | `test_compress_increment_signs` |
| M4 | `compression_arms.py` | GQA×carry 仍报 `GQABlock` | `test_expected_block_classes` |
| M5 | `compression_arms.py` | 压缩行偷换成 fp16（假 int8 臂） | `test_eval_rows_three_int8_shares_gqa_weights` |

结果见 `mutation_report.json`（`all_caught_red` / `all_restored_bitwise` 都须为 true）。
