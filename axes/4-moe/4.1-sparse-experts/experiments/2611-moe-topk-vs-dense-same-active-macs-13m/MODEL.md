# MODEL.md · 2611-moe-topk-vs-dense-same-active-macs-13m

权重**不入库**（`.gitignore` 覆盖 `*.pt`）。本目录的产物与生成方式：

| 产物 | 生成命令 | 说明 |
|---|---|---|
| `ledger.json` | `python3 run.py --stage ledger` | 臂记账（参数量 / 每 token MACs / token-per-param）；无权重 |
| `preflight.json` | `python3 run.py --stage preflight` | E=1 逐位恒等 + 记账自洽 + 路由活性/判别力断言；无权重 |
| `runs/smoke-*/results.json` | `bash train_arms.sh` 前的 20 步冒烟（`--steps 20 --out-root smoke`） | 步时与显存实测；权重不保留 |
| `runs/<臂>/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}` | `bash train_arms.sh <臂> …` | 训练产物；`model.pt` 是本机派生物，不入库 |
| `results.json` | `python3 run.py --stage eval` | 全臂配对 bpc + 判词聚合；无权重 |

- 权重可用同一 `config.json` 的 `arms[].seed` 与同一 `run.py` 重训得到，不是字节保证（GPU 非确定性见
  `harness/methodology.md` §2 的 0.0019 nondeterminism）。
- 数据：`data/enwik8_full.npz`（不入库，见 `README.md` §7）。
- 共享文件指纹：本目录**不含** `pathlm/**` / `train_depth_ar.py` / `tools/**` 的改动；MoE 模块与
  `Block.mlp` 的替换都发生在 `run.py` 内（运行期属性替换）。
