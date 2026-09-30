# MODEL · 2614 KV 消融保留曲线（13M dense，纯推理）

**本实验不产生任何新权重**（纯推理：评测期删除 KV，不改模型权重、不训练）。
下表列出**复用**的被测量权重与运行时强制断言。

## 被测量主干（复用，只读）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 体积 | 54,193,675 B（54.2 MB） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0（2610 记录） |
| 记录 bpc | 1.4836（clean，24×16 窗口）——本实验用同一窗口序复算得 1.483569 |

**sha256 断言**（`run.py::load_trunk`，复用他人 checkpoint 的强制项）：

```python
got = sha256_file(ckpt)                       # 54,193,675 B
assert got == "86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2"
```

不匹配时 `run.py` 直接 `SystemExit`，不会静默使用别的权重（`interpreter` 无从「跑错模型」）。

## 生成命令（2610 侧，本实验未重跑）

```bash
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 三条曲线 + 分解诊断 + 安全半径 + 复现性 + 噪声底 + 判据块 + 成本 |
| `retention_log.jsonl` | 137 行逐配置日志（bpc / 窗口 seed / 耗时） |
| `run_log.txt` | 全电池 stdout（`python3 run.py > run_log.txt 2>&1`） |
| `mutation_report.json` | 4 处「删错位置」变异验证结果 + 还原后 sha256 一致性 |
| `analysis.md` / `stability.md` | 判读 / 稳定性专报 |
