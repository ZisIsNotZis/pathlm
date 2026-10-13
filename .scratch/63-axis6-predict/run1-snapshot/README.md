# run #1 快照（**本地副本，不入库**）

2026-10-13T17:23+08:00 axis-lead 在前任 WP writer 的 run #2 被运行时上限杀死后，把 run #1 的
三份落盘产物拷到这里留证。经 sha256 核对，它们与主家（实验目录）中的同名文件**逐字节相同**
⇒ 证据的主家是实验目录，此处副本**不重复入库**（避免同一事实两份大文件）。

| 文件 | sha256 | 主家 |
|---|---|---|
| `results.json` | `1302b6785630f32815651a02340ecd72b556c7e5ec44657311994ee74c13dbaa` | `axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/results.json` |
| `cycle_log.jsonl` | `d2073370c7985c7b1bc0981e6d8a73231462674bbd2ea49ae0f54ef48a592290` | 同上目录 `cycle_log.jsonl` |
| `run_log.txt` | `ba48b06a0bcee2420c119c61a6c2687dee544225d1e30c29b60fb1a600a2984c` | 同上目录 `run_log.txt` |

核对时间：2026-10-13T17:32+08:00（同一哈希三方比对：快照 / 实验目录；全部 identical）。
