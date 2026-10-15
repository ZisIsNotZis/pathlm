# RESUME — 2673-robustness-13m

WP 6.2.4：温度>0 的无损投机采样（P1）+ 53.9M 对抗臂（P2）。进度逐 job 落盘。

- 已完成 job：['p1_13m_s0', 'p1_13m_s1', 'p2_53m_s0']
- 未完成 job：[]
  - `p1_13m_s0`：τ=0.936643 tokens/fwd=1.936643 α₁=0.605201
  - `p1_13m_s1`：τ=0.959809 tokens/fwd=1.959809 α₁=0.623604
  - `p2_53m_s0`/adv_random_k4：α₁=0.019422 Wilson=[0.0145, 0.0259] identical=True
  - `p2_53m_s0`/adv_bottomk4：α₁=0.0 Wilson=[0.0, 0.0016] identical=True

重跑：`python3 run.py --out . --resume --jobs p1_13m_s0 p1_13m_s1 p2_53m_s0`（detached + ≥60s 让出窗口）
