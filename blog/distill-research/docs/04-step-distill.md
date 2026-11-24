# 04 · 步数蒸馏：把 10 步 ODE 压成 1 步的所有做法

> 为什么这条线最值钱：π0 系 VLA 的动作头每次出 chunk 要积 10 次速度场，"action generation dominates model latency and caps the achievable control frequency"（SnapFlow Abstract，逐字）。压步数 = 直接抬控制频率。
> 本页只讲**动作生成器**（flow matching / diffusion）的步数压缩，不讲语言 token 的投机解码（见 08 章 H）。
> **实现代码见本章 §10（Progressive / Consistency / DMD / Shortcut / SnapFlow / VLA-SSD / FastOPD）。**

---

## 0. 一张图看懂五类方法

| 类型 | 核心思想 | 代表 | 需要教师吗 | 训练轮数 |
|---|---|---|---|---|
| **迭代减半** | 学生学"教师两步的端点"，反复减半 | Progressive Distillation (2202.00512) | 需要（上一轮学生） | log₂N 轮 |
| **自洽约束** | 任意 t 都能一步到 x₀，沿 ODE 轨迹自洽 | Consistency Models (2303.01469)、LCM (2310.04378) | 不需要（用 solver） | 1 轮 |
| **分布匹配** | 用 score 差把学生分布拉向教师/真值 | DMD (2311.18828)、DMD2 (2405.14867) | 需要（score/教师） | 1 轮 |
| **步长条件** | 学"跳 d 步"的平均速度，推理取最大 d | Shortcut (2410.12557)、MeanFlow (2505.13447) | 不需要（从零/继续训练） | 1 轮 |
| **端点自蒸馏** | 教师=自己的多步 Euler 端点 | SnapFlow (2604.05656)、RealtimeWAM/TACD (2610.06617) | 自己 | 1 轮 |

---

## 1. Progressive Distillation（2202.00512）：最稳，但最贵

**做法**：教师 2N 步 → 学生 N 步；学生学 "教师走两步的结果"：

```
x_{t-2Δ} = teacher(teacher(x_t, t), t-Δ)
L = || student(x_t, t) - sg(x_{t-2Δ}) ||²
```

然后学生变成新教师，重复 log₂N 次。10 步 → 1 步要 4 轮（10→5→3→2→1 近似）。

- **优点**：每轮目标明确、稳定；理论上是"把 ODE 离散化误差重新分配"。
- **缺点**：轮数多、累计成本高；每轮都要重新采样数据。
- **VLA 落点**：π0/π0.5 的 flow 头可以用，但工业界更常用一步到位的自蒸馏（SnapFlow），因为 progressive 的 4 轮训练在机器人数据上太贵。

---

## 2. Consistency Models / LCM（2303.01469 / 2310.04378）：一步，但目标弱

**做法**：学 consistency function `f(x_t, t) = x_0`，约束同一 ODE 轨迹上相邻两点的输出一致：

```
L = d( f(x_{t_{n+1}}, t_{n+1}), f(x_{t_n}, t_n) )
```

LCM 的改进：**skipping-step**（一次跳多步）+ **scheduled sampling**（训练时用自己生成的轨迹，缓解误差累积）。

- **优点**：不需要教师网络；1–4 步可用。
- **缺点**：一致性约束只保证"同一轨迹上自洽"，不保证"分布正确"；多峰动作容易被压平。
- **VLA 落点**：FastOPD 的 self-consistency 项就是这个思路的变体（见 §6）。

---

## 3. DMD / DMD2（2311.18828 / 2405.14867）：质量最高，最玄学

**做法**：把"蒸馏"写成分布匹配。学生生成 x，用教师（或真值）的 score 与学生 score 的差做梯度：

```
∇θ L ≈ E_{x,t}[ (s_real(x_t,t) - s_fake(x_t,t)) · ∂x/∂θ ]
```

- DMD 用"教师=真值扩散模型 + 学生=fake"两个 score 的差；
- DMD2 去掉回归损失、加 GAN loss 稳定训练，并引入"反向 KL 修正"。
- **优点**：1 步质量最接近多步；图像生成里是 SOTA 级。
- **缺点**：需要额外的 score 网络/教师、训练不稳、对超参敏感；**动作空间的维度低、模态多，DMD 的"平均化"问题比图像更严重**。
- **VLA 落点**：DyMD（2609.31349）用 DMD 变体把 14B 视频教师蒸成 4 步 1.3B 学生；DIDO（2609.15570）用 DMD + 交互表征引导。

---

## 4. Shortcut / MeanFlow（2410.12557 / 2505.13447）：一次训练，任意步数

**Shortcut**：给网络加一个"步长条件" d：

```
v_θ(x, t, d) ≈ (x_{t+d} - x_t) / d
```

训练时 d 从 {1,2,4,...} 采样，且满足自洽：`v(x,t,2d) = (v(x,t,d) + v(x_{t+d}, t+d, d))/2`。推理时 d=最大 → 一步。

**MeanFlow**：直接学平均速度 `u(x,t,r) = (x_r - x_t)/(r-t)`，并用恒等式

```
u(x,t,r) = v(x,t) - (r-t) · d/dt u(x,t,r)
```

做自监督。不需要两阶段蒸馏，也不需要教师。

- **优点**：单 checkpoint 支持多步数（1/2/4/8），按算力预算切换；训练一次完成。
- **缺点**：训练目标复杂、实现细节多（JVP 计算）；对动作这种低维输出，收益不如图像明显。
- **VLA 落点**：Anchored Path Matching（2611.22592）的"单 ckpt 多预算"就是这个思路的 WAM 版本；FastOPD 的 flow map 单状态监督也接近。

---

## 5. 端点自蒸馏：SnapFlow（2604.05656）

**动机（原文）**：

> "every action chunk is produced by integrating a learned velocity field over about ten sequential action-expert evaluations, so action generation dominates model latency and caps the achievable control frequency"

**做法（原文）**：

> "SnapFlow mixes local flow-matching supervision with shortcut samples whose target is the model's own two-step Euler endpoint under stop-gradient, and adds a single zero-initialized destination-time projection so that one action expert serves both objectives while the pretrained predictor is preserved at initialization; **no separate teacher network or critic is required**"

翻成实现：

```python
# 1) 正常 flow-matching 分支
t = sample_t(); x_t = (1-t)*x0 + t*noise
L_fm = mse(v_theta(x_t, t), noise - x0)

# 2) shortcut 分支：目标是"自己走两步 Euler 的端点"（stop-grad）
with torch.no_grad():
    x_mid = x_t + (t_mid - t) * v_theta(x_t, t)      # 第一步
    x_dst = x_mid + (0 - t_mid) * v_theta(x_mid, t_mid)  # 第二步到 0
L_sc = mse(v_theta(x_t, t, d=0), (x_dst - x_t) / (0 - t))  # 直接预测到终点

L = L_fm + λ * L_sc
```

- **zero-init destination-time projection**：加一个初始化为 0 的 time embedding 分量表示"目标时间"，保证初始化时学生=原模型（不破坏预训练）。
- **结果（正文）**：π0.5（10-NFE 教师）→ 1-NFE 学生 LIBERO **98.75% vs 教师 97.75%**；去噪 **9.6×**、端到端 **274ms→83ms**（每步约 23ms）；SmolVLA 500M 上 MSE **−8.3%**、端到端 **3.56×**；**朴素 1 步只有 96.75%**（说明 shortcut 项值钱）；n_act=5 长程 93% vs 90%；π0 配对评测 +10.5 点。
- **训练**：单卡 A800 约 **12 小时**；LIBERO 4 suite / 40 任务 / 400 episodes；π0.5 与 SmolVLA 共用同一套超参。
- **可叠加性**：与 Shallow-π（2×）、EfficientVLA（1.9×）正交。
- **为什么能反超**：多步积分有离散化误差，1 步反而没有累积误差；shortcut 目标自带"向前看两步"的正则。

---

## 6. 随机单跳：VLA-SSD（2611.20460）

**问题（原文）**：

> "In group-based RL methods like GRPO, every rollout step requires multi-step ODE integration, and the sampler's limited stochasticity leads to rollout groups with low variance."

**做法（原文）**：

> "self-distills a pretrained flow-matching action head into a stochastic one-step policy by conditioning a single-jump generator on a Brownian path."

实现要点：
1. 把 flow map 扩展成 **Itô map**：输入中间动作 + 一段标准布朗运动 `W` 的实现，输出单跳端点；
2. 学生 `a_1 = a_0 + G_θ(a_0; W|s) + m`，`m` 是由布朗路径导出的漂移项——**随机性来自路径，不只来自初始噪声**；
3. 因为有了显式密度（1 次 JVP 可算），可以直接做 GRPO 的 likelihood-ratio：`Â=(R−R̄)/(std+κ)`，`L_clip=−E[min(ρÂ, clip(ρ,1−ε,1+ε)Â)]`。

**结果（摘要/引言）**：
- action head 前向 **÷7.7**；LIBERO 相对降 **5.5%**（摘要口径；引言对 π0.5 报掉 0.3 点，两处口径不一致，未核实）；
- 零方差组比例 **63%→35%**（随机性带来的探索增益）；
- GRPO 后 **93.0%**（较 SFT **+15.9 点**），训练 rollout **3.2×** 快；四步 GRPO 基线 91.5%；
- 摘要另称 π0.5 + GRPO 相对提升 **20.6%**。

**洞察**：这条工作真正卖的不是"快"，而是"**把确定性 flow 变成可做 RL 的随机策略**"。蒸馏在这里是"为了 RL 能跑"的前置步骤——精度略降可以接受，探索性提升更值钱。

---

## 7. On-policy + 步数：FastOPD（2610.02832）

**定位**：把 on-policy 蒸馏（学生自己跑、教师在学生状态上给速度场）与步数压缩（flow map 单状态监督 + self-consistency）合在一起。

**摘要级结论（L1）**：
- π0.5 / LingBot-VLA / MolmoAct2 → 轻量学生；
- LIBERO 2 步保留 **84%**，延迟 **−78.1%**；
- RoboTwin2.0 单步 **+15.9pp**；
- 明示也适用于 WAM。

**为什么值得单独看**：它把"蒸什么"（教师速度场）和"在哪查教师"（学生自己的状态）解耦，是 2026 下半年 on-policy 线的代表配方。详见 06 章。

---

## 8. 横向对比：五种方法在动作头上的实际表现

| 方法 | NFE | 需要教师 | 训练成本 | 多峰动作保持 | 代表数字 |
|---|---|---|---|---|---|
| Progressive | 1（4 轮） | 是（上轮学生） | 高 | 中 | 经典图像 4 步≈原质量 |
| Consistency/LCM | 1–4 | 否 | 中 | 低–中 | 图像 4 步可用 |
| DMD/DMD2 | 1 | 是（score） | 中–高 | 中 | 图像 1 步 SOTA |
| Shortcut/MeanFlow | 1–8 单 ckpt | 否 | 中 | 中 | 单 ckpt 多预算 |
| 端点自蒸馏（SnapFlow） | 1 | 自己 | 低 | 中–高 | LIBERO 98.75 vs 97.75；−3.3× |
| 随机单跳（VLA-SSD） | 1 | 自己 | 低 | 高（保留随机性） | ÷7.7 前向；GRPO +20.6% |

**选型建议**：
- 只要能部署、不追极致质量 → **SnapFlow 式端点自蒸馏**（不需要教师、一次训练）。
- 要做 RL 后训练 → **VLA-SSD 式随机单跳**（保随机性 + 可算似然）。
- 有强视频/世界模型教师 → **DMD 变体**（DyMD/DIDO），但准备好调参。
- 要单 ckpt 多算力档位 → **Shortcut/MeanFlow**。

---

## 9. 实现坑（动作空间特有）

1. **不要对速度场做 MSE 而不对齐 t 和 x_t**：同一 t 才有意义；不同 t 的场不能直接比。
2. **端点 MSE 会让多峰塌缩**：绕障任务里两个可行方向会被平均成"撞中间"。要么加分布项（DMD），要么保留随机性（VLA-SSD）。
3. **stop-gradient 必须有**：SnapFlow 的目标来自自己，不 detach 会自我强化错误。
4. **zero-init 投影很重要**：不然 1 步学生初始化就偏离预训练模型，训练早期崩。
5. **评测要看 NFE-成功率曲线**，不要只报 1 步；2 步常常是性价比拐点。

## 10. 实现代码（逐方法）

> 输入 `x0/x1 [B,H,D]`（动作）、`cond`（视觉语言条件）；输出 `loss: scalar`。`recon` = 按描述重构。

### 10.1 Progressive Distillation（2202.00512）

**物理意义（解释）**：反复「两小步并一大步」。物理上=把 10 次小修压成 1 次大修，每轮都让上一轮学生当老师。

```python
# 输入: x_t, t；输出: loss（每轮减半）
with torch.no_grad():
    x_mid = x_t + (t - 0.5) * teacher(x_t, t)          # 教师走一步
    x_dst = x_mid + 0.5 * teacher(x_mid, t - 0.5)      # 再走一步
loss = F.mse_loss(student(x_t, t), (x_dst - x_t) / (0 - t))
# 学生成为下一轮教师，重复 log2(N) 次
```

### 10.2 Consistency / LCM（2303.01469 / 2310.04378）

**物理意义（解释）**：任意时刻都能一步到终点，且同一条 ODE 轨迹上自洽。物理上=学会「从任何半成品直接交出成品」。

```python
# 输入: x_t, t, t_next；输出: loss
with torch.no_grad():
    x_next = solver_step(x_t, t, t_next)               # ODE 相邻点
loss = F.mse_loss(f_student(x_next, t_next), f_student(x_t, t))
# LCM 额外：skipping-step（一次跳多步）+ scheduled sampling
```

### 10.3 DMD / DMD2（2311.18828 / 2405.14867）

**物理意义（解释）**：让学生的输出分布匹配教师/真值分布。物理上=不要求逐帧像，只要「生成出来的动作集合」像；所以保多峰。

```python
# 输入: cond；输出: loss
with torch.no_grad():
    s_real = score_real(x_t, t, cond)                  # 真值/教师 score
x0 = student(x_t, t, cond)
s_fake = score_fake(x_t, t, cond)
loss = ((s_real - s_fake) * x0).mean()                 # 分布匹配梯度
# DMD2：去掉回归项，加 GAN loss 稳定训练
```

### 10.4 Shortcut / MeanFlow（2410.12557 / 2505.13447）

**物理意义（解释）**：学「跳 d 步的平均速度」，推理取最大 d。物理上=一个网络学会「大步走」，单 checkpoint 支持多步数。

```python
# Shortcut：输入 x, t, d；输出 v
v = student(x, t, d)                                   # d = 本次跳的步长
target = (x_{t+d} - x_t) / d                           # 平均速度
loss = F.mse_loss(v, target)
# 自洽：v(x,t,2d) ≈ 0.5*(v(x,t,d) + v(x_{t+d},t+d,d))
```

### 10.5 SnapFlow（2604.05656）

**物理意义（解释）**：用自己两步 Euler 端点当老师。物理上=用自己的「慢想」教自己的「快答」；stop-grad 防止自我强化错误。

```python
# 输入: x0, cond；输出: loss
t = torch.rand(B, 1); eps = torch.randn_like(x0)
x_t = (1 - t) * x0 + t * eps
L_fm = F.mse_loss(model(x_t, t, t_dst=t, cond=cond), eps - x0)
with torch.no_grad():                                   # 整链 stop-grad
    t_mid = 0.5 * t
    x_mid = x_t + (t_mid - t) * model(x_t, t, t_dst=t_mid, cond=cond)
    x_dst = x_mid + (0 - t_mid) * model(x_mid, t_mid, t_dst=0, cond=cond)
    v_tgt = (x_dst - x_t) / (0 - t)
L_sc = F.mse_loss(model(x_t, t, t_dst=0, cond=cond), v_tgt)
loss = L_fm + lam * L_sc
```

### 10.6 VLA-SSD（2611.20460）

**物理意义（解释）**：布朗路径注入随机性，一步仍有分布。物理上=快答但不失去随机应变；有密度才能做 GRPO。

```python
# 输入: a0 [B,L], state；输出: loss
W  = torch.randn(B, L).cumsum(-1)                      # 布朗路径
a1 = ito(a0, W, state)                                 # 单跳
logp, = jvp(lambda z: ito(z, W, state), (a0,), (torch.ones_like(a0),))
R = reward_fn(a1); A = (R - R.mean()) / (R.std() + kappa)
ratio = (logp - logp_old).exp()
loss = -torch.min(ratio * A, ratio.clamp(1-eps, 1+eps) * A).mean()
```

### 10.7 FastOPD（2610.02832）

**物理意义（解释）**：单点 on-policy + 有限区间自一致。物理上=既在学生会到的地方补课，又保证「一大步 = 两小步」。

```python
# 输入: x1, cond；输出: loss
t   = torch.rand(B, 1) * 0.9 + 0.05
x_t = student.jump(x1, s=1.0, t=t, cond=cond).detach()
L_opd = F.mse_loss(student.velocity(x_t, t, cond),
                   teacher.velocity(x_t, t, cond))
s = t * torch.rand(B, 1); m = 0.5 * (s + t)
x_s = student.jump(x1, s=1.0, t=s, cond=cond).detach()
with torch.no_grad():
    v_s = student.velocity(x_s, s, cond)
    v_m = student.velocity(x_s + (m - s) * v_s, m, cond)
    tgt = 0.5 * (v_s + v_m)
L_sc = F.mse_loss(student.interval_vel(x_s, s, t, cond), tgt)
loss = L_sc + lam * L_opd
```

---

更新时间: 2026-11-21
