# 08 · 蒸馏 trick 总谱：从 Hinton 到 VLA/WAM 的全部招式

> 目标：把「蒸馏」拆成可以单独开关的零件。每个 trick 给 **公式 / 最小代码 / 什么时候值得用 / 什么时候是坑**，并标注它在 VLA/WAM 里的真实落点。
> 标记：★ = 常用（论文里高频出现）；☆ = 少见（但有效或值得知道）；✗ = 常见误用。

---

## 0. 先记住三句话

1. **蒸馏的收益 = 教师信号的信息量 × 学生能吸收的带宽**。教师再强，学生没有可对齐的接口（层数/维度/动作空间不一致）也白搭。
2. **同一篇论文里通常同时开 3–5 个 trick**，消融表里单看某一个的增益往往只有 0.3–1.5 个点。
3. **VLA/WAM 的蒸馏瓶颈不在 loss，而在"对齐什么"**：动作空间异步、时序 chunk 长度不同、教师看不到学生的状态分布——这三件事决定了 trick 的选择。

---

## A. 信号整形（同样是对齐，怎么对齐更有效）

### A1 ★ 温度软化 + KL（Hinton 2015）

```python
logits_s = student(x) / T
logits_t = teacher(x).detach() / T
loss = F.kl_div(F.log_softmax(logits_s, -1), F.softmax(logits_t, -1),
                reduction="batchmean") * T * T
```

- **何时用**：分类/离散动作 token（RT-2 式 VLA、FAST tokenizer）。
- **坑**：T 太大梯度消失；VLA 的"类别"是动作 bin 时，softmax 的语义距离没有物理意义（相邻 bin 未必相邻动作），此时 **regression/flow loss 比 KL 更合理**。Orion-Lite 明确弃用 KL：*"avoiding ... suboptimal distributional metrics"*，改用 L1 回归 latent token。

### A2 ★ Top-k / 只蒸高置信类

只对教师 top-k logits 做 KL，其余置零。用于教师词表巨大（32k+）且长尾噪声多的场景。VLA 里常用于"动作 token 词表"（FAST 的 1024 bins）。

### A3 ☆ Label smoothing KD（LS-KD, 2020）

教师软标签再和 one-hot 做一次平滑：`q = (1-α)·softmax(z_t/T) + α/K`。收益小、实现零成本，适合教师过自信（自动驾驶里常见）。

### A4 ★ DKD：把 KD 拆成 TCKD + NCKD（2203.08679）

```
KD = TCKD(目标类 vs 非目标类) + NCKD(非目标类内部)
```

- 洞察：传统 KD 里 NCKD 被 TCKD 压制，导致"暗知识"学不到。
- VLA 落点：多任务 VLA 的动作分类头；对"该不该动"（TCKD）与"怎么动"（NCKD）解耦有解释力。

### A5 ☆ DIST：相关一致性 + 泰勒展开（2205.10536）

不拟合 logits 值，而拟合 **教师/学生 logits 的相关结构**：

```
L = L_CE + β·L_KD + γ·L_corr,  L_corr = ||ρ(z_s) - ρ(z_t)||²
```

- 适合教师与学生**容量差距大**（7B→0.1B）时，值对不齐但结构可对齐。驾驶 VLA 的 latent planning token 蒸馏可以借这个思路（Orion-Lite 只用 L1，没用相关项，是一个可做的消融）。

### A6 ☆ Whitening / 白化对齐（WTTM 一类）

先对特征做白化再对齐，去掉教师特征的"主导方向"，让学生学到更均衡的表示。适合特征维度高（4096-D 世界模型特征）且各向异性强的情况——THAW-VLA 的 4096 维 Cosmos 特征正是这种形态。

### A7 ☆ RKD 关系蒸馏（1904.05068）

蒸的是**样本间距离结构**：`L = Σ||d(z_s^i,z_s^j) - d(z_t^i,z_t^j)||`。VLA 中可用于"动作 chunk 之间的相似度结构"，但 batch 内构造 O(B²)，大 batch 不划算。

### A8 ☆ CRD 对比蒸馏（1910.10699）

把蒸馏写成对比学习（InfoNCE），教师特征作正样本、其他样本作负样本。小数据（遥操几千条）时比 MSE 更稳，但需要大 batch / memory bank。

---

## B. 对齐位置（蒸哪一层）

### B1 ★ FitNets hint layer（1412.6550）

教师中间层输出经 1×1 conv/linear 投影到学生维度，做 MSE。**要点**：hint 层要选"教师已经形成语义、但还没塌缩到任务"的位置；Orion-Lite 选的是 LLM 输出的 latent planning token，CKT-WAM 选的是 40 层里的第 20 层。

### B2 ★ Attention Transfer（1612.03928）

对齐 attention map 的归一化分布：`L = ||A_s/||A_s|| - A_t/||A_t||||²`。视觉 backbone 蒸馏常用；VLA 里对"看哪里"（驾驶的 ROI）有直接解释力。

### B3 ☆ FSP：层间 Gram 矩阵（CVPR 2017，无 arXiv）

蒸 `G = F_i F_j^T`（层与层的流动），对**深度不同的学生**更鲁棒。适合 7B→0.1B 这种层数不匹配。

### B4 ☆ Jacobian matching

对齐 ∂y/∂x，蒸的是"局部敏感度"。理论上优雅，实际算不动（VLA 输入维度太大），只在极小模型上见过。

### B5 ☆ CKA / CCA 对齐

用 CKA 相似度做损失（而非逐元素 MSE），对旋转/缩放不变。适合跨架构（Transformer teacher → Mamba student）。

### B6 ★ 多层加权 / 层选择搜索

不是所有层都值得蒸。常见做法：每层一个可学习权重 `w_l`，或直接在验证集上搜"蒸哪一层最好"。Orion-Lite 的消融显示 **中间层 > 最后一层**（最后一层过拟合任务，中间层更通用）。

---

## C. 策略/时序（VLA 特有，最容易被忽略）

### C1 ★ On-policy 蒸馏 / DAgger（1011.0686）

学生自己跑，教师对学生**实际到达的状态**给动作标签：

```
s ~ π_student,  a* = π_teacher(s),  L = ||π_student(s) - a*||
```

- 解决 exposure bias：离线蒸馏的学生一旦偏离教师轨迹就崩。
- 成本：每个训练 step 要一次教师前向（教师 7B 时很贵），所以近期工作都在**降采样教师查询**（每 N 步查一次）或用**离线缓存 + 少量在线修正**。

### C2 ☆ Scheduled sampling / 混合策略

按概率 p 用教师动作、1-p 用学生动作 rollout，逐步退火 p→0。比纯 DAgger 稳，实现简单，但 p 的调度很敏感。

### C3 ★ Action chunk 蒸馏的对齐陷阱

教师输出 H_t 步 chunk、学生输出 H_s 步时，直接 MSE 会错位。三种解法：
1. **重采样到同一时间网格**（线性插值）；
2. **只对齐 chunk 的前 k 步**（k = min(H_t,H_s)）；
3. **对齐"执行后的状态"**而不是动作本身（model-based distillation）。

SnapFlow 的 shortcut 目标就是"两步 Euler 端点"，本质是把多步积分的结果当作对齐目标，而不是逐步对齐速度场。

### C4 ★ 分布蒸馏（flow/diffusion 专用）

动作头是 flow matching 时，不能只 MSE 端点（多峰分布被平均成中间态）。做法：
- **速度场匹配**：`L = ||v_s(x_t,t) - v_t(x_t,t)||²`（同 t、同 x_t 才有意义）；
- **分布匹配**：DMD 用得分差 `∇log p_t - ∇log p_s` 做梯度；
- **端点匹配**：只蒸 x_1（SnapFlow、consistency）。

### C5 ★ Verifier / Best-of-N 蒸馏（2610.14885）

教师=verifier 选出来的最优候选。把"测试时搜索"的增益蒸进权重：

```
候选 c_1..c_N ~ π,  c* = argmax V(c),  L = flow_loss(π, c*)
```

- 关键：只对**成功 episode** 蒸馏（拒绝采样），否则把 verifier 的噪声也学进去。
- 效果：SmolVLA 5k 上 54.1%→70.6%（保留 TTC 增益的 74%）。

### C6 ★ 拒绝采样 / STaR 式自举

学生采样 N 条，只保留成功的做 SFT。等价于"用环境当 verifier 的数据蒸馏"。H2E 就是它的任务级版本。

### C7 ☆ Hint-conditioned 生成（AutoVLA 式）

给大模型 GT 动作当 hint 再让它写 CoT，抑制幻觉：

> "GT 动作作 hint 抑制幻觉"（AutoVLA §3.2）

代价是标注带教师偏见；用于造数据可以，用于评估不行。

### C8 ✗ Teacher-forcing 评估陷阱

离线指标（动作 L2）好，不等于闭环好。驾驶 VLA 里 open-loop 与 closed-loop 的排序经常相反（Orion-Lite 引用的 DiMA 就是"只做 open-loop 评估"的反例）。

---

## D. 步数/结构蒸馏（把 10 步压成 1 步）

> 这是 2026 年 VLA 蒸馏最热的一条线，因为**动作头 NFE 直接决定控制频率**。

### D1 ★ Progressive distillation（2202.00512）

迭代减半：教师 2N 步 → 学生 N 步 → 再当教师 → N/2 步……每轮学生学"教师两步的端点"。

```
x_{t-2Δ} = teacher(teacher(x_t, t), t-Δ)   # 目标
L = ||student(x_t, t) - x_{t-2Δ}||²
```

- 优点：稳定、可解释；缺点：要训 log2(N) 轮，每轮都贵。

### D2 ★ Consistency distillation（2303.01469 / LCM 2310.04378）

学"任意 t 都能一步到 x_0"的 consistency function `f(x_t,t)=x_0`，用 ODE solver 的相邻点做自洽约束：

```
L = d(f(x_{t_{n+1}}), f(x_{t_n}))   # 沿同一 ODE 轨迹
```

- LCM 加了 skipping-step 和 scheduled sampling，1–4 步可用。
- VLA 落点：π0/π0.5 的 flow 动作头压步数。

### D3 ★ DMD / DMD2（2311.18828 / 2405.14867）

把"蒸馏"写成**分布匹配**：学生采样 x，教师/真值分布给 score，更新学生使其分布靠近教师：

```
∇θ L ≈ E[ (s_real(x_t) - s_fake(x_t)) · ∂x/∂θ ]
```

- DMD2 去掉回归损失、加 GAN loss 稳定训练。
- 优点：1 步质量高；缺点：训练不稳、需要 reward/score 网络，调参玄学。

### D4 ★ Shortcut models（2410.12557）

训练一个"带步长条件"的网络 `v(x,t,d)`，d 是本次要跳的步长；推理时 d=最大 → 一步。**不需要两阶段蒸馏**，但训练时要采样多种 d。

### D5 ★ MeanFlow（2505.13447）

直接学**平均速度** `u(x,t,r) = (x_r - x_t)/(r-t)`，满足恒等式 `u = v - (r-t)·du/dt`。一步生成，且是"从零训练"而非蒸馏，常被当作 shortcut 的替代。

### D6 ★ 端点自蒸馏 / SnapFlow（2604.05656）

教师=模型自己的两步 Euler 端点（stop-gradient），加一个 zero-init 的 destination-time projection，让一个 action expert 同时服务两个目标：

> "mixes local flow-matching supervision with shortcut samples whose target is the model's own two-step Euler endpoint under stop-gradient ... no separate teacher network or critic is required"

- 结果：π0.5 10-NFE → 1-NFE，LIBERO 98.75% vs 教师 97.75%，延迟 −3.3×。
- **为什么能反超**：1 步消除了积分误差累积，且 shortcut 目标自带"多看一步"的正则。

### D7 ☆ 随机单跳 / VLA-SSD（2611.20460）

把确定性 flow 改成**布朗路径条件**的单跳生成器，从而有随机性、可算似然比：

> "self-distills a pretrained flow-matching action head into a stochastic one-step policy by conditioning a single-jump generator on a Brownian path"

- 结果：action head 前向快 **7.7×**，LIBERO 成功率相对降 **5.5%**；但用它的 rollout 做 GRPO，π0.5 相对提升 **20.6%**。
- 洞察：**蒸馏的收益不一定在精度，可能在"让 RL 跑得动"**。

### D8 ☆ Adversarial distillation（如 ADD/LADD）

判别器区分教师/学生输出分布，学生对抗训练。质量高但不稳，VLA 里少见。

### D9 ★ Token/层剪枝 + 蒸馏

- 视觉 token 剪枝后，用原模型特征做蒸馏恢复精度（POKE 2611.21033 是"probe offline, keep efficient"的剪枝路线）。
- 层跳过（LayerSkip 2404.16710）：训练时随机跳层、推理时早退，配 self-distillation 补偿。

---

## E. 教师侧技巧

### E1 ★ Teacher assistant（TA）

教师太大、学生太小，中间加一个中等模型做"助教"：大→中→小两段蒸。VLA 里等价于 7B→1B→0.3B。**代价**：两段误差叠加；**收益**：容量差距过大时明显更稳。

### E2 ★ Multi-teacher（含冲突感知）

多个教师（不同任务/不同模态）各自出信号，按置信度/梯度冲突加权。2026-11 出现"冲突感知多教师 on-policy 蒸馏"（2611.21094），核心是**动作空间对齐后再聚合**，否则教师之间互相打架。

### E3 ☆ Born-again / 自蒸馏多代

学生训好后当教师再训一个新学生，迭代几代。对小数据正则化有效，但代际误差会累积。

### E4 ★ EMA teacher

教师=学生权重的指数滑动平均，提供更稳的目标（Mean Teacher 1703.01780）。半监督/自蒸馏标配。

### E5 ★ 离线特征缓存（THAW-VLA 式）

教师只跑一次，把特征写进 memory-mapped 缓存，训练时按 (trajectory id, frame idx) 取：

> "At inference ... all distillation modules are dropped, so latency is identical to the undistilled student by construction."

- 收益：训练不需要教师显存；部署零开销。
- 坑：缓存的是**教师看到的状态**，学生偏离后特征失效（off-policy 偏差）。

### E6 ★ 教师集成 / verifier 当教师

不只是"教师模型"，任何能给候选打分的（reward model、PDMS、碰撞检测）都能当教师信号源。

### E7 ★ 特权信息蒸馏（Learning by Cheating 1912.12294）

训练时教师能看到特权信息（真值感知、未来帧、仿真状态），学生只能看传感器。VLP 是它在驾驶里的变体：教师（冻结 LLM）把 GT 感知/规划写成 prompt，学生用对比损失对齐。

---

## F. 学生侧技巧

### F1 ★ LoRA 学生

不全量训学生，只训 LoRA（Verify-Act-Distill 就是把增益蒸进 LoRA）。适合"部署模型不能动、只想补一个能力"。

### F2 ★ 冻结主干 + 只训头/投影器

Orion-Lite 冻结视觉编码器 + QT-Former，只训 0.1B decoder；CKT-WAM 教师学生全冻结，只训 187.4M（1.17%）的 CKT 模块。**性价比最高的做法**，但上限受冻结特征限制。

### F3 ★ 量化感知蒸馏（QAD）

先量化再蒸，或蒸的同时量化（INT4/INT8）。车端部署必做，但论文里常被忽略：**蒸馏损失对量化误差不敏感**，要显式加量化前后一致性项。

### F4 ☆ 结构重参数化 + 蒸馏

把大 kernel/多分支重参数化成小模型，再用蒸馏恢复精度（RepVGG/DBB 系）。视觉 backbone 可用，VLA 的 transformer 不适用。

### F5 ★ Distill-then-RL

先蒸馏（冷启动）再 RL 微调（GRPO/PPO）。AutoVLA 的 RFT、VLA-SSD 的 GRPO 都是这个配方：**蒸馏给"会做"，RL 给"做好"**。

---

## G. 数据侧技巧

### G1 ★ CoT 蒸馏

大模型生成推理链，学生 SFT。AutoVLA（Qwen2.5-VL-72B → 3B，45.6k nuPlan + 7.2k Waymo）、Senna（GPT-4o）、DriveGPT4（ChatGPT）都是这条路线。

### G2 ★ 合成轨迹

DreamGen（12 个新动词：38.3% vs 13.1%）、GR00T N1 的神经轨迹（88h 遥操 → 827h 生成视频，≈10×）。

### G3 ★ 潜动作（latent action）

LAPA 用 VQ-VAE 学帧间离散潜动作，把无动作视频变成可训练标签；GR00T N1 把 VQ-VAE encoder 当 IDM。

### G4 ★ 世界模型生成视频 → 伪动作

Veo-Act（Veo-3 生成未来图 → IDM）、NovaFlow（视频 → 3D object flow → IK）、RIGVid（生成演示 → 6D 跟踪）。

### G5 ☆ 难例挖掘 / 失败回放

只蒸"教师对、学生错"的样本，效率高但需要在线评估。H2E 的"只留成功 session"是它的粗粒度版本。

### G6 ★ 数据混合比例

GR00T N1 后训练真机:神经轨迹 = 1:1，是调出来的经验值；比例错了会"学飞"。

---

## H. 少见但值得知道的偏门招式

### H1 ☆ 闭式权重编辑（2611.11894 Weightsmith）

不训练，直接算一个低秩更新把"指令理解"写进权重：

> "Closed-form instruction distillation copies the per-layer effect of an explicit rewording into the language model's MLP weights."

- 具体：π0.5 在 RoboLab 把 "in the bowl" 绑错物体，回放 21 条误读决策，把"显式改写指令"的逐层效应闭式写进 MLP；
- 结果：128 个 held-out seeds 成功率 **27→106**（修好 81、弄坏 2，p=7×10⁻²²）；同幅度随机方向无效；工具使用 8/64→36/64；只存 **0.16%** 参数的变体保留大部分收益；
- 适用：已部署策略的**定点 bug 修复**（不是能力压缩，也不能提升通用能力）；改成 action expert/head 效果更小。

### H2 ☆ 把教师蒸进 KV cache / prefix

不蒸权重，蒸一段可复用的 prefix（context distillation）。部署时省掉 system prompt 和示例，适合固定指令集的 VLA。

### H3 ☆ 反向蒸馏：小模型教大模型

用学生（快、便宜）生成数据/软标签去**选数据**或**预训练教师**。少见，但在数据筛选里有效。

### H4 ☆ 跨架构蒸馏（Transformer → SSM/Mamba）

CKA/关系蒸馏 + 逐层对齐，学生不是同构 transformer。边缘芯片上 SSM 更省，但工具链不成熟。

### H5 ☆ 跨模态蒸馏（LiDAR/深度 → 纯视觉）

驾驶经典（DriveAdapter、Hydra-MDP 的 teacher-student）。Orion-Lite 的"纯视觉学生 + 7B 教师"就是这条路线的 VLA 版本。

### H6 ★ 蒸馏 + 剪枝 + 量化三明治

顺序一般是：**先蒸馏（保精度）→ 再剪枝（去冗余）→ 最后量化（省显存）**，每步都用前一步模型当教师。反过来做（先量化再蒸馏）精度掉得更多。

### H7 ☆ 用蒸馏做遗忘/去偏

把"不该学的"（隐私、捷径特征）蒸掉：教师不带某特征，学生对齐教师即可遗忘。研究阶段，工业界慎用。

### H8 ☆ Test-time distillation / 自一致性

同一模型多次采样投票，把投票结果蒸回单次推理。和 Verify-Act-Distill 同源，但教师=自己。

### H9 ✗ 把"课程学习/多阶段训练"叫蒸馏

没有教师信号，只是训练调度。分类时容易混。

### H10 ✗ 把"模型合并（model merging）"叫蒸馏

TIES/DARE 是权重空间平均，不需要学生拟合教师输出。效果可能类似，机制不同。

---

## I. 组合配方：三条经过验证的流水线

| 配方 | 组成 | 代表 | 适用 |
|---|---|---|---|
| **驾驶 VLA 车端化** | 数据蒸馏（造 CoT）→ 隐层蒸馏（7B→0.1B decoder）→ RFT（PDMS） | AutoVLA + Orion-Lite | 有大量无标注驾驶数据、要车端实时 |
| **操作 VLA 提速** | 动作头步数蒸馏（10→1）→ 可选 on-policy 修正 | SnapFlow / VLA-SSD | π0 系 flow 动作头、控制频率是瓶颈 |
| **WAM 小部署** | 世界模型特征离线缓存 → 学生 cos 对齐 → 部署丢弃蒸馏模块 | THAW-VLA / CKT-WAM | 有强世界模型教师、部署不能带教师 |

---

## J. 消融优先级（如果只能做 5 个实验）

1. **有没有教师**（对照：同学生同数据，无蒸馏）——最大的 single factor。
2. **对齐层位置**（浅/中/深）。
3. **on-policy vs 离线**（教师查询预算相同）。
4. **步数**（10/4/2/1 NFE 的成功率-延迟曲线）。
5. **蒸馏后 RL**（是否还有增益）。

---

更新时间: 2026-11-21
