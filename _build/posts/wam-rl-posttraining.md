---
title: "从 SFT 到闭环：VLA 与 WAM 的 RL 后训练怎么做"
title_en: "From SFT to Closed Loop: RL Post-training for VLA and WAM"
summary: "把 RL 后训练从 WAM 综述里拆出来单独讲：奖励从哪来、GRPO 与 KL-RL 怎么写、五个代表案例（GigaBrain / VAMPO / EVA / CreFlow / WorldVLN），以及闭环训练的六类参考方法与工程清单。"
summary_en: "A standalone guide to RL post-training for VLA and WAM: where rewards come from, how GRPO and KL-regularized RL are implemented, five case studies, and six references for closed-loop training."
tags: ["RL", "后训练", "闭环训练", "VLA", "WAM"]
tags_en: ["RL", "Post-training", "Closed-loop", "VLA", "WAM"]
---

> 这篇与 WAM 论文站分开：那里按「原理分类」讲模型结构，这里只讲**训练范式**——尤其是 SFT 之后怎么用奖励信号继续优化，以及**闭环训练**（closed-loop training）有哪些可参考的工程做法。
> 证据纪律同前：`[论文 §X]` 为原文小节；`[代码]` 为官方仓库；数值为论文自评（L1/L2/L3 标注）。未给出的量不补写。

## 0. TL;DR

1. VLA / WAM 的 SFT 有三个天花板：**分布漂移**（covariate shift）、**长程复合误差**、**执行性 gap**（视频看着合理但动作不可执行）。RL 后训练针对的是这三件事，而不是「更多演示」。
2. 奖励信号几乎从不直接来自真机执行——五种常见替代：**世界模型价值 / IDM 动作平滑度 / LTL 逻辑验证 / 专家潜距离 / 稀疏 episode 成功**。
3. 算法上只有两大族：**GRPO**（组归一化优势 + 截断代理，无 critic）与 **KL-RL 的闭式解改写为加权 NLL**（advantage-conditioned）。少数走 DiffusionNFT（稀疏奖励）。
4. **闭环训练**不是单一算法：HILR、DAgger 式恢复、失败数据回流、世界模型 rollout、测试时闭环，是五条可组合的工程路径。

---

## 1. 为什么 SFT 之后还要后训练

| 失败模式 | 表现 | 后训练如何介入 |
|---|---|---|
| 分布漂移 | 偏离演示分布后误差累积 | 闭环 rollout 采自己的状态（DAgger/HILR） |
| 长程复合 | 单步像、多步崩 | 段级/子任务级奖励，或世界模型长程评分 |
| 执行性 gap | 视频合理但动作抖动/越限 | IDM 解码后按动作空间约束给奖励（EVA） |
| 指令接地差 | 靠视觉捷径而非指令 | 语言教练重标注 / 视觉指令目标（CoachSI、JEPA-WAM） |

WorldVLN 的原文把动机说得很直白：`[论文 §2]` "Such imitation-based training fits the demonstration distribution but remains vulnerable to covariate shift and accumulated errors."

---

## 2. 后训练的三种形态

1. **离线偏好 / 加权模仿**：不改环境交互，只用「好/坏」标签或优势加权 NLL。
   - GigaBrain：把 KL-RL 闭式解改写成两项加权 NLL（见 §4.1）。
2. **在线 RL**：训练内环生成 rollout、算奖励、更新策略。
   - VAMPO / EVA / CreFlow / WorldVLN。
3. **闭环数据回流**：用执行结果制造新训练数据（不一定是梯度 RL）。
   - HILR、DAgger 恢复、失败数据掩码、语言教练 relabel。

> 注意：**DreamGen 不是 RL**——生成→伪标注→行为克隆是单向数据管线，没有回报回环 `[论文 §2]`。

---

## 3. 奖励从哪来：五种可核实的替代信号

| 奖励 | 代表 | 怎么算 | 代价 |
|---|---|---|---|
| 稀疏 episode 成功 | GigaBrain | 成功 0 / 失败 −C_fail / 其它 −1，再经世界模型价值 n 步 TD 变优势 | 需要世界模型价值网 + 真机日志 |
| 专家潜距离 | VAMPO | `r = −λ1·L1(x̂, x*) + λ2·cos(x̂, x*)`，在视频 VAE 潜空间算 | 完全离线，只需专家视频 |
| 动作平滑/限幅 | EVA | 冻结 IDM 解码关节 → jerk/加速度 Huber + 速度/加速度限幅惩罚 | 需要冻结 IDM |
| LTL 逻辑验证 | CreFlow | 监视器输出二值通过 + 违例 witness（哪帧违反哪条） | 需要离线合成 LTL 规格 |
| 进度/价值标量 | FACT / τ0-WM | 子任务级进度标签 + 蒙特卡洛传播；或 [0,1] 进度值头 | 需要任务分段标注 |

---

## 4. 算法层：两条主线 + 一个变体

### 4.1 KL-RL → 加权 NLL（GigaBrain / RAMP）

`[论文 §3.2.1]` "we extend the KL-regularized reinforcement learning framework to our augmented state space"，闭式解 `π̂(a|S) ∝ π_ref(a|S)·exp(A/β)`；再用 Bayes 把「优势条件化」写成两项 SFT：

```text
A = Σ γ^k r_{t+k} + γ^n v_{t+n} − v_t     # 世界模型价值 n 步 TD
I = 1[A > ε]                              # 二值优势（简化条件）
L = −log π(a|o,z,l) − α·log π(a|I,o,z,l)   # 两项 NLL，不需要策略梯度
```

工程要点：世界模型 token 用 **p=0.2 随机注意力掩码**训练，推理可关闭世界模型；Stage 4 世界模型与策略联合续训，防优势坍缩（A≈0）。

### 4.2 GRPO（VAMPO / EVA / WorldVLN）

组内归一化优势 + PPO 式截断，**没有 critic**：

```text
G 条 rollout（共享初始噪声）→ 每条算奖励 r_g
A_g = (r_g − mean(r)) / (std(r) + ε)       # 组归一化
loss = mean( min(A_g·ρ, A_g·clip(ρ, 1−c, 1+c)) )   # ρ = π_new/π_old
```

三个实现差异：
- **VAMPO**：把多步去噪当 MDP，**只有首步 SDE**（其余 ODE）。原文理由 `[§3.3]`："introducing stochasticity at all denoising steps leads to severe credit assignment issues"。
- **EVA**：Flow-GRPO 只更新生成器 LoRA，奖励来自冻结 IDM。
- **WorldVLN**：段级（非 token 级）优势，`r_j = γ^{j−1}(λ_traj r_traj + λ_task r_task + λ_ref log π_ref)`。

### 4.3 变体：稀疏奖励 DiffusionNFT（CreFlow）

- 二值 LTL 奖励 + 组共享掩码：只在违例帧/实体上加梯度，其余锚定预训练先验；
- 失败样本向「组内正样本均值」纠正（corrective reflow）；
- KL 到冻结参考策略，防止 off-manifold。

---

## 5. 五个案例的完整训练循环（伪代码）

### 5.1 GigaBrain RAMP（VLA 自身）

```text
① 世界模型预训练：4K 小时真机数据，输入 [z_t; Ψ(v_t); Ψ(p_t)]，预测未来帧 offsets{12,24,36,48}
② 优势条件化微调：L = −log π(a|o,z,l) − α·log π(a|I,o,z,l)，掩码 p=0.2
③ HILR rollout：自主执行 + 专家介入，软件自动剔除干预边界伪影
④ 联合续训：世界模型 + 策略同时更新（HILR + base 数据）
```

### 5.2 VAMPO（只训视频预测器）

```text
for each step:
  z0 ~ N(0,I); rollout = sample_denoise(z0, steps=20)   # 首步 SDE，其余 ODE
  r = latent_reward(rollout.final, expert_latent)        # −L1 + cos
  A = group_normalize(r, G=8)
  update VPM by clipped surrogate（lr=1e-6, clip=0.2, adv_clip=2.0）
```

### 5.3 EVA（视频规划器 + IDM 奖励）

```text
q = IDM(video)                             # 冻结 IDM
R = −(λj·Huber(jerk) + λα·Huber(acc) + λv·max(|v|−vmax,0)² + λa·max(|a|−amax,0)²)
update generator LoRA by Flow-GRPO（组优势 + clip + β·KL(π‖π_ref)）
```

### 5.4 CreFlow（稀疏奖励）

```text
(r, witness, mask) = LTL_monitor(rollout)          # 二值 + 违例位置
L_NFT  = 组共享掩码 M 门控的 NFT 损失（正/负分支）
L_CR   = ‖M ⊙ (x̂0 − mean(正样本 x0))‖²             # 只对失败样本
L      = L_NFT + L_CR + β·KL(π‖π_ref)
```

### 5.5 WorldVLN（自回归 WAM 的在线 RL）

```text
for each decision segment j:
  在线 rollout（真观测回灌，替换模型预测的潜）
  r_j = γ^{j−1}(λ_traj·r_traj + λ_task·r_task + λ_ref·log π_ref)
  A_j = (r_j − μ_j)/σ_j                  # 段级组归一化
  update policy by clipped surrogate
```

---

## 6. 闭环训练（closed-loop training）：六类参考方法

> 定义：训练期让策略/世界模型**执行自己的动作并利用执行后果**，而不是只看演示。以下六类都可单独用，也可组合。

### 6.1 人机接力（HILR / DAgger 式干预）

- **做法**：策略自主执行，失败边界由人（或模拟器）接管；记录后做行为克隆，且**软件剔除干预切换处的抖动**。
- **参考**：GigaBrain Stage 3（HILR）；XPACE 用模拟器合成 deviation–recovery 轨迹并仅以 8% 合成恢复数据微调，恢复消融 61.7→86.7（异套件）。
- **要点**：干预点是最值钱的分布边界样本；但 before/after 混淆训练时长（XPACE 自认）。

### 6.2 自生成恢复数据（Recovery / Correction）

- **做法**：在专家轨迹附近注入扰动，由世界模型/模拟器推演恢复路径，过滤后加入训练。
- **参考**：XPACE（simulator 适配自身生成上下文后合成恢复轨迹）；`[摘要]` "synthesize deviation–recovery trajectories around expert demonstrations, and fine-tune the policy on filtered recovery examples"。
- **要点**：过滤（一致性）是关键；恢复数据比例低（8%）即可显著提升。

### 6.3 失败数据闭环（Failure-aware training）

- **做法**：保留失败 rollout，但**掩掉其中的动作模仿损失**，只学「失败后果」与下调的进度；再用进度值头做候选打分。
- **参考**：FACT `[§3.1–3.3]`；DreamAvoid 用成功/失败/边界混合数据训 dream evaluator，测试时选动作。
- **要点**：失败数据不能当正样本；掩码对象选错会把策略拉坏。

### 6.4 世界模型 rollout 闭环（Model-based RL）

- **做法**：在世界模型/模拟器里跑长程 rollout，用预测后果算奖励或优势，再更新策略。
- **参考**：WorldVLN（在线模拟器 + 真实观测回灌）；CreFlow（从行为策略采样 rollout 做 LTL 验证）；GigaBrain Stage 4（联合续训防优势坍缩）。
- **要点**：仿真到真实的偏差需要「真观测回灌」或 KL 锚定来兜底。

### 6.5 语言/语义教练闭环（Relabeling）

- **做法**：冻结 VLM 观察执行并命名「下一步」，成功的 rollout 用原任务指令 hindsight relabel 后微调；动作骨干冻结。
- **参考**：CoachSI（长程未见场景 47.4→59.3，动作骨干冻结）；GR-MG/JEPA-WAM 用生成式视觉指令做目标侧接地。
- **要点**：解决的是「知道下一步」而非「会不会做」，与技能 RL 互补。

### 6.6 测试时闭环（Acting loop / Re-planning）

- **做法**：不改参数，改推理：动作块执行中持续回灌真实观测，按需重规划/评估。
- **参考**：τ0-WM 的 propose–evaluate–revise；DreamAvoid 的 trigger→propose→evaluate；WAMotor/Rolling-WAM 的推理侧加速；DreamZero 7Hz 闭环。
- **要点**：这是「闭环」成本最低的落点；但只改善执行，不改善表征。

### 6.7 反例：开环数据管线

DreamGen：生成→伪标注→行为克隆，无回报回环 `[§2]`。它的收益来自数据分布外扩（43.2%/28.5%），**不是闭环**——评估时要区分。

---

## 7. 工程清单（决定要不要上 RL）

1. **先问奖励**：能否离线算出（潜距离/IDM/进度）？能 → 先做「离线 GRPO」；不能 → 才考虑在线 rollout。
2. **再问数据**：有没有真机失败日志？（失败数据比成功数据更稀缺，价值更高。）
3. **三件套防退化**：KL/参考锚定 + 优势裁剪（adv_clip）+ 混合 SFT（保留一部分原始损失）。
4. **评测口径**：闭环任务看成功率 + 完成时间 + 进度；同时报告「相对 SFT 的增量」，不要只报绝对值。
5. **算力**：在线 RL 的 rollout 成本 ≈ 推理成本 × 组大小 G；G=8、20 步去噪时约为单次推理的 160 倍。

---

## 8. 坑

- **自评数字**：本领域几乎没有第三方复现，跨论文比较必须看协议（trials、任务套件、是否同套件）。
- **奖励 hacking**：IDM 平滑度可被「不动」骗过——EVA 加了限幅惩罚正是为此。
- **闭环≠更好**：世界模型偏差会被 RL 放大（GigaBrain 需要 Stage 4 联合续训）。
- **不要把「post-training」都当 RL**：V-JEPA 2 的「post-training」是加动作条件的微调，不是回报优化。

---

## 9. 参考

- GigaBrain-0.5M*（RAMP，KL-RL→加权 NLL）：arXiv 2602.12099
- VAMPO（GRPO 训视频预测器，含官方代码）：arXiv 2603.19370 · github.com/OpenHelix-Team/VAMPO
- EVA（IDM 奖励 + Flow-GRPO）：arXiv 2603.17808
- CreFlow（LTL 稀疏奖励 + DiffusionNFT）：arXiv 2605.14274
- WorldVLN（自回归 WAM 的在线 GRPO）：arXiv 2605.15964
- DreamAvoid（混合数据评估器，测试时选优）：arXiv 2605.11750
- FACT（act-then-imagine + 失败掩码）：arXiv 2608.10232
- τ0-WM（propose–evaluate–revise）：arXiv 2606.01027
- XPACE（deviation–recovery 自改进）：arXiv 2609.17372
- CoachSI（语言教练 relabel）：arXiv 2611.25529
- DreamGen（开环数据管线，反例）：arXiv 2505.12705
- WAM 论文站：/blog/wam-survey/
