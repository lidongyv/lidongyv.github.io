# 03 · WAM 蒸馏：世界-动作模型怎么被"教会"并"压小"

> WAM（World Action Model）= 把"预测未来"和"输出动作"放在同一个生成模型里（CKT-WAM、Cosmos-Policy、DreamZero、GigaWorld-Policy 等）。
> 它的蒸馏比 VLA 多一层：**既要蒸动作，又要蒸"世界表征"**。本页按"蒸什么"分五条路线，最后给选型。
> **实现级细节见第 12 章《实现手册》（CKT-WAM 的 LQCA、S-VAM 的 decoupler、APM 的梯度投影等）。**

---

## 0. 五条路线

| 路线 | 蒸什么 | 代表 | 部署时教师 |
|---|---|---|---|
| **A. 隐状态 → 上下文 token** | 教师 WAM 的中间层隐状态，压缩成可注入的 token | CKT-WAM (2605.06247) | 消失 |
| **B. 去噪轨迹/端点 → 少步** | 多步 ODE 的端点或整条轨迹 | Anchored Path Matching (2611.22592)、RealtimeWAM/TACD (2610.06617)、DIDO (2609.15570)、WAM-Policy (2611.24011) | 消失 |
| **C. 未来视频表征 → 编码器** | 冻结视频扩散模型的中间/未来表征 | THAW-VLA (2609.24682)、W2A (2611.02600)、DreamFree (2611.04527)、FRED (2610.30320) | 消失 |
| **D. 自蒸馏（自己教自己）** | 自身多步生成的视频取 VFM 特征 | S-VAM (2603.16195)、AutodidactWAM (2610.08119) | 不需要 |
| **E. 数据/标签蒸馏** | 生成视频、潜动作、伪动作 | GR00T N1、LAPA、DreamGen | 只当生成器 |

> **辨析**：Fast-WAM、Rift、LAWA、GigaWorld-Policy、mimic-video、X-WAM 属于训练-推理解耦或调度优化，**没有教师被拟合**，不进本表（见 §7）。

---

## 1. A 路线：CKT-WAM（2605.06247，L3 全文核实）

| 项 | 内容 |
|---|---|
| 教师 | DreamZero-**14B** |
| 学生 | Cosmos-Policy-**2B** |
| 蒸什么 | 教师第 **20 层（共 40 层）** 中间隐状态；教师只做单次前向观测编码（`t*=0`），不生成视频/动作 |
| 压缩 | bottleneck（`d_b=512`）→ **LQCA** 压成 **32+32** query token（generalized / specialized） |
| 注入 | 拼到学生**文本条件序列**尾部，cross-attention 进 DiT；**不动学生去噪骨干** |
| 可训 | 教师+学生**全冻结**，只训 CKT 模块 **187.4M（1.17%）** |
| 配置 | 8×A800，bf16，AdamW（β=0.9/0.95，wd=0.01），cosine + 1k warmup，LR 3e-4 |
| 结果 | LIBERO-Plus 零样本 **86.1%**（逼近全量微调）；真机 4 个长程任务平均 **83.3%** |

**为什么这样设计**（原文 §3 的问题陈述）：跨异构 WAM 迁移难——潜接口不匹配、全量微调贵；输出模仿在动作头不对称时脆弱；逐层隐状态对齐太刚性。所以选"**单次前向 + 少量 query token + 条件序列注入**"。

**代价**：教师仍需在训练时加载（14B）；学生推理时不需要，但训练显存是"14B + 2B + CKT"。

---

## 2. B 路线：把多步去噪压成 1–4 步

### Anchored Path Matching（2611.22592，L1，双臂操作）

- 教师：冻结的 LingBot-VA；**Stage I** 用 AnyFlow 式有限区间蒸馏把整网变成 transition map `Ψ_θ`；**Stage II 冻结 base，只训 video/action 两组 modality-routed LoRA**。
- 蒸什么：在**学生自达状态** `z_r` 上，用教师速度场数值积分同时给出**端点锚** `y_0 = Φ_T^{r→0}(z_r)` 与**首步中间态** `z_s = Φ_T^{r→s}(z_r)`（`s=r−1/K_T`）。
- 损失：`L = L_v + λ_a L_a`，`L_anchor = MSE(Ψ(z_r;r→0), y_0)`，`L_field = MSE(Ψ(z_r;r→s), z_s)`；梯度用 **anchor-priority 投影**解冲突：`g_f^⊥ = g_f − min(⟨g_f,g_a⟩,0)/||g_a||² · g_a`。
- 数字：RoboTwin 2.0 **1-step 80.9% / 4-step 91.4%**（仅 Clean 训练、直接在 Random 评测）；较"继续 base 蒸馏" **+1.7/+1.6pp**；视频 PSNR **30.169→32.317dB**；真机 control cycle **4323.6ms→203.1ms（21.3×）**。

### RealtimeWAM / TACD（2610.06617，L2 部分正文）

- 教师：冻结 WAM，学生与 EMA 共享**同一个冻结 Video Expert 的 block-wise KV**（video 只前向一次）；只蒸 action expert。
- 蒸什么（原文）："adopts the frozen teacher's multi-step rollout endpoint as a numerical proxy for the exact clean endpoint, explicitly aligning the student's velocity prediction with the corresponding interval-averaged teacher velocity"。
- 损失：`L = L_CD + λ L_TA`，`L_TA = E||v_S(a_t,t) − sg[u_T]||²`，`u_T=(a_t−a_0^T)/t`；附录证明它等价于**权重 1/t² 的端点回归**。
- 数字：单次推理 **12.2ms**、H100 **~25×**、精度降 **<1%**；原 Fast-WAM 单次 >300ms。

### DIDO（2609.15570，L2 部分正文）

- 教师：Wan2.2-TI2V-5B 的 4-step 变体，先在机器人演示上微调成"robot-domain teacher"；学生=单步生成器，Stage II 用 MoT 接 action expert。
- 蒸什么：**收敛后的分布**，不是第一步粗输出（原文："Rather than reproducing the coarse prediction obtained by naively truncating the teacher after its first step, DMD trains the single-step generator to match the converged output distribution of the four-step teacher"）；另加**交互表征**：object/gripper/交互/对齐四组 token，bbox 轨迹监督 + DINOv3 多层对齐。
- 损失：`L_DMD = E_t[D_KL(p_θ,t || p_T,t)]` + `λ_obj L_box(b̂^obj,b^obj)` + `λ_grip L_box(b̂^grip,b^grip)` + `λ_align L_align`；动作 `L_act = E||v_θ(a_τ,τ)−(a−ε)||²`。
- 数字：LIBERO **99.0%**、LIBERO-Plus **76.6%**、RoboTwin **92.0%**。

### WAM-Policy（2611.24011，L2 部分正文，四足 loco-manipulation）

- 教师：LingBot-VA（双臂）换成四足接口（ego+俯视双视角、9 维指令）后微调得 `π_θ0`，**冻结为教师**；学生 `θ_S=θ0+Δ`，**Δ 是唯一训练参数（LoRA）**，采样器直接降到 **video 1 步 / action 2 步**。
- 蒸什么（原文）："the student's one-step video stream and two-step action stream are regressed to the targets of the teacher's high-NFE sampler, with a stop-gradient on the plan that conditions the student's action"。
- 损失：`L = L_Z + λ_A L_A`，`L_Z = ρ̄_c(Ẑ−Z_T)`，`L_A = ρ̄_c(Â−A_T)`，`ρ_c(r)=√(r²+c²)−c`（pseudo-Huber，对离群端点更稳）。
- 数字：迁移 **77.0%**；OPSD 学生 **76.7%**（不蒸馏 **69.3%**），采样快 **14×**；Stage2 RL+M-OPD **85.0%**；教师单 chunk **2.226s** 只覆盖 **0.32s** 机器人时间。

---

## 3. C 路线：蒸"未来表征"，让策略不再生成视频

这是 2026 下半年最清晰的一条新路线：**世界模型的价值在于"表征"，不在于"像素"**。

### THAW-VLA（2609.24682，L2 部分正文）

- 教师：冻结 Cosmos3-Nano 理解塔（layer-24、每视图 mean-pool、**4096-D**）；学生：0.8B Qwen3.5-VL。
- 做法：教师对训练帧**离线跑一次** → memory-mapped 缓存；训练时只加视图级 cosine 对齐（λ=0.5）；部署**逐位一致**。
- 数字：LIBERO **97.9%**（+2.6）；RoboCasa-GR1 48.2%→**50.5%**；真机 **28/30**；**32ms / 1.86GB**。

### W2A（2611.02600，L2 部分正文）

- 视频 DiT（Cosmos-Predict2，LoRA 适配机器人视频）→ action DiT（从零训练）；PCA top-K + Q-Former；测试时丢弃视频模型。
- 损失按去噪时间加权：`L = L_FM + λ_dist · E_b[ w(τ_b)·(1 − cos(ẑ^a, z̃^v)) ]`，`w(τ)=exp(−(τ−τ_c)²/2σ_w²)`。
- 数字：LIBERO **95.9%**、LIBERO-Long 93.6%、CALVIN **4.30**；比代表 WAM 快约 **21×**。

### DreamFree（2611.04527，L1）

- 冻结 **14B** 视频扩散 → 只用当前观测的预测性视觉编码器 + action module；蒸未来视频表征（不生成像素）。
- 数字：真实闭环 **11Hz**（14B 视频骨干）；真实+仿真成功率高于 SOTA VLA/WAM。

### FRED（2610.30320，L1）

- **自蒸馏**：同一模型内，以 GT 未来表征为目标，从噪声一步预测；蒸未来视频表征 + 历史观测上下文。
- 数字：纯人类视频任务 **50.0%** vs 最佳竞品 WAM **19.4%**。

### DyMD（2609.31349，L1）

- 14B 视频教师 → 4 步 **1.3B** 学生（DMD 变体，teacher/fake-score 重配）。
- 数字：R-Bench **+9.6pp**、PAI-Bench-G **+5.1**、WorldArena 动作规划 **34% vs 16%**。

**共同实现要点**：
1. 教师**离线**跑（一次或每帧一次），特征进缓存；
2. 学生编码器输出的特征做 **cosine / L2** 对齐，不加生成头；
3. 蒸馏 loss 是**辅助项**（λ 0.5 左右），主 loss 仍是动作；
4. 部署时蒸馏模块全部丢弃。

---

## 4. D 路线：自蒸馏（不引入外部教师）

### S-VAM（2603.16195，L3 全文）

- 教师 = 自身多步生成视频 → 冻结 VFM（DPAv3 几何 + DINOv2 语义）提特征。
- 学生 = 一步去噪的噪声特征经 decoupler（时空 Transformer）映射到 VFM 表征：`L = ||F̃ − Y||²`。
- 关键：用"自己多步生成的视频"而非真值未来帧（同一扩散轨迹，避免错位）。
- 成本：**50k 步、单张 H100**；推理单卡 RTX 3090；CALVIN 平均序列长度 **4.16**（VPP 3.58）。

### AutodidactWAM（2610.08119，L1）

- 自蒸馏：生成视频 + 手姿估计 + IK 恢复动作作目标，只调 action 层（视频 teacher-forced）；DPO + SFT + DTW。
- 数字：原生 **17/10/7% → 75/47/42%**；Oreo 90% pre-grasp / 20% 全任务；held-out 80/30%。
- 启示：暴露了 WAM 的**模态不对称**——"视频像"不等于"动作对"。

---

## 5. E 路线：数据蒸馏（见 07 章）

GR00T N1（88h 遥操 → 827h 生成视频，1:1 混合）、LAPA（VQ-VAE 潜动作）、DreamGen（12 个新动词 38.3% vs 13.1%）、Veo-Act / NovaFlow / RIGVid（L1）。

---

## 6. 一张对照表：WAM 蒸馏的效果与速度

| 工作 | 教师 | 学生 | 蒸什么 | 教师效果 | 学生效果 | 速度 |
|---|---|---|---|---|---|---|
| CKT-WAM | DreamZero-14B | Cosmos-Policy-2B | 第 20 层隐状态 | 未给 | LIBERO-Plus 86.1%；真机 83.3% | 1.17% 可训参数 |
| APM | 多步 WAM | 1/2/4 步 | 端点+首中间态 | 未给 | +2.0pp（1 步） | **21.3×** |
| RealtimeWAM | 多步 WAM | 1 步 | 端点+一致性 | 未给 | 掉点 <1% | **25×**（H100） |
| DIDO | 多步视频 WAM | 1 步 | DMD+交互表征 | 未给 | LIBERO 99.0% | — |
| WAM-Policy | 臂部 WAM | 四足 WAM | 步数+策略 | — | 76.7%（OPD）/85.0%（+RL） | **14×** |
| THAW-VLA | Cosmos3-Nano | 0.8B VLA | layer-24 特征 | — | 97.9%；真机 28/30 | 32ms/1.86GB |
| W2A | 视频扩散 | 动作模型 | 中间去噪表征 | — | LIBERO 95.9% | **21×** |
| DreamFree | 14B 视频扩散 | 编码器+动作头 | 未来视频表征 | — | 高于 SOTA VLA/WAM | **11Hz** |
| S-VAM | 自身多步视频 | 同模型 | VFM 特征 | — | CALVIN 4.16 | 单卡 3090 |

> 注意：这些数字来自不同 benchmark（LIBERO / LIBERO-Plus / CALVIN / RoboTwin / 真机），**不可直接横向比较**；表的作用是看"量级"和"代价结构"。

---

## 7. 辨析：常被误称为"蒸馏"的推理加速

| 方法 | 实际机制 | 为什么不是蒸馏 |
|---|---|---|
| Fast-WAM (2603.16666) | 训练期视频共训塑形表征，推理删未来分支（190ms） | 无教师-学生；同一模型训练/推理不同路径 |
| Rift (2608.11521) | anticipation token 一次前向写"未来 K/V cache" | 缓存/预填，不是知识迁移 |
| GigaWorld-Policy (2603.17240) | 因果设计让视频生成在推理时可选（9× 加速） | 同模型的可选分支 |
| LAWA (2608.24882) | 潜意图序列替代像素未来（−42.9% 延迟） | 表征替换，无教师 |
| mimic-video (2512.15692) | 动作解码器读中间 ODE checkpoint | 早停式加速 |
| X-WAM (2604.26694) | 异步去噪（视频多步、动作少步） | 调度优化 |

**判断标准**：有教师权重/特征/输出被学生显式拟合 → 蒸馏；否则是加速。

---

## 8. 选型建议

1. **有强 WAM 教师、要小部署** → CKT-WAM（改条件序列）或 THAW-VLA（离线缓存、部署零开销）。
2. **只有一个大 WAM、想单步** → S-VAM（自蒸馏，单卡 H100 可做）。
3. **视频模型很强、但不想生成像素** → W2A / DreamFree（蒸未来表征）。
4. **多步 WAM 想保质量** → Anchored Path Matching（单 ckpt 多预算）或 DIDO（DMD 变体）。
5. **没有动作数据** → 数据蒸馏（GR00T N1 / LAPA / DreamGen）。

---

更新时间: 2026-11-21
