# 03 · WAM 蒸馏：世界-动作模型怎么被"教会"并"压小"

> WAM（World Action Model）= 把"预测未来"和"输出动作"放在同一个生成模型里（CKT-WAM、Cosmos-Policy、DreamZero、GigaWorld-Policy 等）。
> 它的蒸馏比 VLA 多一层：**既要蒸动作，又要蒸"世界表征"**。本页按"蒸什么"分五条路线，最后给选型。
> **每个方法的实现代码见本章 §9（CKT-WAM / APM / TACD / DIDO / WAM-Policy / THAW / W2A / DreamFree / FRED / DyMD / S-VAM / AutodidactWAM）。**

---

## 0. 五条路线

| 路线 | 蒸什么 | 代表 | 部署时教师 |
|---|---|---|---|
| **A. 隐状态 → 上下文 token** | 教师 WAM 的中间层隐状态，压缩成可注入的 token | CKT-WAM (2605.06247) | 单次前向（只编码，不生成） |
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
| 压缩 | 共享 bottleneck：`d_tea → d_b=512 → d_stu`（学生文本嵌入宽度）；**LQCA** 再压成 **32+32** 个 query token（generalized / specialized） |
| 注入 | 拼到学生**文本条件序列**尾部，cross-attention 进 DiT；**不动学生去噪骨干** |
| 可训 | 教师+学生**全冻结**，只训 CKT 模块 **187.4M（1.17%）** |
| 配置 | 8×A800，bf16，AdamW（β=0.9/0.95，wd=0.01），cosine + 1k warmup，LR 3e-4 |
| 结果 | LIBERO-Plus 零样本 **86.1%**（逼近全量微调）；真机 4 个长程任务平均 **83.3%** |

**为什么这样设计**（原文 §3 的问题陈述）：跨异构 WAM 迁移难——潜接口不匹配、全量微调贵；输出模仿在动作头不对称时脆弱；逐层隐状态对齐太刚性。所以选"**单次前向 + 少量 query token + 条件序列注入**"。

**模块作用与物理意义**（原文 §3.2–3.4）

| 模块 | 形状 | 作用 | 物理意义 |
|---|---|---|---|
| 教师单次前向（`t*=0`，第 20/40 层） | `H_T [B,N,d_tea]` | 把当前观测编码成"世界理解" | 教师**不想象未来**，只看现在；`t*=0` 是干净端点，避免把某一步的噪声当知识 |
| Bottleneck（down→512→up） | `Z [B,N,d_stu]` | 把教师宽度投到学生文本嵌入宽度 | "翻译"：把教师的世界表征变成学生语言通道能吃的向量；512 是信息瓶颈 |
| LQCA（`Q_g` 32 个、`Q_s` 32 个） | `C_g^0, C_s^0 [B,32,d_stu]` | N 个 token → 32+32 个摘要 | 32 个可学习的"问题"去问教师场景：generalized 问任务无关的（物体/空间/可供性），specialized 问任务相关的（技能/阶段） |
| 广义 adapter（always-on） | `C_g [B,32,d_stu]` | 残差 bottleneck，始终生效 | 所有任务共享的"世界常识"翻译 |
| 路由（`mean(H_T)` → MLP → softmax/M） | `p [B,M]` | 选 top-k 个专家 | 判断"现在处于哪类任务/技能"，决定用哪套翻译 |
| 专家 adapter ×M（top-k 稀疏） | `C_s [B,32,d_stu]` | 各自一个残差 bottleneck | M 套"方言"：不同任务/阶段的世界知识表达；top-k 保证每次只付 k 份成本 |
| 拼接 + `<sep>` | `Ẽ_t [B, L_t+1+64, d_stu]` | 塞进学生**文本条件序列**尾部 | 教师知识走学生已有的"语言条件"通道，cross-attention 消费，不改去噪骨干 |
| 冻结学生 + 只训 CKT | 187.4M（1.17%） | 学生生成先验完全不变 | CKT 只是"同声传译"，不是新模型 |

**推理成本（容易误解）**：教师不是训练期专属，也**不是每个 denoising step 都查**。原文是——教师每次观测只做**一次前向**（只编码当前场景，不生成未来视频/动作），得到 `C_A` 后**整条学生去噪轨迹复用**。所以部署形态是"14B 编码器单次前向 + 2B 学生 + CKT"，比"教师多步生成"便宜得多，但也不是把教师完全删掉。

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

## 9. 实现代码（逐方法）

> 形状约定同第 12 章（`B` batch、`N` token、`D` 动作维、`K` 步数）。标 `recon` 的是按论文描述重构、细节未核实。

### 9.1 CKT-WAM（2605.06247）——隐状态 → 条件 token

**物理意义（解释）**：把 14B 世界模型对当前场景的理解浓缩成 64 条「备注」，贴到 2B 学生的文本条件后面；学生不改结构就继承世界知识。教师只做一次编码前向，备注在整条去噪轨迹复用。

```python
# 输入: obs [B,V,3,H,W], text [B,T], latent [B,Nz,Cl]
# 输出: loss: scalar
with torch.no_grad():
    H_t = teacher.encode(obs, text, t=0, layer=20)      # [B,N,d_tea] 单次前向，只编码
h  = ckt.bottleneck(H_t)                                # [B,N,d_stu] 内部 d_b=512
C0g = ckt.lqca_g(ckt.q_g.expand(B, -1, -1), h, h)       # [B,32,d_stu] 32 个 query 摘要
C0s = ckt.lqca_s(ckt.q_s.expand(B, -1, -1), h, h)       # [B,32,d_stu]
logits = ckt.router(H_t.mean(1))                        # [B,8] 用教师原始特征做路由
w, idx = logits.topk(2, dim=-1); w = w.softmax(-1)      # top-k，配置 k=2
Cg = C0g + ckt.adapter_g(C0g)                           # 常开广义 adapter（残差）
Cs = C0s + sum(ckt.adapters[j](C0s) * w[:, k:k+1, None]
               for k, j in enumerate(idx.T))            # top-k 专家（残差）
cond = torch.cat([text_tokens, ckt.sep, Cg, Cs], dim=1) # 拼到文本条件尾部
t = torch.rand(B, 1)
v = student_dit(add_noise(latent, t), t, cond=cond)     # 学生 DiT 冻结
loss = F.mse_loss(v, target_velocity(latent)) + 0.01 * balance(logits)
```

### 9.2 APM（2611.22592）——端点锚 + 首步中间态

**物理意义（解释）**：蒸「教师从学生当前状态继续走到的终点 + 第一步」。物理上=老师从你走错的地方接着走给你看，而不是从头示范。

```python
# 输入: x1 [B,N,D] 噪声；输出: loss
r   = torch.rand(B, 1)
z_r = rollout(student, x1, K=K, to=r).detach()          # 学生自达状态
s   = r - 1.0 / K_T
with torch.no_grad():
    y0 = teacher.integrate(z_r, r, 0.0)                 # 端点锚
    zs = teacher.integrate(z_r, r, s)                   # 首步中间态
L_a = F.mse_loss(psi(student, z_r, r, 0.0), y0)
L_f = F.mse_loss(psi(student, z_r, r, s),   zs)
# g_a, g_f 用 anchor-priority 投影后合并
```

### 9.3 RealtimeWAM / TACD（2610.06617）——区间平均速度

**物理意义（解释）**：蒸「教师多步走的平均速度」。物理上=把十个小碎步折成一大步，同时用一致性项保证这一步不偏离轨迹。

```python
# 输入: obs [B,V,3,H,W], a0 [B,H,D]；输出: loss
with torch.no_grad():
    kv = video_expert.kv(obs)                           # 只前向一次
t = torch.rand(B, 1); eps = torch.randn_like(a0)
a_t = (1 - t) * a0 + t * eps
v_S = action_expert(a_t, t, kv)
with torch.no_grad():
    a0_T = solver(a_t, t, 0.0, teacher, kv)
    u_T  = (a_t - a0_T) / t.clamp(min=1e-3)
loss = consistency_loss(action_expert, ema, a_t, t, kv) \
     + lam * F.mse_loss(v_S, u_T)
```

### 9.4 DIDO（2609.15570，recon）——DMD + 交互 token

**物理意义（解释）**：蒸「收敛后的交互动力学」，不是第一步的粗预测。物理上=等视频「想清楚」再学，别学半成品；否则交互动态会被截断丢掉。

```python
# 输入: cond, gt_future, gt_box；输出: loss
with torch.no_grad():
    z_T = teacher.sample(cond, steps=4)                 # 4 步机器人域教师
z   = student.sample(cond, steps=1)                     # 单步学生
L_dmd  = dmd_loss(z, teacher, cond)                     # 分布匹配（非端点）
L_box  = smooth_l1(box_head(z), gt_box) + giou_loss(box_head(z), gt_box)
L_dino = F.mse_loss(proj(z), dinov3(gt_future))
loss = L_dmd + lam1 * L_box + lam2 * L_dino
```

### 9.5 WAM-Policy（2611.24011）——video/action 双端点

**物理意义（解释）**：蒸视频/动作两个端点。物理上=把「想象未来」和「动手」都压到 1–2 步；plan 分支 stop-grad，动作分支只看不想。

```python
# 输入: eps_Z [B,N,Cz], eps_A [B,H,D]；输出: loss
Z_hat = student.sample_video(eps_Z, steps=1)                     # 1 步
A_hat = student.sample_action(eps_A, Z_hat.detach(), steps=2)    # 2 步
with torch.no_grad():
    Z_T = teacher.sample_video(eps_Z, steps=K_T)
    A_T = teacher.sample_action(eps_A, Z_T, steps=K_T)
loss = pseudo_huber(Z_hat - Z_T) + lam_A * pseudo_huber(A_hat - A_T)
```

### 9.6 THAW-VLA（2609.24682）——离线特征 + cosine

**物理意义（解释）**：蒸世界模型对每个视角的「整体印象」（全局向量）。物理上=只传「这场景大概是什么」，不传像素布局——空间信息确实丢了，这是有意的粗对齐。

```python
# 输入: obs, act, keys；输出: loss
h   = student(obs)                                   # QwenGR00T 0.8B
f_S = h.image_tokens.mean(dim=1)                     # [B,V,D_s] 沿 token 维池化
f_T = load_cache(keys)                               # [B,V,4096] 教师缓存
L_align = (1 - F.cosine_similarity(p_phi(f_S), f_T, dim=-1)).mean()
loss = flow_matching(h, act) + 0.5 * L_align
```

### 9.7 W2A（2611.02600）——PCA + Q-Former + 时间加权

**物理意义（解释）**：蒸视频 DiT 中间层的「去噪表征」，PCA 压到低维再对齐。物理上=把视频模型脑子里的画面压成低维向量喂给动作模型，不生成像素。

```python
# 输入: future, act, cond；输出: loss
with torch.no_grad():
    Z_v = video_dit.latent(future, layer=L, step=tau)   # [B,N,Cv]
Z_v = (Z_v - mu) @ U                                     # PCA → [B,N,K]
Z_a = action_dit.hidden(add_noise(act, tau), tau, cond)  # [B,N,Ca]
Z_h = qformer(query, Z_a)                                # [B,M,K]
w = torch.exp(-(tau - tau_c) ** 2 / (2 * sigma_w ** 2))
L_d = (w * (1 - F.cosine_similarity(Z_h, Z_v, dim=-1))).mean()
loss = flow_matching(action_dit, act, cond) + lam_dist * L_d
```

### 9.8 DreamFree（2611.04527，recon）——只蒸未来表征

**物理意义（解释）**：蒸「未来视频的表征」而不是视频本身。物理上=只学「未来会变成什么样」的抽象，不把未来画出来。

```python
# 输入: obs, future_frames；输出: loss
with torch.no_grad():
    y = video_dit.encode(future_frames)                 # 冻结 14B 视频模型
z = encoder(obs)                                        # 只看当前观测
loss = (1 - F.cosine_similarity(z, y, dim=-1)).mean() \
     + action_loss(action_head(z), act)
```

### 9.9 FRED（2610.30320，recon）——自蒸馏未来表征

**物理意义（解释）**：自蒸馏「从噪声一步预测未来表征」。物理上=练习「闭眼想象未来」，目标来自真实未来观测，不用教师采样。

```python
# 输入: noise, cond, gt_future；输出: loss
with torch.no_grad():
    y = video_dit.encode(gt_future)                     # 目标来自 GT 未来观测
z = video_dit.denoise(noise, cond, steps=1)             # 从噪声一步预测
loss = F.mse_loss(z, y) + action_loss(...)
```

### 9.10 DyMD（2609.31349，recon）——DMD 变体

**物理意义（解释）**：分布匹配，重点是保住「机器人—物体怎么动」。物理上=别把交互动力学蒸没了；画质持平但动作规划更对。

```python
# 输入: cond；输出: loss
z0 = student(cond, steps=4)                             # 1.3B 学生
L_G = score_diff_loss(z0, teacher, critic)              # 教师/fake score 差
L_c = fm_loss(critic, z0, teacher)
loss = L_G + lam * L_c
```

### 9.11 S-VAM（2603.16195）——自蒸馏 VFM 特征

**物理意义（解释）**：用自己多步生成的视频取 VFM 特征当老师。物理上=自己先「做梦」，再用梦里的几何/语义校准一步预测；用同一条轨迹避免错位。

```python
# 输入: obs；输出: loss
F = concat_interp(denoiser.upsample_features(obs, step=0))
with torch.no_grad():
    V_hat = svd_sample(denoiser, obs, steps=K)
    Y_geo = vfm.dpav3(interp(V_hat)); Y_sem = vfm.dinov2(interp(V_hat))
L = ((decoup.geo(torch.cat([F, Y_geo_ref], -1)) - Y_geo) ** 2).mean() \
  + ((decoup.sem(torch.cat([F, Y_sem_ref], -1)) - Y_sem) ** 2).mean()
```

### 9.12 AutodidactWAM（2610.08119，recon）——视频 + 手姿 + IK

**物理意义（解释）**：视频 + 手姿 + IK 造动作标签。物理上=看视频学动作，不需要真机示教；暴露了「视频像 ≠ 动作对」的模态不对称。

```python
# 输入: obs, video；输出: loss
with torch.no_grad():
    hand = hand_estimator(video)                        # 手姿
    a_ik = ik_solver(hand)                              # IK 恢复动作
L_sft = bc(action_layer(obs), a_ik)
L_dpo = dpo_loss(action_layer(obs), a_ik, a_neg)       # 偏好对齐
loss = L_sft + lam * L_dpo + dtw_align(...)
```

---

更新时间: 2026-11-21
