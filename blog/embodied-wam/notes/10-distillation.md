# WAM × 蒸馏：用大模型知识实现/加速世界-动作模型的论文、方法与伪代码

> 问题：哪些 WAM 是「通过蒸馏大模型」实现的？怎么蒸、蒸什么、效果如何；最近三个月（2026-08-21 ~ 11-21）有哪些新进展。
> 纪律：分类基于论文原文；每个数字标来源级别（L3=全文核实 / L2=部分正文 / L1=摘要级）。**L1 级方法的伪代码是「按摘要重构」，不是原文算法**，已逐条标注。
> 更新：2026-11-21

---

## 0. 结论：现在有五条路线 + 一个「爆发期」

| 路线 | 蒸什么 | 代表 | 一句话 |
|---|---|---|---|
| **A. WAM→WAM 知识转移** | 教师 WAM 中间层隐状态 → 上下文 token | CKT-WAM (2605.06247) | 14B→2B，仅训 1.17%，双骨干冻结 |
| **B. 自蒸馏（压缩去噪步数）** | 多步去噪的结构化先验 → 一步/少步 | S-VAM (2603.16195)、**DIDO (2609.15570)**、**APM (2611.22592)**、**AnyStep-WAM (2609.33748)** | 把 8–35 步蒸成 1–4 步 |
| **C. WAM/VLM → 小策略（表征蒸馏）** | 冻结教师中间特征 → VLA/策略对齐 | THAW-VLA (2609.24682)、**DreamFree (2611.04527)**、**W2A (2611.02600)**、**ForeTime-VLA (2608.20735)** | 部署时不带教师（32ms/21×/11Hz） |
| **D. 数据/标签蒸馏** | 大模型生成视频/潜动作/伪动作 | GR00T N1、LAPA、DreamGen、Veo-Act、NovaFlow、RIGVid | 把「会想象」变「有数据」 |
| **E. 在线/同策略蒸馏（强化学习式）** | 教师在自己轨迹上的条件分布 | **WAM-OPD (2609.34250)**、**HINT-WAM (2611.24789)**、**WAM-Policy (2611.24011)** | on-policy 蒸馏，弥补分布漂移 |
| （辨析）**不是蒸馏** | — | Fast-WAM、Rift、GigaWorld-Policy、LAWA、mimic-video、X-WAM | 训练-推理解耦/调度优化 |

**最近三个月（8–11 月）是「蒸馏爆发期」**：仅 arXiv 就检索到 10+ 篇专门做 WAM 蒸馏的工作——从「压缩去噪步数」（DIDO/APM/AnyStep）到「免想象部署」（DreamFree/W2A/THAW-VLA）、「弹性模型」（LoopedWAM）、「在线蒸馏」（WAM-OPD/HINT-WAM/WAM-Policy）。

---

## 1. 最近三个月新工作速查（全部 arXiv 摘要核实，L1）

| 日期 | 工作 | 蒸什么 / 怎么蒸 | 关键结果 |
|---|---|---|---|
| 11-19 | **LoopedWAM** (2611.25595) | **弹性自蒸馏**：每个专家共享 6-block core 循环（30 层只存 12 block）；跨 action-depth 与采样步预算训练单一 elastic checkpoint | 0.58B；LIBERO **96.1%**、RoboTwin **91.9%**、RoboCasaGR1 **52.2%**；比 Fast-WAM 少 **7.4×** 峰值显存；LIBERO 27.6ms/24.7ms 两个工作点 |
| 11-18 | **Phoenix** (2611.24938) | **双教师蒸馏**：DINOv3 语义 + 几何/时空语义知识 → Phoenix Encoder；CS-VAE 压缩成生成潜变量 | 39k 小时数据；LIBERO-Plus **94.8%**、RoboTwin C2R **72.4%** |
| 11-18 | **HINT-WAM** (2611.24789) | **在线自蒸馏**：学生自己跑去噪轨迹；同一模型以「示教未来视频」为特权信息当教师；用 **energy score** 做分布级蒸馏（不回归速度场） | 无需环境奖励/仿真交互 |
| 11-18 | **WAM-Policy** (2611.24011) | 跨形态适配 + **OPSD**（一视频+两动作步）+ 多教师 **M-OPD**；四足 loco-manip | 77.0%→OPSD 76.7%（**14× 快**）→85.0% |
| 11-17 | **APM** (2611.22592) | **Anchored Path Matching**：从学生到达的状态继续教师到零噪声，取其终点与首中间态为锚点；模态路由 adapter；梯度冲突时「锚优先」 | 单 checkpoint 支持 **1/2/4 步**；RoboTwin Clean +2.0/Rand +1.4；PSNR 30.169→**32.317**；真机延迟 **21.3×** |
| 11-17 | **ReLAP** (2611.22077) | JEPA 空间循环视频预测 + **潜动作蒸馏**（冻结逆动力学教师）+ 循环未来特征预测 | 11M 样本、1.06B 可训；LIBERO **97.9%**、LIBERO-Plus 70.0% |
| 11-19 | **CauSyncVLA** (2611.25667) | 因果潜世界模型 + **动作敏感蒸馏**（保留冻结专家控制响应）；PFWS 单步流校正 | LIBERO-10 93.8%，吞吐 **5.04×** |
| 11-03 | **DreamFree** (2611.04527) | **未来表征蒸馏**：视觉编码器只用当前观测预测冻结视频扩散模型的表征；动作模块从其抽动作信息；推理不生成未来视频 | 14B 骨干下 **11Hz** 闭环；可从小视频/跨本体视频学新任务 |
| 11-02 | **W2A** (2611.02600) | **两阶段**：训练期把视频扩散模型的中间去噪表征蒸进动作模型，测试期完全丢弃视频模型 | LIBERO **95.9%**；比代表性 WAM 快 **~21×** |
| 09-28 | **WAM-OPD** (2609.34250) | **on-policy 蒸馏** + **PWTR**：固定轨迹池（初学生 rollout + 教师 rollout），在存储历史上重新生成去噪路径；用前缀累积分数做代理重要性权重 | 不额外环境交互完成任务适配 |
| 09-27 | **AnyStep-WAM** (2609.33748) | **预算对齐的教师轨迹蒸馏**：区间条件 flow map + 冻结教师转移 + 共享 LoRA；风险-收益调度器按一步预览选最小预算 | Motus/FastWAM/LingBotVA 去噪步 **-60.2%/-49.8%/-85.28%**；一步预算下成功率 **+7.07/+12.08/+8.94** |
| 09-20 | **HapticWAM** (2609.23888) | 触觉教师（凝胶图像+形变/剪切/力/扳手）→ 学生（移除指尖输入，保留生成式接触头） | 真机 3 任务均值 **77%**（41/50，82% pooled） |
| 09-14 | **DIDO** (2609.15570) | **一步蒸馏** = distribution matching + **交互中心表征引导**（夹爪/物体 bbox 视觉推理 token + DINOv3 层对齐） | LIBERO **99.0%**、LIBERO-Plus 76.6%、RoboTwin **92.0%** |
| 08-21 | **ForeTime-VLA** (2608.20735) | **因果未来 token 蒸馏**：当前+未来视频潜压成白化 64-D 目标；8 帧历史编码器预测；4 个 future token + 1 个 phase token 进 VLM 前缀 | 传送带抓取：静止 81.1%、慢速 58.9%；44/90 vs pi0.5 23/90 |
| 08-23 | **DreamMimic** (2608.22278) | 用 RSSM 世界模型做**特权教师蒸馏**的稳定器（预测潜动力学 + 辅助头 + PCG 自适应调度） | 人形 loco-manip（OMOMO/BEHAVE） |

> 注：以上除 CKT/S-VAM/THAW 外均为**摘要级（L1）**；伪代码为按摘要重构。

---

## 2. 实现细节与伪代码

### 2.1 CKT-WAM（L3，原文方法节）

```
def ckt_wam_train(loader, teacher, student, ckt, opt):
    for (obs, text, future_video, future_action) in loader:
        # 1) 教师单次前向（冻结；t*=0；取第 20/40 层）
        H = teacher.encode(obs, text, t=0, stop_grad=True).layer[20]      # [B,N,d_tea]
        # 2) 压到学生维度 + 可学习 query 压缩
        Z  = LN(up(down(H)))                                             # [B,N,d_stu]
        Cg0 = LN(MHCA(Qg, Z@Wk, Z@Wv) + Qg)                              # [B,32,d_stu]
        Cs0 = LN(MHCA(Qs, Z@Wk, Z@Wv) + Qs)                              # [B,32,d_stu]
        # 3) 常开 adapter + 稀疏专家 adapter（top-2/8）
        Cg = Cg0 + up(down(Cg0))
        p  = softmax(router(mean_pool(H)))
        idx = topk(p, k=2)
        Cs = sum((p[m]/sum(p[idx])) * (Cs0 + up_m(down_m(Cs0))) for m in idx)
        CA = concat([Cg, Cs], dim=1)
        # 4) 注入学生条件序列（加 separator）
        Et = concat([student.text_emb, SEP, CA], dim=1)
        # 5) 冻结学生只做前向，损失=学生自身 flow-matching
        v_hat, a_hat = student.denoise(noisy_v, noisy_a, cond=Et)
        loss = L_act(a_hat, a0) + 1.0*L_vid(v_hat, v0) + 0.01*L_bal(p)
        opt.step(loss, params=ckt_only)          # 187.4M / 1.17%

def ckt_wam_inference(obs, text):
    H  = teacher.encode(obs, text, t=0).layer[20]      # 一次
    CA = ckt(H)                                        # 一次
    Et = concat([text_emb, SEP, CA])
    for step in range(K):                              # CA 全步复用，不再查教师
        a = student.step(a, Et)
    return a
```

### 2.2 S-VAM（L3，原文方法节）

```
# Stage 1: 在具身视频上微调 SVD
fit(SVD, embodied_video, steps=100k/40k)

# Stage 2: 自蒸馏（50k 步、单张 H100；SVD 冻结）
for (obs, instr, _) in loader:
    V_hat   = SVD.sample(cond=obs, steps=S)            # 教师：自身多步生成
    Y_geo   = DPAv3(interp(V_hat))
    Y_sem   = DINOv2(interp(V_hat))
    F       = concat([interp(f) for f in SVD.step1.upsample_features])   # 一步特征
    for i in [geo, sem]:
        Y_ref  = Phi_i(interp(obs))
        F_i0   = concat([F, repeat(Y_ref)], dim=ch)
        F_iK   = decoupler_i(F_i0)                     # 时空 transformer
        loss_i = ||F_iK - Y_i||^2
    update(decouplers, loss_geo + loss_sem)            # SVD 冻结

# Stage 3: 动作专家（SVD+decoupler 冻结；QFormer 压缩 + DiT 扩散）
C      = concat([F_geoK, F_semK, F])
F_agg  = FFN(SelfAttn(CrossAttn(Q, C)))
train(DiT, loss=||eps - eps_phi(a_j, F_agg, E, j)||^2)

# 推理：一步特征 → decoupler → 动作专家（不做视频 rollout）
```

### 2.3 THAW-VLA（L2）

```
# 离线：教师只跑一次
for (traj_id, idx) in train_frames:
    cache[traj_id, idx] = Cosmos3_Nano.layer24(mean_pool_views(frame))   # 4096-D

# 训练：学生（0.8B Qwen3.5-VL）只加 cosine 对齐
for batch in loader:
    s = student(obs, instr)                                 # 单次 prefill
    L = L_flow(a) + 0.5 * (1 - cos(mean_pool(s_pooled), cache[traj, idx]))
    update(student)                                         # 教师不进显存

# 部署：与未蒸馏基线逐位一致（4 flow steps；32ms/1.86GB）
```

### 2.4 最近三个月方法（按摘要重构；L1）

**DreamFree（未来表征蒸馏，免想象）**
```
for (obs, future) in data:
    z_t = stopgrad( video_diffusion.encode(future) )   # 冻结教师表征
    z_p = visual_encoder(obs)                          # 学生只看当前观测
    loss = ||z_p - z_t||^2 + L_action
# 推理：动作模块吃 z_p；不生成未来视频（11Hz）
```

**W2A（两阶段表征蒸馏）**
```
# 阶段1：视频模型跑训练帧，取中间去噪表征 h_t
# 阶段2：动作模型 f(a | obs, h_t) 与 h_t 对齐/条件；测试丢弃视频模型
# 结果：LIBERO 95.9；~21× 加速
```

**APM（Anchored Path Matching，少步蒸馏）**
```
for (obs, a0) in data:
    s = student.sample_state(obs)                # 学生到达的状态
    (a_end, a_mid) = teacher.continue_to_zero_noise(s)   # 教师终点 + 首中间态
    pred_end, pred_mid = adapter_end(s), adapter_mid(s)
    L = L_end + L_mid
    if grad_conflict(L_end, L_mid):
        apply_anchor_priority()                  # 保端点梯度、去局部冲突分量
# 一个 checkpoint 支持 1/2/4 步；RoboTwin +2.0/+1.4；21.3× 延迟
```

**DIDO（一步蒸馏 + 交互中心引导）**
```
teacher = multi_step_video_model
student = one_step_generator
z_teacher = teacher.multi_step(cond=obs)                 # 收敛后的交互动态
z_student = student.one_step(obs)
L = DMD(z_student, z_teacher)                            # 分布匹配
  + L_token(夹爪/物体 bbox 视觉推理 token)                # 交互中心
  + L_dino(目标物体多层特征 ↔ DINOv3)                     # 语义锚
# LIBERO 99.0 / LIBERO-Plus 76.6 / RoboTwin 92.0
```

**AnyStep-WAM（预算对齐蒸馏）**
```
for budget k in {1,2,4,8}:
    train interval-conditioned flow map f(x; k) 用冻结教师的真实转移 (x_t -> x_{t+1/k}) 作监督
    shared LoRA 更新
# 推理：一步预览预测难度 → 调度器选最小满足风险的 k
# Motus/FastWAM/LingBotVA 步数 -60.2/-49.8/-85.28%
```

**WAM-OPD（on-policy 蒸馏 + PWTR）**
```
pool = initial_student_rollouts + teacher_rollouts        # 固定轨迹池
for traj in pool:
    for decision t in traj:
        hist = traj.history[:t]
        fresh_path = student.denoise(hist)               # 每次重新采样
        L_t = KL(teacher(a | hist, privileged) || student(a | hist))
        w_t = proxy_importance(prefix_scores(t))         # 前缀累积分数
        update(student, w_t * L_t)
# 不额外环境交互
```

**HINT-WAM（hindsight 自蒸馏）**
```
for demo in demonstrations:
    obs_t = demo.observation
    # 学生用自己的去噪轨迹生成动作
    a_student = student.sample(obs_t)
    # 同一模型 + hindsight 未来视频作条件 → 教师分布
    p_teacher = student.sample(obs_t, cond=hindsight_future_video)
    L = EnergyScore(p_teacher, a_student)             # 分布级，不回归速度场
    update(student)
```

**LoopedWAM（弹性自蒸馏）**
```
core = 6-block shared transformer                     # 循环 5 次 = 30 层
for (depth_budget, step_budget) in budgets:
    y = loop(core, depth_budget)
    L = distill(teacher_30step, y, steps=step_budget)
# 单一 elastic checkpoint：1/2/4 步 × 不同 action-depth
# 0.58B；LIBERO 96.1；27.6ms
```

**ForeTime-VLA（未来 token 蒸馏）**
```
for (obs_hist, future_video, phase, T2T) in data:
    target = whitening(compress64D(video_latent(current) ⊕ video_latent(future)))
    pred   = history_encoder_8frames(obs_hist)        # 预测 target
    L = L_flow + cos(pred, target) + L_relational_geo
      + L_phase + L_T2T + L_action_equiv
    prefix = concat(detach(future_token×4), phase_token)
# 推理：无需未来帧，因果可用；传送带 81.1/58.9%
```

---

## 3. 五条路线总表（含老工作，便于对照）

| 路线 | 工作 | 教师 | 学生 | 机制 | 结果 | 级别 |
|---|---|---|---|---|---|---|
| A | CKT-WAM | DreamZero-14B | Cosmos-Policy-2B | 第 20 层隐状态 → 64 token → 文本条件 | LIBERO-Plus 86.1；真机 83.3 | L3 |
| B | S-VAM | 自身多步视频 + DPAv3/DINOv2 | 一步 decoupler | VFM 特征蒸馏 | CALVIN 4.16 | L3 |
| B | DIDO | 多步视频模型 | 一步生成器 | DMD + 交互 token + DINOv3 | 99.0/76.6/92.0 | L1 |
| B | APM | 教师终点/中间态 | 学生 adapter | 锚点路径匹配 | 1/2/4 步；21.3× | L1 |
| B | AnyStep | 冻结教师转移 | 区间 flow map | 预算对齐蒸馏 | -60.2/-49.8/-85.3% 步 | L1 |
| C | THAW-VLA | Cosmos3-Nano（layer-24） | 0.8B VLA | 离线缓存 + cosine | 97.9；32ms | L2 |
| C | DreamFree | 14B 视频扩散 | 预测编码器 + 动作模块 | 未来表征蒸馏 | 11Hz | L1 |
| C | W2A | 视频模型中间去噪表征 | 动作模型 | 两阶段蒸馏 | 95.9；21× | L1 |
| C | ForeTime-VLA | Fast-WAM 系教师 | pi0.5 | 因果未来 token | 81.1/58.9 | L1 |
| D | GR00T N1 | 视频生成 + VQ-VAE + IDM | WAM | 神经轨迹/潜动作/伪动作 | 88h→827h | L3 |
| D | LAPA / DreamGen / Veo-Act / NovaFlow / RIGVid | 视频/潜动作生成 | WAM/VLA | 数据蒸馏 | — | L1–L2 |
| E | WAM-OPD | 任务教师 | 学生 | on-policy + PWTR | 无额外交互 | L1 |
| E | HINT-WAM | 同模型+hindsight | 同模型 | energy score 自蒸馏 | — | L1 |
| E | WAM-Policy | 教师 WAM | 学生（少步） | OPSD + M-OPD | 76.7→85.0 | L1 |
| — | LoopedWAM | 自身 | 弹性 checkpoint | 弹性自蒸馏 | 96.1；0.58B | L1 |

---

## 4. 选型与辨析

1. **有强教师、要小部署**：CKT-WAM（改条件序列）或 THAW-VLA（离线缓存、部署零开销）。
2. **要免想象/低延迟**：DreamFree（表征蒸馏）、W2A（测试期丢视频模型）、THAW（32ms）。
3. **要少步但保留质量**：APM（1/2/4 步单 checkpoint）、DIDO（一步+交互引导）、AnyStep（按需预算）、LoopedWAM（弹性）。
4. **要在线适配/RL 式**：WAM-OPD、HINT-WAM、WAM-Policy。
5. **不要把 Fast-WAM/Rift/GigaWorld-Policy/LAWA 叫蒸馏**——没有教师被拟合。

## 5. 缺口

- 最近三个月 15 篇新工作均为**摘要级（L1）**，方法节未逐篇精读；伪代码为按摘要重构，**引用公式前需回原文核对**。
- 老工作（CKT/S-VAM/THAW）为 L2–L3，实验表仍有截断未核。
- 「蒸馏」在 WAM 社区尚无统一定义；本页按机制归纳，非官方分类。
