# WAM 蒸馏：工程伪代码（输入/输出/模块级）

> 目标：不看公式也能照着实现。每节给出 —— 角色（教师/学生）、输入、输出、模块表、训练伪代码、推理伪代码、超参、坑。
> 级别：L3=论文方法节逐字核实（CKT-WAM、S-VAM）；L2=部分正文（THAW-VLA）；L1=仅摘要（其余，模块形状多为「摘要未给」）。
> 更新：2026-11-21

---

## 0. 读法

- `[B,T,D]` 形状约定：B=batch，T=序列长度/时间步，D=特征维；`obs`=当前观测（多相机图），`instr`=语言指令，`a`=动作块。
- 「教师」= 冻结的大模型；「学生」= 要部署的模型；「CKT/decoupler/adapter」= 唯一可训练部分。
- 所有 L1 方法的模块名与形状是**按摘要复原**，原文未给的标 `?`。

---

## 1. CKT-WAM（L3）——WAM→WAM 上下文蒸馏

**角色**：教师 DreamZero-14B（冻结）；学生 Cosmos-Policy-2B（冻结）；CKT 模块（可训，187.4M / 1.17%）。

**输入（每个训练样本）**
| 名称 | 形状 | 来源 |
|---|---|---|
| `obs` | 图像 token `[B,N_img,d_tea]` | 多相机观测 |
| `text` | 文本 token `[B,N_txt,d_tea]` | 指令 |
| `future_video` | VAE latent `[B,T_v,d_v]` | 数据 |
| `future_action` | 动作 chunk `[B,T_a,d_a]` | 数据 |

**输出（训练）**：`v_hat, a_hat`（学生预测的干净未来视频/动作 latent）；loss 标量。
**输出（推理）**：`a`（动作 chunk），不生成未来视频。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| teacher encoder | 单次前向取隐状态 | obs,text → `H` | 冻结；第 20/40 层；`t*=0` |
| down/up 投影 | 教师维度→学生维度 | `H[B,N,1280?]` → `Z[B,N,d_stu]` | bottleneck `d_b=512`；LN+激活+dropout |
| LQCA（压缩器） | 把 N 个 token 压成固定 query | `Z` → `Cg0,Cs0` | 两个 query bank；`Kg=Ks=32`；16 头 |
| generalized adapter | 常开残差瓶颈 | `Cg0` → `Cg` | `r_g` 瓶颈（值未给） |
| specialized adapters | 输入相关的 8 个专家 | `Cs0` → `Cs` | `M=8`；`top-2` 路由 |
| router | 从教师隐状态出路由概率 | `mean(H)` → `p[B,8]` | 2 层 MLP |
| 注入器 | 拼到学生文本条件 | `[text_emb, SEP, CA]` | SEP 为可学习 token |
| student denoiser | 冻结，接收新条件 | `noisy_v,noisy_a,Et` → 预测 | 视频/动作双输出 |

**训练伪代码**
```
def ckt_train(loader, teacher, student, ckt, opt):
    for (obs, text, fv, fa) in loader:
        # 教师：一次前向，不生成
        H = teacher.encode(obs_tokens=obs, text=text, t=0)[layer=20].detach()  # [B,N,d_tea]
        Z = LN(up(down(H)))                                                   # [B,N,d_stu]
        Cg0 = LN(MHCA(Qg, Z, Z) + Qg)                                         # [B,32,d_stu]
        Cs0 = LN(MHCA(Qs, Z, Z) + Qs)                                         # [B,32,d_stu]
        Cg = Cg0 + up_g(act(down_g(Cg0)))                                     # [B,32,d_stu]
        p = softmax(router(mean_pool(H)))                                     # [B,8]
        idx = topk(p, 2)
        Cs = sum(renorm(p,m) * (Cs0 + up_m(act(down_m(Cs0)))) for m in idx)   # [B,32,d_stu]
        CA = concat([Cg, Cs], dim=1)                                          # [B,64,d_stu]
        Et = concat([student.text_emb(text), SEP, CA], dim=1)                 # [B,L+65,d_stu]
        # 学生：冻结，只前向；噪声由数据构造
        v_hat, a_hat = student.denoise(noisy_v, noisy_a, cond=Et)
        loss = L_act(a_hat, fa) + 1.0*L_vid(v_hat, fv) + 0.01*L_bal(p)
        opt.step(loss, params=ckt.params)        # 只有 CKT 可训
```
**推理伪代码**
```
def ckt_infer(obs, text, K=10):
    H  = teacher.encode(obs, text, t=0)[20]      # 1 次
    CA = ckt(H)                                   # 1 次
    Et = concat([text_emb, SEP, CA])
    a  = randn([1, H_horizon, d_a])
    for k in range(K):                            # 每步不再查教师
        a = student.step(a, Et)
    return a
```
**超参**：layer 20/40；`d_b=512`；`Kg=Ks=32`；16 头；dropout 0.1；`M=8`、top-2；λ_vid=1、λ_bal=0.01；LR 3e-4；8×A800。
**坑**：①教师必须 `t*=0`（干净端），否则噪声注入进上下文；②CA 一次算、全步复用——不要在每步重算教师；③`top-2` 若不 renorm 会让专家输出幅度随路由变化。

---

## 2. S-VAM（L3）——自蒸馏「多步→一步」

**角色**：教师 = SVD 多步生成 + 冻结 VFM（DPAv3/DINOv2）；学生 = 两个 decoupler；动作专家 = DiT。

**输入**：`obs`（单目 RGB）、`instr`、`future_video`（训练用）。
**输出**：`F_geo, F_sem`（未来几何/语义特征，形状 `[T,C_i,h,w]`）；推理输出 `a`。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| SVD backbone | 一步/多步视频生成 | obs → 视频/latent | 微调后冻结 |
| 一步特征提取 | 取上采样层多尺度特征 | step1 内部 → `F[T,ΣC_l,h,w]` | 插值对齐+concat |
| DPAv3 | 几何教师 | `V_hat` → `Y_geo` | 冻结 |
| DINOv2 | 语义教师 | `V_hat` → `Y_sem` | 冻结 |
| geometric decoupler | 一步特征→几何前瞻 | `[F; Y_ref_geo]` → `F_geo` | K 个时空 block |
| semantic decoupler | 一步特征→语义前瞻 | `[F; Y_ref_sem]` → `F_sem` | K 个时空 block |
| Uni-Perceiver | 压缩成 N 个 token | `[F_geo;F_sem;F]` → `F_agg` | QFormer 式（CrossAttn+SelfAttn+FFN） |
| DiT policy | 动作扩散 | `a_j, F_agg, E` → ε | flow/diffusion |

**训练伪代码（三阶段）**
```
# Stage 1: 微调 SVD（MetaWorld 100k / 真机 40k，4×H100）
for (obs, future_video) in embodied_data:
    loss = SVD.diffusion_loss(future_video, cond=obs)
    update(SVD)

# Stage 2: 自蒸馏（SVD 冻结；50k 步，1×H100）
for (obs, instr) in data:
    V_hat  = SVD.sample(cond=obs, steps=S)                 # 教师视频
    Y_geo  = DPAv3(interp(V_hat))                          # [T,C_g,h,w]
    Y_sem  = DINOv2(interp(V_hat))                         # [T,C_s,h,w]
    F      = concat([interp(f) for f in SVD.step1.upsample_feats])   # [T,ΣC,h,w]
    for i in [geo, sem]:
        Y_ref = Phi_i(interp(obs))                         # [1,C_i,h,w]
        F_i0  = concat([F, repeat(Y_ref,T)], dim=ch)       # [T,ΣC+C_i,h,w]
        F_iK  = decoupler_i(F_i0)                          # [T,C_i,h,w]
        loss_i = mse(F_iK, Y_i)
    update(decouplers, loss_geo + loss_sem)                # SVD/VFM 冻结

# Stage 3: 动作专家（SVD+decoupler 冻结；60k/40k 步，4×H100）
for (obs, instr, a0) in data:
    F      = step1_features(obs)
    F_geoK, F_semK = decouplers(F, obs)
    C      = concat([F_geoK, F_semK, F], dim=ch)
    F_agg  = FFN(SelfAttn(CrossAttn(Q, C)))                # [N,C_agg]
    loss   = diffusion_loss(a0, F_agg, E, j)
    update(DiT)
```
**推理**
```
F = SVD.step1_features(obs)          # 一步，不出视频
F_geo, F_sem = decouplers(F, obs)
F_agg = uniperceiver(F_geo, F_sem, F)
a = diffusion_sample(F_agg, E)       # RTX 3090 可跑
```
**超参**：SVD 骨干；几何教师 DPAv3、语义教师 DINOv2；decoupler 层数 K（值未给）；蒸馏 50k 步。
**坑**：①教师必须用「模型自己多步生成的视频」取特征，用真值未来帧会引入轨迹错位（原文明确）；②`Y_ref` 必须来自**当前观测**，否则 decoupler 退化成纯生成；③Stage 3 冻结 decoupler，否则蒸馏目标会跟着漂。

---

## 3. THAW-VLA（L2）——WAM 特征→VLA 离线蒸馏

**角色**：教师 Cosmos3-Nano（理解塔，冻结，只离线跑一次）；学生 0.8B Qwen3.5-VL（可训）；投影器（训后丢弃）。

**输入**：训练帧 `(traj_id, idx, frame, instr)`；动作标签。
**输出**：学生策略（部署时 32ms/1.86GB；4 flow steps）。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| teacher tower | 提特征 | `frame` → `feat` | layer-24；每视图 mean-pool；4096-D |
| 特征缓存 | 免显存复用 | key=(traj,idx) → `feat` | memory-mapped，训前一次写 |
| 冻结学生（Qwen3.5-VL） | 策略骨干 | obs,instr → hidden | 0.8B；单次 prefill |
| 蒸馏投影器 | 维度对齐 | hidden → 4096-D | 训后丢弃 |
| 对齐损失 | cos 约束 | `s_pooled, feat` → loss | λ=0.5，视图级 |

**训练伪代码**
```
# 0) 离线：教师只跑一次
for (traj, idx, frame) in train_frames:
    cache[traj, idx] = teacher.layer24(mean_pool_views(frame))   # 4096-D

# 1) 学生训练
for (obs, instr, a, traj, idx) in loader:
    h = student.prefill(obs, instr)                    # 单次前向
    s = projector(view_mean_pool(h))                   # 4096-D
    loss = flow_loss(a, h) + 0.5 * (1 - cos(s, cache[traj, idx]))
    update(student)                                    # 教师不在显存
```
**部署**：删投影器与缓存；`student` 与未蒸馏基线结构完全一致（4 flow steps）。
**超参**：λ=0.5；layer-24；4096-D；LIBERO 缓存约 4 卡 1 小时。
**坑**：①缓存 key 必须与数据索引严格对齐，错位会让 cos 对齐到别的帧；②只对齐「视图级 pooled」而非逐 token，否则学生被拉去拟合教师空间结构；③部署等价性要靠「删掉蒸馏模块后参数不变」保证——不要把投影器接进推理图。

---

## 4. DreamFree（L1，摘要级）——免想象的表征蒸馏

**角色**：教师 = 冻结视频扩散模型（14B）；学生 = 视觉编码器 + 动作模块。
**输入**：当前 `obs`；训练时另有 `future` 帧。
**输出**：`z_p`（预测性视觉表征，供动作模块）；部署 11Hz。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| 冻结视频模型 | 提供教师表征 | `future` → `z_t` | 14B，冻结 |
| 学生视觉编码器 | 只看当前观测预测 | `obs` → `z_p` | ? |
| 动作模块 | 从 `z_p` 抽动作信息 | `z_p` → `a` | ? |

**训练伪代码**
```
for (obs, future, a) in data:
    z_t = stopgrad(video_model.encode(future))     # 教师表征
    z_p = visual_encoder(obs)                      # 学生
    loss = mse(z_p, z_t) + L_action(a, action_head(z_p))
    update(visual_encoder, action_head)
```
**推理**：`z_p=visual_encoder(obs)` → `a=action_head(z_p)`；不生成未来帧。
**坑**：摘要未给表征层级/维度；「用当前观测预测未来表征」的瓶颈在编码器容量——太小学不到时序，太大失去免想象意义。

---

## 5. W2A（L1）——两阶段去噪表征蒸馏

**角色**：教师 = 视频扩散模型（训练期用）；学生 = 动作模型（部署）。
**输入**：`obs, instr`；训练期视频模型产中间去噪表征 `h_t`。
**输出**：动作 `a`；LIBERO 95.9%；~21× 加速。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| 视频模型（教师） | 产中间去噪表征 | obs,instr → `h_t` | ? |
| 压缩子空间 | 降维（假设低维够用） | `h_t` → `h'` | ? |
| 动作模型（学生） | 条件动作生成 | `obs, h'` → `a` | ? |

**训练伪代码**
```
# 阶段 1：教师前向，抽中间去噪表征
for (obs, instr, a) in data:
    h_t = video_model.denoise(obs, instr, step=t)      # 中间步
    h'  = compress(h_t)                                 # 低维子空间
    loss = L_action(a, action_model(obs, h'))
    update(action_model, compress)                      # 视频模型冻结
# 阶段 2（可选）：微调动作模型去掉对 h' 的依赖
```
**推理**：无视频模型；`a = action_model(obs, compress(h_t_cache?))`——摘要未说明推理时 `h'` 从哪来（**需回原文确认**：可能是缓存/或用学生自己预测）。
**坑**：这是最需要回原文核对的一段——「测试期丢弃视频模型」后动作模型用什么条件，摘要没说。

---

## 6. APM（L1）——锚点路径匹配少步蒸馏

**角色**：教师 = 多步 WAM（冻结）；学生 = 少步生成器 + 两个模态路由 adapter。
**输入**：`obs, instr, a0`（干净动作）。
**输出**：单 checkpoint 支持 1/2/4 步。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| 学生采样器 | 生成学生状态 | obs → `s` | 少步 |
| 教师续跑器 | 从 `s` 跑到零噪声 | `s` → `a_end, a_mid` | 冻结教师 |
| 端点 adapter | 预测终点 | `s` → `â_end` | ? |
| 中间态 adapter | 预测首中间态 | `s` → `â_mid` | ? |
| 锚优先更新 | 梯度冲突处理 | grads → 更新 | 保端点、去局部分量 |

**训练伪代码**
```
for (obs, instr, a0) in data:
    s = student.sample_state(obs, steps=k)                  # 学生到达的状态
    a_end, a_mid = teacher.continue_to_zero_noise(s)        # 教师锚点
    L_end = mse(adapter_end(s), a_end)
    L_mid = mse(adapter_mid(s), a_mid)
    if cos(grad(L_end), grad(L_mid)) < 0:
        g = grad(L_end) - project(grad(L_mid), grad(L_end)) # 锚优先
    else:
        g = grad(L_end) + grad(L_mid)
    update(adapters, g)
```
**推理**：按部署预算选 1/2/4 步；无需教师查询。
**坑**：锚点必须从**学生自己到达的状态**出发（不是从训练轨迹），否则少步误差会累积；梯度冲突处理是该方法的核心贡献，去掉会退化为普通蒸馏。

---

## 7. DIDO（L1）——一步蒸馏 + 交互中心引导

**角色**：教师 = 多步视频模型；学生 = 一步生成器（仍出未来+动作）。
**输入**：`obs, instr, a0`；教师多步生成 `z_teacher`。
**输出**：一步预测；LIBERO 99.0 / LIBERO-Plus 76.6 / RoboTwin 92.0。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| 多步教师 | 收敛的交互动态 | obs → `z_teacher` | 冻结 |
| 一步学生 | 单步出未来 | obs → `z_student` | ? |
| 交互 token 头 | 夹爪/物体 bbox 视觉推理 token | 特征 → tokens | 监督 bbox |
| DINOv3 对齐 | 目标物体多层特征 | 学生层 → DINOv3 | 冻结 DINOv3 |
| DMD 损失 | 分布匹配 | 学生/教师样本 | distribution matching |

**训练伪代码**
```
for (obs, instr, a0, boxes) in data:
    z_t = teacher.multi_step(obs)                    # 收敛的交互动态
    z_s = student.one_step(obs)
    L = DMD(z_s, z_t)                                # 分布匹配
      + BCE(token_head(feat), boxes)                  # 交互中心
      + mse(proj(student.layers), dinov3(target_obj)) # 语义锚
    update(student, token_head, proj)
```
**坑**：摘要未给学生步数/结构；「交互 token」监督需要 bbox 标注（数据要求）。

---

## 8. AnyStep-WAM（L1）——预算对齐蒸馏 + 调度

**角色**：教师 = 冻结 WAM；学生 = 区间条件 flow map + 共享 LoRA；调度器（轻量）。
**输入**：`obs, instr, a0`；教师真实转移 `(x_t → x_{t+1/k})`。
**输出**：任意预算 1..K 步的动作；平均步数 -60.2/-49.8/-85.28%（Motus/FastWAM/LingBotVA）。

**模块表**
| 模块 | 作用 | 输入→输出 | 规格 |
|---|---|---|---|
| 教师转移采集 | 提供区间监督 | 轨迹 → `(x_t, x_{t+1/k})` | 冻结 |
| 区间条件 flow map | 学每步映射 | `x_t, k` → `x_{t+1/k}` | 共享 LoRA |
| 风险-收益调度器 | 选最小预算 | 一步预览 → 难度/保真度 | 轻量网络 |

**训练伪代码**
```
for (obs, a0) in data:
    traj = teacher.rollout(obs)                       # 冻结教师轨迹
    for k in budgets:
        for (x_t, x_next) in subsample(traj, k):
            loss = mse(flow_map(x_t, k), x_next)      # 区间条件
        update(flow_map, LoRA)
# 推理
preview = student.one_step(obs)
k = scheduler(preview)                                # 选最小满足风险
a = student.sample(obs, steps=k)
```
**坑**：摘要未给调度器输入细节；「一步预览」本身有成本，要算进延迟。

---

## 9. WAM-OPD / HINT-WAM / WAM-Policy（L1）——在线蒸馏家族

共同点：教师监督发生在**学生自己的轨迹**上，而不是固定示教数据。

**WAM-OPD**：固定轨迹池 + 每次重新采样去噪路径 + 前缀重要性权重。
```
pool = initial_student_rollouts + teacher_rollouts
for traj in pool:
    for t in decisions(traj):
        hist = traj.history[:t]
        path = student.denoise(hist)                  # 新路径
        L_t  = KL(teacher(hist, privileged) || student(hist))
        update(student, w_t * L_t)                    # w_t 来自前缀分数
```
**HINT-WAM**：同一模型 + hindsight 未来视频当教师；energy score 取代速度回归。
```
a_s = student.sample(obs)                             # 学生动作
p_t = student.sample(obs, cond=hindsight_video)       # 教师分布
L   = energy_score(p_t, a_s)
```
**WAM-Policy**：OPSD（一视频+两动作步）+ 多教师 M-OPD；四足。
```
# 阶段1：教师 WAM → 学生少步（OPSD）
# 阶段2：任务 RL 教师 + M-OPD 合并多专家
```
**坑**：三者都需要「教师可查询」；HINT 的教师=学生同权重+特权输入，注意不要把特权信息泄漏进部署路径。

---

## 10. LoopedWAM（L1）——弹性 checkpoint

**角色**：单模型多预算（1/2/4 步 × 不同 action-depth）。
**模块**：共享 6-block core 循环（30 层只存 12 block）；视频专家每 query 算一次，动作专家每采样步重复；elastic 自蒸馏跨预算。
```
for (depth, steps) in budgets:
    y = loop(shared_core, depth)
    loss = distill(teacher_30step, y, steps=steps)
```
**坑**：摘要未给 elastic 的训练采样策略；「0.58B 达到 96.1%」意味着循环深度是关键变量。

---

## 11. GR00T N1 数据蒸馏管线（L3）——附：伪动作生成

```
# 1) 潜动作（VQ-VAE）
train(VQVAE, pairs (x_t, x_{t+H}) from human/robot video)
z = VQVAE.encode(x_t, x_{t+H})          # 当作 "LAPA" embodiment 的标签
# 2) 神经轨迹（视频生成）
fine_tune(image2video, 88h teleop); generate 827h with novel prompts
# 3) IDM 伪动作
train(IDM, real_robot_pairs); a_hat = IDM(x_t, x_{t+H})
# 4) 混合训练
for batch in mix(real, latent_action, neural_traj):
    loss = flow_matching(a_or_z, cond=obs, instr)
```
**坑**：IDM 只在低数据上训（防止泄漏）；后训练真机:神经轨迹=1:1。

---

## 12. 实现检查清单（跨方法）

1. **教师何时跑**：CKT（每 batch 一次）/ THAW（训前离线一次）/ S-VAM（每样本多步）/ DreamFree、W2A（训练期）。
2. **学生是否冻结**：CKT 冻结；THAW/S-VAM/DreamFree/W2A 学生可训。
3. **部署是否带教师**：除 CKT 外都不带；CKT 推理仍需教师一次前向。
4. **形状对齐点**：CKT 的 d_stu；THAW 的 4096-D；S-VAM 的 VFM 通道。
5. **缓存一致性**：THAW 的 (traj,idx) 键；S-VAM 的 obs→Y_ref 同步。
6. **数值稳定**：CKT 的 top-k renorm；APM 的梯度投影；S-VAM 的 L2 归一化。
