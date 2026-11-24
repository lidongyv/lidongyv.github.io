# 06 · On-policy 蒸馏与经验蒸馏：2026 下半年的主战场

> 一句话：**离线蒸馏教学生"在教师轨迹上怎么做"，on-policy 蒸馏教学生"在你自己搞砸的地方怎么补救"。**
> 这也是最近三个月（2026-08~11）新论文最密集的方向：VLA-MOPD、FastOPD、REPAIR、TAD、TILDE、WAM-Policy、OPTED、Reflect-VLA 全部落在这一格。
> **每个方法的训练循环见本章 §10（FastOPD / VLA-MOPD / WAM-Policy / TAD / TILDE / H2E / REPAIR / VAD / OPTED / Reflect）。**

---

## 1. 为什么离线蒸馏不够

离线 KD 的致命问题是 **exposure bias / 分布偏移**：

- 训练时学生看到的状态来自教师轨迹；
- 推理时学生自己跑，一旦偏一点就进入"教师从没见过的状态"，误差复合增长；
- 学生越小，偏得越快。

DAgger（1011.0686）的解法是：**让学生自己跑，教师在学生到达的状态上给标签**。代价是每步都要教师前向。VLA 时代这个成本被两个东西摊薄了：

1. 教师可以**批量并行**在仿真里打分（GPU 农场）；
2. 教师查询可以**降采样**（每 N 步查一次）+ 离线缓存。

---

## 2. 四种 on-policy 形态（按教师查询方式分）

| 形态 | 教师给什么 | 查询成本 | 代表 |
|---|---|---|---|
| **A. 动作监督** | 学生状态下教师的动作/速度场 | 每步一次教师前向 | FastOPD、VLA-MOPD、WAM-Policy |
| **B. 轨迹级监督** | 学生轨迹与教师轨迹的分布差异（判别器/reward） | 只需教师轨迹库 | TAD |
| **C. 恢复数据** | 教师从学生失败状态"救回来"的轨迹 | 稀疏（触发式） | TILDE |
| **D. 经验筛选** | 只保留成功的 session | 零教师查询（环境当裁判） | H2E、Verify-Act-Distill、REPAIR |

---

## 3. A 类：逐步动作监督

### FastOPD（2610.02832，L2 部分正文）

- 定位：on-policy 蒸馏 + 步数压缩二合一；**教师只查一个点**（学生自己跳到的状态），而不是逐去噪步查。
- 教师：π0.5 / LingBot-VLA / MolmoAct2 / WAM，**3B–6B**（比学生大 **>7.3×**）；学生：**SmolVLA 450M**（**冻结 VLM 主干 + 只训轻量 action expert**）。
- 损失（原文 Eq.3）：

```
L(θ) = E[ ||u_θ(x_s,s,t) − ½(sg[v_θ(x_s,s)] + sg[v_θ(x̂_m,m)])||² ]      # 有限区间自一致
     + λ·E[ ||v_θ(x̃_t,t) − v_φ(x̃_t,t)||² ]                              # on-policy 速度蒸馏
```

  第一项把"直跳 s→t"对齐"经中点 m 的两段短跳"（stop-grad）；第二项的采样点 `x̃_t = sg[f_θ(x_1,1,t)]` 是学生自己的一步落点。
- 数字：LIBERO 2 步保留 **84%**，延迟 **−78.1%**；RoboTwin2.0 单步 **+15.9pp**；同水平下比常规 OPD **快 5.7×**。
- 关键：它把"在哪查教师"（学生自己的状态）与"蒸什么"（速度场）解耦，是当前最完整的 OPD 配方之一。

### VLA-MOPD（2611.12286，L1）

- 问题：多个任务专家（每个一个 flow VLA）→ 合并成一个共享 VLA，且不能掉点。
- 做法：共享学生在**任务 rollout 中实际访问的观测 + 每个 ODE 中间状态**上，向 instruction 路由的冻结教师查询；对每个 flow step 的 transition 做 **masked local matching**（目标 detach，只在有效 action chunk 维度上算损失）。
- 数字：LIBERO mean success-once **75.4%→98.9%**（端点动作蒸馏基线 98.15%）；RoboTwin 2.0 **73.0%→89.3%**；Piper 真机两轮 **70.0%→90.0%**。
- 消融（可确认 1 条）：加 reference-policy 项把 Object suite 的降幅从 **13.2 点限制到 3.0 点**（缓解遗忘）。
- 为什么重要：多专家合并（model merging 的替代）在 VLA 上第一次做到"ODE 轨迹级"对齐，而不是权重平均。

### WAM-Policy（2611.24011，L1）

- 场景：四足 loco-manipulation；教师=臂部中心预训练的 WAM → 四足学生。
- 两阶段：**OPSD 自蒸馏**（先压采样步数：视频 1 步 + 动作 2 步）→ **RL 调教师 + 多教师 OPD**。
- 数字：迁移 **77.0%**；OPSD 学生 **76.7%** 且采样快 **14×**；RL+M-OPD **85.0%**（QLM-Bench）。
- 启示：WAM 的 on-policy 蒸馏不只是"学动作"，还要先解决**跨 embodiment 的观测/动作空间对齐**。

---

## 4. B 类：只用教师轨迹库（黑盒教师）

### TAD（2610.33650，L1）

- 设定：教师只给**轨迹库**（joint-position 特征），不给动作分数、不给在线查询——最苛刻的黑盒接口。
- 做法：判别器比较教师/学生 rollout 的特征**边缘分布**，产出 reward，做 on-policy 学习。
- 数字：PutOnPlateInScene25Main **26.2%→76.6%** success-once（+50.4 点），success-end 20.3%→58.6%。
- **但**：仍低于白盒 OPD 的 **81.6%/59.4%**，且花了 **5× 交互**、**12.5× batch-passes**。
- 价值：它给出了"黑盒教师蒸馏的预算上限"——**接口越弱，样本效率越差**，这是综述里必须引的反例。

---

## 5. C 类：恢复数据（DAgger 的现代版）

### TILDE（2610.25376，L1）

- 教师：预训练 EFM（embodied foundation model）→ 学生：未做动作预训练的策略。
- 做法：① 在学生**诱导状态**上给教师监督；② 收集**教师恢复轨迹**；③ AutoDAgger 用 reward model 触发干预（只在学生明显要失败时叫教师）。
- 数字：三个真机任务最高 **4.1×** 于 behavior cloning；恢复轨迹采集比 human-in-the-loop 快 **>2×**。
- 启示：on-policy 蒸馏的成本控制关键是**"什么时候叫教师"**，而不是"叫多少次"。

---

## 6. D 类：经验筛选（环境当裁判）

### H2E（2611.16246，L3 部分）

- 教师：frontier LLM planner（gpt-6-astra + place graph + 工具）调用**冻结 8B NavGPT** 组成的 agent；学生：同一个 8B VLA，只训 **LoRA + waypoint head**，视觉编码器与 LLM 冻结。
- 做法：只保留到达目标的 session，把 planner 的子指令/工具移动/停止决策**重标为 VLA 动作**；远离目标的步 mask 掉（`d(p_{t+K}) ≤ d(p_t)+δ`，**δ=0.25m**）；距目标 `r_arr=1m` 内加停机标签（`y_t=0`）；数据混合 **0.60 agent : 0.25 DART : 0.15 R2R-CE**。
- 规模：2395 条成功 session / 57 场景，**332,405** 带标签步、**30,181** 停标签；3000 updates（48,000 samples）。
- 数字：REVERIE-CE val-unseen **35.0%→44.9%**（+9.9）；v3 报 8B VLA **+15.6%/+11.1%**（REVERIE/NavRAG）；零 LLM token，比 agent 快 **3.7×**；100 dev episodes 上 Base 38/59 → H2E **53/57**。
- 消融：去掉 task-level labels 53→47；去掉 progress masking 53→47；改 geodesic targets 46；改 self-imitation 47。
- 价值：证明"**调用 VLA 的 agent 本身可以当教师**"——不需要教师和被蒸馏者同构。

### REPAIR（2611.02777，L1）

- 教师：轻量 RL 专家 + critic；学生：单个 flow-matching VLA。
- 做法：关键阶段采多专家动作候选，用 **critic 加权**成监督分布；离线初始化 + on-policy 自采集。
- 数字：六个真机任务各 30 trials **全部 100%**；末端 jerk 均低于专家辅助执行。
- 价值：把"RL 专家只是补丁"变成"可部署的单一 VLA"；真机 100% 是很强的证据。

### Verify-Act-Distill（2610.14885，L3 部分）

- 教师：固定 verifier 给 flow VLA 的多候选打分（**N=8**，softmax 温度 **λ=0.3**、候选噪声 **α=1.2**）；学生：同一 VLA 的 **LoRA（rank 8, scaling 16, 0.37M 可训）**。
- 做法：成功适应回合的观测 + **聚合** action chunk，用**标准 flow-matching loss** 蒸进 LoRA：`L=E||v_{θ0,φ}(A_τ,τ|o)−(ε−Ā)||²`，`A_τ=(1−τ)Ā+τε`；无 RL、无额外 rollout。verifier 是 6 层 Transformer（hidden 512，60k 步）。
- 训练：LIBERO 40 任务 × 每任务 10 episodes；LoRA 200 步、lr 1e-4、batch 8。
- 数字：SmolVLA 5k 单候选 **54.1%→70.6%**（保留约 74% 的 8 候选增益）；聚合 8 候选 76.4%→**82.3%**；LIBERO-Long **44.0%→57.8%**；10k ckpt 83.2→86.2。
- 价值：把 test-time compute 的增益**写回权重**，是"oracle/Best-of-N 蒸馏"最干净的数字。

---

## 7. 驾驶侧的 on-policy

### OPTED（2609.20756，L1）

- 教师：向量化输入（HD 图 + 框）的 RL 特权教师；学生：相机输入（TransFuser/VaVAM）。
- 做法：闭环 on-policy 经验蒸馏（AlpaSim + 3DGS 在线采样）。
- 数字：DS 提升 **1.6× 与 9.5×**；同等闭环性能少约 **3 个数量级**仿真交互。

### Reflect-VLA（2610.13453，L1）

- 教师：同体渐进教师（Reflective Guidance 作为训练期特权）→ 单次前向 VLA。
- 做法：token 级纠正分布蒸馏；**风险轨迹蒸教师、可靠轨迹自模仿**（双路径）。
- 数字：训练时间较 RL 减少 **90%+**；NAVSIM v1/v2、Bench2Drive（摘要未给分值）。

---

## 8. On-policy 蒸馏的五个工程决定

1. **教师查询频率**：每步查 → 最准但最贵；每 N 步 / 只在"关键阶段"查（REPAIR 用 critic 判断）是主流。
2. **教师是否可并行**：仿真里可以开 1000 个环境并行查教师；真机不行，只能离线缓存 + 少量在线。
3. **mask 策略**：H2E 把"远离目标"的步 mask 掉；VLA-MOPD 对 ODE 每一步做 masked local matching。**mask 比 loss 权重更有效**。
4. **是否混离线数据**：纯 on-policy 会遗忘预训练能力，通常混 30–50% 离线数据。
5. **教师误差放大**：on-policy 下学生会模仿教师的错误并叠加自己的错误；用 critic/verifier 过滤比直接用教师动作稳。

---

## 9. 一句话选型

| 你的约束 | 选 |
|---|---|
| 教师可以进训练循环、要最高上限 | A 类（FastOPD / VLA-MOPD） |
| 教师只能给轨迹、不能查 | B 类（TAD），但接受样本效率差 5–12× |
| 真机、教师调用贵 | C 类（TILDE，触发式恢复） |
| 有环境/verifier 当裁判 | D 类（H2E / REPAIR / Verify-Act-Distill） |

## 10. 实现代码（逐方法）

> 只给"on-policy 那一段"：学生怎么跑、教师在哪查、损失怎么算。完整训练循环见第 12 章。

### 10.1 FastOPD（2610.02832）——A 类

**物理意义（解释）**：在学生落点查教师：只补学生真正会到的地方，训练成本比逐步 OPD 低。

```python
for batch in loader:
    t   = torch.rand(B, 1) * 0.9 + 0.05
    x_t = student.jump(batch.x1, s=1.0, t=t, cond=batch.cond).detach()
    L_opd = F.mse_loss(student.velocity(x_t, t), teacher.velocity(x_t, t))
    # 另加有限区间自一致项 L_sc（见 04 章 §10.7）
    (L_sc + lam * L_opd).backward(); optimizer.step()
```

### 10.2 VLA-MOPD（2611.12286）——A 类

**物理意义（解释）**：在学生的每个 ODE 中间状态查对应专家：不只学终点，还学「中间该怎么走」。

```python
for task in tasks:
    traj = rollout(student, task)                 # 学生自己跑
    teacher = teachers[route(task.instruction)]   # 路由冻结专家
    for obs, act in traj:
        x = torch.randn_like(act)
        for k in range(K):                        # 逐 ODE 中间状态
            with torch.no_grad():
                tgt = transition(teacher, obs, x, k, K)
            loss += masked_match(transition(student, obs, x, k, K), tgt, valid_dims(act))
            x = x + tgt / K
    loss += beta * ref_kl(student, ref, task)     # 防遗忘
```

### 10.3 WAM-Policy（2611.24011）——A 类

**物理意义（解释）**：学生自己采样，教师高 NFE 端点当目标；plan stop-grad，动作以 plan 为条件。

```python
Z_hat = student.sample_video(eps_Z, steps=1)
A_hat = student.sample_action(eps_A, Z_hat.detach(), steps=2)
with torch.no_grad():
    Z_T = teacher.sample_video(eps_Z, steps=K_T)
    A_T = teacher.sample_action(eps_A, Z_T, steps=K_T)
loss = pseudo_huber(Z_hat - Z_T) + lam_A * pseudo_huber(A_hat - A_T)
```

### 10.4 TAD（2610.33650）——B 类

**物理意义（解释）**：判别器当 reward，比较学生/教师 rollout 的特征分布；只有轨迹库也能学行为风格。

```python
rollout = student_rollout(student, env)          # 学生自己跑
f_s = student_features(rollout)
f_t = sample_teacher_features()                  # 教师轨迹库（黑盒）
reward = discriminator(f_s, f_t)
loss = -reward.mean() + bc_loss(student, rollout)
```

### 10.5 TILDE（2610.25376）——C 类

**物理意义（解释）**：失败前叫教师接管，收集恢复轨迹；把「师傅扶一把」变成数据。

```python
for obs in student_rollout:
    if reward_model(obs) < thr:                  # 触发式干预
        with torch.no_grad():
            D_recovery.append((obs, teacher(obs)))
loss = bc(student, D_recovery) + bc(student, D_online)
```

### 10.6 H2E（2611.16246）——D 类

**物理意义（解释）**：agent 成功 session 重标为 VLA 动作；只学能到达目标的轨迹，测试时零 LLM。

```python
for s in successful_sessions:
    for t, step in enumerate(s):
        y = relabel(step)                        # 所有运动重标为 VLA 动作
        if dist(s.p[t], goal) <= 1.0: y = STOP
        m = float(dist(s.p[t+K], goal) <= dist(s.p[t], goal) + 0.25)
        loss += m * F.l1_loss(student(step.obs), y)
```

### 10.7 REPAIR（2611.02777）——D 类

**物理意义（解释）**：critic 给多专家候选加权成监督分布；把偏好蒸进单个 VLA。

```python
with torch.no_grad():
    cands = torch.stack([e(obs) for e in experts])   # 多专家候选
    w = critic(obs, cands).softmax(-1)
    target = (w[..., None, None] * cands).sum(0)
loss = flow_matching(student(obs), target)
```

### 10.8 Verify-Act-Distill（2610.14885）——D 类

**物理意义（解释）**：verifier 选优→聚合→只留成功 episode→蒸进 LoRA；把测试时搜索的收益写回权重。

```python
for b in range(replan_steps):
    cands  = [student.sample(obs) for _ in range(8)]
    scores = verifier(obs, [c[:10] for c in cands])
    A_bar  = sum(torch.softmax(scores/0.3, 0)[i] * cands[i] for i in range(8))
    env.execute([c[:10] for c in cands])
    if env.success: D.append((obs, A_bar))       # 只留成功
loss = flow_matching_lora(student, D)
```

### 10.9 OPTED（2609.20756）——A 类（驾驶）

**物理意义（解释）**：特权教师（HD 图/框）在闭环里给学生相机动作；样本效率来自「在环」而不是离线数据。

```python
obs, priv = env.step()
with torch.no_grad():
    a_t = teacher(priv)                          # 向量化 RL 特权教师
loss = bc_loss(student(obs), a_t)                # 闭环 on-policy
```

### 10.10 Reflect-VLA（2610.13453）——A 类（驾驶）

**物理意义（解释）**：风险轨迹学教师纠正分布，可靠轨迹自模仿；把反思内化。

```python
risk = risk_score(traj)
if risk > thr:
    with torch.no_grad():
        p_t = teacher.token_dist(obs, traj)      # token 级纠正
    loss = kl_div(student.logits(obs), p_t)
else:
    loss = ce_loss(student.logits(obs), traj)    # 自模仿
```

---

更新时间: 2026-11-21
