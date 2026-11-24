# 02 · VLA 蒸馏：每个方法的实现、数字与坑

> VLA = Vision-Language-Action。蒸馏它有两种完全不同的动机：
> **驾驶**——把 7B 级 LLM 的车端推理省掉（延迟/显存是硬约束）；
> **操作**——把 flow/diffusion 动作头的 NFE 压下来（控制频率是硬约束）。
> 这一版把**每个方法的实现代码**放进对应小节（输入/输出 + 核心伪代码），不再只写"做了什么、结果多少"。
> 深度模块拆解（LQCA、decoder 内部形状、训练循环）见第 12 章。

---

## 1. 全景：VLA 蒸馏的 8 种做法

| # | 做法 | 代表 | 教师 | 学生 | 部署带走什么 |
|---|---|---|---|---|---|
| 1 | latent planning token 回归 | **Orion-Lite** (2604.08266) | 7B LLM | 0.1B decoder | 只有学生 |
| 2 | latent 推理特征 + 可摘 LM | **DRIFT** (2611.24719) | <150M LM | 255M 规划器 | 只有规划器 |
| 3 | 特权/未来观测蒸馏 | **FPD-Drive** (2610.13232)、**OPTED** (2609.20756) | 特权教师 | 纯视觉规划器 | 只有学生 |
| 4 | 多教师特征蒸馏 | **ConDenseDrive** (2610.16908) | 3 个 VFM | 单一视觉编码器 | 只有编码器 |
| 5 | 同体渐进 + token 级纠正 | **Reflect-VLA** (2610.13453) | 自身渐进教师 | 单次前向 VLA | 只有学生 |
| 6 | 动作头步数蒸馏 | **SnapFlow** (2604.05656)、**π0-FAST**、**FastOPD** (2610.02832) | 10-NFE / flow 教师 | 1-NFE | 只有学生 |
| 7 | 多专家策略合并（OPD） | **VLA-MOPD** (2611.12286)、**REPAIR** (2611.02777) | 多任务/RL 专家 | 单一 VLA | 只有学生 |
| 8 | 剪枝 + 离线隐层蒸馏 | **2609.19579** | OpenVLA-OFT 7B | 宽度剪枝 63–72% | 剪枝后的学生 |

---

## 2. 驾驶线：省掉推理期的 LLM

### 2.1 Orion-Lite（2604.08266，L3）——最硬的证据

**物理意义（解释）**：蒸的是教师 LLM 对当前场景形成的**规划意图**——不是动作，而是「看到这个场景，老司机心里已经有的路线倾向」。学生用纯视觉重建这个意图，反而去掉了文本 prompt 与长 CoT 带来的推理噪声，所以能反超教师。

**教师结构**：ORION = **EVA-02-L/14 视觉塔 + Vicuna-v1.5 7B LLM + VAE 生成式规划器**。
EVA-02-L/14 是 ViT-L/14 规格（**24 层、1024 宽、16 头、patch 14、304M 参数**；SwiGLU + RoPE；224 预训练、448 微调；层数/宽度按 ViT-L/14 惯例，具体以 checkpoint config 为准；预训练方式是"从 CLIP 视觉编码器做掩码图像建模蒸馏"，见 EVA-02 论文 2303.11331 / 官方 README）。

**实现**（输入 `images [B,V,3,224,224]`、`text [B,T]`、`gt_traj [B,H,3]`；输出 `loss: scalar`）

```python
# ---- 教师：EVA-02-L/14 + Vicuna-7B，冻结，只取 planning token ----
with torch.no_grad():
    T_p = teacher.planning_token(images, text)        # [B,1,C_p]

# ---- 学生：同款视觉塔（冻结）+ 0.1B、6 层 decoder（纯视觉，丢文本） ----
vis = student.vision(images)                          # [B,Nv,1024]
T_s = student.decoder(vis)                            # [B,1,C_p]
pred = student.planner(T_s, vis)                      # [B,H,3]

# ---- 两路损失 ----
L_mimic = (T_s - T_p).abs().mean()                    # 逐元素 L1
L_gt    = traj_losses(pred, gt_traj)                  # col + bd + reg + vae
loss    = L_gt + L_mimic
```

**L1 和 KL 的实现区别**（这是最常被问的问题）

```python
# L1：逐元素比较两个连续向量，要求形状完全一致
L1 = (T_s - T_p).abs().mean()

# KL：比较两个"分布"，先 log_softmax/softmax，再沿类别维求和
p  = F.log_softmax(T_s / T, dim=-1)                   # 学生 log-prob
q  = F.softmax(T_p / T, dim=-1)                       # 教师 prob
KL = F.kl_div(p, q, reduction="batchmean") * T * T    # 温度缩放要乘回 T²
```

| 维度 | L1 | KL |
|---|---|---|
| 输入形状 | 两个**同形状**张量 | 两个**同形状**张量，但最后一维被当作"类别" |
| 是否需要 softmax | 不需要 | 必须（否则不是分布） |
| 语义 | 向量回归，逐元素 | 分布对齐，沿类别求和 |
| 梯度 | ±1，对误差大小不敏感 | 与概率差成正比，教师越自信权重越大 |
| 对称性 | 对称 | **不对称**（`KL(p‖q) ≠ KL(q‖p)`） |
| 数值 | 稳定 | 需要 `log_softmax`，否则 `-inf` |

所以"不都还是 L1 嘛"不成立：**L1 是逐元素绝对值，KL 是分布间的求和**。Orion-Lite 用 L1，是因为 latent planning token 是连续表征、没有概率含义；DiMA 用 KL，是把通道维当分布，两者不能互换（作者原文明确说 KL 是 "suboptimal distributional metrics"）。

**KL 也不是一种实现：KD 的 KL 与 VAE 的 KL 是两个东西**

```python
# (a) KD 的 KL：两个 categorical 分布（logits → softmax），沿类别维求和
p = F.log_softmax(student_logits / T, -1)
q = F.softmax(teacher_logits / T, -1)
KL_cat = F.kl_div(p, q, reduction="batchmean") * T * T

# (b) VAE 的 KL：对角高斯后验 vs N(0, I)，闭式，逐维求和
def kl_gaussian(mu, logvar):
    # 输入: mu [B,D], logvar [B,D]；输出: [B]
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)

# (c) 用 torch.distributions 交叉验证（结果应与 (b) 一致）
KL_gauss = torch.distributions.kl_divergence(
    torch.distributions.Normal(mu, (0.5 * logvar).exp()),
    torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
).sum(-1)
```

| | KD 的 KL | VAE 的 KL |
|---|---|---|
| 分布族 | categorical（softmax 后） | diagonal Gaussian |
| 输入 | logits `[B,C]` | `mu, logvar [B,D]` |
| 计算 | `Σ_c q·log(q/p)` | `−0.5·Σ_d (1 + logvar − mu² − exp(logvar))` |
| 是否采样 | 否 | 否（闭式） |
| 作用 | 对齐教师/学生输出分布 | 把潜变量正则到先验 |
| 本工程出现处 | DiMA、Reflect-VLA、VLA-MOPD 的 ref KL | Orion-Lite 的 `L_vae`（VAE 规划器损失项，通常含高斯 KL；原文 §3.4 未逐项展开） |

看到 "KL" 要先问**是哪个分布族**：KD 的 KL 是类别分布（要 softmax）；VAE 的 KL 是高斯（闭式、对 `mu/logvar` 算）；DMD 里的 KL 则是用 score 差近似，连闭式都没有。

**结果**：Bench2Drive DS **80.6**（教师 77.7）；推理模块 150×；端到端 806→267ms；显存 31→8GB。

---

### 2.2 DRIFT（2611.24719，L1）——把 LM 做成"可摘除的推理支架"

**物理意义（解释）**：把「想清楚再开」的推理过程压成一个 latent 特征，让 255M 规划器「看一眼就有数」。物理上蒸的是**决策前的沉默思考**，不是决策结果。

**实现**（输入 `images`、`text`、`gt_traj`；输出 `loss`）

```python
# 教师：轻量 LM（<150M）出 latent reasoning feature
with torch.no_grad():
    R_t = lm_teacher(images, text)                    # [B,1,Cr]

# 学生：255M 规划器直接出"推理特征 + 轨迹"
R_s, traj = planner(images)                           # R_s [B,1,Cr], traj [B,H,3]

loss = traj_loss(traj, gt_traj) + (R_s - R_t).abs().mean()
# 推理：丢掉 lm_teacher，只跑 planner（255M）
```

**结果**：Bench2Drive DS **97.03** / SR **89.39%**（三次均值），255M 参数。

---

### 2.3 FPD-Drive（2610.13232，L1）——未来观测特权教师

**物理意义（解释）**：教师能偷看未来几帧，学生不能；蒸的是「接下来别的车/人会往哪走」的**未来轨迹先验**，让学生从当前帧推断动态。

**实现**（输入 `obs`、`future_obs`（仅训练期）、`gt_traj`；输出 `loss`）

```python
# 教师能看到"未来观测"，学生只能看当前帧
with torch.no_grad():
    y_t = teacher(obs, future_obs)                    # [B,M,T,2] 多目标未来轨迹
y_s = student(obs)                                    # 纯视觉规划器
loss = traj_loss(y_s, gt_traj) + (y_s - y_t).abs().mean()
# 部署：学生不需要 future_obs
```

**结果**：NAVSIM v1 **94.9 PDMS**；v2 navhard 55.9 EPDMS；HUGSIM HD-Score 45.1。

---

### 2.4 OPTED（2609.20756，L1）——闭环 on-policy 经验蒸馏

**物理意义（解释）**：蒸的是**闭环交互经验**——不是单帧动作，而是「在环里开错之后怎么修」的行为分布；所以同等性能需要的仿真交互少三个数量级。

**实现**（输入 `obs`（相机）、`priv`（HD 图 + 框，仅教师）；输出 `loss`）

```python
for step in range(N):
    obs, priv, done = env.step()
    with torch.no_grad():
        a_t = teacher(priv)                           # 向量化 RL 特权教师
    a_s = student(obs)                                # 相机学生
    loss = bc_loss(a_s, a_t)                          # 闭环 on-policy 蒸馏
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

**结果**：DS 提升 **1.6× / 9.5×**；同等闭环性能少约 **3 个数量级**仿真交互。

---

### 2.5 Reflect-VLA（2610.13453，L1）——双路径自蒸馏

**物理意义（解释）**：风险轨迹上学教师的**纠错意见**（token 级纠正分布），可靠轨迹上自模仿。物理上=把「事后复盘」内化进单次前向。

**实现**（输入 `obs`、`traj`、`risk`；输出 `loss`）

```python
risk = risk_score(traj)                               # 轨迹风险分
if risk > thr:                                        # 风险轨迹 → 学教师纠正
    with torch.no_grad():
        p_t = teacher.token_dist(obs, traj)           # token 级纠正分布
    loss = kl_div(student.logits(obs), p_t)
else:                                                 # 可靠轨迹 → 自模仿
    loss = ce_loss(student.logits(obs), traj)
```

**结果**：训练时间较 RL 减少 **90%+**；NAVSIM v1/v2、Bench2Drive（摘要未给分值）。

---

### 2.6 ConDenseDrive（2610.16908，L1）——多教师特征蒸馏

**物理意义（解释）**：把视觉/语义/几何三个基础模型的「看法」压进一个编码器。物理上=让一个学生同时继承三双眼睛，且推理时只付一双眼睛的钱。

**实现**（输入 `img`；输出 `loss`）

```python
with torch.no_grad():
    f_v = vfm_visual(img)                             # 视觉 VFM
    f_s = vfm_semantic(img)                           # 语义 VFM
    f_g = vfm_geometric(img)                          # 几何 VFM
f = encoder(img)                                      # 学生单编码器
target = torch.cat([f_v, f_s, f_g], dim=-1).detach()
loss = F.mse_loss(f, target) + task_loss(planner(f), gt)
```

**结果**：**2.0×** 加速；NAVSIM v1 **94.5 PDMS**、v2 90.8 EPDMS。

---

### 2.7 数据蒸馏（驾驶，见 07 章）

AutoVLA（Qwen2.5-VL-72B → 3B）、Senna（GPT-4o）、DriveGPT4（ChatGPT）、OmniDrive（GPT-4）、LMDrive（指令改写）。

---

## 3. 操作线：压动作头的 NFE

### 3.1 π0-FAST（PI 官方，无 arXiv，L2）

**物理意义（解释）**：把连续动作的「积分求解」换成离散 token 的「逐字写」。物理上=把连续控制翻译成语言，用自回归换掉 ODE 的串行开销。

**实现**（输入 `obs`；输出 `tokens`）

```python
# 教师：π0 flow 头多步采样出动作
with torch.no_grad():
    a = pi0_flow.sample(obs, steps=10)                # [B,H,D]
    tokens = fast_tokenizer.encode(a)                 # 离散动作 token
# 学生：FAST 自回归专家
loss = F.cross_entropy(ar_expert(obs).logits, tokens)
```

**结果**：推理约 **5×**（官方博客 + openpi 代码）。

---

### 3.2 SnapFlow（2604.05656，L3）

**物理意义（解释）**：蒸的是「自己走两步的终点」。物理上=让学生学会一步到位；反超教师是因为多步积分的离散化误差被消掉了。

**实现**（输入 `x0 [B,H,D]`、`cond`；输出 `loss`）

```python
t   = torch.rand(B, 1)
eps = torch.randn_like(x0)
x_t = (1 - t) * x0 + t * eps

# 分支 1：标准 flow matching
L_fm = F.mse_loss(model(x_t, t, t_dst=t, cond=cond), eps - x0)

# 分支 2：自己的两步 Euler 端点（整链 stop-grad）
with torch.no_grad():
    t_mid = 0.5 * t
    x_mid = x_t + (t_mid - t) * model(x_t, t, t_dst=t_mid, cond=cond)
    x_dst = x_mid + (0 - t_mid) * model(x_mid, t_mid, t_dst=0, cond=cond)
    v_tgt = (x_dst - x_t) / (0 - t)
L_sc = F.mse_loss(model(x_t, t, t_dst=0, cond=cond), v_tgt)

loss = L_fm + lam * L_sc
# 推理：a0 = a1 - model(a1, t=1, t_dst=0, cond)   1 NFE
```

**结果**：LIBERO **98.75% vs 教师 97.75%**；去噪 9.6×、端到端 274→83ms。

---

### 3.3 FastOPD（2610.02832，L2）

**物理意义（解释）**：在学生**自己会到达的状态**上问教师「这里该往哪走」。物理上=补课补在学生真正卡住的地方，而不是教师轨迹上。

**实现**（输入 `x1 [B,H,D]`、`cond`；输出 `loss`）

```python
# S2 学生落点（教师只查这一个点）
t   = torch.rand(B, 1) * 0.9 + 0.05
x_t = student.jump(x1, s=1.0, t=t, cond=cond).detach()

# S3 on-policy 速度蒸馏
L_opd = F.mse_loss(student.velocity(x_t, t, cond),
                   teacher.velocity(x_t, t, cond))

# S4 有限区间自一致
s   = t * torch.rand(B, 1)
m   = 0.5 * (s + t)
x_s = student.jump(x1, s=1.0, t=s, cond=cond).detach()
with torch.no_grad():
    v_s = student.velocity(x_s, s, cond)
    v_m = student.velocity(x_s + (m - s) * v_s, m, cond)
    tgt = 0.5 * (v_s + v_m)
L_sc = F.mse_loss(student.interval_vel(x_s, s, t, cond), tgt)

loss = L_sc + cfg.lam * L_opd
```

**结果**：2 步保留 **84%**、延迟 **−78.1%**；RoboTwin 单步 +15.9pp。

---

### 3.4 VLA-MOPD（2611.12286，L1）

**物理意义（解释）**：多个任务专家各有各的「手感」；蒸的是每个 ODE 中间状态的动作走向。物理上=把几套手感合并成一套，且中间过程也要对。

**实现**（输入 `task`、`student`；输出 `loss`）

```python
traj = rollout(student, task)                         # 学生自采样
teacher = teachers[route(task.instruction)]           # 冻结专家路由
loss = 0
for obs, act in traj:
    x = torch.randn_like(act)
    for k in range(K):                                # 每个 ODE 中间状态
        with torch.no_grad():
            tgt = transition(teacher, obs, x, k, K)
        mask = valid_dims(act)
        loss += masked_match(transition(student, obs, x, k, K), tgt, mask)
        x = x + tgt / K
loss += beta * ref_kl(student, ref, task)             # 防遗忘
```

**结果**：LIBERO 75.4→**98.9%**；RoboTwin 73.0→89.3%；Piper 70→90%。

---

### 3.5 REPAIR（2611.02777，L1）

**物理意义（解释）**：RL 专家 + critic 给「哪个动作更好」的加权。物理上=把多个师傅的偏好蒸成一个徒弟，真机上末端更平顺。

**实现**（输入 `obs`；输出 `loss`）

```python
with torch.no_grad():
    cands = torch.stack([e(obs) for e in experts])    # [E,B,H,D]
    w = critic(obs, cands).softmax(-1)                # critic 加权
    target = (w[..., None, None] * cands).sum(0)
loss = flow_matching(student(obs), target)            # 单 VLA
```

**结果**：六个真机任务各 30 trials **100%**；末端 jerk 低于专家辅助执行。

---

### 3.6 TAD（2610.33650，L1）——黑盒教师

**物理意义（解释）**：教师只给轨迹库，判别器判断「学生跑得像不像教师」。物理上=模仿教师的**行为风格**，而不是逐帧动作。

**实现**（输入 `rollout`、`teacher_feats`；输出 `loss`）

```python
f_s = student_features(rollout)                       # 学生 rollout 特征
f_t = sample_teacher_features()                       # 教师轨迹库特征
reward = discriminator(f_s, f_t)                      # 判别器给 reward
loss = -reward.mean() + bc_loss(student, rollout)     # on-policy 学习
```

**结果**：26.2→**76.6%**，但白盒 OPD 81.6%；花 5× 交互、12.5× batch-passes。

---

### 3.7 TILDE（2610.25376，L1）

**物理意义（解释）**：学生快失败时叫教师接管，收集「救回来」的轨迹。物理上=把「师傅扶一把」的经验变成训练数据。

**实现**（输入 `obs`、`reward_model`；输出 `loss`）

```python
for obs in student_rollout:
    r = reward_model(obs)
    if r < thr:                                       # AutoDAgger 触发
        with torch.no_grad():
            a_t = teacher(obs)
        D_recovery.append((obs, a_t))                 # 教师恢复轨迹
loss = bc(student, D_recovery) + bc(student, D_online)
```

**结果**：三个真机任务最高 **4.1×** 于 behavior cloning；恢复轨迹采集快 **>2×**。

---

### 3.8 剪枝 + 离线隐层（2609.19579，L1）——最实用的基线

**物理意义（解释）**：蒸的是「被剪掉的通道原本负责的表征」。物理上=给截肢后的模型做康复训练：同形状对齐，不让它重新学走路。

**实现**（输入 `teacher`、`loader`；输出 `pruned`）

```python
pruned = prune_width(openvla_oft_7b, ratio=0.63)      # 同构剪枝
cache = {}
with torch.no_grad():
    for batch in loader:
        cache[batch.key] = teacher(batch.obs, return_hidden=True).half()

for batch in loader:
    H_s = pruned(batch.obs, return_hidden=True)       # 同形状
    loss = task_loss(pruned(batch.obs), batch.label) \
         + lam * F.mse_loss(H_s, cache[batch.key].float())
    loss.backward(); optimizer.step()
```

**结果**：63% 剪枝 → LIBERO-Long 93.2%→0.8%，**8 GPU-h** 恢复到 3.5 点内；真机 77.5% vs 59.5%；板载 2.23×。

---

### 3.9 SPARK-VLA（2611.10443，NeurIPS 2026，L1）

**物理意义（解释）**：蒸 3D 专家「注意力放在哪」。物理上=把立体空间感注入 2D 视觉分支，推理时不要 3D 输入。

**实现**（输入 `img`、`action_loss`；输出 `loss`）

```python
with torch.no_grad():
    A_t = expert3d.attention(img)                     # [B,h,N,N] 3D 专家注意力
A_s = student.visual_attention(img)                   # LoRA 视觉分支
L_align = F.mse_loss(A_s, A_t)                        # 对齐注意力响应

# router：用 action-loss 梯度监督选 token
w = router(student.tokens)
L_route = route_supervision(w, action_loss_grad)
loss = L_task + L_align + L_route
```

**结果**：0.5B 主干、约 **53M** 可训练；摘要未给成功率。

---

## 4. 横向总表（VLA）

| 工作 | 场景 | 教师 | 学生 | 蒸什么 | 教师效果 | 学生效果 | 速度/成本 |
|---|---|---|---|---|---|---|---|
| Orion-Lite | 驾驶 | 7B LLM | 0.1B decoder | latent token | DS 77.7 | **DS 80.6** | 150× 模块 / 806→267ms / 31→8GB |
| DRIFT | 驾驶 | <150M LM | 255M 规划器 | latent 推理特征 | — | DS 97.03 / SR 89.39 | 255M |
| FPD-Drive | 驾驶 | 特权教师 | 视觉规划器 | 未来轨迹 | — | 94.9 PDMS (v1) | — |
| OPTED | 驾驶 | 向量化 RL 教师 | 相机学生 | 闭环经验 | — | DS 1.6–9.5× | 交互 −3 数量级 |
| ConDenseDrive | 驾驶 | 3 VFM | 视觉编码器 | 特征 | — | 94.5 PDMS | 2.0× |
| Reflect-VLA | 驾驶 | 同体渐进 | 单次 VLA | token 级纠正 | — | 未给分值 | 训练 −90% |
| SnapFlow | 操作 | π0.5 10-NFE | 1-NFE | 端点 | 97.75% | **98.75%** | −3.3× |
| FastOPD | 操作 | π0.5 等 | 轻量 | 速度场 | — | 2 步保留 84% | −78.1% |
| VLA-MOPD | 操作 | 多专家 | 单一 VLA | ODE 轨迹 | — | 75.4→98.9% | — |
| REPAIR | 操作 | RL 专家 | flow VLA | critic 加权动作 | — | 真机 100% | — |
| TAD | 操作 | 轨迹库 | OpenVLA-OFT 7B | 判别器 reward | — | 26.2→76.6% | 5× 交互 |
| TILDE | 操作 | EFM | 策略 | 恢复轨迹 | — | 4.1× BC | 采集 −2× |
| 2609.19579 | 操作 | 7B | 剪枝 63–72% | 离线 hidden | 93.2% | 教师 −3.5 点内 | 8 GPU-h；2.23× |
| SPARK-VLA | 操作 | 3D 专家 | 0.5B + LoRA | 空间表征 | — | 未给 | 53M 可训 |

> 表内数字来自不同 benchmark，**不可直接横比**；作用是看"量级"和"代价结构"。所有 L1 条目仅摘要级。

---

## 5. 五种"只看描述会写错"的实现细节

1. **Orion-Lite 的 token 取法**：取 planning query 那一个 token，不是 `last_hidden.mean()`。
2. **SnapFlow 的 stop-grad**：两步 Euler 的整条链都要 `no_grad`，漏一步就自我强化。
3. **FastOPD 的教师查询点**：是学生落点 `x_t`，不是数据集里的真值动作。
4. **VLA-MOPD 的 mask**：只对有效动作维算损失，mask 用 float。
5. **剪枝恢复的同构性**：只改宽度、不改层数/接口，否则要加 projector。

---

## 6. 选型

| 约束 | 选 |
|---|---|
| 驾驶、要省推理 LLM | Orion-Lite（0.1B）或 DRIFT（255M） |
| 驾驶、有特权教师/仿真 | OPTED（闭环经验，样本效率最高） |
| 操作、flow 动作头太慢 | SnapFlow（无教师、一次训练）或 FastOPD（可到 2 步） |
| 多个任务专家要合并 | VLA-MOPD（ODE 轨迹级） |
| 真机、有 RL 专家 | REPAIR（critic 加权） |
| 没预算、只想恢复剪坏的模型 | 2609.19579 的离线隐层（8 GPU-h） |

---

更新时间: 2026-11-24
