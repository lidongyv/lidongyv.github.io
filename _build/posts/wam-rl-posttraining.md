---
title: "从 SFT 到闭环：VLA 与 WAM 的 RL 后训练怎么做（实现版）"
title_en: "From SFT to Closed Loop: RL Post-training for VLA and WAM (Implementation Notes)"
summary: "按「输入/输出 → 中间算法 → 伪代码（附物理意义注释）」重写 RL 后训练：GigaBrain RAMP、VAMPO、EVA、CreFlow、WorldVLN，以及六类闭环训练的官方代码级实现（FACT / DreamAvoid / τ0-WM 为仓库源码）。"
summary_en: "A rewrite of RL post-training with inputs/outputs, algorithms and annotated pseudocode: GigaBrain RAMP, VAMPO, EVA, CreFlow, WorldVLN, plus code-grounded closed-loop implementations (FACT, DreamAvoid, τ0-WM)."
tags: ["RL", "后训练", "闭环训练", "VLA", "WAM", "代码"]
tags_en: ["RL", "Post-training", "Closed-loop", "VLA", "WAM", "Code"]
---

> 这篇只讲**训练范式**：SFT 之后如何用奖励信号继续优化，以及**闭环训练**的具体实现。
> 约定：`[论文 §X]`=论文小节；`[代码 path]`=官方仓库文件；**「伪代码」= 可执行结构的函数级代码**，每段带「物理」注释说明这一步对机器人意味着什么；凡没有官方代码的方法，标注「依论文重建」，不冒充源码。
> 与 WAM 论文站（结构分类）分开：那边讲模型长什么样，这里讲怎么训、怎么闭环。

---

## 0. TL;DR

- VLA/WAM 的 SFT 有三个硬伤：**分布漂移**、**长程复合误差**、**执行性 gap**。RL 后训练针对它们，而不是「再多采点数据」。
- 奖励几乎从不来自真机执行，而是五种代理信号：**世界模型价值 / IDM 动作平滑 / LTL 逻辑验证 / 专家潜距离 / 稀疏 episode 成功**。
- 算法只有两大族：**GRPO**（组归一化优势 + 截断代理，无 critic）与 **KL-RL 闭式解 → 加权 NLL**（advantage-conditioned）。稀疏奖励走 DiffusionNFT。
- **闭环训练**有六类实现，其中三类有官方代码可直接读、可跑（FACT、DreamAvoid、τ0-WM）；本文按源码给出输入输出与伪代码。

---

## 1. 为什么后训练（一句话版）

| 失败模式 | 现象 | 后训练的介入点 |
|---|---|---|
| 分布漂移 | 偏离演示后误差累积 | 用策略自己的状态采数据（DAgger/HILR） |
| 长程复合 | 单步像、多步崩 | 段级/子任务级奖励，或世界模型长程评分 |
| 执行性 gap | 未来看着对，动作抖动/越限 | IDM 解码后按运动学约束给奖励（EVA） |
| 指令接地差 | 靠画面捷径而非指令 | 语言教练/视觉目标重标注（CoachSI、JEPA-WAM） |

---

## 2. 五种奖励信号（实现视角）

| 奖励 | 代表 | 输入 | 输出 | 计算位置 |
|---|---|---|---|---|
| 稀疏成功 + 价值 TD | GigaBrain | episode 成功标签、世界模型价值 | 优势 A、二值 I | 训练内环（离线日志） |
| 专家潜距离 | VAMPO | 生成未来潜、专家未来潜 | 标量 r | VAE 潜空间 |
| 动作平滑/限幅 | EVA | 生成视频 → IDM 关节序列 | 标量 R | 关节空间 |
| LTL 逻辑验证 | CreFlow | rollout 帧序列、LTL 规格 | 二值 r + 违例 witness | 监视器 |
| 进度/价值标量 | FACT、τ0-WM | 子任务进度标签、MC 回报 | [0,1] 进度/价值 | 训练标签 |

---

## 3. 五个 RL 后训练方法（输入/输出 + 伪代码）

### 3.1 GigaBrain RAMP：KL-RL 闭式解 → 加权 NLL

**⚠️ 无官方代码**；以下伪代码**依论文 §3.2.1–3.2.2 的公式重建**（Eq.2/3/5/8），标注了每个公式对应代码行。

**输入 / 输出**

| 项 | 内容 |
|---|---|
| 输入 | 机器人数据 D（4K 小时）、episode 成功标签、预训练 VLA π_θ、世界模型 W_φ（Wan2.2 骨干）、HILR 新数据 |
| 输出 | 后训练策略 π_θ*、联合续训后的世界模型 W_φ* |
| 关键超参 | 掩码率 p=0.2（世界模型 token）；稀疏奖励常数 C_fail；TD 的 n、γ；阈值 ε；权重 α —— **数值论文未写** |

```python
# ── Stage 0：世界模型预训练（§3.2.2 Stage 1，Eq.7）──────────────
def train_world_model(D_robot, W, steps):
    """物理：世界模型学「给当前观测+本体+动作，12/24/36/48 步后画面变成什么」。
    它不输出动作，只提供未来潜与价值，供后续策略条件化。"""
    for (o_t, o_future, p_t, v_t, a_t, l) in D_robot:
        z_t   = W.vae.encode(o_t)                      # 当前帧 → 潜 [B,C,T,H,W]
        s_t   = concat([z_t, psi(v_t), psi(p_t)])      # 增强状态 S=(o,z,l)，§3.2.1
        for i in [12, 24, 36, 48]:                     # 四个未来时刻
            eps = randn_like(z_future_i)               # 流匹配噪声
            x   = (1-t)*eps + t*z_future_i             # 线性插值路径
            loss += mse(W.flow(s_t, a_t, t), z_future_i - eps)   # Eq.(7)
        W.update(loss)

# ── Stage 1：稀疏奖励（§3.2.2 Stage 1，Eq.5）────────────────────
def sparse_reward(episode, C_fail):
    """物理：只有「最后成功了」才是 0；失败要付固定代价；每个中间步 −1 表示
    '多花了一步的时间成本'。价值函数因此等价于「负的期望剩余步数」。"""
    for t in range(T):
        if t == T-1 and episode.success:  r_t = 0
        elif t == T-1:                    r_t = -C_fail
        else:                             r_t = -1
    return r

# ── Stage 2：优势 + 二值化 + 加权 NLL（Eq.8 / Eq.3）─────────────
def advantage_and_loss(rollouts, W, pi_theta, gamma, n, eps, alpha):
    """物理：A>ε 表示「这一步比参考策略平均更好」。把 A 二值化成 I 后，
    策略只需学会「在 I=1 的条件下更像专家」，无需策略梯度。"""
    for tau in rollouts:
        for t in range(T):
            A_t = sum(gamma**k * tau.r[t+k] for k in range(n)) \
                  + gamma**n * W.value(tau.s[t+n]) - W.value(tau.s[t])   # Eq.(8)
            I_t = 1.0 if A_t > eps else 0.0                              # 二值优势
            # 两项 NLL：无条件 + 改进条件（KL-RL 闭式解，§3.2.1 Eq.3）
            loss += -log pi_theta(a_t | o_t, z_t, l) \
                    - alpha * log pi_theta(a_t | I_t, o_t, z_t, l)
        mask_world_tokens(pi_theta, p=0.2)   # 随机掩掉世界模型 token，防依赖
        pi_theta.update(loss)

# ── Stage 3：HILR 数据采集（§3.2.2 Stage 3）────────────────────
def hilr_collect(pi_theta, env, human, n_episodes):
    """物理：机器人自己跑，快失败时人接管；记录接管前后的状态-动作，
    并软件剔除「人机切换瞬间」的抖动样本（否则策略会学到突变动作）。"""
    data = []
    for _ in range(n_episodes):
        traj, intervened = [], False
        while not done:
            a = pi_theta.act(obs)
            if human.wants_control(obs):        # 人判断要接管
                a = human.act(obs); intervened = True
            traj.append((obs, a, intervened))
            obs, done = env.step(a)
        data.append(scrub_boundary(traj))       # 去掉切换边界的伪影
    return data

# ── Stage 4：联合续训（§3.2.2 Stage 4）──────────────────────────
def joint_continue(pi_theta, W, D_hilr, D_base):
    """物理：只训策略会让世界模型的优势估计越来越偏（A 塌缩到 0），
    所以世界模型也要跟着新数据继续训，保持「价值判断」与「策略行为」同步。"""
    for batch in mix(D_hilr, D_base, ratio=1):
        W.update(flow_loss(batch))              # 世界模型继续学
        pi_theta.update(weighted_nll(batch))    # 策略继续学
```

---

### 3.2 VAMPO：GRPO 训练视频去噪器（**官方代码**）

**输入 / 输出**

| 项 | 内容 |
|---|---|
| 输入 | 专家视频（预编码为 SVD VAE 潜）、SFT 后的视频预测器 VPM、初始噪声 |
| 输出 | 后训练后的 VPM（动作侧 AGM 冻结） |
| 代码 | `OpenHelix-Team/VAMPO`：`grpo/reward_fn.py`、`step1_train_svd_grpo.py`、`grpo/eas_with_logprob.py` |

```python
# ── 采样：G 条未来，只有首步随机（eas_with_logprob.py）──────────
def rollout_group(vpm, z0, cond, G=8, steps=20):
    """物理：同一条当前观测下，想象 G 种「可能的未来」。
    首步加噪（SDE）让不同想象分叉；后续走 ODE 保证轨迹可比、可信。"""
    trajs, logps = [], []
    for g in range(G):
        z = z0.clone()
        for i, t in enumerate(scheduler.timesteps(steps)):
            if i == 0:                              # 首步：采样噪声
                z, lp = eas_step_with_logprob(vpm, z, t, cond)
                logp = lp
            else:                                   # 其余：确定性 ODE
                z = euler_step(vpm, z, t, cond)
        trajs.append(z); logps.append(logp)
    return trajs, logps

# ── 奖励：与专家未来的潜空间一致性（reward_fn.py）───────────────
def latent_reward(pred_latents, expert_latents):
    """物理：预测的未来和专家演示的未来在「潜空间」有多接近。
    用 L1 保证结构一致，用 cosine 保证方向一致；不解码像素，省算力。"""
    mae    = l1(pred_latents, expert_latents).mean(dim=[2,3,4])
    cosine = cosine_sim(pred_latents.flatten(2), expert_latents.flatten(2))
    return -mae + cosine                    # 默认 mae_weight=cosine_weight=1.0

# ── 更新：组归一化优势 + 截断代理（step1_train_svd_grpo.py）─────
def grpo_update(vpm, trajs, logps, expert_latents, clip=0.2):
    """物理：组内比较——比同组平均更接近专家（优势>0）的想象被强化，
    更差的被压制；截断防止单步更新过猛把 VPM 推离预训练分布。"""
    r    = [latent_reward(z, expert_latents) for z in trajs]      # [G]
    adv  = (r - mean(r)) / (std(r) + 1e-8)                        # 组归一化
    for z, lp, a in zip(trajs, logps, adv):
        ratio = exp(new_logprob(vpm, z) - lp)                      # 重要性比
        loss += min(-a*ratio, -a*clip(ratio, 1-clip, 1+clip))
    vpm.update(loss)                          # 只更新 VPM；VAE/文本编码器冻结
```

---

### 3.3 EVA：IDM 动作平滑度奖励 + Flow-GRPO

**⚠️ 无官方代码**（项目页只有展示）；伪代码依 §4.2/§4.3 重建。

**输入 / 输出**：输入=生成视频 + 冻结 IDM；输出=只更新生成器 LoRA 的梯度。

```python
def eva_reward(video, idm, vmax, amax, lam):
    """物理：把生成的视频解码成关节角序列，惩罚「抖动(jerk)、急加速(acc)、
    超速/超加速度(限幅)」——这是机械臂真实执行时会出问题的三种运动。"""
    q = idm.decode(video)                     # 关节序列 [T, D]
    v = diff(q); a = diff(v); j = diff(a)     # 速度/加速度/jerk
    R = -(lam.j*huber(j) + lam.a*huber(a)
          + lam.v*relu(abs(v)-vmax).pow(2)
          + lam.a2*relu(abs(a)-amax).pow(2))
    return R

def flow_grpo_step(gen, videos, prompts):
    """物理：同一提示词生成多条视频，谁的动作更平滑就强化谁；
    加 KL 约束防止生成器只讨好奖励、丢掉预训练的画面先验。"""
    rs  = [eva_reward(v, idm) for v in videos]
    adv = (rs - mean(rs)) / (std(rs) + 1e-8)
    loss = clipped_surrogate(adv, ratio) - beta * KL(gen, gen_ref)
    gen.update(loss)                          # 仅 LoRA
```

---

### 3.4 CreFlow：LTL 稀疏奖励 + DiffusionNFT + corrective reflow

**⚠️ 无官方代码**；伪代码依 §4.1–§4.3 重建（Eq.5–9）。

**输入 / 输出**：输入=rollout 视频、LTLf 规格；输出=生成器梯度（监视器、IDM 冻结）。

```python
def ltl_monitor(video, spec):
    """物理：用线性时序逻辑检查「抓取→移动→放下」这类时序约束是否满足。
    返回 0/1 奖励 + 违例发生在哪帧（witness），供损失只惩罚违例区。"""
    r, witness, masks = spec.check(frames(video))   # Eq.(5)
    return r, witness, masks                        # r∈{0,1}

def creflow_loss(rollouts, spec, ref_policy, beta):
    """物理：成功的 rollout 正常学；失败的 rollout 不学「复制坏动作」，
    而是把它们的预测往「同组成功样本的均值」推——这是物理上可执行的改进方向。"""
    rs = [ltl_monitor(v, spec)[0] for v in rollouts]
    M  = union_witness_masks(rollouts, spec)        # 只在违例帧/实体上加梯度
    # NFT：正分支强化、负分支用反射目标推开（Eq.7）
    loss_nft = token_masked_nft(rollouts, rs, M)
    # corrective reflow：失败样本向组内正样本均值纠正（Eq.9）
    x_pos = mean([v for v, r in zip(rollouts, rs) if r == 1])
    loss_cr = mse(mask(M) * (x0_hat(failed) - x_pos))
    return loss_nft + loss_cr + beta * KL(gen, ref_policy)
```

---

### 3.5 WorldVLN：自回归 WAM 的段级 GRPO

**⚠️ 无官方代码**；伪代码依 §4.2/§4.3 重建（Eq.9–13）。

**输入 / 输出**：输入=在线模拟器、真观测流；输出=自回归 WAM 的策略梯度。

```python
def segment_rollout(model, env, goal, gamma):
    """物理：在模拟器里走一段，段末用真实观测替换模型想象的潜状态。
    这样误差不会因为长程想象越滚越大（世界模型只能短程可信）。"""
    seg = []
    for j in range(J):
        a = model.act(seg_obs)                       # 动作块
        seg.append(a)
        obs = env.step(a)
        seg_obs = obs if env.is_real else model.predict(obs)  # 真观测回灌
    return seg

def worldvln_step(model, seg, goal, gamma):
    """物理：三种奖励——轨迹像专家、终点离目标近、别偏离参考策略；
    按 γ^{j-1} 衰减，越早的动作越重要（后面还能纠）。"""
    r = sum(gamma**(j-1) * (lam_traj * r_traj(seg[j], goal)
                            + lam_task * r_task(seg[j], goal)
                            + lam_ref * log_ref(seg[j]))
            for j in range(1, J+1))                 # Eq.(9-12)
    A = (r - mean_group(r)) / (std_group(r) + 1e-8)  # 段级组归一化
    model.update(clipped_surrogate(A, ratio))        # 段级 GRPO，Eq.(13)
```

---

## 4. 闭环训练：六类实现（输入/输出 + 算法 + 代码）

> 闭环 = 训练期让策略执行自己的动作，并把执行后果纳入训练。以下六类可组合。

### 4.1 HILR：人机接力

| 项 | 内容 |
|---|---|
| 输入 | 策略 π、环境、人类接管策略、episode 上限 |
| 输出 | 带「干预标记」的轨迹数据集（已剔除切换抖动） |
| 算法 | 策略自主跑 → 人判断要接管 → 记录 (obs, a, intervened) → 剔除切换边界 → 行为克隆 |
| 证据 | GigaBrain §3.2.2 Stage 3（无官方代码） |

```python
def hilr_episode(policy, env, human):
    """物理：只在「机器人快做错」的边界上由人示范，样本价值远高于随机演示；
    但切换瞬间动作会突变，必须剔除，否则策略学到不连续的动作。"""
    traj, intervened = [], False
    while True:
        obs = env.observe()
        if human.takeover(obs):           # 人判断：这一步会失败
            a, intervened = human.act(obs), True
        else:
            a = policy.act(obs)
        traj.append((obs, a, intervened))
        nxt, done = env.step(a)
        if done: break
    return scrub_switch_artifact(traj)    # 去边界伪影
```

### 4.2 DAgger 式恢复数据（XPACE）

| 项 | 内容 |
|---|---|
| 输入 | 专家轨迹、可自适应的模拟器/世界模型、一致性过滤器 |
| 输出 | deviation–recovery 恢复轨迹（占微调数据 ~8%） |
| 算法 | 模拟器适配策略自己的上下文 → 在专家轨迹附近注入偏差 → 推演恢复 → 一致性过滤 → 与专家数据混合微调 |
| 证据 | XPACE 摘要 + 项目页；恢复消融 61.7→86.7（异套件，见原文声明） |

```python
def recovery_data(expert_traj, simulator, policy):
    """物理：真实部署时机器人一定会偏离专家轨迹；这里人为制造偏离，
    再用模拟器推演「怎么回来」，把恢复动作教给策略。"""
    sim = simulator.adapt_to(policy)              # 适配策略自己的视觉上下文
    recs = []
    for tau in expert_traj:
        s_dev = inject_deviation(tau, scale=delta)  # 制造偏离
        tau_rec = sim.rollout_recovery(s_dev)       # 推演恢复路径
        if consistency_filter(tau_rec):             # 只保留物理一致的恢复
            recs.append(tau_rec)
    return recs                                    # 混合专家数据微调
```

### 4.3 失败数据闭环（FACT，**官方代码**）

**输入 / 输出**：输入=成功+失败 rollouts（含 `action_loss_mask`）；输出=策略 + 价值头 + 未来状态预测头。

```python
# 事实：FACT 训练器源码（world_action_model/trainer/wa_casual_trainer.py）
action_loss_mask = batch_dict.get('action_loss_mask')

def _masked_mean(per_sample_loss, mask):
    """物理：失败 episode 的动作不能学（会教坏机器人），
    但它的「后果」（未来状态、价值）必须学——失败是负样本，也是成本信号。"""
    m = mask.reshape(-1)                      # 1=可模仿, 0=失败
    return (per_sample_loss * m).sum() / m.sum().clamp_min(1e-8)

action_loss = _masked_mean(action_loss_raw.flatten(1).mean(dim=1), action_loss_mask)
value_loss  = ((value_pred - value_target) ** 2).mean()   # 全部样本都参与
```

```python
# 训练主循环（按源码结构整理）
def fact_train_step(batch, model):
    """物理：因果顺序是「先动作、后想象」——用真实动作作条件去预测未来状态
    与价值；损失分四路（视觉/动作/未来状态/价值），动作那一路被失败掩码门控。"""
    visual_loss, action_pred, future_state_pred, value_pred = model(batch)
    action_loss = masked_mean(action_loss_raw, batch.action_loss_mask)
    value_loss  = mse(value_pred, batch.value_normalized)      # 失败样本也学
    loss = w_v*visual_loss + w_a*action_loss + w_f*future_state_loss + w_val*value_loss
    return loss
```

### 4.4 DreamAvoid：混合数据价值模型 + 触发式 dreaming（**官方代码**）

**输入 / 输出**：输入=当前 obs、候选动作块、预测的未来帧、任务 id、状态；输出=标量价值（越高越安全）。

```python
# 事实：ValueModel 源码（DreamDojo-da/examples/value_model.py）
class ValueModel(nn.Module):
    """物理：给「当前画面 + 这段动作 + 它导致的未来若干帧」打分。
    分数高 = 这段动作大概率不会失败。冻结 DINOv2 保证视觉特征稳定。"""
    def forward(self, obs, future, action, state, task_id):
        obs_tok = encoder(obs) + frame_emb[0]           # 当前帧 token
        fut_tok = encoder(future) + frame_emb[1] + time_emb[:L]  # 未来帧 token
        seq = transformer([value_token] + obs_tok + fut_tok)     # 时序注意力
        pooled = seq[:, 0]                              # value token 汇总
        x = cat([pooled, proj(action), task_emb(task_id), proj(state)])
        return head(x).squeeze(-1)                      # 标量价值

# 推理：触发 → 候选 → 评估 → 选优（agilex_infer_dream_trigger.py）
def dream_avoid_step(policy, value_model, obs, state, task):
    """物理：只在「关键阶段」才花算力想象（由 trigger 判断），
    否则直接执行策略动作；候选动作各梦一小段未来，选价值最高者。"""
    if not trigger(obs):                                # 未到关键阶段
        return policy.act(obs)
    cands = [policy.sample_action(obs) for _ in range(8)]   # 8 个候选
    best = max(cands, key=lambda a: value_model.score(
        obs, rollout_future(policy, obs, a), a, state, task))
    return best
```

### 4.5 τ0-WM：propose → RCS 排序 → 模拟器精炼（**官方代码**）

**输入 / 输出**：输入=观测、状态、候选数 N、RCS 时间步比例、模拟器；输出=选中动作块（可被模拟器截短执行步数）。

```python
# 事实：TauPolicyTTC.play（web_infer_utils/posttrain_taco_play_withTTC/TauPolicyTTC.py）
def play(obs, state, N_proposals=8, RCS_timestep_ratios=[0.5]):
    """物理：一次生成 N 个候选动作块；用「重去噪一致性」判断哪个候选
    不是偶然采样出来的（越一致越可信）；再用动作条件模拟器推演，
    谁的未来回报高就执行谁，且按回报峰值决定执行几步。"""
    pred = pipeline.infer(obs, state, N_proposals=N_proposals, get_RCS=True)
    cands, rcs = pred['action'], pred['rcs']            # [N,T,C], [N]
    best = cands[rcs.max(dim=0).indices]                # RCS 最高者
    if sim_refine:                                      # 可选的模拟器精炼
        sim_rewards = sim_pipeline.infer(actions=cands, return_reward=True)
        if sim_rewards.max() < 0:                       # 全都差 → 用模拟器潜重生成
            best = re_predict_with_sim_latent(...)
        else:
            step = argmax(sim_rewards) + 1              # 只执行回报峰值前
            best = cands[best_idx, :step]
    return best
```

### 4.6 语言教练 relabel（CoachSI）

| 项 | 内容 |
|---|---|
| 输入 | 冻结 VLM、成功的 rollout、原任务指令 |
| 输出 | 带「下一步指令」标签的轨迹；动作骨干冻结，只训指令跟随 |
| 算法 | VLM 观察轨迹 → 给每段命名下一步 → 用原指令 hindsight relabel → 微调策略 |
| 证据 | CoachSI 摘要（长程未见场景 47.4→59.3） |

```python
def coach_relabel(traj, vlm, task):
    """物理：机器人失败常常不是「不会动」，而是「不知道该干哪一步」；
    让 VLM 把成功轨迹切段命名，等于给策略补上「下一步是什么」的监督。"""
    steps = []
    for seg in segment(traj):
        steps.append(vlm.describe(seg.frames, task, "next step?"))
    return [(seg.obs, seg.action, step) for seg, step in zip(segment(traj), steps)]
```

---

## 5. 六类闭环的对照

| 类 | 数据从哪来 | 奖励/信号 | 更新什么 | 代码 |
|---|---|---|---|---|
| HILR | 人接管边界 | 无（模仿） | 策略 | 无公开 |
| DAgger 恢复 | 模拟器推演偏离-恢复 | 一致性过滤 | 策略 | 项目页 |
| 失败数据（FACT） | 真机失败 rollouts | 掩码 + 价值头 | 策略+价值 | [Bariona/FACT](https://github.com/Bariona/FACT) |
| 混合价值（DreamAvoid） | 成功/失败/边界 | 学出的价值模型 | 价值模型+推理选优 | [XianzheFan/DreamAvoid](https://github.com/XianzheFan/DreamAvoid) |
| 模拟器精炼（τ0-WM） | 候选动作 + 模拟器 | RCS + 模拟回报 | 不更新（推理选优） | [sii-research/tau-0-wm](https://github.com/sii-research/tau-0-wm) |
| 语言教练（CoachSI） | 成功轨迹 | VLM 命名 | 指令跟随 | 项目页 |

---

## 6. 工程清单（决定要不要上 RL）

1. **奖励能否离线算**？能（潜距离/IDM/进度）→ 先做离线 GRPO；不能 → 才考虑在线 rollout。
2. **有没有失败数据**？失败数据比成功数据稀缺，价值更高（FACT 的掩码机制就是为它设计的）。
3. **防退化三件套**：KL/参考锚定、优势裁剪、混合 SFT 损失。
4. **评测**：成功率 + 完成时间 + 进度；报告「相对 SFT 的增量」，不要只报绝对值。
5. **算力**：在线 RL ≈ 单次推理 × 组大小 G（G=8、20 步去噪约 160×）。

---

## 7. 坑（按证据标注）

- **「伪代码」必须能落地**：如果一段代码没有函数签名、没有输入输出、没有数据流，它就不是伪代码，只是步骤清单。
- **公式与代码要能对上**：本文每个公式都映射到一行代码；对不上的地方（GigaBrain 的 β/γ/n/ε）明确标注「论文未给」。
- **自评数字**：本领域几乎无第三方复现；跨论文比较先看协议（trials、任务套件）。
- **奖励 hacking**：IDM 平滑度可被「不动」骗过——EVA 用限幅惩罚堵这个洞。
- **闭环≠更好**：世界模型偏差会被 RL 放大（GigaBrain 需要 Stage 4 联合续训）。
- **不要把「post-training」都当 RL**：V-JEPA 2 的 post-training 是加动作条件的微调。

---

## 8. 参考（含代码）

- GigaBrain-0.5M*（RAMP）：arXiv 2602.12099（无官方代码）
- VAMPO：arXiv 2603.19370 · [OpenHelix-Team/VAMPO](https://github.com/OpenHelix-Team/VAMPO)
- EVA：arXiv 2603.17808
- CreFlow：arXiv 2605.14274
- WorldVLN：arXiv 2605.15964
- FACT：arXiv 2608.10232 · [Bariona/FACT](https://github.com/Bariona/FACT)
- DreamAvoid：arXiv 2605.11750 · [XianzheFan/DreamAvoid](https://github.com/XianzheFan/DreamAvoid)
- τ0-WM：arXiv 2606.01027 · [sii-research/tau-0-wm](https://github.com/sii-research/tau-0-wm)
- CoachSI：arXiv 2611.25529
- DreamGen（开环反例）：arXiv 2505.12705
- WAM 论文站：/blog/wam-survey/
