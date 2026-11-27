---
title: "从 SFT 到闭环：VLA 与 WAM 的 RL 后训练怎么做（实现版）"
title_en: "From SFT to Closed Loop: RL Post-training for VLA and WAM (Implementation Notes)"
summary: "按「物理意义 → 为什么 → 出处 → 优势」四要素重写 RL 后训练：GigaBrain RAMP、VAMPO、EVA、CreFlow、WorldVLN，以及六类闭环训练的官方代码级实现（FACT / DreamAvoid / τ0-WM 为仓库源码）。"
summary_en: "RL post-training rewritten with the four-element code reading: physical meaning, motivation, provenance, trade-offs. Covers GigaBrain RAMP, VAMPO, EVA, CreFlow, WorldVLN, and six closed-loop implementations (FACT, DreamAvoid, τ0-WM from source)."
tags: ["RL", "后训练", "闭环训练", "VLA", "WAM", "代码"]
tags_en: ["RL", "Post-training", "Closed-loop", "VLA", "WAM", "Code"]
---

> 这篇只讲**训练范式**：SFT 之后如何用奖励信号继续优化，以及**闭环训练**的具体实现。
> 约定：`[论文 §X]`=论文小节；`[代码 path]`=官方仓库文件。每段伪代码前有「代码解读」四要素：**物理意义 / 为什么 / 出处 / 优势与代价**——这是本文与「步骤清单式伪代码」的区别。
> 与 WAM 论文站（结构分类）分开：那边讲模型长什么样，这里讲怎么训、怎么闭环。

---

## 0. TL;DR

- VLA/WAM 的 SFT 有三个硬伤：**分布漂移**、**长程复合误差**、**执行性 gap**。RL 后训练针对它们，而不是「再多采点数据」。
- 奖励几乎从不来自真机执行，而是五种代理信号：**世界模型价值 / IDM 动作平滑 / LTL 逻辑验证 / 专家潜距离 / 稀疏 episode 成功**。
- 算法只有两大族：**GRPO**（组归一化优势 + 截断代理，无 critic）与 **KL-RL 闭式解 → 加权 NLL**（advantage-conditioned）。稀疏奖励走 DiffusionNFT。
- 闭环训练三类有官方代码可直接读：FACT、DreamAvoid、τ0-WM；其余标注「依论文重建」。

---

## 1. 为什么后训练（一句话版）

| 失败模式 | 现象 | 后训练的介入点 |
|---|---|---|
| 分布漂移 | 偏离演示后误差累积 | 用策略自己的状态采数据（DAgger/HILR） |
| 长程复合 | 单步像、多步崩 | 段级/子任务级奖励，或世界模型长程评分 |
| 执行性 gap | 未来看着对，动作抖动/越限 | IDM 解码后按运动学约束给奖励（EVA） |
| 指令接地差 | 靠画面捷径而非指令 | 语言教练/视觉目标重标注（CoachSI、JEPA-WAM） |

---

## 2. 五个 RL 后训练方法

### 2.1 GigaBrain RAMP：KL-RL 闭式解 → 加权 NLL

**先给符号表**（shape 中 `C_z`=VAE 潜通道数、`h×w`=潜分辨率、`d_p`=本体维、`H_a`=动作块长、`d_a`=动作维；论文未给的数值标「未给」）：

| 符号 | 形状 | 物理含义 | 从哪来 |
|---|---|---|---|
| `o_t` | `[3,H,W]` | t 时刻 RGB 观测 | 数据 |
| `p_t` | `[d_p]` | 本体感知（关节/夹爪） | 数据 |
| `a_t` | `[H_a,d_a]` | 动作块（策略输出） | 数据/策略 |
| `l` | token ids | 语言指令 | 数据 |
| `success` | bool | episode 级成功标签 | 数据（遥操作日志） |
| `z_t` | `[H′,W′,C′]` | 当前帧的 VAE 时空潜 | `z_t = E_vae(o_t)` |
| `z_future_i` | `[C_z,h,w]` | 未来帧潜，`i∈{12,24,36,48}` 步后 | `E_vae(o_{t+i})`，仅训练期 |
| `v_t` | 标量 | 价值估计≈「负的期望剩余步数」 | 价值头 `W.value(s_t)` |
| `Ψ(·)` | `R → R^{H′×W′×1}`；`R^d → R^{H′×W′×d}` | **spatial tiling projection**：把标量价值/低维本体复制并广播到空间维，与视觉潜对齐 | 论文 §3.2.2（Eq.6） |
| `s_t` | `[H′,W′,C′+1+d]` | 增强状态：潜 + 价值图 + 本体图（通道拼接，Eq.6） | `s_t=[z_t;Ψ(v_t);Ψ(p_t)]` |
| `W.flow(u,τ,s_t,a_t)` | → `[C_z,h,w]` | 流匹配速度场：给定加噪潜、流时间、状态、动作，预测「往干净未来走的速度」 | 世界模型骨干（Wan2.2 + flow matching） |
| `u_τ` | `[C_z,h,w]` | 加噪插值潜 `(1−τ)ε+τz_future` | 训练时构造 |
| `ε`（噪声） | `[C_z,h,w]` | 流匹配噪声 | `randn_like` |
| `r_t` | 标量 | 稀疏奖励（Eq.5） | 由 `success` 与步数算出 |
| `A_t` | 标量 | n 步 TD 优势（Eq.8） | 由 `r_t`、`v_t` 算出 |
| `I_t` | `{0,1}` | 改进指示：`1[A_t>ε_thr]` | 由 `A_t` 与阈值算出 |
| `γ, n, ε_thr, α, C_fail, β` | 标量 | 折扣/TD 步长/二值阈值/权重/失败代价/KL 温度 | **数值论文未给** |

**数据流闭环（先看这个，再看代码）**

```
数据 episode ──► (o_t, o_{t+i}, p_t, a_t, l, success)
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
   VAE 编码 o_t / o_{t+i}           稀疏奖励 r_t（Eq.5）
        │  z_t / z_future_i              │
        ▼                                ▼
   s_t=[z_t;Ψ(v_t);Ψ(p_t)] ──► W.flow 训练（Eq.7）──► 价值头 v_t（TD）
        │                                              │
        └────────────► A_t（Eq.8） ◄──────────────────┘
                        │
                 I_t=1[A_t>ε_thr]
                        │
                        ▼
             策略损失（Eq.3，两项 NLL）
```

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 世界模型学「给当前观测+本体+动作，12/24/36/48 步后画面变成什么」；价值头学「还剩多少步能成功」；策略学「在 I=1 的条件下更像专家」。 |
| 为什么这么做 | 真机奖励极稀疏，直接用策略梯度方差太大；把 KL-RL 闭式解改写成两项 NLL，等价于用优势当条件做加权模仿，训练更稳。 |
| 方法出处 | 论文 §3.2.1–3.2.2（Eq.2/3/5/6/7/8）；**无官方代码，本文依论文重建**；KL-RL 谱系来自 LLM 的 RLHF/DPO，RECAP 是同族特例。 |
| 优势与代价 | 优势：无需 critic、可用离线日志、训练稳定；代价：依赖价值头精度，β/γ/n/ε/α 论文未给需自调。 |

```python
# ── 采样：一条 episode 的字段（success 是 episode 级标签）──────────
def sample_episode(D):
    """返回：o [T,3,H,W]；p [T,d_p]；a [T,H_a,d_a]；l；success: bool。"""
    return {"o": o, "p": p, "a": a, "l": l, "success": success}

# ── 编码：图像 → 潜（VAE 冻结）──────────────────────────────────
def encode(vae, img):                       # img [3,H,W] → [C_z,h,w]
    return vae.encode(img).latent_dist.mode()

# ── 构造增强状态 s_t（Eq.6）────────────────────────────────────
def build_state(z_t, v_t, p_t, psi_v, psi_p):
    """物理：把标量价值与本体向量『画』成与潜同尺寸的通道图，再通道拼接。
    z_t [C_z,h,w]；Ψ(v_t)→[1,h,w]；Ψ(p_t)→[C_p,h,w]；输出 [C_z+1+C_p,h,w]。"""
    v_map = psi_v(v_t).view(1, h, w)        # spatial tiling：标量复制成单通道图
    p_map = psi_p(p_t).view(d, h, w)        # spatial tiling：d 维本体广播到 d 通道
    return cat([z_t, v_map, p_map], dim=0)  # Eq.6 channel-wise concat

# ── 世界模型一步：流匹配（Eq.7）──────────────────────────────────
def world_model_step(W, vae, ep, tau):
    """物理：给『加噪的未来潜』和『当前状态+动作』，预测往干净未来走的速度。
    目标速度 = z_future − ε（流匹配的标准目标）。"""
    z_t   = encode(vae, ep["o"][t])
    v_t   = W.value(build_state(z_t, W.value(build_state(z_t, 0, ep["p"][t]), ep["p"][t]), ep["p"][t]))  # 价值头估计
    s_t   = build_state(z_t, v_t, ep["p"][t], W.psi_v, W.psi_p)
    z_future = [encode(vae, ep["o"][t+i]) for i in (12, 24, 36, 48)]   # 四个未来目标
    eps   = randn_like(z_future[0])                 # ε [C_z,h,w]
    u     = (1 - tau) * eps + tau * z_future[0]     # 插值潜 u_τ
    target = z_future[0] - eps                      # 目标速度
    return mse(W.flow(u, tau, s_t, ep["a"][t]), target)   # Eq.7

# ── 稀疏奖励（Eq.5）：输入是 episode，输出每步 r_t ────────────────
def sparse_reward(ep, C_fail):
    """物理：成功=0（不罚），失败=固定代价，中间每步 −1（时间成本）。
    v 头学到的就是『负的期望剩余步数』。"""
    T = len(ep["a"]); r = [-1.0] * T
    r[T-1] = 0.0 if ep["success"] else -C_fail
    return r                                     # [T]

# ── n 步 TD 优势（Eq.8）──────────────────────────────────────────
def n_step_advantage(r, v, gamma, n):
    """物理：A_t = 这段动作实际拿到的折扣回报 + 未来价值基线 − 当前价值基线。
    A_t>0 表示『这一步比价值函数的预期更好』。"""
    T = len(r); A = [0.0] * T
    for t in range(T):
        k = min(n, T - 1 - t)
        A[t] = sum(gamma**i * r[t+i] for i in range(k)) \
               + gamma**k * v[min(t+k, T-1)] - v[t]     # Eq.8
    return A                                     # [T]

# ── 策略损失（Eq.3）：两项 NLL ───────────────────────────────────
def policy_loss(pi, ep, s_t, A, eps_thr, alpha):
    """物理：第一项=正常模仿；第二项=『在 I=1（这步更好）条件下』模仿。
    I 是优势的二值化，作为额外的条件 token 喂给策略。"""
    I = [1.0 if a > eps_thr else 0.0 for a in A]        # I_t ∈ {0,1}
    loss = 0.0
    for t in range(len(A)):
        loss += -logp(pi, ep["a"][t], cond=(ep["o"][t], s_t[t], ep["l"])) \
                - alpha * logp(pi, ep["a"][t], cond=(ep["o"][t], s_t[t], ep["l"], I[t]))
    return loss                                  # Eq.3
```

**HILR（Human-in-the-Loop Rollout，人机接力 rollout）到底算什么**：它不是公式，是**数据采集过程**——输入策略/环境/人类接管策略，输出带干预标记的轨迹。

```python
def hilr_episode(policy, env, human):
    """物理：机器人自己跑；人只在『即将失败』时接管。
    输出 [(obs, a, intervened)]，并剔除切换瞬间的抖动样本（否则策略学到突变动作）。"""
    traj, obs, done = [], env.reset(), False
    while not done:
        a = policy.act(obs)
        if human.wants_control(obs):
            a, flag = human.act(obs), 1      # 人接管
        else:
            a, flag = a, 0
        traj.append((obs, a, flag))
        obs, done = env.step(a)
    return scrub(traj)                       # 去切换边界伪影

# Stage 4：世界模型 + 策略在 HILR+base 上联合续训（混合比例论文未给）
def joint_continue(W, pi, D_hilr, D_base):
    for batch in sample_mix(D_hilr, D_base):     # ratio 未给
        W.update(world_model_step(W, W.vae, batch, tau=rand()))   # 防优势坍缩
        pi.update(policy_loss(pi, batch, build_states(batch), A=batch.A))
```

### 2.2 VAMPO：GRPO 训练视频去噪器（**官方代码**）

**输入 / 输出**：输入=专家视频（SVD VAE 潜 `[B,F,C,H,W]`）、SFT 后的 VPM、初始噪声；输出=后训练 VPM（动作侧 AGM 冻结）。

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 同一条当前观测下想象 G 种未来；与专家未来越接近的想象越可执行。 |
| 为什么这么做 | 直接对动作策略做 RL 需要真机 rollout；这里把「视频预测器」当策略，奖励来自专家潜距离，**完全离线**、无需环境交互。 |
| 方法出处 | 仓库 `OpenHelix-Team/VAMPO`：`grpo/eas_with_logprob.py`、`grpo/reward_fn.py`、`step1_train_svd_grpo.py`（源码级）；GRPO 谱系来自 DeepSeek-R1 的组相对优势。 |
| 优势与代价 | 优势：离线、无 critic、见效快；代价：只改进视频预测，不直接改动作策略；首步 SDE 的方差需调 clip。 |

#### ① 采样器：`euler_step` 到底怎么算

仓库里没有叫 `euler_step` 的函数；实际是 **diffusers 的 `EulerDiscreteScheduler`**：首步走 `eas_step_with_logprob`（Euler **Ancestral**，即带噪声的 SDE 步），其余步走普通 `scheduler.step`（纯 Euler，`sigma_up=0`，无噪声）。源码核心（`grpo/eas_with_logprob.py`）：

```python
# 输入：model_output=模型预测的噪声 v̂ [B,F,C,H,W]；sample=当前潜 x_σ; sigma=σ_k
# 1) 先从噪声预测反推「干净样本」x0（epsilon 参数化）
pred_original_sample = sample - sigma * model_output        # x0 = x − σ·ε̂
# 2) Euler 的导数 = (x − x0)/σ，即「指向 x0 的方向 / σ」
derivative = (sample - pred_original_sample) / sigma
# 3) 步长：sigma_down 是去掉噪声后的目标 σ；sigma_up 是本步注入的噪声量
sigma_up   = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
dt = sigma_down - sigma
# 4) 确定性部分：Euler 一步
prev_sample_mean = sample + derivative * dt
# 5) 随机部分：只有首步（prev_sample 未提供）才真采样噪声
if first_step:            # 首步：prev_sample 未提供
    noise = randn_tensor(sample.shape)
    prev_sample = prev_sample_mean + noise * sigma_up       # SDE 步
    self._step_index += 1
# 6) 反推本步实际用的噪声，算对数概率（用于 PPO/GRPO 的重要性比）
noise = (prev_sample - prev_sample_mean) / sigma_up
log_prob = (-0.5*noise**2 - torch.log(sigma_up) - 0.5*math.log(2*math.pi)) \
             .mean(dim=tuple(range(2, prev_sample.ndim)))   # 空间维平均 → [B,C]
```

- **为什么 `sigma_up` 随 σ 变化**：ancestral 采样在 σ 大时注入更多噪声（探索），σ 小时几乎不加（收敛）；
- **为什么只首步要 log-prob**：后续 ODE 步 `sigma_up=0`，是确定性的，概率为 1（无梯度信号）；只有首步的随机转移有密度，才有策略梯度。
- 论文原话（为什么不全步随机）：`[§3.3]` "introducing stochasticity at all denoising steps leads to severe credit assignment issues"。

#### ② 奖励：`latent_reward` 的实现（不改像素）

源码（`grpo/reward_fn.py`），形状标注为 `[B,F,C,H,W]`：

```python
def latent_reward_fn(pred_latents, true_latents, vae,
                     decode_chunk_size=8, use_mae=True, use_cosine=True,
                     mae_weight=1.0, cosine_weight=1.0):
    """物理：预测未来与专家未来在潜空间的接近程度。
    L1 管结构（物体在哪、多大），cosine 管方向（整体变化趋势）。"""
    bsz, num_frames = pred_latents.shape[0], pred_latents.shape[1]
    all_mae, all_cos = [], []
    with torch.no_grad():                       # 奖励不回传梯度
        for i in range(0, bsz, decode_chunk_size):
            p = pred_latents[i:i+decode_chunk_size]      # [b,F,C,H,W]
            t = true_latents[i:i+decode_chunk_size]
            if use_mae:
                mae = F.l1_loss(p, t, reduction='none').mean(dim=[2,3,4])  # [b,F]
                all_mae.append(-mae)                                # 负 L1 作奖励
            if use_cosine:
                pf = p.view(p.shape[0], num_frames, -1)             # [b,F,C*H*W]
                tf = t.view(t.shape[0], num_frames, -1)
                cos = (F.normalize(pf, dim=-1) * F.normalize(tf, dim=-1)).sum(-1)  # [b,F]
                all_cos.append(cos)
    reward = mae_weight*torch.cat(all_mae) + cosine_weight*torch.cat(all_cos)  # [B,F]
    return reward                               # 逐帧奖励，训练时再按帧平均/求和
```

注意：repo 里另有 `pixel_reward_fn`——先 `vae.decode` 再算 MAE + **LPIPS**（感知奖励），用于需要像素感知差异的场景；默认的 `latent_reward_fn` 不解码，省算力。

#### ③ GRPO 更新：逐行注释

源码（`step1_train_svd_grpo.py`）：

```python
# 组归一化优势：同一观测的 G=8 条想象里，谁更接近专家
rewards = rewards.reshape(sample_batch_size, num_generations, num_frames)  # [B,G,F]
advantages = (rewards - rewards.mean(dim=1, keepdim=True)) \
             / (rewards.std(dim=1, keepdim=True) + 1e-8)                   # [B,G,F]
# ↑ 物理：没有 critic，用「同组平均」当基线——这就是 GRPO 与 PPO 的区别

# 重要性比：当前策略 vs 采样时策略，对同一条轨迹的概率比
ratio = torch.exp(log_prob - batch_log_probs)          # π_new / π_old，标量/张量
clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)  # 0.8~1.2
# 截断代理：min(未截断, 截断) = 悲观估计，防止单步更新过猛
unclipped_loss = -advantages * ratio                   # 优势>0 时希望 ratio 变大
clipped_loss   = -advantages * clipped_ratio
loss = torch.mean(torch.minimum(unclipped_loss, clipped_loss))
# 诊断指标（不参与反传）
approx_kl = 0.5 * torch.mean((log_prob - batch_log_probs) ** 2)        # 策略漂移
clipfrac  = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())  # 被截断比例
```

- **优势>0**：这条想象比组内平均更接近专家 → 提高其概率（ratio↑）；**优势<0** → 压低；
- **为什么取 min**：未截断项在 ratio 很大时仍线性增长，可能被单条样本带飞；截断项把它压住，取 min 得到保守更新（PPO 的标准做法）；
- **`clip_range=0.2`、`adv_clip_max=2.0`、`lr=1e-6`、`G=8`、`I=20 步`、`guidance=7.5`**：均来自 `scripts/train_calvin_svd_grpo.sh` 的实参；
- **`log_prob` 从哪来**：首步 SDE 的 `eas_step_with_logprob` 返回值（见 ①），后续 ODE 步没有 log-prob。

```python
# 训练主干（按源码结构）
def grpo_step(vpm, expert_latents, cond, G=8, steps=20, clip=0.2):
    z0 = randn_like(expert_latents)                 # 共享初始噪声 [B,F,C,H,W]
    trajs, logps = rollout_group(vpm, z0, cond, G, steps)   # 只首步 SDE
    r   = latent_reward_fn(torch.stack(trajs), expert_latents)  # [B,G,F]
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-8)
    loss = clipped_surrogate(vpm, trajs, logps, adv, clip)     # 见上
    vpm.update(loss)                                # 只更新 VPM；VAE/文本编码器冻结
```

### 2.3 EVA：IDM 动作平滑度奖励 + Flow-GRPO

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 把生成视频解码成关节角，惩罚抖动/急加速/超限——机械臂真实执行时最容易出问题的三种运动。 |
| 为什么这么做 | 视频看着合理不代表能动；用冻结 IDM 把「画面质量」翻译成「动作可执行性」，补上执行性 gap。 |
| 方法出处 | 论文 §4.2–4.3（Eq.3–7）；**无官方代码，依论文重建**；Flow-GRPO 谱系来自 GRPO 在扩散/流模型上的适配。 |
| 优势与代价 | 优势：奖励无需真机、物理可解释；代价：依赖 IDM 精度，平滑奖励可被「不动」骗过（用限幅惩罚堵）。 |

```python
def eva_reward(video, idm, vmax, amax, lam):
    """物理：q 为关节序列；v/a/j 是速度/加速度/jerk。
    平滑项抑制抖动，限幅项抑制超速。"""
    q = idm.decode(video)
    v = diff(q); a = diff(v); j = diff(a)
    return -(lam.j*huber(j) + lam.a*huber(a)
             + lam.v*relu(abs(v)-vmax).pow(2)
             + lam.a2*relu(abs(a)-amax).pow(2))

def flow_grpo_step(gen, videos, prompts):
    """物理：同一提示词多生成，谁动作更平滑就强化谁；KL 防止丢画面先验。"""
    rs  = [eva_reward(v, idm) for v in videos]
    adv = (rs - mean(rs)) / (std(rs) + 1e-8)
    gen.update(clipped_surrogate(adv, ratio) - beta * KL(gen, gen_ref))  # 仅 LoRA
```

### 2.4 CreFlow：LTL 稀疏奖励 + DiffusionNFT + corrective reflow

**输入 / 输出**：输入=rollout 视频、离线合成的 LTLf 规格、冻结的参考策略（Vidar）；输出=生成器梯度（监视器、IDM 冻结）。
**代码解读**

| 项 | 内容 |
|---|---|
| 物理意义 | 用线性时序逻辑检查「抓取→移动→放下」是否按顺序完成；违例帧就是不能学的区域。 |
| 为什么这么做 | 稀疏成功奖励下，失败 rollout 不能整段模仿；把失败样本推向同组成功样本的均值，给出物理可执行的改进方向。 |
| 方法出处 | 论文 §3–§4.3（Eq.2–9）；**无官方代码，依论文重建**；NFT 谱系来自 DiffusionNFT，reflow 来自 Rectified Flow。 |
| 优势与代价 | 优势：只用二值奖励就能训、更新被限制在违例区；代价：需要离线合成 LTL 规格，规格错了奖励就错。 |

#### 公式逐条：Eq.5–Eq.9 分别在说什么

| 公式 | 内容 | 物理意义 |
|---|---|---|
| **Eq.(5)** `(r,𝒱,𝒮)=ℳ_c(x0;y)` | LTL 监视器输出：二值奖励 `r∈{0,1}`、违例 witness `𝒱`（哪一帧违反哪条子句）、实体 mask `𝒮`（SAM3 跟踪的物体） | 「这一遍到底错在哪一帧、哪个物体上」——不是只给一个 0/1 |
| **Eq.(6)** `M = M_T ⊗ M_S` | 组共享掩码：`M_T=⋃` 失败子句的帧，`M_S=⋃` 全部 rollout 的实体 mask | 只在「出错的时空区域」加梯度；整组都成功时 `M≡0`，等价于组相对优势为 0（不更新） |
| **Eq.(7)** `L_CA-NFT(θ)=E[r‖M⊙(v⁺−v̂)‖²+(1−r)‖M⊙(v⁻−v̂)‖²]` | 带掩码的 NFT 损失；`v⁺=(1−β)v_old+βv_θ`（正分支），`v⁻=(1+β)v_old−βv_θ`（负分支） | 好样本：把速度场往「数据速度 `v̂`」拉；坏样本：把速度场**反射**着推离——用模型自身构造正负分支，不需要奖励模型或偏好对 |
| **Eq.(8)** `x̄₀⁺ := (1/|P|)Σ_{j∈P} x₀^(j)` | 组内**成功样本**的干净潜均值（`P`=成功集合） | 「同一指令下，成功的样子平均长什么样」——给失败样本一个具体的靠拢目标 |
| **Eq.(9)** `L_CR(θ)=E_{i∈N}‖M⊙(x̂₀^(i)−x̄₀⁺)‖²` | corrective reflow：失败样本的单步 `x0` 预测 `x̂₀=x_t−t·v_θ` 拉向 `x̄₀⁺` | 「别只告诉我错，告诉我该往哪走」；只对失败样本 `N`，且只在掩码区 |

**为什么 Eq.(8) 不能换成「最近的成功样本」**：论文原话（§4.3）用组内均值是「within-condition positive mean 的无偏估计」，而「nearest-positive selection」会引入 matching bias（挑到谁取决于度量，方差更大、容易过拟合到单条轨迹）。

**为什么 Eq.(7) 要取 min/反射而不是直接回归**：直接对失败样本回归会「学会失败」；反射分支让更新方向与坏样本相反。**为什么 Eq.(9) 只对失败样本**：成功样本已经在 `x̄₀⁺` 里，重复约束没有增益，还会压制多样性。

#### 代码（Eq.5–9 逐行对应）

```python
def ltl_monitor(video, spec):
    """Eq.5：返回 (r, witness, entity_masks)。
    r∈{0,1}；witness=违例的帧/子句；masks=SAM3 实体 mask（[T,H,W]）。"""
    return spec.check(frames(video))          # ℳ_c(x0;y)

def group_mask(witnesses, entity_masks):
    """Eq.6：组共享掩码 = 违例帧 ⊗ 相关实体。
    物理：只惩罚『错的那几帧、那几个物体』；整组全成功 → M=0，训练自动跳过。"""
    M_T = union_frames(witnesses)             # [T]
    M_S = union_entities(entity_masks)        # [T,H,W] 或 token 级
    return M_T.view(-1, 1, 1) * M_S           # [T,H,W]

def ca_nft_loss(vpm, x_t, v_hat, r, M, beta):
    """Eq.7：credit-aware NFT。
    v_hat = 数据速度（ε−x0）；v_θ = 模型输出；v_old = 采样时策略（stop-grad）。"""
    v_plus  = (1 - beta) * v_old + beta * v_theta        # 正分支：更靠近数据
    v_minus = (1 + beta) * v_old - beta * v_theta        # 负分支：反射离开坏样本
    loss_pos = r       * (M.unsqueeze(0) * (v_plus  - v_hat)).pow(2)   # r=1 的样本
    loss_neg = (1 - r) * (M.unsqueeze(0) * (v_minus - v_hat)).pow(2)   # r=0 的样本
    return (loss_pos + loss_neg).mean()

def corrective_reflow(vpm, rollouts, rewards, M, t):
    """Eq.8 + Eq.9：失败样本向组内成功均值纠正。"""
    # Eq.8：成功 rollout 的干净潜均值（同一指令下的『平均成功结局』）
    x0_pos = [r.x0 for r in rollouts if r.reward == 1]          # 每个 [C,F,H,W]
    x_bar_pos = torch.mean(torch.stack(x0_pos), dim=0)
    loss = 0.0
    for r in rollouts:
        if r.reward != 0:            continue                    # 只对失败样本 N
        x0_hat = r.x_t - t * vpm(r.x_t, t, r.cond)              # 单步反推 x0
        loss = loss + (M.unsqueeze(0) * (x0_hat - x_bar_pos)).pow(2).mean()   # Eq.9
    return loss

# 总目标（§4.3）：两项 + 到冻结参考 Vidar 的 KL，λ/β 论文未给
loss = ca_nft_loss(...) + lambda_cr * corrective_reflow(...) \
       + beta_kl * KL(gen, vidar_ref)
```

### 2.5 WorldVLN：自回归 WAM 的段级 GRPO

**输入 / 输出**：输入=在线模拟器、真实观测流、目标（goal）、同一 goal 下的多段 rollout；输出=自回归 WAM 的策略梯度。
**代码解读**

| 项 | 内容 |
|---|---|
| 物理意义 | 段末用真实观测替换模型想象的潜状态，防止长程想象误差累积（世界模型只有短程可信）。 |
| 为什么这么做 | 导航奖励稀疏且长程；拆成段、每段组内比较，能把「哪一段走得更好」变成可学信号。 |
| 方法出处 | 论文 §4.2–4.3（Eq.9–13）；**无官方代码，依论文重建**；段级 GRPO 谱系来自 GRPO 的 segment-level 变体。 |
| 优势与代价 | 优势：真观测回灌显著缓解漂移；代价：需要在线模拟器与真实观测流同步，λ/G/γ 数值未给。 |

#### 奖励 `r_j` 怎么算（Eq.9–12）：三项各管什么

| 项 | 公式（论文 §4.2） | 物理意义 | 少了它会怎样 |
|---|---|---|---|
| `r_traj` | `1/(1+‖a−a*‖)` | **轨迹项**：预测的动作/路点与专家动作 `a*` 的距离；越像越高 | 只学「到终点」，可能学会绕路/危险捷径 |
| `r_task` | `1/(1+终点距目标)` | **任务项**：段末位置离目标越近越高 | 只学「像专家」，换目标就失效 |
| `r_ref` | `log π_ref(a|s)` | **参考项**：参考策略（SFT/预训练策略）下该动作的对数概率 | 策略漂到参考分布外，泛化崩（相当于 KL 约束的奖励版） |

合成方式（Eq.12）：

`r_j = γ^{j−1} · (λ_traj·r_traj + λ_task·r_task + λ_ref·r_ref)`

- `j` 是段序号（第 1 段最早）；`γ^{j−1}` 让**越早的段权重越大**——后面的段还能纠，前面的段错了代价更高；
- `λ_*` 是三项权重，**数值论文未给**；距离用哪种范数（L1/L2）论文未逐字写明；
- `a*` 的来源：专家轨迹（或规划器给出的参考路径），具体定义见论文 §4.2。

#### `A` 是什么、有什么用

`A_j = (r_j − mean_group(r)) / (std_group(r) + ε)`

- **是什么**：段级**组相对优势**——把「这一段的奖励」减去「同一 goal 下所有段的平均奖励」，再除以标准差。GRPO 用它代替 critic/价值网络（PPO 需要学一个 value 基线，GRPO 用组均值当基线）。
- **有什么用**：给策略更新**定方向与幅度**：
  - `A_j > 0`：这一段比同组平均好 → 提高该段动作的概率；
  - `A_j < 0`：比平均差 → 降低概率；
  - 幅度由「好多少（以标准差为单位）」决定，组内比较消除了任务难度带来的绝对分数差异（同一个 goal 的段才可比）。
- **注意**：`A` 不是价值、不是回报，是**归一化后的相对优势**；它只出现在策略损失里，不参与环境交互。

#### 代码（含逐行注释）

```python
def worldvln_step(model, segs, goal, gamma, lam):
    """segs：同一 goal 下的 J 个段（每段含动作 a、段末状态 end）。
    物理：组内比较「哪段走得更好」，用真观测回灌避免长程漂移。"""
    R = []
    for j, seg in enumerate(segs, start=1):
        r_traj = 1.0 / (1.0 + dist(seg.a, seg.a_star))   # 像不像专家轨迹
        r_task = 1.0 / (1.0 + dist(seg.end, goal))       # 离目标多近
        r_ref  = logp_ref(seg.a, seg.s)                  # 参考策略锚定（KL 的奖励版）
        R.append(gamma**(j-1) * (lam.traj*r_traj
                                 + lam.task*r_task
                                 + lam.ref*r_ref))       # Eq.(12)
    R = torch.tensor(R)
    A = (R - R.mean()) / (R.std() + 1e-8)                # 组相对优势（无 critic）
    ratio = torch.exp(logp_new - logp_old)               # 段级重要性比
    loss = -torch.min(ratio * A,
                      torch.clamp(ratio, 1-0.2, 1+0.2) * A).mean()   # Eq.(13) 截断代理
    model.update(loss)
```

#### 与 VAMPO 的 GRPO 有什么不同

| | VAMPO | WorldVLN |
|---|---|---|
| 优化对象 | 视频预测器（VPM） | 自回归 WAM 的策略（整段动作） |
| 比较单位 | 一次去噪轨迹（`G` 条想象） | 一个段（`J` 段 rollout） |
| 奖励 | 与专家未来的潜距离 | 轨迹 + 任务 + 参考三项加权 |
| 基线 | 组内均值（G 条） | 组内均值（J 段） |
| 特殊点 | 只首步 SDE 才有 log-prob | 真观测回灌（防长程漂移） |

---

## 3. 闭环训练：六类实现

### 3.1 HILR：人机接力

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 只在「机器人快做错」的边界上由人示范，样本价值远高于随机演示。 |
| 为什么这么做 | 纯自主 rollout 会反复犯同样的错；人的介入点正是策略的分布边界，最值得学。 |
| 方法出处 | GigaBrain §3.2.2 Stage 3；**无官方代码，依论文重建**；谱系来自 DAgger（干预式数据采集）。 |
| 优势与代价 | 优势：样本效率高；代价：依赖人力和接管时机，切换抖动必须剔除。 |

```python
def hilr_episode(policy, env, human):
    """物理：切换瞬间动作会突变，必须剔除，否则策略学到不连续动作。"""
    traj, intervened = [], False
    while True:
        obs = env.observe()
        if human.takeover(obs):
            a, intervened = human.act(obs), True
        else:
            a = policy.act(obs)
        traj.append((obs, a, intervened))
        nxt, done = env.step(a)
        if done: break
    return scrub_switch_artifact(traj)
```

### 3.2 DAgger 式恢复数据（XPACE）

**DAgger 全称**：**Dataset Aggregation**（Ross, Gordon, Bagnell, AISTATS 2011）。

**为什么需要它（物理直觉）**：行为克隆只在专家轨迹上训练；部署时机器人一旦偏离，就会进入**专家从没去过的状态**，误差越滚越大（covariate shift）。DAgger 的核心是：**在策略自己会去到的状态上，向专家要标签**，再把这些状态-动作聚合回训练集。

**算法（Ross et al. 2011，标准 DAgger）**

```python
# 输入：专家策略 π*、初始演示数据 D、迭代轮数 N
# 输出：在「策略自己访问到的状态分布」上也能正确动作的策略 π
def dagger(expert, D, N):
    pi = train_bc(D)                        # ① 先行为克隆
    for i in range(N):
        states = []                         # ② 用当前策略跑，收集它自己访问到的状态
        for episode in rollouts(pi, env):
            for s in episode.states:
                states.append(s)
        labels = [expert.act(s) for s in states]      # ③ 在这些状态上问专家要动作
        D = D + list(zip(states, labels))             # ④ 聚合（Dataset Aggregation 的名字来源）
        pi = train_bc(D)                    # ⑤ 在聚合数据上重训
    return pi
```

- 关键点：**状态分布来自学生、标签来自专家**；`β_i` 混合（以概率 β 执行专家动作）只是稳定探索的变体；
- 物理意义：把「机器人实际会走到哪里」和「那里该怎么做」对齐，而不是只对齐「专家会走到哪里」。

**XPACE 的映射（工程版 DAgger）**

| DAgger 组件 | XPACE 的实现 |
|---|---|
| 学生状态分布 | 在专家轨迹附近**注入偏差**（inject deviation），模拟策略会偏离到的状态 |
| 专家标签 | 用**模拟器推演恢复路径**（deviation–recovery），得到「怎么回来」的动作 |
| 数据聚合 | 恢复数据只占约 **8%**，与专家演示混合 |
| 额外步骤 | **一致性过滤**：只保留物理一致的恢复轨迹（DAgger 原本没有） |

```python
def recovery_data(expert_traj, simulator, policy):
    """输入：专家轨迹；输出：deviation–recovery 恢复轨迹（用于混合微调）。
    物理：制造『机器人实际会偏离到』的状态，再教它怎么回来。"""
    sim = simulator.adapt_to(policy)              # 模拟器适配策略自己的视觉上下文
    recs = []
    for tau in expert_traj:
        s_dev = inject_deviation(tau, scale=delta) # ① 学生状态：人为偏离
        tau_rec = sim.rollout_recovery(s_dev)      # ② 专家标签：模拟器推演恢复动作
        if consistency_filter(tau_rec):            # ③ 过滤：只留物理一致的
            recs.append(tau_rec)
    return recs                                    # ④ 聚合：与专家数据混合微调
```

**证据与边界**：XPACE 摘要 + 项目页；恢复消融 61.7→86.7（**异套件**，不能与 68.3% 直接比）；无官方代码，以上为依论文描述的重建；`delta`、模拟器细节、过滤阈值论文未给。

---

### 3.3 失败数据闭环（FACT，**官方代码**）

**核心思想**：把「该模仿什么」与「该预测什么」分开——**失败轨迹的动作不学，但失败的后果（未来视频、进度）要学**。论文原话："FACT first proposes an action, then predicts the future video and task-progress value conditioned on the executed action."

**输入 / 输出**：输入=成功演示 `D_s` + 失败 rollout `D_f`（带 episode 结果与失败起点）；输出=同时会「出动作、预测未来、估进度」的因果 WAM。

#### 公式逐条（Eq.1–10）

| 公式 | 内容 | 物理意义 |
|---|---|---|
| **Eq.(1)** `p(a_{t:t+H}\|o_t,ℓ)` | 标准策略：观测+指令 → 动作块 | 动作生成 |
| **Eq.(2)** `p(o'_{t:t+K}, v_t(a_{t:t+H})\|o_t,ℓ,a_{t:t+H})` | **动作条件**的世界分支：给定动作，预测未来视频与进度值 | 「这个动作会导致什么后果」 |
| **Eq.(3)** `p_t = G_t/G_T ∈ [0,1]` | 归一化进度：累积进度/总进度；均匀奖励下 `p_t=t/T` | 跨不同长度 episode 可比的进度 |
| **Eq.(4)** `p(o', v(a_gt)\|o,ℓ,a_gt)` | teacher forcing：世界分支看**干净的真动作** `a_gt`，不看带噪的预测动作 | 把「动作模仿」与「后果预测」解耦 |
| **Eq.(5)** `z=[P_ref‖A_pred‖G_gt‖V_value‖I_future]` | token 顺序：前缀、预测动作、真动作、价值、未来 | 注意力掩码的依据 |
| **Eq.(6)** `v_t = p_{t+H}`（成功）/ `clip(p_{t+H}−λ_fail·1_fail, 0, 1)`（失败） | 失败时把进度目标**下调** `λ_fail`（=1） | 「做了这个动作会失败」→ 进度值应更低 |
| **Eq.(7)** 流匹配损失（逐模态） | 动作/价值/未来三路都用同一 flow-matching | 统一训练 |
| **Eq.(8)** `L_s = w_a L_a + w_v L_v + w_I L_I` | 成功样本：三路都学 | 学行为 + 后果 |
| **Eq.(9)** `L_f = w_v L_v + w_I L_I` | 失败样本：**去掉动作项**，只学价值与未来 | 失败教「后果」，不教「动作」 |
| **Eq.(10)** `a* = argmax_k V_θ(o,ℓ,a^(k))` | 可选：采样 N 个候选动作，用进度值打分选最优 | 推理期用价值做 best-of-N |

**注意力掩码（Figure 3 的要点）**：训练时 `V`、`I` 只 attend 干净的 `G`，不 attend 带噪的 `A`；`A` 也看不到 `G`。这样失败样本的未来/价值监督不会污染动作分支。

#### 代码（对应官方训练器）

```python
# 事实：FACT 训练器（world_action_model/trainer/wa_casual_trainer.py）
action_loss_mask = batch_dict.get('action_loss_mask')       # 1=成功可模仿, 0=失败
def _masked_mean(loss, mask):
    m = mask.reshape(-1)
    return (loss * m).sum() / m.sum().clamp_min(1e-8)        # 失败样本动作项被剔除（Eq.9）

def fact_train_step(batch, model):
    """物理：先动作、后后果。动作项只学成功；未来/价值项成功+失败都学。"""
    pred_action, pred_value, pred_future = model(batch)       # 共享 DiT，动作 adapter
    L_act = masked_mean(action_loss(pred_action, batch.a), batch.action_loss_mask)   # Eq.8/9
    L_val = mse(pred_value, batch.value_target)               # Eq.6 的进度目标，全样本
    L_fut = flow_mse(pred_future, batch.future_latents)
    return 20.0*L_act + 1.0*L_val + 1.0*L_fut                 # w_a=20, w_v=w_I=1（§3.3 实现细节）

# 两阶段推理（§3.3）
def fact_infer(obs, instruction, N=1):
    """Stage 1 只出动作；需要评估后果时进 Stage 2。"""
    a = denoise_action(obs, instruction, steps=20)            # Stage 1
    if N == 1: return a                                       # 动作单用，跳过世界分支
    cands = [denoise_action(obs, instruction) for _ in range(N)]  # 候选
    scores = [value_head(obs, instruction, c) for c in cands]     # Stage 2，Eq.10
    return cands[argmax(scores)]
```

**实现细节（论文 §3.3）**：Wan2.2-5B 初始化；动作 FFN 学习率 `2e-4`、骨干 `2e-5`；`w_a=20`、`w_v=w_I=1`；动作块 `H=48`；`λ_fail=1`；未来帧偏移 `[0, H/4, H/2, 3H/4, H]`；推理 20 步 flow-Euler。

**为什么有效（物理）**：成功数据让模型知道「怎么做对」，失败数据让模型知道「做错会看到什么、进度会掉多少」——后者的价值在于**推理时能识别坏动作**（Eq.10 的候选打分）。

---

### 3.4 DreamAvoid：混合数据价值模型 + 触发式 dreaming（**官方代码**）

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 给「当前画面 + 这段动作 + 它导致的未来若干帧」打分，分数高≈不会失败。 |
| 为什么这么做 | 全时想象太贵；只在关键阶段（trigger 判断）才花算力做梦，平时直接执行。 |
| 方法出处 | 仓库 `XianzheFan/DreamAvoid`：`DreamDojo-da/examples/value_model.py`、`openpi-da/agilex/agilex_infer_dream_trigger.py`（源码级）。 |
| 优势与代价 | 优势：失败/边界数据都能进价值模型，推理可控成本；代价：需要关键阶段标注与触发阈值调参。 |

```python
# 事实：ValueModel 源码
class ValueModel(nn.Module):
    """物理：DINOv2 编码当前帧与未来帧；value token 汇总后与动作/任务/状态拼接。"""
    def forward(self, obs, future, action, state, task_id):
        obs_tok = encoder(obs) + frame_emb[0]
        fut_tok = encoder(future) + frame_emb[1] + time_emb[:L]
        seq = transformer([value_token] + obs_tok + fut_tok)
        x = cat([seq[:, 0], proj(action), task_emb(task_id), proj(state)])
        return head(x).squeeze(-1)

def dream_avoid_step(policy, value_model, obs, state, task):
    """物理：未到关键阶段直接执行；到关键阶段才采样候选、梦未来、选最高价值。"""
    if not trigger(obs): return policy.act(obs)
    cands = [policy.sample_action(obs) for _ in range(8)]
    return max(cands, key=lambda a: value_model.score(
        obs, rollout_future(policy, obs, a), a, state, task))
```

### 3.5 τ0-WM：propose → RCS 排序 → 模拟器精炼（**官方代码**）

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 一次生成 N 个候选动作块；RCS 判断哪个不是偶然采样；模拟器推演谁的未来回报高。 |
| 为什么这么做 | 单次采样不可靠，重去噪一致性是「可信度」的廉价代理；模拟器进一步按回报截断执行步数。 |
| 方法出处 | 仓库 `sii-research/tau-0-wm`：`web_infer_utils/posttrain_taco_play_withTTC/TauPolicyTTC.py`（源码级）；谱系来自 best-of-N + MPPI 式评估。 |
| 优势与代价 | 优势：推理期即闭环、无需训练；代价：N 倍算力，模拟器误差会传导。 |

```python
def play(obs, state, N_proposals=8, RCS_timestep_ratios=[0.5]):
    """物理：RCS 高=更可信；模拟回报峰值决定执行几步。"""
    pred = pipeline.infer(obs, state, N_proposals=N_proposals, get_RCS=True)
    cands, rcs = pred['action'], pred['rcs']
    best = cands[rcs.max(dim=0).indices]
    if sim_refine:
        sim_rewards = sim_pipeline.infer(actions=cands, return_reward=True)
        if sim_rewards.max() < 0:
            best = re_predict_with_sim_latent(...)
        else:
            best = cands[best_idx, :argmax(sim_rewards)+1]
    return best
```

### 3.6 语言教练 relabel（CoachSI）

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 机器人失败常因「不知道该干哪一步」，而不是「不会动」。 |
| 为什么这么做 | 用冻结 VLM 把成功轨迹切段命名，等于补上「下一步」监督，且不动动作骨干。 |
| 方法出处 | CoachSI 摘要（长程 47.4→59.3）；**无公开代码，依论文重建**；谱系来自 hindsight relabeling。 |
| 优势与代价 | 优势：不改动作模型、成本低；代价：只解决指令接地，不解决技能本身。 |

```python
def coach_relabel(traj, vlm, task):
    """物理：VLM 给每段轨迹命名下一步，作为新的指令监督。"""
    steps = [vlm.describe(seg.frames, task, "next step?") for seg in segment(traj)]
    return [(seg.obs, seg.action, s) for seg, s in zip(segment(traj), steps)]
```

---

## 4. 六类闭环对照

| 类 | 数据从哪来 | 奖励/信号 | 更新什么 | 代码 |
|---|---|---|---|---|
| HILR | 人接管边界 | 无（模仿） | 策略 | 无公开 |
| DAgger 恢复 | 模拟器推演偏离-恢复 | 一致性过滤 | 策略 | 项目页 |
| 失败数据（FACT） | 真机失败 rollouts | 掩码 + 价值头 | 策略+价值 | [Bariona/FACT](https://github.com/Bariona/FACT) |
| 混合价值（DreamAvoid） | 成功/失败/边界 | 学出的价值模型 | 价值模型+推理选优 | [XianzheFan/DreamAvoid](https://github.com/XianzheFan/DreamAvoid) |
| 模拟器精炼（τ0-WM） | 候选动作 + 模拟器 | RCS + 模拟回报 | 不更新（推理选优） | [sii-research/tau-0-wm](https://github.com/sii-research/tau-0-wm) |
| 语言教练（CoachSI） | 成功轨迹 | VLM 命名 | 指令跟随 | 项目页 |

---

## 5. 工程清单

1. **奖励能否离线算**？能 → 先做离线 GRPO；不能 → 才考虑在线 rollout。
2. **有没有失败数据**？失败数据比成功数据稀缺，价值更高。
3. **防退化三件套**：KL/参考锚定、优势裁剪、混合 SFT 损失。
4. **评测**：成功率 + 完成时间 + 进度；报告相对 SFT 的增量。
5. **算力**：在线 RL ≈ 单次推理 × G（G=8、20 步约 160×）。

---

## 6. 坑

- 「伪代码」必须能落地：没有函数签名、输入输出、数据流的，只是步骤清单。
- 公式要映射代码行；论文没给的超参写「论文未给」，不补默认值。
- 自评数字：跨论文比较先看协议（trials、任务套件）。
- 奖励 hacking：平滑度可被「不动」骗过，EVA 用限幅惩罚堵。
- 闭环≠更好：世界模型偏差会被 RL 放大。
- 不要把 post-training 都当 RL：V-JEPA 2 的 post-training 是加动作条件的微调。

---

## 7. 术语附录：q 的 shape、LoRA、GRPO

### 7.1 `q` 的 shape（EVA 里的关节序列）

`q = idm.decode(video)`：`video` 形状 `[T_v, 3, H, W]`，解码后

- `q ∈ R^{T×D}`：`T` 个动作帧（通常与视频帧率对齐或降采样），`D`=关节维数；双臂 `D=14`（每臂 6 关节+1 夹爪），单臂 `D=7`；
- `v = diff(q) ∈ R^{(T−1)×D}`（速度）；`a = diff(v) ∈ R^{(T−2)×D}`；`j = diff(a) ∈ R^{(T−3)×D}`；
- 奖励 `R` 是标量：各阶差分先按关节平均，再加权求和。

**`T` 与 `D` 的具体数值 EVA 论文未给**（EVA 依 LVP 初始化、不同任务 D 不同）；能确定的是「逐帧、逐关节」的二维表。

### 7.2 LoRA：全称、原理、实现

**全称**：Low-Rank Adaptation（Hu et al., ICLR 2022，arXiv:2106.09685）。

**原理**：冻结预训练权重 `W0 ∈ R^{d×k}`，只学一个低秩增量 `ΔW = BA`：

`h = W0·x + (α/r)·B·A·x`，其中 `B ∈ R^{d×r}`、`A ∈ R^{r×k}`，`r ≪ min(d,k)`。

- 初始化：`A ~ 高斯(小方差)`、`B = 0` → 训练起点 `ΔW=0`，不破坏原模型；
- 只训练 `A、B`（参数量 `r(d+k)`，通常是原权重的 0.1–1%）；
- 推理时可合并：`W' = W0 + (α/r)·BA`，**零额外延迟**；不合并则相当于旁路。

```python
class LoRALinear(nn.Module):
    """物理：给冻结的线性层加一条低秩旁路，只学「任务增量」。"""
    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)  # [r,k]
        self.B = nn.Parameter(torch.zeros(base.out_features, r))        # [d,r]
        for p in base.parameters():
            p.requires_grad_(False)                    # 冻结原权重

    def forward(self, x):                              # x [..., k]
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)   # [..., d]
```

**为什么在本领域用**：VLA/WAM 骨干动辄数 B 参数，全量微调显存扛不住；LoRA 只训旁路、可并存多任务适配器。**代价**：秩 `r` 限制表达能力，分布偏移大时不如全量微调。
**代码位置**：VAMPO `scripts/train_calvin_svd_grpo.sh` 的 `train_args.use_lora=False, lora_rank=16`（可切）；EVA「只更新生成器 LoRA」；DreamAvoid 的 `openpi-da` 也带 `lora.py`。

### 7.3 GRPO：全称、原理、实现

**全称**：Group Relative Policy Optimization（DeepSeekMath，arXiv:2402.03300）。

**原理**：PPO 需要 value/critic 估计基线；GRPO **用同组样本的均值/标准差代替 critic**：

1. 对同一条件采样 `G` 条输出，得到奖励 `r_1..r_G`；
2. 组内标准化优势：`Â_i = (r_i − mean(r)) / (std(r)+ε)`；
3. 最大化截断代理目标：`J = E[ (1/G) Σ_i min(ρ_i Â_i, clip(ρ_i,1−c,1+c) Â_i) ] − β·D_KL(π‖π_ref)`，其中 `ρ_i = π_θ(o_i)/π_old(o_i)`；
4. 参考策略 KL 防止跑偏（VAMPO 用「只首步 SDE」的结构性约束替代显式 KL）。

```python
def grpo_loss(logp, logp_old, rewards, clip=0.2, beta=0.0, has_ref=False, logp_ref=0.0):
    """物理：组内比较——比同组平均好的样本被强化，差的被压制。
    无 critic：均值/标准差就是基线。"""
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)     # 组相对优势
    ratio = torch.exp(logp - logp_old)                            # 重要性比
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    loss = -torch.min(unclipped, clipped).mean()                  # 悲观（取 min）
    if beta > 0 and has_ref:
        loss = loss + beta * (logp - logp_ref).mean()             # 可选 KL 惩罚
    return loss
```

**与 PPO 的差异**：省掉 value 网络（显存/工程更简单）、基线随组自适应；**代价**：每组要 `G` 条完整采样（算力 ×G），组内奖励方差大时优势噪声大；`G` 太小则基线不稳。
**在 VAMPO 里的实例**：`G=8`、`clip=0.2`、`adv_clip_max=2.0`、`lr=1e-6`、首步 SDE 才有 `log_prob`、20 步去噪、guidance 7.5。


## 8. 参考

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

## 9. 缩写表（全站简写对照）

| 简写 | 全称 | 一句话 |
|---|---|---|
| **WAM** | World Action Model | 联合建模未来状态与动作的具身基础模型 |
| **VLA** | Vision-Language-Action (model) | 视觉-语言-动作模型 |
| **VDM** | Video Diffusion Model | 视频扩散模型 |
| **DiT** | Diffusion Transformer | 扩散 Transformer 骨干 |
| **MoT** | Mixture-of-Transformers | 多专家共享注意力的 Transformer 组合 |
| **VAE** | Variational Autoencoder | 变分自编码器（图像↔潜） |
| **IDM** | Inverse Dynamics Model | 逆动力学模型：视频→动作 |
| **VPM** | Video Prediction Model | VAMPO 里被 RL 优化的视频预测器 |
| **AGM** | Action Generation Model | VAMPO 里冻结的动作生成器 |
| **GRPO** | Group Relative Policy Optimization | 组相对策略优化（无 critic，用组均值当基线） |
| **PPO** | Proximal Policy Optimization | 近端策略优化（需要 critic） |
| **KL** | Kullback–Leibler divergence | 分布散度，用作「别偏离参考策略」的约束 |
| **LoRA** | Low-Rank Adaptation | 低秩适配：冻结主干、只训低秩旁路 |
| **LTL / LTLf** | (Linear) Temporal Logic (over finite traces) | 线性时序逻辑（有限轨迹版），用来写任务约束 |
| **NFT** | Negative-aware FineTuning（DiffusionNFT） | 用正/负分支做扩散模型偏好式微调 |
| **HILR** | **Human-in-the-Loop Rollout** | 人机接力 rollout：自主执行 + 人工纠正，剔除切换伪影 |
| **DAgger** | **Dataset Aggregation**（Ross et al., 2011） | 在线聚合「策略访问到的状态 + 专家动作」的模仿学习 |
| **RAMP** | Reinforcement leArning via world Model-conditioned Policy | GigaBrain 的「世界模型条件化策略」RL 框架 |
| **RECAP** | （π0.6*/π0.5 的 advantage-conditioned RL 框架，论文称 RAMP 的特例） | 用稀疏优势（0/1）条件化的策略改进 |
| **SDE / ODE** | Stochastic / Ordinary Differential Equation | 随机/常微分方程（采样器的随机步/确定步） |
| **MPPI** | Model Predictive Path Integral | 采样型 MPC，用加权样本更新控制分布 |
| **CEM** | Cross-Entropy Method | 交叉熵方法（采样+精英更新） |
| **MPC** | Model Predictive Control | 模型预测控制 |
| **RCS** | Re-denoising Consistency Score | 重去噪一致性分数（τ0-WM 的候选可信度） |
| **ACVS** | 论文已在 §VI 使用；全称在已获取正文中未展开 ⚠️ | τ0-WM 的密集任务进度评估模块 |
| **LAR** | 论文称「局部动作重查询」（Local Action Re-query）⚠️ | τ0-WM 的局部重查询 |
| **TTC** | Test-Time Computation | 测试时计算（τ0-WM 的 TTC 推理） |
| **EEF** | End-Effector | 末端执行器 |
| **SAM3** | Segment Anything Model 3 | 分割基础模型（CreFlow 的实体 mask） |
| **VLM** | Vision-Language Model | 视觉语言模型 |
| **TTT** | Test-Time Training | 测试时训练（WAM-TTT） |
| **OPD** | On-Policy Distillation | 在线策略蒸馏（WAM-OPD） |
| **KV** | Key-Value (cache) | 注意力键值缓存 |

> ⚠️ 标注项：全称未在论文已获取部分出现，按论文用词记录；不臆造展开。
