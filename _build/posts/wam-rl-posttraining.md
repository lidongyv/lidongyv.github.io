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
| `z_t` | `[C_z,h,w]` | 当前帧的 VAE 潜 | `z_t = E_vae(o_t)` |
| `z_future_i` | `[C_z,h,w]` | 未来帧潜，`i∈{12,24,36,48}` 步后 | `E_vae(o_{t+i})`，仅训练期 |
| `v_t` | 标量 | 价值估计≈「负的期望剩余步数」 | 价值头 `W.value(s_t)` |
| `Ψ(·)` | — | 轻量 MLP：把标量价值/本体向量投影到潜通道 | 论文只说「MLP 对齐维度」，结构未给 |
| `s_t` | `[C_z+1+C_p,h,w]` | 增强状态：潜 + 价值图 + 本体图（通道拼接，Eq.6） | `s_t=[z_t;Ψ(v_t);Ψ(p_t)]` |
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
    v_map = psi_v(v_t).view(1, h, w)        # 标量 → 单通道图
    p_map = psi_p(p_t).view(C_p, h, w)      # MLP 投影到 C_p 通道
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

**HILR（Stage 3）到底算什么**：它不是公式，是**数据采集过程**——输入策略/环境/人类接管策略，输出带干预标记的轨迹。

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

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 用线性时序逻辑检查「抓取→移动→放下」是否按顺序完成；违例帧就是不能学的区域。 |
| 为什么这么做 | 稀疏成功奖励下，失败 rollout 不能整段模仿；把失败样本推向同组成功样本的均值，给出物理可执行的改进方向。 |
| 方法出处 | 论文 §4.1–4.3（Eq.5–9）；**无官方代码，依论文重建**；NFT 谱系来自 DiffusionNFT，reflow 来自 Rectified Flow。 |
| 优势与代价 | 优势：只用二值奖励就能训、且更新被限制在违例区；代价：需要离线合成 LTL 规格，规格错了奖励就错。 |

```python
def ltl_monitor(video, spec):
    """物理：返回 0/1 + 违例发生在哪帧（witness），供损失只惩罚违例区。"""
    return spec.check(frames(video))            # Eq.(5)

def creflow_loss(rollouts, spec, ref_policy, beta):
    """物理：成功样本正常学；失败样本向组内成功均值纠正。"""
    rs = [ltl_monitor(v, spec)[0] for v in rollouts]
    M  = union_witness_masks(rollouts, spec)    # 违例帧/实体掩码
    loss_nft = token_masked_nft(rollouts, rs, M)                    # Eq.(7)
    x_pos = mean([v for v, r in zip(rollouts, rs) if r == 1])
    loss_cr = mse(M * (x0_hat(failed) - x_pos))                     # Eq.(9)
    return loss_nft + loss_cr + beta * KL(gen, ref_policy)
```

### 2.5 WorldVLN：自回归 WAM 的段级 GRPO

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 段末用真实观测替换模型想象的潜状态，防止长程想象误差累积（世界模型只有短程可信）。 |
| 为什么这么做 | 导航任务奖励稀疏且长程；拆成段、每段组内比较，能把「哪一段走得更好」变成可学信号。 |
| 方法出处 | 论文 §4.2–4.3（Eq.9–13）；**无官方代码，依论文重建**；段级 GRPO 谱系来自 GRPO 的 segment-level 变体。 |
| 优势与代价 | 优势：真观测回灌显著缓解漂移；代价：需要在线模拟器与真实观测流同步。 |

```python
def segment_rollout(model, env, goal, gamma):
    """物理：走一段，段末用真观测替换想象潜（真观测回灌）。"""
    seg, seg_obs = [], env.reset()
    for j in range(J):
        a = model.act(seg_obs); seg.append(a)
        obs = env.step(a)
        seg_obs = obs if env.is_real else model.predict(obs)   # 回灌
    return seg

def worldvln_step(model, seg, goal, gamma):
    """物理：轨迹像专家 + 终点离目标近 + 别偏离参考策略，按 γ^{j-1} 衰减。"""
    r = sum(gamma**(j-1) * (lam_traj * r_traj(seg[j], goal)
                            + lam_task * r_task(seg[j], goal)
                            + lam_ref * log_ref(seg[j]))
            for j in range(1, J+1))                 # Eq.(9-12)
    A = (r - mean_group(r)) / (std_group(r) + 1e-8)
    model.update(clipped_surrogate(A, ratio))        # Eq.(13)
```

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

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 人为制造「偏离专家轨迹」的状态，再用模拟器推演怎么回来。 |
| 为什么这么做 | 真实部署必然偏离；与其等真机犯错，不如在模拟器里批量生成恢复样本。 |
| 方法出处 | XPACE 摘要 + 项目页（恢复消融 61.7→86.7，异套件）；**无公开代码，依论文重建**；谱系来自 DAgger + 域随机化。 |
| 优势与代价 | 优势：恢复数据仅占 ~8% 就能显著提升；代价：模拟器必须适配策略自己的视觉上下文，否则恢复轨迹不可迁移。 |

```python
def recovery_data(expert_traj, simulator, policy):
    """物理：注入偏差→推演恢复→只留物理一致的样本。"""
    sim = simulator.adapt_to(policy)              # 适配策略视觉上下文
    recs = []
    for tau in expert_traj:
        s_dev = inject_deviation(tau, scale=delta)
        tau_rec = sim.rollout_recovery(s_dev)
        if consistency_filter(tau_rec): recs.append(tau_rec)
    return recs
```

### 3.3 失败数据闭环（FACT，**官方代码**）

**代码解读**
| 项 | 内容 |
|---|---|
| 物理意义 | 失败 episode 的动作不能学（会教坏机器人），但它的后果必须学——失败是负样本，也是成本信号。 |
| 为什么这么做 | 失败数据稀缺且昂贵；只用成功数据会让价值函数看不到「坏在哪里」。 |
| 方法出处 | 仓库 `Bariona/FACT`：`world_action_model/trainer/wa_casual_trainer.py`（源码级）；act-then-imagine 因果序。 |
| 优势与代价 | 优势：失败数据直接进训练、无需额外标注；代价：掩码对象选错（比如把失败样本的动作也学）会把策略拉坏。 |

```python
# 事实：FACT 训练器源码
action_loss_mask = batch_dict.get('action_loss_mask')

def _masked_mean(per_sample_loss, mask):
    """物理：m=1 可模仿、m=0 失败；失败样本动作损失置零。"""
    m = mask.reshape(-1)
    return (per_sample_loss * m).sum() / m.sum().clamp_min(1e-8)

action_loss = _masked_mean(action_loss_raw.flatten(1).mean(dim=1), action_loss_mask)
value_loss  = ((value_pred - value_target) ** 2).mean()   # 全部样本都学
```

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

## 7. 参考

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
