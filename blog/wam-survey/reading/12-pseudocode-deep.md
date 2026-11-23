# WAM 伪代码（工程级）：输入输出 + 模块 + 执行流（12）

> 目的：给出**可直接照着实现**的伪代码。每条包含：①模块表（职责/可训性）②输入表（字段/形状/含义）③输出 ④细粒度伪代码 ⑤数据流说明。
> 依据纪律：`[论文 §X]`=论文小节；`[代码 file]`=官方仓库 verbatim；`[README]`=官方 README；`[未给]`=已获取范围内论文未给出，禁止脑补填充。
> 伪代码约定：`[B,H,D]` 等为张量形状；`·` 表示调用；`→` 表示数据流向。

---

## 1. WAMotor：投机推理（代码级，最细）

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| `predict()` | 包一层模型前向：按候选数复制 KV-cache 并批量预测速度 | 否（推理） | `[代码 speculative/action.py]` |
| `scheduler.step()` | Euler 积分一步 | 否 | `[代码]` |
| `threshold()` | 按步长生成每步验收阈值 | 否 | `[代码]` |
| `draft()` | 用 anchor velocity 线性外推候选轨迹 | 否 | `[代码]` |
| `verify()` | 一次 batched 前向验证全部候选 | 否 | `[代码]` |
| `accept_prefix()` | 取最长合法前缀并决定回退 | 否 | `[代码]` |

**输入**

| 字段 | 形状 | 含义 | 出处 |
|---|---|---|---|
| `latents_action` | `[B, H, D_a]` | 当前动作潜（初始为噪声） | `[代码]` |
| `infer_timesteps_action` | `[N]` | N 个去噪时间点 | `[代码]` |
| `infer_deltas_action` | `[N]` | 相邻时间步差 Δ_k | `[代码]` |
| `context` | `[B, L, C]` | 文本+proprio 条件序列 | `[代码]` |
| `context_mask` | `[B, L]` | 条件有效位 | `[代码]` |
| `video_kv_cache` | `list[dict]`（每层 k/v） | 视频专家预填的 KV | `[代码]` |
| `attention_masks` | `[S, S]` 或列表 | 动作→视频的注意力掩码 | `[代码]` |
| `video_seq_len` | int | 视频 token 数 | `[代码]` |
| `w` | int，默认 10 | 投机窗口上限 | `[代码]` |
| `epsilon` | float，默认 1e-2 | 阈值系数 | `[代码]` |

**输出**：`latents_action`：`[B, H, D_a]`——完成全部 N 步后的动作潜（交给动作解码器）。

**伪代码**

```python
def speculative_denoise_action(a, t, delta, ctx, ctx_mask, video_kv, masks, video_len, w=10, epsilon=1e-2):
    # ---------- 模块 0：阈值表（时间步感知） ----------
    ref = median(abs(delta))                          # 参考步长
    thr = epsilon * (ref / abs(delta).clamp_min(eps)) ** 2   # 步长越大阈值越紧

    # ---------- 模块 1：首步强制标准 Euler（推迟投机） ----------
    u = predict(a, t[0], ctx, ctx_mask, video_kv, masks, video_len)   # [B,H,D]
    a = euler_step(a, u, delta[0])
    m, u_cached = 1, None

    # ---------- 模块 2：循环投机 + 验证 + 提交 ----------
    while m < len(t):
        if u_cached is None:
            u_cached = predict(a, t[m], ctx, ctx_mask, video_kv, masks, video_len)
        if m == len(t) - 1:                            # 最后一步：普通 Euler 收尾
            return euler_step(a, u_cached, delta[m])

        end = min(m + w, len(t) - 1)                   # 本窗口最多看 w 步
        k = end - m                                    # 候选数

        # 2.1 草拟：draft_j = a + u_cached * cumsum(delta)
        draft = [a + u_cached * sum(delta[m:m+j]) for j in 1..k]   # [k*B,H,D]

        # 2.2 验证：一次 batched 前向（KV 按 k 扩展）
        v_hat = predict_batched(draft, t[m+1:end+1], ctx, ctx_mask,
                                video_kv.expand(k), masks, video_len)  # [k*B,H,D]

        # 2.3 逐候选逐步 MSE 判定，接受最长合法前缀
        ok = [mse(v_hat[j], u_cached) < thr[m+1+j] for j in 0..k-1]
        acc = longest_prefix(ok)                       # 0..k

        if acc == k:                                   # 全接受：直接推进，缓存速度
            return_ready = True
            a, u_cached, m = draft[-1], v_hat[-1], end
            continue

        # 2.4 部分接受：在边界做一次标准 Euler 回退
        u_fb = v_hat[acc-1] if acc > 0 else u_cached
        a = euler_step(a, u_fb, delta[m+acc])
        m, u_cached = m + acc + 1, (v_hat[0] if acc == 0 else None)

    return a
```

**数据流说明**
- `predict()` 内部对 `video_kv_cache` 做 `expand`（候选数 × batch），因此验证是**一次前向**而非 k 次；
- `thr` 只依赖 `delta`（先验已知），不依赖模型输出——所以「步长越大越严」是离线可算的；
- 所有步只动 `latents_action`，视频 KV 只读不改（步骤复用）。

---

## 2. WA-JEPA：GDRF + FSCF 动作精炼（无候选搜索）

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| `DINOv2` | 图像→稠密特征 | 否（冻结） | `[论文 §3.2]` |
| `FusePool` | 多层特征融合 + query pooling → scene tokens | 是 | `[论文 §3.2]` |
| `TemporalAttn` | 整合 L 帧历史 | 是 | `[论文 §3.2]` |
| `BiCrossAttn` | 最新帧 ↔ 目标编码，得 context/goal memory | 是 | `[论文 §3.2]` |
| `Proposal T_r` | 提案更新（chunk 间 self-attn + 读 [M,G̃,Ẑ]) | 是 | `[论文 §3.3]` |
| `Decoder D + Range B` | 解出动作 chunk，映射到合法范围 | 是 | `[论文 §3.3]` |
| `WorldModel W_ψ` | 预测每个 chunk 边界的场景 + proprio | 是（共享） | `[论文 §3.3]` |
| `GDRF F_r + gate γ_r` | 残差修正（预测 vs 目标） | 是 | `[论文 §3.3]` |

**输入**

| 字段 | 形状 | 含义 | 出处 |
|---|---|---|---|
| `obs` | `[L, 3, H_img, W_img]` | 最近 L 帧图像 | `[论文 §3.1]` |
| `proprio` | `[L, d_p]` | 每帧本体状态 | `[论文 §3.2]` |
| `goal_img` | `[3,H_img,W_img]` | 目标图像 | `[论文 §3.1]` |
| `R` | int | 精炼阶段数（README：4 提案 / 3 次更新） | `[README]` |
| `H, b, d_a` | int | chunk 数、每 chunk 控制步数、动作维 | `[未给]` |

**输出**：`A[R]`：`[H, b*d_a]`——最终阶段提案（执行用）；中间阶段仅用于精炼。

**伪代码**

```python
def wa_jepa_policy(obs, proprio, goal_img, R=4):
    # ---------- 编码（§3.2） ----------
    Z = [FusePool(DINOv2(o)) for o in obs]            # [L, S, d] 每帧 S 个 scene token
    Zp = temporal_attn(cat(Z, proj(proprio)))         # 融合 proprio 的历史表示
    M, G_hat = bi_cross_attn(Zp[-1], G=G_src)         # context memory [L(S+1), d] / goal memory [S,d]
    Z0, G_bar = Z[-1], pool(goal_img)                 # 预测起点 / discrepancy 参照

    # ---------- 初始提案（§3.3） ----------
    Q = chunk_queries + time_emb + cond(pool(Z0), pool(Z0) - pool(G_bar))
    Z_pred = repeat(Z0, future_steps)                 # FSCF 上下文初始化

    for r in 1..R:                                    # 固定阶段，无候选搜索
        # (a) 提案更新：self-attn 协调 chunk + cross-attn 读 [M, G̃, Ẑ]
        Q_tilde = T_r(Q, [M, G_hat, Z_pred])
        A_r = B(D(LN(Q_tilde)))                       # [H, b*d_a] 合法范围动作

        # (b) 后果预测：时序掩码保证 h 只看前 h 个 chunk
        Z_pred, P_hat = W_psi(M, Z0, A_r)             # Z_pred [H,S,d] / P_hat [H,d_p]

        # (c) GDRF：逐 chunk 用「目标-预测」残差更新提案
        for h in 1..H:
            residual = G_bar - mean(Z_pred[h])         # 与目标场景差
            U = concat(Q_tilde[h], mean(Z_pred[h]), residual, proj(P_hat[h]))
            Q[h] = Q_tilde[h] + sigmoid(gamma_r) * F_r(U)

    return A_r                                         # 只执行最终阶段
```

**数据流说明**：`Z0` 同时喂世界模型与 FSCF；`G_bar` 只用于 GDRF 的残差；每一 stage 都从**同一待执行计划**上修正（不推进环境时间），最终只执行一次。

---

## 3. LoopedWAM：权重共享循环 + 弹性蒸馏

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| `prelude`（3 blocks） | 每专家入口层 | 是 | `[论文 §3.2]` |
| `shared_core`（6 blocks） | 循环复用的核心层 | 是 | `[论文 §3.2]` |
| `coda`（3 blocks） | 每专家出口层 | 是 | `[论文 §3.2]` |
| `MoT attn` | 两专家共享注意力 | 是 | `[论文 §3.2]` |
| `step_cond_MLP` | 把步长 d=1/N 嵌入 timestep 调制（零初始化） | 是 | `[论文 §3.5]` |
| `teacher Φ_T` | 冻结教师采样器（N_T 步 Euler） | 否 | `[论文 §3.5]` |

**输入**：`z_t`（噪声动作 chunk `[1, H, D_a]`）、`context`（文本+proprio）、`K_a`（动作 loop 数 ∈{1,2,4}）、`N`（采样步数 ∈{2,5,10}）、`K_v`（视频 loop 数，训练固定）。
**输出**：学生更新后参数（动作专家 + step_cond_MLP）；推理时输出动作 chunk。

**伪代码**

```python
# 前向：单专家 = prelude(x) -> repeat(core(x), K 次) -> coda(x)
def expert_forward(x, K, d_step, ctx):
    h = prelude(x)
    for _ in range(K):
        h = core(h)                          # 权重共享；层数语义 = L(K)=3+K*6+3
    return coda(h, step=d_step, ctx=ctx)     # step_cond_MLP 注入 1/N

def loopedwam_forward(video_lat, action_noise, ctx, K_v, K_a):
    z_v = expert_video(video_lat, K_v, d=0, ctx)     # 每 query 只算一次，缓存各层 KV
    z_a = expert_action(action_noise, K_a, d=1/N, ctx) # 每采样步跑 K_a 次
    # late alignment：动作 loop r 读视频 loop σ(r)=max(1, K_v-K_a+r)
    return z_a

# 训练：elastic self-distillation（一次更新）
def distill_step(S, teacher, student):
    (K_a, N) = balanced_next(S)              # 每 |S| 次更新访问每个设置一次
    z0 = randn; x, tau = sample_real_action()
    L_local = 0
    for j in 0..N-1:
        v = student.step(z0, K_a=K_a, d=1/N, tau_j)
        z_next_T = teacher.euler(z0, tau_j -> tau_j+1)   # 教师从学生状态出发
        L_local += mse(v, stop_grad((z_next_T - z0) / (tau_next - tau_j)))
        z0 = stop_grad(z0) + (tau_next - tau_j) * v if j < N - m else z0 + (tau_next - tau_j) * v
    L = L_local/N + l_end*mse(student_chunk, teacher_chunk) + l_grip*softplus_margin(grip) + l_anchor*flow_loss(x)
    update(student, L)                       # 只更新动作专家 + step_cond_MLP
```

**数据流说明**：视频专家每 query 只前向一次并缓存 KV（省 7.4× 显存的前提）；动作专家按 `K_a` 复用 core 层数；蒸馏时学生**从自身状态**出发问教师，暴露自身误差。

---

## 4. MetaWAM：频域打包 + slot 路由

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| `Pack(q)` | 信号写入固定频带系数槽（无参数） | 否 | `[论文 §4.2]` |
| `2D-IDCT` / `2D-DCT` | 系数 ↔ 空间潜（每通道正交变换） | 否 | `[论文 §4.2]` |
| `Unpack(F̂)` | 从固定槽读出信号（无学习解码头） | 否 | `[论文 §4.2]` |
| `VDM denoiser Dθ` | 共享去噪主干（含 8 帧潜序列） | 是 | `[论文 §4.1]` |
| `slot router Φ` | 按任务开/关频带，生成细掩码 + 粗掩码 | 否（确定性） | `[论文 §4.4]` |

**输入**：`q=(task, ρ_t, a, ρ_{t+H}, v)`（任务 one-hot、当前 proprio、动作 chunk、未来 proprio、价值）；`obs` 三相机当前帧；`mask_mode ∈ {policy, world, value}`。
**输出**：`z_out^m`（输出 meta-frame）→ `Unpack` 得 `â, ρ̂_{t+H}, v̂`；`z^w_{t+H}` 等未来视觉帧。

**伪代码**

```python
def pack(q):                                 # 模块 1：信号 → 稀疏系数
    F = zeros(C, H, W)                       # 无学习参数
    for s in [task, rho_t, a, rho_tH, v]:
        band = allocate_band(s)              # 互不相交 channel band
        write_row0(F[band], flatten(group(s)))   # 只写 k_H=0 行固定槽
    return F

def unpack(z_m):                             # 模块 2：输出 meta-frame → 信号
    F = dct2_per_channel(z_m)
    return read_slots(F)                     # â, ρ̂_{t+H}, v̂（系数即归一化信号）

def forward(obs, q, mask_mode):
    z_in  = idct2_per_channel(pack(q_cond_only))       # 输入 meta-frame（task+ρ_t 恒为条件）
    z_out = Dtheta(z_in, obs_frames=6, cond=z_in, mask=M_slot)  # 8 帧序列联合去噪
    F_hat = dct2_per_channel(z_out)                    # 可微，无解码头
    return read_slots(F_hat), z_out.future_frames

def router(mask_mode):                       # 模块 3：逐任务开带
    open = {task, rho_t}
    if mask_mode == "world": open |= {a}
    if mask_mode == "value": open |= {a, rho_tH}
    fine = band_mask(open)                   # 细掩码（任务语义全在此）
    coarse = derive_coarse(fine)             # 单通道粗掩码（仅保持字节级兼容）
    return fine, coarse
```

**数据流说明**：输入侧 `z_in` 恒为 clean 条件（tokenizer 预留 blank slot，零成本）；输出侧同一 meta-frame 按路由承载 action/未来 proprio/value；视觉 6 帧走粗掩码，meta-frame 走细掩码。

---

## 5. Bind-WAM：绑定不确定性路由（何时想象）

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| 离线标注器 | Qwen3-VL-32B 识别实体 + SAM 3.1 传播 mask + DINOv3 测身份一致性 | 否（离线） | `[论文 §III-B]` |
| `Router f_θ` | 6.65M ViT，预测 `U_t∈[0,1]` | 是 | `[论文 §III-C]` |
| 共享 WAM | 视频/动作专家 + 混合 Transformer，两条掩码路径 | 是 | `[论文 §III-D]` |
| 路由规则 | `U_t≥τ_U or ΔU_t≥τ_Δ → joint` | 否 | `[论文 §III-E]` |

**输入**：`o_t`（当前观测）、`ℓ`（指令）、`U_{t-1}`（上一决策点不确定性）。
**输出**：动作 chunk（direct 或 joint 路径）。

**伪代码**

```python
# 离线：给每帧打 binding 失效标签（只在训练前跑一次）
def label_episode(traj, instruction):
    entities = qwen3_vl(instruction, frames=uniform(traj, 32))   # 被操作物体 + 锚点
    masks = sam31_prompt(entities).propagate(traj)
    feats = dino_v3(masks)
    y = {}
    for r, e in entities:
        bad = (iou(masks[r], ref_mask[r]) < 0.3) or (not visible(e)) or (cos(feats[r]) < 0.9)
        y[r] = bad
    return max(y.values())                                       # 帧级标签 y_t

# 训练路由器
def train_router(o_t, l, y_t):
    U_hat = router(o_t, l)                                       # 6.65M ViT
    return weighted_bce(U_hat, y_t) + lam * (U_hat - y_t)**2     # [论文 Eq.5]

# 推理：双路径共享 WAM
def act(o_t, l, U_prev, tau_U, tau_delta):
    U_t = router(o_t, l)
    dU = abs(U_t - U_prev)
    mode = "joint" if (U_t >= tau_U or dU >= tau_delta) else "direct"
    if mode == "direct":
        return wam(o_t, l, mask=M_direct)      # 动作只见当前帧（h=0）
    return wam(o_t, l, mask=M_joint)           # 联合去噪未来 H 帧潜 + 动作
```

**数据流说明**：离线标签器（Qwen/SAM/DINO）**推理时完全不需要**；`U_t` 与 `ΔU_t` 双阈值决定是否付出「想象未来」的代价。

---

## 6. Elastic-WAM：时空精炼需求场

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| 冻结 WAM 主干 | 视频/动作联合去噪 | 否 | `[论文 §3]` |
| 需求场 `B0(z|s)` | Beta 分布 `(m0,c0)`：时序精炼曲线 | 是 | `[论文 §3.2]` |
| 空间分数 `Δu_g` | 64 组区域优先级（stop-grad 防互扰） | 是 | `[论文 §3.2]` |
| 全局偏移 `b0` | 总需求振幅 | 是 | `[论文 §3.2]` |
| 决策器 | WHEN / HOW FAR / WHERE 三个函数 | 否 | `[论文 §3.2]` |

**输入**：`s_k`（当前精炼状态：已精炼 token 集合 + 时间位置），视频 token 数 `NF=196`，组数 `G=64`，`n_g∈{1,2,4}`。
**输出**：下一步精炼的三项决策（是否 commit、步长 δ、刷新区域 M*）。

**伪代码**

```python
def need_field(s_k):                           # 模块：需求场求值
    D = sum(n_g / NF * sigmoid(b0(s_k) + stopgrad(du_g(s_k))) for g)
    m0, c0 = net_temporal(s_k)                 # 时序参数
    return D, Beta(m0, c0)                     # 总需求 + 形状

def step(s_k, eps=1e-6, eta=0.5):
    D, B0 = need_field(s_k)
    # WHEN：累积继续概率 ≥ η 触发 World Commit
    Ck = D
    F = 1 - F_prev * clamp(Ck, eps, 1.0)
    if F >= eta or s_k == last:  return "COMMIT"
    # HOW FAR：取 Beta 众数当步长（上下限夹住）
    z = mode(B0)
    delta = 0.05 + z * (min(0.20, sigma_k - sigma_floor) - 0.05)
    # WHERE：精确 token 预算下选最优组
    M_star = topk_groups(score = n_g * du_g(s_k), budget = 96)
    return "REFINE", delta, M_star

def rollout(dense_steps=1):
    s = dense_joint_denoise(steps=dense_steps)      # 先做 1 步 dense
    while True:
        act = step(s)
        if act == "COMMIT": break
        s = sparse_refine(s, delta=act.delta, region=act.M_star)  # 只刷新选中 token
    return continue_action_denoising(cached_video_feats)
```

**数据流说明**：三步决策全部来自同一个需求场（`D` 给 WHEN、Beta 众数给步长、`Δu_g` 给位置）；精炼只刷新选中 token，但注意力上下文保持完整；commit 后复用缓存的视频表征继续动作去噪。

---

## 7. WATO：可微轨迹优化（训练用 TO，部署免 TO）

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| 生成器 `q_φ` | joint WAM，输出 (Ã, Z̃) | 是 | `[论文 §3.1]` |
| 世界模型 `R_θ` | 潜空间前向动力学 | 否（预训练） | `[论文 §2]` |
| 投影 `Π` | 动力学约束下的轨迹优化（只训练时跑） | — | `[论文 §3.3]` |
| 隐式微分 | 把任务损失穿过 TO 解回传给 φ | — | `[论文 Abstract]` |

**输入**：`c=(z_t, g)`（当前潜 + 目标潜）、`τ̃=(Ã,Z̃)`（生成提案）。
**输出**：训练时—更新后的 φ；部署时—直接生成的候选轨迹（世界模型排序）。

**伪代码**

```python
def wato_train_step(c, A_real, Z_real):
    eps = randn_like; s = randn()
    A_tilde, Z_tilde = flow_sample(q_phi, c, eps, s)          # 生成提案
    # 投影：在动力学约束下求 (A,Z) 同时贴近提案与目标
    def objective(A, Z):
        return J(Z, g) + lam_A * dist(A, A_tilde) + lam_Z * dist(Z, Z_tilde)
    A_star, Z_star = solve_constrained(objective, constraint=Z == R_theta(z_t, A))  # TO
    # 隐式微分：把任务损失 J(Z_star, g) 对 φ 求梯度（穿过 argmin 解）
    loss = J(Z_star, g) + imitation_loss(q_phi, A_real, Z_real)
    update(phi, dloss_dphi_via_implicit_diff(loss, A_star, Z_star, c))
    return

def wato_deploy(c, K=8):
    cands = [flow_sample(q_phi, c) for _ in range(K)]           # 免 TO，只生成候选
    return argmin_traj(cands, key=lambda traj: J(R_theta(z_t, traj.A), g))
```

**数据流说明**：TO 只出现在训练图的中间（`Π` 的输出带梯度），部署时换成「生成 K 候选 + 世界模型打分」；因此比 CEM 快 11.5×、世界模型评估少 600×。

---

## 8. Rolling-WAM：滚动窗口错峰去噪

**模块表**

| 模块 | 职责 | 可训 | 出处 |
|---|---|---|---|
| 视频/动作双专家 | 共享注意力的 MoT | 是 | `[03b 笔记]` |
| 噪声调度 `σ(τ)` | 分 chunk 噪声：`σ(τ)=ρτ/[1+(ρ−1)τ]`，ρ=5 | 否 | `[03b 笔记]` |
| 滚动窗口 | K=16 步/chunk、W=5 窗口、N=10 步/chunk | 否 | `[03b 笔记]` |

**输入**：`{chunk_{i−W+1..i}}`（滑动窗口内已执行/待执行 chunk）、当前观测；**输出**：下一 chunk 的 16 步动作（稳态每周期只 refine 2 步）。

**伪代码**

```python
def rolling_infer(window, obs, K=16, W=5, N=10, rho=5):
    # 窗口内每个 chunk 的噪声等级随时间错峰
    for j, chunk in enumerate(window):                      # j=1..W
        σ = rho * tau(j) / (1 + (rho - 1) * tau(j))          # 近端噪声低、远端高
        chunk.add_noise(σ)
    # 只 refine 当前 chunk 与即将接棒的 chunk（其余冻结）
    for step in range(2):                                    # 稳态每周期 2 步
        video, action = denoise_joint(window, obs, steps=N)
    window.pop(0); window.append(new_chunk(action))
    return action[:K]                                        # 执行 16 步
```

**数据流说明**：错峰噪声让「远端 chunk 保持高噪声可继续演化、近端已接近干净可直接执行」；动作只依赖窗口内视频与自身，跨 chunk 动作互不 attend（因果）。

---

## 附：术语与出处对照

| 记号 | 含义 | 首次出现 |
|---|---|---|
| `[B,H,D_a]` | batch × 动作 horizon × 动作维 | WAMotor [代码] |
| `w / epsilon` | 投机窗口 / 验收阈值系数 | WAMotor [代码] w=10, ε=1e-2 |
| `M / G̃ / Z0 / Ḡ` | 上下文记忆 / 目标记忆 / 起点场景 / 目标池化 | WA-JEPA [论文 §3.2] |
| `K_v / K_a / d` | 视频循环数 / 动作循环数 / 步长嵌入 | LoopedWAM [论文 §3.2/§3.5] |
| `F / z^m` | 稀疏系数张量 / meta-frame | MetaWAM [论文 §4.2] |
| `U_t / ΔU_t / τ` | 绑定不确定性 / 变化量 / 阈值 | Bind-WAM [论文 §III-E] |
| `D(s) / B0 / Δu_g` | 总需求 / 时序曲线 / 区域分数 | Elastic-WAM [论文 §3.2] |
| `Π / Fθ(zt)` | 投影算子 / 可行集 | WATO [论文 §3.3] |
| `σ(τ) / K / W / N` | 滚动噪声 / chunk 长 / 窗口 / 步数 | Rolling-WAM [03b 笔记] |
