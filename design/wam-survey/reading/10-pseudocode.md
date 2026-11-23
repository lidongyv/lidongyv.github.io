# WAM 伪代码提取 · 10 个机制

> 生成：2026-11-21 ｜ 代理：pseudocode 提取线（wam-survey-project 阅读线扩展）｜ 任务：为 10 个 WAM 机制写伪代码。

## 说明与纪律

**依据分级**（每个伪代码块标注其一）：
- **A｜论文原 Algorithm**：论文自带 Algorithm box，逐字转写（标注 Algorithm 编号 + 所在小节）。不改变步骤语义。
- **B｜结构化改写**：论文无 Algorithm box，按方法节公式/描述结构化改写执行流（标注「按 §X.X 结构化（非论文原 Algorithm）」）。禁止编造论文没有的步骤；论文未写明的数值/维度标注「原文未写（已获取范围）」。
- **C｜未能获取（L1）**：abs→html→ar5iv 各尝试一次（共两次）仍取不到正文，仅给公式要点，标 L1。

**取文链**：arxiv.org/abs/<id> → arxiv.org/html/<id>（最新版）→ ar5iv.labs.arxiv.org/html/<id>。fetch 失败重试一次；两次失败降级 C。

**格式约定**：
- 每个伪代码块前：论文名 / arXiv 号 / 小节定位 + 依据分级标注。
- 每个伪代码块后：2-4 行「关键行依据」——哪一步对应原文哪句/哪个公式，verbatim 引用（小节号 + 引文）。
- 变量与论文原文一致；循环/分支清晰、可直接读执行流；每条 ≤70 行。
- 第 9、10 条（Rolling-WAM 2609.30247 / FACT 2608.10232）的 verbatim 引文取自已有 L2 笔记 reading/03b-action-decision-wave2.md（该笔记为 L2 级正文核实，引文直接引用）。

**环境限制**：shell 出网被沙箱拦截（实测 Invoke-WebRequest 失败：「基础连接已经关闭」），全部网络取证走 web_fetch；HTML 页可能截断，截断位置逐篇标注。

---

## 1. WAMotor — Accelerating World Action Models via Speculative Inference under Dual Nonuniform Denoising Dynamics（arXiv:2611.25413，cs.RO，2026-11-19）

- 定位：主循环 = §4.1（Preliminaries: Speculative Inference）+ §4.3（Stage I）+ §4.4（Stage II）；依据分级：**B｜按 §4.1–§4.4 结构化（非论文原 Algorithm 转写）**。注：论文附录 B.5「Pseudocode for the Integration of WAMotor in Joint-WAMs」为 Algorithm 所在，但 arxiv HTML 截断于 §4.3 中段（"…uses u1" 处），未取到；验证阈值公式已与官方代码交叉核实（github.com/Overdue-Lin/WAMotor，src/wamotor/models/wan22/speculative/action.py、joint.py）。

```
Algorithm: WAMotor 投机推理主循环（动作流 speculative Euler denoising）
输入: 初始噪声动作 a_0^s = z ~ N(0,I)（H×D）；时刻 t_0..t_{N-1}，Δt_k = t_{k+1} − t_k；
      条件 c（观测/语言/本体/未来表征，WAM 中随去噪演进）；窗口上限 w；验证尺度 ε
输出: 去噪动作 a_N^s

 1: m ← 0；u ← None
 2: ▷ Stage I（§4.3）Reliable Anchor Velocity Selection：先做一步标准 Euler，延迟投机
 3: u_0 ← v_θ(a_0^s, t_0; c_0)
 4: a_1^s ← a_0^s + Δt_0 · u_0                    ▷ 普通 Euler 更新，不投机
 5: m ← 1
 6: ▷ Stage II（§4.4）Time-Step-Aware Verification：按积分步长缩放验收阈值
 7: Δ_ref ← median(|Δt_1|, …, |Δt_{N-1}|)
 8: ε_k ← ε · (Δ_ref / |Δt_k|)² ，k = 1..N−1
 9: while m < N−1:
10:     if u = None: u_m ← v_θ(a_m^s, t_m; c_m^s)          ▷ 计算 anchor velocity
11:     if m = N−1: return a_N^s ← a_{N−1}^s + Δt_{N−1} · u_{N−1}
12:     q ← min(w, N−1−m)                                  ▷ 投机窗口上限
13:     ▷ 用 anchor velocity 起草候选（Eq. 1）
14:     for j = 1..q:  â_{m,j} ← a_m^s + u_m · Σ_{i=0}^{j−1} Δt_{m+i}
15:     ▷ 批量验证：一次模型调用评估全部候选
16:     {v̂_{m,j}} ← v_θ({â_{m,j}}, {t_{m+j}}; {ĉ_{m,j}})
17:     r_{m,j} ← u_m − v̂_{m,j}                            ▷ velocity residual
18:     accept_j ← max-per-token MSE(r_{m,j}) < ε_{m+j}    ▷ 每个 token 都必须通过
19:     ℓ ← 最长连续接受前缀长度（j=1..ℓ 全过，j=ℓ+1 拒绝）
20:     if ℓ = q:                                          ▷ 整窗接受：直接推进 q 步
21:         a_{m+q}^s ← â_{m,q}；u ← v̂_{m,q}；m ← m+q
22:     else if ℓ > 0:                                     ▷ 部分接受：回退一步标准 Euler
23:         a_{m+ℓ}^s ← â_{m,ℓ}
24:         a_{m+ℓ+1}^s ← a_{m+ℓ}^s + Δt_{m+ℓ} · v̂_{m,ℓ}
25:         m ← m+ℓ+1；u ← None
26:     else:                                              ▷ 首候选即拒绝
27:         a_{m+1}^s ← a_m^s + Δt_m · u_m
28:         m ← m+1；u ← v̂_{m,1}   ▷ v̂_{m,1} 恰在新状态处算出，复用为下一 anchor
29: return a_N^s
```

**关键行依据**：
- 行 3–5（延迟一步再投机）：§3.1 verbatim「Given this nonuniform velocity smoothness, WAMotor defers speculation until after the first standard Euler update, using the velocity at the updated state as the initial anchor.」（代码注释同义："Always take the first transition with ordinary Euler integration."）
- 行 14（起草）：§4.1 Eq.(1) verbatim「â_{m,j} = a_m^s + u_m Σ_{i=0}^{j−1} Δt_{m+i}, j=1⋯q_m」。
- 行 16–19（批量验证与最长前缀提交）：§4.1 verbatim「a batched model call evaluates v̂_{m,j}=v_θ(â_{m,j},t_{m+j};ĉ_{m,j}) for all candidates, yielding velocity residuals r_{m,j}=u_m−v̂_{m,j}... the longest accepted prefix, of length ℓ, is committed」；Def. 4.1「∥·∥_A² denotes maximum per-token MSE over the blocks, which ensures the acceptance criterion applies to every token.」
- 行 6–8、18（阈值随步长缩放）：Intro verbatim「Time-Step-Aware Verification rule based on the quadratic relationship between squared velocity residuals and integration time step size, tightening verification for larger steps and relaxing it for smaller ones」；§4.4 正文在截断区，阈值公式按官方代码逐行核实：`thresholds = ε * (Δ_ref / |Δt_k|)²`（action.py）。理论保证：Theorem 4.3「Trajectory δ-error tolerance can be given by a weighted sum of anchor velocity reuse residuals plus the propagated conditioning-error contribution.」

**取文记录**：arxiv HTML v1 成功但截断（§4.3 中段）；ar5iv 未收录（重定向回 arxiv）。Joint-WAM 版为同一循环作用于耦合 video+action 状态，验证只判动作流 MSE、每步重置首帧 latent（joint.py）。

---

## 2. WA-JEPA — Goal-Aware Predictive Feedback Action Refinement for a World–Action Model（arXiv:2611.19183，cs.CV，2026-11-14）

- 定位：动作精炼循环 = §3.1（公式化）+ §3.2（场景-目标公共表示）+ §3.3（GDRF+FSCF 双路径反馈，Eq.1–3）；依据分级：**B｜按 §3.1–§3.3 结构化（非论文原 Algorithm；已获取范围内无 Algorithm box）**。截断位置：§4 开头（"This setting examines predictive feedback..." 句处）。R 数值：论文截断区未取到，代码 README 实锤「four proposals with three feedback updates」→ **R=4（4 提案、3 次反馈更新；H=5 chunk × b=5 步）**（github.com/Dovisya/WA-JEPA README，Short-horizon "Shared defaults"）。

```
Algorithm: WA-JEPA 动作精炼循环（proposal → prediction → 双路径反馈，无候选搜索）
输入: 观测历史 {o_τ, p_τ}_{τ=t−L+1}^t、目标图 o_g；精炼阶段数 R（固定）
输出: 执行动作序列 A^(R)（仅最终阶段提案；A ∈ R^{H×b·d_a}：H 个 chunk，每 chunk b 个 d_a 维控制）

 1: ▷ §3.2 场景与目标的公共特征空间（共享 DINOv2 + 融合/池化参数）
 2: Z_τ ← QueryPool(Fuse(DINOv2(o_τ)))，τ = t−L+1..t   ▷ 每帧 S 个 scene tokens
 3: 嵌入标准化 proprio 为额外 token → 时序注意力整合观测历史
 4: (M, G̃) ← BiCrossAttn(最新历史帧表示, G)   ▷ 上下文记忆 M、目标记忆 G̃
 5: 保留 Z_0（世界模型预测起点）与 G（discrepancy 参照）
 6: ▷ §3.3 初始提案：chunk queries + 时序嵌入 + 条件信号
 7: Q^(1) ← chunk queries + 时序嵌入 + cond(pool(Z_0), pool(Z_0) − pool(G))
 8: Ẑ^(0) ← 沿未来时间轴重复 Z_0                    ▷ FSCF 上下文初始化
 9: for r = 1..R:                                   ▷ 固定数量前向阶段，无候选搜索
10:     ▷ (a) 提案更新：自注意力协调 chunk 间，交叉注意力读 [M, G̃, Ẑ^(r−1)]（Eq.1；FSCF 在此注入）
11:     Q̃^(r) ← T_r( Q^(r), [M, G̃, Ẑ^(r−1)] )
12:     A^(r) ← B( D( LN(Q̃^(r)) ) )                  ▷ 共享解码器 + 合法动作范围映射
13:     ▷ (b) 预测提案后果（Eq.2）：共享世界模型预测每个 chunk 边界的场景+本体
14:     (Ẑ^(r), P̂^(r)) ← W_ψ(M, Z_0, A^(r))           ▷ 时序掩码：预测 h 只看前 h 个 chunk
15:     ▷ (c) GDRF 逐 chunk 残差修正（Eq.3）
16:     for h = 1..H:
17:         U_h^(r) ← [Q̃_h^(r), Ẑ̄_h^(r), Ḡ − Ẑ̄_h^(r), E_p(P̂_h^(r))]   ▷ 场景 token 平均后与目标比较
18:         Q_h^(r+1) ← Q̃_h^(r) + σ(γ_r) · F_r(U_h^(r))               ▷ 可学习门控残差
19: 执行 A^(R)（部署只取最终阶段解码的提案；新观测到来后开启下一次决策）
```

**关键行依据**：
- 行 11–12（提案更新与 FSCF 注入）：§3.3 Eq.(1) verbatim「Q̃^{(r)} = T_r(Q^{(r)}, [M, G̃, Ẑ^{(r−1)}])」「A^{(r)} = B(D(LN(Q̃^{(r)})))」；FSCF：「FSCF supplies the preceding proposal's predictions as this additional context.」
- 行 14（后果预测）：§3.3 Eq.(2) verbatim「(Ẑ^{(r)}, P̂^{(r)}) = W_ψ(M, Z_0, A^{(r)})」；「Temporal masks restrict prediction h to the first h action chunks... predictions across future times can be computed in parallel.」
- 行 16–18（GDRF）：§3.3 Eq.(3) verbatim「U_h^{(r)} = [Q̃_h^{(r)}, Ẑ̄_h^{(r)}, Ḡ−Ẑ̄_h^{(r)}, E_p(P̂_h^{(r)})]」「Q_h^{(r+1)} = Q̃_h^{(r)} + σ(γ_r)F_r(U_h^{(r)})」；「A learned gate σ(γ_r)... controls the correction magnitude.」
- 行 9、19（固定阶段、无候选搜索、同一计划迭代）：§1 verbatim「The process uses a fixed number of forward stages, making predictions part of deployment-time action optimization without candidate-action search.」「refine the same plan before execution」；§3.3「only the final-stage action sequence is used for execution after refinement.」

**取文记录**：arxiv HTML v1 成功；§1–§3 全文取到、§4 起截断。R 数值未取到（截断区），代码仓库 github.com/Dovisya/WA-JEPA 可补。

---

## 3. LoopedWAM — Looped World Action Models for Elastic Inference（arXiv:2611.25595，cs.CV，2026-11-19）

- 定位：elastic self-distillation 训练 = §3.5（正文 + Eq.5/6；论文在 §3.5 引用「Algorithm 1」总结一次更新，但 Algorithm box 恰在截断点之后未取到）；依据分级：**B｜按 §3.5 结构化（非论文原 Algorithm 转写；Algorithm 1 box 未取到，按 §3.5 文字 + Eq.5/6 重建）**。设置集 𝒮：K_a ∈ {1,2,4} × N ∈ {2,5,10}（§1）；m（端到端回传步数）与 λ 权重数值在截断区未取到。

```
Algorithm: LoopedWAM-E elastic self-distillation（一次训练更新）
前置: 教师 = 训练好的 LoopedWAM（冻结，N_T 步 Euler 采样器）；学生 = 教师拷贝，动作专家可训；
     视频专家/VAE/文本编码器/proprio 投影冻结且与教师共享（一次视频前向供两者）；
     零初始化 step-condition MLP：把步长 d=1/N 的嵌入加进动作专家的 timestep 调制
输入: 设置集 𝒮 = {(K_a, N)}（K_a ≤ K 为动作 loop 数，N 为采样步数）；平衡序列
输出: 更新后学生 θ（单 checkpoint 覆盖所有 (K_a, N) 运行点）

 1: (K_a, N) ← 平衡序列下一个设置     ▷ 每 |𝒮| 次更新恰好访问 𝒮 中每个设置一次
 2: 采样噪声 z_0（学生与教师共享）；采样真实动作 x 与 flow time τ（供 ℒ_anchor）
 3: ℒ_local ← 0
 4: for j = 0..N−1:                          ▷ 学生 N 个宏步，τ_{j+1} = τ_j − 1/N
 5:     v̂_j ← Student(z_j; K_a, d=1/N, τ_j)          ▷ 动作专家跑 K_a 个 loop
 6:     z_{j+1}^T ← Φ_T(z_j; τ_j→τ_{j+1})   ▷ 冻结教师细 Euler（步长 1/N_T），从学生自身状态出发
 7:     ℒ_local += ‖v̂_j − sg[(z_{j+1}^T − z_j) / (τ_{j+1} − τ_j)]‖²    ▷ Eq.5；sg 截断教师侧梯度
 8:     if j < N−m: z_{j+1} ← sg(z_j) + (τ_{j+1}−τ_j)·v̂_j   ▷ 前 N−m 步：状态起点截断梯度
 9:     else:       z_{j+1} ← z_j + (τ_{j+1}−τ_j)·v̂_j       ▷ 最后 m 步：端到端回传
10: ℒ_local ← ℒ_local / N
11: ℒ_end   ← MSE(学生最终 chunk 的前 H_x 个执行动作, 教师从同一噪声出的 chunk 对应动作)
12: ℒ_grip  ← softplus margin(上述 H_x 个动作的夹爪指令)
13: ℒ_anchor← Eq.(1) flow loss：学生以 K_a=K、d=0 对真实动作 x 回归速度场
14: ℒ ← ℒ_local + λ_end·ℒ_end + λ_grip·ℒ_grip + λ_anchor·ℒ_anchor     ▷ Eq.6
15: 用 ℒ 更新学生（仅动作专家 + step-condition MLP 参数）
```

**关键行依据**：
- 行 1（平衡序列）：§3.5 verbatim「Settings are drawn from a balanced sequence that visits each setting of 𝒮 once per |𝒮| updates.」
- 行 5–7（局部目标：教师从学生自身状态出发）：§3.5 Eq.(5) verbatim「ℒ_local = (1/N) Σ_{j=0}^{N−1} ‖v̂_j − sg[(Φ_T(z_j; τ_j→τ_{j+1}) − z_j)/(τ_{j+1}−τ_j)]‖²」；「Following the teacher from the student's own states, rather than along the teacher's trajectory, exposes the student to its own errors and trains it to correct them.」
- 行 8–9（梯度截断）：§3.5 verbatim「Gradients flow through all loops of a step, but the state is detached at the start of steps j ≤ N−m, so only the last m steps are back-propagated end to end.」
- 行 11–14（终点/夹爪/锚点项）：§3.5 verbatim「An endpoint loss ℒ_end (mean squared error) matches the H_x executed actions of the student's final chunk to the teacher's chunk from the same noise, ℒ_grip applies a softplus margin to the gripper command of those actions, and ℒ_anchor applies Eq. 1 on real actions to the student at K_a=K and d=0」；Eq.(6)「ℒ = ℒ_local + λ_end ℒ_end + λ_grip ℒ_grip + λ_anchor ℒ_anchor」。

**取文记录**：arxiv HTML v1 成功；§1–§3.5 全文取到，§3.5 末尾 Algorithm 1 的描述句截断（"…and evaluates the teacher at its fine steps for the local targets an…"）。代码/模型：github.com/anhdao69/LoopedWAM（未打开）。

---

## 4. MetaWAM — From Video Frames to Robot Meta-Frames（arXiv:2611.14365，cs.CV，2026-11-10）

- 定位：频率打包 = §4.2（Eq.2/3）；slot-aware routing = §4.4；8 帧潜序列 = §4.1 Eq.(1)「z = (z^m_in, z^w_t, z^p_t, z^s_t, z^m_out, z^w_{t+H}, z^p_{t+H}, z^s_{t+H})」；依据分级：**B｜按 §4.2/§4.4 结构化（非论文原 Algorithm；TOC 无 Algorithm 小节）**。截断位置：§5.1 中段。

```
块 A：Distribution-Aware Spectral Lifting —— 信号打包进统一 meta-frame（§4.2）
输入: 归一化机器人信号 q = (task, ρ_t, a, ρ_{t+H}, v)；固定频带划分
      F = [F_task, F_ρt, F_a, F_ρt+H, F_v]（潜通道维的不相交分区，每信号一带）
输出: 空间潜 meta-frame z^m

 1: F ← 0（稀疏系数张量，C×H×W）
 2: for 每个信号 s ∈ {task, ρ_t, a, ρ_{t+H}, v}:
 3:     F_s ← 把展平/分组后的信号值写入其通道带的 k_H=0 行上的固定系数 slot（其余置 0）
 4: z^m ← 𝒯_2D^{-1}(F)      ▷ 每通道正交 2D 逆 DCT，系数分布到 H×W 网格
 5: 输入 meta-frame z^m_in：task + ρ_t 恒为 clean 条件（复用 tokenizer 预留空白 slot）
 6: 输出 meta-frame z^m_out：a / ρ_{t+H} / v 按任务路由成为条件或去噪目标

读取（Unpack，解析式、无学习解码头）:
 7: F̂ ← 𝒯_2D(ẑ^m_out)                    ▷ 对去噪后输出 meta-frame 做空间 2D DCT
 8: (â, ρ̂_{t+H}, v̂) ← Unpack(F̂)          ▷ 固定 slot 读出并恢复形状；系数即归一化信号

块 B：Slot-Aware Task Routing —— 同一 meta-frame 承载策略/世界模型/值预测（§4.4）
输入: 样本任务指示 (m_π, m_w, m_v)（policy / world-model / value 样本）
输出: 带级掩码 M_slot 与粗掩码 M_coarse

 1: M_slot ← Φ(m_π, m_w, m_v)             ▷ 确定性：open 一个带=clean 条件，close=去噪目标
 2: if 策略样本 (m_π=1):    open {task, ρ_t}；a、ρ_{t+H}、v 全为目标
 3: if 世界模型样本 (m_w=1): 再 open {a}；ρ_{t+H} 仍为目标
 4: if 值样本 (m_v=1):      再 open {ρ_{t+H}}；仅 v 为目标
 5: M_coarse ← 由 M_slot 确定性派生的单通道粗掩码
 6: 视觉 6 帧走 M_coarse（原视频去噪接口，不含任务语义）; meta-frame 走 M_slot
 7: 监督: ℒ = ℒ_diff + λ_act·ℒ_act + λ_prop·ℒ_prop + λ_val·ℒ_val + λ_off·ℒ_off   ▷ §4.5 Eq.5
 8: ℒ_off ← E[mean_{C,H,W} |(1−S) ⊙ F̂|]   ▷ 惩罚已分配 slot 之外的系数；λ_off = 10^{-4}
```

**关键行依据**：
- 块A 行 1–3（固定频带 slot）：§4.2 Eq.(2) verbatim「F = [F_task, F_ρt, F_a, F_ρt+H, F_v]」；「We construct F = Pack(q) by flattening or grouping each signal and directly writing its values into fixed coefficient slots on the k_H=0 row of its channel band; all unused entries are zero... The allocation is fixed across training and inference and introduces no learned parameters.」
- 块A 行 4、7–8（提升与解析读取）：§4.2 verbatim「We obtain the spatial meta-frame z^m = 𝒯_2D^{-1}(F) by applying an orthonormal 2D inverse DCT per channel.」；Eq.(3) verbatim「F̂ = 𝒯_2D(ẑ^m), (â, ρ̂_{t+H}, v̂) = Unpack(F̂)」；「The selected coefficients directly represent the normalized signals, so no further inverse DCT is applied.」
- 块B 行 1–4（逐任务开带）：§4.4 verbatim「M_slot = Φ(m_π, m_w, m_v), where Φ deterministically opens a band (clean conditioning) or closes it (denoising target)」；「a policy sample opens only the task and current-proprioception bands; a world-model sample additionally opens the action band, with the future state still a target; a value sample further opens the future-proprioception band, leaving only value as a target.」
- 块B 行 5–6（双层掩码）：§4.4 verbatim「we deterministically derive a single-channel coarse mask for the original video-denoising path... The fine mask carries all task information; the coarse mask carries none and exists only to keep byte-level compatibility with the backbone.」；行 7–8：§4.5 Eq.(5)(6)，λ_off=10^{-4}。

**取文记录**：arxiv HTML v1 成功；§1–§5.1 中段取到。无 Algorithm box（已获取范围）。代码：github.com/showlab/MetaWAM（未打开）。

---

## 5. Bind-WAM — Binding-Uncertainty-Guided Selective Future Imagination（arXiv:2611.15368，cs.RO，2026-11-11）

- 定位：离线标注+预测器训练 = §III-B/III-C（Eq.2–5）；推理路由 = §III-D/III-E（Eq.6–9）；依据分级：**B｜按 §III-B–§III-E 结构化（非论文原 Algorithm；TOC 无 Algorithm 小节）**。截断位置：§IV-B Table II 内（Bind-WAM 自身行处）。

```
块 A：binding uncertainty 离线标注 → 轻量预测器训练（§III-B/C）
 1: ▷ 离线标注（仅用完整演示；Qwen3-VL-32B / SAM 3.1 / DINOv3 ViT-L/16）
 2: 每条演示均匀采样 32 帧；Qwen 识别任务相关角色实体（被操作物 + 关系锚点）
 3: SAM 3.1 从实体文本 prompt 生成候选实例 mask 并沿轨迹传播
 4: for 每角色 r:
 5:     IoU_r ← Qwen 验证参考 mask 与候选轨迹的参考帧重叠度
 6:     v_t^r ← 当前可见性；s_t^r ← 当前帧与参考帧 DINO 特征的 cosine 相似度
 7:     y_t^r ← 1 if (IoU_r < 0.3) or (v_t^r = 0) or (s_t^r < 0.9) else 0    ▷ Eq.2
 8: y_t ← max_r y_t^r                                                    ▷ Eq.3 帧级标签
 9: ▷ 预测器训练（仅用当前观测+指令；标注模型不参与推理）
10: U_t ← f_θ(o_t, ℓ)，U_t ∈ [0,1]                                      ▷ Eq.4
11: ℒ_bind ← BCE_w(Û_t, y_t) + λ·(Û_t − y_t)²                          ▷ Eq.5 类加权 BCE + Brier

块 B：选择性未来想象（推理时路由，共享 WAM 两路径）（§III-D/E）
输入: o_t, e_t, ℓ；上一时刻不确定性 U_{t−1}；阈值 τ_U、τ_Δ
输出: 动作 chunk a_{t:t+H}

 1: U_t ← f_θ(o_t, ℓ)
 2: ΔU_t ← |U_t − U_{t−1}|                                              ▷ Eq.8
 3: if U_t ≥ τ_U  or  ΔU_t ≥ τ_Δ:                                       ▷ Eq.9
 4:     走 π_joint：联合去噪未来视觉 latent + 动作 chunk
 5:         动作 token 可见全部视频 token（M_joint(Z^a, Z_h^v)=1, h=0..H）
 6: else:
 7:     走 π_direct：仅从当前观测直接去噪动作
 8:         动作 token 仅可见当前帧（M_direct(Z^a, Z_h^v)=1 if h=0 else 0）
 9: 执行 a_{t:t+H}；保存 U_t → U_{t−1}
```

**关键行依据**：
- 块A 行 7–8（三类绑定失败的标签）：§III-B Eq.(2) verbatim「y_t^r = 1 if IoU_r<0.3 ∨ v_t^r=0 ∨ s_t^r<0.9, 0 otherwise」；Eq.(3)「y_t = max_r y_t^r」；「This label captures three concrete binding failures: reference mismatch, invisibility, and identity drift.」
- 块A 行 9–11（预测器）：§III-C Eq.(4) verbatim「U_t = f_θ(o_t, ℓ), U_t ∈ [0,1]」；Eq.(5)「ℒ_bind = BCE_w(Û_t, y_t) + λ(Û_t − y_t)²」；§III-B「Qwen, SAM, DINO, action signals, and trajectory context are used exclusively during offline annotation and are not required at inference time.」
- 块B 行 3–8（两路径与掩码差异）：§III-D Eq.(6) verbatim「M_direct(Z^a, Z_h^v) = 1, h=0; 0, h>0」、Eq.(7)「M_joint(Z^a, Z_h^v) = 1, h=0,…,H」；「Both paths share the video expert, action expert, and mixed transformer, and differ only in their action-to-video attention masks.」
- 块B 行 1–3（绝对值与变化量双阈值）：§III-E Eq.(8) verbatim「ΔU_t = |U_t − U_{t−1}|」；Eq.(9)「π(a_{t:t+H}|o_t,e_t,ℓ) = π_joint if U_t ≥ τ_U ∨ ΔU_t ≥ τ_Δ, π_direct otherwise」。

**取文记录**：arxiv HTML v1 成功；§I–§IV-B 前半取到。τ_U、τ_Δ 数值在截断区未取到（实验/附录）。无 Algorithm box（已获取范围）。

---

## 6. Elastic-WAM — Adaptive Spatiotemporal Imagination for Efficient Robot Control（arXiv:2611.14257，cs.RO，2026-11-10）

- 定位：需求场学习 = §3.3；推理应用 = §3.2（WHEN/HOW FAR/WHERE，Eq.7–9）+ §3.4/Fig.2；依据分级：**B｜按 §3.1–§3.3 结构化（非论文原 Algorithm；TOC 无 Algorithm 小节）**。截断位置：§3.3 中段 p*_stop(s) 定义处（阶段 2/3 具体监督与 §3.4 未取到）。场几何：未来视频 token N_F=196，划分为 G=64 个保时序/视角边界的组，n_g ∈ {1,2,4}（§3.1）。

```
块 A：三阶段 action-grounded 学习（§3.3；预训练 WAM 冻结）
 1: ▷ 阶段 1：时序剖面 (m_0, c_0) —— 训练「推进多远」
 2: for 每个采样的循环状态 s:
 3:     y_commit ← 立即 World Commit → 完整 Action Finish 所得动作序列
 4:     for 各候选去噪距离 δ:
 5:         y_δ ← 从 s 再做一步联合去噪(距离 δ) → 强制 World Commit → 完整 Action Finish
 6:         E_δ ← 0.6·E_seq(s, y_δ) + 0.4·E_Δ(s, y_δ)          ▷ Eq.10 终点动作误差
 7:     {E_δ} → 去噪距离上的软目标（误差越低概率越高），监督 B_0(z|s) = Beta(z; m_0, c_0)
 8:     E_continue^π(s) ← Σ_δ P_π(δ|s)·E_δ(s)                  ▷ Eq.11 继续精炼的期望误差
 9:     A_stop(s) ← E_continue^π(s) − E_commit(s)              ▷ 停止优势信号（p*_stop 定义在截断区）
10: ▷ 阶段 2：相对空间优先级 {u_g^rel} —— 训练「刷新哪些区域」（对下游动作预测的贡献）
11: ▷ 阶段 3：全局偏移 b_0 —— 训练整体需求 D(s)（何时停止世界精炼）
12:     （阶段 2/3 的具体监督形式在截断区未取到；顺序按 §3.3 首句）

块 B：推理 —— 需求场引导的递归稀疏世界精炼（§3.2 + §3.4/Fig.2）
输入: 状态 s_0；学好的场 (m_0, c_0, {Δu_g}, b_0)；η=0.5；δ_min=0.05；K_0=96
输出: 动作 chunk

 1: s_0 ← 第一步 dense 联合去噪（视频+动作标准一步）     ▷ 推理以 dense 步开场
 2: k ← 0
 3: loop:
 4:     a_g ← sigmoid( b_0(s_k) + stopgrad[Δu_g(s_k)] )；D(s_k) ← Σ_g (n_g/N_F)·a_g   ▷ Eq.6/4
 5:     C_k ← D(s_k)；ΔH_k ← −log(clamp(C_k, ε, 1))；F_k ← 1 − Π_{j≤k} C_j          ▷ Eq.7
 6:     if F_k ≥ η: 触发 World Commit，停止世界精炼；break                        ▷ WHEN
 7:     z_k ← m_0(s_k)；Δσ_k ← δ_min + z_k·(δ_max − δ_min)                          ▷ HOW FAR
 8:         δ_max ← min(0.20, σ_k − σ_floor)                                        ▷ Eq.8
 9:     M*_k ← argmax_{M∈{0,1}^G} Σ_g n_g·M_g·Δu_g(s_k)  s.t. Σ_g n_g·M_g = K_0    ▷ WHERE Eq.9
10:     对 M*_k 选中的组刷新未来视频 token，噪声前进 Δσ_k（保持完整逻辑注意力上下文）
11:     s_{k+1} ← 更新后状态；k ← k+1
12: 世界提交后：动作继续去噪，使用缓存的世界表示 → 输出动作 chunk
```

**关键行依据**：
- 块B 行 4–6（WHEN）：§3.2 verbatim「We interpret the global refinement demand D(s_k) as the conditional continuation probability and construct the cumulative World Commit probability across refinement steps」；「We trigger World Commit when F_k ≥ η, with η = 0.5.」；Eq.(6)「a_g(s) = sigmoid(b_0(s) + stopgrad[Δu_g(s)])」。
- 块B 行 7–8（HOW FAR）：§3.2 verbatim「we normalize the global refinement field to obtain the temporal progression distribution and use its mode to determine the denoising step size」；「We set δ_min = 0.05 and δ_max = min(0.20, σ_k − σ_floor) to constrain the denoising progression within the admissible noise interval.」
- 块B 行 9–10（WHERE）：§3.2 Eq.(9) verbatim「M*_k = argmax_{M∈{0,1}^G} Σ_g n_g M_g Δu_g(s_k), s.t. Σ_g n_g M_g = K_0」；「We use an externally specified budget of K_0 = 96 tokens per sparse update by default, while the learned spatial scores determine which regions to refresh.」
- 块A 行 2–9（阶段 1）：§3.3 verbatim「we compare immediate World Commit with continuations of different denoising distances from the same recurrent state s. Each continuation performs one additional joint denoising step, followed by a forced World Commit and complete Action Finish」；Eq.(10)「E(s,y) = 0.6E_seq(s,y) + 0.4E_Δ(s,y)」；「We convert the candidate errors into a soft target over denoising distances, assigning higher probabilities to candidates with lower terminal errors, to supervise the temporal distribution B_0(z|s)」；Eq.(11)「E_continue^π(s) = Σ_δ P_π(δ|s)E_δ(s), A_stop(s) = E_continue^π(s) − E_commit(s)」。
- 块A 行 10–12（阶段序）：§3.3 verbatim「we learn the refinement field in three stages, successively optimizing the temporal parameters (m_0,c_0), the relative spatial scores u_g^rel, and the global offset b_0.」块B 行 1、12：Abstract「we start with a dense joint denoising step, use the learned field to guide subsequent sparse world refinement, and continue action denoising after world refinement stops.」

**取文记录**：arxiv HTML v1 成功；§1–§3.3 中段取到。r.jina.ai 回退不可达。阶段 2/3 损失细节、p*_stop(s)、§3.4 在截断区。项目页 li.haitian.com/Elastic-WAM（未取）。

---

## 7. WATO — Joint World Action Models through Differentiable Trajectory Optimization（arXiv:2611.17626，cs.LG，2026-11-13）

- 定位：训练循环 = §2（世界模型/代价）+ §3.1（ℒ_WAF）+ §3.3（动力学约束投影 Eq.4）；部署 = Intro/Abstract 描述；依据分级：**B｜按 §2/§3.1/§3.3 + Intro 结构化（非论文原 Algorithm；§3.4–§3.6「Learning Through the Optimized Trajectory / The WATO Objective / Training and Deployment」在截断区未取到，隐式微分步与部署块依据 Intro/Abstract verbatim）**。截断位置：§3.3 d_Z 定义处。

```
块 A：WATO 训练循环（可微轨迹优化；TO 仅训练期使用）
输入: 离线视觉轨迹数据集 D（reward-free）；冻结世界模型 E_θ, ℛ_θ；WAF 生成器 q_φ（flow matching）
 1: ▷ WAF 先以轨迹模仿预热：ℒ_WAF = E[MSE(v_φ^A(X_s,s;c), A−ε_A) + λ_WAF·MSE(v_φ^Z(X_s,s;c), Z−ε_Z)]   ▷ Eq.3
 2: for 每个训练窗口:
 3:     c ← (z_t, g)，z_t = E_θ(o_t)，g = E_θ(窗口末帧观察)
 4:     (Ã, Z̃) ~ q_φ(A, Z | c)                              ▷ 生成参考轨迹 τ̃ = (Ã, Z̃)
 5:     ▷ 动力学约束轨迹投影（Eq.4，逐样本解 TO；Z̃ 不是执行对象而是参考）
 6:     τ* ← Π_{θ,g}(τ̃) = argmin_{(A,Z)∈ℱ_θ(z_t)}  J(Z,g) + λ_A·d_A(A,Ã) + λ_Z·d_Z(Z,Z̃)
 7:         ℱ_θ(z_t) = {(A,Z) : A ∈ 𝒰^H, Z = ℛ_θ(z_t,A)}     ▷ 轨迹必须与世界模型一致
 8:         d_A(A,Ã) = H^{-1} Σ_{h=0}^{H-1} ‖A_h − Ã_h‖²      ▷ 动作参考项
 9:     ▷ 隐式微分：把任务损失对优化解 τ* 的梯度回传到生成参考（Ã, Z̃）
10:     以该梯度更新 φ（保留 Eq.3 模仿项；§3.5 的损失组合式在截断区未取到）
11: return 任务感知、动力学一致的生成器 q_φ

块 B：WATO 部署（免迭代 TO：候选生成 + 世界模型排序）
输入: c = (z_t, g)；候选数 K；冻结世界模型 ℛ_θ
 1: for k = 1..K:
 2:     (Ã_k, Z̃_k) ~ q_φ(A, Z | c)            ▷ 直接生成候选世界-动作轨迹
 3:     Ẑ_k ← ℛ_θ(z_t, Ã_k)                  ▷ 世界模型评估动作序列的后果
 4:     score_k ← J(Ẑ_k, g) = ‖ẑ_{k,H} − g‖²  ▷ §2 终点目标代价（排序准则）
 5: k* ← argmin_k score_k                    ▷ lightweight best-of-K 选择
 6: 执行 Ã_{k*} 的一个前缀段，从新观测重规划（receding horizon，§2）
```

**关键行依据**：
- 块A 行 4–8（投影问题）：§3.3 Eq.(4) verbatim「Π_{θ,g}(τ̃) = argmin_{(A,Z)∈ℱ_θ(z_t)} J(Z,g) + λ_A d_A(A,Ã) + λ_Z d_Z(Z,Z̃)」；「define the set of trajectories consistent with the pretrained world model as ℱ_θ(z_t) = {(A,Z) : A ∈ 𝒰^H, Z = ℛ_θ(z_t,A)}」；「We use d_A(A,Ã) = H^{-1} Σ_{h=0}^{H-1} ‖A_h − Ã_h‖²」（d_Z 的归一化形式在截断区）。角色：§3.3「The task cost drives the trajectory toward the goal. The action reference preserves how the WAM proposes to move, while the state reference preserves where it proposes to go.」
- 块A 行 9–10（隐式微分训练）：Intro verbatim「WATO differentiates the task loss through this solution using implicit differentiation... providing task gradients to both generated actions and states.」「Flow matching provides offline trajectory imitation, while trajectory optimization makes the joint WAM task-aware.」；行 1：§3.1 Eq.(3)。
- 块B 行 1–5（候选+排序）：Intro verbatim「At deployment, WATO directly generates candidate world-action trajectories and evaluates their action sequences with the world model, i.e. action-conditioned dynamics. Deployment performs a lightweight best-of-K selection over these proposals, but no iterative trajectory optimization is required.」；行 4：§2 verbatim「For a latent trajectory Z=(z_1,…,z_H), we use the terminal goal cost J(Z,g)=‖z_H−g‖².」
- 块B 行 6：§2 verbatim「Control proceeds in receding horizon by repeatedly generating an action trajectory, executing a segment, and planning from the new observation.」

**取文记录**：arxiv HTML v1 成功；§1–§3.3 中段取到（§3.4/§3.5/§3.6、附录 D 隐式微分细节在截断区）。无代码链接（abs 页）。排序准则按 §2 J 与 Intro 句组合推断（已标注）。

---

## 8. ResWAM — Learning What Changes, Not What Stays（arXiv:2611.19559，cs.RO，2026-11-14）

- 定位：残差目标 = §4.1（Eq.1/2）；联合建模与双支路 = §5.1（Eq.4 + Fig.4）；依据分级：**B｜按 §4.1/§5.1 + Fig.4 结构化（非论文原 Algorithm；§5.2「Training and inference mechanisms」在截断区未取到，ℒ_a/ℒ_r 具体公式未取到，不脑补）**。截断位置：§5.1 Fig.4 caption 末尾（"…the flow-matching losses supervised by" 处）。

```
Algorithm: ResWAM 训练 —— DINOv3 残差目标 + 双支路 DiT 条件流匹配（§4.1/§5.1）
输入: 演示窗口 (o_t, o_{t+H}, q_t, c)；冻结 DINOv3 编码器 ℰ；VLM 骨干；双支路 DiT（动作支路+残差支路）
 1: ▷ 目标构造：只监督「变化」，不监督完整未来（冻结 DINOv3 空间）
 2: Z_t ← ℰ(o_t)；Z_{t+H} ← ℰ(o_{t+H})        ▷ Eq.1 视觉 patch tokens（N×D）
 3: Δ_t ← Z_{t+H} − Z_t                       ▷ Eq.2 视觉 token 残差目标（抑制静态背景内容）
 4: ▷ 前向：VLM 编码当前图像 + 本体 + 语言指令
 5: h ← VLM(o_t, q_t, c)
 6: ▷ 双支路 DiT 联合去噪（Fig.4）：两支路各自 cross-attend 到 h 与当前 DINOv3 tokens Z_t，
 7: ▷ 并经 joint attention 交换信息
 8: (Â_t, Δ̂_t) ← DualBranchDiT(噪声动作, 噪声残差, flow time s; h, Z_t)
 9: ▷ 条件流匹配监督：动作 chunk 与残差各自速度回归（ℒ_a、ℒ_r 具体式在 §5.2 截断区未取到）
10: ℒ ← ℒ_a(Â_t; A_t) + ℒ_r(Δ̂_t; Δ_t)
11: 更新 Θ（VLM + 双支路 DiT；总 2.3B，双支路 DiT 0.5B）
12: 全程无像素重建、无未来解码（Abstract）；§5.2 推理机制在截断区未取到
```

**关键行依据**：
- 行 2–3（残差目标）：§4.1 Eq.(1)(2) verbatim「Z_{t+H} = ℰ(o_{t+H}) ∈ R^{N×D}」「Δ_t = Z_{t+H} − Z_t ∈ R^{N×D}」；「the residual suppresses redundant static scene content while emphasizing temporal changes, providing a more focused supervision signal for action-induced visual changes.」
- 行 6–8（双支路联合去噪）：§5.1 Fig.4 caption verbatim「The action and residual branches denoise the predicted action chunk Â_t and residual Δ̂_t by cross-attending to the VLM output and the current DINOv3 tokens Z_t, exchanging information via joint attention.」；Eq.(4)「π_Θ(Â_t, Δ̂_t | o_t, q_t, c)」。
- 行 9–11（联合 CFM 训练）：Abstract verbatim「a lightweight 0.5B dual-branch DiT that jointly trains an action branch and a residual branch via conditional flow matching」；代码 README：「RynnBrain-1.1-2B VLM, frozen DINOv3-B/16, Dual-Stream DiT, flow objective」。
- 动机（为何残差）：§1 verbatim「supervising complete future observations weakens policy learning because consecutive frames over an action chunk change very little... the future loss is largely satisfied by copying static background tokens rather than capturing how the robot and objects move in response to actions.」

**取文记录**：arxiv HTML v1 成功；§1–§5.1 取到、§5.2 起截断。代码 github.com/OPTML-Group/ResWAM（README 已取，训练脚本未打开）。

---

## 9. Rolling-WAM — World Action Models with Rolling Imagination（arXiv:2609.30247，cs.RO，v2 2026-10-05）

- 定位：滚动窗口错峰噪声去噪与执行 = §III-A（联合分布 Eq.1）/§III-B（rolling 噪声调度 Eq.2–4）/§III-C（执行循环）；依据分级：**B｜按 §III-A/B/C 结构化（非论文原 Algorithm；L2 笔记 03b 未记录 Algorithm box，TOC 亦未见）**。verbatim 引文均转引自已核实正文的 L2 笔记 reading/03b-action-decision-wave2.md（该篇 L2 级，§I–§IV-B 前半完整）。

```
Algorithm: Rolling-WAM 滚动窗口错峰噪声去噪与执行（§III-A/B/C；数值 §IV-B）
输入: 条件 c_t = (o_t, s_t, ℓ)；窗口 W=5 个 chunk、每 chunk K=16 个动作（H=WK=80）；
      每 chunk N=10 去噪步；shifted 噪声调度 σ(τ) = ρτ/[1+(ρ−1)τ]，ρ=5
 1: ▷ episode 开头：初始化模式（Eq.4）——按错峰公式置各 chunk 初始噪声
 2: for j = 1..W: σ_j ← σ_j^init(τ) = σ( min{1, τ+(j−1)/W} )
 3: ▷ 稳态滚动循环
 4: repeat:
 5:     ▷ 本周期对窗口内全部 W 个 chunk 按其滚动噪声水平推进联合去噪（flow time τ: 1→0）
 6:     for 去噪步 = 1..N:
 7:         σ_j ← σ_j^rolling(τ) = σ((j−1+τ)/W)，j=1..W   ▷ Eq.2：chunk j 每周期从 σ(j/W) 降到 σ((j−1)/W)
 8:         Euler 更新各 chunk（§III-C Eq.5）；视频/动作联合 flow matching
 9:     ▷ 最近 chunk 已 clean：σ_1^rolling(0) = σ(0)
10:     执行 chunk 1 的 K=16 个动作
11:     ▷ 窗口滑动（§III-C）
12:     移除已执行 chunk；其余 chunk 保留其噪声水平（σ_{j+1}^rolling(0) = σ_j^rolling(1)，连续无重初始化）
13:     追加新 chunk j=W ← 高斯噪声
14: until 任务结束
```

**关键行依据**：
- 行 2（初始化模式）：§III-B Eq.(4) verbatim「σ_j^init(τ) = σ(min{1, τ+(j−1)/W})」（L2 笔记逐字；训练时 rolling/初始化模式按概率 β=0.8/0.2 采样，§III-D/§IV-B）。
- 行 7（错峰调度）：§III-B Eq.(2)(3) verbatim「σ_j^rolling(τ) = σ((j−1+τ)/W)」；L2 笔记：「每个周期 chunk j 从 σ(j/W) 降到 σ((j−1)/W)」；「σ_{j+1}^rolling(0) = σ_j^rolling(1)」。
- 行 9–13（执行与滑动）：§III-C verbatim「Once the first chunk is clean, the robot executes its K actions. The window then slides over. It removes the executed chunk, retains the remaining predictions, and appends a new chunk initialized with Gaussian noise.」
- 数值与注意力约束：§IV-B verbatim「Unless otherwise specified, we use N=10 denoising steps per chunk, a window of W=5 chunks, and K=16 actions per chunk.」；§III-D verbatim「every action chunk [can] attend to visual features across the entire prediction window. Direct action-to-action attention is restricted to the same chunk, and video tokens do not attend to action tokens.」；稳态每周期每 chunk 等效 N/W=2 步（§IV-B，4.5× 重规划加速口径）。

**取文记录**：未重新 fetch——直接引用 L2 笔记 reading/03b-action-decision-wave2.md 第 8 节（该笔记 L2 级，arxiv HTML v2 截断于 Table I LingBot-VA 行；本文引用部分均在已获取范围内）。

---

## 10. FACT — Failure-Aware Causal Training for World-Action Models（arXiv:2608.10232，cs.RO，v2 2026-10-13）

- 定位：act-then-imagine 因果训练 + 失败数据掩码 = §3.1（进度值）/§3.2（teacher-forced 因果）/§3.3（损失与两阶段推理）；依据分级：**B｜按 §3.1–§3.3 结构化（非论文原 Algorithm；L2 笔记 03b 未记录 Algorithm box）**。verbatim 引文均转引自已核实正文的 L2 笔记 reading/03b-action-decision-wave2.md（L2 级，§1–§3 全含实现细节）。

```
块 A：FACT 训练 —— act-then-imagine 因果监督 + 失败数据掩码（§3.1–§3.3）
输入: 演示/rollout 窗口 (o_t, ℓ, a_{t:t+H}, 后续观测, 进度 p)；token 序 z = [P_ref‖A_pred‖G_gt‖V_value‖I_future]
 1: ▷ teacher-forced 因果（§3.2）：世界分支用干净真实动作做条件
 2: G_gt ← a^gt_{t:t+H}（干净真实动作）；A_pred ← 噪声化预测动作
 3: mask：V_value 与 I_future 只 attend 到 {P, G_gt}；A_pred 不可见 G_gt
 4: ▷ 未来视频监督点：当前帧 + 4 个未来 offset [0, H/4, H/2, 3H/4, H]
 5: ▷ 失败感知值目标（§3.3 Eq.6；H=48，λ_fail=1）
 6: if 窗口为成功数据: v_t ← p_{t+H}
 7: else（失败 rollout）: v_t ← clip( p_{t+H} − λ_fail·1_fail(t+H), 0, 1 )
 8: ▷ 按数据来源分支出损失（失败数据掩掉动作模仿项）
 9: if 成功演示:  ℒ ← w_a·ℒ_a + w_v·ℒ_v + w_I·ℒ_I
10: else（失败）: ℒ ← w_v·ℒ_v + w_I·ℒ_I          ▷ Eq.9 L_Df：只监督后果
11: 更新 θ（共享骨干：世界损失直接塑形动作生成；动作适配器挂每层 FFN 后）

块 B：FACT 两阶段推理 —— act-then-imagine（§3.3）
 1: Stage 1: 去噪 [P_state, P_ref, A_noisy]，K_denoise=20 步 flow-Euler → â_{t:t+H}
 2: if 只需动作: 返回 â，跳过世界预测（action-only 模式；prefix KV-cache 复用）
 3: Stage 2（可选候选打分）: Stage 1 并行采样 N 个候选 a^(k)
 4:     v̂^(k) ← V_θ(o_t, ℓ, a^(k))（值头预测各候选的执行后进度）
 5: 执行 a* = argmax_k v̂^(k)                     ▷ Eq.10，无需单独 critic
```

**关键行依据**：
- 块A 行 1–3（teacher-forced 因果）：§3.2 verbatim「we introduce a clean ground-truth action condition a^gt_{t:t+H} for the world branch」；「During training, value and future-video tokens attend to the clean ground-truth action G rather than the noisy predicted action A, while A cannot see G.」（token 序 Eq.5 z=[P_ref‖A_pred‖G_gt‖V_value‖I_future]）
- 块A 行 5–7（失败感知值目标）：§3.3 Eq.(6)「v_t = p_{t+H}（成功）或 clip(p_{t+H} − λ_fail·1_fail(t+H), 0, 1)（失败）」；「The action chunk length is H=48 and λ_fail=1.」；进度定义 §3.1 Eq.(3)「p_t = G_t/G_T ∈ [0,1]」。
- 块A 行 8–10（失败掩码）：§3.3 Eq.(9) verbatim「L_Df = w_v L_v + w_I L_I」+「failure data teaches the model the consequences and lower action-conditioned progress of failed actions without making those actions policy targets.」；行 4：§3.3「未来视频监督 = 当前帧 + 4 个未来 offset [0, H/4, H/2, 3H/4, H]」。
- 块B 行 1–5（两阶段推理与候选打分）：§3.3 verbatim「Stage 1 denoises [P_state, P_ref, A_noisy] for K_denoise flow-Euler steps and returns a clean action chunk â_{t:t+H}.」「Unless otherwise specified, inference uses 20 flow-Euler denoising steps.」「If only an action is required, FACT stops here and skips world prediction.」；候选打分 Eq.(10)「执行 a* = argmax_k v̂^(k)」+「This selection rule uses the value head to score the future implied by each candidate action, without training a separate critic.」

**取文记录**：未重新 fetch——直接引用 L2 笔记 reading/03b-action-decision-wave2.md 第 6 节（L2 级；该笔记截断于 Table 1 Motus 行，本文引用部分均在已获取范围内）。动作维度 d_a、候选数 N 原文未写（已获取范围）。

---

*（全文完：10 个机制伪代码 + 依据分级说明。生成 2026-11-21。）*

