# WAM 调研补充线：RL post-training 专项精读（6 篇）

> 生成：2026-11-12 ｜ 代理：07-rl-posttraining ｜ 任务：把 6 篇「真·策略优化 RL」WAM/视频动作模型论文从综述一行摘要（L1）升到 L2/L3，每完成一篇立即落盘。
> 取文链：arxiv.org/abs/<id> → arxiv.org/html/<id>（最新版）→ ar5iv.labs.arxiv.org/html/<id>。
> 铁律：只依据论文原文（英文 verbatim + 小节号）或代码（仓库+文件+行）；找不到标「原文未写（已获取范围）」/「未取到」（标截断位置）；数值逐字抄录；禁止脑补。
> 来源级别约定（沿用 README）：L3=全文核实（含代码）/ L2=部分正文 / L1=仅摘要。
> 环境限制：shell 出网被沙箱拦截（实测 Invoke-WebRequest：「基础连接已经关闭」），全部网络取证走 web_fetch；HTML 页被工具截断，截断位置逐篇标注。

## 本线背景与结论分层说明

- 定位：WAM 调研语料的补充线。已有语料已核实价值学习（WAV / τ0-WM / DreamAvoid）与失败感知模仿（FACT / XPACE）；本线补「真·策略优化 RL」一手证据（KL-regularized RL / GRPO / 稀疏奖励 / 导航 RL），DreamGen 作「非 RL」对照组。
- 结论分层：L3=正文+代码双核实；L2=正文（或截断正文）；L1=仅摘要（fetch 两次失败降级）。每篇「来源级别」行注明截断位置与未取到区间。
- 判定口径（本线）：算 RL 的充分条件 = 用回报/优势/奖励信号驱动策略更新（含 advantage-conditioned 的加权 NLL 变形，因它等价于 KL-regularized RL 闭式解）；「世界模型生成轨迹→伪标注→SFT」无回报反馈回路，不算 RL（DreamGen 单列对照）。
- 每篇 ≤60 行；文末附 6 篇对比总表 + 共性模式归纳（仅基于原文）。

---

## 1. GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning（arXiv:2602.12099）

### 元信息
- v2（latest，2026-02-26 修订；v1 2026-02-12），cs.CV；GigaBrain Team（25 人团队署名）。
- 项目页 https://gigabrain05m.github.io ；官方 GitHub 未在 abs 页列出。
- 一句话：GigaBrain-0.5（>10,000 小时机器人操作数据预训练 VLA）之上叠加 RAMP（Reinforcement leArning via world Model-conditioned Policy）四阶段闭环。

### 来源级别
L2（arxiv HTML v2 正文：Abstract、§1、§2、§3 全，§3.2.2 Stage 1–4 完整）。**截断位置**：§3.2.2 末「Inference. During deployme」处；**未取到**：Inference 段、§4 Experiment 全文（4.1/4.2 的 AWR/RECAP 对比数值与成功率表）、§5、References。

### 【RL 机制】
1. 算法框架 = **KL-regularized RL**。§3.2.1："we extend the KL-regularized reinforcement learning framework to our augmented state space 𝐒=(𝐨,𝐳,𝑙), where 𝐳 represents the latent representation extracted by the world model." 最优解闭式 Eq.(2)："π̂(a|S) ∝ π_ref(a|S) exp(A^{π_ref}(S,a)/β)"。β 具体数值**原文未写（已获取范围）**。
2. **二值优势指示 I 从哪来**：世界模型价值估计 + n-step TD。§3.2.2 Stage 2 Eq.(8)："A(s_t,a_t)=Σ_{k=0}^{n−1} γ^k r_{t+k} + γ^n v_{t+n} − v_t"；"advantages are discretized into a binary indicator I=𝟙(A(s_t,a_t)>ε) with threshold ε"。n/γ/ε 数值**原文未写（已获取范围）**。
3. 目标改写：§3.2.1 引入 I 并假设 p(I|a,S) ∝ exp(A/β)，Bayes 得 "exp(A^{π_ref}(S,a)/β) ∝ π_ref(a|I,S)/π_ref(a|S)"，最终为加权 NLL Eq.(3)："L(θ)=E_D[−log π_θ(a|o,z,l) − α log π_θ(a|I,o,z_t,l)]"，"where I=𝟙[A(o,z,l,a)>ε] serves as the improvement signal"。→ RL 目标被写成「无条件生成 + 改进条件生成」两项 SFT。
4. 奖励：稀疏、episode 级。§3.2.2 Stage 1 Eq.(5)："r_t = 0 if t=T and episode succeeds; −C_fail if t=T and episode fails; −1 otherwise"；"we derive sparse rewards from episode-level success labels such that the value function corresponds to the negative expected steps-to-completion"。C_fail 取值未写。
5. 世界模型如何注入政策：Stage 2 政策接收 𝒲_ϕ 预测的两路信号——未来状态 token z_future（"Future state tokens are projected via a lightweight MLP to align their dimensionality with the policy's visual encoder outputs"）与价值 v_t；骨干 Wan2.2 + flow matching（Eq.(7)），输入 latent s_t=[z_t;Ψ(v_t);Ψ(p_t)] channel-wise 拼接（Eq.(6)），未来帧 {o_{t+i}}_{i∈{12,24,36,48}} 经预训练 VAE 编码；"We utilize 4K hours real robot manipulation data to train the world model"。
6. 更新范围：Stage 2 "initializes the policy from the pretrained GigaBrain-0.5 checkpoint and further fine-tunes it"；Stage 4 "the world model 𝒲_ϕ is jointly trained with HILR dataset and base data"（防 advantage 坍缩 A≈0）。**哪些参数冻结：原文未写（已获取范围）**；仅提 Knowledge Insulation（Driess et al., 2025）隔离语言/动作优化。
7. 正则：随机注意力掩码——"stochastic attention masking that randomly suppresses world model tokens with probability p=0.2 during training"，Stage 4 对 I 与 z_future 都掩码；推理时世界模型 "only a single denoising step during inference"。
8. 数据闭环：Stage 3 HILR（自主执行+专家介入，软件自动去除干预边界 artifact）→ Stage 4 用 HILR+base 数据继续训练。

### 【与 SFT 的衔接】
- Stage 2 直接从 GigaBrain-0.5 checkpoint 初始化（无重训）；Eq.(3) 即加权 NLL 的 SFT 形式；世界模型输出仅作条件 token（MLP 对齐维度后拼入），不改主干架构（Stage 1："This approach requires no architectural modifications to the underlying Diffusion Transformer"）。
- 理论：§3.2.1 Eq.(4) 证明 RECAP 是 RAMP 特例："π_RECAP(a|o,I)=∫_z π_RAMP(a|o,z,I) p(z|o,I) dz"；信息论理由 "H(a|o,z,I) ≤ H(a|o,I)"。

### 【代码依据】
无。abs 页无代码链接，仅项目页（视频）；无法代码核实。

### 【why 追问】
- 为什么不直接用策略梯度？§2.3："scaling policy gradient methods to large-scale VLAs remains challenging due to training instability and sample inefficiency" → 走 advantage-conditioned（RECAP 路线）。
- 为什么二值化 advantage？§3.2.2："To simplify conditioning while preserving preference structure, advantages are discretized into a binary indicator"；§1 对比 RECAP："RECAP only uses sparse advantages (0 or 1) as input, providing limited information gain"，RAMP 以 z 补密度。
- 为什么推理时可脱离世界模型？掩码 p=0.2 + 单步去噪，"forces the policy to maintain robust performance even when world model inputs are partially or fully unavailable"。

### 【关键数值】
"over 10,000 hours of robotic manipulation data"（GigaBrain-0.5 预训练，Abstract）；"approximately 30%"（RAMP 相对 RECAP，Laundry Folding / Box Packing / Espresso Preparation，Abstract）；4K hours（世界模型训练数据）；p=0.2；未来帧 i∈{12,24,36,48}；γ/n/ε/β/α/C_fail 未写。

### 【一页总结】
问题：VLA 短视（"myopic observations"，§1）。方法：RAMP 四阶段（世界模型预训练 → 政策以 (I, z_future, v) 为条件微调 → HILR rollout → 共同持续训练），把 KL-regularized RL 闭式解经 Bayes 改写为加权条件生成目标；优势 = 世界模型价值 n-step TD 的二值化。证据：相对 RECAP ≈30% 提升（Abstract；§4 数值未取到）。定位：**优势条件化的 RL**（非策略梯度），世界模型承担价值+未来态预测。

### 【待核实】
§4.2 的 AWR/RECAP 成功率对比数值；Inference 段内容；β/γ/n/ε 数值；参数冻结声明；GitHub 是否发布；C_fail。

---

## 2. VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models（arXiv:2603.19370）

### 元信息
- v1（唯一版本），2026-03-19，cs.RO；Zirui Ge, Pengxiang Ding 等 19 人（浙大等 13 单位）。
- 项目页 https://vampo-robot.github.io/VAMPO ；代码 https://github.com/OpenHelix-Team/VAMPO ；模型 https://huggingface.co/williammmgezju
- 一句话：把视频动作模型的 VPM（视频预测模型）多步去噪当 MDP，用 GRPO + expert-latent 一致性奖励做 post-training。

### 来源级别
L2（arxiv HTML v1 与 ar5iv 互补：Abstract、§1、§2、§3.1、§3.2 全，§3.3 至 reward 公式开头）。**截断位置**：arxiv HTML 截于 §3.3「the VPM must produce a group of G diverse candidate future representations」；ar5iv 截于 §3.3「r = −λ_L1 ‖x_σ0−x_0‖_1 + λ_cos x_σ0·x_0/…」公式处。**未取到**：§3.3 的 GRPO 目标 J(θ) 公式与优势归一化细节、§4 Experiments 数值、§5–§10（含 §7 Training Details、§9 Ablations）。附加渠道 r.jina.ai 连续两次失败（已按规则停试）。

### 【RL 机制】
1. 奖励来源与定义：expert-latent 一致性（无环境交互、非对抗）。§3.3："we compute a verifiable relative latent-consistency reward: r = −λ_L1 ‖x_σ0−x_0‖_1 + λ_cos (x_σ0·x_0)/(‖x_σ0‖‖x_0‖)"（截于该公式；L1 距离+cosine 两项已取到）；x_0 为 expert 未来片段经 VAE 编码（§1："a terminal reward that measures consistency with expert visual dynamics, obtained by encoding the ground-truth future sequence into the same latent space"）。λ_L1/λ_cos 数值未取到。
2. 优化算法：GRPO（Xue et al. 2025）。§3.3："we employ GRPO to fine-tune the VPM"；Algorithm 1：每轮从 D 采条件 c_v=(o,l)+expert x_0 → 共享初始噪声 x_σI∼N(0,σ_I²I) → G 条 rollout（"first step SDE, remaining steps ODE"）→ 逐条算 r_g → "Compute advantages A_g via group normalization" → 最大化 J(θ)（公式未取到）→ 周期性同步 π_θold←π_θ。G 数值未取到。
3. MDP 化（DDPO 式）：§3.1 状态 s_i=(x_σi,σ_i,c_v)、动作 a_i=x_{σi−1}、转移确定性、奖励仅末步："R(s_i,a_i) ≜ 𝟙[i=0]·r(x_σ0)"；目标 Eq.(9)："J(π)=E_{τ∼π}[Σ_{i=0}^I R(s_i,a_i)]"。
4. Euler Hybrid 为什么只在第一步注 SDE 噪声：§3.3 两条原文——(a) credit assignment："introducing stochasticity at all denoising steps leads to severe credit assignment issues: since the entire trajectory shares the same reward signal, the policy may maximize the return by exploiting later denoising actions that are irrelevant to action modeling, rather than genuinely improving the early-step visual representation"；(b) 稳定性/效率："limiting stochasticity to a single denoising step preserves the pre-trained temporal consistency of the video model and reduces the number of steps involved in backpropagation"。公式 Eq.(15)："p(s_0:I−1,a_1:I|s_I)=π^eas(a_I|s_I) Π_{i=1}^{I−1} δ(a_i−f_eds(s_i))"；§1 补充：首步 SDE 给出 "tractable Gaussian transition for low-variance policy-gradient estimation via reparameterization"（Eq.(14) 高斯转移密度）。
5. 训练哪个组件：仅 VPM（denoiser D_θv）。§3.3 明示 GRPO 微调 VPM；AGM 是否冻结**原文未写（已获取范围）**（§7 Training Details 未取到）。
6. KL 或正则：未见 KL 项（已获取范围）；正则=「只首步随机」的采样结构本身。
7. 数据来源：expert demonstrations D={(o,l,a,N)}（§2.2），离线数据集即可支撑（奖励只需 N 的 VAE 编码）；RL 阶段无真实环境交互的证据（已获取范围）。

### 【与 SFT 的衔接】
- 预训练=监督目标：VPM 用 EDM 式 L_Video=E‖D_θv(x_σi,σ_i,c_v)−x_0‖²（Eq.(3)）；AGM=diffusion policy（Chi et al. 2025）+DDIM，L_Action（Eq.(6)）；推理时 ĥ 取自 D_θv 首步多层 hidden states（"we extract the predictive visual representation ĥ from the multi-layer hidden states of D_θv at the first denoising step"，沿 Hu et al. 2024）。
- RL 是「post-training」：在预训练 VPM 上以 hybrid sampler 展开去噪轨迹、以 expert latent 为真值打分——把 likelihood-surrogate 换成 task-relevant reward（§2.3 Solution）。

### 【代码依据】
链接已见（github.com/OpenHelix-Team/VAMPO），本代理 shell 出网被沙箱拦截未拉取；**父代理已用 web_fetch 完成代码级取证（L3）**，见 [08-rl-code-forensics.md](08-rl-code-forensics.md)：GRPO 目标=POM 式截断代理（minimum(unclipped, clipped)）、组归一化优势、`sde_start=0, sde_end=1` 首步 SDE、latent 奖励 −MAE+cosine、超参（G=8/I=20/lr=1e-6/clip 0.2/adv_clip 2.0/guidance 7.5）。

### 【why 追问】
- 为什么不直接对 AGM 做 RL？动机是视觉表征误差会被 AGM 放大（§2.3："these discrepancies can be amplified…leading to compounding errors over time"），故奖励打在 VPM 预测 latent 上，间接改善动作。
- 为什么用 SVD/EDM？§2.2："We instantiate VPM with Stable Video Diffusion (SVD)…with EDM-style design and sampling"——沿用 Hu et al. 2024 范式；EDM 连续 σ 给出解析高斯转移（Eq.(14)），是策略梯度可用的前提。
- 为什么「verifiable / non-adversarial」？奖励是闭式 latent 距离而非学出的判别器（Abstract："a verifiable non-adversarial reward"）。

### 【关键数值】
G（组采样数）、I（去噪步数）、λ_L1/λ_cos、σ 调度、仿真/真机成功率数值：均**未取到**（截于 §3.3 中部）。已获取范围无定量结果。

### 【一页总结】
问题：ELBO 类 surrogate 只优化全局似然而非控制关键的视觉动态精度（"objective mismatch"，§2.3）。方法：DDPO 式 MDP + Euler Hybrid sampler（仅首步 SDE）+ GRPO + expert-latent L1/cos 奖励，只 post-train VPM。证据：§4 数值未取到。定位：**扩散世界模型内部的策略优化**（把去噪器当策略训），不是机器人动作策略的 RL。

### 【待核实】
§3.3 GRPO 目标公式与 group normalization 细节；G/I/λ 数值；§4 仿真环境与真机成功率；§7 训练细节（AGM 是否冻结、数据量）；GitHub 代码与论文一致性（L3 升级路径）。

---

## 3. CreFlow: Corrective Reflow for Sparse-Reward Embodied Video Diffusion RL（arXiv:2605.14274）

### 元信息
- v1（唯一版本），2026-05-14，cs.CV；Zhenyang Ni, Qi Zhu 等 10 人（Northwestern / UCSD / Oxford）。
- 一句话：用 LTL（Linear Temporal Logic）监视器当稀疏奖励，把「NFT 负向感知微调 + 同组正样本均值纠正」两个损失做成局域化 corrective reflow，post-train 视频生成器。

### 来源级别
L2（arxiv HTML v1 + ar5iv 互补：Abstract、§1、§2、§3（Eq.1–4）、§4.1、§4.2（Eq.6–7）、§4.3（Eq.8–9）全；**截断位置**：§4.3 末「the full CreFlow objective reads」总目标公式处）。**未取到**：总目标公式（λ 权重/KL 系数）、§5 Experiments 数值（5.1–5.3）、§6、Appendix A 证明正文。

### 【RL 机制】
1. 奖励来源与定义：LTL 监视器 ℳ_c 给**二值 + 定位诊断**。§4.1 Eq.(5)："(r,𝒱,𝒮)=ℳ_c(x_0;y)"：r∈{0,1} 代理 Succ；𝒱=逐 clause 违例记录（witness W_k⊆ℰ_c×[1,T]，模板化抽取，如 "𝐆(p→q) produces W_k={(e,t): b_p(t)∧¬b_q(t)}"）；𝒮=逐实体 SAM3 mask 图集（m_e=⋃_t SAM3(e,t)）。最终奖励："r(x_0;y)=𝟏[σ⊨Φ_c]∈{0,1}"（state trace σ 满足 LTLf 合取 Φ_c=⋀_{k=1}^K φ_k）。
2. LTL 规格怎么来：离线 coding agent 每任务合成一次 (ℰ_c,𝒫_c,Φ_c)——实体集、实体绑定谓词库（如 "Grasp(arm,cup)"、"In(cup,drawer)"）、四族模板（"persistence 𝐆 p, terminal placement 𝐅𝐆 p, causal coupling 𝐆(p→q), and ordering p 𝐔 q"）；"run once per task by a coding agent and frozen during RL"。状态提升三模型分工：SAM3（mask/轨迹）、IDM（机器人/夹爪状态）、VLM（属性检查如抽屉开合）。
3. 优化算法：online RL + **DiffusionNFT 代理损失**（§3 Eq.2–4）：rectified-flow 插值 x_t=(1−t)x_0+tε, v=ε−x_0；正/负分支 "v_θ+=(1−β)v_old+βv_θ, v_θ−=(1+β)v_old−βv_θ"；L_NFT=E[r_i‖v_θ+(x_t,t,y)−v‖²+(1−r_i)‖v_θ−(x_t,t,y)−v‖²]。"the r_i=0 branch keeps v_old as an anchor and trains v_θ toward the reflected target of a failed rollout"。
4. Credit-aware NFT：Eq.(6) 组共享掩码 M=M_T⊗M_S，M_T=⋃_{i,k}π_t(W_k^(i))（仅失败 clause 的帧）、M_S=⋃_{i,e}m_e^(i)（全部 rollout 实体并集）；"If the whole group succeeds, M_T=∅ and M≡0, matching the vanishing group-relative advantage"。Eq.(7)："L_NFT^CA(θ)=E_{i,t,ε}[r_i‖M⊙(v_θ+−v^(i))‖²+(1−r_i)‖M⊙(v_θ−−v^(i))‖²]"，"M≡1 recovers Eq.(4) exactly"。
5. Corrective reflow：Eq.(8) 组内正样本均值 "x̄_0+ := (1/|𝒫|) Σ_{j∈𝒫} x_0^(j)"（"unbiased estimator of the within-condition positive mean"；全正集聚合还 "eliminates the matching-bias term that nearest-positive selection…would introduce"）；反解 rectified-flow 得一步预测 "x̂_0^(i) := x_t^(i) − t·v_θ(x_t^(i),t,y)"；Eq.(9)："L_CR(θ)=E_{i∈𝒩,t,ε}[‖M⊙(x̂_0^(i)−x̄_0+)‖²]"，**只作用于失败样本**（i∈𝒩）。
6. KL 约束：有。§4.3 末："a KL regularizer to a frozen reference policy π_ref with parameters θ_ref (the pretrained Vidar policy used to initialize post-training)"——总目标公式与 KL 系数被截断（**未取到**）。
7. 训练哪个组件：仅视频生成器。"a frozen inverse-dynamics decoder ℐ_IDM maps the rollout to robot actions a=ℐ_IDM(x_0). Only the video generator parameters θ are updated during post-training"（§3）；监视器 RL 期间冻结。
8. 数据来源：D_task 任务上下文 y=(c,I)（指令+首帧）；每 online 迭代从行为策略 p_θold 采样 rollout（§3："In each online iteration, we sample rollouts from the behavior policy p_θold(⋅|y)"）；基础模型=预训练 flow-matching TI2V（Vidar 视频即策略管线）。

### 【与 SFT 的衔接】
- 起点：预训练 Vidar 政策 = KL 参考（θ_ref，冻结）；post-training 在其上用 online rollout 打分微调。
- 与普通 SFT 的区别：r_i=1 分支退化为 rejection-style flow-matching（§3 "the r_i=1 branch reduces to a rejection-style flow-matching update on successful rollouts"），r_i=0 分支用反射目标推开坏样本——即「奖励加权的重流（reflow）」而非均匀模仿。

### 【代码依据】
无。abs 页未见代码链接（仅 TeX source）；未做代码级核实。

### 【why 追问】
- 为什么不直接执行 Succ（真机/仿真）当奖励？§3："Evaluating Succ requires executing decoded actions on a real robot or high-fidelity simulator, so it is too expensive for the inner training loop"→ 用 LTL 监视器当代理奖励。
- 为什么掩码要组共享而非逐样本？§4.2："so that the positive and negative branches of the loss compete on a common support—successful rollouts are reinforced at exactly the coordinates where failed rollouts' drift is penalized"。
- 为什么不用 NFT 自带的反射目标？§4.3/Appendix A.5：正样本均值直接给出改进方向，方差更低——"the within-group positive mean directly supplies the reward-induced improvement direction, eliminating the EMA-plug-in noise floor that NFT incurs"。
- 为什么二值奖励不会全场拉偏？§4.2：全视频更新会 off-manifold drift/policy collapse，掩码把梯度限制在违例区，其余区域锚定预训练先验。

### 【关键数值】
"23.8 percentage points"（下游执行成功率提升，八个双手操作任务，Abstract；§5 细节未取到）；"eight RoboTwin tasks"（§1）；β>0（NFT 强化强度，数值未写）；λ_CR/KL 系数未取到。LTL 模板族=4。

### 【一页总结】
问题：似然代理目标训练的 VGM 会生成「看着合理、物理上不可执行」的操控视频。方法：离线 coding agent 合成 LTLf 规格 → 监视器输出二值奖励+时空违例 witness → 组共享掩码 M 门控 NFT 损失（Eq.7）+ 失败样本向组内正均值纠正（Eq.9）+ KL 到预训练参考。证据：8 个 RoboTwin 任务 +23.8pp 下游执行成功率（Abstract）。定位：**稀疏二值奖励下的视频生成器 online RL**（非策略梯度，是 reward-weighted flow-matching 变体）。

### 【待核实】
总目标公式（各损失权重与 KL 系数）；§5 数值表（与哪些 reward 方法比对齐度、各任务成功率）；λ/β/N（组大小）数值；Appendix A 证明细节。

---

## 4. EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards（arXiv:2603.17808）

### 元信息
- v3（latest，2026-10-18 修订；v1 2026-03-18），cs.RO + cs.AI；Ruixiang Wang 等 6 人（CUHK-Shenzhen、DexForce）。
- 项目页 https://eva-project-page.github.io/ ；代码链接未在 abs 页列出。
- 一句话：冻结 IDM 解码生成视频→用动作平滑度/限幅违例当奖励→Flow-GRPO 只更新视频生成器 LoRA。

### 来源级别
L2（arxiv HTML v3 正文：Abstract、§1、§2、§3、§4（Eq.1–7 全）、§5.1–5.4 大部；**截断位置**：§5.4 Table 3「Base-generated data | 92 | 18 | 53 | 54.3」行处）。**未取到**：Table 3 的 EVA 行与均值、§5.5 Ablations、§6、Appendix A–H（含 C 训练超参、D IDM 消融、E RoboTwin 全表、F 真机协议）。

### 【RL 机制】
1. 奖励来源与定义 = **IDM 解码动作的平滑度+限幅惩罚**。§4.2：冻结 IDM 从生成视频 V 解码关节位置指令 A={q_t}；"We compute velocity v_t=(q_{t+1}−q_t)/Δt, acceleration α_t=(v_{t+1}−v_t)/Δt, and jerk j_t=(α_{t+1}−α_t)/Δt"。平滑度用 Huber：Eq.(3) "Huber(x;δ)=½x² (|x|≤δ), δ(|x|−½δ) (|x|>δ)"；Eq.(4) "𝒫_α=E_t[Huber(α_t;δ_α)], 𝒫_j=E_t[Huber(j_t;δ_j)]"。限幅违例 Eq.(5)："𝒫_vel=E_t‖max(|v_t|−v_max,0)‖₂², 𝒫_acc=E_t‖max(|α_t|−α_max,0)‖₂²"。总奖励 Eq.(6)："R=−(λ_j𝒫_j+λ_α𝒫_α+λ_{v-lim}𝒫_vel+λ_{a-lim}𝒫_acc)"。δ/λ/v_max/α_max 数值**未写（已获取范围）**。
2. 优化算法：**Flow-GRPO**（Liu et al. 2025a，§4.3）。每个条件采样 G 条视频、冻结 IDM 算 R_i；组相对优势 "Â_i=(R_i−μ_R)/(σ_R+ε)"；裁剪目标 Eq.(7)："E[(1/G)Σ_{i=1}^G min(r_i(θ)Â_i, clip(r_i(θ),1−ε,1+ε)Â_i) − βD_KL(π_θ‖π_ref)]"，r_i(θ)=当前/行为策略转移似然比。G/ε/β 数值未写（已获取范围；Appendix C 未取到）。
3. 训练哪个组件：**仅生成器 LoRA**。§4.3："Only generator LoRA (Hu et al., 2021) parameters are updated"；IDM 全程冻结（§4.1："The trained IDM is frozen throughout video-model post-training"）；"The fine-tuned model follows the same sampling algorithm as the reference model"。
4. KL 约束：Eq.(7) 含 −βD_KL(π_θ‖π_ref)（"the KL term regularizes deviation from the reference policy"）。**π_ref 具体指向哪个 checkpoint：原文未明写（已获取范围）**，按 §5.1 流程应为 SFT 后的 Base；待核实。
5. 数据来源：训练时不执行动作（§1："incorporating action-space constraints into its rollout distribution without executing each training rollout on a robot"）；IDM 在机器人演示上监督回归训练（Eq.(2)："L_IDM=E[Σ_t‖f_ϕ(I_{t−k:t+k})−q_t^gt‖₂²]"）。
6. IDM 架构：§4.1 "convolutional backbone, spatial softmax, and an MLP action head (Levine et al., 2016)"；输入以 t 为中心 ±k 帧窗口，输出 q_t。

### 【与 SFT 的衔接】
- 与 LVP 初始化的关系（§5.1）："We initialize the Wan2.1-14B video backbone from the pretrained Large Video Planner (LVP)"→ 在机器人数据上 diffusion forcing SFT → Base → "Starting from this checkpoint, we use EVA to perform RL post-training"→ EVA。即 LVP→SFT→RL 三级管线；RL 阶段 LoRA 叠加于 SFT checkpoint。
- 与普通 SFT 的区别：SFT 只有 L_FM（Eq.(1) flow-matching）；EVA 增加动作空间反馈（经冻结 IDM），视觉与动作共享同一对齐信号（Abstract："links visual and action-level improvements through a shared alignment signal"）。

### 【代码依据】
无。abs 页无代码链接（仅项目页与 TeX source）；未做代码级核实。

### 【why 追问】
- 为什么用 Huber 而不是 L2 惩罚加速度/jerk？§4.2 "To penalize non-smooth motions, we apply a robust Huber penalty"——对离群尖峰稳健（阈值内二次、阈值外线性）。
- 为什么奖励只来自 IDM 动作而不含视觉项？§5.2 给了证据性解释："action-space feedback can discourage visually inconsistent robot motion without directly supervising task semantics or object interactions"——Kinematic 提升最大（+20.9）、Interaction 只小幅上升。
- 为什么是 LoRA-only？未直接解释（已获取范围）；与 Wan2.1-14B 大模型微调成本和 KL 锚定一致，属待核实。
- 为什么不逐 rollout 真机执行？§1/§3：执行成本太高，用 IDM 动作反馈作 executability 代理。

### 【关键数值】
Kinematic 通过率：Vidar 67.6 / Base 70.5 / EVA 91.4（+20.9）；Perfect：62.9/68.1/83.8（Table 1，210 prompts 人评）。执行成功率：RoboTwin 21 任务均值 45.81→52.29（+6.48，18/21 任务提升）；LIBERO Spatial 46.5→53.5、Goal 49.5→57.0、Object 63.5→70.0（Table 2）。数据：RoboTwin 1,050 demos/21 任务/50 trials；LIBERO 200 trials/suite；真机 AgileX CobotMagic 250 demos、5 seen+5 OOD、20 trials。Table 3 下游 VLA（π0.5）Base 生成数据均值 54.3（EVA 行未取到）。

### 【一页总结】
问题：视频规划器生成的 rollout 有 embodiment 伪影（arm 变形/时序跳变），解码动作不平滑或越限——"executability gap"。方法：冻结 IDM 当反馈器，把解码动作的 jerk/acc Huber 惩罚与速度/加速度限幅惩罚合成奖励，Flow-GRPO（组优势+clip+KL）只训生成器 LoRA。证据：真机/仿真执行成功率全面提升（RoboTwin +6.48、LIBERO +6.5~7.5），且作为数据引擎生成视频-动作对可提升下游 VLA。定位：**视频世界模型的 action-space 奖励对齐 RL**（无环境执行、无真值动作轨迹）。

### 【待核实】
Table 3 EVA 行数值；§5.5 消融（各惩罚项贡献）；δ/λ/v_max/α_max/G/ε/β/LoRA rank 超参（Appendix C）；π_ref 是否=Base 的明文；真机成功率数值（Figure 5 图内数值）；代码仓库是否存在。

---

## 5. WorldVLN: Autoregressive World Action Model for Aerial Vision-Language Navigation（arXiv:2605.15964）

### 元信息
- v1（唯一版本），2026-05-15，cs.RO + cs.CV；Baining Zhao 等 16 人（清华等）。
- 代码与 demo：https://embodiedcity.github.io/WorldVLN/
- 一句话：把 latent 自回归视频 backbone 改造为「预测短程世界状态转移→解码 waypoint 动作→真实观测回灌」闭环 WAM，Stage 2 用 Action-aware GRPO 在线 rollout 优化段级决策（首个面向自回归 WAM 的 GRPO）。

### 来源级别
L2（arxiv HTML v1 + ar5iv 互补：Abstract、§1、§2、§3、§4.1、§4.2 全（Eq.1–13）、§5 设置与 Table 1 前半；**截断位置**：arxiv HTML 截于 §4.2.2「where μ_j and σ_j…」；ar5iv 截于 §5.1 Table 1「OpenVLA-UAV | Fixed | 45.24…」行处）。**未取到**：Table 1 余行（WorldVLN 自身行）、Table 2（IndoorUAV）、§5.2–5.4（案例/消融/真机）、§6、Appendix A.4.3（GRPO 超参 λ/G/γ/K）。

### 【RL 机制】
1. 奖励从哪来：**三段式段级奖励**，只来自模拟器在线 rollout 的几何/任务结果（Eq.9）："r_j^(i)=γ^{j−1}(λ_traj r_traj,j^(i)+λ_task r_task,j^(i)+λ_ref r_ref,j^(i))"。
   - 轨迹奖励（Eq.10，局部几何监督，靠专家动作）："r_traj,j^(i)=1/(1+‖a_{(j−1)K:jK−1}^(i)−a_{(j−1)K:jK−1}^{⋆(i)}‖)"。
   - 任务奖励（Eq.11，全局结局）："r_task,j^(i)=1/(1+‖(x_T,y_T,z_T)−(x^⋆,y^⋆,z^⋆)‖₂)"——rollout 终点到真值目标距离的倒数映射。
   - 参考奖励（Eq.12，防漂移）："r_ref,j^(i)=log π_ref(a|o_≤t,a_<(j−1)K,ℓ)"，理由："preventing the latent world prediction from drifting too far away from the original navigation dynamics prior"。
   - 时间衰减："γ^{j−1}, 0<γ<1…earlier decisions receive larger weights, as they affect a longer chain of future observations and actions"。
2. GRPO 对自回归 WAM 的实现：**段级（非 token 级）**。每组 G 条在线 rollout、每条 n 个决策段（每段 K 个 waypoint）；rollout 用真实观测回灌（Eq.6："Instead of continuing generation with the model-predicted latent ẑ…, WorldVLN replaces it with the real latent z…in the autoregressive context"）。段级优势组归一化："A_j^(i)=(r_j^(i)−μ_j)/(σ_j+ε), where μ_j and σ_j are the group statistics of {r_j^(i)}_{i=1}^G for the j-th decision segment"。裁剪目标 Eq.(13)："J_GRPO=E_{i,j}[min(ρ_j^(i)A_j^(i), clip(ρ_j^(i),1−ε_clip,1+ε_clip)A_j^(i))]"，"ρ_j^(i)=π_θ(a|h_j^(i))/π_old(a|h_j^(i))"，h_j^(i)=第 i 条 rollout 第 j 段前的历史。
3. 训练哪个组件：Eq.(13) 更新 "the policy induced by the autoregressive backbone and action decoder"（§4.2.2 原文）；动作解码器 D_ϕ 与 backbone 是否同步更新/冻结拆分**原文未写（已获取范围，Appendix A.4.3 未取到）**。
4. KL/正则：无显式 KL 项；用 λ_ref·r_ref（参考策略对数似然奖励）替代（Eq.9/12）。
5. 数据来源：RL 阶段纯模拟器交互（"the model performs online autoregressive rollouts in the simulator"，§4.2.2）；轨迹奖励需专家动作 a^⋆；任务奖励需真值目标坐标。
6. 模型结构：backbone=text encoder ψ + video VAE ℰ_vid + latent 自回归 Transformer p_θ；ẑ_{t+1:t+K}∼p_θ(·|e_ℓ,z_≤t)（Eq.3）；a_{t:t+K−1}=D_ϕ(ẑ_{t+1:t+K})（Eq.4）；动作 a_t=(Δx_t,Δy_t,Δz_t,Δψ_t)∈ℝ⁴（Eq.1）。

### 【与 SFT 的衔接】
- 两阶段：Stage 1 监督——backbone 用指令-导航视频对恢复原视频生成目标 L_wm=−Σ log p_θ(z|e_ℓ,z_≤t)（Eq.7）；动作解码器 L_act=Σ‖D_ϕ(ℰ_vid(o))−a^⋆‖（Eq.8），且 "initialize the decoder with features from the video decoder and a learning-based visual odometry backbone"。Stage 2 在 Stage 1 checkpoint 上做 Action-aware GRPO（§4.2 结构：先 SFT 接地、后 RL 对齐结局）。
- 动机（§2 Post-training）："Such imitation-based training fits the demonstration distribution but remains vulnerable to covariate shift and accumulated errors"。

### 【代码依据】
链接已见（embodiedcity.github.io/WorldVLN/ 含 code），但本代理 shell 出网被沙箱拦截，未拉仓库核实（未做 L3）。

### 【why 追问】
- 为什么 WAM 化而不是直接 VLA？§1："their web-scale visual-linguistic priors are well suited for recognizing objects and parsing instructions, but not for modeling how the world evolves under the agent's own actions"。
- 为什么 autoregressive 而不是扩散式整段生成？§1 Model Structure："Most video generation backbones generate an entire clip in a bidirectional manner, whereas embodied navigation requires a causal observe–act–update loop"。
- 为什么段级而非 token 级优势？§4.2.2 结构本身给出答案：奖励天然定义在决策段上（每段解码 K 个 waypoint），段内 waypoint 共享同一奖励；时间衰减 γ^{j−1} 解决多段 credit assignment。
- 为什么用参考奖励而不是 KL？§4.2.2 解释：保想象力（"preserve the implicit imagination capability learned during supervised world-action training"）——等价目标的实现选择。

### 【关键数值】
"12%+ success-rate gains"（相对 VLA baselines，Abstract）；动作 4 维（Eq.1）；K=预测视野、G=组规模、γ/λ 数值未取到（Appendix A.4.3 未取到）；backbone=InfinityStar、解码器初始化自 Wan VAE + TSformer-VO（§5 设置）；训练 8×NVIDIA A800 80GB、rollout 在 RTX 4090（§5 设置）；基准 UAV-Flow 与 IndoorUAV。

### 【一页总结】
问题：VLA 缺「世界如何随动作演化」的建模；扩散视频模型又是双向整段生成、与因果 observe-act 循环错位。方法：latent 自回归 WAM（预测短程世界状态→解码 waypoint→真观测回灌闭环）+ 两阶段训练（SFT 接地 → Action-aware GRPO：模拟器在线 rollout、段级三奖励、时间衰减、组归一化优势、裁剪目标）。证据：两个公开基准 12%+ SR 提升、真机 zero-shot（Abstract；表细节未取到）。定位：**唯一一篇对「自回归 WAM」整体做在线 RL 的论文**（其余各篇 RL 打在扩散/流视频生成器或 VLA 上）。

### 【待核实】
Table 1 WorldVLN 行与 Table 2 数值；§5.3 消融（GRPO 贡献、autoregressive 必要性）；λ_traj/λ_task/λ_ref/γ/G/K/n 超参（A.4.3）；backbone 与 action decoder 在 Stage 2 的冻结范围；真机成功率数值。

---

## 6. DreamGen: Unlocking Generalization in Robot Learning through Video World Models（arXiv:2505.12705）【对照组：非 RL】

### 元信息
- v2（latest，2025-06-17 修订；v1 2025-05-19），cs.RO + cs.AI + cs.LG；NVIDIA（Joel Jang, Yuke Zhu, Linxi Fan 等 27 人）。
- 代码：https://github.com/NVIDIA/GR00T-Dreams ；网站：https://research.nvidia.com/labs/gear/dreamgen
- 一句话：4 阶段合成数据管线（微调视频世界模型→生成视频→伪动作标注→下游策略模仿训练），无任何回报/RL 回路。

### 来源级别
L2（arxiv HTML v2 正文：Abstract、§1、§2（2.1–2.4 全）、§3.1、§3.2 至 Table 1；**截断位置**：§3.2 Environment Generalization 的 Table 1 末尾）。**未取到**：§3.2 环境泛化文字尾、§4 DreamGen Bench、§5 Related Work、§6–7、Appendix A–J。

### 【RL 机制】
（本小节改为「为什么它**不是** RL」——这是本篇在本线的用途。）
1. 管线 = 生成→伪标注→**模仿学习**，无回报信号。§1 四步 recipe："(1) We fine-tune video world models on a target robot…(2) we prompt the model…to generate large volumes of robot videos…(3) we then extract pseudo-actions using either a latent action model or an inverse dynamics model (IDM); (4) finally, we use the resulting video-action sequence pairs, dubbed neural trajectories, for training downstream visuomotor policies."
2. 下游训练是纯行为克隆。§2.4："we train the policies to generate â_{t:t+H}"（条件：图像观测+指令）；"We condition state information with zero values"；策略 = Diffusion Policy / π0 / GR00T N1。共训比例 "a sampling ratio of 1:1"；行为/环境泛化实验 "we only use neural trajectories for policy training"。
3. 无在线回环：视频世界模型微调用 LoRA（§2.1："we use Low-Rank Adaptation (LoRA) by default"），生成用人工采集的首帧（§2.2："we manually take new initial frames"），生成结果不会经执行成败回馈给生成器——数据流是单向的。
4. 伪动作两法：IDM（"diffusion transformers with SigLIP-2 vision encoder and train with a flow matching objective"，两帧条件→action chunk，滑窗 H 步标注）；LAPA 潜在动作（VQ-VAE，"use the pre-quantized continuous embedding as the latent action following GR00T N1"）。
5. 全文无 reward/advantage/policy-gradient/value 字样（已获取范围）；被同批论文引用时的定位也是「数据引擎」——EVA §2："As data engines, DreamGen and GigaWorld generate manipulation videos paired with inferred actions to expand policy-training data"；GigaBrain §2.2 亦将 DreamGen 归入「生成视频→IDM 推动作」管线。
6. 判定：按本线口径（用回报/优势驱动策略更新才算 RL），DreamGen 不满足；其下游训练是 SFT，世界模型微调是标准视频扩散损失。**属「世界模型生成→伪标注→模仿」对照组**。

### 【与 SFT 的衔接】
- 与 SFT 的关系不是「衔接」而是「就是 SFT」：世界模型 = 视频生成 SFT（LoRA）；下游策略 = 行为克隆。§2.4 明确训练目标即监督回归/生成伪动作序列。

### 【代码依据】
链接已见（github.com/NVIDIA/GR00T-Dreams），本代理 shell 出网被沙箱拦截，未拉仓库核实。

### 【why 追问】
- 为什么不用 RL 也能大幅泛化？§3.2："just given the initial frame and the language instruction, we observe that the video world model can generalize in generating videos of totally unseen behaviors"——泛化来自视频先验而非回报；这正是它被当作 RL 对照组的理由：提升来自数据分布外扩，不来自策略优化。
- 为什么伪标注而非真值动作？生成模型只输出视频（Abstract："Since these models generate only videos, we recover pseudo-action sequences"）。
- 它和 CreFlow/EVA 的差别在哪？CreFlow/EVA 把「监视器/IDM 信号」作为回报回灌生成器（有反馈回路）；DreamGen 的 IDM 信号只用于给下游策略做监督标签（无回路）。

### 【关键数值】
"22 new behaviors"（行为泛化，Abstract）；14 个新行为任务均值 11.2%→43.2%、13 个新环境任务 0.0→28.5%（Table 1，GR00T N1 基座）；RoboCasa 合成数据放大 "333×"、仅用 neural trajectories 训练 24 任务均值 "20.6%"；真机三形态：GR1 4 任务 37%→46.4%、Franka 3 任务 23%→37%、SO-100 2 任务 21%→45.5%（§3.1，各任务仅 10–13 条真机轨迹）；世界模型微调数据 2,884 条 GR1 pick-and-place（§3.2）；基座视频模型 WAN2.1（§2.1）。

### 【一页总结】
问题：遥操作数据贵。方法：微调视频世界模型→首帧+指令生成视频→IDM/LAPA 伪标注→下游 visuomotor 策略模仿训练（4 步单向管线）。证据：新行为 43.2%、新环境 28.5%（从 ~0 起步）。定位：**数据合成管线（data engine），非 RL**——无回报、无策略梯度、无在线回环；下游是 SFT，世界模型更新是标准视频扩散 SFT。对照组意义：区分「世界模型用于扩大模仿数据」与「世界模型/策略被回报信号优化」两条技术线。

### 【待核实】
§3.2 环境泛化正文细节；§4 DreamGen Bench（8 模型评分与下游相关性）；下游策略训练超参（Appendix）；GR00T-Dreams 仓库是否包含任何 RL 组件（代码级确认其纯 SFT 性质，L3 升级路径）。

---

## 6 篇对比总表（奖励来源 × 算法 × 训练组件 × 数据 × KL/正则）

| 论文 | 奖励来源/定义 | 优化算法 | 训练组件 | 冻结范围 | 数据 | KL/正则 |
|---|---|---|---|---|---|---|
| GigaBrain 2602.12099 | sparse episode reward（Eq.5：成功 0/失败 −C_fail/步 −1）→ 世界模型价值 n-step TD 优势（Eq.8）→ 二值 I=𝟙[A>ε] | KL-regularized RL 闭式解改写为加权 NLL（Eq.3，含 α 权重改进项） | policy（GigaBrain-0.5 全模型）+ 世界模型（Stage 4 共同训练） | 冻结范围未写；Knowledge Insulation | 预训练>10K h + 4K h（世界模型）+ HILR rollout | 掩码 p=0.2（I 与 z_future）；单步去噪 |
| VAMPO 2603.19370 | expert-latent 一致性：r=−λ_L1‖x_σ0−x_0‖₁+λ_cos·cos（§3.3，截断于公式） | GRPO（组采样 G、共享噪声、组归一化优势，Algorithm 1） | 仅 VPM（SVD 去噪器） | AGM 冻结状态未写 | expert demonstrations D={(o,l,a,N)}（离线） | 无 KL；仅首步 SDE（Euler Hybrid）当正则 |
| CreFlow 2605.14274 | LTL 监视器二值 r=𝟏[σ⊨Φ_c] + 违例 witness + SAM3 图集（Eq.5） | online RL：credit-aware NFT（Eq.7）+ corrective reflow（Eq.9） | 仅视频生成器（flow TI2V） | IDM 冻结、监视器冻结（规格一次合成） | D_task（首帧+指令），online rollout | 有 KL 到冻结 π_ref（预训练 Vidar；系数未取到） |
| EVA 2603.17808 | IDM 解码动作：jerk/acc Huber + 速度/加速度限幅惩罚（Eq.3–6），R=负加权和 | Flow-GRPO（G 组、组优势、clip、−βD_KL，Eq.7） | 仅生成器 LoRA | IDM 冻结（"frozen throughout"）；其余未写 | 机器人演示（训 IDM）+ 生成 rollout（不执行） | KL 到 π_ref（Eq.7；π_ref 身份未明写） |
| WorldVLN 2605.15964 | 三奖励：轨迹 1/(1+‖a−a*‖)+任务 1/(1+终点距目标)+参考 log π_ref（Eq.10–12），γ^{j−1} 时间衰减 | Action-aware GRPO（段级组归一化优势+clip，Eq.13） | 自回归 backbone + action decoder（"policy induced by the autoregressive backbone and action decoder"） | 拆分冻结未写 | 模拟器在线 rollout + 专家动作/真值目标 | 无 KL；λ_ref·log π_ref 奖励替代 |
| DreamGen 2505.12705【对照】 | 无奖励 | 无 RL（下游=行为克隆；世界模型=LoRA 视频 SFT） | 下游策略（DP/π0/GR00T N1）与世界模型各自 SFT | 无 RL 回路 | 遥操作数据微调世界模型→生成→伪标注 | 无 |

## 共性模式归纳（仅基于原文）

1. **奖励全部是「免执行」的代理信号**：GigaBrain 用世界模型价值（episode 级成功标签）+HILR；VAMPO 用 expert-latent 距离；CreFlow 用 LTL 监视器对生成视频的逻辑验证；EVA 用冻结 IDM 解码动作的平滑度/限幅；只有 WorldVLN 用模拟器在线 rollout 的几何/任务结果。五篇 RL 无一在训练内环执行真机动作。
2. **被优化的对象是「视频/世界生成器」，不是下游动作策略**：VAMPO 训 VPM、CreFlow 训 TI2V 生成器、EVA 训视频模型 LoRA、GigaBrain 训含世界模型条件输入的 policy、WorldVLN 训自回归 WAM 整体。共同逻辑：改善预测的视觉/世界表征 → 下游动作（经 IDM/解码器）随之变好。
3. **优势/奖励几乎总是被归一化或二值化后使用**：GigaBrain 二值化 I（"To simplify conditioning while preserving preference structure"）；VAMPO/EVA/WorldVLN 用组归一化优势（GRPO 系）；CreFlow 用二值 r + 组共享掩码（组全成功则掩码归零，"matching the vanishing group-relative advantage"）。
4. **KL/参考锚定是标配**：CreFlow 显式 KL 到冻结预训练参考；EVA 的 Flow-GRPO 含 −βD_KL；GigaBrain 的理论起点就是 KL-regularized RL 闭式解；WorldVLN 用 log π_ref 奖励项；VAMPO 用「仅首步随机」的采样结构保时序一致性（结构级锚定）。
5. **RL 一律以 SFT checkpoint 为起点**（post-training 语义一致）：GigaBrain Stage 2 从 GigaBrain-0.5 初始化；VAMPO 在预训练 VPM 上 post-train；CreFlow 参考=预训练 Vidar；EVA 为 LVP→diffusion-forcing SFT→RL 三级；WorldVLN Stage 1 SFT 接地后才 GRPO。
6. **对照组结论**：DreamGen 证明「生成→伪标注→模仿」无需回报即可解锁泛化（43.2%/28.5%），但它是数据外扩不是策略优化；与 CreFlow/EVA 的差别在于 IDM/监视器信号是否形成回灌生成器的闭环。

<!-- QA_LOG -->
（占位：后续按「逐机制 why 追问」格式追加 Q&A 记录，替换锚点使用本注释内文本。）
<!-- /QA_LOG -->

