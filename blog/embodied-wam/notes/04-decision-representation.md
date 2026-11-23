# 精读线④：决策/规划表征 —— 9 篇具身智能 World Action Model 论文精读笔记

> 生成时间：2026-10-26
> 取文方式：每篇先 `https://arxiv.org/abs/<id>`，再 `https://arxiv.org/html/<id>`（最新版）；截断则补 `https://ar5iv.labs.arxiv.org/html/<id>`。
> 证据纪律：所有引文为原文 verbatim（附小节号）；截断处标注「未取到」+位置；数值只来自原文；无原文依据处写「原文未写（已获取范围）」。

---

## 1. WAV — World-Value-Action Model: Implicit Planning for Vision-Language-Action Systems（arXiv:2604.14732）

**元信息**
- 作者：Runze Li, Hongyin Zhang, Junxi Jin, Qixin Zeng, Zifeng Zhuang, Yiqi Tang, Shangke Lyu, Donglin Wang（Westlake University 等）
- 版本：v1 2026-04-16；v3 2026-10-12（最新，即本次取文版本）
- 状态：Accepted at NeurIPS 2026；Code: https://github.com/Win-commit/WAV ；Project page: https://win-commit.github.io/wavpage/

**来源级别：L3（方法节原文，覆盖至 4.3 第二段）**
- 已获取原文：Abstract、Sec 3 Preliminaries、Sec 4.1（Lemma 4.1 / Proposition 4.2 / Corollary 4.3 全文）、Sec 4.2（架构与三个 flow-matching loss、Eq 3）、Fig 1 caption、Algorithm 1（全文）、Sec 4.3 第一、二段。
- 截断位置：`arxiv.org/html/2604.14732v3` 截断于 4.2「action decoder」公式 Eq (3) 之前；`ar5iv.labs.arxiv.org/html/2604.14732` 更深，截断于 4.3 第二段句「Each latent sample is denoised by the video generation module 𝒲 to produce video feature chunking: 𝐱1:H1」处。PDF 不可用（web_fetch 不支持 application/pdf）。
- 未取到（在截断之后）：Eq (4)/(5) 更新公式、M/N/K/K1/K2 具体数值、价值函数训练数据构造细节（App B）、消融实验（Sec 5.3）、附录理论证明（App A）。

**【决策机制】原文引文 + 逐条指路**

结构（Sec 4.1 末 + Fig 1）：
> "Latent planning resolves the feasibility bottleneck by reshaping the search distribution, while iterative inference resolves the remaining optimization challenge within the feasible manifold."（Sec 4.1 末段）
> Fig 1 caption（verbatim）: "The proposed WAV model decomposes planning and control into three tightly coupled components: 1) a language-conditioned video generation module for predicting future visual trajectories; 2) a trajectory value module for long-horizon evaluation; and 3) an action module for generating executable robot actions. Together, these components enable implicit planning in latent space, where action generation arises from inference over predicted futures guided by trajectory value."

1. **决策范式 = 潜空间 MPPI 式迭代推断**（Sec 4.3，verbatim 首段）：
> "Inspired by Model Predictive Path Integral (MPPI) ([47]) control, we perform iterative latent-space trajectory optimization during inference. Instead of fixing latent priors to standard Gaussians, we adaptively update the noise distributions of the video and value modules based on trajectory evaluations. We maintain two independent Gaussian noise distributions: f_vid^(k)=𝒩(𝝁_vid^(k), diag((𝝈_vid^(k))²)), and a trajectory value flow noise distribution f_val^(k)=𝒩(𝝁_val^(k), diag((𝝈_val^(k))²)), where k∈{1,…,K} denotes the iteration index."

2. **采样**（Sec 4.3 第二段，verbatim，截断于句尾）：
> "At iteration k, we first sample M groups of video latent noise: {𝐳_vid^(m)}_{m=1}^{M}∼f_vid^(k−1). Each latent sample is denoised by the video generation module 𝒲 to produce video feature chunking: 𝐱1:H1"（截断，后文为每个视频特征采样 N 个价值潜噪声并评分——见 Algorithm 1）

3. **算法伪代码**（Algorithm 1 "WAV: Latent Trajectory Planning"，全文 verbatim 要点）：
   - 输入：Video generator module 𝒲、trajectory value module 𝒱、initial flow noise distributions f_vid^(0), f_val^(0)、samples (M,N)、elites (K1,K2)。
   - for iteration k=1,…,K：
     1. Sample video latent noise 𝐳_vid^(m)∼f_vid^(k−1)
     2. Generate video feature 𝐱^(m)
     3. for each video feature 𝐱^(m)：Sample value latent noise 𝐳_val^(m,n)∼f_val^(k−1)；Evaluate values 𝐔^(m,n)；Compute score φ^(m)
     4. Update f_vid^(k) using top-K1 samples by Eq. (4)
     5. Update f_val^(k) using top-K2 samples by Eq. (5)
   - 输出：Sample optimized latent noise 𝐳_vid^⋆, 𝐳_val^⋆；Predict action by Eq. (3)
   - → 结构即「M 组视频想象 × 每组 N 个价值评估 → top-K1/K2 精英更新双高斯 → 迭代 K 次 → 解码动作」；**M/N/K/K1/K2 的数值原文未取到**（位于 5.3 消融「Effect of Iteration Counts K and Sampling Sizes M,N」「Effect of Elite Selection (K1,K2)」与附录 C，均在截断之后）。

4. **动作解码**（Sec 4.2，Eq (3) verbatim）：
> "𝐳_act^(i) = ℬ_i^act(𝐳_act^(i−1), CrossAttn(𝐳_act^(i−1), 𝐱_i)), and 𝐚_i = ℬ_i^act(𝐳_act^(i), CrossAttn(𝐳_act^(i), 𝐮_i))."
（动作 token 先交叉注意力到视频特征、再交叉注意力到价值特征 𝐮_i，即价值引导动作解码。）

5. **价值函数训练数据**（Sec 4.2 "Training via Flow Matching" verbatim）：
> "x¹,v¹,a¹ denote samples from the target data distributions, including the target video trajectories, the return v¹=∑_{i=0}^{H} γ^i R(s_{t+i}, a_{t+i}), and the target robot action distribution. Here we use a rule-based dense reward function similar to that used in previous work–ReinboT ([53]) (details in Appendix Sec. B)."
> 三阶段 loss（verbatim）：ℒ_vid = 𝔼[‖v_θ(t,l,o,x^t)−(x¹−x⁰)‖₂²]（先训视频，冻结）；ℒ_val = 𝔼[‖v_θ(t,l,o,z_vid,v^t)−(v¹−v⁰)‖₂²]（价值模块预测累积折扣回报）；ℒ_act = 𝔼[‖v_θ(t,l,o,z_vid,z_val,a^t)−(a¹−a⁰)‖₂²]（全模块联合）。
> 语言条件：frozen T5-XXL encoder → cross-attention 注入 DiT；多视角 i∈{h,l,r}，x̂_{t:t+N}=𝒲({v₀^(i), v_t̂^(i), z^(i)}_i, 𝒯(g))。

6. **理论支撑**（Sec 4.1，全部 verbatim 取得）：
- Lemma 4.1 (Vanishing Feasible Mass in Trajectory Space)：𝒳=𝒮^H×𝒜^H，D=H(dim𝒮+dim𝒜)，ℳ_traj 内在维 d≪D，则 μ(𝒩_ε(ℳ_traj))/μ(𝒳) ≤ exp(−cH)（Eq (1)）。
- Proposition 4.2 (Latent Reweighting of the Search Distribution)：Pr_{z∼f_θ(s_t)}[𝒲_θ(z)∈ℳ_traj] / Pr_{a∼Unif(𝒜^H)}[Φ(a)∈ℳ_traj] ≥ exp(cH)(1−δ)（Eq (2)）。
- Corollary 4.3 (Necessity of Iterative Latent Planning)：对任意随 H 亚指数增长的采样预算 N，从 P_latent 一次性采样「does not, in general, guarantee with high probability the discovery of a near-optimal trajectory. Iterative inference procedures that adaptively reweight the latent distribution are therefore necessary to concentrate probability mass on high-value feasible trajectories.」

**【why 追问】**
1. 为何要测试时迭代计算（而非一次前向）？Corollary 4.3（原文）：可行性只解决「采样到可行轨迹」的概率问题（Prop 4.2），但可行集内轨迹价值差异巨大，一次性潜采样在亚指数预算下无法保证发现近优轨迹，必须迭代地把概率质量集中到高价值可行轨迹。这是「可行性瓶颈 + 最优性瓶颈」双论据。
2. 为何在潜空间而非动作空间搜索？Lemma 4.1（原文）：动作空间均匀探索时可行轨迹概率随 H 指数衰减 exp(−cH)；Prop 4.2（原文）：潜生成器将可行概率比提升 ≥ exp(cH)(1−δ)——「This advantage arises not from dimensionality reduction, but from reweighting the search toward dynamically and semantically valid regions, effectively acting as a probabilistic projection onto the feasible trajectory manifold.」（4.1 原文）
3. 为何选 MPPI 式精英更新而非梯度优化？（推导，基于原文 4.3）动作空间高维、可行流形为「thin manifold」（4.1 原文），免梯度采样-评分-精英更新天然适配该结构；原文直接声明「Inspired by MPPI」而未给出与其他优化器的对比——待核实。
4. 为何用视频生成器兼作世界模型与先验？（原文 4.2）"𝒲 captures both task intent and spatiotemporal dynamics"，且 Fig 1 将「预测未来视觉轨迹」作为价值与动作的共同基底。

**【关键数值】**
- 已取得：γ∈(0,1]（Cor 4.3）；T5-XXL 冻结语言编码器；DiT 主干；多视角 {head,left,right}。
- 未取到（截断后，待核实）：K（迭代数）、M（视频样本组数）、N（每组价值样本数）、K1/K2（精英数）、规划视界 H、推理开销数值（App B.3 有「Inference Cost and Accuracy–Compute Trade-off」小节，原文未取到）。

**【一页总结】**
WAV 把 VLA 的决策定义为潜空间推断：视频生成模块产生未来轨迹候选（M 组），轨迹价值模块对每组采样 N 个价值评估，按 top-K1/K2 精英更新两个流匹配噪声分布（MPPI 式），迭代 K 次后解码动作。理论三件套（Lemma/Prop/Cor）为「潜空间重加权解决可行性、迭代推断解决最优性」提供指数级论据。价值监督为 ReinboT 风格规则密集奖励 + 累积折扣回报的 flow matching。4.3 后半（Eq 4/5 具体更新式）与全部超参数值在本次取文截断之后。

**【待核实】**
1. Eq (4)/(5)（top-K1/K2 更新公式）原文；
2. M/N/K/K1/K2 数值与规划视界（Sec 5.3 / App C.3）；
3. 价值训练数据构造（App B：ReinboT 规则奖励的分解细节）；
4. App A 中 Lemma/Prop/Cor 的证明（A.2–A.4）；
5. 推理计算开销与准确率-计算权衡（App B.3）。

---

## 2. τ0-WM — A Unified Video-Action World Model for Robotic Manipulation（arXiv:2606.01027）

**元信息**
- 作者：Pengfei Zhou, Shengcong Chen, Di Chen, …, Jianlan Luo（Shanghai Innovation Institute / AGIBOT Finch；Corresponding: Jianlan Luo）
- 版本：v1 2026-05-31；v2 2026-08-23（最新，本次取文版本）；CC BY-NC-SA 4.0
- Project: https://tau0-wm.github.io/

**来源级别：L3（决策机制方法节原文已完整取得）**
- 已获取原文：Abstract、Sec I–VI 全部正文（含 VI-A RCS 公式 Eq 6–7、VI-B LAR 公式 Eq 8–11）、Algorithm 1/Algorithm 2 全文、Fig 1/Fig 2/Fig 3 caption。
- 截断位置：`arxiv.org/html/2606.01027v2` 截断于 Algorithm 1 第 4 行（第 3 行为 "candidate budget N, threshold γ"）；`ar5iv.labs.arxiv.org/html/2606.01027` 更深，截断于 VI-B 末句 "Instead of directly executing the corresponding action, we perform a second policy query conditioned on the selected future rollout. Specifically, the rollout latent" 处（Algorithm 2 已在该句之前取得，内容完整）。
- 未取到（截断后）：VI-B 末段剩余一句、VII 实验与消融、附录 -A2（部署细节，N/γ 数值所在）。

**【决策机制】原文引文 + 逐条指路**

整体定位（Intro，verbatim）：
> "At inference time, τ0-WM allocates additional computation to action selection rather than executing the first feed-forward prediction. The model first samples multiple action chunks from VAM and ranks them with a re-denoising consistency score, which measures whether a candidate is consistent with the learned conditional action distribution. When the selected candidate appears unreliable, τ0-WM invokes ACVS to simulate the futures induced by candidate actions and estimate their task progress. The most promising imagined future is then used to condition a second VAM query, producing a refined action chunk. This yields a proposal–evaluation–revision procedure in which future prediction is used directly as a mechanism for improving robot actions before execution."

1. **propose–evaluate–revise 三阶段**（Sec VI 首段，verbatim）：
> "τ0-WM adopts a coarse-to-fine test-time computation strategy. It first samples multiple action candidates from VAM and applies a lightweight self-consistency filter to identify reliable candidates. Only when the sampled candidates appear unreliable does the system invoke ACVS for more expensive rollout-based evaluation and action rectification. This design preserves real-time performance in most situations while retaining the ability to recover from difficult states. The overall procedure is summarized in Alg. 1."

2. **Algorithm 1 "Test-Time Computation"**（全文 verbatim 要点）：
> 输入：VAM F_θ, ACVS G_ϕ；current context 𝒞_t；candidate budget N, threshold γ。
> 1) Sample N candidate actions {ā^(i)}_{i=1}^{N}；2) for i=1,…,N: Compute S_RCS^(i)；3) i^⋆←arg max_i S_RCS^(i)；4) if S_RCS^(i^⋆)≥γ then return ā^(i^⋆)；5) （否则）for i=1,…,N: Evaluate candidate using ACVS, Compute rollout value J^(i)；6) j^⋆←arg max_i J^(i)；7) return LAR(ẑ^(j^⋆))。
> **Algorithm 2 "Low-quality Action Rectification"**（全文 verbatim 要点）：
> 1) selected rollout latent ẑ^(j^⋆)；2) Convert ẑ^(j^⋆) into future conditioning；3) Re-query VAM with 𝐨_t, 𝐩, 𝐬_t and the selected future condition；4) Generate refined action chunk ã；5) return ã。
> 即：**N 个候选 → RCS 排名 → 最优候选 RCS≥γ 则直接执行；否则全部候选进 ACVS 评分 → 取最高 J 的想象未来 → 用该未来条件再查询 VAM 精修动作**。

2b. **RCS 定义**（Sec VI-A，verbatim）：
> "Given the current context 𝒞_t=(𝐨_t,𝐩,𝐬_t), VAM samples N candidate action chunks {ā^(i)}_{i=1}^{N}. For each candidate, we randomly sample K flow timesteps and re-noise the action according to the same flow-matching process used during training. The re-noised action is then evaluated by VAM's action vector field, producing an average re-denoising error ℰ_RCS^(i). We define the Re-denoising Consistency Score (RCS) as S_RCS^(i) = −ℰ_RCS^(i) (Eq 6), and select i^⋆ = arg max_i S_RCS^(i) (Eq 7)."
> "RCS serves as a lightweight distributional filter. It favors candidates that are more consistent with the learned conditional action manifold while introducing negligible computational overhead compared with rollout-based evaluation."（**K 个 flow 时间步的具体数值原文未取到**，应在附录 -A2）

2c. **LAR 触发与评分**（Sec VI-B，verbatim）：
> "When the selected candidate satisfies S_RCS^(i^⋆) < γ (Eq 8), where γ denotes a reliability threshold, ACVS is invoked to evaluate all candidate actions. For each candidate action chunk, ACVS predicts an imagined rollout and a dense reward trajectory (ẑ^(i), r̂^(i)) = G_ϕ(𝐨_{t−M:t}, 𝐩, ā^(i)) (Eq 9). The rollout value is computed as J^(i) = max_{0≤q<H_a} r̂^(i)_{t+q} (Eq 10), where J^(i) measures the maximum task progress achieved by the imagined rollout. The highest-value rollout j^⋆ = arg max_i J^(i) (Eq 11) is selected as the most promising future."
> "Instead of directly executing the corresponding action, we perform a second policy query conditioned on the selected future rollout. Specifically, the rollout latent"（此处截断，续见 Algorithm 2：转 future conditioning → 重查 VAM → 精修动作）

3. **策略接口 VAM**（Sec IV-A，Eq (1) verbatim）：
> "F_θ(𝐨_t, 𝐩, 𝐬_t) → (ẑ_{t+1:t+H_v}, â_{t:t+H_a−1})"（联合预测未来视频潜 + 连续动作块；H_v 视频视界、H_a 动作视界）
- 架构（IV-B）：Wan2.2-TI2V-5B 实例化；Wan VAE 编码多视角并按宽度拼接成 latent canvas；视频分支 = 原 Wan video DiT 主干（5B 参数）；动作分支 = 0.5B DiT 式 action decoder，经 cross-attention 耦合；合计 5.5B VAM。
- 训练（IV-C）：flow matching，Eq (2)：ℒ_VAM = 𝔼[λ_z‖f_θ^z(𝐳̃,u_z,𝐜_t,𝐩)−𝐯_𝐳‖₂² + λ_a‖f_θ^a(ã,u_a,𝐬_t,𝐡)−𝐯_𝐚‖₂²]，"In all experiments, we simply set λ_z=λ_a=1."

4. **评估接口 ACVS**（Sec V-A，Eq (3) verbatim）：
> "G_ϕ(𝐨_{t−M:t}, 𝐩, ā_{t:t+H_a−1}) → (ẑ_{t+1:t+H_v}, r̂_{t:t+H_a−1})"（给定候选动作块，预测未来视频潜 + 密集奖励轨迹；"ACVS is not an action policy; it treats the candidate action chunk as a clean condition and evaluates the future it induces."）
- 动作条件注入（V-B，Eq (4) verbatim）："𝐜_ℓ^a=ψ_D(𝐛_ℓ), 𝐦_ℓ^a=ψ_6D(𝐛_ℓ)"，注入 diffusion-time embedding 与 AdaLN 调制；"we follow the action-conditioned design of Cosmos"。
- 奖励构造（V-C，verbatim）："We decompose each manipulation task into subtasks and assign progress labels at the subtask level. Frame-level rewards are then estimated through Monte Carlo propagation within each subtask segment, producing dense supervision rather than a single terminal success label." 失败段赋负奖励；额外加入 failure-heavy 与 recovery 轨迹训练模拟器。
- 训练（V-D）：ℒ_ACVS = 𝔼[λ_z‖g_ϕ^z(𝐳̃,u_z,𝐜_{t−M:t},𝐩,ā)−𝐯_𝐳‖₂² + λ_r‖g_ϕ^r(𝐫̃,u_r,𝐡)−𝐯_𝐫‖₂²]，λ_z=λ_r=1。

5. **训练数据（混合监督）**（Sec III，verbatim 数值）：约 27,300 小时 = 17.8K 小时真机遥操作（AGIBOT-G01、ARX、双臂 Franka）+ 6.5K 小时 UMI 风格演示（Gen-DAS Grippers）+ 3.0K 小时第一人称人类视频；modality-specific supervision masks（真机提供动作监督、UMI 提供弱动作信号、egocentric 仅视频动力学）。

**【why 追问】**
1. 为何 coarse-to-fine（先 RCS 再 ACVS）？（Sec VI 原文）"This design preserves real-time performance in most situations while retaining the ability to recover from difficult states." —— ACVS rollout 昂贵，仅在候选不可靠时触发。
2. 为何需要采样+排名而非单次前向？（Sec VI 原文）"Pre-training on large-scale heterogeneous interaction data makes the conditional action distribution inherently multimodal: for the same instruction and scene, the robot may complete the task through multiple feasible action sequences." —— 多模态分布下需挑选「精度/稳健性/成功率」更优的解。
3. 为何失败数据只训练 ACVS 不训练策略？（V-C 原文）"While such data may be suboptimal as direct policy supervision, it is particularly valuable for simulator learning because it exposes the model to off-distribution actions, failed interactions, and recovery behaviors..." —— 失败轨迹对「区分进展与伪成功」的评估器价值大于对策略的价值。
4. 为何 VAM 与 ACVS 共享主干？（II-B 原文）"In contrast, τ0-WM does not use the simulator as a separate module. Its ACVS shares the action interface and backbone configuration with the VAM..." —— 共享预测表征使「想象」与「执行」同分布，评估才可信（推导，基于原文）。

**【关键数值】**
- 已取得（原文）：λ_z=λ_a=1、λ_z=λ_r=1；5B（Wan DiT）/0.5B（Action DiT）/5.5B（VAM 合计）参数；27,300 小时 = 17.8K（真机遥操作）+ 6.5K（UMI）+ 3.0K（egocentric）；J^(i)=max 任务进度（Eq 10）；RCS=−ℰ_RCS（Eq 6）。
- 未取到（截断后）：N（候选数）、γ（可靠性阈值）、RCS 的 K 个 flow 时间步、H_v/H_a 数值、ACVS 调用频率、推理延迟（附录 -A2/-B）。

**【一页总结】**
τ0-WM 的决策结构是「VAM 提议 → RCS 轻量自一致筛选（候选不可靠时触发）→ ACVS 动作条件视频模拟+密集进度评分 → 用最优想象未来条件第二次 VAM 查询精修」。测试时计算分两级：廉价的重去噪一致性排名 + 昂贵的模拟器整流，仅按需触发以保证实时性。任务进度以「子任务级标签 + 段内蒙特卡洛传播」构造密集奖励，失败数据专门强化评估器。Algorithm 1 全文、N/γ 数值与 LAR 细节在取文截断之后。

**【待核实】**
1. N、γ、K（RCS 重去噪时间步数）、H_v、H_a 数值（附录 -A2 部署细节）；
2. VI-B 末段最后一句话（ar5iv 截断于 "the rollout latent"）；
3. VII-B 消融（RCS/LAR 各自增益）与 VII-A 主结果数值。

---

## 3. VLP — Video Language Planning（arXiv:2310.10625）

**元信息**
- 作者：Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, Jonathan Tompson（Google DeepMind / MIT / UC Berkeley）
- 版本：v1 2023-10-16（唯一版本）；CC BY 4.0；Project: https://video-language-planning.github.io/

**来源级别：L3（Algorithm 1 全文 + Sec 2.1/2.2/2.3 原文取得；A/D/B/H 具体数值未取到）**
- 截断位置：`arxiv.org/html/2310.10625v1` 截断于 Sec 3.1 Long-Horizon Video Synthesis 表（UniPi 行的 "Reward" 列处）。
- 未取到（截断后）：Sec 3.2/3.3/4/5 正文与 App A.5 Planning Details（A/D/B/H 数值所在）。

**【决策机制】原文引文 + 逐条指路**

结构（Fig 1 caption，verbatim）：
> "Video Language Planning uses forward tree search via vision-language models and text-to-video models to construct long-horizon video plans. From an image observation, the VLM policy (top left) generates next-step text actions, which a video model converts into possible future image sequences (top right). Future image states are evaluated using a VLM heuristic function (bottom left), and the best sequence is recursively expanded with tree search (middle). Video plans can be converted to action execution with goal-conditioned policies (bottom right)."

1. **搜索目标**（Sec 2.2，Eq (1) verbatim）：
> "we propose to search for a sequence of actions to reach g, corresponding to finding a long-horizon video plan x_{1:H} which optimizes x*_{1:H} = argmax_{x_{1:H}∼f_VM,π_VLM} H_VLM(x_H, g)."

2. **树搜索算法**（Algorithm 1 "Decision Making with VLP"，全文 verbatim 要点）：
   - 超参：Text-Branching factor A、Video-Branching factor D、Planning Beams B、Planning horizon H。
   - plans ← [[x₀] ∀ i∈{1…B}]（初始化 B 个 plan beam）；
   - for h=1…H：for b=1…B：x←plans[b][−1]；a_{1:A}←π(x,g)（生成 A 个文本动作）；video_branches ← [f_VM(a, a_i) for i in (1…A) for j in (1…D)]（**每个动作生成 D 条视频，共 A×D 个分支**）；plans[b].append(argmax(video_branches, H_VLM))；
   - max_idx, min_idx ← argmax(plans, H_VLM), argmin(plans, H_VLM)；plans[min_idx] ← plans[max_idx]（周期性替换最低价值 beam）；
   - plan ← argmax(plans, H_VLM)（返回最高价值计划）。
- Sec 2.2（verbatim）："we use a tree-search procedure based on parallel hill climbing ([Selman & Gomes, 2006])…we first sample a set of A actions using π_VLM(x,g), and for each action we synthesize D different videos using f_VM(x,a). We then use our heuristic function H_VLM(x,g) to select the generated video with the highest heuristic among the A×D generated videos…every 5 steps, we discard the beam with the lowest value and replicate its video plan with the beam with the highest value."

3. **policy / value（heuristic）双模型**（Sec 2.1，verbatim）：
> 策略："a VLM policy π_VLM(x,g)→a…query the VLM for a natural language action to take given…the natural language goal and a tokenized embedding of the current image"（基于 PaLM-E 风格实现；两策略：示例标注 or 在短轨迹片段上微调 PaLM-E）。
> 价值/heuristic："we finetune a PaLM-E model using long trajectory snippets x_{1:H} which accomplish a long horizon goal g, and train it to predict, given an image in the subtrajectory x_t, the number of steps left until the end of the trajectory snippet. The negated number of predicted steps to goal completion from VLM is used to implement H_VLM(x,g) (so that high heuristic value corresponds to being close to goal completion)."
> 动态模型："a text-to-video model f_VM(x,a), which takes an image x and a short horizon text instruction a and outputs a short synthesized video x_{1:S} starting at the image observation x₀"（在图像轨迹片段 + 语言标签上训练）。

4. **视频生成如何进入搜索**：作为动态模型把抽象文本动作落地为具体未来图像状态，再被 H_VLM 评分（Fig 1 + Sec 2.1 原文）。防利用（Sec 2.2，verbatim）：
> "Preventing Exploitative Model Dynamics. When our planning procedure optimizes the VLM heuristic function H_VLM(x,g) it can exploit irregularities in the dynamics model f_VM(x,a) to get artificially high estimates…we discard generated videos from f_VM(x,a) if they increase the heuristic estimate H_VLM(x,g) above a fixed threshold."（阈值数值原文未写于已获取范围）

5. **执行与重规划**（Sec 2.3，verbatim）：
> "we use a short-horizon goal-conditioned policy π_control(x,x_g)…For each frame in our video plan x_{1:H}, the goal-conditioned policy is executed for a fixed pre-specified number of timesteps."
> "Replanning…we use receding horizon control strategy…where we generate videos plans with a fixed horizon (that might not fully complete the task), and then repeatedly regenerate/replan video plans with the same horizon after a fixed number of action executions."

**【why 追问】**
1. 为何要搜索而非直接执行 π_VLM？（Sec 2.2 原文）"directly applying the π_VLM to infer text actions a to reach goal g is not sufficient, as π_VLM is not able to perform sufficiently accurate long-horizon reasoning to select actions that are helpful in the long run. Furthermore, there are many possible low-level image sub-sequences that correspond to different ways to perform a, but it is critical to select one that is consistent with the rest of the actions that must be taken to reach g."
2. 为何视频模型必须进入决策回路？（Sec 1 原文）VLMs "continue to struggle to reason over dynamics e.g., how objects may move or collide with one another over time"；而视频模型 "can model the low-level visual dynamics of objects in ways that are more information-rich than text"。
3. 为何需要 heuristic 剪枝与防利用阈值？树搜索分支 A×D 指数增长，需 H_VLM 剪枝；但 H_VLM 可被动态模型的虚假视频（物体瞬移/遮挡）欺骗，故设阈值丢弃（Sec 2.2 原文）。
4. 为何用 beam/hill-climbing 而非 MCTS 式回溯？（推导，非原文）VLP 无回传价值机制，采用并行爬山 + 周期性最差 beam 替换——原文明确引 Selman & Gomes 2006 的 parallel hill climbing，属"显式搜索但无回溯"结构。

**【关键数值】**
- 已取得（原文）：A/D/B/H 为算法超参（**具体数值未取到**，在 App A.5）；beam 替换周期 = 每 5 步；heuristic 防利用阈值数值未给（原文未写于已获取范围）。
- 计算扩展（Fig 3 verbatim 数据，make line 任务）：Beams/Branch 1/1/1→4%，1/1/4→10%，1/4/4→22%，2/4/4→56%（Fig 3 caption 另提 "Beam 2, Branch 16" 的 extensive planning）；"VLP scales positively with more compute budget"。
- Table 1（计划成功率，Sim/Real：Move Area 58%/78%、Group Color 98%/100%、Make Line 66%/56%）；UniPi 2/4/2 与 4/12/4；VLP No Value Function 10/42/8 与 20/64/4。

**【一页总结】**
VLP = 「VLM 策略提议 A 个文本动作 × 视频动态模型每动作合成 D 条视频 = A×D 候选 → VLM heuristic（步数负值）评分选最优 → B 条并行 beam 递归扩展 H 步 → 每 5 步最差 beam 复制最优 beam」的并行爬山式前向树搜索；规划产物（视频子目标序列）由 goal-conditioned 策略按帧执行，并以滚动时域重规划。测试时计算量与 A×D×B 成正比，图 3 展示成功率随分支预算单调上升。

**【待核实】**
1. A、D、B、H 的具体数值（App A.5 Planning Details）；
2. heuristic 防利用「fixed threshold」数值；
3. Sec 3.2 执行成功率与 3.3 泛化细节（截断后）；
4. π_control 每帧执行的「fixed pre-specified number of timesteps」数值。

---

## 4. 3D-ALP — 3D-Anchored Lookahead Planning（arXiv:2604.11302）

**元信息**
- 作者：Bronislav Sidik, Dror Mizrahi（Huawei Technologies, Israel R&D Center）
- 版本：v1 2026-04-13（唯一版本）；5 pages, 1 figure, 1 table

**来源级别：L3（正文全文取得；仅 References 尾部截断）**
- 截断位置：`arxiv.org/html/2604.11302v1` 截断于 References 列表第 [12] 条（Luo et al.）处，正文无损。

**【决策机制】原文引文 + 逐条指路**

结构（Sec 3.1，verbatim）：
> "3D-ALP consists of four components: a Kinematic Bridge (FK), a World Model oracle (InSpatio-WorldFM), a Hybrid Scorer, and a MCTS Engine. They operate at two timescales. At planning time, the MCTS engine samples candidate joint actions from the kinematic bridge, queries the world model for predicted frames at each resulting c2w, scores those frames with the hybrid scorer, and backpropagates values to select the best action. At execution time, the selected action is physically executed, the real camera frame is observed, and the 3D anchor is updated: 𝐜𝟐𝐰_{t+1}=FK(𝐪_{t+1}) (Eq 1)…Crucially, the MCTS tree is re-rooted at the executed action's child node, preserving all previously computed subtree values."

1. **节点/状态**：c2w 位姿（SE(3) 相机锚）为节点；"The anchor 𝐜𝟐𝐰_t∈SE(3) encodes the camera position and orientation in the world frame. When the robot moves, the anchor is updated via FK — not reset."（Sec 3.2）
2. **世界模型 rollout oracle**（Sec 3.2）："Uses a 3D-consistent world model (InSpatio-WorldFM) to render predicted frames from any c2w query, enabling MCTS rollouts in imagined 3D space." 锚漂移修正 Eq (2)（verbatim）："𝐳_ref ← 0.7·Enc(𝐈_real)+0.3·𝐳_ref"，"The blend ratio α=0.7 was selected by a line search over α∈{0.5,…,0.9} on a held-out validation trajectory, minimising anchor drift at step 5."
3. **评分**（Sec 3.3，Eq (3) verbatim）："S_total = S_semantic · max(0, 1−‖𝐜𝟐𝐰[:3,3]−𝐜𝟐𝐰_goal[:3,3]‖₂)"；消融用纯几何评分 Eq (4)："S_exact = max(0, 1−d_3D)"。
4. **回溯/四个结构修复**（Sec 3.4，verbatim 要点）：
   - (F1) Zero-action exploitation trap："UCB1 selects the most-visited child…the zero action ("stay still") accumulates visits before exploration, causing stutter. Fix: select by Max-Q value, filtering zero-magnitude actions explicitly."
   - (F2) Tree depth decay："After advance_root, reused children retain their original depth values, shrinking the effective lookahead horizon to zero after a few steps. Fix: recursive reset_depths after every re-rooting."
   - (F3) Standard averaging penalty："UCT backpropagates the mean score. One perfect path through the tree is diluted by poor sibling branches. Fix: Max-MCTS — backpropagate max_value rather than mean."
   - (F4) UCB1 constant mismatch："The standard c=√2≈1.414 is calibrated for binary scores {0,1}. With continuous kinematic distance scores (Δ∼0.01–0.05), the exploration term overwhelms exploitation…Fix: c=0.02, empirically calibrated so the exploitation signal dominates while maintaining meaningful exploration (verified in 'mcts_engine.py')."
5. **超参确认**（Sec 4.1 Setup，verbatim）：
> "3D-ALP uses lookahead depth D=2 and branching factor B=4 (the number of candidate joint actions sampled per node), a 2 s planning budget, and 10 physical actions per step. The greedy baseline samples the same number of actions per step and executes the one minimising immediate distance."

**【why 追问】**
1. 为何必须测试时搜索（MCTS）而非贪心反应式？（Sec 1 原文）"This is not a limitation of model capacity; it is a limitation of the architecture — there is no mechanism for persistent scene memory." 贪心在记忆步骤 0.006 vs 3D-ALP 0.650。
2. 为何用 persistent c2w 锚而非 POMDP 随机信念？（Sec 2 原文）"the 3D anchor provides a deterministic belief update via forward kinematics, replacing the stochastic belief propagation typical in POMDP solvers"；且树中保留遮挡目标的历史 c2w 节点值，"the 3D anchor does not reset when an object becomes occluded"。
3. 为何 UCT 需要四个修复？（Sec 3.4 原文）连续机器人操作（连续分数、零动作、re-rooting）与离散棋类（二进制分数）结构不同——F1–F4 逐条给出失败模式与修复。
4. 为何要 hybrid 几何-语义评分？（Sec 3.3/5 原文）"A gripper floating 15 cm above a target object can score identically to a gripper touching it."（VLM 深度盲）；"Florence-2 returns flat scores (≈0.0) for all nodes in a planning cycle…so that UCB1 reduces to pure exploration."

**【关键数值】**
- 已取得（原文）：D=2、B=4、2 s 规划预算、每步 10 个物理动作、c=0.02（vs √2）、α=0.7（0.3 权重旧参考）、渲染 ≈2400 ms/frame（RTX A6000）、2 s 预算 ≈5 nodes/cycle vs 理论容量 20（D=2×B=4）、4-GPU 并行 ≈600 ms、30 episodes × 3 seeds。
- 结果（Table 1）：Greedy Memory SR 0.006±0.008 / Step5 0.000；MCTS D=1 0.539±0.064 (+0.533, 82% of gain) / Step5 0.622；3D-ALP D=2 **0.650±0.109** (+0.645) / **Step5 0.822**（vs greedy 0.000）；p<0.001。
- Phase 0–1：SSIM=1.000、ORB=1.000 (391/391)、bridge latency 0.5 ms、angular error 0.00°。

**【一页总结】**
3D-ALP = 显式 MCTS（UCT 改良）+ 3D 世界模型作 rollout oracle + 混合几何-语义评分。决策结构：FK 桥采样候选关节动作（B=4）→ 世界模型渲染未来帧 → 评分 → Max-Q/Max-MCTS 回溯（c=0.02）→ 深度 D=2 前瞻 → 执行后树 re-root 保留子树值（持续空间记忆）。消融把增益分解为「树搜索空间记忆 +0.533（82%）」与「更深前瞻 +0.111（17%）」；瓶颈是 VLM/渲染（2400 ms/帧），Phase 2 计划换 JEPA 潜空间评分。

**【待核实】**
1. mcts_engine.py 中 UCT 实现的更多细节（作者未在文中给出全部代码）；
2. 真实机器人验证（原文声明 "Real-robot validation…is future work"）；
3. Phase 2（LeWorldModel/JEPA）实现进展（本文仅为计划）。

---

## 5. DreamAvoid — Critical-Phase Test-Time Dreaming（arXiv:2605.11750）

**元信息**
- 作者：Xianzhe Fan, Yuxiang Lu, Shenyuan Gao, Xiaoyang Wu, Ruihua Han, Manling Li, Hengshuang Zhao（HKU / HKUST / Northwestern）
- 版本：v1 2026-05-12；v2 2026-09-23（最新，本次取文版本）；23 pages, 7 figures；Code: https://github.com/XianzheFan/DreamAvoid

**来源级别：L3（Sec 2 pilot + Sec 3.1–3.3 全原文含公式与数值；3.4 后半与第 4–7 节、附录未取到）**
- 截断位置：`arxiv.org/html/2605.11750v2` 截断于 Sec 3.4 句 "We deploy the initial version of DreamAvoid (trained only on 𝒟_teleop" 处。
- 未取到：3.4 的 DA-ABL 循环细节、Sec 4–7、App C.2 Table 5（σ 数值）、App D.3 延迟分析。

**【决策机制】原文引文 + 逐条指路**

结构（Sec 3 开头，verbatim）：
> "The core idea is that the robot directly executes the frozen base VLA policy during most time steps, and only triggers test-time dreaming when it predicts an imminent transition into a critical phase. DreamAvoid consists of three modules: Dream Trigger, Action Proposer, and Dream Evaluator."

1. **Dream Trigger（触发条件）**（Sec 3.1，verbatim 要点）：
> "a lightweight Dream Trigger p_t=f_φ(𝒪_{t−K+1:t}, s_t)…inputs are (1) single- or multi-view observations from the recent K frames…and (2) the current proprioceptive state s_t. Its output p_t∈[0,1] is an onset-detection score. We implement the trigger with a frozen DINOv2 encoder and a three-layer MLP."
> 软标签："ỹ_t = Sigmoid((t−t_crit)/β)"；损失："ℒ_trigger = −1/T Σ_t [w_+ ỹ_t log p_t + (1−ỹ_t) log(1−p_t)]"，"w_+=N_−/N_+"。
> 干预规则："The intervention condition is c_t^test = (1−d_t)𝟙[p_t≥γ]…If c_t^test=0, the system executes the base-policy action chunk. Otherwise, the scheduler sets d_t=1 and launches one cycle of action proposal, candidate evaluation, and selection."（每阶段独立 prediction head 与 flag）
> 数值（verbatim）："We use β=5 frames and γ=0.5, with asynchronous trigger polling at 1 Hz and serialized evaluator calls."；每任务 100 个训练视频约 26,000–34,000 帧。
2. **Action Proposer（候选生成）**（Sec 3.2，verbatim 要点）：
> 基座策略为 flow matching："the base policy is modeled using flow matching…During standard inference…it typically outputs only one default action chunk. To obtain multiple candidate action chunks during testing, DreamAvoid replaces this deterministic ODE inference process with its corresponding stochastic SDE sampling process."
> ODE："du = v_θ(u,ρ∣o_t,l)dρ"；SDE："du = v_θ(u,ρ∣o_t,l)dρ + σ(ρ)dw_ρ"，"σ(ρ) controls the sampling noise intensity, and w_ρ denotes the Wiener process. In practice, we adopt a constant noise scheduling strategy. The specific values of σ for different tasks can be found in Table 5, Appendix C.2."（σ 数值未取到）
> 定位："all candidate actions still originate from the velocity field of the same pre-trained policy; therefore, they are not blind searches in an unconstrained action space, but are guided by the task semantics and action priors encoded in the base policy."
3. **Dream Evaluator（候选评分）**（Sec 3.3，verbatim 要点）：
> 世界模型："for each candidate action chunk u_t^(k)…we use the distilled DreamDojo to predict its corresponding future observation sequence ŷ_t^(k) = 𝒲_ψ(o_t, u_t^(k))"。
> 价值标签："Robometer…predicting frame-level task progress η∈[0,1]…We use z_t = η_{t+H} − η_t, with z_t=−1 for labeled irreversible failures."
> 混合训练（real-dream joint training）：真实未来序列 (o_t,u_t,y_t^true)→z_t 与梦想序列 (o_t,u_t,ŷ_t)→z_t 双输入。
> 批采样（DA-ABL）："20% of the data is sampled from the terminal state horizons of successful completions, 40% is drawn from the additionally collected boundary and failure data…and the remaining 40% comes from routine steady progression phases."
> 损失："ℒ_VE = 1/|ℬ| Σ_{i∈ℬ} L_δ(v̂^(i), z^(i)) + λ_rank/|𝒫| Σ_{(i,j)∈𝒫} max(0, −(v̂^(i)−v̂^(j)) + α(z^(i)−z^(j)))"（Huber + 连续 margin ranking）。
> 候选数（verbatim）："During testing, the Dream Evaluator selects the candidate action with the highest predicted progress delta v̂^(k). At deployment, the distilled value model scores all **eight** candidates in approximately 40 ms; Robometer is used offline to generate training labels." → **候选数 = 8，评分耗时 ≈40 ms**。

**【why 追问】**
1. 为何选择性触发而非 always-on？（Sec 1 原文）"Always-on inference can be costly; executing heavy rollouts at every replan incurs additional computation even during routine execution…This motivates selective intervention that concentrates computation on consequential decisions while retaining the base policy during routine execution."
2. 为何要显式时空 dreaming（世界模型渲染）而非 VLM 直接读数值？（Sec 2 pilot 原文）直接语义评估 57.5%（joint）/67.5%（EEF）vs 显式 dreaming 80.0%；"By explicitly simulating numerical actions forward into visual futures, the world model serves as a critical spatiotemporal bridge. It translates abstract numerical errors into explicit visual deviations."
3. 为何混合成功/失败/边界数据训练评估器？（Sec 1 原文）"If trained solely on success data, the world model may generate optimistic hallucinations due to a lack of negative boundary constraints."（3.4 称失败与边界数据让世界模型拥有 "nightmares"）
4. 为何 SDE 采样而非动作空间盲搜索？（Sec 3.2 原文）候选仍出自同一策略的速度场，"guided by the task semantics and action priors"；且"multiple seemingly reasonable action chunks might lead to drastically different downstream outcomes"（关键相位的小误差放大）。

**【关键数值】**
- 已取得（原文）：候选数 8；评分 ≈40 ms；β=5 frames、γ=0.5、轮询 1 Hz；批采样比 20/40/40；失败目标 z=−1；真实世界总成功率 72.5% vs 基座 48.8% vs GPC-RANK 54.4%（摘要）；pilot 准确率 80.0%（vs 57.5/67.5）；每任务 100 视频 26–34K 帧。
- 未取到：σ（Table 5）、K（触发器帧窗口）、H（动作块长度）、延迟分解（App D.3）。

**【一页总结】**
DreamAvoid 是「按需触发的测试时想象」：冻结基座 VLA 正常执行；轻量 Dream Trigger（DINOv2+MLP，软标签 onset 检测，p_t≥γ=0.5 触发，每阶段一次性）检测关键相位；Action Proposer 把 flow-matching 的 ODE 推理换成 SDE 采样得到 8 个候选动作块；Dream Evaluator（蒸馏 DreamDojo 世界模型 + Robometer 标签训练的价值模型，Huber+ranking loss，real-dream 联合训练）对 8 个候选的短视界未来「梦想」打分（约 40 ms），选最高进度增量者执行。失败/边界数据（20/40/40 采样）专门强化评估器的失败意识。

**【待核实】**
1. Sec 3.4 DA-ABL 自主边界学习循环的后半段原文；
2. σ（App C.2 Table 5）、K、H 数值；
3. Sec 4 真实实验与 Sec 5 仿真基准细节（截断后）；
4. App D.3 延迟分析与 call counts。

---

## 6. PointWorld — Scaling 3D World Models（arXiv:2601.03782）

**元信息**
- 作者：Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei（Stanford University / NVIDIA）
- 版本：v1 2026-01-07（唯一版本）；Project: https://point-world.github.io/

**来源级别：L3（Sec 3.1 + Sec 3.2 MPC/MPPI 集成原文取得；K/T/β 数值与 Sec 5.4 实验未取到）**
- 截断位置：`arxiv.org/html/2601.03782v1` 截断于 Sec 3.1 Training Objective 的 Eq (1) 之后（"where" 一词处）；`ar5iv.labs.arxiv.org/html/2601.03782` 更深，截断于 Sec 4 Model Evaluation Protocol 句 "we focus on the metric on moving points (" 处。
- 未取到（截断后）：Sec 5.4（Model-Based Planning 实验）、App A.6.1（K/T/β 数值所在）。

**【决策机制】原文引文 + 逐条指路**

定位（Intro，verbatim）：
> "To demonstrate PointWorld's potential for manipulation, we integrate it with a model-predictive controller (MPC) for action inference on a real robot. As PointWorld predicts scene dynamics jointly over short action chunks in a single forward pass at a real-time latency (0.1s), it provides a natural and efficient integration with sampling-based MPC (e.g., MPPI [12])."

1. **chunked 动态（0.1s/rollout 的依据）**（Sec 3.1，verbatim）：
> "we adopt a multi-step (chunked) formulation…the model predicts future states over a horizon H in a single forward pass ℱ^H_θ:(𝐬_t, 𝐚_{t:t+H−1})→𝐬_{t+1:t+H}, which improves temporal consistency and amortizes computation. **We use H=10 steps and 0.1s per step.**"
> "This chunked formulation delivers extremely efficient inference capable of evaluating many candidate trajectories with a real-time latency (**0.1 s per batched forward pass**), which stands in contrast to pixel-based approaches that typically require seconds-long inference due to the use of diffusion objectives."
2. **状态/动作表示**（Sec 3.1）：状态 = 场景点流 {(𝐩_{t,i}, 𝐟^S_i)}；动作 = 机器人点流（URDF + FK 前向传播，仅采样 gripper 几百点），"an embodiment-agnostic description of interaction geometry over the horizon"。
3. **训练目标**（Sec 3.1，verbatim 要点）：movement weighting m_{k,i}=σ(κ(δ_{k,i}−τ)) 归一化为 w_{k,i}；aleatoric uncertainty（预测 log-variance s_{k,i}）；Eq (1)：½Σ_{k,i}^{H,N_S} w_{k,i}(ρ_δ(P̂−P)·e^{−s_{k,i}} + s_{k,i})（Huber on 3D residual × uncertainty weight + uncertainty reg.）。
4. **MPC/MPPI 集成**（Sec 3.2，verbatim）：
> "we integrate PointWorld in an MPC framework with a sampling-based planner MPPI ([Williams et al. 2017]) that plans a sequence of T end-effector pose targets in SE(3) given a cost function defined in the model's state space."
> 采样："We then sample K action perturbations ℓ_{1:K} using a time-correlated (cubic-spline) noise distribution, which are added to a nominal end-effector trajectory. For each sampled trajectory 𝐄^(ℓ)_{1:T}, the corresponding robot point-flow actions 𝐚^(ℓ)_{1:T} are constructed, scene flows are rolled out by PointWorld conditioned on 𝐚^(ℓ)_{1:T}, and a trajectory cost J^(ℓ) is accumulated."
> 更新："The nominal trajectory is iteratively refined by computing exponentiated weights ω_ℓ ∝ exp(−J^(ℓ)/β) over samples and updating the nominal as a weighted average of sampled trajectories, where β is non-negative temperature."（标准 MPPI 指数权重更新；**K、T、β 数值原文未取到**，在 App A.6.1）
5. **代价函数**（Sec 3.2，Eq (2) verbatim）：
> 任务代价："c_task(s_k) = 1/|ℐ_task| Σ_{i∈ℐ_task} ‖𝐩_{k,i} − 𝐠_i‖₂². Such pointwise goal costs apply broadly across rigid, deformable, and articulated objects. Task-relevant points can be specified by either human via GUI or by VLMs."
> 全局优化问题："argmin_{𝐄_{0:T}} Σ_{k=1}^{T} [c_task(s_k) + c_ctrl(𝐄_k)]  s.t.  𝐬_{1:T} = ℱ^T_θ(s_0, 𝐚_{1:T}), 𝐄_0 = 𝐄_measured, where c_ctrl subsumes path length and reachability regularization."

**【why 追问】**
1. 为何 0.1s 推理是决策结构的前提？（原文）MPPI 需要评估大量候选轨迹；像素扩散模型"typically require seconds-long inference"不可用于采样 MPC；chunked 点云回归单次前向把延迟降到 0.1s，从而 "evaluating many candidate trajectories" 可行。
2. 为何 chunked（H=10）而非单步递推？（原文）"improves temporal consistency and amortizes computation"。
3. 为何动作也用点流？（原文）跨具身学习 + 保证动作"fully, rather than partially, observable"（遮挡接触情形，如抱大箱时）；且直接条件于机器人几何而非关节空间。
4. 为何选 MPC 而非学策略？（原文）"a single pre-trained checkpoint enables a real-world Franka robot to perform…without requiring any demonstrations or post-training"——世界模型 + MPC 允许零演示泛化（推导基于 Intro/摘要原文）。

**【关键数值】**
- 已取得（原文）：H=10 步、0.1 s/step、0.1 s per batched forward pass；数据集 ≈2M 轨迹 / ≈500 小时（DROID + BEHAVIOR-1K）；DINOv3 冻结特征；gripper 点采样"a few hundred points per gripper"；MPPI 为 T 步 SE(3) 末端姿态序列 + K 个三次样条时间相关扰动 + 指数权重 ω∝exp(−J/β)。
- 未取到：K（扰动数）、T（MPC 视界）、β（温度）数值（App A.6.1）；τ、κ 数值。

**【一页总结】**
PointWorld 把决策表达为「0.1s 级 3D 点流世界模型 + 采样基 MPC（MPPI）」：状态与动作统一为 3D 点流，chunked 单次前向预测 H=10 步全场景位移，使 MPPI 能在 0.1s/rollout 的预算下评估大量候选动作序列，实现在野单张 RGB-D 零演示的推/变形体/关节体/工具使用。MPC 集成细节（代价函数、采样数）的原文在本次取文截断之后。

**【待核实】**
1. K、T、β 数值与 App A.6.1（Model-Based Planning 细节）；
2. Sec 5.4 Model-Based Planning 实验与真实机器人结果；
3. τ、κ（movement weighting 阈值/温度）数值。

---

## 7. WAP — World Action Planner（arXiv:2607.27599）

**元信息**
- 作者：Xiangcheng Zhang, Runhan Huang, Yilun Du（Harvard University）
- 版本：v1 2026-07-30；v2 2026-10-02（最新，本次取文版本）；CC BY 4.0；Project: https://worldactionplanner.github.io/

**来源级别：L3（Sec 3.1 + Sec 3.2 前半 + Algorithm 1 全文取得；3.2.2 后半与超参数 N 未取到）**
- 截断位置：`arxiv.org/html/2607.27599v2` 截断于 Sec 3.2.2 句 "This procedure is illustrated in Figure" 处。
- 未取到（截断后）：3.2.2 余下（global/local 优化细节）、Sec 4 理论、Sec 5 实验、App C.2.3 Search Details（N 等超参）、App B.6（bound 实例化数值）。

**【决策机制】原文引文 + 逐条指路**

结构（Abstract，verbatim）：
> "The search proceeds in a coarse-to-fine manner. First, the agent performs global action optimization by reasoning over imagined world-model rollouts to identify potential failures and refine the proposed action plan. It then performs local action search, comparing the imagined future outcomes of neighboring candidates to select the best action for execution."

1. **Algorithm 1 "World Action Planner (WAP)"**（全文 verbatim 要点）：
   - 输入：VLM Agent、robot controller φ、world model WM、optional policy π；Task description ℓ、initial state s、env。
   - 循环：
     - Agent Action Proposal：g = Agent.Proposal(s, ℓ)（VLM 提议原始动作目标）；a = φ(g, s_proprio)（翻译为机器人动作）；
     - Global Optimization：ŝ_next = WM(s, a)（世界模型想象）；Δg = Agent.Eval(ŝ_next, ℓ)（VLM 反馈）；g ← g + Δg；a = φ(g, s_proprio)；
     - Local Search and Refinement：g_1,…,g_N = GridSearch(g)；a_i = φ(g_i, s_proprio)；ŝ_i = WM(s, a_i)；if use policy：ŝ_i = WM(ŝ_i, π(ŝ_i))（叠加策略 rollout 想象）；i* = Agent.Rank(ŝ_1,…,ŝ_N, ℓ)；
     - Execute：s, done = env.Step(s, a_{i*})；if use policy：s, done = env.Step(s, π(s))。
2. **外环 VLM + 内环动作条件 WM 想象**（Sec 3.2.2 开头，verbatim）：
> "Instead, we organize search into a coarse-to-fine hierarchy that operates in a low-dimensional goal space. Guided global optimization first uses agentic feedback to correct large, semantically meaningful errors in the proposed trajectory, while physically grounded local search subsequently concentrates computation on a compact neighborhood of plausible actions for fine-grained refinement."
> 动机（verbatim）："Generic sampling methods such as Best-of-N and CEM are therefore inefficient in this setting: they explore large action spaces through weakly structured proposal distributions, causing many world-model evaluations to be spent on physically implausible candidates, redundant variations, or actions that repeat the same underlying failure."
3. **世界模型的条件方式**（Sec 3.1，verbatim）：
> "we first compute the future robot joint configurations induced by the actions and render them as pose-skeleton images from the corresponding camera viewpoints. This representation makes the robot's future motion explicit in the visual input, allowing the world model to focus on predicting how the environment responds to that motion."
> "We adopt multi-view prediction by concatenating third-person and wrist-view camera feeds into a grid…we train the model using diffusion-forcing with the flow-matching objective…keeping the pose-image tokens noise-free for conditioning."
4. **策略先验组合**（Sec 3.2.1，verbatim）：
> "When policy priors are available for local manipulation, such as grasping, the agent flexibly composes them in states where the policy remains in distribution…the agent first navigates to the vicinity of the target object and then searches for states from which the local grasping policy is likely to succeed by imagining future policy rollouts with the world model."
> VLM 原始动作（verbatim）："The VLM generates a sequence of primitive actions, such as MOVE, ROTATE, GRASP, and RELEASE…We then triangulate the corresponding 2D pixel coordinates across views to recover the target position in 3D space, eliminating the need for explicit depth information."

**【why 追问】**
1. 为何不用 MPPI/Best-of-N 而做 agentic 分层搜索？（Sec 3.2.2 原文）高维连续动作空间 + 稀疏成功集："little continuous guidance for optimization"；弱结构提议分布浪费世界模型评估于"physically implausible candidates, redundant variations, or actions that repeat the same underlying failure"。摘要/Intro 另声称分层搜索"achieves substantially greater planning efficiency than conventional action-sampling methods such as MPPI and Best-of-N"（原文）。
2. 为何世界模型要 pose-image 条件而非低维动作向量？（Sec 3.1 原文）"low-dimensional action representations can entangle robot kinematics with environment dynamics, making generalization to novel actions difficult"——姿态骨架图让动作在视觉输入中显式化，模型可复用物体交互动力学泛化到未见动作。
3. 为何用 VLM 做提议+评估+排序的外环？（Sec 3.2.1 原文）"we leverage the broad world knowledge of foundation VLMs to propose action primitives in settings where demonstration data is unavailable"。
4. 为何「if use policy」叠加策略 rollout？（Sec 3.2.1 原文）把策略当"modular action priors"，先导航到目标附近再在想象中搜索策略可能成功的状态——扩展策略分布外泛化能力。

**【关键数值】**
- 已取得（原文）：局部候选数 N（GridSearch 生成 g_1..g_N，**数值未取到**，在 App C.2.3）；动作原语集合 {MOVE, ROTATE, GRASP, RELEASE}；三角化恢复 3D 目标位置（无深度信息）。
- 未取到：N、global 优化轮数、VLM 具体型号、世界模型规模、规划耗时（App E Wall Clock Time）。

**【一页总结】**
WAP = 「VLM 外环提议/评估/排序 + 动作条件（pose-image）视频世界模型想象」的粗-细两层搜索：粗层全局优化在目标空间用 Agent.Proposal→WM 想象→Agent.Eval 反馈闭环修正大语义错误；细层局部搜索对目标做 GridSearch 生成 N 个邻域候选，逐一到动作并 WM 想象（可选叠加策略 rollout），由 Agent.Rank 选最优执行。与 MPPI/Best-of-N 的对立论据是「弱结构提议分布浪费想象预算」，分层搜索把计算集中到低维目标空间。

**【v1 版本补充（ar5iv 渲染为 v1：标题 "Generalizable Decision-Making"，2 位作者）】**
- v1 Algorithm 1 措辞略有差异：Agent.ProposeActions(s,ℓ) / Agent.Optimize(ŝ_next,ℓ) / GridSearch(a)（v2 为 Agent.Proposal / Agent.Eval / GridSearch(g)）；结构一致：提议→全局优化（VLM 反馈）→局部搜索（网格候选 + WM 想象 + Agent.Rank）→执行。
- Global Optimization（v1 Sec 3.2，verbatim）："we imagine the action trajectory using our action-conditioned world model and prompt the VLM to evaluate whether the actions are safe and aligned with the intended goal based on the imagined rollout video. If a trajectory is deemed suboptimal, the VLM provides high-level semantic feedback to refine the action sequence. For example, if the gripper risks a collision, the VLM may suggest increasing the height for clearance; similarly, if an object is dropped behind the box, the model might suggest a forward adjustment."
- Local Search（v1 Sec 3.2，verbatim）："Although VLMs exhibit robust semantic understanding of spatial relationships, they inherently lack absolute metric grounding, such as directly outputting precise physical coordinates or distances from visual inputs. To overcome this, we transition to a discriminative selection process: we employ a grid search to sample a set of candidate actions and leverage the VLM's evaluative capabilities to identify the optimal trajectory, using the imagined rollout videos from the action-conditioned world model."
- Policies as Tools（v1 Sec 3.2，verbatim）："we treat existing imitation learning policies as modular tools, enabling the integration of diverse models, such as diffusion policies, VLAs, or WAMs…synthesize novel actions using our full model-based planning system in OOD scenarios."
- Sec 4 理论（v1，verbatim 要点）：Theorem 1——模型基算法 K 幕数据后对任意任务 c 次优间隙 Õ(1/√K)，且存在实例使模仿学习跨任务平均次优间隙 ≥ Ω(|𝒞|/K)；Theorem 2（线性函数逼近）——模型基 Õ(1/(κ√K))；Theorem 3——多任务下最优策略无法由任务特征 ψ(c) 的 ≤|𝒜| 次多项式实现（单任务则为线性）；"training a generalist expert policy is often difficult due to distinct behaviors across tasks, whereas a generalizable multi-task reward model can be effectively obtained using pretrained vision and language models"。
- Sec 5.1 世界模型设置（v1，verbatim）：Wan-T2V-1.3B 主干、10K 步微调、batch 64、21 帧历史 @7FPS → 20 帧未来 @20FPS、4 视角 2×2 网格（224px/视角）、20 diffusion steps；训练数据以 MPPI 算法风格高斯噪声扰动演示轨迹。

**【待核实】**
1. Sec 3.2.2 后半原文（global/local 优化具体步骤与 Figure 3 说明）；
2. N 数值与 App C.2.3 Search Details 超参；
3. Sec 4 理论（模型基规划 vs 模仿学习的泛化界）与 B.5 WAP 误差界；
4. Sec 5 实验结果与 App E 规划耗时。

---

## 8. Cosmos 3 — Omnimodal World Models for Physical AI（arXiv:2606.02800）

**元信息**
- 作者：NVIDIA（200+ 作者）；版本：v1 2026-06-01，v4 2026-06-23（最新，本次取文版本）；CC BY 4.0；OpenMDW-1.1 许可；Repo: https://github.com/nvidia/cosmos ；HF: https://huggingface.co/collections/nvidia/cosmos3

**来源级别：L2+（Abstract/Intro/Fig 1-2 caption/Table 1/Sec 2 架构开头取得；Reasoner 具身 CoT 与 2D 轨迹规划机制原文未取到）**
- 截断位置：`arxiv.org/html/2606.02800v4` 首次请求超时（30s），重试成功但正文仅取到目录（TOC 巨大，占满额度）；`ar5iv.labs.arxiv.org/html/2606.02800` 截断于 Sec 2.1.1 句 "The VAE compresses the input video temporally by 4× and spatially by" 处。
- 未取到（超长技术报告，决策相关小节在取文范围之外）：Sec 3.1.2 SFT「Robotics and embodied AI」（Reasoner 具身 CoT 数据）、Sec 4.1.2、Sec 6.1「Robotics」（Reasoner 评测）、Sec 6.3.2「Cosmos 3 Reasoner as Prompt Upsampler」、Sec 2.1.3/2.2/2.3（动作编码与生成模式细节）。

**【决策机制】原文引文 + 逐条指路**

结构（Fig 1 caption，verbatim）：
> "Cosmos 3 serves as a general-purpose backbone for Physical AI. By jointly modeling language, image, video, audio, and action for both understanding and generation, Cosmos 3 unifies a wide range of model classes within a single network architecture, including vision-language models, image generation models, audio-visual generation models, policy or world-action models, forward dynamics models, and inverse dynamics models."

1. **双塔衔接（官方博客，次级来源，verbatim）**（https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/ ）：
> "**Reasoner tower**: A vision-language model (VLM) that interprets multimodal observations like images, videos, and text. This tower uses an autoregressive architecture to interpret the input and understand motion, object interactions, and other physical context. This serves as the 'brain' that reasons about the world before any generation happens. **Generator tower**: Generates future observations and action sequences. This tower uses a diffusion-based process to generate physics-aware video and action outputs that are conditioned on the reasoner tower's understanding. **The reasoner can be called independently, but the generator always activates both towers for guided generation.**"
> Fig 3 caption（博客，verbatim）："…an autoregressive reasoner tower that takes in text, image, video, audio, and action inputs is connected to a diffusion-based generator tower that outputs text, image, video, audio, and action. **Information from the reasoner tower feeds unidirectionally into the generator tower, which enables coherent generation.**"
2. **WAM 模式（论文 Intro，verbatim）**：
> "Depending on the input-output configuration, Cosmos 3 seamlessly transitions between multiple operational modes: it can operate as…a world-action model for joint action prediction and environmental simulation. By unifying perception, simulation, and execution without architectural modifications, Cosmos 3 eliminates the need for fragmented, task-specific pipelines."
3. **动作作为一等模态（论文 Sec 2 开头，verbatim）**：
> "Beyond language, vision (image and video), and audio, Cosmos 3 treats action as a core modality, introducing a dedicated class of action tokens. These action tokens bridge the physical world with language-based reasoning and video-based world modeling, linking directly to physically grounded control signals for real-world interaction."
4. **具身 CoT 与 2D 轨迹规划**：TOC 显示内容位于 Sec 3.1.2 SFT 子项「Robotics and embodied AI」、Sec 6.1「Reasoner Evaluation > Robotics」、Sec 6.3.2「Cosmos 3 Reasoner as Prompt Upsampler」、附录 G.1 贡献分组含「2D Grounding」——**这些小节的正文原文未取到**。Reasoner→Generator 的衔接方式从已取到文本归纳为：Reasoner 先做理解/推理（自回归塔），其表征单向馈入 Generator 扩散塔，引导视频/动作生成（依据：博客 Fig 3 与双塔描述；论文原文细节未取到）。

**【why 追问】**
1. 为何把推理与生成合一？（论文 Intro 原文）"Under the current paradigm, the robot must stitch together a disjointed suite of models…This fragmented architecture is suboptimal and computationally wasteful."；"understanding requires reasoning about the future evolution of the world and the consequences of actions, while generation relies on a compact, structured representation of the world and agent behaviors."
2. 为何生成必须激活双塔而非独立？（博客原文）"the generator always activates both towers for guided generation"——生成由推理塔的理解条件化，保证「物理感知的视频与动作输出」。（次级来源，论文同义的原文在 6.3.2 未取到）
3. 为何把动作做成 token 模态？（论文 Sec 2 原文）使动作「bridge the physical world with language-based reasoning and video-based world modeling」——动作与语言/视觉共享表征才能统一推理与执行。（推导自原文）

**【关键数值】**
- 已取得（论文 Table 1）：Cosmos3-Super/Nano Reasoning-Robotics 57.8/55.1；FD: Robot 26.0*/25.5*（vs Ctrl-World 23.0）；Policy: Robot Nano 39.7*（vs π0.5 28.1，Super 无策略值）。
- 已取得（官方博客）：Nano 16B / Super 64B 参数；博客 Table 1 模态表：「Video|Text → Video|Action = World action model, video action model, VLA, policy model」「Action|Video|Text → Video = Action-conditioned world model」。
- 未取到：具身 CoT 长度/格式、2D 轨迹规划的表示与推理步数、推理-生成接口的量化配置。

**【一页总结】**
Cosmos 3 的决策结构是「双塔 MoT」：自回归 Reasoner 塔（VLM，理解+推理+具身 CoT/2D grounding）单向馈入扩散 Generator 塔，由推理表征条件化地生成未来视频与动作（WAM 模式 = 联合动作预测 + 环境模拟）。动作被设计为一等 token 模态。本次取文仅达架构总述层（正文截断于 2.1.1）；Reasoner 的具身 CoT 与 2D 轨迹规划机制正文（3.1.2/4.1.2/6.1/6.3.2）未取到，需全文本地文件补齐。

**【待核实】**
1. Sec 3.1.2「Robotics and embodied AI」中 Reasoner SFT 的具身 CoT 数据构造原文；
2. 2D 轨迹规划/2D Grounding 的表示与推理格式（G.1「2D Grounding」贡献组对应正文）；
3. Sec 6.3.2 Reasoner as Prompt Upsampler 的 prompt contract（附录 B.1 模板）；
4. Sec 2.1.3 动作表示与 tokenization、Sec 2.2.2 生成模式切换细节。

---

## 9. Meta WAM 官方博客

**URL**：https://ai.meta.com/blog/wam-world-action-model/
**获取结果：未能获取。**

尝试记录（按时间序）：
1. pwsh `Invoke-WebRequest`/`curl.exe` 直连 → 沙箱阻断网络（"无法连接到远程服务器"；TLS 亦失败）；
2. web_fetch `https://ai.meta.com/blog/wam-world-action-model/` → 2 次 "TypeError: fetch failed"（网络层失败）；
3. web_fetch `https://r.jina.ai/https://ai.meta.com/blog/wam-world-action-model/`（代理）→ "TypeError: fetch failed"；
4. web_fetch `https://web.archive.org/web/2026/https://ai.meta.com/blog/wam-world-action-model/` → "TypeError: fetch failed"；
5. 按任务要求改试 Hugging Face papers 搜索：web_fetch `https://huggingface.co/papers?q=world+action+model` → "TypeError: fetch failed"；web_search 未返回 Meta WAM 对应的 HF papers 页面（仅出现无关条目：MaskWAM 2606.13515、世界行动模型综述 2606.20781、MA-WAM 2609.31281）。

按铁律，不凭记忆补写任何博客内容。后续可用本地浏览器/代理重取该 URL 后补充本节。

---

# 汇总对比表：决策结构 × 代表论文 × 测试时计算量 × 依据

| 决策结构 | 代表论文 | 测试时计算量 | 依据（原文） |
| --- | --- | --- | --- |
| 潜空间 MPPI 式迭代推断（隐式规划） | WAV 2604.14732 | K 迭代 × M 组视频样本 × 每组 N 个价值样本 → top-K1/K2 精英更新（M/N/K 数值未取到） | Algorithm 1；Sec 4.3 |
| propose–evaluate–revise（轻量 RCS + 按需 ACVS 整流） | τ0-WM 2606.01027 | N 候选 × K 个 flow 时间步重去噪（轻量）；RCS<γ 时追加 N 次 ACVS rollout + 二次 VAM 查询（N/γ/K 数值未取到） | Algorithm 1/2；Eq 6–11 |
| 前向树搜索（并行爬山 beam，无回溯） | VLP 2310.10625 | 每步 A×D 条视频分支 × B 条 beam × H 视界；每 5 步 beam 替换（数值未取到） | Algorithm 1；Sec 2.2 |
| UCT-MCTS + 3D 世界模型 rollout oracle | 3D-ALP 2604.11302 | D=2、B=4 → 理论 20 节点/周期；2 s 预算实际 ≈5 节点（渲染 ≈2400 ms/帧） | Sec 4.1 Setup；Sec 5 |
| 关键相位按需触发 + 梦想评分 | DreamAvoid 2605.11750 | 触发后 8 候选 ≈40 ms 评分；γ=0.5、1 Hz 轮询、每阶段一次 | Sec 3.1/3.3 |
| 采样 MPC（标准 MPPI，0.1 s/rollout） | PointWorld 2601.03782 | K 个三次样条扰动 × T 步视界，指数权重更新（K/T/β 数值未取到）；0.1 s/批前向 | Sec 3.2 Eq 2 |
| VLM 外环优化 + WM 想象 + 局部网格搜索（粗-细分层） | WAP 2607.27599 | 全局 VLM 反馈轮次 + 局部 N 个网格候选 WM 想象（N 数值未取到） | Algorithm 1；Sec 3.2.2 |
| Reasoner（具身 CoT/2D 规划）→ Generator 双塔引导生成 | Cosmos 3 2606.02800 | 未取到（报告过大，正文仅至 2.1.1） | Intro/Fig 1；官方博客 Fig 3 |
| （官方博客） | Meta WAM | 未能获取 | — |

# 趋势（原文论据）

**1. 测试时计算成为决策主线。**
- VLP（Abstract）："VLP scales with increasing computation budget where more computation time results in improved video plans"；Fig 3 数据：beams/branch 1/1/1→4%、1/1/4→10%、1/4/4→22%、2/4/4→56%（make line 任务）。
- WAV（Corollary 4.3）：一次性潜采样在亚指数预算下无法保证近优轨迹，"Iterative inference procedures that adaptively reweight the latent distribution are therefore necessary"。
- τ0-WM（Sec VI）："selecting a high-quality action becomes an important deployment-time problem"（条件分布多模态）。
- DreamAvoid（Sec 1）："introducing additional computation during test time to endow VLAs with a 'dreaming' capability emerges as a natural choice"。

**2. 测试时计算从 always-on 转向按需/分级（coarse-to-fine）。**
- τ0-WM（Sec VI）："This design preserves real-time performance in most situations while retaining the ability to recover from difficult states."
- DreamAvoid（Sec 1）："Always-on inference can be costly…This motivates selective intervention that concentrates computation on consequential decisions."
- WAP（Abstract/Sec 3.2.2）："coarse-to-fine" 分层，明确反对 MPPI/Best-of-N 的"weakly structured proposal distributions"。

**3. 隐式推断 vs 显式搜索的两极及其混合。**
- 隐式端：WAV（Abstract）："Rather than performing explicit trajectory optimization…Action generation is then formulated as inference in this latent space"；τ0-WM 的 RCS 为隐式自一致过滤（Sec VI-A）。
- 显式端：VLP（Fig 1 caption "forward tree search"；Sec 2.2 "parallel hill climbing"）；3D-ALP（UCT-MCTS 四修复）；PointWorld（标准 MPPI）；WAP（agentic 分层搜索）。
- 混合证据：τ0-WM 把隐式 RCS 与显式 ACVS rollout 按可靠性阈值串接（Algorithm 1）；DreamAvoid 用生成式"梦想" + 判别式价值排序（Sec 3.3）。

**4. 世界模型在决策中的角色分化。**
- rollout oracle：3D-ALP（"Uses a 3D-consistent world model…enabling MCTS rollouts in imagined 3D space"）、PointWorld（MPC 内 MPPI）、WAP（WM 想象 + VLM 评分）、DreamAvoid（DreamDojo 渲染未来帧）。
- 价值/评估载体：WAV（trajectory value module 的 flow matching）、τ0-WM（ACVS 密集进度评分）。
- 统一基座：Cosmos 3（"world-action model for joint action prediction and environmental simulation"）。

**5. 失败意识进入评估器训练。**
- τ0-WM（Sec V-C）：失败段负奖励 + failure-heavy/recovery 轨迹训练模拟器；DreamAvoid（Sec 3.3/3.4）：失败与边界数据 20/40/40 采样、z=−1、"If trained solely on success data, the world model may generate optimistic hallucinations"。

# 未能获取项列表（按篇）

1. WAV：Eq (4)/(5) 更新公式、M/N/K/K1/K2 数值、App B 奖励构造、App C.3 训练配置（arxiv v3 HTML 截断于 4.2 Eq(3) 前；ar5iv 截断于 4.3 第二段；PDF 不被 web_fetch 支持）。
2. τ0-WM：N、γ、K（RCS 时间步）、Hv/Ha 数值（App -A2）、VI-B 末句、Sec VII 消融（ar5iv 截断于 VI-B 末段）。
3. VLP：A/D/B/H 数值（App A.5）、heuristic 防利用阈值、Sec 3.2/3.3（截断于 3.1 表格）。
4. 3D-ALP：仅 References 尾部截断，正文无缺失。
5. DreamAvoid：Sec 3.4 后半、σ（App C.2 Table 5）、K、H、Sec 4–7、App D.3（截断于 3.4 中部）。
6. Cosmos 3：Reasoner 具身 CoT 与 2D 轨迹规划机制正文（Sec 3.1.2/4.1.2/6.1/6.3.2 未取到；正文仅至 2.1.1）。
7. PointWorld：K、T、β 数值（App A.6.1）、Sec 5.4、τ/κ（ar5iv 截断于 Sec 4）。
8. WAP：Sec 3.2.2 后半、N 数值（App C.2.3）、Sec 5 结果（v2 截断于 3.2.2 中；ar5iv 为 v1，截断于 5.1 中）。
9. Meta WAM 博客：未能获取（网络层失败，见第 9 节尝试记录）。

<!-- QA_LOG -->
