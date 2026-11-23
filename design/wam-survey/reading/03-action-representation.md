# WAM 精读线③：动作表征（10 篇深度笔记）

> 任务：对 10 篇具身智能 World Action Model 论文做【动作表征】轴深度精读。
> 生成时间：2026-10-26（Asia/Shanghai）。
> 取文通道：arxiv abs 页 → arxiv.org/html/<id>v1 或最新版 → ar5iv.labs.arxiv.org/html/<id>（平台 web_fetch 对全文单次返回约 5 万字符即截断，fragment/anchor 无效，PDF 不支持，pwsh 直连网络被沙箱阻断；无法按节续抓）。每篇标注「已获取范围」与截断位置，截断处之后的内容一律标「未取到」，禁止脑补。
> 来源级别定义：
> - **L3**：动作空间/维度/chunk/离散连续/生成机制的关键字段均在已获取原文中，有 verbatim 引文+小节号。
> - **L2**：机制有原文引文，但个别字段（如维度数值、chunk 长度）在已获取范围内原文未写。
> - **代码依据**：GitHub 官方仓库 README/config，明确标注 repo 与文件路径。
> - 图按「图编号 + verbatim caption + URL」记录。

---

## 1. GR-1（arXiv:2312.13139）

**元信息**
- 标题：Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation
- 作者：Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, Tao Kong（ByteDance Research）
- 版本：v1 2023-12-20；v2 2023-12-21（latest）。ICLR 2024。
- 取文：abs 页 + arxiv.org/html/2312.13139v2（截断于 §4.1 中段「...ABCD→\rightarrowD and ABC」）+ ar5iv（截断于 §4.2.2 开头）。附录 A（A.1 Network and Training Details）未取到。
- 代码仓：https://github.com/bytedance/GR-1

**来源级别：L3**（§3.2.1/§3.2.3/§4.1 动作定义全取到；仅附录 A.1 的实现细节与视频预测间隔 Δt 数值未取到）

**【动作表征机制】原文引文+逐条指路**

- **动作空间（6D EE pose 增量 + 二值夹爪）**，§4.1 "Experiment Setup"：
  > "For action prediction, similar to [Mees et al. 2022c], we train GR-1 to predict delta XYZ positions and delta Euler angles for arm actions and binary gripper actions."
  → 绝对/增量：**增量**（delta）；EE 位姿（非关节）；夹爪二值。
- **状态空间定义（输入侧）**，§3.2.1 "Robot state input"：
  > "The robot state \mathbf{s} contains the 6D pose of the robot end-effector \mathbf{s}_arm \in SE(3) and a binary status of the gripper \mathbf{s}_gripper \in {0,1}. We use linear layers to encode them"
- **输出头与损失**，§3.2.3 "Outputs"：
  > "The outputs from the [ACT] tokens are passed through linear layers to predict the arm and gripper actions"
  > "Since the arm action is continuous, we use Smooth-L1 loss L_arm for training. Gripper actions are optimized using Binary Cross Entropy (BCE) loss L_gripper."
  → 连续回归头（非扩散、非 token 离散化）；总损失 §3.3 Eq.(3)：L_finetune = L_arm + L_gripper + L_video。
- **预测长度（无 chunk）**：§3.2.1 微调期序列格式
  > "(l, \mathbf{s}_{t-h}, \mathbf{o}_{t-h}, [OBS], [ACT], l, \mathbf{s}_{t-h+1}, ..., l, \mathbf{s}_t, \mathbf{o}_t, [OBS], [ACT])"
  → 每个时间步仅一个 [ACT] token，§3.1 映射为单一动作 a_t：π(l, o_{t-h:t}, s_{t-h:t}) → o_{t+Δt}, a_t。即**单步动作预测（horizon=1，无 action chunk）**。
- **归一化**：正文未写动作数值归一化方式（已获取范围）。**未写**。
- **生成机制**：因果 GPT 式 transformer，[ACT]/[OBS] token 被掩蔽（§3.2.2），linear 头直接回归；无 AR 迭代生成（单 token 一步出）。
- 视频侧预测间隔 Δt（§3.1 Eq.1/2 中的 o_{t+Δt}）：正文未给数值，A.1 未取到 → **未写（A.1 未取到）**。

**【代码依据】**（GitHub raw：https://raw.githubusercontent.com/bytedance/GR-1/main/README.md ，2026-10-26 获取）
- README 仅含安装/评估说明，无动作空间数值；评估脚本 evaluate_calvin.sh + 权重 snapshot_ABCD.pt / snapshot_ABC.pt。动作细节需读 utils/ 下 config（本次未逐文件抓取，标注待核实）。

**【why 追问】（基于原文）**
1. 为何用 delta EE 而非关节位置？原文未给出直接论证，§4.1 显式表述为「similar to Mees et al. 2022c」（沿用 CALVIN/HULC 惯例）；§2.1 批评 keypoint+planner 路线 "making them less flexible compared to predicting continuous actions"——即选择直接连续动作是为避免分层规划的不灵活。
2. 为何不预测 action chunk？原文未讨论 chunk（§3.2.1 序列中每步 1 个 [ACT]），无相关动机论述 → 原文未写。

**【关键数值】（逐字抄录）**
- "6D pose of the robot end-effector \mathbf{s}_arm \in SE(3)"（§3.2.1）
- "binary status of the gripper \mathbf{s}_gripper \in {0,1}"（§3.2.1）
- "delta XYZ positions and delta Euler angles ... binary gripper actions"（§4.1）
- "Smooth-L1 loss L_arm ... Binary Cross Entropy (BCE) loss L_gripper"（§3.2.3）
- CALVIN：34 tasks、ABCD→D 94.9%（vs 88.9%）、avg. len 4.21；ABC→D 85.4%；10% 数据 77.8%；训练集 "over 20k expert trajectories"（§4.1）
- 预训练：Ego4D "800,000 video clips, containing 8M frames"（§3.3）

**【一页总结】** GR-1 的动作表征是最朴素的一档：连续 6 维 ΔXYZ+Δ欧拉角 + 1 维二值夹爪，由 GPT 因果 transformer 的 [ACT] token 经线性层一步回归（Smooth-L1 + BCE），无 chunk、无离散化、无扩散、无归一化说明。它把「未来视频预测」当动作学习的代理任务，动作本身保持 CALVIN 原生格式。

**【待核实】** ① A.1 中 Δt、动作归一化、模型尺寸（未取到）；② 代码 config 中是否对动作做归一化（GitHub bytedance/GR-1，未逐文件核对）；③ 真机实验（§4.2.1-4.2.2 部分被截断）的动作格式是否与 CALVIN 一致（截断于 §4.2.2 开头，标注未取到）。

---

## 2. DiT4DiT（arXiv:2603.10448）

**元信息**
- 标题：DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control
- 作者：Teli Ma, Jia Zheng, Zifan Wang, Chunli Jiang, Andy Cui, Junwei Liang, Shuo Yang（Mondo Robotics, HKUST(GZ), HKUST）
- 取文：ar5iv 页（截断于 §4.3 Algorithm 1 第 21 行「v*_action ← ε − a^0_t」处；§4.4 之后、实验、附录 A 未取到）。代码 release 页：dit4dit.github.io（本次未抓取细节）。
- 无 arxiv html 第二遍抓取（预算控制）。

**来源级别：L2**（§4.2/§4.3 机制引文齐备；**动作空间维度、chunk 长度、归一化在已获取范围内原文未写**）

**【动作表征机制】原文引文+逐条指路**

- **Action DiT 来源与输入**，§4.2 "Action DiT"：
  > "we employ a dedicated action diffusion transformer adapted from the GR00T-N1."
  > "The input sequence to this DiT is a concatenation of proprioceptive state embeddings, encoded noisy action trajectories, and a set of learnable “future tokens” that serve as compressed queries for the motion planning task."
  > "The network terminates with a linear projection that predicts the velocity vector field of the action sequence, allowing the final trajectory to be synthesized via iterative numerical integration during inference."
  → 动作以「action trajectories（序列/chunk 形式）」输入；预测**速度场**。
- **生成机制（流匹配）**，§4.1 "Flow matching"：
  > "we employ a first-order Euler discretization to perform the numerical integration."（Eq.4-5：d x/d τ = v_θ(x,τ)，x_1∼N(0,I)，x_{τ−Δτ} = x_τ − Δτ·v_θ(x_τ,τ)）
- **动作时间步 τa ~ Beta**，§4.3 "Tri-timestep scheme"：
  > "the action DiT relies on a third, independent timestep, τ_a. ... τ_a is drawn from a Beta distribution during training (τ_a = 1−σ, where σ ∼ Beta(α,β)). This biased continuous-time sampling strategy allocates more training capacity to the most critical stages of the flow trajectory."
  → α、β 数值在已获取范围内**未写**。
- **视频侧**：τ_v ∼ U[0,1]；特征提取 τ_f 固定（"we forward the context frames through the denoising backbone at a fixed timestep, denoted as τ_f"；Algorithm 1 中训练时 τ_f ∼ U{0/T,…,T/T} 采样，推理时固定，正文称 "stable visual conditions are extracted at a fixed deterministic timestep"——图 3 caption）。
- **联合损失**，§4.3 Eq.(10)：Action FM loss：‖v^action_φ(a_t^{τ_a}, τ_a | h_t^{τ_f}, s) − (ε − a_t^0)‖²；Video FM loss 加权 λ。
- **动作空间维度**：**未写（已获取范围 §1–§4.3）**。仅知「GR00T-N1 适配」（GR00T-N1 的具体动作格式未在本文复述）。
- **chunk 长度**：**未写**。仅「action trajectories」「action sequence」「future tokens」等表述。
- **归一化**：**未写**（已获取范围）。

**【代码依据】** 原文 §1：「We release code and models at https://dit4dit.github.io/」；本次未抓取该页面/GitHub → 待核实。

**【why 追问】（基于原文）**
1. 为何动作时间步用 Beta 偏置而非均匀？§4.3：τ_v 均匀是「forcing it to learn the full denoising trajectory」，而 τ_a 偏置为「allocates more training capacity to the most critical stages of the flow trajectory」——动作只需关键控制阶段的容量，视频需全轨迹。
2. 为何动作 DiT 要「adapted from GR00T-N1」而非新造？原文未给出动机陈述（仅 §4.2 一句交代）；可推为复用成熟动作流匹配专家，但原文未明说 → 原文未写理由。

**【关键数值】（逐字抄录）**
- "reaching average success rates of 98.6% on LIBERO and 50.8% on RoboCasa GR1"（Abstract）
- "improves sample efficiency by over 10× and speeds up convergence by up to 7×"（Abstract）
- 视频骨干 "Cosmos-Predict2.5-2B"；文本 "Cosmos-Reason1"（§4.2）
- τ_v ∼ U[0,1]；τ_a = 1−σ, σ∼Beta(α,β)（α,β 未写）（§4.3）
- 视频条件帧 T_cond、未来帧 T_v（"T_v represents the horizon of future frames"，§4.2，数值未写）

**【一页总结】** DiT4DiT 的动作表征核心是「动作流匹配 DiT」：从视频 DiT 在固定流时间步 τ_f 提取的中间特征经 cross-attention 注入，动作侧以独立时间步 τ_a=1−Beta(α,β) 训练速度场、Euler 积分生成动作轨迹。动作空间维度/chunk/归一化三个关键字段在已获取正文中均未写明，是本组中「动作空间定义最不透明」的论文之一。

**【待核实】** ① §4.4 及附录中动作维度/chunk/归一化（未取到）；② Beta(α,β) 参数值（未取到）；③ dit4dit.github.io / GitHub 代码中动作空间配置（代码依据待补）。

---

## 3. τ0-WM（arXiv:2606.01027）

**元信息**
- 标题：τ0-WM: A Unified Video-Action World Model for Robotic Manipulation
- 作者：Pengfei Zhou, Shengcong Chen, Di Chen 等（Shanghai Innovation Institute, AGIBOT Finch）
- 取文：ar5iv 页（截断于 §VI-B LAR 处「the rollout latent」；§VII 实验、附录未取到）。

**来源级别：L2**（§IV VAM 接口/损失完整；**H_v、H_a 数值与动作空间定义在已获取范围内原文未写**）

**【动作表征机制】原文引文+逐条指路**

- **VAM 接口：未来潜变量 + 连续动作 chunk**，§IV-A：
  > "VAM predicts a future latent trajectory together with an executable action chunk: F_θ(\mathbf{o}_t, \mathbf{p}, \mathbf{s}_t) → (\hat{\mathbf{z}}_{t+1:t+H_v}, \hat{\mathbf{a}}_{t:t+H_a−1}), where \hat{\mathbf{z}} denotes future video latents over horizon H_v and \hat{\mathbf{a}} denotes a continuous action chunk over horizon H_a."
  → **连续动作 chunk**；H_a、H_v 数值在已获取范围**未写**。
- **动作分支结构**，§IV-B：
  > "The action branch is a 0.5B-parameter DiT-style action decoder coupled to the video transformer. Together, they form a 5.5B-parameter Video Action Model."
  > "At matched transformer stages, the action tokens first model temporal dependencies within the action horizon and then cross-attend to intermediate video features."
  → 视频骨干 Wan2.2-TI2V-5B；动作 token 内部先做时间建模再 cross-attend 视频特征。
- **联合流匹配目标**，§IV-C Eq.(2)：
  > "ℒ_VAM = E[ λ_z‖f^z_θ(z̃, u_z, c_t, p) − v_z‖² + λ_a‖f^a_θ(ã, u_a, s_t, h) − v_a‖² ]"；"In all experiments, we simply set λ_z = λ_a = 1."
- **部署：receding-horizon 执行**，§IV-D：
  > "In action-only deployment, only the predicted action chunk is generated and executed in a receding-horizon manner"
- **动作空间定义（EE/关节/维度）**：**未写（已获取范围）**。仅知输入 robot state s_t、数据来自 AGIBOT-G01/ARX/dual-arm Franka（§III）。
- **ACVS 动作条件注入**（动作→视频方向），§V-B：
  > "temporally aligned actions are grouped into an action block b_ℓ and projected through lightweight MLPs: c^a_ℓ = ψ_D(b_ℓ), m^a_ℓ = ψ_6D(b_ℓ), which are injected into the diffusion-time embedding and AdaLN modulation embedding, respectively."
  > "we follow the action-conditioned design of Cosmos"
- **测试时计算**：采样 N 个动作候选 → RCS 重去噪一致性打分 → 低于阈值 γ 则 ACVS rollout + J = max_{0≤q<H_a} r̂_{t+q}（Eq.10）→ LAR 二次 VAM 查询修正。

**【代码依据】** 原文未提公开代码（Abstract/§1 无 repo 链接，已获取范围）→ 无。

**【why 追问】（基于原文）**
1. 为何动作预测要附在未来视频潜变量预测上？§IV-A："Future visual prediction serves not only as an auxiliary objective but also as a mechanism for learning transferable interaction dynamics from heterogeneous data sources, including videos without action annotations, while action prediction grounds the learned representation in executable robot control."——视频预测为跨数据源迁移，动作预测为可执行接地。
2. 为何测试时要对动作做重去噪一致性筛选？§VI："Pre-training on large-scale heterogeneous interaction data makes the conditional action distribution inherently multimodal... selecting a high-quality action becomes an important deployment-time problem."

**【关键数值】（逐字抄录）**
- "approximately 27,300 hours"（Abstract）；"17.8K hours ... AGIBOT-G01, ARX ... dual-arm Franka; 6.5K hours ... UMI-style ... Gen-DAS Grippers; 3.0K hours ... egocentric"（§III）
- "0.5B-parameter DiT-style action decoder ... 5.5B-parameter Video Action Model"（§IV-B）
- λ_z = λ_a = 1（§IV-C）；λ_z = λ_r = 1（§V-D）
- H_v、H_a：**未写**（已获取范围）

**【一页总结】** τ0-WM 的动作表征＝「共享视频扩散骨干上的连续动作 chunk 流匹配」：Action DiT 分支在动作 horizon 内先做时间建模、再 cross-attend 视频中间特征，与未来视频潜变量联合流匹配（λ_z=λ_a=1）；执行 receding-horizon，测试时用 RCS+ACVS 对候选动作做评估与修正。动作空间的物理定义（EE/关节维度）与 H_a/H_v 数值未在已获取正文中出现。

**【待核实】** ① H_v/H_a 具体数值（未取到，可能在 §VII/附录）；② 动作空间定义（EE 位姿或关节；维数；绝对/增量）（未取到）；③ 动作/状态归一化（未写）。

---

## 4. LAPA（arXiv:2410.11758）

**元信息**
- 标题：Latent Action Pretraining From Videos
- 作者：Seonghyeon Ye, Joel Jang, Byeongguk Jeon, Sejune Joo, Jianwei Yang, Baolin Peng, Ajay Mandlekar, Reuben Tan, Yu-Wei Chao, Yuchen Lin, Lars Liden, Kimin Lee, Jianfeng Gao, Luke Zettlemoyer, Dieter Fox, Minjoon Seo（UW, KAIST, Microsoft Research, NVIDIA, Allen Institute for AI）
- 取文：ar5iv 页（截断于 §5.2 中段「we also analyze the latent actions learned from the Open-X embodiment...」；附录 A「Latent Action Quantization Details」、B 实验设置未取到）。
- 代码/模型：latentactionpretraining.github.io（已开源）。

**来源级别：L2**（§3 全章 + §4.6/§5.1 拿到 8^4 关键数值；附录 A 中窗口 H、序列长 s、词表 |C| 的逐条细节未取到，但 §4.6/§5.1/§5.2 给出生成空间 8^4 的直接原文）

**【动作表征机制】原文引文+逐条指路**

- **潜动作：VQ 离散码（VQ-VAE 学自相邻帧）**，§3.1 "Latent Action Quantization"：
  > "the encoder takes the current frame x_t and the future frame x_{t+H} of a video with a fixed window size H and outputs the latent action z_t"
  > "The VQ-VAE objective enables the latent action z_t to be discrete tokens (codebooks), making it easy for VLMs to predict z_t."
  > "The latent action is represented using s sequences from |C| codebook vocabulary space. The sequence length s is designated by the kernel size, stride and padding value of a CNN network which is used right before the vector quantization process."
  → 窗口 H 与 s、|C| 数值在附录 A（未取到）；正文未给具体数值。
- **潜动作预测头**，§3.2：
  > "we attach a separate latent action head (a single MLP layer) of vocab size |C|."
- **潜动作 → 真实动作的映射（微调阶段）**，§3.3 "Action Finetuning"：
  > "To map latent actions to actual robot actions, we finetune LAPA on a small set of labeled trajectories that contain ground truth actions (delta end-effector). For action prediction, we discretize the continuous action space for each dimension of the robot so that the number of data points allocated for each bin is equal following [Kim et al. 2024; Brohan et al. 2023]. We discard the latent action head (a single MLP layer) and replace it with a new action head to generate ground truth actions."
  → 真实动作空间＝**delta end-effector**，逐维离散化（等频分箱，沿用 OpenVLA/RT-2 方式）；丢弃潜动作头换新头。
- **生成空间大小（主实验）**，§4.6：
  > "the action space for LAPA is much smaller than that for OpenVLA (8^4 vs. 256^7)"（原文 LaTeX 为 $8^4$ vs $256^7$；ar5iv 渲染错乱为「848^{4} vs. 2567256^{7}」，按 LaTeX 语义还原为 8^4 vs 256^7）
- **主实验潜动作配置**，§5.1：
  > "Except for Language Table, we maintain the generation space of LAPA at 8^4 throughout all of our main experiments."
  → 即 4 个 token × 词表 8（s=4, |C|=8）；§5.2 示例潜动作 "[1,1,3,2]"、" [3,2,0,1]" 亦为 4 元组。
- **离散/连续**：预训练阶段**离散**（VQ 码）；微调阶段输出**离散化**的真实动作 token。
- **生成机制**：自回归 VLM（LWM-Chat-1M 7B）预测潜动作码/动作 token（非扩散）。

**【代码依据】** 原文 Abstract："We will open-source the model checkpoints and code at latentactionpretraining.github.io"（本次未抓取仓库 → 待核实）。

**【why 追问】（基于原文）**
1. 为何用离散 VQ 码当潜动作？§3.1："The VQ-VAE objective enables the latent action z_t to be discrete tokens (codebooks), making it easy for VLMs to predict z_t"——为让 VLM 直接做离散 token 预测；§1 类比 BPE："Analogous to Byte Pair Encoding... this can be seen as learning to tokenize atomic actions without requiring predefined action priors (e.g., end-effector positions, joint positions)."
2. 为何生成空间选 8^4 而非更大？§5.1："we observe that the optimal scale of the latent action space depends on the complexity of the action dimension contained in the pretraining dataset. For example, increasing the latent action sequence length is less effective compared to increasing the vocabulary for Language Table."——尺度按数据集动作复杂度取舍；§4.6 还论证小动作空间是预训练效率来源（"the coarse-grained actions of LAPA"）。

**【关键数值】（逐字抄录）**
- "8^4 vs. 256^7"（§4.6）
- "we maintain the generation space of LAPA at 8^4 throughout all of our main experiments"（§5.1）
- 预训练效率："8 H100 GPUs for 34 hours with a batch size of 128 (total of 272 H100-hours) ... OpenVLA required a total of 21,500 A100-hours"（§4.6）
- 骨干："the 7B Large World Model (LWM-Chat-1M)"（§4.2）
- 预训练数据：181k trajectories（Language Table）、440k real-world（cross-env）、Bridgev2、Open-X、Something-Something v2（220K videos，§4.5）
- 微调数据：Language Table 1k/7k；SIMPLER 100 trajectories；真机每任务 150 trajectories（§4.1）

**【一页总结】** LAPA 的动作表征是两段式：无标签视频上 VQ-VAE 学「帧间潜动作」离散码（主实验 8^4＝4 token×词表 8），VLM 自回归预测这些码；下游微调换新头直接输出逐维等频离散化的 delta end-effector 动作。潜动作只做「中间表征」，真实执行仍落在离散化的 ΔEE 空间（256^7 级，与 OpenVLA 同规格）。

**【待核实】** ① 附录 A 中窗口 H、s、|C|、量化模型尺寸（未取到）；② 各维度 bin 数（§3.3 说沿用 Kim 2024/Brohan 2023 的离散化，即 256 bins，但本文未复述数值——引原文时写「未复述，参见 OpenVLA」）；③ 代码仓库中 VQ codebook 配置（待抓 latentactionpretraining.github.io）。

---

## 5. V-JEPA 2（arXiv:2506.09985）

**元信息**
- 标题：V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning
- 作者：Mahmoud Assran 等（FAIR at Meta、Mila 等，Yann LeCun 在列）
- 取文：ar5iv 页（截断于 §3.1 "Loss function" 的 rollout loss 公式「ℒ_rol」处；**§3.2 训练细节、§4 Planning: Zero-shot Robot Control（image-goal 规划与动作采样）、§5-9、附录未取到**）。
- 代码：https://github.com/facebookresearch/vjepa2

**来源级别：L2**（§3.1 动作定义/输入/损失前半完整；image-goal 规划采样动作的 §4 未取到——这是任务指定重点，需补代码或标注）

**【动作表征机制】原文引文+逐条指路**

- **AC 模型定位**，§1：
  > "we train an action-conditioned world model, V-JEPA 2-AC, on a small set of interaction data... Our action-conditioned world model is a 300M-parameter transformer network employing a block-causal attention mechanism, which autoregressively predicts the representation of the next video frame conditioned on an action and previous states."
- **动作定义：7 维 ΔEE**，§3.1 "Model inputs"：
  > "The robot's end-effector state in each observation is denoted by the sequence (s_k)_{k∈[16]}, where s_k is a real-valued 7D vector defined relative to the base of the robot. The first three dimensions of s_k encode the cartesian position of the end-effector, the next three dimensions encode its orientation in the form of extrinsic Euler angles, and the last dimension encodes the gripper state."
  > "We construct a sequence of actions (a_k)_{k∈[15]} by computing the change in end-effector state between adjacent frames. Specifically, each action a_k is a real-valued 7-dimensional vector representing the change in end-effector state between frames k and k+1."
  → **连续、增量**：Δ位置 3 + Δ外蕴欧拉角 3 + Δ夹爪 1；16 帧 clip 内共 15 个动作（T=15）。
- **采样率/序列**，§3.1：
  > "sampled with resolution 256×256 and a frame-rate of 4 frames-per-second (fps), yielding 16 frame clips"
  > "The sequence of feature maps, end-effector states, and actions are temporally interleaved as (a_k, s_k, z_k)_{k∈[15]}"
- **表征空间**：V-JEPA 2 编码器冻结，特征图 "have the shape 16×16×1408"（ViT-g）；预测在潜表征空间（L1，teacher-forcing，Eq.2）＋两步 rollout loss（公式被截断）。
- **归一化**：未写（已获取范围）。
- **生成机制**：block-causal 自回归表征预测（非扩散、非离散）。
- **image-goal 规划时如何采样动作**：§4 "Planning: Zero-shot Robot Control" **未取到**（截断于 §3.1）。已知线索（§1）："enabling picking and placing of objects using planning with image goals"、"planning within a model predictive control loop"（Fig.1 caption）。具体动作采样机制 → 未取到。

**【代码依据】** 论文页眉标注 "Code: https://github.com/facebookresearch/vjepa2"（本次未抓取 → 待核实）。

**【why 追问】（基于原文）**
1. 为何动作定义为帧间 ΔEE 而非绝对位姿？§3.1 直接以帧间差分构造动作（"computing the change in end-effector state between adjacent frames"），未给论证；§3 开篇说明控制动作对应 end-effector commands（"where control actions correspond to end-effector commands"）——为匹配 DROID 数据集原生信号，原文未展开。
2. 为何用潜表征预测而非像素生成？§1："in contrast to approaches based on video generation, the JEPA approach focuses on learning representations for predictable aspects of a scene... while ignoring unpredictable details that generative objectives emphasize"——潜空间预测避免像素级不可预测细节，且 §1 指出 "the computational cost of planning by generating video" 是前人瓶颈。

**【关键数值】（逐字抄录）**
- "less than 62 hours of unlabeled robot videos from the Droid dataset"（Abstract）
- "300M-parameter transformer ... block-causal attention"（§1）
- "256×256 ... 4 frames-per-second ... 16 frame clips"（§3.1）
- "7D vector ... extrinsic Euler angles"（§3.1）；"T=15"（Eq.2 下方）
- "16×16×1408"（§3.1）
- DROID 视频 "typically 3–4 seconds long, of a 7-DoF Franka Emika Panda arm equipped with a two-finger gripper"（§3.1）

**【一页总结】** V-JEPA 2-AC 的动作表征＝连续 7 维 ΔEE（位置/外蕴欧拉角/夹爪），在 4fps×16 帧的 clip 上取 15 个帧间差分动作，与端效器状态、冻结 ViT-g 特征图按 (a_k,s_k,z_k) 交错输入 300M block-causal 预测器，L1 teacher-forcing + 两 step rollout 损失，自回归预测下一帧表征。image-goal 规划的动作采样机制（§4）未取到，是本文待补的核心缺口。

**【待核实】** ① §4 的 MPC/image-goal 规划中动作如何采样（未取到）；② 动作归一化方式（未写）；③ vjepa2 仓库中 AC 训练/规划配置（代码依据待抓）；④ rollout loss 完整公式（截断）。

---
## 6. Action Images（arXiv:2604.06168）

**元信息**
- 标题：Action Images: End-to-End Policy Learning via Multiview Video Generation
- 作者：Haoyu Zhen, Zixian Gao, Qiao Sun, Yilin Zhao, Yuncong Yang, Yilun Du, Pengsheng Guo, Tsun-Hsuan Wang, Yi-Ling Qiao, Chuang Gan（UMass Amherst, NVIDIA, Harvard, Genesis AI）
- 取文：ar5iv 页（截断于 §4.1 Table 2 表头；§4 实验、附录未取到）。方法节 §3.1–3.3 完整。
- 项目页：ActionImages.github.io

**来源级别：L3**（§3.1 编码、§3.2 解码、§3.3 训练全部逐字取到）

**【动作表征机制】原文引文+逐条指路**

- **动作定义：7-DoF 绝对 EE 位姿 + 夹爪开度**，§3.1：
  > "At each time step t, the robot action is \mathbf{a}_t = [\mathbf{p}_t, \boldsymbol{\theta}_t, g_t] ∈ ℝ^7, where \mathbf{p}_t ∈ ℝ^3 is the end-effector position, \boldsymbol{\theta}_t ∈ ℝ^3 denotes its orientation, and g_t ∈ ℝ is the gripper openness."
  → **绝对位姿**（非增量）。
- **编码：7-DoF → 三个语义 3D 点 → 多视角 RGB 热力图**，§3.1：
  > "We convert this 7-DoF action into three semantic 3D points: a position point, a normal point, and an up point. The position point is the end-effector position, \mathbf{q}^{pos}_t = \mathbf{p}_t."
  > "\mathbf{q}^{up}_t = \mathbf{p}_t + ℓ \mathbf{R}(\boldsymbol{\theta}_t) \mathbf{e}_x, \mathbf{q}^{normal}_t = \mathbf{p}_t + ℓ \mathbf{R}(\boldsymbol{\theta}_t)(−\mathbf{e}_z)"（Eq.1）
  > "The red channel encodes the position point, the green channel encodes the normal point, and the blue channel encodes the up point together with the gripper openness"
  > "we inject the binary gripper openness signal into low-response regions: A^{(v)}_t(i,j,3) = Ã^{(v)}_t(i,j,3) if Ã>0.25, else 0.25·g_t"（Eq.5）
  > "Stacking these frames over time yields an action video for each view, A^{(v)} = {A^{(v)}_1,…,A^{(v)}_T} ∈ ℝ^{T×H×W×3}."（Eq.6）
  → 动作＝T×H×W×3 像素视频，与观测视频同构。
- **解码回 7-DoF（如何恢复动作）**，§3.2：
  > 夹爪："ĝ_t = (1/0.25)·(1/|Ω_t|) Σ_{(i,j,v)∈Ω_t} A^{(v)}_t(i,j,3), Ω_t = {(i,j,v) | A^{(v)}_t(i,j,3) < 0.25}"（Eq.7，蓝通道低响应区平均）
  > 位姿：主视角热图质心（Eq.8）→ ray casting + 侧视热图匹配选 3D 点（Eq.9）→
  > "\hat{\mathbf{e}}^x_t = norm(\hat{\mathbf{q}}^{up}_t − \hat{\mathbf{q}}^{pos}_t) and \hat{\mathbf{e}}^z_t = norm(\hat{\mathbf{q}}^{pos}_t − \hat{\mathbf{q}}^{normal}_t), then obtain \hat{\mathbf{e}}^y_t = \hat{\mathbf{e}}^z_t × \hat{\mathbf{e}}^x_t, from which the end-effector orientation \hat{\boldsymbol{\theta}}_t is determined. The final decoded action is â_t = [\hat{\mathbf{p}}_t, \hat{\boldsymbol{\theta}}_t, ĝ_t]."
- **统一模型训练**，§3.3：
  > "We first encode both streams into the backbone latent space by the 3D-VAE, and then concatenate them temporally to form a single input sequence X_v = [V^{(v)}_{1:T}, A^{(v)}_{1:T}] ∈ ℝ^{(2T)×h×w×c}"（Eq.10）
  > 四种 mask 模式：joint generation / action-conditioned video / video-to-action labeling / video-only。
  > 目标："We fine-tune the pretrained backbone using a flow matching objective on the masked latent tokens. The target velocity is defined as \mathbf{v} = \boldsymbol{\epsilon} − \mathbf{X}."（Eq.11）
  → 生成机制＝**流匹配**；骨干 Wan 2.2。
- **多视角理由**，§1："A single view often provides only a ambiguous projection of motion, making it difficult for the model to infer the full action consistently from pixels alone. Using multiple views makes the pixel-grounded action image more reconstructable"
- 数据集：DROID 80k/2 views；RLBench 180k/4 views；BridgeV2 30k/1-4 views（§3.3 Table 1）。

**【代码依据】** 项目页 ActionImages.github.io（本次未抓取）→ 待核实。

**【why 追问】（基于原文）**
1. 为何把动作做成像素图像？§1："the model's predictive knowledge of the world is only indirectly connected to acting... the burden of generalization is shifted to a specialized control module, which is often exactly where transfer breaks down."；§3.1 Benefits："it makes action prediction spatially grounded: the model learns control through visible robot-arm motion rather than through abstract action tokens."
2. 为何能容忍像素编码的信息损失？§3.2 Discussion："the remaining decoding error is dominated not by representation mismatch, but by discretization... finer ray sampling and higher image resolution directly improve the fidelity of the decoded action."

**【关键数值】（逐字抄录）**
- "7-DoF"、"ℝ^7"、"ℝ^{T×H×W×3}"（§3.1）
- 蓝通道阈值 0.25（Eq.5/7）
- "H=10"? 否——chunk/帧长 T 数值未在 §3 出现（数据集帧数未写）→ 未写。
- 骨干 "Wan 2.2"；"v = ε − X"（§3.3）

**【一页总结】** Action Images 把 7-DoF 绝对 EE 动作渲染成多视角 3 通道 RGB 高斯热图视频（红=位置点、绿=法向点、蓝=上向点+夹爪开度），与观测视频在 latent 空间时域拼接，用 Wan 2.2 流匹配统一建模；解码用「质心+ray casting+侧视匹配」恢复 6D 位姿、蓝通道低响应均值恢复夹爪。动作维度的「存放形式」是像素，但信息内容仍是 7-DoF。

**【待核实】** ① 动作视频帧率/帧长 T 与 chunk 时长（§4 实验未取到）；② 高斯 σ、ℓ 取值（§3.1 未给数值）；③ 实验表格（§4.1 截断）。

---

## 7. PointWorld（arXiv:2601.03782）

**元信息**
- 标题：PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation
- 作者：Wenlong Huang, Yu-Wei Chao, Arsalan Mousavian, Ming-Yu Liu, Dieter Fox, Kaichun Mo, Li Fei-Fei（Stanford, NVIDIA）
- 取文：ar5iv 页（截断于 §4「Model Evaluation Protocol」中段；§5 实验/消融、附录未取到）。方法节 §3.1–3.2 完整。
- 项目页：point-world.github.io（代码/数据集/权重将开源）

**来源级别：L3**（§3.1 状态/动作表示、§3.2 转回机器人动作全取到）

**【动作表征机制】原文引文+逐条指路**

- **总体：状态与动作统一为 3D 点流**，§3.1：
  > "We model environment dynamics as a neural network F_θ : S × A → S... we adopt a multi-step (chunked) formulation... F^H_θ : (s_t, a_{t:t+H−1}) → s_{t+1:t+H}, which improves temporal consistency and amortizes computation. We use H=10 steps and 0.1s per step."
  → **chunk 长度 H=10 步、0.1s/步**。
- **动作：robot point flows（URDF 前向运动学传播的表面点）**，§3.1 "Action Representation"：
  > "given a sequence of joint configurations {q_{t+k}}^H_{k=0}, we sample robot surface points once at time t, attach each to its corresponding link, and propagate them with forward kinematics to obtain an ordered set of N_R robot points {(r_{t+k,j}, f^R_{t+k,j})}^{N_R}_{j=1} at each time step t+k... We treat this collection as the action at time t+k and denote it by a_{t+k}."
  > "for efficiency, we sample robot point flows from only the grippers (a few hundred points per gripper depending on its geometry)"
  → 动作维度＝N_R×3（每 gripper 几百点），**非固定低维向量**；embodiment-agnostic。
- **点采样方式（场景侧）**，§3.1 "State Representation"：
  > "from one or a few calibrated RGB-D views, we mask robot pixels via forward kinematics (using the URDF and joint configuration) and back-project the remaining pixels to obtain p_{t,i}."
- **位移表示**：预测每点每步 3D 位移（Huber on 3D residual，Eq.1），movement weight w_{k,i}（σ(κ(δ_{k,i}−τ)) 归一化）+ aleatoric 不确定性。
- **如何转回机器人动作**，§3.2 "PointWorld for Robotic Manipulation"：
  > "we integrate PointWorld in an MPC framework with a sampling-based planner MPPI that plans a sequence of T end-effector pose targets in SE(3) given a cost function defined in the model's state space."
  > "We then sample K action perturbations ℓ_{1:K} using a time-correlated (cubic-spline) noise distribution, which are added to a nominal end-effector trajectory. For each sampled trajectory E^{(ℓ)}_{1:T}, the corresponding robot point-flow actions a^{(ℓ)}_{1:T} are constructed, scene flows are rolled out by PointWorld... exponentiated weights ω_ℓ ∝ exp(−J^{(ℓ)}/β)"
  → 动作方向反转：在 SE(3) 端效器轨迹空间采样 → 转成点流喂给模型 → 按点级目标代价择优。
- 特征：scene 点用 frozen DINOv3；robot 点用 temporal embedding（§3.1 Dynamics Prediction）。

**【代码依据】** §1 贡献声明 "Code, dataset, and pre-trained checkpoints will be open-sourced"（point-world.github.io）→ 待核实。

**【why 追问】（基于原文）**
1. 为何动作要表示为 3D 点流？§3.1："This is an intentional design for ensuring “imagined actions” are fully, rather than partially, observable while being represented in an embodiment-agnostic way—crucial in cases where contact occurs in occluded regions"；§1："By representing actions as 3D point flows instead of embodiment-specific action spaces (e.g., joint positions), this formulation directly conditions on physical geometries of robots, while seamlessly integrating learning across embodiments."
2. 为何选 H=10、0.1s/步？§3.1：chunk 化 "improves temporal consistency and amortizes computation"；§1/Abstract："real-time (0.1s) inference speed"、"predicts scene dynamics jointly over short action chunks in a single forward pass at a real-time latency (0.1s)"——0.1s 匹配实时推理延迟。具体为什么是 10 步原文未展开。

**【关键数值】（逐字抄录）**
- "We use H=10 steps and 0.1s per step"（§3.1）
- "a few hundred points per gripper"（§3.1）
- "∼2M trajectories or ∼500 hours"（Abstract）；"over 60% of DROID (nearly 200 hours...)"（§4）
- 外参中位误差 "1.8 cm and 1.9 degrees"（§4）；点云 grid downsampling "1.5cm"（Fig.6 caption）
- "a single forward pass at a real-time latency (0.1s)"（§1）

**【一页总结】** PointWorld 把动作定义为「URDF 前向运动学传播的 gripper 表面点流」（每 gripper 几百点，0.1s×10 步 chunk），与场景点云拼接后由点云骨干单前向预测全场景逐点位移（movement-weighted Huber + 不确定性）。执行时经 MPPI 在 SE(3) 端效器轨迹空间采样、用同一模型 roll out 打分，把「3D 点流动作」转回机器人可执行命令。动作空间完全几何化，无离散、无扩散。

**【待核实】** ① §5.2 中动作表示消融（gripper-only vs 全机点、点数量）（未取到）；② MPPI 的 T、K、β 数值（§3.2 未给数值，附录未取到）；③ N_S、N_R 具体点数（未写）。

---

## 8. Motus（arXiv:2512.13030）

**元信息**
- 标题：Motus: A Unified Latent Action World Model
- 作者：Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, Jun Zhu（Peking University, Tsinghua, Horizon Robotics）
- 取文：ar5iv 页（截断于 §5.2 Table 2 表头；§5 实验、附录未取到）。方法节 §4.1–4.3 完整。
- 项目页：motus-robotics.github.io/motus

**来源级别：L3**（§4.1/4.2/4.3 的 14 维潜动作管线与整流流机制全取到）

**【动作表征机制】原文引文+逐条指路**

- **总体：光流潜动作（连续）+ 目标机器人真动作**，§4.2：
  > "we introduce latent actions that encode the motion learned directly from pixels... these latent actions allow the model to absorb motion knowledge from various sources such as internet videos, egocentric human demonstrations, and multi-robot trajectories"
- **14 维潜动作管线（DPFlow → DC-AE → 投影）**，§4.2 "Optical Flow Based Representation"：
  > "optical flows are computed by DPFlow and then converted into RGB images."
  > "we employ a deep convolutional variational autoencoder (DC-AE) that reconstructs the flow while encoding it into four 512-dimensional tokens. A lightweight encoder then projects these concatenated 4×512 features into a 14-dimensional vector, roughly matching the scale of typical robot action spaces."
  → **潜动作维数=14**（4×512 → 14）。
- **分布对齐（潜动作→真动作）**，§4.2 "Training and Distribution Alignment"：
  > "task-agnostic data uses Curobo to collect image-action pairs by randomly sampling the target robot's action space"
  > "we mix 90% unlabeled data for self-supervised reconstruction with 10% labeled trajectories for weak action supervision"
  > "ℒ = ℒ_recon + λ_a‖a_real − a_pred‖² + βℒ_KL"（Eq.2）
- **生成机制：rectified flow + UniDiffuser 式独立时间步**，§4.1：
  > "Motus jointly predicts chunks of videos and actions with rectified flow-based objectives"（l_action + l_obs，τ_a, τ_o ∼ U(0, T_τ)）
  > "By allocating different timesteps and noise scales to videos and actions... a UniDiffuser-like scheduler to capture heterogeneous data distributions and adaptively switch between various embodied foundation models during inference (e.g., VLA, World Model, IDM, VGM, Joint Prediction)."
- **Action-Dense Video-Sparse**，§4.1：
  > "we downsample the video frames so that the number of video tokens and action tokens remains balanced—for example, by setting the video frame rate to one-sixth of the action frame rate."
- **光流为何是动作表示（原文理由）**，§2.2：
  > "inspired by optical flow as a universal motion expression, we use it to align cross-embodiment behaviors and learn latent actions to facilitate large-scale pretraining."
  > "Initially, RGB images are used for supervision, but this introduces task-irrelevant appearance information."（评述前人工作，引出光流）
- 专家构成：Wan 2.2 5B（生成）、Qwen3-VL-2B（理解）、action expert（同深度 Transformer + AdaLN + Tri-model Joint Attention）（§4.1）。
- 真实动作空间：目标机器人动作（§3 脚注 1："In joint position control, proprioception and action share the same representation space"）；chunk 长度 k（§3 "predicts the next k actions (action chunking)"）数值未写。
- 三阶段训练 + 六层数据金字塔（§4.3 Table 1）。

**【代码依据】** 项目页 motus-robotics.github.io/motus（本次未抓取）→ 待核实。

**【why 追问】（基于原文）**
1. 为何潜动作取 14 维？§4.2："a 14-dimensional vector, roughly matching the scale of typical robot action spaces. This dimensional correspondence ensures that the latent representation can align naturally with real robotic controls and act as a bridge between perception and action."——维度对齐真实动作空间以便弱监督对齐。
2. 为何用光流而非 RGB/DINOv2/keypoint 重建潜动作？§2.2：RGB 重建引入 task-irrelevant appearance；光流是 "a universal motion expression"、"pixel-level 'delta action'"（§1），跨 embodiment 对齐行为。

**【关键数值】（逐字抄录）**
- "four 512-dimensional tokens"、"4×512"、"14-dimensional vector"（§4.2）
- "one-sixth of the action frame rate"（§4.1）
- "90% unlabeled ... 10% labeled"（§4.2）
- τ_a, τ_o ∼ U(0, T_τ)（§4.1）
- 仿真 "+15% improvement over X-VLA and a +45% improvement over π0.5"；真机 "+11~48%"（Abstract）
- 真实动作维/chunk k：**未写**（已获取范围）

**【一页总结】** Motus 的动作表征分两层：跨 embodiment 通用层是「光流潜动作」——DPFlow 光流→RGB→DC-AE 编码成 4×512 tokens→投影为 14 维连续向量，以 90% 无标签+10% 弱标签（含 Curobo 任务无关采样）做分布对齐（重建+对齐+KL）；执行层是目标机器人的原生动作空间（Stage 3 SFT 直接输出真动作）。生成机制为整流流，视频/动作独立时间步（UniDiffuser 式调度），视频帧率取动作帧率的 1/6 保持 token 平衡。

**【待核实】** ① chunk 长度 k、T_τ、λ_a/β 数值（未写/未取到）；② 14 维潜动作内部构成（是否显式分组，未写）；③ Stage 3 的真实动作空间细节（joint 维数，未取到）。

---

## 9. UWM（arXiv:2504.02792）

**元信息**
- 标题：Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets
- 作者：Chuning Zhu, Raymond Yu, Siyuan Feng, Benjamin Burchfiel, Paarth Shah, Abhishek Gupta（University of Washington, Toyota Research Institute）
- 取文：ar5iv 页（截断于 §III-B 第 4 条 Inverse Dynamics 的公式「a_{t−1} = 1/√α_t (」处；§III-C 之后、§IV 实验（含动作 tokenizer 细节）、附录未取到）。
- 代码：weirdlabuw.github.io/uwm

**来源级别：L2**（四模式时间步置位机制逐字全取到；**动作 token 形式/维度/chunk 在已获取范围内原文未写**——动作离散化细节在 §IV 实验部分，未取到）

**【动作表征机制】原文引文+逐条指路**

- **核心机制：独立扩散时间步 = 掩蔽/条件化开关**，§III-B：
  > "Time-steps closer to T (fully noised) indicate full masking, while timesteps closer to 0 (unnoised) indicate no masking."
  > "Fixing the timestep for either t_a or t_o′ to T marginalizes the corresponding variable a, o′, while setting the timestep to 0 performs conditioning."
- **耦合评分网络**，§III-B：
  > "a UWM models a coupled noise prediction network s_θ(o, a_{t_a}, o′_{t_o′}, t_a, t_o′)... with t_a and t_o′ referring to the decoupled steps of the diffusion process with respect to actions and next-observations respectively."
- **训练**：t_a, t_o′ ∼ U(0,T) 独立采样，损失 w_a‖ε^θ_a−ε_a‖² + w_o′‖ε^θ_o′−ε_o′‖²（Eq.1）。
- **四模式时间步置位（原文逐字）**：
  1) Policy（边缘化 o′）："To sample from the policy p(a|o), we marginalize out the next observation o′ by setting t_o′=T and o′_T∼N(0,I). We perform the reverse diffusion process on actions going from t_a=T,…,1 with a_T∼N(0,I)"（Eq.2）
  2) Video Prediction（边缘化 a）："set t_a=T and a_T∼N(0,I)... reverse diffusion process on next observations going from t_o′=T,…,1"（Eq.3）
  3) Forward Dynamics（条件 a）："we condition on a particular action a by setting t_a=0 and a_0=a... reverse diffusion process on next observations going from t_o′=T,…,1"（Eq.4）
  4) Inverse Dynamics（条件 o′）："we condition on a particular next observation o′ by setting t_o′=0 and o_0=o... reverse diffusion process on actions going from t_a=T,…,1"（Eq.5 截断）
- **生成机制**：DDPM（epsilon prediction，ᾱ_t 调度），非流匹配。
- **动作空间/维度/chunk/离散化**：**未写（已获取范围）**。§III-A 设定 "a dataset of (observation, action, next-observation) pairs"；动作的 token 化/离散化细节在未取到的 §IV。

**【代码依据】** 论文页脚 "code are available at https://weirdlabuw.github.io/uwm/"（本次未抓取）→ 待核实。

**【why 追问】（基于原文）**
1. 为何把时间步当掩蔽用？§III-B："noising input tokens by setting the inference timestep for diffusion appropriately can naturally induce a form of partial masking"——同一模型经时间步置位即可边缘化/条件化任意模态，无需多模型或训练技巧。
2. 为何要解耦两个时间步？§III-B："training such a joint noise prediction network s_θ(o,a_t,o′_t,t) does not accomplish flexible inference since it can only sample from the joint distribution of (a,o′)"——耦合时间步只能采样联合分布，解耦才解锁四种分布。

**【关键数值】（逐字抄录）**
- "t_a, t_o′ ∼ U(0,T)"（Eq.1）
- 动作/视频预测损失权重 w_a、w_o′（数值未写）
- "Time-steps closer to T... closer to 0"（§III-B）
- 动作离散化相关数值：**未取到**

**【一页总结】** UWM 的动作表征关键不在动作空间本身，而在「时间步作开关」：单一耦合评分网络对动作与下一帧分别持独立扩散时间步 t_a、t_o′；推理时 t=0 条件化（注入干净数据）、t=T 边缘化（纯噪声）——组合出 policy/前向动力学/逆动力学/视频生成四模式。动作侧具体 token 形式（离散 VQ/维度/chunk）未出现在已获取正文范围。

**【待核实】** ① §IV 中动作离散化（VQ tokenizer、codebook、chunk 长度、数据集配置）（未取到）；② w_a/w_o′ 数值（未写）；③ 逆动力学模式完整公式（Eq.5 截断）。

---

## 10. DreamZero（arXiv:2602.15922）

**元信息**
- 标题：World Action Models are Zero-shot Policies
- 作者：Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng 等（NVIDIA；Project Leads: Yuke Zhu, Linxi "Jim" Fan, Joel Jang）
- 取文：ar5iv 页（截断于 §4.1 中段「PolaRiS」处；§4.2 之后、附录 C「Model and Training details」（含动作空间定义，很可能在此）未取到）。§3 全章取到。
- 代码：github.com/dreamzero0/dreamzero

**来源级别：L2**（§3.1 联合建模、§3.2 48 步@30Hz 全取到；**动作空间的具体定义（EE/关节维度）在已获取范围未写，在附录 C**）

**【动作表征机制】原文引文+逐条指路**

- **视频与动作联合建模**，§3.1 "Problem Formulation"：
  > "DreamZero jointly predicts video \mathbf{o}_{l:l+H} and actions \mathbf{a}_{l:l+H} conditioned on language instruction \mathbf{c}, proprioceptive state \mathbf{q}_l and visual observation... H>0 is a fixed horizon"
  > "joint prediction of video and action is a decomposition of (1) autoregressive video prediction and (2) action prediction from an inverse-dynamics model (IDM)"（Eq.1）
- **动作 token/维度**：正文用「normalized actions」（Eq.2 "\mathbf{a}^k_{t_k} = t_k \mathbf{a}^k_1 + (1−t_k)\mathbf{a}^k_0... and \mathbf{z}^k_1 and \mathbf{a}^k_1 are a clean video latent vector and a normalized action"）——**具体空间（EE/关节、维数、绝对/增量）未写（已获取范围，应在附录 C）**。
- **chunk 长度**，§3.1：
  > "DreamZero is trained to predict video frames in a chunk manner; each chunk has a fixed number of latent frames K to match the action horizon."
- **48 步@30Hz 的含义**，§3.2.2 "Asynchronous Closed-Loop Execution"：
  > "In our experiments, we deploy policies at an action horizon of 48 steps at 30Hz control frequency (1.6 seconds per chunk) for bimanual manipulation robots."
  → 动作 chunk＝48 步×1/30s＝1.6 秒；控制率 7Hz 指 chunk 级推理节拍（Abstract："real-time closed-loop control at 7Hz"）。
- **生成机制：流匹配 + 教师强制分块去噪**，§3.1：
  > "we employ flow-matching as the training objective"
  > "DreamZero shares the denoising timestep between video and action modality for faster convergence"
  > "we apply teacher forcing... the model is trained to denoise the noisy current chunk conditioned on the clean previous chunks."（Eq.3：v^k := [z^k_1, a^k_1] − [z^k_0, a^k_0]，K 个 chunk 求和，w(t_k) 加权）
  > 每个 chunk 内所有帧共享 t_k，不同 chunk 独立 timestep（Eq.2 上方："All frames within the same chunk share the same timestep t_k, while different chunks are assigned independent timesteps."）
- **DreamZero-Flash 解耦噪声调度**，§3.2.5：
  > "DreamZero-Flash closes this gap by biasing video timesteps toward high-noise states via t^{video}_k = 1−η, where η ∼ Beta(α,β) with α>β. In practice, we use Beta(7,1)... yielding E[t^{video}_k]=0.125 (predominantly noisy), while action timesteps remain uniform"
  > "we reduce the diffusion steps from four to one, cutting inference from ∼350ms to ∼150ms"
- 推理：KV cache，执行后以真值观测替换预测帧（§3.1 Model Inference）；异步执行（§3.2.2）。
- 基线扩散步数：16（§3.2.1："iterative denoising across 16 diffusion steps"）。
- 动作平滑：Savitzky-Golay 滤波，2× 上采样（§3.2.5 "Action Chunk Smoothing"）。

**【代码依据】** §1："We open-source our model weights, inference code, and code to run... at https://github.com/dreamzero0/dreamzero"（本次未抓取 → 待核实）。

**【why 追问】（基于原文）**
1. 为何视频与动作共享同一时间步（DreamZero 主体）？§3.1："DreamZero shares the denoising timestep between video and action modality for faster convergence at the beginning of training."——收敛快；而 Flash 变体又解耦（Beta 偏置视频向高噪），为匹配少步推理时「视频仍带噪」的训练-推理分布，两处动机均在原文。
2. 为何 chunk 取 48 步@30Hz？§3.2.2 唯一动机是异步闭环的缓冲时间："This structure transforms the latency constraint from 'inference must complete before the robot moves' to 'inference must complete before the current action chunk expires.'... we target inference latency below approximately 200ms"——1.6s chunk 给 200ms 级推理留出充分重叠；具体为何 48 步原文未展开。

**【关键数值】（逐字抄录）**
- "action horizon of 48 steps at 30Hz control frequency (1.6 seconds per chunk)"（§3.2.2）
- "7Hz"（Abstract）；"38× inference speedup"（§1）
- "5.7 seconds per action chunk" → "150ms"（§3.2.1/3.2.5）；"16 diffusion steps"→"from four to one"（§3.2.1/3.2.5）
- "Beta(7,1)... E[t^{video}_k]=0.125"（§3.2.5）
- "14B autoregressive video diffusion model"（Abstract）；预训练骨干 "Wan"（image-to-video diffusion backbone, §1）
- 数据："∼500 hours of teleoperation data across 22 unique environments"（§4.1）；"7.2K episodes"（Fig.6 caption）；跨 embodiment："12 minutes"/"20 minutes"/"30 minutes"（§1）

**【一页总结】** DreamZero 把动作与视频做成同一流匹配去噪对象：视频 latent 与「归一化动作」共享时间步（chunk 内所有帧共享 t_k，chunk 间独立），教师强制逐 chunk 自回归去噪，动作与视频由 14B Wan 骨干联合生成；执行采用 48 步@30Hz＝1.6s 的 chunk + 异步闭环 + KV 真值回填；Flash 变体用 Beta(7,1) 把视频时间步偏置到高噪，实现 1 步去噪推理。动作空间的物理定义（EE/关节维度）在正文未写，指向附录 C。

**【待核实】** ① 附录 C 中动作空间定义、归一化方法、K 值（未取到）；② 各 embodiment（AgiBot G1 bimanual / Franka）的动作维数（未取到）；③ w(t_k) 形式（未写）。

---

## 汇总对比表（10 篇 × 动作空间 × 维度 × chunk × 离散/连续 × 生成机制）

| # | 论文 | 动作空间 | 维度 | chunk 长度 | 离散/连续 | 生成机制 |
|---|------|----------|------|-----------|-----------|----------|
| 1 | GR-1 | ΔXYZ + Δ欧拉角（EE）+ 二值夹爪（§4.1） | 6+1 | 无 chunk（单步 a_t，§3.2.1） | 连续（Smooth-L1）+ 二值（BCE） | 因果 GPT [ACT] 头一步回归 |
| 2 | DiT4DiT | **未写**（GR00T-N1 适配，§4.2） | **未写** | **未写**（"action trajectories"） | 连续（速度场） | 流匹配；τ_a=1−Beta(α,β)、Euler 积分 |
| 3 | τ0-WM | **未写**（连续动作 chunk，§IV-A） | **未写** | H_a（**数值未写**） | 连续 | 流匹配（λ_z=λ_a=1）；Action DiT cross-attend 视频特征 |
| 4 | LAPA | 潜动作=VQ 码（预训练）；ΔEE 离散 bin（微调，§3.3） | 潜：8^4（4 码×8 词表）；真动作：256 bins×7（"256^7"） | 视频窗 H（**数值未写**，附录 A） | 潜：离散；真：离散化 | 自回归 VLM（LWM-7B）token 预测 |
| 5 | V-JEPA 2-AC | ΔEE：Δ位置 3 + Δ外蕴欧拉 3 + Δ夹爪 1（§3.1） | 7 | 15 步（4fps×16 帧 clip，T=15） | 连续 | 300M block-causal 自回归表征预测（L1 + 2-step rollout） |
| 6 | Action Images | 绝对 EE pose（pos/θ 6）+ 夹爪开度，渲染为多视角 3 通道 RGB 热图视频（§3.1） | 7-DoF → T×H×W×3 像素 | 与视频同帧率 T（**数值未写**） | 连续（像素化） | 流匹配（Wan 2.2，v=ε−X）；几何解码回 7-DoF |
| 7 | PointWorld | 3D robot point flows（URDF FK 传播的 gripper 表面点，§3.1） | N_R×3（每 gripper 几百点） | H=10 步@0.1s | 连续（3D 点） | 单前向回归（movement-weighted Huber+不确定性）；MPC-MPPI 转回 EE 命令 |
| 8 | Motus | 光流潜动作（跨 embodiment）→ 目标机器人动作（Stage 3） | 潜动作 14 维（DC-AE 4×512→14） | k 步 chunk（**数值未写**） | 连续 | 整流流（rectified flow），UniDiffuser 式独立 τ_a/τ_o；视频帧率=动作 1/6 |
| 9 | UWM | **未写**（已获取范围；离散化细节在 §IV） | **未写** | **未写** | **未写**（按 §III 为连续高斯扩散对象） | DDPM；独立时间步 t_a/t_o′ 置位切换四模式 |
| 10 | DreamZero | **未写**（"normalized actions"，定义在附录 C） | **未写** | 48 步@30Hz＝1.6s（§3.2.2） | 连续（流匹配对象） | 流匹配+教师强制 chunk 自回归；Flash：视频 t=1−Beta(7,1)、动作均匀 |

> 标注说明：「未写」= 该字段在本次已获取原文范围内未出现；「未取到」= 位于截断点之后的章节（如附录 C、§IV）。

## 趋势与争议（仅基于原文）

**趋势**
1. **流匹配/整流流取代 DDPM 成为主流生成机制**：DiT4DiT（§4.1）、τ0-WM（§IV-C）、Action Images（§3.3）、Motus（§4.1）、DreamZero（§3.1）全部采用流匹配/整流流速度场目标；仅 UWM（2025-04）用 DDPM ε 预测（§III-B）。
2. **多模态时间步/噪声调度解耦**：UWM（双时间步，§III-B）→ Motus（UniDiffuser 式独立 τ_a/τ_o，§4.1）→ DiT4DiT（三时间步 τ_v/τ_f/τ_a，§4.3）→ DreamZero-Flash（Beta(7,1) 偏置视频时间步，§3.2.5）。但 DreamZero 主体明确「共享时间步以获得更快收敛」（§3.1），Flash 才解耦——共享 vs 解耦是显式张力点。
3. **动作的「像素化/几何化」以换取跨 embodiment 迁移**：Action Images（动作→多视角 RGB 热图）、PointWorld（动作→3D 点流）、Motus（动作→光流潜码）都把动作压到与视觉同构的空间；三篇的原文理由高度一致（跨视角/跨 embodiment、用视频骨干先验、避免独立动作模块的迁移瓶颈）。
4. **无标签视频上的「潜动作预训练」成为动作表征新增层**：LAPA（离散 VQ 码）、Motus（连续 14 维）、V-JEPA 2-AC（潜表征空间中的 ΔEE 条件）都在动作之上加了一层从像素/流学到的中间表征，再用少量真动作标签接地。
5. **动作执行形态收敛到 chunk 化**：除 GR-1（单步）外，所有给出 chunk 信息的论文均为 action chunk（τ0-WM H_a、PointWorld H=10、DreamZero 48 步、Motus k 步、V-JEPA 2-AC 15 步），与视频预测 horizon 对齐是共性设计（τ0-WM Eq.1、DreamZero Eq.1、PointWorld §3.1）。

**争议/分歧**
1. **离散 vs 连续**：LAPA 主张离散 VQ 码「making it easy for VLMs to predict」（§3.1），Motus 主张连续 14 维「roughly matching the scale of typical robot action spaces」（§4.2）——同一条「潜动作」路线下的两种相反选择，各自引文理由明确但无第三方裁决。
2. **绝对 vs 增量位姿**：GR-1/V-JEPA 2-AC 用 ΔEE（§4.1/§3.1），Action Images 用绝对 EE pose（§3.1）——增量便于策略回归、绝对便于像素几何编码，两篇均未互相讨论。
3. **动作空间写明的透明度悬殊**：10 篇中 4 篇（DiT4DiT、τ0-WM、UWM、DreamZero）在正文已获取范围内未写明动作空间的物理定义/维度；其中 DiT4DiT、DreamZero 的动作细节疑似分别外置到「GR00T-N1 适配」（§4.2）与附录 C——这对复现与跨论文对比构成实质性障碍。
4. **时间步语义的方向约定不一致**：UWM 以 t=T 为噪声（DDPM 惯例，§III-B），流匹配系（DiT4DiT/τ0-WM/Motus/DreamZero）以 τ=1 为噪声（§4.1 等）——机制等价但记法相反，跨论文读源码时需注意。

## 每篇来源级别与截断位置（速查）

| 论文 | 级别 | 截断位置 |
|------|------|----------|
| GR-1 | L3 | arxiv html v2 截断于 §4.1 中段；ar5iv 截断于 §4.2.2 开头；附录 A 未取到 |
| DiT4DiT | L2 | ar5iv 截断于 §4.3 Algorithm 1 第 21 行；§4.4 之后未取到 |
| τ0-WM | L2 | ar5iv 截断于 §VI-B LAR 段；§VII 之后未取到 |
| LAPA | L2 | ar5iv 截断于 §5.2 中段；附录 A/B 未取到 |
| V-JEPA 2 | L2 | ar5iv 截断于 §3.1 rollout loss 公式；§3.2/§4 未取到 |
| Action Images | L3 | ar5iv 截断于 §4.1 Table 2；§4 之后未取到 |
| PointWorld | L3 | ar5iv 截断于 §4「Model Evaluation Protocol」中段；§5 未取到 |
| Motus | L3 | ar5iv 截断于 §5.2 Table 2；§5 之后未取到 |
| UWM | L2 | ar5iv 截断于 §III-B 逆动力学公式；§III-C/§IV 未取到 |
| DreamZero | L2 | ar5iv 截断于 §4.1 PolaRiS 处；附录 C 未取到 |

**动作空间（物理定义或维度）在已获取正文中未写明的论文清单**：DiT4DiT、τ0-WM、UWM、DreamZero（共 4 篇）；另有 LAPA/Motus 的「真实执行动作维数」也未在已获取范围写明（其潜动作层写明，但下游真动作细节分别在 §3.3/Stage 3 一笔带过）。

<!-- QA_LOG -->

