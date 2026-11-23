# WAM 精读线①：输入表征（10 篇）

> 生成时间：2026-10-26（Asia/Shanghai）
> 取文方式：每篇 arXiv abs 页 + arxiv.org/html 全文（最新版本）。所有 HTML 长文均在实验中段被截断；每篇下方标注截断位置与「未取到」范围。
> 铁律执行声明：所有引文为论文原文 verbatim（LaTeX 已做最小化转写，如 `4×`、`p(a1:H | o,l)`），数值均逐字抄录自已获取正文；未获取内容一律标注，无脑补。

---

## 1. WorldVLA — Towards Autoregressive Action World Model

- **arXiv**：2506.21539（cs.RO）｜**提交**：v1 2025-06-26（仅 v1）
- **机构**：DAMO Academy, Alibaba Group；Hupan Lab；Zhejiang University
- **代码**：https://github.com/alibaba-damo-academy/WorldVLA（abs 页 Comments）
- **来源级别**：L3（§3 方法全文核实）／L2（§4.2 起截断）｜**截断位置**：§4.2.3 开头（"Table 3: Ablation study of action model." 处，Table 3 及之后未取到）

### 【输入表征机制】原文引文

**§3.2 Architecture（三 tokenizer、共享词表）**
> "We initilize the model from Chameleon (Team, 2024) since it is a unified model for image understanding and generation. Three tokenizers are involved, including an image tokenizer, a text tokenizer, and an action tokenizer. The image tokenizer is a VQ-GAN model (Esser et al., 2021) with additional perceptual losses to specific image regions, e.g., faces and salient objects (Gafni et al., 2022). The compression ratio of the image tokenizer is 16 and the codebook size is 8192. The image tokenizer generates 256 tokens for 256×256 images and 1024 tokens for 512×512 images. The action tokenizer discretizes each dimension of continuous robot actions into one of 256 bins, with bin widths determined by the range of the training data (Kim et al., 2024; Brohan et al., 2023). The actions are represented as 7 tokens, including 3 relative positions, 3 relative angles and 1 absolute gripper states. The text tokenizer is a trained BPE tokenizer (Sennrich et al., 2015) with a vocabulary size of 65,536, which includes 8192 image tokens and 256 action tokens. All texts, actions, and images are discretized into tokens and are trained under the autoregressive manner."

**§3.3 Training Strategy → Action Model Data（文本模板 + token 序列）**
> "The text inputs are \"What action should the robot take to + task instruction + ?\". The overall token sequence is: [BOS]{text}[BOI]{image}...{image}[EOI]×M [EOS][BOA]{action}...[EOA]×K [EOS] ... The input contains M images and the output contains K actions. We only calculate the loss of action tokens L_action."

**§3.3 World Model Data（世界模型分支的输入序列）**
> "The text inputs are \"Generate the next frame based on the current image and the action.\". The overall token sequence is: [BOS]{text} [BOI]{image}[EOI][BOA]{action}[EOA][EOS] [BOI]{image}[EOI][EOS]×N. The next frame prediction conditioned on the action repeats N times, and we only calculate the loss of generated image tokens L_world."

**§3.3 Attention Mask（action chunk 掩码设计）**
> "This modified mask ensures that current actions rely solely on textual and visual inputs, while prohibiting access to prior actions. Such a design enables the autoregressive framework to generate multiple actions in parallel, aligning with methodologies presented in (Kim et al., 2025; Black et al., 2024). The world model part adheres to the conventional causal attention mask."

**§4.1 Training Setting（输入规模配置）**
> "The action model utilizes a default input image count of M=2. The action chunk size is set to K=10 for the LIBERO Long task and K=5 for the remaining three LIBERO tasks under default configuration. To minimize computational expenditure, the world model operates with a single round N=1. The parameter α is fixed at 0.04 in the experimental setup."

### 逐条中文指路
- **观测编码格式/维度/token 数**：VQ-GAN 离散 token，压缩比 16，codebook 8192；256×256→256 tokens，512×512→1024 tokens（§3.2）。
- **语言进入方式**：BPE 离散 token 直接进 LLM 序列（词表 65,536，含 8192 图像 + 256 动作 token）；固定文本模板拼接任务指令（§3.2/§3.3）。
- **proprio 注入**：原文未写。动作（7 tokens：3 相对位置 + 3 相对角度 + 1 绝对夹爪状态）只作为**输出**与世界模型输入，未见本体状态作为策略输入的描述（已获取范围内）。
- **多相机处理**：原文未写（已获取范围内，全文仅单图表述）。
- **时间组织**：历史帧数 M=2（图像序列 [BOI]...[EOI]×M）；action chunk K=5/10；世界模型单轮 N=1；自回归因果掩码（action 分支用并行掩码）。

### 【why 追问】
1. **为什么三种模态要共享词表？** 原文答（§3.2）：「The tokens from different modalities are set to share the same vocabulary so that understanding and generation across these modalities can be unified within a single LLM architecture.」（引文见 §1 引言段）。
2. **为什么 action chunk 掩码要屏蔽先前动作？** 原文答（§3.3 Attention Mask）：MLLM 在动作域泛化弱，"errors originating from earlier actions propagate to subsequent actions under the default attention mask, resulting in performance degradation"；屏蔽后当前动作只依赖文本与视觉输入，可并行解码。
3. **为什么 α 只有 0.04？** 原文答（§3.3 Training Objective）：「Since the image tokens (256 tokens for 256×256 images and 1024 tokens for 512×512 images) are much more than the action tokens (7 tokens), we use α to balance the loss contribution.」
4. **为什么 512 分辨率更好？** 原文答（§4.2）：「This phenomenon is primarily attributable to the pretraining regimen of the Chameleon backbone, whose image tokenization module and the large language model components are inherently optimized at 512×512 resolution. Additionally, higher resolution naturally provides a greater level of detailed visual information, which is particularly crucial for robotic grasping tasks.」——原文未写（已获取范围）部分：proprio/多相机设计的动机。

### 【关键数值】（verbatim 自正文）
- 图像压缩比 16；codebook 8192；词表 65,536；动作 bin 数 256；动作 token 数 7
- 256×256 → 256 tokens；512×512 → 1024 tokens
- M=2、K=10（Long）/K=5（其余）、N=1、α=0.04
- Table 2（LIBERO，离散动作模型）：WorldVLA(256)：Spatial 85.6 / Object 89.0 / Goal 82.6 / Long 59.0 / Avg 79.1；WorldVLA(512)：87.6 / 96.2 / 83.4 / 60.0 / 81.8；OpenVLA：84.7 / 88.4 / 79.2 / 53.7 / 76.5

### 【一页总结】
问题：VLA 把动作只当输出、世界模型不能直接出动作，二者割裂。方法：Chameleon 初始化，图像/文本/动作三 tokenizer 离散化共词表，同一 LLM 自回归联合建模动作与下一帧；混合动作模型数据与世界模型数据，损失 L = L_action + αL_world（α=0.04）。实验：LIBERO 四套件上 WorldVLA(512) 均值 81.8 超过离散 OpenVLA 76.5（无预训练）；世界模型支线 FVD 降 10%（摘要）。关键发现：自回归逐 token 生成 action chunk 掉点 10–50%，屏蔽先前动作的掩码可挽回 4–23%。Takeaway：离散自回归 + 共享词表 + 并行 action 掩码是输入表征层的核心设计，输入极简（文本+图像，无 proprio）。

### 【图】
- Figure 1："(a) Action model generates actions based on image understanding; (b) World model generates the image based on image and action understanding; (c) Action World Model unifies both image and action understanding and generation." — https://arxiv.org/html/2506.21539v1
- Figure 2："Overview of WorldVLA. WorldVLA integrates two distinct but complementary functional components: an action model and a world model. The action model is responsible for generating actions conditioned on both textual and visual data. The world model functions to predict the subsequent environmental state (e.g., the next visual frame) by leveraging textual information, current image, and current action." — 图文件 https://arxiv.org/html/2506.21539v1（2506.21539v1/fig_1_3.png）
- Figure 3："Attention mask mechanism of (a) default action model, (b) our proposed action model, and (c) world model."
- Table 1（§2）："Comparason of different action and video generative models. T: Text; V: Video; A: Action."（Action World Model 行：Input T+V+A、Output V+A）

### 【待核实/原文未写】
- proprio/机器人状态输入：原文未写（已获取范围）
- 多相机：原文未写（已获取范围）
- 图像 tokenizer 的 patch/分辨率细节、训练数据规模：§4 后半未取到
- 历史帧时间间隔（帧率）：原文未写（已获取范围）

---

## 2. GR-1 — Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation

- **arXiv**：2312.13139（cs.RO）｜**提交**：v1 2023-12-20，v2 2023-12-21（读取 v2）
- **机构**：ByteDance Research
- **项目页**：https://GR1-Manipulation.github.io
- **来源级别**：L3（§3 方法全文核实）／L2（§4.1 起截断；附录 A 未取到）｜**截断位置**：§4.1 中部（"We perform experiments on two splits of data: ABCD→D and ABC" 处）

### 【输入表征机制】原文引文

**§3.2.1 Inputs（三模态编码器）**
> "Language input. The language l is encoded via a text encoder (Fig. 2(a)). Following (Shridhar et al., 2022; Shridhar et al., 2023), we employ CLIP (Radford et al., 2021) as the language encoder."
> "Visual input. Visual observations o are encoded via a Vision Transformer (ViT) which has been pre-trained with MAE (He et al., 2022) (Fig. 2(c)). The output CLS token z_o^CLS is used as a global representation of the image. The output patch tokens z_o^{p_{1:i}} are used as local representations which are further processed with a perceiver resampler (Jaegle et al., 2021) to reduce the token number."
> "Robot state input. The robot state s contains the 6D pose of the robot end-effector s_arm ∈ SE(3) and a binary status of the gripper s_gripper ∈ {0,1}. We use linear layers to encode them (Fig. 2(b))."

**§3.2.1 Inputs（token 序列与时间注入）**
> "During video generative pre-training, the tokens are sequenced as: (l, o_{t-h}, [OBS], l, o_{t-h+1}, [OBS], ..., l, o_t, [OBS]). When finetuning on robot data, the tokens are sequenced as: (l, s_{t-h}, o_{t-h}, [OBS], [ACT], l, s_{t-h+1}, ..., l, s_t, o_t, [OBS], [ACT]). Note that the language token is repeated in each timestep to avoid being overwhelmed by other modalities. To inject temporal information, we add learned relative timestep embeddings to the tokens. All modalities at a timestep share the same timestep embedding."

**§3.2.2 Network（因果掩码）**
> "We follow the causal attention mechanism typically used in generative pre-trained models, except that all the [ACT] and [OBS] tokens are masked (Fig. 1). That is, during pre-training, a token can attend to all the tokens in the previous positions except the [OBS] tokens; during finetuning, a token can attend to all the tokens in the previous positions except the [ACT] and [OBS] tokens."

**§3.3 Training（冻结编码器）**
> "In both pre-training and finetuning phases, we freeze the CLIP text encoder and MAE image encoder."

### 逐条中文指路
- **观测编码格式/维度/token 数**：MAE 预训练 ViT；CLS token 作全局表示，patch tokens 经 Perceiver Resampler 降 token 数（数量未写）；[OBS] 为可学习「观测预测 token」。
- **语言进入方式**：CLIP 文本编码器（冻结）；语言 token 每个时间步重复注入序列，防止被其他模态淹没。
- **proprio 注入**：6D 末端位姿（SE(3)）+ 夹爪二值状态，线性层编码，以 s_t 序列进入 transformer——最明确的 proprio 注入路线之一。
- **多相机处理**：原文未写（CALVIN/真实机器人多为单视角设置；未描述多相机拼接）。
- **时间组织**：历史窗口 h 帧 + 学到的相对 timestep embedding（同时间步所有模态共享同一 embedding）；[OBS] 掩码 token 预测未来帧、[ACT] token 预测动作。

### 【why 追问】
1. **为什么语言 token 每个时间步重复？** 原文答（§3.2.1）：「the language token is repeated in each timestep to avoid being overwhelmed by other modalities.」
2. **为什么用 MAE 预训练 ViT + CLIP 文本编码器并冻结？** 原文答（§3.3）：「In both pre-training and finetuning phases, we freeze the CLIP text encoder and MAE image encoder.」——冻结理由未明说（推测为保持预训练表征稳定，标注：原文未写明确理由）。
3. **为什么 patch token 要过 Perceiver Resampler？** 原文答（§3.2.1）：「to reduce the token number」——压缩比未写。
4. **为什么预训练任务用视频预测而非掩码建模？** 原文答（§1）：「The ability to forecast future frames based on past images and language instructions empowers the robot to anticipate forthcoming events, thereby facilitating the generation of relevant and suitable actions.」

### 【关键数值】（verbatim 自正文）
- 机器人状态：6D 位姿 + 1D 二值夹爪
- Ego4D 预训练：「We crop a short clip with a duration of 3s from each video. With this strategy, a total of 800,000 video clips, containing 8M frames, are collected.」（§3.3）
- Table 1（CALVIN ABCD→D）：GR-1 1/2/3/4/5 连做率 0.949/0.896/0.844/0.789/0.731，Avg.Len. 4.21（HULC 3.06）
- 图像分辨率 / 历史长度 h / patch token 数量：未取到（附录 A.1 未获取）

### 【一页总结】
问题：机器人数据稀缺多模态，能否用大规模视频生成预训练赋能语言条件操作。方法：GPT 风格因果 transformer，输入 = CLIP 语言 token（每步重复）+ MAE-ViT 图像 token（CLS + Perceiver 降采样 patch）+ 线性编码的 6D 位姿/夹爪状态，输出 [ACT] 动作与 [OBS] 未来帧（像素 MSE）；Ego4D 80 万片段预训练 → 机器人数据微调（冻结两个编码器）。实验：CALVIN 连做 5 任务成功率 94.9%（vs 88.9%）、平均长度 4.21；10% 数据 77.8%。Takeaway：输入层三模态 + 每步重复语言 token + 相对时间 embedding 是该文输入设计的关键；proprio 以状态序列显式进入。

### 【图】
- Figure 1："Overview of GR-1. GR-1 is first pre-trained on the task of video prediction with a large-scale video dataset. It is then finetuned on robot data to learn multi-task visual robot manipulation."
- Figure 2："Encoders and Decoders. (a) Language encoder. (b) Robot state encoder. (c) Image encoder. (d) Image decoder. (e) Action decoder." — https://arxiv.org/html/2312.13139v2（2312.13139v2/encoder_decoder.png）

### 【待核实/原文未写】
- 图像分辨率、patch 大小、Perceiver token 数量、历史窗口 h 帧数、[OBS] 预测帧数 Δt：附录 A.1 未取到
- 多相机处理：原文未写（已获取范围）
- 冻结编码器的设计动机：原文未写明理由

---

## 3. Motus — A Unified Latent Action World Model

- **arXiv**：2512.13030（cs.CV）｜**提交**：v1 2025-12-15，v2 2025-12-25（读取 v2）
- **机构**：Peking University；Tsinghua University（BNRist、THBI、Tsinghua-Bosch Joint ML Center）；Horizon Robotics
- **项目页**：https://motus-robotics.github.io/motus
- **来源级别**：L3（§3–4.2 核实）／L2（§4.3 起截断；§9 Implementation Details 未取到）｜**截断位置**：§4.3 标题处（"Motus Training."）

### 【输入表征机制】原文引文

**§3 Problem Formulation → Embodied Policies（proprio 显式定义）**
> "For each embodiment, the task defines an action a∈A, an observation o∈O (visual input), a language instruction ℓ∈L, and the proprioception of the robot p, where A, O and L denote the action space, the observation space, and the language instruction space respectively. The task typically provides an expert dataset D_expert = {{ℓ, p_1, o_1, a_1, ..., p_N, o_N, a_N}}, which contains robot proprioception, visual observations, and actions collected by an expert over N timesteps, along with corresponding language annotations for each trajectory."
> "At each timestep t, the policy predicts the next k actions (action chunking) based on the current observation and proprioception, modeling the distribution p_θ(a_{t+1:t+k} | o_t, p_t, ℓ) or p_θ(a_{t+1:t+k} | o_t, ℓ)."

**§4.1 Experts Details（三专家与输入归属）**
> "For the generative expert, we employ Wan 2.2 5B as the video foundation model for its accessibility and ease of use. We extend its self-attention context to create a cross-modal Tri-model Joint Attention mechanism. For the action expert, we construct a Transformer block of the same depth as Wan. Each block comprises AdaLN for injecting rectified flow timesteps, a Feed-Forward Network (FFN), and the Tri-model Joint Attention for cross-expert interaction. We select Qwen3-VL-2B for our understanding expert due to its inherent capabilities in 3D grounding, spatial understanding, and precise object localization, which are crucial for robotic manipulation. The input to this expert is taken from the last-layer corresponding tokens of the VLM."

**§4.1 Action-Dense Video-Sparse Prediction（帧率权衡）**
> "we downsample the video frames so that the number of video tokens and action tokens remains balanced—for example, by setting the video frame rate to one-sixth of the action frame rate."

**§4.2 Optical Flow Based Representation（latent action 编码）**
> "optical flows are computed by DPFlow and then converted into RGB images. To compress this high-dimensional representation into a control-level space, we employ a deep convolutional variational autoencoder (DC-AE) that reconstructs the flow while encoding it into four 512-dimensional tokens. A lightweight encoder then projects these concatenated 4×512 features into a 14-dimensional vector, roughly matching the scale of typical robot action spaces."

**§4.2 Training and Distribution Alignment（监督比例）**
> "we mix 90% unlabeled data for self-supervised reconstruction with 10% labeled trajectories for weak action supervision, where the labeled portion includes both task-agnostic data and standard robot demonstrations."

### 逐条中文指路
- **观测编码格式/维度/token 数**：视觉观察经 Wan2.2-5B 视频 VAE 进 latent（token 数与分辨率在 §9 未取到）；Qwen3-VL-2B 理解专家取 VLM 最后一层对应 token 作为输入。
- **语言进入方式**：语言指令 ℓ 为任务定义要素；注入路径细节（文本编码器/交叉注意力）在已获取范围内未写明 → 原文未写（已获取范围）。
- **proprio 注入**：p_t 显式为策略输入之一 p_θ(a | o_t, p_t, ℓ)；注入实现细节（token 化方式）原文未写（已获取范围，疑在 §9）。
- **多相机处理**：原文未写（已获取范围）。
- **时间组织**：Action-Dense Video-Sparse——视频帧率 = 动作帧率的 1/6 以平衡 token 数；latent action 以光流（DPFlow）→ DC-AE（4×512 tokens）→ 14 维向量压缩。

### 【why 追问】
1. **为什么用光流定义 latent action？** 原文答（§1）："We introduce latent actions, which encode motion patterns from optical flow as a pixel-level 'delta action'. This representation bridges visual dynamics with control signals, enabling the action expert to be pretrained on diverse unlabeled videos and robot trajectories."（另见 §2.2："inspired by optical flow as a universal motion expression, we use it to align cross-embodiment behaviors and learn latent actions to facilitate large-scale pretraining"）。
2. **为什么视频帧率要降到动作的 1/6？** 原文答（§4.1）："(1) low training and inference efficiency, (2) redundant video frame predictions, and (3) an imbalance in the Tri-modal Joint Attention mechanism—where the number of video tokens significantly exceeds that of action tokens. This imbalance causes the model to overfit to video prediction, thereby weakening its action prediction capability."
3. **为什么 latent action 压缩到 14 维？** 原文答（§4.2）：「a 14-dimensional vector, roughly matching the scale of typical robot action spaces. This dimensional correspondence ensures that the latent representation can align naturally with real robotic controls and act as a bridge between perception and action.」
4. **为什么 90/10 混合？** 原文答（§4.2）：「Dimensional correspondence and weak action supervision jointly drive the latent-action distribution to align with the real action distribution, allowing motion priors learned from videos to naturally map to executable controls.」——90%/10% 比例的消融依据未写。

### 【关键数值】（verbatim 自正文）
- Wan 2.2 5B（生成专家）；Qwen3-VL-2B（理解专家）
- latent action：DC-AE 编码 4×512 tokens → 14 维向量
- 视频:动作帧率 = 1:6
- 无监督:有监督 = 90%:10%
- 摘要：仿真 +15%（vs X-VLA）、+45%（vs π0.5）；真机 +11~48%

### 【一页总结】
问题：理解/世界模型/控制被拆成孤立模型，异构无标签视频无法喂给动作专家。方法：MoT 三专家（Wan2.2-5B 生成 + Qwen3-VL-2B 理解 + 动作专家）共享 Tri-model Joint Attention；UniDiffuser 风格调度器统一 5 种分布（VLA/WM/IDM/VGM/联合）；光流→DC-AE→14 维 latent action，用 10% 有标签弱监督对齐真实动作空间；三阶段训练 + 六层数据金字塔。实验：RoboTwin/LIBERO/真机均 SOTA（摘要数值见上）。Takeaway：输入表征以「光流 latent action」跨 embodiment 共享运动知识为核心，视频稀疏化（1/6 帧率）解决 token 失衡。

### 【图】
- Figure 1："Motus Architecture. Here, a_t...a_{t+k} are actions, z_t...z_{t+k} are latent actions, and τ_v and τ_a are the rectified flow timesteps for the video generation model and the action expert, respectively."
- Figure 2："Action-Dense Video-Sparse Prediction. The sampling rates for video frames and actions differ."
- Figure 3："The Latent Action VAE." — https://arxiv.org/html/2512.13030v2（2512.13030v2/vae.png）

### 【待核实/原文未写】
- 图像分辨率、VAE latent token 数/维度、语言注入实现、proprio token 化方式：§9 Implementation Details 未取到
- 多相机：原文未写（已获取范围）
- 90/10 比例选择依据：原文未写

---

## 4. Fast-WAM — Do World Action Models Need Test-time Future Imagination?

- **arXiv**：2603.16666（cs.CV）｜**提交**：v1 2026-03-17，v2 2026-03-23（读取 v2）
- **机构**：IIIS, Tsinghua University；Galaxea AI
- **项目页**：https://yuantianyuan01.github.io/FastWAM/
- **来源级别**：L3（§3 + §4.1 全文核实）／L2（§4.3.1 截断；附录仅 A.1 RoboTwin 结果表，无输入细节章节）｜**截断位置**：§4.3.1 Table 2 末行（LIBERO 表 Average 列被截）

### 【输入表征机制】原文引文

**§3.2 Architecture（主干与 token 分组）**
> "Fast-WAM is built on top of the video Diffusion Transformer (DiT) from Wan2.2-5B, which serves as the world modeling backbone. We also reuse its pretrained text encoder and video VAE: task language is encoded by the built-in T5 encoder and provided to all tokens through cross-attention, while visual observations are mapped into latent video tokens by the pretrained VAE."
> "We organize the input tokens into three groups: clean latent tokens of the first observation frame, which serve as the shared visual anchor; noisy latent tokens of future video frames, which are used only during training for video modeling; and action tokens, which are processed by the action expert for action generation. All token groups attend to the language embeddings through cross-attention."
> "A structured attention mask controls the information flow between these groups. During training, future noisy video tokens attend bidirectionally within the video branch and can access the clean first-frame tokens; action tokens attend bidirectionally within the action branch and can also access the clean first-frame tokens. Crucially, action tokens cannot attend to future video tokens, and the clean first-frame tokens do not attend to any other tokens."
> "At inference time, Fast-WAM removes the future video branch entirely: only the clean first-frame latent tokens are retained and passed through the video backbone once to produce latent world features for the action expert."

**§4.1 Implementation Details（帧率/多相机/超参）**
> "We use pretrained Wan2.2-5B as the backbone, including its video DiT, text encoder, and video VAE. The action expert shares the same architecture as the video branch but uses a reduced hidden dimension of d_a=1024, resulting in a 1B action expert and a total model size of 6B parameters. We set the action horizon to h=32. Video frames are temporally downsampled by 4×, resulting in 9 video frames per chunk. Images from multiple cameras are concatenated into a single image before being fed into the VAE."
> "We use the same flow-matching formulation for both the video and action branches. Following [36], we adopt a logit-normal distribution over t as the noise schedule during both training and inference. During inference, we use 10 denoising steps with classifier-free guidance (CFG) scale set to 1.0. We use AdamW with a learning rate of 1×10^-4, weight decay of 0.01, and a cosine annealing schedule for all training settings. Training is conducted in mixed precision with gradient clipping at 1.0. All latency numbers are measured on a single NVIDIA RTX 5090D V2 32GB GPU."
> "For Fast-WAM-IDM, we follow [3] and apply noise augmentation to the ground-truth video tokens with probability p=0.5."

### 逐条中文指路
- **观测编码格式/维度/token 数**：Wan2.2-5B 预训练 video VAE latent；「clean latent tokens of the first observation frame」作共享视觉锚点，未来帧 noisy latent 仅训练期存在（token 数与 VAE 下采样率未逐字给出，未取到）。
- **语言进入方式**：Wan2.2 内置 T5 编码器，经 cross-attention 注入**所有** token 组。
- **proprio 注入**：原文未写——方法节与实现细节均未提及机器人状态/proprio 输入（已获取范围）。**代码补充（2026-10-26 官方仓库实锤）**：论文未提 ≠ 模型没有。`fastwam.py` 中 `self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)`，`_append_proprio_to_context` 把 proprio 投影为 **1 个 token**（`proprio_token = self.proprio_encoder(proprio.unsqueeze(1)) # [B, 1, D]`）并 `torch.cat([context, proprio_token], dim=1)` **追加到 T5 文本条件序列尾部**，经 cross-attention 同时喂给视频与动作专家；`build_inputs` 取 `proprio[:, 0, :]`（观测时刻状态，LIBERO 8 维 / RoboTwin 14 维）。→ 论文表述不全，输入实为：首帧图像 latent + T5 文本 + 1 个 proprio token。
- **多相机处理**：「Images from multiple cameras are concatenated into a single image before being fed into the VAE.」（verbatim，§4.1）；各相机分辨率与拼接布局：未取到/原文未写（已获取范围）。
- **时间组织**：temporal downsample 4× → 每 chunk 9 帧；action horizon h=32；训练掩码：动作 token 不能看未来视频 token，首帧 token 不看任何其他 token。

### 【why 追问】
1. **为什么推理时删掉未来视频分支？** 原文答（§3.2）：「Since no future noisy video tokens are instantiated and no explicit future video denoising is performed, Fast-WAM incurs substantially lower inference cost than standard imagine-then-execute WAMs.」（更深层动机见 §1：视频预测的价值可能在训练期塑造表征而非测试期显式生成）。
2. **为什么动作 token 禁止 attend 未来视频 token？** 原文答（§3.2）：「This ensures that both video modeling and action prediction are grounded in the same visual context while preventing future information from leaking into the action branch.」
3. **为什么多相机要拼成单图再进 VAE？** 原文未写理由（§4.1 仅陈述事实）——标注：原文未写。
4. **为什么 temporal downsample 是 4×（9 帧）？** 原文未写理由；仅陈述 "resulting in 9 video frames per chunk"——标注：原文未写。

### 【关键数值】（verbatim 自正文）
- Wan2.2-5B 主干；动作专家 d_a=1024、1B；总 6B 参数
- action horizon h=32；temporal downsample 4× → 9 帧/chunk
- 推理：10 步去噪、CFG=1.0；LR 1×10^-4、weight decay 0.01；RTX 5090D V2 32GB
- IDM 变体：视频 token 噪声增强概率 p=0.5
- 摘要：190 ms 延迟、比 imagine-then-execute WAM 快 4× 以上
- Table 1（RoboTwin）：Fast-WAM Clean 91.88 / Rand. 91.78 / Avg 91.8；Fast-WAM w.o. video co-train 82.76 / 84.80 / 83.8；Motus 88.66/87.02/87.8
- Table 2（LIBERO，部分）：Fast-WAM Spatial 98.2 / Object 100.0 / Goal 97.0 / Long 95.2（Average 列截断未取到）

### 【一页总结】
问题：WAM 的价值来自训练期视频共训还是测试期显式想象？方法：保留训练期视频共训（joint flow matching，L = L_act + λL_vid），推理时丢弃未来视频分支，只用首帧 clean latent 单次前向出动作；受控变体 Joint / IDM / 无视频共训对照。实验：RoboTwin 91.8（无 embodied 预训练，追平 Motus 87.8 并超越）、LIBERO 各套件 95.2–100.0；去掉视频共训掉 8–12 点。Takeaway：输入表征层的关键结论——WAM 增益主要来自训练期视频共训塑造的 latent 世界表征，而非测试期未来帧输入；多相机拼单图是唯一的显式多视角处理。

### 【图】
- Figure 1："Three representative WAM paradigms. (A) Joint-modeling WAMs denoise future video and action tokens together. (B) Causal WAMs first generate future observations and then condition action prediction on the generated future representation. (C) Fast-WAM retains video co-training during training but removes explicit future generation at inference time, directly predicting actions from latent world representations in a single forward pass."
- Figure 2："Fast-WAM architecture and the structured attention mask used to disentangle video co-training from action generation." — https://arxiv.org/html/2603.16666v2（2603.16666v2/model_arch.png）
- Figure 3："Real-world towel-folding task on the Galaxea R1 Lite platform. Folding a deformable object requires long-horizon planning and precise closed-loop manipulation, making it a challenging benchmark for evaluating both task success and execution efficiency."

### 【待核实/原文未写】
- 各相机分辨率、拼接布局（2×1？横竖？）：未取到（v2 全文已获取范围内未出现）
- VAE latent 维度/token 数、T5 版本：未取到（继承 Wan2.2-5B 配置，未逐字写明）
- proprio/本体状态输入：论文原文未写（已获取范围）；**代码补充**：有——1 个 proprio token 追加进文本条件序列（见上「proprio 注入」）。论文与代码在此处不一致，原因未知（待与作者/附录核对）。
- temporal stride 具体值（downsample 4× 对应 stride）：仅 "temporally downsampled by 4×" 原文表述，stride 帧间隔未写

---

## 5. V-JEPA Policy — Building Effective World-Action Models on Predictive Visual Latents

- **arXiv**：2609.37250（cs.CV）｜**提交**：v1 2026-09-29，v2 2026-10-10（读取 v2）；19 pages, 5 figures, 11 tables
- **机构**：Tsinghua；SJTU；Fudan；USTC；Washington Univ. in St. Louis；TeleAI（China Telecom）；Gamma Robotics
- **代码/项目页**：https://github.com/breez3young/VJEPA-Policy ｜ https://breez3young.github.io/vjepa-policy/
- **来源级别**：L3（§2–3 + §5.1 核实）／L2（§5.2 起截断；附录 A–H 未取到）｜**截断位置**：§5.1 末段之后（"Network configurations and benchmark-specific training settings are detailed in Appendix A.2." 处）

### 【输入表征机制】原文引文

**§2 Preliminaries → Problem formulation（输入定义）**
> "At time t, the robot receives RGB observations o_t = (I_t^1, ..., I_t^n) from n camera views, its proprioceptive state q_t ∈ R^{d_s}, and a task instruction ℓ. Here, I_t^v denotes the visual observation from camera v. We learn a policy π(a_t | o_t, q_t, ℓ) to imitate demonstrated action chunks a_t = (a_t, ..., a_{t+H-1}) ∈ R^{H×d_a}, where H is the action horizon and d_a is the action dimension."

**§3.1 Instruction-Conditioned Latent Prediction（视觉与条件注入）**
> "We use the frozen V-JEPA 2.1 encoder to encode the observed context and define the future latent targets (Mur-Labadia et al., 2026). To match the encoder's temporal tubelet size of two, we form a minimal two-frame observed context for each camera view by pairing the current frame with an earlier observation."
> "We independently encode each camera view and concatenate their context representations and future targets across views to form Z_t^c and Z_t^+, respectively."
> "We encode the task instruction ℓ with a frozen T5 encoder (Raffel et al., 2020) and concatenate the resulting text tokens with one state token projected from q_t to form a conditioning sequence. Following the conditioning layout of Wan (Wan et al., 2025), each predictor block applies visual self-attention (Eq. 3), cross-attention to this sequence, and a feed-forward network. Both context tokens and future queries therefore receive instruction and proprioceptive state information."
> "Learnable view embeddings are added to both context tokens and future queries to distinguish their associated cameras."

**§3.2 Coupling Prediction and Action（MoT + 状态注入）**
> "Conditioned on C_φ, the instruction ℓ, and proprioceptive state q_t, a diffusion-transformer-based action expert v_ψ models continuous action chunks using conditional flow matching."
> "Both the predictor and action expert are conditioned on the instruction ℓ and proprioceptive state q_t through cross-attention. Each module projects the T5 instruction embeddings and proprioceptive state into its own hidden space."

**§3.3 Inference（imagine-then-act）**
> "At each decision step, a single predictor pass constructs the future-informed context interface C_φ. We cache this interface and integrate v_ψ from Gaussian noise at τ=0 to a clean predicted action chunk at τ=1 using 10 Euler steps, evaluating only the action expert at each step."

**§5.1 Implementation details（编码器规模）**
> "By default, V-JEPA Policy uses a frozen V-JEPA 2.1 ViT-L visual encoder and a frozen T5-XXL text encoder... The visual encoder, predictor, and action expert contain approximately 0.3B, 0.5B, and 0.1B parameters, respectively, totaling 0.9B parameters, of which 0.6B parameters are trainable."

### 逐条中文指路
- **观测编码格式/维度/token 数**：冻结 V-JEPA 2.1（ViT-L）latent 空间；tubelet=2 → 每相机最小两帧上下文；未来目标为未来 tubelet 位置的归一化 target token；各相机独立编码后跨视角拼接。
- **语言进入方式**：冻结 T5（T5-XXL）编码指令 → 文本 token 与 1 个状态 token 拼接为条件序列 → 每层 cross-attention（「Following the conditioning layout of Wan」）。
- **proprio 注入**：q_t ∈ R^{d_s} 投影为 1 个 state token，与文本 token 拼接后经 cross-attention 同时喂给 predictor 与 action expert；d_s 数值未写。
- **多相机处理**：n 视角 RGB；每视角独立编码 + learnable view embedding 区分相机；上下文与未来 target 跨视角拼接（n 与分辨率未写，附录未取到）。
- **时间组织**：tubelet=2 决定双帧上下文；3D RoPE（temporal/height/width 坐标，坐标局部于各视角）；未来 K 个偏移 δ_1<...<δ_K 的观测作预测目标（K/δ 数值未写）；推理 10 Euler 步、单次 predictor 前向（imagine-then-act）。

### 【why 追问】
1. **为什么不继承完整视频生成模型、只继承 latent 空间？** 原文答（§1）：核心问题句 "Does effective WAM learning require inheriting a complete pretrained visual generative model, or can a predictive visual latent space learned at scale provide a sufficient foundation?"；§5.3 结论（未取到细节）：V-JEPA 预测性 latent 在分布偏移下优于判别式/重建式/视频理解式基础模型（gap 随分布偏移扩大，数值表未取到）。
2. **为什么上下文用最小两帧？** 原文答（§3.1）：「To match the encoder's temporal tubelet size of two, we form a minimal two-frame observed context for each camera view by pairing the current frame with an earlier observation.」
3. **为什么动作专家接收的是 layer-wise context KV 而非未来 latent 预测值？** 原文答（§3.2）：「Through bidirectional interactions with future queries in the predictor, context states at deeper layers incorporate the evolving future-query representations.」→ 未来信息在推理期只以中间 KV 状态形式存在，不需显式未来帧。
4. **为什么 action loss 要回传进 predictor？** 原文答（§3.3）：「The action loss also backpropagates through C_φ into the predictor, so its context states are jointly shaped by the supervision of both the future prediction and the action generation.」

### 【关键数值】（verbatim 自正文）
- V-JEPA 2.1 ViT-L 视觉编码器（0.3B）+ predictor 0.5B + action expert 0.1B = 0.9B 总参（0.6B 可训）
- 冻结 T5-XXL；tubelet=2；推理 10 Euler 步；λ_future = 1（Eq. 9）
- LIBERO 10 epochs；RoboCasa-GR1 50k steps、batch 256
- 摘要结果：LIBERO 97.25%、LIBERO-Plus 79.25%（DROID 预测器预训练后 91.50%）、RoboCasa-GR1 50.92%、真机 45%（预训练后 80%）
- n、d_s、H、K、δ：未取到（附录 A.2 未获取）

### 【一页总结】
问题：WAM 学习是否必须继承完整预训练视觉生成模型，还是预测性 latent 空间已足够？方法：冻结 V-JEPA 2.1 编码器定义视觉 latent 空间；从零联合训练指令条件未来 latent 预测器（V-JEPA 2 predictor 结构 + cross-attention 条件）与 flow-matching 动作专家（MoT 共享注意力）；动作专家用 predictor 的 layer-wise 未来信息上下文 KV 状态做条件。实验：LIBERO 97.25 / LIBERO-Plus 79.25 / RoboCasa-GR1 50.92，与代表性 WAM/VLA 基线相当；DROID 视频-指令对（无动作标签）预训练 predictor 后 LIBERO-Plus 升到 91.50。Takeaway：输入表征 = 冻结预测性 latent + 每视角 view embedding + 「文本 token + 1 个状态 token」cross-attention 条件序列；证明生成模型本体可被 latent 空间替代。

### 【图】
- Figure 1："Overview of V-JEPA Policy. A frozen V-JEPA 2.1 encoder defines the visual latent space. The future predictor jointly processes context tokens and learnable future queries in a single forward pass, providing layer-wise, future-informed context key–value states to a flow-matching action expert. Both modules receive language and proprioceptive conditioning and are jointly trained from scratch with future-latent regression and action flow matching." — https://arxiv.org/html/2609.37250v2（2609.37250v2/V-JEPA-Policy-zy-v1.png）

### 【待核实/原文未写】
- 相机数 n、分辨率、tubelet 时空尺寸、d_s、H、K、δ 偏移：附录 A.2 未取到
- 预训练 predictor 的 DROID 数据规模/过滤：附录 E 未取到
- 视觉基础模型对比（DINOv3/Wan-VAE/Video-LLM 等）细节表：§5.3 截断未取到

---

## 6. JEPA-WAM — Connecting Generated Visual Instructions to World Action Models through JEPA Latent Representations

- **arXiv**：2609.20277（cs.AI）｜**提交**：v1 2026-08-04（仅 v1）；11 pages, 4 figures
- **机构**：HTML 作者块未标注单位（abs 页作者 Liu/Zhu/Su/Zhang/Ma/Huang/Xu）
- **项目页/代码**：未列出（abs 页无 Comments）
- **来源级别**：L3（§3 + 实现细节全文核实）／L2（§5.2 表截断；附录 A–D 未取到）｜**截断位置**：§5.2 Table 1 中部（"PnP carrot | 50.0 | 90.0" 行）

### 【输入表征机制】原文引文

**§3.1 Generated Visual-Instruction Banks（生成式视觉指令库）**
> "Let ℓ denote a language instruction and let G_φ be a pretrained text-to-image generator. For each unique instruction, we construct an offline visual-instruction bank B_ℓ = {g_ℓ^(k) = G_φ(ℓ; ε_k)}_{k=1}^K, where ε_k is a random sampling seed and K is the bank size. Different seeds produce varied appearances, compositions, and spatial layouts under the same language condition."
> "The generator is used only for offline bank construction and is neither fine-tuned nor jointly optimized with the robot policy."
> "During training, each trajectory segment associated with instruction ℓ is paired with an image uniformly sampled from B_ℓ."

**§3.2 Latent visual semantics（JEPA 编码 + 压缩）**
> "Let E_J denote a frozen V-JEPA 2.1 encoder and let P_ψ denote a learnable projector... To accommodate the temporal tubelet size of two used by V-JEPA 2.1, we duplicate each static visual instruction to form a two-frame clip."
> "Z_ℓ^(k) = P_ψ(Pool_{8×8}(E_J([g_ℓ^(k), g_ℓ^(k)]))) ∈ R^{64×d_c}, where Pool_{8×8} spatially compresses the dense V-JEPA features into 64 tokens and P_ψ maps them into the WAM representation dimension d_c. Spatial pooling reduces the conditioning cost while retaining coarse spatial organization. The V-JEPA encoder remains frozen, whereas the projector is optimized jointly with the robot policy."

**§3.2 World-action modeling（条件序列组装）**
> "Let C_ℓ ∈ R^{L×d_c} denote the language conditioning sequence, and let P_s(s) ∈ R^{1×d_c} denote the projected token of the current robot state s. The complete conditioning sequence is C = [C_ℓ; P_s(s); Z_ℓ]. The video expert uses C to model task-guided visual dynamics, while the action expert uses the same sequence to predict the denoising direction of the action trajectory. We otherwise retain the directional masked-attention structure of Fast-WAM."

**§3.2 Implementation details（生成器/编码器/库规模）**
> "we instantiate the image generator G_φ with the compact 0.39B DreamLite model and the visual encoder E_J with the 80M-parameter V-JEPA 2.1 ViT-B/16 checkpoint... The complete inference stack, including image generation and JEPA encoding, runs on a single NVIDIA RTX 5090. DreamLite uses a fixed prompt template across all tasks, with only the task instruction and sampling seed varied. Unless otherwise specified, the main JEPA-WAM model uses a bank of 1,024 generated images per instruction."

**§4.1 Task Suite and Data Distribution（数据规模）**
> "All policies are trained on the same set of 1,370 real-robot demonstrations. Each PnP task contains 150 demonstrations, as do Sort Toys and Wipe Table, whereas each Give Me task contains 80 demonstrations."

### 逐条中文指路
- **观测编码格式/维度/token 数**：本体观测走 Fast-WAM 骨架（WAM backbone，方向性掩码注意力结构原文指 Fast-WAM；backbone 名称未逐字写明）；**新增**的视觉指令目标经 V-JEPA 2.1（ViT-B/16，80M）→ 8×8 池化 → 64 tokens。
- **语言进入方式**：双通道——①原始文本指令的 conditioning 序列 C_ℓ（L tokens）；②文本→DreamLite（0.39B）生成任务完成图（bank K=1024）→ V-JEPA latent 目标 token Z_ℓ（64 tokens）。二者拼接为 C = [C_ℓ; P_s(s); Z_ℓ] 后经 cross-attention 同时喂视频与动作专家。
- **proprio 注入**：当前机器人状态 s 投影为 1 个 token P_s(s)（维度与 d_c 对齐）插入条件序列。
- **多相机处理**：原文未写（已获取范围）。
- **时间组织**：静态视觉指令复制成两帧以匹配 V-JEPA tubelet=2；部署时指令 latent 在指令变更时才重算并跨控制步复用（"avoiding repeated image generation and JEPA encoding for every action query"）。

### 【why 追问】
1. **为什么要用生成图像扩充文本指令？** 原文答（§1）：「rich visual-action trajectories are often paired with sparse and repetitive language annotations, allowing policies to identify tasks from visual context and motion regularities rather than grounding the instruction itself.」即数据结构性失衡导致语言捷径。
2. **为什么生成图要用 V-JEPA 编码而不是直接当目标图？** 原文答（§1）：「Directly conditioning the policy on raw generated pixels would therefore entangle task-relevant semantics with nuisance visual variation.」V-JEPA 2.1 "learns predictive representations in latent space rather than reconstructing raw pixels, while its dense features retain explicit spatial structure and semantic coherence"。
3. **为什么 bank 要 1024 张且训练时随机采样？** 原文答（§3.1）：「This stochastic pairing exposes the policy to multiple visual realizations of the same task semantics without requiring additional robot demonstrations or manually constructed language paraphrases.」（bank 规模消融在 §5.3，未取到）
4. **为什么生成器与 V-JEPA 都选轻量变体？** 原文答（§3.2 Impl.）：「We select these lightweight variants to keep the complete conditioning pipeline on a single device and eliminate inter-GPU communication during closed-loop control.」

### 【关键数值】（verbatim 自正文）
- DreamLite 0.39B（T2I 生成器）；V-JEPA 2.1 ViT-B/16（80M）
- bank K = 1,024 张/指令；8×8 池化 → 64 tokens；目标 latent Z_ℓ ∈ R^{64×d_c}
- 训练数据：1,370 条真机示范（PnP/Sort/Wipe 各 150，Give Me 各 80）
- 摘要结果：ID 87.3% / OOD-S 74.5% / OOD-I 80.9%；分别超过 π0 与 Fast-WAM 至少 10.0 / 27.3 / 14.5 个百分点
- Table 1（部分，截断前）：JEPA-WAM ID：Give bread 100.0 / Give pear 60.0 / Give egg 80.0 / Give carrot 90.0 / PnP car 100.0 / PnP bread 100.0…

### 【一页总结】
问题：纯文本指令条件下 WAM 指令跟随弱（数据语言稀疏重复→视觉捷径）。方法：每条指令离线生成 K=1024 张任务完成图（DreamLite），冻结 V-JEPA 2.1 编码 + 8×8 池化压缩为 64 个目标 token，与文本序列、1 个机器人状态 token 拼接成条件序列 C，经 cross-attention 同时条件视频专家与动作专家（保留 Fast-WAM 方向性掩码）。实验：自建 11 任务真机指令跟随基准，ID/OOD-S/OOD-I = 87.3/74.5/80.9%，全面超过 π0 与 Fast-WAM。Takeaway：输入表征的「语言注入」升级为「文本 + 生成式视觉指令（JEPA latent）」双通道；生成图负责语义多样性，JEPA latent 负责剥离外观干扰。

### 【图】
- Figure 1："Overview of JEPA-WAM. For each language instruction ℓ, we construct a visual-instruction bank B_ℓ containing K images sampled from an instruction-conditioned text-to-image generator. One visual instruction is randomly sampled and encoded by frozen V-JEPA 2.1 and a learnable projector into compact representations Z_ℓ at each training step. These representations condition the video and action experts together with the task instruction and current observations. The right panel shows the self- and cross-attention masks used during training and inference." — https://arxiv.org/html/2609.20277v1（2609.20277v1/Figures/jepa_wam_architecture.png）
- Figure 2："UMAP visualization of frozen V-JEPA 2.1 representations for 704 generated images from the 11 benchmark tasks... The embedding reveals task-level structure among generated visual hypotheses."
- Figure 3："Overview of the real-robot instruction-following benchmark..."（见正文 §3.2 引文后）

### 【待核实/原文未写】
- WAM 主 backbone（文中只说保留 Fast-WAM 掩码结构，backbone 名称未逐字出现于已获取范围）
- C_ℓ 文本编码器类型（T5？）、L、d_c 数值：未取到/原文未写（已获取范围）
- 状态 s 维度、图像分辨率、bank 消融结果（§5.3）：未取到
- 多相机：原文未写（已获取范围）

---

## 7. NavWAM — A Navigation World Action Model for Goal-Conditioned Visual Navigation

- **arXiv**：2606.13494（cs.RO）｜**提交**：v1 2026-06-11（仅 v1）
- **机构**：The University of Tokyo；National Institute of Informatics；AIRoA；ATR
- **项目页**：https://dachii-azm.github.io/navwam/
- **来源级别**：L3（§3–4 方法全文核实）／L2（§5.2 表截断；附录 A/B 未取到）｜**截断位置**：§5.2 Table 2 首行（"Sup. heads | Infer. | ATE | RPE"）

### 【输入表征机制】原文引文

**§3 Problem Setup（输入定义：图像目标）**
> "At time t, the robot observes an RGB image o_t and receives a goal specification g. In this work, we focus on image-goal navigation, where g is a target image captured at the goal location. The policy outputs an executable action chunk a_{t:t+H-1} = (a_t, ..., a_{t+H-1}) over a control horizon H."

**§4 Method → Backbone（主干与 latent frame 原则）**
> "We instantiate NavWAM with Cosmos Predict2, a pretrained diffusion-transformer video world model. Cosmos Predict2 provides the causal VAE, DiT backbone, and text-conditioning interface used to encode and denoise the latent sequence. Following the latent-frame modeling principle of Cosmos Policy, we encode non-visual robot variables as latent frames rather than attaching separate action or value heads."

**§4.1 World-Action Latent Canvas（九帧 latent 画布）**
> "NavWAM represents navigation as denoising a fixed nine-frame latent canvas... The bottom four frames in the canvas are observed and condition the denoising process: a blank frame required by the causal VAE's temporal compression, the current robot state s_t, a goal frame containing the goal specification g, and the current egocentric observation o_t. In our image-goal setting, g is the target image captured at the goal location and is encoded as the goal-image frame. The top five frames are generated as prediction targets: the executable action chunk a_{t:t+H-1}, the future state s_{t+H}, two future egocentric observations o_{t+H-1} and o_{t+H}, and the goal-progress value v_{t+H}."
> "Image frames are encoded as standard video latents through the causal VAE. Non-image variables, including states, action chunks, and the scalar value, are normalized and broadcast over the latent spatial grid. Their predictions are recovered by averaging the denoised entries of the corresponding frame."

**§4.2 Training Objective（模式划分与上采样权重）**
> "We use three modes: a policy mode that conditions on the current state, current observation, and goal, and predicts the action chunk, future state, future observations, and value; a world-model mode that additionally conditions on the action chunk and predicts the resulting future state, future observations, and value; and a value mode that conditions on the future frames and predicts the value. In our main setting, samples are assigned to these modes with a 50/25/25 split."
> "We upweight the action frame relative to the other prediction frames so that the low-dimensional action signal is not dominated by the high-dimensional image reconstruction loss."

**§5.1 Real-world Evaluation Setup（分辨率）**
> "We deploy NavWAM and two baselines, NWM and OmniVLA, on a Diablo mobile robot equipped with an egocentric RGB camera. All methods receive 224×224 RGB observations."

### 逐条中文指路
- **观测编码格式/维度/token 数**：Cosmos Predict2 causal VAE latent（latent 维度/token 数未写）；当前观测与目标图各占一帧 latent frame；图像目标 g 即目标位置拍摄的图像。
- **语言进入方式**：本文为 image-goal 导航，**无语言指令**；原文明示语言目标可经 goal frame 或 text-conditioning 接口扩展但未实现（"Although the same canvas could in principle be extended to language- or object-specified goals through the goal frame or text-conditioning interface, we focus on image-goal navigation in this work."）。
- **proprio 注入**：机器人状态 s_t 归一化后 broadcast 到 latent 空间网格、作为 4 个「观测帧」之一；预测的未来状态 s_{t+H} 亦为 latent frame（s 的定义在附录 A.3，未取到）。
- **多相机处理**：真机为单目 egocentric RGB（224×224）；多相机未提及（已获取范围）。
- **时间组织**：固定 9 帧 latent canvas（4 观测 + 5 生成）；未来观测取 t+H−1 与 t+H 两帧；动作 chunk 整段放一帧；训练三模式 50/25/25 混合（policy / world-model / value）。

### 【why 追问】
1. **为什么把动作/状态/价值编码成 latent frame 而不是独立头？** 原文答（§4 Backbone）：「This preserves the pretrained video-model interface while using a common latent representation for actions, states, values, and future images.」
2. **为什么动作帧要上加权？** 原文答（§4.2）：「so that the low-dimensional action signal is not dominated by the high-dimensional image reconstruction loss.」
3. **为什么三种模式 50/25/25 混合？** 原文答（§4.2）：「A single set of weights therefore learns action generation, action-conditioned future prediction, and value estimation.」——比例选取依据未写。
4. **为什么未来观测预测两帧（t+H−1, t+H）？** 原文未写明理由（附录 A.1 有 frame layout，未取到）。

### 【关键数值】（verbatim 自正文）
- 9 帧 latent canvas：4 观测帧（空白帧 + 状态 + 目标 + 当前观测）+ 5 生成帧（动作 chunk + 未来状态 + 未来观测×2 + 价值）
- 训练模式比例 50/25/25；真机观测 224×224 RGB
- 2B 参数视频 backbone（§1：「using a 2B-parameter video backbone」）
- Table 1（部分）：NavWAM ATE 0.324 / RPE 0.099；NavWAM w/ FT 0.192 / 0.070；NWM 0.453 / 0.107（h 未在已获取范围注明）
- H（控制时域）、latent 分辨率：未取到（附录 B）

### 【一页总结】
问题：导航世界模型只是预测模块，需要外部 CEM 规划器才能闭环。方法：把未来观测、目标进度价值、动作 chunk 放进同一 9 帧 latent canvas 联合去噪（Cosmos Predict2 causal VAE + DiT），非视觉量归一化 broadcast 到 latent 网格；三模式训练（50/25/25）共享一组权重；推理默认 policy 模式免搜索。实验：离线 ATE/RPE 优于 NWM 与 Cosmos Predict2 规划式基线，真机 24 集闭环图像目标导航。Takeaway：输入表征 = 「一切皆 latent frame」——观测、目标图、状态、动作、价值统一为 latent 帧；与操纵 WAM 的 token 拼接路线形成对照。

### 【图】
- Figure 1："Prior navigation world models predict future views for candidate actions and rely on external planning. NavWAM instead predicts future egocentric views, goal-progress values, and executable action chunks within one policy representation, turning visual foresight into a closed-loop navigation policy." — https://arxiv.org/html/2606.13494v1（2606.13494v1/teaser_navwam.png）
- Figure 2："NavWAM overview."（2606.13494v1/overview6.png）

### 【待核实/原文未写】
- 机器人状态 s_t 具体组成、latent 分辨率/token 数、H 值：附录 A.3/B 未取到
- 9 帧 canvas 帧序布局：附录 A.1 未取到
- 多相机：原文未写（已获取范围）
- 语言目标扩展：原文明示未实现

---

## 8. LaWAM — Latent World Action Models for Efficient Dynamics-Aware Robot Policies

- **arXiv**：2606.15768（cs.RO）｜**提交**：v1 2026-06-14（仅 v1）
- **机构**：Tsinghua；Jilin Univ.；Nankai Univ.；Peking Univ.；HIT；Zhongguancun Academy；Striding.AI；Infinigence AI
- **项目页/代码**：未列出
- **来源级别**：L3（§1–3 方法全文核实）／L2（§4.1 表截断；附录 C 未取到）｜**截断位置**：§4.1 Table 1 中部（π0.5 行 "3.5B | 220 |" 处）

### 【输入表征机制】原文引文

**§3.1 Problem Formulation（冻结视觉特征空间）**
> "Let o denote the current observation, l the task instruction, and a_{1:T} an action chunk over a fixed physical horizon τ. A standard VLA directly models p(a_{1:T} | o, l)... LaWAM keeps the future-conditioned structure but represents the future in a frozen visual feature space. Let f_ψ be the frozen encoder, and define u = f_ψ(o) and u_T = f_ψ(o_T)."
> "At inference time, the future feature u_T is unavailable, so the policy must predict the latent action before LaWM can predict the subgoal. LaWAM therefore factors action generation as p(a_{1:T}, û_T, ẑ | o, l) = p_θ(ẑ | o, l) [Policy Prior] p_ω(û_T | u, ẑ) [LaWM] p_η(a_{1:T} | o, l, u, û_T) [Action Expert]."

**§3.2 Latent World Model（latent action 与状态辅助监督）**
> "Each training example contains a current observation o and a horizon observation o_T sampled after the physical interval τ. After encoding them as (u, u_T), the inverse-dynamics encoder infers a latent action z ∼ q_φ(z | u, u_T). The decoder then uses (u, z) to predict the horizon feature, ũ_T = LaWM_ω(u, z)."
> "We add one auxiliary signal during this pretraining stage. A lightweight predictor g maps the current end-effector state s and latent action z to the horizon state s_T, encouraging z to encode embodied motion rather than only visual appearance change. After pretraining, we discard this auxiliary head and keep the decoder as LaWM."
> "Because both u_T and s_T are defined by the same physical interval τ, the learned subgoal corresponds to a consistent amount of elapsed motion rather than a dataset-specific frame offset."

**§1 / §3.3（视觉编码器与 Alternate-DiT）**
> "We obtain LaWM by revisiting latent action models (LAMs) in the latent space of frozen visual encoders, such as DINOv3."
> "The action expert uses this predicted future through an Alternate-DiT design. One stream carries the semantic context from the VLM backbone, while the other carries the latent dynamics context formed by (u, û_T)."
> "For mixed-frequency robot data, LaWAM keeps each dataset at its native control frequency and uses physical-time encoding, as detailed in Appendix C.3."

**§4（预训练数据规模）**
> "LaWM is pretrained on roughly 3,000 hours of robot videos and 1,500 hours of egocentric human videos, while LaWAM policy integration uses only robot trajectories with language instructions; human videos contribute only through the dynamics prior learned by LaWM."
> "LaWAM uses a 230M LaWM in place of the 5B WAN backbone used by such models, reducing world-modeling parameters by about 95%. This keeps the full LaWAM model size at 2.3B and yields 187 ms latency, up to 24× faster than pixel-space WAMs."

### 逐条中文指路
- **观测编码格式/维度/token 数**：冻结视觉基础模型（如 DINOv3）特征空间，u = f_ψ(o)、u_T = f_ψ(o_T)；latent action z 经 IDM 编码器推断（维度未写）；预测 subgoal û_T 为特征而非像素。
- **语言进入方式**：指令 l 作为 policy prior p_θ(ẑ | o, l) 与动作专家的条件；具体文本编码器（VLM backbone 种类）在附录 C.2，未取到。
- **proprio 注入**：末端执行器状态 s 仅用于 LaWM 预训练期的辅助头 g(s, z) → s_T（训练后丢弃）；策略测试期是否注入 proprio 原文未写（已获取范围）。
- **多相机处理**：原文未写（已获取范围）。
- **时间组织**：物理时距 τ 对齐（u_T 与 s_T 同一 τ → 子目标对应一致流逝的运动量）；混合频率数据保持原生控制频率 + physical-time encoding（附录 C.3 未取到）。

### 【why 追问】
1. **为什么未来要用冻结特征空间而不是像素？** 原文答（§1）：「many methods predict future images or videos, allocating substantial modeling capacity to pixel-level synthesis rather than compact action-relevant dynamics. Second, iterative future generation introduces considerable inference latency... Third, effective future prediction for manipulation should expose the state change relevant to the next action chunk, rather than merely generating visually plausible task-consistent futures.」
2. **为什么 LAM 的解码器要保留下来而不是丢弃？** 原文答（§1）：「Prior latent-action-based VLAs mainly use this framework to learn embodiment-agnostic action representations, treating the decoder only as an auxiliary training component and discarding it after pretraining. In contrast, we repurpose the decoder as the core predictive module of LaWM, using latent actions to generate future observation features that serve as latent visual subgoals for action generation.」
3. **为什么要加末端状态辅助头？** 原文答（§3.2）：「encouraging z to encode embodied motion rather than only visual appearance change.」
4. **为什么用 Alternate-DiT 双流？** 原文答（§3.3）：「Alternating between these streams lets the expert combine task intent with the predicted scene evolution when denoising the action chunk.」

### 【关键数值】（verbatim 自正文）
- LaWM 230M；LaWAM 总计 2.3B；187 ms/action-chunk；最快比像素级 WAM 快 24×
- 预训练：≈3,000 小时机器人视频 + 1,500 小时自我中心人类视频
- 摘要：LIBERO 98.6% SR；RoboTwin 91.22% SR
- 参照（§1）：LingBot-VA 单次推理 4482 ms；π0.5 220 ms
- latent action 维度、τ、DINOv3 特征维度/token 数：未取到（附录 C）

### 【一页总结】
问题：WAM 需要前视，但像素级视频生成冗余且慢（LingBot-VA 4482ms）。方法：两阶段——先训 latent action model（DINOv3 等冻结特征空间，IDM 编码器推断 z，解码器保留为 LaWM），再训 LaWAM（policy prior 预测 ẑ → LaWM 解码 latent 视觉子目标 û_T → Alternate-DiT 动作专家在语义流与动态流之间交替去噪）。实验：LIBERO 98.6%、RoboTwin 91.22%、真机三任务，187ms 低延迟。Takeaway：输入表征以「预测特征级子目标」替代「生成像素级未来」，世界模型参数量降 95%；proprio 仅作 LaWM 训练辅助信号。

### 【图】
- Figure 2："Overview of LaWAM. Stage 1 learns a latent-action-conditioned world model from visual transitions: an inverse-dynamics encoder infers latent actions, and the decoder is retained as LaWM to predict future observation features. Stage 2 integrates LaWM into a VLA policy: latent-action distillation teaches the policy to drive LaWM, whose predicted latent visual subgoal is passed to an Alternate-DiT action expert for subgoal-conditioned action generation." — https://arxiv.org/html/2606.15768v1（2606.15768v1/LaWAM_overview.png）
- Figure 3："Subgoal-guided chunk execution. The top row shows observations within one executed LIBERO chunk together with the predicted latent subgoal; the bottom row overlays subgoal-derived robot-arm heatmaps, illustrating how the executed motion approaches the predicted subgoal region."

### 【待核实/原文未写】
- VLM backbone（Qwen?）、文本编码器、latent action 维度、特征 token 数、τ 数值、分辨率：附录 C.1/C.2/C.3 未取到
- 测试期 proprio 是否注入：原文未写（已获取范围）
- 多相机：原文未写（已获取范围）

---

## 9. PALM — Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation

- **arXiv**：2601.07060（cs.RO）｜**提交**：v1 2026-01-11，v2 2026-04-04（读取 v2）；CVPR 2026
- **机构**：Univ. of Pennsylvania；UIUC；NTU；Univ. of Oxford；MIT
- **项目页**：https://plan-lab.github.io/palm
- **来源级别**：L3（§3.1–3.2 核实）／L2（§3.3 起截断；附录 A 未取到）｜**截断位置**：§3.3 Dynamic Prediction 段中（"The selected trajectories are then rasterized at t" 处）
- **注**：PALM 是 VLA 框架（非严格 WAM，摘要自述 "a VLA framework"），但其 affordance foresight 结构与 WAM 未来预测同族，按清单收录。

### 【输入表征机制】原文引文

**§3.2 Multi-Modal Encoders（三模态编码）**
> "PALM processes three synchronized inputs: a language instruction l, an image observation o_t, and a robot state s_t. Instructions are embedded using a CLIP text encoder, observations are encoded with a Masked Autoencoder and downsampled by a Perceiver Resampler to retain task-relevant visual tokens. The robot state is projected through a lightweight MLP. The resulting tokens are concatenated into a single sequence with time index t to form a unified multimodal sequence passed to the transformer backbone."

**§3.2 Backbone and Learnable Queries（查询集）**
> "The backbone follows a GPT-2-style transformer to integrate the token sequence with causal and cross-modal attention. On top of this backbone, PALM introduces two learnable query sets that instantiate the affordance and progress-aware policy: ♢ Fine-grained affordance queries comprise four subqueries <Global>, <Local>, <Spatial>, and <Dynamic>, that attend over language, vision, and state tokens to extract task-relevant representation at complementary scales and produce the affordance-centric latent F̂_{t+n} = f_aff(l, o_t, s_t) ∈ F. ♢ Action–progress queries pool control-relevant context and condition on F̂_{t+n} to support progress-aware inverse-dynamics decoding..."

**§3.2 Decoders（DiT 头）**
> "The action head is a denoising diffusion transformer that conditions on the action–progress queries and the affordance latent to generate a horizon-n trajectory: (â_{t:t+n-1}, p̂_{t:t+n-1}) = DiT(l, o_t, s_t, F̂_{t+n}), where â is the predicted action sequence and p̂ the predicted sequence of scalar progress values at each step."

**§3.3 各 affordance 头的监督来源（Global 为例，verbatim）**
> "Targets are obtained by first resolving the referent via Grounding DINO and then segmenting the instance with SAM to obtain a binary mask on the observation lattice."（Local 用 GLOVER++ 接触热图；Spatial 用 SpatialVLM + RoboPoint；Dynamic 用 CoTracker 网格跟踪）

### 逐条中文指路
- **观测编码格式/维度/token 数**：MAE 编码 + Perceiver Resampler 降采样（token 数未写）；单一图像观测 o_t。
- **语言进入方式**：CLIP 文本编码器（冻结与否未写明于已获取范围）。
- **proprio 注入**：机器人状态 s_t 经轻量 MLP 投影为 token 进入统一序列（维度未写）。
- **多相机处理**：原文未写（已获取范围）。
- **时间组织**：时间索引 t 的 token 序列 + 因果/跨模态注意力；未来偏移 t+n 的 affordance foresight F̂_{t+n} 为条件；动作与连续进度 p ∈ [0,1] 联合解码 horizon n 序列。

### 【why 追问】
1. **为什么预测 affordance 而非未来像素？** 原文答（§1 图 1 注释）："In contrast to vanilla VLAs that directly map inputs to actions or to predictive methods that forecast dense future images, PALM introduces learnable queries to forecast a structured set of future affordances."（另见 §1：结构化 affordance 线索与显式状态跟踪缺失导致长程失败模式）。
2. **为什么 progress 要连续标量且与动作联合解码？** 原文答（§3.1）：「the policy jointly decodes an action a_t alongside a scalar p_t that encodes progress within the current subtask and serves as a temporal regularizer that mitigates phase ambiguity and enables smooth subtask transitions without relying on separate planners or hierarchical controllers.」
3. **为什么 affordance 子查询只 attend 上下文 token 而不互相 attend？** 原文答（§3.2 Structured Attention，图 2 注释）："Affordance subqueries attend only to shared context tokens to stay disentangled, while both query types use causal attention to preserve temporal consistency."
4. **为什么监督目标要用 Grounding DINO/SAM/SpatialVLM/CoTracker 等外部模型生成？** 原文答（§3.3）：「Ground-truth targets are obtained by...」——即用现成工具合成真值标签；选型理由（鲁棒性/成本）原文未写。

### 【关键数值】（verbatim 自正文）
- 4 个 affordance 子查询 <Global>/<Local>/<Spatial>/<Dynamic>；progress 空间 P ⊂ [0,1]
- 摘要：LIBERO-LONG 91.8% SR；CALVIN ABC→D 平均长度 +12.5%；真机 2× 提升
- 预训练数据（图 1 注释）：BridgeDataV2、DROID、EPIC-KITCHENS、RoboCerebra
- MAE/Perceiver token 数、MLP 状态维度、n 时域：未取到（附录 A）

### 【一页总结】
问题：长程任务中 VLA 缺「该碰哪个物体、放哪、动多少」的结构化交互线索与子任务进度跟踪。方法：CLIP 文本 + MAE/Perceiver 视觉 + MLP 状态三模态 token 进 GPT-2 式 transformer；4 个 affordance 子查询预测未来 t+n 的全局目标掩码/接触热图/放置点集/动态掩码（外部模型合成监督）；DiT 头在 affordance latent 条件下联合解码动作与连续进度。实验：LIBERO-LONG 91.8%、CALVIN +12.5%、真机三泛化设置 2× 提升。Takeaway：输入表征之外新增「结构化未来 affordance」条件支路，用外部视觉模型（Grounding DINO/SAM/SpatialVLM/CoTracker）构建监督，proprio 以 MLP token 显式注入。

### 【图】
- Figure 1："In contrast to vanilla VLAs that directly map inputs to actions or to predictive methods that forecast dense future images, PALM introduces learnable queries to forecast a structured set of future affordances. Conditioned on these affordances, a diffusion-based policy jointly decodes the robot's action and a continuous progress value..." — https://arxiv.org/html/2601.07060v2（2601.07060v2/main_graph.png）
- Figure 2："PALM Overview. (a) Model Architecture: Given a language instruction l, observation o_t, and robot state s_t, PALM encodes each modality using frozen encoders to obtain text, visual, and state tokens... (b) Structured Attention: Affordance subqueries attend only to shared context tokens to stay disentangled, while both query types use causal attention to preserve temporal consistency."（2601.07060v2/pipeline.png）

### 【待核实/原文未写】
- 图像分辨率、Perceiver token 数、状态维度、动作维度、horizon n：附录 A.1/A.2 未取到
- Dynamic affordance 损失函数（截断处之后）：未取到
- 多相机：原文未写（已获取范围）
- CLIP/MAE 是否冻结：图 2 注释写 "frozen encoders"，正文 §3.2 未逐字重复——以图注为准（原文两处略有差异，标注待核实）

---

## 10. UniPi — Learning Universal Policies via Text-Guided Video Generation

- **arXiv**：2302.00111（cs.AI）｜**提交**：v1 2023-01-31，v3 2023-11-20（读取 v3）；NeurIPS 2023
- **机构**：MIT；Google DeepMind；UC Berkeley；Georgia Tech；Univ. of Alberta
- **项目页**：https://universal-policy.github.io/
- **来源级别**：L3（§2–3 全文核实）／L2（§4.2 起截断；附录 A.1/A.2 未取到）｜**截断位置**：§4.2 前表格头（"| Frame | Frame | Temporal | Condition |" 处）

### 【输入表征机制】原文引文

**§2.2 Unified Predictive Decision Process（输入抽象）**
> "we define a UPDP to be a tuple G = ⟨X, C, H, ρ⟩, where X denotes the observation space of images, C denotes the space of textual task descriptions, H ∈ N is a finite horizon length, and ρ(·|x_0, c) : X × C → Δ(X^H) is a conditional video generator. That is, ρ(·|x_0, c) is a conditional distribution over H-step image sequences determined by the first frame x_0 and the task description c."

**§3.1 Conditional Video Synthesis（首帧条件方式）**
> "we found it more effective to explicitly train a constrained video synthesis model by providing the first frame of each video as explicit conditioning context during training."
> "we provide, as additional context, the observed image when denoising each frame in the synthesized video. In particular, we re-purpose a temporal super-resolution video diffusion architecture, and provide as context the conditioned visual observation tiled across time, as the opposed to a low temporal-resolution video for denoising at each timestep. In this model, we directly concatenate each intermediate noisy frame with the conditioned observed image across sampling steps, which serves as a strong signal to maintain the underlying environment state across time."
> "To train the text-conditioned video generation model, we utilize the video diffusion algorithm in [19], where pretrained language features from T5 are encoded."

**§2.3 Diffusion Models for UPDP（CFG）**
> "We leverage classifier-free guidance and use ŝ(τ_k, k | c, x_0) = (1+ω)s(τ_k, k | c, x_0) − ω s(τ_k, k) as the denoiser in the reverse process for sampling, where ω controls the strength of the text and first-frame conditioning."

**§3.2 Inverse Dynamics（动作提取）**
> "We train a small model to estimate actions given input images. The training of the inverse dynamics is independent from the planner and can be done on a separate, smaller and potentially suboptimal dataset generated by a simulator."
> "For computational efficiency, we use an open-loop controller in all our experiments in this paper."

### 逐条中文指路
- **观测编码格式/维度/token 数**：像素级视频扩散（video diffusion 算法，Imagen Video 系）；当前帧 x_0 作为首帧条件 + tiled 上下文（每个中间噪声帧与观测图逐时间步拼接）；分辨率/帧数在附录 A.1 未取到。
- **语言进入方式**：预训练 T5 语言特征编码任务描述 c，作为文本条件（细节在附录 A.1）。
- **proprio 注入**：原文未写——UniPi 规划器只有首帧图像 + 文本；逆动力学模型输入为图像序列（未见本体状态输入，已获取范围）。
- **多相机处理**：原文未写（已获取范围）。
- **时间组织**：H 步图像轨迹；分层规划（先稀疏「抽象」视频再时间超分）；视频生成与动作提取分离（open-loop 执行）。

### 【why 追问】
1. **为什么固定首帧靠训练期显式条件而非测试期采样修改？** 原文答（§3.1）：「However, we found that this performed poorly and led to subsequent frames in the video plan to deviate significantly from the original observed image. Instead, we found it more effective to explicitly train a constrained video synthesis model...」
2. **为什么要把观测图 tile 到每个时间步？** 原文答（§3.1）：「which serves as a strong signal to maintain the underlying environment state across time.」（环境一致性）
3. **为什么规划与动作提取分离？** 原文答（§2.2）：「This design choice isolates planning decisions from action-specific mechanisms, allowing the planner to be environment and agent agnostic.」
4. **为什么要分层规划？** 原文答（§3.1）：「directly generating a set of actions to reach a goal state quickly becomes intractable due to the exponential blow-up of the underlying search space.」

### 【关键数值】（verbatim 自正文）
- 组合规划环境：「We train the video model on 200k example videos of generated language instructions in the train set.」（§4.1）；语言划分 70% 训练 / 30% 测试
- Table 1（组合环境任务完成率）：UniPi Seen：Place 59.1±2.5 / Relation 53.2±2.0；Novel：Place 60.1±3.9 / Relation 46.1±3.0（最优基线 Image+TT Novel Relation 9.1±2.5）
- 分辨率/帧数/CFG ω 数值：未取到（附录 A.1）

### 【一页总结】
问题：不同环境状态/动作空间各异，如何构建跨环境通用策略并组合泛化到新语言目标。方法：UPDP 抽象——文本条件视频生成器 ρ(x_{1:H} | x_0, c) 当规划器（首帧显式条件 + 观测图时间维 tile + 分层时间超分），生成的图像轨迹交给小逆动力学模型提取动作（open-loop）。实验：组合摆放任务 Seen/Novel 全面碾压 transformer BC/TT/Diffuser；多环境 CLIPort 迁移与真机生成（真实机器人视频合成）成立。Takeaway：输入表征最简：首帧 + T5 文本；环境一致性靠「观测图 tile 进每个去噪步」的输入构造实现；无 proprio、无 latent 编码。

### 【图】
- Figure 1："Text-Conditional Video Generation as Universal Policies. Text-conditional video generations enables us to train general purpose policies on wide sources of data (simulated, real robots and YouTube) which may be applied to downstream multi-task settings requiring combinatorical language generalization, long-horizon planning, or internet-scale knowledge." — https://arxiv.org/html/2302.00111v3（2302.00111v3/overview.png）
- Figure 2："Given an input observation and text instruction, we plan a set of images representing agent behavior. Images are converted to actions using an inverse dynamics model."（2302.00111v3/method_overview.png）

### 【待核实/原文未写】
- 视频分辨率/帧数/帧率、T5 版本、CFG ω、时间超分层数：附录 A.1 未取到
- proprio：原文未写（已获取范围）
- 多相机：原文未写（已获取范围）

---

# 汇总：输入表征主流选项对比表（10 篇）

| 论文 | 编码路线 | token 形式 | 语言注入 | proprio 处理 | 多相机 | 时间组织 |
|---|---|---|---|---|---|---|
| WorldVLA 2506.21539 | 离散 VQ 自回归（Chameleon LLM） | 图像 VQ-GAN 离散 token（256/1024）、动作 7 离散 token、文本 BPE；三模态共享词表 | BPE 离散 token + 固定文本模板进序列 | 未写（动作仅作输出） | 未写 | 历史 M=2 帧；chunk K=5/10；世界模型 N=1；并行 action 掩码 |
| GR-1 2312.13139 | 连续表征 + 因果 GPT | CLS + Perceiver 降采样 patch tokens；[OBS]/[ACT] 可学习 token | CLIP 文本编码器（冻结），每时间步重复注入 | 6D 位姿 + 夹爪二值，线性层序列注入 | 未写 | 历史 h 帧 + 相对 timestep embedding |
| Motus 2512.13030 | 连续 latent + 整流流 MoT（Wan2.2-5B + Qwen3-VL-2B） | 视频 VAE latent + 4×512 latent action tokens（光流 DC-AE→14 维） | 理解专家（Qwen3-VL-2B）last-layer tokens；细节未写 | p_t 为策略显式输入（实现未写） | 未写 | 视频帧率=动作 1/6（稀疏化） |
| Fast-WAM 2603.16666 | 连续 latent + flow matching MoT（Wan2.2-5B DiT） | 首帧 clean latent 锚点 + 训练期未来 noisy latent + 动作 tokens | Wan2.2 内置 T5，cross-attention 进所有 token 组 | 未写（已获取范围） | 多相机拼单图入 VAE（分辨率未写） | downsample 4×→9 帧/chunk；h=32；推理单前向 |
| V-JEPA Policy 2609.37250 | 冻结预测性 latent（V-JEPA 2.1）+ flow matching MoT | V-JEPA 上下文 latent tokens + 可学习未来 query | 冻结 T5-XXL，文本 tokens 与状态 token 拼接后 cross-attention（Wan 布局） | q_t∈R^{d_s} 投影为 1 token 与文本拼接 | n 视角独立编码 + learnable view embedding 拼接 | tubelet=2 双帧上下文；3D RoPE；未来 δ 偏移目标；推理 10 Euler 步 |
| JEPA-WAM 2609.20277 | Fast-WAM 骨架 + 生成式视觉指令（JEPA latent 目标） | 观测 latent + 64 个目标 token（V-JEPA 2.1 ViT-B/16 + 8×8 池化） | 双通道：文本序列 C_ℓ + T2I（DreamLite 0.39B，bank K=1024）生成图→JEPA latent | 状态 s 投影 1 token P_s(s) | 未写 | 静态图复制两帧匹配 tubelet=2；指令 latent 跨步复用 |
| NavWAM 2606.13494 | 连续 latent + diffusion DiT（Cosmos Predict2 causal VAE） | 9 帧 latent canvas（观测/目标/状态/动作/价值各为 latent frame） | 无（image-goal）；语言扩展未实现 | 状态归一化 broadcast 到 latent 网格，占 1 观测帧 | 单目 224×224 | 固定 9 帧 canvas；未来观测 2 帧；三模式训练 50/25/25 |
| LaWAM 2606.15768 | 冻结特征 latent（DINOv3 等）+ flow matching Alternate-DiT | 特征向量 u/u_T + latent action z → 预测特征子目标 û_T（非像素） | 指令 l 条件 policy prior 与动作专家（编码器未写） | 末端状态 s 仅作 LaWM 训练辅助头（训练后丢弃） | 未写 | 物理时距 τ 对齐；混合频率数据原生频率 + physical-time encoding |
| PALM 2601.07060 | 连续表征 + GPT-2 式 transformer + DiT 头 | MAE + Perceiver 视觉 tokens + CLIP 文本 tokens + MLP 状态 tokens + 4 类 affordance queries | CLIP 文本编码器 | s_t 经轻量 MLP 投影为 token | 未写 | 因果/跨模态注意力；未来 t+n affordance foresight；动作+进度联合解码 horizon n |
| UniPi 2302.00111 | 像素级视频扩散（规划器）+ 逆动力学模型 | 像素帧（首帧条件 + 观测图时间维 tile） | 预训练 T5 语言特征 | 未写（已获取范围） | 未写 | H 步图像轨迹；分层时间超分；open-loop 执行 |

## 趋势与争议点（仅基于原文）

**趋势 1｜像素 → latent 迁移**：从 UniPi（像素视频规划）与 GR-1（像素 MSE 预测）到 Fast-WAM / V-JEPA Policy / JEPA-WAM / LaWAM / Motus，未来预测目标从像素帧迁到 VAE/JEPA/DINO 特征 latent。Fast-WAM §1 明确把价值归于「improving world representations during training」；V-JEPA Policy 追问「whether a predictive visual latent space ... can instead provide a sufficient foundation for effective WAM learning without inheriting a complete pretrained visual generative model」。

**趋势 2｜语言注入收敛到 T5/Wan 式 cross-attention 与「文本+状态 token 拼接条件序列」**：Fast-WAM（T5 cross-attention）、V-JEPA Policy（T5 + state token，明确「Following the conditioning layout of Wan」）、JEPA-WAM（C = [C_ℓ; P_s(s); Z_ℓ]）、UniPi（T5）四篇同源；WorldVLA（离散 BPE）与 GR-1/PALM（CLIP）是另两支。

**趋势 3｜proprio 三档**：显式 token 注入（GR-1 线性层、V-JEPA Policy 1 个状态 token、JEPA-WAM 1 token、PALM MLP、Motus 数据集要素；Fast-WAM 代码侧亦属此档——1 token 追加文本条件序列，见其节代码补充）／latent frame 化（NavWAM broadcast 到 latent 网格）／仅作训练辅助或原文未写（LaWAM 辅助头、WorldVLA、UniPi 原文未写）。

**趋势 4｜多相机是普遍空白**：10 篇中仅 Fast-WAM（拼单图）与 V-JEPA Policy（分视角编码 + view embedding）给出多相机方案，其余均未写；Fast-WAM 未给出相机分辨率细节。

**争议 A｜测试期是否需要显式未来想象**：Fast-WAM 断言不需要（去掉未来分支仍 91.8 RoboTwin，而去掉视频共训掉到 83.8）；V-JEPA Policy 持「latent 级 imagine-then-act」（预测器单前向出未来信息 KV 再条件动作）折中；JEPA-WAM/LaWAM 进一步把「未来」压缩成目标 token / 特征子目标。

**争议 B｜离散 vs 连续动作表征**：WorldVLA §4.1 原文：「Discrete models inherently exhibit inferior performance, as the tokenization process of actions may lead to information loss.」——但同一段 Table 2 中离散 WorldVLA(512) 81.8 又超过部分连续基线；离散派依赖共享词表统一多模态。

**争议 C｜语言指令是否被视觉捷径劫持**：JEPA-WAM §1 指出现有 WAM「limited instruction-following ability when conditioned solely on text instructions」，归因于「sparse and repetitive language annotations」；其解法是把语言升级为生成式视觉指令（文本→T2I 图→JEPA latent），即「语言不必直接进模型，可先转成视觉语义」。

**争议 D｜视频 token 与动作 token 的注意力结构**：WorldVLA（action 屏蔽先前 action）、Fast-WAM（action 禁止看未来视频、首帧不看任何 token）、JEPA-WAM（沿用 Fast-WAM 方向性掩码）三家都对掩码做了专门设计，共识是「未来信息泄漏进动作分支有害」，但掩码方案互不兼容、无统一基准。

<!-- QA_LOG -->
（占位：精读过程中产生的问答记录后续追加于此）
