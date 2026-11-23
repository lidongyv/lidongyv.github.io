# HumanoidWM 精读笔记

> 唯一数据源：`auto_docs/papers/txt/HumanoidWM.txt`（arXiv 2506.01182，51,053 字符，12 页）。行号均为该 txt 的行号。引文逐字复制（含原文拼写错误，如 L301 "passing passing"、L443 "seperately"、L456 "token steam"、L661 "very benefitial"，系原文如此）。所有事实性陈述附原句；论文未写处标「原文未写」；解读标【推断】。

## 1. 原文信息

- 标题全称（L1–3）："Humanoid World Models : Open World Foundation Models for Humanoid Robotics"
- 作者（L4）："Qasim Ali * 1 Aditya Sridhar * 1 Shahbuland Matiana 1 Alexander Wong 1 Mohammad Al-Sharman 1"
- 共同一作与单位（L40）："*Equal contribution 1University of Waterloo, Waterloo, Canada."
- 通讯（L41–42）："Correspondence to: Qasim Ali <m45ali@uwaterloo.ca>, Aditya Sridhar <a27sridh@uwaterloo.ca>."
- ID 与版本（L77）："arXiv:2506.01182v2  [cs.RO]  8 Jul 2025"
- 发表状态：原文未写（txt 中无任何会议/期刊接收信息；仅有 arXiv v2）。
- 数据/代码链接：原文未写（全文未给出代码或权重下载地址）。

## 2. Abstract 全文逐字（L5–28）

> "Humanoid robots, with their human-like form,
> are uniquely suited for interacting in environ-
> ments built for people. However, enabling hu-
> manoids to reason, plan, and act in complex open-
> world settings remains a challenge. World mod-
> els, models that predict the future outcome of a
> given action, can support these capabilities by
> serving as a dynamics model in long-horizon
> planning and generating synthetic data for policy
> learning. We introduce Humanoid World Mod-
> els (HWM), a family of lightweight, open-source
> models that forecast future egocentric video con-
> ditioned on humanoid control tokens. We train
> two types of generative models, Masked Trans-
> formers and Flow-Matching, on 100 hours of
> humanoid demonstrations. Additionally, we ex-
> plore architectural variants with different atten-
> tion mechanisms and parameter-sharing strategies.
> Our parameter-sharing techniques reduce model
> size by 33–53% with minimal impact on perfor-
> mance or visual fidelity. HWMs are designed to
> be trained and deployed in practical academic and
> small-lab settings, such as 1–2 GPUs."

## 3. 问题与动机

**人形机器人为何需要推理能力（L30–38）：**

> "Autonomous humanoid robots have the potential to transform both industry and daily life by automating tasks that are dull, dangerous, or physically demanding (Goswami & Vadakkepat, 2019). Their human-like morphology allows them to operate in spaces built for people, interact naturally with humans, and easily learn from teleoperated demonstrations. However, in order to navigate the complexity and unpredictability of real-world environments these agents require sophisticated reasoning capabilities."

**为何不能靠大模型直接解决（L43–49）：**

> "While large multimodal models (Zhang et al., 2024; Black et al., 2024) show some promise in reasoning tasks, they often lack the accuracy, reliability, and robustness needed for embodied AI in open-world settings (Tong et al., 2024; Pfeifer & Iida, 2004). As a result, they struggle to meet the demands of embodied agents that must act safely and effectively in dynamic, unstructured environments"

**世界模型的动机（L53–64）：**

> "One avenue for improving humanoid intelligence and control is through World Models—predictive models trained to forecast future outcomes based on past observations and actions (Ha & Schmidhuber, 2018). In our setting, these function as action-conditioned video generators: they predict future visual states as sequences of frames, enabling agents to simulate the consequences of candidate actions (Du et al., 2023). By simulating the outcomes of actions without trying them in the real world, world models can be used for long-horizon planning by using them as a dynamics model (Yang et al., 2024) and enable more data-efficient policy learning through synthetic rollouts (Yang et al., 2023)."

**研究空白与研究问题（L65–80）：**

> "Despite recent progress in video generation, most models are built for entertainment applications (Liu et al., 2024; Polyak et al., 2024), emphasizing visual appeal over physical, ego-centric plausibility. Many world models remain closed-source, require large-scale compute for training or inference, or are not designed for humanoid robots (Yang et al., 2023; NVIDIA et al., 2025; Zhu et al., 2024). As a result, there is a clear gap: few open-source models are both physically grounded for humanoid robots and lightweight enough to train and deploy on modest academic hardware (e.g., 2–3 GPUs). To address this, we ask: Can we build a physically plausible, humanoid-specific world model that can be trained and deployed on as little as two GPUs?"

**与 VLA 的区别——原文的表述（L139–157）：**

> "Recent approaches include Vision-Language-Action (VLA) models that predict actions from video and language inputs (Black et al., 2024; Kim et al., 2024), and large language or vision-language models used as high-level planners (Liang et al., 2023; Li et al., 2024; Wang et al., 2024). However, these methods often struggle with spatial reasoning (Tong et al., 2024), continuous sensorimotor processing, and require complex prompting strategies (Li et al., 2024)."

> "In contrast, video generation models offer a more grounded alternative for embodied agents. Video captures fine-grained physical and temporal structure that language alone cannot express (Yang et al., 2024). When trained to predict future video frames given past observations and actions, these models serve as World Models—internal simulators that forecast the outcomes of potential action sequences (Ha & Schmidhuber, 2018; Du et al., 2023). This enables agents to plan through imagination, reason counterfactually, and generate synthetic experience for policy training."

即：VLA 是「输入视频+语言 → 输出动作」的直接动作回归；世界模型是「输入过去视频+动作 → 输出未来视频」的动力学模拟，供规划与合成数据使用。原文未对 VLA 做实验对比。

## 4. 方法逐机制

**任务形式化（L284–296）：**

> "The goal is to predict plausible future video frames given a sequence of past video frames and associated actions. Formally, the model predicts a sequence of f future RGB frames vf ∈ Rf ×3×H×W , conditioned on p past frames vp ∈ Rp×3×H×W , p past actions ap ∈ Rz, and f future actions af ∈ Rz. Action vectors include joint angles, velocities, and gripper states of the humanoid. ... we train our generative models in a V AE's compressed latent space: vp and vf are encoded into latent representations Lp and Lf . Masked-HWM uses a VQ-V AE to quantize latents into tokens from a finite vocabulary of size s, while Flow-HWM uses a continuous V AE."

**数据来源与规模（L525–534）：**

> "We train our models on the 1xGPT dataset (1X Technologies, 2024), which contains 100 hours of egocentric video captured from the Humanoid EVE Android executing various tasks. Video frames are recorded at 30 Hz, with each frame paired with a corresponding action vector a ∈ R25 representing movement velocities, hand closure states, and pitch-yaw-roll angles of the joints (wrist, knee, elbow, shoulder, neck, and hip). We train both models to generate f = 8 future frames conditioned on p = 9 past frames at H = 256× W = 256resolution."

**动作表示**：R25 向量（速度、手爪开合、各关节 pitch-yaw-roll），动作序列经 MLP 嵌入到与视频 token 同维的 h 维空间（L347–350）。

**Masked-HWM（掩码视频建模）**（L298–342）：VQ-VAE 量化后，过去+未来 latent 沿时间拼接；训练时对 L 注入 Copilot-4D 风格随机 token 替换噪声（比率 ~U(0, ρmax)），并对未来 latent 按帧阈值掩码；损失为掩码位置交叉熵：

> "The training objective is to minimize the cross-entropy loss between the predicted tokens ˆLf at the masked locations M and the true tokens at those same locations"（L319–321）

推理为逐 latent 帧生成、帧内 K 步并行细化（L331–342）。训练配置（L556–573）："We tokenize all video frames using the NVIDIA Cosmos DV8x8x8 tokenizer, which applies 8× compression in both spatial and temporal dimensions, reducing 256 × 256 RGB frames to 32 × 32 latent grids."；"Models are trained for 60,000 steps using the AdamW optimizer, with a learning rate linearly decayed from 3e−5 after 100 warmup steps. We use 24 transformer layers, 8 heads, 512-dimensional tokens, and an MLP hidden size of 2048."；"Training is performed on a single NVIDIA A6000 with batch size 16. During inference, we use K = 2 decoding iterations."；ρmax = 0.2（L563–564）。

**Flow-HWM（流匹配）**（L358–398）：在连续 latent 空间学习速度场；线性插值路径与目标：

> "Xt = tX1 + (1− (1 − σmin)t)X0, (1)"（L373）；训练目标为预测速度与真值的期望 MSE（L386–390）；"We adopt classifier-free guidance (Ho & Salimans, 2022)"（L392–393）；推理从高斯噪声积分 t=0→1，一阶 Euler ODE（L395–398）。训练配置（L604–630）："We use the Cosmos Continuous 8x16x16 tokenizer that spatially compresses frames by a factor of 16 (from 256 × 256 to 16 × 16), and performs 8× temporal compression."；"Models are trained with d = 17 transformer layers and h = 1172-dimensional tokens using the AdamW optimizer, a learning rate of 1e−4, cosine learning rate scheduler, and batch size 128. Training runs for 150,000 steps across 2 NVIDIA A6000 GPUs. We use patch sizes plw = 2, pt = 1."；"We initialize final linear layers with Xavier initialization, as zero-initialization (as in prior works (Peebles & Xie, 2023)) led to instability. No learning rate warmup is used, as it degraded convergence. During inference, we apply 50 denoising steps with a classifier-free guidance scale of 3.0."

**Transformer Block 设计（L416–523）**：三条设计轴——(1) 注意力结构 joint vs split、(2) token 流间参数共享、(3) 流分组。Base Block：四条 token 流（vp, vf, ap, af）各自参数 + 联合注意力；Masked 版用分解式空间/时间注意力（factorized attention），"During temporal attention, all tokens from various streams [ap, af , L] jointly attend to one another along the temporal dimension. Spatial attention is applied seperately only to the video tokens. Rotary Position Embeddings (RoPE) (Su et al., 2023) are used during spatial and temporal attention."（L440–445）。Flow 版受 SD3 启发，3D RoPE 用于视频 token、1D RoPE 用于动作 token（L446–468）。参数共享两种：Full Sharing（四条流共享调制标量、QKV、MLP）、Modality Sharing（视频流内共享、动作流内共享）（L485–491）；实现上前 4 层保留独立参数、后 l−4 层共享（L493–499）。Split Attention：各流先自注意力，再以 vf 为 query、其余流为 K/V 做交叉注意力（L509–519）。

## 5. 实验与主结果

**评测协议（L545–554）：**

> "We evaluate our models using Fr´echet Inception Distance (FID) (Heusel et al., 2018) and PSNR (Hor´e & Ziou, 2010), computed over 21,000 generated frames from a held-out validation set. To isolate the impact of the generative model from the V AE's reconstruction quality, all ground truth frames are passed through the same V AE encoder-decoder used during training."

**表 1：Masked-HWM 四变体（L539–544，全部数值抄录）**

| 指标 | Split Attention | Base Block | Modality Sharing | Full Sharing |
|---|---|---|---|---|
| 模型大小 (Billion) | 0.220 | 0.321 | 0.237 | 0.195 |
| 峰值 GPU 显存 (GB) | 2.22 | 2.63 | 2.30 | 2.12 |
| 每秒样本数 | 2.09 | 2.27 | 2.25 | 2.36 |
| FID | 15.31 | 10.13 | 11.67 | 14.21 |
| PSNR (dB) | 29.37 | 29.02 | 28.97 | 28.66 |

**表 2：Flow-HWM 四变体（L610–615，全部数值抄录）**

| 指标 | Split Attention | Base Block | Modality Sharing | Full Sharing |
|---|---|---|---|---|
| 模型大小 (Billion) | 0.944 | 1.36 | 0.886 | 0.648 |
| 峰值 GPU 显存 (GB) | 4.37 | 5.94 | 4.41 | 3.25 |
| 每秒样本数 | 1.11 | 1.69 | 1.89 | 1.91 |
| FID | 111.12 | 111.59 | 112.75 | 110.73 |
| PSNR (dB) | 20.50 | 20.42 | 20.50 | 20.43 |

**主结论原文（L587–602；L644–661）：**

> "The Base Block achieved the best FID score (10.13), indicating the strongest visual quality among all configurations. The Split Attention variant obtained the highest PSNR (29.37 dB), but slightly underperformed in FID, suggesting better pixel-level fidelity at the cost of less realistic global structure."（Masked）

> "Modality Sharing matched Base Block quality with 26% fewer parameters, showing that intra-modality sharing suffices. Full Sharing, though slightly lower in quality, had the smallest footprint and fastest inference (2.36 samples/sec), making it ideal for efficiency-focused settings."

> "Full Sharing offered the best trade-off, outperforming the Base Block in all metrics while halving the parameter count. It achieved the lowest FID (110.73), fastest inference (1.91 samples/sec), and minimal memory use (3.25 GB)."

> "Overall, none of the Flow-HWM variants outperformed the Masked-HWM models in either visual quality or sampling speed, suggesting that masked video modeling is a more effective generative paradigm for our dataset and compute constraints."

**基线对比——Cosmos（L227–234）：**

> "NVIDIA's Cosmos (NVIDIA et al., 2025) is a notable exception—an open-source, high-fidelity video-to-video model. However, it is not designed for humanoid embodiments and is prohibitively resource-intensive. Its smallest variant (7B parameters) requires 8 NVIDIA H100 GPUs for training and over 40GB of VRAM for inference. On our compute setup of 2 NVIDIA A6000s, generating 121 frames video took over an hour, making finetuning or deployment impractical."

原文未写与 UniSim/IraSim 等其他世界模型的数值对比实验。

**物理/结构规律的定性证据（L575–583，Masked；L634–642，Flow）：**

> "The model learns both structural elements of the scene, such as furniture and small objects that the robot is manipulating, and overall textures. Larger parts of the robot's appendage, including arms and wheels, are generally modeled accurately. However, precision elements, such as fingers, are often slightly blurry and entangled. Further, the visual quality of generated images is robust to lighting, as shown in the 3rd (lighter) and 4th (darker) sequences."

> "As shown in Figure 5, the generated videos successfully capture overall scene structure (walls, floors, and doors). However, the outputs exhibit noticeable blurriness and artifacting like spotted patches. Visual quality tends to degrade in later frames, with the model struggling to preserve straight edges and rounded shapes. While the arms are often rendered with high fidelity, the model defaults to a canonical arm appearance and fails to represent unusual or out-of-distribution arm configurations (as seen in the top video strip)."

原文未使用「物理规律涌现（emergence of physics laws）」这一表述。

## 6. 消融

原文未写独立的消融（Ablation）章节。事实上承担消融职能的是 Table 1 / Table 2 对四种架构变体（Split Attention / Base Block / Modality Sharing / Full Sharing）的对照，即注意力机制（joint vs split）与参数共享策略两轴的消融。核心结论原句（L103–109）：

> "Importantly, we found that parameter-sharing strategies yielded near-identical performance to their non-shared counterparts while reducing parameter counts by 33–53%, significantly lowering computational requirements. These results suggest that architectural and efficiency trade-offs—particularly attention design and parameter sharing—can be leveraged to build lightweight, performant world models without compromising quality."

训练细节类消融原句（L626–628）："We initialize final linear layers with Xavier initialization, as zero-initialization (as in prior works (Peebles & Xie, 2023)) led to instability. No learning rate warmup is used, as it degraded convergence."——其余（如 warmup 消融数值）原文未写。

## 7. 局限 / 未来工作

原文未写专门的 Limitations / Future Work 章节。可视为事实性局限的散点陈述：

- 泛化声明（L130–132）："Our world models are trained specifically for humanoids, but can easily be extended to other embodiments."（原文未写该扩展的实验证据。）
- Masked 定性缺陷（L579–581）："However, precision elements, such as fingers, are often slightly blurry and entangled."
- Flow 定性缺陷（L636–642）："the outputs exhibit noticeable blurriness and artifacting like spotted patches. Visual quality tends to degrade in later frames, with the model struggling to preserve straight edges and rounded shapes. ... the model defaults to a canonical arm appearance and fails to represent unusual or out-of-distribution arm configurations"
- 结论（L662–668）："Humanoid World Models demonstrate that it is possible to build physically plausible, efficient predictive models tailored for humanoid robotics using modest computational resources. Through effective parameter-sharing strategies and compressive V AEs, HWM enables open-world reasoning for embodied agents in compute-constrained settings."
- 影响声明（L669–674）："This work advances video-based world models for humanoid robots, aiming to make predictive simulation more computationally accessible. While our models support progress in embodied AI, they are not intended for unsafe or unethical deployment."

## 8. 叙事关联（≤5 条推断）

1. **「世界模型 vs VLA」路线的直接文献证据**：原文明确把 VLA 判为有缺陷的路线，把视频生成判为更 grounded 的路线（见第 3 节两段引文）。支持原句（L144–147）："However, these methods often struggle with spatial reasoning (Tong et al., 2024), continuous sensorimotor processing, and require complex prompting strategies (Li et al., 2024)."；以及（L150–153）："Video captures fine-grained physical and temporal structure that language alone cannot express (Yang et al., 2024). When trained to predict future video frames given past observations and actions, these models serve as World Models—internal simulators that forecast the outcomes of potential action sequences"。【推断】该文构成「用世界模型承接 VLA 所不擅长的空间/物理推理」这一叙事的具体实验论据：VLA 直接映射动作，世界模型在动作后果层面做模拟。
2. **世界模型 = 慢脑（深思式规划器）**：支持原句（L61–64）："world models can be used for long-horizon planning by using them as a dynamics model (Yang et al., 2024) and enable more data-efficient policy learning through synthetic rollouts (Yang et al., 2023)."；以及（L155–157）："This enables agents to plan through imagination, reason counterfactually, and generate synthetic experience for policy training."【推断】「imagination / counterfactual / long-horizon」对应慢脑的推演式认知功能；但本文只交付动力学模型，规划器与策略本身原文未写（未实现、未实验）。
3. **世界模型 = 快脑的合成数据源**：支持原句（L13–15）："serving as a dynamics model in long-horizon planning and generating synthetic data for policy learning"；以及表 1/2 的「每秒样本数」指标（L542/L543，2.09–2.36；L613，1.11–1.91）。【推断】快脑（反应式策略）从世界模型 rollout 中学习是原文宣称的用途之一；「samples per second」是快脑数据吞吐的代理指标，但论文原文未写任何策略训练实验，此对应为推断。
4. **慢脑本地化部署诉求**：支持原句（L26–28）："HWMs are designed to be trained and deployed in practical academic and small-lab settings, such as 1–2 GPUs."；及 Cosmos 对比（L230–234）："Its smallest variant (7B parameters) requires 8 NVIDIA H100 GPUs for training and over 40GB of VRAM for inference. On our compute setup of 2 NVIDIA A6000s, generating 121 frames video took over an hour"。【推断】若慢脑（世界模型）无法与机器人同地部署，慢/快脑分工就只能在云端成立；本文的参数共享（33–53% 降参）正是把慢脑压进机载/实验室算力预算的努力。
5. **快脑的形态约束与开放世界缺口**：支持原句（L36–38）："However, in order to navigate the complexity and unpredictability of real-world environments these agents require sophisticated reasoning capabilities."；以及（L43–45）："they often lack the accuracy, reliability, and robustness needed for embodied AI in open-world settings"。【推断】按慢/快脑框架，快脑（直接感知-动作通路）在开放世界不可靠，故需慢脑（世界模型）补齐「reasoning」；本文以 1xGPT/EVE 的 100 小时遥操作数据（L525–528）训练该慢脑，但只验证了视频预测保真度（FID/PSNR），未验证其在下游规划/策略中的收益——该闭环证据原文未写。

## 9. 图注 / 表注逐字抄录

**Figure 1（L50–52）：**

> "Figure 1. Overview of Humanoid World Models (HWM). Given past video observations and humanoid control tokens (joint angles, velocities, etc.), it predicts future video observations."

**Figure 2（L431–436）：**

> "Figure 2. Architecture of a single transformer block in Masked-HWM (Base Block variant). Video and action streams are processed independently, with video streams also receiving Spatial Attention. All streams interact via joint Temporal Attention. RoPE is applied per attention type (2D for spatial and 1D for temporal). Each stream uses distinct MLP weights in the feedforward stage."

**Figure 3（L459–464）：**

> "Figure 3. Architecture of a single transformer block in Flow-HWM (Base Block variant). Each token stream (past/future video and actions) uses separate weights for timestep modulation, QKV projection, and feedforward MLPs. Joint Attention integrates all streams. RoPE is applied by modality: 3D for video tokens, 1D for action tokens."

**Figure 4（L584–586）：**

> "Figure 4. Four sample videos from the Base Block Variant of Masked-HWM. Top row: generated frames; bottom row: ground truth."

**Figure 5（L632–633）：**

> "Figure 5. Sample videos from the Base Variant of Flow-HWM. Top row: generated frames; bottom row: ground truth."

**Table 1（L537–538）：**

> "Table 1. Performance of Masked-HWM variants. Base Block yields the best FID; Split Attention gives the highest PSNR. Parameter sharing improves efficiency with minimal quality loss."

**Table 2（L608–609）：**

> "Table 2. Performance of Flow-HWM variants. Full Sharing achieves the best FID, memory usage, and speed even when compared to the Base Variant; Split Attention yields the highest PSNR."
