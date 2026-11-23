# WAM 精读线⑤：目标表征（Goal Representation）

> 9 篇具身智能 World Action Model 论文的【目标（goal）表征】轴深度精读。
> 来源级别：L3=方法节全文核实 / L2=部分正文 / L1=仅摘要。所有引文为论文原文（英文 verbatim + 小节号）。
> 「原文未写（已获取范围）」= 论文在已获取范围内确实没写；「未取到」= 工具截断导致没拿到（标注位置）。
> 取文通道：arxiv.org/abs/<id> → arxiv.org/html/<id>vN → ar5iv.labs.arxiv.org/html/<id>（截断回退）。shell 出网被沙箱拦截、TeX 源返回 gzip（web_fetch 不支持），无法本地抠 PDF。
> 更新：2026-10-26 23:40

---

## 1. JEPA-WAM — arXiv:2609.20277（v1）

- 元信息：JEPA-WAM: Connecting Generated Visual Instructions to World Action Models through JEPA Latent Representations。Tianbin Liu, Jian Zhu, Taiyi Su 等；cs.AI；v1 2026-08-04；11 页 4 图。HTML：https://arxiv.org/html/2609.20277v1
- 来源级别：**L3**（Abstract、Sec 1-3 全、Sec 4-5.2 开头）。**截断**：Sec 5.2 实验表格（Give bread/pear… 表）处；附录 A（生成 prompt 模板）、C（λ_v/λ_a 数值）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = 每个语言指令一个「生成图像 bank」的 latent 编码**（不是单一 goal image，也不是文本）。

- Sec 3.1「Generated Visual-Instruction Banks」，Eq. (1)：
  > "For each unique instruction, we construct an offline visual-instruction bank
  > ℬ_ℓ = {g_ℓ^(k) = G_φ(ℓ; ε_k)}_{k=1}^K,  (1)
  > where ε_k is a random sampling seed and K is the bank size."
- Sec 3.1（为何用 bank 而非单张）：
  > "A single generated image captures only one possible realization of task completion and may contain incidental appearance or layout patterns. Rather than treating one generated image as a deterministic visual goal, we associate each language instruction with a bank of diverse task-completion hypotheses."
  > "The resulting bank forms a finite approximation of the generator-induced conditional distribution p_φ(g|ℓ)."
- Sec 3.1（生成器只做离线建库，不训练；生成图非像素精确预测）：
  > "The generator is used only for offline bank construction and is neither fine-tuned nor jointly optimized with the robot policy. Because generation is conditioned on the language instruction rather than the current observation, the resulting images represent task-level completion hypotheses rather than pixel-accurate predictions of the future robot scene."

**编码器 = 冻结 V-JEPA 2.1 + Pool 8×8 + 可学习投影**。Sec 3.2「Latent visual semantics」，Eq. (2)：

  > "Let E_J denote a frozen V-JEPA 2.1 encoder and let P_ψ denote a learnable projector… To accommodate the temporal tubelet size of two used by V-JEPA 2.1, we duplicate each static visual instruction to form a two-frame clip. For each generated visual instruction g_ℓ^(k), we compute
  > Z_ℓ^(k) = P_ψ(Pool_{8×8}(E_J([g_ℓ^(k), g_ℓ^(k)]))) ∈ ℝ^{64×d_c},  (2)
  > where Pool_{8×8} spatially compresses the dense V-JEPA features into 64 tokens and P_ψ maps them into the WAM representation dimension d_c."
  > "Spatial pooling reduces the conditioning cost while retaining coarse spatial organization. The V-JEPA encoder remains frozen, whereas the projector is optimized jointly with the robot policy."

- Eq. (3)（bank 编码诱导的经验 latent 分布，训练即从该分布采样）：
  > "p̂_K(Z|ℓ) = (1/K) Σ_{k=1}^K δ(Z − Z_ℓ^(k)),  (3)
  > where δ(·) denotes the Dirac delta function. This formulation captures that JEPA-WAM is trained with stochastic samples from a distribution of latent task hypotheses rather than with a single fixed visual reference."

**条件化方式 = 拼接进条件序列，cross-attention 进 video 与 action 两个 expert**。Sec 3.2「World-action modeling」，Eq. (4)-(5)：

  > "Let C_ℓ ∈ ℝ^{L×d_c} denote the language conditioning sequence, and let P_s(s) ∈ ℝ^{1×d_c} denote the projected token of the current robot state s. The complete conditioning sequence is
  > C = [C_ℓ; P_s(s); Z_ℓ].  (4)
  > The video expert uses C to model task-guided visual dynamics, while the action expert uses the same sequence to predict the denoising direction of the action trajectory. We otherwise retain the directional masked-attention structure of Fast-WAM. Consequently, the latent visual semantics influence action generation directly through the action expert and indirectly through the conditioned video representations."
  > Training: "min_{θ,ψ} E_{(τ,ℓ)∼D} E_{k∼Uniform({1,…,K})} [L_WAM(θ; τ, ℓ, Z_ℓ^(k))],  (5)"
  > "We retain the original joint flow-matching objective L_WAM = λ_v L_video + λ_a L_action,  (6)… Only θ and ψ are optimized; the text-to-image generator G_φ and V-JEPA encoder E_J remain fixed."

- 部署（Sec 3.2「Training and inference」）：
  > "At deployment, a visual instruction is generated and encoded when a language instruction is received or changed. Its latent representation is then reused across subsequent control steps, avoiding repeated image generation and JEPA encoding for every action query."

### why 追问（基于原文）

1. **为何不用纯文本条件？** Abstract：
   > "rich visual-action trajectories are often paired with sparse and repetitive language annotations, allowing policies to identify tasks from visual context and motion regularities rather than grounding the instruction itself."
   Sec 1："Yet this language signal is static and substantially less diverse than the temporally dense visual-action supervision available throughout the trajectory."（文本信号静态且远不如时序密集的视觉-动作监督多样。）
2. **为何选 V-JEPA 2.1 编码（而非直接像素/CLIP 类编码）？** Sec 1：
   > "Directly conditioning the policy on raw generated pixels would therefore entangle task-relevant semantics with nuisance visual variation. We consequently require a representation that abstracts away low-level appearance differences while preserving the spatially structured semantics needed to identify target objects, relations, and task outcomes. V-JEPA 2.1 is particularly suited to this role because it learns predictive representations in latent space rather than reconstructing raw pixels, while its dense features retain explicit spatial structure and semantic coherence."
   Sec 2.3："JEPA-WAM does not train the V-JEPA predictor or introduce an additional predictive objective. Instead, it reuses the frozen encoder's dense features as conditioning for the WAM."（只用冻结编码器，不引入新预测目标。）
3. **为何生成图像而不要求真实 goal image？** Sec 2.2："JEPA-WAM uses this capability to construct instruction-specific visual banks without requiring user-provided goal images, additional robot trajectories, or joint training of the image generator."；Sec 5.4 有「Generated Visual Instructions vs. Real Keyframes」对照实验（数值未取到）。
4. **为何选 DreamLite（0.39B）？** Sec 3.2「Implementation details」："We select these lightweight variants to keep the complete conditioning pipeline on a single device and eliminate inter-GPU communication during closed-loop control. The complete inference stack, including image generation and JEPA encoding, runs on a single NVIDIA RTX 5090."（轻量以单卡闭环，非性能最优论。）

### 关键数值

- K = **1,024** 张/指令（"the main JEPA-WAM model uses a bank of 1,024 generated images per instruction"，Sec 3.2 Implementation details）
- 每张图编码为 **64 tokens**（Eq. 2）；V-JEPA 2.1 **ViT-B/16、80M** 参数；生成器 **DreamLite 0.39B**
- 训练数据 **1,370** 条真机演示（PnP/Sort/Wipe 各 150，Give Me 各 80）；11 个行为重叠任务
- 成功率 ID/**87.3%**、OOD-S/**74.5%**、OOD-I/**80.9%**；优于 π0 与 Fast-WAM ≥10.0/27.3/14.5 pp（Abstract）

### 一页总结

JEPA-WAM 把「目标」定义为**指令条件的生成图像 bank（K=1024）在冻结 V-JEPA 2.1 latent 空间的 64-token 紧凑编码**。生成（DreamLite）只管离线建库、不参与训练；V-JEPA 编码抑制生成图与真实场景之间的外观差异、保留空间语义；目标 latent 以 [文本; 状态; Z_ℓ] 形式同时 cross-attention 进 video expert 与 action expert。训练时每条轨迹从 bank 均匀采样一个目标（Eq. 5），形成「latent 目标分布上的随机条件化」。动机链条：文本稀疏重复→VLA 学捷径→补多样视觉假设→但生成图外观偏→JEPA 预测性 latent 兜底。

### 待核实

- λ_v / λ_a 数值（附录 C，未取到）；生成 prompt 模板（附录 A，未取到）；Sec 5.3 Goal-Bank Scaling、5.4 生成 vs 真实关键帧的数值（截断）。

---

## 2. Act2Goal — arXiv:2512.23541（v1）

- 元信息：Act2Goal: From World Model To General Goal-conditioned Policy。Pengfei Zhou 等，Agibot Research；cs.RO；v1 2025-12-29。HTML：https://arxiv.org/html/2512.23541v1 ；ar5iv：https://ar5iv.labs.arxiv.org/html/2512.23541
- 来源级别：**L3**（Abstract、Sec I-IV.A 完整）。**截断**：Sec IV（Task 3 描述处，ar5iv）；V-A Implementation Details（VAE 型号、MSTH 超参 K/P/r/M）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = image-goal（纯视觉，去掉语言）**，以 VAE latent z_g 与当前观测 z_t 沿 hidden-state 序列拼接，条件 flow-matching 世界模型。Sec III-A：

  > "our goal-conditioned world model builds upon the Genie Envisioner architecture with key modifications tailored for goal-conditioned policy learning. We introduce a goal visual condition that is concatenated with the current observation along the hidden states sequence, while removing all language-conditioning components to create a purely vision-based model."
  > "z_pred = f_θ(z_t, z_g, ε),  (1)
  > where z_t and z_g are VAE-compressed latents of the current observation and goal respectively, ε represents random noise inputs… f_θ is the flow matching model that generates the latent frames between current observation and goal."

**MSTH 远端对数稀疏采样公式**。Sec III-B，Eq. (5)（上下文与理由逐字）：

  > "Given a total imagined trajectory length K, a proximal horizon P, and a vision sampling stride r, MSTH partitions the future trajectory into two segments.
  > The proximal segment consists of high-frequency short-horizon visual states {s_{t+kr}}_{k=1}^{P/r}, which capture fine-grained local dynamics. The distal segment contains M sparsely sampled visual states {s_{t+d_m}}_{m=1}^M, where the indices d_m are determined by logarithmic spacing:
  > d_m = P + ⌊(K−P)/log(M+1) · log(m+1)⌋,  m = 1,…,M.  (5)
  > This logarithmic sampling results in increasing temporal intervals as the horizon extends, providing coarse but goal-aligned long-term guidance."
  > "Proximal actions are predicted at every timestep… In contrast, distal actions {a_{t+d_m}}_{m=1}^M are aligned with the distal visual states and serve as long-horizon guidance. During deployment, only the proximal actions are executed, while distal predictions remain latent and guide long-term goal adherence."

- 注：「完整推导」——原文只给定义 + 上式 + 一句话理由，无更长的逐项推导；「K-P 区间内对数间隔、近密远疏、远端仅作锚点不执行」即其全部论证。
- 动作专家条件化（Sec III-A，Eq. 3-4）：
  > "a_pred = g_φ(c_w, c_p, ζ),  (3)
  > where… c_w = {h_world^1, …, h_world^L} represents the layered transition features, and c_p is the proprio state condition."
  > 架构上（Fig. 3 caption）："the robot state and multi-scale features from the world model are fed via cross-attention into isomorphic Action DiT blocks"。
- 训练（Sec III-C，Eq. 6-8）：Stage1 = L_v + λ·L_a，**λ = 0.1**；Stage2 只用 L_a（行为克隆端到端）。

### why 追问（基于原文）

1. **为何 image-goal 而非语言？** Sec I：
   > "While natural language offers a flexible interface for diverse tasks, it often lacks the necessary granularity for fine-grained manipulation… Visual goals offer a more precise alternative by directly encoding object configurations, spatial relations, and terminal constraints, avoiding linguistic ambiguity and explicit reward engineering."
2. **为何需要目标条件的世界模型（而非直接 GCP）？** Sec I：
   > "Without an explicit model of visual transitions toward the goal, such policies tend to overfit demonstrated state–action mappings and must rely on dense supervision to compensate for the lack of structured intermediate guidance."
3. **为何 MSTH 多尺度（而非均匀采样）？** Sec I：
   > "A general GCP must balance global consistency—maintaining fidelity to the long-term goal—with local reactivity—responding robustly to perturbations… Full-trajectory planning provides global coherence but is brittle under deviations; short-horizon control improves robustness but easily loses directional alignment in extended tasks."
4. **为何 online 阶段能免奖励？** Sec III-D + Algorithm 1："we automatically relabel the goal condition as the robot's observation at the end of the inference step"（g′ ← o′；HER 式 goal relabel + LoRA）。

### 关键数值

- λ = **0.1**（Eq. 8）；真机 OOD 成功率 **0.30 → 0.90**（几分钟自主交互，Abstract）
- RoboTwin 2.0：Easy Act2Goal 0.62/0.80/0.64/0.52；Hard 0.13/0.43/0.13/0.15（TABLE I）
- 真机三任务 ID **0.93/0.75/0.45**、OOD **0.90/0.48/0.30**（TABLE II）
- 仿真 goal image 获取方式（Sec IV-A）："the final goal image was obtained via fixing the environment seed… extracting it from successfully executed trajectories under this seed"
- MSTH 超参 K/P/r/M、VAE 型号：V-A 截断，**未取到**

### 一页总结

Act2Goal 是纯视觉 goal-conditioned：goal 图像经 VAE 压缩成 z_g 与当前观测 z_t 拼接，条件一个 Genie-Envisioner 式 flow-matching 世界模型，想象「当前→目标」的中间视觉轨迹；MSTH 把该轨迹近端密集（P/r 帧、逐帧动作）远端对数稀疏（Eq. 5，仅作全局锚点），世界模型逐层特征 c_w 经 cross-attention 进入同构 Action DiT。目标图像既是任务规格（消除语言歧义）又是世界模型的条件，HER 式 relabel 让目标可以随时换成「已到达的状态」实现免奖励在线自改进。

### 待核实

- V-A：VAE 型号、MSTH 的 K/P/r/M 与 N（去噪步数）数值（未取到；README 全局待核实 #5 一致）；IV-B/C（在线自改进、MSTH 消融）数值（截断）。

---

## 3. NavWAM — arXiv:2606.13494（v1）

- 元信息：NavWAM: A Navigation World Action Model for Goal-Conditioned Visual Navigation。Daichi Azuma 等，东大；cs.RO；v1 2026-06-11。HTML：https://arxiv.org/html/2606.13494v1 ；项目页：https://dachii-azm.github.io/navwam/
- 来源级别：**L3**（主文 Abstract→Conclusion 完整）。**截断**：References 之后（ar5iv 两轮同点）；附录 A.1（Latent Canvas Frame Layout）、A.2（Goal-Progress Value v_{t+H}）**未取到**。

### 目标表征机制（原文 + 指路）

**目标形式 = image-goal（goal 位置拍摄的目标图），作为 9 帧 latent canvas 的第 3 帧（底排 4 帧之一）**。Sec 4.1：

  > "NavWAM represents navigation as denoising a fixed nine-frame latent canvas… The bottom four frames in the canvas are observed and condition the denoising process: a blank frame required by the causal VAE's temporal compression, the current robot state s_t, a goal frame containing the goal specification g, and the current egocentric observation o_t. In our image-goal setting, g is the target image captured at the goal location and is encoded as the goal-image frame. The top five frames are generated as prediction targets: the executable action chunk a_{t:t+H-1}, the future state s_{t+H}, two future egocentric observations o_{t+H-1} and o_{t+H}, and the goal-progress value v_{t+H}."
  > （即 9 帧布局：底 4 = [blank pad, state, **goal**, obs]，顶 5 = [action, future state, future obs ×2, **value**]。）

- 编码方式（Sec 4.1）："Image frames are encoded as standard video latents through the causal VAE. Non-image variables, including states, action chunks, and the scalar value, are normalized and broadcast over the latent spatial grid. Their predictions are recovered by averaging the denoised entries of the corresponding frame."（Cosmos Predict2 的 causal VAE + DiT，非视觉变量也压成 latent frame，不用单独 head。）
- 联合分布（Sec 3.2，Eq. 3）：
  > "p_θ(a_{t:t+H-1}, s_{t+H}, o_{t+H-1:t+H}, v_{t+H} | o_t, g).  (3)… v_{t+H} is a goal-conditioned scalar value that estimates progress toward the goal."
- **goal-progress value 的监督来源**：主文机制（Sec 4.2）：
  > "For navigation, the value frame is trained to represent goal progress rather than a generic reward-to-go. This encourages the model to estimate whether the predicted future state moves the robot closer to the specified goal."
  > 三模式（Sec 4.2 Multi-mode Conditioning）："a policy mode that conditions on the current state, current observation, and goal, and predicts the action chunk, future state, future observations, and value; a world-model mode that additionally conditions on the action chunk…; and a value mode that conditions on the future frames and predicts the value… samples are assigned to these modes with a 50/25/25 split."
  > ⚠️ value 帧的**具体标量标签**（如何由数据算出的 progress 真值）在附录 A.2「Goal-Progress Value v_{t+H}」——**未取到**（截断于 References 之后；项目页亦未给公式）。主文只有「表示 goal progress、由 value mode 用 future frames 条件预测」的机制描述。
- 训练目标（Sec 4.2，Eq. 4）：L_diff = E_{σ,ε}[w(σ)‖x_0 − F_θ(x_σ, σ, c)‖²₂]，"We upweight the action frame relative to the other prediction frames so that the low-dimensional action signal is not dominated by the high-dimensional image reconstruction loss."（动作帧权重上浮，具体倍率未见主文。）
- 推理（Sec 4.3）：policy mode 输出动作块，receding-horizon 执行；未来帧与 value 仅作可解释前视，不参与执行（best-of-N 采样为可选，主结果不用）。

### why 追问（基于原文）

1. **为何 image-goal 而非语言/物体目标？** Sec 3："In this work, we focus on image-goal navigation, where g is a target image captured at the goal location." Sec 4.1："Although the same canvas could in principle be extended to language- or object-specified goals through the goal frame or text-conditioning interface, we focus on image-goal navigation in this work."（可扩展但本文只做 image-goal；无更强的反语言论证。）
2. **为何 value 帧代表 goal progress 而非通用 reward-to-go？** Sec 4.2（引文见上）——目标是让模型显式估计「预测未来是否更接近指定 goal」，是导航场景的定制化。
3. **为何把 goal 放进共享 latent 序列而不是旁路编码器？** Sec 4 Overview："The current observation, image goal, robot state, action chunk, future egocentric observations, and goal-progress value are assigned to different latent frames in the same diffusion-transformer sequence. This turns goal-conditioned navigation into a joint denoising problem over future perception, progress estimation, and executable motion." Sec 2："NavWAM avoids this separation by jointly predicting action chunks, future egocentric observations, and goal-conditioned values within one policy representation."（把「目标感知」与「控制」做成同一个去噪问题，避免外部规划器。）

### 关键数值

- 9 帧 latent canvas；模式比例 **50/25/25**；backbone Cosmos Predict2（**2B**，causal VAE）
- go stanford：ATE 0.324（无 FT）→ **0.192**（w/ FT）；RPE 0.099 → 0.070；subject consistency 0.524（NWM）→ **0.668**（NavWAM）、FT 后 0.635
- 头部消融（Table 2）：仅图像+planning 0.326/0.569（h=4/8）；+action+state 0.107/0.287；+value 全量 **0.076/0.192**
- 真机（Diablo，24 集/4 环境）：NavWAM **19/24 = 79.2%** vs OmniVLA 58.3% vs NWM 16.7%；224×224 RGB；vs OmniVLA（7B）在 sit 上 ATE 0.077 vs 0.086

### 一页总结

NavWAM 把「目标」放在共享 latent canvas 的**底排第 3 帧**（goal frame），与 state/obs 同为条件帧；顶排 5 帧是预测目标，其中**第 9 帧是 goal-progress 标量 value**。目标图经 causal VAE 压成视频 latent，与动作/状态/value 用同一 DiT 联合去噪；value 帧被显式训练成「接近目标的进度估计」（policy/world-model/value 三模式 50/25/25）。goal-progress value 的精确标签公式未取到（附录 A.2）。

### 待核实

- 附录 A.1/A.2：canvas 帧布局细节与 **v_{t+H} 标量标签的构造公式（监督来源）**——未取到；动作帧上浮权重倍率（主文未给）；A.5 三模式细节、B 超参（未取到）。

---

## 4. GR-MG — arXiv:2408.14368（v2）

- 元信息：GR-MG: Leveraging Partially-Annotated Data via Multi-Modal Goal-Conditioned Policy。Peiyan Li 等，中科院自动化所 + ByteDance；cs.RO，RA-L；v2 2024-12-23。HTML：https://arxiv.org/html/2408.14368v2 ；ar5iv（全文完整）：https://ar5iv.labs.arxiv.org/html/2408.14368
- 来源级别：**L3 全文**（ar5iv 全篇到 References）。

### 目标表征机制（原文 + 指路）

**目标形式 = InstructPix2Pix 生成的 subgoal 图像（中间态而非终态）+ 文本双条件**。Sec III-B1：

  > "Specifically, we use InstructPix2Pix, a diffusion-based image-editing model, as the network. Following the approach in Susie, we generate a sub-goal representing an intermediate state rather than the final state, and update the sub-goal at fixed time intervals during the rollout."

- **进度文本格式**（Sec III-B1，verbatim 例子）：
  > "To incorporate progress information into the generation model, we append a sentence to the task description, e.g., "pick up the red block. And 60% of the instruction has been completed.". We use T5-Base to encode the text."
- **10 档离散化**（Sec III-C）：
  > "As a rough task progress can already be very informative and can bring about more stable training, we discretize the task progress predicted from the policy from 0% to 100% into 10 bins before passing it to the goal image generation model."
- 进度真值来源（Sec III-B1 / III-C）：
  > "During training, the progress of a frame can be easily obtained from its timestep in the trajectory (or video)."
  > "The input task progress in the training of the goal image generation model is computed from the timestep of the image frame in the video."
- **目标图像采样与更新节奏**（Sec III-C / III-D）："We sample the frame which is N steps away from the current frame in the trajectory/video as the goal image." "During inference, the task progress is initially set as zero… we run the goal image generation model only every n<N steps in a closed-loop manner for efficiency."（N、n 的**具体数值原文未写**，仅用符号。）
- **目标图像注入策略（Sec III-B2）**：
  > "To inject the goal image condition, we use MAE to tokenize the goal image generated by the goal image generation model and append the tokens at the front of the input token sequence (Fig. 2). These tokens can be attended by all the subsequent tokens for conditioning."
- **[PROG] 查询 token**（Sec III-B2）：
  > "in order to predict task progress, a [PROG] query token is included. The output embedding of the token is passed through linear layers to regress the progress value."
- **cVAE 动作轨迹头**（Sec III-B2）：
  > "we follow recent work to predict action trajectories instead of a single action as in GR-1 via a conditional variational autoencoder (cVAE). Specifically, we use a VAE encoder to encode the action trajectory into a style vector embedding. We concatenate the style vector embedding, the output embedding of the [ACT] token, and k learnable token embeddings together and input them to a transformer for predicting an action trajectory of k steps."
- 无文本标签数据的训练（Sec III-C）："if data w/o text labels is available, we first train the policy with the data, where a null string is provided as the text condition."；policy 初始化自 GR-1 的 Ego4d 生成式视频预训练权重。

### why 追问（基于原文）

1. **为何文本 + 图像双条件（而非只用生成图）？** Sec I：
   > "Since GR-MG is conditioned on both the text and goal image, the policy can still rely on the text condition to guide action prediction even if the generated goal image is inaccurate, substantially improving the robustness of the model."（生成图不准时文本兜底。）
2. **为何把 progress 注入目标生成？** Sec III-B1：
   > "The insight is that robot manipulation is a sequential decision process instead of static image editing. Additionally, the current observation can be ambiguous in the case where there are identical observations throughout the rollout of a task, e.g., moving an object back-and-forth. The progress information can provide valuable global information for goal image generation."
3. **为何要 10 档离散化？** Sec III-C（引文见上）：粗略进度已足够 + 训练更稳定（"a rough task progress can already be very informative and can bring about more stable training"）。
4. **为何目标设为 N 步后的中间态而非终态？** Sec III-B1："we generate a sub-goal representing an intermediate state rather than the final state, and update the sub-goal at fixed time intervals"（沿 Susie 做法，闭环滚动更新）。

### 关键数值

- 进度离散 **10 bins**；CALVIN 1/5 连做成功率 **96.8% / 64.4%**、平均长度 3.35→**4.04**（100% 数据）；10% 数据 + 无文本数据：1-task **90.3%**、avg len **3.11**（vs GR-1 全数据 3.06）
- 真机 58 任务：Simple **68.7%→78.1%**；Generalization **44.4%→60.6%**；few-shot 10/30-shot **17.5%/37.5%**
- 进度消融（Table II）：GR-MG MSE 903.139 / PSNR 19.121 / SSIM 0.730 / CD-ResNet50 0.946（vs w/o progress 965.347/18.821/0.721/0.945）
- 无文本数据 1M 帧 vs 全标注 0.1M 帧（10% 设置）；真机 18k demos/37 任务训练
- N（goal 提前步数）、n（生成间隔）数值：**原文未写**（仅符号 N、n<N）

### 一页总结

GR-MG 的目标 = InstructPix2Pix 依据「当前帧 + 文本 + 进度句」编辑出的 N 步后 subgoal 图像；进度以文本句子 "…And X% of the instruction has been completed." 注入、由 [PROG] token 回归、离散化到 10 档再喂回生成器形成闭环。策略侧：MAE 编码 goal image 放在 token 序列前端（所有后续 token 可见），[ACT]+[PROG] 查询 token 出 cVAE 动作轨迹与进度；训练时目标图从轨迹 N 步后采样（HER 式自监督），无文本时用 null string。设计核心：文本与图像互为保险，进度提供时序全局信息。

### 待核实

- policy 侧 [PROG] 回归的**进度标签公式**（正文只说 "regress the progress value"，生成侧明确用视频时间步；策略侧标签是否同为归一化时间步**原文未写明**）；N、n 数值（原文未写）；[OBS] future-image 头细节沿用 GR-1（未展开）。

---

## 5. LaWAM — arXiv:2606.15768（v1）

- 元信息：LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies。Jialei Chen 等，清华等；cs.RO；v1 2026-06-14。HTML：https://arxiv.org/html/2606.15768v1
- 来源级别：**L3**（主文 Sec 1-3 完整、4.1 开头）。**截断**：Table 1（LIBERO）处；附录 C.1/C.2（架构细节）、C.5（λ 数值、训练超参）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = latent visual subgoal û_T（未来观测特征，非像素）**，由潜动作 z 经 LaWM 解码得到。Sec 3.1：

  > "LaWAM keeps the future-conditioned structure but represents the future in a frozen visual feature space. Let f_ψ be the frozen encoder, and define u = f_ψ(o) and u_T = f_ψ(o_T). We first learn a latent action model over such feature pairs:
  > z ∼ q_φ(z | u, u_T),   ũ_T = LaWM_ω(u, z).  (2)
  > The distribution q_φ is the latent-action posterior: it acts as a latent inverse-dynamics model, inferring z from the observed transition (u, u_T). The decoder then predicts the horizon feature from the current feature and z, and we retain this decoder as the Latent World Model (LaWM)."

- 编码器：Sec 1 / Sec 2："in the latent space of frozen visual encoders, such as DINOv3"（冻结视觉基础模型特征空间，如 DINOv3）。
- **LaWM 训练目标**（Sec 3.2，Eq. 4）：
  > "We train the latent action model with a forward-prediction objective and KL regularization,
  > L_LAM = L_wm + L_aux + β D_KL(q_φ(z | u, u_T) ‖ N(0, I)),  (4)
  > where L_wm = ‖ũ_T − u_T‖²₂ trains LaWM to match the horizon feature, and L_aux = ‖g(s, z) − s_T‖²₂ trains the auxiliary state predictor. The KL term regularizes the latent-action space so that it can later be modeled by the policy prior."
  > "A lightweight predictor g maps the current end-effector state s and latent action z to the horizon state s_T, encouraging z to encode embodied motion rather than only visual appearance change. After pretraining, we discard this auxiliary head."
- **潜动作 z 的推断分布**：q_φ(z|u,u_T)，KL 拉到 N(0,I)（Eq. 4）；"the inferred latent action z serves as the teacher signal for the stage-two policy prior"（Sec 3.2）。
- 推理分解（Sec 3.1，Eq. 3）：
  > "p(a_{1:T}, û_T, ẑ | o, l) = p_θ(ẑ | o, l) [Policy Prior] · p_ω(û_T | u, ẑ) [LaWM] · p_η(a_{1:T} | o, l, u, û_T) [Action Expert].  (3)
  > …We use u_T for the training target and û_T for the policy-driven subgoal."
- **subgoal 如何条件动作专家**（Sec 3.3）：
  > "The action expert uses this predicted future through an Alternate-DiT design. One stream carries the semantic context from the VLM backbone, while the other carries the latent dynamics context formed by (u, û_T). Alternating between these streams lets the expert combine task intent with the predicted scene evolution when denoising the action chunk."
- Stage-2 目标（Sec 3.3，Eq. 5）：
  > "L_LaWAM = λ_distill L_distill + λ_wm L_wm + L_act.  (5)
  > Here L_distill = E[‖ẑ − z‖²₂], L_wm = ‖û_T − u_T‖²₂ supervises the policy-driven subgoal, and L_act denotes the conditional flow-matching loss for a_{1:T} given (o, l, u, û_T)."（另有 Knowledge Insulation 防动作梯度覆盖 LaWM。）
- 物理时间对齐（Sec 3.2）："Because both u_T and s_T are defined by the same physical interval τ, the learned subgoal corresponds to a consistent amount of elapsed motion rather than a dataset-specific frame offset."（τ 的具体数值在附录 C，未取到。）

### why 追问（基于原文）

1. **为何 latent 而非像素视频目标？** Sec 1：
   > "many methods predict future images or videos, allocating substantial modeling capacity to pixel-level synthesis rather than compact action-relevant dynamics… effective future prediction for manipulation should expose the state change relevant to the next action chunk, rather than merely generating visually plausible task-consistent futures."
2. **为何复用 LAM 的 decoder 当世界模型（而非另训）？** Sec 2：
   > "as concurrently observed by Garrido et al., the decoder of a latent action model already implements a latent action-conditioned world model. LaWAM builds on this observation by repurposing the decoder as a policy-facing latent dynamics interface, where latent actions are expanded into embodiment-grounded future observation features that directly condition action generation."
   Sec 2（对比既有做法）："Prior latent-action-based VLAs mainly use this framework to learn embodiment-agnostic action representations, treating the decoder only as an auxiliary training component and discarding it after pretraining. In contrast, we repurpose the decoder as the core predictive module of LaWM."
3. **为何需要 L_aux 状态回归？** Sec 3.2："encouraging z to encode embodied motion rather than only visual appearance change"（防止 z 只编码外观变化）。
4. **为何用 KL 正则？** Sec 3.2："The KL term regularizes the latent-action space so that it can later be modeled by the policy prior."（为 Stage-2 的策略先验可学。）

### 关键数值

- LaWM **230M** vs 像素 WAM 的 5B WAN backbone（**约少 95%** 世界模型参数）；总模型 **2.3B**；**187 ms**/动作块；比像素 WAM 快**最多 24×**
- 预训练数据：约 **3,000 h** 机器人视频 + **1,500 h** 自中心人类视频（人类视频只贡献 LaWM 动力学先验）
- LIBERO **98.6%** SR；RoboTwin **91.22%**；对照：LingBot-VA 4482 ms vs π0.5 220 ms
- z 维度、τ 数值、λ_distill/λ_wm：附录 C 未取到

### 一页总结

LaWAM 的目标是「一个动作块之后」的**潜在视觉 subgoal û_T = LaWM_ω(u, ẑ)**：Stage-1 在冻结 VFM（如 DINOv3）特征空间训潜动作模型（后验 q_φ(z|u,u_T) + 前向解码 + 状态回归辅助头 + KL 正则，Eq. 4），把 decoder 留下当 LaWM；Stage-2 训练策略先验 p_θ(ẑ|o,l) 预测潜动作，经 LaWM 单次前向解码出 û_T，再以 (u, û_T) 为「动力学流」与 VLM 语义流交替进入 Alternate-DiT 动作专家（Eq. 5：蒸馏 + subgoal 回归 + flow matching）。目标彻底不落像素，换取 187ms/24× 的推理效率。

### 待核实

- 附录 C.1/C.2：LaWM/LaWAM 架构（z 维度、特征网格、Alternate-DiT 细节）；C.5：λ_distill/λ_wm、β、τ 数值——未取到（截断于 Table 1）。

---

## 6. V-JEPA Policy — arXiv:2609.37250（v2）

- 元信息：V-JEPA Policy: Building Effective World-Action Models on Predictive Visual Latents。Yang Zhang 等；cs.CV；v2 2026-10-10（19 页 5 图 11 表）。HTML：https://arxiv.org/html/2609.37250v2
- 来源级别：**L3**（Sec 2 Preliminaries、3.1-3.3 完整，Eq. 1-9；Sec 5.1 开头）。**截断**：Sec 5.1 Implementation details 段尾；附录 A.2（网络配置/训练配方，含 δ 偏移数值）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = 未来 V-JEPA 2.1 latent（K 个未来观测的预测性特征），以 mask-token 查询做单次前向预测**。

- 目标构造（Sec 2 Problem formulation，Eq. 2）：
  > "o_t^+ = (o_{t+δ_1}, …, o_{t+δ_K}),  0 < δ_1 < ⋯ < δ_K,  (2)
  > denote K future observations from the same camera views, sampled at temporal offsets δ_1, …, δ_K… In our V-JEPA Policy, the latent representations of o_t^+ serve as prediction targets, and future prediction is learned jointly with action generation."
- 监督细节（Sec 3.1「A fixed visual space」）：
  > "To match the encoder's temporal tubelet size of two, we form a minimal two-frame observed context for each camera view by pairing the current frame with an earlier observation… Following the construction of context and target representations in V-JEPA, we designate its future tubelet positions as the masked region. The context path thus removes tokens at future positions before the Transformer blocks of the encoder, and encodes only observed tokens. The target path instead processes the complete unmasked clip with the same frozen encoder. We normalize its output tokens across the feature dimension and select the future positions as prediction targets. We independently encode each camera view and concatenate their context representations and future targets across views to form Z_t^c and Z_t^+, respectively."
- **mask-token 查询**（Sec 3.1「Future-query prediction」）：
  > "We project Z_t^c to the predictor's hidden dimension and concatenate the result with future-query tokens Δ_t^+. These query tokens repeat a single learnable mask embedding at all future target positions. Bidirectional self-attention jointly updates both groups, allowing context states to incorporate information from the evolving future-query representations."
- **3D RoPE**（Sec 3.1）：
  > "To distinguish token positions, we utilize 3D rotary position embeddings (RoPE) based on temporal, height, and width coordinates in the visual latent grid… Attn_3D(Q, K, V) = softmax(R_3D(Q) R_3D(K)ᵀ / √d_h) V,  (3)
  > …Coordinates are local to each camera view. Learnable view embeddings are added to both context tokens and future queries to distinguish their associated cameras."
- 指令/状态条件化（Sec 3.1）：冻结 T5 文本 token + 1 个 state token 组成条件序列，每层 predictor block 做 cross-attention；输出（Eq. 4）：
  > "Ẑ_t^+ = P_φ(Z_t^c, Δ_t^+ | ℓ, q_t).  (4)"
- **目标如何条件动作（Sec 3.2）**——不是用最终预测 latent，而是用 predictor 的逐层 context K/V：
  > "We connect future prediction to action generation through the predictor's layer-wise context states rather than its final future-latent predictions… At predictor layer j, let K_c^(j) and V_c^(j) denote the key and value projections of the normalized block input at context positions… The resulting layer-wise interface is C_φ = {(R_3D(K_c^(j)), V_c^(j))}_{j=1}^L… To maintain the spatiotemporal position information of the context tubelets when conditioning the action expert, we preserve the 3D RoPE transformation applied to the keys… while the values remain unrotated. This interface contains only keys and values at context positions."
  > Eq. 6：动作 token 的 Q 同时 attend [R_3D(K_c^(j)); K_a^(j)] 与 [V_c^(j); V_a^(j)]；"In contrast, predictor tokens cannot attend to action tokens."（MoT 共享注意力，单向：predictor→action。）
- **L1 回归与联合目标（Sec 3.3）**：
  > "L_future = E_D ‖P_φ(Z_t^c, Δ_t^+ | ℓ, q_t) − Z_t^+‖_1.  (7)"
  > "L_action = E ‖v_ψ(a_t^τ, τ; C_φ, ℓ, q_t) − (a_t − ε)‖²₂.  (8)"
  > "L(φ, ψ) = L_action + λ_future L_future,  (9) with λ_future = 1. The action loss also backpropagates through C_φ into the predictor, so its context states are jointly shaped by the supervision of both the future prediction and the action generation."
- 推理（Sec 3.3）：imagine-then-act：一次 predictor 前向构造 C_φ 并缓存，动作专家 10 步 Euler 积分出动作块。
- 动作流匹配构造（Eq. 5）：a_t^τ = (1−τ)ε + τ a_t。

### why 追问（基于原文）

1. **为何用 V-JEPA latent 而非继承预训练生成器？** Sec 1 的研究问题 + 动机：
   > "Does effective WAM learning require inheriting a complete pretrained visual generative model, or can a predictive visual latent space learned at scale provide a sufficient foundation?"
   Sec 2 Preliminaries："The V-JEPA family of models learns visual representations through self-supervised prediction in feature space, encouraging the modeling of predictable spatiotemporal structure and dynamics rather than pixel-level generation."
   Sec 5.3（对照结论，摘要）：“Comparing visual foundations under the same downstream framework and training budget identifies V-JEPA latents as more effective than the discriminative, reconstructive, and video-understanding-oriented alternatives, particularly under distribution shifts.”
2. **为何动作专家用逐层 context K/V 而非最终未来 latent？** Sec 3.2："Through bidirectional interactions with future queries in the predictor, context states at deeper layers incorporate the evolving future-query representations."（更深层的 context 状态已吸收未来查询信息，比最后输出更「未来知情」。）
3. **为何未来预测监督用 L1？** 沿用 V-JEPA 2 的 mask-denoising 目标（Sec 2，Eq. 1 原文即 L1："L_predict = (1/|M|) Σ_{i∈M} ‖P_φ(E_θ(x), Δ_y)_i − sg(E_θ̄(y)_i)‖_1"）；本文 Eq. 7 直接继承该 L1 形式——「同框架同预算」的对照公平性使然。
4. **为何 DROID 无动作预训练能迁移？** Sec 1："Task-specific demonstrations cover only a limited range of behaviors and visual transitions. We therefore pretrain only the instruction-conditioned future-latent predictor on diverse DROID video–instruction pairs, using future-prediction supervision without action labels while keeping the V-JEPA encoder frozen."

### 关键数值

- 总参 **0.9B**（encoder 0.3B + predictor 0.5B + expert 0.1B），**0.6B 可训**；V-JEPA 2.1 **ViT-L**、T5-XXL 冻结
- λ_future = **1**；推理 10 Euler 步；LIBERO 10 epochs；RoboCasa 50k 步、batch 256
- LIBERO **97.25%**、LIBERO-Plus **79.25%**、RoboCasa-GR1 **50.92%**、真机双任务 **45%**
- DROID 无动作预测器预训练：LIBERO-Plus 79.25→**91.50**（超过延长训练的 81.64）；真机 45%→**80%**
- δ_1..δ_K 与 K 的数值：附录 A.2 未取到

### 一页总结

V-JEPA Policy 把「目标」定义为**未来 K 个观测在冻结 V-JEPA 2.1 latent 空间的归一化特征 Z_t^+**：用单次前向的 mask-query 预测器（重复 learnable mask embedding + 3D RoPE + T5/状态 cross-attention）回归出 Ẑ_t^+（L1，Eq. 7），再把 predictor 逐层、经 3D RoPE 旋转的 context K/V 作为未来知情接口 C_φ 喂给 flow-matching 动作专家（MoT，Eq. 6）。核心主张：**预测性视觉 latent 本身即可支撑 WAM，无需继承完整视觉生成器**；且同 latent 空间能承载 DROID 无动作视频的「未来建模知识」迁移。

### 待核实

- 附录 A.2：δ_1..δ_K 的取值与 K、K/V 层数、predictor/expert 隐藏维度等网络配置（未取到）；B（真机系统）与 C-G（对照/消融细节）未取到。

---

## 7. PALM — arXiv:2601.07060（v2）

- 元信息：PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation。Yuanzhe Liu 等，UIUC/Penn 等；cs.RO，CVPR 2026；v2 2026-04-04。HTML：https://arxiv.org/html/2601.07060v2 ；ar5iv：https://ar5iv.labs.arxiv.org/html/2601.07060
- 来源级别：**L3**（主文 Sec 1-3.4 至 Eq. 10 完整）。**截断**：Eq. 10 之后（progress 标签来源句/后续段落）；Sec 4 实验与附录 A（Training Details）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = 结构化「未来 affordance 预测」latent F̂_{t+n}（4 类子查询）+ 连续子任务进度标量 p_t**（不生成 goal 图像）。Sec 3.2：

  > "Fine-grained affordance queries comprise four subqueries <Global>, <Local>, <Spatial>, and <Dynamic>… and produce the affordance-centric latent
  > F̂_{t+n} = f_aff(l, o_t, s_t) ∈ ℱ.  (1)"

- 4 类 affordance 的**监督来源**（Sec 3.3，逐条）：
  - Global（Eq. 3）："Targets are obtained by first resolving the referent via **Grounding DINO** and then segmenting the instance with **SAM** to obtain a binary mask on the observation lattice." → L_FL（focal）+ L_Dice。
  - Local（Eq. 4-5）："Following **GLOVER++**, annotated contact points are converted into Gaussian heatmaps that serve as continuous supervision targets." → L_FL + L_KL（ℓ1 归一化后的 KL）。
  - Spatial（Eq. 6）："Ground-truth targets are obtained by first converting the instruction into spatial semantics using **SpatialVLM** and then sampling executable 2D coordinates with **RoboPoint**." → 集合匹配损失（min over m 的最近点 L2）。
  - Dynamic（Eq. 7）："an N×N visual grid of query points is placed on the first frame of a short history at t−δ. **CoTracker** follows each point forward in time. We compute cumulative displacement for each trajectory and retain those that exceed a threshold… The selected trajectories are then rasterized at t+n into a dynamic region." → masked VAE 重构损失（负对数似然 + β KL）。
- **progress 标量如何融入**（Sec 3.4 + Eq. 2/8/9/10）：
  > "At each time step PALM first infers the active subtask stage from the affordance latent and derives a stage embedding. Conditioned on this embedding, the model predicts a scalar p_t ∈ [0,1] that quantifies within-stage completion. We append this scalar to the action output so the policy jointly predicts (a_t, p_t) under a shared multimodal context."
  > "(â_{t:t+n−1}, p̂_{t:t+n−1}) = DiT(l, o_t, s_t, F̂_{t+n}),  (2)"
  > "The DiT head jointly models action distributions and progress values… L_DiT = E_{t_d,ε}‖ε − ε_θ(ỹ_{t:t+n−1,t_d} | l, o_t, s_t, F̂_{t+n}, t_d)‖²₂,  (10)
  > where y_{t:t+n−1} is the target action–progress vector over steps t to t+n−1…"（progress 与动作拼接成同一扩散目标联合去噪。）
  > 动机（Sec 3.4）："visually similar observations may correspond to different actions depending on stage, and p_t disambiguates these cases by providing a continuous indicator of 'where we are'"。
  > ⚠️ **p_t 的真值标签来源**（数据侧如何标注 progress）在 Eq. 10 之后的正文/附录 A.2——**未取到**。已获取范围内只有「预测 within-stage completion ∈[0,1]」与联合去噪机制。
- 编码器（Sec 3.2）：CLIP 文本编码器；MAE 视觉编码 + Perceiver Resampler 下采样；state 经轻量 MLP；GPT-2 式 backbone；动作头为 denoising DiT。

### why 追问（基于原文）

1. **为何目标用 affordance 而非 goal 图像/未来帧？** Sec 1：
   > "they lack internal representations that disambiguate which object should be targeted next, which part or region is relevant for interaction, where items should be placed or moved, or what motion is appropriate for the upcoming step."
   Fig. 1 caption："In contrast to vanilla VLAs that directly map inputs to actions or to predictive methods that forecast dense future images, PALM introduces learnable queries to forecast a structured set of future affordances."
2. **为何引入连续 progress 标量？** Sec 1："Without a persistent online notion of advancement, the policy cannot reliably decide whether to continue, switch stages, or terminate."；Sec 3.4（引文见上）：视觉相似但阶段不同时用 p_t 消歧。
3. **为何 4 类 affordance 分别用 4 套外部模型打标？** Sec 3.3：各自覆盖互补尺度——Global 语义对象定位、Local 接触几何、Spatial 放置候选、Dynamic 运动轨迹（"complementary affordance representations that capture object relevance, contact geometry, spatial placements, and motion dynamics"，Abstract）。
4. **为何 affordance 子查询要与 context 解耦？** Fig. 2 caption："Affordance subqueries attend only to shared context tokens to stay disentangled, while both query types use causal attention to preserve temporal consistency."

### 关键数值

- LIBERO-LONG **91.8%**；CALVIN ABC→D 平均长度 **+12.5%**；真机三类泛化 **2×** 于基线
- 预训练数据：BridgeDataV2、DROID（机器人）+ EPIC-KITCHENS、RoboCerebra（长程视频），后接人工标注 affordance 数据微调
- 动作+进度联合输出：horizon-n DiT；progress 空间 P ⊂ [0,1]
- progress 真值标签来源、N×N 网格尺寸、β 等：未取到（截断）

### 一页总结

PALM 的「目标」不是单张图像，而是**四类结构化 affordance 预测**（Global/Local/Spatial/Dynamic，分别由 Grounding DINO+SAM、GLOVER++、SpatialVLM+RoboPoint、CoTracker 提供监督）组成的 latent F̂_{t+n}，外加一个**与动作联合去噪的连续 within-subtask 进度标量 p_t ∈ [0,1]**（Eq. 2/8/10）。条件化路径：affordance 查询→F̂_{t+n}→action-progress 查询→DiT 输出 (â, p̂)；推理时 affordance head 摘除。p_t 充当「阶段消歧」的时间正则器，解决长程任务的重复动作/跳步/提前终止。

### 待核实

- **p_t 的 ground-truth 标签构造**（Eq. 10 之后/附录 A.2）——未取到；Dynamic 的 N、位移阈值、β 数值——未取到；Sec 4 实验表与附录 C（progress 阈值消融）——未取到。

---

## 8. CoT-VLA — arXiv:2503.22020（v1）

- 元信息：CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models。Qingqing Zhao 等，NVIDIA/Stanford/MIT；cs.CV，CVPR 2025；v1 2025-03-27。HTML：https://arxiv.org/html/2503.22020v1
- 来源级别：**L3**（Sec 1-3.3 完整，Eq. 1-6 + Algorithm 1）。**截断**：Sec 4 Experiments 起（ablation 表格数值）；supplementary Sec 6.x（n_l/n_u、超参）未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = 自回归生成的未来图像帧 ŝ_{t+n}（像素空间 visual goal），先「想」后「动」**。Sec 3.1：

  > "our approach operates in two sequential phases:
  > ŝ_{t+n} ∼ P_θ(s_{t+n} | s_t, l)  (2)
  > {â_t, …, â_{t+m}} ∼ P_θ({a_t, …, a_{t+m} | s_t, l, s_{t+n})  (3)
  > where we first predict a subgoal image ŝ_{t+n}, n frames ahead, as an intermediate visual reasoning step (Equation 2). Then we generates a sequence of m actions to achieve this subgoal state (Equation 3). This enables the model to 'think visually' first by explicitly reasoning about desired future states before predicting the actions."
  > 数据分工："The visual reasoning step, Equation (2), is trained on both robot demonstrations D_r and action-less videos D_v, and the action generation step, Equation (3), is trained on robot demonstrations D_r only."

- 序列组织（Sec 3.3）：训练序列 "(l, s_t, s_{t+n})"（视觉步）与 "(l, s_t, s_{t+n}, a_t, …, a_{t+m})"（动作步）；测试（Algorithm 1）：sample ŝ_{t+n} → sample [â_t..â_{t+m}] → 执行 m 步 → t ← t+m+1 → 更新观测（闭环）。
- 视觉 token 化（Sec 3.2）：VILA-U 底座，256×256 输入，每图 **16×16×4** tokens、residual depth D=4（RQ-VAE 深度 transformer）；视觉损失（Eq. 4）：
  > "L_visual = −Σ_j Σ_{d=1}^D log P_δ(k_{jd} | k_{j,<d})"。
- 动作 token 化（Sec 3.3）：每动作 **7 tokens**、每维 256 离散 bins（1st–99th 百分位均分）、复用文本词表中最低频 256 个 token 作 bin token；**动作用 full attention（双向），图像/文本用 causal attention**（hybrid attention，Fig. 3）；L_action = −Σ log P_θ（Eq. 5）；总损失 L = L_action + L_visual（Eq. 6）。
- 目标帧采样（Sec 3.3 Pretraining）："we use subgoal images at future timestep n uniformly sampled from a dataset-specific range [n_l, n_u]"；"We use an action chunk size of 10."（n_l/n_u 数值在 supplementary，未取到。）

### why 追问（基于原文）

1. **为何用 subgoal 图像做中间推理（而非文本 CoT/关键点）？** Sec 1：
   > "we explore subgoal images as an intermediate reasoning step before action generation. These images capture the state of the model's reasoning process and are naturally available within robot demonstration datasets."
   Sec 2："This approach leverages demonstration videos as natural intermediate reasoning states without requiring additional annotations."（对比：文本描述/关键点/框需要额外预处理管线。）
2. **为何 subgoal 生成可以吃无动作视频？** Sec 1："since subgoal image generation does not require action annotations, this unlocks the potential of using abundant video data for improved visual reasoning and understanding."（EPIC-KITCHENS-100、SSv2 进视觉步训练。）
3. **为何动作用 full attention、视觉用 causal？** Sec 1："we use causal attention with next-token prediction for text and image generation, and leverage full attention to predict all action dimensions at once."（生成式因果 vs 动作维度的并行交互；图 3 有 [x]/[θ]/[g] 特殊 token。）
4. **为何自回归生成目标而非扩散/潜空间？** Sec 3.2：直接继承 VILA-U 的「统一离散 token + 自回归 next-token」框架，视觉生成与理解同一骨干，无需另接生成器。

### 关键数值

- 底座 **7B**（VILA-U）；动作块 m=**10**；每动作 7 tokens、256 bins；图像 256×256 → 16×16×4 tokens（D=4）
- LIBERO 平均 **81.13%**（vs OpenVLA fine-tuned 76.5%、Diffusion Policy 72.4%）；真机 +17%、仿真 +6%（vs SOTA，Abstract）
- 训练：LLM backbone + projector + depth transformer 可训，vision tower 冻结；OpenX（第三方视角、单臂 7-DoF）+ EPIC-KITCHENS + SSv2
- n_l/n_u（subgoal 视野范围）数值：supplementary 未取到

### 一页总结

CoT-VLA 的「目标」是**自回归生成的 n 帧后 subgoal 图像 ŝ_{t+n}**（像素空间、VILA-U 离散 token）：先以 causal attention 按式 (2) 生成 visual goal，再以 full attention 依式 (3) 条件 (s_t, l, ŝ_{t+n}) 生成 m=10 步动作块，闭环滚动（Algorithm 1）。训练上，视觉步可吃无动作视频（EPIC-KITCHENS/SSv2），动作步只吃机器人演示；混合注意力（因果/双向）与动作 chunking 是两项关键设计。目标表征的选择动机：演示视频天然提供中间推理状态、无需额外标注。

### 待核实

- n_l/n_u 数值与数据集相关超参（Sec 6.x，截断）；Sec 4.3 消融（Visual CoT/Hybrid Attention/Chunking/Pretraining 各贡献数值，截断）；Sec 4.4「更好的视觉推理→更好的动作」数值（截断）。

---

## 9. SWEET — arXiv:2605.19319（v1）

- 元信息：SWEET: Sparse World Modeling with Image Editing for Embodied Task Execution。Yiren Song 等，NUS Show Lab；cs.CV；v1 2026-05-19。HTML：https://arxiv.org/html/2605.19319v1
- 来源级别：**L3**（Sec 1-3.6 完整 + Sec 4.1/4.2 至效率讨论）。**截断**：Table 1（FLUX-Kontext vs Wan2.2 的 MSE/PSNR/SSIM/LPIPS 数值）与 Sec 4.3/4.4 未取到。

### 目标表征机制（原文 + 指路）

**目标形式 = 图像编辑模型逐帧生成的「任务关键帧序列」（稀疏视觉计划），相邻帧对作视觉目标喂扩散策略**。

- 问题定义（Sec 3.1）：
  > "Each trajectory is represented by a sparse sequence of task-relevant keyframes 𝒦 = {k_0, k_1, …, k_M}… These keyframes summarize the main semantic stages of a manipulation task, such as approach, contact, grasp, transport, and release."
  > "𝒦̂ = E_θ(o_0, 𝒞),   Â_{t:t+H−1} = π_φ(o_t, k̂_{m+1}).
  > Here, E_θ denotes the recursive application of the image-editing planner over the subtask conditions, π_φ denotes the goal-conditioned action predictor, and H is the action horizon."
- 关键帧监督来源（Sec 3.2）：**人工标注**：
  > "For each episode, we manually annotate a sparse set of task-critical keyframes… each annotated transition contains (k_m, k_m^arr, l_m, k_{m+1}, A_m). We use (k_m^arr, l_m, k_{m+1}) to supervise the image-editing planner, while A_m is used for training the subgoal-conditioned action predictor."
  > "we adopt human annotation for keyframe selection in this work"（VLM 自动抽取关键帧不可靠；箭头可让 Gemini-3 生成）。
  - 箭头空间提示："The spatial cue is rendered on the source keyframe as an arrow-overlaid image k_m^arr, indicating the coarse image-space motion direction of the gripper. Its color encodes the gripper-state change, such as open-to-close or close-to-open, and a circle is used instead of an arrow when the gripper mainly acts in place with negligible displacement."
- 规划器（Sec 3.3，Eq. 1）："k̂_{m+1} = E_θ(k_m, l_m, r_m).  (1)"；底座 **FLUX Kontext（FLUX.1-Kontext-dev）LoRA 微调**；"we randomly drop the arrow condition during training while always keeping the text description"；"We optimize the planner with the native image-editing objective of FLUX Kontext and update only the inserted LoRA parameters."
- **相邻帧对如何条件扩散策略**（Sec 3.4）：
  > "Given the current observation o_t and a target keyframe k̂_{m+1}, the predictor outputs an action chunk:
  > Â_t = π_φ(o_t, k̂_{m+1}).  (2)"
  > "we adopt the Diffusion Policy architecture and modify it into a start–goal conditioned policy… The current observation and predicted subgoal image are encoded independently by a visual encoder, and their features are concatenated as the conditioning input to the action diffusion model. The policy predicts action chunks of length 16"（H=16）。
  > 训练（Eq. 4）："φ* = argmin_φ L_act(o_t, k_{m+1}, A_t^H)"（标准 DDPM 去噪目标，receding-horizon）；"each subgoal is tracked in a closed-loop, receding-horizon manner"（计划开环一次生成、逐 subgoal 闭环跟踪）。
- 真实↔编辑 domain gap 处理（Sec 3.5）：
  > "we construct synthetic edited transitions. For each ground-truth transition, we use the visual planner to generate an edited target keyframe k̃_{m+1}, and pair it with the same action supervision from the original transition… we filter low-quality edited samples according to visual consistency and task plausibility… We then combine these filtered edited transitions with the original clean transitions, and train the action predictor with a 1:1 mixture of both."
- 推理（Sec 3.6）：一次性生成全部关键帧（one-shot 稀疏计划），逐子目标闭环执行；"For more complex procedural or long-horizon tasks, the same planner–predictor interface can be naturally extended to a feedback-enabled setting."

### why 追问（基于原文）

1. **为何用图像编辑而非视频生成做世界模型？** Abstract + Sec 4.2（对照实验）：
   > "we conduct a controlled comparison between the video generation model Wan2.2 and the image editing model FLUX-Kontext under the same robotic data setting, and find that image editing produces more reliable task-level keyframes with better visual fidelity and substantially lower inference cost."
   Sec 2.3："Such supervision encourages the model to preserve the source scene while applying semantically meaningful changes, making it naturally aligned with robotic keyframe prediction: the background, object identity, and scene layout should remain stable, while the robot and task-relevant objects evolve according to language or spatial guidance."
2. **为何稀疏关键帧而非密集 rollout？** Abstract："dense video generation is computationally expensive and often unnecessary for many manipulation tasks, whose progress can be summarized by a small number of task-relevant visual states."（效率差：3 帧计划约 10 s vs Wan2.2 的 81 帧 >400 s，Sec 4.2。）
3. **为何加箭头空间引导？** Sec 1："The arrow provides a coarse high-level spatial hint, which can be specified by a user or predicted by a vision-language or vision-reasoning model."（语言之外的低成本空间提示，可随机丢弃以支持纯语言。）
4. **为何做 mixed training？** Sec 3.5："During standard training, the predictor observes clean transitions… At inference time, however, the target subgoal is generated by the image-editing planner… shifting the input distribution of the action predictor. To reduce this mismatch…"

### 关键数值

- 动作块 H=**16**；LoRA rank **32**、6K 步、lr 1e-4（规划器）；action predictor lr 1e-4、batch 32
- 编辑步数：RoboMimic Lift=**2**（接触、抬起）、Can/Square=**3**；DROID 全部 **3**
- 数据：DROID 700 样本/约 **2,100** 标注子任务（10+ 环境）；RoboMimic 600 样本/**1,600** 子任务；每任务 200 次评估
- 效率：FLUX-Kontext 3 关键帧计划约 **10 s** vs Wan2.2 81 帧 **>400 s**（H20）
- Table 1 的 MSE/PSNR/SSIM/LPIPS 数值、Sec 4.3 执行成功率：截断未取到

### 一页总结

SWEET 的目标 = 图像编辑模型（FLUX-Kontext，LoRA）从初始帧**逐帧编辑出的任务关键帧序列**（语言 + 可选箭头空间提示条件），构成稀疏视觉计划；随后每个「相邻关键帧对 (o_t, k̂_{m+1})」经独立视觉编码 + 特征拼接，条件一个 Diffusion-Policy 式动作预测器输出 H=16 的动作块（Eq. 2/4），逐子目标闭环滚动。关键帧由人工标注（approach/contact/grasp/transport/release 等语义阶段），动作监督取两关键帧间的动作段；1:1 混合「真实 + 过滤编辑」过渡缓解 real-to-edited 分布差。主张：图像编辑（而非视频生成）是更可靠、更便宜的任务级稀疏世界模型。

### 待核实

- Table 1 定量数值（FLUX-Kontext vs Wan2.2 的 MSE/PSNR/SSIM/LPIPS，截断）；Sec 4.3 RoboMimic 全管线成功率与 endpoint MSE、Sec 4.4 消融（截断）；关键帧过滤的具体标准/阈值（Sec 3.5 只给定性描述）。

---

# 汇总对比表（9 篇 × 目标形式 × 编码器 × 条件化位置 × 原文理由）

| # | 论文 | 目标形式 | 编码器 | 条件化位置 | 原文理由（关键词） |
|---|---|---|---|---|---|
| 1 | JEPA-WAM 2609.20277 | 每指令生成图像 bank（K=1024）的 latent 分布（Eq. 1/3） | 冻结 V-JEPA 2.1（ViT-B/16, 80M）+ Pool 8×8 + 可学习投影 → 64 tokens（Eq. 2） | 条件序列 C=[C_ℓ; P_s(s); Z_ℓ]（Eq. 4），cross-attention 进 video 与 action 两 expert | 文本稀疏重复→捷径；生成图外观偏→JEPA「预测性 latent + 显式空间结构」抑制外观差异（Sec 1/2.3） |
| 2 | Act2Goal 2512.23541 | image-goal（纯视觉），VAE latent z_g 沿序列拼接（Eq. 1） | Genie-Envisioner 的 VAE（video encoder；型号 V-A 未取到） | z_g 与 z_t 拼接条件 flow-matching 世界模型；世界模型逐层特征 c_w 经 cross-attention 进同构 Action DiT（Eq. 3） | 视觉目标「直接编码物体构型/空间关系/终态约束」，避开语言歧义与奖励工程（Sec I） |
| 3 | NavWAM 2606.13494 | image-goal（goal 位置拍摄图），9 帧 canvas 第 3 帧 | Cosmos Predict2 causal VAE（视频 latent）；非视觉变量广播到 latent 网格 | 底排 4 帧条件联合去噪；goal 帧与 state/obs 同序列（Sec 4.1）；value 帧为预测目标 | 导航需「视角变化的前视 + 接近目标的进度」，共享 latent 序列免外部规划器（Sec 2/4） |
| 4 | GR-MG 2408.14368 | InstructPix2Pix 生成的 N 步后 subgoal 图像 + 文本双条件 | MAE 编码 goal image（token 前置）；T5-Base 编码进度文本 | goal tokens 置输入序列前端、可被所有后续 token attend（Sec III-B2） | 生成图不准时文本兜底（Sec I）；进度句解决静态编辑 vs 序贯决策的矛盾（III-B1） |
| 5 | LaWAM 2606.15768 | latent visual subgoal û_T = LaWM_ω(u, ẑ)（未来观测特征） | 冻结 VFM（如 DINOv3）特征空间；潜动作 z 由 q_φ(z\|u,u_T) 推断 | 策略先验 p_θ(ẑ\|o,l)→LaWM 解码→(u, û_T) 为动力学流进 Alternate-DiT 动作专家（Eq. 3） | 像素视频冗余且慢；「动作块相关的状态变化」而非「视觉上合理的未来」（Sec 1） |
| 6 | V-JEPA Policy 2609.37250 | 未来 K 个观测的 V-JEPA latent Z_t^+（Eq. 2） | 冻结 V-JEPA 2.1 ViT-L（目标路径完整 clip；context 路径删未来 token） | predictor 逐层 context K/V（3D RoPE 旋转键）作 C_φ 喂 flow-matching 动作专家（MoT，Eq. 6） | 预测性 latent 空间或可替代完整视觉生成器（Sec 1 核心问题）；分布偏移下更稳（Sec 5.3） |
| 7 | PALM 2601.07060 | 4 类 affordance latent F̂_{t+n} + 连续进度标量 p_t∈[0,1] | CLIP（文本）、MAE+Perceiver（视觉）、MLP（状态）；GPT-2 式 backbone | affordance 查询→F̂_{t+n}→action-progress 查询→DiT 联合去噪 (â,p̂)（Eq. 2/8） | 直接映射缺乏「下一对象/接触区/放置点/运动」的显式表征；p_t 消除阶段歧义（Sec 1/3.4） |
| 8 | CoT-VLA 2503.22020 | 自回归生成的 subgoal 图像 ŝ_{t+n}（像素空间，n 帧后） | VILA-U 统一离散视觉 token（256×256→16×16×4，RQ-VAE 深度 4） | 序列 (l, s_t, ŝ_{t+n}, a_t..a_{t+m})；视觉/文本 causal、动作 full attention（Eq. 2/3，Alg. 1） | 演示视频天然是中间推理状态、免额外标注；先「视觉思考」再行动（Sec 1） |
| 9 | SWEET 2605.19319 | 图像编辑逐帧生成的任务关键帧序列（稀疏计划） | FLUX-Kontext（LoRA rank 32）；动作侧独立视觉编码器 | 相邻帧对 (o_t, k̂_{m+1}) 各自编码后特征拼接，条件 Diffusion Policy（Eq. 2/4） | 图像编辑比视频生成「更可靠的关键帧 + 更高保真 + 更低推理成本」（Abstract/Sec 4.2） |

---

# 趋势与争议（仅基于原文）

## 趋势

1. **目标载体谱系：像素 ↔ latent ↔ 结构化信号**。一端是像素目标：Act2Goal、NavWAM（image-goal）、GR-MG、CoT-VLA、SWEET（生成/真实图像目标与关键帧）；中间是预测性视觉 latent：JEPA-WAM、V-JEPA Policy（V-JEPA 系）、LaWAM（DINOv3 系）；另一端是结构化目标：PALM（4 类 affordance + progress 标量，完全不做图像目标）。三篇明确拒绝「纯文本目标」：JEPA-WAM（语言稀疏重复→捷径）、Act2Goal（语言缺细粒度）、GR-MG（语言与视觉异域且抽象）。
2. **目标的两种职能**：①「任务规格」（goal specification：Act2Goal、NavWAM、SWEET、GR-MG 的 goal image 定义「要什么」）；②「动态预测/中间推理」（LaWAM 的 û_T、V-JEPA Policy 的 Z_t^+、CoT-VLA 的 ŝ_{t+n} 是「未来会怎样」的模型预测）。JEPA-WAM 二者兼具（生成图=规格，JEPA latent=语义抽离）。
3. **条件化方式收敛到 cross-attention 及其变体**：JEPA-WAM（拼接序列 + 双 expert cross-attention）、Act2Goal（世界模型层特征 c_w cross-attention）、V-JEPA Policy（layer-wise context K/V + MoT）、LaWAM（Alternate-DiT 双流）；序列前置拼接（GR-MG 的 MAE tokens）、自回归序列内联（CoT-VLA）、共享 latent 帧联合去噪（NavWAM）、特征拼接 + 扩散（SWEET、PALM 联合去噪）。
4. **progress/value 标量作为目标附属信号兴起**：NavWAM（goal-progress value 帧）、GR-MG（[PROG] token + 10 档）、PALM（连续 p_t 联合去噪）都把「距离目标多近」作为与目标共存的连续监督；三篇的 progress 标签都指向轨迹时间进度（GR-MG 明写「timestep in the trajectory」，NavWAM/PALM 的精确标签公式未取到）。
5. **生成目标 vs 真实观察的 domain gap 被显式治理**：SWEET（过滤 + 1:1 混合训练）、GR-MG（文本兜底）、JEPA-WAM（JEPA 编码抽离外观差异）、NavWAM（不用生成目标，直接用真实 goal 图）。
6. **目标监督来源分层**：现成生成器（DreamLite、InstructPix2Pix、FLUX-Kontext、VILA-U）无需新标注；离线专家模型自动打标（PALM 的 4 类 affordance）；人工标注（SWEET 关键帧）；轨迹时间步自监督（GR-MG 进度、LaWAM 的 u_T、V-JEPA Policy 的未来 latent）。

## 争议

1. **「要不要像素」分歧**：LaWAM 断言像素空间 WAM「把大量建模能力花在像素级合成上」、SWEET 用实验主张图像编辑优于视频生成、V-JEPA Policy 主张无需继承视觉生成器；但 JEPA-WAM 仍需现成 T2I 生成器（DreamLite）产生目标图、CoT-VLA/SWEET/GR-MG 都以生成图像为中间目标。效率与任务适用性（指令跟随 vs 效率优先）可能是分歧根源。
2. **未来预测的「位置」**：V-JEPA Policy 刻意用 predictor 的逐层 context K/V（而非最终未来 latent）条件动作；LaWAM 用最终 subgoal 特征；Fast-WAM（被 JEPA-WAM 引用为基线）在测试时直接去掉未来生成——「未来信号放在哪里、要不要保留」尚无共识。
3. **goal-progress 标签的监督定义不透明**：NavWAM 的 v_{t+H} 与 PALM 的 p_t 的标签公式均在我截断位置之外（附录 A.2 / Eq.10 之后），已获取范围只到机制层——这是本线最重要的跨论文待核实项。

---

# 全局待核实清单（本线）

1. NavWAM 附录 A.2：goal-progress value v_{t+H} 的标量标签构造（监督来源）——未取到（截断于 References 之后）
2. PALM：progress p_t 的真值标签来源（Eq. 10 之后 / 附录 A.2）——未取到
3. GR-MG：N（goal 提前步数）与 n（生成间隔）数值——原文未写；policy 侧 [PROG] 回归标签公式——原文未写明
4. Act2Goal V-A：VAE 型号、MSTH 超参 K/P/r/M、去噪步数 N——未取到（同 README 全局 #5）
5. LaWAM 附录 C.1/C.2/C.5：z 维度、τ、λ_distill/λ_wm、β——未取到（截断于 Table 1）
6. JEPA-WAM 附录 A/C：prompt 模板、λ_v/λ_a——未取到（截断于 Sec 5.2）
7. V-JEPA Policy 附录 A.2：δ_1..δ_K 数值与网络配置——未取到（截断于 Sec 5.1）
8. CoT-VLA supplementary 6.x：n_l/n_u 数值——未取到（截断于 Sec 4）
9. SWEET Table 1：MSE/PSNR/SSIM/LPIPS 数值、Sec 4.3 执行成功率——未取到（截断于 Table 1）

<!-- QA_LOG -->
