# WAM 扩展精读线 A：输入/目标表征（第二梯队）· 重发

> 生成时间：2026-10-27（Asia/Shanghai）。本文件为增量落盘版本；每完成 2 篇即写入，最后统一 polish。
> 铁律说明：结论只依据论文原文或代码事实；引文为英文 verbatim + 小节号；数值逐字抄录；找不到写「原文未写（已获取范围）」或「未取到（截断位置）」。
> 来源级别定义：**A** = arXiv HTML 全文获取成功且正文完整；**B** = 获取不完整（标注截断位置）；**C** = 仅 abs/摘要页。取证路径：arxiv.org/abs/<id> → arxiv.org/html/<id>（最新版）→ ar5iv.labs.arxiv.org/html/<id>；HTML 本地化后抽文本（lxml）供逐字引用。

---

## 1. GR-2（arXiv 2410.06158）

### 元信息
- 标题：GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation
- 版本：v1，2024-10-08；Tech Report（CC BY 4.0）；机构：ByteDance Research；作者按字母序
- 主页：https://gr2-manipulation.github.io
- 来源级别：**A**。arxiv.org/html/2410.06158v1 获取成功，§1–§5 正文完整；web_fetch 输出仅截断于 References 列表末尾（对输入/目标表征无影响）。本地备份：`auto_docs/reading/txt/2410.06158.txt`（49,540 字符）、`auto_docs/reading/pdfs/2410.06158.pdf`（16.2MB，未做文本抽取）。

### 【输入表征】
- 形式化（§2 Methods，Eq. (1)）：
  > "Specifically, we want to train a universal policy π that takes as inputs a language instruction l, a sequence of environment observation 𝐨_{t−h:t}, and a sequence of robot states 𝐬_{t−h:t}. The policy outputs an action trajectory 𝐚_{t:t+k} in an end-to-end manner: 𝐚_{t:t+k}=π(l,𝐨_{t−h:t},𝐬_{t−h:t}), where h and k denote the length of the observation history and the action trajectory, respectively."
- 各模态编码（§2.1 Model & Training）：
  > "We use a frozen text encoder [6] to tokenize the language instruction. For the image frames in the video, we employ a VQGAN [7] to convert each image into discrete tokens. The VQGAN is trained on a large corpus of Internet data as well as in-domain robot data and is kept frozen during the training process."（[6]=CLIP 文本编码器，参考文献列表可见；[7]=VQGAN）
  > "Robot states contain the position and rotation of the end-effector, as well as the binary gripper state. The states are encoded via linear layers, which are trainable during the fine-tuning stage."
- 多视角与输出（§2.1）：
  > "It takes as inputs the tokenized language instruction, the image sequences captured from multiple views, and the robot state sequence. The outputs include future images of each view and an action trajectory. The action trajectory is generated with a conditional VAE (cVAE) [17,18,19]. We empirically found that generating action trajectories rather than single-step actions is crucial for both trajectory smoothing and real-time performance."
- 任务设定（§2）： "We consider language-conditioned visual robot manipulation as our approach towards generalist robot manipulation, as language is one of the most flexible ways for a human to specify tasks for a robot."

### 【目标表征】
- 显式 goal 条件：**原文未写（已获取范围：HTML v1 全文）**。GR-2 不使用 goal image / goal embedding 条件；任务由 language instruction 指定。
- 隐式「未来视觉即目标/规划」依据（§3.4 Autoregressive Video Generation）：
  > "GR-2 is pre-trained on a vast number of diverse videos, enabling it to predict future states within the image space. As a result, this video generation capability can effectively act as a planner for action generation. That is, after generating the visual trajectory, an action trajectory can be subsequently inferred based on the visual trajectory."
  > "We highlight that the generated videos align with the real-world rollouts faithfully. This indicates that the predicted action is trying to 'replay' the trajectory in the predicted video."
- 对 goal-image 路线的态度（§4 Related Work）： "Another line of works proposed to condition the policy with a goal image instead of a language [45–49]. And previous methods have also explored aligning the latent space of goal images and languages to enable both goal image condition and language condition during training [27,31,33]. GR-2 is a language-conditioned generalist robot manipulation agent."（即：GR-2 明确选择语言条件而非 goal 条件。）

### 【代码依据】
- 论文全文（HTML v1）未提供 GitHub 链接（获取范围：§1–§5 及 References）。项目页 https://gr2-manipulation.github.io 存在（abs 页标注）。
- 待核实：项目页/仓库是否含代码（留待后续 GitHub 取证轮）。

### 【why 追问】
1. 为何预测未来帧能帮助动作预测？原文动机（§2.1）是「未来视频预测先验 → 更准确的动作预测」： "Our goal in the pre-training stage is to equip GR-2 with the capability to predict future videos. This enables the model to develop a strong prior for predicting future events, thereby enhancing its ability to make accurate action predictions."——但原文未提供机制性解释（如预测帧内潜表征如何被动作头消费）；cVAE 动作头与视频 token 的耦合方式（同一 transformer 输出两种 token？损失如何加权？）在 HTML 全文未给出细节（未写）。
2. 语言指令 token 由冻结 CLIP 编码，机器人状态由可训练线性层编码——为何状态在微调阶段才可训练？原文只写 "trainable during the fine-tuning stage"（§2.1），未解释理由。
3. §3.4 称「改进视频生成即可持续改进动作预测」（"This property brings about a simple approach to continuously improving action prediction by iteratively improving video generation."）——是否意味着动作与视频预测在训练中强耦合、且视频质量是动作性能的上界？原文未给出该上界的定量验证。

### 【关键数值】（逐字抄录）
- 预训练数据：38 million video clips、over 50 billion tokens（Abstract）；GR-1→GR-2 预训练视频从 0.8 million 增至 38 million（§4）。
- 模型规模：default 230M total / 95M trainable（§3）；Scaling 四档 trainable：30M (GR-2-S)、95M (GR-2-B)、312M (GR-2-L)、719M (GR-2-XL)（§3.5）。
- 多任务：105 tasks / 8 skills；~40,000 trajectories，平均 400/task；数据稀缺设置约 1/8 数据 ≈ 50 trajectories/task（§3.1）。
- 成功率（§3.1）：Simple 97.7%（105 tasks）；Unseen Backgrounds 71.4%、Unseen Environments 71.7%（"doubling those of GR-1"）；GR-2 w/ DA：Unseen Environments 87.0%、三个泛化设置平均 74.7%；50 traj/task 时 Simple 73.9%；Unseen Manipulation 55.8%。
- Bin picking（§3.2）：GR-1 33.3% → GR-2 79.0%（平均成功率）；训练 55 objects / ~94,000 pick-and-place trajectories；评测 122 objects（55 seen + 67 unseen）；Cluttered 设置物体数 12–17（两倍于训练）。
- CALVIN（§3.3）：1 task 94.9%→98.6%；5 tasks 73.1%→85.9%；average length 4.21→4.64。
- 部署（§2.2）：7-DoF Kinova Gen3 + Robotiq 2F-85；双相机（静态 head + 末端腕部）；WBC 以 200 Hz 执行关节动作。

### 【一页总结】
GR-2 是 ByteDance 的生成式 VLA 技术报告：GPT 式 transformer，把「语言指令（冻结 CLIP token）+ 历史视频帧（VQGAN 离散 token，微调时多视角）+ 机器人状态（EE 位姿 + 二值夹爪，线性层）」作为输入，同时自回归输出「下一帧视频 token + 动作轨迹（cVAE）」。没有显式 goal 表征；其「目标」功能由语言指令（任务说明）与「预测视频作为动作 planner」（§3.4）隐式承担。核心卖点是 38M clips/50B tokens 的视频生成预训练带来的泛化（unseen bg/env 相对 GR-1 翻倍）。对「WAM 输入/目标表征」调查的意义：目标=语言指令；未来视觉=内部规划器而非输入条件。

### 【待核实】
- cVAE 动作头与视频 token 的具体接口/损失权重：HTML 全文未写。
- 历史长度 h、动作轨迹长度 k 的数值：原文未写。
- 代码仓库是否存在（bytedance/GR-2？）：待 GitHub 取证。

---

## 2. VPP（arXiv 2412.14803）

### 元信息
- 标题：Video Prediction Policy: A Generalist Robot Policy with Predictive Visual Representations
- 版本：v2，2025-05-04（v1 2024-12-19）；ICML 2025 Spotlight Paper；作者：Yucheng Hu, Yanjiang Guo（equal contribution）等（Tsinghua IIIS / Shanghai AI Lab / RobotEra / UC Berkeley）
- 主页：https://video-prediction-policy.github.io
- 来源级别：**A**。arxiv.org/html/2412.14803v2 获取成功；首次 web_fetch 输出截断于 §5.1 中部（Table 2 附近），后经本地 HTML→txt 全量抽取补齐（HTML 273,032 字节；txt 66,599 字符 / 911 行）。本地备份：`auto_docs/reading/txt/2412.14803.txt`、`auto_docs/reading/pdfs/2412.14803.pdf`（18.3MB）。

### 【输入表征】
- 核心概念「predictive visual representations」（§1 Introduction）：
  > "These representations are typically structured as a tensor with dimensions (T,H,W), explicitly representing 1 current step and (T−1) predicted future steps ([Blattmann et al., 2023a]), where H and W correspond to the height and width of the image representation. In contrast, previous vision encoders do not explicitly capture future representations… Based on this distinction, we refer to these latent variables within the video diffusion model as 'predictive visual representations'."
- TVP 模型改造（§4.1）：基座 SVD 1.5B，加入 CLIP 语言特征、视频分辨率改为 16×256×256：
  > "We chose the open-sourced Stable Video Diffusion (SVD) model ([Blattmann et al., 2023a]) with 1.5 billion parameters as our foundation. … We augment the model to incorporate CLIP ([Radford et al., 2021]) language feature l_emb using cross-attention layers. Furthermore, we adjust the output video resolution to 16×256×256… In this setup, the initial observation s0 is concatenated channel-wise with each predicted frame as a condition."
- 单步前向作「编码器」（§4.2）： "Specifically, we concatenate the current image s0 with the final noised latent q(x_{t′}|x0) (typically white noise) and input this combination into the TVP model. We then directly leverage the latent features."
- 特征聚合 L_m → F_p（§4.2，Eq.）：up-sampling 层特征插值到 Wp×Hp 后沿通道拼接： "The final predictive visual representation F_p∈ℝ^{T×(∑_m C_m)×Wp×Hp} is given by: F_p=concate((L′0,L′1,…,L′m),dim=1)"；多视角时分别预测 static 与 wrist 视图 F_p^static, F_p^wrist。
- Video Former（§4.2，Eq. (5)）：learnable tokens Q[0:T,0:L] 对每帧做 Spat-Attn（输入为 (F^static_p[i], F^wrist_p[i])），再做 Temp-Attn + FFN 得到 Q′′；diffusion policy 头以 Q′′ 与 l_emb 为条件输出动作序列 a0（§4.2 Action Generation）。
- 推理节奏（§5.1 Policy Roll-out Details）： "our approach uses the TVP model as an encoder rather than a denoiser, ensuring that each observation is processed through the TVP model only once, which takes less than 160 ms. … achieve a significantly higher frequency of 7-10 Hz with consumer-level NVIDIA RTX 4090 GPU. Additionally, we implement action chunking (Chi et al., 2023) with 10 steps."

### 【目标表征】
- 显式 goal 表征：**原文未写（已获取范围：HTML v2 全文）**。VPP 不用 goal image/embedding 条件；任务由语言指令（CLIP l_emb）指定。
- 隐式「未来视觉即目标」依据（§1）：
  > "Our key insight is that the downstream policy can implicitly learn the inverse dynamics model by tracking the robot's movements within the predictive representation. As long as the video model accurately predicts future scenarios for diverse tasks, the policy can generate appropriate actions by tracking robot arm's position implicitly."
- 与 goal-conditioned 方法的对比（§2 Related Works）： "SuSIE ([Black et al., 2023]) conditions its control policy on a predicted future keyframe generated by InstructPix2Pix… while UniPi ([Du et al., 2024]) learns the inverse dynamics between two generated frames. These methods rely on a single future prediction step to determine actions… Additionally, they denoise the final future image which is time-consuming and lead to low control frequency."（即 VPP 明确以「多帧未来潜表征」取代「单帧 goal/keyframe」。）
- 泛化机制解读（§5.3 Generalization Analysis）： "the low-level policy learns a robust inverse dynamics model that only needs to implicitly track the movement of the robot in the predicted future, without the need to focus on new objects or backgrounds."

### 【代码依据】
- 论文全文未见 GitHub 链接（获取范围：HTML v2 全文）；项目页 https://video-prediction-policy.github.io（abs 页标注）。待 GitHub 取证。

### 【why 追问】
1. 为何「T 帧潜表征（1 当前 + T−1 未来）」比单帧 goal 好？原文论据是单帧方法 "may not accurately capture the complexities of physical dynamics"（§2），且单步表征已含运动信息（§5.1 Figure 4："single-step representation already conveys valuable information, such as the movement of objects and the robot arm"）。但未给「T 的选择（16 帧）与下游成功率」的消融曲线——16 帧是否为关键超参，原文未写。
2. 第一前向步（近乎白噪声输入）的潜表征为什么已经够用？原文自称 "The first forward step, while not yielding a clear video, still provides a rough trajectory of future states and valuable guidance"（§4.2），并说 "This insight is verified in our experiment section and shown in Fig 4"——但 §5 只有可视化，没有「one-step vs 30-step 表征」的下游成功率定量对比（检索 HTML 全文未见此类表格）。
3. 语言指令 l_emb 同时注入 TVP 与 diffusion policy 头（Eq. (6) 条件含 l_emb）——目标语义与未来视觉如何分工？原文未做剥离实验（未写）。

### 【关键数值】（逐字抄录）
- CALVIN ABC→D（§5.1 + Table 1）：VPP avg length 4.33；全文行文称 "improved the previous state-of-the-art result from an average task completion length of 3.35 to 4.33"；Table 1 中 Vidman 3.42、RoboUniview 3.65、GR-1 3.06；10% ABC 数据时 VPP 3.25（超全部 100% 基线）。注：Abstract 写 "18.6% relative improvement on the Calvin ABC-D … compared to the previous state-of-the-art"（与 3.65→4.33 一致）；§1 写 "41.5% improvement in the Calvin ABC→D benchmark compared to the previous SOTA method"（与 3.06→4.33 一致）——两处口径不一致（见【待核实】）。
- MetaWorld（§5.1）：50 tasks 单策略；"outperforming the strongest GR-1 baseline by 10.8% in average success rate"（Table 2 逐任务列于 §5.1）。
- 消融（Table 3）：视觉表征替换为 Stable-VAE 2.58 / VC-1 1.23 / Voltron 1.54（VDM 4.33）。
- 消融（Table 4）：w/o Internet data 3.97；w/o Calvin video 3.31；w/o Internet & w/o SVD Pretrain 1.63；w/o Video Former 3.86（延迟 ~450ms）；w/o Feature Agg 3.60（延迟 ~140ms；完整 VPP ~140ms）。
- 数据与算力（§5.1）：TVP 微调用 193,690 human manipulation trajectories（SSv2）+ 179,074 robotic manipulation trajectories（DROID）+ 下游任务视频；2–3 days on eight A100；第二阶段策略 6–12 hours on four A100。
- 真机（§5.3 + Table 5）：Franka Panda 2,000 trajectories / 30+ tasks / 6 categories，seen 0.85（DP 0.42 / Susie 0.56 / GR-1 0.52）、unseen 0.73（0.25 / 0.46 / 0.38）；Xarm + 12-DoF Xhand 4,000 trajectories / 100+ tasks / 13 categories，seen 0.75（0.28 / 0.45 / 0.32）、unseen 0.60（0.11 / 0.28 / 0.15）、tool-use 0.68（0.05 / 0.23 / 0.15）；评测 200+ rollouts（Panda）、500+ rollouts（dexterous）。

### 【一页总结】
VPP 把预训练视频扩散模型（SVD 1.5B，改为文本条件、16×256×256）当作「预测性视觉编码器」：一次前向得到 (T,H,W) 的「当前+未来」潜表征，Video Former 聚合成固定 token，Diffusion Policy 头以此输出动作序列（chunk=10，~140ms/7–10Hz，单张 4090）。没有显式 goal 条件；其目标机制是「在预测的未来里隐式跟踪机器人运动学」的逆动力学学习。对 WAM 调研的意义：目标=语言指令 + 隐式未来视觉（作为目标状态的表征性替代），是「future representation 即目标」路线的早期代表。

### 【待核实】
- Abstract（18.6%）与 §1（41.5%）的 CALVIN 提升口径不一致：18.6%=(4.33−3.65)/3.65（vs RoboUniview），41.5%=(4.33−3.06)/3.06（vs GR-1）——两者都能由文中数值算出，但「previous SOTA」指代不同，且 §5.1 行文写 3.35（与 Table 1 的 3.42/3.65 均不符）。待与 v1 或官方版本核对。
- 帧数 T=16 与动作 chunk=10 的对齐关系：原文未写。
- 代码仓库：待 GitHub 取证。

---

## 3. UVA（arXiv 2503.00200）

### 元信息
- 标题：Unified Video Action Model
- 版本：v3，2025-04-24（v1 2025-02-28）；CC BY 4.0；作者：Shuang Li, Yihuai Gao, Dorsa Sadigh, Shuran Song；主页 https://unified-video-action-model.github.io
- 来源级别：**A**。arxiv.org/html/2503.00200v3 本地化成功（HTML 253,576 字节；txt 80,554 字符 / 471 行，§I–§X 与 References 完整）。本地备份：`auto_docs/reading/html/2503.00200.html`、`txt/2503.00200.txt`、`pdfs/2503.00200.pdf`（4.1MB）。

### 【输入表征】
- 统一 video-action token 序列（§III Problem Statement + §III-A）：输入为历史观测 {𝐎t−h+1,…,𝐎t} 与动作 chunk {𝐀t−h,…,𝐀t−1}；每帧经 VAE 编码为 N 个 d 维视觉 token，动作 chunk（每帧对应 L 个高频动作）重复 M 次后经 FC 转为 N 个 d 维动作 token：
  > "We first process the historical image observations through a pre-trained VAE encoder (kl-f16) [35] to obtain their latent representations. … We repeat the action chunk M times to match the number of visual tokens as shown in Figure 2."（§III-A）
- 未来帧同法编码并随机 mask 作为输入（§III-B）："Future observation frames … are passed through a VAE encoder to extract latent representations … Some tokens are randomly masked out during training."；h 步 latents 时序拼接为 N×h 序列 → Transformer → joint video-action latent {𝐙t+1,…,𝐙t+h}（每步 N tokens）。"To minimize information leakage across different frames, we consistently mask the same positions across all video frames."
- 语言指令（可选，Libero 类任务，§III-B）："the language input is encoded into a d-dimensional token using the CLIP [33] text encoder. To emphasize the language, this token is repeated M times and appended to the N×h video-action tokens"（经 cross-attention 融合，取前 N×h 个输出为 joint latent）。

### 【目标表征】
- 显式 goal 模块：**原文未写（已获取范围：v3 全文）**。任务目标由输入侧定义（TABLE I caption）："In the multi-task scenario, the goal can be defined through the image observations (PushT-M) or language descriptions (Libero10)."
- 「未来视觉」的定位（§III-C）：联合 latent 同时条件两个头；训练时视频监督帮助 latent 捕获场景信息，"During policy inference, where speed is crucial, the decoupled design allows us to skip video generation and decode only the actions."（即未来视觉是训练期监督/内部表征，不是推理期显式目标输入。）
- 未来视频可作规划器（§I、§VII）：masked training 使单一模型可 "simultaneously performing both low-level control and high-level planning"；forward dynamics 用途：以预测帧选动作轨迹，DP-C 38% → 60%（GT 模拟器上界 75%，Fig. 9）。

### 【代码依据】
- 论文两处提到代码但全文未给 URL（获取范围：v3 全文）："Please check out our code for details."（§X-A）；"Our model builds on the pretrained model (MAR-B) released by [27]"（§X-A）。待 GitHub 取证。

### 【why 追问】
1. 为何视频监督能提升动作精度？原文机制："video generation helps the latent representations 𝐙 capture more detailed scene information, which benefits action prediction"（§III-C）；证据是视觉扰动鲁棒性（Fig. 5 goal color：UVA 64% vs UniPi 40% / OpenVLA 32%）与 UVA-action 消融降分（Table II）。但 latent 内部视频/动作信息的逐 token 分工分析原文未写。
2. 为何解耦解码快？denoising 仅发生在轻量头而非整个网络（§III-C）；§X-D 分解：VAE 40ms + Transformer 40ms + Action Diffusion(16 步) 15ms = 95ms/16 动作。但动作头仅以聚合 latent 为条件、不参与视频重建，是否损失细粒度未来信息？原文未写。
3. 语言 token 输出被丢弃（只取前 N×h 个输出，§III-B）——语言仅经 attention 间接影响 latent；该设计无消融（原文未写）。

### 【关键数值】（逐字抄录）
- 规模：UVA 0.5B（vs π0 3.3B、π0-FAST 3.0B）（§V-D）。
- 仿真 Table II（成功率）：UVA PushT 0.98 / Toolhang 0.88 / PushT-M 0.88 / Libero10 0.90 @0.23s；DP-C 0.91/0.95/0.68/0.53 @0.50s；DP-T 0.78/0.76/0.63/0.58 @0.36s；π0 Libero10 0.85 @0.09s；UVA-action 0.45/0.62/0.46/0.86 @0.22s。
- 多任务：§V-D "UVA surpasses the best baseline by 20% on the PushT-M task and by 5% on the Libero10 benchmark."
- 真机 Table III（UMI，OOD 多任务）：UVA Cup 0.65 / Towel 0.70 / Mouse 0.80 @95ms；DP-UMI 0.50/0.70/0.40 @70ms；单任务 Cup：DP-UMI 0.95 vs UVA 0.85；§V-D "achieving a 15% higher success rate on the Cup task and a 40% higher success rate on the Mouse task compared to DP-UMI."
- 视觉扰动（Fig. 5，PushT）：GoalColor UVA 0.64 / UniPi 0.40 / OpenVLA 0.32；BgColor UVA 0.35 / DP-C 0.12。
- 推理分解（Table VI，RTX 3080）：Action Diffusion 16 步 15ms / 100 步 93ms；Video Diffusion 16 步 100ms / 100 步 625ms；UVA(16 步) 95ms、(100 步) 173ms。
- 视频生成 FVD（Table IV）：Libero10 UniPi 56.55 / UVA-1step 89.36 / UVA-8step 51.10；Cup UniPi 71.37 / UVA-1step 51.34 / UVA-8step 29.72。
- Inverse dynamics（Table V，L2）：UVA 0.75cm/1.11°；UniPi 1.92cm/2.21°；SLAM 0.41cm/0.30°。
- 无动作视频（§X-E）：Human Video 3,175 videos 预训练后 Libero10 30-test 0.93→0.97、500-test 0.90→0.91。

### 【一页总结】
UVA = 联合 video-action latent + 解耦扩散头 + masked 多目标训练：历史图像（VAE latent→N tokens）与历史动作 chunk（重复 M 次→N tokens）进共享 Transformer 得 joint latent Z，两个轻量扩散头分别解动作/视频；推理跳过视频头，0.5B 参数、95ms/16 动作。目标表征无独立模块：多任务目标由图像（PushT-M）或语言（CLIP token，Libero10）在输入侧定义；未来视频是训练期监督与（可选）forward-dynamics 规划器，而非推理期显式目标输入。对 WAM 调研的意义：joint latent 即「未来状态+动作」统一表征的早期代表。

### 【待核实】
- 代码仓库 URL（全文未给）；M 的数值（语言/动作 token 重复次数）未写明。
- 「20%/5%」口径：按 Table II 验算为绝对百分点（0.88−0.68=0.20 vs DP-C；0.90−0.85=0.05 vs π0），原文未说明。

---

## 4. Genie Envisioner（arXiv 2508.05635）

### 元信息
- 标题：Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation
- 版本：v3，2025-11-04（v1 2025-08-07）；CC BY 4.0；作者：Yue Liao, Pengfei Zhou, Siyuan Huang, Donglin Yang 等（AgiBot Genie 团队）；主页 https://genie-envisioner.github.io
- 来源级别：**A**。arxiv.org/html/2508.05635v3 本地化成功（HTML 244,344 字节；txt 101,688 字符 / 336 行，§1–§9+References 完整）。txt 两处数字重复（"2,9672,967 hours"、"200200ms"）系 LaTeXML MathML 重复渲染，已核对 HTML 原 markup：实为 2,967 hours、200 ms。本地备份：`auto_docs/reading/html/2508.05635.html`、`txt/2508.05635.txt`、`pdfs/2508.05635.pdf`（26.2MB）。

### 【输入表征】
- GE-Base 三条件输入（§2.1）："At each step t, the world foundation model 𝒲 generates the next chunk 𝐱1:N(t) conditioned on three components: the initial visual observation 𝐱0, the language instruction q, and a sparse memory 𝐱̂0:t−1, constructed by sparsely sampling long-term historical frames from previous steps."
- 视觉编码（§2.1）：三视角（head + 双腕 vl,vr）经共享视频编码器 ℰ 得 latent visual tokens；每视角独立 noise map，配 2D rotary positional embedding + view-specific learnable embedding + timestep encoding 进 DiT 骨干。
- 指令编码（§2.1）："the instruction q is processed using a frozen T5-XXL encoder (Raffel et al., 2020), producing a set of text embeddings 𝒯(q). These are integrated into the visual token stream through cross-attention layers within the DiT."
- GE-Act 输入（§3.1）："At each step, the foundation model processes visual tokens derived from initial observations 𝐱0 and sparsely sampled historical frames 𝐱̂t−1, conditioned on instruction embeddings 𝒯(q)"；"a lightweight 160M-parameter autoregressive action decoder"；训练输入为 4 帧@5Hz 记忆、输出 54 步@30Hz 动作（§3.2）。
- GE-Sim 动作输入（§5.1）：双臂每步 14 维 [x,y,z,roll,pitch,yaw,o]×2，K 步轨迹 𝐀∈ℝK×14；Pose2Image（相机内外参投影姿态为图像）与历史帧 latent 逐元素相加（Eq.），temporal motion deltas 与冻结 CLIP 参考图作 style anchor 注入 cross-attention。

### 【目标表征】
- 任务目标 = 语言指令，无独立 goal 模块：GE-Base 是 "instruction-conditioned video diffusion model"（Abstract）；"The generation proceeds iteratively until the task specified by the instruction is fully executed"（§2.3）。
- 未来视频的角色分工：GE-Base 生成指令对齐的未来视频（"an embodied predictive representation"（§2.1））；GE-Act 中未来视觉退化为 cached latent 条件——"The video DiT performs a single flow-matching denoising step per inference pass to generate visual latent tokens, which are then cached and reused throughout the action generation phase."（§3.3）。
- 立场声明（§7 Related Work）："Our framework takes a different approach by using vision-language inputs to condition a generative world model, enabling predictive reasoning and planning through internal simulation."；EWMBench 显式度量 "instruction-action alignment"（Abstract）。

### 【代码依据】
- Abstract："All code, models, and benchmarks will be released publicly."；全文未见 GitHub URL（获取范围：v3 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何世界模型预训练对动作头有效？§3.4：from-scratch 或通用视频模型（LTX-Video）初始化近零成功率；GE-Base 域内初始化 "achieves 64 SR and 81 E2E, which further improve to 76% and 89% when combined with general video pretraining"——机制归结为 "strong representations"，latent 内目标语义的探针分析原文未写。
2. 为何加入 robot state 对通用视频预训练模型反而有害？原文仅一句 "due to short-cut learning effects"（§3.4），无机制展开。
3. 视频 5Hz vs 动作 30Hz（1:6）解耦（§3.3）：为何稀疏未来帧够支撑稠密动作？原文理由为降 latent 维度、免处理高频视觉（§3.3），无帧数消融。

### 【关键数值】（逐字抄录）
- 数据（§2.2）：AgiBot-World-Beta "approximately one million" episodes、"2,967 hours"；Stage I 57 帧@3–30Hz→8 latent 帧（32×A100 ≈7 天）；Stage II 9 帧@5Hz+4 记忆帧→2 latent 帧（≈3 天）。
- GE-Act（§3.2/§3.3）：160M 参数；4 帧@5Hz→54 步@30Hz；推理 video DiT 1 步 + action 5 步，"the forward pass for 54 steps is completed in 200 ms on an onboard NVIDIA RTX 4090 GPU"。
- 预训练分析（§3.4，305 demos/40,000 steps，Table 1）：from-scratch E2E 0.15 / SR 0.30；GE-Base 初始化 E2E 0.81 / SR 0.64；+视频自适应 E2E 0.89 / SR 0.76（均 w/ robot state 口径）。
- 任务集（§3.4）：Make a sandwich / Pour a cup of tea / Clean the table / Heat food in the microwave / Pack laundry detergent 共 5 个；AgiBot G1 上 SR/E2E 具体数值仅在 Figure 8 柱状图 → 未取到（HTML 文本无数值）。
- Few-shot（§4）：Agilex 折叠任务每任务 250 demos（约 1 小时）；Dual Franka cloth folding 250 episodes（约 1 小时）；RoboTwin 200 demos（4 任务×50）all-in-one；UniVLA/GR00T N1 在 Agilex 折叠任务 "achieving a success rate of 0%"（§4.2）。

### 【一页总结】
GE 是「世界模型→动作」层级平台：GE-Base（LTX-Video/COSMOS2 2B 三视角视频扩散）以「初始观测 + 稀疏记忆 + T5-XXL 指令」为条件自回归生成未来视频；GE-Act 以 160M flow-matching 头把相同 latent 条件直接映射为动作（训练期冻结 GE-Base 只训头；推理视频仅 1 步 denoising 并缓存，200ms/54 步）；GE-Sim 反向以动作条件生成视频做闭环仿真。目标表征 = 语言指令（无 goal 图像/嵌入）；未来视觉在 GE-Act 中是被缓存的隐式条件、在 GE-Sim 中是仿真媒介。对 WAM 调研意义：指令为唯一任务目标源，世界模型 latent 是目标与动作的共享表征。

### 【待核实】
- 5 任务 SR/E2E 数值（仅图内）；代码仓库 URL（承诺 release 但未给）；GE-Act 与 GE-Base hidden dim 比值（原文未写）。

## 5. LingBot-VA / Causal World Modeling for Robot Control（arXiv 2601.21998）

### 元信息
- 标题：Causal World Modeling for Robot Control（模型名 LingBot-VA）
- 版本：v2，2026-03-22（v1 2026-01-29）；作者：Lin Li, Qihang Zhang, Yiming Luo 等（Ant Group / Robbyant）；项目页 https://technology.robbyant.com/lingbot-va ；代码 https://github.com/robbyant/lingbot-va
- 来源级别：**A**。arxiv.org/html/2601.21998v2 本地化成功（HTML 830,681 字节；txt 105,183 字符 / 3,347 行，§1–§6+References+附录完整）。本地备份：`auto_docs/reading/html/2601.21998.html`、`txt/2601.21998.txt`、`pdfs/2601.21998.pdf`（7.8MB）。

### 【输入表征】
- 视觉输入（§3.2 Video-Action State Encoding）："We leverage a causal video VAE [79] to compress visual observations into compact latent tokens zt=E(ot∣o<t)∈ℝN×C"；动作输入："we project action vectors to token embeddings at∈ℝD via a lightweight MLP ϕ(⋅)"。
- 统一交织序列（§3.3 Video Sparsification）：视频按 τ=4 时间下采样，每帧配 τ 个连续动作："forming a unified sequence [zt,at,1,at,2,…,at,τ,zt+1,…] for joint modeling"；世界模型条件含动作历史（Eq. (8)，§3.2）："Conditioning on action history thus provides knowledge of how the robot has moved and interacted with objects"。
- 指令输入（§4.2）："Task instructions are encoded using a frozen T5 text encoder [59] and injected via cross-attention."
- 实现数值（§4.2）：Wan2.2-5B 视频流 dv=3072/30 层；动作流同深度 da=768（4× 更小）、约 +350M 参数、总 5.3B；causal VAE 4×16×16 压缩 + patchify /2 → N=192 tokens/帧（多视角沿宽度拼接，视角数未写明）；动作编解码为单层 MLP hidden 256。

### 【目标表征】
- 显式 goal 模块：**原文未写（已获取范围：v2 全文）**。任务目标 = 语言指令（T5，cross-attention）。
- 核心目标机制 = 「期望的未来观测」作为逆动力学条件（§3.2）："Rather than directly predicting actions from current observations, we employ an inverse dynamics model that infers actions by conditioning on desired future observations, enabling the policy to reason about what action leads to a desired visual outcome."（Eq. (9)：at:t+K−1∼gψ(⋅|ẑt+1:t+K,z≤t,a<t)）
- 视频即隐式目标证据（§4.3.1）："the value of video generation as implicit guidance. The generated video futures provide rich predictive signals about object dynamics and state transitions, which inform the action model to produce more physically plausible manipulation trajectories."
- 闭环更新目标（§3.4）：FDM-grounded 步骤以真实观测 zt−1 重新想象 zt，"we force the model to re-align with environmental feedback before predicting zt+1"——目标未来随闭环观测持续刷新。

### 【代码依据】
- abs 页标注代码仓库 https://github.com/robbyant/lingbot-va；论文正文未再给 URL（获取范围：v2 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何要显式建模未来（而非 feedforward VLA）？原文动机是「表征纠缠」（§1）："requiring a single neural network to simultaneously learn visual scene understanding, physical dynamics, and motor control from a unified supervision signal."；且物理因果性（§3.2）："The physical world, however, is inherently causal and autoregressive: the present state depends only on the past"。数据效率证据：10 demos 时 LingBot-VA 比 π0.5 高 15.6%（Make Breakfast）/10.3%（RoboTwin Easy）progress（§4.5.1）。
2. 为何视频只需部分去噪（s=0.6）就够动作解码？§3.3 Noisy History Augmentation：动作预测不需要完全去噪的视频表征；训练期注入 p=0.5、saug∈[0.5,1] 的噪声。无 s 值消融曲线（原文未写）。
3. 为何 MoT 双流而非单流？§3.3："preventing interference between modality-specific feature representations"；但双流 vs 单流同参数量对比实验未做（消融只有 WAN 初始化、async、动作流初始化，未见 MoT 消融）。

### 【关键数值】（逐字抄录）
- 预训练（§4.2）：1.4T tokens；K∈[1,4] 随机采样、部署 K=4；推理 Euler：视频 3 步（s→0.6）、动作 10 步（s→1.0）；video CFG 5.0 / action CFG 1.0。
- 后训练（§4.2）：50 demos、3K steps、lr 1×10⁻⁵。
- RoboTwin 2.0（§4.3.2，Table 1）："LingBot-VA achieves an average success rate of 92.9% (Easy) and 91.6% (Hard)"；50 任务均值 92.93 (+4.2) / 91.55 (+4.6)；Horizon=3 增益 +8.2% (Easy) / +9.1% (Hard)。
- LIBERO（Table 2）：Spatial 98.5±0.3 / Object 99.6±0.3 / Goal 97.2±0.2 / Long 98.5±0.5 / Avg 98.5。
- 真机（§4.3.1）：6 任务（Make Breakfast、Pick Screws、Insert Tubes、Unpack Delivery、Fold Clothes、Fold Pants），每任务 50 demos；具体成功率仅 Figure 5 → 未取到（HTML 文本无数值）；§6 结论："achieving over 20% improvement on challenging tasks compared to π0.5 with only 50 demonstrations for adaptation."
- 消融（Table 3，RoboTwin Easy）：LingBot-VA 92.9；FDM-grounded async 90.4；naive async 74.3；WAN 初始化 80.6；异步比同步快 "2× faster"（§4.4）。
- 记忆任务（§4.5.2，Fig. 9）：Wipe Plate / Search Box 具体数值仅图内 → 未取到。

### 【一页总结】
LingBot-VA 是「因果自回归视频-动作世界模型」：causal VAE 把观测压成 zt（N=192 tokens/帧），动作经 MLP 成 token，按 τ=4 交织进单一因果序列，双流 MoT（视频流 Wan2.2-5B + 768 维动作流，共 5.3B）以 teacher forcing + flow matching 同时学「视觉动力学 + 逆动力学」，指令（冻结 T5）cross-attention 注入。目标表征 = 语言指令 + 模型自预测的「期望未来观测」（逆动力学条件），执行中由 FDM-grounded 闭环用真实观测刷新。意义：与 UVA 的联合 latent 相比，这里未来视觉是**推理期显式生成的逆动力学目标**，是 WAM「预测未来→解码动作」的闭环版。

### 【待核实】
- 多视角数量（N=192 由视图拼接而来，但视角数未写明）；真机 6 任务逐任务数值（仅 Fig. 5）。
- §6 结论 "92.0% on RoboTwin 2.0" 与 §4.3.2 "92.9% (Easy)" 口径不一致（原文如此，待与官方表核对）。
- LIBERO-Goal 97.2 低于 Spatial/Object/Long：Goal 套件的目标表征未单独分析（原文未写）。

---

## 6. Rift（arXiv 2608.11521）

### 元信息
- 标题：Keep the Future, Drop the Rollout: RIFT for World Action Models
- 版本：v3，2026-09-25（v1 2026-08-12；Comments: "Added real-world experiments and updated the project URL"）；CC BY-NC-ND 4.0；作者：Chushan Zhang, Jinguang Tong, Xuesong Li, Yikai Wang, Hongdong Li
- 来源级别：**A**。arxiv.org/html/2608.11521v3 本地化成功（HTML 428,482 字节；txt 70,326 字符 / 1,463 行，§1–§6+References+附录完整）。本地备份：`auto_docs/reading/html/2608.11521.html`、`txt/2608.11521.txt`、`pdfs/2608.11521.pdf`（15.0MB）。

### 【输入表征】
- 输入三要素（§4.2）：first-frame tokens f0(o)（观测 o）+ 语言指令 l（经原条件路径）+ learned anticipation tokens E∈ℝm×d 占据未来时空位置：
  > "Let f0(o) denote the first-frame tokens extracted from observation o; the remaining observation and instruction l enter through the backbone's original conditioning path."
- Anticipation tokens 替代 rollout 未来 token（§4.2）："Rift keeps Fast-WAM-Joint's architecture. For a video stack with hidden width d and L layers, it replaces rolled-out future tokens with learned anticipation tokens E∈ℝm×d."；full alignment m=n(Tlat−1)：LIBERO m=196、RoboTwin 2.0 m=240；动作 chunk 水平 H=32。
- 单趟 CachePrefill（§4.2，Eq. (2)–(3)）："A single forward pass through the video expert computes the keys and values of all anticipation tokens"；â1:H=ActionDenoise(o,l;𝒞ϕ(o,l))；"Deployment requires one prefill per chunk, with no rollout or VAE decoding."
- 注意力掩码（§4.2）："first-frame tokens attend only the observed frame, anticipation tokens attend the first frame and one another, and video tokens never attend action tokens. The cache is therefore action-independent."

### 【目标表征】
- 本文核心即「目标表征」的因果界定：动作专家读取的是**未来 K/V cache（内容+位置组织）**，而非 rollout 的动态过程。Finding 2（§3.5）："For Joint and Cosmos-2, final-clean K/V replay gives 1.9/1.7 cm EE-ADE and 97.9/98.2% success; the corresponding Oracle policies reach 98.4/98.4%."
- 内容与位置敏感（Finding 1，§3.4）：masking 得 11.8–20.4 cm EE-ADE、成功率 98.4–98.6%→0.0–32.0%；"Position also matters: spatial shuffle yields 5.0–19.8 cm and 0.0–84.5% success across all four, while temporal swap yields 15.6–16.3 cm and 0.0–69.0% on Joint, IDM, and LingBot-VA."
- 目标未来的多模态性（§4.3）："Under a multimodal conditional distribution, direct ℓ2 regression has a conditional-mean optimum and can average distinct valid futures; we instead use conditional flow matching (Lipman et al. 2023) as a distributional auxiliary objective."——未来（目标）不是单点，需分布监督。
- 语言指令 l 仍为条件之一，但与未来 cache 的贡献分离实验：原文未写（已获取范围：v3 全文）。

### 【代码依据】
- 全文与 abs 页均未显示代码/项目 URL（abs Comments 仅提 "updated the project URL"）。待 GitHub 取证。

### 【why 追问】
1. 为何保留未来条件比纯当前观测多 1.5–2.0pp 且几乎不加延迟？§5.2："Compared with rollout-free Fast-WAM and PFD, Rift improves success by 2.0 and 1.5 percentage points, respectively, at comparable latency."——即「显式未来表征」本身有增量；但为何动作专家不能在内部自行推理未来（即该增量为何必须显式外化）原文未解释。
2. 为何 conditional FM 优于 L2 回归（+0.4pp）？§4.3 理论理由为未来多模态、条件均值会模糊；§5.3：Rift-L2 98.37±0.12 vs Rift 98.8±0.17；m 扫描（Fig. 5）：L2 从 97.08%→98.37%，FM 从 m=4 起反超、m=196 峰值 98.78%。
3. 干预严重度为何因接口而异（temporal swap 对 Joint/IDM 更致命、spatial shuffle 对 LingBot-VA 更致命）？原文仅报现象："Severity differs by interface"（§3.4），无机制分析。

### 【关键数值】（逐字抄录）
- LIBERO（Table 1，A800 ms/chunk）：Rift 98.8±0.17 @248ms；Fast-WAM 96.8±0.27 @236；PFD 97.3±0.12 @257；Fast-WAM-Joint 98.4±0.26 @780；Fast-WAM-IDM 98.6±0.34 @1081；LingBot-VA 98.5±0.08 @2270；"Rift requires only 247.9 ms per action chunk, reducing latency by 68.2% to 89.1%"；速度增益 3.1–9.2×（Abstract）。
- LIBERO-Plus OOD（10,030 variants，无再训练）：Rift 81.1%，"+9.7 percentage-point gain over Fast-WAM-IDM"（§5.2）。
- RoboTwin 2.0（Table 2）：Rift 92.9/92.6（AVG 92.8）；PFD 92.5/92.1；LingBot-VA 92.4/91.4；Fast-WAM 91.9/91.6；Fast-WAM-Joint 91.0/91.1。
- 真机（Galaxea R1 Lite，3 RGB 视角，RTX 4090，Table 3）：Stack Baskets 36/50、Object Classification 22/50、Tube Mixing 10/50，总 SR 45.3%（Fast-WAM-Joint 39.3%、Fast-WAM 26.7%）；420 ms/chunk（Joint 1069、Fast-WAM 404）；"a +6.0 percentage-point improvement over the strongest evaluated baseline"；"60.7% lower action-chunk latency"（§1）。
- 干预（§3，每 cell 2,000 paired trials × 40 LIBERO 任务）：masking 18.7 cm EE-ADE、成功率 98.4%→9.7%；spatial shuffle 14.3 cm/65.2%；temporal swap 15.6 cm/0.7%；final-clean replay 1.9 cm/97.9%。
- 训练（§4.3）：每步两个前向共享视频专家；课程 70% 后第一帧 latent 噪声线性升至 0.3×、0.06×σ；λFM=λprobe=1 前 70% 后 cosine 衰减至 0.2；LIBERO 20k-step budget（§5.1）。

### 【一页总结】
Rift 先做因果干预（对 4 个 WAM 的 future K/V cache 做 masking/重排/替换，配对闭环测 EE-ADE 与成功率），证明：动作专家依赖未来表征的内容与位置组织、但不依赖 rollout 的动态演化——单个 fixed final-clean cache 即可近似保留执行（1.9cm/97.9%）。据此把「未来表征」从生成式 rollout 抽象为 learned anticipation tokens 的单趟 CachePrefill（m=196/240），conditional FM 做分布级目标监督。意义：10 篇中唯一**直接量化并操纵目标表征内部组织**的工作——「目标=未来 K/V cache」，为 WAM 输入/目标表征提供因果证据（内容敏感、位置敏感、固定可复用）。

### 【待核实】
- 代码/项目 URL（未显示）；语言指令 l 与未来 cache 的贡献分离（原文未写）；Fig. 5 中 m=2 的精确数值（仅起点 97.08% 可见）。

---

## 7. WAM-TTT（arXiv 2607.06988）

### 元信息
- 标题：WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time
- 版本：v3，2026-10-23（v1 2026-07-08）；作者：Yusen Feng, Bingchen Han, Jiangran Lyu, Kai Liu 等（北大/北京通用人工智能等）；主页：abs 页未列（获取范围：abs+HTML）
- 来源级别：**A**。arxiv.org/html/2607.06988v3 本地化成功（HTML 451,940 字节；txt 99,058 字符 / 845 行，§1–§5+References+附录完整）。本地备份：`auto_docs/reading/html/2607.06988.html`、`txt/2607.06988.txt`、`pdfs/2607.06988.pdf`（11.4MB）。

### 【输入表征】
- 基础 WAM 输入（§1）："Once deployed, their behavior is largely determined by the pretrained weights and a limited conditioning interface, such as language instructions, goal images, or short observation histories"——WAM-TTT 在此基础上**新增第四种输入通道**：人类视频→fast-weight 记忆。
- 人类视频输入的处理方式（Abstract/§3.2）："Rather than treating human videos as trajectories to imitate, WAM-TTT absorbs them into a lightweight adaptive memory inside a frozen WAM through self-supervised video prediction."
- TTT 残差只加在 video expert 上（§3.1，Eq. (1)）："𝒛(ℓ+1)=𝒛̂(ℓ+1)+Δ𝒛TTT(ℓ), 𝒙(ℓ+1)=𝒙̂(ℓ+1)"——动作 token 不变，人类线索经视频流间接进入 joint attention 影响动作专家。
- 记忆读写（§3.2）：人类 token hφ(ℓ)（按归一化相位 φ=t/Tr 对齐）投影为 Keys/Values，机器人 token 投影为 Queries；KVM 重构损失（Eq. (3)）ℒKVM=‖fW(Kh)−Vh‖²₂；测试期只用人类侧目标（Eq. (7)）：ℒTTT=ℒvg+λΣℓℒKVM，内层 SGD 更新快权重，WAM 全冻结。

### 【目标表征】
- 目标 = 「人类视频所演示的目标行为/任务变体」，通过记忆吸收而非显式 goal 特征（§1/Abstract）："steering robot foundation models (RFMs) toward new task variants or user-preferred behaviors"；"At test time, only unlabeled human videos are required to adapt the memory"。
- 推理时目标条件（§3.3，Eq. (9)）："𝒂t:t+k∼pΘWAM,θTTT,WN(𝒂t:t+k∣𝒐t,𝒈)"——基础 WAM 原有目标条件 𝒈（语言指令/goal 图像，§1）之外，人类视频以快权重形式注入；两者如何交互原文未做剥离实验（已获取范围：v3 全文）。
- 目标表征为何放记忆而非 in-context（§4.2）：WAM-ICL（同样的人类视频作上下文）avg 仅 7.1% vs WAM-TTT 46.2%；"feeding the same human videos as in-context tokens fails to transfer skill to unseen home environments, whereas absorbing them as fast-weight memory does."
- 泛化保持（§4.4）："wam-ttt steers the visual dynamics used for action generation without overwriting the pretrained action prior."

### 【代码依据】
- 全文未见 GitHub URL（获取范围：v3 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何 in-context 人类视频会失效而快权重记忆有效？§4.2 推测："our WAM was not pretrained to condition action denoising on full deployment-time human videos; learning this conditioning through meta-training alone may require more data."——记忆路径经视频预测损失（自监督）吸收，绕过了动作级配对需求。
2. 为何 KVM 重构目标必要？消融（Table 3）：w/o Memory Recon 在 Table Bussing 66.7 vs 完整 100.0、Swap Place 72.0 vs 88.9；w/o Meta Training 两任务均 0.0/9.0——即「人-机对齐的读写接口」必须预训练。
3. 为何 TTT 快权重不破坏泛化（§4.4，Table 4：Lighting 66.0 vs LDA 54.0、Spatial 56.0 vs 28.0）？原文归因于 "task-specific adaptation without overwriting the WAM's transferable structure"（附录 L590）；但适应信号与原有目标条件 𝒈 的冲突边界未刻画（Limitations (2)：未empirically 表征漂移边界）。

### 【关键数值】（逐字抄录）
- 主结果（Table 1，New 家居环境，9 任务 25 trials/cell，progress %）：WAM-TTT avg 46.2；LDA 32.5（+13.7）；WAM-Cotrain 25.3（+20.9）；EgoScale 15.0（+31.2）；π0.5 14.8（+31.4）；WAM-ICL 7.1（+39.1）；9 任务赢 7、Flip Steak 平（10.0）、Stamp Paper 输（8.3 vs LDA 33.3）。
- 接口对比（Table 2，Sharpa Swap Place）：WAM-TTT 66.7；LingBot-style ICL 50.0；WAM no adapt 44.4；XSkill-style SAT 10.4；UniSkill-style 15.2。
- 消融（Table 3，10 trials）：Table Bussing：WAM-TTT 100.0 / WAM-LoRA 30.0 / w/o Meta 9.0 / w/o Memory Recon 66.7 / w/o TTT 40.0；Swap Place：88.9 / 0.0 / 0.0 / 72.0 / 74.1。
- 泛化保持（Table 4，Deliver Drink）：Lighting WAM-TTT 66.0（LDA 54.0、WAM-ICL 12.0、π0.5 28.0）；Spatial 56.0（28.0/20.0/0.0）。
- 数据（§4.1）：2,286 paired human-robot episodes、9 任务；3 具身（Unitree G1、Galbot gripper、Galbot Sharpa）；人类视频 GoPro 采集、无姿态估计。
- 附录（E.2，backbone 消融）：VLM 条件后端负载关键——去掉 VLM −28 pts（DiT-only 72 vs 100）、随机初始化 VLM −20 pts（80）、冻结 VLM −46 pts（54）。

### 【一页总结】
WAM-TTT 给冻结 WAM（LDA）的 video expert 添加 TTT fast-weight 残差层：元训练期用 2,286 对相位对齐的人-机演示学「人类 Keys/Values→机器人 Queries」的读写接口（KVM 重构 + 视频预测内外层目标）；测试期只需无标注人类视频，内层 SGD 更新快权重，WAM/动作专家/慢投影全冻结。目标表征 = 「人类演示视频」以记忆残差形式注入视觉动力学（第 4 种输入通道，与 𝒈 并存）。意义：目标不再只是语言/图像，而是**可在线吸收的行为分布**；对 WAM 调研而言是「目标表征的运行时更新机制」。

### 【待核实】
- 基础 WAM（LDA [36]）中 𝒈 的具体形式（语言或 goal 图像）主文未展开（属外部文献）；𝒈 与 TTT 记忆的交互消融（原文未写）；代码仓库。

---

## 8. HOST（arXiv 2607.20033）

### 元信息
- 标题：HOST: Robots Acquire Manipulation Skills in Seconds from a Single Human Video
- 版本：v4，2026-08-20（v1 2026-07-22）；作者：Guangyan Chen, Meiling Wang, Te Cui 等；主页：abs 页未列（获取范围：abs+HTML）
- 来源级别：**A**。arxiv.org/html/2607.20033v4 本地化成功（HTML 543,797 字节；txt 145,158 字符 / 655 行，§1–§4+References+附录完整）。本地备份：`auto_docs/reading/html/2607.20033.html`、`txt/2607.20033.txt`、`pdfs/2607.20033.pdf`（23.1MB）。

### 【输入表征】
- 策略输入（§4.1）：πθ(at∣τd,ℓ,ot-K:tr,str)——视频演示 τd + 语言指令 ℓ + 近期机器人观测 + 本体感状态。
- 视觉共享 latent（§4.3 Input representation）："All visual elements, including the video demonstration window 𝒲, recent robot observations, and the future observations 𝒯to, are encoded into a shared latent space by the Wan VAE (47), yielding video demonstration tokens zd, robot observation tokens zo, and target observation tokens yo."
- 因果 token 序列（§4.3，Eq. (12)）：[zd∥zo∥yσp,yp∥yσo,yo∥yσa]——三组生成目标（定位 token yp、未来观测 token yo、动作 token ya）以 noisy/clean 对形式按 p→o→a 因果序排列，flow matching 联合训练。
- 语言/状态条件（Eq. (14)）："c=[UMT5(ℓ)|Ms·str]"，每层 cross-attention；架构为 Wan2.2 视频骨干上的双专家 MoT（视频专家 + 缩减宽度的动作专家，共享 self-attention）。

### 【目标表征】
- 核心立场：视频演示是「主动目标驱动者」而非被动上下文（§1/§2.8）："the video demonstration should actively determine the robot's prediction target."；"turning the video from a passive conditioning signal into the active driver of the robot's prediction."
- 目标的双层结构（§4.3）：先把机器人进度定位到共享任务进度流形，再 "translate its upcoming progression into the robot's own future observations conditioned on the localized segment, and derive robot actions from these future observations"——目标 = 「演示视频的后续进展」→「机器人自身未来观测」→ 动作。
- 进度流形（§4.2）：Qwen3-VL-Embedding-8B 帧嵌入 + Smooth DTW + temporal cycle consistency，自监督无帧级标注；"Progress alignment reduces the mean absolute progress error at human-annotated events from 0.079 under matching based on clock time to 0.006"（§2.8）。
- 技能外部化（§3）："Each video keeps the acquired skill in external context rather than encoding it in the policy weights."——目标（技能）驻留在外部演示记忆，可检索复用（Eq. (20) 指令+场景相似度检索）。

### 【代码依据】
- 全文未见 GitHub URL（获取范围：v4 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何预测目标要耦合到演示的未来进展而非执行轨迹的固定偏移？§1 三点结构失配：temporal asynchrony、state discrepancies、scarce paired supervision；证据链（§2.8）：整段视频条件 0.21 → timestamp 窗口 0.29 → progress 窗口 0.45 → 目标耦合 0.62。
2. 为何要「先预测机器人未来观测、再解码动作」？§4.3："Directly deriving robot actions from the video demonstration forces the model to resolve intractable cross-domain ambiguities"；消融（§2.9）：直接映射 0.34 → +localization 0.43 → +未来观测 0.55 → 因果级联 0.62。
3. 为何两阶段训练（同具身→人机）？§2.10：仅 Stage 2 人机配对训练 "performs poorly"（0% 条件）；Stage 1 数据量单调提升成功率（Fig. 10D）——同具身配对数据供给「跟随演示程序」的基础能力。

### 【关键数值】（逐字抄录）
- 总成绩（§1/§2.1）：62% 平均成功率（novel tasks）、约 29s/技能；"exceeding the strongest one-shot visual imitation baseline by 43% and the strongest zero-shot imitation learning baseline by 45%"；vs 50 demos SFT：Wall-OSS+SFT 仅 56%（§2.5），HOST "using 50 times fewer demonstrations, acquiring new skills 507 times faster"（SFT 4.0–4.9 h/任务）。
- 保留率（§2.6）：HOST 保持原水平；π0.5+SFT 保留 20%、HOST-base+SFT 22%、Wall-OSS+SFT 43%。
- 数据（§2.2）：Stage 1 193,462 robot trajectories / 229 tasks；Stage 2 5,847 human video demos；50 novel tasks × 20 trials；双臂 ARX R5 ×2、3 RGB 相机（双腕+第三人称）、20 维动作空间。
- 扰动鲁棒（§2.7，相对默认 62% 的损失）：lighting −1%、object substitution −4%、scene replacement −6%、human disturbance −9%。
- 定位精度（§2.9）：progress MAE 0.013；进度对齐 MAE 0.079±0.062 → 0.006±0.008（§2.8 Fig. 9D）。

### 【一页总结】
HOST 是「单视频一次技能获取」框架：输入（视频演示+指令+近期观测+本体感）进 Wan VAE 共享 latent，双专家 MoT 以因果 token 序列联合生成「进度定位 → 机器人未来观测 → 动作 chunk」（flow matching）；训练目标经共享任务进度流形与演示视频的未来进展耦合，使人类视频成为主动目标驱动者；两阶段训练（同具身机器人-机器人配对 → 人-机配对）缓解配对数据稀缺。目标表征 = 人类视频的「未来进展」显式翻译为「机器人自身未来观测」，再级联解码动作。意义：10 篇中唯一把**跨具身视频目标翻译**做成显式中间表征的框架（human video → robot future obs → action 三级因果链）。

### 【待核实】
- 代码仓库 URL；语言指令 ℓ 标注为可选（§4.1 "optional language instruction"），其与视频目标的贡献分离实验未做；demb 维度数值原文未写。

---

## 9. Pelican-Unify 1.0（arXiv 2605.15153）

### 元信息
- 标题：Pelican-Unify 1.0: A Unified Embodied Intelligence Model (UEI) for Understanding, Reasoning, Imagination and Action
- 版本：v2，2026-05-21（v1 2026-05-14）；作者：Yi Zhang, Yinda Chen, Che Liu 等（Pelican 团队）；主页：abs 页未列（获取范围：abs+HTML）
- 来源级别：**A**。arxiv.org/html/2605.15153v2 本地化成功（HTML 306,561 字节；txt 59,542 字符 / 616 行，§1–§6+References 完整）。本地备份：`auto_docs/reading/html/2605.15153.html`、`txt/2605.15153.txt`、`pdfs/2605.15153.pdf`（40.9MB）。

### 【输入表征】
- 多模态上下文（§2.2，Eq. (2)）："ct=(a<t,l,o≤t,s≤t)，which bundles the action history, language instruction, environment observations, and robot proprioceptive state."
- 共享嵌入器（§2.2）："video frames o≤t are encoded by a 3D video VAE ℰv"；"the action history a<t is embedded by a lightweight MLP ℰa"；语言经 text tokenizer；状态经 linear projection；全部进入 VLM（Qwen3-VL 初始化）token 空间——ℰv/ℰa 同时被 UFG 复用："understanding and generation therefore share one modality embedding space."
- 模型整体映射（§2.1，Eq. (1)）：(τt,v̂t:t+H,ât:t+H)=ℳΘ(a<t,l,o≤t,s≤t)——CoT 轨迹 + 想象未来视频 + 动作 chunk 三输出。

### 【目标表征】
- 显式「目标状态」= 推理压缩成的 loop state z（§2.3，Eq. (5)）："We then summarize the trace into a dense loop state z=Pϕ(hτt)，where hτt is the VLM hidden state at the end of τt … z is the only interface through which downstream future generation accesses the model's understanding and reasoning."——z「指定未来应当发生什么」。
- CoT 双轨结构（§2.3）："Video CoT—a description and explanation of how the scene is expected to evolve"；"Action CoT—an explanation and decomposition of the motor program that should realize that future"——目标以语言形式显式化为「未来场景描述+动作计划」，再压缩为 z 条件 UFG。
- 统一生成（§2.4–2.5）：UFG（Wan2.2 初始化 DiT）以同一 z（cross-attention）联合去噪未来视频与动作 chunk；视频侧额外以观测前缀为条件（Eq. (9) 前缀保持 clean、仅未来区加噪）；"The action stream takes no further condition beyond z."
- 组合泛化的目标机制（§4.1）："VLA baselines fail at this transition … because their action distributions carry no representation of 'what should happen after A is done'. The imagination face … can render the post-A scene state and re-condition on it."——目标=「A 完成后的未来场景」可被想象面再条件化。
- 人类评估口径（§3.3）："The system treats first-frame conditions as manipulation goals and commits to completing them."

### 【代码依据】
- 全文未见 GitHub URL（获取范围：v2 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何统一训练不牺牲专家能力？§3 证据：VLM 8 基准 64.7（基线 Qwen3-VL-4B 58.2）、RoboTwin 93.5、WorldArena 66.03 rank1；§5.1 归因于 "coupled training signal"（loop-closed data）。但 z 内部「语义/预测/动作」三种信息的可分离性无探针实验（原文未写）。
2. 为何 CoT 必须是「视频+动作」双轨？§2.3："the model is forced to think about 'what should happen' and 'what should be done' under a single causal pass."——无单轨 CoT 消融（原文未写）。
3. 为何视频侧要加观测前缀条件而动作侧不加（§2.5）？原文仅陈述设计（"The action stream takes no further condition beyond z."），无消融。

### 【关键数值】（逐字抄录）
- VLM 基准（§3.1，Table 1）：avg 64.7（8 基准；Qwen3-VL-4B-Instruct 58.2；MolmoAct 27.5）；Where2Place +28.2、PhyX +20.6（vs 4B 基线）。
- RoboTwin（§3.2，Table 2）：93.5 avg（Clean 93.6 / Randomized 93.3）；AIM 93.1、LingBot-VA 92.3、Fast-WAM 91.9、Motus 87.9、starVLA 88.3、π0.5 79.8；MotuBrain 95.9（第一）；31/50 任务 ≥95%、39/50 ≥90%、15 任务 100%。
- WorldArena（§3.3，Table 3）：EWM 66.03（rank #1）；3D Accuracy 98.13、Motion Quality 62.69（均 rank #1）、Visual Quality 63.43、Content Consistency 60.33、Physics Adherence 61.51、Controllability 59.28。
- 人类盲评（Table 4，0–2 标度）：overall mean 1.76（rank #1）、Task Success 1.81、Controllability 2.00；Seedance2.0 1.55。
- 真机（§4）：UR5e 组合任务（plug RJ45 + waterproof，训练无组合演示）；Tienkung 人形：5 seen 任务（各约 300 video-action episodes）+ 3 unseen 任务（各 50 video sequences）zero-shot——逐任务成功率仅 Fig. 5 可视化 → 未取到（HTML 文本无数值表）。

### 【一页总结】
Pelican-Unify 1.0 是「理解-推理-想象-动作」单环训练的统一模型：单 VLM（Qwen3-VL 初始化）把「场景+指令+视觉上下文+动作历史」映射进共享语义空间，自回归生成双轨 CoT（Video CoT + Action CoT）后把末隐藏态投影为 loop state z；UFG（Wan2.2 DiT）以 z 为唯一条件、共享去噪主干联合生成未来视频与动作 chunk（视频侧加观测前缀条件），语言/视频/动作三损失全部回传 z。目标表征 = 「语言化未来推理（CoT）」压缩成的 z——10 篇中最显式的「可语言监督的目标状态」，且目标是理解、推理、想象、动作共享的单一接口。

### 【待核实】
- 代码仓库 URL；单轨 CoT 消融、z 信息可分离性探针（原文未写）；Tienkung 零样本逐任务成功率（仅图内）；动作头/视频头 λ 权重数值（Eq. (14) 只写加权和，未给值）。

---

## 10. WALL-WM（arXiv 2606.01955）

### 元信息
- 标题：WALL-WM: Carving World Action Modeling at the Event Joints
- 版本：v2，2026-09-06（v1 2026-06-01）；CC BY-NC-ND 4.0；作者：Shalfun Li, Victor Yao, Charles Yang 等（WALL 团队）；主页：abs 页未列（获取范围：abs+HTML）
- 来源级别：**A**。arxiv.org/html/2606.01955v2 本地化成功（HTML 601,655 字节；txt 177,473 字符 / 1,019 行，§1–§8+References+附录完整）。本地备份：`auto_docs/reading/html/2606.01955.html`、`txt/2606.01955.txt`、`pdfs/2606.01955.pdf`（43.6MB）。

### 【输入表征】
- 模型映射（§3.1）：pθ(𝐕e,𝐚e∣𝐕0,𝐬,ce)——当前多视角观测 𝐕0（每相机一 keyframe）+ 本体感状态 𝐬 + per-event caption ce → 事件对齐的未来多视角视频 𝐕e 与末端轨迹 𝐚e（长度随事件变化）。
- 视频塔（§3.2）：Wan 单视角 DiT 扩展多视角：Camera RoPE（每相机可学习旋转身份）+ sight-cone masking + tube patch masking（均仅训练期）；Wan-style v-prediction flow matching。
- 动作塔（§3.3）："A shared MLP encoder ℰa:ℝDa→ℝd embeds both the proprioceptive state and the noisy actions"；动作块 (a) 自注意力 (b) 对状态 token 的专用 cross-attention (c) 对配对视频层特征的逐层单向 cross-attention——"Coupling runs in one direction only: each action block cross-attends to per-layer video features, and no auxiliary branch is added to the video DiT."；相对位姿动作流（anchor token=0 固定时间原点）。
- 语言侧（§3.4）：Qwen3.5-9B VLM 文本 token → 𝒫Q 投影 + DiT 原生文本 MLP → cℓ 注入 cross-attention；Staircase latent CoT：Kc 个连续潜在推理状态，深度并行 relay（relay depth Nr），由冻结 Qwen3.5-0.8B 做 latent-to-text 重构监督（ℒCoT，Eq. (19)）。

### 【目标表征】
- 目标 = 「下一个语义事件的文本描述」ce（§3.1/§5.5.1）：event mode 下 "a VLM, human, or agent proposes the next-event description for the current observation, and WALL-WM executes the corresponding variable-length video–action segment before observing the next state, so that rollout follows the natural duration of each task stage rather than a fixed control horizon."
- 粒度不匹配论（Abstract）："Language describes semantic goals and events, vision evolves through continuous scene dynamics, and actions operate at control-level timescales; forcing all three into the same fixed-length prediction window turns VLA training into short-horizon correlation fitting."——事件是语言/视觉/动作对齐的原子学习单元。
- 统一模式的目标三源（§5.5.2）：同一 WAM 推理接口可切换三种文本条件源——"reusing one encoded global instruction"（gradient-continuous）/ 每 chunk 一条原子指令 / VLM-CoT staircase 潜状态，"Because every source only replaces the cross-attention text input, sources can be switched mid-rollout without altering denoiser state."
- 目标层级分工（§8）："Our preference is to factor information from high-level tasks into per-event textual descriptions, and then from per-event language into vision and action."

### 【代码依据】
- 全文未见 GitHub URL（获取范围：v2 全文）。待 GitHub 取证。

### 【why 追问】
1. 为何固定长度 chunk 是「短程相关拟合」？§1/§3.1："each sample is an atomic (𝐕e,𝐚e) event carved out of a long-horizon episode rather than a fixed-length chunk"；证据：事件模式比 fixed-length unified（WALL-WM-U-Scratch）在 Diverse 高出 12.86 分、Generalization 高出 35.25 分（§7.2.1/7.2.4）。但 chunk 长度与事件边界的定量错配分析原文未写。
2. 为何动作塔用「单向逐层耦合」而不回传梯度到视频塔？§3.3 只陈述设计（"no auxiliary branch is added"）；§5.1 主配方视频流冻结训练动作塔（asymmetric 1-to-Nd，anchor s⋆=45）——但单向信息流的消融（双向耦合 vs 单向）未做。
3. 为何 anchor s⋆=45（§3.3/§5.5）？原文："We pin s⋆ at the step that best balances faithful visual structure with usable cross-attention evidence"——具体选择依据的曲线未给出。

### 【关键数值】（逐字抄录）
- 真机 Task Progress（§7.2，0–100）：Diverse 75.86（WALL-WM-U-Scratch 63.00 / π0.5 55.64 / DreamZero 39.97 / LingBot-VA 29.71）；Reasoning 71.60（59.50/56.40/32.70/31.60）；Dexterous 32.00（U-Scratch 31.25）；Generalization 53.75（U-Scratch 18.50 / DreamZero 28.50 / π0.5 24.00）。
- 消融（Table 4）：事件+VI-SA vs 无 VI-SA fixed-length：Reasoning avg 32.6→71.6（Press Button in Order 0→64）；Generalization avg 22.0→53.75（Push Cleaning Cloth 26→78）。
- RoboTwin 零样本（附录 9.5，Table 5）：平均 15.2%（76/500 episodes），26/50 任务至少一次成功；click_bell 90%、press_stapler 80%、click_alarmclock 60%。
- 推理参数（§3.3/§5.5）：Nd=50 步，anchor s⋆=45；相机数 Nv=3；动作去噪时动作每步读同一 anchored video forward。
- 数据（§4）：四尺度 caption（Task/Subtask/Action/Segment）+ cluster-balanced sampling + contact-rich recovery data；Muon 优化器大规模预训练（Abstract）。
- 规模（§8）："Our current WALL-WM family spans model sizes from below 10B parameters to the tens-of-billions regime."

### 【一页总结】
WALL-WM 把「事件」设为语言/视觉/动作对齐的原子学习单元：以「当前多视角观测 + 本体感 + 下一事件描述 ce」为条件，联合去噪「事件对齐的未来多视角视频 + 末端轨迹」（长度随事件可变）；视频塔（Wan DiT + Camera RoPE/视线锥掩码/管道掩码）冻结时动作塔经单向逐层 cross-attention 训练（anchor s⋆ 协议省去全视频去噪）。目标表征 = **下一事件文本描述**（人类/VLM 给出）→ 事件对齐的未来视觉+动作；固定长度场景则以 Staircase 潜在 CoT 作为 chunk 级目标结构。意义：10 篇中唯一以「语义事件」为目标的粒度重构工作，把目标表征从指令/帧提升到事件级。

### 【待核实】
- 代码仓库 URL；模型参数量具体值（仅 "below 10B to tens-of-billions"）；anchor s⋆ 选择曲线、双向耦合消融（原文未写）；真机平台型号（"internally developed deployment platform suite"，未具体命名）。

---

## 汇总对比表（10 篇：本文件 8 篇 + GR-2、VPP）

| # | 论文 | 输入表征（原文依据） | 目标表征（原文依据） | 未来视觉的角色 | 来源级别 |
|---|---|---|---|---|---|
| 1 | GR-2 | 语言(冻结 CLIP)+历史视频帧(VQGAN token)+状态(线性层)（§2.1） | 仅语言指令；无显式 goal（§3.4 明确拒绝 goal-image 路线） | 训练期视频生成，§3.4 称其"act as a planner for action generation" | A |
| 2 | VPP | 语言(CLIP)+当前帧→SVD 潜表征 (T,H,W)（§4.1–4.2） | 仅语言指令 l_emb；无显式 goal | 单步前向的「预测性视觉表征」直接作动作头输入（encoder 而非 denoiser） | A |
| 3 | UVA | 历史观测 VAE latent(N tokens)+动作 chunk(重复 M)+可选语言(CLIP)（§III-A/B） | 图像(PushT-M)或语言(Libero10)，Table I caption；无独立 goal 模块 | 训练期监督 + joint latent Z；推理跳过视频头（§III-C） | A |
| 4 | Genie Envisioner | 初始观测三视角+稀疏记忆+T5-XXL 指令（§2.1）；GE-Act 同 latent 条件 | 语言指令；"until the task specified by the instruction is fully executed"（§2.3） | GE-Base 生成；GE-Act 中退化为 1 步 denoising 缓存条件（§3.3）；GE-Sim 作仿真 | A |
| 5 | LingBot-VA | causal VAE latent(N=192/帧)+动作 MLP token 交织(τ=4)+T5 指令（§3.2–3.3） | 语言指令 + 自预测「期望未来观测」作逆动力学条件（§3.2） | 推理期显式生成（部分去噪 s=0.6），FDM-grounded 闭环刷新 | A |
| 6 | Rift | first-frame tokens+指令+anticipation tokens E(m=196/240)（§4.2） | 「未来 K/V cache」：内容+位置敏感、fixed final-clean 可复用（§3.4–3.5） | 一次前向 CachePrefill 构造，无 rollout、无 VAE 解码 | A |
| 7 | WAM-TTT | 基础 WAM 观测+目标 𝒈 + TTT fast-weight 记忆（人类视频吸收）（§3.1） | 人类演示视频（任务变体/用户偏好）；"absorbing them as fast-weight memory"（§4.2） | 人类视频预测损失驱动记忆更新，经视频流间接影响动作专家 | A |
| 8 | HOST | 人类视频窗口+指令+近期观测+本体感（Wan VAE 共享 latent）（§4.1/4.3） | 演示视频的未来进展→机器人自身未来观测（共享进度流形）（§4.2–4.3） | 显式中间目标：因果级联 p→o→a 三级 | A |
| 9 | Pelican-Unify | 动作历史+指令+观测+本体感→共享嵌入器→VLM（§2.2） | 双轨 CoT(视频+动作)→loop state z，"specifies what future should happen"（§2.3） | UFG 以 z 联合生成未来视频+动作 chunk（同去噪过程） | A |
| 10 | WALL-WM | 多视角观测+本体感+事件 caption ce（§3.1）；Staircase 潜 CoT（§3.4） | 下一事件描述 ce（事件模式）/全局指令+潜 CoT（统一模式）（§5.5） | 事件对齐的未来视频+轨迹联合去噪（可变长度） | A |

## 趋势（仅基于原文）

1. **目标表征从「语言指令」向「未来视觉/未来表征」迁移**：GR-2/VPP（语言为主，未来视频是训练期信号）→ UVA（joint latent 统一未来+动作）→ LingBot-VA（推理期显式生成未来观测作逆动力学目标）→ Rift（未来 K/V cache 成为目标表征本身）→ Pelican-Unify（语言化 CoT 压缩为 z 再驱动未来生成）→ WALL-WM（事件级目标）。原文直接证据：Rift §3.5 "One fixed final-clean K/V cache nearly preserves execution"；LingBot-VA §3.2 "infer actions by conditioning on desired future observations"。
2. **「目标=未来」的去 rollout 化/轻量化**：GE-Act 把未来视频降为 1 步 denoising 缓存（§3.3）；LingBot-VA 部分去噪 s=0.6（§3.3/§4.2）；Rift 一次前向 CachePrefill（§4.2）——生成式 rollout 向「一次性目标表征」收敛，延迟从 rollout 方法的 780–2270ms 降至 248–420ms（Rift Table 1/3）。
3. **目标来源多元化**：语言指令（GR-2/VPP/UVA/GE/LingBot/Pelican）→ 人类视频（WAM-TTT：fast-weight 吸收；HOST：唯一做跨具身目标翻译——human video→robot future obs→action）→ 事件描述（WALL-WM）。WAM-TTT 与 HOST 的共同点（§1/§4.2）：把视频当目标载体而非被动上下文。
4. **目标粒度对齐成为显式问题**：WALL-WM Abstract 明言语言/视觉/动作三粒度不匹配（"forcing all three into the same fixed-length prediction window turns VLA training into short-horizon correlation fitting"）；HOST 以共享任务进度流形对齐人机轨迹（§4.2）；UVA/GE/LingBot 以「视频低帧率 vs 动作高帧率」的比率解耦（UVA 每帧 L 动作、GE 5Hz/30Hz=1:6、LingBot τ=4）。
5. **共同缺口（10 篇原文均未覆盖）**：除 Rift 的 cache 干预外，无一篇做「语言目标 vs 未来视觉目标」的贡献剥离实验；目标表征内部结构（z/latent/cache 中任务语义的布局）普遍无探针分析；所有模型的代码仓库均在取证中（见各篇【待核实】）。

---

<!-- QA_LOG -->
- 取证方式：复用前两轮已本地化的 HTML（arxiv.org/html/<id><最新版>），经 lxml 抽文本为 txt（auto_docs/reading/html/、txt/、pdfs/），逐字引用以 txt+HTML 原 markup 双源核对；版本经 abs 页确认全部为最新版（UVA v3、GE v3、LingBot v2、Rift v3、WAM-TTT v3、HOST v4、Pelican v2、WALL-WM v2）。
- 网络状态：本会话 8 个 abs 页 web_fetch 全部 200，无 L1 降级案例；8 篇来源级别均为 A（全文完整，含 LaTeXML 页脚、References、附录）。
- 渲染伪影处理：GE 篇 txt 出现 "2,9672,967"、"200200ms"（MathML mn+annotation 重复），已按 HTML 原 markup 核为 2,967 / 200；UVA 篇表格交叉引用 "Table 3.4" 为 LaTeXML 渲染产物（对应 §3.4 Analysis 之 Table 1），未采信。
- 明确「未取到」项（仅图内数值）：GE AgiBot G1 五任务 SR/E2E（Fig. 8）；LingBot 真机 6 任务逐任务成功率（Fig. 5）、记忆任务数值（Fig. 9）；Pelican Tienkung 零样本逐任务成功率（Fig. 5）；Rift m=2 精确值（Fig. 5）。
- 口径冲突已记录：LingBot §6 "92.0% on RoboTwin 2.0" vs §4.3.2 "92.9% (Easy)"；UVA 20%/5% 为绝对百分点（可自 Table II 验算）。
- 待后续 GitHub 取证（8 篇全文均未给代码 URL；LingBot abs 页有 github.com/robbyant/lingbot-va 待核）。


