# DriveWAM 精读笔记

> 数据源：`auto_docs/papers/txt/DriveWAM.txt`（947 行，arXiv 2605.28544）。所有英文引用逐字复制自 txt 并注行号；txt 为 PDF 抽取文本，含排版产物：`DriveW AM`=DriveWAM、`V oyager`=Voyager、`NA VSIM`=NAVSIM、`W A`=WA、`V AE`=VAE，引用时不作修正。

## 1. 原文信息

- **标题全称**（L1–2）："DriveW AM: Video Generative Priors Enable Scalable / World-Action Modeling for Autonomous Driving"
- **作者**（L3）："Chen Shi1∗ Jinrui Xu1∗ Shaoshuai Shi2 Kehua Sheng2 Bo Zhang2 Li Jiang1†"；脚注（L43）："∗:Equal Contribution. Work done during an internship at V oyager Research, Didi Chuxing.†: Corresponding author."
- **机构**（L4）："1The Chinese University of Hong Kong, Shenzhen 2V oyager Research, Didi Chuxing"（香港中文大学（深圳）；滴滴出行 Voyager Research）
- **ID/状态**（L44–45）："Preprint."；"arXiv:2605.28544v1 [cs.CV] 27 May 2026"（arXiv 预印本 v1，未标注会议/期刊录用信息）
- **项目页**（L5）："Project Page:https://chenshi3.github.io/drivewam.github.io/"

## 2. Abstract 全文逐字（L6–24）

> Pretrained foundation models have become an important basis for end-to-end autonomous driving. In contrast to vision-language models pretrained primarily on static image-text pairs, video generative models capture temporal dynamics and motion priors that are naturally suited for driving. We present DriveWAM, a driving world-action model that adapts a pretrained video diffusion transformer into an autoregressive video-action policy. DriveW AM organizes video and action streams into a unified temporal token sequence and trains them under a joint flow-matching objective, preserving the pretrained video-generation architecture while adapting its large-scale video priors to action generation. To incorporate high-level scene understanding, we introduce scene-evolving driving guidance, where a frozen VLM produces chunk-specific semantic intent to guide video-action generation. To keep long-horizon rollout bounded, we further introduce selective KV memory, which maintains bounded modality-aware video and action memory pools through relevance-redundancy cache selection at inference time. Experiments on NA VSIM and the PhysicalAI-Autonomous-Vehicles benchmark show that DriveWAM achieves strong planning performance, and a data-scaling study from 4k to 100k driving clips further confirms the scaling potential of world-action modeling for end-to-end autonomous driving.

## 3. 问题与动机（为什么视频生成先验能规模化 world-action modeling）

论文的动机链条分三层。

**第一层：VLM 先验的缺口。** 端到端驾驶系统越来越多地以预训练基础模型为策略骨干，但主流 VLA 路线基于图像-文本预训练的 VLM，缺少时序先验：

> "Recent end-to-end autonomous driving systems increasingly leverage pretrained foundation models as policy backbones." (L26–27)

> "driving decisions also require temporally dense visual cues such as spatial layout, motion continuity, and how the scene may evolve in the near future. Since VLM backbones are pretrained primarily on image-text data rather than video dynamics, VLM-centric policies must acquire these temporal priors largely from downstream driving data." (L30–33)

**第二层：视频生成模型是互补的基础。** 视频生成模型在大规模视频上学习物体持续性、运动模式与场景演化，天然适配动态决策问题——这是「视频生成先验」的核心论据：

> "Video generative models offer a complementary foundation. They are pretrained on large-scale videos to model object persistence, motion patterns, and scene evolution, making them naturally suited for dynamic decision problems." (L34–36)

> "Recent VLA-based driving methods [3, 14, 15] have begun to incorporate future image or video generation to improve spatio-temporal awareness, but visual generation is often used as an auxiliary signal or a modular component on top of a VLM-centric policy." (L36–38)

**第三层：机器人学 WA 模型的迁移依据 + 驾驶域的三大难点。** 论文明确援引机器人学 world-action 模型作为范式来源，随后列出三个非平凡的适应难点（对应其三大贡献）：

> "In parallel, world-action (WA) models in robotics [16–21] show that pretrained video foundation models can be adapted more directly for action prediction and planning." (L39–40)

> "First, a video foundation model is pretrained for visual generation rather than ego-action control, so turning it into an autoregressive video-action policy requires preserving its future-generation prior while coupling it to continuous action prediction. Second, video foundation models capture near-future dynamics but lack high-level semantic planning, whereas the appropriate driving decision depends on route intent, right-of-way, and decision-relevant traffic participants. Third, deploying such autoregressive policies over long horizons requires persistent historical context, but full KV caching grows with horizon length and sliding-window caching may discard old yet critical evidence." (L41–51)

既有驾驶域 WA 方法的缺口与「规模化」落点：

> "Existing driving-oriented world-action methods [22–24] often rely on separate planners, discrete video tokenizers, or customized generation architectures, leaving open how to directly adapt a modern video foundation model into a semantically guided and scalable end-to-end driving policy." (L51–54)

## 4. 方法逐机制

### 4.1 视频生成先验如何注入（backbone 复用 + 保留原生目标）

DriveWAM 直接把预训练视频扩散 Transformer 当作策略核心，从 checkpoint 初始化并全量微调：

> "DriveW AM uses a flow-matching video diffusion transformer as the policy core and formulates driving as autoregressive video-action generation." (L57–58)

> "DriveW AM uses Wan2.2-TI2V-5B [40] as the video backbone, initialized from the base checkpoint released by [19]. Unless otherwise specified, we fine-tune the full video diffusion transformer together with the newly introduced action and ego-state modules." (L415–417)

视频先验的注入点有两个：(a) 观测 chunk 用预训练 VAE 编码，且 VAE latent 经「从预训练视频扩散 Transformer 继承的 latent input embedding」映射，构成统一表示；(b) 视频分支保留原生未来视觉预测目标，把视频模型「改造成策略先验」：

> "Each observed video chunk is encoded by the pretrained V AE [40], and each ego-action chunk, represented as normalized ego-frame translation and yaw increments, is embedded by an MLP action encoder E a, as follows: zk = V AE(xk), u k =E a(ak), H k ={(z i, ui)}i≤k. (1)" (L216–219)

> "In practice, the V AE latents zk are also mapped to dimension d by the latent input embedding layer inherited from the pretrained video diffusion transformer, yielding a unified representation for video-action generation." (L222–224)

> "Notably, this conditioning repurposes the video model as a policy prior, with the backbone retaining its native future-visual-prediction objective while the predicted future is shaped by driving history, ego state, and semantic intent." (L235–237)

### 4.2 世界模型与动作模型的耦合（先想象未来、再读动作：inverse dynamics）

耦合方式是把驾驶任务分解为「未来世界建模 + 逆动力学动作生成」，视频、动作共享同一 Transformer，在联合 flow-matching 目标下训练。视频分支（式 2）在上下文 (H_k, e_k, g_k) 条件下预测视频速度场；动作分支（式 3）在**生成出的未来世界 latent** 条件下预测动作速度场，经 MLP 动作解码器 Da 输出：

> "Given observed video-action history and ego state, the model first generates future video latents and then decodes ego actions conditioned on the generated future latent, realizing inverse-dynamics action generation." (L58–60)

> "Both video and action streams share the same transformer and are trained under a joint flow-matching objective [25], preserving the pretrained spatio-temporal generative prior while learning to convert imagined future world evolution into executable ego motion." (L60–62)

> "Actions are generated by an inverse-dynamics flow on the same diffusion transformer. We perturb the next action chunk directly in the normalized action space and embed it with the MLP action encoder Ea to obtain uk+1,τ . Conditioned on the future world latent and the current driving context, the shared transformer predicts the action velocity as: ˆvak+1,τ =D a(Tω(uk+1,τ ; ˜zk+1, Hk, ek, gk, τ)), (3)" (L238–242)

> "The conditioning latent ˜zk+1 is the clean future video latent zk+1 during teacher-forced training and the generated latent ˆzk+1 during inference. This design grounds action generation in the predicted world evolution, so the action decoder behaves as an inverse-dynamics readout of the predicted future rather than an independent trajectory head. We use noisy-history augmentation [19] to reduce this train-test mismatch." (L244–248)

联合目标（式 4）与训练方式（整段 clip 单次前向 + 因果 teacher-forcing 掩码）：

> "We train the video and action branches with a joint flow-matching objective: L =E k,τ [∥ˆvzk+1,τ −v zk+1,τ ∥2 2 +β a ∥ˆvak+1,τ −v ak+1,τ ∥2 2], (4) where βa controls the balance between future world modeling and action generation. The video term preserves the pretrained spatio-temporal generative prior during policy adaptation, while the action term teaches the shared backbone to decode this prior into executable ego motion." (L249–254)

> "During training, we process all chunks of a clip in a single forward pass for efficiency. The video-action tokens are arranged in temporal order and denoised in parallel under a causal teacher-forcing mask (Figure 2), which realizes the conditional dependencies in Eqs. 2 and 3 while preserving the causal pattern used during inference [45, 19, 44]." (L255–258)

> "At inference, DriveWAM rolls out one chunk at a time. Given history Hk, the model first samples the future video latent ˆzk+1 and then samples the action chunk ˆak+1 conditioned on this generated future. When the next real observation becomes available, it is encoded and appended to the history to form H k+1, keeping long-horizon rollout grounded in observed driving context." (L259–262)

### 4.3 场景演化引导（冻结 VLM，chunk 级语义意图）

> "At each decision step k, the frozen Qwen3-VL-8B [9] receives only causally available information: the latest observation xk, a recent ego trajectory ak, and the route command c k for the upcoming horizon. It produces a concise guidance text as follows: gk = ΦVLM(xk, ak, ck), (5) which summarizes the current road context and provides ego behavior guidance for the upcoming horizon, such as proceeding, yielding, stopping, or merging. Since no observation from the target chunk is used, gk provides semantic intent for predicting (xk+1, ak+1) without leaking future information." (L274–281)

> "During training, guidance texts are precomputed and cached; during inference, the VLM is queried once per decision step and reused across all denoising steps, keeping the semantic condition aligned with the current prediction horizon." (L281–283)

引导注入方式与防泄漏设计：

> "We therefore apply an additional block-diagonal text mask, which allows video-action tokens of target chunk k+ 1 to attend only to the guidance tokens of gk. This keeps semantic conditioning temporally localized and prevents cross-chunk leakage." (L287–289)

> "Thus, the VLM acts as a semantic guide, while the video foundation model remains responsible for dense temporal prediction." (L68–69)

### 4.4 选择性 KV 记忆（推理时、免训练、模态分离内存池）

> "To keep long-horizon inference bounded without discarding useful context, DriveW AM adopts an inference-time, training-free selective KV memory inspired by FlowCache [26], retaining a compact, prediction-relevant approximation of H k during rollout." (L334–336)

> "Video tokens are numerous and encode scene context, while action tokens are compact and encode ego-motion history. A single global cache would therefore be dominated by visual tokens and may under-preserve motion context. DriveW AM decomposes Hk into two bounded modality pools H vk and H ak , with |H vk | ≤Bv and |H ak | ≤Ba, where Bv and Ba are the video and action memory budgets." (L338–342)

打分规则：相关性 = 当前 query 对缓存 token 的平均注意力质量；冗余 = 与其他缓存 key 的平均相似度；留存分数 s = λ·relevance − (1−λ)·redundancy（式 6–7，L344–359）：

> "We measure relevance ρmj by the average attention mass assigned to token j from current queries, and redundancy η mj by its average similarity to other cached keys: … The final retention score is: smj =λρ mj −(1−λ)η mj , (7) where λ∈[0,1] balances relevance and redundancy, and tokens with low scores are evicted." (L347–348, L357–359)

> "As shown in Figure 3, this criterion has a natural driving-oriented interpretation: repeated road surfaces, sky, buildings, and other static background regions tend to be filtered out, while prediction-relevant cues such as moving vehicles and lane geometry are more likely to be retained." (L360–362)

> "Selective KV memory is applied only at inference time and does not change the training objective or model parameters." (L363–364)

推理超参：

> "For inference, following [19], we use an Euler ODE solver with 3 steps for video tokens and 10 steps for action tokens. The video solver integrates the flow trajectory from τ= 1 to τ= 0.6 , while the action solver integrates from τ= 1 to τ= 0 . For selective KV memory, we follow FlowCache [26] and set λ= 0.07. The video and action cache capacities are set to 448 and 160 tokens, respectively." (L455–458)

### 4.5 训练数据与模态/视角设置

输入模态（无 LiDAR、无多视角）：

> "A driving clip contains synchronized streams of camera images, ego actions, and ego states." (L146)

> "The causally available conditions include the historical context Hk (video and action tokens of all observed chunks up to k), the ego state ek at the end frame of chunk k (e.g., velocity, acceleration, and curvature), and a textual guidance g k for the predicted chunk." (L213–214)

物理 AI 基准上只用前视单目：

> "We use the front-view camera stream and ego-motion labels." (L408)

训练数据：

> "NA VSIM[27] is a standard end-to-end planning benchmark derived from OpenScene [60, 61], with 103k trainval samples and 12k test samples." (L402–403)

> "PhysicalAI-Autonomous-Vehicles is a large-scale real-world driving benchmark released with Alpamayo-R1 [28]. It contains approximately 1,700 hours of driving logs, organized into 306,152 clips of 20 seconds each, with 153,625 clips for training, 90,928 for validation, and 61,599 for testing. We use the front-view camera stream and ego-motion labels. To focus on non-trivial driving scenarios, we use a VLM to tag each clip with a scene description and filter out simple scenes. Finally, we select 100k clips from the training split, and construct a curated 1,000-clip test subset from the test split." (L406–411)

训练配置：

> "All models are trained at 256×448 resolution on 48 NVIDIA H20 GPUs. We use AdamW [62] with β= (0.9,0.95) , weight decay 0.1, learning rate 1×10−5, and per-device batch size 1. The action loss weight is set to βa = 1.0. DriveWAM uses a 4-second chunk for video-action generation." (L446–448)

> "On the PhysicalAI-Autonomous-Vehicles benchmark, we train for 50k iterations. Each training sample is a 12-second segment randomly cropped from a 20-second clip. The video stream is downsampled to 1 Hz, while ego actions remain at 10 Hz." (L452–454)

数据筛选（Appendix A）：

> "We first retain all clips with interest score no smaller than 2.0, preserving rare-event and interaction-rich cases. We then uniformly sample 50% of the remaining lower-score clips, so that ordinary driving scenarios are still represented without dominating the training distribution." (L754–756)

## 5. 实验与主结果

### 5.1 NAVSIM v1（表 1 全表抄录）

> "As shown in Table 1, DriveWAM achieves a PDMS of 90.1 using only a single front-view camera, outperforming all competing methods under comparable training settings. We attribute this to the underlying video generative backbone, which provides effective spatio-temporal priors for modeling scene geometry, motion dynamics, and fine-grained action prediction." (L462–465)

| Method | Ref | Sensors | NC↑ | DAC↑ | TTC↑ | C.↑ | EP↑ | PDMS↑ |
|---|---|---|---|---|---|---|---|---|
| Human | – | – | 100 | 100 | 100 | 99.9 | 87.5 | 94.8 |
| UniAD [54] | CVPR'23 | MV | 97.8 | 91.9 | 92.9 | 100.0 | 78.8 | 83.4 |
| TransFuser [55] | TPAMI'23 | MV & L | 97.7 | 92.8 | 92.8 | 100.0 | 79.2 | 84.0 |
| PARA-Drive [56] | CVPR'24 | MV | 97.9 | 92.4 | 93.0 | 99.8 | 79.3 | 84.0 |
| LAW [57] | ICLR'25 | SV | 96.4 | 95.4 | 88.7 | 99.9 | 81.7 | 84.6 |
| DiffusionDrive [58] | CVPR'25 | MV & L | 98.2 | 96.2 | 94.7 | 100.0 | 82.2 | 88.1 |
| WoTE [59] | ICCV'25 | MV & L | 98.5 | 96.8 | 94.4 | 99.9 | 81.9 | 88.3 |
| ReCogDrive∗ [4] | ICLR'26 | MV | 98.1 | 94.7 | 94.2 | 100.0 | 80.9 | 86.5 |
| DriveVLA-W0 [3] | ICLR'26 | SV | 98.7 | 96.2 | 95.5 | 100.0 | 82.2 | 88.4 |
| AutoVLA [37] | NeurIPS'25 | MV | 98.4 | 95.6 | 98.0 | 99.9 | 81.9 | 89.1 |
| DriveDreamer-Policy [14] | arXiv'26 | MV | 98.4 | 97.1 | 95.1 | 100.0 | 83.5 | 89.2 |
| DriveVLA-W0† [3] | ICLR'26 | SV | 98.7 | 99.1 | 95.3 | 99.3 | 83.3 | 90.2 |
| Epona [24] | ICCV'25 | SV | 97.9 | 95.1 | 93.8 | 99.9 | 80.4 | 86.2 |
| WorldDrive [22] | arXiv'26 | SV | 98.4 | 95.8 | 95.2 | 99.8 | 83.3 | 89.0 |
| **DriveWAM** | – | SV | **98.3** | **98.1** | **95.2** | **100.0** | **84.3** | **90.1** |

### 5.2 PhysicalAI-Autonomous-Vehicles（表 2 全表抄录）

> "We evaluate DriveWAM on the large-scale PhysicalAI-Autonomous-Vehicles benchmark, comparing against the WA-based VaV AM [23], trained on approximately 1,700 hours of OpenDV [63] driving data, and the VLA-based Alpamayo-1.5 [28], trained on roughly 80,000 hours of data containing the PhysicalAI-Autonomous-Vehicles training set." (L466–470)

> "As reported in Table 2, DriveW AM achieves ADE/FDE of 0.47/1.35 at 3 seconds and 0.83/2.47 at 4 seconds, substantially outperforming both baselines." (L471–472)

| Method | Source | Sensors | # Params | ADE@3s↓ | FDE@3s↓ | ADE@4s↓ | FDE@4s↓ |
|---|---|---|---|---|---|---|---|
| VaVAM∗ [23] | Valeo | SV | 1.3B | 2.31 | 4.32 | – | – |
| Alpamayo-1.5 [28] | NVIDIA | SV | 10B | 0.80 | 2.31 | 1.44 | 4.18 |
| **DriveWAM** | – | SV | 5B + 8B | **0.47** | **1.35** | **0.83** | **2.47** |

### 5.3 推理效率（表 6 全表抄录）

> "We analyze the per-chunk inference cost of DriveW AM and compare it against Alpamayo-1.5 [28] on a single NVIDIA H20 GPU. As shown in Table 6, each inference pass consists of three stages: (1) VLM guidance generation, (2) video generation, and (3) action denoising." (L900–902)

> "DriveW AM queries a frozen Qwen3-VL-8B once per 4-second chunk, taking 125 ms with the default vLLM compilation. … Alpamayo-1.5 processes a substantially larger number of visual tokens per query, which accounts for its higher VLM latency of 570 ms." (L903–906)

> "DriveW AM generates a 4-second video clip using a 3-step Euler ODE solver over the video tokens, taking 372 ms. Alpamayo-1.5 does not perform explicit video generation." (L907–908)

> "By default, DriveW AM uses 10 denoising steps for action tokens, taking 765 ms. We find that reducing the steps from 10 to 5 incurs negligible change in trajectory metrics, while reducing action denoising time to 374 ms. The 5-step variant (DriveW AM∗) brings the total per-chunk cost to approximately 871 ms, comparable to Alpamayo-1.5's 900 ms, while additionally producing a jointly generated future video." (L909–913)

| Method | VLM (ms) | Video Gen (ms) | Action (ms) | ADE@4s↓ | FDE@4s↓ |
|---|---|---|---|---|---|
| Alpamayo-1.5 | 570 | — | 330 | 1.44 | 4.18 |
| DriveWAM (Ours) | 125 | 372 | 765 | 0.83 | 2.47 |
| DriveWAM∗ (Ours) | 125 | 372 | 374 | 0.84 | 2.45 |

## 6. 消融（全部抄录）

**场景演化引导（表 3）**：

> "Replacing the global prompt with scene-evolving guidance consistently improves trajectory prediction at every training data scale, reducing ADE@4s from 1.21 to 1.01 with 4k clips and from 0.92 to 0.83 with 100k clips, while also yielding consistent reductions in FDE@4s. These results indicate that high-level scene reasoning provides a complementary semantic conditioning to the low-level W A backbone. We also observe that the benefit does not vanish as training data grows." (L491–495)

| # Clips | # Iters | SE Guidance | ADE@4s↓ | FDE@4s↓ |
|---|---|---|---|---|
| 4k | 50k | ✗ | 1.21 | 3.65 |
| 4k | 50k | ✓ | 1.01 | 2.95 |
| 20k | 50k | ✗ | 0.95 | 2.94 |
| 20k | 50k | ✓ | 0.94 | 2.65 |
| 100k | 50k | ✗ | 0.92 | 2.75 |
| 100k | 50k | ✓ | 0.83 | 2.47 |

**视频骨干初始化与联合视频监督（表 4）**：

> "training entirely from scratch removes the large-scale spatio-temporal priors inherited from video pretraining, and degrades ADE@4s/FDE@4s to 1.10/3.26. Initializing from the pretrained backbone but removing video supervision also performs poorly, yielding 1.23/3.79, suggesting that action-only adaptation fails to preserve the generative video priors needed for WA policy learning. The full configuration combines pretrained initialization with joint video-action flow-matching supervision and achieves the best performance." (L532–538)

| Pretrained init. | Video sup. | ADE@4s↓ | FDE@4s↓ |
|---|---|---|---|
| ✗ | ✓ | 1.10 | 3.26 |
| ✓ | ✗ | 1.23 | 3.79 |
| ✓ | ✓ | 0.83 | 2.47 |

**KV 记忆策略（表 5）**：

> "As shown in 1st and 3rd rows, selective KV memory largely closes the accuracy gap to full caching, achieving 0.89/2.52 ADE@4s/FDE@4s, while FIFO degrades substantially to 1.40/3.47." (L541–543)

> "full caching requires 3.07 GB memory and 17.37 GFLOPs per step, whereas selective KV memory reduces them to 0.25 GB and 1.44 GFLOPs, yielding over 12× reductions." (L547–548)

| KV memory | ADE@4s↓ | FDE@4s↓ | Mem. (GB)↓ | GFLOPs↓ |
|---|---|---|---|---|
| Full | 0.83 | 2.47 | 3.07 | 17.37 |
| FIFO | 1.40 | 3.47 | 0.25 | 1.05 |
| Selective | 0.89 | 2.52 | 0.25 | 1.44 |

**数据扩展（图 5 / 表 3）**：

> "We investigate the data scalability of DriveW AM by varying the training set size from 4k to 20k and 100k clips under a fixed 50k-iteration training procedure. As shown in Table 3 and Figure 5, both ADE@4s and FDE@4s improve significantly with more data, regardless of whether scene-evolving guidance is applied. This consistent scaling trend reflects the effectiveness of the video-action modeling as a scalable policy foundation, and suggests that DriveW AM has not yet saturated at the current data scale." (L520–529)

## 7. 局限/未来工作

**原文未写。** 全文（含正文、结论、附录 A–D）经检索无 "limitation" 或 "future work" 字样，没有独立的局限性或未来工作章节；结论章（L548–556）也无未来工作句。结论全文逐字：

> "We present DriveW AM, a unified world-action policy that adapts a pretrained video foundation model directly into an end-to-end driving policy. DriveWAM introduces scene-evolving driving guidance that injects chunk-specific semantic intent through temporally localized cross-attention, and selective KV memory that maintains modality-aware video and action memory pools via relevance-redundancy selection at inference time. Experiments on NA VSIM and the PhysicalAI-Autonomous-Vehicles benchmark show that DriveWAM achieves strong planning performance, and a data-scaling study from 4k to 100k clips further confirms its scalability." (L548–556)

两处可作「作者自行披露的边界」（非作者自称局限，仅客观陈述）：数据未饱和（"suggests that DriveW AM has not yet saturated at the current data scale." L528–529）；推理成本与 VLA 相当（"The 5-step variant (DriveW AM∗) brings the total per-chunk cost to approximately 871 ms, comparable to Alpamayo-1.5's 900 ms" L911–913）。

## 8. 叙事关联【推断】：「world action model」术语在驾驶域的定义与用法

1. **【推断】WA 的定义判据 = 把世界建模与动作预测放进同一个生成式 backbone 联合训练，动作由「想象的世界」解码出来（inverse-dynamics readout），而非世界模型作辅助分支。** 这是论文对驾驶域 WA 的界定核心。
   > "DriveW AM instead builds directly on a pretrained video diffusion transformer and adapts both video and action streams under a unified flow-matching objective." (L124–125)
   > "This design grounds action generation in the predicted world evolution, so the action decoder behaves as an inverse-dynamics readout of the predicted future rather than an independent trajectory head." (L246–247)

2. **【推断】术语中的 "world" 指显式生成的未来视觉演化（future world modeling），而非隐式状态表征。** 模型先生成未来视频 latent、再以其为条件解码动作。
   > "the model first generates future video latents and then decodes ego actions conditioned on the generated future latent, realizing inverse-dynamics action generation" (L58–60)
   > "which factors the driving task into future world modeling and inverse-dynamics action generation" (L225–226)

3. **【推断】论文把 WA 与 VLA 构造为两条对立的路线：VLA 以 VLM 为核心、视觉生成为辅助；WA 以视频生成模型为核心、VLM 降级为语义引导。**
   > "Although these designs improve the spatio-temporal awareness of VLA-based driving, their policy core remains VLM-centric, with visual generation serving as an auxiliary branch rather than the policy backbone." (L107–109)
   > "Thus, the VLM acts as a semantic guide, while the video foundation model remains responsible for dense temporal prediction." (L68–69)

4. **【推断】「scalable」在本文是双重的：数据规模化（4k→100k 未饱和）与推理长时域可规模化（有界 KV 内存），这也是 "Scalable World-Action Modeling" 题目的落点。**
   > "a data-scaling study from 4k to 100k driving clips further confirms the scaling potential of world-action modeling for end-to-end autonomous driving." (L22–24)
   > "To keep long-horizon inference bounded without discarding useful context, DriveW AM adopts an inference-time, training-free selective KV memory inspired by FlowCache [26]" (L334–336)

5. **【推断】论文隐含承认纯视频先验不足以做驾驶决策：WA 骨干仍需高层语义（路由意图、让行权、相关交通参与者），故引入冻结 VLM 的 chunk 级引导——即 WA 与 VLM 是分工互补而非互相替代。**
   > "In driving, the appropriate future is determined not only by short-term dynamics but also by route intent, traffic participants, and other decision-level semantics. For example, at an intersection, multiple future evolutions may be visually plausible from the current observation, while the desired one depends on the high-level driving intent." (L266–269)

## 9. 图注/表注逐字抄录

**图注**：

- Figure 1 (L209–212): "Figure 1: Overview of DriveWAM, which adapts a pretrained video generation backbone into a unified video-action policy. Building on this backbone, DriveW AM uses a frozen VLM to provide chunk-specific scene-evolving guidance for high-level scene reasoning and introduces selective KV memory to preserve compact prediction-relevant history for long-horizon rollout."
- Figure 2 (L310–312): "Figure 2: Attention mask used during Drive-WAM training. Colored entries indicate allowed attention; blank entries are masked."
- Figure 3 (L325–327): "Figure 3: Video-token retention under selective KV memory. Columns 2 and 3 visualize the tokens retained after query chunks 4 and 5."
- Figure 4 (L484–485): "Figure 4: Qualitative results on NA VSIM (left) and PhysicalAI-Autonomous-Vehicles benchmark (right). The predicted ego trajectories are consistent with the jointly generated future scenes."
- Figure 5 (L518–519): "Figure 5: Data scaling on PhysicalAI-Autonomous-Vehicles."
- Figure 6 (L805–808): "Figure 6: Representative scene tagging results for dataset curation. For each clip, the left panel shows Qwen3-VL-8B detected scene attributes, events, and the resulting interest score, while the right panel shows sampled front-view frames. High-score clips capture rare or interaction-rich scenarios, whereas low-score clips represent ordinary driving."
- Figure 7 (L863–864): "Figure 7: Examples of scene-evolving VLM guidance. The guidance adapts to changing scene context and route intent, such as pedestrians, traffic lights, construction barriers."
- Figure 8 (L933–935): "Figure 8: Qualitative results on NA VSIM (top two rows) and PhysicalAI-Autonomous-Vehicles (bottom two rows) benchmarks. Each row shows the predicted ego trajectory alongside the jointly generated future frames at T=1,2,3,4."

**表注**：

- Table 1 (L370–371): "Table 1: Comparison on NA VSIM v1.∗: results with imitation learning. †: trained with multiple trajectory anchors from [53]. MV: multi-view cameras; SV: single-view camera; L: LiDAR."
- Table 2 (L390–392): "Table 2: Comparison on our curated 1,000-clip test subset of PhysicalAI-Autonomous-Vehicles benchmark. # Params denotes the number of model parameters. SV: single-view camera. ∗: evaluated using the released checkpoint, which only supports up to 3s prediction."
- Table 3 (L420–421): "Table 3: Ablation of scene-evolving (SE) driving guidance under different training data scales on the PhysicalAI-Autonomous-Vehicles benchmark.✗: fixed global prompt as text conditioning."
- Table 4 (L429–431): "Table 4: Ablation of video backbone initialization and joint video supervision. All models are trained on 100k clips for 50k iterations."
- Table 5 (L436–438): "Table 5: Ablation of KV memory strategies. ADE/FDE are measured on 20s clips, while KV memory and GFLOPs are profiled under a 300s clip."
- Table 6 (L918–919): "Table 6: Per-chunk inference cost and trajectory prediction accuracy on a single H20 GPU.∗ indicates action denoising steps reduced from 10 to 5."
