# DTWM 精读笔记（Dexterous Tactile World Model）

> 数据源：`auto_docs/papers/txt/DTWM.txt`（1383 行，27 页，arXiv:2609.34286）
> 行号均指该 txt 文件行号。txt 转存存在空格粘连（如 L31 "cansynthesize"、L220 "or36frames"），下列引文按词边界还原空格，其余逐字；换行处连字符（如 L14-15 "resid-ual"）已按词合并。
> 标注约定：所有事实性陈述附论文英文原句+行号；论文没有的写「原文未写」；解读/引申标注【推断】。

---

## 1. 原文信息

| 项 | 内容 | 行号 |
|---|---|---|
| 标题 | "Dexterous Tactile World Model"（txt 中粘连为 "DEXTEROUSTACTILEWORLDMODEL"） | L2 |
| 作者 | "Ziyao Zeng1 Xiatao Sun1 Hao Wang2 Yueyang Pan3 Zhengxiang Yu4 Fengyu Yang1 Tianyu Liu1 Zhiwen Fan2 Daniel Rakita1" | L3-4 |
| 单位 | "1Yale University 2Texas A&M University 3University of California, Los Angeles 4University of Washington" | L5-6 |
| 编号 | "arXiv:2609.34286v1 [cs.CV] 28 Sep 2026" | L51 |
| 项目页 | "Project page: https://adonis-galaxy.github.io/dtwm-project-page/" | L49 |
| 版式 | Preprint，27 页（页码至 27） | L1、L1383 |

## 2. Abstract 全文逐字（L7-27）

> "World models for manipulation are typically trained from video, yet the events that determine how manipulation unfolds, such as making and releasing contact, are difficult to observe visually and are often easier to sense through touch. We present the Dexterous Tactile World Model (DTWM), a video world model for future-frame prediction of egocentric manipulation from both observed video and tactile signals from a glove worn on each hand. We condition a pretrained video diffusion transformer on each hand's tactile signal through a zero-initialized residual at the corresponding hand location in the video tokens, while a causal mask prevents predicted frames from accessing future information. Compared with a vision-only model matched in architecture, parameters, and training, DTWM reduces the underestimation of hand motion from 23% to 9%, while reducing the perceptual error in the hand region by 7.4% across three training runs per model. The benefit also increases over the prediction horizon, with the improvement in the later predicted chunks being about 4.1x larger than in the first. DTWM also outperforms other visual-tactile world models under the same setting, and training with touch improves future-frame prediction even when no touch is available at inference. Ablations show that the model benefits from both the magnitude and spatial location of force: replacing the tactile signal with binary contact states, either per hand or per location, increases prediction error. The observed course of the force indicates whether the interaction will persist or change."

## 3. 问题与动机（简）

- 世界模型从视频训练，但决定操作走向的事件（建立/解除接触）视觉上难观察、触觉易感：
  - L8-10（同 Abstract）："World models for manipulation are typically trained from video, yet the events that determine how manipulation unfolds, such as making and releasing contact, are difficult to observe visually and are often easier to sense through touch."
  - L39-43："All three uses require accurate predictions where the outcome is decided: at contact. In manipulation, whether a grasp holds, an object slips, or a finger reaches a surface determines what happens next. Yet world models are trained mainly from video, where contact is hard to observe. The hand often occludes the contact region, and touching an object can look nearly identical to pressing firmly against it."
- 视觉歧义导致生成失败：L44-45："The failure is visible in the generated video: predicted hands may lose their fingers, merge with the objects they hold, or move less than they should (Figures 1 and 3)."
- 触觉给出直接信号：L47-48："Touch provides a direct signal of these events. A pressure-sensing glove records where and how firmly each hand makes contact, including contacts that are hidden from the camera."
- 新语料使「双触觉世界模型」成为可能：L79-81："Recent datasets pair egocentric videos of everyday manipulation with such tactile measurements from both hands (Zhou et al., 2026a). We therefore condition a world model on dexterous touch: given the observed video frames and the tactile signals from both hands, it predicts how the manipulation unfolds next."
- 与既有视觉-触觉世界模型的差异：本文从过去观测（无动作/无未来姿态）预测**人类**自我中心操作，且触觉表示未与图像区域对齐：
  - L88-89："We instead predict egocentric human manipulation from past observations alone, so the model must also infer how the hands themselves will move."
  - L90-93："Finally, they represent touch as global tactile features, additional tokens or a separate tactile image, none aligned with the image region where contact occurs. We call this missing alignment the indexing mismatch: a glove reading is indexed by patches of skin, a video transformer by places in the image, and the touched skin can be anywhere in the frame."
- 核心解法：L94-98："DTWM resolves the mismatch by placing the observed tactile readings at the observed hand location, calculated using rendered hand skeleton (Figure 2). We embed each hand's reading, spread it over the video tokens around that location with a Gaussian footprint, and add it as a residual inside a pretrained video diffusion transformer. The residual is computed by a zero-initialized projection, so the pretrained model is unchanged at initialization and the pathway is learned rather than imposed."

## 4. 实现细节（重点）

### 4.1 数据：双手手套触觉 + 自我中心视频（采集 / 规模 / 频率）

- 语料：L782-785："All experiments use the EgoTouch corpus (Zhou et al., 2026a), released as the HuggingFace dataset zhouzhoujy/EgoTouch. It records everyday bimanual manipulation with a head-mounted wide-angle RGB camera and a pressure-sensing glove on each hand. Our copy contains 1933 episode directories from five scenes (Home, Office, Outdoor, Retail and Workbench)."
- 每 episode 提供的流：L785-789："Each episode provides frame-aligned egocentric RGB video at 480×640 and 30 fps, two wrist-mounted RGB streams, three sets of 3D hand-joint annotations (WiLoR, HaMeR and Rokoko), per-hand 21×21 pressure grids, raw 256-dimensional glove vectors, hand segmentation masks, wrist tracker poses and an episode-level contact label."
- 规模（时长/帧数/clip 数）：L793-796："The clip planner retains 1926 of the 1933 episodes. Three episodes are shorter than one clip and four contain video files that cannot be decoded. The retained episodes contain 2,183,052 frames, or 20.21 hours at 30 fps. Each egocentric stream is divided into contiguous, non-overlapping windows of 49 frames, yielding 43,604 clips of 1.63 seconds."
- 存储与训练分辨率：L796-798："Clips are stored at 432×320 together with a per-clip pressure tensor of shape (49,2,21,21) and the on-hand mask. The loader resizes all frames to a single 384×512 resolution, and inference applies the same resizing, so the evaluation geometry matches training."
- 频率细节：L799-800："The stored container rate of 15 fps is metadata only; the frames are consecutive 30 fps samples, so the 36 predicted frames span 1.2 seconds." 触觉采样与视频逐帧同步（每帧每手一读）：L194-195（Fig. 2 caption）："We sample one reading from both gloves (255 pressure taxels and 30 finger-flexion channels in total) per video frame."（即 30 Hz 级同步；原文未写触觉传感器的独立采样率。）
- 划分：L297-302："Its 1,926 episodes yield 43,604 clips of 49 frames, of which 38,600 are used for training. Our main evaluation uses 3,000 clips from recording sessions excluded from training. We divide these by task: 2,728 in-distribution (ID) clips have the same, a near-duplicate, or a related task in training; 272 out-of-distribution (OOD) clips cover three tasks with no close training counterpart (Table 3)."
- session holdout：L803-806："we additionally apply a session holdout: every episode that shares a recording session with an evaluation episode, meaning the same scene and task with consecutive timestamps at most 300 seconds apart, is removed from training. This removes 588 episodes and 4,704 clips."
- 训练集构成：L813-814："The training split consists of the remaining 38,600 clips from 1338 episodes and 174 task types, amounting to 17.51 hours of video, with a scene composition of 53.3% Home, 16.1% Retail, 13.4% Workbench, 12.8% Outdoor and 4.4% Office."
- OOD 三个任务：L821-823："The out-of-distribution group consists of three tasks, pushing a cart, dragging a chair and using a thermometer, with 272 of the 3,000 clips."
- 任务指令模板：L825-828："Every clip is paired with a task instruction generated from a fixed template, 'egocentric first-person view of two hands {task}, hand-object manipulation', in which underscores in the task name are replaced by spaces. Instructions contain between 9 and 15 words, with a mean of 11.2. The instruction is always provided, and no model drops it during training."
- 手部检测缺失率：L850-851："Hand detections are missing on 10.65% of frames for the left hand and on 12.14% for the right."

触觉通道细节：
- L225-227："The tactile signal is a pressure grid for each hand and frame, indexed somatotopically: each channel is a fixed patch of skin. The transformer is indexed retinotopically, by its token grid. We resolve the mismatch by giving each hand's signal the location of that hand in this grid."
- L228-231："For each pixel frame t and hand h, the sensor provides a hand-shaped 21×21 raster whose 217 on-hand cells partly share raw glove channels. We remove the duplicates. This leaves 138 channels for the left hand and 147 for the right."
- 网格定义与归一化：L859-862："The 21×21 grid is a hand-shaped visualization raster and not the physical sensor layout. Exactly 217 of its 441 cells are finite and lie on the hand; the other 224 are undefined... The released data are baseline-corrected and rescaled to [0,1] per episode (Section A)."
- 通道构成：L874-876："Each hand therefore has 15 flexion channels, three per digit, together with 123 pressure taxels on the left and 132 on the right."（123+132=255 pressure + 30 flexion，与 L194-195 一致。）
- 标准化：L246-247："We standardize each channel with fixed statistics and clip the result to [−8,8]; some channels measure finger flexion (Section B)."
- binary contact 定义：L251-253："Section 4.3 replaces this signal with coarser representations, down to a single contact bit per hand and frame, where a hand counts as being in contact when more than 2% of its pressure taxels exceed a normalized pressure of 0.05 (Section G)."

### 4.2 架构

- 总体：L206-210："We formulate manipulation prediction as block-causal chunked frame prediction with a pretrained flow-matching (Lipman et al., 2023; Liu et al., 2023) video diffusion transformer (Peebles & Xie, 2023; Wan et al., 2025) with 30 blocks and width d = 3072. A causal video autoencoder and a (1,2,2) patchifier map a clip of L = 49 frames at 384×512 to T = 13 latent frames on a 12×16 token grid, that is, 2496 video tokens (Section C)."
- chunk 与 block-causal：L211-215："The prediction unit is a chunk of C = 4 latent frames, or 16 pixel frames. Attention is block-causal (Yin et al., 2025): a token attends bidirectionally inside its own chunk and causally to earlier chunks. Given the observation, pixel frames 0 to K, that is latent frames 0 to m = K/4, the model denoises the remaining latents in one pass. Observed latents stay clean with a per-token timestep of zero, and predicted latents share one sampled timestep (Chen et al., 2024)."
- 预训练骨架（表 4）：L898-903："DTWM finetunes Wan2.2-Fun-5B-InP (Wan et al., 2025), a flow-matching video transformer with 5,001,016,512 parameters, hidden dimension 3072, 30 blocks, 24 attention heads of dimension 128 and a feed-forward width of 14,336. A training example is a clip of 49 frames at 384×512, which the causal video autoencoder compresses by a factor of four in time and sixteen in space into 48-channel latents; a non-overlapping (1,2,2) patchifier turns these into 2496 video tokens."
- 输入通道扩展（手部骨架）：L903-908："Apart from text cross-attention and the tactile residual, all conditioning is by channel concatenation. The pretrained inpainting input layer reads 100 channels, namely 48 noisy latents, a 4-channel observation mask and 48 reference latents. We extend the patch embedding to 148 channels by appending the encoded hand-skeleton stream, copy the pretrained weights into the first 100 input channels and initialize the 48 new channels to zero, so the extended model is functionally identical to the pretrained model at initialization."
- 条件只含观测区间内信息：L221-223："The conditioning contains only information available before the predicted interval (Figure 2). The hand skeleton is encoded by a widened, zero-initialized patch embedding (Section C), and no future hand pose or action is provided."
- 冻结与参数规模：L101-102："The backbone, the autoencoder and the text encoder stay frozen, and the tactile pathway adds only a few linear maps."
- 触觉嵌入：L248-250："The standardized states ht is embedded as e_ht = W_h s_ht + frame(t) + hand(h), using a linear map per hand and learned frame and hand embeddings. Embeddings are averaged within each temporal group of the autoencoder, so that t subsequently indexes latent frames."
- 手部定位（注入位置）：L254-258："The image location of each hand comes from the rendered skeleton the model already receives, produced from per-frame 3D joints (Potamias et al., 2025) under a fixed pinhole projection. The pixel centroid of each hand's color gives normalized coordinates (y_ht, x_ht) ∈ [0,1]² and a validity flag v_ht, which is zero when the hand is not drawn. Both are reduced to latent frames by a validity-weighted mean (Section D)."
- 高斯足迹（Eq. 1）：L259-264："At cell p = (i,j) of the H×W token grid, each hand places an isotropic Gaussian that is gated by validity and normalized to unit mass: g_ht(p) = exp(−((i−y_ht H)²+(j−x_ht W)²)/(2σ²)), ĝ_ht(p) = v_ht g_ht(p)/(Σ_p′ v_ht g_ht(p′) + ε), σ = 1.8. (1) The normalization makes the injected mass independent of σ, and an undetected hand contributes nothing."
- 残差注入（Eq. 2）：L265-271："Let u(ℓ)_t,p denote the video token at latent frame t and cell p after block ℓ. Each injection block updates it as u(ℓ)_t,p ← u(ℓ)_t,p + λ Σ_{h∈{L,R}} ĝ_ht(p)(W(ℓ) LN(ℓ)(e_ht) + b(ℓ)), where LN(ℓ) is a LayerNorm, W(ℓ) ∈ R^{d×d} and b(ℓ) are a linear map and bias, and λ is a scale fixed at one and stored as a buffer so that the pathway can be disabled exactly at inference (Section 4.3). We inject at blocks {0,6,12,18,24}."
- zero 初始化 + causal mask：L272-274："W(ℓ) and b(ℓ) are initialized to zero, so the residual is zero at initialization while the projection receives gradient from the first step (Section C). Setting v_ht = 0 for every t > m removes the residual from all predicted latents, in training and evaluation."
- 触觉只喂给观测帧：L99-100："The reading is supplied only for the observed frames, and a causal mask keeps every predicted frame from attending to later ones."

### 4.3 训练目标与配置（超参表）

- 训练设置概述：L351-357："We freeze the video autoencoder and text encoder, and adapt the pretrained transformer with rank-64 LoRA (Hu et al., 2022). The widened patch embedding, the tactile embedding, and the LayerNorm and projection that add the tactile residual at five blocks are trained at full rank. The flow-matching loss assigns weights of 0, 1, and 0.2 to observed latent frames, the first predicted chunk, and later frames, respectively. A spatial weight emphasizes the hands; future ground-truth hand locations are used only to weight the loss. All models train for one epoch over 38,600 clips using AdamW at 3×10⁻⁵ after a 100-step warmup (Section C)."
- 训练期 K 采样：L219-220："During training K is drawn uniformly from {4,8,...,44}. At deployment K = 12, the model predicts three chunks, or 36 frames, and all of them are evaluated (Section 4.1)."
- 可训练参数划分：L911-916："The autoencoder, the text encoder and the transformer weights are frozen. Adaptation uses two disjoint sets of parameters. The first is a LoRA of rank 64 and scale 64 (Hu et al., 2022) on all 306 linear modules of the transformer. The second consists of densely trained modules: the extended patch embedding, the tactile embedding and the five-block spatial injector. Modules of the tactile pathway are excluded from the LoRA, so the pathway is trained at full rank and does not share LoRA capacity."
- 优化器：L922-925："A single AdamW optimizer (Loshchilov & Hutter, 2019) updates both parameter sets with a learning rate of 3×10⁻⁵, β = (0.9,0.999), ε = 10⁻⁸ and weight decay 0.01. The learning rate is constant after a linear warmup of 100 steps, and gradients are clipped to a global norm of 1.0. Training uses bf16 precision with gradient checkpointing."
- 分布式与硬件：L926-935："Each run uses two GPUs with accelerate and DeepSpeed ZeRO-2 (Rajbhandari et al., 2020), with optimizer states offloaded to the CPU. Each GPU processes one clip per step and the parameters are updated after every step, so the effective batch size is two clips and one epoch comprises 19,300 updates... Each run uses two NVIDIA RTX 6000 Ada GPUs with 49 GB of memory, of which about 38 GB per GPU are used, and takes between 5 h 20 and 6 h 30."
- 损失空间加权：L988-991："Spatially, the weight is multiplied by 1 + 5.0H, where H ∈ [0,1] is the maximum over both hands of a unit-peak Gaussian of width 2.4 latent cells centered at the hand's latent location. This weighting uses ground-truth hand locations of the predicted frames only as supervision, and it is never provided as input."
- 文本与噪声调度：L908-910："Text conditioning uses a frozen umt5-xxl encoder with a 512-token context. The noise schedule is a flow-matching Euler discrete scheduler with 1000 training timesteps and shift 12.0, and the same scheduler is used at inference (Section D)."
- 对照 baseline（关键）：L317-320："The vision-only baseline uses the same architecture, parameter count, and training recipe as DTWM, but its tactile readings are zeroed before the embedding during both training and inference. Its tactile pathway remains trainable and retains the same injection sites and hand locations, so the comparison tests the contribution of the sensor readings."

**表 4 逐项（L941-974，全部超参）：**

| 组 | 项 | 值 |
|---|---|---|
| Backbone | Pretrained model | Wan2.2-Fun-5B-InP (flow-matching video transformer) |
| Backbone | Transformer parameters | 5,001,016,512 (after the channel extension) |
| Backbone | Hidden dimension, blocks, heads | 3072, 30, 24 (head dimension 128) |
| Backbone | Feed-forward dimension | 14,336 |
| Backbone | Patchifier | 3D convolution, kernel and stride (1,2,2) |
| Backbone | Video autoencoder | 48 latent channels, 4× temporal and 16× spatial compression |
| Backbone | Clip length and resolution | 49 frames, 384×512 |
| Backbone | Latent and token grid | 13×24×32 latents, 13×12×16 = 2496 tokens |
| Backbone | Patch-embedding channels | 148: 48 noise, 4 mask, 48 reference, 48 hand skeleton |
| Backbone | Text encoder | umt5-xxl, frozen, 512 tokens |
| Backbone | Noise schedule | flow-matching Euler discrete, 1000 training timesteps, shift 12.0 |
| Trained params | LoRA rank, scale, targets | 64, 64, all 306 linear modules of the transformer |
| Trained params | LoRA parameters | 164,261,888 |
| Trained params | Dense parameters | patch embedding 1,821,696; tactile embedding 4,349,725; injector 47,232,000 |
| Trained params | Total trainable | 217,665,309 (4.35%), identical for DTWM, the control and every tactile input |
| Optimization | Optimizer | AdamW, β=(0.9,0.999), ε=10⁻⁸, weight decay 0.01 |
| Optimization | Learning rate | 3×10⁻⁵, constant after a 100-step warmup |
| Optimization | Gradient clipping | global norm 1.0 |
| Optimization | Precision | bf16, gradient checkpointing |
| Optimization | Clips per GPU × GPUs | 1 × 2 |
| Optimization | Effective batch size | 2 clips per update |
| Optimization | Updates per epoch | 19,300 |
| Optimization | Budget | 1 epoch, 38,600 clips |
| Systems | Software | accelerate 1.2.1, DeepSpeed 0.19.2 (ZeRO-2, CPU optimizer offload), PyTorch 2.6.0 |
| Systems | Hardware per run | 2 NVIDIA RTX 6000 Ada (49 GB), about 38 GB used per GPU |
| Systems | Duration per run | 5 h 20 to 6 h 30 |

### 4.4 推理流程

- 预测任务：L306-309："Given RGB frames 0 to 12, the corresponding tactile readings and hand skeletons, and a task instruction, the model predicts frames 13 to 48 (1.2 seconds). No future hand pose, action, or tactile reading is provided. All 36 future frames are generated in one sampling pass with block-causal attention, in chunks covering frames 13 to 28, 29 to 44, and 45 to 48."
- K=12 切分：L1098-1100："A 49-frame clip is divided at K = 12: pixel frames 0 to 12, that is, thirteen frames spanning 0.43 seconds and four latent frames, form the observation, and the model denoises the remaining nine latent frames, corresponding to 36 pixel frames, in a single pass."
- 观测 latents 恒干净：L1100-1101："Observed latents are held at their clean encodings with timestep zero throughout sampling."
- chunk 因果结构：L1101-1104："Attention is block-causal over chunks of four latent frames, so the first predicted chunk (pixel frames 13 to 28) attends only to observed latents, the second (frames 29 to 44) attends in addition to the first, and the third (frames 45 to 48) attends to everything before it. The hand skeleton is black and the tactile residual is zero after frame K, so no future information is supplied."
- 采样参数：L1105-1106："Sampling uses 30 flow-matching steps with scheduler shift 12.0 and classifier-free guidance 6.0 against the negative prompt 'bad detailed'."
- 足迹宽度差异（训练 1.8 vs 推理 3.0）：L1106-1108："The footprint width at evaluation is the inference default σ = 3.0, whereas training uses σ = 1.8; evaluating one checkpoint at both widths changes LPIPS by +0.0005 on the held-out-episode split, and all models share the same evaluation width."
- 评分前重采样：L1109-1110："Generated frames at 384×512 are resized bilinearly to the stored 320×432 resolution before scoring. All 36 predicted frames are scored."
- 指标定义：LPIPS（AlexNet backbone）与 LPIPS hand：L1112-1117："LPIPS uses the AlexNet backbone with the calibrated linear head, inputs mapped to [−1,1], computed per frame and averaged over the scored frames. LPIPS hand applies the same metric to one crop per clip, defined as the union of rendered skeleton pixels over the scored window, enlarged by 25% of its height and width on each side and resized to 128×128. Because the enlarged crop usually contains the grasped object, it measures the manipulation region rather than the hands alone, and it is undefined for the few clips without a usable skeleton."
- 统计方法：L1120-1124："The unit of pairing is the clip... The vector of per-clip differences, with fewer entries where LPIPS hand is undefined, is resampled with replacement 20,000 times with a fixed generator. We report the mean difference and the 2.5 and 97.5 percentiles of the resampled means."

## 5. 伪代码

⚠ **依据原文重建（非原文伪代码）**：全文检索 "Algorithm / Pseudo / pseudocode" 零匹配，原文无 Algorithm 环境。以下依据正文逐字描述重建，每步标注行号；原文未写的步骤细节一律不补。

### 5.1 训练（单步，per gradient step）

```
输入: 49 帧 clip (384×512)、每帧每手 21×21 触觉栅格 s、任务指令 τ、手部骨架流
1.  采样 K ~ Uniform({4,8,...,44})                                   [L219]
2.  编码观测帧 0..K 为干净 latent（latent 帧 0..m, m=K/4），
    观测 token 时间步置 0；预测 latent 共享同一采样时间步           [L213-215]
3.  手部骨架经扩宽至 148 通道的 zero-initialized patch embedding 编码 [L221-222, L903-907]
4.  触觉: 通道标准化后 clip 到 [-8,8]                                [L246-247]
    e_ht = W_h s_ht + frame(t) + hand(h)（每手一个线性映射）         [L248-249]
    （触觉嵌入在 autoencoder 各时间组内平均: 49 帧 → 13 latent 帧）  [L249-250]
5.  由渲染骨架得手部质心 (y_ht, x_ht) 与有效性 v_ht，经
    validity-weighted mean 降到 latent 帧                            [L254-258]
6.  按 Eq.(1) 计算归一化高斯足迹 ĝ_ht(p)，σ=1.8（训练值）           [L259-264]
7.  causal 处理: 对 t > m 令 v_ht = 0（预测 latent 无触觉残差）      [L273-274]
8.  在块 ℓ ∈ {0,6,12,18,24} 注入:
    u ← u + λ Σ_h ĝ_ht(p) (W(ℓ) LN(ℓ)(e_ht) + b(ℓ)), λ=1
    （W,b 初始化为零）                                               [L266-272]
9.  文本经冻结 umt5-xxl 编码（512 tokens）                          [L908-909]
10. flow-matching Euler 调度（1000 训练时间步, shift 12.0）去噪      [L909-910]
11. loss: 观测 latent 权重 0 / 第一预测 chunk 权重 1 / 后续帧 0.2；
    空间上 × (1 + 5.0H)，H 为手部位置宽度 2.4 latent cell 的
    单位峰值高斯最大值（只用 GT 手部位置加权，不作输入）            [L354-355, L987-991]
12. 更新: AdamW lr=3e-5（100 步 warmup 后恒定）、grad clip 1.0、
    bf16；LoRA rank/scale=64 覆盖全部 306 个线性模块 +
    密集模块（patch embedding / 触觉嵌入 / 五块注入器）全秩训练     [L912-925]
```

### 5.2 推理（部署设置，K=12）

```
输入: 观测帧 0..12（=4 latent 帧，0.43 秒）+ 观测期触觉 + 手部骨架 + 任务指令
1.  K = 12；观测帧编码为干净 latent，时间步恒为 0                    [L1098-1101]
2.  触觉仅取观测帧（t ≤ K）的读数并嵌入                            [L99-100]
3.  K 之后的骨架渲染为黑、触觉残差为 0（不提供未来信息）            [L1104-1105]
4.  block-causal 注意力（chunk = 4 latent 帧）:
    chunk1(帧13-28) 只看观测; chunk2(帧29-44) 再看 chunk1;
    chunk3(帧45-48) 看全部先前 token                                 [L1101-1104]
5.  一次采样去噪剩余 9 个 latent: 30 个 flow-matching 步,
    shift 12.0, CFG 6.0, 负提示 "bad detailed"                      [L1105-1106]
6.  足迹宽度用推理默认 σ = 3.0（训练为 1.8）                        [L1106-1107]
7.  解码得 36 帧 (13..48)；384×512 双线性缩至 320×432 后评分        [L1109-1110]
```

## 6. 主结果数值表

**表 1（L339-348）：held-out set 预测质量（单次训练 run）**

| 模型 | ID (2728) LPIPS↓ | ID LPIPS hand↓ | OOD (272) LPIPS↓ | OOD LPIPS hand↓ | Hand(3000) IoU↑ | Hand flow dir.↑ |
|---|---|---|---|---|---|---|
| Vision-only baseline | 0.4243 | 0.3273 | 0.4882 | 0.3765 | 0.475 | 0.039 |
| TouchWorld | 0.4899 | 0.3979 | 0.5341 | 0.4204 | 0.376 | 0.031 |
| VT-WM | 0.4237 | 0.3324 | 0.4854 | 0.3797 | 0.462 | 0.032 |
| FeelWorld | 0.4488 | 0.3465 | 0.5071 | 0.3964 | 0.430 | 0.027 |
| **DTWM (ours)** | **0.4157** | **0.3227** | **0.4774** | **0.3668** | **0.476** | **0.041** |

- 相对 baseline 的降幅：L360-361："Relative to the vision-only baseline, full-frame LPIPS decreases by 2.0% on ID clips and 2.2% on OOD clips."
- 相对其他 VT 世界模型：L109-110："We also outperform three existing visual-tactile world models under the same setting, with a perceptual error lower by 1.8% to 14.7%."

**核心三个数字的出处：**
- 23%→9%：L471-472："The underestimation of optical-flow magnitude falls from 23% to 9%, and the silhouette area ratio moves closer to one."（表 10，L1216-1221：flow magnitude ratio 0.771→0.912；silhouette area ratio 0.920→1.032）
- 7.4%：L106-108："The vision-only model underestimates how much the hands move by 23%, and DTWM by 9%. The perceptual error in the hand region falls by 7.4% across three training runs per model, against 5.1% over the whole frame."（表 6，L1158-1164：held-out episodes LPIPS hand 0.3817±0.0439 → 0.3536±0.0119，差 −0.0281；LPIPS 0.4905→0.4654，差 −0.0252）
- 4.1×：L479-480："On held-out episodes, the mean per-frame improvement after the first chunk is about 4.1 times that within the first chunk."（表 8，L1186-1200：held-out episodes 首 chunk 帧13-20 差 −0.0083；帧37-48 差 −0.0399）

**表 8（L1186-1200）：增益随预测窗口增长（三 run 平均）**

| 评分窗口 | control | DTWM | 差值 |
|---|---|---|---|
| 帧 13-20（首 chunk 前 8 帧） | 0.2727 | 0.2644 | −0.0083 |
| 帧 13-28 | 0.3593 | 0.3501 | −0.0092 |
| 帧 13-36 | 0.4265 | 0.4087 | −0.0178 |
| 帧 13-48 | 0.4905 | 0.4654 | −0.0252 |
| 帧 21-28 | 0.4459 | 0.4358 | −0.0101 |
| 帧 29-36（第二 chunk） | 0.5610 | 0.5260 | −0.0350 |
| 帧 37-48 | 0.6185 | 0.5786 | −0.0399 |

**表 9（L1205-1211）：3,000-clip 集上全部 6 项手部指标领先既有模型（图 1 雷达图）**

| 模型 | LPIPS↓ | LPIPS hand↓ | IoU↑ | centroid(px)↓ | DTW(px)↓ | flow dir.↑ |
|---|---|---|---|---|---|---|
| TouchWorld | 0.494 | 0.400 | 0.376 | 28.0 | 14.2 | 0.031 |
| FeelWorld | 0.454 | 0.351 | 0.430 | 25.1 | 11.9 | 0.027 |
| VT-WM | 0.429 | 0.337 | 0.462 | 23.3 | 11.2 | 0.032 |
| **DTWM** | **0.421** | **0.327** | **0.476** | **21.9** | **10.4** | **0.041** |

**表 10（L1212-1221）：DTWM vs vision-only 的手部行为指标**

| 指标 | vision-only | DTWM | 差值 |
|---|---|---|---|
| silhouette IoU↑ | 0.475 | 0.476 | +0.001 |
| silhouette area ratio (ideal 1) | 0.920 | 1.032 | +0.111 |
| trajectory DTW (px)↓ | 10.41 | 10.37 | −0.04 |
| flow direction agreement↑ | 0.039 | 0.041 | +0.003 |
| flow magnitude ratio (ideal 1) | 0.771 | 0.912 | +0.141 |

**三 run 稳定性：** 表 6（L1158-1164）：held-out episodes LPIPS 0.4905±0.0328 vs 0.4654±0.0109；training episodes 0.5024±0.0417 vs 0.4777±0.0170。表 7（L1170-1179）："DTWM wins 7 of the 9 run pairings on each split, and its worst run is below the baseline's mean."

**PSNR/SSIM 反序的警示（原文自述）：** L1228-1232："Blurring the output of DTWM with a Gaussian of increasing width raises its PSNR from 14.94 to 15.30 and its SSIM from 0.535 to 0.571, while its LPIPS rises from 0.412 to 0.625; the blurred prediction scores higher on PSNR on 98% of the clips and on SSIM on 100%."（即 PSNR/SSIM 奖励「画得更少」，vision-only 在这些指标上反而更高；L1222-1224："On every split the vision-only baseline has the higher PSNR and SSIM while DTWM has the lower LPIPS..."）

## 7. 消融

**表 2（L370-380）：触觉输入消融（单次 run）**

| 触觉输入 | ID LPIPS | ID LPIPS hand | OOD LPIPS | OOD LPIPS hand |
|---|---|---|---|---|
| zeroed reading (vision-only baseline) | 0.4243 | 0.3273 | 0.4882 | 0.3765 |
| pathway disabled at inference | 0.4208 | 0.3297 | 0.4838 | 0.3754 |
| one contact bit per hand | 0.4307 | 0.3318 | 0.5014 | 0.3864 |
| per-taxel contact, no force | 0.4200 | 0.3246 | 0.4823 | 0.3745 |
| **DTWM (ours, full reading)** | **0.4157** | **0.3227** | **0.4774** | **0.3668** |

- 压力幅值的作用：L405-409："Thresholding each pressure taxel keeps the spatial pattern of contact and the finger-flexion channels but removes pressure magnitude. This increases full-frame and hand-crop LPIPS on both splits."
- binary 单比特更差（甚至差于 vision-only）：L409-414："Replacing the reading with one contact bit per hand, which also discards where the hand is pressed and how its fingers are flexed, further increases error and makes the model worse than the vision-only baseline. The model therefore uses more than whether a hand is in contact."
- 力的轨迹可预测释放：L415-420："In the training corpus, a hand whose force holds steady rarely releases its grasp within the next chunk, whereas a hand whose force rises or decays is 1.8 to 2.5 times as likely to release it (Figure 6 and Section G). The trend remains predictive after accounting for the current force level."（表 14，L1365-1382：steady 释放率 6.6%、decaying 16.3%、rising 11.6%（首 chunk）；力变化>20% 的 chunk 占 46.1%，接触状态变化仅 16.8%（单手）/5.0%（双手）；logistic AUC：current force+area 0.723 → +trend 0.735，第二 chunk 0.627→0.649）
- 无触觉推理仍受益（训练期触觉的迁移）：L425-430："Setting λ = 0 in Equation (2) disables the tactile residual of the trained full-reading model. This worsens all four perceptual scores relative to using the full reading, but full-frame LPIPS remains below the vision-only baseline on both ID and OOD clips. We attribute this to training on paired touch and video. Every training clip shows which visual appearance corresponds to which contact, where the hand touches and how firmly, and the model learns this association, which remains useful when the tactile pathway is off."
- 其他 VT 世界模型的对照实现（同信号、同骨架、同训练，仅注入机制不同）：L363-367："TouchWorld renders it as image panels, VT-WM appends it as tokens, and FeelWorld adds a gated global feature. DTWM instead adds each hand's reading at that hand's image location, and it has the lowest perceptual error of the four designs. With the same signal and training, how the tactile signal is fed to the model therefore affects prediction quality."

## 8. 局限（逐字，L496-500）

> "Limitations. To our knowledge, the corpus we use is the only open dataset that pairs egocentric video of everyday human manipulation with tactile signals from both hands, and at about twenty hours it is small for video models. Hand locations come from an off-the-shelf reconstruction that misses some frames. Finally, the model predicts human manipulation, so it cannot yet be placed directly inside a robot planner or policy. Extending it to a robot embodiment is a natural next step."

## 9. 叙事关联【推断】（≤4 条）

1. 【推断】DTWM 为「触觉补视觉世界模型盲区」叙事提供了**人类双手手套**数据的独立证据：不同于 VT-WM/FeelWorld/ViTacWorld（机器人指尖/夹爪、动作条件）与 TouchWorld（同语料但适配机器人手并供给策略子目标）；依据 L84-89（"Three are trained on robot trajectories... We instead predict egocentric human manipulation from past observations alone"）与 L168-171（"DTWM is instead a standalone world model of egocentric human manipulation..."）。
2. 【推断】"training with touch improves future-frame prediction even when no touch is available at inference"（L22-24、L425-430）可直接支撑「触觉=训练期对齐/监督信号、不必是推理期输入」的叙事；原文将其归因于配对训练让模型学到「视觉外观↔接触状态」的关联（L429-430）。
3. 【推断】zero-initialized residual（L97-98、L272-273）与原文对「gated projection 为何训练不动」的讨论（L978-986："The gradient with respect to W(ℓ) then contains the factor tanh(γ(ℓ)), which is zero at initialization, so the projection receives no gradient and retains its random values."）是「增量通路不破坏预训练先验」叙事的典型实例，与 zero-conv/LLaMA-Adapter 同源。
4. 【推断】增益沿预测时域增长（4.1×，L479-484："It is consistent with touch resolving uncertainty about the observed interaction: an error in that estimate barely affects the first predicted frames, but later chunks attend to earlier ones and inherit their errors."）支持「早期状态估计误差沿 block-causal 链累积、触觉在源头上消除不确定性」的世界模型叙事。
