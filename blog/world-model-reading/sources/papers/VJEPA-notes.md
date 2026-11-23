# V-JEPA 论文精读笔记

> 数据源：`auto_docs/papers/txt/VJEPA.txt`（1705 行）。所有英文引用逐字复制自该文件，行号即 txt 行号。论文未写之处标注「原文未写」，笔者的解读标注【推断】。

## 1 原文信息

- 标题：*Revisiting Feature Prediction for Learning Visual Representations from Video*（行 1–2）
- 作者（行 3–4）："Adrien Bardes1,2,3, Quentin Garrido1,4, Jean Ponce3,5,6, Xinlei Chen1, Michael Rabbat1, Yann LeCun1,5,6, Mahmoud Assran1,†, Nicolas Ballas1,†"，†为共同通讯："†Joint last author"（行 7）
- 机构（行 5–6）："1FAIR at Meta, 2Inria, 3École normale supérieure, CNRS, PSL Research University, 4Univ. Gustave Eiffel, CNRS, LIGM, 5Courant Institute, New York University, 6Center for Data Science, New York University"
- ID：arXiv:2404.08471v1 [cs.CV] 15 Feb 2024（行 92）；"Date: April 15, 2024"（行 16）
- 发表状态：原文未写（txt 中无任何会议接收信息，仅 arXiv 预印本）
- 其他（行 17–19）："Correspondence: {abardes, massran, ballasn}@meta.com"；"Code: https://github.com/facebookresearch/jepa"；"Blogpost: Click here"

## 2 Abstract 全文逐字（行 8–15）

> "This paper explores feature prediction as a stand-alone objective for unsupervised learning from video and introduces V-JEPA, a collection of vision models trained solely using a feature prediction objective, without the use of pretrained image encoders, text, negative examples, reconstruction, or other sources of supervision. The models are trained on 2 million videos collected from public datasets and are evaluated on downstream image and video tasks. Our results show that learning by predicting video features leads to versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters; e.g., using a frozen backbone. Our largest model, a ViT-H/16 trained only on videos, obtains 81.9% on Kinetics-400, 72.2% on Something-Something-v2, and77.9% on ImageNet1K."（"and77.9%"无空格为原文如此）

## 3 问题与动机：为什么特征预测优于像素重建

核心问题（行 87–90）："How effective is feature prediction as a stand-alone objective for unsupervised learning from video with modern tools?"

理论源头是预测特征原则（行 29–32）："One related hypothesis is based on the predictive feature principle (Rao and Ballard, 1999), which posits that representations of temporally adjacent sensory stimuli should be predictive of each other."

特征预测优于像素重建的原文论证（行 201–207）："Approaches that predict in pixel space must dedicate significant model capacity and compute to capture all the low-level detail in the visual input. By contrast, approaches that predict in latent space have the flexibility to eliminate irrelevant or unpredictable pixel-level details from the target representation (Vondrick et al., 2016)."

已有证据（行 208–214）："Predicting in representation space has been shown to lead to versatile representations that perform well across many downstream tasks through linear probing or low-shot adaptation (Assran et al., 2023; Oquab et al., 2023; Assran et al., 2022), while demonstrating an efficiency gain during pretraining compared to pixel level reconstruction (Assran et al., 2023; Baevski et al., 2022b,a)."

扩展到视频模态的定位（行 215–219）："The works of Baevski et al. (2022a,b) additionally show that predicting in representation space results in competitive end-to-end fine-tuning performance in the image, audio and text domains. In this work, we extend these findings to the video modality."

**动机链条解读**：本文的论证结构是——① 认知科学假设：相邻感觉刺激的表征应互相可预测（行 29–32 的 predictive feature principle）；② 工程代价论证：像素预测必须为"低层细节"付出容量与算力，潜空间预测则可以把无关或不可预测的像素细节排除出目标表征（行 202–207）；③ 已有证据：表征空间预测在图像/音频/文本上都得到通用表征、预训练更高效、微调也具竞争力（行 208–214, 215–219）；④ 本文的增量贡献：把这一结论验证到视频模态，且不用预训练图像编码器、文本、负样本、重建或任何其他监督（行 9–10）。论文反复强调的对比对象是 masked autoencoding 一系的像素重建（VideoMAE/VideoMAEv2/Hiera/OmniMAE），并给出"训练样本数少一个量级"的效率证据（见第 5 节 Table 5）。

## 4 方法逐机制

**总览**：V-JEPA 是双编码器（x-encoder / y-encoder，即 context/target）+ 预测器的 JEPA 结构（Figure 2、Figure 3）。x 是保留（未掩码）的可见区域，y 是被掩码掉的目标区域；网络从 x 的表征预测 y 的表征，预测器额外吃进 z（这里 z ← ∆y，即 y 的时空位置信息，行 254–256）。

### 4.1 JEPA 思想与架构

（行 237–246）："The main idea behind a JEPA is to learn by predicting the representation of an input y from the representation of another input x. The basic architecture is made up of an encoder, Eθ(·), which computes the representation of the inputs, and a predictor, Pϕ(·), which predicts the representation of y from the representation of x, conditioned on a variable z indicating the transformation (or corruption) between x and y. Conditioning on z enables the generation of distinct predictions for various transformations of x."

视频分词（行 363–370）："we split the video clip into a 3D grid of L spatio-temporal patches, where a patch consists of a 16 × 16 pixel block spanning 2 consecutive frames; we refer to these spatio-temporal patches as tokens."

掩码施加位置（行 371–375）："We apply masking at the input of the x-encoder, and at the output of the y-encoder to construct contextualized targets (Baevski et al., 2022b)."

**解读**：编码器是标准 ViT；输入是 16 帧、帧间跳 4 帧（≈3 秒）的短片；patch 是 16×16 像素跨 2 帧的时空块。掩码的施加点有两个关键设计：x-encoder 输入处直接"丢弃 token"（不做补零），y-encoder 则看完整视频、在输出端取被掩码位置的表征作为"上下文化目标"——目标由此带上了未掩码区域的上下文信息，这是 data2vec 式的 contextualized target 设计。

### 4.2 训练目标与防坍缩

（行 257–266）："Naively implementing the objective using the regression minimizeθ,ϕ ∥Pϕ(Eθ(x), ∆y) − Eθ(y)∥1, would admit a trivial solution, where the encoder outputs a constant representation, regardless of its input. In practice, we use the following modified objective to prevent representation collapse, minimizeθ,ϕ ∥Pϕ(Eθ(x), ∆y) − sg(Eθ(y))∥1, (1) where sg(·) denotes a stop-gradient operation, which does not backpropagate through its argument, and Eθ(·) is an exponential moving average of the network Eθ(·)."

与 I-JEPA 的差别（行 272–275）："the objective in equation(1) is similar to the loss of Assran et al. (2023) used for image pretraining, but we modify it to use an ℓ1 regression, which we found to be more stable."

理论动机（行 286–300）："P ⋆(Eθ(x)) = argminP ∥P (Eθ(x)) − Y ∥1 = median(Y |Eθ(x))."；"∇θE∥P ⋆(Eθ(x)) − Y ∥1 = ∇θMAD(Y |Eθ(x)), where MAD(· |Eθ(x)) is the median absolute deviation of a random variable conditioned on Eθ(x). Thus, in the case where the predictor is optimal, the encoder must learn to capture as much information about the video as possible to minimize the deviation of the target. The hypothesis is that incorporating an exponential moving average to compute the representation of y ensures that the predictor evolves faster than the encoder and remains close to optimal, thereby preventing collapse."

**解读**：naive 回归目标 ∥Pϕ(Eθ(x), ∆y) − Eθ(y)∥1 存在平凡解（编码器输出常数），所以目标侧加 stop-gradient 且 y 的编码用 x-encoder 的 EMA（指数滑动平均）网络计算——这是 BYOL 防坍缩策略在 ℓ1 回归下的变体；与 I-JEPA 的唯一区别是把平方误差换成更稳定的 ℓ1。理论上（对 BYOL 分析的简单改写）：最优预测器是条件中位数 median(Y|Eθ(x))，此时编码器梯度等于最小化条件 MAD，因此编码器被驱动去捕获尽可能多的视频信息；EMA 使预测器比编码器进化更快、始终接近最优，从而防坍缩。

### 4.3 掩码策略：multi-block

（行 341–349）："To sample y from a video, we sample several (possibly overlapping) spatially continuous blocks with various aspect ratios and repeat the spatial blocks across the entire temporal dimension of the video; x is taken to be the complement. Masking a large continuous block that covers the full temporal dimension limits information leakage due to the spatial and temporal redundancy of videos, and results in a harder prediction task (Tong et al., 2022)."

（行 350–360）："We leverage two types of masks: short-range masks, where we take the union of 8 randomly sampled target blocks covering 15% of each frame, and long-range masks, where we take the union of 2 randomly sampled target blocks covering 70% of each frame. In both cases, the aspect ratio for all sampled blocks is randomly chosen in the range (0.75, 1.5). Given that both short-range and long-range masks are produced by sampling many blocks and taking their union, the result is an average masking ratio of ∼ 90%. We refer to our masking strategy as multi-block"

**解读**：掩码逻辑是"抽块取并集"：短程掩码 8 块 ×15%，长程掩码 2 块 ×70%，块宽高比在 (0.75, 1.5) 随机，最终平均掩掉约 90% 的 token。空间块沿整个时间维度重复延伸，原因（行 346–349）是视频在时空上高度冗余，只在个别帧打洞会信息泄漏、预测任务太简单。长短程两个掩码构成两种难度互补的任务，并由一次 y-encoder 前向共享目标（见 4.4）。

### 4.4 预测器与多掩码预测

（行 375–385）："The encoder is parameterized using standard ViT networks, while the predictor is a narrow transformer implemented using 12 blocks with an embedding dimension of 384. Taking inspiration from masked autoencoders (He et al., 2021), our predictor takes as input the sequence of embeddings produced by the x-encoder as well as a sequence of learnable mask tokens with positional embeddings indicating the spatio-temporal positions of the y tokens. The output of the predictor is an embedding vector for each mask token"

多掩码（行 1406–1409）："To increase the efficiency of V-JEPA, we use a multi-masking strategy (Caron et al., 2020; Baevski et al., 2022a), which enables us to amortize the cost of the target computation. As mentioned in Section 3, for a given video clip, we sample 2 different masks, short-range and long-range. While we need to forward propagate the x-encoder and predictor separately for each mask, we only need to compute the y-representation once."

### 4.5 预训练数据与设置

（行 388–396）："We combine the videos from HowTo100M (HT) (Miech et al., 2019), Kinetics-400/600/700 (K710) (Kay et al., 2017), and Something-Something-v2 (SSv2) (Goyal et al., 2017), and remove any overlap with the validation sets of Kinetics-400/600/700 and Something-Something-v2, resulting in approximately 2 million videos."（该数据集名为 VideoMix2M，行 390）

（行 396–405）："We train a ViT-L/16, a ViT-H/16, and a ViT-H/16384 transformer model on VideoMix2M. We use a batch size of 3072 for the ViT-L/16 and ViT-H/16 models, and a batch size of 2400 for the ViT-H/16384 model. Each model takes as input a video clip of 16 frames sampled with a frame-skip of 4, corresponding to roughly 3 second clips on average. The ViT-L/16 and ViT-H/16 process the video at a spatial resolution of 224, while the ViT-H/16384 uses an input resolution of 384"

**解读**：VideoMix2M 由三个公开数据集拼合（HowTo100M + Kinetics-400/600/700 + SSv2），并去除与各验证集的重叠，得到约 200 万视频。共训练三个规模的模型（ViT-L/16、ViT-H/16、ViT-H/16@384），训练量统一为 90,000 迭代。优化细节（行 1442–1449）值得注意：y-encoder 是 x-encoder 的 EMA（动量 0.998 线性升至 1.0），且所有调度（学习率/权重衰减/EMA）按 112,500 迭代计算、只训 90,000 迭代——"We found the last 25% of the default scheduler period to update hyper-parameters too aggressively, and simply truncating the schedulers improved performance."（行 1448–1449）。预训练全程不使用 [cls] token（行 1420）。

训练 90,000 迭代、AdamW、EMA 动量 0.998→1.0 等见 Table 8（行 1370–1405）。

## 5 实验与主结果（数值逐字抄录，top-1 %）

### 5.1 冻结评估：vs 像素预测基线（Table 5，行 593–609）

| Method | Arch. | #Samples Seen | Iter. | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 | K400-ft | SSv2-ft |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OmniMAE | ViT-L/16 | 2400M | 1170K | 65.6 | 60.6 | 14.4 | 75.1 | 59.8 | 66.1 | 84.0 | 74.2 |
| VideoMAE | ViT-L/16 | 410M | 400K | 77.8 | 65.5 | 21.6 | 71.1 | 59.3 | 64.6 | 85.4 | 74.3 |
| Hiera | Hiera-L | 770M | 1500K | 75.5 | 64.2 | 15.8 | 68.9 | 58.5 | 56.9 | 87.3 | 75.1 |
| V-JEPA | ViT-L/16 | 270M | 90K | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 | 85.6 | 75.1 |

表注关键句（行 597–601）："In frozen evaluation, V-JEPA outperforms the baselines on all downstream tasks, except ImageNet, where the model achieves74.8% compared to 75.1% of an OmniMAE model trained directly on ImageNet."；"The V-JEPA results are achieved while processing significantly fewer examples during pretraining."

**读表**：同为 ViT-L/16（或参数量相近的 Hiera-L），V-JEPA 只用 270M 样本 / 90K 迭代，冻结评估在 K400（80.8）、SSv2（69.5）、AVA（25.6）、Places205（60.3）、iNat21（67.8）五项全部最高，唯 IN1K 以 74.8 输给在 ImageNet 上直接训练的 OmniMAE 的 75.1；微调上 K400 85.6 优于全部 ViT-L 基线，SSv2 75.1 与 Hiera-L 持平。

### 5.2 与 SOTA 对比（Table 6，行 610–636；冻结+attentive probe）

| Method | Arch. | Params. | Data | K400 | SSv2 | AVA | IN1K | Places205 | iNat21 |
|---|---|---|---|---|---|---|---|---|---|
| I-JEPA | ViT-H/16 512 | 630M | IN22K | 79.7 | 50.0 | 19.8 | 84.4 | 66.5 | 85.7 |
| OpenCLIP | ViT-G/14 | 1800M | LAION | 81.8 | 34.8 | 23.2 | 85.3 | 70.2 | 83.6 |
| DINOv2 | ViT-g/14 | 1100M | LVD-142M | 83.4 | 50.6 | 24.3 | 86.2 | 68.4 | 88.8 |
| MVD | ViT-L/16 | 200M | IN1K+K400 | 79.4 | 66.5 | 19.7 | 73.3 | 59.4 | 65.7 |
| OmniMAE | ViT-H/16 | 630M | IN1K+SSv2 | 71.4 | 65.4 | 16.0 | 76.3 | 60.6 | 72.4 |
| VideoMAE | ViT-H/16 | 630M | K400 | 79.8 | 66.2 | 20.7 | 72.3 | 59.1 | 65.5 |
| VideoMAEv2 | ViT-g/14 | 1100M | Un.Hybrid | 71.2 | 61.2 | 12.9 | 71.4 | 60.6 | 68.3 |
| Hiera | Hiera-H | 670M | K400 | 77.0 | 64.7 | 17.5 | 71.4 | 59.5 | 61.7 |
| **V-JEPA** | ViT-L/16 | 200M | VideoMix2M | 80.8 | 69.5 | 25.6 | 74.8 | 60.3 | 67.8 |
| **V-JEPA** | ViT-H/16 | 630M | VideoMix2M | 82.0 | 71.4 | 25.8 | 75.9 | 61.7 | 67.9 |
| **V-JEPA** | ViT-H/16384 | 630M | VideoMix2M | 81.9 | 72.2 | 25.0 | 77.4 | 62.8 | 72.6 |

正文定量结论（行 754–762）："Our H/16 model outperforms the largest publicly available VideoMAE, VideoMAEv2, OmniMAE, MVD, and Hiera models by at least +5 points in motion understanding (Something-Something-v2), +2 points in action recognition (Kinetics-400), +5 points on action detection (AVA), +1 point on object recognition (ImageNet-1K), +2 points in scene recognition (Places205), and +0.2 points on fine-grained recognition (iNaturalist)."

vs 图像模型（行 767–771）："On tasks that require a fine-grained understanding of motion (Something-Something-v2), the V-JEPA models provide a major improvement (over +21 points) compared to large-scale image baselines, such as DINOv2, OpenCLIP, and I-JEPA."

ImageNet（行 785–788, 805–806）："V-JEPA achieves a score of 77.4% on ImageNet using a one-layer attentive probe, which can be further improved to 77.9% using a two-layer attentive probe."

训练速度（行 763–765）："V-JEPA achieves this performance with a roughly 2× speedup compared to the large pixel prediction models."

**读表**：与视频模型比，V-JEPA H/16 在六个下游任务全面领先（正文行 754–762 给出 +5/+2/+5/+1/+2/+0.2 的最低优势）；与图像模型比，SSv2 上领先 21 点以上，但 K400 上 DINOv2 以 83.4 仍居首（正文解释：Kinetics 很多类别仅凭外观线索即可判别，行 781–784）。表 6 注意各模型参数量差异极大（OpenCLIP 1800M、DINOv2 1100M、VideoMAEv2 1100M vs V-JEPA 630M），且 I-JEPA512 与 V-JEPA384 分别在 512/384 分辨率评估。

### 5.3 标签效率（Table 7，行 789–804；mean ± std）

| Method | K400 5% | 10% | 50% | SSv2 5% | 10% | 50% |
|---|---|---|---|---|---|---|
| MVD ViT-L/16 | 62.6 ± 0.2 | 68.3 ± 0.2 | 77.2 ± 0.3 | 42.9 ± 0.8 | 49.5 ± 0.6 | 61.0 ± 0.2 |
| VideoMAE ViT-H/16 | 62.3 ± 0.3 | 68.5 ± 0.2 | 78.2 ± 0.1 | 41.4 ± 0.8 | 48.1 ± 0.2 | 60.5 ± 0.4 |
| VideoMAEv2 ViT-g/14 | 37.0 ± 0.3 | 48.8 ± 0.4 | 67.8 ± 0.1 | 28.0 ± 1.0 | 37.3 ± 0.3 | 54.0 ± 0.3 |
| V-JEPA ViT-H/16 | 67.0 ± 0.2 | 72.1 ± 0.1 | 80.2 ± 0.2 | 51.9 ± 0.3 | 57.5 ± 0.4 | 67.3 ± 0.2 |
| V-JEPA ViT-H/16384 | 68.2 ± 0.2 | 72.8 ± 0.2 | 80.6 ± 0.2 | 54.0 ± 0.2 | 59.3 ± 0.5 | 67.9 ± 0.2 |

（行 833–839）："the performance of the largest V-JEPA model on K400 drops by 12% to 68.2% top-1 when we reduce the number of labeled examples by a factor of 10× (from roughly 287 examples per class to 29 examples per class). By contrast, VideoMAEv2 drops by 30% to 37.0% top-1, VideoMAE drops by 15.9% to 62.3% top-1, and MVD drops by 14.6% to 62.6% top-1."（SSv2：V-JEPA 最大模型掉 13.9% 至 54.0%，行 840–847）

**读表**：低样本设定取 5%/10%/50% 训练集各 3 个随机切分、共 9 组实验，报均值±标准差。标签越少，V-JEPA 相对基线的优势越大——这是"特征预测比像素重建更标签高效"的直接证据。

### 5.4 全微调（Table 15，行 1625–1635）

| Method | Arch. | Data | #Samples Seen | K400 (16×5×3) | SSv2 (16×2×3) |
|---|---|---|---|---|---|
| VideoMAEv1 | ViT-L/16 | K400\|SSv2 | 380M\|410M | 85.4 | 74.3 |
| VideoMAEv1 | ViT-H/16 | K400\|SSv2 | 380M\|410M | 86.6 | 74.8 |
| VideoMAEv2 | ViT-H/16 | Un.Hybrid | 1600M | 86.9 | 76.8 |
| MVD | ViT-L/16 | K400+IN1K | 2400M | 86.4 | 76.7 |
| MVD | ViT-H/16 | K400+IN1K | 2400M | 87.2 | 77.3 |
| V-JEPA | ViT-L/16 | VideoMix2M | 270M | 85.6 | 75.1 |
| V-JEPA | ViT-H/16 | VideoMix2M | 270M | 86.6 | 77.0 |

（行 1595–1601）："With a ViTiH/16 architecture, V-JEPA outperforms by 1.2% VideoMAE and +0.3% VideoMAEv2 on the SSv2 dataset, while obtaining comparable performance on K400."（"ViTiH/16"为原文如此）

样本效率（Table 16，行 1654–1662）：OpenCLIP ViT-G/14 LAION-2B 39000M；DINOv2 ViT-g/14 LVD 142M 1900M；VideoMAEv2 ViT-g/14 UnlabeledHybrid 1600M；V-JEPA ViT-H/16 384 VideoMix2M 210M。正文（行 1605–1607）："Notably, our results with V-JEPA are obtained while processing an order of magnitude fewer samples than previous methods, and notably two orders of magnitude fewer samples than OpenCLIP."

### 5.5 结论（原文要点，行 895–912）

（行 895–905）："In this work, we explored the effectiveness of feature prediction as a stand-alone objective for unsupervised learning from video and introduced V-JEPA, a collection of vision models trained solely using a self-supervised feature prediction objective. The V-JEPA models demonstrate the ability to solve various downstream image and video tasks without adaption of the model parameters, and outperform previous video representation learning approaches in frozen evaluation on action recognition, spatio-temporal action detection, and image classification tasks."

（行 905–912）："we show that pretraining V-JEPA on videos is particularly effective for solving downstream tasks requiring fine-grained motion understanding, while large-scale image models trained on internet scale datasets fall short on such tasks. Finally, we empirically observed that V-JEPA models are label-efficient learners, and exhibit good performance on downstream tasks, even when only few labeled examples are available."

## 6 消融重点

第 4 节自问"what to predict from what?"（行 461–462），依次隔离四个设计变量：预测目标（特征 vs 像素）、预训练数据分布、下游特征池化、掩码策略。以下抄录关键数值。

**Table 1 特征 vs 像素目标**（行 407–416，ViT-L/16，VideoMix2M 90K iters）：Pixels：K400 68.6 / SSv2 66.0 / IN1K 73.3 / K400-ft 85.4；Features：73.7 / 66.2 / 74.8 / 85.6。表注（行 410–411）："Predicting in feature space provide a consistent improvement over pixel space prediction."

**Table 2 数据分布**（行 417–428，90K iters）：ViT-L/16：K710 700K → 75.8/63.2/73.7（Avg 70.9）；K710+SSv2 900K → 72.9/67.4/72.8（71.0）；K710+HT 1900K → 74.5/64.2/74.8（71.1）；VideoMix2M 2000K → 73.7/66.2/74.8（71.5）。ViT-H/16：K710+SSv2 900K → 75.7/66.8/73.7（72.0）；VideoMix2M 2000K → 74.0/68.5/75.9（72.8）。（行 511–513）："average performance across tasks monotonically increases as we increase the size of the pretraining dataset"

**Table 3 池化**（行 487–500）：V-JEPA ViT-L/16：K400 Avg 56.7 → Att 73.7；SSv2 Avg 50.1 → Att 66.2。表注（行 494–495）："Using adaptive pooling with a cross-attention layer leads to improvements of +17.3 points on K400 and +16.1 points on SSv2."（正文行 562–564 写 "+17 points on K400 and +16.1 points on SSv2"）

**Table 4 掩码策略**（行 540–559，ViT-L/16，K710+SSv2）：random-tube[0.9] 51.5/46.4/55.6；causal multi-block[6] 61.3/49.8/66.9；causal multi-block[12] 71.9/63.6/72.2；multi-block 72.9/67.4/72.8。正文（行 582–591）："the best results are obtained by sampling x using a multi-block strategy, wherein the network is forced to make predictions after removing large continuous blocks in the video."

**Figure 8 掩码消融**（26 组实验，ViT-B/16，K400）："We found that low spatial or temporal coverage results in a trivial prediction task, which degrades downstream performance. Based on those results, we sample masks that remove roughly 90% of the frame and extend along the entire temporal dimension of the clip by default."（行 1641–1644）；"We find that sampling several blocks to perform better than sampling a single large block."（行 1647）；"We find that sampling two masks for each clip, with different spatial block sizes for each, to be more effective than sampling just a single mask. We hypothesize that this masking strategy induces complementary tasks."（行 1649–1651）

**Table 12 线性 vs attentive probe**（行 1568–1574）：VideoMAE ViT-L/16 K400 52.5→77.8、SSv2 41.3→61.2；V-JEPA ViT-L/16 56.7→80.8、50.1→69.5。

**Table 14 时间覆盖**（行 1616–1624）：VideoMAE 1 Clip 69.4 → 8 Clips 77.8；V-JEPA 73.7 → 80.9。

**小结**：四个消融共同指向——① 在表征空间而非像素空间回归（+约 4–5 点冻结 K400）；② 数据越大平均性能越好（L/16：Avg 70.9→71.5；H/16：72.0→72.8）；③ 冻结评估必须用可学习的 cross-attention 池化（K400 +17.3、SSv2 +16.1），因为 ℓ1 预测目标无归一化、没有理由保证特征线性可分（行 529–531："Since the prediction objective in equation(1) is unnormalized, there is no a priori reason for the encoder to yield a linearly separable subspace (Chen et al., 2020)."）；④ multi-block 掩码（72.9/67.4/72.8）全面优于 random-tube 与因果式掩码，且要保持约 90% 时空覆盖率。

## 7 局限 / 未来工作（原文逐字）

论文无独立 Limitations 章节（原文未写）。正文中的相关陈述：

（行 807–812）："we hypothesize that the datasets used to train V-JEPA and other video models are too constrained and lack the visual diversity of the internet-scale pretraining data used by the images models; as such, there is value in focusing future work on building diverse publicly available video datasets."

（行 1607–1608）："We believe that further investment towards improving the video pretraining data distribution could lead to substantial gains in downstream image and video tasks."

## 8 叙事关联：JEPA 线对「世界模型该输出什么本体」的论证【推断】

1. **输出本体=潜表征而非像素**：【推断】V-JEPA 是 JEPA 家族从音频（data2vec）、图像（I-JEPA）到视频的延展，用同一目标函数在不同模态验证「预测表征」这一本体论选择。支持原句（行 163–172）："The idea of learning a representation by predicting missing information in feature space is also core to the joint-embedding predictive architecture (JEPA) (LeCun, 2022), which combines a siamese encoder with a predictor network. JEPAs have been successfully instantiated in several modalities, such as with audio data (Baevski et al., 2022b) and image data (Zhou et al., 2021; Oquab et al., 2023; Assran et al., 2023). In this work, we extend this paradigm to video data"
2. **像素细节在损失中被"合法抛弃"**：【推断】特征预测使世界模型可以把与任务无关/不可预测的像素细节排除在目标表征之外，这是「世界模型输出应是抽象潜状态」的直接论证。支持原句（行 202–207）："Approaches that predict in pixel space must dedicate significant model capacity and compute to capture all the low-level detail in the visual input. By contrast, approaches that predict in latent space have the flexibility to eliminate irrelevant or unpredictable pixel-level details from the target representation"
3. **潜表征必须"信息最大化"而非"随机丢弃"**：【推断】ℓ1 回归 + median 理论动机说明该目标让编码器捕获尽可能多的视频信息，即潜表征是压缩但信息保留的抽象本体。支持原句（行 294–296）："in the case where the predictor is optimal, the encoder must learn to capture as much information about the video as possible to minimize the deviation of the target"
4. **模型明言自己不是生成模型**：【推断】解码到像素仅用于可视化验证"表征里装了什么"，本体仍停在潜空间，反向支持 JEPA「世界模型无需生成像素」的路线。支持原句（行 888–890）："V-JEPA is not a generative model and the decoder does not have access to the context (first row), so we do not expect samples to exactly match the input. This experiment qualitatively illustrates what information is encoded and predicted by V-JEPA."
5. **视频提供静态图像没有的动态概念**：【推断】从视频学习潜表征能建模「运动」这类时间性本体，这是纯图像预训练（含大型 CLIP/DINOv2 系）难以覆盖的，为「世界模型本体应含时间维度」提供证据。支持原句（行 771–773）："Self-supervised pretraining from videos allows to model dynamic concepts that are not easily learned from static image datasets."

## 9 主要图注/表注抄录

- **Figure 1**（行 81–85）："V-JEPA models pretrained on video learn versatile visual representations. It performs well on motion-based tasks (Something-Something-v2) and appearance-based tasks (Kinetics 400) without adaptation of the model’s parameters, i.e., using the same frozen backbone for both tasks."
- **Figure 2**（行 228–232）："Joint-Embedding Predictive Architectures are trained to predict the representation of an input y from the representation of another input x. The additional variable z provides the predictor with information about the transformation that computes y from x."
- **Figure 3**（行 330–336）："V-JEPA. Training operates on a video clip of T frames with spatial resolution H × W, flattened into a sequence of L tokens. … The outputs of the predictor are then regressed to the prediction targets using an L1 loss. The prediction targets correspond to the output of the y-encoder."
- **Figure 4**（行 688–693）："SSv2 fine-tuning performance vs. Samples Seen. … V-JEPA outperforms all pixel-reconstruction methods using a ViT-L/16 and matches the Hiera-L performance while seeing significantly less samples during pretraining."
- **Figure 5**（行 744–753）："SSv2 frozen-evaluation performance vs. Pretraining Time. Wallclock times for all methods are measured on a single GPU with a batch size of 10 clips, using the official codebases for VideoMAE and VideoMAEv2, and linearly extrapolated assuming a global batch size of 2400 samples. … V-JEPA outperforms pixel-reconstruction methods while training significantly faster."
- **Figure 6**（行 893 及 882–892）："(a) Visualization Methodology. We train a conditional diffusion model to decode the V-JEPA feature-space predictions to interpretable pixels; the pretrained V-JEPA encoder and predictor networks are kept frozen in this process. The decoder is only fed the representations predicted for the missing regions of the video, and does not have access to the unmasked regions of the video."；"(b) Visualizations. … Some of the samples also demonstrate an understanding of object-permanence"（行 874–876）
- **Figure 7**（行 1346–1349）："To convert a video clip of size 16 × 224 × 224 × 3 into a 1D token sequence, we apply a 3D convolution comprising d filters of size 2 × 16 × 16 with a temporal stride of 2 and a spatial stride of 16, resulting in a tensor of shape 8 × 14 × 14 × d."
- **Figure 8**（行 1695–1699）："Masking Strategy Ablation. … We find it important to maintain a high spatial and temporal masking ratio during pretraining."
- **Figure 9**（行 1703–1704）："Each mask is constructed by sampling several (possibly overlapping) blocks and taking their union."
- **Table 5 表注**（行 593–601）、**Table 6 表注**（行 610–617）、**Table 7 表注**（行 789–795）：见第 5 节；**Table 1–4 表注**见第 6 节。
