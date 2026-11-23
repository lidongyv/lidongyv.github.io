# DINO-WM 论文精读笔记

> 唯一数据源：`auto_docs/papers/txt/DINOWM.txt`（共 1207 行，页码 1–21，arXiv:2411.04983v2）。本文所有行号指该 txt 文件行号；英文引用均为逐字复制自 txt；【推断】为笔者解读；「原文未写」表示论文中不存在该内容。

## 1. 原文信息

- **标题**（L1–2、L86）："DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning"
- **作者**（L3）："Gaoyue Zhou 1 Hengkai Pan 1 Yann LeCun1 2 Lerrel Pinto 1"
- **机构/通讯**（L40–41）："1Courant Institute, New York University 2Meta AI. Correspondence to: Gaoyue Zhou <gz2123@nyu.edu>."（上标 1=NYU，2=Meta AI）
- **ID/版本**（L85）："arXiv:2411.04983v2  [cs.RO]  1 Feb 2025"
- **发表状态**：原文未写（txt 中无任何会议/期刊录用信息，仅 arXiv v2，日期 2025-02-01）。项目页（L133–136）："Code and models for DINO-WM are open-sourced to ensure reproducibility and videos of planning are made available on our anonymous project website: https://dino-wm.github.io."（原文自称 anonymous，暗示投稿匿名版本）
- **资助**（L666–669）："This work was supported by grants from Honda, Hyundai, NSF award 2339096 and ONR awards N00014-21-1-2758 and N00014-22-1-2773. LP is supported by the Packard Fellowship."

## 2. Abstract 全文逐字（L5–32）

> "The ability to predict future outcomes given control actions is fundamental for physical reasoning. However, such predictive models, often called world models, remains challenging to learn and are typically developed for task-specific solutions with online policy learning. To unlock world models' true potential, we argue that they should 1) be trainable on offline, pre-collected trajectories, 2) support test-time behavior optimization, and 3) facilitate task-agnostic reasoning. To this end, we present DINO World Model (DINO-WM), a new method to model visual dynamics without reconstructing the visual world. DINO-WM leverages spatial patch features pre-trained with DINOv2, enabling it to learn from offline behavioral trajectories by predicting future patch features. This allows DINO-WM to achieve observational goals through action sequence optimization, facilitating task-agnostic planning by treating goal features as prediction targets. We demonstrate that DINO-WM achieves zero-shot behavioral solutions at test time on six environments without expert demonstrations, reward modeling, or pre-learned inverse models, outperforming prior state-of-the-art work across diverse task families such as arbitrarily configured mazes, push manipulation with varied object shapes, and multi-particle scenarios."

（注："remains challenging" 为原文语法，逐字保留。）

## 3. 问题与动机（逐字引用）

**前馈策略的泛化困境**（L52–57）："However, it is neither feasible nor efficient to learn solutions for all potential tasks and environments in advance. Instead of learning the solutions to all possible tasks during training, an alternate is to fit a dynamics model on training data and optimize task-specific behavior at runtime."

**在线 vs 离线世界模型的取舍**（L75–78）："the online world model is only accurate in the cover of the policy that was being optimized. Hence, while it can be used to train powerful task-specific policies, it requires re-training for every new task even in the same environment." 离线则需要强辅助信息（L87–92）："methods in this domain require strong auxiliary information which can take the form of expert demonstrations …, structured keypoints …, access to pretrained inverse models … or dense reward functions …, all of which reduce the generality of using offline world models."

**中心问题**（L93–95）："The central question to building better offline world models is if there is alternate auxiliary information that does not compromise its generality?"

**为什么在预训练视觉特征上建世界模型**（L98–105）："DINO-WM models the world dynamics on compact embeddings of the world, rather than the raw observations themselves. For the embedding, we use pretrained patch-features from the DINOv2 model, which provides both a spatial and object-centric representation prior. We conjecture that this pretrained representation enables robust and consistent world modeling, which relaxes the necessity for task-specific data coverage." 感知层的理由（L304–311）："we argue instead that it can be inefficient and often not possible to learn a good observation model from scratch when facing a new environment, as perception is a general task that benefits from large-scale internet data. Therefore, we use the pre-trained DINOv2 model as our world model's observation model, leveraging its strong spatial understanding for tasks like object detection, semantic segmentation, and depth estimation (Oquab et al., 2024)."

**测试时零额外信息**（L112–114）："we can simply use model predictive control with inference-time optimization to reach desired goals without any extra information during testing."

## 4. 方法逐机制

**问题形式化（POMDP）**（L226–230）："The POMDP is defined by the tuple (O, A, p), where O represents the observation space, and A denotes the action space. The dynamics of the environment is modeled by the transition distribution p(ot+1 | o≤t, a≤t), which predicts future observations based on past actions and observations."

**模型三组件**（L248–258，原文公式逐字）：

```
Observation model: zt ∼ encθ(zt | ot)
Transition model: zt+1 ∼ pθ(zt+1 | zt−H:t, at−H:t)
Decoder model: ˆot ∼ qθ(ot | zt)   (optional for visualization)
```

**DINOv2 特征空间 / patch 级潜状态**：编码器冻结（L311–312）："The observation model remains frozen during training and testing." 潜状态为 patch 嵌入（L312–315）："At each time step t, it encodes an image ot to patch embeddings zt ∈ RN×E, where N denotes the number of patches, and E denotes the embedding dimension." 具体形状（L1148–1149）："We use an encoder based on DINOv2, which extracts features with a shape of (14 × 14, 384) from input images resized to 196 × 196 pixels."

**预测头（transition model，ViT 改造）**（L318–334）："We adopt the ViT architecture (Dosovitskiy et al., 2021) for the transition model due to its suitability for processing patch features. We remove the tokenization layer, as it operates on patch embeddings, effectively transforming it into a decoder-only transformer." 因果注意力（L331–334）："we implement a causal attention mechanism in the ViT model, enabling the model to predict latents autoregressively at a frame level. Specifically, each patch vector zit for the latent state zt attends to {zit−H:t−1}Ni=1." 与 IRIS 的 token 级预测区别（L338–343）："We argue that predicting at a frame level and treating patch vectors of one observation as a whole better captures global structure and temporal dynamics, modeling dependencies across the entire observation rather than isolated tokens, leading to improved temporal generalization." 动作条件化（L348–351）："we concatenate the K-dimensional action vector, mapped from the original action representation using a multi-layer perceptron (MLP), to each patch vector zit for i = 1, . . . , N." 本体感受（L351–353）："When proprioceptive information is available, we incorporate it similarly by concatenating it to the observation latents, thereby integrating it into the latent states." 训练（L357–366）："We train the world model with teacher forcing. … we compute Lpred = ∥pθ (encθ(ot−H:t), ϕ(at−H:t)) − encθ (ot+1)∥2 (1) … Note that our world model training is entirely performed in latent space, without the need to reconstruct the original pixel images." 架构参数（L1149–1150）："The ViT backbone has a depth of 6, 16 attention heads, and an MLP dimension of 2048, amounting to approximately 19M parameters."

**可选解码器**（L281–289、L368–380）："Note that our decoder is entirely optional, as the training objectives for the decoder is independent for training the rest part of the world model. This eliminates the need to reconstruct images both during training and testing, which reduces computational costs…" 实现（L368–374）："we use a stack of transposed convolution layers to decode the patch representations back to image pixels, similar as in (Razavi et al., 2019). … Lrec = ∥qθ(zt) − ot∥2 , where zt = encθ(ot) (2)"

**规划器：MPC + CEM（非 MCTS）**：原文未写 MCTS/蒙特卡洛树搜索（全文无此词）。论文规划器为（L396–406）："We employ model predictive control (MPC), which facilitates planning by considering the outcomes of future actions. We utilize the cross-entropy method (CEM) to optimize the sequence of actions at each iteration. The planning cost is defined as the mean squared error (MSE) between the current latent state and the goal's latent state, given by C = ∥ˆzT − zg∥2 , where ˆzt = p(ˆzt−1, at−1), ˆz0 = enc(o0), zg = enc(og)." CEM 流程见 A.5.1（L1066–1077）：采样 N 条动作序列→世界模型潜空间前滚→取代价最低 K 条更新分布均值/协方差→迭代，执行前 k 个动作。备选 GD（L408–414）："Since our world model is differentiable, a possibly more efficient approach is to optimize this objective through gradient descent (GD) … However, we empirically observe that CEM outperforms GD in our experiments with full results in Appendix A.5.3."

**目标条件（goal conditioning）**：训练时模型不含任务信息（L290–296）："DINO-WM models only the information available from offline trajectory data in an environment, in contrast to recent online RL world models that also require task-relevant information, such as rewards (…), discount factors (…), and termination conditions (…)." 目标仅在测试时以观测图像形式进入代价（L236–240、L401–406）："At test time, our system starts from an arbitrary environment state and is provided with a goal observation in the form of an RGB image … and is asked to perform a sequence of actions a0, ..., aT to reach the goal state."

## 5. 实验与主结果（全部数值，不四舍五入）

**环境**（L435–446）："These environments include maze navigation (Maze, Wall), fine-grained control for tabletop pushing (PushT) and robotic arm control (Reach), and deformable object manipulation with an XArm (Rope, Granular)." 测试协议（L477–482）："For Maze, Reach, PushT, and Wall environments, we sample 50 initial and goal states and measure the success rate across all instances. Due to the environment stepping time for the Rope and Granular environments, we evaluate the Chamfer Distance (CD) on 10 instances for them." 观测（L444–445）："Observations in all environments are RGB images of size (224, 224)." 环境细节（A.1）：Maze=PointMaze（D4RL，"force-actuated 2-DoF ball"，L964–967）；Wall=两房间一门（L968–972）；Reach=Reacher（DMC，"a 2-joint robotic arm"，要求整条手臂匹配目标位姿，L973–975）；PushT=推 T 形块（L977–985，目标 25 步内可达，"both the agent and the T to match the target locations"）；Rope/Granular=XArm + 软绳/约百粒子，Nvidia Flex 仿真（L986–994）。

**真机 Kinova 结果：原文未写。** 全文无 "Kinova"、无真机实验章节；机械臂环境均为仿真（Reacher 来自 DMC；Rope/Granular 是 Nvidia Flex 仿真中的 XArm）。【推断】任务描述中的「真机 Kinova」可能与本论文之外的材料混淆。

**基线**（L455–457）："For IRIS, DreamerV3, and TD-MPC2, we train the models with our offline datasets without any reward or task information, and perform MPC on the learned world model for solving downstream tasks." A VDC 仅定性比较（L469–473）。

**表注原文**（L485–488）："Table 1. Planning results for offline world models on six control environments."（表头：Model / Maze SR↑ / Wall SR↑ / Reach SR↑ / PushT SR↑ / Rope CD↓ / Granular CD↓）

| Model | Maze SR↑ | Wall SR↑ | Reach SR↑ | PushT SR↑ | Rope CD↓ | Granular CD↓ |
|---|---|---|---|---|---|---|
| IRIS | 0.74 | 0.04 | 0.18 | 0.32 | 1.11 | 0.37 |
| DreamerV3 | 1.00 | 1.00 | 0.64 | 0.30 | 2.49 | 1.05 |
| TD-MPC2 | 0.00 | 0.00 | 0.00 | 0.00 | 2.52 | 1.21 |
| Ours | 0.98 | 0.96 | 0.92 | 0.90 | 0.41 | 0.26 |

要点（L493–501）：简单环境与 DreamerV3 相当："on simpler environments such as Wall and PointMaze, DINO-WM is on par with state-of-the-art world models like DreamerV3. However, DINO-WM significantly outperforms prior work at manipulation environments where rich contact information and object dynamics need to be accurately inferred for task completion." TD-MPC2 归因："the lack of reward signal makes it difficult to learn good latent representations."

**指标定义引用**：SR 原文未给形式化定义（仅 L477–479 引文）；CD 原文未写定义（仅 L480–481 引文）。LPIPS（L615–620）："We report the Learned Perceptual Image Patch Similarity (LPIPS) (Zhang et al., 2018) on the world models' predicted future frames, which assesses perceptual similarity by comparing deep representations of images, with lower scores reflecting closer visual similarity." SSIM（L1116–1117）："SSIM measures the perceived quality of images by evaluating structural information and luminance consistency between predicted and ground-truth images, with higher values indicating greater similarity."

**编码器选择**（表注 L520–521："Table 2. Planning results for world models with various pre-trained encoders."）

| Model | Maze SR↑ | Wall SR↑ | Reach SR↑ | PushT SR↑ | Rope CD↓ | Granular CD↓ |
|---|---|---|---|---|---|---|
| R3M | 0.94 | 0.34 | 0.40 | 0.42 | 1.13 | 0.95 |
| ResNet | 0.98 | 0.12 | 0.06 | 0.20 | 1.08 | 0.90 |
| DINO CLS | 0.96 | 0.58 | 0.60 | 0.44 | 0.84 | 0.79 |
| DINOPatch (Ours) | 0.98 | 0.96 | 0.92 | 0.90 | 0.41 | 0.26 |

解读（L538–543）："We posit that patch-based representations better capture spatial information, in contrast to models like R3M, ResNet, and DINO CLS, which reduce observations to a single global feature vector, losing crucial spatial details necessary for manipulation tasks."

**泛化到未见配置**（表注 L565–566："Table 3. Planning results for offline world models on three suites with unseen environment configurations."）

| Model | WallRandom SR↑ | PushObj SR↑ | GranularRandom CD↓ |
|---|---|---|---|
| IRIS | 0.06 | 0.14 | 0.86 |
| DreamerV3 | 0.76 | 0.18 | 1.53 |
| R3M | 0.40 | 0.16 | 1.12 |
| ResNet | 0.40 | 0.14 | 0.98 |
| DINO CLS | 0.64 | 0.18 | 1.36 |
| Ours | 0.82 | 0.34 | 0.63 |

原文叙述（L552–559）："From Table 5, we observe that DINO-WM demonstrates significantly better performance in WallRandom, indicating that model has effectively learned the general concepts of walls and doors, even when they are positioned in locations unseen during training."（注：此处正文写 "Table 5"，但对应表格实际编号为 Table 3，附录 Table 5 是数据规模消融——原文编号笔误【原文如此】。）GranularRandom（L576–586）："the agent encounters fewer than half the particles present during training, resulting in out-of-distribution images compared to the training instances. Nevertheless, DINO-WM accurately encodes the scene and successfully gathers the particles into a designated square location with the lowest Chamfer Distance (CD) compared to the baselines, demonstrating better scene understanding. We hypothesize that this is due to DINO-WM 's observation model encoding the scene as patch features, making the variance in particle number still within the distribution for each image patch."

**LPIPS 主表**（表注 L622："Table 4. Comparison of world models on LPIPS metrics."）

| Method | PushT LPIPS↓ | Wall LPIPS↓ | Rope LPIPS↓ | Granular LPIPS↓ |
|---|---|---|---|---|
| R3M | 0.045 | 0.008 | 0.023 | 0.080 |
| ResNet | 0.063 | 0.002 | 0.025 | 0.080 |
| DINO CLS | 0.039 | 0.004 | 0.029 | 0.086 |
| A VDC | 0.046 | 0.030 | 0.060 | 0.106 |
| Ours | 0.007 | 0.0016 | 0.009 | 0.035 |

**LPIPS+SSIM 附录全表**（表注 L1120："Table 9. Comparison of world models across different environments on LPIPS and SSIM metrics."）

| Method | PushT LPIPS↓ | Wall LPIPS↓ | Rope LPIPS↓ | Granular LPIPS↓ | PushT SSIM↑ | Wall SSIM↑ | Rope SSIM↑ | Granular SSIM↑ |
|---|---|---|---|---|---|---|---|---|
| R3M | 0.045 | 0.008 | 0.023 | 0.080 | 0.956 | 0.994 | 0.982 | 0.917 |
| ResNet | 0.063 | 0.002 | 0.025 | 0.080 | 0.950 | 0.996 | 0.980 | 0.915 |
| DINO CLS | 0.039 | 0.004 | 0.029 | 0.086 | 0.973 | 0.996 | 0.980 | 0.912 |
| A VDC | 0.046 | 0.030 | 0.060 | 0.106 | 0.959 | 0.983 | 0.979 | 0.909 |
| Ours | 0.007 | 0.0016 | 0.009 | 0.035 | 0.985 | 0.997 | 0.985 | 0.940 |

**推理/规划耗时**（表注 L1138–1140 见第 9 节；L1128–1137）："Table 10 reports the time required on an NVIDIA A6000 GPU for a single inference step, the environment rollout time for advancing one step in the simulator, and the overall planning time for generating an optimal action sequence using the Cross-Entropy Method (CEM)." 数值：Inference (Batch 32)=0.014 s；Simulation Rollout (Batch 1)=3.0 s；Planning (CEM, 100x10)=53.0 s（L1142–1144）。

**Intro 两个百分比声明**（L121–127）："On LPIPS metrics for our hardest tasks, this improves upon prior state-of-the-art work by 56% (See Section 4.7). … we show high success for reaching arbitrary goals on our hardest tasks, improving upon prior work by 45% on average (See Section 4.3)."

**生成式视频模型定性对比**（L592–596）："while A VDC can generate visually realistic future images, these images lack physical plausibility. Large, unrealistic changes can occur within a single timestep, and the model struggles to reach the exact goal state." 动作条件版 A VDC（L1109–1110）："the action-conditioned diffusion model diverges from the ground truth observations over long-term predictions, making it insufficient for accurate task planning."

## 6. 消融（全部抄录）

**A.4.1 数据规模（表注 L1020–1022）**："Table 5. Planning performance and prediction quality on PushT with DINO-WM trained on datasets of various sizes. SSIM and LPIPS are measured on the predicted future latents after decoding. We observe consistent improvement in performance as we increase the dataset size."（数据集 200→18500 条轨迹，L1015–1017）

| Dataset Size | SR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| n=200 | 0.08 | 0.949 | 0.056 |
| n=1000 | 0.48 | 0.973 | 0.013 |
| n=5000 | 0.72 | 0.981 | 0.007 |
| n=10000 | 0.88 | 0.984 | 0.006 |
| n=18500 | 0.92 | 0.987 | 0.005 |

**A.4.2 因果注意力掩码（表注 L1039–1040）**："Table 6. Comparison of DINO-WM with and without causal attention mask on PushT. We train models with varying historyh, representing the number of past observations the model takes as input."（"historyh" 为 txt 原样排版）解读（L1034–1038）："When h = 1where the model with and without this causal mask is equivalent, both models get decent and equivalent success rate. However, as we increase the history length, we see a rapid drop in the w/o mask case, since the model can cheat during training by attending to future frames, which are not available at test time. Adding the causal mask solves this issue…"

| | h = 1 | h = 2 | h = 3 |
|---|---|---|---|
| w/o mask | 0.76 | 0.36 | 0.08 |
| with mask | 0.76 | 0.88 | 0.92 |

**A.4.3 解码器重建损失（表注 L1052–1053）**："Table 7. Comparison of DINO-WM trained with and without loss from the decoder on PushT, highlighting the advantage of disentangling feature learning from reconstruction objectives."

| | Success Rate |
|---|---|
| w/o decoder loss | 0.92 |
| with decoder loss | 0.80 |

**A.5.3 规划优化器对比（表注 L1100）**："Table 8. Planning results of DINO-WM"（L1096–1099：CEM=开环一次执行无纠错；GD=开环执行全部动作；MPC=可重规划+滚动时域 CEM）

| | PointMaze | Push-T | Wall | Rope | Granular |
|---|---|---|---|---|---|
| CEM | 0.8 | 0.86 | 0.74 | NA | NA |
| GD | 0.22 | 0.28 | NA | NA | NA |
| MPC | 0.98 | 0.90 | 0.96 | 0.41 | 0.26 |

（注：第 2 节的编码器对比 Table 2 亦属消融，数值见第 5 节。）

## 7. 局限/未来工作（原文逐字，L641–654）

> "Limitations and Future Work: First, DINO-WM assumes having access to offline datasets with sufficient state-action coverage, which can be challenging to obtain for highly complex environments. This can potentially be addressed by combining DINO-WM with exploration strategies and updating the model as new experiences are available. Second, DINO-WM still relies on the availability of ground truth actions from agents, which may not always be feasible when training with vast video data from the internet. Lastly, while we currently plan in action space for downstream task solving, an extension of this work could involve developing a hierarchical structure that integrates high-level planning with low-level control policies to enable solving more fine-grained control tasks."

（另，规划器层面 L415–417："We hypothesize that incorporating regularizations during training and in the planning objectives could further improve performance, and leave this for future work."）

## 8. 叙事关联【推断】

1. **零样本规划对「世界模型不需要像素重建」的论证**。论文把"不重建视觉世界"作为方法定义本身（L15–17："a new method to model visual dynamics without reconstructing the visual world"），并声称训练与测试均免重建（L283–285："This eliminates the need to reconstruct images both during training and testing"）。且解码器损失回传反而降低规划成功率（Table 7：0.92 → 0.80）。【推断】DINO-WM 在 6 环境上的零样本规划成功，构成"潜空间预测足以支撑行为决策、像素级重建非必需"的直接经验证据，支持快脑层不做像素重建、只做潜轨迹预测的设计立场。
2. **对快脑层轨迹预测的架构启示**。支持原句：L331–334（帧级因果注意力自回归）、L338–343（帧级整体预测优于 token 级）、L364–366（"our world model training is entirely performed in latent space, without the need to reconstruct the original pixel images"）。【推断】"冻结的通用视觉编码器 + 轻量 ViT 潜空间动力学预测器"可作为快脑轨迹预测模块的候选结构模板；h=3 优于 h=1（Table 6）提示短历史上下文即可捕获速度/加速度/动量等动力学量。
3. **对快脑层计算效率的启示**。支持原句：L1132–1134（"The inference time of DINO-WM remains constant across environments due to the fixed model size and input image resolution, resulting in significant speedup over traditional simulation rollouts."）；Table 10：推理 0.014 s（批 32）vs 仿真 3.0 s/步。【推断】用潜空间动力学模型替代物理仿真器滚动有明确加速收益；但 Planning (CEM, 100x10)=53.0 s 显示零阶 CEM 优化仍是闭环瓶颈，快脑实时规划需更轻量优化器（原文也承认 GD 更高效但效果差，L408–414）。
4. **任务无关内核 + 目标条件化解耦 → 零样本**。支持原句：L290–296（训练只用离线轨迹信息，无奖励/折扣/终止条件）；L21–23（"facilitating task-agnostic planning by treating goal features as prediction targets"）；目标仅在测试时进入代价 C = ∥ˆzT − zg∥2（L401–406）。【推断】"任务无关的动力学内核 + 测试时注入的目标特征"的解耦方式，为快脑"通用预测层 + 任务相关目标模块"的分离提供模板。
5. **预训练感知 + patch 级表征对 OOD 的鲁棒性**。支持原句：L304–307（"perception is a general task that benefits from large-scale internet data"）；L576–586（GranularRandom 粒子数减半仍以最低 CD 成功，作者归因于 patch 特征使粒子数方差仍在单 patch 分布内）。【推断】快脑可复用大规模预训练视觉特征，仅训练轻量预测器；patch 级（非全局向量）表征对未见场景配置更鲁棒（Table 2/Table 3 中 R3M/ResNet/DINO CLS 全局向量在复杂任务上显著掉点）。

## 9. 图注/表注逐字抄录

**图注**（逐字，txt 原样）：

- Figure 1（L191–193）："Figure 1. We present DINO-WM , a method for training visual models by using pretrained DINOv2 embeddings of image frames (a). Once trained, given a target observation oT , we can directly optimize agent behavior by planning through DINO-WM using model predictive control (b). The use of pretrained embeddings significantly improves performance over prior state-of-the-art world models (c)."
- Figure 2（L278–280）："Figure 2. Architecture of DINO-WM . Given observations ot−k:t, we optimize the sequence of actions at:T −1 to minimize the predicted loss to the desired goal og. All forward computation is done in the latent space z. Here pθ indicates DINO-WM 's dynamics model, which is used for making future predictions."
- Figure 3（L452–454）："Figure 3. We evaluateDINO-WM on six environment suites, from left to right, top to bottom: Maze, Reach, Wall, Push-T, Rope Manipulation, and Granular Manipulation."（"evaluateDINO-WM" 缺空格为 txt 原样）
- Figure 4（L535–537）："Figure 4. Open-loop rollouts of world models on Push-T and Granular. Given the first frame and action sequence, each model predicts future frames, reconstructed by its decoder. For each environment, the bottom row denotes the ground truth. DINO-WM (Ours) rollouts are bolded and are visually indistinguishable from the ground truth observations."
- Figure 5（L553–555）："Figure 5. Training and testing visualizations for WallRandom, PushObj and GranularRandom. Test setups are highlighted in blue boxes."
- Figure 6（L605）："Figure 6. Plans generated by DINO-WM and A VDC."
- Figure 7（L1093–1094）："Figure 7. Open-loop rollout on PushT with DINO-WM and action-conditioned A VDC (A VDC-AC). For each trajectory, the model is given the first frame as well as sequence of actions. The world models performs open-loop rollout with these actions."
- Figure 8（L1180–1182）："Figure 8. Planning visualizations for PointMaze, Push-T, and Granular, on randomly sampled initial and goal configurations. The task is defined by Start and Goal, denoting the initial and goal observations. Final shows the final state the system arrives at after planning with each world model. For comparison, we show the best performing world models DINO CLS and DreamerV3."
- Figure 9（L1199–1200）："Figure 9. Trajectories planned with DINO-WM on all six environments. For each environment, the top (shaded) row shows the environment's observation after executing the planned actions, and the bottom row shows the world model's imagined observations."
- Figure 10（L1203）："Figure 10. Trajectories planned with DINO-WM on PushT with the same initial states but different goal states."
- Figure 11（L1206）："Figure 11. Trajectories planned with DINO-WM on PointMaze with the same initial states but different goal states."

**表注**（图注已列于各数据节处的，此处汇总其余）：

- Table 10（L1138–1140）："Table 10. Inference time and planning time for DINO-WM . Inference time represents the time required for a single forward pass for one step, while environment rollout time measures the simulator's speed for advancing one step. Planning time corresponds to Cross-Entropy Method (CEM) with 100 samples per iteration and 10 optimization steps."
- Table 11（L1157–1160）："Table 11. Environment-dependent hyperparameters forDINO-WM training. We report the number of trajectories in the dataset under Dataset Size, and the length of trajectories under Traj. Len."（"forDINO-WM" 缺空格为 txt 原样）数值（L1161–1169）：PointMaze H=3/Frameskip=5/2000/100；Reacher 3/5/3000/100；Push-T 3/5/18500/100-300；PushObj 3/5/20000/100；Wall 1/5/1920/50；WallRandom 1/5/10240/50；Rope 1/1/1000/5；Granular 1/1/1000/5。
- Table 12（L1170）："Table 12. Shared hyperparameters for DINO-WM training" 数值（L1171–1179）：Image size 224；Optimizer AdamW；Decoder lr 3e-4；Predictor lr 5e-5；Action encoder lr 5e-4；Action emb dim 10；Epochs 100；Batch size 32。

**其余表注**（已在上文逐字给出）：Table 1（L485–488）、Table 2（L520–521）、Table 3（L565–566）、Table 4（L622）、Table 5（L1020–1022）、Table 6（L1039–1040）、Table 7（L1052–1053）、Table 8（L1100）、Table 9（L1120）。

---

**结论**（L630–640，逐字）："We introduce DINO-WM , a simple yet effective technique for modeling visual dynamics in latent space without the need for pixel-space reconstruction. We have demonstrated that DINO-WM captures environmental dynamics and generalizes to unseen configurations, independent of task specifications, enabling visual reasoning at test time and generating zero-shot solutions for downstream tasks through planning. DINO-WM takes a step toward bridging the gap between task-agnostic world modeling and reasoning and control, offering promising prospects for generic world models in real-world applications."

**Impact Statement**（L657–661，逐字）："This paper presents work whose goal is to facilitate the learning and applications of task-agnostic world models. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here."
