# 驾驶 VLA 的 L / A / V 拆解对比

> 调研日期：2026-10-26 · 用途：《VLA 在 L4 的可执行落地方案》方法细节支撑
> 核心问题：L（语言）如何实现？A（动作）如何表达？V（视觉）是显式监督还是 compact token？
> 所有结论附原文章节引用，逐条经论文全文/源码核实（子代理 B 负责 LMDrive/CarLLaVA/Senna，其余由主代理直接精读）；全部引用逐字摘自原文。

## TL;DR

- **A（动作）只有两种主表达**：① 数值回归头出 BEV waypoint（UniAD/VAD/Senna/LMDrive，MLP 或 query 解码，速度快）；② 数值转文本 token 自回归（DriveVLM/EMMA/OmniDrive，控制量同理 DriveGPT4）——两者在论文里有明确对照（EMMA §2 讨论 "text conversion" vs "special tokens"，选文本）。
- **V（视觉监督）有三档**：全显式（UniAD 五任务全监督、VAD 地图/运动向量+约束、Omni-Q 检测 queries）→ 半显式（LMDrive 仅预训练头、Senna E2E 侧有/VLM 侧无）→ 零显式（CarLLaVA、DriveGPT4、Omni-L、Senna-VLM、EMMA 基础版）。**EMMA 以纯自监督（唯一监督=ego 未来轨迹）在 nuScenes 上反超 DriveVLM-Dual 6.4%**，是「compact token 无中间监督可行」的最硬证据——但代价是 Gemini 预训练 + 1.1M 级数据 + 推理重。
- **L（语言）有四种位置**：无语言（UniAD/VAD，3 类导航命令嵌入）→ 仅输入指令（LMDrive 拼接 token；CarLLaVA 连输入都没有、语言只是可选输出装饰）→ 双向（DriveVLM/Senna 输出 meta-action 文本，再转成低层条件）→ 一切皆文本（EMMA/DriveGPT4 输入输出全文本化）。**语言作为中间层能涨点**：EMMA 消融 meta-decision +3.0%、critical object +1.5%（Table 3）。
- **语言输入是把双刃剑（评测证据）**：ICR-Drive——指令措辞微扰即大幅掉分（misleading ΔDS −33.64/−32.50）；Language2Drive——93.2% 的成功 rollout 换一条等价指令就失败，且 "Training may allow language to remain unnecessary for action selection"（J.2）。用语言控制车辆必须配固定词典 + 成对反事实评测。
- 与你的问题直接对应的三句话：A 的表达=waypoint（车场景无一例外，控制量只有 DriveGPT4 一家）；V 是否显式监督=工程权衡而非理论必需（显式=可 debug、贵；compact=数据便宜、黑盒）；L 的实现=「指令 token 注入」或「决策文本输出」，两种都在量产方案中出现过（LMDrive/Senna/DriveVLM-Dual）。

## 一、分类框架（survey 原文定义）

依据 [Vision-Language-Action Models for Autonomous Driving: Past, Present, and Future](https://arxiv.org/abs/2512.16760)（arXiv 2512.16760，WorldBench Team）：

- **两大范式（Abstract）**："_End-to-End VLA_, which integrates perception, reasoning, and planning within a single model, and _Dual-System VLA_, which separates slow deliberation (via VLMs) from fast, safety-critical execution (via planners)"。Dual-System 的机制原文（§2.2.2）："the VLM functions as a high-level reasoning module that produces intermediate guidance—textual rationales or structured latent intents, which a downstream planner converts into low-level actions. This 'slow thinking + fast execution' architecture..."。本组对应：End-to-End = EMMA/DriveGPT4/LMDrive/CarLLaVA/VLDrive/BEVDriver；Dual-System = DriveVLM-Dual/Senna/VLP。
- **动作头四型（§2.3，比「文本/数值」二分类更细）**：
  - **Language Head (LH)**："directly utilizes the VLM's inherent text-generation capabilities to produce actions in the language space... output either free-form textual commands (e.g., 'turn left') or a sequence of discretized action tokens... widely adopted in textual action generators like DriveMLM and DriveGPT4"。
  - **Regression (REG)**："employs a decoder structure followed by a regressor (typically a Multi-Layer Perceptron) to directly predict continuous values... steering angles, throttle/brake values, or trajectory waypoints. Representative methods... LMDrive and DriveGPT4-V2"。
  - **Trajectory Selection (SEL)**："evaluates a set of candidate trajectories and selects the optimal one... utilized by methods like WoTE and SeerDrive"。
  - **Trajectory Generation (GEN)**："synthesizes actions through probabilistic modeling, most notably using diffusion models or variational autoencoder... Prominent examples include ORION and DiffVLA"。
- **动作空间四类（§2.4）**：Discrete trajectory representations / Continuous trajectory representations / Direct control representations / Language representations。
- **语言输入定义（§2.1.3）**："a sequence of tokens representing the driving task or goal (e.g., 'turn left at the next intersection')"。
- **survey 记载、本表尚未逐篇精读的代表作（§1）**：DriveMLM、GPT-Driver、Diff-VLA、InsightDrive（"integrates causal language reasoning with MPC, assigning why to the VLM and how to the planner"）、AutoVLA（"fast/slow thinking and GRPO-based optimization"）、SimLingo（"explicitly studies language-action alignment"）、DriveGPT4-V2、ORION、WoTE、SeerDrive——后续可按需扩读。



## 二、逐篇拆解（引用均逐字摘自论文原文）

### 2.1 端到端非语言基线：UniAD / VAD（无 L，V 显式监督，A=waypoint 回归）

#### UniAD（2212.10156，CVPR 2023）

- **L**：无语言。导航信号转为可学习嵌入："we convert the raw navigation signals (i.e., turn left, turn right and keep forward) into three learnable embeddings, named command embeddings"（§2.4）。
- **A**：未来 waypoint 回归。"As the ego-vehicle query from MotionFormer already expresses its multimodal intentions, we equip it with command embeddings to form a 'plan query'. We attend plan query to BEV features B to make it aware of surroundings, and then decode it to future waypoints τ̂"（§2.4）。推理期再用牛顿法优化避碰："we optimize τ̂ based on Newton's method"（§2.4, Eq.9-11，用 OccFormer 占用图，仅推理时）。
- **V**：**显式全栈监督**（Table 1 中 Det/Track/Map/Motion/Occ/Plan 六项全 ✓）。TrackFormer（"additional track queries are introduced to track agents across frames"，§2.1）、MapFormer（"panoptic segmentation of the map... lanes, dividers and crossings as things, and the drivable area as stuff"，§2/§2.1）、MotionFormer（"predicts all agents' multimodal future movements, i.e., top-k possible trajectories"，§2.2）、OccFormer（"predicts multi-step future occupancy with agent identity preserved"，§2）。BEV 由 "an off-the-shelf BEV encoder in BEVFormer" 生成（§2）。训练期 GT 轨迹经 "non-linear smoother... make them physically feasible"（§2.2，仅训练）。推理 555.6ms/1.8 FPS（A100，VAD Table 1 转引）。

#### VAD（2303.12077，ICCV 2023）

- **L**：无语言。"VAD uses three kinds of driving commands: turn left, turn right and go straight"（§3.2）。
- **A**：MLP 头回归轨迹："outputs the planning trajectory V̂_ego ∈ R^{Tf×2}. VAD adopts a simple MLP-based planning head"（§3.2）。VAD-Base 224.3ms/4.5 FPS、VAD-Tiny 59.5ms/16.8 FPS（RTX 3090，Table 1）。
- **V**：**显式感知监督 + 感知结果作显式规划约束**。map queries "predicts map vectors V̂_m ∈ R^{Nm×Np×2} and the class score for each map vector. Three kinds of map elements are considered: lane divider, road boundary, and pedestrian crossing"（§3.1）；agent queries 解码属性并 "predicts future trajectories of each agent, represented as multi-modality motion vectors V̂_a ∈ R^{Na×Nk×Tf×2}"（§3.1）。Loss："Manhattan distance is adopted to calculate the regression loss between the predicted map points and the ground truth map points... focal loss is used as the map classification loss"；"we use l1 loss as the regression loss to predict agent attributes (location, orientation, size, etc.)"（§3.4）。三条训练期约束："the ego-agent collision constraint... ego-boundary overstepping constraint... ego-lane direction constraint"（§1/§3.3）。

### 2.2 VLM 规划 + 传统执行：DriveVLM(-Dual) / Senna（L 双向，A 文本→数值两级）

#### DriveVLM / DriveVLM-Dual（2402.12289）

- **L 输出侧**：CoT 三段——"scene description, scene analysis, and hierarchical planning"（§1）。环境描述为结构化语言："E={E_weather, E_time, E_road, E_lane}"（§3.2）。关键物体："The category and coordinates are mapped to their corresponding language token_id in the language modality"（§3.2）——检测框坐标也转成文本 token。
- **A**：文本自回归生成数值 waypoint。"DriveVLM progressively generates driving plans, in three stages: meta-actions, decision description, and trajectory waypoints"；meta-actions "fall into 17 categories"（§3.4）；"Trajectory Waypoints W... wi=(xi,yi)... **We map these numerical waypoints into language tokens for auto-regressive generation**"（§3.4）——waypoint 数值直接当文本 token 生成，无回归头。
- **Dual（快慢系统）**：3D 检测结果 "used as language prompts for the model"（§3.5）；"After obtaining a trajectory from DriveVLM at low frequency, denoted as W_slow, we take it as a reference trajectory for a classical planner for high-frequency trajectory refinement... W_fast = Planner([W_slow, f])"（§3.5, Eq.1）；"the two branches operate asynchronously in a slow-fast manner"（§3.5）。量产车部署验证。
- **V**：图像序列入 VLM，VLM 本身**无感知回归 loss**（感知以文本描述/QA 监督）。骨干："Qwen-VL... 9.6 billion parameters, including a visual encoder (1.9 billion), a vision-language adapter (0.08 billion), and a large language model (Qwen, 7.7 billion). Images are resized to a resolution of 448×448"；输入 "a sequence of images at the current time T s, T-1 s, T-2 s, and T-3 s"（§5.1）。
- 数据监督：SUP-AD，"except for waypoints, which can be auto-labeled from the vehicle's IMU recordings"（§4.3）；描述/分析/规划人工标注；keyframe 取 "0.5s to 1s earlier than the actual maneuver"（§4.3）；co-tuning Talk2Car/BDDX/Drama/SUTD/LLAVA（§5.1）。

#### Senna（2410.22313）

- **L 输入+输出**："The input scene information includes multi-view image sequence, user instruction, and navigation command. The user instruction is fed into Senna-VLM as prompts"（§III）。输出高层决策文本 meta-action：横向 {Left, Straight, Right} × 纵向 {Accelerate, Keep, Decelerate, Stop}（§III-A），由 GT 轨迹规则化生成。
- **L→A 链路**：语言决策经 Meta-action Encoder 的可学习 embedding 映射为特征 e_act，作为 Senna-E2E 的额外交互 token（§III-B）——VLM 语言输出不是轨迹本身，而是条件信号。
- **A**：Senna-E2E（扩展 VADv2）"planning tokens 注意力交互预测轨迹 V=Φ(I, e_nav, e_act) ∈ R^{T×2}"（§III-C）。训练用 GT meta-action、推理用 VLM 预测（§III-D）。开环轨迹，无 PID 后处理；E2E 各任务 loss 公式原文未写（仅报 L2 displacement error + 碰撞率，§IV-A）。
- **V**：VLM 用 CLIP ViT-L/14，224×224 每图 576 tokens；Driving Vision Adapter 用可学习 queries 压缩，"最优 128 tokens/图、6 视角"（§III-B；不压缩会 "model collapse"）。VLM **无检测 loss**，感知监督全是自动生成的 planning-oriented QAs（红绿灯状态、VRU 距离、他车意图、场景描述，GPT-4o 生成，§III-A）；E2E 侧含 VADv2 感知/预测多任务监督（§III-C）。
- 训练三阶段（§III-D）：① Mix Pre-training（仅训 Driving Vision Adapter）→ ② Driving Fine-tuning（除 Vision Encoder 外全参数微调）→ ③ Planning Fine-tuning（仅 meta-action planning QA）。数据：DriveX 1000K clips × 3s + nuScenes（§IV-A）。

### 2.3 语言条件闭环：LMDrive / CarLLaVA（L 输入指令；A=waypoint+双 PID）

#### LMDrive（2312.07488，CVPR 2024）

- **L 输入层，无语言决策输出**："we apply the LLaMA tokenizer to convert the instruction into textual tokens"（§4.2），与视觉 token 拼接进 LLM。指令数据：CARLA 规则专家跑 2.5k 路线采集后解析标注，56 种模板指令（follow/turn/others + notice），"we utilized ChatGPT API to generate eight different variants"（§3）；共 64K clips、464K notice 指令（§3）。"the LLM predicts the action tokens"（§4.2）——直接出动作，不产文本决策。
- **A**：LLM → 2 层 MLP adapter → 未来 5 个 waypoint（"C,H,W,N are set as 256,50,50,5"，§5.1）+ 指令完成 flag；"we use two PID controllers for latitudinal and longitudinal control to track the heading and velocity of predicted waypoints"（§4.2）。损失：L1 waypoint + 完成标志交叉熵（§4.2）。waypoint 地平线/坐标制原文未写。
- **V**：4 路 RGB（左右 60° + 前视 focus-view 裁剪）+ 64 线 LiDAR（§3）；ResNet-50 + transformer 融合，PointPillars→BEV queries，输出三类 token（BEV / waypoint / traffic light，§4.1）。**显式感知监督只在 vision encoder 预训练阶段**："the vision encoder is pre-trained on perception tasks by adding additional prediction heads"（CenterPoint 检测 + 灯色分类 + waypoint 回归，§4.1）；微调时丢头、每帧 406 token 经 Q-Former（M=4）压缩，LLM 冻结："the Q-Former and Adapters are trainable and the other components are frozen"（§4.3）。

#### CarLLaVA（2406.10165）

- **L 仅输出侧（可选），无语言输入**："The model inputs include camera images, the next two target points, and the ego vehicle's speed"（§3）——导航靠数值目标点。语言仅是可选驾驶 commentary："we auto-regressively sample the language explanation after generating the path and waypoints"（§3），标签由规则专家逻辑生成（IDM 前车 + 专家启发式，§3）；作者声明 "This is not intended as an actual explanation"（§4）。语言→动作无链路。
- **A**：半解耦双表示："getting the advantages of the path for better lateral control and the waypoints for better longitudinal control"（§3）——time-conditioned waypoints 供纵向 PID、space-conditioned path 供横向 PID；learnable queries → MLP 预测 waypoint 差分、累加得 waypoint，MSE loss（§3）。无 GRU。waypoint 数量/频率原文未写。
- **V**：**零辅助感知标签**。LLaVA-NeXT 编码器（CLIP ViT-L-336px + anyres 高分块），单前视 C1T1（变体 C1T2/C2T1）；"our model is the only model on the leaderboard working only with camera images and without the usage of additional auxiliary labels"（§4, Table 1）。
- 训练：PDM-light 特权规则专家采集 2.9M 样本 @5fps（§3），分桶采样（每 epoch 650K，§4）；LLaMA-50M/350M 从零训、TinyLLaMA-1B LoRA（§4）。

### 2.4 纯 VLA 大模型：DriveGPT4（控制量即语言）

#### DriveGPT4（2310.01412）

- **L**：QA 交互式。动作描述/理由/控制预测三类问答（§3）；ChatGPT 用特权信息（YOLOv8 检测框 + GT 控制信号 + caption）生成 40K 额外 QA（§3），共 56K 指令样本。
- **A**：**控制量与文本共用同一 tokenizer，控制量被当作语言**。"texts and control signals utilize the same text de-tokenizer, signifying that control signals can be interpreted as a language"（§4.1，引 RT-2）；"DriveGPT4 should predict control signals in the next step (i.e., (v_{N+1}, Δ_{N+1}))"；"The turning angle represents the relative angle between the current frame and the previous frame"；"Predicted control signals are embedded in the output texts using a fixed format"（§4.1）——速度+转角以固定文本格式输出，无回归头、无 PID。
- **V**：**无显式感知监督**（检测仅作为 ChatGPT 生成 QA 时的特权信息，不进训练 loss）。"Videos are uniformly sampled into a fixed number of images, and a video tokenizer based on Valley is employed to convert video frames into text domain tokens"（§4.1）；CLIP 视觉编码器输出 "F_i ∈ R^{257×d}"（全局 + 256 patch），时间特征=全局特征拼接、空间特征=Pooling（§4.1, Eq.1-2）。
- 训练：预训练 "593K image-text pairs from the CC3M dataset and 703K video-text pairs from the WebVid-2M... Only the projector is trained"（§4.2）；mix-finetune "56K video-text instruction-following data... in conjunction with 223K general instruction-following data"（§4.2）。

### 2.5 一切皆文本：EMMA / DriveGPT4 之外的两篇（EMMA 精读 + OmniDrive）

#### EMMA（2410.23262，Waymo）

- **L**：所有非传感器输入输出都是纯文本。"EMMA accepts camera images and plain text for other non-vision inputs such as high-level driving commands and historical context"（§1）；"All sensor data are represented as stitched images or videos as V; all router commands, driving context, and task-specific prompts are represented as T; all output tasks are presented as language outputs O"（§2）。坐标表示选型有明确论述："The first is direct text conversion to floating-point numbers... RT-2 exemplifies this approach in robotic control. The second approach uses special tokens to represent each location or action... MotionLM leverages this method... **We opt for the text representation** so that all tasks can share the same unified language representation space"（§2）。
- **A**：未来 BEV waypoint 以纯文本自回归生成。"represented as a set of future trajectory waypoints for the ego vehicle in the same BEV space: O_trajectory = {(x_t,y_t)}_{t=1}^{T_f}... where all output waypoints are also represented as plain text"（§2.1）；历史 ego 状态同样 "All waypoint coordinates are represented as plain text without specialized tokens"（§2.1）；intent 命令 "derived from the router, includes directives such as 'go straight', 'turn left', 'turn right'"（§2.1）。"These generated trajectories are then transformed into vehicle-specific control actions such as acceleration and turning"（§2.1）——轨迹→控制信号的下游转换。多轨迹采样："we also sample multiple trajectories using a Top-K decoding strategy, up to K=24... select the one with the lowest average L2 distance"（WOMD，§3.1.1）；nuScenes 上 "we report only top-1 predictions"（§3.1.2）。
- **L 中间层（CoT）**：先出理由再出轨迹。"asking the model to articulate its decision rationale O_rationale while predicting the final future trajectory waypoints"（§2.2）；R1 场景描述、R2 关键物体（"we require the model to identify their precise 3D/BEV coordinates"）、R3 行为描述、R4 meta 决策（"includes 12 categories of high-level driving decisions"）（§2.2）。理由标签全自动生成："the driving rationale captions are generated using an automated tool without any additional human labels... Meta driving decisions are computed using a heuristic algorithm that analyzes the ego vehicle's ground-truth trajectory"（§2.2）。**消融证据**：meta-decision +3.0%、critical object +1.5%、四件套 +6.7%（Table 3）；且 "we can predict O_trajectory first and apply early stopping during inference for time-critical applications"（§2.2）。
- **V**：**基础规划版零感知监督（自监督）**。"Self-supervised: the only required supervision is the future locations of the ego vehicle. No dedicated human labels are needed. Camera-only... HD map free"（§2.1）；nuScenes 上 "our self-supervised EMMA achieves state-of-the-art results in planning... outperforms previous BEV-Planner by 17.1%... even compared to DriveVLM-Dual that heavily uses intermediate perception and human labels, our self-supervised EMMA still improves the average L2 metric by 6.4%"（§3.1.2, Table 2）。Generalist 版才有任务输出监督：3D 检测 "We follow Pix2Seq and formulate the output 3D bounding boxes as O_boxes = set{text(x,y,z,l,w,h,θ,cls)}... writing floating-point numbers with two decimal places"（§2.3）、road graph（"a text-encoded road graph represented as waypoints"，§2.3）、temporary blockage（§2.3）。模型："We leverage the smallest size of Gemini, i.e., Gemini 1.0 Nano-1"（§3）。局限（作者自述）："it can process only a small amount of image frames, does not incorporate accurate 3D sensing modalities like LiDAR or radar and is computationally expensive"（§1）。

#### OmniDrive（2405.01533，NVIDIA）

- **L**：数据集视角的工作——反事实 Q&A 生成。关键帧选择（CLIP embedding + K-means，再按未来轨迹聚类 200 中心，§2.1）；GPT-4 输入为拼接图 + caption + 模拟/专家轨迹 + 需关注物体（"We also identify an object as 'close', if its minimum distance to the trajectory is smaller than 10 meters in the next 3 seconds"，§2.2）。四类 QA：scene description / attention / counterfactual reasoning / decision making and planning / general conversation（§2.3）。
- **A**：轨迹在 Q&A 中文本化。"Simulated trajectory: [PT, (+4.85, -0.08), (+9.71, -0.22), …]"（Table 1, Prompt type 2）；"Answer: The most suitable trajectory to follow would be [PT, (+0.76, +0.02), (+1.45, +0.03), …]"（Table 1, Conversation type 3）——动作即文本坐标列表。训练只有文本生成 loss："In both stages, we calculate the text generation loss"（§3.3）。Open-loop 规划用 L2/碰撞/驶出率评测（Table 2）。
- **V 监督对比（两种 agent 架构）**：Omni-L = LLaVA 式 MLP 对齐，多视角 flatten + 3D 位置编码（"for training stability, these position encoding weights are initialized to zero"，§3.2）——**无感知监督**；Omni-Q = StreamPETR BEV + Q-Former，"the perception queries Q_d are used to predict the categories and coordinates of the foreground elements. The carrier queries Q_c are sent to a MLP to align with the dimension of LLM tokens"（§3.1）——**有显式感知监督**。结论："Our findings indicate that migrating 2D VLMs to 3D is a more straightforward approach compared to integrating traditional 3D perception stacks into VLMs"（§1）；但规划指标上 Omni-Q++（0.33 avg L2）优于 Omni-L++（0.40）（Table 2）。视觉编码器："Our model uses EVA-02-L as the vision encoder"（§4.1）。训练两阶段：2D-Pretraining + 3D-Finetuning（§3.3）。

### 2.6 2024-2026 补位：VLP / BEVDriver / GenAD / DriveVLA-W0 / VLDrive / ICR-Drive

> 存在性核实（arXiv API）：**EMMA-2 不存在**（all:"EMMA-2" 0 结果，无官方博客）；「Driving with LLM Foresight」「Verbalized Vehicle Behavior Planning」「CarOmni」均不存在；真实 VLP = CVPR 2024 arXiv 2401.05577。留证全文见 `evidence/` 目录：`vlp_2401.05577.txt`、`bevdriver_2503.03074.txt`、`genad_2403.09630.txt`、`drivevlaw0_2510.12796.txt`、`vl_drive_full.txt`、`icr_drive_full.txt`。

#### VLP: Vision Language Planning（2401.05577，CVPR 2024）

- **L**：**语言只训练期用、推理时完全移除**。"VLP is only active during training, ensuring no additional parameters or computations are introduced during inference"（Fig.2 caption）。注入=冻结 LLM 编码 GT 化 prompt，对比损失对齐 BEV 特征与规划特征；LM 冻结只训适配层："we freeze the LM and only train the adaptation layer F_bev"（§3.2）。指令数据=nuScenes GT 标注本身，无人工语言标注。
- **A**：不新增动作头，原 ADS（VAD/UniAD 类）规划头保留："P×2 indicate the planned waypoints in the next P timesteps"（§3.3, Eq.4）。
- **V**：image backbone→BEV encoder；**有全套显式感知监督**（3D tracking/mapping/motion/occupancy/planning，§3.1），ALP 用 GT 3D bbox+全景分割 mask 池化对齐。
- 训练：nuScenes；首次跨城泛化实验（贡献 4）。

#### BEVDriver（2503.03074，IROS 2025）

- **L**：LangAuto 导航指令（7 个离散指令映射自然语言）→ 改造 BLIP-2 Q-Former："adapted for the BEV inputs to be aligned with the navigation tokens in the natural language space employing 32 learnable query tokens with 768 dimensions"（§III-B）→ LLM（LoRA rank 16）。
- **A**：**GRU adapter 用 LLM 最后隐层出 5 个未来 waypoint**："A GRU adapter network predicts five future waypoints using the last hidden layer of the LLM, while another 2-layer MLP adapter determines a completion flag... two PID controllers... regulate braking, throttling, and steering"（§III-B）。
- **V**：InterFuser 编码器（ResNet-50 图像 + ResNet-18 LiDAR → transformer → 256 维）；**显式感知监督四损失**（L1 检测、CE 红绿灯、focal+加权 CE 分割 α=0.9/γ=5、对比时序对齐，§III-A）；推理丢 decoder："During inference, we discard the decoder... 5.37M parameters and an inference frequence of 18Hz"（§III-A）。
- 训练：LMDrive 15k 序列 + 自采 2k 语义分割序列；BEV 编码器冻结为主、LLM 走 LoRA；Llama-7b / Llama-3.1-8B 双 backbone。

#### GenAD（2403.09630，CVPR 2024，OpenDriveLab）

- **L**：文本是扩散条件："c is the text condition that guides the denoising process, which is a concatenation of context and command"（§3.1）；command 由 14 类动作分类器打标+词典映射，context 由 BLIP-2/GPT 生成；CLIP text encoder 冻结。
- **A**：无原生动作头；Planning 扩展："extract spatiotemporal features of two historical frames through the UNet encoder of the frozen GenAD... feed them to a multi-layer perceptron (MLP) to predict future waypoints... the training process of our planner can be sped up by 3400 times compared to an end-to-end planning model UniAD"（§3.3）。
- **V**：SDXL latent diffusion（256×448、2Hz clip）；**无感知监督**，纯未来帧噪声预测自监督。
- 训练：OpenDV-2K 2059h；Stage1 图像域迁移（全参 UNet）→ Stage2 视频预测（冻结 Stage1 UNet，只训 temporal reasoning blocks）。

#### DriveVLA-W0（2510.12796，2025，中科院自动化所）

- **L**："High-level driving language instructions (L_t) are processed using the VLM's native tokenizer"（§3.1）；深交错序列 S_t=[L_{t-H},V_{t-H},A_{t-H-1},…,L_t,V_t,A_{t-1}]（§3.1）。
- **A**：**waypoint 轨迹经 FAST tokenizer 离散化为动作 token 自回归**："we use the FAST tokenizer to convert continuous waypoint trajectories into a sequence of discrete tokens, denoted A_{t−1}"（§3.1）；交叉熵 L_Action=−Σ log P(a_i|S_t,a_<i)（Eq.1）。500M MoE Action Expert 三种解码：query-based（MLP 直接回归连续 waypoint、L1）、autoregressive（CE）、flow matching（MSE + ODE 采样，§3.3）。评测：3 秒 6-waypoint @2Hz，ADE+碰撞率（§4.1）。
- **V**：双 archetype——VQ=Emu3-8B 离散视觉 token / ViT=Qwen2.5-VL-7B 连续特征（§3.1）；**无感知监督**，世界模型自监督（AR 离散 token 预测或扩散未来 latent 去噪，L_Total=L_Action+α L_WM，§3.2）；推理时世界模型分支绕过，延迟降至基线 63.1%。
- 训练：Stage1 6VA 序列 WM+action 联合预训练；Stage2 2VA 序列仅 action loss、不冻结 backbone（§4.2）；in-house 70M 帧/1M clips；NAVSIM v1 单目 PDMS 93.0、v2 EPDMS 86.1（Table 1/2）。

#### VLDrive（2511.06256，2025）

- **L**：指令在 LLM 输入层注入："process the combination of visual tokens of all frames … and corresponding text embedding F_t … tokenized from navigation instructions"（§3.1）；轻量 LM："LLaMA* (a 4-layer lite version of LLaMA) and TinyLLaMA"（1.3B，§4.1）；**无文本输出**（隐层→MLP→轨迹→PID，§3.1/Fig.2）；DDIA 使指令↔视觉交叉注意力不受 RoPE 位置编码影响（§3.5）。指令数据=LangAuto 64K clips（§4.1），生成方式原文未写。
- **A**：轨迹 L1 回归："L_way supervises trajectory prediction by minimizing the L1 norm between predicted and ground-truth paths"（§3.6）；维度/地平线/频率原文未写；PID 转转向+加速度（§3.1）。
- **V**："We adopt the visual encoder in LMDrive, producing 106 visual tokens (N=106) and kept it fixed during training"（§4.1）；Gumbel-Softmax 动态剪枝（保留率 R=0.3，§3.2/§4.1）+ MEFA(Q-Former) 聚合；**无检测/占用/建图监督**，唯一辅助是训练期自监督 token 重建 L_rec（MSE 重建被剪 token，仅训练时用，§3.6）。

#### ICR-Drive（2604.05378，2026，纯评测框架）

- 性质：不训模型，只变指令文本："Both agents are evaluated using their publicly released checkpoints with no retraining or fine-tuning; only the instruction text is varied"（§4.2）。
- 基线机制（论文转述）：LMDrive = "integrates visual encodings with natural-language instructions using a Q-Former and a large language model backbone (e.g., Vicuna-7B), followed by a trajectory and control prediction head trained via behavioral cloning on LangAuto"（§4.2）；BEVDriver = "aligned with natural language instructions via a Q-Former and passed to a Llama-7B backbone for waypoint prediction"（§4.2）。
- 扰动：规则模板 4 族（paraphrase/ambiguity/noise/misleading）× K=8 变体，每路线 32 条反事实指令；另有 GPT-4o 生成变体（§4.3/§4.4.1）。
- 结论："goal-preserving perturbations (paraphrase, ambiguity, noise) can still induce measurable degradation, while goal-conflicting directives (misleading, authority-framed overrides) expose pronounced susceptibility to goal override"（§5）；LangAuto-Tiny 上 LMDrive misleading ΔDS −33.64、BEVDriver −32.50（Table 1）；"DS changes are largely RC-driven"（§4.4.1）。

### 2.7 Language2Drive：指令保真度评测基准（2610.16634，2026，清华 AIR 等）

- 性质：闭环指令跟随评测基准（非模型）。核心评测单元 PCU（paired counterfactual unit）："Both experimental conditions are initialized with the same map, ego/actor states, history, non-language inputs, and simulation seed. The human language instruction changes"（§2.2）。
- 指标：SISR（单指令严格成功率）与 CFPS（成对反事实双成功）；"Success requires the complete requested contract, rejection of its paired alternative, and all applicable safety and execution conditions"（§2.4）。
- 四个语言控制 scope："(A) Atomic actions, (B) Persistent behaviors, (C) Scene-grounded instructions, and (D) Long-horizon programs"（§1/Fig.1）。环境：CARLA-Language2Drive（6 张定制地图：地下停车场/港口/中国城市道路/美式城市/园区/高速）+ AlpaSim 重建真实场景 + nuScenes/NAVSIM/WOD-E2E 开环（§1）。
- **核心发现（原文）**："Across principal CARLA systems, **93.2% of successful rollouts lack a successful opposite-instruction counterpart**"（§1）；"In reconstructed scenes, Alpamayo 1.5 achieves 100% Safety and a reference-path DrivingScore of 97.17, yet zero CFPS"（§1）；"no evaluated CARLA system completes both alternatives under all three tested language expressions"（§1）。被评测模型全名单在 §3（页面截断未读到，仅确认含 Alpamayo 1.5）。
- 失效机制（J.2 节标题，仅读到标题）："Training may allow language to remain unnecessary for action selection"；"Language understanding may not be preserved through planning and control"——即**语言在训练中可能根本没参与动作选择**，这是 L 真正起作用前必须排除的隐患。

## 三、L / A / V 三轴总表

| 论文 | L 如何实现 | A 如何表达 | V 是否有显式监督 |
|---|---|---|---|
| UniAD | 无语言；3 类导航命令 → 可学习嵌入（§2.4） | 未来 waypoint τ̂ 回归（plan query→BEV 解码，§2.4）；推理期牛顿法避碰 | ✓ 全栈显式：检测+跟踪+建图+运动预测+占用（Table 1） |
| VAD | 无语言；3 类驾驶命令（§3.2） | MLP 头回归 V̂_ego∈R^{Tf×2}（§3.2） | ✓ 显式：地图向量+agent 属性/轨迹；感知结果作 3 条训练期规划约束（§3.3） |
| DriveVLM(-Dual) | 输出 CoT 文本（描述/分析/层级规划）；17 类 meta-action；坐标转文本 token（§3.2/§3.4） | waypoint 数值→语言 token 自回归（§3.4）；Dual 下 W_slow 交传统规划器高频细化（§3.5） | ✗ VLM 无感知回归 loss（QA 文本监督）；Dual 把 3D 检测结果转文本 prompt 注入（§3.5） |
| Senna | 输入：指令 prompt；输出：meta-action 文本（3 横向×4 纵向），可学习 embedding 注入 E2E（§III-A/B） | VADv2 轨迹 V∈R^{T×2} 回归（§III-C） | 混合：VLM 无检测 loss（QA 监督）；E2E 侧有 VADv2 感知/预测多任务（§III-C） |
| LMDrive | 指令 token（LLaMA tokenizer）与视觉 token 拼接（§4.2）；无文本决策输出 | LLM→2 层 MLP adapter→5 waypoint+完成 flag（§5.1）；双 PID 跟踪（§4.2） | 半显式：vision encoder 预训练有 CenterPoint 检测+灯色+waypoint 头（§4.1），微调丢头 |
| CarLLaVA | 无语言输入（目标点+车速）；可选输出 commentary（§3） | waypoint（纵向 PID）+ path（横向 PID）双表示，MSE（§3） | ✗ 零辅助标签（§4, Table 1） |
| DriveGPT4 | QA 交互；ChatGPT 生成 40K QA（§3） | 控制量（速度+转角）以固定文本格式生成，与文本共用 tokenizer（§4.1） | ✗ 无（YOLOv8 检测仅作 QA 生成特权信息，不进 loss） |
| EMMA | 一切非传感器输入输出=纯文本（§2）；CoT R1-R4 后出轨迹（§2.2） | BEV waypoint 纯文本自回归（§2.1）；WOMD top-K=24 采样（§3.1.1） | 基础规划版 ✗ 自监督（唯一监督=ego 未来轨迹，§2.1）；Generalist 版检测/路图文本输出监督（§2.3） |
| OmniDrive | 反事实 Q&A 数据集（GPT-4 生成，§2） | 轨迹=文本坐标列表（QA 答案，Table 1） | 架构对比：Omni-L ✗ / Omni-Q ✓（检测 queries，§3.1-3.2） |
| VLP | 训练期冻结 LLM 编码 GT prompt+对比对齐，推理移除（Fig.2 caption） | 原 ADS 头不变：P×2 waypoints（§3.3 Eq.4） | ✓ 全套 P3 感知监督（§3.1） |
| BEVDriver | 导航指令 token+Q-Former 对齐 BEV 进 LLM（LoRA，§III-B） | GRU 用 LLM 隐层出 5 waypoint+完成 flag；双 PID（§III-B） | ✓ 检测/红绿灯/分割四损失，推理丢 decoder（18Hz，§III-A） |
| GenAD | 文本条件 c=context+command 注入扩散（§3.1） | 无原生头；冻结 UNet 编码器+MLP 回归 waypoint（§3.3） | ✗ 无（未来帧预测自监督） |
| DriveVLA-W0 | 指令原生 token 与视觉/动作 token 深交错（§3.1） | waypoint→FAST 离散 token 自回归；Action Expert 三解码（回归/AR/流匹配，§3.1/§3.3） | ✗ 无（世界模型自监督） |
| VLDrive | 指令 token 注入 LLM（1.3B），无文本输出（§3.1/§4.1） | 轨迹 L1 回归→PID（§3.6/§3.1） | ✗ 无检测/占用监督；仅训练期自监督 token 重建（§3.6） |
| ICR-Drive（评测） | 只改指令文本、模型不重训（§4.2） | 基线：LMDrive BC 轨迹+控制头 / BEVDriver waypoint（§4.2） | 基线无感知监督（§1 冻结视觉编码器） |
| Language2Drive（评测） | PCU 固定场景只换指令（§2.2） | SISR/CFPS 机器可验证标准（§2.4） | 不涉及（纯评测） |

**规律**：① A=waypoint 是全组共识（控制量仅 DriveGPT4）；文本化 waypoint（DriveVLM/EMMA/OmniDrive）与回归 waypoint（UniAD/VAD/Senna/LMDrive/VLDrive/BEVDriver）两条路在 EMMA §2 有直接对照；survey 进一步分 LH/REG/SEL/GEN 四型动作头（§2.3）。② V 的显式监督随「e2e 纯度」递减：UniAD/VAD 全显式 → LMDrive/Senna 半 → EMMA 基础版/CarLLaVA 零；零监督阵营靠 MLLM 预训练+大数据兜底，另有世界模型自监督流派（GenAD/DriveVLA-W0，无感知头）。③ L 的位置决定架构：L 在输入=指令条件策略（LMDrive/VLDrive/BEVDriver）；L 在输出=决策器（DriveVLM/Senna）；L 全文本化=通才模型（EMMA）；L 仅训练期=蒸馏教师（VLP，推理零开销）。④ **语言输入是把双刃剑**：ICR-Drive 显示措辞微扰即大幅掉分（misleading ΔDS −33.64/−32.50，Table 1），Language2Drive 显示 93.2% 的成功 rollout 换一条等价指令即失败（§1）——用语言控制车辆必须配固定词典 + 成对反事实评测。

## 四、对 robovan L4 方案的启示

1. **A 的选型**：低速场景直接用 LMDrive 式「小头回归 5-10 个 waypoint + 双 PID 跟踪」最省事；不推荐 DriveGPT4 式直接出控制量（精度与频率都受 tokenizer 粒度限制）。若要做文本决策+数值执行的分层，Senna 的 meta-action embedding 注入是最干净的现成方案。
2. **V 的监督策略（对应你方案里「感知是 debug 辅助」）**：三档里最贴合的是 **LMDrive 式两段法**——vision encoder 预训练阶段挂显式感知头（可 debug、可复用 two-stage 标注资产），微调阶段丢头压缩成 406 token（Q-Former M=4）保持端到端；EMMA 自监督路线（ego 轨迹自动标注，无需人工标签）适合做数据兜底与快速冷启动，但黑盒性最强。
3. **L 的位置（对应「遥操新增模式：切 VLA / 切 two-stage」）**：DriveVLM-Dual 的「VLM 低频轨迹 W_slow 作为传统规划器输入、两分支异步运行」是现成的架构证明（§3.5 原文 + 量产车部署）；Senna 证明 meta-action 词典（3×4=12 类）已够用，不必做 17 类或自由文本。语言中间层有量化收益（EMMA +3.0%/+1.5%），且支持 early stopping（§2.2）。
4. **成本账的论文证据**：EMMA 自监督（无人工标签）反超 DriveVLM-Dual（大量人工标注）6.4%（§3.1.2）——直接支撑你方案里「VLA 数据成本可大幅低于 two-stage」的判断；但注意 EMMA 自己承认 "computationally expensive"（§1），推理成本要单独算。

## 五、RL post-training 怎么做（驾驶 VLA + 通用 VLA，原文依据）

### 5.1 驾驶侧范式：AutoVLA（2506.13757，NeurIPS 2025，UCLA）

- 位置：在 SFT 之后。"we propose reinforcement fine-tuning (RFT), utilizing Group Relative Policy Optimization (GRPO) with verifiable planning reward functions"（Abstract）。
- 目标（Eq.3-4）：GRPO 对每个场景采样 G 条输出，组内标准化优势 A_i=(r_i−mean)/std，带 KL 正则："𝒥_GRPO(θ)=E[1/G Σ_i (𝒥_i^R − β D_KL(π_θ‖π_ref))]"，π_ref 是 SFT 检查点。工程简化："We perform a single policy update at each step, allowing the use of a simplified objective without the need for clipping or tracking the old policy"（§4.1）。
- 奖励（§3.4）："r = r_Driving − λ_r r_CoT"——nuPlan 用 **PDMS**（安全/舒适/效率等驾驶质量评分器）；Waymo E2E 用 **ADE**；r_CoT 惩罚过长推理链（换取快/慢思考自适应，省时延）。
- 可训参数与超参（§4.1）：RFT 走 **LoRA**；lr 3e-5；β=0.04；6000 步；骨干 Qwen2.5-VL-3B（动作 token：K-disk 聚类 2048 个 (Δx,Δy,Δθ) 离散 token，每 token 0.5s、共 10 个=5s 地平线，§3.1）。
- 效果（§4.2/Fig.5）：RFT 后 **PDMS +10.6%、推理时长 −66.8%**；Best-of-N（采样 6 条 + oracle 评分器选优）PDMS 92.12。**注：oracle = 用测试期特权真值（地图/他车 3D 框）打分的裁判，Best-of-N 是评估上限而非部署口径；部署口径看 One-shot（80.54→89.11）。**
- 驾驶侧相关 RFT 工作（其 Related Work）：AlphaDrive（GRPO 只做高层 meta-action）、Gen-Drive / TrajHF（RFT 对齐安全约束与人类驾驶偏好）、RAD（3D Gaussian splatting 生成场景 + 闭环 RL 训练）。

### 5.2 通用 VLA 侧全景（TEMPO 2608.07314 §II-B 综述 + TEMPO 方法）

TEMPO 论文 Related Work 对 RL post-training 的原始分类（全部原文）：
- **奖励信号**："VLA-RL uses a learned process reward, RIPT-VLA adopts sparse binary success feedback, and VLA-RFT employs simulator-verified rewards"。
- **交互数据**："ConRFT uses real-robot intervention trajectories, whereas RECAP combines cross-embodiment demonstrations, policy-generated experience, and human corrections"。
- **优化机制**："VLA-RL uses PPO to update the LoRA parameters of an autoregressive OpenVLA policy, whereas SimpleVLA-RL applies GRPO to optimize all parameters of an action-token policy"；"iRe-VLA keeps the vision-language backbone unchanged and updates only the action head during the RL stage, which is then alternated with supervised optimization"。
- **TEMPO 本身**：冻结 VLM backbone（"directly updating a large vision-language backbone... may also disturb its pretrained representations"）；**语义投影层（低频更新）+ 动作专家（高频更新）两个模块级 TD3 回路**，各自 replay buffer/twin critics/target networks，共享稀疏成功奖励 r_t=1/0（§III-B）；"settings with ρ>1 generally outperform ρ=1"（动作侧更新更频，§III-E）。
- 动机（§I）："SFT optimizes action prediction under the demonstration distribution rather than closed-loop success under the policy-induced state distribution"——RL post-training 解决的是分布失配与误差累积。
- 工程栈：NVIDIA IsaacLab 官方 VLA RL post-training 教程：https://isaac-sim.github.io/IsaacLab/develop/source/experimental-features/rlinf_vla_posttraining.html （Isaac GR00T 支持 GR00T N1 的 RL 后训练）。

### 5.3 通用 recipe（综合）

1. SFT/行为克隆起手（warm start，遥操数据即可）；
2. 选可训子集：驾驶多用 LoRA / action head；机器人多用 action expert / 语义投影层；**VLM backbone 一律冻结**（TEMPO §I、AutoVLA §4.1）；
3. 奖励：驾驶 = 可验证评分器（PDMS/ADE/闭环 DS）或人类偏好（TrajHF）；机器人 = 稀疏成功 / 过程奖励 / 仿真验证；
4. 算法：GRPO（组内相对优势 + KL→π_ref）、PPO（OpenVLA LoRA）、模块级 TD3（TEMPO 双回路）；
5. 推理侧配合：Best-of-N + oracle 评分（AutoVLA）；
6. 坑：开环 ADE/PDMS 奖励会被 reward hacking（AutoVLA 用 RFS/闭环 Bench2Drive 交叉验证）；且 RFT 只奖驾驶质量时，**语言可控性不会自动变好**（Language2Drive J.2："Training may allow language to remain unnecessary for action selection"）——要把「指令一致性」显式做成奖励项。

### 5.4 对 L4 robovan 方案的映射

- 遥操数据 SFT 之后接 RFT：车队规则化实现 PDMS 式评分器 + GRPO + LoRA + KL；奖励里加推理长度惩罚省时延；上线前用 Best-of-N。
- 闭环奖励若难拿，先用「开环 ADE + 安全惩罚」起步（AutoVLA Waymo 分支做法），再逐步换闭环；参照 RAD 用 3DGS/仿真生成长尾场景做闭环 RL。

## 六、具体实现蓝图（论文/代码事实 + 工程合成标注）

> 标注规则：`[论文]`=逐字原文依据；`[代码]`=开源仓库文件事实；`[合成]`=基于以上事实的工程拼装建议，非论文原话。

### 6.1 系统拓扑（三个进程）

```
[采样 worker]──G 条候选轨迹──▶[奖励服务]──标量奖励──▶[GRPO 训练器(LoRA)]
      ▲                                                    │
      └────────── 新策略权重（每 N 步回灌）────────────────┘
```
- 采样 worker：离线=遥操/回放数据逐场景组采样 G 条（[论文] AutoVLA §3.4 "we sample a set of G candidate outputs from the old policy"）；在线=仿真器交互。
- 奖励服务：无状态，输入(场景, 轨迹) → 标量。[代码] 开环直接部署 [autonomousvision/navsim](https://github.com/autonomousvision/navsim)（NeurIPS 2024 官方仓库）；闭环用 CARLA DS。
- 训练器：[代码] [verl](https://verl.readthedocs.io/en/v0.5.x/examples/multi_modal_example.html) 已支持 Qwen2.5-VL 多模态 GRPO；或按 AutoVLA 单步更新简化自写 LoRA 循环。

### 6.2 组件一：动作 token 化（[论文] AutoVLA §3.1）

- K-disk 聚类把连续轨迹 (Δx, Δy, Δθ) 量化为 **K=2048** 个动作 token；每 token 0.5s、**10 token = 5s 地平线**；`<action_0>…<action_k>` 扩展进 VLM 词表，推理时用 codebook 解码回轨迹。[代码] 仓库含 `codebook_cache/` 与 `models/`、`scripts/`、`config/`（[ucla-mobility/AutoVLA](https://github.com/ucla-mobility/AutoVLA)）。

### 6.3 组件二：奖励实现（[论文] NAVSIM 2406.15349 §3 原文）

- 公式："PDMS = (∏_{m∈{NC,DAC}} score_m) × (Σ_{w∈{EP,TTC,C}} weight_w × score_w / Σ weight_w)"——**乘法惩罚 × 加权平均**。
- NC（无责碰撞）、DAC（可行驶区域合规）是硬惩罚：违规记 0（静态物碰撞记 0.5），场景 PDMS 直接归零。
- EP（ego 前进，按 PDM-Closed 规则规划器的上界归一）、TTC（4s 内低于阈值记 0）、C（加速度/加加速度超阈值扣分）；权重 **EP=5, TTC=5, C=2**。
- 执行机制（原文）：非反应式——"driving agents are only queried in the initial frame of each scene. Afterwards, the planned trajectory is kept fixed"；"An LQR controller is applied at each simulation iteration... and a kinematic bicycle model propagates the ego vehicle. We execute this pipeline at 10Hz over the 4s trajectory horizon"。
- AutoVLA 用法（§3.4）："r = r_Driving − λ_r r_CoT"——r_Driving=PDMS（nuPlan）或 ADE（Waymo E2E）；r_CoT=推理长度惩罚。
- [合成] 车队版：fork navsim 仓库，把 OpenScene 真值换成自车数据（地图/检测框/ego 轨迹），PDMS 即可作为遥操数据的奖励评分器；先用「ADE+碰撞/越界惩罚」冷启动再换 PDMS。

### 6.4 组件三：GRPO 训练循环（[论文] AutoVLA §3.4/§4.1 + [合成] 伪代码）

- 目标（Eq.3-4，原文）：组内标准化优势 A_i=(r_i−mean)/std，带 clip 的 ratio 项 + KL："𝒥_GRPO(θ)=E[1/G Σ_i (𝒥_i^R − β D_KL(π_θ‖π_ref))]"；π_ref=SFT 检查点。简化："We perform a single policy update at each step, allowing the use of a simplified objective without the need for clipping or tracking the old policy"（§4.1）。
- 超参（§4.1）：LoRA；lr 3e-5；β=0.04；6000 步；组大小越大越好（Fig.5b）。
- 伪代码（[合成]）：

```
ref = load(SFT_checkpoint)          # π_ref，冻结
policy = attach_lora(ref)           # 只训 LoRA
for step in range(6000):
    batch = sample_scenarios(B)     # 场景（图像+指令+ego状态）
    for q in batch:
        O = [policy.rollout(q) for _ in range(G)]      # G 条候选，含推理链+动作token
        R = [reward_server.score(q, decode(o)) for o in O]   # PDMS − λ·len(reasoning)
    A = (R - mean(R)) / std(R)      # 组内标准化优势
    loss = -mean(min(ratio·A, clip(ratio)·A)) + β·KL(policy||ref)
    loss.backward(); optimizer.step()   # 单次更新，无 old policy 跟踪
# 推理侧
best = argmax_{o in N_samples} oracle_score(o)   # Best-of-N（AutoVLA N=6）
```

### 6.5 遥操数据与门禁

- 遥操接管=天然负样本：[论文] TEMPO §II-B "ConRFT uses real-robot intervention trajectories"（[ConRFT 2502.05450](https://arxiv.org/abs/2502.05450)）——把「VLA 行驶→被人工接管」的片段做成低奖励/负样本进 RL 数据，正好匹配你们「遥操数据→VLA」的闭环。
- 门禁一（奖励有效性）：[论文] NAVSIM §4.1——先用 navmini（396 场景）验证你的 PDMS 实现与闭环分数的 Spearman/Pearson 相关性，防止奖励错位（"PDMS is better correlated to the CLS for every planner type"）。
- 门禁二（语言可控性）：若指令参与输入，奖励里显式加「指令一致性」项 + 用 PCU 成对评测验收（Language2Drive/ICR-Drive 的警示，见 §2.7/§5.3）。

### 6.6 GRPO 的 forward / backward / loss 逐行实现（[代码] verl core_algos.py 事实）

- **三个 log-prob 来源**（一次训练步内的三套前向）：
  1. **rollout**：旧策略 π_old 对每个场景采样 G 条 response，记录 `old_log_prob`（形状 BS×L_resp）；
  2. **奖励→优势**（verl `compute_grpo_outcome_advantage` 代码原文）：
     ```python
     scores = token_level_rewards.sum(dim=-1)          # outcome 奖励（标量；该函数注释：only consider outcome supervision）
     # 按 prompt uid 分组：
     scores[i] = (scores[i] - mean(group)) / (std(group) + 1e-6)
     advantages = scores.unsqueeze(-1) * response_mask # 标量优势广播到每个 response token
     ```
  3. **训练前向**：π_θ 重算 `log_prob`（整条 prompt+response），π_ref 重算 `ref_log_prob`（冻结、可缓存）。VLA 的动作 token 与文本 token 共用同一 softmax（AutoVLA 动作 token 进了 VLM 词表），所以动作 token 的 log-prob 直接取同一分布。
- **loss**（verl `compute_policy_loss_*` 代码事实）：
  ```python
  negative_approx_kl = log_prob - old_log_prob
  ratio = torch.exp(negative_approx_kl)
  pg_losses = -advantages * ratio            # 或 clip 版：-min(r, clip(r,1-ε,1+ε)) * A（AutoVLA Eq.4 即 clip 形式）
  # KL 惩罚（verl kl_penalty_forward，三种估计器）：
  #   k1      = logprob - ref_logprob
  #   k2/mse  = 0.5 * (logprob - ref_logprob)**2
  #   k3      = exp(clamp(ref-logprob,±20)) - clamp(ref-logprob,±20) - 1   # 低方差估计，verl 默认 k3
  loss = agg_loss((pg_losses + beta*kl) * response_mask)   # 默认 token-mean：Σmasked / Σmask，prompt 全掩
  ```
- **backward**：
  - 梯度只流向可训子集（AutoVLA=LoRA，§4.1；TEMPO=语义投影层+动作专家）；π_ref、π_old、advantages 全部 no_grad；
  - **AutoVLA 的「单步更新」简化**（§4.1 原文）的含义：每批数据只做一次梯度更新就丢弃 → ratio 初值≈1、clip 与 old-policy 跟踪都不再需要（对应上轮伪代码里无 clip 分支）；若一批数据要复用多次更新，就必须保留 clip+old policy（DeepSeekMath 原版 GRPO 形式）。
  - GRPO 无 critic：基线来自组内均值（[论文] DeepSeekMath 2402.03300 Abstract："GRPO foregoes the critic model, instead estimating the baseline from group scores"），所以没有 value loss、没有 GAE——这是它与 PPO 实现上最大的差别。

---

更新时间: 2026-11-12 13:50
