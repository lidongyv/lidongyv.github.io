# WAM 扩展精读线 B：动作/决策表征（第二梯队）·重发

> 生成：2026-10-27 ｜ 代理：03b 重发轮 ｜ 上一轮同任务代理中途失败无产出，本轮增量落盘。
> 取文链：arxiv.org/abs/<id> → arxiv.org/html/<id>（最新版）→ ar5iv.labs.arxiv.org/html/<id>（截断回退）。
> 铁律：只依据论文原文（英文 verbatim + 小节号）或代码（仓库+文件+行）；找不到标「原文未写（已获取范围）」/「未取到」（标截断位置）；数值逐字抄录（MathJax 渲染重复如 `95.8%95.8\%` 已归一为 `95.8%`）。
> 来源级别约定（沿用 README）：L3=全文核实 / L2=部分正文 / L1=仅摘要。
> 环境限制：shell 出网被沙箱拦截（实测 Invoke-WebRequest 失败：连接关闭），全部网络取证走 web_fetch；HTML 页有截断（截断位置逐篇标注）。

论文清单（10 篇）：MotuBrain 2604.27792 ｜ GigaWorld-Policy 2603.17240 ｜ Kairos 2606.16533 ｜ LAWA 2608.24882 ｜ DreamWAM 2608.04996 ｜ FACT 2608.10232 ｜ THAW-VLA 2609.24682 ｜ Rolling-WAM 2609.30247 ｜ X-WAM 2604.26694 ｜ OpenWAM 2609.07398

---

## 1. MotuBrain — An Advanced World Action Model for Robot Control（arXiv:2604.27792）

### 元信息
- 版本/日期：v5（latest），2026-07-15 提交；v1 2026-04-30。cs.RO。
- 作者：Motubrain Team（含 Chendong Xiang, Fan Bao, Jun Zhu 等 20 人，团队署名）。
- 项目页：https://www.genspi.com/en/motubrain ；GitHub：https://github.com/shengshu-ai/Motubrain
- 一句话：UniDiffuser 联合建模 video+action 的统一 World Action Model，三流 Mixture-of-Transformers，一个模型支持 VLA/世界模型/视频生成/逆动力学/联合预测五种推理模式，云 GPU 推理 11Hz。

### 来源级别
L2（arxiv HTML v5 正文：§1–§2.4.1 完整、§2.4.2 开头一段后被截断）。**截断位置**：§2.4.2「Real-time Inference and Execution」中公式「Let the current action chunk be」处；**未取到**：§2.4.2 后半（RTC 融合细节）、§3 Evaluations 全文（3.1/3.2/3.3 真机数值细节）、§4 Conclusion、§5 Contributors、References。

### 【动作表征】
1. **形式：连续动作 tokens + flow matching**，与视频同构。§2.1："The video and action streams are both trained with flow matching, predicting the velocity fields of video latents and action tokens, respectively."
2. **统一跨本体动作表征 = 相对末端执行器（EEF）chunk**。§2.2："In the second stage, we initialize from the first-stage checkpoint and train only the action branch on heterogeneous-embodiment data, while freezing the video branch. At this stage, we use a unified action representation across embodiments." 公式(2)(3)(4)：E^rel = E^abs ⊖ s，e_i^rel = (p_i−p_s, R_s^−1 R_i, g_i)；"The raw input pose is provided in quaternion format, while the training target uses a 6D rotation representation. **Each end-effector action therefore has dimension 10, consisting of position, rotation, and gripper state. We only normalize the gripper dimension to [−1,1] and keep the remaining dimensions in their original physical scales.**"
3. **chunk/生成机制**：§2.3（Non-AR）："the VAE temporally compresses every τ consecutive video frames into a single latent frame, and the action stream operates at f_va actions per raw video frame. Each latent frame therefore corresponds to S_a = f_va τ action tokens. Consequently, predicting K future video latents amounts to generating K S_a actions, enabling high-frequency control while keeping the visual stream compact."（τ、f_va、K 的具体数值：原文未写（已获取范围））
4. **AR 长程：chunk 级分解 + 块因果注意力**。§2.3："We partition each episode into non-overlapping chunks and process all chunks in parallel during training with a block-causal attention mask."；"Importantly, our AR model does not involve clean action tokens, because doing so would break the unified relative-EEF action representation."
5. **执行侧后处理**：§2.4.1 Action Smoothing："Each chunk is upsampled to twice its original temporal resolution, smoothed with a Savitzky-Golay filter, and then downsampled back"；Frequency-aware interpolation："the action sequence is interpolated according to the ratio between the model action frequency and the low-level control frequency."

### 【决策表征】
1. **决策 = 同分布联合生成未来视频+动作**（五种推理模式，Table 1）：VLA p(a_{t+1:t+k}|o_t,ℓ)、WM p(o_{t+1:t+k}|o_t,a)、IDM p(a|o_{t:t+k})、VGM p(o|o_t,ℓ)、Joint p(o,a|o_t,ℓ)。§2.1："a single training procedure supports inference over five distributions: vision-language-action policy modeling, world modeling, video generation, inverse dynamics, and joint video-action prediction"。
2. **无额外规划器/双系统**。§1（原文）："while retaining the ability to solve long-horizon and dexterous manipulation tasks without relying on an additional VLM planner, a dual-system decomposition, external memory, or retry-specific data."
3. **V2A 式非对称依赖 → 动作-only 推理（决策的部署形态）**。§2.3："Under this attention pattern, action tokens attend to video and language tokens, while video tokens never attend to action tokens... this asymmetric dependency makes it possible to use an action-only suffix during inference: after a short joint denoising prefix, the video stream can be frozen and only the action stream continues to update while attending to the cached visual-language context."；§2.4.1 公式(9)：t<N 联合去噪 Φ_joint，t≥N 冻结视频 latent，仅动作流更新。
4. **AR 部署时的闭环决策流**。§2.3："At deployment, the model rolls out sequentially, using the newly observed frame as clean context for the next chunk."
5. **决策与执行的解耦 + RTC 风格衔接**。§2.4.2："Motubrain therefore decouples the model inference loop from the robot action execution loop: the controller executes the current action chunk at the target control frequency, while the world action model asynchronously generates the next chunk from the latest observation."（RTC 融合细节在截断区内，未取到）

### 【代码依据】
GitHub https://github.com/shengshu-ai/Motubrain（abs/正文首页已链接；本线尚未打开仓库取证——待核实清单）。

### 【why 追问】
- Q1：动作维度 10 = 位置3 + 6D旋转 + 夹爪1，但「f_va actions per raw video frame」与「S_a = f_va τ action tokens」的具体数值未写明——若不拿到 f_va/τ，无法判断 11Hz 下实际执行的是多少步动作（chunk 粒度未知），这正是代码取证的目标。
- Q2：AR 模式「不引入 clean action tokens 以免破坏相对-EEF 表征」——为什么 clean action token 会破坏相对表征？（原文只断言、未给机制论证，§2.3。）
- Q3：动作流 timeshift=1、视频流 timeshift=6（SNR-based 采样）——动作更均匀采样是「把噪声更多地推给视频」的决策，能否量化动作/视频各自承担多少建模容量？

### 【关键数值】（逐字抄录）
- 动作维度 10（位置3+旋转6D+夹爪1）；仅夹爪归一化到 [−1,1]，其余保持物理尺度（§2.2）。
- timeshift：video=6，action=1（§2.2/§2.4.1）；推理步数 50→30（§2.4.1）。
- 推理加速表（Table 2，Non-AR 端到端）：Baseline 50步×95.0ms/步=4.90s（0.20Hz,1.00×）→ +Noise sampling 30步 2.90s（1.69×）→ +torch.compile 0.98s（32.7ms/步，5.00×）→ +FP8 0.88s（5.57×）→ +DiT cache 0.20s（5.00Hz，24.5×）→ +V2A-style 0.09s（**11.11Hz，54.4×**）。
- RoboTwin 2.0：clean 95.8%、randomized 96.1%（§1）；新本体适配仅需 50–100 条同本体轨迹（§1）。
- 预训练细节：多视图辅助视角随机丢弃概率 0.1；条件帧扰动概率 p=0.5，s_aug~U[0.3,0.7]（§2.2 公式(1)）。
- H-bridge 注意力：仅中间 50% 层做全 V-A 联合注意力，底部/顶部各 25% 层解耦（§2.1）。
- 云端 GPU 远程推理，实时推理频率 11Hz；速度提升「essentially lossless」：优化前后平均成功率波动在 sub-percent 内（§2.4.1）。

### 【一页总结】
MotuBrain 把动作表征定义为「相对 EEF 的 10 维连续动作 tokens（位置3+6D旋转+夹爪1，仅夹爪归一化）」，经 flow matching 与视频 latent 同构联合去噪；动作流以 f_va/帧、每视频 latent 对应 S_a=f_va·τ 个动作 token 的方式生成 chunk（具体数值未写）。决策表征 = 五种模式共享的「未来视频+动作联合分布」，没有独立规划器；部署时用 V2A 不对称注意力在短联合前缀后冻结视频流、纯动作续推理，配合 DiT cache/FP8/compile 做到 11Hz、54.4× 加速，RoboTwin2.0 95.8/96.1。

### 【待核实】
- f_va、τ、K（动作 chunk 长度）数值：原文未写（已获取范围）→ 需代码取证（github.com/shengshu-ai/Motubrain）。
- §3 真机实验细节（任务、成功率表）在截断区 → 可补 ar5iv 或 PDF。
- 动作平滑的 Savitzky-Golay 参数（窗长/阶数）原文未写（已获取范围）。
- 「50–100 trajectories」适配的人形本体具体任务未取到（§3 截断）。

---

## 2. GigaWorld-Policy — An Efficient Action-Centered World–Action Model（arXiv:2603.17240）

### 元信息
- 版本/日期：v2（latest），2026-03-21 提交（Comments: Added references）；v1 2026-03-18。cs.CV。CC BY-NC-SA 4.0。
- 作者：GigaAI / GigaWorld Team（24 人，alphabetical：Angen Ye … Zheng Zhu）。
- 项目页：https://gigaai-research.github.io/GigaWorld-Policy/ ；abs 页未列 GitHub 仓库（已获取范围内无代码链接）。
- 一句话：action-centered WAM——训练时用未来视频作密集监督+推理信号、因果掩码防未来帧泄漏进动作；推理时未来视频生成可选（action-only），比 Motus 快 9×、成功率高 7%。

### 来源级别
L2（arxiv HTML v2 正文：Abstract、§1、§2、§3.1、§3.2 完整，§3.3 中段截断）。**截断位置**：§3.3 Training Objective 处「we combine the video and action objectives using scalar weights λ_video」；**未取到**：§3.3 后段（λ 权重、数据集细节）、§3.4 Inference、§4 Experiments（4.1–4.5）、§5 Conclusion、Appendix A/B/C、References。

### 【动作表征】
1. **动作 = action chunk（长度 p）+ 连续 action tokens**。§3.1："Conditioned on these inputs, the policy predicts an action chunk of length p, a_{t:t+p−1}=(a_t,a_{t+1},…,a_{t+p−1})."；§3.2："we embed proprioceptive states and actions into the pre-trained model's hidden dimension via linear projections, yielding state tokens T_s and action tokens T_a, respectively."（连续、线性投影嵌入；**未见离散化/量化**）
2. **动作与「动作潜条件 c_t」联合采样**。§3.1 公式(2)："(a_{t:t+p−1}, c_t) ∼ g_Θ(·|o_t, s_t, l)"，"Here, c_t is an action latent conditioning signal that is used to guide visual forecasting."
3. **未来观测按步长 Δ 稀疏预测，与动作 horizon 对齐**。§3.1 公式(3)+正文："(o_{t+Δ}, o_{t+2Δ}, …, o_{t+KΔ}) ∼ g_Θ(·|o_t,s_t,l,c_t)，where Δ denotes the temporal stride between predicted observations, and K=⌊p/Δ⌋"；§3.2："we predict one future observation every Δ steps along the action horizon"。（p、Δ 的具体数值：原文未写（已获取范围））
4. **位置编码分型**。§3.2："visual tokens adopt a 2D positional encoding over the image grid, whereas proprioceptive state and action tokens use a 1D temporal positional encoding."
5. **训练目标：flow matching（动作与视频同构）**。§3.3 公式(6)(7)(9)：x^(s)=(1−s)ϵ+sx，目标速度 ẋ=x−ϵ；L_action = E_{s,ϵ}[∥g_Θ(a^(s),s|T_s,T_o,T_l) − ȧ^(s)∥²]。
6. **输入形态**：§3.1 "multi-view RGB observations o_t={o_t^v}_{v∈S} from a fixed set of camera viewpoints S={left,front,right}" + 语言指令 l + proprioceptive state s_t；§3.2 三视角合成为单张 composite 图（公式(4)）。

### 【决策表征】
1. **决策 = 两个耦合条件分布的采样**（动作分布 + 动作条件的未来视觉分布）。§3.1："we adopt a world-modeling perspective that learns how visual observations evolve under an executed action chunk. We implement our method as a single unified model g_Θ that parameterizes two complementary conditional distributions"（公式(2)(3)）。
2. **因果掩码决定信息流（核心决策结构）**。§3.2："we pack all modalities into one token sequence T_t=[T_o; T_s; T_a; T_f] and use a causal attention mask to control information flow"；"(i) T_s and T_o may attend to each other, but cannot attend to T_a or T_f; (ii) action tokens in T_a may attend to the tokens {T_s,T_o}, but cannot attend to T_f; (iii) future-video tokens in T_f may attend to {T_s,T_o,T_a}"；"This masking scheme prevents information leakage from predicted future frames into action generation, while allowing future-frame prediction to leverage both observations and actions."
3. **语言不在因果序列内**。§3.2："the language instruction is not included in the unified self-attention sequence; instead, l is provided as an external conditioning signal via cross-attention."
4. **推理时未来视频可选 → 决策开销可剥离**。§1："at inference time, explicit future-video prediction is optional: the model can be executed in an action-only mode that directly produces control commands without rolling out long sequences of video tokens."
5. **动机（为何这样设计决策）**。§1："errors in video prediction can propagate to action decoding... particularly when small early inaccuracies compound over time"；§3.2："since dense frame-by-frame prediction is frequently unnecessary due to strong temporal continuity and redundancy in adjacent observations, we only forecast a sparse set of future frames"。

### 【代码依据】
未提供：abs 页与已获取正文范围内无代码仓库链接（项目页在 GitHub Pages，但未见源码 repo 声明）。待核实：gigaai-research 组织下是否有源码仓库。

### 【why 追问】
- Q1：动作维度/本体接口从未出现——「linear projections」把动作嵌入 hidden dim，但动作的物理维度（EEF 6/7D？关节角？）与归一化方式未写（已获取范围）。这是「动作空间未写明」清单的重要成员；§4/Appendix A 可能在后文（截断区）。
- Q2：p 与 Δ 未给数值：K=⌊p/Δ⌋ 决定未来帧数量，而「0.36s per inference」的加速比依赖 p 的取值——若 p 小，9× 加速的含金量不同。
- Q3：c_t 与 a_t 联合采样后，训练时 c_t 是否以教师强制（ground-truth action 派生）还是以采样值进入视频分支？若以采样值，视频监督对动作梯度的路径与 FACT 的 action-conditioned 因果有何异同？

### 【关键数值】（逐字抄录）
- 骨干：5B-parameter diffusion Transformer，初始化自 Wan 预训练视频模型（§3.2）。
- 具身预训练数据 ≈10,000 hours（§3.3）；Table 1 明细：EgoDex 800、EGO4D 3,500、RoboMind 300、RDT 25、DROID 350、ATARA 10、Agibot 2,500、Open X-Embodiment 3,500、Something-Something V2 200（合计 11,185，与正文「approximately 10,000」不一致，原文如此）。
- 推理：9× 比 Motus 快，**down to 0.36s per inference**（§1）；成功率 +7%；较 π0.5 在 RoboTwin 2.0 上 +95%；「under comparable speed and VLA settings」+20%（§1）。
- 3 个相机视角（left/front/right）合成单张 composite 图（§3.2）。
- 稀疏未来帧：每 Δ 步预测 1 帧（Δ 数值未写）。

### 【一页总结】
GigaWorld-Policy 的动作表征 = 长度 p 的连续 action chunk（线性投影成 tokens、1D 时序位置编码、flow matching 去噪），并额外联合采样一个「动作潜条件 c_t」用于驱动未来视频预测；未来观测按 Δ 步长稀疏预测 K=⌊p/Δ⌋ 帧（p、Δ、动作维度均未写明）。决策表征 = 单序列 [T_o;T_s;T_a;T_f] 上的分块因果注意力：动作只能看当前观测+状态、看不到未来帧，未来帧可以看动作——训练时视频监督正则化动作，推理时可完全去掉视频分支，实现 0.36s/次、9× Motus 加速且成功率 +7%。

### 【待核实】
- 动作空间维度与归一化：原文未写（已获取范围，§4/Appendix 在截断区）→ 若补取后仍无，列入「动作空间未写明清单」。
- p、Δ 数值：原文未写（已获取范围）。
- §4.1 推理速度对比的硬件条件（Fig 1 标注 A100）与 Motus 对齐细节在截断区。
- λ_video/λ_action 权重数值在截断区内。

---

## 3. Kairos — A Regret-Aware Native World-Action Model Stack for Physical AI（arXiv:2606.16533）

### 元信息
- 版本/日期：v3（latest），2026-07-03（报告封面日期 August 24, 2026）；v1 2026-06-15。cs.AI（cs.CV 交叉）。技术报告体量（PDF 37.6MB）。
- 作者：Kairos Team（Fei Wang, Shan You, Qiming Zhang … Dacheng Tao, Xiaogang Wang，23 人）。
- 代码/权重：GitHub https://github.com/kairos-agi/kairos ；HF https://huggingface.co/kairos-agi ；ModelScope kairos-team/kairos30；平台页 kairos.acerobotics.com。
- 一句话：4B 原生跨本体世界-动作模型栈；「control-sufficient state」（控制充分状态）替代全像素未来模拟；Cross-Embodiment Data Curriculum + Hybrid Linear Temporal Attention + Deployment-Aware System Co-Design；regret（后悔/物理代价）为设计准则。

### 来源级别
L2 混合：论文正文仅 Abstract+§1 前半（HTML 截断于 §1 代价函数 J_H 定义处）**截断位置**：§1「We use … J_H(a_{t:t+H−1}|H_t,g) … T(he…」处；§2 Model/§3 Pretraining/§5 Inference/§6 Evaluations/理论附录均**未取到**。代码侧已取证：README + benchmarks/robotwin + benchmarks/libero_plus + 两个 dataset_stats.json（一手）。

### 【动作表征】
1. **论文侧（已获取范围）**：§1 形式化中动作空间为抽象集合——"let A denote the action space and let A^H denote the space of length-H future action sequences"；具体动作接口（维度/格式/是否 chunk）所在小节（预计 §2.1.3/§3.4.2/§5.4）**未取到**（截断于 §1）。Abstract 定性："joint world-action prediction"、"predicts future visual states and executable robot actions"（README）。
2. **代码侧（一手，仓库 kairos-agi/kairos）**：
   - benchmarks/robotwin/robotwin_dataset_stats.json：`action.default.stepwise_min/max` 为 **32 步 × 14 维**（每步 14 维 = 双臂 7+7；首行 min/max 值范围可见位置量纲 ±7.5m 级与旋转 ±8.8rad 级、第 7/14 维 gripper∈[0,1]）。计数方法：文件 action 段内 `stepwise_min` 与 `stepwise_q01` 各行首元素行数 66 = 2×32 + global_min/global_q01 各 1；`stepwise_max` 33 = 32 + global_max 1。同文件 `"num_episodes": 95262, "num_transition": 21121454`。
   - benchmarks/libero_plus/libero_plus_dataset_stats.json：`action.default.stepwise_min` 为 **32 步 × 7 维**（每步 7 维；范围逐字：dims1-3∈[−0.9375, 0.9375]，dim4∈[−0.2582142949104309, 0.3557142913341522]，dims5-6∈[−0.375, 0.375]，dim7∈[−1.0, 1.0]）。同文件 `"num_episodes": 14347, "num_transition": 2238036`。
   - 生成机制：README §6.5 "Kairos training uses Flow Matching"；默认配方 "samples videos to 832x480, 81 frames, and 16 FPS"；条件模式 t2v/ti2v/i2v/null2v。benchmarks/robotwin/README：`NUM_INFERENCE_STEPS` 默认 **30**（WAM 采样步数）；`EVAL_NUM_EPISODES` 100；`CHUNK_SIZE` 默认 5（**每 chunk 的 episode 数，系评测分批参数，非动作 horizon**——防误读）。
   - benchmarks/robotwin/README 声明："当前开源配置在 RoboTwin 上的成功率为 **clean: 96.9%**、**random: 95.2%**"（代码侧，非论文正文数值）。

### 【决策表征】
1. **决策对象 = control-sufficient state**。§1（原文 verbatim）："the objective of an embodied world model is not to reproduce all future sensory information, but to learn a compact internal state that is sufficient for predicting task-relevant future variables. We refer to this internal representation as a control-sufficient state."；形式化："given the observation–action history H_t, task goal g, and candidate future action sequence a_{t:t+H−1}, an embodied world model should maintain an internal state Z_t that preserves the information needed to predict task-relevant future states, failure events, task progress, physical cost, and the discrepancy between imagined and real outcomes."（Z_t = f(H_t)，f: H→Z 压缩函数）
2. **决策评价 = regret（条件期望物理代价）**。§1："we use J_H(a_{t:t+H−1} | H_t, g) to denote the conditional expected physical cost of executing a candidate future action sequence a_{t:t+H−1} ∈ A^H under the current history H_t and goal g."（后续 regret 定义/界限在截断区，未取到）；Abstract："a robot does not pay a cost when its predicted pixels are slightly blurry; it pays a cost when a cup slips, a collision occurs, a task fails, a human must intervene, or a safety boundary is crossed."（§1 同义段）
3. **状态内容清单**（Abstract）："should learn and maintain the information most relevant to embodiment control: object state, spatial relations, contact conditions, task progress, action consequences, failure boundaries, and deployment uncertainty."
4. **决策基础设施（代码侧）**：README——"Kairos maintains persistent world states, where understanding provides causal interpretation, generation unfolds plausible futures, and prediction produces executable robot actions"；Hybrid Linear Temporal Memory = Sliding-Window + Dilated Sliding-Window + Gated Linear Attention（local/mid-range/global 三通路）；Deployment-Aware：efficient kernels/quantization/token streaming。

### 【代码依据】
- kairos-agi/kairos：README.md（模型规格/训练/部署）；benchmarks/robotwin/README.md + robotwin_dataset_stats.json（动作 32×14 维、96.9/95.2）；benchmarks/libero_plus/README.md + libero_plus_dataset_stats.json（动作 32×7 维）。上述均为 raw.githubusercontent.com 一手抓取（首次即成功，未用回退）。
- 注意：动作「语义」未在代码侧明示（14 维 = 双臂各 7 维的拆分、7 维 = 3 平移+3 旋转+夹爪？）——从数值范围仅可推断量纲，**未获代码注释/文档直接写明**，列入待核实。

### 【why 追问】
- Q1：Kairos 论文在 §1 就宣称「不必模拟所有像素」，但其动作接口仍采用 RoboTwin/LIBERO 原生连续 EEF 接口（32 步×14/7 维 chunk，代码侧）——「control-sufficient state」与「原样动作接口」之间谁在承担 regret 感知（3.4.1 Regret Alignment Training 内容在截断区，未取到）？
- Q2：32 步 chunk 与 81 帧/16FPS 视频配方（≈5.06s 视频）的时间对齐关系未写明：动作 32 步对应多少秒、多少视频 latent 帧？README 只给视频侧采样。
- Q3：LIBERO-Plus 动作 dim4∈[−0.258,0.356] 明显不是对称范围——该维是什么物理量（夹爪？旋转残差？），JSON 无语义标注，只能从 eval 客户端代码确认（未取到）。

### 【关键数值】（逐字抄录）
- 模型 4B 参数（README）；Kairos 4B：Memory 23.5GB，Complexity **2.3 PFLOPs**，A800 上 TI2V 720P/5s：1GPU **43.3s**、4GPU **9.5s**；对照 Cosmos 2.5 14B 2526.0s/687.2s、Wan 2.2 5B 201.0s/85.0s、Lingbot 28B 5525.0s/1436.0s（README §5.2.2）。
- 480P 实时推理（distilled kairos-4B-robot）：NV-A800 23.5GB，1GPU 11.7s/4GPU 3.0s；NV-RTX5090 13.9GB，11.4s/5.7s（README §5.2.1）。
- 训练配方：832×480、81 帧、16FPS、Flow Matching（README §6.5）；WAM 采样 30 步、每任务 100 episodes（benchmarks/robotwin）。
- 数据：RoboTwin 统计文件 95,262 episodes / 21,121,454 transitions；LIBERO-Plus 14,347 episodes / 2,238,036 transitions（JSON 原文）。
- 长期生成：§8.1.4 标题 "Multi-Timescale Memory Beyond **15-Second** Generation"（暗示当前生成 horizon ≈15s，正文未取到）。

### 【一页总结】
Kairos 的决策表征 = 由历史 H_t 压缩出的「control-sufficient state」Z_t（保留物体状态/空间关系/接触/任务进度/动作后果/失败边界/部署不确定性的信息），决策质量以 regret（候选动作序列 a_{t:t+H−1} 的条件期望物理代价 J_H）衡量；动作表征在论文侧 §1 中只是抽象集合 A^H（具体接口在截断区），代码侧实锤为 RoboTwin 2.0 **32 步×14 维**、LIBERO-Plus **32 步×7 维**的连续动作 chunk（flow matching 生成、30 推理步、数据集统计归一化）。全栈强调部署：4B/2.3 PFLOPs，A800 单卡 TI2V 720P/5s 43.3s（4 卡 9.5s），蒸馏版 480P 可达 11.7s。

### 【待核实】
- §2.1.3 Joint World-Action Modeling、§3.4.2、§5.4 Inference Modes 中动作接口/生成机制的论文原文（截断未取到）→ 可补取 PDF 或等 ar5iv。
- 14 维/7 维动作的逐维语义（代码无注释明示）。
- RoboTwin/LIBERO 动作 chunk 32 步与视频 81 帧的时间对齐。
- §3.4.1 Regret Alignment Training 的具体损失与数据来源。

---

## 4. LAWA — Latent Action as Intention Enables Efficient Future Imagination for World Action Models（arXiv:2608.24882）

### 元信息
- 版本/日期：v2（latest），2026-09-01；v1 2026-08-25。cs.RO。
- 作者：Xiang Li, Yupeng Zheng（corresponding/Project Leader）, Songen Gu, Huailiang Ma, Feng Yu, Yuhang Zheng, Xian Nie, Shanshuai Yuan, Yujie Zang, Weize Li, Shuai Tian, Moyang Liu, Ya-Qin Zhang, Wenchao Ding（corresponding）。机构：CollegeAI THU / AIR THU / CASIA / TARS Robotics / FDU / Southeast University / SJTU / NUS / BUAA。
- 项目页：https://getterupper.github.io/LAWA ；"Code and models will be released"（Abstract）。
- 一句话：WAM 三范式对比（Fast-WAM / Joint-WAM / LAWA）——用「离散 tokenizer 产生的紧凑潜动作」作为未来意图的操作化表征，测试时保留未来想象但去掉未来视频生成。

### 来源级别
L2（arxiv HTML v2：Abstract、§1、§2、§3 全部、§4.1-§4.2 开头获全；**截断位置**：§4.2 Table 2 中 Fast-WAM 行处）。未取到：§4.2 后段（LIBERO-Plus 数值表其余行）、§4.3-4.5（paradigm 对比/消融/真机）、§5 Conclusion、Appendix。

### 【动作表征】
1. **可执行动作 = action chunk**。§3："We consider a policy that maps the current observation o_t and language instruction c to an action chunk a_{t+1:t+H}."（chunk 长度 H、动作维度、归一化：**原文未写（已获取范围）**）
2. **潜动作 = 离散 codebook tokenizer 的产物**（ViPRA 基础上）。§3.1："the tokenizer maps an observation sequence o_{t:t+H} to discrete latent action tokens l_{t+1:t+H} that capture dynamic transitions without using action labels."；编码：DINOv2 提取 patch tokens → 非因果 transformer（factorized spatial/temporal attention）→ F_k∈R^{N×d}；"We encode the transition by differencing F_k and F_{k−1}, and spatially compress the result into L tokens h_k ∈ R^{L×d_q}. Each compressed token is assigned to its nearest entry in a learnable codebook C={e_j}_{j=1}^K."（L、K、d_q 数值未写）
3. **训练中代码是离散 target，测试时连续松弛**。§3.2："The tokenizer targets are discrete codebook embeddings, but the latent expert performs flow matching in their continuous embedding space. **We do not apply nearest-neighbor projection during or after denoising**; the action expert conditions on the evolving continuous latent state. Thus, discrete describes the tokenizer's target set, whereas test-time latent imagination is a continuous relaxation anchored to that set."
4. **tokenizer 监督（动作-视觉对齐机制）**：因果前向解码器把 l_k 作额外空间 token 与 o_{k−1} patch 拼接重建 o_k（§3.1）；SAM 风格 mask 解码器以潜动作为 prompt 预测手/机械臂 mask（目标由 SAM 2 自动生成）；§3.3 损失 L_tok = L_1 + λ_perc L_perc + λ_mask L_mask（mask 用 BCE+Dice+IoU）；"we adopt noise-substitution quantization to avoid additional codebook loss terms. Low-usage entries are periodically refreshed to prevent codebook collapse."
5. **动作流生成**：§3.3 三模态 flow matching（vid/lat/act 独立噪声与 shifted flow time τ_m），y^m_{τ_m}=(1−τ_m)y^m+τ_m ϵ^m，v^m=ϵ^m−y^m；L_LAWA = λ_vid L_vid + λ_lat L_lat + λ_act L_act。
6. **无动作标注预训练**（动作表征的数据来源）：§3.1 action-free egocentric pre-training——per-source frame sampling 对齐运动速度分布；加权重采样使机器人视频约占预期训练样本 **20%**。

### 【决策表征】
1. **决策对象 = 「未来意图」= 潜动作序列**。§1："our central idea is to treat a temporally structured sequence of latent actions as future intentions. Here, intention is an operational term for the predicted sequence of transition targets available to the action expert."
2. **结构化注意力掩码（三专家信息流）**。§3.2（原文）："current-observation tokens z_t cannot access the future, whereas noisy future-video tokens attend to all video tokens. Latent action tokens attend only to the current observation and latent action sequence; action tokens attend to the current observation, latent action sequence, and action sequence. This mask lets the action expert use the evolving future intention without leaking future-video information."；推理时："At inference, future-video tokens are omitted and the current-observation features are cached, so iterative denoising evaluates only the latent action and action experts. The same visibility mask is maintained across training and inference."
3. **决策的部署形态**：§1（Abstract）："LAWA jointly denoises a continuous latent state anchored to these targets with executable action chunks while omitting the future-video branch at inference."
4. **动机链（为何弃 RGB 未来而留潜动作）**：§1："video reconstruction can favor static appearance over the small interaction dynamics that determine manipulation success"；§3 对比 Fast-WAM 局限："At inference, Fast-WAM encodes the current observation once and omits video denoising, gaining speed but losing an explicit representation of how the task should progress."

### 【代码依据】
未发布：Abstract 明言 "Code and models will be released"（截至 2026-09-01 v2 无仓库；abs 页无代码链接）。项目页 https://getterupper.github.io/LAWA 未取到。

### 【why 追问】
- Q1：「discrete tokenizer targets + continuous flow-matching relaxation」是 LAWA 的核心设计——如果训练目标就是连续空间（codebook embedding 空间），「离散」的实际作用只剩 codebook 作为紧凑 anchor 的熵约束；为什么不直接对连续 transition embedding 做 flow matching（去掉 codebook）？原文在已获取范围没有消融「codebook 有无」。
- Q2：动作 chunk 维度与长度 H 全程未写明（已获取范围）——LAWA 的「意图」是 l_{t+1:t+H} 与动作 chunk 同长？潜动作每步 L 个 token、动作每步 1 个向量，两者时间分辨率是否一致？
- Q3：mask 预测只监督「手/机械臂区域」，但意图需要编码「物体状态变化」——原文 Table 6 消融（截断区）应能回答 mask 监督是否足以让潜动作保留物体侧信息，待补。

### 【关键数值】（逐字抄录）
- RoboCasa（24 桌面任务；full = 24,000 轨迹=每任务 1,000；few-shot = 10% 子集=每任务 100 demo；每任务 50 trials）：LAWA few-shot **65.6%** / full **80.8%**；matched Fast-WAM 56.0/76.3（+9.6/+4.5 pts）；matched Joint-WAM 64.1/78.8；对照 DIAL 58.3/70.2、Being-H0.7 49.2、DiT4DiT 50.8、LDA-1B 55.4、GR00T N1.6 47.6（Table 1）。
- LIBERO-Plus 零样本（原文）：LAWA 总 **74.4%**；matched Fast-WAM 60.0%（+14.4 pts）；matched Joint-WAM 70.4%；OpenVLA-OFT 69.6%；OpenVLA 15.6%；camera-viewpoint 增益 44.3 pts、sensor-noise 增益 27.5 pts（§4.2 正文）。
- 延迟：比 matched Joint-WAM 低 **42.9%**（Abstract/§4）。
- 真机：以 full-data Fast-WAM 的 **25%** 演示量即超越（Abstract）。
- 预训练：机器人视频 ≈20% 期望训练样本（§3.1）；tokenizer 目标 mask 由 SAM 2 自动生成（§3.1）。

### 【一页总结】
LAWA 把「未来想象」从 RGB 观测空间搬进紧凑潜动作空间：离散 tokenizer（DINOv2+差分+codebook，动作无关视频预训练）把视觉 transition 压缩成潜动作 token 序列（=未来意图），策略训练时视频/潜动作/动作三专家联合 flow matching，结构化掩码让动作专家只能从「演进的潜意图」借力、绝不接触未来视频 token；推理时砍掉未来视频分支，只联合去噪潜意图+动作 chunk。结果：RoboCasa few-shot 65.6%（超 Fast-WAM 9.6 pts、与 Joint-WAM 持平），推理延迟比 Joint-WAM 低 42.9%。动作 chunk 维度/长度 H 与 codebook 大小 K 原文未写（已获取范围）。

### 【待核实】
- 动作 chunk 维度、长度 H：原文未写（已获取范围）→ 等代码发布（"will be released"）。
- Table 2 其余行、§4.3-4.5（paradigm 消融、真机数值、Table 6）在截断区。
- codebook 大小 K、每步 token 数 L、d_q 数值未写。
- 与扩展 C 线遗留数据交叉核对：README 全局待核实清单中「WLA RoboTwin 训练协议（2500 clean + 25k randomized demos、100k 步、100 trials/任务；WLA-0 RoboTwin2.0 Clean 92.94%）」——**注意**：该处「WLA」是否即本 LAWA 尚待与 C 线确认，本线未在一手材料中见到这些数值，先不采信。

---

## 5. DreamWAM — Beyond RGB Future Prediction for World Action Models（arXiv:2608.04996）

### 元信息
- 版本/日期：v1（唯一版），2026-08-05。cs.RO。
- 作者：Shanglin Yuan, Weiheng Zhao, Xin Shi, Haoyi Jiang, Xianda Guo, Liu Liu, Wenyu Liu, Wei Sui, Xinggang Wang（HUST / D-Robotics / Wuhan University / Horizon Robotics）。
- 代码：https://github.com/hustvl/DreamWAM（Abstract："The code and models are publicly released"）。
- 一句话：把 WAM 的「未来预测」从 RGB 扩展为四视图结构化未来（appearance/motion/geometry/semantics），训练时 RGB+光流联合潜去噪、几何/语义走 gated residual 支路；部署仍 RGB-only，与 Fast-WAM 接口一致。

### 来源级别
L2（arxiv HTML v1：Abstract、Intro、Related Work、Method 全、实验设置与 Table 1 全；**截断位置**：Table 2（LIBERO-Plus）WAM 行起始处）。未取到：§Results 正文、§Ablation、§Conclusion、Appendix A/B（target 构造细节、真机协议）。

### 【动作表征】
1. **动作 = action chunk a_{1:N}，ActionDiT 用 flow matching 去噪**。Method Preliminaries："Given a language instruction c, a current observation o_0, future observations o_{1:H}, and an action chunk a_{1:N}, a WAM jointly learns future prediction and action generation."；"the ActionDiT denoises the action trajectory"。
2. **动作策略数值规格**。§Experimental Setup（原文）："All matched variants share the Wan2.2-5B VideoDiT, ActionDiT, training data, and **a 32-step action policy that executes 10 steps before replanning with 10 denoising steps**."
3. **动作维度：原文未写（已获取范围）**——未出现动作维度/本体接口表述（DreamWAM 基于 Fast-WAM，LIBERO 语境，但本文 HTML 已获取部分确实未写明动作维数）。
4. **部署形态（动作生成路径不变）**。§Training Objective and Inference："In the main configuration, the VideoDiT denoises the RGB latent while the ActionDiT denoises the action chunk through shared attention. The motion input is inactive; the residual branches, feature-prediction heads, RAFT, Depth Anything V3, and DINOv2 are removed... The final control output is the denoised action chunk, with no auxiliary representation prediction or online teacher inference."
5. **训练损失中的动作项**：L = λ_rgb L_rgb^FM + λ_mot L_mot^FM + λ_geo L_geo^pred + λ_sem L_sem^pred + λ_act L_act^FM + λ_g L_gate（公式(4)；λ 数值未写）。光流通道推理时填零："the optical-flow channels are filled with zeros and are therefore inactive"（Appendix A.2 标题同）。

### 【决策表征】
1. **未来状态 = 四视图结构化表示（决策的依据对象）**。Method 公式(1)："DreamWAM represents the future as Y_{1:H} = {z^rgb_{1:H}, z^mot_{1:H}, f^geo_{1:H}, f^sem_{1:H}}, where the four terms describe appearance, motion, geometry, and semantics, respectively. These views provide complementary descriptions of future state changes rather than a perfectly disentangled or complete physical state."
2. **四视图的目标来源（逐字）**：Appearance=Wan2.2 video VAE 编码 RGB；Motion=RAFT 估计相邻未来帧稠密光流、同一 VAE 编码到 RGB latent 网格（"without claiming metric 3D scene flow"）；Geometry=Depth Anything V3 特征、与 VideoDiT 网格时空对齐；Semantics=DINOv2 patch 特征（"preserves object- and region-level consistency rather than serving as a semantic segmentation label"）。
3. **异质建模分两条通路（核心决策架构）**：RGB+motion 同 timestep 联合去噪、沿 channel 拼接（公式(2)）；geometry+semantics 用 gated residual 支路："h_ℓ = h̄_ℓ + Σ_j g^j_ℓ(h̄_ℓ, c) ⊙ R^j_ℓ(h̄_ℓ), f̂^j = P_j(h_L)"（公式(3)），理由（原文）："Forcing the backbone to model them through the same state can therefore interfere with the video-generation prior inherited from pretraining."；L_gate 为 gate 激活的 ℓ1 惩罚。
4. **世界-动作耦合**。§World-Action Coupling："The VideoDiT models the future-state representation, while the ActionDiT denoises the action trajectory. The experts retain separate parameters but exchange information through shared attention at paired layers."；"The additional future views are not optimized as isolated auxiliary tasks; they reshape the VideoDiT representation that participates in action denoising."
5. **动机**。Intro："an RGB-only prediction objective entangles task-relevant state transitions—such as object displacement, contact, and goal completion—with variations in texture, illumination, background, and viewpoint that need not change the required action."

### 【代码依据】
github.com/hustvl/DreamWAM 已声明开放（Abstract）。本线尚未打开仓库取证（动作维度待查 config）——待核实清单。

### 【why 追问】
- Q1：光流用「同一 VAE 编码」与 RGB 同网格联合去噪——RAFT 流是 2 通道矢量场，VAE 通道拼接后 RGB 与流的 latent 分布如何对齐？原文只说「same dense spatiotemporal structure」，未给归一化/缩放细节（附录 A.1 在截断区）。
- Q2：动作 chunk 32 步、执行 10 步后重规划——为什么 world 侧未来 horizon H（帧数）与动作 32 步的对齐关系未写明？H 的数值在已获取范围未出现。
- Q3：gated residual 支路只在「selected VideoDiT layers」生效，哪些层被选中（表/数值）在截断区；该选择与 HBridge（MotuBrain 中间 50%）的「层位置决定模态耦合强度」思路是否一致，可作横向追问。

### 【关键数值】（逐字抄录）
- LIBERO：4 个 suite ×10 任务、每 suite 500 demo、每任务 50 rollouts（每 seed 2,000）；LIBERO-Plus 每 seed 10,030 episodes、7 种扰动（camera viewpoint/robot appearance/language/lighting/background/image noise/object layout）。
- LIBERO 平均成功率（Table 1）：OpenVLA 76.50、π0 94.15、π0.5 96.85、Motus 97.70、LingBot-VA 98.45、Fast-WAM 97.30、**DreamWAM-uncond 98.40**、Fast-WAM-Joint 98.00、**DreamWAM 98.90**。
- LIBERO-Plus（Abstract）：no-rollout 51.36→63.44、joint 69.16→75.47（"improvements across all seven shifts in both settings"）。
- 真机（AgileX PiPER 双臂，每任务/扰动 30 trials）：标准 4 任务平均 90.8→96.7；不可见扰动（光照/背景/布局）74.4 vs Fast-WAM-Joint 55.6（Abstract/§Setup）。
- 训练：8×NVIDIA H20、bfloat16、lr 1×10⁻⁵、batch 16（§Setup）；Wan2.2-5B VideoDiT。
- 动作策略：32 步 chunk、执行 10 步、10 去噪步（§Setup）。

### 【一页总结】
DreamWAM 不改动作接口（Fast-WAM 的 VideoDiT-ActionDiT 共享注意力 + 32 步 chunk/执行 10 步/10 去噪步），改的是「未来想象的对象」：把决策依据的未来状态写成四视图 Y={RGB latent, 光流 latent, 几何特征, 语义特征}，RGB+光流同 timestep 联合 flow matching 去噪，几何/语义经 gated residual 支路（ℓ1 gate 正则）只塑形、不替换视频骨干；所有 beyond-RGB 监督仅训练时存在，部署 RGB-only 输出动作 chunk。收益集中在分布外：LIBERO-Plus 七扰动全升（no-rollout 51.36→63.44、joint 69.16→75.47），真机不可见扰动 74.4% vs 55.6%。

### 【待核实】
- 动作维度（原文未写，已获取范围）→ 代码 github.com/hustvl/DreamWAM 取证。
- 未来 horizon H、光流归一化、gated 支路层选择、λ 权重（截断区/未写）。
- Table 2 其余行与消融表（截断区）。

---

## 6. FACT — Failure-Aware Causal Training for World-Action Models（arXiv:2608.10232）

### 元信息
- 版本/日期：v2（latest），2026-10-13；v1 2026-08-10。cs.RO（cs.AI/cs.LG 交叉）。CC BY-NC-ND 4.0。
- 作者：Quanquan Peng*, Yutong Liang*, Rui Yan, Nicklas Hansen, Xiaolong Wang（UCSD）。项目页 https://fact-wam.github.io 。abs 页无代码仓库链接。
- 一句话：因果 WAM——先出动作、再以「执行过的动作」为条件预测未来视频与任务进度值；失败 rollout 掩掉动作模仿损失、只监督后果，把坏动作变成有效的未来 target；推理可选用进度值给候选动作打分。

### 来源级别
L2（arxiv HTML v2：Abstract、§1、§2、§3 全部含实现细节、§4 设置与 Table 1 前半；**截断位置**：Table 1（RoboTwin）Motus 行之后）。未取到：Table 1 余下（FACT 自身数值）、§4.1 正文、§4.2 真机、§4.3 消融、§5-6、Appendix A-H。

### 【动作表征】
1. **动作 = 连续 action chunk，明确维度框架**。§3.1："The goal is to choose an action chunk a_{t:t+H} ∈ R^{H×d_a} that advances the task specified by ℓ."；**H=48 写明**。§3.3 Implementation details："The action chunk length is H=48 and λ_fail=1."（**d_a 动作维数：原文未写（已获取范围）**；本体为双臂——主视角+左右腕三视图，RoboTwin 语境）
2. **观测组成**。§3.1："o_t = (I^main_t, I^wristL_t, I^wristR_t, s_t), where the images are multi-view RGB observations and s_t ∈ R^{d_s} is the robot proprioceptive state."（d_s 未写）
3. **动作 token 的生成机制：flow matching + 动作适配器**。§3.3 公式(7)：z_τ^x = (1−τ)z_0^x + τ z_1^x，预测速度场 u_θ^x；§3.2："We attach a lightweight action adapter to robot tokens after the feed-forward network in each transformer block, giving the model action-specific capacity while preserving the pretrained video pathway."
4. **两阶段推理（动作解码流程）**。§3.3："Stage 1 denoises [P_state, P_ref, A_noisy] for K_denoise flow-Euler steps and returns a clean action chunk â_{t:t+H}."；"Unless otherwise specified, inference uses **20 flow-Euler denoising steps**."；action-only 模式："If only an action is required, FACT stops here and skips world prediction."（prefix KV-cache 复用）
5. **动作训练的数据分离**。§3.3：成功演示监督 L_a；失败 rollout 掩掉动作模仿项（公式(9) L_Df = w_v L_v + w_I L_I）——"failure data teaches the model the consequences and lower action-conditioned progress of failed actions without making those actions policy targets."

### 【决策表征】
1. **决策对象 = 动作条件的（未来视频 + 任务进度值）**。§3.1 公式(2)："p_θ(o'_{t:t+K}, v_t(a_{t:t+H}) | o_t, ℓ, a_{t:t+H})，where o'_{t:t+K} denotes future-video observations over the prediction horizon and v_t(a_{t:t+H}) ∈ [0,1] is the predicted task progress after executing action chunk a_{t:t+H}."
2. **进度值定义（决策量化）**。§3.1 公式(3)：p_t = G_t/G_T ∈ [0,1]（G_t = Σ r_k，uniform progress reward 下 p_t = t/T）；"p_t=0 denotes the beginning of the task and p_t=1 denotes completion."
3. **teacher-forced 动作条件（因果分离核心）**。§3.2："we introduce a clean ground-truth action condition a^gt_{t:t+H} for the world branch"；token 序列 z = [P_ref ‖ A_pred ‖ G_gt ‖ V_value ‖ I_future]（公式(5)）；"During training, value and future-video tokens attend to the clean ground-truth action G rather than the noisy predicted action A, while A cannot see G."
4. **失败感知值目标**。§3.3 公式(6)：v_t = p_{t+H}（成功）或 clip(p_{t+H} − λ_fail·1_fail(t+H), 0, 1)（失败），λ_fail=1。
5. **可选候选打分（决策接口）**。§3.3 公式(10)：Stage 1 并行采样 N 个候选 a^(k)，Stage 2 预测进度分 v̂^(k)=V_θ(o_t,ℓ,a^(k))，执行 a* = argmax_k v̂^(k)；"This selection rule uses the value head to score the future implied by each candidate action, without training a separate critic."
6. **世界监督直接塑造动作（共享骨干）**。§3.2："this shared backbone lets future-outcome losses affect action generation instead of being confined to a separate world expert, as in Mixture-of-Transformers (MoT) designs."

### 【代码依据】
未提供：abs 页与已获取正文无代码链接（仅项目页 fact-wam.github.io）。待核实项目页是否带仓库。

### 【why 追问】
- Q1：FACT 的价值目标是「归一化任务进度 p_t=t/T」，但 progress reward r_k「uniform」意味着它不区分「正确的慢」与「错误的绕路」——进度值能否真正区分「坏动作」依赖 r_k 的定义（原文只说 uniform、未给任务级 reward 构造，附录可能没有）。
- Q2：λ_fail=1 使失败动作的进度目标直接减 1 后 clip——这是「失败惩罚」还是「取消进度的错误归因」？clip 后与成功轨迹低进度段不可分，值头如何避免把「刚起步」与「失败」混为一谈？
- Q3：动作维度 d_a 与 proprio d_s 未写（已获取范围）——RoboTwin 双臂通常 14 维，但需代码/附录实锤；候选打分 N 的数值也未见。

### 【关键数值】（逐字抄录）
- 动作 chunk H=48；λ_fail=1；未来视频监督 = 当前帧 + 4 个未来 offset [0, H/4, H/2, 3H/4, H]（§3.3）。
- 推理默认 20 flow-Euler 步（§3.3）；骨干 WAN2.2-5B；lr：action FFN 2×10⁻⁴、WAN backbone 2×10⁻⁵；损失权重 w_a=20、w_v=w_I=1（§3.3）。
- RoboTwin（Table 1 已取部分）：π0 65.9/58.4（avg 62.2）、X-VLA 72.9/72.8（72.9）、π0.5 82.7/76.8（79.8）、Gigaworld-Policy 87.0/85.0（86.0）、Motus 88.7/87.0（87.8）；FACT 自身行未取到（截断）。
- 值头输出 [0,1]；值条件 = 执行后的任务进度。

### 【一页总结】
FACT 的动作表征：H=48 的连续动作 chunk（flow matching、20 步 Euler、动作 FFN 适配器挂在 WAN2.2-5B 每层 FFN 后），维度未写；决策表征：token 序 [P‖A_pred‖G_gt‖V_value‖I_future] 上 teacher-forced 因果——价值与未来视频只条件于「干净的真实动作」G，训练时成功数据三损失全开、失败数据掩掉动作模仿只留价值+视频监督（失败进度目标减 λ_fail 后 clip 到 [0,1]）；推理两阶段（先出动作，可选再出后果），价值头可对 N 个采样候选打分取 argmax——把「失败 rollout」转化为动作条件的后果监督，并声称可减少「坏动作下的成功偏置未来幻觉」（消融在截断区）。

### 【待核实】
- 动作维度 d_a、proprio 维度 d_s、候选数 N：原文未写（已获取范围）。
- FACT 自身 RoboTwin/真机数值（Table 1 截断后部分）。
- §4.3 消融（"Failure data reduces future hallucination"、"Value-guided candidate scoring"）在截断区。
- 进度 reward r_k 的构造细节（已获取范围未写）。

---

## 7. THAW-VLA — Think Like a World Model, Act Like a VLA（arXiv:2609.24682）

### 元信息
- 版本/日期：v3（latest），2026-10-09；v1 2026-09-21。cs.RO（cs.CV 交叉）。
- 作者：Trung Dao, Sankalp Yamsani, Jaden Park, Joohyung Kim, Yong Jae Lee（UW-Madison / UIUC）。项目页 https://thaw-vla.trung-dt.com/ 。abs 页无代码链接。
- 一句话：把「世界模型的内在特征」蒸馏进紧凑 VLA 策略——训练时冻结世界模型跑一遍训练帧缓存特征，学生只加一个 cosine 对齐项（λ=0.5）；部署图与未蒸馏基线完全一致（0.8B、32ms、1.86GB、RTX 5090）。

### 来源级别
L2（arxiv HTML v3：Abstract、§I、§II、§III 全部、§IV-A、§IV-B 开头；**截断位置**：Table I（LIBERO）OpenVLA-OFT 行处）。未取到：Table I 余下、§IV-C/IV-D/IV-E/IV-F（真机与消融表）、§V、References。

### 【动作表征】
1. **动作 = K 步 chunk，GR00T 式 flow-matching 动作专家**。§III-A："The student is a QwenGR00T policy: a Qwen3-VL-family vision-language backbone followed by a GR00T-style flow-matching action expert."；"The action expert predicts a chunk of K future actions a_{t:t+K} by velocity regression along a flow path."（K 数值：原文未写（已获取范围））
2. **动作学习目标**（公式(1)）：L_act = E_{o_t,τ,ϵ} ∥v_θ(a^τ, τ, h(o_t)) − (ϵ − a_{t:t+K})∥²，a^τ = τϵ + (1−τ)a_{t:t+K}；**全程只用演示动作监督**："Note that L_act is supervised by ground-truth demonstration actions throughout; no teacher action targets are used. We deliberately exclude the action channel: teacher and student do not share an action parameterization."
3. **基准动作接口（RoboCasa-GR1）**。§IV-A："24 task environments of the GR1 Fourier humanoid (single ego camera, **29-D bimanual action, 58-D state**)"。LIBERO 动作维度未写。
4. **推理规格**。§III-A："At inference the student runs one backbone prefill and **four flow steps**, and all distillation modules are dropped, so latency is identical to the undistilled student by construction."
5. **观测输入**。§III-A："observations o_t (one or multiple camera views, a language instruction, and proprioceptive state)"，骨干输出 h(o_t)∈R^{L×D_s}（L 个 prefix token）。

### 【决策表征】
1. **核心命题：决策鲁棒性来自「表征」而非「生成未来」**。§I（原文）："What a world model knows about physical scenes lives in its internal features; generating the future is merely the objective that produced them, so the grounding can be inherited while the generative machinery is left behind."
2. **对齐机制（公式(2)(3)）**：对每相机视图 c 的图像 token span 做 mean-pool 得 f^S_c(o_t)∈R^{D_s}；两层 MLP 投影器 p_φ: R^{D_s}→R^{D_t}；L_align = 1 − cos(p_φ(f^S_c), sg[f^T_c])，总损失 L_act + λ_align L_align，**λ_align = 0.5 everywhere**；对齐「方向」而非逐点复制："the student is asked to agree with those targets in direction rather than reproduce them exactly... the two backbones need not share a feature space or a dimensionality."
3. **教师 = Cosmos3-Nano 的理解塔**。§III-C："We extract its Qwen3-VL-8B reasoner from the unified checkpoint, dropping the other towers, and get the image-token hidden states at layer 24, mean-pooled per view, giving D_t=4096 per view. The tower is deterministic: there is no diffusion timestep or sampling." 备选教师：Fast-WAM、V-JEPA2-AC（§IV-E 消融）。
4. **离线缓存（决策蒸馏的成本结构）**。§III-B："The teacher is run once, ahead of training; targets are written to a memory-mapped cache keyed (trajectory id, base index) order. Hence, no teacher weights are loaded during training... one cache serves multiple students... about an hour on four GPUs for LIBERO."
5. **部署等价性声明**。§I："the deployed policy is architecturally identical to the undistilled baseline, down to the number of flow steps"；"every gain is attributable to the representation rather than to added capacity or test-time compute."

### 【代码依据】
未提供：abs 页与已获取正文无代码仓库（仅项目页）。待核实项目页。

### 【why 追问】
- Q1：教师特征取「layer 24 的 image-token hidden states」并 mean-pool 到每视图 1 个 4096 维向量——视图级 pool 丢弃了空间结构，为什么「空间无关的全局表征」仍能传递 grounded 信息？原文未解释层/池化选择的依据（是否扫过层位置/池化方式，消融在截断区）。
- Q2：学生只对齐图像 token span、不对齐语言/proprio token——世界模型的「物理 grounding」是否主要位于视觉通路，原文未论证。
- Q3：32ms/1.86GB 是 0.8B Qwen3.5-VL 的部署指标，但 LIBERO 97.9% 已接近 π0.5 水平（96.9 于其他论文口径/本表 94.3-94.2）——「蒸馏带来的 +2.6」在什么数据量下饱和（§IV-E 截断区）。

### 【关键数值】（逐字抄录）
- 学生 0.8B（Qwen3.5-VL backbone）；LIBERO 平均 **97.9%**（+2.6 于同结构未蒸馏学生）；RoboCasa-GR1 **48.2%→50.5%**，距同结构 44B 模型 4.3 点；真机单臂 pick-and-place **28/30** 追平 4B π 风格策略（参数为其 1/5）。
- 部署：**32 ms**、**1.86 GB**（RTX 5090，32GB 卡）；对照 π0.5 65ms（同卡）；DreamZero 3s/inference（H100）、45.9GB。
- 训练：4×A100-80GB、DeepSpeed ZeRO-2、effective batch 256；教师缓存生成 LIBERO ≈1 小时/4 GPU；λ_align=0.5；教师 D_t=4096、layer 24。
- 数据：LIBERO ~273k 演示步、双视角、每任务 50 episodes（每 suite 500 trials）；RoboCasa-GR1 24 任务×1000 demo、24k episodes、6.02M steps、每环境 20 episodes。
- 评测协议：4 次评估（2 seeds × 2 GPUs：A100-80GB 与 RTX 5090）报告均值±std（§IV-A 明确写因非确定性）。
- 动作：GR1 29-D 双臂动作、58-D state（§IV-A）。

### 【一页总结】
THAW-VLA 不改动作表征（QwenGR00T 的 K 步 chunk + flow matching、4 流步推理），改的是学生表征的「grounding 来源」：冻结世界模型（Cosmos3-Nano 理解塔，layer-24 特征、每视图 mean-pool、4096 维）对训练帧离线跑一次成缓存，训练只加 λ=0.5 的视图级 cosine 对齐项；教师不进显存、投影器训后丢弃、部署图与基线逐位一致。收益：0.8B 学生在 LIBERO 97.9%（+2.6）、RoboCasa-GR1 50.5%（+2.3），真机 28/30 追平 4B 级策略；动作维度只在 RoboCasa-GR1 写明 29-D 双臂，LIBERO 动作维度未写。

### 【待核实】
- LIBERO 动作维度、K（chunk 长度）数值：原文未写（已获取范围）。
- Table I 其余行（WAM 对比）、§IV-E 消融（学生规模/骨干/对齐层/教师族切换的具体数值）在截断区。
- 代码仓库是否发布。

---

## 8. Rolling-WAM — World Action Models with Rolling Imagination（arXiv:2609.30247）

### 元信息
- 版本/日期：v2（latest），2026-10-05；v1 2026-09-24。cs.RO（cs.AI/cs.CV 交叉）。CC BY 4.0。Comments："10 pages, 7 figures, 5 tables. Under review."
- 作者：Yinghua Zhou, Junjie Ye, Yiqi Zhao, Hao Dong, Celina Shiyu Wang, Ruohai Ge, Tingyi Yang, Basile Van Hoorick, Gaurav Sukhatme, Vitor Guizilini, Yue Wang（USC / Brown / Fudan / Toyota Research Institute）。
- 项目页：https://rolling-wam.github.io/ 。abs 页无代码链接。
- 一句话：把联合 video-action 去噪「摊到」连续重规划周期上——滑动窗口内各 chunk 保持错峰噪声水平，每周期只把最近 chunk 完全去噪，远端 chunk 渐进细化；稳态每周期只需 N/W 步。

### 来源级别
L2（arxiv HTML v2：Abstract、§I、§II、§III 全部、§IV-A、§IV-B 全、Table I 开头；**截断位置**：Table I（LIBERO）LingBot-VA 行处）。未取到：Table I 余下（Rolling-WAM 自身 LIBERO 行）、Table II（RoboTwin）、§IV-C/IV-D/IV-E/IV-F 数值正文、§V。

### 【动作表征】
1. **动作 = 长度 H 的动作序列，按 chunk 切分**。§III-A/§III-B："It predicts an action sequence a_{t:t+H−1} over a horizon of H steps."；"Each action chunk a^(j) = a_{t+(j−1)K : t+jK−1} contains K actions... The total prediction horizon is therefore H=WK."；**K=16、W=5、N=10**（§IV-B："Unless otherwise specified, we use N=10 denoising steps per chunk, a window of W=5 chunks, and K=16 actions per chunk."）
2. **真机动作接口（Unitree G1）**。§IV-A3（原文）："The policy receives a 320×224 egocentric RGB image and a 43-dimensional robot state. **It predicts 78-dimensional actions comprising a 64-dimensional SONIC motion latent and 7-dimensional commands for each hand.**"；执行 10Hz。
3. **动作专家规格**。§IV-B："The action expert has 30 layers and hidden dimension d_a=1024 (approximately 1B parameters). Its backbone is initialized by interpolating the video expert's weights."（视频专家 = Wan2.2-TI2V-5B）
4. **动作生成机制：flow matching + 分 chunk 噪声**。§III-D 公式(6)(7)：X̃_j = (1−σ_j)X_j + σ_j ϵ_j（视频/动作独立高斯噪声）；目标速度 ϵ_j − X_j；每 chunk 损失 ℓ_j^q = ∥[f_θ^q(X̃_{1:W}, σ; c_t)]_j − (ϵ_j^q − X_j^q)∥²₂；§III-C Euler 更新（公式(5)）；shifted noise schedule σ(τ)=ρτ/[1+(ρ−1)τ]，**ρ=5**。
5. **动作注意力结构**。§III-D："every action chunk [can] attend to visual features across the entire prediction window. Direct action-to-action attention is restricted to the same chunk, and video tokens do not attend to action tokens."
6. **执行循环**。§III-C："Once the first chunk is clean, the robot executes its K actions. The window then slides over. It removes the executed chunk, retains the remaining predictions, and appends a new chunk initialized with Gaussian noise."

### 【决策表征】
1. **决策对象 = 联合分布 p_θ(a_{t:t+H−1}, v_{t+1:t+H} | c_t)，c_t=(o_t, s_t, ℓ)**（§III-A 公式(1)）；v_{t+1:t+H} 为未来视频 latent（按视频采样率覆盖同一物理区间）。
2. **rolling 噪声调度（决策的时间结构）**。§III-B 公式(2)(3)：σ_j^rolling(τ) = σ((j−1+τ)/W)——每个周期 chunk j 从 σ(j/W) 降到 σ((j−1)/W)，且 σ_{j+1}^rolling(0) = σ_j^rolling(1)（保留 chunk 恰好处在新位置的起点噪声）；初始化模式公式(4)：σ_j^init(τ) = σ(min{1, τ+(j−1)/W})。
3. **渐进决策的直觉表述**。§I（原文）："This resembles the intuition behind human planning: near-term actions are concrete, while more distant plans remain provisional and are refined as new observations arrive."
4. **训练对齐推理**。§III-D："we sample τ∼U(0,1) and select rolling or initialization mode with probabilities β and 1−β"（β=0.8，§IV-B："Initialization and rolling modes are sampled with probabilities 0.2 and 0.8"）；activity mask b_j 排除初始化模式中保持纯噪声的 chunk；损失 L = E[(1/W)Σ_j b_j w(σ_j)(λ_v ℓ_j^v + λ_a ℓ_j^a)]，λ_v=λ_a=1。
5. **与 Fast-WAM 类「去未来」路线的立场差异**。§I："Some methods achieve faster inference by omitting future-video generation at test time, at the cost of losing explicit visual predictions that could inform action generation."；§III-C："This approach is orthogonal to other efficiency techniques for WAMs and can be combined to further reduce inference latency."

### 【代码依据】
未提供：abs 页与已获取正文无代码仓库链接（仅项目页）。待核实项目页/后续发布。

### 【why 追问】
- Q1：稳态每周期 N/W=2 步（N=10, W=5），但总 window 计算量 = 每步要过 5 个 chunk——「4.5× 加速」是每周期步数口径，单步前向 FLOPs 反而变大（窗口更长）；原文声称「assuming fixed N and sufficient GPU parallelism to process the extended prediction window」——那么 4.5× 的墙钟收益成立条件（批处理/并行）在截断的 IV-D 中，需核实硬件口径。
- Q2：K=16 动作/ chunk 与视频 chunk 的采样率对齐关系（视频 latent 每 chunk 几帧）未在已获取范围写明；H=WK=80 步动作对应多少物理秒（LIBERO 20Hz 语境 vs G1 10Hz 语境）？
- Q3：动作专家跨 chunk 无直接注意力（仅同 chunk），远端动作只能经视频特征间接获取「上一 chunk 动作」信息——若视频预测出错，远端动作的连贯性是否受损（消融 IV-F 在截断区）。

### 【关键数值】（逐字抄录）
- LIBERO：4 suite×10 任务×500 demo；10 epochs、batch 128、每任务 50 rollouts；Abstract：平均 **98.1%**。
- RoboTwin 2.0：50 双臂任务、**2,500 clean + 25,000 randomized** demos、5 epochs、batch 1024、每任务每 setting 100 rollouts；Abstract：**93.3%**。
- G1 真机：3 任务（Doll Placement/Plate Stacking/Bead Pouring）、每任务 50 demo @10Hz、7,500 训练步、batch 192、每任务 20 trials；**78 维动作（64 维 SONIC motion latent + 每手 7 维命令）**、43 维状态、320×224 ego RGB。
- 效率：**4.5× steady-state replanning speedup**（Abstract）；N=10、W=5、K=16 → 稳态每周期 N/W=**2** 步（§IV-B）；CFG scale 1。
- 训练：AdamW lr 10⁻⁴、weight decay 10⁻²、5% linear warmup + cosine、BF16；β=0.8/0.2；ρ=5；λ_v=λ_a=1。
- 跨线交叉（README 全局待核实清单，扩展 C 线项目页来源，本线未复核）：G1 真机 85.0%、延迟 215 vs 978/548/296/285 ms——与本线无关性声明：**未在本线一手材料中验证**。

### 【一页总结】
Rolling-WAM 的动作表征：H=W×K 步动作 chunk 流（K=16 动作/ chunk、W=5 chunk 窗口、N=10 步/chunk），G1 真机为 78 维（64 维 SONIC 潜运动 + 每手 7 维），flow matching 生成、动作专家 30 层 d_a=1024（≈1B）。决策表征：联合 video-action 分布上的「滚动想象」——窗口内 chunk 按 σ((j−1+τ)/W) 错峰去噪，每周期只把最近 chunk 降噪到底、远端保留部分噪声并随新观测继续演进（初始化模式补齐 episode 开头）；训练在相同噪声剖面上做（rolling 概率 0.8），动作 chunk 可见全窗口视频但不可见其他 chunk 动作。稳态每周期 2 步，4.5× 重规划加速，LIBERO 98.1 / RoboTwin 93.3。

### 【待核实】
- Table I/II 中 Rolling-WAM 自身行与 IV-D 延迟表、IV-E 真机成功率、IV-F 消融（截断区）。
- 视频 chunk 采样率与动作 K=16 的时间对齐（已获取范围未写）。
- 4.5× 墙钟口径的硬件条件（IV-D 截断）。
- 代码是否发布。

---

## 9. X-WAM — Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising（arXiv:2604.26694）

### 元信息
- 版本/日期：v2（latest），2026-05-07；v1 2026-04-29。cs.RO（cs.AI/cs.CV 交叉）。
- 作者：Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, Huaping Liu（Tsinghua / Xiaomi Robotics / PKU / CASIA）。项目页 https://sharinka0715.github.io/X-WAM/ 。abs 页无代码链接。
- 一句话：统一 4D 世界-动作模型——同一框架内做未来多视角 RGB-D 视频生成 + 3D 重建 + 动作执行；深度支路 = 复制预训练 DiT 最后几块 + 单向注意力；ANS 异步噪声采样让动作少步速解、视频满步生成。

### 来源级别
L2（arxiv HTML v2：Abstract、§1、§2、§3.1-§3.4 大部；**截断位置**：§3.4 末「we employ the UniPC multistep scheduler…」句中）。未取到：§3.4 尾、§4 Experiments、§5、Appendix A-E（其中 **B.2「State and action representation」整节在截断区**）。

### 【动作表征】
1. **动作 = K 步未来动作，与视频/状态异步频率**。§3.1："X-WAM takes a language instruction c, the initial proprioceptive state s_0, and multi-view initial RGB observations O_0 as conditions, and jointly predicts future RGB videos O_{1:H}, depth videos D_{1:H}, proprioceptive states s_{1:H}, and actions a_{1:K}"; "given 1 conditioning RGB frame and 1 initial state, X-WAM predicts **H=8** future RGB frames, H=8 future states, and **K=32** future actions... the actions are uniformly distributed across the same time span at **K/H=4× the video frame rate**."
2. **动作的嵌入/解码：对称 MLP 投影**。§3.1："proprioceptive states and robot actions are projected into the latent space via learnable MLPs: z_s=MLP_s(s) and z_a=MLP_a(a). After denoising, the clean latents are decoded back to the original physical spaces via symmetric MLPs: ŝ=MLP_s^dec(z_s) and â=MLP_a^dec(z_a)."
3. **统一序列表示**。§3.1 公式(1)：Z = [z_{O0}, z_{O1:H}, z_{s0}, z_{s1:H}, z_{a1:K}]，"processed with bidirectional full attention"；初始观测/状态固定 t=0（clean）；状态与动作用与视频 token 相同的时间 RoPE，加可学习视角嵌入。
4. **跨数据集的统一动作接口**。§3.4（原文）："We define a universal interface to represent robot states and actions across heterogeneous datasets. **The state is defined as the poses and gripper positions of a dual-arm end-effector, while the action is defined as the corresponding changes in end-effector poses and gripper positions. For single-arm robots, we treat the single arm as the left arm and do not supervise the right-arm output.**"（动作维度数值：原文未写（已获取范围）；附录 B.2「State and action representation」在截断区，待补）
5. **生成机制：flow matching + ANS 异步去噪**。§3.4：flow matching 预测速度场 v=ϵ−z^0，L_m=∥f_θ^m(z_m^{t_m}, t_m) − (ϵ_m − z_m^0)∥²（t_s = t_a）；§3.3："We allocate T_a denoising steps for proprioceptive states and actions, and T_O denoising steps for video (T_a<T_O)... After T_a forward passes, the noise-free actions are obtained and can be immediately dispatched to the downstream robot for execution... the remaining T_O−T_a steps are continued, during which the actions serve as a clean modality... the inference process naturally becomes an action-conditioned world model."（T_a/T_O 数值未写（已获取范围））
6. **训练期耦合噪声采样**（公式(4)）：(t_O, t_a) ~ {t_a=0, t_O~U(0,1) w.p. p} ∪ {t_a~U(0,1), t_O = t_a + (1−t_a)·b, b~Beta(1.5,1) w.p. 1−p}（p 数值未写）。

### 【决策表征】
1. **决策目标：四合一统一框架**。§1："X-WAM (Figure 1), a unified 4D World Action Model that simultaneously targets four objectives within a single architecture: high-fidelity video generation, 3D spatial reconstruction, high policy success rate, and efficient action execution."
2. **深度支路：结构复制 + 单向注意力**。§3.2："given a model with N DiT blocks, we replicate the final M blocks (M<N) to construct an auxiliary depth prediction branch."；interleaved 执行公式(3)："DepthBlock_j attends to the main branch's input Z_m^{(j−1)} at the same layer via cross-attention, while the main branch remains unaffected by depth tokens. We term this asymmetric connectivity unilateral attention: the depth branch can read from the main branch, but not vice versa, thereby strictly preserving the integrity of the pretrained weights."；深度以 inverse depth MSE 监督；深度图单通道复制 3 次成伪 RGB 经同一 VAE。
3. **腕部相机位姿的推导式决策结构**。§3.1："X-WAM predicts the end-effector pose T_ee ∈ SE(3) and derives the wrist camera pose via the fixed hand-to-eye calibration matrix T_h2e: T_wrist = T_ee · T_h2e."
4. **ANS 动机**。§3.3："independently sampling each modality's timestep introduces training configurations that never arise at test time (e.g., t_O<t_a), degrading training efficiency."；Beta(1.5,1) "rescaled to [t_a,1], biases t_O toward higher noise levels, reflecting the fact that video typically requires more denoising steps than actions."
5. **深度建模的路线论证**。§1："concatenating depth maps as additional tokens along the sequence dimension doubles the sequence length, incurring quadratic growth in attention cost; alternatively, fusing depth along the channel dimension shifts the input distribution away from the pretrained manifold, substantially increasing learning difficulty."

### 【代码依据】
未提供：abs 页与已获取正文无代码仓库链接（仅项目页）。待核实。

### 【why 追问】
- Q1：动作 K=32 与视频 H=8「K/H=4×」、状态与视频帧对齐——但附录 B.2 明确要写「State and action representation」的归一化/维度细节，恰在截断区；双臂 14 维或含 SE(3) 流形处理方式待补。
- Q2：深度支路「推理时可灵活开关、不必每步参与」——开关策略（每几步算一次深度）的数值协议未取到（§3.2 只说可以 toggle）。
- Q3：ANS 的 p（动作-free 视频生成分支的采样概率）与 Beta(1.5,1) 形状共同决定训练分布与推理分布的对齐度——p 的取值影响「纯世界模型模式」占比，未取到。

### 【关键数值】（逐字抄录）
- 预测规格：H=8 未来 RGB 帧 + H=8 状态 + K=32 动作；动作频率 = 视频帧率 ×4（§3.1）。
- 预训练：**over 5,800 hours** 机器人数据（真实+仿真、多任务）（Abstract/§3.4）；骨干 Wan2.2-TI2V-5B（§3.1）。
- 策略成绩：RoboCasa **79.2%**、RoboTwin 2.0 **90.7%** 平均成功率（Abstract）。
- 深度分支：复制 DiT 最后 M 块（M<N；M 数值未写）；inverse depth MSE。
- 调度：UniPC multistep scheduler（Wan2.2 推荐），视频与状态/动作分 scheduler 实例（截断处）。
- 真机：耳麦包装（earphone packing）实验（§1 贡献 3；细节在 Appendix D，未取到）。

### 【一页总结】
X-WAM 的动作表征：K=32 步动作 chunk（4× 视频帧率，H=8 帧），动作与状态经对称 MLP 进出 DiT 潜空间、统一序列 [z_O0, z_O1:H, z_s0, z_s1:H, z_a1:K] 双向全注意力联合去噪；统一跨数据集接口 = 双臂 EEF 位姿+夹爪的「变化量」（维度/归一化在截断附录）。决策表征：四目标统一（视频生成+3D 重建+策略+高效执行）；深度支路用「复制最后 M 块 + unilateral attention（深度读主支路、主支路不受扰）」保证 3D 空间意识不伤预训练先验；ANS 让动作 T_a 步先解出并立即执行、视频继续 T_O 步完成（此后推理自然退化为动作条件世界模型），训练期从联合分布采样 t_O≥t_a 对齐推理分布。RoboCasa 79.2 / RoboTwin 2.0 90.7。

### 【待核实】
- 动作/状态维度与归一化：附录 B.2「State and action representation」在截断区，**原文未写（已获取范围）**。
- T_a、T_O、p、M、λ 权重数值：未取到。
- §4 实验细节与 §A 算法（截断区）；代码是否发布。

---

## 10. OpenWAM — An Open, Modular Exploration Towards Systematic World–Action Model Pretraining（arXiv:2609.07398）

### 元信息
- 版本/日期：v1（唯一版），2026-09-07。cs.RO。CC BY 4.0。
- 作者：Yuran Wang, Siqiao Huang, … Shanghang Zhang, Hao Dong, Lin Shao, Hang Zhao（24 人，北大系合作）。
- 代码/权重：GitHub https://github.com/OpenWAM-Official/OpenWAM ；HF https://huggingface.co/OpenWAM ；项目页 https://openwam-official.github.io/ （abs Comments 原文）。
- 一句话：把 WAM 预训练变成受控实验项目的开放研究栈——OpenWAM-Infra（可组合模块）+ OpenWAM-Study（三条设计原则）+ OpenWAM-α（按原则扩规模的 6,400h 预训练模型）。
- 取文特殊性：**该论文无 arXiv 原生 HTML 版**（abs 页仅有 PDF + TeX Source 链接）→ 按回退链使用 **ar5iv HTML**（成功）。

### 来源级别
L2（ar5iv HTML：Abstract、§1、§2、§3.1、§3.2 前部；**截断位置**：§3.2 Training Formulation 处「one joint forward pass of the architecture (v̂_z, v̂_a) = v_θ(z^{t_v}, a^{t_a}, t_v, t_a, c)」）。未取到：§3.2 后半、§3.3-3.4、§4 Study 正文、§5 OpenWAM-α 细节（5.1 架构/训练/部署、5.2 数据、5.3/5.4 结果）、§6、Appendix B/C/D。

### 【动作表征】
1. **动作空间：统一 80-D**。§1（原文）："Pretrained on 518M frames (≈6,400 hours) of egocentric human and robot data through a **unified 80-D action space**, OpenWAM-α demonstrates that the recipe distilled from controlled settings remains effective when scaled across heterogeneous data, domains, and embodiments."（80 维的具体构成/归一化：原文未写（已获取范围，§5.1/Appendix B 截断））
2. **动作 = action chunk a_{1:H}**。§3.2 样本契约："Define a sample as (ℓ, o_{1:T}, a_{1:H}, q, m), a language instruction, a video window, an action chunk, an optional proprioceptive state, and a per-dimension validity mask."（m 为逐维有效性掩码——多数据集动作对齐的关键机制）
3. **动作流的去噪结构**。§3.2："Each stream is noised to its own timestep, t_v for video and t_a for actions... a^{t_a} = t_a·a + (1−t_a)ϵ_a"（t=0 纯噪声、t=1 clean，与多数论文方向相反，原文如此）。
4. **动作流的注意力：双向（chunk 内互见）**。§3.1："M_{A←A} adopts bidirectional attention, in which the noisy action tokens of a chunk are mutually visible so that the predicted actions inform one another."（视频流为 first-frame causal；动作流与视频流之间的交叉掩码有 4 种模式，见下）
5. **动作骨干两种形态**。§3.1："For the action backbone, OpenWAM-Infra offers two options: a separate set of parameters residing in ActionDiT, or a shared video backbone in which action tokens join the video token sequence and are processed jointly."

### 【决策表征】
1. **决策架构 = 可组合三件套 C(E, S, M)**。§3.1：视觉编码器 E（冻结；Wan2.2-VAE / FLUX.2-VAE / DINOv3 / V-JEPA 2.1 + 可选 S-VAE）；流骨干 S（world W + action A + 可选 understanding U；视频骨干五选：Wan2.1-VACE-1.3B、Cosmos-Predict2.5-2B、Cosmos3-Edge-4B、Wan2.2-TI2V-5B、Wan2.1-I2V-14B；VLM=Qwen3-VL 族）；可见性注意力掩码 M。
2. **世界-动作信息流的四种显式模式（决策表征的核心枚举）**。§3.1：M = [[M_{V←V}, M_{V←A}], [M_{A←V}, M_{A←A}]]，"the two cross-modality blocks then define the four attention mask modes...: **mutual** enables both...; **action-sees-video** enables only M_{A←V}...; **video-sees-action** enables only M_{V←A}...; and **isolated** disables both, denoising the two modalities independently."
3. **三族架构**。§3.1：Single-System（Vanilla/MoE，动作 token 注入共享序列；代表 Cosmos Policy/DreamZero）、Dual-System（Joint self-attention / Joint cross-attention / **IDM**——"the video trajectory is generated first and the actions are predicted from it"；代表 Fast-WAM/LingBot-VA）、Tri-System（+只读 VLM 理解流；代表 Motus）。
4. **三条设计原则（Study 结论，决策设计的纲领）**。§1（原文）："upstream knowledge transfers through a sufficiently capable generative backbone and a compact, information-rich latent space; world–action synergy requires dedicated action capacity, explicit world-to-action information flow, and synchronized joint denoising; and embodied pretraining principally improves out-of-domain generalization, with one-stage co-training over egocentric and robot data integrating world coverage and action grounding."
5. **联合训练目标（截断处）**：视频/动作各自 timestep、一次联合前向 (v̂_z, v̂_a) = v_θ(z^{t_v}, a^{t_a}, t_v, t_a, c) 的 flow-matching 目标（完整损失公式未取到）。

### 【代码依据】
仓库已发布：https://github.com/OpenWAM-Official/OpenWAM（abs Comments 明示；本线未打开仓库取证，80-D 动作空间的维度构成可在仓库中核实）。

### 【why 追问】
- Q1：80-D 统一动作空间如何容纳「单臂 7D / 双臂 14D / 灵巧手高维」——逐维有效性掩码 m 是否意味着「80 维是全集、未用维度掩掉」？这一机制直接决定 OpenWAM 的跨本体动作表征哲学（原文 §5.1/附录截断，待代码取证）。
- Q2：t=0 噪声、t=1 clean 的约定与主流（t=1 噪声）相反，只是记号还是影响与预训练模型兼容（需与 Wan/Cosmos 原生约定对齐）？
- Q3：四种掩码模式的对照实验中「action-sees-video」是否在所有场景胜出（§4.2.2 结论未取到）——若只在部分场景胜出，「原则二」的普适性边界是什么？

### 【关键数值】（逐字抄录）
- 动作空间：统一 **80-D**（§1）；OpenWAM-α 预训练 **518M frames ≈6,400 hours** 自我中心人类+机器人数据（§1）。
- 8 个仿真基准、5 类本体（单臂/双臂/灵巧手等）；真机：单臂+双臂+灵巧手（§1/§5.4 标题）。
- 可选视频骨干参数规模梯度：1.3B → 2B → 4B → 5B → 14B（Wan2.1-VACE-1.3B / Cosmos-Predict2.5-2B / Cosmos3-Edge-4B / Wan2.2-TI2V-5B / Wan2.1-I2V-14B，§3.1）。
- 掩码模式：4 种（mutual/action-sees-video/video-sees-action/isolated）；intra 掩码固定：视频 first-frame causal、动作 bidirectional（§3.1）。
- 每个仿真基准的具体分数表（§5.3.1、Appendix D）：**未取到**（截断）。

### 【一页总结】
OpenWAM 把 WAM 的动作/决策表征「参数化」为受控变量：动作 = a_{1:H} chunk（统一 80-D 空间、逐维有效性掩码、双向 chunk 内注意力、独立 t_a 去噪、ActionDiT 或共享骨干两种实现）；决策 = 组合 C(E,S,M) 显式枚举的架构族（单/双/三系统）与四种世界-动作可见性掩码（mutual / action-sees-video / video-sees-action / isolated）。Study 提炼三原则：①足够强的生成骨干+紧凑信息丰富的潜空间承载上游知识；②协同需要专用动作容量+显式 world→action 信息流+同步联合去噪；③具身预训练主要提升 OOD 泛化，自我中心+机器人数据一站式共训最优。OpenWAM-α 按此配方 518M 帧预训练。§5/附录细节截断，80-D 构成待代码取证。

### 【待核实】
- 80-D 动作空间的维度构成/归一化（§5.1/Appendix B 截断）→ GitHub 仓库取证。
- §4.2.2 掩码模式对照结论、§4.2.3 推理期信息流、§5.3 基准分数表（截断）。
- 完整 flow-matching 损失（§3.2 截断处公式）。

---

## 汇总对比表（10 篇）

| # | 论文 | 动作空间（维度） | 动作生成机制 | chunk 规格 | 决策表征核心 | 来源级别 | 截断位置 |
|---|---|---|---|---|---|---|---|
| 1 | MotuBrain 2604.27792 | **10-D**（相对 EEF：位置3+6D旋转+夹爪1；仅夹爪归一化 [−1,1]）§2.2 | flow matching 连续 action tokens；视频/动作 timeshift 6/1 | AR chunk 级分解；S_a=f_va·τ（f_va/τ/K 数值未写）；S-G 平滑 | UniDiffuser 五模式联合分布；V2A 非对称→动作-only 推理；无额外规划器 | L2 | §2.4.2 公式处 |
| 2 | GigaWorld-Policy 2603.17240 | **未写**（3 视角合成图输入） | flow matching；action tokens+潜条件 c_t 联合采样 | 长度 p（数值未写）；未来帧每 Δ 步 1 帧 | 单序列 [T_o;T_s;T_a;T_f] 分块因果掩码：动作不可见未来帧 | L2 | §3.3 λ 公式处 |
| 3 | Kairos 2606.16533 | 论文侧**未取到**；代码实锤 **RoboTwin 32步×14维 / LIBERO-Plus 32步×7维** | flow matching；WAM 采样 30 步 | 32 步（代码 JSON 行数） | control-sufficient state Z_t=f(H_t)；regret J_H；Hybrid Linear Temporal Attention | L2 混合（正文仅 §1；代码一手） | §1 J_H 定义处 |
| 4 | LAWA 2608.24882 | **未写**（动作维度、chunk H、codebook K 均未写） | 离散 codebook target 的连续松弛 flow matching；推理无视频分支 | 与潜意图序列同长（数值未写） | 潜动作序列=「未来意图」；三专家结构化掩码；latent-only 推理 | L2 | §4.2 Table 2 Fast-WAM 行 |
| 5 | DreamWAM 2608.04996 | **未写** | ActionDiT flow matching；10 去噪步 | **32 步** chunk、执行 10 步重规划 | 四视图未来 Y={RGB,光流,几何,语义}；gated residual 支路；VideoDiT-ActionDiT 共享注意力 | L2 | Table 2（LIBERO-Plus）WAM 行 |
| 6 | FACT 2608.10232 | **未写**（d_a、d_s 未写） | flow matching；20 Euler 步；动作 FFN adapter | **H=48**；未来帧=当前+4 offsets | 因果序列 [P‖A‖G‖V‖I]；值头 v∈[0,1] 任务进度；失败数据掩动作损失；可选候选打分 argmax | L2 | Table 1（RoboTwin）Motus 行 |
| 7 | THAW-VLA 2609.24682 | LIBERO **未写**；RoboCasa-GR1 **29-D** 双臂动作/58-D state | GR00T 式 flow matching；4 流步推理 | K（数值未写） | 表征蒸馏：学生视图级 pooled 特征 cos 对齐冻结教师缓存（λ=0.5，Cosmos3-Nano layer-24/4096-D） | L2 | Table I OpenVLA-OFT 行 |
| 8 | Rolling-WAM 2609.30247 | G1 **78-D**（64 维 SONIC 潜运动+每手 7 维）；LIBERO/RoboTwin **未写** | flow matching + 分 chunk 噪声（σ(τ)=ρτ/[1+(ρ−1)τ], ρ=5） | **K=16/ chunk、W=5 窗口、N=10 步/chunk**（稳态每周期 2 步） | 滚动窗口错峰噪声 σ((j−1+τ)/W)；动作可见全窗口视频、跨 chunk 动作互不可见 | L2 | Table I LingBot-VA 行 |
| 9 | X-WAM 2604.26694 | 语义=双臂 EEF 位姿+夹爪「变化量」；**维度未写**（附录 B.2 截断） | flow matching + ANS（T_a<T_O；Beta(1.5,1) 耦合采样）；MLP 出入潜空间 | **K=32**、视频 H=8（动作=4× 视频帧率） | 统一序列双向全注意力；深度支路 unilateral attention；T_ee→T_wrist 推导 | L2 | §3.4 UniPC 句处 |
| 10 | OpenWAM 2609.07398 | **统一 80-D**（构成未写）；逐维有效性掩码 | flow matching、独立 t_a、双向 chunk 内注意力 | a_{1:H}（H 未写） | 可组合 C(E,S,M) + 四种世界-动作掩码模式（mutual/action-sees-video/video-sees-action/isolated）；三原则 | L2（ar5iv 回退） | §3.2 联合前向公式处 |

**动作空间未写明的论文清单（正文维度未给数值）**：GigaWorld-Policy（p 亦未写）、LAWA、DreamWAM、FACT（d_a）、THAW-VLA（LIBERO 侧）、X-WAM（维度数值，附录截断）、Kairos（论文侧，仅代码侧实锤）；Rolling-WAM 仅 G1 真机侧写明 78-D、仿真侧未写。已写明：MotuBrain 10-D、OpenWAM 80-D、THAW-VLA(GR1) 29-D。

## 趋势（仅基于原文，10 篇已获取范围内）

1. **连续动作 chunk + flow matching 已是绝对主流（10/10）**：所有论文的动作生成均为 flow-matching 速度场回归；仅 LAWA 额外引入离散 codebook 作为训练 target，但测试时明确「do not apply nearest-neighbor projection」，即测试态仍是连续。未出现自回归动作 token 方案。
2. **「动作高频、视频低频」成为显式共识**：MotuBrain（每视频 latent 帧对应 S_a=f_va·τ 个动作 token）、X-WAM（K/H=4×）、GigaWorld-Policy（每 Δ 步只预测 1 帧）、FACT（H=48 动作 vs 4 个未来帧 offset）、Rolling-WAM（chunk 级错峰）都在动作与视频的时间分辨率上解耦。
3. **决策表征的重心从「是否生成未来」转向「信息流方向」**：GigaWorld/FACT/LAWA 用因果掩码禁止未来视频→动作泄漏；MotuBrain 用 V2A 非对称使动作-only 推理可行；DreamWAM 用共享注意力让「四视图未来表征」塑形动作；X-WAM/Kairos 保留双向联合；OpenWAM 直接把四种掩码模式做成受控变量并给出「explicit world-to-action information flow」原则——即「world→action 单向」被多数新作采纳，但并非全部。
4. **「不放弃未来想象、只压缩它的表征」是 2026 后半的共同命题**：LAWA 原文 "future imagination need not be discarded"；THAW-VLA 主张 "grounding can be inherited while the generative machinery is left behind"；Rolling-WAM 把去噪摊到多个周期；DreamWAM 把未来从 RGB 换成四视图——效率动机（-42.9% 延迟、4.5×、32ms/0.8B、54.4×/11Hz、9×）主导了动作/决策表征的演化方向。
5. **「坏动作的后果」开始进入决策表征**：FACT 把失败 rollout 转为动作条件后果监督并训练 [0,1] 进度值头做可选候选打分；Kairos 以 regret（条件期望物理代价 J_H）为设计准则——决策表征从「成功演示分布」向「动作质量标量」扩展。
6. **跨本体统一动作空间成为目标但实现不透明**：MotuBrain（相对 EEF 统一表征）、X-WAM（统一 EEF 变化量接口）、OpenWAM（统一 80-D + 逐维掩码）、Kairos（跨本体课程）都声称统一，但除 MotuBrain（10-D 细节齐全）与 OpenWAM（80-D，构成未写）外，多数论文的维度/归一化细节缺失——这是本轮最集中的「原文未写」区。

---

## 环境与取证记录

- shell 出网被沙箱拦截（Invoke-WebRequest 实测失败："基础连接已经关闭"），全部网络取证走 web_fetch；HTML 页工具截断位置逐篇标注于「来源级别」。
- GitHub raw 取证成功：kairos-agi/kairos（README + benchmarks/robotwin + benchmarks/libero_plus + 两个 dataset_stats.json），一次成功未用回退。
- OpenWAM 无 arXiv 原生 HTML → ar5iv 回退成功。
- 本线未做：MotuBrain/DreamWAM/OpenWAM 仓库内部逐文件取证（列入各篇待核实）；Kairos/LAWA/FACT/Rolling-WAM/X-WAM 等截断区内容未补取（context 预算约束，按规则标注「未取到」而非脑补）。

<!-- QA_LOG -->

