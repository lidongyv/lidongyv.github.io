# TouchWorld 精读笔记

> 数据源：`auto_docs\papers\txt\TouchWorld.txt`（arXiv:2607.07287v2，776 行）。所有英文引用均为 txt 原文逐字复制，行号为 txt 行号；txt 提取自 PDF，个别断词/缺空格处已按原文复原（如 "intro-duce"→"introduce"）。论文未写的一律标「原文未写」，本人解读标【推断】。数值未做任何四舍五入。

## 1. 原文信息
- 标题："TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation"（L1-2）
- 作者（L3-4）：Jianyi Zhou、Feiyang Hong、Yunhao Li（∗ 并列一作）、Yicheng Zhao、Yongjue Cen、Zirui Liu、Jiakang Huang、Zirui Chen、Ruiyang Zhang、Weizhuo Zhu、Xuhua Song、Shuo Yang✉；"∗Equal contribution, ✉Corresponding author"（L7）
- 机构线索："1Harbin Institute of Technology, Shenzhen"、"2PHANES AI"（L5-6）
- arXiv ID："arXiv:2607.07287v2  [cs.RO]  9 Jul 2026"（L42）；"Date:July 10, 2026"（L28）
- 通讯："Correspondence:shuoyang@hit.edu.cn"（L29）；"Project Page:https://phanes-lab.github.io/TouchWorld-website/"（L30）
- 发表状态：原文未写（txt 仅见 arXiv 编号，无会议/期刊录用信息）。

## 2. Abstract 原文全文（L9-27）
"Dexterous manipulation in everyday environments requires both anticipation and reaction: a robot must predict how contact should evolve while rapidly correcting local errors caused by slip, misalignment, unstable grasping, or force mismatch. Vision and language provide semantic and geometric guidance, but they cannot reliably reveal hidden contact states such as force, slip, and contact stability. Although tactile sensing exposes these physical cues, most existing policies treat touch as a low-frequency observation stream within a monolithic action model, coupling slow task reasoning, action generation, and fast contact feedback in a single loop. We introduce TouchWorld, a predictive-and-reactive tactile foundation model for dexterous manipulation. TouchWorld uses a hierarchical policy that separates vision-language subtask planning, tactile world-model prediction, visuo-tactile goal-conditioned action generation, and high-frequency tactile residual refinement. A High-Level Planning Layer produces executable subtasks and predicts tactile subgoals; a Visuo-Tactile Goal-Conditioned Policy generates nominal action chunks; and a Tactile-Conditioned Refinement Policy performs online residual correction using recent tactile and proprioceptive feedback. By using touch as both a predictive contact reference and a fast feedback signal, TouchWorld preserves the semantic generalization of vision-language-action policies while improving local contact adaptation. Across six long-horizon and contact-rich dexterous manipulation tasks, TouchWorld achieves 65.0% success in the clean setting and 53.7% success under human perturbations, outperforming the strongest baseline by 15.7 and 18.5 percentage points, respectively."

## 3. 问题与动机（为什么需要 predictive AND reactive）
- L33-34："Many daily skills, such as watering, power-plug insertion, cup insertion and tissue pulling, require the robot to anticipate how contact should evolve and to react when the actual contact deviates from expectation."
- L38-40（多时标控制问题的直接表述）："This naturally introduces a multi-timescale control problem. Semantic task reasoning evolves slowly, visuo-tactile action generation operates at an intermediate action-chunk rate, and tactile feedback must support fast local correction when contact changes."
- L86-88（现有单块模型缺陷）："most existing systems still generate actions through a single monolithic model. Tactile observations, when available, are often appended as additional input tokens and processed at the same rate as visual-language context [25]."
- L91-92（缺陷后果）："As a result, a policy may understand the task and produce plausible motions, yet still fail when stable grasping, force regulation, slip recovery, or precise insertion requires rapid tactile feedback."
- L94-95（双通路方案）："TouchWorld is implemented as a hierarchical manipulation policy that uses touch in two complementary ways: a predictive pathway anticipates future contact-aware goals, and a reactive pathway corrects local execution errors online."
- L109-111（核心问题）："The central question of this work is whether a tactile foundation policy can become more robust in contact-rich, long-horizon manipulation by explicitly separating semantic planning, tactile world-model prediction, visuo-tactile action generation, and tactile-conditioned feedback refinement."

## 4. 方法逐机制
总体（L130-148，式 1-4）："TouchWorld predicts the executed action at through the High-Level Planning Layer and two action policies."；四模块 πsubtask / πworld / πgoal / πtactile，nominal 动作块 Ât:t+H−1 经残差修正为 Ãτ:τ+W−1。四阶段训练、非端到端："TouchWorld is trained in four stages. We do not back-propagate through the entire hierarchy end to end; instead, each module is first trained with the supervision signal that matches its time scale, and the final refinement stage couples the trained nominal VLA policy with the reactive tactile layer."（L265-267）

① Subtask Planner（High-Level Planning Layer，1 Hz）
- backbone："We instantiate the Subtask Planner from Qwen3-VL-4B-Instruct [23] and adapt it from annotated teleoperation traces."（L271-272）
- 角色："Rather than producing low-level commands, it updates the semantic phase of the task at a slow rate and emits an executable subtask for downstream policy conditioning."（L153-154）；输出 osubt={ℓ, ℓsubt, rt}（L198）。
- 数据规模："The Subtask Planner dataset contains 128,866 supervision records."（L276）；"The training target is the current executable subtask rather than low-level robot commands"（L276-277）。

② Tactile World Model（predictive 能力）
- backbone："We instantiate the Tactile World Model from Wan2.2-TI2V-5B [19] and train it separately as a predictive subgoal generator."（L280-281）
- 人类数据预训练（真实采集）："We first pretrain it on large-scale EgoTouch human interaction video data [34], which provides synchronized egocentric and wrist-mounted videos, bimanual hand pose, and dense bilateral palm pressure maps."（L281-283）
- 机器人数据微调："After human tactile pretraining, we fine-tune the Tactile World Model on 10 hours of robot demonstrations, corresponding to approximately 1.08 million robot frames at 30 FPS."（L288-289）
- 预测目标格式："each sample pairs a source observation with a future 17-frame visual-tactile target at 384 × 224 resolution."（L292-293）
- 跨形态接口："Because the human tactile layout differs from the robot tactile glove, we use the image-form tactile representation as the shared prediction interface: EgoTouch pressure maps and robot tactile observations are both converted into visual-tactile goal grids."（L285-287）
- 刷新规则："These predictions are updated only when the Subtask Planner detects a new subtask or a meaningful task-state change, so stable task phases reuse the previous tactile subgoal"（L204-206）。

③ Visuo-Tactile Goal-Conditioned Policy（10 Hz，nominal 动作块）
- 架构："The Visuo-Tactile Goal-Conditioned Policy is a tactile diffusion Transformer policy."（L209）
- 触觉 token 化（图像化）："For the nominal VLA policy, tactile observations are converted into a unified image representation and processed together with visual observations by the vision-language branch."（L211-212）
- 预测目标："A diffusion Transformer action expert then performs flow-matching action generation conditioned on the visual, tactile-image, language, and proprioceptive context."（L212-214）
- flow matching："During training, noisy actions are sampled from the interpolation between data actions and Gaussian noise. During inference, the model starts from Gaussian noise and integrates the predicted velocity field to produce a nominal action sequence."（L227-230）
- 动作规格："the policy predicts a 120-dimensional action vector over a 32-step action horizon. The action vector consists of two 48-dimensional arm-hand action representations, a 9-dimensional head action, and 15 reserved dimensions."（L733-735）；"the nominal VLA policy predicts action chunks with H = 32"（L346）。
- 提示词（式 6，L220）：ℓpolicy = "Task: ℓ ⊕ Current subtask: ℓsubt"（"We compose it from both the original task instruction and the current executable subtask"）。

④ Tactile-Conditioned Refinement Policy / TRT（30 Hz，reactive 能力）
- 作用："The Tactile-Conditioned Refinement Policy converts the nominal action chunk into a locally corrected action window. It operates at a faster feedback rate than the Visuo-Tactile Goal-Conditioned Policy so that contact changes can influence control without waiting for the next nominal chunk."（L232-234）
- 残差公式（式 7-8，L238-239）：ΔAτ:τ+W−1 = fϕ(Âτ:τ+W−1, sτ−k:τ, Xτ−k:τ, ct)；Ãτ:τ+W−1 = Âτ:τ+W−1 + ΔAτ:τ+W−1。
- 触觉编码："Different tactile signal types are encoded with modality-specific lightweight encoders before being fused by the residual Transformer: image-like tactile maps are processed as tactile images, matrix tactile readings are encoded with convolutional layers, and low-dimensional tactile states are embedded with Fourier features and MLPs."（L242-245）
- 执行："At deployment, the controller executes the first C actions from the corrected window and then refreshes the residual prediction with newly observed tactile feedback."（L245-246）；"The refinement layer uses a sliding nominal-action lookahead window with W = 16 and commits the first C = 4 corrected actions before refreshing the residual prediction."（L347-348）
- 残差子空间："the residual layer is trained only on a 58-dimensional tactile-sensitive action subspace covering the two wrist pose blocks and selected hand joints."（L758-759）

⑤ 损失函数
- 反馈细化损失（式 9，L323-324）："Lfb = ‖fϕ(Âτ:τ+W−1, sτ−k:τ, Xτ−k:τ, ct) − (A∗τ:τ+W−1 − Âτ:τ+W−1)‖²₂ (9)"（txt 范数脚标显示为 "22"，按式 9 复原）；"Here Lfb is the feedback-refinement loss and A∗τ:τ+W−1 is the demonstrated high-frequency action window."（L325）
- 作用范围："The loss is applied in the tactile-sensitive residual action subspace, while action dimensions outside this subspace follow the nominal VLA output."（L326-327）
- nominal 策略训练："We train the action decoder with imitation learning and flow matching"（L299-300）。
- TRT 训练："the residual actor and tactile feedback encoder are optimized with masked MSE between the corrected action window and the demonstration action window."（L763-764）；规模："dmodel = 512, eight layers, eight attention heads"（L761-762）。

## 5. 实验与主结果
- 设置："For each task, we collect 200 teleoperated training trajectories and conduct 100 real-robot evaluation rollouts."（L361-362）；指标定义："For real-robot experiments, we report success rate over repeated rollouts in both settings."（L363）
- 扰动设置："In the perturbation setting, the operator applies disturbances such as target displacement, unstable contact, or grasp interference during execution."（L357-358）
- 基线（L370-374）：Pi-0.5（VLA baseline）；FTP-1（"the previous monolithic tactile policy baseline without the proposed predictive-and-reactive hierarchy"）；GR00T N1.7（generalist baseline）。

Table 1（L418-431）Per-task manipulation success rates (%)；原注："The best and second-best results in each setting are highlighted in bold and underline, respectively."

| Method | Water Flower | Tabletop Clearing | Cup Insertion | Power Plug Insertion | Pot Wiping | Tissue Pulling | Avg. |
|---|---|---|---|---|---|---|---|
| Clean: Pi-0.5 | 52 | 66 | 36 | 12 | 39 | 39 | 40.7 |
| Clean: FTP-1 | 56 | 60 | 48 | 32 | 57 | 43 | 49.3 |
| Clean: GR00T N1.7 | 50 | 58 | 33 | 18 | 36 | 41 | 39.3 |
| Clean: TouchWorld | 72 | 76 | 66 | 45 | 70 | 61 | 65.0 |
| Pert.: Pi-0.5 | 34 | 44 | 24 | 6 | 28 | 30 | 27.7 |
| Pert.: FTP-1 | 39 | 42 | 34 | 20 | 42 | 34 | 35.2 |
| Pert.: GR00T N1.7 | 32 | 36 | 21 | 9 | 26 | 32 | 26.0 |
| Pert.: TouchWorld | 60 | 62 | 52 | 35 | 57 | 56 | 53.7 |

主结果表述："In the clean setting, TouchWorld reaches 65.0% average success, improving over the strongest baseline by 15.7 percentage points. Under human perturbations, TouchWorld reaches 53.7% average success, improving over the strongest baseline by 18.5 percentage points."（L378-380）；"The gains are especially clear on power plug insertion, pot wiping, and tissue pulling, where tactile prediction and fast local correction are most important."（L380-382）

Table 2（L432-444）Tactile World Model 预测精度（均为 %，越高越好）；指标定义（L448-451）："We evaluate the predicted tactile sequence with temporal contact accuracy over the 17-frame window, and evaluate the terminal tactile goal with contact IoU and volumetric IoU after thresholding the pressure map at τ."（阈值 τ 的具体数值原文未写）；对照基线："a persistence baseline that copies the current tactile observation as the future goal and a nearest-neighbor baseline that retrieves a subgoal from the training set using the task and subtask label."

| Method | Temporal Contact Acc. | Contact IoU | Volumetric IoU |
|---|---|---|---|
| Current tactile copy | 70.4 | 31.8 | 24.6 |
| Nearest-neighbor subgoal | 77.5 | 39.2 | 31.0 |
| Tactile World Model | 86.3 | 52.7 | 43.8 |

结论："Table 2 shows that the Tactile World Model predicts substantially more accurate contact timing and terminal tactile geometry than simple persistence or retrieval baselines."（L452-453）

Table 3（L469-482）Subtask Planner 变体评估（%）；指标定义（L459-460）："We therefore report subtask correctness, downstream execution success, and transition F1 over task phases."

| Planner | Subtask Acc. | Execution Success | Transition F1 |
|---|---|---|---|
| Zero-shot Qwen3-VL-4B | 43 | 34 | 62 |
| Zero-shot Qwen3-VL-32B | 69 | 54 | 71 |
| SFT Qwen3-VL-4B w/o Memory | 73 | 60 | 76 |
| Memory-Augmented SFT Qwen3-VL-4B (Ours) | 88 | 65 | 82 |

结论："The memory-augmented 4B Subtask Planner outperforms the zero-shot 32B planner, suggesting that task-phase supervision and execution history are more important than model scale alone for this interface."（L462-464）

## 6. 消融
四个消融（L401-408 原文）：
- "• Subtask Planner: remove subtask decomposition and condition the policy directly on the original task prompt."
- "• Tactile World Model: remove future visual-tactile goal prediction and condition the policy only on the current observation and current subtask."
- "• Tactile-Conditioned Refinement Policy: remove the residual feedback layer and execute nominal action chunks directly."
- "• Tactile input: remove tactile observations from the policy input to measure the contribution of touch sensing."

结论（L409-413 原文）："Removing tactile input causes the largest degradation, confirming that contact observations are essential for this benchmark. Removing the Tactile-Conditioned Refinement Policy particularly hurts the human perturbation setting, where the system must correct local execution errors online. Removing the Subtask Planner mainly reduces long-horizon consistency, while removing the Tactile World Model weakens contact-aware goal conditioning."
注：各消融的具体数值在 Figure 5 堆叠柱状图中，txt 未含柱高数字（原文未写）。

## 7. 局限 / 未来工作（L489-520 逐字）
- "First, our real-robot evaluation focuses on six representative contact-rich tasks. These tasks cover planning, insertion, wiping, and soft-object handling, but they do not yet exhaust the diversity of household manipulation or deformable-object interactions. Scaling the benchmark to more objects, workspaces, and task compositions is an important next step."
- "Second, the Tactile World Model currently predicts short-horizon visual-tactile subgoals. This is sufficient for contact-aware subtask execution, but longer-horizon prediction remains challenging when object motion, hand occlusion, or human perturbations produce multiple plausible futures. Future work can improve this component with uncertainty-aware subgoal prediction or by generating multiple candidate tactile subgoals for downstream selection."
- "Third, TouchWorld is tied to the sensing layout used in our robot platform. ... Although this design is modular, transferring to a different tactile sensor or hand morphology still requires calibration, normalization, and likely a small amount of adaptation data."
- "Finally, the hierarchy introduces scheduling hyperparameters, including the High-Level Planning Layer update rate, the Tactile World Model refresh rule, the nominal action chunk length, and the residual commit interval. We use a fixed scheduling profile in this work, which worked well for our tasks. More adaptive scheduling could further reduce computation and improve responsiveness when contact dynamics change rapidly."
- 结论中的未来工作（L529-530）："Future work will scale the task suite, improve uncertainty-aware tactile goal prediction, and study broader transfer across tactile sensors and robot hands."

## 8. 与叙事框架的关联（全部【推断】，附支持原句）
1. 脊髓层（毫秒级在线反应）证据【推断】：三级时标 1 Hz / 10 Hz / 30 Hz（Figure 1 标注，L55-59："High-Level Planning Layer (1 Hz)"、"Visuo-Tactile Goal-Conditioned Policy Layer (10 Hz)"、"Reactive Tactile Refinement Layer (30Hz)"）；反应层直入控制环："the refinement policy provides a fast feedback path when real contact deviates from the prediction."（L105-106）。论文未用任何生物类比，此对应属推断。
2. 触觉「预测+反应」与视觉世界模型的差异【推断】：触觉世界模型预测的是接触结果而非视觉场景："it predicts future visual-tactile subgoals that describe the expected contact outcome of the current subtask."（L203-204）；视觉/语言"cannot reliably reveal hidden contact states such as force, slip, and contact stability."（L11-12）；触觉信息局部短程："tactile histories are short-range, local, and most informative near contact transitions such as slip, impact, or insertion misalignment."（L254-255）。
3. 「力学数据必须真实采集」的证据【推断】：训练数据全部真实采集——人类侧："large-scale EgoTouch human interaction video data [34], which provides synchronized egocentric and wrist-mounted videos, bimanual hand pose, and dense bilateral palm pressure maps."（L281-283）；机器人侧："we fine-tune the Tactile World Model on 10 hours of robot demonstrations, corresponding to approximately 1.08 million robot frames at 30 FPS."（L288-289）；硬件真实："the demonstrator wears a Meta Quest headset ... and uses Meta Quest Touch Plus controllers together with a Wuji Glove"（L338-341）。全文未出现任何仿真/合成力学数据来源。
4. 图像化触觉作为跨 embodiment 桥梁【推断】："Because the human tactile layout differs from the robot tactile glove, we use the image-form tactile representation as the shared prediction interface: EgoTouch pressure maps and robot tactile observations are both converted into visual-tactile goal grids."（L285-287）
5. 局限与「真实采集」叙事自洽【推断】："transferring to a different tactile sensor or hand morphology still requires calibration, normalization, and likely a small amount of adaptation data."（L501-502）。

## 9. 图注 / 表注（原文逐字）
- Figure 1（L82-85）："Conceptual overview of TouchWorld. The High-Level Planning Layer contains a Subtask Planner that produces executable subtasks and a Tactile World Model that predicts visual-tactile subgoals. A visuo-tactile goal-conditioned policy generates nominal action chunks, and a tactile-conditioned refinement policy refines the final action online using high-frequency tactile feedback."
- Figure 2（L193-197）："TouchWorld architecture. The High-Level Planning Layer runs at a slow semantic rate and contains the Subtask Planner, which produces executable subtasks, and the Tactile World Model, which predicts visual-tactile subgoals. The Visuo-Tactile Goal-Conditioned Policy generates nominal action chunks at an intermediate rate, and the Tactile-Conditioned Refinement Policy refines execution inside the high-frequency robot control loop using tactile and proprioceptive feedback."
- Figure 3（L312-314）："Hardware platform for TouchWorld. The human operator provides visual and hand-motion inputs through a Meta Quest headset, Meta Quest Touch Plus controllers, and a Wuji Glove. The robot platform uses Wuji dexterous hands and a JQ-Industries tactile glove for contact-rich manipulation with tactile feedback."
- Figure 4（L397-398）："Real-robot task suite for evaluating TouchWorld. The six tasks cover long-horizon planning, precision insertion, continuous contact regulation, soft-object handling, and recovery from human perturbations."
- Figure 5（L466-468）："Stacked ablation results for TouchWorld. Each bar reports the six-task average success rate, with colored segments showing the contribution of each task to the average. Solid bars denote clean rollouts, while hatched bars denote human-perturbed rollouts that require corrective behavior."
- Figure 6（L512-513）："Qualitative inference demonstration of TouchWorld. For each task, the model progresses from the initial scene through a sequence of executable subtasks and predicts tactile subgoals that support contact-aware manipulation."
- Table 1（L418-420）："Per-task manipulation success rates (%) on the real-robot TouchWorld benchmark. Each task is evaluated in both a clean setting and a human perturbation setting. The best and second-best results in each setting are highlighted in bold and underline, respectively."
- Table 2（L432-434）："Tactile World Model prediction accuracy on held-out subgoal segments. Ground truth is the tactile pressure observation collected when the robot reaches the corresponding annotated subgoal state. All numbers are percentages, and higher is better for all metrics."
- Table 3（L469-471）："Evaluation of Vision-Language Subtask Planner variants. We compare zero-shot planners, supervised fine-tuning without memory, and the memory-augmented planner used by TouchWorld. Metrics are reported in percentage."

---
附注：Abstract 全文、9 条图注/表注、3 张数值表、局限段落为强制逐字内容，合计已超出 2500–3500 字目标；正文中文讲解已尽量压缩，请以结构完整性为准。
