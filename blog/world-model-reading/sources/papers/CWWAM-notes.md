# CW-WAM 精读笔记

> 数据源：`auto_docs\papers\txt\CWWAM.txt`（唯一来源）。所有英文引用逐字复制自该 txt 并注明行号；论文未写之处标注「原文未写」；个人解读以【推断】标注。数值一律不四舍五入，照抄原文。

## 1) 原文信息

- 标题：CW-WAM: A Structured Contact-Wrench World Action Model for Force-Sensitive Manipulation（L1–2；txt 第 1 行作「CW-W AM」，系换行断词伪影，其余各处均作 CW-WAM）
- 作者：Houzhu Ding, Xinda Qi, Luqi Wang, Ming Yang（L3）；机构：Ant Group（L4）
- 编号：arXiv:2610.30117v1 [cs.RO] 21 Oct 2026（L100）
- 发表状态：txt 仅给出 arXiv v1 预印本信息；是否被会议/期刊录用【原文未写】
- Index Terms（L67–68）："world action models, force-sensitive manipulation, force–torque sensing, admittance control"

## 2) Abstract 全文逐字（L45–66）

> Abstract—Force-sensitive manipulation requires contact and load information that is not provided explicitly by visual observations. We present CW-W AM, a world action model that jointly predicts video, motion, and contact wrench. Inspired by hybrid motion–force control, it contains a unified interaction module that treats both motion and contact wrench as modeling targets and predicts time-aligned motion–wrench chunks as control references. A pair-causal attention mechanism structures the motion–wrench dependencies, while a long fixed-length wrench memory retains past interaction information. The predicted wrench chunk acts as a force reference for a 20-Hz outer-loop admittance controller within a slow–fast control system that maintains force feedback and smooth motion between model queries. CW-W AM improves contact prediction and increases the insertion success rate from 25% to 70% for square pegs with 1-mm clearance per side and from 40% to 95% for power adapters, compared with WAM without wrench input. It also improves lifting-height accuracy across different loads. Our framework with an admittance controller reduces force-reference RMSE from 4.8 to 2.7 N for peg insertion and from 2.0 to 1.3 N for wiping, respectively, with smaller force fluctuations during successful execution. Project website:cw-wam-project.github.io.

（注：Abstract 内两处 "CW-W AM" 与标题同样为断词伪影，一并保留以忠实转录。）

## 3) 问题与动机

论文的出发点是：力敏感操作所需的接触与负载信息无法由视觉显式提供，而「看起来」成功的外观极具欺骗性（卡死的方销看起来像已完全就位、外观相同的物体重量差异巨大），必须靠腕部力/力矩传感提供物理交互信息；进一步地，联合预测「动作+力」可以把预测力用于引导反馈校正或评估接触结果，因此要预测每个动作伴随的 wrench 并用它引导执行；接触由指令运动、先前交互与环境共同决定，这要求把二者结构化地联合建模。

- L45–46："Force-sensitive manipulation requires contact and load information that is not provided explicitly by visual observations."
- L72–76："Yet a square peg jammed during insertion may appear fully seated, and visually similar objects may differ substantially in weight. Such force-sensitive tasks require physical interaction information that wrist force–torque sensing can provide."
- L77–78："Force feedback improves contact-rich manipulation by complementing visual observations [2], [3] and enabling rapid motion correction [4]."
- L79–83："Joint action–wrench prediction extends these benefits: predicted forces guide learned feedback correction [5] or help evaluate contact outcomes [6]. These advances motivate WAMs that predict the wrench associated with each action and use it to guide execution."
- L84–85："Contact depends on commanded motion, prior interaction, and the environment."
- L85–89："To capture these dependencies, we design an interaction module inspired by hybrid motion–force control. The module pairs past commands with measured wrenches (Fig. 1) and predicts future motion and wrench using separate tokens, prediction heads, and training losses."
- L89–91："High-rate wrench encoding captures contact changes within motion intervals, while longer memory retains earlier wrench measurements."

## 4) 方法逐机制

### 4.1 contact-wrench 表示（结构分解）

预测目标分两条流：动作流与力流，且一一配对、时间对齐。动作是 10 维（位置 + 6-D 连续旋转 + 夹爪），预测 wrench 是 6 维——去偏置、去工具重力的 TCP 系力与力矩（力矩以 TCP 原点为参考），这正是「六维力/力矩」的量化本体；训练时每个动作目标与同一时间戳的补偿后 wrench 配对。

- L247–249："CW-WAM jointly predicts future video, motion, and contact wrench for motion–force control."
- L250（式 (1)）："pθ(Z +t , At, Wt |V t, st, ℓ,HAWt ,H Wt )."（hats 表示预测，txt 排版中帽号与下标分离；Vt、st、ℓ 分别为 RGB 观测、机器人状态、语言指令，HAWt 与 HWt 为近期动作–wrench 历史与更长 wrench 记忆）
- L254–257："Outputs are video latents Z +t , motion targets At = (at,i)Hi=1, and wrenches Wt = (wt,i)Hi=1, with hats denoting predictions. Training pairs each motion target with the compensated wrench at the same timestamp."
- L258–259："Each action at,i ∈R 10 contains position, continuous 6-D rotation [27], and gripper position."（「∈R 10」为 txt 断词，实为 at,i ∈ R10）
- L260–262："Position and rotation specify a delta from the tool center point (TCP) pose at observation time t, converted to an absolute base-frame pose."
- L263–266："Each predicted wrench ˆwt,i ∈R 6 represents bias-corrected, tool-gravity-compensated force and torque in the TCP frame defined by its paired predicted pose, with torque referenced to the TCP origin."

### 4.2 世界模型与动作联合

模型建立在 DreamZero 视频–动作骨干（Wan2.2-TI2V-5B DiT）上：三视角 RGB 经 VAE 合成 160×320 composite，状态经 MLP，视频/状态/历史/未来动作–wrench token 共享自注意力，并以 OpenCLIP 图像特征与 UMT5 语言特征做交叉注意力。交互模块把过去的运动指令与同一时间区间的实测 wrench 配对（近期 24 个区间、每区间最多 16 个高频 wrench 样本）；另有长达 150 区间的仅 wrench 长记忆（LTM），保留先前载荷张力或持续阻力。未来运动与 wrench token 共享 Transformer 但用各自编码器、预测头与损失；pair-causal（PC）下三角注意力使每个未来 wrench 能关注其配对动作与更早的配对，反之则不能。训练用 flow matching，三模态噪声共享噪声级别 σ，按有效通道归一化损失（动作 10 通道、wrench 6 通道，λm=1）。

- L267–272："CW-WAM extends DreamZero [1] using a Wan2.2-TI2V-5B DiT [28]. A VAE encodes three RGB views as a 160×320 composite; an MLP encodes robot state. Video, state, history, and future action–wrench tokens share self-attention, conditioned on OpenCLIP image and UMT5 language features through cross-attention."
- L274–278："The interaction module pairs past motion commands with measured wrenches from the same time interval. Recent history spans Hs intervals at rate fh, with boundaries bi =t−(H s − i)/fh. Each interval (bi−1, bi] associates up to Kw high-rate wrench samples with a motion-target token."
- L279–283："A temporal wrench encoder processes each sample's 6-D value, relative sampling time, and inter-sample interval. Validity masks exclude padding from attention and pooling. The action encoder produces a separate token, and both tokens receive the same history-interval encoding:"
- L284（式 (2)）："HAWt = [A−1 , W−1 , . . . , A−Hs , W−Hs ]."
- L285–289："Long-term wrench memory (LTM) extends history to Hℓ wrench-only intervals using the same encoder with additional window-time and history embeddings. It retains preceding payload tension or sustained resistance. Both histories overlap and end at the current observation."
- L290–293："Future motion and wrench tokens share the Transformer but use separate encoders, prediction heads, and losses. Pair-causal (PC) attention uses a lower-triangular mask over [A+1 , W+1 , . . . , A+H , W+H ]. Thus, W +i attends to A+i , but not conversely; both attend to earlier pairs."
- L298–299："All future tokens update jointly at each denoising step."
- L301–303："Training uses flow matching with normalized action and wrench targets. For each modality m∈ {V, A, W}, target xm is mixed with independent noise ϵm ∼ N(0, I)at a shared noise levelσwithin each block:"
- L311–313："Valid-channel normalization averages motion and wrench losses over their 10 and 6 valid channels, respectively, with λm = 1 for all modalities."

### 4.3 训练数据与训练配置

- L428–430："Training uses task-specific language instructions and 450 demonstrations: 213 peg insertions, 97 adapter insertions, 64 whiteboard wipes, and 76 spring-scale transports."
- L450–458："Data are collected by bilateral teleoperation with two 7-DoF Franka arms. Leader joint positions command the follower, whose external-joint-torque estimates provide force feedback to the operator. A wrist-mounted LinkTouch six-axis F/T sensor records wrench at 150 Hz. We remove sensor bias and calibrated tool gravity, transform the wrench to the TCP frame with moment about the TCP origin, and apply a causal second-order Butterworth low-pass filter with a 10-Hz cutoff. The resulting wrench represents external contact and grasped-payload loads."
- L459–462："Two RealSense D405 cameras and one D455 provide three RGB views. Motion commands, robot state, and images are recorded at 15 Hz and aligned with the wrench stream by timestamps."
- L463–466："Training uses eight NVIDIA H20 GPUs and AdamW with a global batch size of 8, a peak learning rate of 10−5, and weight decay of 10−5. We use 5% linear warmup followed by cosine decay."
- L316–319："We train the DiT and modality projectors for 50k steps with task-balanced sampling, keeping VAE, OpenCLIP, and UMT5 frozen. We use H=H s = 24, Hℓ = 150, fh = 15 Hz, and Kw = 16: 1.6 s of recent history and 10 s of wrench memory."
- L320–323："At inference, four Euler steps jointly generate two video latent frames and 24 action–wrench pairs spanning 1.6 s, while observed latents remain fixed. Control uses the paired predictions without decoding video to RGB."

### 4.4 慢快执行与控制（配套机制）

预测 wrench 作为导纳控制的力参考，在模型查询之间维持力反馈：20 Hz 外环导纳修正 + 1 kHz 关节阻抗跟踪。

- L325–326："We apply task-weighted admittance corrections to nominal motion at 20 Hz, with joint-impedance tracking at 1 kHz."
- L339–341："We use the predicted wrench as a feedback reference to maintain the interaction expected along the jointly predicted motion."
- L346–348（式 (5)–(7)）："ew = ˆw(ϕw)−w meas, (5) M∆¨x+C∆ ˙x+Ka∆x=−S(e w +k I η), (6) ˙η=e w −λη. (7)"（ϕw 为按进度对齐选择的预测 wrench，S 为对角反馈权重矩阵，η 为带衰减率 λ 的误差积分）
- L364–370："The feedback-weighting matrix S reflects each task's contact constraints (Table I). For peg insertion, smaller Fz and τz weights favor lateral and tilt corrections while reducing changes to nominal insertion motion. Adapter insertion uses larger axial-force and torque weights. Wiping enables only Fz feedback, leaving tangential motion and orientation to the nominal trajectory."
- L372–375："On a single NVIDIA RTX 5090, four-step model queries typically take 0.5–0.6 s, including observation processing, inference, and postprocessing, but excluding network communication."
- L376–379："Chunks execute sequentially (Fig. 3). While awaiting the next prediction, the robot holds the final nominal pose with admittance active during contact. An adaptive-duration C2 bridge joins each new chunk to the current commanded pose, matching velocity and acceleration."
- L383–385："Matched progress ϕw cannot move backward or exceed scheduled progress ϕ, accommodating execution lag while preserving reference order."

## 5) 实验与主结果

任务设定：方形销插入（38 mm 方销入 40 mm 孔，单边 1 mm 间隙、插入深度约 50 mm）、电源适配器插入、白板擦拭、弹簧秤搬运（盒子 30 g 或 830 g 且外观相同）。

- L398–400："The robot inserts a 38-mm square peg into a 40-mm hole with 1-mm clearance per side and an insertion depth of about 50 mm."
- L421–425："The robot grasps a spring scale attached to a box and transfers the box between bins. The boxes weigh either 30 g or 830 g but look identical. The heavier box stretches the spring farther, requiring a higher lift to clear the rim."
- L539–540："Each method uses 20 trials per contact task and 10 per load (Sec. V-F)."

### 5.1 开环预测（Table II，全部数值照抄）

评估设置（L488–491）："Table II evaluates 50k-step checkpoints on 373 24-step windows from 22 test episodes, using four denoising steps and seed 42. Windows use stride 24 plus a final tail window." 接触定义（L491–492）："Contact means ∥F∥ 2 ≥5 N; dynamic windows contain contact and satisfy maxi,j ∥Fi −F j∥2 ≥2 N."

| 模型 | e_p [mm] | e_R [°] | e_F [N] | e_I [N s] |
|---|---|---|---|---|
| WAM | 26.97 | 2.85 | – | – |
| WAM + Wrench (padded) | 26.32 | 2.76 | 3.15 | 2.33 |
| WAM + Wrench (valid) | 25.50 | 2.80 | 3.03 | 2.21 |
| CW-WAM w/o LTM and PC | 22.88 | 2.25 | 2.92 | 2.11 |
| CW-WAM w/o LTM | 19.92 | 2.02 | 2.84 | 2.17 |
| CW-WAM | 20.69 | 2.07 | 2.79 | 2.01 |

（指标定义见第 9 节表注与 L495–505；e_p、e_R、e_F、e_I 分别为接触位置误差、SO(3) 姿态误差、动态力向量 RMSE、力幅值积分误差。）

### 5.2 真机任务（Fig. 4，全部数值照抄；列序为 WAM / WAM+Wrench / CW-WAM w/o LTM / CW-WAM / CW-WAM+Adm.，L678）

| 指标 | WAM | WAM+Wr | CW−LTM | CW-WAM | CW+Adm |
|---|---|---|---|---|---|
| 插销插入成功 | 5/20 | 8/20 | 7/20 | 14/20 | 14/20 |
| 插销释放成功 | 0/20 | 1/20 | 6/20 | 11/20 | 12/20 |
| 适配器插入成功 | 8/20 | 9/20 | 16/20 | 19/20 | 19/20 |
| 适配器释放成功 | 0/20 | 0/20 | 13/20 | 16/20 | 15/20 |
| 白板擦拭成功 | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 |
| 力参考 RMSE·插销 [N] | — | 15.15 | 4.09 | 4.84 | 2.75 |
| 力参考 RMSE·适配器 [N] | — | 2.53 | 9.72 | 8.37 | 5.24 |
| 力参考 RMSE·擦拭 [N] | — | 3.60 | 2.65 | 1.99 | 1.34 |
| 预测vs演示 RMSE·插销 [N] | — | 2.64 | 1.35 | 1.27 | 1.14 |
| 预测vs演示 RMSE·适配器 [N] | — | 2.62 | 2.71 | 2.99 | 2.20 |
| 预测vs演示 RMSE·擦拭 [N] | — | 1.30 | 1.46 | 1.54 | 1.56 |
| 接触力波动·插销 [N] | 4.23 | 3.40 | 1.40 | 2.15 | 1.11 |
| 接触力波动·适配器 [N] | 2.36 | 1.01 | 2.50 | 2.94 | 2.10 |
| 接触力波动·擦拭 [N] | 1.44 | 1.13 | 1.11 | 1.13 | 0.87 |
| 位置跟踪 RMSE·插销 [mm] | 6.53 | 12.49 | 3.23 | 2.80 | 2.71 |
| 位置跟踪 RMSE·适配器 [mm] | 23.63 | 25.26 | 14.45 | 8.75 | 11.55 |
| 位置跟踪 RMSE·擦拭 [mm] | 45.47 | 47.00 | 43.02 | 40.29 | 36.04 |

说明：图注称「Quality rows compare all 20 trials (light, hatched) with successful subsets (dark bars)」，即每行应有全部 20 次与成功子集两组值，但 txt 提取层每格只保留一组数值；其中插销位置跟踪 12.49/2.80 与力参考 15.15/4.84 对应正文「from 12.5 to 2.8 mm」「from 15.2 to 4.8 N」的成功子集表述，适配器 25.26/8.75 对应「from 25.3 to 8.8 mm」，故该组数值【推断】为成功子集（dark bars）值；全 20 次一组数值在 txt 中未保留。

正文关键结果：

- L696–700："The interaction module improves insertion and release (Fig. 4). Relative to WAM + Wrench, CW-WAM w/o LTM raises adapter insertion from 9/20 to 16/20 and release from 0/20 to 13/20. Peg release improves from 1/20 to 6/20, with insertion at 7/20 versus 8/20."
- L701–706："Longer wrench memory improves both insertion tasks: peg insertion rises from 7/20 to 14/20 and release from 6/20 to 11/20; adapter insertion rises from 16/20 to 19/20 and release from 13/20 to 16/20. It also lowers force-reference error in adapter insertion and wiping, demonstrating the task value of longer interaction context (Fig. 4)."
- L707–709："In successful peg trials, CW-WAM reduces position-tracking RMSE from 12.5 to 2.8 mm and force-reference RMSE from 15.2 to 4.8 N relative to WAM + Wrench (Fig. 4)."
- L711–719："For adapter insertion, CW-WAM improves completion and reduces position-tracking RMSE from 25.3 to 8.8 mm. Its higher force-reference RMSE and contact-force variation relative to WAM + Wrench (Figs. 4 and 5(c)) are consistent with the observation that most successful WAM + Wrench trials rely on precise, one-shot alignment without adjustment, followed by full insertion. In contrast, CW-WAM's learned search and firm insertion behavior improves insertion success while producing higher contact forces."
- L720–723："In the whiteboard-wiping task, all methods remove the ink in 20/20 trials with simple patterns. For more complicated patterns, such as circles and letters, the average removal rate is 85.9% across the six selected trace patterns."
- 摘要/结论口径（L60–61、L841–843）：插销插入成功率 25%→70%（5/20→14/20）、适配器插入 40%→95%（8/20→19/20），对比对象为无 wrench 输入的 WAM。

### 5.3 负载自适应抬升（Fig. 6，全部数值照抄）

- L762–764："CW-WAM reproduces load-dependent lifting heights more accurately than the other models, even when the boxes look identical."
- L765–767："Its median peak heights are 423.6 and 481.8 mm for light and heavy loads, respectively, close to the demonstration medians of 420.9 and 492.7 mm (Fig. 6)."
- L768–769："WAM reaches 464.4 and 447.3 mm, showing no higher lift for heavier loads."
- L770–773："WAM + Wrench gives an intermediate response at 431.3 and 467.9 mm: its absolute differences from the demonstration medians are 10.4 and 24.8 mm, compared with 2.7 and 10.9 mm for CW-WAM."

### 5.4 预测 wrench 反馈（Fig. 7，全部数值照抄）

- L775–779："Admittance further reduces force-reference RMSE and measured-force variation compared with CW-WAM alone (Fig. 4). Peg RMSE decreases from 4.8 to 2.7 N and wiping RMSE from 2.0 to 1.3 N, while insertion and wiping success rates remain unchanged."
- L782–784："For the representative trials in Fig. 7, force-reference RMSEs are 3.88 and 2.41 N without feedback (a, b), and 2.69 and 1.20 N with predicted-wrench tracking (e, f)."（图 7(c)(d) 轴标另给出跟踪演示中位数的 RMSE 4.11 N 与 2.79 N）
- L785–788："Predicted-wrench tracking also yields the lowest median contact-force variation, which decreases from 1.86 to 1.39 to 0.94 N for peg insertion and from 1.11 to 0.92 to 0.85 N for wiping."
- L788–791："The larger reduction in peg insertion, where contact changes during search and seating, highlights the value of predicted wrench references for smoother contact. Torque errors and variations are generally small across tasks."

## 6) 消融

变体定义（L468–486）：

- L468–470："WAM has no wrench input or output. WAM + Wrench encodes pooled wrench history and predicts concatenated 16-D action–wrench vectors."
- L471–473："CW-WAM w/o LTM introduces the interaction module: explicit paired command and wrench history, separate future streams, and PC attention."
- L475–477："Full CW-WAM adds LTM. CW-WAM w/o LTM and PC replaces only PC with bidirectional attention, isolating attention structure within the no-LTM module."
- L478–486："All variants share training data, backbone initialization, and a 50k-step schedule with task-balanced sampling. WAM omits wrench loss. WAM + Wrench uses either padded-width normalization, dividing masked squared-error sums by 32, or CW-WAM's valid-channel normalization (Sec. III-C), with architecture and other settings unchanged. The latter tests the interaction module under the same loss normalization. PC and LTM ablations retain the CW-WAM objective: PC is tested without LTM, and LTM with PC retained."

消融结果（L506–521，全部数值照抄）：

- L506–507："Valid-channel normalization lowers WAM + Wrench errors from 26.32 to 25.50 mm and from 3.15 to 3.03 N."
- L508–512："Under the same normalization, CW-WAM without LTM achieves 19.92 mm and 2.84 N: paired reductions of 5.58 mm (95% CI [3.65, 7.84]) and 0.19 N (95% CI [0.04, 0.33]). Orientation error also decreases from 2.80 ◦ to 2.02 ◦, and dynamic torque-vector RMSE from 0.0908 to 0.0837 N m."
- L513–515："These gains support structured interaction modeling beyond loss normalization. CIs use 10,000 paired episode bootstrap resamples, stratified by task, with trained models held fixed."
- L516–517："PC reduces no-LTM contact-position and orientation errors by 2.96 mm (95% CI [2.05, 3.89]) and 0.23 ◦ (95% CI [0.12, 0.35]), respectively."
- L518–521："Full CW-WAM achieves the lowest dynamic force-vector RMSE (2.79 N) and force-magnitude integral error (2.01 N s). The no-LTM variant retains slightly lower position and orientation errors. Real-robot tests evaluate how these gains translate into interaction and task completion."

（对照 Table II：CW-WAM w/o LTM and PC = 22.88 mm / 2.25°，即 PC 消融点的 22.88→19.92 mm 差 2.96 mm、2.25°→2.02° 差 0.23°，与上述 CI 一致。）

## 7) 局限 / 未来工作（L850–857，原文逐字）

> "Our evaluation covers four tasks within the demonstration distribution, using manually tuned feedback gains and predefined compliant directions. Future work will examine how controller settings affect performance and explore learning additional contact parameters such as the feedback-weighting matrix and contact frame, alongside motion and wrench prediction for end-to-end closed-loop control. We will also evaluate generalization to new objects and tasks."

解读：局限 = 仅示范分布内 4 任务 + 手工整定反馈增益 + 预定义柔顺方向；未来 = 研究控制器设定影响、学习 S 矩阵与接触坐标系等接触参数、端到端闭环控制、新物体/新任务泛化。

## 8) 叙事关联【推断】

前提说明：「丁文伯」「等效六维力」字样在论文中未出现【原文未写】；论文第一作者为 Houzhu Ding（L3）。以下仅就「六维力/力矩预测」与「等效六维力」这一概念表述的可对应性给出推断，每条附支持原句。

1. 【推断】论文预测的 wrench 是六维（3 力 + 3 矩）的量，与「等效六维力」所指向的「六维力/力矩」量化对象一致。——支持：L263–266："Each predicted wrench ˆwt,i ∈R 6 represents bias-corrected, tool-gravity-compensated force and torque in the TCP frame defined by its paired predicted pose, with torque referenced to the TCP origin."
2. 【推断】该六维量经过传感器偏置去除与工具重力补偿（含 10 Hz Butterworth 滤波），是「净外部接触 + 夹持载荷」而非传感器原始读数，构成「等效」（去自重/去偏置后的有效力）意义上的六维力。——支持：L454–458："We remove sensor bias and calibrated tool gravity, transform the wrench to the TCP frame with moment about the TCP origin, and apply a causal second-order Butterworth low-pass filter with a 10-Hz cutoff. The resulting wrench represents external contact and grasped-payload loads."
3. 【推断】「等效六维力」在论文的对应物不只是观测，而是可预测、可作控制参考的闭环量：预测 wrench 直接充当 20 Hz 导纳外环的力参考。——支持：L55–57（Abstract）："The predicted wrench chunk acts as a force reference for a 20-Hz outer-loop admittance controller within a slow–fast control system that maintains force feedback and smooth motion between model queries."；L339–341："We use the predicted wrench as a feedback reference to maintain the interaction expected along the jointly predicted motion."
4. 【推断】六维力被结构化地嵌入世界动作模型：与 10 维动作（含 6-D 连续旋转）配对、时间对齐地联合预测，说明「六维力」是动作模型的一等输出而非后处理量。——支持：L254–257："Outputs are video latents Z +t , motion targets At = (at,i)Hi=1, and wrenches Wt = (wt,i)Hi=1, with hats denoting predictions. Training pairs each motion target with the compensated wrench at the same timestamp."；L258–259（R10 动作定义）。
5. 【推断】六维力数据来自腕部六轴 F/T 传感器（150 Hz）并与 15 Hz 的运动/图像流按时间戳对齐，为「等效六维力」提供了可采集、可对齐、可监督学习的完整数据链路证据。——支持：L453–454："A wrist-mounted LinkTouch six-axis F/T sensor records wrench at 150 Hz."；L459–462："Motion commands, robot state, and images are recorded at 15 Hz and aligned with the wrench stream by timestamps."

## 9) 图注 / 表注逐字抄录

图 1（L42–44）："Fig. 1. CW-WAM helps robots use contact forces to guide their movements. It learns from past motions and wrist force measurements to predict both how to move and the forces expected during contact. These predictions guide motion corrections during execution. Peg insertion success increases from 25% with WAM to 70% with CW-WAM; adding force feedback to CW-WAM reduces force-tracking error from 4.8 to 2.7 N."

图 2（L180–183）："Fig. 2. CW-WAM jointly predicts video, motion, and wrench from visual observations, language, robot state, and interaction history. Paired motion–wrench tokens capture recent interactions, while long-term wrench memory retains earlier contact information. Pair-causal attention links each future wrench to its paired action and earlier pairs. The model jointly denoises the three streams using separate prediction heads. Predicted motion and wrench trajectories provide paired references for robot control."

图 3（L294–297）："Fig. 3. Slow–fast motion–force control. Admittance corrects nominal base-frame poses before IK. Joint-trajectory updates preserve C2 continuity. Wrench feedback is compensated; TCP-pose feedback optionally aligns wrench progress to measured TCP pose."

表 I（L327–337，含内容照抄）：
"TABLE I　SELECTED ADMITTANCE GAINS"
表头："Task　Feedback weights　M　C　Ka　kI　λ"
- Peg insertions：sT = diag(1,1,0.05)、sR = diag(1,1,0.2)；M = 5 / 0.1，C = 75 / 1.5，Ka = 200 / 4，kI = 0.5 / 0.5，λ = 0.5 / 0.5
- Adapter insertions：sT = diag(1,1,0.5)、sR = diag(1,1,1)；M = 5 / 0.1，C = 75 / 1.5，Ka = 200 / 4，kI = 0.1 / 1，λ = 0.1 / 1
- Whiteboard wipings：sT = diag(0,0,1)、sR = diag(0,0,0)；M = 3 / –，C = 34.64 / –，Ka = 100 / –，kI = – / –，λ = – / –
- 表注（L336–337）："S= blkdiag(s T , sR);s T ands R weight translational and rotational feedback, respectively.k I andλare ins −1."

表 II（L431–434）："TABLE II　OPEN-LOOP PREDICTION ACROSS FOUR TASKS. PADDED AND VALID DENOTE PADDED-WIDTH AND VALID-CHANNEL LOSS NORMALIZATION. BOLD:LOWEST ERROR; –:NO WRENCH OUTPUT"

图 4（L680–681）："Fig. 4. Task outcomes and execution quality. Top: insertion or marker-removal success out of 20 trials; hatched bars show successful release. Quality rows compare all 20 trials (light, hatched) with successful subsets (dark bars); these groups coincide for wiping. A dash indicates no wrench prediction."

图 5（L755–760）："Fig. 5. Contact-force curves across tasks. Colored curves show measured force magnitudes across successful trials. Thick blue and green curves show CW-WAM without and with admittance, respectively. The black curve and gray band show the demonstration reference. Curves are aligned by contact phase, with low-force intervals compressed. Adapter unloading includes only trials with successful release."

图 6（L829–833）："Fig. 6. Load-dependent lifting under light and heavy payloads. Curves show representative end-effector height trajectories. Dashed lines indicate approximate heights needed to clear the bin rim. The lower panels show peak heights from all ten trials per method and load. Diamonds mark the trials shown in the trajectories. Gray lines and bands show demonstration references."

图 7（L944–949）："Fig. 7. Force and torque tracking for peg insertion (left) and whiteboard wiping (right). (a–f) Measured and reference magnitudes, with three-axis force RMSE labels. Each example is selected from the trial whose force variation is closest to the group median. (g,h) Force variation (solid bars, left axis) and torque variation (hatched bars, right axis). Aggregate bars summarize 10 trials per condition."

附注·致谢（L858–861）："Under the authors' guidance, OpenAI Codex assisted with polishing the Abstract and Sections I–VI, code development, and plotting scripts supporting Figs. 1, 4–7."
